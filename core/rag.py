"""
RAG（检索增强生成）模块
=======================
负责文档的加载、切分、向量化存储与相似度检索。

技术栈：
- LangChain：文档加载（PyPDFLoader / TextLoader）+ 切分（RecursiveCharacterTextSplitter）
- ChromaDB：本地持久化向量数据库（数据保存在项目根目录 ./chroma_db）

嵌入模型（通过环境变量 EMBEDDING_PROVIDER 配置，默认 auto）：
- ollama：使用 Ollama 的多语言模型 bge-m3（默认优先；需本地运行 Ollama 并已
          ollama pull bge-m3，中文语义区分能力优于英文模型）
- openai：使用 OpenAI 的 text-embedding-3-small（需设置 OPENAI_API_KEY，可选 OPENAI_BASE_URL）
- local ：使用 ChromaDB 内置本地嵌入模型 ONNXMiniLM-L6-v2（免密钥、无需联网）
- auto  ：优先 ollama（可用时），否则有 OPENAI_API_KEY 用 openai，否则降级为 local

注意：已存在的向量库必须沿用其创建时的嵌入方式（向量维度不同不能混用），
模块会自动检测并沿用；集合为空（无向量）时允许按新配置重建，不会丢数据。
"""

import os
import uuid
from datetime import datetime
from tempfile import NamedTemporaryFile

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

import chromadb
from chromadb.utils import embedding_functions

from core.cache import (
    OLLAMA_EMBEDDING_MODEL,
    bump_rag_version,
    ollama_base_url,
    ollama_embeddings_available,
)

# ====================== 常量配置 ======================
# 项目根目录（本模块位于 core/ 下，父目录即项目根目录）
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")   # 向量库本地持久化目录
COLLECTION_NAME = "documents"                      # 向量集合名称
CHUNK_SIZE = 1000                                  # 切分块大小
CHUNK_OVERLAP = 200                                # 切分块重叠大小
TOP_K_DEFAULT = 3                                  # 默认检索返回片段数

# 服务器模式配置（Docker 部署时由 compose 注入）：
# 设置 CHROMA_HOST 后自动切换为连接独立 ChromaDB 服务；未设置则用本地内嵌模式
CHROMA_HOST = os.environ.get("CHROMA_HOST", "").strip()
CHROMA_PORT = int(os.environ.get("CHROMA_PORT", "8000"))

# ====================== 模块级懒加载单例 ======================
_client = None       # ChromaDB 持久化客户端（首次使用时创建）
_collection = None   # 文档向量集合
_provider = None     # 当前向量库使用的嵌入方式："ollama" / "openai" / "local"
_status_msg = ""     # 状态提示信息（如自动降级提示、警告）


# ====================== 内部辅助函数 ======================
def _get_client():
    """获取 ChromaDB 客户端（懒加载单例，自动识别运行模式）

    运行模式：
    - 本地模式（未设置 CHROMA_HOST）：内嵌 PersistentClient，数据存 ./chroma_db
    - 服务器模式（设置 CHROMA_HOST）：HttpClient 连接独立 ChromaDB 服务（Docker 部署）

    Returns:
        chromadb.PersistentClient 或 chromadb.HttpClient: 向量数据库客户端
    """
    global _client
    if _client is None:
        if CHROMA_HOST:
            _client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        else:
            _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


def _get_embedding_function(provider):
    """根据嵌入方式创建对应的嵌入函数对象

    Args:
        provider: 嵌入方式，取值为 "ollama" / "openai" / "local"

    Returns:
        嵌入函数对象；local 方式返回 None（ChromaDB 会使用内置默认嵌入模型）

    Raises:
        RuntimeError: ollama 方式但 Ollama 嵌入不可用时抛出
    """
    if provider == "ollama":
        if not ollama_embeddings_available():
            raise RuntimeError(
                "Ollama 嵌入不可用（未安装 langchain-ollama / ollama，或服务未启动）。"
                "请先启动 Ollama 并执行 ollama pull bge-m3，"
                "或将 EMBEDDING_PROVIDER 设为 local 使用内置本地模型。"
            )
        # 用 ChromaDB 原生 OllamaEmbeddingFunction：它能被正确序列化进集合配置、
        # 重启后自动重建；langchain 的 ChromaLangchainEmbeddingFunction 在
        # chromadb 1.5.9 的 query 路径会抛 TypeError，故不采用
        return embedding_functions.OllamaEmbeddingFunction(
            url=ollama_base_url(), model_name=OLLAMA_EMBEDDING_MODEL)
    if provider == "openai":
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            api_base=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model_name="text-embedding-3-small",
        )
    return None  # None 表示使用 ChromaDB 内置的 ONNXMiniLM-L6-v2 本地模型


def _resolve_provider():
    """解析实际使用的嵌入方式

    规则：
    1. 向量库已有数据（片段数 > 0）→ 必须沿用其创建时记录的嵌入方式
       （否则向量维度不匹配），配置与之冲突时给出提示；
       空集合不构成约束——按新配置重建即可，不会丢数据。
    2. 向量库不存在或为空 → 按环境变量 EMBEDDING_PROVIDER 决定
       （auto 时优先 Ollama bge-m3，其次 openai，最后 local）。

    Returns:
        str: 实际使用的嵌入方式（"ollama" / "openai" / "local"）

    Raises:
        RuntimeError: 已有向量库需要 OpenAI 密钥但未配置时抛出
    """
    global _status_msg
    env_provider = os.environ.get("EMBEDDING_PROVIDER", "auto").strip().lower()

    # 检查向量库是否已有数据，并读取其创建时使用的嵌入方式
    existing_provider = None
    try:
        col = _get_client().get_collection(COLLECTION_NAME)
        if col.count() > 0:  # 空集合不锁定嵌入方式
            existing_provider = (col.metadata or {}).get("embedding_provider")
    except Exception:
        existing_provider = None  # 集合不存在，属于首次运行

    if existing_provider:
        if env_provider not in ("auto", existing_provider):
            _status_msg = f"⚠️ 向量库由「{existing_provider}」嵌入创建，已自动沿用该方式（配置的 {env_provider} 被忽略）"
        if existing_provider == "openai" and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(
                "向量库由 OpenAI 嵌入创建，但未设置 OPENAI_API_KEY 环境变量，无法加载。"
                "请设置密钥后重启应用，或先「清空所有文档」再改用其他嵌入。"
            )
        return existing_provider

    # 首次运行 / 空向量库：按配置决定嵌入方式
    if env_provider == "ollama":
        if ollama_embeddings_available():
            return "ollama"
        _status_msg = "⚠️ Ollama 嵌入不可用，已自动降级为本地内置嵌入模型"
        return "local"
    if env_provider == "openai":
        return "openai"
    if env_provider == "local":
        return "local"
    # auto：优先 Ollama（多语言，中文效果最好），其次 OpenAI，最后本地
    if ollama_embeddings_available():
        return "ollama"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "local"


def _get_collection():
    """获取（或创建）文档向量集合，自动处理嵌入方式一致性

    Returns:
        chromadb.Collection: 文档向量集合
    """
    global _collection, _provider
    if _collection is None:
        _provider = _resolve_provider()
        client = _get_client()
        # 空集合若嵌入方式与当前配置不一致，先删除再重建：
        # get_or_create_collection 不会更新已有集合的 metadata，
        # 不删除会导致嵌入方式永远停留在旧值（集合为空，删除不丢数据）
        try:
            existing = client.get_collection(COLLECTION_NAME)
            existing_provider = (existing.metadata or {}).get("embedding_provider")
            if existing.count() == 0 and existing_provider != _provider:
                client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass  # 集合不存在：直接走创建流程
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_get_embedding_function(_provider),
            metadata={"embedding_provider": _provider, "hnsw:space": "cosine"},
        )
    return _collection


def _load_text_file(path):
    """读取 TXT/MD 文本文件，自动识别常见中文编码

    Args:
        path: 文本文件路径

    Returns:
        list[Document]: 包含全文的单个文档对象

    Raises:
        ValueError: 文件编码无法识别或内容为空时抛出
    """
    # 优先用 LangChain 的 TextLoader（自动检测编码）
    try:
        docs = TextLoader(path, autodetect_encoding=True).load()
        if docs and docs[0].page_content.strip():
            return docs
    except Exception:
        pass

    # 兜底方案：手动尝试常见中文编码
    with open(path, "rb") as f:
        raw = f.read()
    for encoding in ("utf-8", "gb18030", "utf-16"):
        try:
            text = raw.decode(encoding).strip()
            if text:
                return [Document(page_content=text, metadata={"source": os.path.basename(path)})]
        except (UnicodeDecodeError, UnicodeError):
            continue
    raise ValueError("无法识别文档编码，请将文件另存为 UTF-8 格式后重试")


# ====================== 对外核心函数 ======================
def load_document(file):
    """加载上传的文档（PDF/TXT/MD）并切分成片段

    Args:
        file: Streamlit 的 UploadedFile 对象（含 name 与 getvalue 方法），
              或本地文件路径字符串

    Returns:
        list[Document]: 切分后的文档片段列表，每个片段带有 doc_id/file_name/来源等元数据

    Raises:
        ValueError: 文档无有效文本内容（如扫描版 PDF）时抛出
    """
    # 统一处理两种入参：路径字符串 / 上传文件对象
    if isinstance(file, str):
        file_name = os.path.basename(file)
        tmp_path, need_clean = file, False
    else:
        file_name = getattr(file, "name", "未命名文档")
        suffix = os.path.splitext(file_name)[1].lower()
        with NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(file.getvalue())
        tmp_path, need_clean = tmp_file.name, True

    try:
        # 第一步：按格式加载文档
        suffix = os.path.splitext(file_name)[1].lower()
        if suffix == ".pdf":
            docs = PyPDFLoader(tmp_path).load()
            for d in docs:  # 页码从 0 开始，转成从 1 开始便于展示
                if "page" in d.metadata:
                    d.metadata["page"] = int(d.metadata["page"]) + 1
        else:  # txt / md
            docs = _load_text_file(tmp_path)

        # 第二步：切分（中文友好分隔符 + 块间重叠）
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("文档内容为空，无法提取文本（扫描版 PDF 暂不支持）")

        # 第三步：为每个片段附加文档级元数据（用于列表展示与删除）
        doc_id = uuid.uuid4().hex
        upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for chunk in chunks:
            chunk.metadata.update(
                {"doc_id": doc_id, "file_name": file_name, "upload_time": upload_time}
            )
        return chunks
    finally:
        if need_clean and os.path.exists(tmp_path):
            os.remove(tmp_path)  # 清理临时文件


def add_to_vectorstore(docs):
    """将文档片段批量写入向量库（内置去重：同名文件已存在时拒绝入库）

    Args:
        docs: load_document 返回的片段列表

    Returns:
        int: 成功写入的片段数量

    Raises:
        ValueError: 文档内容为空，或同名文件已入库时抛出
    """
    if not docs:
        raise ValueError("文档内容为空，没有可入库的片段")
    collection = _get_collection()
    # 去重保护：同名文件已入库时拒绝再次写入（界面层已拦截，此处作为兜底防线）
    file_name = (docs[0].metadata or {}).get("file_name", "")
    if file_name:
        existing = collection.get(where={"file_name": file_name}, include=["metadatas"])
        if existing.get("ids"):
            raise ValueError(f"文件《{file_name}》已存在，请勿重复上传")
    collection.add(
        ids=[uuid.uuid4().hex for _ in docs],
        documents=[d.page_content for d in docs],
        metadatas=[d.metadata for d in docs],
    )
    bump_rag_version()  # 知识库内容变化：缓存键版本 +1，旧缓存自然失效
    return len(docs)


def search(query, top_k=TOP_K_DEFAULT):
    """检索与问题最相关的文档片段（相似度检索）

    Args:
        query: 用户问题文本
        top_k: 返回的片段数量，默认 3

    Returns:
        list[dict]: 片段列表，每项包含 content/source/score/doc_id 等字段；
                    向量库为空时返回空列表
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    result = collection.query(
        query_texts=[query],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    results = []
    documents = result.get("documents")[0]
    metadatas = result.get("metadatas")[0]
    distances = result.get("distances")[0]
    for content, meta, dist in zip(documents, metadatas, distances):
        source = meta.get("file_name", "未知文档")
        if meta.get("page"):  # PDF 片段带页码
            source += f"（第{meta['page']}页）"
        results.append({
            "content": content,
            "source": source,
            "score": round(1 - dist, 4),  # cosine 距离转相似度（越接近 1 越相关）
            "doc_id": meta.get("doc_id", ""),
            "upload_time": meta.get("upload_time", ""),
        })
    return results


def get_document_list():
    """列出已上传的文档（按 doc_id 聚合，统计每个文档的片段数）

    Returns:
        list[dict]: 文档列表（按上传时间倒序），每项包含
                    doc_id/file_name/chunks/upload_time
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    data = collection.get(include=["metadatas"])
    docs_map = {}
    for meta in data.get("metadatas", []):
        doc_id = meta.get("doc_id")
        if not doc_id:
            continue
        if doc_id not in docs_map:
            docs_map[doc_id] = {
                "doc_id": doc_id,
                "file_name": meta.get("file_name", "未知文档"),
                "chunks": 0,
                "upload_time": meta.get("upload_time", ""),
            }
        docs_map[doc_id]["chunks"] += 1
    return sorted(docs_map.values(), key=lambda d: d["upload_time"], reverse=True)


def delete_document(doc_id):
    """按文档ID删除单个文档的所有向量片段

    先按 doc_id 查出全部片段 ID，再按 ID 精确删除：部分 ChromaDB 版本
    对 where 过滤删除支持不佳（可能静默失败导致"点击没反应"），按 ID
    删除更可靠，且能返回实际删除数量供界面提示。

    Args:
        doc_id: 文档唯一标识（load_document 时生成）

    Returns:
        int: 实际删除的片段数量（0 表示该文档不存在或已被删除）

    Raises:
        ValueError: doc_id 为空时抛出
    """
    if not doc_id:
        raise ValueError("文档ID为空，无法删除")
    collection = _get_collection()
    # 第一步：查出该文档的全部片段 ID（顺带确认文档是否存在）
    existing = collection.get(where={"doc_id": doc_id}, include=["metadatas"])
    ids = existing.get("ids") or []
    if not ids:
        return 0
    # 第二步：按 ID 精确删除
    collection.delete(ids=ids)
    bump_rag_version()  # 知识库内容变化：缓存键版本 +1，旧缓存自然失效
    return len(ids)


def invalidate_collection_cache():
    """重置集合单例缓存，强制下次访问重新从 ChromaDB 加载最新数据

    删除文档后调用，确保 get_document_list 等函数能立即反映删除结果
    （避免 _get_collection 缓存的集合对象返回旧数据）。
    """
    global _collection
    _collection = None


def clear_all():
    """清空向量库中的全部文档（删除整个集合，下次入库时自动重建）"""
    try:
        _get_client().delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # 集合本来就不存在时无需处理
    bump_rag_version()  # 知识库内容变化：缓存键版本 +1，旧缓存自然失效
    global _collection, _provider, _status_msg
    _collection = None  # 重置单例，让下次访问重新创建集合
    _provider = None
    _status_msg = ""


def get_rag_status():
    """获取 RAG 当前状态（供界面展示）

    Returns:
        tuple: (是否可用, 状态说明文字, 当前嵌入方式, 向量片段总数)
    """
    try:
        collection = _get_collection()
        return True, _status_msg, _provider, collection.count()
    except Exception as e:
        return False, f"❌ 向量库加载失败：{str(e)}", None, 0
