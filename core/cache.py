"""
Redis 缓存模块
==============
以「问题 + 系统提示词 + 知识库版本」为 key（MD5 哈希）缓存 AI 回答，
减少重复 API 调用，降低成本和延迟。

缓存键附加因子（保证回答不过期错乱）：
- system_prompt_hash：当前系统提示词的 MD5，界面保存新提示词时更新；
- RAG_VERSION：RAG 知识库版本号，上传/删除/清空文档时 +1。
两者任一变化，缓存键随之变化，旧缓存自然失效，无需手动清空。

语义缓存（v2）：精确匹配未命中时，把问题转成向量与缓存中的历史问题
做余弦相似度比较，意思相近的问题也能命中缓存，提升命中率（实测本地模型
下同义改写相似度约 0.6~0.99、无关问题约 0.45，默认阈值 0.95 无安全误命中）。
回答与问题向量一起以 JSON 存入缓存值；嵌入不可用或调用失败时自动降级为
纯精确匹配。

配置（全部通过环境变量，均有默认值）：
- REDIS_HOST：Redis 地址，默认 127.0.0.1（避免 Windows 下 localhost
  解析出 IPv6 地址导致连接挂起翻倍）
- REDIS_PORT：Redis 端口，默认 6379
- REDIS_PASSWORD：Redis 密码，默认无
- REDIS_DB：Redis 数据库编号，默认 0
- CACHE_TTL：缓存过期秒数，默认 3600（1 小时）
- OPENAI_API_KEY：语义缓存嵌入源开关，配置后调用 OpenAI
  text-embedding-3-small；未配置则用 ChromaDB 内置本地模型
  ONNXMiniLM-L6-v2（离线、免密钥）。DeepSeek 未提供嵌入接口，
  因此不使用 DEEPSEEK_API_KEY 做嵌入。

启动 Redis（Docker，一行命令）：
    docker run -d --name redis -p 6379:6379 redis:7

降级策略：Redis 未安装或连接失败时，所有读写静默跳过，
主应用流程完全不受影响（仅少一次缓存加速）。
每次可用性检查有硬超时保护（守护线程 ping，超时即判不可用），
检查失败后冷却时间指数退避，绝不阻塞主流程。
"""

import hashlib
import json
import logging
import math
import os
import threading
import time

from core.config import SEMANTIC_CACHE_THRESHOLD

try:
    import redis as redis_lib
    REDIS_LIB_AVAILABLE = True
except ImportError:
    REDIS_LIB_AVAILABLE = False

logger = logging.getLogger("ai_chat.core.cache")

# ====================== 常量配置 ======================
KEY_PREFIX = "ai_cache"                                    # 缓存键前缀（避免与其它应用冲突）
CACHE_TTL = int(os.environ.get("CACHE_TTL", 3600))         # 缓存过期秒数，默认 1 小时
CHECK_INTERVAL = 30                                        # 检查成功时的冷却时间（秒）
FAIL_COOLDOWN_MAX = 1800                                   # 检查失败时冷却时间的上限（30 分钟）
PING_DEADLINE = 1.5                                        # ping 硬超时（秒）：超过即判定不可用

# ====================== 缓存键附加因子 ======================
# 同一问题的回答不仅取决于问题本身，还取决于系统提示词与 RAG 知识库内容：
# - system_prompt_hash：当前系统提示词的 MD5（初始为空提示词的哈希，
#   每次发送消息前按当前提示词刷新；界面保存新提示词时立即更新）；
# - RAG_VERSION：知识库版本号（上传/删除/清空文档时 +1，见 core/rag.py）。
# 两者任一变化，缓存键随之变化，旧缓存自然失效。
system_prompt_hash = hashlib.md5(b"").hexdigest()
RAG_VERSION = 0


def update_system_prompt_hash(prompt: str) -> None:
    """系统提示词变化时更新其哈希（缓存键随之变化，旧缓存自然失效）"""
    global system_prompt_hash
    system_prompt_hash = hashlib.md5(prompt.strip().encode("utf-8")).hexdigest()


def bump_rag_version() -> None:
    """RAG 文档上传/删除后调用：知识库版本 +1，旧缓存自然失效"""
    global RAG_VERSION
    RAG_VERSION += 1

# ====================== 语义缓存（相似问题命中） ======================
# 精确缓存要求问题文本完全相同才命中（实测命中率约 44%）；语义缓存在精确
# 未命中时，把当前问题转成向量，与 Redis 中同模型、同提示词/知识库版本的
# 历史条目做余弦相似度比较，最相似且达到阈值（SEMANTIC_CACHE_THRESHOLD，
# 见 core/config.py）的条目视为命中，返回其答案，显著提升命中率。
#
# 嵌入来源（与 RAG 一致，自动选择）：
# - 配置了 OPENAI_API_KEY：调用 OpenAI text-embedding-3-small（在线）；
# - 未配置：ChromaDB 内置本地模型 ONNXMiniLM-L6-v2（离线、免密钥，
#   首次使用自动下载模型文件）。
# 注意：DeepSeek 未提供嵌入接口，因此不使用 DEEPSEEK_API_KEY 做嵌入。
# 嵌入调用失败时 _embedding_failed 置位，本次进程内不再重试，自动降级为
# 纯精确匹配，原有缓存功能不受任何影响。
SEMANTIC_ENABLED = True          # 语义缓存总开关（按需置 False 可完全关闭）
_embedding_fn = None             # 嵌入函数懒加载单例
_embedding_failed = False        # 嵌入失败标记（失败后本次进程内不再重试）


def get_embedding(text: str):
    """获取文本的向量表示

    Args:
        text: 问题文本

    Returns:
        list[float] 或 None: 问题向量；嵌入不可用或调用失败时返回 None
        （调用方应降级为精确匹配）
    """
    global _embedding_fn, _embedding_failed
    if not SEMANTIC_ENABLED or _embedding_failed:
        return None
    try:
        if _embedding_fn is None:
            from chromadb.utils import embedding_functions
            if os.getenv("OPENAI_API_KEY"):
                _embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=os.environ["OPENAI_API_KEY"],
                    api_base=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    model_name="text-embedding-3-small",
                )
            else:
                _embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        vector = _embedding_fn([text.strip()])[0]
        return [float(v) for v in vector]
    except Exception as e:
        logger.warning("语义缓存嵌入失败，本次进程内降级为精确匹配：%s", e)
        _embedding_failed = True
        _embedding_fn = None
        return None


def _cosine_similarity(vec_a, vec_b) -> float:
    """两个向量的余弦相似度（0~1，越接近 1 越相似）

    纯 Python 实现：向量只有几百维、缓存条目量级很小，性能足够，
    无需引入 scikit-learn / numpy 依赖。维度不一致、零向量等异常输入返回 0。
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _encode_cache_value(answer: str, embedding, model: str) -> str:
    """把「回答 + 问题向量 + 模型」打包成 JSON 字符串（新版缓存值格式）"""
    return json.dumps(
        {"answer": answer, "embedding": embedding, "model": model},
        ensure_ascii=False,
    )


def _extract_answer(raw):
    """从缓存值中取出回答文本（兼容新旧两种存储格式）

    - 新版 JSON：{"answer": ..., "embedding": ..., "model": ...} → 取 answer；
    - 旧版纯文本：直接是回答文本，原样返回。
    """
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
        if isinstance(data, dict) and "answer" in data:
            return data["answer"]
    except (ValueError, TypeError):
        pass
    return raw


def find_similar_in_cache(question_embedding, entries, threshold=SEMANTIC_CACHE_THRESHOLD):
    """在缓存条目中查找与当前问题语义相似的历史问题（纯计算，便于单元测试）

    Args:
        question_embedding: 当前问题的向量
        entries: 候选缓存条目列表，每项形如
                 {"answer": str, "embedding": list[float], "model": str}
        threshold: 余弦相似度阈值，>= 阈值才视为命中

    Returns:
        dict 或 None: 最相似且达到阈值的条目（其 answer 即缓存回答）；
        无任何条目达标时返回 None
    """
    best, best_score = None, 0.0
    for entry in entries:
        emb = entry.get("embedding")
        if not emb:
            continue
        score = _cosine_similarity(question_embedding, emb)
        if score >= threshold and score > best_score:
            best, best_score = entry, score
    return best

_client = None                          # Redis 客户端（懒加载单例）
# 可用性缓存：available 最近一次检查结果；checked_at 上次检查时间；
# cooldown 当前冷却时长（失败时指数退避增长，成功后恢复默认值）
_availability = {"available": False, "checked_at": 0.0, "cooldown": CHECK_INTERVAL}


# ====================== 内部辅助函数 ======================
def _get_client():
    """获取 Redis 客户端（懒加载单例，创建失败返回 None）

    Returns:
        redis.Redis 或 None: 客户端对象；redis 库未安装/参数非法时返回 None
    """
    global _client
    if _client is not None:
        return _client
    if not REDIS_LIB_AVAILABLE:
        return None
    try:
        _client = redis_lib.Redis(
            host=os.environ.get("REDIS_HOST", "127.0.0.1"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            password=os.environ.get("REDIS_PASSWORD") or None,
            db=int(os.environ.get("REDIS_DB", "0")),
            socket_connect_timeout=1,   # 连接超时 1 秒，失败快速降级
            socket_timeout=1,           # 读写超时 1 秒
            decode_responses=True,      # 直接返回字符串
        )
    except Exception:
        _client = None
    return _client


def _current_factor() -> str:
    """当前提示词/知识库版本因子（缓存键第三段）

    语义缓存扫描也按此因子隔离：提示词或知识库变化后，扫描只落在
    当前因子下的条目，绝不会跨版本匹配旧回答。
    """
    return hashlib.md5(f"{system_prompt_hash}:{RAG_VERSION}".encode()).hexdigest()[:8]


def _make_key(question, model):
    """生成缓存键：前缀 + 模型名 + 提示词/知识库版本因子 + 问题（去首尾空白）的 MD5

    带模型名是为了防止不同模型对同一问题的回答互相污染；
    附加因子（system_prompt_hash + RAG_VERSION）保证系统提示词或知识库
    变化后旧缓存键不再命中，回答不会过期错乱。

    Args:
        question: 用户问题原文
        model: 当前模型名

    Returns:
        str: 缓存键
    """
    digest = hashlib.md5(question.strip().encode("utf-8")).hexdigest()
    return f"{KEY_PREFIX}:{model}:{_current_factor()}:{digest}"


# ====================== 对外接口 ======================
def _ping_with_deadline(client):
    """在守护线程中执行 Redis ping，超过硬超时即判定不可用

    背景：Windows 上连接无监听的端口时，防火墙可能丢弃 SYN 包导致连接
    长时间挂起（实测单地址 15 秒、localhost 双地址 26 秒），redis-py 的
    socket_connect_timeout 无法有效中断该过程。线程化 + 硬超时保证调用方
    最多等待 PING_DEADLINE 秒。

    Args:
        client: redis 客户端对象

    Returns:
        bool: 硬超时内 ping 成功返回 True，否则 False
    """
    result = []

    def _worker():
        try:
            result.append(bool(client.ping()))
        except Exception:
            result.append(False)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(PING_DEADLINE)
    return result[0] if result else False


def is_redis_available():
    """检查 Redis 是否可用（带冷却时间，避免每次请求都 ping）

    冷却策略：
    - 检查成功 → 30 秒后重新检查（保持对宕机的感知）
    - 检查失败 → 冷却时间指数退避（30 秒 → 2 分钟 → 8 分钟 → 最多 30 分钟），
      避免 Redis 宕机或网络异常时反复拖慢对话
    每次检查本身有硬超时保护（守护线程 ping，见 _ping_with_deadline）。

    Returns:
        bool: True 表示 Redis 已连接可用
    """
    if not REDIS_LIB_AVAILABLE:
        return False
    now = time.time()
    if now - _availability["checked_at"] < _availability["cooldown"]:
        return _availability["available"]
    try:
        client = _get_client()
        if client is None or not _ping_with_deadline(client):
            raise ConnectionError("Redis ping 失败")
        _availability["available"] = True
        _availability["cooldown"] = CHECK_INTERVAL  # 恢复后回到正常检查频率
    except Exception:
        _availability["available"] = False
        _availability["cooldown"] = min(
            _availability["cooldown"] * 4, FAIL_COOLDOWN_MAX)  # 失败退避
        _client = None  # 重置客户端，下次检查时重建连接
    _availability["checked_at"] = now
    return _availability["available"]


def get_cached_response(question, model="default"):
    """获取缓存的 AI 回答（先精确匹配，未命中时做语义匹配）

    语义匹配：把问题转成向量，扫描同模型、同提示词/知识库版本因子下的
    全部缓存条目，余弦相似度最高且 >= SEMANTIC_CACHE_THRESHOLD 的条目
    视为命中（比较逻辑见 find_similar_in_cache）。
    嵌入不可用/失败时自动降级为纯精确匹配，行为与旧版完全一致。

    Args:
        question: 用户问题原文
        model: 当前模型名（参与缓存键，防止跨模型混用）

    Returns:
        str 或 None: 命中时返回缓存的回答文本；未命中/Redis 不可用时返回 None
    """
    if not is_redis_available():
        return None
    try:
        client = _get_client()
        # 1. 精确匹配（原有逻辑，保持原样）
        raw = client.get(_make_key(question, model))
        if raw:
            return _extract_answer(raw)

        # 2. 语义匹配：扫描当前模型 + 当前因子下的历史条目
        question_embedding = get_embedding(question)
        if question_embedding is None:
            return None
        entries = []
        prefix = f"{KEY_PREFIX}:{model}:{_current_factor()}:"
        for key in client.scan_iter(match=f"{prefix}*", count=200):
            value = client.get(key)
            try:
                data = json.loads(value) if value else None
            except (ValueError, TypeError):
                continue  # 旧版纯文本条目没有向量，跳过
            if (isinstance(data, dict) and data.get("embedding")
                    and data.get("model") == model):
                entries.append(data)
        hit = find_similar_in_cache(question_embedding, entries)
        return hit["answer"] if hit else None
    except Exception:
        return None


def set_cached_response(question, response, model="default"):
    """缓存 AI 回答（默认 1 小时过期），失败时静默跳过

    写入格式：{"answer": 回答, "embedding": 问题向量, "model": 模型}（JSON）。
    嵌入不可用/失败时退化为旧版纯文本格式（精确匹配仍可用，
    只是该条目不参与语义匹配）。

    Args:
        question: 用户问题原文
        response: AI 的完整回答文本
        model: 当前模型名
    """
    if not is_redis_available() or not response:
        return
    try:
        embedding = get_embedding(question)
        value = (_encode_cache_value(response, embedding, model)
                 if embedding is not None else response)
        _get_client().setex(_make_key(question, model), CACHE_TTL, value)
    except Exception:
        pass  # 缓存写入失败不影响主流程


def clear_cache():
    """清空本应用的全部缓存（按前缀匹配删除）

    Returns:
        int: 删除的缓存条数（Redis 不可用时返回 0）
    """
    if not is_redis_available():
        return 0
    try:
        client = _get_client()
        keys = list(client.scan_iter(match=f"{KEY_PREFIX}:*"))
        if keys:
            client.delete(*keys)
        return len(keys)
    except Exception:
        return 0


def get_cache_status():
    """获取缓存服务状态（供界面展示）

    Returns:
        tuple: (状态说明文字, 状态级别)，级别取值：
               "ok"          Redis 已连接，缓存生效
               "no_lib"      redis 库未安装
               "unreachable" Redis 未启动或连接失败（已自动降级）
    """
    if not REDIS_LIB_AVAILABLE:
        return "未安装 redis 库，缓存不可用（pip install redis）", "no_lib"
    if is_redis_available():
        host = os.environ.get("REDIS_HOST", "127.0.0.1")
        port = os.environ.get("REDIS_PORT", "6379")
        return f"Redis 已连接（{host}:{port}），缓存生效中", "ok"
    return "Redis 不可用，已自动跳过缓存", "unreachable"
