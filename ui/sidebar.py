"""侧边栏 UI（仅负责渲染）：API 配置、高级参数、RAG 文档管理、工具/缓存设置、对话管理"""

import json
import logging
import os
from datetime import datetime

import streamlit as st

from core import (
    AGENT_AVAILABLE,
    CACHE_AVAILABLE,
    MODELS_AVAILABLE,
    RAG_AVAILABLE,
    TOOLS_AVAILABLE,
    AppConfig,
    MODEL_LIST,
    clear_all_conversations_from_db,
    clear_conversation_messages_from_db,
    delete_conversation_from_db,
    estimate_tokens,
    save_session_to_file,
)
from ui.components import toast

logger = logging.getLogger("ai_chat.ui.sidebar")

# 以下功能模块依赖第三方库，缺失时对应 UI 分支自动降级（与重构前行为一致）
try:
    from core.rag import (
        add_to_vectorstore,
        clear_all,
        delete_document,
        get_document_list,
        get_rag_status,
        invalidate_collection_cache,
        load_document,
    )
except ImportError:
    pass
try:
    from core.tools import get_tool_names
except ImportError:
    pass
try:
    from core.cache import CACHE_TTL, clear_cache, get_cache_status
except ImportError:
    pass
try:
    from core.models import get_model_config, get_provider_config, list_ollama_models, list_providers
except ImportError:
    pass


# ====================== 多模型配置界面辅助函数 ======================
def _apply_model_defaults(model_name: str) -> None:
    """切换模型后，把温度/最大长度同步为该模型的独立默认值

    Args:
        model_name: 模型名称
    """
    if MODELS_AVAILABLE:
        cfg = get_model_config(model_name, provider_key=st.session_state.provider)
        st.session_state.temperature = cfg["temperature"]
        st.session_state.max_tokens = cfg["max_tokens"]


def _render_ollama_model_picker() -> None:
    """Ollama 模型选择：动态获取本地已安装模型，服务不可用时退化为手动输入"""
    base_url = st.session_state.base_urls.get("ollama", "http://localhost:11434/v1")
    cache_key = f"ollama_models_{base_url}"
    if cache_key not in st.session_state:
        models, err = list_ollama_models(base_url)
        st.session_state[cache_key] = (models, err)

    models, err = st.session_state[cache_key]
    if models:
        model_idx = (models.index(st.session_state.current_model)
                     if st.session_state.current_model in models else 0)
        st.session_state.current_model = st.selectbox(
            "选择模型（本地已安装）", models, index=model_idx,
            help="模型列表来自 Ollama 服务，下拉选择即可切换")
        if st.button("🔄 刷新模型列表", use_container_width=True):
            st.session_state.pop(cache_key)
            st.rerun()
    else:
        st.warning(err or "未检测到已安装模型")
        st.session_state.current_model = st.text_input(
            "模型名称（手动输入）",
            value=st.session_state.current_model,
            placeholder="如 qwen2.5:7b、llama3:8b",
            help=f"请先执行 `ollama pull 模型名` 安装模型（服务地址：{base_url}）"
        )
        if st.button("🔄 重新检测", use_container_width=True):
            st.session_state.pop(cache_key)
            st.rerun()


def _render_provider_config() -> None:
    """渲染 API 配置面板：提供方切换 + 动态配置字段 + 模型选择"""
    provider_keys = [k for k, _ in list_providers()]
    provider_labels = [label for _, label in list_providers()]
    idx = (provider_keys.index(st.session_state.provider)
           if st.session_state.provider in provider_keys else 0)
    selected_label = st.selectbox(
        "模型提供方", provider_labels, index=idx,
        help="切换提供方后自动套用该提供方默认模型与参数；Ollama 为本地免费模型，无需密钥"
    )
    new_provider = provider_keys[provider_labels.index(selected_label)]

    # 提供方切换：更新会话状态，并套用新提供方的默认模型与参数
    if new_provider != st.session_state.provider:
        st.session_state.provider = new_provider
        cfg = get_provider_config(new_provider)
        st.session_state.current_model = cfg["default_model"]
        _apply_model_defaults(st.session_state.current_model)
        save_session_to_file(st.session_state)
        st.rerun()

    cfg = get_provider_config(st.session_state.provider)

    # ---- 动态配置字段：需密钥的提供方显示密钥框；本地提供方显示服务地址框 ----
    if cfg.get("api_key_env"):
        key_value = st.text_input(
            f"{cfg['label'].split(' ', 1)[-1]} API Key",
            value=st.session_state.api_keys.get(st.session_state.provider, ""),
            type="password",
            placeholder=f"未填写时自动读取环境变量 {cfg['api_key_env']}",
            help=cfg.get("help", "")
        )
        st.session_state.api_keys[st.session_state.provider] = key_value
    else:
        base_url = st.text_input(
            "Ollama 服务地址",
            value=st.session_state.base_urls.get(st.session_state.provider, cfg["base_url"]),
            help=cfg.get("help", "") + "；Docker 部署时填 http://host.docker.internal:11434/v1"
        )
        st.session_state.base_urls[st.session_state.provider] = base_url

    # ---- 模型选择 ----
    if st.session_state.provider == "ollama":
        _render_ollama_model_picker()
    else:
        model_list = cfg["models"]
        model_idx = (model_list.index(st.session_state.current_model)
                     if st.session_state.current_model in model_list else 0)
        select_model = st.selectbox("选择模型", model_list, index=model_idx,
                                    help="reasoner 为推理模型，回复更深入但耗时更长")
        if select_model != st.session_state.current_model:
            st.session_state.current_model = select_model
            _apply_model_defaults(select_model)
            save_session_to_file(st.session_state)

    # ---- 当前模型默认参数提示 ----
    model_cfg = get_model_config(st.session_state.current_model,
                                 provider_key=st.session_state.provider)
    caption = (f"默认参数：temperature={model_cfg['temperature']}，"
               f"最大长度={model_cfg['max_tokens']}")
    if not model_cfg["supports_tools"]:
        caption += "；不支持工具调用"
    st.caption(caption)

    if st.button("✅ 保存配置", use_container_width=True):
        save_session_to_file(st.session_state)
        st.success("配置保存成功！")
        st.rerun()


def _render_legacy_api_config() -> None:
    """多模型模块不可用时的兜底配置面板（仅 DeepSeek，保证应用不崩溃）"""
    if "api_key" not in st.session_state:
        st.session_state.api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if "base_url" not in st.session_state:
        st.session_state.base_url = "https://api.deepseek.com"
    api_key = st.text_input(
        "DeepSeek API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="请输入你的API密钥"
    )
    base_url = st.text_input("接口地址", value=st.session_state.base_url)
    model_idx = (MODEL_LIST.index(st.session_state.current_model)
                 if st.session_state.current_model in MODEL_LIST else 0)
    select_model = st.selectbox("选择模型", MODEL_LIST, index=model_idx)
    if st.button("✅ 保存配置", use_container_width=True):
        st.session_state.api_key = api_key
        st.session_state.base_url = base_url
        st.session_state.current_model = select_model
        save_session_to_file(st.session_state)
        st.success("配置保存成功！")
        st.rerun()


# ====================== 功能开关联动（缓存 / RAG / 工具调用） ======================
# Streamlit 的 on_change 回调先于 session_state 更新执行：回调内读到的
# 是切换前的旧值，因此「旧值为 False」即表示用户本次操作是「开启」。
def _on_cache_enabled_change() -> None:
    """缓存开关联动：开启缓存时自动关闭 RAG、工具调用与 Agent 模式"""
    if not st.session_state.cache_enabled:  # 旧值为 False → 本次为开启
        st.session_state.rag_enabled = False
        st.session_state.tools_enabled = False
        st.session_state.agent_mode_enabled = False


def _on_rag_enabled_change() -> None:
    """RAG 开关联动：开启文档检索时自动关闭缓存（RAG 与工具调用不互斥）"""
    if not st.session_state.rag_enabled:
        st.session_state.cache_enabled = False


def _on_tools_enabled_change() -> None:
    """工具开关联动：开启工具调用时自动关闭缓存；关闭工具时同步关闭 Agent 模式"""
    if not st.session_state.tools_enabled:  # 旧值为 False → 本次为开启
        st.session_state.cache_enabled = False
    else:  # 本次为关闭：Agent 模式依赖工具调用，一并关闭
        st.session_state.agent_mode_enabled = False


def _delete_conversation(idx: int) -> None:
    """删除指定索引的历史对话（删除按钮的 on_click 回调）

    注意：Streamlit（≥1.37）中在 on_click 回调内调用 st.rerun() 会被当作
    no-op 并输出警告。回调只做状态修改并置位 _need_rerun 标记，真正的
    st.rerun() 统一由 app.py 顶层消费（详见 app.py 说明）。删除按钮被点击
    后 Streamlit 本就会自动触发一次重跑，因此界面能立即刷新。
    1. 删除后按被删位置修正 current_conversation_index，防止越界与错位：
       - 删除的是当前对话之前的条目 → 当前索引 -1（列表前移）
       - 删除的是当前对话本身 → 切换到相邻条目（优先后一个；
         删的是最后一条则退到前一条，pop 后的列表用 min 统一处理）
       - 删除的是当前对话之后的条目 → 当前索引不变
    """
    conversations = st.session_state.conversations
    if not (0 <= idx < len(conversations)):
        logger.error("删除对话参数越界：idx=%s，共 %d 条", idx, len(conversations))
        return
    if len(conversations) <= 1:
        logger.warning("拒绝删除：仅剩 1 条对话，至少保留一条")
        return

    current_idx = st.session_state.current_conversation_index
    removed = conversations.pop(idx)
    # MySQL 主存储时同步删除数据库中的对应对话（JSON 降级模式下为空操作）
    delete_conversation_from_db(removed.get("id", ""))

    if idx < current_idx:
        # 删除的是当前对话之前的条目：当前对话随列表前移一位
        st.session_state.current_conversation_index = current_idx - 1
    elif idx == current_idx:
        # 删除的是当前对话：pop 后原位置即"后一个"对话；
        # 若删的是最后一条，则退到新的最后一条（即前一条）
        new_idx = min(idx, len(conversations) - 1)
        st.session_state.current_conversation_index = new_idx
        st.session_state.messages = conversations[new_idx].get("messages", [])
        st.session_state.conversation_id = conversations[new_idx].get("id", "")
    # idx > current_idx：当前对话位置不变，无需处理

    st.session_state._conv_deleted_name = removed.get("name", "")
    save_session_to_file(st.session_state)
    st.session_state._need_rerun = True  # app.py 顶层消费后执行真正的 st.rerun()


# ====================== 侧边栏渲染 ======================
def render_sidebar() -> None:
    """渲染侧边栏所有功能"""
    # 文档上传"处理完成"标记：各处理分支只负责置 True + st.rerun()，
    # 清空上传器统一在这里完成。新版 Streamlit 禁止给 file_uploader 的
    # key 直接赋值（即使实例化之前也会报 StreamlitValueAssignmentNotAllowedError），
    # 改用 del 删除 key：widget 实例化时会重新创建该 key，等价于清空上传器。
    if st.session_state.pop("_doc_uploader_done", False):
        if "doc_uploader" in st.session_state:
            del st.session_state["doc_uploader"]
    # 历史对话删除成功提示：_delete_conversation 回调里置标志，本处渲染 toast
    if deleted_name := st.session_state.pop("_conv_deleted_name", None):
        toast(f"已删除对话「{deleted_name}」", icon="🗑️")
    with st.sidebar:
        st.markdown("## 🤖 AI 智能助手")
        st.divider()

        # 主题切换
        st.toggle("🌙 夜间模式", key="dark_mode")

        # API配置面板（提供方切换 + 动态配置字段 + 模型选择）
        with st.expander("🔑 API 配置", expanded=True):
            if MODELS_AVAILABLE:
                _render_provider_config()
            else:
                _render_legacy_api_config()

        # 高级设置
        with st.expander("⚙️ 高级参数设置", expanded=st.session_state.show_settings):
            new_prompt = st.text_area(
                "系统提示词",
                value=st.session_state.system_prompt,
                height=160,
                help="自定义AI角色和回答风格"
            )
            if st.button("💾 保存提示词", use_container_width=True):
                st.session_state.system_prompt = new_prompt
                save_session_to_file(st.session_state)
                st.success("提示词更新成功！")
                st.rerun()

            st.session_state.temperature = st.slider(
                "创造性 (Temperature)", 0.0, 2.0, st.session_state.temperature, 0.1
            )
            st.session_state.max_tokens = st.number_input(
                "最大回复长度", 100, 8000, st.session_state.max_tokens, 100
            )
            st.session_state.max_context_msg = st.number_input(
                "最大上下文消息数",  10, 100, st.session_state.max_context_msg, 5,
                help="限制上下文长度，防止卡顿、超限"
            )

        # 文档管理（RAG）
        with st.expander("📄 文档管理"):
            if not RAG_AVAILABLE:
                st.error(
                    "⚠️ RAG 依赖未安装，请执行：\n```\n"
                    "pip install langchain langchain-community "
                    "langchain-text-splitters chromadb pypdf\n```"
                )
            else:
                # RAG 状态获取带异常保护：向量库损坏时不拖垮整个侧边栏
                try:
                    ready, status_msg, provider, doc_count = get_rag_status()
                except Exception as e:
                    logger.error("RAG 状态获取失败：%s", e)
                    ready, status_msg, provider, doc_count = False, f"状态获取失败：{str(e)}", None, 0
                if provider:
                    st.caption(f"当前嵌入方式：{'OpenAI text-embedding-3-small' if provider == 'openai' else '本地模型（免密钥）'}")
                if status_msg:
                    st.caption(status_msg)

                st.toggle("🔍 提问时检索文档", key="rag_enabled",
                          disabled=st.session_state.cache_enabled,
                          on_change=_on_rag_enabled_change,
                          help="开启后，提问会先从已上传文档中检索相关内容再交给AI回答；"
                               "开启时自动关闭 Redis 缓存，且缓存开启期间本开关被禁用")

                upload_file = st.file_uploader(
                    "上传文档（PDF / TXT / MD）",
                    type=["pdf", "txt", "md"],
                    disabled=not ready,
                    help="上传后自动切分并向量化；支持 PDF、TXT、Markdown 格式",
                    key="doc_uploader",  # 固定 key；清空动作在 render_sidebar 顶部完成（widget 实例化之前）
                )
                if upload_file is not None:
                    # 重复检查：同名文件已入库时跳过，防止创建重复记录
                    existing_names = set()
                    if ready:
                        try:
                            existing_names = {doc["file_name"] for doc in get_document_list()}
                        except Exception as e:
                            logger.error("读取文档列表失败：%s", e)
                    if upload_file.name in existing_names:
                        st.warning(f"⚠️ 《{upload_file.name}》已存在，已跳过上传（如需更新请先删除原文档）")
                        st.session_state["_doc_uploader_done"] = True
                        st.rerun()
                    # 10MB 上限保护：超大文件会导致内存与页面卡死
                    if upload_file.size > AppConfig.MAX_UPLOAD_SIZE:
                        st.error(f"❌ 文件超过 {AppConfig.MAX_UPLOAD_SIZE // (1024 * 1024)}MB 限制，无法上传")
                        st.session_state["_doc_uploader_done"] = True
                        st.rerun()
                    try:
                        with st.spinner(f"正在处理《{upload_file.name}》..."):
                            chunks = load_document(upload_file)
                            if not chunks:
                                st.error(f"❌ 《{upload_file.name}》未解析出任何内容，请检查文件格式")
                                st.session_state["_doc_uploader_done"] = True
                                st.rerun()
                            # 加载与向量化分开捕获：定位失败环节更准确
                            try:
                                n_added = add_to_vectorstore(chunks)
                            except Exception as e:
                                logger.error("文档向量化失败：%s", e)
                                st.error(f"❌ 文档向量化失败：{str(e)}")
                                st.session_state["_doc_uploader_done"] = True
                                st.rerun()
                        st.success(f"✅ 《{upload_file.name}》入库成功（{n_added} 个片段）")
                        st.session_state["_doc_uploader_done"] = True
                        st.rerun()
                    except Exception as e:
                        logger.error("文档加载失败：%s", e)
                        st.error(f"❌ 文档加载失败：{str(e)}")
                        st.session_state["_doc_uploader_done"] = True
                        st.rerun()

                # 文档列表与统计
                doc_list = []
                if ready:
                    try:
                        doc_list = get_document_list()
                    except Exception as e:
                        st.error(f"❌ 读取文档列表失败：{str(e)}")
                c1, c2 = st.columns(2)
                c1.metric("已上传文档", len(doc_list))
                c2.metric("向量片段总数", doc_count)

                for doc in doc_list:
                    col_info, col_del = st.columns([5, 1])
                    with col_info:
                        st.caption(f"📄 {doc['file_name']}（{doc['chunks']} 个片段）")
                    with col_del:
                        if st.button("🗑️", key=f"del_doc_{doc['doc_id']}",
                                     help="删除此文档", use_container_width=True):
                            try:
                                n_deleted = delete_document(doc["doc_id"])
                            except Exception as e:
                                # 删除异常：记录日志并向界面显示具体错误信息
                                logger.error("删除文档失败：%s", e)
                                st.error(f"❌ 删除《{doc['file_name']}》失败：{str(e)}")
                            else:
                                # 删除成功后强制刷新向量库缓存，再 rerun 重新加载文档列表
                                invalidate_collection_cache()
                                if n_deleted > 0:
                                    st.success(f"✅ 已删除《{doc['file_name']}》（{n_deleted} 个片段）")
                                else:
                                    st.warning(f"⚠️ 未在向量库中找到《{doc['file_name']}》的片段，可能已被删除")
                                st.rerun()

                if st.button("🧹 清空所有文档", use_container_width=True,
                             disabled=doc_count == 0):
                    try:
                        clear_all()
                    except Exception as e:
                        logger.error("清空文档失败：%s", e)
                        st.error(f"❌ 清空失败：{str(e)}")
                    else:
                        st.success("✅ 已清空所有文档")
                        st.rerun()

        # 工具调用（Function Calling）
        with st.expander("🔧 工具调用"):
            if not TOOLS_AVAILABLE:
                st.error("⚠️ 工具模块加载失败，请检查 core/tools.py")
            else:
                st.toggle("🔧 启用工具调用", key="tools_enabled",
                          disabled=st.session_state.cache_enabled,
                          on_change=_on_tools_enabled_change,
                          help="开启后，AI 可调用时间、计算器、网络搜索等工具，回答需要实时信息或精确计算的问题；"
                               "开启时自动关闭 Redis 缓存，且缓存开启期间本开关被禁用")
                st.caption("可用工具：" + "、".join(get_tool_names()))
                if MODELS_AVAILABLE:
                    model_cfg = get_model_config(
                        st.session_state.current_model,
                        provider_key=st.session_state.provider)
                    if not model_cfg["supports_tools"]:
                        st.caption(
                            f"⚠️ 当前模型 {st.session_state.current_model} 标记为不支持工具调用："
                            "提问时仍会先尝试，请求失败后自动降级并在回答中明确提示"
                        )

                # Agent 模式（LangGraph）：在工具调用之上提供规划/反思/自主结束，
                # 替代固定 3 轮的工具循环；依赖工具开关，关闭工具时自动关闭
                if AGENT_AVAILABLE:
                    st.toggle("🧠 Agent 模式（规划 + 反思）", key="agent_mode_enabled",
                              disabled=not st.session_state.tools_enabled,
                              help="开启后使用 LangGraph Agent 替代固定 3 轮的工具循环："
                                   "AI 自主规划工具调用步骤、反思工具结果，并自主决定何时结束。"
                                   "需先开启「启用工具调用」。")
                    if st.session_state.agent_mode_enabled:
                        st.slider("最大规划步数", 1, 10,
                                  value=st.session_state.agent_max_steps,
                                  key="agent_max_steps",
                                  help="Agent 最多进行多少轮模型决策；达到上限时返回已收集的信息")
                else:
                    st.caption("⚠️ LangGraph Agent 未安装（pip install langgraph langchain-openai）")

        # 缓存设置（Redis）
        with st.expander("📦 缓存设置"):
            if not CACHE_AVAILABLE:
                st.error("⚠️ 缓存模块加载失败，请检查 core/cache.py")
            else:
                st.toggle("📦 启用Redis缓存", key="cache_enabled",
                          on_change=_on_cache_enabled_change,
                          help="相同问题命中缓存时直接返回，减少API调用成本；Redis不可用时自动跳过。"
                               "开启时自动关闭「提问时检索文档」与「启用工具调用」")
                status_text, status_level = get_cache_status()
                if status_level == "ok":
                    st.caption("✅ " + status_text)
                else:
                    st.caption("⚠️ " + status_text)
                st.caption(f"缓存有效期：{CACHE_TTL // 60} 分钟")
                if st.button("🧹 清空缓存", use_container_width=True):
                    n = clear_cache()
                    st.success(f"✅ 已清空 {n} 条缓存")

        # 数据管理
        with st.expander("📁 对话数据管理"):
            if st.button("📤 导出当前对话", use_container_width=True):
                export_data = {
                    "conversation_id": st.session_state.conversation_id,
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "system_prompt": st.session_state.system_prompt,
                    "messages": st.session_state.messages
                }
                st.download_button(
                    "📥 下载JSON文件",
                    data=json.dumps(export_data, ensure_ascii=False, indent=2),
                    file_name=f"对话记录_{st.session_state.conversation_id}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="json_export"
                )
                text_content = "\n\n".join([f"{m['role']}：{m['content']}" for m in st.session_state.messages])
                st.download_button(
                    "📥 下载TXT文件",
                    data=text_content,
                    file_name=f"对话记录_{st.session_state.conversation_id}.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="txt_export"
                )
                # Markdown 导出：带标题、元信息与角色分节，便于阅读与存档
                md_lines = [
                    "# 对话记录\n",
                    f"- 对话ID：{st.session_state.conversation_id}",
                    f"- 导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"- 系统提示词：{st.session_state.system_prompt}\n",
                ]
                for m in st.session_state.messages:
                    role = "👤 用户" if m["role"] == "user" else "🤖 AI"
                    md_lines.append(f"## {role}\n\n{m['content']}\n")
                md_content = "\n".join(md_lines)
                st.download_button(
                    "📥 下载Markdown文件",
                    data=md_content,
                    file_name=f"对话记录_{st.session_state.conversation_id}.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="md_export"
                )

            upload_file = st.file_uploader("导入JSON对话", type="json")
            if upload_file:
                # 10MB 上限保护：超大文件会导致内存与页面卡死
                if upload_file.size > AppConfig.MAX_UPLOAD_SIZE:
                    st.error(f"❌ 文件超过 {AppConfig.MAX_UPLOAD_SIZE // (1024 * 1024)}MB 限制，无法导入")
                else:
                    try:
                        import_data = json.load(upload_file)
                        new_conv_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                        new_conv = {
                            "id": new_conv_id,
                            "name": f"导入对话_{new_conv_id}",
                            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "messages": import_data.get("messages", [])
                        }
                        st.session_state.conversations.append(new_conv)
                        st.session_state.current_conversation_index = len(st.session_state.conversations) - 1
                        st.session_state.messages = new_conv["messages"]
                        save_session_to_file(st.session_state)
                        st.success("✅ 对话导入成功！")
                        st.rerun()
                    except Exception as e:
                        logger.error("对话导入失败：%s", e)
                        st.error(f"❌ 导入失败：{str(e)}")

        # ====================== 对话管理区域 (重新设计，对齐美观) ======================
        st.markdown("### 📚 对话管理")

        # 三个操作按钮：新建、清空当前、全部清空
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➕ 新建", use_container_width=True):
                new_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_conv = {
                    "id": new_id,
                    "name": f"对话_{new_id}",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "messages": []
                }
                st.session_state.conversations.append(new_conv)
                st.session_state.current_conversation_index = len(st.session_state.conversations) - 1
                st.session_state.messages = []
                st.session_state.conversation_id = new_id
                save_session_to_file(st.session_state)
                st.rerun()
        with col2:
            if st.button("🗑️ 清空当前", use_container_width=True):
                st.session_state.messages = []
                conv_idx = st.session_state.current_conversation_index
                if st.session_state.conversations and 0 <= conv_idx < len(st.session_state.conversations):
                    st.session_state.conversations[conv_idx]["messages"] = []
                    # MySQL 主存储时同步清空数据库中的该对话消息（JSON 降级模式下为空操作）
                    clear_conversation_messages_from_db(st.session_state.conversations[conv_idx]["id"])
                else:
                    # 防御：索引越界时直接索引会崩溃，导致后面的 st.rerun() 不执行
                    logger.error("清空当前：索引 %s 越界（共 %d 条），跳过对话内消息清空",
                                 conv_idx, len(st.session_state.conversations))
                save_session_to_file(st.session_state)
                st.success("✅ 已清空当前对话")
                st.rerun()
        with col3:
            if st.button("🧹 全部清空", use_container_width=True):
                # MySQL 主存储时先清空数据库中的全部对话（JSON 降级模式下为空操作）；
                # 随后 save_session_to_file 会把新的默认对话同步回数据库
                clear_all_conversations_from_db()
                st.session_state.conversations = [{
                    "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "name": "默认对话",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "messages": []
                }]
                st.session_state.current_conversation_index = 0
                st.session_state.messages = []
                save_session_to_file(st.session_state)
                st.rerun()

        st.markdown("#### 📜 历史对话")

        # 对话搜索框：按名称/创建时间/消息内容关键词过滤历史对话
        search_keyword = st.text_input(
            "🔍 搜索对话", key="conv_search",
            placeholder="按名称或内容搜索...",
        ).strip().lower()

        # ========== 优化后的历史对话列表 ==========
        # 使用容器让列表更整齐
        history_container = st.container()

        # 预计算匹配集合：搜索时只渲染命中的对话
        matched_ids = None
        if search_keyword:
            matched_ids = set()
            for conv in st.session_state.conversations:
                haystack = " ".join([
                    conv.get("name", ""),
                    conv.get("created_at", ""),
                    " ".join(m.get("content", "") for m in conv.get("messages", [])),
                ]).lower()
                if search_keyword in haystack:
                    matched_ids.add(conv["id"])

        shown_count = 0
        with history_container:
            for idx, conv in enumerate(st.session_state.conversations):
                if matched_ids is not None and conv["id"] not in matched_ids:
                    continue
                shown_count += 1
                is_current = (idx == st.session_state.current_conversation_index)

                # 使用两列布局：左侧名称（可编辑/切换），右侧删除按钮
                # 为了让删除按钮和命名框对齐，采用自定义比例
                col_name, col_del = st.columns([5, 1])

                with col_name:
                    if is_current:
                        # 当前对话：显示可编辑输入框（更直观修改名称）
                        new_name = st.text_input(
                            "",
                            value=conv["name"],
                            # key 用对话 id 而非索引：删除后索引位移不会串位
                            key=f"conv_name_edit_{conv.get('id', idx)}",
                            label_visibility="collapsed",
                            placeholder="对话名称"
                        )
                        if new_name != conv["name"]:
                            st.session_state.conversations[idx]["name"] = new_name
                            save_session_to_file(st.session_state)
                            st.rerun()
                    else:
                        # 非当前对话：点击名称切换对话
                        display_name = conv["name"][:22] + "..." if len(conv["name"]) > 22 else conv["name"]
                        if st.button(
                                display_name,
                                key=f"switch_{idx}",
                                use_container_width=True,
                                help=f"创建于 {conv['created_at']}"
                        ):
                            st.session_state.current_conversation_index = idx
                            st.session_state.messages = conv["messages"]
                            st.session_state.conversation_id = conv["id"]
                            save_session_to_file(st.session_state)
                            st.rerun()

                with col_del:
                    # 仅保留删除按钮；仅剩一条对话时禁用（至少保留一条）。
                    # 删除逻辑放在 on_click 回调 _delete_conversation 中：
                    # 回调里修改状态并置位 _need_rerun 标记，由 app.py 顶层
                    # 统一执行 st.rerun()（回调内直接 rerun 会被 Streamlit
                    # 当作 no-op 并警告）。
                    del_disabled = len(st.session_state.conversations) <= 1
                    st.button(
                        "🗑️",
                        key=f"del_conv_{idx}",
                        help="删除此对话",
                        disabled=del_disabled,
                        use_container_width=True,
                        on_click=_delete_conversation,
                        args=(idx,),
                    )

                # 可选：显示创建时间的小灰字（美观）
                st.caption(f"📅 {conv['created_at']}")
                st.markdown("---")  # 轻量分隔线

            if search_keyword and shown_count == 0:
                st.caption("🔍 未找到匹配的对话")

        # 对话统计（含 Token 用量预估）
        with st.expander("📊 对话统计"):
            total = len(st.session_state.messages)
            user_cnt = sum(1 for m in st.session_state.messages if m["role"] == "user")
            ai_cnt = total - user_cnt
            # Token 用量：按消息内容粗略估算（中文约 1 token/字，其他约 4 字符/token）
            user_tokens = sum(estimate_tokens(m["content"])
                              for m in st.session_state.messages if m["role"] == "user")
            ai_tokens = sum(estimate_tokens(m["content"])
                            for m in st.session_state.messages if m["role"] != "user")
            st.metric("总消息数", total)
            c1, c2 = st.columns(2)
            c1.metric("用户提问", user_cnt)
            c2.metric("AI回复", ai_cnt)
            st.metric("预估 Token 总量", user_tokens + ai_tokens)
            c3, c4 = st.columns(2)
            c3.metric("用户 Token", user_tokens)
            c4.metric("AI Token", ai_tokens)
            st.metric("对话总数", len(st.session_state.conversations))
