"""
core 包：业务逻辑层（不依赖 Streamlit，会话状态由调用方注入）。

统一导出：配置（config）、会话持久化（session）、对话核心流程（chat），
以及自 modules/ 迁移而来的 RAG / 工具 / 缓存 / 多模型 / Agent 模块，
并集中提供 *_AVAILABLE 可用性开关——依赖缺失时应用照常运行，仅对应功能不可用。
"""

from core.chat import call_ai_api_stream, create_ai_client, run_tool_loop, trim_context_messages
from core.config import AppConfig, DEFAULT_SYSTEM_PROMPT, MODEL_LIST, QUICK_QUESTIONS
from core.session import init_session_state, load_session_from_file, save_session_to_file

# ====================== 文本工具模块（Token 估算等） ======================
# 纯函数工具；模块文件缺失时提供等价兜底实现，保证统计功能不挂
try:
    from core.text_utils import estimate_tokens
except ImportError:
    import re

    def estimate_tokens(text: str) -> int:
        """粗略估算文本 Token 数（兜底实现，正常走 core/text_utils.py）"""
        if not text:
            return 0
        cjk_chars = len(re.findall(r"[一-鿿　-〿＀-￯]", text))
        other_chars = len(text) - cjk_chars
        return cjk_chars + other_chars // 4

# ====================== RAG 模块 ======================
# 依赖未安装时自动降级：应用照常运行，仅文档管理功能不可用
try:
    from core.rag import (
        add_to_vectorstore,
        clear_all,
        delete_document,
        get_document_list,
        get_rag_status,
        invalidate_collection_cache,
        load_document,
        search,
    )
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ====================== 工具模块 ======================
# 同样自动降级：工具不可用时应用照常运行
try:
    from core.tools import (
        build_tools_ack,
        build_tools_directive,
        execute_tool,
        get_available_tools,
        get_tool_names,
        has_tool_denial,
    )
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False

# ====================== LangGraph Agent 模块 ======================
# Agent 依赖 langgraph / langchain-openai；未安装时自动降级为普通工具循环模式
try:
    from core.agent import run_agent
    AGENT_AVAILABLE = True
except ImportError:
    AGENT_AVAILABLE = False

# ====================== 缓存模块 ======================
try:
    from core.cache import (
        CACHE_TTL,
        clear_cache,
        get_cache_status,
        get_cached_response,
        set_cached_response,
    )
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# ====================== 多模型模块 ======================
try:
    from core.models import (
        extract_reasoning,
        get_model_config,
        get_provider_config,
        list_ollama_models,
        list_providers,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# ====================== MySQL 会话存储模块 ======================
# pymysql 缺失或连接失败时自动降级为本地 JSON，模块本身可无条件导入
from core.db import (
    USER_ID,
    clear_all_conversations_from_db,
    clear_conversation_messages_from_db,
    db_available,
    delete_conversation_from_db,
)

# ====================== 对外导出的公共接口 ======================
# 显式声明再导出（含 *_AVAILABLE 可用性开关），供 ui/ 与 app.py 统一从这里导入
__all__ = [
    # config
    "AppConfig", "DEFAULT_SYSTEM_PROMPT", "MODEL_LIST", "QUICK_QUESTIONS",
    # session
    "init_session_state", "load_session_from_file", "save_session_to_file",
    # chat
    "call_ai_api_stream", "create_ai_client", "run_tool_loop", "trim_context_messages",
    # text_utils
    "estimate_tokens",
    # rag
    "RAG_AVAILABLE", "add_to_vectorstore", "clear_all", "delete_document",
    "get_document_list", "get_rag_status", "invalidate_collection_cache",
    "load_document", "search",
    # tools
    "TOOLS_AVAILABLE", "build_tools_ack", "build_tools_directive", "execute_tool",
    "get_available_tools", "get_tool_names", "has_tool_denial",
    # agent
    "AGENT_AVAILABLE", "run_agent",
    # cache
    "CACHE_AVAILABLE", "CACHE_TTL", "clear_cache", "get_cache_status",
    "get_cached_response", "set_cached_response",
    # models
    "MODELS_AVAILABLE", "extract_reasoning", "get_model_config",
    "get_provider_config", "list_ollama_models", "list_providers",
    # db
    "USER_ID", "db_available", "delete_conversation_from_db",
    "clear_conversation_messages_from_db", "clear_all_conversations_from_db",
]
