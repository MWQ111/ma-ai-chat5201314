"""会话状态管理：持久化与初始化（不依赖 Streamlit，session_state 由调用方注入）

存储策略：
- MySQL（core/db.py，SQLAlchemy ORM）为主存储 —— 界面（Streamlit）与 app_api 共用同一数据；
- 本地 JSON（session_data/session_cache.json）保留为降级备份 —— MySQL 不可用时自动回退，
  界面/API 数据仍可通过各自的读写路径独立工作。
"""

import json
import logging
import os
from datetime import datetime

from core.config import AppConfig, DEFAULT_SYSTEM_PROMPT
from core.db import USER_ID, SessionDB, db_available, sync_session_to_db

logger = logging.getLogger("ai_chat.core.session")

# 提供方/模型组合校验依赖多模型模块；缺失时跳过校验，不影响启动（与重构前行为一致）
try:
    from core.models import get_provider_config
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


def _fmt_time(value) -> str:
    """把 ORM/MySQL 返回的时间统一为界面展示用的字符串"""
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _load_conversations_from_db() -> list | None:
    """从 MySQL 读取全部对话并转为 UI 结构（参考 app_api.py 的 ORM → dict 转换）

    返回形如 [{id, name, created_at, messages: [{role, content}]}]；
    MySQL 不可用、库为空或读取失败时返回 None（上层回退本地 JSON）。
    """
    if not db_available():
        return None
    dbs = SessionDB()
    try:
        conv_rows = dbs.get_conversations(USER_ID)
        if not conv_rows:
            return None
        result = []
        for conv in conv_rows:
            msgs = dbs.get_messages(conv.id)
            result.append({
                "id": conv.id,
                "name": conv.title or f"对话 {conv.id}",
                "created_at": _fmt_time(conv.created_at),
                "messages": [
                    {"role": m.role, "content": m.content}
                    for m in msgs
                    if m.role in ("user", "assistant")
                ],
            })
        return result
    except Exception as e:
        logger.warning("MySQL 读取对话失败，回退本地 JSON：%s", e)
        return None
    finally:
        dbs.close()


def _sync_conversations_to_db(conversations) -> None:
    """把 UI 结构中的对话列表增量同步到 MySQL

    UI 约定用 name 字段、数据库用 title 字段；直接透传原始对话 dict 给
    core.db 层（db 层自行做 name→title 映射并过滤消息），这样 db 层为
    本地新建对话插入主键后能回写到原 dict（下次即可走增量更新，避免重复插入）。
    """
    if not db_available():
        return
    sync_session_to_db(USER_ID, conversations)


def save_session_to_file(session_state) -> None:
    """持久化会话：MySQL 为主存储（ORM 增量同步），JSON 同时保留为降级备份

    - MySQL 可用 → 写 MySQL + 写 JSON 备份；
    - MySQL 不可用 / 写入失败 → 只写 JSON（降级路径，行为与改造前一致）。

    Args:
        session_state: 会话状态对象（st.session_state，由调用方注入）
    """
    if db_available():
        try:
            _sync_conversations_to_db(session_state.conversations)
        except Exception as e:
            logger.warning("MySQL 写入失败，降级到本地 JSON：%s", e)
    _save_session_to_file(session_state)


def _save_session_to_file(session_state) -> None:
    """写本地 JSON 文件（原有实现，作为 MySQL 不可用时的降级存储与备份）

    原子写入：先写临时文件再替换，避免中断损坏；带格式版本号；
    任何写入失败只记录日志，不影响应用继续运行。
    """
    try:
        save_dir = AppConfig.SESSION_FILE_DIR
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, AppConfig.SESSION_FILE_NAME)
        tmp_path = save_path + ".tmp"
        save_data = {
            "version": AppConfig.SESSION_FILE_VERSION,  # 文件格式版本号（旧文件无此字段也兼容）
            "conversations": session_state.conversations,
            "system_prompt": session_state.system_prompt,
            "temperature": session_state.temperature,
            "max_tokens": session_state.max_tokens,
            "current_conversation_index": session_state.current_conversation_index,
            "provider": session_state.get("provider", "deepseek"),
            "current_model": session_state.get("current_model", "deepseek-chat"),
            "agent_mode_enabled": session_state.get("agent_mode_enabled", False),
            "agent_max_steps": session_state.get("agent_max_steps", 5),
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, save_path)  # 原子替换：写入中断也不会损坏原文件
        logger.info("会话已保存到 %s（对话 %d 条，当前索引 %s）",
                    save_path, len(save_data["conversations"]),
                    save_data["current_conversation_index"])
    except Exception as e:
        logger.warning("会话保存失败：%s", e)


def load_session_from_file() -> dict | None:
    """从本地 JSON 文件加载持久化会话数据（降级存储；文件缺失或损坏时返回 None）"""
    save_path = os.path.join(AppConfig.SESSION_FILE_DIR, AppConfig.SESSION_FILE_NAME)
    if os.path.exists(save_path):
        try:
            with open(save_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("会话文件读取失败，回退默认值：%s", e)
            return None
    return None


def init_session_state(session_state) -> None:
    """初始化所有会话状态，支持持久化加载

    加载优先级（数据一致性规则，MySQL 为主存储）：
    1. MySQL 有数据 → 以 MySQL 为准（对话列表来自 ORM；提示词/温度等设置类字段仍读 JSON）；
    2. MySQL 空但 JSON 有历史 → 从 JSON 加载，并尝试同步进 MySQL（空库迁入）；
    3. 两者都不可用/为空 → 使用默认值，首次保存时再写入存储。

    Args:
        session_state: 会话状态对象（st.session_state，由调用方注入，
            保证本模块不依赖 Streamlit）
    """
    # ===== 存储加载：MySQL（主）→ JSON（降级）=====
    # 只在“会话首次初始化”（conversations 尚未注入 session_state）时读取持久化
    # 存储；此后的 rerun 均沿用内存状态，不再重复查 MySQL / 读 JSON 文件
    #（旧实现每次 rerun 都会重新 load_session_from_file()，造成无效 I/O）。
    cache_data = None
    if "conversations" not in session_state:
        db_convs = _load_conversations_from_db()
        if db_convs is not None:
            # MySQL 有数据：对话列表以 MySQL 为准，其余设置字段合并 JSON（缺失则用默认值）
            file_cache = load_session_from_file()
            merged = dict(file_cache) if isinstance(file_cache, dict) else {}
            merged["conversations"] = db_convs
            merged["current_conversation_index"] = 0
            cache_data = merged
        else:
            # MySQL 不可用 / 库为空：降级到本地 JSON
            cache_data = load_session_from_file()
            # 库空但 JSON 有历史：迁入 MySQL，之后界面与 API 共用同一份数据
            if cache_data and db_available():
                try:
                    _sync_conversations_to_db(cache_data.get("conversations") or [])
                except Exception as e:
                    logger.warning("JSON → MySQL 迁入失败（不影响本次运行）：%s", e)

    if "dark_mode" not in session_state:
        session_state.dark_mode = AppConfig.DEFAULT_DARK_MODE

    # ===== 多模型配置（提供方 / 密钥 / 接口地址 / 当前模型） =====
    if "provider" not in session_state:
        session_state.provider = "deepseek" if not cache_data else cache_data.get("provider", "deepseek")

    if "api_keys" not in session_state:
        # 各提供方密钥：从环境变量预填；界面修改的值仅保存在内存中（不写入磁盘）
        session_state.api_keys = {
            "deepseek": os.environ.get("DEEPSEEK_API_KEY", ""),
            "openai": os.environ.get("OPENAI_API_KEY", ""),
            "ollama": "",  # 本地模型无需密钥
        }

    if "base_urls" not in session_state:
        # 各提供方接口地址（OpenAI 兼容格式）
        session_state.base_urls = {
            "deepseek": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "openai": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "ollama": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        }

    if "current_model" not in session_state:
        session_state.current_model = "deepseek-chat" if not cache_data else cache_data.get("current_model", "deepseek-chat")

    # ===== 会话数据安全加载：校验对话列表与索引有效性（修复 IndexError） =====
    # 持久化数据可能被手工修改或由旧版本写出，导致 conversations 为空、
    # 索引越界或类型错误；先校验再取值，任何异常情况都回退到安全值。
    cache_convs = cache_data.get("conversations", []) if cache_data else []
    cache_idx = cache_data.get("current_conversation_index", 0) if cache_data else 0
    convs_valid = (isinstance(cache_convs, list) and len(cache_convs) > 0
                   and all(isinstance(c, dict) for c in cache_convs))
    idx_valid = convs_valid and isinstance(cache_idx, int) and 0 <= cache_idx < len(cache_convs)
    safe_idx = cache_idx if idx_valid else 0  # 索引越界时钳制到 0（保留历史数据）

    if "messages" not in session_state:
        if convs_valid:
            msgs = cache_convs[safe_idx].get("messages", [])
            session_state.messages = msgs if isinstance(msgs, list) else []
        else:
            # 对话列表为空或损坏：安全回退为空列表（对话结构在下方自动重建）
            session_state.messages = []

    if "system_prompt" not in session_state:
        session_state.system_prompt = DEFAULT_SYSTEM_PROMPT if not cache_data else cache_data.get("system_prompt",
                                                                                                 DEFAULT_SYSTEM_PROMPT)

    if "temperature" not in session_state:
        session_state.temperature = (
            AppConfig.DEFAULT_TEMPERATURE
            if not cache_data
            else cache_data.get("temperature", AppConfig.DEFAULT_TEMPERATURE)
        )

    if "max_tokens" not in session_state:
        session_state.max_tokens = (
            AppConfig.DEFAULT_MAX_TOKENS
            if not cache_data
            else cache_data.get("max_tokens", AppConfig.DEFAULT_MAX_TOKENS)
        )

    if "conversation_id" not in session_state:
        session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "conversations" not in session_state:
        if convs_valid:
            session_state.conversations = cache_convs
        else:
            # 对话列表为空或损坏：自动重建默认对话结构
            session_state.conversations = [
                {
                    "id": session_state.conversation_id,
                    "name": f"对话 {session_state.conversation_id}",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "messages": []
                }
            ]

    if "current_conversation_index" not in session_state:
        # 索引越界时钳制到 0；对话列表已重建时索引置 0
        session_state.current_conversation_index = safe_idx if convs_valid else 0

    if "show_settings" not in session_state:
        session_state.show_settings = False

    if "max_context_msg" not in session_state:
        session_state.max_context_msg = AppConfig.MAX_CONTEXT_MSGS

    if "rag_enabled" not in session_state:
        session_state.rag_enabled = True

    if "tools_enabled" not in session_state:
        session_state.tools_enabled = True

    if "cache_enabled" not in session_state:
        session_state.cache_enabled = True

    # ===== Agent 模式（LangGraph 规划 + 反思 + 自主结束，需先开启工具调用） =====
    if "agent_mode_enabled" not in session_state:
        session_state.agent_mode_enabled = (
            bool(cache_data.get("agent_mode_enabled", False)) if cache_data else False
        )

    if "agent_max_steps" not in session_state:
        session_state.agent_max_steps = (
            int(cache_data.get("agent_max_steps", 5)) if cache_data else 5
        )

    # 联动约束兜底：缓存开启时 RAG 与工具调用必须关闭。
    # 正常交互由 on_change 回调维护，这里兜底处理默认值冲突与旧版持久化数据。
    if session_state.cache_enabled:
        session_state.rag_enabled = False
        session_state.tools_enabled = False

    # ===== 校验提供方/模型组合是否有效（旧版持久化数据或模型列表变更后自动回退默认） =====
    if MODELS_AVAILABLE:
        try:
            cfg = get_provider_config(session_state.provider)
            if cfg is None:
                session_state.provider = "deepseek"
                cfg = get_provider_config("deepseek")
            if cfg["models"] and session_state.current_model not in cfg["models"]:
                session_state.current_model = cfg["default_model"]
        except Exception:
            pass  # 校验失败不影响启动，保持原值
