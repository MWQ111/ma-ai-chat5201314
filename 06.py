import streamlit as st
import os
import re
import time
import json
import uuid
import logging
from typing import Any, List, Optional, Tuple
from openai import OpenAI
from datetime import datetime

# ====================== 日志系统 ======================
# 仅在根日志器无处理器时配置（Streamlit 通常已配置好，避免重复输出）；
# 模块内统一使用 logger 记录关键路径的运行与降级信息。
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
logger = logging.getLogger("ai_chat")

# ====================== 环境变量加载 ======================
# 使用 find_dotenv(usecwd=True) 定位当前工作目录下的 .env 文件
# （以 `streamlit run 06.py` 启动时工作目录即项目根目录，比默认的
# 按调用栈向上查找更可靠；文件不存在时静默跳过）。
# 必须放在功能模块导入之前：cache.py 等在导入时会读取环境变量。
try:
    from dotenv import load_dotenv, find_dotenv
    # find_dotenv 自动向上查找 .env；参数异常/未安装时整体降级，不影响启动
    _dotenv_path = find_dotenv(usecwd=True, raise_error_if_not_found=False)
    if _dotenv_path:
        load_dotenv(_dotenv_path)
    else:
        load_dotenv()
    logger.info("已加载环境变量文件：%s", _dotenv_path or ".env（默认路径）")
except Exception as e:
    logger.warning("环境变量文件加载跳过：%s", e)

# ====================== RAG 模块导入 ======================
# 依赖未安装时自动降级：应用照常运行，仅文档管理功能不可用
try:
    from modules.rag_module import (
        load_document,
        add_to_vectorstore,
        search,
        get_document_list,
        delete_document,
        clear_all,
        get_rag_status,
    )
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ====================== 工具模块导入 ======================
# 同样自动降级：工具不可用时应用照常运行
try:
    from modules.tools import get_available_tools, get_tool_names, execute_tool
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False

# ====================== 缓存模块导入 ======================
try:
    from modules.cache import (
        get_cached_response, set_cached_response, clear_cache,
        get_cache_status, CACHE_TTL,
    )
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# ====================== 多模型模块导入 ======================
try:
    from modules.models import (
        get_model_config, list_providers, get_provider_config,
        list_ollama_models, extract_reasoning,
    )
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

# ====================== 全局配置 ======================
# 页面全局配置
st.set_page_config(
    page_title="马氏AI会话智能体",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/help',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "# 马氏AI智能助手\n一个基于DeepSeek的强大AI对话系统"
    }
)

# ====================== 应用配置（集中管理） ======================
class AppConfig:
    """应用级常量配置：所有可调参数集中在此，便于统一维护"""

    # ---- 会话持久化 ----
    SESSION_FILE_DIR = "./session_data"        # 会话数据目录
    SESSION_FILE_NAME = "session_cache.json"   # 会话数据文件名
    SESSION_FILE_VERSION = 2                   # 会话文件格式版本号（结构变更时递增）

    # ---- 默认会话参数 ----
    DEFAULT_DARK_MODE = True                   # 启动时默认夜间模式
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2000
    MAX_CONTEXT_MSGS = 50                      # 最大上下文消息数

    # ---- 消息与回答保护 ----
    MAX_MESSAGE_LENGTH = 10000                 # 单条用户消息最大字符数
    MAX_RENDER_MESSAGES = 50                   # 聊天区最多渲染的消息条数（虚拟滚动）
    STREAM_STALL_TIMEOUT = 30                  # 流式响应相邻数据块最大间隔（秒）
    MAX_TOOL_RESULT_CHARS = 4000               # 工具返回结果最大字符数
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024         # 上传文件最大字节数（10MB）

    # ---- 系统默认提示词 ----
    DEFAULT_SYSTEM_PROMPT = """你是一位可爱且专业的AI助理喔~。你的特点：
1. 回答要亲切友好，使用适当的语气词（呢、哦、呀）
2. 提供准确、有用的信息
3. 在不确定时坦诚说明
4. 适当使用表情符号增加亲和力
5. 回答要简洁明了，避免过于冗长"""

    # ---- 快速提问模板 ----
    QUICK_QUESTIONS = [
        "你好！请介绍一下你自己",
        "帮我写一段Python代码示例",
        "解释一下什么是人工智能",
        "给我一些学习建议"
    ]

    # ---- 兜底模型列表（仅多模型模块不可用时使用；正常情况用 modules/models.py 配置） ----
    MODEL_LIST = [
        "deepseek-chat",      #聊天模型
        "deepseek-reasoner"   #推理模型
    ]


# 向后兼容的模块级别名（其余代码仍可直接引用这些名字）
DEFAULT_SYSTEM_PROMPT = AppConfig.DEFAULT_SYSTEM_PROMPT
QUICK_QUESTIONS = AppConfig.QUICK_QUESTIONS
MODEL_LIST = AppConfig.MODEL_LIST

st.logo("./resources/gdutlogo.png")       #

# ====================== 工具函数 ======================
def save_session_to_file() -> None:
    """持久化会话数据到本地文件（原子写入：先写临时文件再替换，避免中断损坏）

    带格式版本号；任何写入失败只记录日志，不影响应用继续运行。
    """
    try:
        save_dir = AppConfig.SESSION_FILE_DIR
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, AppConfig.SESSION_FILE_NAME)
        tmp_path = save_path + ".tmp"
        save_data = {
            "version": AppConfig.SESSION_FILE_VERSION,  # 文件格式版本号（旧文件无此字段也兼容）
            "conversations": st.session_state.conversations,
            "system_prompt": st.session_state.system_prompt,
            "temperature": st.session_state.temperature,
            "max_tokens": st.session_state.max_tokens,
            "current_conversation_index": st.session_state.current_conversation_index,
            "provider": st.session_state.get("provider", "deepseek"),
            "current_model": st.session_state.get("current_model", "deepseek-chat")
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, save_path)  # 原子替换：写入中断也不会损坏原文件
    except Exception as e:
        logger.warning("会话保存失败：%s", e)


def load_session_from_file() -> Optional[dict]:
    """从本地文件加载持久化会话数据（文件缺失或损坏时返回 None）"""
    save_path = os.path.join(AppConfig.SESSION_FILE_DIR, AppConfig.SESSION_FILE_NAME)
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("会话文件读取失败，回退默认值：%s", e)
            return None
    return None


def init_session_state() -> None:
    """初始化所有会话状态，支持持久化加载"""
    cache_data = load_session_from_file()

    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = AppConfig.DEFAULT_DARK_MODE

    # ===== 多模型配置（提供方 / 密钥 / 接口地址 / 当前模型） =====
    if "provider" not in st.session_state:
        st.session_state.provider = "deepseek" if not cache_data else cache_data.get("provider", "deepseek")

    if "api_keys" not in st.session_state:
        # 各提供方密钥：从环境变量预填；界面修改的值仅保存在内存中（不写入磁盘）
        st.session_state.api_keys = {
            "deepseek": os.environ.get("DEEPSEEK_API_KEY", ""),
            "openai": os.environ.get("OPENAI_API_KEY", ""),
            "ollama": "",  # 本地模型无需密钥
        }

    if "base_urls" not in st.session_state:
        # 各提供方接口地址（OpenAI 兼容格式）
        st.session_state.base_urls = {
            "deepseek": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "openai": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "ollama": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        }

    if "current_model" not in st.session_state:
        st.session_state.current_model = "deepseek-chat" if not cache_data else cache_data.get("current_model", "deepseek-chat")

    # ===== 会话数据安全加载：校验对话列表与索引有效性（修复 IndexError） =====
    # 持久化文件可能被手工修改或由旧版本写出，导致 conversations 为空、
    # 索引越界或类型错误；先校验再取值，任何异常情况都回退到安全值。
    cache_convs = cache_data.get("conversations", []) if cache_data else []
    cache_idx = cache_data.get("current_conversation_index", 0) if cache_data else 0
    convs_valid = (isinstance(cache_convs, list) and len(cache_convs) > 0
                   and all(isinstance(c, dict) for c in cache_convs))
    idx_valid = convs_valid and isinstance(cache_idx, int) and 0 <= cache_idx < len(cache_convs)
    safe_idx = cache_idx if idx_valid else 0  # 索引越界时钳制到 0（保留历史数据）

    if "messages" not in st.session_state:
        if convs_valid:
            msgs = cache_convs[safe_idx].get("messages", [])
            st.session_state.messages = msgs if isinstance(msgs, list) else []
        else:
            # 对话列表为空或损坏：安全回退为空列表（对话结构在下方自动重建）
            st.session_state.messages = []

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT if not cache_data else cache_data.get("system_prompt",
                                                                                                     DEFAULT_SYSTEM_PROMPT)

    if "temperature" not in st.session_state:
        st.session_state.temperature = AppConfig.DEFAULT_TEMPERATURE if not cache_data else cache_data.get("temperature", AppConfig.DEFAULT_TEMPERATURE)

    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = AppConfig.DEFAULT_MAX_TOKENS if not cache_data else cache_data.get("max_tokens", AppConfig.DEFAULT_MAX_TOKENS)

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "conversations" not in st.session_state:
        if convs_valid:
            st.session_state.conversations = cache_convs
        else:
            # 对话列表为空或损坏：自动重建默认对话结构
            st.session_state.conversations = [
                {
                    "id": st.session_state.conversation_id,
                    "name": f"对话 {st.session_state.conversation_id}",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "messages": []
                }
            ]

    if "current_conversation_index" not in st.session_state:
        # 索引越界时钳制到 0；对话列表已重建时索引置 0
        st.session_state.current_conversation_index = safe_idx if convs_valid else 0

    if "show_settings" not in st.session_state:
        st.session_state.show_settings = False

    if "max_context_msg" not in st.session_state:
        st.session_state.max_context_msg = AppConfig.MAX_CONTEXT_MSGS

    if "rag_enabled" not in st.session_state:
        st.session_state.rag_enabled = True

    if "tools_enabled" not in st.session_state:
        st.session_state.tools_enabled = True

    if "cache_enabled" not in st.session_state:
        st.session_state.cache_enabled = True

    # ===== 校验提供方/模型组合是否有效（旧版持久化数据或模型列表变更后自动回退默认） =====
    if MODELS_AVAILABLE:
        try:
            cfg = get_provider_config(st.session_state.provider)
            if cfg is None:
                st.session_state.provider = "deepseek"
                cfg = get_provider_config("deepseek")
            if cfg["models"] and st.session_state.current_model not in cfg["models"]:
                st.session_state.current_model = cfg["default_model"]
        except Exception:
            pass  # 校验失败不影响启动，保持原值


init_session_state()


# ====================== 美化CSS（尽早注入，保证流式回答期间样式恒定） ======================
# 重要：本块必须在所有页面元素渲染之前注入（紧跟 init_session_state）。
# AI 流式回答期间脚本会长时间阻塞在流式循环里，若 CSS 写在脚本末尾，
# 回答完成前浏览器拿不到样式，页面会退回 Streamlit 默认主题（跟随系统
# 深浅色），出现"思考回答时是夜间模式、完成后才恢复"的闪变问题。
# 因此：① 提前注入；② 除页面与气泡外，侧边栏/按钮/输入框/折叠面板等
# 控件底色也随模式一起切换，保证深浅两种模式下界面颜色始终一致。
# ---- 颜色方案：日间/夜间两套变量完全分离，集中定义便于维护 ----
accent_color = "#667eea"    # 品牌强调色（两种模式通用：激活边框/高亮）
accent_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"  # 用户气泡/滚动条渐变
if st.session_state.dark_mode:
    # ============ 夜间模式 ============
    bg_color = "#0e1117"            # 页面/侧边栏背景
    surface_color = "#1a1d23"       # 卡片/控件底色（气泡、输入框、折叠面板、指标卡、下拉列表）
    hover_color = "#232833"         # 按钮悬停底色
    text_color = "#e8edf5"          # 正文/标题文字
    muted_color = "#9aa3b2"         # 次要文字（caption、占位提示、时间戳）
    border_color = "#2a3040"        # 控件描边/分割线/滚动条轨道
    input_bg = "#262a33"            # 输入类控件底色（打字框/文本框/选择框，略浅于卡片底）
    input_border = "#2d333b"        # 输入类控件描边
    code_bg = "#161b22"             # 代码底色
    alert_bg = "#20242e"            # 提示框底色
else:
    # ============ 日间模式 ============
    bg_color = "#f5f7fb"            # 页面/侧边栏背景
    surface_color = "#ffffff"       # 卡片/控件底色
    hover_color = "#e9edf5"         # 按钮悬停底色
    text_color = "#1a1a2e"          # 正文/标题文字
    muted_color = "#5a6472"         # 次要文字
    border_color = "#d8dde6"        # 控件描边/分割线/滚动条轨道
    input_bg = "#ffffff"            # 输入类控件底色
    input_border = "#dce0e8"        # 输入类控件描边
    code_bg = "#f0f2f6"             # 代码底色
    alert_bg = "#eef2f8"            # 提示框底色

st.markdown(f"""
<style>
    /* ============ ① 全局基础：页面 / 顶部工具条 / 主容器 / 字体 ============ */
    html, body, .stApp {{
        background-color: {bg_color};
    }}
    .stApp {{
        color: {text_color};
    }}
    /* 顶部工具条：与页面同底，避免切换模式后残留系统主题色 */
    header[data-testid="stHeader"] {{
        background-color: {bg_color};
    }}
    /* 主内容容器透明，透出页面底色 */
    [data-testid="stMainBlockContainer"] {{
        background-color: transparent;
    }}
    /* 主视图容器与底部栏（聊天输入框所在区域）：跟随页面底色。
       stBottom 在 1.57 中自带主题底色，是日间模式下输入框周围
       仍发黑的元凶，显式覆盖 */
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
    }}
    [data-testid="stBottom"] {{
        background-color: {bg_color};
        border-top-color: {border_color};
    }}
    /* 全局字体 */
    body, .stMarkdown, .stTextInput, .stTextArea,
    .stChatMessage, .stExpander, .stMetric, .stButton {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, "Noto Sans SC", "Microsoft YaHei", sans-serif !important;
    }}

    /* ============ ② 标题 / 正文 / 次要文字 / 分割线 ============ */
    [data-testid="stHeading"], [data-testid="stHeading"] *,
    [data-testid="stText"], .stMarkdown, .stMarkdown p, .stMarkdown li {{
        color: {text_color};
    }}
    /* 次要文字（caption/说明/时间戳）：每种模式独立的中灰，保证可读 */
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
    .stCaption {{
        color: {muted_color} !important;
    }}
    .stMarkdown hr {{
        margin: 8px 0;
        opacity: 0.4;
        border-color: {border_color};
    }}

    /* ============ ③ 侧边栏及其内部全部元素 ============ */
    section[data-testid="stSidebar"] {{
        background-color: {bg_color};
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        background-color: transparent;
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stHeading"],
    section[data-testid="stSidebar"] label {{
        color: {text_color};
    }}
    /* 控件标签（开关/勾选框/输入框标题等）：文字颜色统一覆盖 */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] p {{
        color: {text_color} !important;
    }}
    /* 侧边栏按钮圆角与悬停动效 */
    section[data-testid="stSidebar"] .stButton > button {{
        border-radius: 10px;
        transition: 0.2s;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        transform: translateY(-1px);
    }}

    /* ============ ④ 按钮（普通 / 下载 / 链接 / 表单提交） ============ */
    [data-testid="stButton"] > button,
    [data-testid="stDownloadButton"] > button,
    [data-testid="stLinkButton"] > button,
    [data-testid="stFormSubmitButton"] {{
        background-color: {surface_color} !important;
        color: {text_color} !important;
        border-color: {border_color} !important;
    }}
    [data-testid="stButton"] > button:hover,
    [data-testid="stDownloadButton"] > button:hover,
    [data-testid="stLinkButton"] > button:hover,
    [data-testid="stFormSubmitButton"]:hover {{
        background-color: {hover_color} !important;
        border-color: {accent_color} !important;
        color: {accent_color} !important;
    }}

    /* ============ ⑤ 输入框（文本 / 数字 / 文本域） ============ */
    /* 外层容器置透明，杜绝主题底色残留（1.57 中这些控件已不用
       baseweb，可见盒子就是 input/textarea 元素本身） */
    [data-testid="stTextInput"],
    [data-testid="stNumberInput"],
    [data-testid="stTextArea"] {{
        background-color: transparent !important;
    }}
    /* 底色 + 文字 + 光标 + 描边全部覆盖（-webkit-text-fill-color 兜底
       浏览器特殊渲染如自动填充） */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
        caret-color: {text_color};
        border-color: {input_border} !important;
    }}
    /* 占位提示文字：两种模式各自的中灰 */
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stNumberInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {{
        color: {muted_color} !important;
        opacity: 1;
    }}
    /* 聚焦时描边高亮为品牌色 */
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {{
        border-color: {accent_color} !important;
    }}
    /* 数字输入框的加减按钮 */
    [data-testid="stNumberInput"] button {{
        color: {text_color} !important;
        background-color: transparent !important;
    }}

    /* ============ ⑥ 下拉选择框（闭合状态 + 展开列表） ============ */
    /* 外层容器置透明，杜绝主题底色残留 */
    [data-testid="stSelectbox"] {{
        background-color: transparent !important;
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"] span,
    [data-testid="stSelectbox"] [data-baseweb="select"] input {{
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
    }}
    /* 展开的下拉列表（渲染在 body 级 Portal 中）：底色与文字跟随模式 */
    [data-testid="stSelectboxVirtualDropdown"] [data-baseweb="menu"] {{
        background-color: {input_bg} !important;
    }}
    [data-testid="stSelectboxVirtualDropdown"] [data-baseweb="menu"] li {{
        color: {text_color} !important;
    }}

    /* ============ ⑦ 聊天输入框（打字框） ============ */
    /* stChatInputTextArea 可能落在 textarea 本身或其包装元素上，
       两种情况的选择器都写上，保证命中；内层输入框置透明底色，
       露出容器底色，避免深浅色混搭 */
    [data-testid="stChatInput"] {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
    }}
    /* 中间包装层一律透明，透出容器底色；baseweb 输入根层
       （data-baseweb="input"）在 1.57 中自带主题底色，是打字框
       在日间模式下仍发黑的元凶，显式覆盖为控件底色 */
    [data-testid="stChatInput"] div {{
        background-color: transparent !important;
    }}
    [data-testid="stChatInput"] [data-baseweb="input"] {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
    }}
    [data-testid="stChatInputTextArea"],
    [data-testid="stChatInputTextArea"] textarea,
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {{
        background-color: transparent !important;
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
        caret-color: {text_color};
    }}
    [data-testid="stChatInputTextArea"]::placeholder,
    [data-testid="stChatInputTextArea"] textarea::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {muted_color} !important;
        opacity: 1;
    }}

    /* ============ ⑧ 折叠面板（思考过程/参考来源/对话统计） ============ */
    [data-testid="stExpander"] {{
        background-color: {surface_color} !important;
        border: 1px solid {border_color};
    }}
    [data-testid="stExpander"] summary {{
        color: {text_color};
        background-color: transparent;
    }}
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary span {{
        color: {text_color};
    }}
    /* 展开内容区：1.57 中该层自带主题底色（日间模式下展开后
       整块变黑的元凶），强制透明，透出折叠面板的控件底色 */
    [data-testid="stExpanderDetails"] {{
        background-color: transparent !important;
    }}

    /* ============ ⑨ 聊天消息气泡 ============ */
    .stChatMessage {{
        padding: 1rem;
        border-radius: 16px;
        margin: 0.6rem 0;
    }}
    /* 用户消息气泡：紫色渐变 + 白色文字。
       选择器说明：Streamlit 1.57 实际 DOM 中没有 stChatMessage-user /
       stChatMessage-assistant 这两个 testid（旧选择器匹配不到任何元素），
       消息角色信息在内容区的 aria-label 上（"Chat message from user" /
       "Chat message from assistant"），因此用 :has() 匹配。 */
    [data-testid="stChatMessage-user"],
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {{
        background: {accent_gradient} !important;
        color: white !important;
    }}
    [data-testid="stChatMessage-user"] [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage-user"] p,
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) p {{
        color: white !important;
    }}
    /* AI 回复气泡：卡片底 + 模式文字色 + 品牌色左边线 */
    [data-testid="stChatMessage-assistant"],
    [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) {{
        background: {surface_color} !important;
        color: {text_color} !important;
        border-left: 4px solid {accent_color};
    }}
    /* AI 回复内部文字显式继承气泡文字颜色（内容区与段落两层都覆盖） */
    [data-testid="stChatMessage-assistant"] [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage-assistant"] p,
    [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) p {{
        color: {text_color} !important;
    }}

    /* ============ ⑩ 指标卡（对话统计） ============ */
    [data-testid="stMetric"] {{
        background-color: {surface_color};
        border: 1px solid {border_color};
        border-radius: 10px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {muted_color} !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {text_color} !important;
    }}

    /* ============ ⑪ 提示框（info/success/warning/error） ============ */
    /* 语义颜色由图标与 ❌/⚠️/✅ 表情符号传达，底色统一为模式中性色，
       保证两种模式下文字都清晰可读 */
    [data-testid="stAlertContainer"] {{
        background-color: transparent;
    }}
    [data-testid="stAlert"] {{
        background-color: {alert_bg} !important;
        border: 1px solid {border_color};
    }}
    [data-testid="stAlert"] .stMarkdown,
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{
        color: {text_color} !important;
    }}

    /* ============ ⑫ 复选框 / 开关 / 单选框 / 滑块 ============ */
    /* st.toggle 与 st.checkbox 共用 stCheckbox testid（1.57 前端包核实） */
    [data-testid="stCheckbox"] label div,
    [data-testid="stRadio"] label div,
    [data-testid="stSlider"] {{
        color: {text_color} !important;
    }}

    /* ============ ⑬ 文件上传器 ============ */
    [data-testid="stFileUploader"] {{
        background-color: transparent;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {surface_color} !important;
        border-color: {border_color} !important;
    }}
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] small {{
        color: {text_color} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: {hover_color} !important;
        color: {text_color} !important;
        border-color: {border_color} !important;
    }}

    /* ============ ⑭ Tab 标签页 ============ */
    [data-testid="stTabs"] [data-baseweb="tab-list"],
    .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent;
        border-bottom-color: {border_color};
    }}
    [data-testid="stTabs"] [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] {{
        color: {muted_color} !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {text_color} !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {accent_color};
    }}

    /* ============ ⑮ 代码块 / 表格 / 进度条 / 弹层 / 表单 ============ */
    [data-testid="stCode"], [data-testid="stCode"] pre,
    [data-testid="stCode"] code, .stMarkdown code {{
        background-color: {code_bg} !important;
        color: {text_color} !important;
    }}
    [data-testid="stDataFrame"] {{
        background-color: transparent;
    }}
    [data-testid="stProgress"] {{
        background-color: {border_color};
    }}
    [data-testid="stSpinner"] {{
        color: {text_color};
    }}
    [data-testid="stToast"] {{
        background-color: {surface_color} !important;
        border: 1px solid {border_color};
    }}
    /* 帮助提示浮层 / 弹窗内容：跟随模式 */
    [data-testid="stTooltipContent"],
    [data-testid="stPopoverBody"] {{
        background-color: {surface_color};
        color: {text_color};
    }}
    [data-testid="stForm"] {{
        background-color: transparent;
        border-color: {border_color};
    }}

    /* ============ ⑯ 滚动条美化 ============ */
    ::-webkit-scrollbar {{width: 8px; height: 8px;}}
    ::-webkit-scrollbar-track {{background: {border_color}; border-radius: 4px;}}
    ::-webkit-scrollbar-thumb {{background: {accent_gradient}; border-radius: 4px;}}

    /* 历史对话中删除按钮保持对齐 */
    .stColumn > div {{
        display: flex;
        align-items: center;
        height: 100%;
    }}
</style>
""", unsafe_allow_html=True)


# ====================== 客户端初始化 ======================
def create_ai_client() -> OpenAI:
    """创建统一的 AI API 客户端

    三家提供方（DeepSeek / OpenAI / Ollama）均通过 OpenAI SDK 调用，
    由 modules/models.py 提供统一配置；多模型模块不可用时自动退化为
    DeepSeek 环境变量配置（保证应用不崩溃）。

    超时说明：客户端 read 超时即"两次数据块之间的最大等待时间"，
    统一收紧到 STREAM_STALL_TIMEOUT 秒，配合流循环内的停滞检查实现
    30 秒超时保护（超时只影响卡死的连接，不影响正常的长回复）。
    """
    if MODELS_AVAILABLE:
        cfg = get_model_config(
            st.session_state.current_model, provider_key=st.session_state.provider)
        api_key = (st.session_state.api_keys.get(st.session_state.provider, "")
                   or cfg["api_key"])
        base_url = (st.session_state.base_urls.get(st.session_state.provider, "")
                    or cfg["base_url"])
        # 客户端 read 超时取 min(模型配置, 停滞超时)：任何情况下断流最多 30 秒即可发现
        timeout = min(cfg["timeout"], AppConfig.STREAM_STALL_TIMEOUT)
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # 兜底：多模型模块不可用时退回 DeepSeek 环境变量配置
    return OpenAI(
        api_key=st.session_state.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=st.session_state.get("base_url", "https://api.deepseek.com"),
        timeout=AppConfig.STREAM_STALL_TIMEOUT,
    )


# ====================== API核心函数 ======================
def call_ai_api_stream(messages: List[dict], tools: Optional[List[dict]] = None) -> Tuple[Any, Optional[str]]:
    """流式调用AI接口

    Args:
        messages: 完整的消息列表
        tools: 工具定义列表，传入时启用函数调用（None 表示不带工具）

    Returns:
        tuple: (流式响应对象, 错误信息或None)
    """
    max_retry = 2
    retry_count = 0

    while retry_count <= max_retry:
        try:
            client = create_ai_client()
            stream = client.chat.completions.create(
                model=st.session_state.current_model,
                messages=messages,
                stream=True,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens,
                tools=tools,
            )
            return stream, None
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            logger.warning("接口调用失败，重试 %d/%d：%s", retry_count, max_retry, e)
            if retry_count > max_retry:
                if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
                    return None, f"❌ API密钥无效或未填写！请检查「{st.session_state.provider}」提供方的密钥配置。"
                elif "quota" in error_msg.lower() or "balance" in error_msg.lower():
                    return None, "❌ 账户配额不足，请检查账户余额！"
                elif "timeout" in error_msg.lower():
                    return None, "❌ 请求超时，请检查网络后重试！"
                else:
                    return None, f"❌ 接口请求失败：{error_msg}"


def trim_context_messages(messages: List[dict]) -> List[dict]:
    """裁剪上下文消息（仅保留最近 max_context_msg 条，系统提示词始终保留）"""
    if len(messages) > st.session_state.max_context_msg:
        system_msg = messages[0] if messages and messages[0]["role"] == "system" else None
        new_msgs = messages[-st.session_state.max_context_msg:]
        if system_msg:
            new_msgs.insert(0, system_msg)
        return new_msgs
    return messages


_CJK_RE = re.compile(r"[一-鿿　-〿＀-￯]")


def estimate_tokens(text: str) -> int:
    """粗略估算文本 Token 数（中文字符约 1 token/字，其他字符约 4 字符/token）"""
    if not text:
        return 0
    cjk_chars = len(_CJK_RE.findall(text))
    other_chars = len(text) - cjk_chars
    return cjk_chars + other_chars // 4


def run_tool_loop(
    messages: List[dict],
    content_placeholder: Any,
    tool_placeholder: Any,
    tools: Optional[List[dict]] = None,
) -> Tuple[str, List[str], Optional[str], str]:
    """带工具调用的多轮对话循环（Function Calling 主流程）

    流程：
    1. 携带工具定义调用 AI（流式），同时累积正文与 tool_calls 增量
    2. 若 AI 返回 tool_calls → 逐个执行工具，结果以 tool 消息追加回对话，进入下一轮
    3. 若 AI 直接回复正文 → 结束循环，返回最终回复
    4. 最多循环 MAX_TOOL_ROUNDS 轮，防止死循环
    5. 模型不支持工具调用时报错 → 自动去掉 tools 重试一次（兼容 reasoner 等模型）
    6. 流式 30 秒停滞保护（客户端 read 超时 + 循环内检查双保险），中断时保留已接收内容
    7. 工具结果超过 4000 字符自动截断，防止挤爆上下文窗口

    Args:
        messages: 完整的 API 消息列表（system + 历史对话，本函数会就地追加中间消息）
        content_placeholder: 用于流式渲染回复内容的 st.empty 占位符
        tool_placeholder: 用于显示工具调用状态的 st.empty 占位符
        tools: 工具定义列表；None 表示不启用工具（单轮普通对话）

    Returns:
        tuple: (最终回复文本, 使用过的工具名列表, 错误信息或None, 思考过程文本)
    """
    MAX_TOOL_ROUNDS = 3
    tools_used = []
    reasoning_text = ""             # 累积的模型思考过程（如 deepseek-reasoner）
    reasoning_placeholder = None    # 思考过程实时展示占位符（首次收到思考内容时创建）
    allow_retry_without_tools = tools is not None  # 首次调用带工具时，报错可降级重试

    for _ in range(MAX_TOOL_ROUNDS):
        if tools:
            content_placeholder.markdown("🤔 正在思考是否调用工具...")

        stream, err = call_ai_api_stream(messages, tools=tools)
        if err:
            # 模型不支持工具调用（如 deepseek-reasoner）时，去掉 tools 降级重试一次
            if allow_retry_without_tools:
                allow_retry_without_tools = False
                tools = None
                stream, err = call_ai_api_stream(messages, tools=None)
            if err:
                return "", tools_used, err, reasoning_text

        # 累积流式输出：思考过程 + 正文 + 工具调用增量（按 index 拼接参数片段）
        # 停滞保护：30 秒未收到新数据即中止本轮（客户端 read 超时兜底彻底断流，
        # 此检查兜底"有数据但极慢"的情况），避免界面无限转圈
        full_content = ""
        tool_calls = {}
        last_chunk_time = time.monotonic()
        try:
            for chunk in stream:
                if time.monotonic() - last_chunk_time > AppConfig.STREAM_STALL_TIMEOUT:
                    logger.warning("流式响应停滞超过 %s 秒，中止本轮接收",
                                   AppConfig.STREAM_STALL_TIMEOUT)
                    if full_content:
                        return full_content, tools_used, "⚠️ 响应中断：部分内容已返回", reasoning_text
                    return "", tools_used, "❌ 流式响应超时（30 秒无数据），请重试", reasoning_text
                last_chunk_time = time.monotonic()
                delta = chunk.choices[0].delta
                # 思考过程增量：reasoner 模型先流式输出思考，再输出最终回答
                reasoning_piece = extract_reasoning(chunk)
                if reasoning_piece:
                    reasoning_text += reasoning_piece
                    if reasoning_placeholder is None:
                        # 首次收到思考内容时才创建可折叠面板，普通模型零开销
                        with st.expander("💭 思考过程", expanded=True):
                            reasoning_placeholder = st.empty()
                    reasoning_placeholder.markdown(reasoning_text)
                if delta.content:
                    full_content += delta.content
                    content_placeholder.markdown(full_content + "▌")
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index or 0
                        entry = tool_calls.setdefault(idx, {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if tc.id:
                            entry["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                entry["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                entry["function"]["arguments"] += tc.function.arguments
        except Exception as e:
            # 流中途断连（如 read 超时）：保留已接收内容降级返回，不崩溃
            logger.warning("流式接收中断：%s", e)
            if full_content:
                return full_content, tools_used, "⚠️ 流式响应中断，已返回部分内容", reasoning_text
            return "", tools_used, f"❌ 流式响应中断：{e}", reasoning_text

        # 没有工具调用 → AI 已给出最终回答
        if not tool_calls:
            return full_content, tools_used, None, reasoning_text

        # AI 请求调用工具：执行并把结果回传
        call_list = list(tool_calls.values())
        messages.append({
            "role": "assistant",
            "content": full_content or None,
            "tool_calls": call_list,
        })
        for call in call_list:
            name = call["function"]["name"]
            tools_used.append(name)
            tool_placeholder.markdown("🔧 调用了工具：" + "、".join(tools_used))
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            logger.info("执行工具：%s，参数：%s", name,
                        call["function"]["arguments"][:200])
            result = execute_tool(name, arguments)
            # 工具结果截断保护：超长结果会挤爆上下文窗口，限制为 4000 字符
            if len(result) > AppConfig.MAX_TOOL_RESULT_CHARS:
                logger.warning("工具 %s 结果过长（%d 字符），已截断至 %s 字符",
                               name, len(result), AppConfig.MAX_TOOL_RESULT_CHARS)
                result = (result[:AppConfig.MAX_TOOL_RESULT_CHARS]
                          + "…（结果过长，已截断）")
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })

    return "", tools_used, "❌ 工具调用轮数超过上限，请简化问题后重试", reasoning_text


# ====================== 页面渲染函数 ======================
def display_chat_messages() -> None:
    """渲染聊天记录（虚拟滚动：仅渲染最近 MAX_RENDER_MESSAGES 条，超长会话不卡顿）"""
    if not st.session_state.messages:
        st.markdown("### 💡 快捷提问")
        cols = st.columns(2)
        for idx, question in enumerate(QUICK_QUESTIONS):
            with cols[idx % 2]:
                if st.button(question, key=f"quick_{idx}", use_container_width=True):
                    st.session_state.pending_message = question
                    st.rerun()
        return

    # 虚拟滚动：只渲染最近 50 条消息，避免超长会话时页面卡死
    all_messages = st.session_state.messages
    if len(all_messages) > AppConfig.MAX_RENDER_MESSAGES:
        st.caption(f"📜 共 {len(all_messages)} 条消息，仅渲染最近 {AppConfig.MAX_RENDER_MESSAGES} 条")
        all_messages = all_messages[-AppConfig.MAX_RENDER_MESSAGES:]

    for i, msg in enumerate(all_messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                # 历史消息中带有思考过程时先展示（可折叠，最终回答在下方）
                if msg.get("reasoning"):
                    with st.expander("💭 思考过程", expanded=False):
                        st.markdown(msg["reasoning"])
                st.markdown(msg["content"])
                # 历史消息中带有引用来源时一并展示
                if msg.get("sources"):
                    with st.expander("📚 参考来源", expanded=False):
                        for i, src in enumerate(msg["sources"], 1):
                            st.caption(f"📄 来源{i}：{src['source']}　·　相似度 {src['score']:.2f}")
                # 历史消息中带有工具调用记录时一并展示
                if msg.get("tools_used"):
                    st.caption("🔧 调用了工具：" + "、".join(msg["tools_used"]))
                # 历史消息中命中缓存时一并展示
                if msg.get("cache_hit"):
                    st.caption("📦 来自缓存")


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
        save_session_to_file()
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
            save_session_to_file()

    # ---- 当前模型默认参数提示 ----
    model_cfg = get_model_config(st.session_state.current_model,
                                 provider_key=st.session_state.provider)
    caption = (f"默认参数：temperature={model_cfg['temperature']}，"
               f"最大长度={model_cfg['max_tokens']}")
    if not model_cfg["supports_tools"]:
        caption += "；不支持工具调用"
    st.caption(caption)

    if st.button("✅ 保存配置", use_container_width=True):
        save_session_to_file()
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
        save_session_to_file()
        st.success("配置保存成功！")
        st.rerun()


# ====================== 侧边栏渲染 ======================
def render_sidebar() -> None:
    """渲染侧边栏所有功能"""
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
                save_session_to_file()
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
                st.error("⚠️ RAG 依赖未安装，请执行：\n```\npip install langchain langchain-community langchain-text-splitters chromadb pypdf\n```")
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
                          help="开启后，提问会先从已上传文档中检索相关内容再交给AI回答")

                upload_file = st.file_uploader(
                    "上传文档（PDF / TXT / MD）",
                    type=["pdf", "txt", "md"],
                    disabled=not ready,
                    help="上传后自动切分并向量化；支持 PDF、TXT、Markdown 格式"
                )
                if upload_file is not None:
                    # 10MB 上限保护：超大文件会导致内存与页面卡死
                    if upload_file.size > AppConfig.MAX_UPLOAD_SIZE:
                        st.error(f"❌ 文件超过 {AppConfig.MAX_UPLOAD_SIZE // (1024 * 1024)}MB 限制，无法上传")
                    else:
                        try:
                            with st.spinner(f"正在处理《{upload_file.name}》..."):
                                chunks = load_document(upload_file)
                                if not chunks:
                                    st.error(f"❌ 《{upload_file.name}》未解析出任何内容，请检查文件格式")
                                    st.stop()
                                # 加载与向量化分开捕获：定位失败环节更准确
                                try:
                                    n_added = add_to_vectorstore(chunks)
                                except Exception as e:
                                    logger.error("文档向量化失败：%s", e)
                                    st.error(f"❌ 文档向量化失败：{str(e)}")
                                    st.stop()
                            st.success(f"✅ 《{upload_file.name}》入库成功（{n_added} 个片段）")
                            st.rerun()
                        except Exception as e:
                            logger.error("文档加载失败：%s", e)
                            st.error(f"❌ 文档加载失败：{str(e)}")

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
                                delete_document(doc["doc_id"])
                            except Exception as e:
                                logger.error("删除文档失败：%s", e)
                                st.error(f"❌ 删除失败：{str(e)}")
                            else:
                                st.success(f"✅ 已删除《{doc['file_name']}》")
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
                st.error("⚠️ 工具模块加载失败，请检查 modules/tools.py")
            else:
                st.toggle("🔧 启用工具调用", key="tools_enabled",
                          help="开启后，AI 可调用时间、计算器、网络搜索等工具，回答需要实时信息或精确计算的问题")
                st.caption("可用工具：" + "、".join(get_tool_names()))
                if MODELS_AVAILABLE:
                    model_cfg = get_model_config(
                        st.session_state.current_model,
                        provider_key=st.session_state.provider)
                    if not model_cfg["supports_tools"]:
                        st.caption(f"⚠️ 当前模型 {st.session_state.current_model} 不支持工具调用，已自动跳过")

        # 缓存设置（Redis）
        with st.expander("📦 缓存设置"):
            if not CACHE_AVAILABLE:
                st.error("⚠️ 缓存模块加载失败，请检查 modules/cache.py")
            else:
                st.toggle("📦 启用Redis缓存", key="cache_enabled",
                          help="相同问题命中缓存时直接返回，减少API调用成本；Redis不可用时自动跳过")
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
                    f"# 对话记录\n",
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
                        save_session_to_file()
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
                save_session_to_file()
                st.rerun()
        with col2:
            if st.button("🗑️ 清空当前", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.conversations:
                    st.session_state.conversations[st.session_state.current_conversation_index]["messages"] = []
                save_session_to_file()
                st.success("✅ 已清空当前对话")
                st.rerun()
        with col3:
            if st.button("🧹 全部清空", use_container_width=True):
                st.session_state.conversations = [{
                    "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "name": "默认对话",
                    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "messages": []
                }]
                st.session_state.current_conversation_index = 0
                st.session_state.messages = []
                save_session_to_file()
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
                            key=f"conv_name_edit_{idx}",
                            label_visibility="collapsed",
                            placeholder="对话名称"
                        )
                        if new_name != conv["name"]:
                            st.session_state.conversations[idx]["name"] = new_name
                            save_session_to_file()
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
                            save_session_to_file()
                            st.rerun()

                with col_del:
                    # 仅保留删除按钮，移除重新生成功能
                    # 如果只有一条对话，禁用删除或提示，但保留删除逻辑（至少保留一个对话）
                    del_disabled = len(st.session_state.conversations) <= 1
                    if st.button(
                            "🗑️",
                            key=f"del_conv_{idx}",
                            help="删除此对话",
                            disabled=del_disabled,
                            use_container_width=True
                    ):
                        if len(st.session_state.conversations) > 1:
                            # 删除当前对话时，需要切换到其他对话
                            if idx == st.session_state.current_conversation_index:
                                # 切换到前一个或后一个
                                new_idx = idx - 1 if idx > 0 else 0
                                # 但要保证新索引有效
                                if new_idx >= len(st.session_state.conversations) - 1:
                                    new_idx = 0
                                st.session_state.current_conversation_index = new_idx
                                st.session_state.messages = st.session_state.conversations[new_idx]["messages"]
                                st.session_state.conversation_id = st.session_state.conversations[new_idx]["id"]
                            # 执行删除
                            st.session_state.conversations.pop(idx)
                            save_session_to_file()
                            st.rerun()
                        else:
                            st.warning("至少保留一个对话")

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


# ====================== 主逻辑 ======================
render_sidebar()

# 页面头部
st.title("🤖 马氏-AI 智能助手")
st.divider()

# 处理快捷提问和输入
pending_msg = st.session_state.pop("pending_message", None)
input_msg = st.chat_input("💬 输入问题，按Enter发送...")
process_msg = input_msg or pending_msg

# 渲染聊天界面
display_chat_messages()

# 处理用户消息
if process_msg and process_msg.strip():
    if not process_msg.strip():
        st.warning("请勿发送空消息！")
        st.stop()

    # 消息长度限制：超长输入会挤爆上下文窗口，直接拦截
    if len(process_msg) > AppConfig.MAX_MESSAGE_LENGTH:
        st.warning(f"⚠️ 消息过长（{len(process_msg)} 字符），"
                   f"最多允许 {AppConfig.MAX_MESSAGE_LENGTH} 字符，请精简后再发送")
        st.stop()

    with st.chat_message("user", avatar="👤"):
        st.markdown(process_msg)

    st.session_state.messages.append({"role": "user", "content": process_msg})

    # RAG 检索：先从向量库中查找与问题相关的文档片段
    rag_sources = []
    if RAG_AVAILABLE and st.session_state.rag_enabled:
        try:
            rag_sources = search(process_msg, top_k=3)
        except Exception as e:
            logger.warning("文档检索失败：%s", e)
            st.warning(f"⚠️ 文档检索失败：{str(e)}")

    # 将检索到的片段拼接到系统提示词中，引导AI基于文档回答
    api_msgs = [{"role": "system", "content": st.session_state.system_prompt}]
    if rag_sources:
        context_text = "\n\n".join(
            f"【文档片段{i}】（来源：{s['source']}）：\n{s['content']}"
            for i, s in enumerate(rag_sources, 1)
        )
        api_msgs[0]["content"] += (
            "\n\n【文档检索上下文】\n"
            "以下是从用户上传文档中检索到的相关内容，请优先基于这些内容回答用户问题；"
            "如果文档中没有与问题相关的内容，请如实告知，再结合你自己的知识回答：\n\n"
            + context_text
        )
    api_msgs += st.session_state.messages
    api_msgs = trim_context_messages(api_msgs)

    with st.chat_message("assistant", avatar="🤖"):
        res_placeholder = st.empty()
        tool_placeholder = st.empty()

        # 判断是否启用工具调用（当前模型不支持时自动跳过，如 deepseek-reasoner）
        tools_for_call = None
        model_supports_tools = True
        if MODELS_AVAILABLE:
            model_supports_tools = get_model_config(
                st.session_state.current_model,
                provider_key=st.session_state.provider
            )["supports_tools"]
        if (TOOLS_AVAILABLE and st.session_state.tools_enabled and model_supports_tools):
            tools_for_call = get_available_tools()

        full_response = ""
        tools_used = []
        reasoning_text = ""
        cache_hit = False
        err = None

        # 缓存查找：仅普通对话可用缓存（有RAG检索或启用工具时跳过，避免旧答案/旧时间）
        use_cache = (CACHE_AVAILABLE and st.session_state.cache_enabled
                     and not rag_sources and tools_for_call is None)
        if use_cache:
            # 缓存降级保护：Redis 异常时跳过缓存直接走 API，不影响正常对话
            try:
                cached = get_cached_response(process_msg, model=st.session_state.current_model)
            except Exception as e:
                logger.warning("缓存读取失败，跳过缓存：%s", e)
                cached = None
            if cached:
                full_response = cached
                cache_hit = True
                res_placeholder.markdown(full_response)
                st.caption("📦 来自缓存")

        if not cache_hit:
            full_response, tools_used, err, reasoning_text = run_tool_loop(
                api_msgs, res_placeholder, tool_placeholder, tools_for_call
            )
            if err:
                if full_response:
                    # 已收到部分内容：保留内容并附上中断提示，而不是覆盖掉
                    res_placeholder.markdown(full_response)
                    st.warning(err)
                else:
                    res_placeholder.error(err)
            elif use_cache and full_response:
                # 回答生成后写入缓存，供下次相同问题直接命中（失败只记日志，不影响对话）
                try:
                    set_cached_response(process_msg, full_response,
                                        model=st.session_state.current_model)
                except Exception as e:
                    logger.warning("缓存写入失败：%s", e)

        # 在回复下方显示检索到的参考来源
        if rag_sources:
            with st.expander("📚 参考来源", expanded=False):
                for i, src in enumerate(rag_sources, 1):
                    st.caption(f"📄 来源{i}：{src['source']}　·　相似度 {src['score']:.2f}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": rag_sources,
        "tools_used": tools_used,
        "cache_hit": cache_hit,
        "reasoning": reasoning_text,
    })
    if st.session_state.conversations:
        st.session_state.conversations[st.session_state.current_conversation_index][
            "messages"] = st.session_state.messages.copy()
    save_session_to_file()

st.divider()