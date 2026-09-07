"""
app.py — Streamlit 应用入口（只做导入与组装，具体逻辑见 core/ 与 ui/）

结构：
- core/：业务逻辑层（不依赖 Streamlit）：配置（config）、会话持久化（session）、
  对话核心流程（chat），以及自 modules/ 迁移而来的 RAG / 工具 / 缓存 /
  多模型 / Agent 模块（统一在 core/__init__.py 导出，含 *_AVAILABLE 开关）
- ui/：UI 层（仅依赖 Streamlit）：侧边栏（sidebar）、聊天界面（chat）、
  通用组件（components，全局主题 CSS / toast 等）

运行：streamlit run app.py
"""

import logging
import os

import streamlit as st

# ====================== 日志系统 ======================
# 仅在根日志器无处理器时配置（Streamlit 通常已配置好，避免重复输出）；
# 各模块内统一使用 logging.getLogger("ai_chat.*") 记录关键路径的运行与降级信息。
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
logger = logging.getLogger("ai_chat")

# ====================== 环境变量加载 ======================
# 使用 find_dotenv(usecwd=True) 定位当前工作目录下的 .env 文件
# （以 `streamlit run app.py` 启动时工作目录即项目根目录，比默认的
# 按调用栈向上查找更可靠；文件不存在时静默跳过）。
# 必须放在功能模块导入之前：core/cache.py 等在导入时会读取环境变量。
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

# ====================== 业务逻辑层与 UI 层导入 ======================
# noqa: E402 —— 有意放在环境变量加载之后（core/cache.py 等在导入时读取环境变量）
from core import db_available, init_session_state  # noqa: E402
from ui.chat import display_chat_messages, handle_user_message  # noqa: E402
from ui.components import inject_custom_css  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402

# ====================== 页面全局配置 ======================
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

st.logo("./resources/gdutlogo.png")

# ====================== 初始化与组装 ======================
init_session_state(st.session_state)

# ====================== 回调内 rerun 统一处理入口 ======================
# Streamlit ≥1.37 把 on_click/on_change 回调内调用的 st.rerun() 视为 no-op
# 并输出 "Calling st.rerun() within a callback is a no-op." 警告（本项目运行于
# 1.57）。因此所有回调只负责修改状态并置位 _need_rerun 标记，真正触发重跑的
# st.rerun() 统一收敛到本处执行：置位 → 顶层立即 rerun → 新一次运行 pop 掉标记。
if st.session_state.pop("_need_rerun", False):
    st.rerun()

# 美化CSS：必须在所有页面元素渲染之前注入（紧跟 init_session_state），
# 保证 AI 流式回答期间样式恒定（原因详见 ui/components.py 注释）。
inject_custom_css()

render_sidebar()

# 页面头部
st.title("🤖 马氏-AI 智能助手")

# 存储后端状态提示：MySQL 连接失败时自动降级为本地 JSON（见 core/db.py）
st.caption("🗄️ 存储后端：MySQL" if db_available()
           else "💾 存储后端：本地 JSON（MySQL 连接失败，已自动降级）")
st.divider()

# 处理快捷提问和输入
pending_msg = st.session_state.pop("pending_message", None)
input_msg = st.chat_input("💬 输入问题，按Enter发送...")
process_msg = input_msg or pending_msg

# 渲染聊天界面
display_chat_messages()

# 处理用户消息（完整流程见 ui/chat.py）
if process_msg and process_msg.strip():
    handle_user_message(process_msg)

st.divider()


# ====================== 本地启动入口（仅 `python app.py` 时生效） ======================
# ⚠️ 不能用 `if __name__ == "__main__"` 单独判断：Streamlit 执行脚本时
# __name__ 同样是 "__main__"（见 streamlit 源码 script_runner.py 的官方注释），
# 该块会在**每次 rerun** 时重复执行，导致每次点击都额外启动一个
# streamlit 子进程。必须再判断当前是否处于 Streamlit 运行环境：
# 有 ScriptRunContext 说明是 `streamlit run` 启动的，跳过本块。
if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:  # 无 Streamlit 运行上下文 = 纯 python 启动
        os.system("streamlit run app.py --server.headless true")
