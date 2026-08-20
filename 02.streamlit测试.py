import streamlit as st
import os
from openai import OpenAI
from datetime import datetime
import json

# 创建OpenAI对象
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

# 设置页面配置
st.set_page_config(
    page_title="AI 智能助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/help',
        'Report a bug': 'https://github.com/your-repo/issues',
        'About': "# AI智能助手\n一个基于DeepSeek的强大AI助手"
    }
)

# 系统提示词（可配置）
DEFAULT_SYSTEM_PROMPT = """你是一位可爱且专业的AI助理喔~。你的特点：
1. 回答要亲切友好，使用适当的语气词（如：呢、哦、呀）
2. 提供准确、有用的信息
3. 在不确定时坦诚说明
4. 适当使用表情符号增加亲和力
5. 回答要简洁明了，避免过于冗长"""


# 初始化session state
def init_session_state():
    """初始化所有session state变量"""
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 2000


init_session_state()

# 侧边栏配置
with st.sidebar:
    st.header("⚙️ 配置选项")

    # 系统提示词编辑
    with st.expander("📝 系统提示词", expanded=False):
        new_system_prompt = st.text_area(
            "编辑AI的角色设定",
            value=st.session_state.system_prompt,
            height=200,
            help="修改这里可以改变AI的回答风格和角色"
        )
        if st.button("保存提示词"):
            st.session_state.system_prompt = new_system_prompt
            st.success("提示词已更新！")
            st.rerun()

    # 模型参数
    with st.expander("🔧 高级参数", expanded=False):
        temperature = st.slider(
            "创造性 (Temperature)",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.1,
            help="值越高回答越有创造性，值越低越保守"
        )
        st.session_state.temperature = temperature

        max_tokens = st.number_input(
            "最大回复长度",
            min_value=100,
            max_value=4000,
            value=st.session_state.max_tokens,
            step=100,
            help="AI回复的最大token数"
        )
        st.session_state.max_tokens = max_tokens

    # 对话管理
    with st.expander("💾 对话管理", expanded=False):
        if st.button("🔄 清空对话历史", use_container_width=True):
            st.session_state.messages = []
            st.success("对话历史已清空！")
            st.rerun()

        if st.button("💾 导出对话记录", use_container_width=True):
            export_data = {
                "conversation_id": st.session_state.conversation_id,
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages,
                "system_prompt": st.session_state.system_prompt
            }
            st.download_button(
                label="📥 下载JSON文件",
                data=json.dumps(export_data, ensure_ascii=False, indent=2),
                file_name=f"chat_history_{st.session_state.conversation_id}.json",
                mime="application/json"
            )

        if st.button("📂 导入对话记录", use_container_width=True):
            uploaded_file = st.file_uploader(
                "选择JSON文件",
                type=['json'],
                key="import_file"
            )
            if uploaded_file:
                import_data = json.load(uploaded_file)
                st.session_state.messages = import_data.get("messages", [])
                st.session_state.system_prompt = import_data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
                st.success("对话记录导入成功！")
                st.rerun()

    # 使用统计
    with st.expander("📊 使用统计", expanded=False):
        total_messages = len(st.session_state.messages)
        user_messages = sum(1 for m in st.session_state.messages if m["role"] == "user")
        assistant_messages = sum(1 for m in st.session_state.messages if m["role"] == "assistant")

        st.metric("总消息数", total_messages)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("用户消息", user_messages)
        with col2:
            st.metric("AI回复", assistant_messages)

        if total_messages > 0:
            st.progress(user_messages / total_messages, text="对话占比")


# 主界面
def display_chat_messages():
    """显示聊天消息"""
    for messages in st.session_state.messages:
        if messages["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(messages["content"])
        elif messages["role"] == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(messages["content"])


def call_ai_api(messages):
    """调用AI API"""
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # 使用正确的模型名称
            messages=messages,
            stream=False,
            temperature=st.session_state.temperature,
            max_tokens=st.session_state.max_tokens,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ API调用出错：{str(e)}"


# 显示标题和头部
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.title("🤖 你好，我是你的AI智能助手")
    st.markdown("*我可以回答问题、提供建议、帮助你解决问题*")
    st.divider()

# 显示Logo（如果有的话）
logo_path = "./resources/AILogo.jpg"
if os.path.exists(logo_path):
    st.logo(logo_path)

# 显示聊天记录
display_chat_messages()

# 消息输入框
message = st.chat_input(
    "💬 输入您的问题...（支持多行输入，按Enter发送）",
    key="user_input"
)

# 处理用户输入
if message and message.strip():
    # 显示用户消息
    with st.chat_message("user", avatar="👤"):
        st.markdown(message)

    # 保存用户消息
    st.session_state.messages.append({"role": "user", "content": message})

    # 显示AI思考指示器
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🤔 思考中..."):
            # 准备API消息
            api_messages = [
                {"role": "system", "content": st.session_state.system_prompt},
                *[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
            ]

            # 调用API
            response_content = call_ai_api(api_messages)

            # 显示回复
            st.markdown(response_content)

            # 保存AI回复
            st.session_state.messages.append(
                {"role": "assistant", "content": response_content}
            )

# 底部提示
st.divider()
st.caption("✨ AI智能助手 | 基于DeepSeek API | 您的对话将被用于改善服务质量")

# 添加一些CSS美化
st.markdown("""
<style>
    /* 自定义滚动条 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #888;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
    }

    /* 聊天消息样式 */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)



















