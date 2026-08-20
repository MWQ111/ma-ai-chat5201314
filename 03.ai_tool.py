import streamlit as st
import os
from openai import OpenAI
#创建OpenAI对象
client = OpenAI(
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com")
#设置页面配置
st.set_page_config(
    page_title="AI 智能体",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
#系统提示词
system_prompt = "我是一名可爱的ai助理喔~."
#初始化聊天信息
if "messages" not in st.session_state:
    st.session_state.messages = []

#展示聊天信息
for message in st.session_state.messages:
    if message["role"] == "system":
        st.write(message["content"])
    elif message["role"] == "user":
        st.chat_message("user").write(message["content"])
    elif message["role"] == "assistant":
        st.chat_message("assistant").write(message["content"])
#大标题
st.title("你好,我是你的AI助理")
st.header("你可以提问任何问题")
st.subheader("我都会回复你")
# logo
st.logo("./resources/AILogo.jpg")

#消息输入框
message = st.chat_input("请输入您要问的问题：", key="user_input")

#处理用户输入
if message:
    st.chat_message("user").write(message)
    print("————————>调用AI大模型，提示词：",message)
    #保存用户输入
    st.session_state.messages.append({"role": "user", "content": message})
    #调用AI大模型
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        stream=False,
        reasoning_effort="high",
        extra_body={"thinking": {"type": "enabled"}}
    )
    #输出大模型返回结果
    print("<————————AI大模型返回结果：",response.choices[0].message.content)
    st.chat_message("assistant").write(response.choices[0].message.content)
    #保存大模型返回结果
    st.session_state.messages.append(
        {"role": "assistant", "content": response.choices[0].message.content}
    )