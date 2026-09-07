import requests

# 测试 GET /conversations
response = requests.get("http://localhost:5000/conversations")
print("GET /conversations：", response.json())

# 测试 POST /conversations（新建对话）
response = requests.post(
    "http://localhost:5000/conversations",
    json={"title": "API测试对话"}
)
print("POST /conversations：", response.json())

# 如果新建成功，取 conversation_id 测试新增消息
if response.status_code == 200:
    conv_id = response.json().get("conversation_id")
    if conv_id:
        msg_response = requests.post(
            f"http://localhost:5000/messages/{conv_id}",
            json={"role": "user", "content": "你好，这是通过API发的消息"}
        )
        print("POST /messages：", msg_response.json())