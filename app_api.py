import json
import threading
import uuid

from flask import Flask, request, jsonify, Response, stream_with_context
import streamlit as st
import os

from core.chat import call_ai_api_simple, call_ai_api_simple_stream
from core.db import SessionDB
app = Flask(__name__)

# 正在进行的流式生成任务登记表：request_id -> {"cancel_event": threading.Event}
# /chat/stream 启动时登记，流结束 / 客户端断开 / /cancel 取消时移除；
# /cancel 接口据此按 request_id 中断正在进行的 AI 流式生成。
active_streams: dict[str, dict] = {}

# 手动初始化 session_state（给 API 用）
if not hasattr(st, 'session_state'):
    st.session_state.api_keys = {
        "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "ollama": "",
    }
    st.session_state.base_urls = {
        "deepseek": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "openai": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "ollama": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    }
    st.session_state.provider = "deepseek"
    st.session_state.current_model = "deepseek-chat"
    st.session_state.temperature = 0.7
    st.session_state.max_tokens = 2000

@app.route("/")
def hello():
    return "AI API 已启动！"
@app.route("/ping", methods=["GET"])
def ping():
    return "pong"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data:
        return jsonify({"error": "请发送 JSON 格式的请求"}), 400

    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "message 字段不能为空"}), 400

    messages = [{"role": "user", "content": user_message}]
    full_response = call_ai_api_simple(messages)

    return jsonify({
        "response": full_response,
        "user_message": user_message
    })

@app.route("/ask", methods=["GET"])
def ask():
    question = request.args.get("q", "")
    if not question:
        return "请传入 q 参数，比如 /ask?q=你好"
    messages = [{"role": "user", "content": question}]
    full_response = call_ai_api_simple(messages)
    return full_response


def _sse(data: dict, event: str | None = None) -> str:
    """把数据包装成一条 SSE 消息（Server-Sent Events 协议）"""
    payload = json.dumps(data, ensure_ascii=False)
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {payload}\n\n"


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """流式对话接口（SSE）：边生成边推送，支持通过 /cancel/<request_id> 中断

    与 /chat 的区别：/chat 一次性返回完整回答，无法中途取消；本接口以 SSE
    逐块推送文本增量，并在响应首帧携带 request_id 供取消使用。

    请求体：{"message": "...", "history": [{"role": "...", "content": "..."}, ...]}
    返回体：text/event-stream，事件序列：
      event: start —— data: {"request_id": "..."}
      event: delta —— data: {"delta": "文本增量"}
      event: done  —— data: {"cancelled": true|false}
    """
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "message 字段不能为空"}), 400

    history = data.get("history") or []
    history = [{"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
               for m in history if isinstance(m, dict)]
    messages = history + [{"role": "user", "content": user_message}]

    request_id = uuid.uuid4().hex
    cancel_event = threading.Event()
    active_streams[request_id] = {"cancel_event": cancel_event}

    def generate():
        try:
            # 首帧：告知客户端本次生成的 request_id（用于调用 /cancel）
            yield _sse({"request_id": request_id}, event="start")
            for piece in call_ai_api_simple_stream(messages, cancel_event=cancel_event):
                # 客户端断开 或 收到取消：提前结束本轮推送
                if cancel_event.is_set() or request.environ.get("werkzeug.server.disconnect"):
                    break
                yield _sse({"delta": piece}, event="delta")
            yield _sse({"cancelled": cancel_event.is_set()}, event="done")
        finally:
            active_streams.pop(request_id, None)  # 流结束后务必注销，避免登记表膨胀

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/cancel/<request_id>", methods=["POST"])
def cancel_stream(request_id):
    """取消正在进行的 AI 流式生成

    按 request_id 在 active_streams 中找到对应的取消事件并置位，生成器会在
    下一个数据块到达时提前结束并关闭连接。request_id 由 /chat/stream 首帧返回。
    """
    task = active_streams.get(request_id)
    if not task:
        return jsonify({"error": "未找到该请求，可能已完成或已取消",
                        "request_id": request_id}), 404
    task["cancel_event"].set()
    return jsonify({"cancelled": True, "request_id": request_id})


@app.route("/conversations", methods=["GET"])
def get_conversations():
    try:
        db = SessionDB()
        result = db.get_conversations("default_user")
        db.close()
        # 把 ORM 对象转成字典
        data = []
        for conv in result:
            data.append({
                "id": conv.id,
                "user_id": conv.user_id,
                "title": conv.title,
                "created_at": conv.created_at
            })
        return jsonify({"conversations": data})
    except Exception as e:
        print("查询对话列表失败：", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/conversations/<int:conv_id>/messages", methods=["GET"])
def get_messages(conv_id):
    db = SessionDB()
    result = db.get_messages(conv_id)
    db.close()
    return jsonify({"messages": result})

@app.route("/conversations", methods=["POST"])
def create_conversation():
    data = request.get_json()
    title = data.get("title", "新对话")
    db = SessionDB()
    conv_id = db.create_conversation("default_user", title)
    db.close()
    return jsonify({"conversation_id": conv_id, "title": title})

@app.route("/conversations/<int:conv_id>", methods=["PUT"])
def update_conversation(conv_id):
    data = request.get_json()
    title = data.get("title", "")
    if not title:
        return jsonify({"error": "title 字段不能为空"}), 400
    db = SessionDB()
    db.update_conversation_title(conv_id, title)
    db.close()
    return jsonify({"conversation_id": conv_id, "title": title})

@app.route("/conversations/<int:conv_id>", methods=["DELETE"])
def delete_conversation(conv_id):
    db = SessionDB()
    db.delete_conversation(conv_id)
    db.close()
    return jsonify({"conversation_id": conv_id})

@app.route("/messages/<int:conv_id>", methods=["POST"])
def create_message(conv_id):
    data = request.get_json()
    role = data.get("role", "")
    content = data.get("content", "")
    if not role or not content:
        return jsonify({"error": "role 和 content 字段不能为空"}), 400
    db = SessionDB()
    message_id = db.add_message(conv_id, role, content)
    db.close()
    return jsonify({"message_id": message_id, "role": role, "content": content})

if __name__ == "__main__":
    # threaded=True 让 /cancel 在 /chat/stream 推送期间也能被并发处理，
    # 否则取消请求会被占用中的单线程阻塞，无法真正中断生成。
    app.run(debug=True, port=5000, threaded=True)