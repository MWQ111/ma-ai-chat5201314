"""app_api.py 集成测试（Flask test_client）

不启动真实 HTTP 服务，也**不依赖真实的 AI API / MySQL / Redis**：
- ``call_ai_api_simple`` / ``call_ai_api_simple_stream`` 被替换为确定性的假实现；
- ``SessionDB`` 被替换为内存假实现（conversations 列表 + messages 列表）。

响应状态码以「实测行为」为准，几处与直觉不同的地方已单独标注：
- ``GET /ask`` 不带 q 参数时返回 200 + 提示文本，而非 400；
- 请求体**完全缺失**（无 JSON）时 Flask 在进入视图前即以 415 拒绝；
  只有带合法 JSON body 但**缺字段**时，才由视图返回 400。
"""

import pytest

import app_api

MOCK_AI_REPLY = "【mock 回复】这是测试替身，不会发起真实请求"
CREATED_AT = "2026-01-01 00:00:00"


class _Conversation:
    """替身 ORM 对象：字段口径与 core/db.py 中的会话模型一致"""

    def __init__(self, conv_id, user_id, title, created_at):
        self.id = conv_id
        self.user_id = user_id
        self.title = title
        self.created_at = created_at


class _Message:
    """替身 ORM 对象：字段口径与 core/db.py 中的消息模型一致"""

    def __init__(self, msg_id, conversation_id, role, content, created_at):
        self.id = msg_id
        self.conversation_id = conversation_id
        self.role = role
        self.content = content
        self.created_at = created_at


@pytest.fixture
def fake_store():
    """内存存储，替代真实 MySQL 表（每个用例独立，避免相互污染）"""
    return {"conversations": [], "messages": [], "next_conv_id": 1, "next_msg_id": 1}


@pytest.fixture
def ai_calls(monkeypatch):
    """替换 AI 调用为确定返回值，并记录每次调用的入参"""
    calls = []

    def _fake_call_ai(messages, *args, **kwargs):
        calls.append(messages)
        return MOCK_AI_REPLY

    monkeypatch.setattr(app_api, "call_ai_api_simple", _fake_call_ai)
    return calls


@pytest.fixture
def api_client(monkeypatch, fake_store, ai_calls):
    """已注入假 DB 与假 AI 的 Flask test_client"""

    class FakeSessionDB:
        """内存假实现：接口与真实 SessionDB 对齐，每次请求都会被 new 一个，
        故状态放在闭包 fake_store 中跨实例共享"""

        def get_conversations(self, user_id):
            return [c for c in fake_store["conversations"] if c.user_id == user_id]

        def get_messages(self, conversation_id):
            return [m for m in fake_store["messages"] if m.conversation_id == conversation_id]

        def create_conversation(self, user_id, title):
            conv_id = fake_store["next_conv_id"]
            fake_store["next_conv_id"] += 1
            fake_store["conversations"].append(_Conversation(conv_id, user_id, title, CREATED_AT))
            return conv_id

        def add_message(self, conversation_id, role, content):
            msg_id = fake_store["next_msg_id"]
            fake_store["next_msg_id"] += 1
            fake_store["messages"].append(_Message(msg_id, conversation_id, role, content, CREATED_AT))
            return msg_id

        def update_conversation_title(self, conversation_id, title):
            for conv in fake_store["conversations"]:
                if conv.id == conversation_id:
                    conv.title = title

        def delete_conversation(self, conversation_id):
            fake_store["conversations"] = [c for c in fake_store["conversations"] if c.id != conversation_id]
            fake_store["messages"] = [m for m in fake_store["messages"] if m.conversation_id != conversation_id]

        def close(self):
            pass

    monkeypatch.setattr(app_api, "SessionDB", FakeSessionDB)
    app_api.app.config["TESTING"] = True
    with app_api.app.test_client() as client:
        yield client


def _new_conversation(client, title="用例对话"):
    """建一条对话并返回其 id"""
    resp = client.post("/conversations", json={"title": title})
    assert resp.status_code == 200
    return resp.get_json()["conversation_id"]


class _BrokenDB:
    """构造即抛异常的假 DB，用于验证接口的 500 降级分支"""

    def __init__(self):
        raise RuntimeError("模拟数据库连接失败")


# ====================== 健康检查 ======================
def test_index_returns_running_banner(api_client):
    resp = api_client.get("/")
    assert resp.status_code == 200
    assert "AI API 已启动" in resp.get_data(as_text=True)


def test_ping_returns_pong(api_client):
    resp = api_client.get("/ping")
    assert resp.status_code == 200
    assert resp.get_data(as_text=True) == "pong"


# ====================== GET /ask ======================
def test_ask_without_q_returns_hint_not_error(api_client):
    """当前实现：不带 q 时返回 200 + 提示文本（不是 400）"""
    resp = api_client.get("/ask")
    assert resp.status_code == 200
    assert "q 参数" in resp.get_data(as_text=True)


def test_ask_with_q_returns_ai_reply(api_client, ai_calls):
    resp = api_client.get("/ask?q=你好")
    assert resp.status_code == 200
    assert resp.get_data(as_text=True) == MOCK_AI_REPLY
    assert ai_calls == [[{"role": "user", "content": "你好"}]]


# ====================== GET /conversations ======================
def test_list_conversations_empty(api_client):
    resp = api_client.get("/conversations")
    assert resp.status_code == 200
    assert resp.get_json() == {"conversations": []}


def test_list_conversations_structure(api_client):
    _new_conversation(api_client, "列表用例")
    resp = api_client.get("/conversations")
    assert resp.status_code == 200
    data = resp.get_json()["conversations"]
    assert len(data) == 1
    assert set(data[0]) == {"id", "user_id", "title", "created_at"}
    assert data[0]["title"] == "列表用例"
    assert data[0]["user_id"] == "default_user"


def test_list_conversations_db_error_returns_500(api_client, monkeypatch):
    monkeypatch.setattr(app_api, "SessionDB", _BrokenDB)
    resp = api_client.get("/conversations")
    assert resp.status_code == 500
    assert "error" in resp.get_json()


# ====================== POST /conversations ======================
def test_create_conversation_ok(api_client):
    resp = api_client.post("/conversations", json={"title": "新建用例"})
    assert resp.status_code == 200
    assert resp.get_json() == {"conversation_id": 1, "title": "新建用例"}


def test_create_conversation_without_title_uses_default(api_client):
    resp = api_client.post("/conversations", json={})
    assert resp.status_code == 200
    assert resp.get_json()["title"] == "新对话"


def test_create_conversation_persists(api_client):
    conv_id = _new_conversation(api_client)
    listed = api_client.get("/conversations").get_json()["conversations"]
    assert [c["id"] for c in listed] == [conv_id]


def test_create_conversation_without_json_body_returns_415(api_client):
    """无 JSON body 时 Flask 在进入视图前即拒绝（415），属框架行为而非视图逻辑"""
    resp = api_client.post("/conversations")
    assert resp.status_code == 415


# ====================== GET /conversations/<id>/messages ======================
def test_get_messages_empty_for_new_conversation(api_client):
    conv_id = _new_conversation(api_client)
    resp = api_client.get(f"/conversations/{conv_id}/messages")
    assert resp.status_code == 200
    assert resp.get_json() == {"messages": []}


def test_get_messages_structure(api_client):
    conv_id = _new_conversation(api_client)
    api_client.post(f"/messages/{conv_id}", json={"role": "user", "content": "你好"})
    resp = api_client.get(f"/conversations/{conv_id}/messages")
    assert resp.status_code == 200
    messages = resp.get_json()["messages"]
    assert len(messages) == 1
    assert set(messages[0]) == {"id", "conversation_id", "role", "content", "created_at"}
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "你好"
    assert messages[0]["conversation_id"] == conv_id


def test_get_messages_db_error_returns_500(api_client, monkeypatch):
    monkeypatch.setattr(app_api, "SessionDB", _BrokenDB)
    resp = api_client.get("/conversations/1/messages")
    assert resp.status_code == 500
    assert "error" in resp.get_json()


# ====================== POST /messages/<id> ======================
def test_create_message_ok(api_client):
    conv_id = _new_conversation(api_client)
    resp = api_client.post(f"/messages/{conv_id}", json={"role": "assistant", "content": "回复内容"})
    assert resp.status_code == 200
    assert resp.get_json() == {"message_id": 1, "role": "assistant", "content": "回复内容"}


@pytest.mark.parametrize("payload", [
    {"content": "缺 role"},
    {"role": "user"},
    {},
    {"role": "", "content": "空 role"},
    {"role": "user", "content": ""},
])
def test_create_message_invalid_payload_returns_400(api_client, payload):
    conv_id = _new_conversation(api_client)
    resp = api_client.post(f"/messages/{conv_id}", json=payload)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_create_message_without_json_body_returns_415(api_client):
    resp = api_client.post("/messages/1")
    assert resp.status_code == 415


# ====================== PUT / DELETE /conversations/<id> ======================
def test_update_conversation_ok(api_client):
    conv_id = _new_conversation(api_client, "旧标题")
    resp = api_client.put(f"/conversations/{conv_id}", json={"title": "新标题"})
    assert resp.status_code == 200
    assert resp.get_json() == {"conversation_id": conv_id, "title": "新标题"}
    listed = api_client.get("/conversations").get_json()["conversations"]
    assert listed[0]["title"] == "新标题"


def test_update_conversation_missing_title_returns_400(api_client):
    conv_id = _new_conversation(api_client)
    resp = api_client.put(f"/conversations/{conv_id}", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_delete_conversation_ok(api_client):
    conv_id = _new_conversation(api_client)
    resp = api_client.delete(f"/conversations/{conv_id}")
    assert resp.status_code == 200
    assert resp.get_json() == {"conversation_id": conv_id}
    assert api_client.get("/conversations").get_json()["conversations"] == []


# ====================== POST /chat ======================
def test_chat_ok(api_client, ai_calls):
    resp = api_client.post("/chat", json={"message": "你好"})
    assert resp.status_code == 200
    assert resp.get_json() == {"response": MOCK_AI_REPLY, "user_message": "你好"}
    assert ai_calls == [[{"role": "user", "content": "你好"}]]


@pytest.mark.parametrize("payload", [{}, {"message": ""}])
def test_chat_invalid_payload_returns_400(api_client, payload):
    resp = api_client.post("/chat", json=payload)
    assert resp.status_code == 400
    assert "error" in resp.get_json()


def test_chat_without_json_body_returns_415(api_client):
    resp = api_client.post("/chat")
    assert resp.status_code == 415


# ====================== POST /cancel/<request_id> ======================
def test_cancel_unknown_request_returns_404(api_client):
    resp = api_client.post("/cancel/no-such-request")
    assert resp.status_code == 404
    assert "error" in resp.get_json()


# ====================== POST /chat/stream（SSE） ======================
def test_chat_stream_emits_sse_events(api_client, monkeypatch):
    def _fake_stream(messages, cancel_event=None, **kwargs):
        yield from ("你", "好")

    monkeypatch.setattr(app_api, "call_ai_api_simple_stream", _fake_stream)

    resp = api_client.post("/chat/stream", json={"message": "hi"})
    assert resp.status_code == 200
    assert resp.mimetype == "text/event-stream"
    body = resp.get_data(as_text=True)
    assert "event: start" in body        # 首帧下发 request_id
    assert '"delta": "你"' in body        # 分块推送，而非一次性文本
    assert '"delta": "好"' in body
    assert "event: done" in body
    assert '"cancelled": false' in body


def test_chat_stream_without_message_returns_400(api_client):
    resp = api_client.post("/chat/stream", json={})
    assert resp.status_code == 400
    assert "error" in resp.get_json()
