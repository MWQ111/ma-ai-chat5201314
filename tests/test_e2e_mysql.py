"""端到端测试：MySQL 存储路径真实跑通（插入 → 查询验证 → 清理）

证明「MySQL 这条路径真的跑通」的直接证据：不 mock、不降级，对真实数据库
执行完整的写入 / 读取 / 删除闭环。MySQL 不可用时自动跳过（跳过不算失败），
本地 JSON 降级环境下测试套件依然绿色。
"""

import uuid

import pytest

from core.db import USER_ID, SessionDB, db_available

# 测试数据使用唯一标题，避免与真实数据混淆；无论断言成败都会在 finally 中清理
TEST_TITLE = f"_e2e_pytest_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def db():
    """提供真实 SessionDB 连接；MySQL 不可用时跳过用例"""
    if not db_available():
        pytest.skip("MySQL 不可用（连接失败或表缺失），跳过端到端测试")
    db = SessionDB()
    yield db
    db.close()


def test_mysql_conversation_and_message_roundtrip(db):
    """端到端闭环：创建对话 → 写入消息 → 查询校验 → 删除对话 → 确认清理"""
    conv_id = None
    try:
        # 1. 插入一条测试对话
        conv_id = db.create_conversation(USER_ID, TEST_TITLE)
        assert isinstance(conv_id, int)

        # 2. 插入两条测试消息（一问一答）
        msg1_id = db.add_message(conv_id, "user", "端到端测试：你好")
        msg2_id = db.add_message(conv_id, "assistant", "端到端测试：收到！")
        assert msg1_id and msg2_id

        # 3. 查询并验证消息：顺序、角色、内容与写入一致
        msgs = db.get_messages(conv_id)
        assert [m.role for m in msgs] == ["user", "assistant"]
        assert msgs[0].content == "端到端测试：你好"
        assert msgs[1].content == "端到端测试：收到！"

        # 4. 查询对话列表：包含新对话且标题一致（同时验证列表缓存在写入后已失效）
        titles = {c.title for c in db.get_conversations(USER_ID)}
        assert TEST_TITLE in titles
    finally:
        # 5. 清理测试数据（断言失败也要清理，避免污染真实库；
        #    清理异常不掩盖真实断言结果）
        if conv_id is not None:
            try:
                db.delete_conversation(conv_id)
            except Exception:
                pass

    # 6. 确认清理干净（delete 会失效列表缓存，此处读到的是最新数据）
    titles = {c.title for c in db.get_conversations(USER_ID)}
    assert TEST_TITLE not in titles
