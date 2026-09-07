
# core/db.py
import logging
from types import SimpleNamespace

from sqlalchemy import create_engine, Column, Integer, String, Text, TIMESTAMP, ForeignKey, func
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

logger = logging.getLogger("ai_chat.core.db")

# 1. 连接 MySQL
# 把 "你的密码" 改成你自己的 MySQL root 密码
DATABASE_URL = "mysql+pymysql://root:123456mwq@localhost:3306/ma_ai_chat"
USER_ID = "default_user"
engine = create_engine(DATABASE_URL, echo=False)

# 检测 MySQL 是否可用（导入时探测一次；运行中故障由调用方捕获后走 JSON 降级）
_db_available = False
try:
    with engine.connect() as conn:
        _db_available = True
except Exception as e:
    logger.warning("MySQL 连接失败，应用将降级到本地 JSON 存储：%s", e)


def db_available() -> bool:
    """MySQL 是否可用（False 时上层自动走 JSON 降级）"""
    return _db_available


# ====================== 会话列表轻量内存缓存 ======================
# 背景：每次 rerun / 每次 API 请求都全量查库读取对话列表，属于无谓的重复 I/O。
# 这里为“某用户的对话列表”提供一个进程内缓存（按 user_id 存为轻量元组，
# 命中时包装成 SimpleNamespace 返回，避免 ORM 对象跨会话失效问题）。
# 一致性：所有会增删改对话行的路径（create/update/delete/全量清空/增量同步）
# 都会调用 invalidate_conversation_cache() 使缓存失效，保证读到最新数据。
# 注意：缓存仅存在于当前进程内，多进程部署（Flask + Streamlit 各自进程）
# 各维护一份；它只减少重复查询，不承担跨进程强一致。
_conversation_cache: dict[str, list] = {}
_CONVERSATION_CACHE_MAX = 32


def invalidate_conversation_cache(user_id: str | None = None) -> None:
    """使会话列表缓存失效（不传 user_id 时清空全部缓存）"""
    if user_id is None:
        _conversation_cache.clear()
    else:
        _conversation_cache.pop(user_id, None)


def _cache_conversations(user_id: str, rows: list) -> list:
    """把查询结果缓存为轻量元组，返回原始行"""
    if len(_conversation_cache) >= _CONVERSATION_CACHE_MAX:
        _conversation_cache.clear()
    _conversation_cache[user_id] = [
        (r.id, r.user_id, r.title, r.created_at) for r in rows
    ]
    return rows

SessionLocal = sessionmaker(bind=engine)

# 2. 定义 ORM 模型的基类
Base = declarative_base()

# 3. 定义 conversations 表对应的类
class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(50))
    title = Column(String(255))
    created_at = Column(TIMESTAMP, server_default=func.now())

# 4. 定义 messages 表对应的类
class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"))
    role = Column(String(20))
    content = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())

    conversation = relationship("Conversation", backref="messages")

# 5. 数据库操作类
class SessionDB:
    def __init__(self):
        self.session = SessionLocal()

    def create_conversation(self, user_id, title):
        conv = Conversation(user_id=user_id, title=title)
        self.session.add(conv)
        self.session.commit()
        invalidate_conversation_cache(user_id)
        return conv.id

    def add_message(self, conversation_id, role, content):
        msg = Message(conversation_id=conversation_id, role=role, content=content)
        self.session.add(msg)
        self.session.commit()
        return msg.id

    def get_conversations(self, user_id):
        """读取某用户的对话列表（带进程内缓存，见模块级缓存说明）"""
        cached = _conversation_cache.get(user_id)
        if cached is not None:
            return [SimpleNamespace(id=c[0], user_id=c[1], title=c[2], created_at=c[3])
                    for c in cached]
        rows = (self.session.query(Conversation)
                .filter_by(user_id=user_id)
                .order_by(Conversation.id.asc()).all())
        return _cache_conversations(user_id, rows)

    def get_messages(self, conversation_id):
        return self.session.query(Message).filter_by(conversation_id=conversation_id).order_by(Message.id.asc()).all()

    def update_conversation_title(self, conversation_id, title):
        conv = self.session.get(Conversation, conversation_id)
        if conv:
            conv.title = title
            self.session.commit()
            invalidate_conversation_cache()  # 标题变更影响列表展示，整体失效即可

    def delete_conversation(self, conversation_id):
        # 先删消息再删对话，避免 ORM 对子记录做 NULL 化/级联删除带来的约束问题
        self.session.query(Message).filter_by(conversation_id=conversation_id).delete(synchronize_session=False)
        conv = self.session.get(Conversation, conversation_id)
        if conv:
            self.session.delete(conv)
        self.session.commit()
        invalidate_conversation_cache()

    def delete_all_conversations(self, user_id):
        """清空用户的所有对话和消息（先删消息，避免 ORM 级联问题）"""
        ids = [c.id for c in self.session.query(Conversation.id).filter_by(user_id=user_id)]
        if ids:
            self.session.query(Message).filter(Message.conversation_id.in_(ids)).delete(synchronize_session=False)
            self.session.query(Conversation).filter(Conversation.id.in_(ids)).delete(synchronize_session=False)
        self.session.commit()
        invalidate_conversation_cache(user_id)

    def close(self):
        self.session.close()


# 6. 模块级函数（供 session.py / UI 调用；MySQL 不可用或失败时自动降级为空操作）
def load_conversations_from_db(user_id):
    """读取某用户全部对话的 ORM 对象列表（不可用/失败时返回 []）"""
    if not db_available():
        return []
    db = SessionDB()
    try:
        return db.get_conversations(user_id)
    except Exception:
        return []
    finally:
        db.close()


def _sync_messages_incremental(db, conv_id: int, local_msgs: list[dict]) -> None:
    """按“内容前缀匹配”增量同步某个对话下的消息

    只处理 role 为 user/assistant 的消息（与全量版口径一致）。消息在
    界面里只追加、整段清空、不单独编辑/删除，因此远程与本地通常共享
    相同前缀：从首个不一致处起，删除远程多余的消息、追加本地新增的消息，
    中间不重复读写的部分保持原样，避免整条对话重建。

    Args:
        db: SessionDB 实例
        conv_id: 目标对话的数据库主键
        local_msgs: 本地消息列表 [{role, content}, ...]
    """
    remote_msgs = db.get_messages(conv_id)
    n = len(remote_msgs)
    i = 0
    while (i < n and i < len(local_msgs)
           and remote_msgs[i].role == local_msgs[i].get("role")
           and (remote_msgs[i].content or "") == (local_msgs[i].get("content") or "")):
        i += 1
    if i < n:
        # 远程有本地没有（或内容不一致）的尾部消息：先删掉再按本地补齐
        extra_ids = [m.id for m in remote_msgs[i:]]
        db.session.query(Message).filter(Message.id.in_(extra_ids)).delete(synchronize_session=False)
        db.session.commit()
    for m in local_msgs[i:]:
        db.add_message(conv_id, m.get("role"), m.get("content") or "")


def sync_session_to_db(user_id, conversations):
    """把内存中的对话列表增量同步到 MySQL（不再每次全删重建）

    匹配依据是 conversations 元素里的稳定 id：
    - 从 MySQL 载入的对话，其 id 即数据库主键（int），可直接定位对应行并增量更新；
    - 本地新建的对话 id 是字符串时间戳（数据库里没有对应行），插入后把新主键
      回写到该 dict（调用方随后写回内存/JSON，之后即可走增量更新）。

    同步规则（保持原有“UI 内存列表为准”的语义）：
    1. 本地有主键且存在于远程 → 仅按需更新标题，消息做内容前缀 diff，不重建；
    2. 本地没有主键（或主键在远程不存在）→ 新建对话并回写 id，逐条插入消息；
    3. 远程存在但本地列表中没有 → 删除（覆盖“界面上删除了但未同步”等场景）。

    Args:
        user_id: 目标用户
        conversations: UI 结构对话列表，元素为 {id?, name/title, messages: [...]}
    """
    if not db_available():
        return
    db = SessionDB()
    try:
        # 始终读取最新数据做 diff，避免上一轮缓存导致误判删除/重复插入
        invalidate_conversation_cache(user_id)
        remote_by_id = {row.id: row for row in db.get_conversations(user_id)}
        seen_ids: set = set()

        for conv in conversations:
            title = conv.get("title") or conv.get("name") or "对话"
            local_msgs = [
                {"role": m.get("role"), "content": m.get("content") or ""}
                for m in (conv.get("messages") or [])
                if m.get("role") in ("user", "assistant")
            ]
            cid = conv.get("id")
            if isinstance(cid, int) and cid in remote_by_id:
                # 已落库的对话：按需更新标题，消息做增量 diff
                seen_ids.add(cid)
                row = remote_by_id[cid]
                if row.title != title:
                    conv_obj = db.session.get(Conversation, cid)
                    if conv_obj is not None:
                        conv_obj.title = title
                        db.session.commit()
                _sync_messages_incremental(db, cid, local_msgs)
            else:
                # 本地新建对话：插入并把主键回写，方便下一次走增量更新
                new_id = db.create_conversation(user_id, title)
                conv["id"] = new_id
                _sync_messages_incremental(db, new_id, local_msgs)

        # 删除远程有、但本地列表里已不存在的对话（先取快照，避免边遍历边删）
        for rid in list(remote_by_id):
            if rid not in seen_ids:
                db.delete_conversation(rid)
        invalidate_conversation_cache(user_id)
    finally:
        db.close()


def clear_all_conversations_from_db(user_id=USER_ID):
    """清空用户所有对话（MySQL 不可用/失败时为空操作）"""
    if not db_available():
        return
    db = SessionDB()
    try:
        db.delete_all_conversations(user_id)
    except Exception as e:
        logger.warning("清空全部对话失败：%s", e)
    finally:
        db.close()


def clear_conversation_messages_from_db(conversation_id):
    """清空某个对话下的所有消息（MySQL 不可用/无效 id 时为空操作）"""
    if not db_available():
        return
    db = SessionDB()
    try:
        db.session.query(Message).filter_by(conversation_id=conversation_id).delete(synchronize_session=False)
        db.session.commit()
    except Exception as e:
        logger.warning("清空对话消息失败：%s", e)
    finally:
        db.close()


def delete_conversation_from_db(conversation_id):
    """删除某个对话及其所有消息（MySQL 不可用/无效 id 时为空操作）"""
    if not db_available():
        return
    db = SessionDB()
    try:
        db.delete_conversation(conversation_id)
    except Exception as e:
        logger.warning("删除对话失败：%s", e)
    finally:
        db.close()