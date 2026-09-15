"""app.py 应用级集成测试（Streamlit AppTest）

覆盖：深浅双模式渲染无异常、对话搜索过滤、消息长度拦截、
会话原子写入（版本号）、损坏会话文件降级。
所有用例均不触发真实 API 调用（长度拦截在发请求前 st.stop）。
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_FILE = PROJECT_ROOT / "app.py"
SESSION_FILE = PROJECT_ROOT / "session_data" / "session_cache.json"


def _load_app() -> AppTest:
    """启动应用并完成首次渲染"""
    at = AppTest.from_file(str(APP_FILE), default_timeout=60)
    at.run()
    return at


def _all_css(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_dark_mode_renders_without_exception():
    at = _load_app()
    assert not at.exception
    css = _all_css(at)
    assert "stAppViewContainer" in css and "#0e1117" in css


def test_light_mode_renders_without_exception():
    at = _load_app()
    at.toggle(key="dark_mode").set_value(False)
    at.run()
    assert not at.exception
    css = _all_css(at)
    assert "#f5f7fb" in css and "#1a1a2e" in css


SEARCH_TEST_TITLE = "zzz_pytest_搜索用例专用标题_zzz"


def test_conversation_search_filter():
    """搜索按名称过滤对话。

    不依赖存储中已有的历史数据：用例内先新建一个标题已知的对话再搜索，
    结束后删除，避免污染真实会话数据（MySQL 可用时会真实落库）。
    """
    at = _load_app()

    # 1. 新建对话（新建后自动成为当前对话）
    for b in at.get("button"):
        if b.label == "➕ 新建":
            b.click()
            break
    at.run()
    assert not at.exception

    conv_id = at.session_state["conversations"][-1]["id"]

    # 2. 重命名为用例专属标题（当前对话的名称是可编辑输入框）
    at.text_input(key=f"conv_name_edit_{conv_id}").set_value(SEARCH_TEST_TITLE)
    at.run()
    assert not at.exception

    try:
        # 3. 命中：搜索该标题，不应出现「未找到」提示
        at.text_input(key="conv_search").set_value(SEARCH_TEST_TITLE)
        at.run()
        assert not any("未找到匹配的对话" in c.value for c in at.caption)

        # 4. 未命中：搜索不存在的关键词，应出现提示且不抛异常
        at.text_input(key="conv_search").set_value("zzz不存在的关键词zzz")
        at.run()
        assert any("未找到匹配的对话" in c.value for c in at.caption)
        assert not at.exception
    finally:
        # 5. 清理：先清空搜索（被过滤掉的对话不渲染删除按钮），再删除本用例新建的对话
        at.text_input(key="conv_search").set_value("")
        at.run()
        idx = next(
            (i for i, c in enumerate(at.session_state["conversations"])
             if c.get("id") == conv_id),
            None,
        )
        if idx is not None:
            at.button(key=f"del_conv_{idx}").click()
            at.run()


def test_delete_conversation_keeps_theme_mode():
    """删除对话不应改变当前主题模式。

    回归用例：早期实现在 app.py 顶层用 _need_rerun 标记提前 st.rerun()，该中断
    发生在 render_sidebar() 之前，导致本轮未渲染的 dark_mode 开关其 session_state
    被回收，下一次运行回落为 AppConfig.DEFAULT_DARK_MODE（True）—— 表现为浅色
    模式下删除对话后界面跳回夜间模式。

    自建自删：只操作本用例创建的对话，不触碰既有数据。
    """
    at = _load_app()
    at.toggle(key="dark_mode").set_value(False)
    at.run()
    assert at.session_state["dark_mode"] is False

    # 新建一条用例专属对话（新建后自动成为当前对话，位于列表末尾）
    for b in at.get("button"):
        if b.label == "➕ 新建":
            b.click()
            break
    at.run()
    assert not at.exception
    conv_id = at.session_state["conversations"][-1]["id"]

    # 走真实 on_click 回调删除它
    idx = next(i for i, c in enumerate(at.session_state["conversations"])
               if c.get("id") == conv_id)
    at.button(key=f"del_conv_{idx}").click()
    at.run()

    assert not at.exception
    assert at.session_state["dark_mode"] is False  # 主题模式未被重置
    assert not any(c.get("id") == conv_id for c in at.session_state["conversations"])


def test_message_length_limit():
    """超过 10000 字符的消息被拦截，且不发 API 请求"""
    at = _load_app()
    at.chat_input[0].set_value("长" * 10001)
    at.run()
    assert any("消息过长" in w.value for w in at.warning)
    assert not at.exception


def test_new_conversation_persisted_with_version():
    """新建对话触发原子写入：带版本号、无 .tmp 残留、新对话出现在文件中。

    自建自删：用例结束必须删掉自己新建的对话。conftest 的夹具只备份/还原
    JSON 文件，还原不了 MySQL —— 若不清理，每跑一次就往真实库里留一条空对话。
    """
    at = _load_app()
    before = 1
    before_names = set()
    if SESSION_FILE.exists():
        with open(SESSION_FILE, encoding="utf-8") as f:
            conversations = json.load(f).get("conversations", [])
            before = len(conversations)
            before_names = {c.get("name") for c in conversations}
    for b in at.get("button"):
        if b.label == "➕ 新建":
            b.click()
            break
    at.run()
    assert not at.exception
    conv_id = at.session_state["conversations"][-1]["id"]

    try:
        assert SESSION_FILE.exists()
        with open(SESSION_FILE, encoding="utf-8") as f:
            saved = json.load(f)
        saved_conversations = saved.get("conversations", [])
        assert saved.get("version") == 2
        # 应用启动会把 MySQL 中已有的会话并入 session_state（文件可能滞后于库），
        # 因此只断言「数量不少于 +1」且「出现了一个新对话」，不做精确计数
        assert len(saved_conversations) >= before + 1
        assert before_names < {c.get("name") for c in saved_conversations}
        assert not Path(str(SESSION_FILE) + ".tmp").exists()
    finally:
        # 清理：删除本用例新建的对话。走应用自身的删除回调，它会同时清掉
        # MySQL 行（delete_conversation_from_db）与 JSON 备份（save_session_to_file）
        idx = next(
            (i for i, c in enumerate(at.session_state["conversations"])
             if c.get("id") == conv_id),
            None,
        )
        if idx is not None:
            at.button(key=f"del_conv_{idx}").click()
            at.run()


def test_corrupted_session_file_degrades():
    """损坏的会话文件不拖垮应用：自动回退默认会话（夹具负责恢复原文件）"""
    SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)
    SESSION_FILE.write_text("{broken json", encoding="utf-8")
    try:
        at = _load_app()
        assert not at.exception
        assert at.session_state["conversations"]  # 已回退到默认会话
    finally:
        SESSION_FILE.unlink(missing_ok=True)
