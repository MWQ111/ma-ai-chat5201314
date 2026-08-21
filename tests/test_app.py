"""06.py 应用级集成测试（Streamlit AppTest）

覆盖：深浅双模式渲染无异常、对话搜索过滤、消息长度拦截、
会话原子写入（版本号）、损坏会话文件降级。
所有用例均不触发真实 API 调用（长度拦截在发请求前 st.stop）。
"""

import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_FILE = PROJECT_ROOT / "06.py"
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


def test_conversation_search_filter():
    at = _load_app()
    # 命中：默认对话名包含"默认"，不应出现未找到提示
    at.text_input(key="conv_search").set_value("默认")
    at.run()
    assert not any("未找到匹配的对话" in c.value for c in at.caption)
    # 未命中：显示提示且不抛异常
    at.text_input(key="conv_search").set_value("zzz不存在的关键词zzz")
    at.run()
    assert any("未找到匹配的对话" in c.value for c in at.caption)
    assert not at.exception


def test_message_length_limit():
    """超过 10000 字符的消息被拦截，且不发 API 请求"""
    at = _load_app()
    at.chat_input[0].set_value("长" * 10001)
    at.run()
    assert any("消息过长" in w.value for w in at.warning)
    assert not at.exception


def test_new_conversation_persisted_with_version():
    """新建对话触发原子写入：带版本号、无 .tmp 残留、数量 +1"""
    at = _load_app()
    before = 1
    if SESSION_FILE.exists():
        with open(SESSION_FILE, encoding="utf-8") as f:
            before = len(json.load(f).get("conversations", []))
    for b in at.get("button"):
        if b.label == "➕ 新建":
            b.click()
            break
    at.run()
    assert not at.exception
    assert SESSION_FILE.exists()
    with open(SESSION_FILE, encoding="utf-8") as f:
        saved = json.load(f)
    assert saved.get("version") == 2
    assert len(saved.get("conversations", [])) == before + 1
    assert not Path(str(SESSION_FILE) + ".tmp").exists()


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
