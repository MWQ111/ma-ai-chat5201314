"""
pytest 全局夹具
===============
会话文件备份/恢复：每个测试前后还原 session_cache.json，
防止测试（如新建对话、损坏文件降级）污染用户真实数据。
"""

import shutil
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SESSION_FILE = PROJECT_ROOT / "session_data" / "session_cache.json"


@pytest.fixture(autouse=True)
def _protect_session_file():
    """测试前后备份/恢复会话文件；原本不存在时测试结束后清理"""
    backup = SESSION_FILE.with_name(SESSION_FILE.name + ".bak_test")
    had_session = SESSION_FILE.exists()
    if had_session:
        shutil.copy2(SESSION_FILE, backup)
    yield
    if had_session:
        shutil.copy2(backup, SESSION_FILE)
        backup.unlink()
    elif SESSION_FILE.exists():
        SESSION_FILE.unlink()
