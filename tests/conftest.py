"""
pytest 全局夹具
===============
会话文件备份/恢复：每个测试前后还原 session_cache.json，
防止测试（如新建对话、损坏文件降级）污染用户真实数据。
"""

import shutil
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# pytest 控制台脚本（pytest 而非 python -m pytest）不会把当前工作目录加入 sys.path，
# 导致 tests/ 下 `from core import ...` 报 ModuleNotFoundError；
# conftest.py 由 pytest 最先导入，在此统一把项目根目录加入 sys.path。
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 与 app.py / app_api.py 一致：先加载 .env 再导入业务模块。
# core/db.py 在导入时读取 DATABASE_URL，而 pytest 收集测试模块的顺序不定，
# 若某个测试模块先导入 core 就会拿到错误的默认连接串（本地密码与默认值不同）；
# conftest.py 由 pytest 最先导入，在这里加载 .env 可保证测试与真实运行环境一致
# （.env 不存在或 dotenv 不可用时静默跳过，不影响测试收集）。
try:
    from dotenv import load_dotenv, find_dotenv
    _dotenv_path = find_dotenv(usecwd=True, raise_error_if_not_found=False)
    if _dotenv_path:
        load_dotenv(_dotenv_path)
    else:
        load_dotenv()
except Exception:
    pass

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
