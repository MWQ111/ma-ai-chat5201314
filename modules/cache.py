"""
Redis 缓存模块
==============
以用户问题为 key（MD5 哈希）缓存 AI 回答，减少重复 API 调用，降低成本和延迟。

配置（全部通过环境变量，均有默认值）：
- REDIS_HOST：Redis 地址，默认 127.0.0.1（避免 Windows 下 localhost
  解析出 IPv6 地址导致连接挂起翻倍）
- REDIS_PORT：Redis 端口，默认 6379
- REDIS_PASSWORD：Redis 密码，默认无
- REDIS_DB：Redis 数据库编号，默认 0
- CACHE_TTL：缓存过期秒数，默认 3600（1 小时）

启动 Redis（Docker，一行命令）：
    docker run -d --name redis -p 6379:6379 redis:7

降级策略：Redis 未安装或连接失败时，所有读写静默跳过，
主应用流程完全不受影响（仅少一次缓存加速）。
每次可用性检查有硬超时保护（守护线程 ping，超时即判不可用），
检查失败后冷却时间指数退避，绝不阻塞主流程。
"""

import hashlib
import os
import threading
import time

try:
    import redis as redis_lib
    REDIS_LIB_AVAILABLE = True
except ImportError:
    REDIS_LIB_AVAILABLE = False

# ====================== 常量配置 ======================
KEY_PREFIX = "ai_cache"                                    # 缓存键前缀（避免与其它应用冲突）
CACHE_TTL = int(os.environ.get("CACHE_TTL", 3600))         # 缓存过期秒数，默认 1 小时
CHECK_INTERVAL = 30                                        # 检查成功时的冷却时间（秒）
FAIL_COOLDOWN_MAX = 1800                                   # 检查失败时冷却时间的上限（30 分钟）
PING_DEADLINE = 1.5                                        # ping 硬超时（秒）：超过即判定不可用

_client = None                          # Redis 客户端（懒加载单例）
# 可用性缓存：available 最近一次检查结果；checked_at 上次检查时间；
# cooldown 当前冷却时长（失败时指数退避增长，成功后恢复默认值）
_availability = {"available": False, "checked_at": 0.0, "cooldown": CHECK_INTERVAL}


# ====================== 内部辅助函数 ======================
def _get_client():
    """获取 Redis 客户端（懒加载单例，创建失败返回 None）

    Returns:
        redis.Redis 或 None: 客户端对象；redis 库未安装/参数非法时返回 None
    """
    global _client
    if _client is not None:
        return _client
    if not REDIS_LIB_AVAILABLE:
        return None
    try:
        _client = redis_lib.Redis(
            host=os.environ.get("REDIS_HOST", "127.0.0.1"),
            port=int(os.environ.get("REDIS_PORT", "6379")),
            password=os.environ.get("REDIS_PASSWORD") or None,
            db=int(os.environ.get("REDIS_DB", "0")),
            socket_connect_timeout=1,   # 连接超时 1 秒，失败快速降级
            socket_timeout=1,           # 读写超时 1 秒
            decode_responses=True,      # 直接返回字符串
        )
    except Exception:
        _client = None
    return _client


def _make_key(question, model):
    """生成缓存键：应用前缀 + 模型名 + 问题（去除首尾空白后）的 MD5 哈希

    带模型名是为了防止不同模型对同一问题的回答互相污染。

    Args:
        question: 用户问题原文
        model: 当前模型名

    Returns:
        str: 缓存键
    """
    digest = hashlib.md5(question.strip().encode("utf-8")).hexdigest()
    return f"{KEY_PREFIX}:{model}:{digest}"


# ====================== 对外接口 ======================
def _ping_with_deadline(client):
    """在守护线程中执行 Redis ping，超过硬超时即判定不可用

    背景：Windows 上连接无监听的端口时，防火墙可能丢弃 SYN 包导致连接
    长时间挂起（实测单地址 15 秒、localhost 双地址 26 秒），redis-py 的
    socket_connect_timeout 无法有效中断该过程。线程化 + 硬超时保证调用方
    最多等待 PING_DEADLINE 秒。

    Args:
        client: redis 客户端对象

    Returns:
        bool: 硬超时内 ping 成功返回 True，否则 False
    """
    result = []

    def _worker():
        try:
            result.append(bool(client.ping()))
        except Exception:
            result.append(False)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(PING_DEADLINE)
    return result[0] if result else False


def is_redis_available():
    """检查 Redis 是否可用（带冷却时间，避免每次请求都 ping）

    冷却策略：
    - 检查成功 → 30 秒后重新检查（保持对宕机的感知）
    - 检查失败 → 冷却时间指数退避（30 秒 → 2 分钟 → 8 分钟 → 最多 30 分钟），
      避免 Redis 宕机或网络异常时反复拖慢对话
    每次检查本身有硬超时保护（守护线程 ping，见 _ping_with_deadline）。

    Returns:
        bool: True 表示 Redis 已连接可用
    """
    if not REDIS_LIB_AVAILABLE:
        return False
    now = time.time()
    if now - _availability["checked_at"] < _availability["cooldown"]:
        return _availability["available"]
    try:
        client = _get_client()
        if client is None or not _ping_with_deadline(client):
            raise ConnectionError("Redis ping 失败")
        _availability["available"] = True
        _availability["cooldown"] = CHECK_INTERVAL  # 恢复后回到正常检查频率
    except Exception:
        _availability["available"] = False
        _availability["cooldown"] = min(
            _availability["cooldown"] * 4, FAIL_COOLDOWN_MAX)  # 失败退避
        _client = None  # 重置客户端，下次检查时重建连接
    _availability["checked_at"] = now
    return _availability["available"]


def get_cached_response(question, model="default"):
    """获取缓存的 AI 回答

    Args:
        question: 用户问题原文
        model: 当前模型名（参与缓存键，防止跨模型混用）

    Returns:
        str 或 None: 命中时返回缓存的回答文本；未命中/Redis 不可用时返回 None
    """
    if not is_redis_available():
        return None
    try:
        return _get_client().get(_make_key(question, model))
    except Exception:
        return None


def set_cached_response(question, response, model="default"):
    """缓存 AI 回答（默认 1 小时过期），失败时静默跳过

    Args:
        question: 用户问题原文
        response: AI 的完整回答文本
        model: 当前模型名
    """
    if not is_redis_available() or not response:
        return
    try:
        _get_client().setex(_make_key(question, model), CACHE_TTL, response)
    except Exception:
        pass  # 缓存写入失败不影响主流程


def clear_cache():
    """清空本应用的全部缓存（按前缀匹配删除）

    Returns:
        int: 删除的缓存条数（Redis 不可用时返回 0）
    """
    if not is_redis_available():
        return 0
    try:
        client = _get_client()
        keys = list(client.scan_iter(match=f"{KEY_PREFIX}:*"))
        if keys:
            client.delete(*keys)
        return len(keys)
    except Exception:
        return 0


def get_cache_status():
    """获取缓存服务状态（供界面展示）

    Returns:
        tuple: (状态说明文字, 状态级别)，级别取值：
               "ok"          Redis 已连接，缓存生效
               "no_lib"      redis 库未安装
               "unreachable" Redis 未启动或连接失败（已自动降级）
    """
    if not REDIS_LIB_AVAILABLE:
        return "未安装 redis 库，缓存不可用（pip install redis）", "no_lib"
    if is_redis_available():
        host = os.environ.get("REDIS_HOST", "127.0.0.1")
        port = os.environ.get("REDIS_PORT", "6379")
        return f"Redis 已连接（{host}:{port}），缓存生效中", "ok"
    return "Redis 不可用，已自动跳过缓存", "unreachable"
