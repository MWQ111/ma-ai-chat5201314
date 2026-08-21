"""modules/cache 单元测试：缓存键规则 + 无 Redis 时的静默降级"""

from modules.cache import (
    _make_key,
    get_cache_status,
    get_cached_response,
    set_cached_response,
)


def test_make_key_deterministic_and_model_separated():
    k1 = _make_key("你好", "deepseek-chat")
    k2 = _make_key("你好", "deepseek-chat")
    k3 = _make_key("你好", "deepseek-reasoner")
    assert k1 == k2 != k3  # 同模型同问题键一致；跨模型隔离
    assert _make_key(" 你好 ", "m") == _make_key("你好", "m")  # 首尾空白不影响


def test_get_cache_degrades_without_redis():
    """Redis 未启动时返回 None（首次可用性探测可能耗时 1-2 秒，属预期）"""
    assert get_cached_response("任意问题", model="deepseek-chat") is None


def test_set_cache_degrades_silently():
    """写入失败静默跳过，绝不抛异常"""
    set_cached_response("任意问题", "回答", model="deepseek-chat")


def test_get_cache_status_degrades():
    """状态接口始终返回 (说明, 级别)，级别为已知枚举值"""
    _, level = get_cache_status()
    assert level in ("ok", "no_lib", "unreachable")
