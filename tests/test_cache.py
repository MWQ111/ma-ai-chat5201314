"""core/cache 单元测试：缓存键规则 + 无 Redis 时的静默降级 + 语义缓存纯函数"""

from core.cache import (
    _cosine_similarity,
    _encode_cache_value,
    _extract_answer,
    _make_key,
    find_similar_in_cache,
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


def test_make_key_invalidates_on_prompt_and_rag_version_change():
    """提示词或知识库版本变化后，同一问题的缓存键必须改变（旧缓存自然失效）"""
    import core.cache as cache_mod
    old_hash, old_version = cache_mod.system_prompt_hash, cache_mod.RAG_VERSION
    try:
        k_base = _make_key("你好", "m")
        cache_mod.update_system_prompt_hash("自定义提示词")
        k_prompt = _make_key("你好", "m")
        cache_mod.bump_rag_version()
        k_version = _make_key("你好", "m")
        assert k_base != k_prompt != k_version  # 任一因子变化，键都不同
        # 因子不变时键保持确定，且跨模型依然隔离
        assert _make_key("你好", "m") == k_version
        assert _make_key("你好", "n") != k_version
    finally:
        cache_mod.system_prompt_hash = old_hash
        cache_mod.RAG_VERSION = old_version


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


# ====================== 语义缓存（纯函数，不依赖 Redis/嵌入模型） ======================
def test_cosine_similarity_math():
    """余弦相似度计算：同向为 1、正交为 0、异常输入为 0"""
    assert abs(_cosine_similarity([1.0, 0.0], [1.0, 0.0]) - 1.0) < 1e-9
    assert abs(_cosine_similarity([1.0, 0.0], [0.0, 1.0])) < 1e-9  # 正交 → 0
    assert abs(_cosine_similarity([1.0, 1.0], [2.0, 2.0]) - 1.0) < 1e-9  # 同方向不同模长 → 1
    assert abs(_cosine_similarity([1.0, 2.0], [2.0, 1.0]) - 0.8) < 1e-9  # (1*2+2*1)/(√5·√5)=0.8
    assert _cosine_similarity([], [1.0, 2.0]) == 0.0    # 空向量
    assert _cosine_similarity([1.0], [1.0, 2.0]) == 0.0  # 维度不一致
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0  # 零向量


def test_find_similar_in_cache_returns_best_above_threshold():
    """相似度达标才命中，且返回最相似的一条"""
    entries = [
        {"answer": "关于天气的回答", "embedding": [1.0, 0.0], "model": "m"},
        {"answer": "关于代码的回答", "embedding": [0.9, 0.44], "model": "m"},  # 相似度约 0.898
        {"answer": "无关回答", "embedding": [0.0, 1.0], "model": "m"},          # 正交
    ]
    # 阈值 0.9：只有第一条达标
    hit = find_similar_in_cache([1.0, 0.0], entries, threshold=0.9)
    assert hit is not None and hit["answer"] == "关于天气的回答"
    # 当前问题与所有条目都不相似（反向/正交）→ None
    assert find_similar_in_cache([-1.0, 0.0], entries, threshold=0.9) is None
    # 条目缺 embedding（旧版纯文本不会进入候选，这里验证不报错）
    assert find_similar_in_cache([1.0, 0.0], [{"answer": "x"}], threshold=0.5) is None
    # 空候选 → None
    assert find_similar_in_cache([1.0, 0.0], [], threshold=0.5) is None


def test_cache_value_format_roundtrip_and_legacy_compat():
    """新版 JSON 值可正确取出回答；旧版纯文本值原样兼容"""
    payload = _encode_cache_value("语义缓存的回答", [0.1, 0.2, 0.3], "deepseek-chat")
    assert _extract_answer(payload) == "语义缓存的回答"
    assert _extract_answer("旧版纯文本回答") == "旧版纯文本回答"  # 旧格式原样返回
    assert _extract_answer(None) is None
