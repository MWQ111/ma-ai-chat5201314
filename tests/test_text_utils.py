"""modules/text_utils 单元测试：Token 估算"""

from modules.text_utils import estimate_tokens


def test_empty_text():
    """空文本返回 0"""
    assert estimate_tokens("") == 0


def test_pure_cjk():
    """10 个中文字符 ≈ 10 token"""
    assert estimate_tokens("你好世界你好世界你好") == 10


def test_pure_ascii():
    """8 个英文字符 ≈ 2 token"""
    assert estimate_tokens("abcdefgh") == 2


def test_mixed_text():
    """中英混合：2 个中文（2 token）+ 4 个英文（1 token）"""
    assert estimate_tokens("你好abcd") == 3


def test_cjk_punctuation_counts_as_cjk():
    """全角标点按中文计"""
    assert estimate_tokens("你好，世界！") == 6


def test_none_degrades_to_zero():
    """非法输入（None）安全返回 0 而非崩溃"""
    assert estimate_tokens(None) == 0
