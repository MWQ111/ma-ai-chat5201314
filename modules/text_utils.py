"""
文本工具模块
============
提供与 AI 文本处理相关的纯函数工具（无第三方依赖、无副作用，便于单元测试）。

设计要点：
- Token 估算采用"中文 1 字 ≈ 1 token、其他 4 字符 ≈ 1 token"的工程近似，
  用于界面展示用量统计，不追求与官方分词完全一致
"""

import re

# 中文及全角符号字符区间：CJK 统一表意文字（含扩展）、中文标点、全角字符
_CJK_RE = re.compile(r"[一-鿿　-〿＀-￯]")


def estimate_tokens(text: str) -> int:
    """粗略估算文本 Token 数（中文字符约 1 token/字，其他字符约 4 字符/token）

    Args:
        text: 待估算的文本

    Returns:
        int: 估算的 Token 数；空文本返回 0
    """
    if not text:
        return 0
    cjk_chars = len(_CJK_RE.findall(text))
    other_chars = len(text) - cjk_chars
    return cjk_chars + other_chars // 4
