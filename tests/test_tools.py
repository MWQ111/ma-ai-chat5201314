"""modules/tools 单元测试：安全计算器、时间工具与工具执行器"""

import pytest

from modules.tools import (
    _preprocess_expression,
    calculate,
    execute_tool,
    get_available_tools,
    get_current_time,
    get_tool_names,
)


class TestCalculate:
    """安全数学计算：正确性 + 注入防护（AST 白名单）"""

    def test_basic_arithmetic(self):
        assert calculate("2+3*4") == "14"
        assert calculate("(1+2)*3") == "9"
        assert calculate("10/4") == "2.5"
        assert calculate("2**10") == "1024"

    def test_sqrt(self):
        assert calculate("√16") == "4"
        assert calculate("√(1+3)") == "2"

    def test_division_by_zero(self):
        assert "除数不能为 0" in calculate("1/0")

    def test_injection_blocked(self):
        """危险表达式必须被白名单拦截，绝不执行"""
        evil = [
            "__import__('os').system('dir')",
            "().__class__",
            "open('/etc/passwd')",
            "1;2",
        ]
        for expr in evil:
            assert calculate(expr).startswith("错误"), expr

    def test_invalid_expression(self):
        assert calculate("√").startswith("错误")
        assert calculate("1+").startswith("错误")

    def test_huge_exponent_blocked(self):
        assert "指数过大" in calculate("2**100001")


def test_preprocess_expression():
    assert _preprocess_expression("√16 + 1") == "sqrt(16) + 1"
    # 括号形式会保留原括号（求值结果不受影响）
    assert _preprocess_expression("√(1+3)") == "sqrt((1+3))"
    with pytest.raises(ValueError):
        _preprocess_expression("√x")


def test_get_current_time():
    result = get_current_time()
    assert "Asia/Shanghai" in result
    assert "错误" in get_current_time("不存在的时区")


def test_execute_tool():
    assert execute_tool("calculate", {"expression": "1+1"}) == "2"
    assert "未知工具" in execute_tool("no_such_tool", {})
    assert "参数不正确" in execute_tool("calculate", {})


def test_tool_definitions():
    """工具定义遵循 OpenAI function calling 协议结构"""
    names = [t["function"]["name"] for t in get_available_tools()]
    assert {"get_current_time", "calculate"} <= set(names)
    assert len(get_tool_names()) >= 2
