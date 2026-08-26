"""modules/tools 单元测试：安全计算器、时间工具与工具执行器"""

import pytest

from modules.tools import (
    _preprocess_expression,
    build_tools_ack,
    build_tools_directive,
    calculate,
    execute_tool,
    get_available_tools,
    get_current_time,
    get_tool_names,
    has_tool_denial,
)


class TestCalculate:
    """安全数学计算：正确性 + 注入防护（AST 白名单）"""

    def test_basic_arithmetic(self):
        # === FIX: 返回值带明确前缀，模型不再误判需要继续计算 ===
        assert calculate("2+3*4") == "计算结果：14"
        assert calculate("(1+2)*3") == "计算结果：9"
        assert calculate("10/4") == "计算结果：2.5"
        assert calculate("2**10") == "计算结果：1024"

    def test_sqrt(self):
        assert calculate("√16") == "计算结果：4"
        assert calculate("√(1+3)") == "计算结果：2"

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
            # === FIX: 失败前缀改为"计算错误" ===
            assert calculate(expr).startswith("计算错误"), expr

    def test_invalid_expression(self):
        assert calculate("√").startswith("计算错误")
        assert calculate("1+").startswith("计算错误")

    def test_huge_exponent_blocked(self):
        assert "指数过大" in calculate("2**100001")


def test_preprocess_expression():
    assert _preprocess_expression("√16 + 1") == "sqrt(16) + 1"
    # 括号形式会保留原括号（求值结果不受影响）
    assert _preprocess_expression("√(1+3)") == "sqrt((1+3))"
    # === FIX: ^ 统一规范为 **（幂运算） ===
    assert _preprocess_expression("3^33") == "3**33"
    assert _preprocess_expression("2^10 + 1") == "2**10 + 1"
    with pytest.raises(ValueError):
        _preprocess_expression("√x")


def test_caret_power_operator():
    """=== FIX: 模型/用户常用 ^ 表示幂，必须与 ** 等价 ==="""
    assert calculate("3^33") == "计算结果：5559060566555523"
    assert calculate("2^10") == "计算结果：1024"


def test_get_current_time():
    result = get_current_time()
    assert "Asia/Shanghai" in result
    assert "错误" in get_current_time("不存在的时区")


def test_execute_tool():
    # === FIX: calculate 返回值带"计算结果："前缀 ===
    assert execute_tool("calculate", {"expression": "1+1"}) == "计算结果：2"
    assert "未知工具" in execute_tool("no_such_tool", {})
    assert "参数不正确" in execute_tool("calculate", {})


def test_tool_definitions():
    """工具定义遵循 OpenAI function calling 协议结构"""
    names = [t["function"]["name"] for t in get_available_tools()]
    assert {"get_current_time", "calculate"} <= set(names)
    assert len(get_tool_names()) >= 2


def test_has_tool_denial():
    """识别助手"否认工具能力"的历史回复（模型会模仿旧回复拒绝调用工具）"""
    denials = [
        "我这边确实没有调用时间工具的能力哦",
        "我这边确实没有权限调用时间查询工具",
        "目前在这个对话环境里，我这边并没有开放调用外部工具",
        "我现在真的没有看时间的小功能呢",
        "我没有实时查看时间的功能呢",
        "我这边确实没有调用时间工具的能力",
        "暂时还没有调用时间工具的功能哦",
        "我目前确实没有调用时间查询工具的能力",
    ]
    for text in denials:
        assert has_tool_denial([{"role": "assistant", "content": text}]), text
    # 正常回复与用户消息不应误判
    assert not has_tool_denial([{"role": "assistant", "content": "现在是 22:24（北京时间）"}])
    assert not has_tool_denial([{"role": "user", "content": "为什么不调用工具？"}])
    assert not has_tool_denial([])


def test_build_tools_directive():
    """指令声明工具可用并要求必须调用（用于破除对旧否认回复的模仿）"""
    directive = build_tools_directive()
    assert "get_current_time" in directive
    assert "calculate" in directive
    assert "必须调用" in directive
    assert "没有工具" in directive


def test_build_tools_ack():
    """确认回合声明工具已接入（插入历史末尾，示范调用工具的行为模式）"""
    ack = build_tools_ack()
    assert "工具" in ack
    assert "接入" in ack
