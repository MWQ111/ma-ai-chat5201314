"""
工具调用（Function Calling）模块
================================
为 AI 提供外部工具能力：获取当前时间、安全数学计算、网络搜索。

工具定义遵循 OpenAI 兼容的 function calling 协议，
DeepSeek / OpenAI 等模型均可直接使用。

注意事项：
- 网络搜索依赖 ddgs 库（DuckDuckGo，免 API 密钥），未安装时自动从工具列表剔除
- deepseek-reasoner 模型不支持工具调用，由主程序判断跳过（并带自动重试兜底）
- 数学计算使用 AST 白名单解析，杜绝 eval 注入风险
"""

import ast
import math
import operator
import re
from datetime import datetime
from zoneinfo import ZoneInfo

# 网络搜索库：优先新版 ddgs，兼容旧版 duckduckgo_search
try:
    from ddgs import DDGS
except ImportError:
    try:
        from duckduckgo_search import DDGS
    except ImportError:
        DDGS = None
DDGS_AVAILABLE = DDGS is not None


# ====================== 工具实现函数 ======================
def get_current_time(timezone="Asia/Shanghai"):
    """获取指定时区的当前日期时间

    Args:
        timezone: IANA 时区名称，如 Asia/Shanghai、America/New_York，默认北京时间

    Returns:
        str: 格式化后的日期时间字符串（含星期与时区）；时区名无效时返回错误说明
    """
    try:
        tz = ZoneInfo(timezone)
    except Exception:
        return f"错误：未知时区「{timezone}」，请使用 IANA 时区名（如 Asia/Shanghai、America/New_York）"
    now = datetime.now(tz)
    weekdays = ["一", "二", "三", "四", "五", "六", "日"]
    return now.strftime(f"%Y-%m-%d %H:%M:%S 星期{weekdays[now.weekday()]}（{timezone}）")


# 开方符号规范化：√16 -> sqrt(16)，√(1+3) -> sqrt(1+3)
_SQRT_PATTERN = re.compile(r"√\s*(\d+(?:\.\d+)?|\([^()]*\))")


def _preprocess_expression(expression):
    """预处理表达式：将 √ 符号规范为 sqrt() 函数调用

    Args:
        expression: 原始表达式字符串

    Returns:
        str: 规范化后的表达式

    Raises:
        ValueError: √ 用法不正确时抛出
    """
    expr = _SQRT_PATTERN.sub(r"sqrt(\1)", expression)
    if "√" in expr:
        raise ValueError("√ 用法不正确，请写成 √16 或 √(1+3) 的形式")
    return expr


_ALLOWED_FUNCS = {"sqrt": math.sqrt}


def _safe_eval_node(node):
    """递归计算 AST 节点，只允许白名单内的运算与函数

    Args:
        node: ast 节点

    Returns:
        int/float: 节点计算结果

    Raises:
        ValueError: 遇到白名单之外的语法（函数调用、属性访问等）时抛出
    """
    # 顶层表达式包装节点（ast.parse(mode="eval") 的返回结构）
    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)
    # 数字常量
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    # 正负号
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_eval_node(node.operand)
        return +value if isinstance(node.op, ast.UAdd) else -value
    # 二元运算：+ - * / **
    if isinstance(node, ast.BinOp):
        op_map = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
        }
        if type(node.op) in op_map:
            left = _safe_eval_node(node.left)
            right = _safe_eval_node(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > 100000:
                raise ValueError("指数过大，请控制在 100000 以内")
            return op_map[type(node.op)](left, right)
        raise ValueError(f"不支持的运算符：{type(node.op).__name__}")
    # 函数调用：仅允许白名单函数，且参数格式固定
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Name)
            and node.func.id in _ALLOWED_FUNCS
            and len(node.args) == 1
            and not node.keywords
        ):
            return _ALLOWED_FUNCS[node.func.id](_safe_eval_node(node.args[0]))
        raise ValueError("不支持的函数调用")
    raise ValueError(f"不支持的语法：{type(node).__name__}")


def calculate(expression):
    """安全计算数学表达式（支持 + - * / ** √，禁止一切危险语法）

    Args:
        expression: 数学表达式字符串，如 "2+3*4"、"2**10"、"√16"、"(1+2)*3"

    Returns:
        str: 计算结果；表达式非法、除零等错误时返回错误说明（不抛异常）
    """
    try:
        expr = _preprocess_expression(expression.strip())
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval_node(tree)
        # 整数值的浮点结果（如 4.0）显示为整数
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "错误：除数不能为 0"
    except (ValueError, SyntaxError) as e:
        return f"错误：表达式不合法（{e}）"
    except Exception as e:
        return f"错误：计算失败（{e}）"


def search_web(query, max_results=5):
    """使用 DuckDuckGo 搜索网络并返回结果摘要（免 API 密钥）

    Args:
        query: 搜索关键词
        max_results: 最多返回的结果条数，默认 5（上限 10）

    Returns:
        str: 搜索结果文本（标题+链接+摘要）；服务不可用或失败时返回错误说明
    """
    if not DDGS_AVAILABLE:
        return "错误：网络搜索不可用（未安装 ddgs 库，可执行 pip install ddgs）"
    try:
        max_results = max(1, min(int(max_results), 10))
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(
                    f"标题：{r.get('title', '')}\n链接：{r.get('href', '')}\n摘要：{r.get('body', '')}"
                )
        if not results:
            return f"未找到与「{query}」相关的结果"
        return "\n\n".join(results)
    except Exception as e:
        return f"错误：搜索失败（{e}）"


# ====================== 工具定义（OpenAI 兼容协议） ======================
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间。当用户询问现在几点、今天几号、星期几、某个时区的时间时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA 时区名称，如 Asia/Shanghai、America/New_York，默认 Asia/Shanghai",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式。支持 + - * / **（幂）和 √（开方），例如 2+3*4、2**10、√16。当用户要求计算数学题时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "要计算的数学表达式，如 2+3*4"}
                },
                "required": ["expression"],
            },
        },
    },
]

# 网络搜索工具（ddgs 不可用时不会注册）
SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "搜索网络获取实时信息。当用户询问新闻、最新资讯、时事或你不知道的最新内容时使用。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "max_results": {"type": "integer", "description": "最多返回的结果条数，默认 5"},
            },
            "required": ["query"],
        },
    },
}

# 工具名 -> 实现函数的映射（execute_tool 使用）
_EXECUTORS = {
    "get_current_time": get_current_time,
    "calculate": calculate,
    "search_web": search_web,
}


# ====================== 对外接口 ======================
def get_available_tools():
    """获取当前可用的工具定义列表（网络搜索依赖 ddgs，未安装时自动剔除）

    Returns:
        list[dict]: OpenAI 兼容的工具定义列表
    """
    tools = list(TOOLS)
    if DDGS_AVAILABLE:
        tools.append(SEARCH_TOOL)
    return tools


def get_tool_names():
    """获取工具的中文展示名列表（供界面显示）

    Returns:
        list[str]: 工具中文名列表
    """
    names = ["⏰ 获取当前时间", "🧮 数学计算"]
    if DDGS_AVAILABLE:
        names.append("🌐 网络搜索")
    return names


def execute_tool(name, arguments):
    """按工具名执行对应函数并返回结果字符串

    Args:
        name: 工具名称（AI 返回的 tool_calls 中的函数名）
        arguments: 工具参数字典（AI 返回的 JSON 已解析）

    Returns:
        str: 工具执行结果；任何错误都以错误说明字符串返回（便于 AI 理解并转述给用户）
    """
    func = _EXECUTORS.get(name)
    if func is None:
        return f"错误：未知工具「{name}」"
    try:
        return str(func(**arguments))
    except TypeError:
        return f"错误：工具「{name}」的参数不正确：{arguments}"
    except Exception as e:
        return f"错误：工具「{name}」执行失败：{e}"
