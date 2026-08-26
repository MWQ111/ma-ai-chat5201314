"""
工具调用（Function Calling）模块
================================
为 AI 提供外部工具能力：获取当前时间、安全数学计算、网络搜索。

工具定义遵循 OpenAI 兼容的 function calling 协议，
DeepSeek / OpenAI 等模型均可直接使用。

注意事项：
- 网络搜索依赖 ddgs 库（DuckDuckGo，免 API 密钥），未安装时自动从工具列表剔除
- deepseek-chat / deepseek-reasoner 均已实测支持工具调用；个别不支持的模型由
  主程序先尝试、失败后在回答中明确提示降级（绝不静默跳过）
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
    """预处理表达式：将 √ 规范为 sqrt()、^ 规范为 **（幂运算）

    === FIX: 模型与用户常用 3^33 表示 3 的 33 次方，而 Python 中 ^ 是按位异或，
    直接求值会报"不支持的运算符"并浪费一整轮工具调用，这里统一转换为 ** ===

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
    expr = expr.replace("^", "**")  # === FIX: 兼容 ^ 幂运算写法 ===
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
    """安全计算数学表达式（支持 + - * / **（或 ^）√，禁止一切危险语法）

    Args:
        expression: 数学表达式字符串，如 "2+3*4"、"2**10"、"3^33"、"√16"、"(1+2)*3"

    Returns:
        str: 成功返回 "计算结果：xxx"，失败返回 "计算错误：xxx"。
        === FIX: 带明确的前缀，模型不会把裸数字误解为"还需要继续计算" ===
    """
    try:
        expr = _preprocess_expression(expression.strip())
        tree = ast.parse(expr, mode="eval")
        result = _safe_eval_node(tree)
        # 整数值的浮点结果（如 4.0）显示为整数
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        # === FIX: 成功结果带明确前缀，避免 AI 误判需要再次调用工具 ===
        return f"计算结果：{result}"
    except ZeroDivisionError:
        # === FIX: 错误信息统一 "计算错误：" 前缀 ===
        return "计算错误：除数不能为 0"
    except (ValueError, SyntaxError) as e:
        return f"计算错误：表达式不合法（{e}）"
    except Exception as e:
        return f"计算错误：计算失败（{e}）"


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
            "description": ("计算数学表达式。支持 + - * /、** 或 ^（幂）和 √（开方），"
                           "例如 2+3*4、2**10、3^33、√16。当用户要求计算数学题时使用。"),
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


# ====================== 工具启用辅助（提示词与历史自愈） ======================
# 历史对话中若出现过助手否认工具能力的回复（多见于未启用工具时或模型
# 被误判为不支持工具时），模型会"模仿"自己的旧回复，即使本轮请求里带了
# 工具定义也继续拒绝调用。通过两招破除这种锚定：
#   ① 系统指令（build_tools_directive）声明工具可用并要求必须调用；
#   ② 检测到否认历史时，在历史末尾追加一条"工具已接入"的助手确认回合
#      （build_tools_ack）——最近的助手回合具有最强的示范效应。

# 工具在提示词中的用途说明（build_tools_directive 使用）
_TOOL_DIRECTIVE_DESCRIPTIONS = {
    "get_current_time": "查询当前日期和时间",
    "calculate": "计算数学表达式",
    "search_web": "搜索网络实时信息",
}

# 助手"否认工具能力"回复的识别模式（仅检测 assistant 角色消息）。
# 误判的代价只是多注入一条确认回合，对回答无副作用，宁可多检不漏检。
_TOOL_DENIAL_PATTERNS = [
    re.compile(r"没有.{0,10}(调用|开放|连接|接入|使用|权限).{0,10}(工具|时间|天气|网络)"),
    re.compile(r"没有.{0,8}(查看|看|查).{0,6}(时间|天气|新闻).{0,6}(工具|功能|能力)"),
    re.compile(r"(工具|功能).{0,8}(未开放|不可用|没有接入|已关闭|没.{0,4}开放)"),
    re.compile(r"无法.{0,6}(调用|使用|查询|访问).{0,6}(工具|时间|天气|网络)"),
]


def has_tool_denial(messages):
    """检测历史消息中是否存在助手否认工具能力的回复

    Args:
        messages: API 消息列表（dict 列表，含 role/content 字段）

    Returns:
        bool: 存在否认回复时返回 True（主程序据此追加工具确认回合）
    """
    for m in messages:
        if m.get("role") != "assistant":
            continue
        text = m.get("content") or ""
        if any(p.search(text) for p in _TOOL_DENIAL_PATTERNS):
            return True
    return False


def build_tools_directive():
    """生成系统提示词附加指令：声明工具可用，并要求在相应场景必须调用

    Returns:
        str: 追加到系统提示词末尾的指令文本
    """
    tool_lines = "、".join(
        f"{t['function']['name']}（{_TOOL_DIRECTIVE_DESCRIPTIONS.get(t['function']['name'], '')}）"
        for t in get_available_tools()
    )
    return (
        "\n\n【工具调用】系统已为你接入以下工具（这是系统级事实，历史对话中助手声称"
        "“没有工具/没有权限”的说法均已过时无效）："
        + tool_lines
        + "。当用户询问当前时间、日期、星期几，或要求计算数学表达式，或需要实时信息时，"
        "必须调用对应工具获取结果，禁止声称自己没有相关能力。"
    )


def build_tools_ack():
    """生成插入历史末尾的"工具已接入"助手确认回合

    最近的助手回合对模型行为有最强示范效应：在历史中出现过否认工具的回复时，
    追加本回合可让模型切换到"先调用工具"的行为模式。

    Returns:
        str: 确认回合的文本
    """
    return (
        "好的，收到！我现在已经接入了工具调用能力（时间查询、数学计算、网络搜索），"
        "遇到需要实时信息或精确计算的问题时，我会直接调用对应工具来回答。"
    )


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
