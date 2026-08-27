"""
工具调用（Function Calling）模块
================================
为 AI 提供外部工具能力：获取当前时间、安全数学计算、网络搜索。

工具定义遵循 OpenAI 兼容的 function calling 协议，
DeepSeek / OpenAI 等模型均可直接使用。

注意事项：
- 网络搜索使用 Tavily API（需在 .env 中配置 TAVILY_API_KEY），Tavily 失败时
  自动降级到备用源 pixserp（需配置 PIXSERP_API_KEY）
- deepseek-chat / deepseek-reasoner 均已实测支持工具调用；个别不支持的模型由
  主程序先尝试、失败后在回答中明确提示降级（绝不静默跳过）
- 数学计算使用 AST 白名单解析，杜绝 eval 注入风险
"""

import ast
import logging
import math
import operator
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

# 网络搜索：Tavily API（API Key 从环境变量 TAVILY_API_KEY 读取；
# 主程序启动时已通过 load_dotenv 加载 .env，因此这里按调用时懒加载客户端）
try:
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None

# pixserp 备用搜索源走 HTTP 接口；requests 是 streamlit 的既有依赖，
# 个别环境缺失时降级为「备用源不可用」（主源 Tavily 不受影响）
try:
    import requests
except ImportError:
    requests = None


# ====================== 搜索内部缓存（Redis） ======================
# search_web 工具的内部缓存：与全局「回答缓存」（cache_enabled 开关）完全独立，
# 不受该开关与互斥逻辑影响；Redis 不可用时自动降级为直接调用 Tavily API。
# 复用 modules/cache.py 的 Redis 连接（同一份 HOST/PORT/PASSWORD 配置与
# 可用性冷却机制），此处不重复创建连接。
logger = logging.getLogger("ai_chat.tools")
SEARCH_CACHE_TTL = int(os.environ.get("SEARCH_CACHE_TTL", 600))  # 搜索缓存过期秒数，默认 600

try:
    from modules.cache import (
        is_redis_available as _redis_available,
        _get_client as _get_redis_client,
    )
except ImportError:
    # 缓存模块缺失时整体降级：搜索直接走 Tavily API
    _redis_available = None
    _get_redis_client = None


def _search_cache_get(key):
    """从 Redis 读取搜索缓存；Redis 不可用或读取失败返回 None（静默降级）"""
    if _redis_available is None or not _redis_available():
        return None
    try:
        return _get_redis_client().get(key)
    except Exception:
        return None


def _search_cache_set(key, value):
    """写入搜索缓存；Redis 不可用或写入失败静默跳过（不影响搜索主流程）"""
    if _redis_available is None or not _redis_available():
        return
    try:
        _get_redis_client().setex(key, SEARCH_CACHE_TTL, value)
    except Exception:
        pass


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


# ====================== 备用搜索源：pixserp ======================
# pixserp 没有独立的 REST 搜索端点，走 OpenAI 兼容的 chat/completions 接口
# （实测：POST {base}/chat/completions，模型 pixserp-fast，响应的
# message.citations 中含 title/url/snippet 字段）。地址与模型均可通过
# 环境变量覆盖，密钥绝不硬编码。
PIXSERP_BASE_URL = os.environ.get("PIXSERP_BASE_URL", "https://pixserp.com/api/v1").rstrip("/")
PIXSERP_MODEL = os.environ.get("PIXSERP_MODEL", "pixserp-fast")
PIXSERP_TIMEOUT = int(os.environ.get("PIXSERP_TIMEOUT", 30))  # 请求超时秒数


def _get_tavily_client():
    """懒加载 Tavily 客户端（每次搜索时创建，保证 .env 已加载后再读取密钥）

    Returns:
        tuple: (TavilyClient 或 None, 错误说明或 None)；出错时客户端为 None
    """
    if TavilyClient is None:
        return None, "错误：网络搜索不可用（未安装 tavily-python，可执行 pip install tavily-python）"
    api_key = os.environ.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return None, "错误：网络搜索未配置 TAVILY_API_KEY（请在项目根目录 .env 中设置）"
    return TavilyClient(api_key=api_key), None


def _search_with_tavily(query, max_results):
    """调用 Tavily 搜索（主源），成功返回格式化结果行列表，失败抛异常

    Returns:
        list[str]: 每条结果一行「标题/链接/摘要」文本；空列表表示无结果
    """
    client, err = _get_tavily_client()
    if err:
        raise RuntimeError(err)
    # Tavily 支持 search_depth=basic/advanced；basic 速度快、消耗低，够用
    response = client.search(query=query, max_results=max_results, search_depth="basic", timeout=30)
    results = response.get("results", []) if isinstance(response, dict) else []
    return [
        f"标题：{r.get('title', '')}\n链接：{r.get('url', '')}\n摘要：{r.get('content', '')}"
        for r in results
    ]


def _search_with_pixserp(query, max_results):
    """调用 pixserp 搜索（备用源），成功返回格式化结果行列表，失败抛异常

    走 OpenAI 兼容接口：POST {PIXSERP_BASE_URL}/chat/completions，Bearer 认证；
    响应的 message.content 为直接答案，message.citations 提供来源引用
    （title/url/snippet 字段），两者拼成与 Tavily 一致的文本格式。

    Returns:
        list[str]: 「答案」+ 每条引用一行「标题/链接/摘要」文本；空列表表示无结果
    """
    if requests is None:
        raise RuntimeError("错误：备用搜索源不可用（未安装 requests，可执行 pip install requests）")
    api_key = os.environ.get("PIXSERP_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("错误：备用搜索源未配置 PIXSERP_API_KEY（请在项目根目录 .env 中设置）")
    resp = requests.post(
        f"{PIXSERP_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": PIXSERP_MODEL,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": 800,
        },
        timeout=PIXSERP_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    message = choices[0].get("message", {}) if choices else {}
    content = (message.get("content") or "").strip()
    citations = message.get("citations") or []
    lines = []
    if content:
        lines.append(f"答案：{content}")
    for c in citations[:max_results]:
        lines.append(f"标题：{c.get('title', '')}\n链接：{c.get('url', '')}\n摘要：{c.get('snippet', '')}")
    return lines


def search_web(query, max_results=5):
    """使用 Tavily API 搜索网络并返回结果摘要（带 Redis 内部缓存）

    搜索链路：Tavily（主源）→ 失败自动降级 pixserp（备用源）→ 两个都失败
    返回友好错误。缓存说明：键格式 search:{query}:{max_results}，过期时间
    SEARCH_CACHE_TTL（默认 600 秒）；Redis 不可用时自动降级为直接调用搜索
    API，不影响搜索。主源或备用源的成功结果都写入同一缓存。

    Args:
        query: 搜索关键词
        max_results: 最多返回的结果条数，默认 5（上限 10）

    Returns:
        str: 搜索结果文本（标题+链接+摘要）；两个源都不可用时返回错误说明
    """
    try:
        max_results = max(1, min(int(max_results), 10))
    except (TypeError, ValueError):
        max_results = 5  # 参数异常时回退默认值，避免整次搜索失败

    # ① 先查缓存：命中直接返回，完全不调用搜索 API
    cache_key = f"search:{query}:{max_results}"
    cached = _search_cache_get(cache_key)
    if cached:
        logger.info("搜索缓存命中：%s", cache_key)
        return cached
    logger.info("搜索缓存未命中：%s", cache_key)

    # ② 主源 Tavily
    try:
        lines = _search_with_tavily(query, max_results)
        logger.info("Tavily 搜索成功：%s（%d 条结果）", cache_key, len(lines))
    except Exception as e:
        # ③ Tavily 失败（超时/额度用完/返回错误等）：自动切换到备用源 pixserp
        logger.warning("Tavily 搜索失败：%s（%s），切换到备用源 pixserp", cache_key, e)
        try:
            lines = _search_with_pixserp(query, max_results)
            logger.info("pixserp 搜索成功：%s（%d 条结果）", cache_key, len(lines))
        except Exception as e2:
            logger.error("pixserp 搜索也失败：%s（%s）", cache_key, e2)
            return (f"❌ 网络搜索失败：主搜索源 Tavily 与备用源 pixserp 均不可用，请稍后重试。"
                    f"Tavily 错误：{e}；pixserp 错误：{e2}。")

    if not lines:
        return f"未找到与「{query}」相关的结果"

    # ④ 成功结果写入缓存（仅缓存成功结果；写入失败静默跳过）
    result_text = "\n\n".join(lines)
    _search_cache_set(cache_key, result_text)
    return result_text


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

# 网络搜索工具（Tavily 主源 + pixserp 备用，始终注册）
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
    """获取当前可用的工具定义列表（时间 / 计算 / Tavily 网络搜索）

    Returns:
        list[dict]: OpenAI 兼容的工具定义列表
    """
    return TOOLS + [SEARCH_TOOL]


def get_tool_names():
    """获取工具的中文展示名列表（供界面显示）

    Returns:
        list[str]: 工具中文名列表
    """
    return ["⏰ 获取当前时间", "🧮 数学计算", "🌐 网络搜索"]


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
