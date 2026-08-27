"""
LangGraph Agent 模块
====================
在现有 Function Calling 工具层之上，用 LangGraph 实现带「规划 + 反思 + 自主结束」
能力的 Agent，替代普通模式下固定轮数的工具循环。

图结构：:

    START → call_model ──(请求工具且未达上限)──→ tool ──┐
               │                                        │
               └─────(直接回答 / 达到步数上限)──→ finalize ←┘
                                                       ↓
                                                      END

节点职责：
- call_model_node（规划）：AI 决策节点——结合对话历史与已收集的工具结果，
  由模型自主决定下一步是调用工具还是直接回答；
- tool_node（执行）：执行模型请求的工具，结果以 ToolMessage 回传；
- 反思：工具结果回传后的下一轮 call_model 即是对结果的反思与再决策，
  由 LangGraph 的循环天然实现，无需独立节点；
- should_continue + finalize_node（自主结束）：模型不再发起工具调用时立即结束；
  达到 max_steps 上限时返回已收集的信息并提示。

设计要点：
- 不重写工具层：工具定义与执行完全复用 modules/tools.py 的
  get_available_tools / execute_tool；
- 模型适配：ChatOpenAI 统一接口（DeepSeek / OpenAI / Ollama 均为 OpenAI
  兼容协议），配置从 st.session_state 读取（provider / current_model /
  temperature / api_keys / base_urls），与主程序普通模式共用同一套配置；
- run_agent 返回与主程序 run_tool_loop 相同的四元组
  (最终回复, 已用工具列表, 错误信息, 思考过程)，主程序仅需按模式开关改调
  本函数，其余渲染 / 缓存 / 持久化逻辑完全复用。
"""

import json
import logging
import operator
import os
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from modules.tools import execute_tool

logger = logging.getLogger("ai_chat.agent")

# ====================== Agent 配置常量 ======================
# 默认最大规划步数（可用环境变量 AGENT_MAX_STEPS 覆盖，界面侧边栏可再调整）
DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", 5))
# 工具结果截断上限：与普通模式 AppConfig.MAX_TOOL_RESULT_CHARS 保持一致。
# 注意 agent.py 不能 import APP.py（APP.py 导入本模块，会循环导入），
# 因此此处独立读取环境变量（同名键可由部署方统一配置）。
AGENT_MAX_TOOL_RESULT_CHARS = int(os.environ.get("AGENT_MAX_TOOL_RESULT_CHARS", 4000))

# 追加到系统提示词的 Agent 行为指令（规划 + 反思 + 自主结束）
AGENT_SYSTEM_DIRECTIVE = (
    "\n\n【Agent 模式指令】你现在是一个具备「规划 + 反思 + 自主结束」能力的智能体，请遵循：\n"
    "1. 规划：回答需要外部信息或精确计算时，先想清楚需要哪些信息，再调用相应工具（一次可请求多个）；\n"
    "2. 反思：每获得一批工具结果，先检查是否足以回答用户问题——足够就立即给出最终回答，不足再补充调用其他工具；\n"
    "3. 自主结束：一旦能够回答，立刻停止调用工具并输出最终回答，禁止反复调用同一工具。"
)


# ====================== Agent 状态定义 ======================
class AgentState(TypedDict, total=False):
    """Agent 图状态（messages / tool_results 用 operator.add 归约自动追加）"""

    messages: Annotated[list, operator.add]       # 完整对话（含工具消息）
    current_step: int                             # 已执行的规划步数（模型调用次数）
    max_steps: int                                # 最大规划步数上限
    tool_results: Annotated[list, operator.add]   # 工具调用记录（名称/参数/结果摘要）
    is_done: bool                                 # 是否已结束
    final_answer: str                             # 最终回答
    # ---- 内部字段（run_agent 注入，节点间共享，不参与用户对话） ----
    tools: list                                   # 工具定义列表（bind_tools 使用）
    llm_config: dict                              # ChatOpenAI 构建参数


# ====================== 模型配置解析 ======================
def _resolve_llm_config(config: dict | None) -> dict:
    """解析模型调用配置（ChatOpenAI 参数）

    优先使用显式传入的 config（测试 / 调用方覆盖用）；否则从 st.session_state
    读取多模型配置（provider / current_model / temperature / api_keys /
    base_urls），会话状态为空时回退 modules/models.py 的模型默认配置；
    非 Streamlit 环境（单元测试 / CLI）回退环境变量，保证模块可独立使用。

    Returns:
        dict: 含 model / api_key / base_url / temperature / max_tokens 字段
    """
    if config:
        return {
            "model": config.get("model", "deepseek-chat"),
            "api_key": config.get("api_key", ""),
            "base_url": config.get("base_url", ""),
            "temperature": config.get("temperature"),
            "max_tokens": config.get("max_tokens"),
        }
    try:
        import streamlit as st  # 仅在 Streamlit 运行时读取会话状态

        ss = st.session_state
        provider = ss.get("provider", "deepseek")
        model = ss.get("current_model", "deepseek-chat")
        api_key = ss.get("api_keys", {}).get(provider, "")
        base_url = ss.get("base_urls", {}).get(provider, "")
        temperature = ss.get("temperature")
        max_tokens = ss.get("max_tokens")
        # 与主程序 create_ai_client 一致：会话状态为空时回退模型配置 / 环境变量
        try:
            from modules.models import get_model_config
            cfg = get_model_config(model, provider_key=provider)
        except Exception:
            cfg = {}
        return {
            "model": model,
            "api_key": api_key or cfg.get("api_key", "") or os.environ.get("DEEPSEEK_API_KEY", ""),
            "base_url": base_url or cfg.get("base_url", "") or "https://api.deepseek.com",
            "temperature": temperature if temperature is not None else cfg.get("temperature"),
            "max_tokens": max_tokens if max_tokens is not None else cfg.get("max_tokens"),
        }
    except Exception as e:
        # 非 Streamlit 环境（测试 / CLI）：回退环境变量，保证模块可独立使用
        logger.warning("读取会话状态失败（%s），回退环境变量配置", e)
        return {
            "model": os.environ.get("AGENT_MODEL", "deepseek-chat"),
            "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
            "base_url": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            "temperature": None,
            "max_tokens": None,
        }


def _build_llm(cfg: dict) -> ChatOpenAI:
    """用 ChatOpenAI 统一接口构建模型（DeepSeek / OpenAI / Ollama 均兼容）"""
    kwargs: dict = {"model": cfg["model"], "api_key": cfg["api_key"]}
    if cfg.get("base_url"):
        kwargs["base_url"] = cfg["base_url"]
    if cfg.get("temperature") is not None:
        kwargs["temperature"] = cfg["temperature"]
    if cfg.get("max_tokens"):
        kwargs["max_tokens"] = cfg["max_tokens"]
    return ChatOpenAI(**kwargs)


def _extract_reasoning(msg: AIMessage) -> str:
    """提取模型思考过程（deepseek-reasoner 等推理模型）

    langchain-openai 会把 DeepSeek 的 reasoning_content 透传到
    additional_kwargs / response_metadata，两者都尝试读取。
    """
    for key in ("reasoning_content", "reasoning"):
        for bucket in (msg.additional_kwargs, msg.response_metadata or {}):
            value = bucket.get(key)
            if value:
                return value if isinstance(value, str) else str(value)
    return ""


# ====================== 图节点 ======================
def call_model_node(state: AgentState) -> dict:
    """AI 决策节点（规划）：调用模型，由其决定下一步是调用工具还是直接回答

    每轮执行时把当前已收集的工具结果一并送入模型，模型基于结果进行
    「反思」并再决策——这正是 Agent 反思能力的来源。
    """
    step = state.get("current_step", 0) + 1
    max_steps = state.get("max_steps", DEFAULT_MAX_STEPS)
    logger.info("Agent 规划第 %d/%d 步：调用模型决策", step, max_steps)

    llm = _build_llm(state.get("llm_config") or _resolve_llm_config(None))
    tools = state.get("tools") or []
    if tools:
        llm = llm.bind_tools(tools)

    response = llm.invoke(state["messages"])
    decision = (
        "调用工具 " + "、".join(t["name"] for t in response.tool_calls)
        if response.tool_calls else "直接回答"
    )
    logger.info("Agent 规划第 %d/%d 步：模型决定 %s", step, max_steps, decision)
    return {"messages": [response], "current_step": step}


def tool_node(state: AgentState) -> dict:
    """工具执行节点：执行模型请求的工具调用，结果以 ToolMessage 回传

    复用 modules.tools.execute_tool（与普通模式同一套实现，不重写工具层）。
    """
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return {}  # 防御：正常流程不会走到这里（should_continue 已把关）

    tool_msgs = []
    records = []
    for tc in last.tool_calls:
        name, call_id = tc["name"], tc["id"]
        try:
            args = json.loads(tc["args"]) if isinstance(tc["args"], str) else (tc["args"] or {})
        except json.JSONDecodeError:
            args = {}
        logger.info("Agent 执行工具：%s，参数：%s", name, str(args)[:200])
        result = execute_tool(name, args)
        # 工具结果截断保护：超长结果会挤爆上下文窗口（与普通模式一致）
        if len(result) > AGENT_MAX_TOOL_RESULT_CHARS:
            logger.warning("工具 %s 结果过长（%d 字符），已截断至 %d 字符",
                           name, len(result), AGENT_MAX_TOOL_RESULT_CHARS)
            result = result[:AGENT_MAX_TOOL_RESULT_CHARS] + "…（结果过长，已截断）"
        logger.info("Agent 工具 %s 执行完成，结果：%s", name, result[:200])
        tool_msgs.append(ToolMessage(content=result, tool_call_id=call_id, name=name))
        records.append({"name": name, "arguments": args, "result_summary": result[:500]})
    return {"messages": tool_msgs, "tool_results": records}


def should_continue(state: AgentState) -> str:
    """路由判断：模型是否继续请求工具，是否达到步数上限

    Returns:
        str: "tool"（继续执行工具）/ "finalize"（结束并汇总）
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        if state.get("current_step", 0) < state.get("max_steps", DEFAULT_MAX_STEPS):
            return "tool"
        logger.warning("Agent 达到步数上限 %d，结束循环并返回已收集信息",
                       state.get("max_steps", DEFAULT_MAX_STEPS))
    return "finalize"


def finalize_node(state: AgentState) -> dict:
    """结束节点：汇总最终回答；达到步数上限时返回已收集的信息并提示"""
    max_steps = state.get("max_steps", DEFAULT_MAX_STEPS)
    last = state["messages"][-1] if state.get("messages") else None
    limit_hit = isinstance(last, AIMessage) and bool(last.tool_calls)
    if limit_hit:
        collected = state.get("tool_results") or []
        summary = "\n".join(
            f"- 工具「{r['name']}」（参数：{r['arguments']}）：{r['result_summary']}"
            for r in collected
        ) or "（无）"
        partial = (last.content or "").strip()
        final_answer = (
            f"⚠️ Agent 已达到最大规划步数（{max_steps} 步），未能完成最终回答。\n"
            f"已收集的信息：\n{summary}\n"
            + (f"最后一轮模型输出：{partial}\n" if partial else "")
            + "建议：简化问题后重试，或在侧边栏调大「最大规划步数」。"
        )
    else:
        final_answer = (last.content or "") if isinstance(last, AIMessage) else ""
    logger.info("Agent 结束：%s", "达到步数上限" if limit_hit else "模型自主结束")
    return {"is_done": True, "final_answer": final_answer}


# ====================== 图构建（模块加载时编译一次，全应用复用） ======================
def _build_graph():
    """构建 LangGraph 状态图：call_model ⇄ tool 循环 + finalize 出口"""
    builder = StateGraph(AgentState)
    builder.add_node("call_model", call_model_node)
    builder.add_node("tool", tool_node)
    builder.add_node("finalize", finalize_node)
    builder.add_edge(START, "call_model")
    builder.add_conditional_edges(
        "call_model", should_continue,
        {"tool": "tool", "finalize": "finalize"},
    )
    builder.add_edge("tool", "call_model")
    builder.add_edge("finalize", END)
    return builder.compile()


_AGENT_GRAPH = _build_graph()


# ====================== 对外入口 ======================
def run_agent(
    messages: list[dict],
    content_placeholder: Any = None,
    tool_placeholder: Any = None,
    tools: list[dict] | None = None,
    max_steps: int | None = None,
    config: dict | None = None,
) -> tuple[str, list[str], str | None, str]:
    """LangGraph Agent 主入口：规划 → 工具执行 → 反思循环，自主决定何时结束

    返回与主程序 run_tool_loop 相同的四元组
    (最终回复文本, 使用过的工具名列表, 错误信息或None, 思考过程文本)，
    主程序在 Agent 模式开关打开时改调本函数即可，其余逻辑完全复用。

    Args:
        messages: 完整的 API 消息列表（system + 历史对话；本函数不就地修改它）
        content_placeholder: 渲染最终回答的 st.empty 占位符（可为 None，便于测试）
        tool_placeholder: 展示 Agent 步骤状态的 st.empty 占位符（可为 None）
        tools: 工具定义列表；None 表示无工具（Agent 退化为单轮回答）
        max_steps: 最大规划步数（None 时用 DEFAULT_MAX_STEPS）
        config: 显式模型配置字典（测试 / 调用方覆盖用）；None 时从
                st.session_state 读取多模型配置

    Returns:
        tuple: (最终回复, 已用工具列表, 错误信息或None, 思考过程文本)
    """
    try:
        llm_config = config if config is not None else _resolve_llm_config(None)
        if not llm_config.get("api_key"):
            return "", [], "❌ API密钥无效或未填写！请检查当前提供方的密钥配置。", ""
        steps = max(1, int(max_steps or DEFAULT_MAX_STEPS))

        # 在系统提示词后追加 Agent 行为指令；浅拷贝避免污染主程序的消息列表
        run_messages = [dict(m) for m in messages]
        if run_messages and run_messages[0].get("role") == "system":
            sys_msg = dict(run_messages[0])
            sys_msg["content"] = (sys_msg.get("content") or "") + AGENT_SYSTEM_DIRECTIVE
            run_messages[0] = sys_msg

        logger.info("Agent 启动：max_steps=%d，工具数=%d，消息数=%d",
                    steps, len(tools or []), len(run_messages))
        if tool_placeholder is not None:
            tool_placeholder.markdown("🧠 Agent 模式：开始规划…")

        initial_state: AgentState = {
            "messages": run_messages,
            "current_step": 0,
            "max_steps": steps,
            "tools": tools or [],
            "llm_config": llm_config,
        }

        # 流式观察图执行过程：实时更新界面占位符并收集工具调用记录
        final_answer = ""
        tools_used: list[str] = []
        reasoning_text = ""
        for chunk in _AGENT_GRAPH.stream(
            initial_state, config={"recursion_limit": steps * 3 + 5}
        ):
            if "call_model" in chunk:
                update = chunk["call_model"]
                ai_msgs = update.get("messages") or []
                if ai_msgs and isinstance(ai_msgs[0], AIMessage):
                    ai_msg = ai_msgs[0]
                    reasoning_text += _extract_reasoning(ai_msg)
                    if ai_msg.tool_calls:
                        names = "、".join(t["name"] for t in ai_msg.tool_calls)
                        if tool_placeholder is not None:
                            tool_placeholder.markdown(
                                f"🧠 Agent 规划第 {update.get('current_step', '?')} 步："
                                f"请求调用工具 {names}")
                    elif ai_msg.content and content_placeholder is not None:
                        content_placeholder.markdown(str(ai_msg.content) + "▌")
            elif "tool" in chunk:
                records = chunk["tool"].get("tool_results") or []
                tools_used.extend(r["name"] for r in records)
                if tool_placeholder is not None:
                    tool_placeholder.markdown("🔧 Agent 执行工具：" + "、".join(tools_used))
            elif "finalize" in chunk:
                final_answer = chunk["finalize"].get("final_answer", "")
                if content_placeholder is not None:
                    content_placeholder.markdown(final_answer)
                if tool_placeholder is not None:
                    tool_placeholder.markdown(
                        "🧠 Agent 完成（调用工具：" + ("、".join(tools_used) or "无") + "）")
        return final_answer, tools_used, None, reasoning_text
    except Exception as e:
        logger.error("Agent 运行失败：%s", e)
        return "", [], f"❌ Agent 运行失败：{e}", ""
