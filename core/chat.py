"""对话核心流程：AI 客户端、流式调用与工具循环（不依赖 Streamlit）

设计说明：界面占位符（st.empty 等）由调用方以参数注入，会话配置由调用方
以 session_state 参数注入；思考过程展示面板通过 reasoning_placeholder_factory
回调创建。因此本模块在非 Streamlit 环境（单元测试）也可独立运行。
"""

import json
import logging
import os
import time
from typing import Any

from openai import OpenAI

from core.config import AppConfig

logger = logging.getLogger("ai_chat.core.chat")

# 多模型配置与思考过程提取依赖 models 模块；缺失时提供兜底，不影响流式接收
try:
    from core.models import extract_reasoning, get_model_config
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False

    def extract_reasoning(_chunk) -> str:
        """模型思考过程提取兜底实现（models 模块缺失时返回空字符串）"""
        return ""

# 工具执行依赖 tools 模块；工具模块不可用时不会走到工具执行分支（由调用方保证）
try:
    from core.tools import execute_tool
except ImportError:
    execute_tool = None


def create_ai_client(session_state) -> OpenAI:
    """创建统一的 AI API 客户端

    三家提供方（DeepSeek / OpenAI / Ollama）均通过 OpenAI SDK 调用，
    由 core/models.py 提供统一配置；多模型模块不可用时自动退化为
    DeepSeek 环境变量配置（保证应用不崩溃）。

    超时说明：客户端 read 超时即"两次数据块之间的最大等待时间"，
    统一收紧到 STREAM_STALL_TIMEOUT 秒，配合流循环内的停滞检查实现
    30 秒超时保护（超时只影响卡死的连接，不影响正常的长回复）。

    Args:
        session_state: 会话状态对象（st.session_state，由调用方注入）
    """
    if MODELS_AVAILABLE:
        cfg = get_model_config(
            session_state.current_model, provider_key=session_state.provider)
        api_key = (session_state.api_keys.get(session_state.provider, "")
                   or cfg["api_key"])
        base_url = (session_state.base_urls.get(session_state.provider, "")
                    or cfg["base_url"])
        # 客户端 read 超时取 min(模型配置, 停滞超时)：任何情况下断流最多 30 秒即可发现
        timeout = min(cfg["timeout"], AppConfig.STREAM_STALL_TIMEOUT)
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    # 兜底：多模型模块不可用时退回 DeepSeek 环境变量配置
    return OpenAI(
        api_key=session_state.get("api_key") or os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url=session_state.get("base_url", "https://api.deepseek.com"),
        timeout=AppConfig.STREAM_STALL_TIMEOUT,
    )


# ====================== API核心函数 ======================
def call_ai_api_stream(session_state, messages: list[dict], tools: list[dict] | None = None) -> tuple[Any, str | None]:
    """流式调用AI接口

    Args:
        session_state: 会话状态对象（st.session_state，由调用方注入）
        messages: 完整的消息列表
        tools: 工具定义列表，传入时启用函数调用（None 表示不带工具）

    Returns:
        tuple: (流式响应对象, 错误信息或None)
    """
    max_retry = 2
    retry_count = 0

    while retry_count <= max_retry:
        try:
            client = create_ai_client(session_state)
            stream = client.chat.completions.create(
                model=session_state.current_model,
                messages=messages,
                stream=True,
                temperature=session_state.temperature,
                max_tokens=session_state.max_tokens,
                tools=tools,
            )
            return stream, None
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            logger.warning("接口调用失败，重试 %d/%d：%s", retry_count, max_retry, e)
            if retry_count > max_retry:
                if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
                    return None, f"❌ API密钥无效或未填写！请检查「{session_state.provider}」提供方的密钥配置。"
                elif "quota" in error_msg.lower() or "balance" in error_msg.lower():
                    return None, "❌ 账户配额不足，请检查账户余额！"
                elif "timeout" in error_msg.lower():
                    return None, "❌ 请求超时，请检查网络后重试！"
                else:
                    return None, f"❌ 接口请求失败：{error_msg}"


def trim_context_messages(messages: list[dict], session_state) -> list[dict]:
    """裁剪上下文消息（仅保留最近 max_context_msg 条，系统提示词始终保留）

    Args:
        messages: 完整的消息列表
        session_state: 会话状态对象（st.session_state，由调用方注入）
    """
    if len(messages) > session_state.max_context_msg:
        system_msg = messages[0] if messages and messages[0]["role"] == "system" else None
        new_msgs = messages[-session_state.max_context_msg:]
        if system_msg:
            new_msgs.insert(0, system_msg)
        return new_msgs
    return messages


def run_tool_loop(
    session_state,
    messages: list[dict],
    content_placeholder: Any,
    tool_placeholder: Any,
    tools: list[dict] | None = None,
    reasoning_placeholder_factory: Any = None,
) -> tuple[str, list[str], str | None, str]:
    """带工具调用的多轮对话循环（Function Calling 主流程）

    流程：
    1. 携带工具定义调用 AI（流式），同时累积正文与 tool_calls 增量
    2. 若 AI 返回 tool_calls → 逐个执行工具，结果以 tool 消息追加回对话，进入下一轮
    3. 若 AI 直接回复正文 → 结束循环，返回最终回复
    4. 最多循环 MAX_TOOL_ROUNDS 轮，防止死循环
    5. 工具请求失败（如模型不支持工具调用）→ 自动去掉 tools 重试一次，降级原因随最终回答明确提示（绝不静默）
    6. 流式 30 秒停滞保护（客户端 read 超时 + 循环内检查双保险），中断时保留已接收内容
    7. 工具结果超过 4000 字符自动截断，防止挤爆上下文窗口
    8. === FIX: 工具结果回传后追加"停止调用"系统消息，强制 AI 直接给出最终回答，防止无限循环 ===
    9. === FIX: 达到轮数上限时返回最后一轮已有内容并附上限提示，不再直接丢弃 ===

    Args:
        session_state: 会话状态对象（st.session_state，由调用方注入）
        messages: 完整的 API 消息列表（system + 历史对话，本函数会就地追加中间消息）
        content_placeholder: 用于流式渲染回复内容的 st.empty 占位符
        tool_placeholder: 用于显示工具调用状态的 st.empty 占位符
        tools: 工具定义列表；None 表示不启用工具（单轮普通对话）
        reasoning_placeholder_factory: 思考过程展示面板工厂（无参数回调，返回面板
            内部占位符；None 表示不实时展示思考过程）。由 UI 层传入，避免本模块
            依赖 Streamlit

    Returns:
        tuple: (最终回复文本, 使用过的工具名列表, 错误信息或None, 思考过程文本)
    """
    MAX_TOOL_ROUNDS = 3
    tools_used = []
    reasoning_text = ""             # 累积的模型思考过程（如 deepseek-reasoner）
    reasoning_placeholder = None    # 思考过程实时展示占位符（首次收到思考内容时创建）
    allow_retry_without_tools = tools is not None  # 首次调用带工具时，报错可降级重试
    degrade_notice = None           # 工具请求失败降级为普通对话时的提示（最终回答中展示）
    # === FIX: full_content 提升到循环外定义，达到轮数上限时能返回最后一轮已有内容 ===
    full_content = ""

    for round_idx in range(MAX_TOOL_ROUNDS):
        # === FIX: 每轮开始记录日志：轮次与累计已调用的工具 ===
        logger.info("工具循环第 %d/%d 轮开始（已调用工具：%s）",
                    round_idx + 1, MAX_TOOL_ROUNDS, "、".join(tools_used) or "无")
        if tools:
            content_placeholder.markdown("🤔 正在思考是否调用工具...")

        stream, err = call_ai_api_stream(session_state, messages, tools=tools)
        if err:
            # 工具请求失败（如模型不支持工具调用）时，去掉 tools 降级重试一次，
            # 并把降级原因记入 degrade_notice 随最终回答展示——绝不静默降级。
            # 若历史中已含 tool 消息则不能去工具重试（接口会拒绝不配套的 tool 消息），
            # 直接返回错误。
            if (allow_retry_without_tools
                    and not any(m.get("role") == "tool" for m in messages)):
                allow_retry_without_tools = False
                degrade_notice = (
                    "⚠️ 工具调用请求失败，已自动降级为普通对话（本轮不再调用工具）："
                    + err.replace("❌", "").strip()
                )
                tools = None
                stream, err = call_ai_api_stream(session_state, messages, tools=None)
            if err:
                return "", tools_used, err, reasoning_text

        # 累积流式输出：思考过程 + 正文 + 工具调用增量（按 index 拼接参数片段）
        # 停滞保护：30 秒未收到新数据即中止本轮（客户端 read 超时兜底彻底断流，
        # 此检查兜底"有数据但极慢"的情况），避免界面无限转圈
        full_content = ""
        tool_calls = {}
        last_chunk_time = time.monotonic()
        try:
            for chunk in stream:
                if time.monotonic() - last_chunk_time > AppConfig.STREAM_STALL_TIMEOUT:
                    logger.warning("流式响应停滞超过 %s 秒，中止本轮接收",
                                   AppConfig.STREAM_STALL_TIMEOUT)
                    if full_content:
                        stall_err = "⚠️ 响应中断：部分内容已返回"
                        return (full_content, tools_used,
                                f"{degrade_notice}；{stall_err}" if degrade_notice else stall_err,
                                reasoning_text)
                    return "", tools_used, "❌ 流式响应超时（30 秒无数据），请重试", reasoning_text
                last_chunk_time = time.monotonic()
                delta = chunk.choices[0].delta
                # 思考过程增量：reasoner 模型先流式输出思考，再输出最终回答
                reasoning_piece = extract_reasoning(chunk)
                if reasoning_piece:
                    reasoning_text += reasoning_piece
                    if reasoning_placeholder is None and reasoning_placeholder_factory is not None:
                        # 首次收到思考内容时才创建可折叠面板（由 UI 层提供的
                        # Streamlit 工厂完成），普通模型零开销
                        reasoning_placeholder = reasoning_placeholder_factory()
                    if reasoning_placeholder is not None:
                        reasoning_placeholder.markdown(reasoning_text)
                if delta.content:
                    full_content += delta.content
                    content_placeholder.markdown(full_content + "▌")
                if delta.tool_calls:
                    for tc in delta.tool_calls:
                        idx = tc.index or 0
                        entry = tool_calls.setdefault(idx, {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        })
                        if tc.id:
                            entry["id"] = tc.id
                        if tc.function:
                            if tc.function.name:
                                entry["function"]["name"] += tc.function.name
                            if tc.function.arguments:
                                entry["function"]["arguments"] += tc.function.arguments
        except Exception as e:
            # 流中途断连（如 read 超时）：保留已接收内容降级返回，不崩溃
            logger.warning("流式接收中断：%s", e)
            if full_content:
                cut_err = "⚠️ 流式响应中断，已返回部分内容"
                return (full_content, tools_used,
                        f"{degrade_notice}；{cut_err}" if degrade_notice else cut_err,
                        reasoning_text)
            return "", tools_used, f"❌ 流式响应中断：{e}", reasoning_text

        # === FIX: 最终回答检测——本轮 AI 未请求任何工具，即视为已给出最终答案，立即返回 ===
        if not tool_calls:
            logger.info("工具循环第 %d 轮：AI 未请求工具，直接给出最终回答", round_idx + 1)
            return full_content, tools_used, degrade_notice, reasoning_text

        # AI 请求调用工具：执行并把结果回传
        if execute_tool is None:
            # 防御：工具模块不可用时不应走到这里（调用方不会传入 tools）
            return "", tools_used, "❌ 工具模块不可用，无法执行工具调用", reasoning_text
        call_list = list(tool_calls.values())
        # === FIX: 记录本轮 AI 请求调用的工具清单 ===
        call_names = [c["function"]["name"] for c in call_list]
        logger.info("工具循环第 %d 轮：AI 请求调用工具 %s",
                    round_idx + 1, "、".join(call_names))
        messages.append({
            "role": "assistant",
            "content": full_content or None,
            "tool_calls": call_list,
        })
        for call in call_list:
            name = call["function"]["name"]
            tools_used.append(name)
            tool_placeholder.markdown("🔧 调用了工具：" + "、".join(tools_used))
            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            logger.info("执行工具：%s，参数：%s", name,
                        call["function"]["arguments"][:200])
            result = execute_tool(name, arguments)
            # 工具结果截断保护：超长结果会挤爆上下文窗口，限制为 4000 字符
            if len(result) > AppConfig.MAX_TOOL_RESULT_CHARS:
                logger.warning("工具 %s 结果过长（%d 字符），已截断至 %s 字符",
                               name, len(result), AppConfig.MAX_TOOL_RESULT_CHARS)
                result = (result[:AppConfig.MAX_TOOL_RESULT_CHARS]
                          + "…（结果过长，已截断）")
            # === FIX: 记录工具执行结果（截断日志，避免超长结果刷屏） ===
            logger.info("工具 %s 执行完成（第 %d 轮），结果：%s",
                        name, round_idx + 1, result[:200])
            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })
        # === FIX: 工具结果全部回传后，追加"停止调用"系统消息，强制 AI 下一轮
        # 直接给出最终回答，不再发起新的工具调用（同一轮内的并行多工具调用不受影响），
        # 从根源上杜绝"反复调用工具直到轮数上限"的无限循环 ===
        messages.append({
            "role": "system",
            "content": "你已获得所有工具结果，请直接回答用户的问题，不要再调用任何工具。",
        })

    # === FIX: 轮数上限保护：返回最后一轮已有内容并附上限提示，而不是直接丢弃 ===
    logger.warning("工具循环达到 %d 轮上限仍未收敛（已调用：%s）",
                   MAX_TOOL_ROUNDS, "、".join(tools_used))
    if full_content:
        return (full_content, tools_used,
                "⚠️ 工具调用已达上限（3轮），已返回部分结果，回答可能不完整",
                reasoning_text)
    return ("", tools_used,
            "⚠️ 工具调用已达上限（3轮），未能完成回答，请简化问题后重试",
            reasoning_text)
def call_ai_api_simple(messages):
    """纯 API 调用（直接从环境变量读配置，绕过 st.session_state）"""
    from openai import OpenAI
    import os

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        return "错误：未设置 DEEPSEEK_API_KEY 环境变量"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=30
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=2000,
    )

    full_content = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            full_content += chunk.choices[0].delta.content

    return full_content


def call_ai_api_simple_stream(messages, cancel_event=None, model=None):
    """纯 API 流式迭代生成器（供 app_api 的 SSE / 取消接口使用）

    与 call_ai_api_simple 同款环境变量配置（绕过 st.session_state），但逐块
    yield 文本增量，且在每两块数据之间检查 cancel_event —— 被取消时提前
    终止生成，从而支持“停止生成”类需求。

    Args:
        messages: 完整的消息列表
        cancel_event: 可选的 threading.Event；置位后在下一个数据块到达时停止
        model: 可选的模型名覆盖（默认读 DEEPSEEK_MODEL，缺省 deepseek-chat）

    Yields:
        逐个文本增量（str）
    """
    import os

    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    if not api_key:
        yield "错误：未设置 DEEPSEEK_API_KEY 环境变量"
        return

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=30)
    response = client.chat.completions.create(
        model=model or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=messages,
        stream=True,
        temperature=0.7,
        max_tokens=2000,
    )

    for chunk in response:
        if cancel_event is not None and cancel_event.is_set():
            return  # 已在取消登记中被中断：提前结束生成
        delta = chunk.choices[0].delta
        if delta and delta.content:
            yield delta.content