import os
import re
import time
import json
import logging
from typing import Any, Optional
from openai import OpenAI
from datetime import datetime

import chainlit as cl

# ====================== 日志系统 ======================
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
logger = logging.getLogger("ai_chat")

# ====================== 环境变量加载 ======================
try:
    from dotenv import load_dotenv, find_dotenv

    _dotenv_path = find_dotenv(usecwd=True, raise_error_if_not_found=False)
    if _dotenv_path:
        load_dotenv(_dotenv_path)
    else:
        load_dotenv()
    logger.info("已加载环境变量文件：%s", _dotenv_path or ".env（默认路径）")
except Exception as e:
    logger.warning("环境变量文件加载跳过：%s", e)

# ====================== 文本工具模块导入 ======================
try:
    from modules.text_utils import estimate_tokens
except ImportError:
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        cjk_chars = len(re.findall(r"[一-鿿　-〿＀-￯]", text))
        other_chars = len(text) - cjk_chars
        return cjk_chars + other_chars // 4

# ====================== RAG 模块导入 ======================
try:
    from modules.rag_module import (
        load_document,
        add_to_vectorstore,
        search,
        get_document_list,
        delete_document,
        invalidate_collection_cache,
        clear_all,
        get_rag_status,
    )

    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ====================== 工具模块导入 ======================
try:
    from modules.tools import (
        get_available_tools, get_tool_names, execute_tool,
        has_tool_denial, build_tools_directive, build_tools_ack,
    )

    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False

# ====================== 缓存模块导入 ======================
try:
    from modules.cache import (
        get_cached_response, set_cached_response, clear_cache,
        get_cache_status, CACHE_TTL,
    )

    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# ====================== 多模型模块导入 ======================
try:
    from modules.models import (
        get_model_config, list_providers, get_provider_config,
        list_ollama_models, extract_reasoning,
    )

    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False


# ====================== 应用配置 ======================
class AppConfig:
    SESSION_FILE_DIR = "./session_data"
    SESSION_FILE_NAME = "session_cache.json"
    SESSION_FILE_VERSION = 2

    DEFAULT_DARK_MODE = True
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2000
    MAX_CONTEXT_MSGS = 50

    MAX_MESSAGE_LENGTH = 10000
    MAX_RENDER_MESSAGES = 50
    STREAM_STALL_TIMEOUT = 30
    MAX_TOOL_RESULT_CHARS = 4000
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024

    DEFAULT_SYSTEM_PROMPT = """你是一位可爱且专业的AI助理喔~。你的特点：
1. 回答要亲切友好，使用适当的语气词（呢、哦、呀）
2. 提供准确、有用的信息
3. 在不确定时坦诚说明
4. 适当使用表情符号增加亲和力
5. 回答要简洁明了，避免过于冗长"""

    QUICK_QUESTIONS = [
        "你好！请介绍一下你自己",
        "帮我写一段Python代码示例",
        "解释一下什么是人工智能",
        "给我一些学习建议"
    ]

    MODEL_LIST = [
        "deepseek-chat",
        "deepseek-reasoner"
    ]


DEFAULT_SYSTEM_PROMPT = AppConfig.DEFAULT_SYSTEM_PROMPT
QUICK_QUESTIONS = AppConfig.QUICK_QUESTIONS
MODEL_LIST = AppConfig.MODEL_LIST


# ====================== 会话持久化 ======================
def save_session_to_file(conversations: list, current_index: int, provider: str, model: str,
                         system_prompt: str, temperature: float, max_tokens: int) -> None:
    """保存会话数据到文件"""
    try:
        save_dir = AppConfig.SESSION_FILE_DIR
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, AppConfig.SESSION_FILE_NAME)
        tmp_path = save_path + ".tmp"
        save_data = {
            "version": AppConfig.SESSION_FILE_VERSION,
            "conversations": conversations,
            "system_prompt": system_prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "current_conversation_index": current_index,
            "provider": provider,
            "current_model": model
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, save_path)
    except Exception as e:
        logger.warning("会话保存失败：%s", e)


def load_session_from_file() -> dict | None:
    """从文件加载会话数据"""
    save_path = os.path.join(AppConfig.SESSION_FILE_DIR, AppConfig.SESSION_FILE_NAME)
    if os.path.exists(save_path):
        try:
            with open(save_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning("会话文件读取失败：%s", e)
            return None
    return None


# ====================== 客户端初始化 ======================
def create_ai_client(provider: str, api_keys: dict, base_urls: dict, current_model: str) -> OpenAI:
    """创建统一的 AI API 客户端"""
    if MODELS_AVAILABLE:
        cfg = get_model_config(current_model, provider_key=provider)
        api_key = (api_keys.get(provider, "") or cfg["api_key"])
        base_url = (base_urls.get(provider, "") or cfg["base_url"])
        timeout = min(cfg["timeout"], AppConfig.STREAM_STALL_TIMEOUT)
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    return OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        base_url="https://api.deepseek.com",
        timeout=AppConfig.STREAM_STALL_TIMEOUT,
    )


# ====================== API核心函数 ======================
def call_ai_api_stream(messages: list[dict], tools: list[dict] | None,
                       provider: str, api_keys: dict, base_urls: dict,
                       current_model: str, temperature: float, max_tokens: int) -> tuple[Any, str | None]:
    """流式调用AI接口"""
    max_retry = 2
    retry_count = 0

    while retry_count <= max_retry:
        try:
            client = create_ai_client(provider, api_keys, base_urls, current_model)
            stream = client.chat.completions.create(
                model=current_model,
                messages=messages,
                stream=True,
                temperature=temperature,
                max_tokens=max_tokens,
                tools=tools,
            )
            return stream, None
        except Exception as e:
            retry_count += 1
            error_msg = str(e)
            logger.warning("接口调用失败，重试 %d/%d：%s", retry_count, max_retry, e)
            if retry_count > max_retry:
                if "API key" in error_msg.lower() or "authentication" in error_msg.lower():
                    return None, f"❌ API密钥无效或未填写！请检查「{provider}」提供方的密钥配置。"
                elif "quota" in error_msg.lower() or "balance" in error_msg.lower():
                    return None, "❌ 账户配额不足，请检查账户余额！"
                elif "timeout" in error_msg.lower():
                    return None, "❌ 请求超时，请检查网络后重试！"
                else:
                    return None, f"❌ 接口请求失败：{error_msg}"


def trim_context_messages(messages: list[dict], max_context_msg: int) -> list[dict]:
    """裁剪上下文消息"""
    if len(messages) > max_context_msg:
        system_msg = messages[0] if messages and messages[0]["role"] == "system" else None
        new_msgs = messages[-max_context_msg:]
        if system_msg:
            new_msgs.insert(0, system_msg)
        return new_msgs
    return messages


# ====================== 工具调用循环 ======================
async def run_tool_loop(
        messages: list[dict],
        tools: list[dict] | None,
        provider: str,
        api_keys: dict,
        base_urls: dict,
        current_model: str,
        temperature: float,
        max_tokens: int,
        max_context_msg: int,
        msg: cl.Message,
) -> tuple[str, list[str], str | None, str]:
    """带工具调用的多轮对话循环"""
    MAX_TOOL_ROUNDS = 3
    tools_used = []
    reasoning_text = ""
    allow_retry_without_tools = tools is not None
    degrade_notice = None
    full_content = ""

    for round_idx in range(MAX_TOOL_ROUNDS):
        logger.info("工具循环第 %d/%d 轮开始", round_idx + 1, MAX_TOOL_ROUNDS)

        stream, err = call_ai_api_stream(
            messages, tools, provider, api_keys, base_urls,
            current_model, temperature, max_tokens
        )

        if err:
            if (allow_retry_without_tools
                    and not any(m.get("role") == "tool" for m in messages)):
                allow_retry_without_tools = False
                degrade_notice = f"⚠️ 工具调用请求失败，已自动降级为普通对话：{err.replace('❌', '').strip()}"
                tools = None
                stream, err = call_ai_api_stream(
                    messages, None, provider, api_keys, base_urls,
                    current_model, temperature, max_tokens
                )
            if err:
                return "", tools_used, err, reasoning_text

        full_content = ""
        tool_calls = {}
        last_chunk_time = time.monotonic()

        try:
            for chunk in stream:
                if time.monotonic() - last_chunk_time > AppConfig.STREAM_STALL_TIMEOUT:
                    logger.warning("流式响应停滞超过 %s 秒", AppConfig.STREAM_STALL_TIMEOUT)
                    if full_content:
                        return full_content, tools_used, "⚠️ 响应中断：部分内容已返回", reasoning_text
                    return "", tools_used, "❌ 流式响应超时（30 秒无数据）", reasoning_text
                last_chunk_time = time.monotonic()
                delta = chunk.choices[0].delta

                reasoning_piece = extract_reasoning(chunk)
                if reasoning_piece:
                    reasoning_text += reasoning_piece

                if delta.content:
                    full_content += delta.content
                    await msg.stream_token(delta.content)

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
            logger.warning("流式接收中断：%s", e)
            if full_content:
                return full_content, tools_used, "⚠️ 流式响应中断", reasoning_text
            return "", tools_used, f"❌ 流式响应中断：{e}", reasoning_text

        if not tool_calls:
            logger.info("AI 未请求工具，直接给出最终回答")
            return full_content, tools_used, degrade_notice, reasoning_text

        call_list = list(tool_calls.values())
        call_names = [c["function"]["name"] for c in call_list]
        logger.info("AI 请求调用工具：%s", "、".join(call_names))

        messages.append({
            "role": "assistant",
            "content": full_content or None,
            "tool_calls": call_list,
        })

        for call in call_list:
            name = call["function"]["name"]
            tools_used.append(name)
            await msg.stream_token(f"\n\n🔧 正在调用工具：{name}...\n")

            try:
                arguments = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                arguments = {}
            logger.info("执行工具：%s，参数：%s", name, str(arguments)[:200])

            try:
                result = execute_tool(name, arguments)
            except Exception as e:
                result = f"工具执行失败：{str(e)}"

            if len(result) > AppConfig.MAX_TOOL_RESULT_CHARS:
                result = result[:AppConfig.MAX_TOOL_RESULT_CHARS] + "…（结果过长，已截断）"

            await msg.stream_token(f"✅ 工具执行完成\n")

            messages.append({
                "role": "tool",
                "tool_call_id": call["id"],
                "content": result,
            })

        messages.append({
            "role": "system",
            "content": "你已获得所有工具结果，请直接回答用户的问题，不要再调用任何工具。",
        })

        messages = trim_context_messages(messages, max_context_msg)

    logger.warning("工具循环达到 %d 轮上限", MAX_TOOL_ROUNDS)
    if full_content:
        return full_content, tools_used, "⚠️ 工具调用已达上限（3轮），回答可能不完整", reasoning_text
    return "", tools_used, "⚠️ 工具调用已达上限（3轮），请简化问题后重试", reasoning_text


# ====================== 发送工具栏 ======================
async def send_toolbar():
    """发送顶部工具栏（纯文字）"""
    actions = [
        cl.Action(name="settings", value="open", description="⚙️ 设置"),
        cl.Action(name="new_chat", value="create", description="➕ 新建对话"),
        cl.Action(name="history", value="show", description="📚 历史对话"),
        cl.Action(name="upload", value="file", description="📄 上传文档"),
        cl.Action(name="clear_current", value="clear", description="🗑️ 清空当前对话"),
        cl.Action(name="clear_all", value="clear_all", description="🧹 全部清空"),
    ]

    await cl.Message(
        content="",
        actions=actions,
        author="System"
    ).send()


# ====================== Action 回调 ======================
@cl.action_callback("settings")
async def on_settings(action: cl.Action):
    """点击设置 → 显示设置面板"""
    provider = cl.user_session.get("provider", "deepseek")
    model = cl.user_session.get("current_model", "deepseek-chat")
    temp = cl.user_session.get("temperature", 0.7)
    max_tokens = cl.user_session.get("max_tokens", 2000)
    system_prompt = cl.user_session.get("system_prompt", AppConfig.DEFAULT_SYSTEM_PROMPT)
    rag_enabled = cl.user_session.get("rag_enabled", True)
    tools_enabled = cl.user_session.get("tools_enabled", True)
    cache_enabled = cl.user_session.get("cache_enabled", True)

    # 获取可用提供方
    provider_actions = []
    if MODELS_AVAILABLE:
        for p_key, p_label in list_providers():
            is_active = "✅ " if p_key == provider else ""
            provider_actions.append(
                cl.Action(
                    name=f"set_provider_{p_key}",
                    value=p_key,
                    description=f"{is_active}{p_label}"
                )
            )

    # 获取可用模型（根据当前提供方）
    model_actions = []
    if MODELS_AVAILABLE:
        cfg = get_provider_config(provider)
        if cfg and cfg.get("models"):
            for m in cfg["models"]:
                is_active = "✅ " if m == model else ""
                # 检查是否支持工具
                try:
                    m_cfg = get_model_config(m, provider_key=provider)
                    tool_tag = " 🔧" if m_cfg.get("supports_tools", False) else ""
                except:
                    tool_tag = ""
                model_actions.append(
                    cl.Action(
                        name=f"set_model_{m}",
                        value=m,
                        description=f"{is_active}{m}{tool_tag}"
                    )
                )

    # 构建设置消息
    settings_msg = (
        f"## ⚙️ 设置面板\n\n"
        f"**当前配置**\n"
        f"- 提供方：{provider}\n"
        f"- 模型：{model}\n"
        f"- 温度：{temp}\n"
        f"- 最大长度：{max_tokens}\n"
        f"- RAG检索：{'✅ 启用' if rag_enabled else '❌ 禁用'}\n"
        f"- 工具调用：{'✅ 启用' if tools_enabled else '❌ 禁用'}\n"
        f"- 缓存：{'✅ 启用' if cache_enabled else '❌ 禁用'}\n\n"
        f"---\n\n"
        f"**选择提供方：**"
    )

    await cl.Message(content=settings_msg, actions=provider_actions).send()

    await cl.Message(
        content=f"**选择模型（{provider}）：**",
        actions=model_actions
    ).send()

    # 高级参数调整
    param_actions = [
        cl.Action(name="set_temp_high", value="high", description="温度 ↑ 0.9"),
        cl.Action(name="set_temp_low", value="low", description="温度 ↓ 0.3"),
        cl.Action(name="set_tokens_up", value="up", description="最大长度 ↑ 500"),
        cl.Action(name="set_tokens_down", value="down", description="最大长度 ↓ 500"),
        cl.Action(name="toggle_rag", value="toggle", description=f"{'禁用' if rag_enabled else '启用'} RAG"),
        cl.Action(name="toggle_tools", value="toggle", description=f"{'禁用' if tools_enabled else '启用'} 工具"),
        cl.Action(name="toggle_cache", value="toggle", description=f"{'禁用' if cache_enabled else '启用'} 缓存"),
    ]

    await cl.Message(
        content="**调整参数：**",
        actions=param_actions
    ).send()

    # 系统提示词
    await cl.Message(
        content=f"**系统提示词：**\n```\n{system_prompt[:200]}{'...' if len(system_prompt) > 200 else ''}\n```\n"
                f"（修改提示词请编辑代码中的 `AppConfig.DEFAULT_SYSTEM_PROMPT`）"
    ).send()


@cl.action_callback("set_provider_*")
async def set_provider(action: cl.Action):
    """切换提供方"""
    provider = action.value
    cl.user_session.set("provider", provider)

    if MODELS_AVAILABLE:
        cfg = get_provider_config(provider)
        if cfg:
            cl.user_session.set("current_model", cfg.get("default_model", "deepseek-chat"))

    await cl.Message(content=f"✅ 已切换到：**{provider}**").send()
    await send_toolbar()


@cl.action_callback("set_model_*")
async def set_model(action: cl.Action):
    """切换模型"""
    model = action.value
    cl.user_session.set("current_model", model)

    # 同步默认参数
    if MODELS_AVAILABLE:
        try:
            cfg = get_model_config(model, provider_key=cl.user_session.get("provider", "deepseek"))
            cl.user_session.set("temperature", cfg.get("temperature", 0.7))
            cl.user_session.set("max_tokens", cfg.get("max_tokens", 2000))
        except:
            pass

    await cl.Message(content=f"✅ 已切换到模型：**{model}**").send()


@cl.action_callback("set_temp_high")
async def set_temp_high(action: cl.Action):
    temp = cl.user_session.get("temperature", 0.7) + 0.2
    temp = min(temp, 2.0)
    cl.user_session.set("temperature", temp)
    await cl.Message(content=f"✅ 温度已调整为：**{temp:.1f}**").send()


@cl.action_callback("set_temp_low")
async def set_temp_low(action: cl.Action):
    temp = cl.user_session.get("temperature", 0.7) - 0.2
    temp = max(temp, 0.0)
    cl.user_session.set("temperature", temp)
    await cl.Message(content=f"✅ 温度已调整为：**{temp:.1f}**").send()


@cl.action_callback("set_tokens_up")
async def set_tokens_up(action: cl.Action):
    tokens = cl.user_session.get("max_tokens", 2000) + 500
    tokens = min(tokens, 8000)
    cl.user_session.set("max_tokens", tokens)
    await cl.Message(content=f"✅ 最大长度已调整为：**{tokens}**").send()


@cl.action_callback("set_tokens_down")
async def set_tokens_down(action: cl.Action):
    tokens = cl.user_session.get("max_tokens", 2000) - 500
    tokens = max(tokens, 100)
    cl.user_session.set("max_tokens", tokens)
    await cl.Message(content=f"✅ 最大长度已调整为：**{tokens}**").send()


@cl.action_callback("toggle_rag")
async def toggle_rag(action: cl.Action):
    current = cl.user_session.get("rag_enabled", True)
    cl.user_session.set("rag_enabled", not current)
    status = "启用" if not current else "禁用"
    await cl.Message(content=f"✅ RAG检索已：**{status}**").send()


@cl.action_callback("toggle_tools")
async def toggle_tools(action: cl.Action):
    current = cl.user_session.get("tools_enabled", True)
    cl.user_session.set("tools_enabled", not current)
    status = "启用" if not current else "禁用"
    await cl.Message(content=f"✅ 工具调用已：**{status}**").send()


@cl.action_callback("toggle_cache")
async def toggle_cache(action: cl.Action):
    current = cl.user_session.get("cache_enabled", True)
    cl.user_session.set("cache_enabled", not current)
    status = "启用" if not current else "禁用"
    await cl.Message(content=f"✅ 缓存已：**{status}**").send()


@cl.action_callback("new_chat")
async def on_new_chat(action: cl.Action):
    """新建对话"""
    cl.user_session.set("messages", [])

    conv_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    conversations = cl.user_session.get("conversations", [])
    conversations.append({
        "id": conv_id,
        "name": f"对话_{conv_id}",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": []
    })
    cl.user_session.set("conversations", conversations)
    cl.user_session.set("current_conversation_index", len(conversations) - 1)

    await cl.Message(content="✅ **新建对话成功！** 开始输入你的问题吧。").send()
    await send_toolbar()


@cl.action_callback("history")
async def on_history(action: cl.Action):
    """显示历史对话列表"""
    conversations = cl.user_session.get("conversations", [])

    if not conversations:
        await cl.Message(content="📚 暂无历史对话").send()
        return

    actions = []
    current_idx = cl.user_session.get("current_conversation_index", 0)

    for i, conv in enumerate(conversations):
        is_current = i == current_idx
        label = f"{'✅ ' if is_current else ''}{conv['name']} ({len(conv.get('messages', []))}条)"
        actions.append(
            cl.Action(
                name=f"switch_conv_{i}",
                value=str(i),
                description=label
            )
        )

    # 添加删除按钮
    if len(conversations) > 1:
        for i, conv in enumerate(conversations):
            if i != current_idx:
                actions.append(
                    cl.Action(
                        name=f"delete_conv_{i}",
                        value=str(i),
                        description=f"🗑️ 删除 {conv['name']}"
                    )
                )

    await cl.Message(
        content=f"📚 **历史对话**（共 {len(conversations)} 个）\n\n点击切换：",
        actions=actions
    ).send()


@cl.action_callback("switch_conv_*")
async def switch_conversation(action: cl.Action):
    """切换对话"""
    idx = int(action.value)
    conversations = cl.user_session.get("conversations", [])

    if 0 <= idx < len(conversations):
        cl.user_session.set("current_conversation_index", idx)
        messages = conversations[idx].get("messages", [])
        cl.user_session.set("messages", messages)

        await cl.Message(
            content=f"✅ 已切换到：**{conversations[idx]['name']}**（{len(messages)} 条消息）"
        ).send()
        await send_toolbar()


@cl.action_callback("delete_conv_*")
async def delete_conversation(action: cl.Action):
    """删除对话"""
    idx = int(action.value)
    conversations = cl.user_session.get("conversations", [])

    if 0 <= idx < len(conversations):
        name = conversations[idx]['name']
        conversations.pop(idx)
        cl.user_session.set("conversations", conversations)

        # 如果删除的是当前对话，切换到第一个
        current_idx = cl.user_session.get("current_conversation_index", 0)
        if current_idx >= len(conversations):
            cl.user_session.set("current_conversation_index", 0)
            if conversations:
                cl.user_session.set("messages", conversations[0].get("messages", []))

        await cl.Message(content=f"✅ 已删除：**{name}**").send()
        await send_toolbar()


@cl.action_callback("clear_current")
async def clear_current(action: cl.Action):
    """清空当前对话"""
    cl.user_session.set("messages", [])
    conversations = cl.user_session.get("conversations", [])
    current_idx = cl.user_session.get("current_conversation_index", 0)

    if 0 <= current_idx < len(conversations):
        conversations[current_idx]["messages"] = []
        cl.user_session.set("conversations", conversations)

    await cl.Message(content="✅ 已清空当前对话").send()
    await send_toolbar()


@cl.action_callback("clear_all")
async def clear_all(action: cl.Action):
    """全部清空"""
    conv_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    default_conv = {
        "id": conv_id,
        "name": "默认对话",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": []
    }

    cl.user_session.set("conversations", [default_conv])
    cl.user_session.set("current_conversation_index", 0)
    cl.user_session.set("messages", [])

    await cl.Message(content="🧹 已全部清空，重新开始！").send()
    await send_toolbar()


@cl.action_callback("upload")
async def on_upload(action: cl.Action):
    """上传文档"""
    if not RAG_AVAILABLE:
        await cl.Message(
            content="⚠️ RAG 依赖未安装，请执行：\n```\npip install langchain langchain-community chromadb pypdf\n```"
        ).send()
        return

    await cl.Message(
        content="📄 **上传文档**\n\n"
                "请使用右上角的文件上传按钮（📎）上传文档。\n\n"
                "支持格式：PDF、TXT、MD\n"
                "最大大小：10MB\n\n"
                "上传后会自动切分并向量化，之后提问时可以检索文档内容。"
    ).send()


# ====================== Chainlit 生命周期 ======================
@cl.on_chat_start
async def start():
    """用户开始新对话时初始化"""
    cl.user_session.set("messages", [])
    cl.user_session.set("conversations", [])
    cl.user_session.set("current_conversation_index", 0)
    cl.user_session.set("conversation_id", datetime.now().strftime("%Y%m%d_%H%M%S"))

    cl.user_session.set("provider", "deepseek")
    cl.user_session.set("current_model", "deepseek-chat")
    cl.user_session.set("api_keys", {
        "deepseek": os.environ.get("DEEPSEEK_API_KEY", ""),
        "openai": os.environ.get("OPENAI_API_KEY", ""),
        "ollama": "",
    })
    cl.user_session.set("base_urls", {
        "deepseek": os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "openai": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "ollama": os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    })

    cl.user_session.set("system_prompt", AppConfig.DEFAULT_SYSTEM_PROMPT)
    cl.user_session.set("temperature", AppConfig.DEFAULT_TEMPERATURE)
    cl.user_session.set("max_tokens", AppConfig.DEFAULT_MAX_TOKENS)
    cl.user_session.set("max_context_msg", AppConfig.MAX_CONTEXT_MSGS)
    cl.user_session.set("rag_enabled", True)
    cl.user_session.set("tools_enabled", True)
    cl.user_session.set("cache_enabled", True)

    # 加载历史会话
    cache_data = load_session_from_file()
    if cache_data:
        convs = cache_data.get("conversations", [])
        if convs:
            cl.user_session.set("conversations", convs)
            cl.user_session.set("current_conversation_index",
                                cache_data.get("current_conversation_index", 0))
            cl.user_session.set("provider", cache_data.get("provider", "deepseek"))
            cl.user_session.set("current_model", cache_data.get("current_model", "deepseek-chat"))
            cl.user_session.set("system_prompt", cache_data.get("system_prompt", AppConfig.DEFAULT_SYSTEM_PROMPT))
            cl.user_session.set("temperature", cache_data.get("temperature", AppConfig.DEFAULT_TEMPERATURE))
            cl.user_session.set("max_tokens", cache_data.get("max_tokens", AppConfig.DEFAULT_MAX_TOKENS))

            idx = cl.user_session.get("current_conversation_index")
            if 0 <= idx < len(convs):
                messages = convs[idx].get("messages", [])
                cl.user_session.set("messages", messages)

    # 发送欢迎消息 + 工具栏
    await cl.Message(
        content="🤖 **马氏AI智能助手**\n\n有什么可以帮你的吗？"
    ).send()

    await send_toolbar()


# ====================== 消息处理 ======================
@cl.on_message
async def main(message: cl.Message):
    """处理用户每条消息"""
    user_content = message.content.strip()

    if not user_content:
        return

    if len(user_content) > AppConfig.MAX_MESSAGE_LENGTH:
        await cl.Message(
            content=f"⚠️ 消息过长（{len(user_content)} 字符），最多允许 {AppConfig.MAX_MESSAGE_LENGTH} 字符").send()
        return

    # 获取当前状态
    messages = cl.user_session.get("messages", [])
    conversations = cl.user_session.get("conversations", [])
    current_idx = cl.user_session.get("current_conversation_index", 0)
    provider = cl.user_session.get("provider", "deepseek")
    current_model = cl.user_session.get("current_model", "deepseek-chat")
    api_keys = cl.user_session.get("api_keys", {})
    base_urls = cl.user_session.get("base_urls", {})
    system_prompt = cl.user_session.get("system_prompt", AppConfig.DEFAULT_SYSTEM_PROMPT)
    temperature = cl.user_session.get("temperature", AppConfig.DEFAULT_TEMPERATURE)
    max_tokens = cl.user_session.get("max_tokens", AppConfig.DEFAULT_MAX_TOKENS)
    max_context_msg = cl.user_session.get("max_context_msg", AppConfig.MAX_CONTEXT_MSGS)
    rag_enabled = cl.user_session.get("rag_enabled", True)
    tools_enabled = cl.user_session.get("tools_enabled", True)
    cache_enabled = cl.user_session.get("cache_enabled", True)

    messages.append({"role": "user", "content": user_content})
    cl.user_session.set("messages", messages)

    if conversations and current_idx < len(conversations):
        conversations[current_idx]["messages"] = messages.copy()
    else:
        conv_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        conversations.append({
            "id": conv_id,
            "name": f"对话_{conv_id}",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "messages": messages.copy()
        })
        current_idx = len(conversations) - 1
        cl.user_session.set("conversations", conversations)
        cl.user_session.set("current_conversation_index", current_idx)

    # RAG检索
    rag_sources = []
    if RAG_AVAILABLE and rag_enabled:
        try:
            rag_sources = search(user_content, top_k=3)
        except Exception as e:
            logger.warning("文档检索失败：%s", e)

    # 工具调用
    tools_for_call = None
    if TOOLS_AVAILABLE and tools_enabled:
        tools_for_call = get_available_tools()

        # 检查模型是否支持工具
        if MODELS_AVAILABLE:
            try:
                model_cfg = get_model_config(current_model, provider_key=provider)
                if not model_cfg.get("supports_tools", False):
                    await cl.Message(
                        content=f"⚠️ 当前模型 {current_model} 标记为不支持工具调用，将尝试调用，失败后自动降级。"
                    ).send()
            except:
                pass

    # 组装API消息
    api_msgs = [{"role": "system", "content": system_prompt}]

    if rag_sources:
        context_text = "\n\n".join(
            f"【文档片段{i}】（来源：{s['source']}）：\n{s['content']}"
            for i, s in enumerate(rag_sources, 1)
        )
        api_msgs[0]["content"] += (
                "\n\n【文档检索上下文】\n"
                "以下是从用户上传文档中检索到的相关内容，请优先基于这些内容回答用户问题；"
                "如果文档中没有与问题相关的内容，请如实告知，再结合你自己的知识回答：\n\n"
                + context_text
        )

    if tools_for_call is not None:
        api_msgs[0]["content"] += build_tools_directive()

    api_msgs += [{"role": m["role"], "content": m["content"]} for m in messages]

    if tools_for_call is not None and has_tool_denial(api_msgs):
        api_msgs.append({"role": "assistant", "content": build_tools_ack()})

    api_msgs = trim_context_messages(api_msgs, max_context_msg)

    # 创建响应消息
    msg = cl.Message(content="")
    await msg.send()

    # 检查缓存
    cache_hit = False
    full_response = ""
    tools_used = []
    reasoning_text = ""
    err = None

    use_cache = (CACHE_AVAILABLE and cache_enabled
                 and not rag_sources and tools_for_call is None)

    if use_cache:
        try:
            cached = get_cached_response(user_content, model=current_model)
            if cached:
                full_response = cached
                cache_hit = True
                await msg.stream_token(cached)
                await msg.send()
                await cl.Message(content="📦 来自缓存").send()
        except Exception as e:
            logger.warning("缓存读取失败：%s", e)

    if not cache_hit:
        full_response, tools_used, err, reasoning_text = await run_tool_loop(
            api_msgs, tools_for_call, provider, api_keys, base_urls,
            current_model, temperature, max_tokens, max_context_msg, msg
        )

        if err:
            if full_response:
                await msg.stream_token(f"\n\n{err}")
            else:
                await cl.Message(content=err).send()

        if use_cache and full_response and not err:
            try:
                set_cached_response(user_content, full_response, model=current_model)
            except Exception as e:
                logger.warning("缓存写入失败：%s", e)

    await msg.send()

    if reasoning_text:
        await cl.Message(
            content=f"💭 **思考过程**\n\n{reasoning_text}",
            author="System"
        ).send()

    if tools_used:
        await cl.Message(
            content=f"🔧 调用了工具：{', '.join(tools_used)}",
            author="System"
        ).send()

    if rag_sources:
        source_text = "📚 **参考来源**\n\n"
        for i, src in enumerate(rag_sources, 1):
            source_text += f"- 来源{i}：{src['source']}　·　相似度 {src['score']:.2f}\n"
        await cl.Message(content=source_text, author="System").send()

    messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": rag_sources,
        "tools_used": tools_used,
        "cache_hit": cache_hit,
        "reasoning": reasoning_text,
    })
    cl.user_session.set("messages", messages)

    if conversations and current_idx < len(conversations):
        conversations[current_idx]["messages"] = messages.copy()
        cl.user_session.set("conversations", conversations)

    save_session_to_file(
        conversations, current_idx, provider, current_model,
        system_prompt, temperature, max_tokens
    )

    # 显示工具栏
    await send_toolbar()


# ====================== 文件上传处理 ======================
@cl.on_file_upload
async def handle_file_upload(file: cl.File):
    """处理文档上传"""
    if not RAG_AVAILABLE:
        await cl.Message(content="⚠️ RAG 依赖未安装，请先安装依赖").send()
        return

    # 检查文件大小
    if file.size > AppConfig.MAX_UPLOAD_SIZE:
        await cl.Message(
            content=f"❌ 文件超过 {AppConfig.MAX_UPLOAD_SIZE // (1024 * 1024)}MB 限制，无法上传"
        ).send()
        return

    # 检查文件类型
    allowed_types = ["pdf", "txt", "md"]
    file_ext = file.name.split(".")[-1].lower() if "." in file.name else ""
    if file_ext not in allowed_types:
        await cl.Message(
            content=f"❌ 不支持的文件类型：{file_ext}\n支持格式：PDF、TXT、MD"
        ).send()
        return

    # 检查文件是否已存在
    try:
        existing_docs = get_document_list()
        existing_names = [doc["file_name"] for doc in existing_docs]
        if file.name in existing_names:
            await cl.Message(
                content=f"⚠️ 《{file.name}》已存在，请勿重复上传（如需更新请先删除原文档）"
            ).send()
            return
    except Exception as e:
        logger.warning("读取文档列表失败：%s", e)

    try:
        await cl.Message(content=f"📄 正在处理《{file.name}》...").send()

        # 读取文件内容
        # 注意：cl.File 没有直接的 path 属性，需要处理
        # 这里假设 file.content 是文件内容
        if hasattr(file, 'content'):
            # 如果是上传的内存文件
            import io
            file_like = io.BytesIO(file.content) if isinstance(file.content, bytes) else io.StringIO(file.content)
            chunks = load_document(file_like, filename=file.name)
        else:
            # 如果是本地文件路径
            chunks = load_document(file.path)

        if not chunks:
            await cl.Message(content=f"❌ 《{file.name}》未解析出任何内容，请检查文件格式").send()
            return

        n_added = add_to_vectorstore(chunks)

        await cl.Message(
            content=f"✅ 《{file.name}》入库成功（{n_added} 个片段）"
        ).send()

    except Exception as e:
        logger.error("文档加载失败：%s", e)
        await cl.Message(content=f"❌ 文档加载失败：{str(e)}").send()

    await send_toolbar()