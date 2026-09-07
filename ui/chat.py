"""聊天界面 UI：消息渲染 + 用户消息处理流程（RAG 检索 → 缓存/Agent/工具循环 → 持久化）"""

import base64
import logging

import streamlit as st

from core import (
    AGENT_AVAILABLE,
    CACHE_AVAILABLE,
    RAG_AVAILABLE,
    TOOLS_AVAILABLE,
    AppConfig,
    QUICK_QUESTIONS,
    save_session_to_file,
)
from core.chat import run_tool_loop, trim_context_messages

logger = logging.getLogger("ai_chat.ui.chat")

# 以下功能模块依赖第三方库，缺失时对应处理分支自动降级（与重构前行为一致）
try:
    from core.rag import search
except ImportError:
    pass
try:
    from core.tools import build_tools_ack, build_tools_directive, get_available_tools, has_tool_denial
except ImportError:
    pass
try:
    from core.cache import get_cached_response, set_cached_response
except ImportError:
    pass
try:
    from core.agent import run_agent
except ImportError:
    pass


def render_copy_button(content: str) -> None:
    """在 AI 回复下方渲染一个“复制”按钮

    用 st.iframe（srcdoc 模式）生成一个自包含的小 iframe：点击时在 iframe 内部把
    整段回复写入剪贴板（优先 navigator.clipboard，失败回退 execCommand），并在
    按钮旁短暂显示“✅ 已复制”提示。内容以 base64 内嵌，避免换行/引号等特殊字符
    破坏 HTML/JS 字符串。服务端不接触剪贴板，因此远程部署也能复制到“客户端”剪贴板。

    Args:
        content: 要复制到剪贴板的回复原文
    """
    if not content:
        return
    b64 = base64.b64encode(content.encode("utf-8")).decode("ascii")
    dark = bool(st.session_state.get("dark_mode", False))
    # 配色与 ui/components.py 注入的全局主题保持一致（两套变量内联进 iframe）
    if dark:
        bg, text_c, border, accent, hover = "#1a1d23", "#9aa3b2", "#2a3040", "#667eea", "#232833"
    else:
        bg, text_c, border, accent, hover = "#ffffff", "#5a6472", "#d8dde6", "#667eea", "#e9edf5"
    html = f"""
<style>
  body {{ margin: 0; background: transparent; font-family: -apple-system, "Segoe UI", Roboto, "Noto Sans SC", "Microsoft YaHei", sans-serif; }}
  .row {{ display: flex; justify-content: flex-end; align-items: center; gap: 8px; }}
  button {{
    border: 1px solid {border}; background: {bg}; color: {text_c};
    border-radius: 8px; font-size: 12px; line-height: 22px;
    padding: 0 12px; cursor: pointer;
  }}
  button:hover {{ border-color: {accent}; color: {accent}; background: {hover}; }}
  #tip {{ font-size: 12px; color: {text_c}; opacity: 0; transition: opacity .2s; }}
  #tip.show {{ opacity: 1; }}
</style>
<div class="row">
  <span id="tip"></span>
  <button id="copy-btn">📋 复制</button>
</div>
<script>
  var b64 = "{b64}";
  var btn = document.getElementById("copy-btn");
  var tip = document.getElementById("tip");
  function showTip(t) {{
    tip.textContent = t;
    tip.classList.add("show");
    setTimeout(function () {{ tip.classList.remove("show"); }}, 1500);
  }}
  function copyFallback(t) {{
    var ta = document.createElement("textarea");
    ta.value = t;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.focus();
    ta.select();
    var ok = false;
    try {{ ok = document.execCommand("copy"); }} catch (e) {{}}
    document.body.removeChild(ta);
    showTip(ok ? "✅ 已复制" : "⚠️ 复制失败");
  }}
  btn.addEventListener("click", function () {{
    var text;
    try {{ text = decodeURIComponent(escape(atob(b64))); }}
    catch (e) {{ text = b64; }}
    try {{
      if (navigator.clipboard && window.isSecureContext) {{
        navigator.clipboard.writeText(text).then(
          function () {{ showTip("✅ 已复制"); }},
          function () {{ copyFallback(text); }}
        );
        return;
      }}
    }} catch (e) {{}}
    copyFallback(text);
  }});
</script>
"""
    st.iframe(html, height=32)


# ====================== 回复完成后的提示队列 ======================
# 回复流程结束时 handle_user_message 会调用一次顶层 st.rerun()，让刚写进
# 历史列表的消息由 display_chat_messages() 重新渲染（从而自动挂出复制按钮）。
# 但 st.warning/st.error 这类“瞬时提示”随 rerun 刷新会消失，这里先把它们
# 记入队列，供下一次渲染在消息列表下方补显示一次，避免错误提示丢失。
def _notice_warning(text: str) -> None:
    """把一条警告记入回复提示队列（下次刷新时显示）"""
    st.session_state.setdefault("_reply_notices", []).append(("warning", text))


def _notice_error(text: str) -> None:
    """把一条错误记入回复提示队列（下次刷新时显示）"""
    st.session_state.setdefault("_reply_notices", []).append(("error", text))


def _drain_reply_notices() -> None:
    """在历史消息列表之后渲染并清空上次回复遗留的提示"""
    for kind, text in st.session_state.pop("_reply_notices", []):
        if kind == "error":
            st.error(text)
        else:
            st.warning(text)


def _reasoning_panel_factory():
    """思考过程展示面板工厂：首个思考片段到达时创建可折叠面板并返回其内部占位符

    由 core.chat.run_tool_loop 在流式接收期间回调（本层负责 Streamlit 渲染，
    core 层因此无需依赖 Streamlit）。
    """
    with st.expander("💭 思考过程", expanded=True):
        return st.empty()


# ====================== 页面渲染函数 ======================
def display_chat_messages() -> None:
    """渲染聊天记录（虚拟滚动：仅渲染最近 MAX_RENDER_MESSAGES 条，超长会话不卡顿）"""
    if not st.session_state.messages:
        st.markdown("### 💡 快捷提问")
        cols = st.columns(2)
        for idx, question in enumerate(QUICK_QUESTIONS):
            with cols[idx % 2]:
                if st.button(question, key=f"quick_{idx}", use_container_width=True):
                    st.session_state.pending_message = question
                    st.rerun()
        return

    # 虚拟滚动：只渲染最近 50 条消息，避免超长会话时页面卡死
    all_messages = st.session_state.messages
    if len(all_messages) > AppConfig.MAX_RENDER_MESSAGES:
        st.caption(f"📜 共 {len(all_messages)} 条消息，仅渲染最近 {AppConfig.MAX_RENDER_MESSAGES} 条")
        all_messages = all_messages[-AppConfig.MAX_RENDER_MESSAGES:]

    for i, msg in enumerate(all_messages):
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant", avatar="🤖"):
                # 历史消息中带有思考过程时先展示（可折叠，最终回答在下方）
                if msg.get("reasoning"):
                    with st.expander("💭 思考过程", expanded=False):
                        st.markdown(msg["reasoning"])
                st.markdown(msg["content"])
                # 历史消息中带有引用来源时一并展示
                if msg.get("sources"):
                    with st.expander("📚 参考来源", expanded=False):
                        for i, src in enumerate(msg["sources"], 1):
                            st.caption(f"📄 来源{i}：{src['source']}　·　相似度 {src['score']:.2f}")
                # 历史消息中带有工具调用记录时一并展示
                if msg.get("tools_used"):
                    st.caption("🔧 调用了工具：" + "、".join(msg["tools_used"]))
                # 历史消息中命中缓存时一并展示
                if msg.get("cache_hit"):
                    st.caption("📦 来自缓存")
                # 复制按钮：每条 AI 回复底部（仅在有正文时展示）
                if msg.get("content"):
                    render_copy_button(msg["content"])

    # 上次回复遗留的提示（若刚完成一次流式回复并触发了刷新，这里会补显示一次）
    _drain_reply_notices()


# ====================== 用户消息处理流程 ======================
def handle_user_message(process_msg: str) -> None:
    """处理用户消息：长度校验 → RAG 检索 → 缓存/Agent/工具循环 → 渲染回复 → 持久化

    Args:
        process_msg: 用户输入（或快捷提问）的原始文本
    """
    if not process_msg.strip():
        st.warning("请勿发送空消息！")
        st.stop()

    # 消息长度限制：超长输入会挤爆上下文窗口，直接拦截
    if len(process_msg) > AppConfig.MAX_MESSAGE_LENGTH:
        st.warning(f"⚠️ 消息过长（{len(process_msg)} 字符），"
                   f"最多允许 {AppConfig.MAX_MESSAGE_LENGTH} 字符，请精简后再发送")
        st.stop()

    with st.chat_message("user", avatar="👤"):
        st.markdown(process_msg)

    st.session_state.messages.append({"role": "user", "content": process_msg})

    # RAG 检索：先从向量库中查找与问题相关的文档片段
    rag_sources = []
    if RAG_AVAILABLE and st.session_state.rag_enabled:
        try:
            rag_sources = search(process_msg, top_k=3)
        except Exception as e:
            logger.warning("文档检索失败：%s", e)
            _notice_warning(f"⚠️ 文档检索失败：{str(e)}")

    # 判断是否启用工具调用：启用后一律尝试（deepseek-chat / reasoner 均已实测支持
    # 工具调用）；个别不支持工具的模型由 run_tool_loop 在请求失败后自动降级，
    # 并在回答中给出明确提示，绝不静默跳过
    tools_for_call = None
    if TOOLS_AVAILABLE and st.session_state.tools_enabled:
        tools_for_call = get_available_tools()

    # 组装 API 消息列表：
    # 1) 仅保留 role/content —— 会话消息携带 sources/tools_used/reasoning 等展示字段，
    #    原样发送会污染接口请求；
    # 2) 启用工具时，把工具调用指令追加到系统提示词（声明工具可用并要求必须调用）；
    # 3) 历史中出现过助手"否认工具能力"的回复时，追加一条"工具已接入"的确认回合，
    #    破除模型对旧回复的模仿（否则即使请求携带了工具定义，模型也会继续拒绝调用）。
    api_msgs = [{"role": "system", "content": st.session_state.system_prompt}]
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
    api_msgs += [{"role": m["role"], "content": m["content"]}
                 for m in st.session_state.messages]
    if tools_for_call is not None and has_tool_denial(api_msgs):
        api_msgs.append({"role": "assistant", "content": build_tools_ack()})
    api_msgs = trim_context_messages(api_msgs, st.session_state)

    with st.chat_message("assistant", avatar="🤖"):
        res_placeholder = st.empty()
        tool_placeholder = st.empty()

        full_response = ""
        tools_used = []
        reasoning_text = ""
        cache_hit = False
        err = None

        # 缓存查找：仅普通对话可用缓存（有RAG检索或启用工具时跳过，避免旧答案/旧时间）
        use_cache = (CACHE_AVAILABLE and st.session_state.cache_enabled
                     and not rag_sources and tools_for_call is None)
        if use_cache:
            # 缓存降级保护：Redis 异常时跳过缓存直接走 API，不影响正常对话
            try:
                cached = get_cached_response(process_msg, model=st.session_state.current_model)
            except Exception as e:
                logger.warning("缓存读取失败，跳过缓存：%s", e)
                cached = None
            if cached:
                full_response = cached
                cache_hit = True
                res_placeholder.markdown(full_response)
                st.caption("📦 来自缓存")

        if not cache_hit:
            # Agent 模式（LangGraph）：规划 + 反思 + 自主结束，替代固定轮数的工具循环；
            # 仅在 Agent 模块可用、开关打开且启用了工具时生效，否则回退普通模式
            if (AGENT_AVAILABLE and st.session_state.agent_mode_enabled
                    and tools_for_call is not None):
                full_response, tools_used, err, reasoning_text = run_agent(
                    api_msgs,
                    content_placeholder=res_placeholder,
                    tool_placeholder=tool_placeholder,
                    tools=tools_for_call,
                    max_steps=st.session_state.agent_max_steps,
                )
            else:
                full_response, tools_used, err, reasoning_text = run_tool_loop(
                    st.session_state, api_msgs, res_placeholder, tool_placeholder,
                    tools_for_call, reasoning_placeholder_factory=_reasoning_panel_factory,
                )
            if err:
                if full_response:
                    # 已收到部分内容：保留内容并附上中断提示，而不是覆盖掉
                    res_placeholder.markdown(full_response)
                    _notice_warning(err)  # rerun 刷新后在下一条消息下方补显示
                else:
                    _notice_error(err)  # rerun 刷新后在下一条消息下方补显示
            elif use_cache and full_response:
                # 回答生成后写入缓存，供下次相同问题直接命中（失败只记日志，不影响对话）
                try:
                    set_cached_response(process_msg, full_response,
                                        model=st.session_state.current_model)
                except Exception as e:
                    logger.warning("缓存写入失败：%s", e)

        # 在回复下方显示检索到的参考来源
        if rag_sources:
            with st.expander("📚 参考来源", expanded=False):
                for i, src in enumerate(rag_sources, 1):
                    st.caption(f"📄 来源{i}：{src['source']}　·　相似度 {src['score']:.2f}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": rag_sources,
        "tools_used": tools_used,
        "cache_hit": cache_hit,
        "reasoning": reasoning_text,
    })
    if st.session_state.conversations:
        conv_idx = st.session_state.current_conversation_index
        if 0 <= conv_idx < len(st.session_state.conversations):
            st.session_state.conversations[conv_idx]["messages"] = st.session_state.messages.copy()
        else:
            # 防御：索引越界时只记录日志，不崩溃（正常路径不会走到这里）
            logger.error("当前对话索引 %s 越界（共 %d 条），跳过消息同步",
                         conv_idx, len(st.session_state.conversations))
    save_session_to_file(st.session_state)

    # 流式回复已写入历史（messages）并持久化。复制按钮是 display_chat_messages()
    # 渲染“历史消息”时挂上去的，而本条消息在本轮开始时尚未存在；这里在主脚本
    # 顶层（非回调、非循环）调用一次 st.rerun()，让新一轮把本条消息按历史渲染，
    # 复制按钮随即自动出现。回复完成即刷新一次，不会重复触发。
    st.rerun()
