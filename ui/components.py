"""通用 UI 组件（仅依赖 Streamlit）：全局主题 CSS 注入、toast 轻提示等"""

import streamlit as st


def toast(message: str, icon: str | None = None) -> None:
    """toast 轻提示封装：st.toast 的薄封装，统一图标前缀（供各界面模块复用）"""
    st.toast(f"{icon} {message}" if icon else message)


def inject_custom_css() -> None:
    """注入全局主题 CSS（深浅两套变量 + 全部控件样式）

    重要：本函数必须在所有页面元素渲染之前调用（紧跟 init_session_state）。
    AI 流式回答期间脚本会长时间阻塞在流式循环里，若 CSS 在脚本末尾才注入，
    回答完成前浏览器拿不到样式，页面会退回 Streamlit 默认主题（跟随系统
    深浅色），出现"思考回答时是夜间模式、完成后才恢复"的闪变问题。
    因此：① 提前注入；② 除页面与气泡外，侧边栏/按钮/输入框/折叠面板等
    控件底色也随模式一起切换，保证深浅两种模式下界面颜色始终一致。
    """
    # ---- 颜色方案：日间/夜间两套变量完全分离，集中定义便于维护 ----
    accent_color = "#667eea"    # 品牌强调色（两种模式通用：激活边框/高亮）
    accent_gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"  # 用户气泡/滚动条渐变
    # 聊天输入框描边：两种模式通用（浅/深背景下都醒目，故不随模式变化）
    input_ring = "#e53e3e"          # 常态描边：醒目红
    input_ring_focus = "#c53030"    # 聚焦描边：加深，配合加粗提示当前输入焦点
    if st.session_state.dark_mode:
        # ============ 夜间模式 ============
        bg_color = "#0e1117"            # 页面/侧边栏背景
        surface_color = "#1a1d23"       # 卡片/控件底色（气泡、输入框、折叠面板、指标卡、下拉列表）
        hover_color = "#232833"         # 按钮悬停底色
        text_color = "#e8edf5"          # 正文/标题文字
        muted_color = "#9aa3b2"         # 次要文字（caption、占位提示、时间戳）
        border_color = "#2a3040"        # 控件描边/分割线/滚动条轨道
        input_bg = "#262a33"            # 输入类控件底色（打字框/文本框/选择框，略浅于卡片底）
        input_border = "#2d333b"        # 输入类控件描边
        code_bg = "#161b22"             # 代码底色
        alert_bg = "#20242e"            # 提示框底色
    else:
        # ============ 日间模式 ============
        bg_color = "#f5f7fb"            # 页面/侧边栏背景
        surface_color = "#ffffff"       # 卡片/控件底色
        hover_color = "#e9edf5"         # 按钮悬停底色
        text_color = "#1a1a2e"          # 正文/标题文字
        muted_color = "#5a6472"         # 次要文字
        border_color = "#d8dde6"        # 控件描边/分割线/滚动条轨道
        input_bg = "#ffffff"            # 输入类控件底色
        input_border = "#dce0e8"        # 输入类控件描边
        code_bg = "#f0f2f6"             # 代码底色
        alert_bg = "#eef2f8"            # 提示框底色

    st.markdown(f"""
<style>
    /* ============ ① 全局基础：页面 / 顶部工具条 / 主容器 / 字体 ============ */
    html, body, .stApp {{
        background-color: {bg_color};
    }}
    .stApp {{
        color: {text_color};
    }}
    /* 顶部工具条：与页面同底，避免切换模式后残留系统主题色 */
    header[data-testid="stHeader"] {{
        background-color: {bg_color};
    }}
    /* 主内容容器透明，透出页面底色 */
    [data-testid="stMainBlockContainer"] {{
        background-color: transparent;
    }}
    /* 主视图容器与底部栏（聊天输入框所在区域）：背景完全由应用内 dark_mode
       决定，不跟随 Streamlit 原生主题。1.57 的底部 DOM 为
       stBottom > stBottomBlockContainer > stChatInput，其中
       stBottomBlockContainer 自带原生主题底色，是「应用内已切浅色、底部
       仍发黑」的元凶；故把 stBottom 的全部直接子元素与伪元素一并覆盖，
       并禁用可能残留的原生渐变/阴影。 */
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
    }}
    [data-testid="stBottom"],
    [data-testid="stBottom"] > div,
    [data-testid="stBottomBlockContainer"] {{
        background-color: {bg_color} !important;
        background-image: none !important;
        border-top-color: {border_color};
    }}
    [data-testid="stBottom"]::before,
    [data-testid="stBottom"]::after,
    [data-testid="stBottomBlockContainer"]::before,
    [data-testid="stBottomBlockContainer"]::after {{
        background: none !important;
        background-image: none !important;
    }}
    /* 全局字体 */
    body, .stMarkdown, .stTextInput, .stTextArea,
    .stChatMessage, .stExpander, .stMetric, .stButton {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, "Noto Sans SC", "Microsoft YaHei", sans-serif !important;
    }}

    /* ============ ② 标题 / 正文 / 次要文字 / 分割线 ============ */
    [data-testid="stHeading"], [data-testid="stHeading"] *,
    [data-testid="stText"], .stMarkdown, .stMarkdown p, .stMarkdown li {{
        color: {text_color};
    }}
    /* 次要文字（caption/说明/时间戳）：每种模式独立的中灰，保证可读 */
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
    .stCaption {{
        color: {muted_color} !important;
    }}
    .stMarkdown hr {{
        margin: 8px 0;
        opacity: 0.4;
        border-color: {border_color};
    }}

    /* ============ ③ 侧边栏及其内部全部元素 ============ */
    section[data-testid="stSidebar"] {{
        background-color: {bg_color};
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        background-color: transparent;
    }}
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stHeading"],
    section[data-testid="stSidebar"] label {{
        color: {text_color};
    }}
    /* 控件标签（开关/勾选框/输入框标题等）：文字颜色统一覆盖 */
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] span,
    [data-testid="stWidgetLabel"] p {{
        color: {text_color} !important;
    }}
    /* 侧边栏按钮（历史对话切换等）：默认/悬停/按下三态都显式跟随应用内
       深浅模式配色，不跟随原生主题。1.57 中 stButton 的 testid 落在外层
       壳上、真按钮是其后代，必须用后代选择器（旧的 `.stButton > button`
       因真按钮是孙代元素而从未命中） */
    section[data-testid="stSidebar"] [data-testid="stButton"] button {{
        background-color: {surface_color} !important;
        color: {text_color} !important;
        border-color: {border_color} !important;
        border-radius: 10px;
        transition: 0.2s;
    }}
    section[data-testid="stSidebar"] [data-testid="stButton"] button:hover {{
        background-color: {hover_color} !important;
        border-color: {accent_color} !important;
        color: {accent_color} !important;
        transform: translateY(-1px);
    }}
    section[data-testid="stSidebar"] [data-testid="stButton"] button:active {{
        background-color: {hover_color} !important;
        border-color: {accent_color} !important;
        color: {text_color} !important;
        transform: none;
    }}

    /* ============ ④ 按钮（普通 / 下载 / 链接 / 表单提交 / 按钮组） ============ */
    /* 1.57 实测 DOM：stButton / stLinkButton / stDownloadButton 的 testid 都落在
       **外层壳**上，真按钮是内层 baseweb <button>，中间还隔着 BaseButton 的
       tooltip 容器层，因此必须用**后代**选择器。旧规则用直接子代 `> button`，
       从未命中真按钮 —— 这就是「浅色模式下侧边栏历史对话按钮仍发黑」的根因
       （真按钮一直穿着原生主题的深色）。外壳若不是 button 本身则置透明，
       杜绝主题底色残留 */
    [data-testid="stButton"] button,
    [data-testid="stDownloadButton"] button,
    [data-testid="stLinkButton"] button,
    [data-testid="stFormSubmitButton"] button,
    [data-testid="stButtonGroup"] button,
    button[data-testid="stButton"],
    button[data-testid="stDownloadButton"] {{
        background-color: {surface_color} !important;
        color: {text_color} !important;
        border-color: {border_color} !important;
    }}
    [data-testid="stButton"] button:hover,
    [data-testid="stDownloadButton"] button:hover,
    [data-testid="stLinkButton"] button:hover,
    [data-testid="stFormSubmitButton"] button:hover,
    [data-testid="stButtonGroup"] button:hover {{
        background-color: {hover_color} !important;
        border-color: {accent_color} !important;
        color: {accent_color} !important;
    }}
    /* 按下态：底色同悬停、文字压回正文色，明确「正在按压」 */
    [data-testid="stButton"] button:active,
    [data-testid="stDownloadButton"] button:active,
    [data-testid="stLinkButton"] button:active,
    [data-testid="stFormSubmitButton"] button:active,
    [data-testid="stButtonGroup"] button:active {{
        background-color: {hover_color} !important;
        border-color: {accent_color} !important;
        color: {text_color} !important;
    }}
    /* 禁用态：次要文字色 + 恢复禁用光标 */
    [data-testid="stButton"] button:disabled,
    [data-testid="stDownloadButton"] button:disabled,
    [data-testid="stLinkButton"] button:disabled,
    [data-testid="stFormSubmitButton"] button:disabled,
    [data-testid="stButtonGroup"] button:disabled {{
        color: {muted_color} !important;
        cursor: not-allowed;
    }}
    /* 聚焦态：禁用浏览器/baseweb 默认焦点环（含主题色 box-shadow 环），
       改用品牌色 2px 环标记键盘焦点 */
    [data-testid="stButton"] button:focus-visible,
    [data-testid="stDownloadButton"] button:focus-visible,
    [data-testid="stLinkButton"] button:focus-visible,
    [data-testid="stFormSubmitButton"] button:focus-visible,
    [data-testid="stButtonGroup"] button:focus-visible {{
        outline: none !important;
        box-shadow: 0 0 0 2px {accent_color} !important;
    }}
    [data-testid="stButton"]:not(button):not(a),
    [data-testid="stDownloadButton"]:not(button):not(a),
    [data-testid="stLinkButton"]:not(button):not(a),
    [data-testid="stFormSubmitButton"],
    [data-testid="stButtonGroup"] {{
        background-color: transparent !important;
    }}

    /* ============ ⑤ 输入框（文本 / 数字 / 文本域 / 下拉选择） ============ */
    /* 外层容器置透明，杜绝主题底色残留（1.57 中文本类控件已不用
       baseweb，可见盒子就是 input/textarea 元素本身） */
    [data-testid="stTextInput"],
    [data-testid="stNumberInput"],
    [data-testid="stTextArea"],
    [data-testid="stSelectbox"],
    [data-testid="stMultiSelect"] {{
        background-color: transparent !important;
    }}
    /* 1.57 的可见盒子还有 *RootElement / *Container 一层（testid 与控件同名
       那层只是外壳），只改外壳会让内层继续露原生底色 */
    [data-testid="stTextInputRootElement"],
    [data-testid="stTextAreaRootElement"],
    [data-testid="stNumberInputContainer"] {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
    }}
    /* 底色 + 文字 + 光标 + 描边全部覆盖（-webkit-text-fill-color 兜底
       浏览器特殊渲染如自动填充） */
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stTextArea"] textarea {{
        background-color: {input_bg} !important;
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
        caret-color: {text_color};
        border-color: {input_border} !important;
    }}
    /* 占位提示文字：两种模式各自的中灰 */
    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stNumberInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {{
        color: {muted_color} !important;
        opacity: 1;
    }}
    /* 聚焦时描边高亮为品牌色，同时禁用浏览器默认 focus 环（outline）与
       baseweb 的 focus box-shadow，聚焦指示统一由品牌描边承担 */
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus,
    [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
    [data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within {{
        border-color: {accent_color} !important;
        outline: none !important;
        box-shadow: none !important;
    }}
    /* baseweb select 根元素自带 tabindex，focus-visible 会落在外层 div 上 */
    [data-testid="stSelectbox"] [data-baseweb="select"]:focus-visible,
    [data-testid="stMultiSelect"] [data-baseweb="select"]:focus-visible {{
        outline: none !important;
        box-shadow: none !important;
    }}
    /* 数字输入框的加减按钮 */
    [data-testid="stNumberInput"] button,
    [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInputStepUp"] {{
        color: {text_color} !important;
        background-color: transparent !important;
        border-color: {border_color} !important;
    }}
    /* ---- 下拉选择（selectbox / multiselect）----
       1.57 中它们仍走 baseweb（data-baseweb="select"/"menu"/"tag"），
       控件本体不在 stSelectbox 内部而由 baseweb 渲染，故各层都要显式覆盖 */
    [data-testid="stSelectbox"] [data-baseweb="select"] > div:first-child,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div:first-child {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"] *,
    [data-testid="stMultiSelect"] [data-baseweb="select"] * {{
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"] svg,
    [data-testid="stMultiSelect"] [data-baseweb="select"] svg {{
        fill: {text_color} !important;
    }}
    /* 多选已选中的标签（tag） */
    [data-testid="stMultiSelect"] [data-baseweb="tag"] {{
        background-color: {hover_color} !important;
        color: {text_color} !important;
    }}
    [data-testid="stMultiSelect"] [data-baseweb="tag"] svg {{
        fill: {text_color} !important;
    }}

    /* ============ ⑥ 下拉选择框（闭合状态 + 展开列表） ============ */
    /* 外层容器置透明，杜绝主题底色残留 */
    [data-testid="stSelectbox"] {{
        background-color: transparent !important;
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
    }}
    [data-testid="stSelectbox"] [data-baseweb="select"] span,
    [data-testid="stSelectbox"] [data-baseweb="select"] input {{
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
    }}
    /* （下拉列表等 Portal 浮层统一收敛到 ⑯ 处理，避免两处规则互相覆盖） */

    /* ============ ⑦ 聊天输入框（打字框） ============ */
    /* 红色描边：两种模式通用，保证输入框在浅/深背景下都醒目可辨。
       stChatInputTextArea 可能落在 textarea 本身或其包装元素上，
       两种情况的选择器都写上，保证命中；内层输入框置透明底色，
       露出容器底色，避免深浅色混搭 */
    [data-testid="stChatInput"] {{
        background-color: {input_bg} !important;
        border: 2px solid {input_ring} !important;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }}
    /* 中间包装层一律透明，透出容器底色；baseweb 输入根层
       （data-baseweb="input"）在 1.57 中自带主题底色，是打字框
       在日间模式下仍发黑的元凶，显式覆盖为控件底色 */
    [data-testid="stChatInput"] div {{
        background-color: transparent !important;
    }}
    [data-testid="stChatInput"] [data-baseweb="input"] {{
        background-color: {input_bg} !important;
        border: 2px solid {input_ring} !important;
    }}
    /* 聚焦态：描边加深并加一圈外发光，明确提示当前输入焦点；
       同时禁用浏览器/baseweb 默认 focus 环，聚焦指示统一由红色描边承担 */
    [data-testid="stChatInput"]:focus-within,
    [data-testid="stChatInput"]:focus-within [data-baseweb="input"] {{
        border-color: {input_ring_focus} !important;
        box-shadow: 0 0 0 1px {input_ring_focus} !important;
        outline: none !important;
    }}
    [data-testid="stChatInputTextArea"],
    [data-testid="stChatInputTextArea"] textarea,
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] input {{
        background-color: transparent !important;
        color: {text_color} !important;
        -webkit-text-fill-color: {text_color} !important;
        caret-color: {text_color};
    }}
    [data-testid="stChatInputTextArea"]::placeholder,
    [data-testid="stChatInputTextArea"] textarea::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {{
        color: {muted_color} !important;
        opacity: 1;
    }}

    /* ============ ⑧ 折叠面板（思考过程/参考来源/对话统计） ============ */
    [data-testid="stExpander"] {{
        background-color: {surface_color} !important;
        border: 1px solid {border_color};
    }}
    [data-testid="stExpander"] summary {{
        color: {text_color} !important;
        background-color: transparent !important;
    }}
    /* 悬停/展开态：1.57 会给 summary 加主题色底，需显式压平 */
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary:focus,
    [data-testid="stExpander"][open] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary svg {{
        color: {text_color} !important;
        background-color: transparent !important;
        fill: {text_color} !important;
    }}
    /* 展开内容区：1.57 中该层自带主题底色（日间模式下展开后
       整块变黑的元凶），强制透明，透出折叠面板的控件底色 */
    [data-testid="stExpanderDetails"] {{
        background-color: transparent !important;
    }}

    /* ============ ⑨ 聊天消息气泡 ============ */
    .stChatMessage {{
        padding: 1rem;
        border-radius: 16px;
        margin: 0.6rem 0;
    }}
    /* 用户消息气泡：紫色渐变 + 白色文字。
       选择器说明：Streamlit 1.57 实际 DOM 中没有 stChatMessage-user /
       stChatMessage-assistant 这两个 testid（旧选择器匹配不到任何元素），
       消息角色信息在内容区的 aria-label 上（"Chat message from user" /
       "Chat message from assistant"），因此用 :has() 匹配。 */
    [data-testid="stChatMessage-user"],
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) {{
        background: {accent_gradient} !important;
        color: white !important;
    }}
    [data-testid="stChatMessage-user"] [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage-user"] p,
    [data-testid="stChatMessage"]:has([aria-label="Chat message from user"]) p {{
        color: white !important;
    }}
    /* AI 回复气泡：卡片底 + 模式文字色 + 品牌色左边线 */
    [data-testid="stChatMessage-assistant"],
    [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) {{
        background: {surface_color} !important;
        color: {text_color} !important;
        border-left: 4px solid {accent_color};
    }}
    /* AI 回复内部文字显式继承气泡文字颜色（内容区与段落两层都覆盖） */
    [data-testid="stChatMessage-assistant"] [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage-assistant"] p,
    [data-testid="stChatMessage"]:has([aria-label="Chat message from assistant"]) p {{
        color: {text_color} !important;
    }}

    /* ============ ⑩ 指标卡（对话统计） ============ */
    [data-testid="stMetric"] {{
        background-color: {surface_color} !important;
        border: 1px solid {border_color};
        border-radius: 10px;
    }}
    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {{
        color: {muted_color} !important;
    }}
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {{
        color: {text_color} !important;
    }}
    [data-testid="stMetricDelta"] {{
        color: {accent_color} !important;
    }}

    /* ============ ⑪ 提示框（info/success/warning/error） ============ */
    /* 语义颜色由图标与 ❌/⚠️/✅ 表情符号传达，底色统一为模式中性色，
       保证两种模式下文字都清晰可读 */
    [data-testid="stAlertContainer"],
    [data-testid="stAlertContent"] {{
        background-color: transparent !important;
    }}
    [data-testid="stAlert"] {{
        background-color: {alert_bg} !important;
        border: 1px solid {border_color};
    }}
    [data-testid="stAlert"] .stMarkdown,
    [data-testid="stAlert"] p,
    [data-testid="stAlertTitle"],
    [data-testid="stAlertContent"],
    [data-testid="stAlertContent"] *,
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{
        color: {text_color} !important;
    }}

    /* ============ ⑫ 复选框 / 开关 / 单选框 / 滑块 ============ */
    /* st.toggle 与 st.checkbox 共用 stCheckbox testid（1.57 前端包核实） */
    [data-testid="stCheckbox"],
    [data-testid="stCheckbox"] label div,
    [data-testid="stRadio"],
    [data-testid="stRadioGroup"],
    [data-testid="stRadio"] label div,
    [data-testid="stRadio"] label span,
    [data-testid="stSlider"] {{
        color: {text_color} !important;
    }}
    /* 复选框/开关的方块、单选框的圆点：未选中用模式描边色，选中用品牌色，
       避免选中后露出原生主题主色。
       ⚠️ 选择器依据浏览器实测 DOM（1.57，st.toggle）：
         <label data-baseweb="checkbox">
           <div>…</div>            ← 可视方块（第 1 个 div，位于 input **之前**）
           <input type="checkbox">
           <div>…</div>            ← 文字标签（input 之后）
         </label>
       可视方块在 input 之前，只能按 `> div:first-of-type` 定位；早期写的
       `input + div` 按相邻兄弟语义命中的其实是**文字标签**（会把开关文字
       涂上底色），已一并订正。
       radio 的 DOM 未实测，暂沿用旧选择器（尽力而为，匹配不到不影响其它样式） */
    [data-testid="stCheckbox"] [data-baseweb="checkbox"] > div:first-of-type {{
        border-color: {border_color} !important;
        background-color: {input_bg} !important;
    }}
    [data-testid="stCheckbox"] [data-baseweb="checkbox"]:has(input:checked) > div:first-of-type {{
        background-color: {accent_color} !important;
        border-color: {accent_color} !important;
    }}
    [data-testid="stRadio"] input + div {{
        border-color: {border_color} !important;
        background-color: {input_bg} !important;
    }}
    [data-testid="stRadio"] input:checked + div {{
        background-color: {accent_color} !important;
        border-color: {accent_color} !important;
    }}
    /* 滑块：滑块头（baseweb 的 role="slider"）、悬浮数值气泡、刻度文字 */
    [data-baseweb="slider"] [role="slider"] {{
        background-color: {accent_color} !important;
    }}
    [data-testid="stSliderThumbValue"] {{
        background-color: {accent_color} !important;
        color: {surface_color} !important;
    }}
    [data-testid="stSliderTickBar"] {{
        color: {muted_color} !important;
    }}
    /* ---- 聚焦态：禁用原生焦点环，改用品牌色自绘 ----
       浏览器实测（devtools，1.57 st.toggle）：紫色焦点框画在
       <label data-baseweb="checkbox"> **这一层** —— baseweb 的 focus ring 是
       prop 驱动的 box-shadow，画在 label 上、与 :focus-visible 无关。
       早期只压了内部 div，所以没盖住它。现把 label 层与其内部所有层一并
       压掉，再用 :has(input:focus-visible) 在键盘聚焦时给可视方块重画
       品牌色 2px 环（只环住方块/滑块头本体，不包住整行文字）。
       radio 未实测但同属 baseweb 家族，按同样方式处理。 */
    [data-testid="stCheckbox"] [data-baseweb="checkbox"],
    [data-testid="stCheckbox"] [data-baseweb="checkbox"] *,
    [data-testid="stRadio"] [data-baseweb="radio"],
    [data-testid="stRadio"] [data-baseweb="radio"] *,
    [data-testid="stCheckbox"] *:focus-visible,
    [data-testid="stRadio"] *:focus-visible,
    [data-testid="stRadioGroup"] *:focus-visible,
    [data-baseweb="slider"] *:focus-visible {{
        box-shadow: none !important;
        outline: none !important;
    }}
    /* 品牌色重画的环写在后面且特异性更高（:has 计入其参数 (0,1,1)），
       胜过上面那条通配清零规则，保证键盘聚焦仍可见 */
    [data-testid="stCheckbox"] [data-baseweb="checkbox"]:has(input:focus-visible) > div:first-of-type,
    [data-testid="stRadio"] [data-baseweb="radio"]:has(input:focus-visible) > div:first-of-type {{
        box-shadow: 0 0 0 2px {accent_color} !important;
    }}
    [data-baseweb="slider"] [role="slider"]:focus-visible {{
        box-shadow: 0 0 0 2px {accent_color} !important;
    }}

    /* ============ ⑬ 文件上传器 ============ */
    [data-testid="stFileUploader"],
    [data-testid="stFileUploaderDropzoneInput"] {{
        background-color: transparent !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {surface_color} !important;
        border-color: {border_color} !important;
    }}
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] small,
    [data-testid="stFileUploaderDropzoneInstructions"],
    [data-testid="stFileUploaderDropzoneInstructions"] * {{
        color: {text_color} !important;
    }}
    /* 拖拽区图标：默认取主题主色，统一改为次要文字色 */
    [data-testid="stFileUploaderDropzone"] svg {{
        fill: {muted_color} !important;
        color: {muted_color} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: {hover_color} !important;
        color: {text_color} !important;
        border-color: {border_color} !important;
    }}
    /* 已上传文件列表：行背景与文件名文字 */
    [data-testid="stFileUploader"] ul,
    [data-testid="stFileUploader"] li,
    [data-testid="stFileUploader"] li small {{
        background-color: transparent !important;
        color: {text_color} !important;
    }}

    /* ============ ⑭ Tab 标签页 ============ */
    [data-testid="stTabs"] [data-baseweb="tab-list"],
    .stTabs [data-baseweb="tab-list"] {{
        background-color: transparent;
        border-bottom-color: {border_color};
    }}
    [data-testid="stTabs"] [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] {{
        color: {muted_color} !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"],
    .stTabs [data-baseweb="tab"][aria-selected="true"] {{
        color: {text_color} !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-highlight"] {{
        background-color: {accent_color};
    }}

    /* ============ ⑮ 代码块 / 表格 / 进度条 / 表单 ============ */
    [data-testid="stCode"], [data-testid="stCode"] pre,
    [data-testid="stCode"] code,
    [data-testid="stErrorCodeBlock"], [data-testid="stErrorCodeBlock"] pre,
    [data-testid="stErrorCodeBlock"] code,
    .stMarkdown code {{
        background-color: {code_bg} !important;
        color: {text_color} !important;
        border-color: {border_color} !important;
    }}
    /* 表格说明：stDataFrame 内部是 canvas 渲染的 glide-data-grid，单元格画在
       canvas 上、不产生 DOM 节点，CSS 无法着色（列头/网格线由前端 theme 决定，
       属于本文件覆盖不到的区域）。此处只能保证外层容器与包边跟随模式。 */
    [data-testid="stDataFrame"],
    [data-testid="stDataFrameResizable"],
    [data-testid="stDataFrameGlideDataEditor"] {{
        background-color: transparent !important;
        border-color: {border_color} !important;
    }}
    [data-testid="stProgress"] {{
        background-color: {border_color} !important;
    }}
    [data-testid="stSpinner"] {{
        color: {text_color} !important;
    }}
    [data-testid="stForm"] {{
        background-color: transparent !important;
        border-color: {border_color} !important;
    }}

    /* ============ ⑯ Portal 浮层（渲染在 body 下，不继承页面底色） ============
       这些层不位于页面容器内，是最容易残留原生主题颜色的区域：
       下拉菜单、右上角主菜单、模态框、提示气泡、toast。 */
    /* 下拉菜单面板（selectbox / multiselect 的展开层） */
    [data-testid="stSelectboxVirtualDropdown"] [data-baseweb="menu"],
    [data-baseweb="popover"] [data-baseweb="menu"] {{
        background-color: {input_bg} !important;
        border: 1px solid {border_color} !important;
    }}
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] [role="option"] {{
        background-color: transparent !important;
        color: {text_color} !important;
    }}
    [data-baseweb="menu"] li:hover,
    [data-baseweb="menu"] [role="option"]:hover,
    [data-baseweb="menu"] [aria-selected="true"] {{
        background-color: {hover_color} !important;
        color: {text_color} !important;
    }}
    [data-testid="stSelectboxVirtualDropdownEmpty"] {{
        background-color: {input_bg} !important;
        color: {muted_color} !important;
    }}
    /* 右上角主菜单（汉堡菜单）按钮与弹出面板 */
    [data-testid="stMainMenuPopover"],
    [data-testid="stMainMenuList"],
    [data-testid="stMainMenuItem"] {{
        background-color: {surface_color} !important;
        color: {text_color} !important;
    }}
    [data-testid="stMainMenuItemLabel"] {{
        color: {text_color} !important;
    }}
    [data-testid="stMainMenuItem"] svg,
    [data-testid="stMainMenuButton"] svg {{
        fill: {text_color} !important;
        color: {text_color} !important;
    }}
    [data-testid="stMainMenuDivider"] {{
        border-color: {border_color} !important;
    }}
    /* 模态框 / 弹窗：只改对话框本体，避免连遮罩一起染成不透明底色 */
    [data-baseweb="modal"] [role="dialog"],
    [data-testid="stDialog"] [role="dialog"] {{
        background-color: {surface_color} !important;
        color: {text_color} !important;
    }}
    /* 帮助提示气泡 / 弹窗正文 / toast */
    [data-baseweb="tooltip"],
    [data-testid="stTooltipContent"],
    [data-testid="stPopoverBody"] {{
        background-color: {surface_color} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    [data-baseweb="toaster"] [data-baseweb="notification"],
    [data-testid="stToast"] {{
        background-color: {surface_color} !important;
        color: {text_color} !important;
        border: 1px solid {border_color} !important;
    }}
    [data-baseweb="toaster"] [data-baseweb="notification"] *,
    [data-testid="stToast"] * {{
        color: {text_color} !important;
    }}

    /* ============ ⑰ 滚动条美化 ============ */
    ::-webkit-scrollbar {{width: 8px; height: 8px;}}
    ::-webkit-scrollbar-track {{background: {border_color}; border-radius: 4px;}}
    ::-webkit-scrollbar-thumb {{background: {accent_gradient}; border-radius: 4px;}}

    /* 历史对话中删除按钮保持对齐 */
    .stColumn > div {{
        display: flex;
        align-items: center;
        height: 100%;
    }}
</style>
    """, unsafe_allow_html=True)
