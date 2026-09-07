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
    /* 主视图容器与底部栏（聊天输入框所在区域）：跟随页面底色。
       stBottom 在 1.57 中自带主题底色，是日间模式下输入框周围
       仍发黑的元凶，显式覆盖 */
    [data-testid="stAppViewContainer"] {{
        background-color: {bg_color};
    }}
    [data-testid="stBottom"] {{
        background-color: {bg_color};
        border-top-color: {border_color};
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
    /* 侧边栏按钮圆角与悬停动效 */
    section[data-testid="stSidebar"] .stButton > button {{
        border-radius: 10px;
        transition: 0.2s;
    }}
    section[data-testid="stSidebar"] .stButton > button:hover {{
        transform: translateY(-1px);
    }}

    /* ============ ④ 按钮（普通 / 下载 / 链接 / 表单提交） ============ */
    [data-testid="stButton"] > button,
    [data-testid="stDownloadButton"] > button,
    [data-testid="stLinkButton"] > button,
    [data-testid="stFormSubmitButton"] {{
        background-color: {surface_color} !important;
        color: {text_color} !important;
        border-color: {border_color} !important;
    }}
    [data-testid="stButton"] > button:hover,
    [data-testid="stDownloadButton"] > button:hover,
    [data-testid="stLinkButton"] > button:hover,
    [data-testid="stFormSubmitButton"]:hover {{
        background-color: {hover_color} !important;
        border-color: {accent_color} !important;
        color: {accent_color} !important;
    }}

    /* ============ ⑤ 输入框（文本 / 数字 / 文本域） ============ */
    /* 外层容器置透明，杜绝主题底色残留（1.57 中这些控件已不用
       baseweb，可见盒子就是 input/textarea 元素本身） */
    [data-testid="stTextInput"],
    [data-testid="stNumberInput"],
    [data-testid="stTextArea"] {{
        background-color: transparent !important;
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
    /* 聚焦时描边高亮为品牌色 */
    [data-testid="stTextInput"] input:focus,
    [data-testid="stNumberInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {{
        border-color: {accent_color} !important;
    }}
    /* 数字输入框的加减按钮 */
    [data-testid="stNumberInput"] button {{
        color: {text_color} !important;
        background-color: transparent !important;
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
    /* 展开的下拉列表（渲染在 body 级 Portal 中）：底色与文字跟随模式 */
    [data-testid="stSelectboxVirtualDropdown"] [data-baseweb="menu"] {{
        background-color: {input_bg} !important;
    }}
    [data-testid="stSelectboxVirtualDropdown"] [data-baseweb="menu"] li {{
        color: {text_color} !important;
    }}

    /* ============ ⑦ 聊天输入框（打字框） ============ */
    /* stChatInputTextArea 可能落在 textarea 本身或其包装元素上，
       两种情况的选择器都写上，保证命中；内层输入框置透明底色，
       露出容器底色，避免深浅色混搭 */
    [data-testid="stChatInput"] {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
    }}
    /* 中间包装层一律透明，透出容器底色；baseweb 输入根层
       （data-baseweb="input"）在 1.57 中自带主题底色，是打字框
       在日间模式下仍发黑的元凶，显式覆盖为控件底色 */
    [data-testid="stChatInput"] div {{
        background-color: transparent !important;
    }}
    [data-testid="stChatInput"] [data-baseweb="input"] {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
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
        color: {text_color};
        background-color: transparent;
    }}
    [data-testid="stExpander"] summary:hover,
    [data-testid="stExpander"] summary span {{
        color: {text_color};
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
        background-color: {surface_color};
        border: 1px solid {border_color};
        border-radius: 10px;
    }}
    [data-testid="stMetricLabel"] {{
        color: {muted_color} !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {text_color} !important;
    }}

    /* ============ ⑪ 提示框（info/success/warning/error） ============ */
    /* 语义颜色由图标与 ❌/⚠️/✅ 表情符号传达，底色统一为模式中性色，
       保证两种模式下文字都清晰可读 */
    [data-testid="stAlertContainer"] {{
        background-color: transparent;
    }}
    [data-testid="stAlert"] {{
        background-color: {alert_bg} !important;
        border: 1px solid {border_color};
    }}
    [data-testid="stAlert"] .stMarkdown,
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p {{
        color: {text_color} !important;
    }}

    /* ============ ⑫ 复选框 / 开关 / 单选框 / 滑块 ============ */
    /* st.toggle 与 st.checkbox 共用 stCheckbox testid（1.57 前端包核实） */
    [data-testid="stCheckbox"] label div,
    [data-testid="stRadio"] label div,
    [data-testid="stSlider"] {{
        color: {text_color} !important;
    }}

    /* ============ ⑬ 文件上传器 ============ */
    [data-testid="stFileUploader"] {{
        background-color: transparent;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background-color: {surface_color} !important;
        border-color: {border_color} !important;
    }}
    [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] small {{
        color: {text_color} !important;
    }}
    [data-testid="stFileUploaderDropzone"] button {{
        background-color: {hover_color} !important;
        color: {text_color} !important;
        border-color: {border_color} !important;
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

    /* ============ ⑮ 代码块 / 表格 / 进度条 / 弹层 / 表单 ============ */
    [data-testid="stCode"], [data-testid="stCode"] pre,
    [data-testid="stCode"] code, .stMarkdown code {{
        background-color: {code_bg} !important;
        color: {text_color} !important;
    }}
    [data-testid="stDataFrame"] {{
        background-color: transparent;
    }}
    [data-testid="stProgress"] {{
        background-color: {border_color};
    }}
    [data-testid="stSpinner"] {{
        color: {text_color};
    }}
    [data-testid="stToast"] {{
        background-color: {surface_color} !important;
        border: 1px solid {border_color};
    }}
    /* 帮助提示浮层 / 弹窗内容：跟随模式 */
    [data-testid="stTooltipContent"],
    [data-testid="stPopoverBody"] {{
        background-color: {surface_color};
        color: {text_color};
    }}
    [data-testid="stForm"] {{
        background-color: transparent;
        border-color: {border_color};
    }}

    /* ============ ⑯ 滚动条美化 ============ */
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
