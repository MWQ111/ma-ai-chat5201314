"""
ui 包：UI 层（仅依赖 Streamlit，只负责渲染与交互组装）。

- sidebar：侧边栏（API 配置 / 高级参数 / RAG 文档管理 / 工具与缓存设置 / 对话管理）
- chat：聊天界面（消息渲染 + 用户消息处理流程）
- components：通用组件（全局主题 CSS 注入、toast 轻提示等）

业务逻辑一律位于 core/ 包（不依赖 Streamlit）。
"""
