"""应用级常量配置（不依赖 Streamlit）：所有可调参数集中在此，便于统一维护"""


class AppConfig:
    """应用级常量配置：所有可调参数集中在此，便于统一维护"""

    # ---- 会话持久化 ----
    SESSION_FILE_DIR = "./session_data"        # 会话数据目录
    SESSION_FILE_NAME = "session_cache.json"   # 会话数据文件名
    SESSION_FILE_VERSION = 2                   # 会话文件格式版本号（结构变更时递增）

    # ---- 默认会话参数 ----
    DEFAULT_DARK_MODE = True                   # 启动时默认夜间模式
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2000
    MAX_CONTEXT_MSGS = 50                      # 最大上下文消息数

    # ---- 消息与回答保护 ----
    MAX_MESSAGE_LENGTH = 10000                 # 单条用户消息最大字符数
    MAX_RENDER_MESSAGES = 50                   # 聊天区最多渲染的消息条数（虚拟滚动）
    STREAM_STALL_TIMEOUT = 30                  # 流式响应相邻数据块最大间隔（秒）
    MAX_TOOL_RESULT_CHARS = 4000               # 工具返回结果最大字符数
    MAX_UPLOAD_SIZE = 10 * 1024 * 1024         # 上传文件最大字节数（10MB）

    # ---- 系统默认提示词 ----
    DEFAULT_SYSTEM_PROMPT = """你是一位可爱且专业的AI助理喔~。你的特点：
1. 回答要亲切友好，使用适当的语气词（呢、哦、呀）
2. 提供准确、有用的信息
3. 在不确定时坦诚说明
4. 适当使用表情符号增加亲和力
5. 回答要简洁明了，避免过于冗长"""

    # ---- 快速提问模板 ----
    QUICK_QUESTIONS = [
        "你好！请介绍一下你自己",
        "帮我写一段Python代码示例",
        "解释一下什么是人工智能",
        "给我一些学习建议"
    ]

    # ---- 兜底模型列表（仅多模型模块不可用时使用；正常情况用 core/models.py 配置） ----
    MODEL_LIST = [
        "deepseek-chat",      #聊天模型
        "deepseek-reasoner"   #推理模型
    ]

# 向后兼容的模块级别名（其余代码仍可直接引用这些名字）
DEFAULT_SYSTEM_PROMPT = AppConfig.DEFAULT_SYSTEM_PROMPT
QUICK_QUESTIONS = AppConfig.QUICK_QUESTIONS
MODEL_LIST = AppConfig.MODEL_LIST
