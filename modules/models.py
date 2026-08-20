"""
多模型配置模块
===============
集中管理各模型提供方（DeepSeek / OpenAI / Ollama）的配置信息，
提供统一的配置查询接口，主应用通过它动态切换模型。

设计要点：
- 三家提供方统一走 OpenAI SDK 调用（Ollama 提供 OpenAI 兼容接口），无需额外依赖
- API 密钥只从环境变量读取，绝不硬编码；界面输入的值仅保存在会话内存中
- 每个模型有独立的 temperature / max_tokens 默认值，切换模型时自动套用
- Ollama 支持动态获取本地已安装模型列表（服务不可用时优雅降级为手动输入）
"""

import json
import os
import urllib.request

# ====================== 提供方配置表 ======================
# key 为提供方标识；字段说明：
#   label          - 界面显示名称
#   api_key_env    - 密钥对应的环境变量名（本地模型无密钥要求时留空）
#   base_url       - OpenAI 兼容接口地址
#   default_model  - 首次切换到该提供方时默认选中的模型
#   models         - 固定模型列表（Ollama 为空列表表示动态获取本地模型）
#   help           - 界面提示文案
MODEL_CONFIGS = {
    "deepseek": {
        "label": "🔵 DeepSeek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-chat",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "help": "密钥获取：platform.deepseek.com → API Keys",
    },
    "openai": {
        "label": "🟢 OpenAI",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o",
        "models": ["gpt-4o", "gpt-4o-mini"],
        "help": "密钥获取：platform.openai.com → API Keys",
    },
    "ollama": {
        "label": "🟠 Ollama（本地）",
        "api_key_env": "",
        "base_url": "http://localhost:11434/v1",
        "default_model": "qwen2.5:7b",
        "models": [],
        "help": "安装：ollama.com 下载后执行 `ollama pull qwen2.5:7b`",
    },
}

# ====================== 各模型独立默认参数 ======================
# 不在表中的模型（如 Ollama 动态获取的本地模型）使用 DEFAULT_PARAMS
MODEL_PARAMS = {
    "deepseek-chat":     {"temperature": 0.7, "max_tokens": 4096, "supports_tools": True},
    "deepseek-reasoner": {"temperature": 0.7, "max_tokens": 8000, "supports_tools": False},  # 推理模型不支持工具调用
    "gpt-4o":            {"temperature": 0.7, "max_tokens": 4096, "supports_tools": True},
    "gpt-4o-mini":       {"temperature": 0.7, "max_tokens": 4096, "supports_tools": True},
}
DEFAULT_PARAMS = {"temperature": 0.7, "max_tokens": 4096, "supports_tools": True}

# API 请求超时（秒）：reasoner 等推理模型出字较慢，超时设置宽裕
API_TIMEOUT = 120

# 无密钥提供方的占位密钥（OpenAI SDK 要求 api_key 参数非空）
LOCAL_API_KEY_PLACEHOLDER = "ollama"


# ====================== 对外函数 ======================
def list_providers():
    """列出所有可用的模型提供方

    Returns:
        list[tuple]: (提供方标识, 界面显示名称) 列表，顺序与配置表一致
    """
    return [(key, cfg["label"]) for key, cfg in MODEL_CONFIGS.items()]


def get_provider_config(provider_key):
    """获取指定提供方的配置信息

    Args:
        provider_key: 提供方标识（deepseek / openai / ollama）

    Returns:
        dict: 提供方配置；未知标识返回 None
    """
    return MODEL_CONFIGS.get(provider_key)


def find_provider_of_model(model_name):
    """按模型名反查所属提供方（无法识别时返回默认提供方）

    Args:
        model_name: 模型名称，如 deepseek-chat / gpt-4o / qwen2.5:7b

    Returns:
        str: 提供方标识；无法识别时返回 "deepseek"
    """
    for key, cfg in MODEL_CONFIGS.items():
        if model_name in cfg["models"]:
            return key
    return "deepseek"


def get_model_config(model_name, provider_key=None):
    """获取指定模型的完整调用配置（统一入口）

    解析规则：
    1. 未显式指定提供方时，按模型名自动匹配；Ollama 等动态模型建议显式传入
    2. API 密钥从环境变量读取（无密钥要求的提供方使用占位密钥）
    3. temperature / max_tokens 取该模型独立默认值，未单独配置的用 DEFAULT_PARAMS

    Args:
        model_name: 模型名称
        provider_key: 提供方标识；缺省时按模型名自动匹配

    Returns:
        dict: 包含 model / provider / api_key / base_url / temperature /
              max_tokens / supports_tools / timeout 字段

    Raises:
        ValueError: 提供方标识无效时抛出
    """
    if provider_key is None:
        provider_key = find_provider_of_model(model_name)
    cfg = MODEL_CONFIGS.get(provider_key)
    if cfg is None:
        raise ValueError(f"未知的提供方标识：{provider_key}")

    params = MODEL_PARAMS.get(model_name, dict(DEFAULT_PARAMS))
    env_key = cfg.get("api_key_env", "")
    # 无密钥要求的提供方（本地模型）使用占位密钥，保证 OpenAI SDK 不报错
    api_key = os.environ.get(env_key, "") if env_key else LOCAL_API_KEY_PLACEHOLDER
    return {
        "model": model_name,
        "provider": provider_key,
        "api_key": api_key,
        "base_url": cfg["base_url"],
        "temperature": params["temperature"],
        "max_tokens": params["max_tokens"],
        "supports_tools": params["supports_tools"],
        "timeout": API_TIMEOUT,
    }


def list_ollama_models(base_url=None, timeout=3):
    """动态获取 Ollama 本地已安装的模型列表

    通过 Ollama 的 OpenAI 兼容接口 GET /models 查询，与对话调用同一条链路。

    Args:
        base_url: Ollama 服务地址（默认取配置表中的 base_url）
        timeout: 请求超时秒数（本地服务短超时，保证服务不可用时界面不卡顿）

    Returns:
        tuple: (模型名列表, 错误信息) — 服务不可用时返回 ([], 错误说明)，绝不抛异常
    """
    base = (base_url or MODEL_CONFIGS["ollama"]["base_url"]).rstrip("/")
    try:
        with urllib.request.urlopen(f"{base}/models", timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = [m.get("name", "") for m in data.get("data", []) if m.get("name")]
        return sorted(models), ""
    except Exception as e:
        return [], f"无法连接 Ollama 服务：{str(e)}"


def extract_reasoning(chunk):
    """从流式响应片段中提取模型的思考内容（如 deepseek-reasoner）

    部分模型（DeepSeek reasoner）在输出最终回答前会先流式输出思考过程：
    优先取 SDK 已解析的显式字段，否则从透传的原始 JSON 字段
    （model_extra）中读取，均无则返回空串。

    Args:
        chunk: OpenAI SDK 的流式响应片段（ChatCompletionChunk）

    Returns:
        str: 本片段的思考内容增量（无则空字符串）
    """
    delta = chunk.choices[0].delta
    text = getattr(delta, "reasoning_content", None)
    if text is None:
        text = (getattr(delta, "model_extra", None) or {}).get("reasoning_content")
    return text or ""
