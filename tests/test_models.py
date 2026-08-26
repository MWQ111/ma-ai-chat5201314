"""modules/models 单元测试：多模型配置与思考过程提取（纯函数，不依赖网络）"""

import os
from types import SimpleNamespace

import pytest

from modules.models import (
    extract_reasoning,
    find_provider_of_model,
    get_model_config,
    get_provider_config,
    list_ollama_models,
    list_providers,
)


def test_list_providers_order():
    providers = list_providers()
    assert [k for k, _ in providers] == ["deepseek", "openai", "ollama"]


def test_get_provider_config():
    assert get_provider_config("deepseek")["default_model"] == "deepseek-chat"
    assert get_provider_config("unknown") is None


def test_find_provider_of_model():
    assert find_provider_of_model("deepseek-chat") == "deepseek"
    assert find_provider_of_model("gpt-4o-mini") == "openai"
    # Ollama 模型列表为空（动态获取本地模型），反查不到时兜底返回 deepseek
    assert find_provider_of_model("qwen2.5:7b") == "deepseek"
    assert find_provider_of_model("不存在的模型") == "deepseek"  # 兜底默认


def test_get_model_config_deepseek():
    cfg = get_model_config("deepseek-chat", provider_key="deepseek")
    assert cfg["base_url"] == "https://api.deepseek.com"
    assert cfg["supports_tools"] is True
    # 密钥只从环境变量读取，绝不硬编码
    assert cfg["api_key"] == os.environ.get("DEEPSEEK_API_KEY", "")


def test_get_model_config_reasoner_supports_tools():
    """deepseek-reasoner 已实测支持工具调用（2026-08 通过真实 API 验证）"""
    cfg = get_model_config("deepseek-reasoner", provider_key="deepseek")
    assert cfg["supports_tools"] is True


def test_get_model_config_ollama_placeholder_key():
    """本地模型无密钥，使用占位密钥保证 OpenAI SDK 不报错"""
    cfg = get_model_config("qwen2.5:7b", provider_key="ollama")
    assert cfg["api_key"] == "ollama"


def test_get_model_config_unknown_provider_raises():
    with pytest.raises(ValueError):
        get_model_config("deepseek-chat", provider_key="not-a-provider")


def test_extract_reasoning_explicit_field():
    """SDK 已解析的显式字段优先"""
    delta = SimpleNamespace(reasoning_content="在思考", model_extra={})
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=delta)])
    assert extract_reasoning(chunk) == "在思考"


def test_extract_reasoning_model_extra():
    """透传字段兜底（reasoning_content 在 model_extra 中）"""
    delta = SimpleNamespace(
        reasoning_content=None, model_extra={"reasoning_content": "透传思考"}
    )
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=delta)])
    assert extract_reasoning(chunk) == "透传思考"


def test_extract_reasoning_none():
    """普通模型无思考内容时返回空串"""
    delta = SimpleNamespace(reasoning_content=None, model_extra={})
    chunk = SimpleNamespace(choices=[SimpleNamespace(delta=delta)])
    assert extract_reasoning(chunk) == ""


def test_list_ollama_models_unreachable():
    """服务不可用时不抛异常，返回 (空列表, 错误说明)"""
    models, err = list_ollama_models("http://127.0.0.1:1", timeout=1)
    assert models == []
    assert "无法连接" in err
