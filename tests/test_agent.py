"""
modules/agent 单元测试：LangGraph Agent 的规划 / 反思 / 自主结束流程

使用 FakeLLM 替换 ChatOpenAI（monkeypatch _build_llm），全部用例
不触发真实 API 调用；工具执行走真实的 execute_tool（本地计算）。
"""

import copy

from langchain_core.messages import AIMessage

from modules import agent as agent_module
from modules.agent import AGENT_SYSTEM_DIRECTIVE, run_agent
from modules.tools import get_available_tools


class FakeLLM:
    """模拟 ChatOpenAI：按预设序列返回 AIMessage（每次深拷贝，避免图执行中复用污染）"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.bound_tools = None
        self.invoked_messages = []

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    def invoke(self, messages):
        self.invoked_messages.append(list(messages))
        return copy.deepcopy(self.responses.pop(0))


def _fake_llm_factory(fake):
    return lambda cfg: fake


TEST_CONFIG = {
    "model": "deepseek-chat",
    "api_key": "test-key",
    "base_url": "https://api.deepseek.com",
    "temperature": 0.7,
    "max_tokens": 100,
}

BASE_MESSAGES = [
    {"role": "system", "content": "你是一个测试助手"},
    {"role": "user", "content": "帮我算一下 2+2"},
]


class TestAgentFlow:
    """Agent 主流程：规划 → 调用工具 → 反思 → 自主结束"""

    def test_plans_calls_tool_then_answers(self, monkeypatch):
        """模型先请求调用计算工具，拿到结果后给出最终回答"""
        fake = FakeLLM([
            AIMessage(content="", tool_calls=[{
                "name": "calculate", "args": {"expression": "2+2"}, "id": "call_1"}],
                additional_kwargs={"reasoning_content": "先算一下"}),
            AIMessage(content="2+2 的答案是 4"),
        ])
        monkeypatch.setattr(agent_module, "_build_llm", _fake_llm_factory(fake))

        messages = copy.deepcopy(BASE_MESSAGES)
        answer, tools_used, err, reasoning = run_agent(
            messages, tools=get_available_tools(), max_steps=3, config=TEST_CONFIG)

        assert err is None
        assert answer == "2+2 的答案是 4"
        assert tools_used == ["calculate"]
        assert reasoning == "先算一下"
        # bind_tools 携带了工具定义（复用 get_available_tools）
        assert fake.bound_tools and any(
            t["function"]["name"] == "calculate" for t in fake.bound_tools)
        # 系统提示词追加了 Agent 行为指令
        assert AGENT_SYSTEM_DIRECTIVE in fake.invoked_messages[0][0]["content"]
        # 不修改调用方的 messages（浅拷贝保护）
        assert messages[0]["content"] == "你是一个测试助手"

    def test_max_steps_returns_collected_info(self, monkeypatch):
        """模型反复请求工具直到步数上限：返回已收集的信息并提示"""
        tool_call_msg = AIMessage(content="", tool_calls=[{
            "name": "calculate", "args": {"expression": "2+2"}, "id": "call_1"}])
        fake = FakeLLM([tool_call_msg, tool_call_msg])  # 每轮都请求工具
        monkeypatch.setattr(agent_module, "_build_llm", _fake_llm_factory(fake))

        answer, tools_used, err, _ = run_agent(
            BASE_MESSAGES, tools=get_available_tools(), max_steps=2, config=TEST_CONFIG)

        assert err is None
        assert "达到最大规划步数" in answer
        assert "已收集的信息" in answer
        assert "calculate" in answer  # 已收集信息包含工具记录
        # max_steps 限制的是模型规划步数：第 2 次决策即触发上限，工具执行 1 轮
        assert tools_used == ["calculate"]

    def test_no_tools_direct_answer(self, monkeypatch):
        """无工具时 Agent 退化为单轮回答，不进入工具循环"""
        fake = FakeLLM([AIMessage(content="直接回答")])
        monkeypatch.setattr(agent_module, "_build_llm", _fake_llm_factory(fake))

        answer, tools_used, err, _ = run_agent(
            BASE_MESSAGES, tools=None, config=TEST_CONFIG)

        assert err is None
        assert answer == "直接回答"
        assert tools_used == []


class TestAgentErrors:
    """错误处理：模型异常 / 密钥缺失均返回友好错误，不崩溃"""

    def test_llm_exception_returns_error(self, monkeypatch):
        """模型调用抛异常：返回友好错误，而不是让异常击穿到界面"""

        class BoomLLM:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages):
                raise RuntimeError("模型服务炸了")

        monkeypatch.setattr(agent_module, "_build_llm", lambda cfg: BoomLLM())
        answer, tools_used, err, _ = run_agent(
            BASE_MESSAGES, tools=None, config=TEST_CONFIG)
        assert answer == "" and tools_used == []
        assert "Agent 运行失败" in err and "模型服务炸了" in err

    def test_missing_api_key_returns_friendly_error(self):
        """未配置密钥时直接返回提示，不发起任何模型调用"""
        cfg = dict(TEST_CONFIG, api_key="")
        answer, tools_used, err, _ = run_agent(
            BASE_MESSAGES, tools=None, config=cfg)
        assert answer == "" and tools_used == []
        assert "API密钥无效或未填写" in err
