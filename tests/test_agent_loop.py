"""Agent 循环的行为测试。

用 MockLLM 替换真实模型，脚本化地喂入「模型该返回什么」，
从而在没有 API Key 的情况下验证循环本身：
  - 无工具调用时直接收敛
  - 单工具调用 -> 回灌 -> 二次决策
  - 一轮内多个工具并发执行
  - 幻觉工具名 / 非法参数能被回灌并让模型自我纠正
  - 达到轮次上限时能安全退出

直接运行：python tests/test_agent_loop.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.tools.builtin  # noqa: F401  触发工具注册
from app.agent import Agent
from app.llm import LLMClient, LLMResponse, ToolCall
from app.registry import registry


class MockLLM(LLMClient):
    """按脚本依次返回预设响应，并把每次收到的 messages 存下来供断言。"""

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.seen_messages: list[list[dict]] = []

    async def create_message(self, system, messages, tools=None):
        self.seen_messages.append(messages)
        if not self.script:
            # 脚本用尽仍然被调用，说明轮次控制有问题
            raise AssertionError("Mock 脚本已耗尽，但模型仍在被调用")
        return self.script.pop(0)


def _answer(text: str) -> LLMResponse:
    return LLMResponse(
        text=text,
        tool_calls=[],
        raw_content=[{"type": "text", "text": text}],
        input_tokens=10,
        output_tokens=5,
    )


def _call_tools(*calls: ToolCall) -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=list(calls),
        raw_content=[
            {
                "type": "tool_use",
                "id": c.id,
                "name": c.name,
                "input": c.arguments,
            }
            for c in calls
        ],
        input_tokens=20,
        output_tokens=8,
    )


async def test_direct_answer():
    llm = MockLLM([_answer("1 加 1 等于 2")])
    agent = Agent(llm, registry, verbose=False)
    result = await agent.run("1+1 等于几")

    assert result.rounds == 1, f"应 1 轮收敛，实际 {result.rounds}"
    assert result.answer == "1 加 1 等于 2"
    assert result.tool_calls == [], "不该有工具调用"
    print("  [PASS] 无工具调用时一轮收敛")


async def test_single_tool_then_answer():
    llm = MockLLM(
        [
            _call_tools(ToolCall("c1", "calculator", {"expression": "(12+8)*3/5"})),
            _answer("结果是 12"),
        ]
    )
    agent = Agent(llm, registry, verbose=False)
    result = await agent.run("帮我算 (12+8)*3/5")

    assert result.rounds == 2, f"应 2 轮，实际 {result.rounds}"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].tool_name == "calculator"
    assert "12" in result.tool_calls[0].result
    # 第二轮请求里必须带上第一轮的工具结果
    second_round_msgs = llm.seen_messages[1]
    tool_result_blocks = [
        b
        for m in second_round_msgs
        if isinstance(m["content"], list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert len(tool_result_blocks) == 1, "工具结果未正确回灌"
    assert tool_result_blocks[0]["tool_use_id"] == "c1"
    print("  [PASS] 单工具调用 -> 结果回灌 -> 二次决策")


async def test_parallel_tool_calls():
    llm = MockLLM(
        [
            _call_tools(
                ToolCall("c1", "query_sales", {"city": "成都", "top_n": 2}),
                ToolCall("c2", "get_current_time", {}),
                ToolCall("c3", "calculator", {"expression": "100*1.13"}),
            ),
            _answer("三家数据汇总完毕"),
        ]
    )
    agent = Agent(llm, registry, verbose=False)
    result = await agent.run("成都销量前2、现在几点、100*1.13")

    assert len(result.tool_calls) == 3, f"应执行 3 个工具，实际 {len(result.tool_calls)}"
    assert result.rounds == 2, "三个工具应在同一轮内并发完成"
    # 回灌顺序必须与模型请求顺序一致
    second_round_msgs = llm.seen_messages[1]
    ids = [
        b["tool_use_id"]
        for m in second_round_msgs
        if isinstance(m["content"], list)
        for b in m["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert ids == ["c1", "c2", "c3"], f"回灌顺序错乱: {ids}"
    print("  [PASS] 一轮内多工具并发执行且回灌顺序正确")


async def test_hallucinated_tool_recovery():
    """模型调用不存在的工具，错误应回灌给模型而不是中断循环。"""
    llm = MockLLM(
        [
            _call_tools(ToolCall("c1", "query_weather", {"city": "成都"})),
            _call_tools(ToolCall("c2", "query_sales", {"city": "成都", "top_n": 2})),
            _answer("成都销量前两名是春熙路店和天府三街店"),
        ]
    )
    agent = Agent(llm, registry, verbose=False)
    result = await agent.run("成都卖得最好的两家店")

    assert result.rounds == 3, f"应 3 轮（幻觉->纠正->作答），实际 {result.rounds}"
    assert "不存在" in result.tool_calls[0].result, "幻觉工具名应返回友好错误"
    assert result.tool_calls[1].tool_name == "query_sales"
    print("  [PASS] 幻觉工具名被回灌后模型自我纠正")


async def test_max_rounds_guard():
    """模型陷入无限调用时，必须在轮次上限处安全退出。"""
    looping = [
        _call_tools(ToolCall(f"c{i}", "get_current_time", {})) for i in range(10)
    ]
    llm = MockLLM(looping)
    agent = Agent(llm, registry, max_rounds=3, verbose=False)
    result = await agent.run("现在几点")

    assert result.rounds == 3, f"应在 3 轮截断，实际 {result.rounds}"
    assert len(result.tool_calls) == 3
    assert "最大工具调用轮次" in result.answer
    print("  [PASS] 无限调用被轮次上限安全截断")


async def main() -> int:
    tests = [
        test_direct_answer,
        test_single_tool_then_answer,
        test_parallel_tool_calls,
        test_hallucinated_tool_recovery,
        test_max_rounds_guard,
    ]
    print("Agent 循环测试：")
    for t in tests:
        await t()
    print(f"\n全部 {len(tests)} 项通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
