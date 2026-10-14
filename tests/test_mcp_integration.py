"""MCP 接入测试：验证 Agent 能通过协议调用远端工具。

这些测试真的会拉起 MCP Server 子进程走一遍完整握手，
不是打桩——协议层的坑（握手时序、会话生命周期、错误跨越边界）
只有跑通真实链路才能暴露。

运行：python tests/test_mcp_integration.py
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import Agent
from app.llm import LLMClient, LLMResponse, ToolCall
from app.mcp_client import MCPClient, _render_result
from app.registry import ToolRegistry


class ScriptedLLM(LLMClient):
    """按脚本返回预设响应，并把收到的工具清单存下来供断言。"""

    def __init__(self, script: list[LLMResponse]) -> None:
        self.script = list(script)
        self.seen_tools: list[list[dict] | None] = []

    async def create_message(self, system, messages, tools=None):
        self.seen_tools.append(tools)
        if not self.script:
            raise AssertionError("脚本耗尽但模型仍被调用")
        return self.script.pop(0)


def _answer(text: str) -> LLMResponse:
    return LLMResponse(text=text, tool_calls=[], raw_content=[{"type": "text", "text": text}])


def _call(name: str, args: dict, call_id: str = "c1") -> LLMResponse:
    return LLMResponse(
        text="",
        tool_calls=[ToolCall(call_id, name, args)],
        raw_content=[
            {"type": "tool_use", "id": call_id, "name": name, "input": args}
        ],
    )


async def test_handshake_and_tool_listing():
    """握手后应拿到完整的工具清单与正确的 Schema。"""
    async with MCPClient() as c:
        names = c.names()
        assert "calculator" in names, f"缺少 calculator: {names}"
        assert "search_policy" in names, f"缺少 search_policy: {names}"
        assert len(names) == 4, f"应有 4 个工具，实际 {len(names)}"

        schemas = {s["name"]: s for s in c.all_schemas()}
        calc = schemas["calculator"]["input_schema"]
        assert calc["required"] == ["expression"], f"必填项错误: {calc}"
        assert "expression" in calc["properties"]

        # 选填参数不能被标成必填
        sales = schemas["query_sales"]["input_schema"]
        assert sales["required"] == ["city"], f"必填项错误: {sales}"
        assert "top_n" in sales["properties"]

        # docstring 描述应被保留
        assert "数学表达式" in schemas["calculator"]["description"]
    print(f"  [PASS] MCP 握手与工具清单正确（{len(names)} 个工具）")


async def test_call_tools_over_protocol():
    """通过协议真实调用各工具。"""
    async with MCPClient() as c:
        cases = [
            ("calculator", {"expression": "(12+8)*3/5"}, "12"),
            ("calculator", {"expression": "100*1.13"}, "113"),
            ("query_sales", {"city": "成都", "top_n": 2}, "春熙路店"),
            ("get_current_time", {}, "当前时间"),
            ("search_policy", {"keyword": "年假"}, "年休假"),
        ]
        for name, args, expect in cases:
            o = await c.execute(name, args)
            assert o.succeeded, f"{name} 调用失败: {o.text}"
            assert expect in o.text, f"{name} 返回不含 {expect}: {o.text[:80]}"
    print(f"  [PASS] {len(cases)} 次协议调用均成功且内容正确")


async def test_error_text_survives_boundary():
    """异常信息必须跨越 MCP 边界保留，否则模型无法自我纠正。"""
    async with MCPClient() as c:
        o = await c.execute("calculator", {"expression": "1/0"})
        assert "ZeroDivisionError" in o.text, f"异常类型丢失: {o.text}"
        assert "Error executing tool" not in o.text, "仍是 MCP 的通用错误"

        o2 = await c.execute("calculator", {"expression": "undefined_var"})
        assert "ValueError" in o2.text, f"异常类型丢失: {o2.text}"
    print("  [PASS] 异常详情跨越 MCP 边界保留")


async def test_injection_blocked_and_reported():
    """代码注入必须被拦截，且给出可理解的错误。"""
    async with MCPClient() as c:
        o = await c.execute(
            "calculator", {"expression": '__import__("os").system("echo pwned")'}
        )
        assert not o.succeeded, "注入不该返回成功"
        assert "pwned" not in o.text, f"注入被执行了: {o.text}"
        assert "不支持的语法" in o.text, f"错误信息对模型无用: {o.text}"
    print("  [PASS] 注入被拦截且错误信息可读")


async def test_hallucinated_mcp_tool():
    """调用不存在的 MCP 工具应有友好提示。"""
    async with MCPClient() as c:
        o = await c.execute("fly_to_moon", {})
        assert not o.succeeded
        assert "不存在" in o.text and "calculator" in o.text
    print("  [PASS] 幻觉工具名返回可用工具清单")


async def test_agent_uses_mcp_tools():
    """端到端：Agent 通过 MCP 调用工具并给出最终答案。"""
    llm = ScriptedLLM(
        [
            _call("query_sales", {"city": "成都", "top_n": 2}),
            _answer("成都前两名是春熙路店和天府三街店"),
        ]
    )
    async with MCPClient() as mcp:
        agent = Agent(llm=llm, registry=None, mcp=mcp, verbose=False)
        result = await agent.run("成都销量前两名")

    assert result.rounds == 2, f"应 2 轮，实际 {result.rounds}"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].succeeded, "MCP 工具调用应成功"
    assert "春熙路店" in result.tool_calls[0].result
    assert result.tool_success_rate == 100.0

    # 模型看到的工具清单应包含 MCP 工具
    seen = llm.seen_tools[0]
    assert seen is not None and any(t["name"] == "query_sales" for t in seen)
    print("  [PASS] Agent 通过 MCP 完成完整 2 轮闭环")


async def test_agent_merges_both_sources():
    """本地工具与 MCP 工具应能同时提供给模型。"""
    reg = ToolRegistry(timeout=2.0)
    from app.tools import core

    reg.register(core.get_current_time)  # 本地只放一个

    llm = ScriptedLLM(
        [
            _call("calculator", {"expression": "6*7"}, "c1"),
            _answer("42"),
        ]
    )
    async with MCPClient() as mcp:
        agent = Agent(llm=llm, registry=reg, mcp=mcp, verbose=False)

        # 先断言模型能同时看到两种来源的工具。
        # 注意 get_current_time 两边都有，因此清单 5 条、唯一名 4 个。
        schemas = agent._all_schemas() or []
        all_names = [s["name"] for s in schemas]
        unique = set(all_names)
        assert len(all_names) == 5, f"应为 1 本地 + 4 MCP 共 5 条，实际 {all_names}"
        assert unique == {"calculator", "get_current_time", "query_sales", "search_policy"}
        assert all_names.count("get_current_time") == 2, "本地与 MCP 应各贡献一条"

        result = await agent.run("算 6*7")
        assert result.tool_calls[0].succeeded, result.tool_calls[0].result
        assert "42" in result.tool_calls[0].result

        # 模型实际收到的清单也应包含两者
        seen = [t["name"] for t in llm.seen_tools[0] or []]
        assert "get_current_time" in seen and "calculator" in seen
    print("  [PASS] 本地与 MCP 两种来源的工具可共存")


async def test_mcp_tool_lookup_prefers_local():
    """同名工具时本地优先，避免歧义。"""
    reg = ToolRegistry(timeout=2.0)

    def calculator(expression: str) -> str:
        """本地版本的计算器。

        Args:
            expression: 表达式
        """
        return "本地实现被调用"

    reg.register(calculator)

    async with MCPClient() as mcp:
        agent = Agent(llm=ScriptedLLM([]), registry=reg, mcp=mcp, verbose=False)
        outcome = await agent._execute("calculator", {"expression": "1+1"})
        assert "本地实现被调用" in outcome.text, f"未优先用本地: {outcome.text}"
    print("  [PASS] 同名工具本地优先")


async def test_render_result_handles_empty():
    """空结果不应让 Agent 拿到空字符串而困惑。"""

    class FakeResult:
        is_error = False
        structured_content = None
        content = []

    assert _render_result(FakeResult()) == "(工具返回了空结果)"
    print("  [PASS] 空结果有占位文案")


async def main() -> int:
    tests = [
        test_render_result_handles_empty,
        test_handshake_and_tool_listing,
        test_call_tools_over_protocol,
        test_error_text_survives_boundary,
        test_injection_blocked_and_reported,
        test_hallucinated_mcp_tool,
        test_agent_uses_mcp_tools,
        test_agent_merges_both_sources,
        test_mcp_tool_lookup_prefers_local,
    ]
    print("MCP 接入测试（会真实拉起 Server 子进程）：")
    for t in tests:
        result = t()
        if asyncio.iscoroutine(result):
            await result
    print(f"\n全部 {len(tests)} 项通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
