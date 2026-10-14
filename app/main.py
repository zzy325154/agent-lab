"""命令行入口：跑一个交互式 Agent 会话。

用法：
    python -m app.main                       # 本地工具，交互模式
    python -m app.main "成都销量前3"          # 本地工具，单次执行
    python -m app.main --mcp                 # 工具改由 MCP Server 提供
    python -m app.main --mcp "现在几点"       # MCP 模式单次执行
    python -m app.main --compare "成都销量前3" # 同一问题跑两种模式对比
"""

import asyncio
import sys

from app.agent import Agent
from app.llm import build_client
from app.mcp_client import MCPClient
from app.registry import registry

# 导入即注册：工具写在 core.py、由 builtin.py 打上装饰器
import app.tools.builtin  # noqa: F401


async def run_once(query: str, use_mcp: bool, verbose: bool = True) -> None:
    llm = build_client()

    if use_mcp:
        async with MCPClient() as mcp:
            agent = Agent(llm=llm, registry=None, mcp=mcp, verbose=verbose)
            print(f"工具来源: MCP 协议（Server 子进程）")
            print(f"可用工具: {', '.join(mcp.names())}")
            result = await agent.run(query)
    else:
        agent = Agent(llm=llm, registry=registry, verbose=verbose)
        print(f"工具来源: 本地进程内")
        print(f"可用工具: {', '.join(registry.names())}")
        result = await agent.run(query)

    print("\n" + "=" * 50)
    print("最终回答：", result.answer)
    print(f"轮次 {result.rounds}｜工具调用 {len(result.tool_calls)} 次"
          f"｜成功率 {result.tool_success_rate:.0f}%"
          f"｜tokens {result.input_tokens}+{result.output_tokens}")


async def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    query = " ".join(args).strip()

    if "--compare" in flags:
        if not query:
            print("--compare 需要带上问题，例如：python -m app.main --compare \"成都销量前3\"")
            return
        print("#" * 55)
        print("# 本地工具")
        print("#" * 55)
        await run_once(query, use_mcp=False)
        print()
        print("#" * 55)
        print("# MCP 工具")
        print("#" * 55)
        await run_once(query, use_mcp=True)
        return

    if query:
        await run_once(query, use_mcp="--mcp" in flags)
        return

    # 交互模式
    use_mcp = "--mcp" in flags
    llm = build_client()

    if use_mcp:
        async with MCPClient() as mcp:
            agent = Agent(llm=llm, registry=None, mcp=mcp)
            print(f"工具来源: MCP｜已注册: {', '.join(mcp.names())}")
            await _repl(agent)
    else:
        agent = Agent(llm=llm, registry=registry)
        print(f"工具来源: 本地｜已注册: {', '.join(registry.names())}")
        await _repl(agent)


async def _repl(agent: Agent) -> None:
    print("\n输入问题开始对话，输入 exit 退出。\n")
    history: list[dict] = []
    while True:
        try:
            user_input = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        result = await agent.run(user_input, history=history)
        print(f"\nAgent > {result.answer}\n")

        history.append({"role": "user", "content": user_input})
        history.append(
            {
                "role": "assistant",
                "content": [{"type": "text", "text": result.answer}],
            }
        )


if __name__ == "__main__":
    asyncio.run(main())
