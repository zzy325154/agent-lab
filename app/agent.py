"""Agent 核心循环：规划 -> 调用工具 -> 观察结果 -> 再决策。

不用任何 Agent 框架，暴露每一步的细节。手写一遍之后，
再去看 LangChain / LangGraph 的源码或 MCP 协议，会清楚每一层在解决什么问题。
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from app.context import ContextManager, ContextStats
from app.llm import LLMClient
from app.registry import ToolRegistry
from app.resilience import FailureKind, ToolOutcome

if TYPE_CHECKING:  # 避免循环导入，仅用于类型标注
    from app.mcp_client import MCPClient

SYSTEM_PROMPT = """你是一个可以调用外部工具来完成任务的助手。

工作方式：
1. 先判断问题是否需要工具。需要就直接调用，不要凭空猜测数据。
2. 一次可以调用多个互不依赖的工具。
3. 拿到工具结果后，判断信息是否足够回答。不够就继续调用。
4. 信息足够后，用简洁自然的中文给出最终答案。

注意：工具报错时，读懂错误信息并调整参数重试，不要重复同样的无效调用。"""


@dataclass
class ToolCallRecord:
    """一次工具调用的完整记录，用于日志追溯和成功率统计。"""

    round_index: int
    tool_name: str
    arguments: dict
    result: str
    elapsed_ms: float
    attempts: int = 1
    succeeded: bool = True


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    rounds: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    context_stats: list[ContextStats] = field(default_factory=list)

    @property
    def tool_time_ms(self) -> float:
        """所有工具调用的耗时之和。并发执行时它会大于墙钟时间，这是正常的。"""
        return sum(r.elapsed_ms for r in self.tool_calls)

    @property
    def saved_tokens(self) -> int:
        return sum(s.saved_tokens for s in self.context_stats)

    @property
    def saved_pct(self) -> float:
        """整轮对话中，按最后一次裁剪的统计口径衡量节省比例。"""
        if not self.context_stats:
            return 0.0
        last = self.context_stats[-1]
        return last.saved_pct

    @property
    def tool_success_rate(self) -> float:
        if not self.tool_calls:
            return 100.0
        ok = sum(1 for r in self.tool_calls if r.succeeded)
        return ok / len(self.tool_calls) * 100

    @property
    def total_retries(self) -> int:
        """多出来的尝试次数总和。"""
        return sum(max(0, r.attempts - 1) for r in self.tool_calls)


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        registry: ToolRegistry | None = None,
        mcp: "MCPClient | None" = None,
        max_rounds: int = 8,
        verbose: bool = True,
        context: ContextManager | None = None,
    ) -> None:
        if registry is None and mcp is None:
            raise ValueError("至少需要一种工具来源（registry 或 mcp）")
        self.llm = llm
        self.registry = registry
        self.mcp = mcp
        self.max_rounds = max_rounds
        self.verbose = verbose
        self.context = context or ContextManager()

    def _all_schemas(self) -> list[dict] | None:
        """合并两种来源的工具定义，喂给模型。

        对模型而言这些都是「可调用的工具」，它不需要知道
        哪个在本地、哪个在 MCP Server 进程里。
        """
        schemas: list[dict] = []
        if self.registry is not None:
            schemas.extend(self.registry.all_schemas())
        if self.mcp is not None:
            schemas.extend(self.mcp.all_schemas())
        return schemas or None

    async def _execute(self, name: str, arguments: dict):
        """按工具来源分发执行。

        MCP 调用虽然走独立会话，但接口同样是 awaitable，
        因此仍可与本地工具一起放进 asyncio.gather。
        区别在于 MCP 侧受单一会话串行化，并发收益有限——
        这是 stdio 单向传输的固有约束，远程 HTTP 传输没有这个问题。
        """
        if self.registry is not None and self.registry.get(name) is not None:
            return await self.registry.execute_tool(self.registry.get(name), arguments)
        if self.mcp is not None and name in self.mcp.tools:
            return await self.mcp.execute(name, arguments)

        # 两边都没有：幻觉工具名
        available = []
        if self.registry is not None:
            available.extend(self.registry.names())
        if self.mcp is not None:
            available.extend(self.mcp.names())
        return ToolOutcome(
            text=f"错误：工具 `{name}` 不存在。可用工具：{', '.join(available)}",
            attempts=0,
            succeeded=False,
            kind=FailureKind.PERMANENT,
        )

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    async def run(self, user_input: str, history: list[dict] | None = None) -> AgentResult:
        messages = list(history or [])
        messages.append({"role": "user", "content": user_input})

        records: list[ToolCallRecord] = []
        context_stats: list[ContextStats] = []
        total_in = total_out = 0
        tools_schema = self._all_schemas()

        for round_index in range(1, self.max_rounds + 1):
            # 每次请求前都过一遍上下文管理：工具结果会持续累积，
            # 只在开头裁一次是不够的。
            system, payload, stats = self.context.build(SYSTEM_PROMPT, messages, tools_schema)
            context_stats.append(stats)
            if stats.saved_tokens > 0:
                self._log(
                    f"--- 第 {round_index} 轮：上下文压缩，"
                    f"{stats.original_tokens} -> {stats.final_tokens} tokens "
                    f"（省 {stats.saved_pct:.0f}%，压缩工具结果 {stats.compressed_results} 处，"
                    f"丢弃历史 {stats.dropped_turns} 轮）---"
                )
            else:
                self._log(f"\n--- 第 {round_index} 轮：请求模型 ---")

            response = await self.llm.create_message(
                system=system,
                messages=payload,
                tools=tools_schema,
            )
            total_in += response.input_tokens
            total_out += response.output_tokens

            messages.append({"role": "assistant", "content": response.raw_content})

            # 模型没有请求工具，说明它准备好给最终答案了
            if not response.tool_calls:
                self._log(f"模型给出最终答案（共 {round_index} 轮）")
                return AgentResult(
                    answer=response.text,
                    tool_calls=records,
                    rounds=round_index,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    context_stats=context_stats,
                )

            self._log(f"模型请求调用 {len(response.tool_calls)} 个工具")

            # 同一轮内多个工具互不依赖，并发执行
            new_records = await asyncio.gather(
                *(
                    self._run_one_tool_call(call, round_index)
                    for call in response.tool_calls
                )
            )
            records.extend(new_records)

            # 工具结果必须按原顺序回灌，模型依赖 tool_use_id 对应
            tool_results = []
            for call, rec in zip(response.tool_calls, new_records):
                self._log(f"  <- {call.name} 返回 {_preview(rec.result)}")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": call.id,
                        "content": rec.result,
                    }
                )

            messages.append({"role": "user", "content": tool_results})

        # 达到轮次上限仍未收敛，强行收尾但如实说明
        self._log(f"达到最大轮次 {self.max_rounds}，强制结束")
        return AgentResult(
            answer="已达到最大工具调用轮次，未能完成任务。请把问题拆得更具体一些。",
            tool_calls=records,
            rounds=self.max_rounds,
            input_tokens=total_in,
            output_tokens=total_out,
            context_stats=context_stats,
        )

    async def _run_one_tool_call(self, call, round_index: int) -> ToolCallRecord:
        self._log(f"  -> 调用 {call.name}({json.dumps(call.arguments, ensure_ascii=False)})")
        started = asyncio.get_event_loop().time()

        outcome = await self._execute(call.name, call.arguments)
        if outcome.attempts > 1:
            self._log(
                f"     （经历 {outcome.attempts} 次尝试，"
                f"{'成功' if outcome.succeeded else '最终失败'}）"
            )

        elapsed_ms = (asyncio.get_event_loop().time() - started) * 1000
        return ToolCallRecord(
            round_index=round_index,
            tool_name=call.name,
            arguments=call.arguments,
            result=outcome.text,
            elapsed_ms=elapsed_ms,
            attempts=outcome.attempts,
            succeeded=outcome.succeeded,
        )

    # ------------------------------------------------------------------
    # 流式执行：在 run() 基础上把过程事件暴露出来
    # ------------------------------------------------------------------

    async def stream(
        self,
        user_input: str,
        history: list[dict] | None = None,
    ):
        """流式执行，产出过程事件。

        事件类型：
            ("round_start", {"round": n})
            ("text_delta", {"text": "..."})       模型正在输出的正文增量
            ("tool_start",  {"name": ..., "arguments": {...}})
            ("tool_end",    {"name": ..., "succeeded": bool, "attempts": n,
                             "elapsed_ms": f, "preview": "..."})
            ("round_end",   {"round": n, "tool_calls": [names]})
            ("done",        AgentResult)
            ("error",       {"message": "..."})

        与非流式 run() 的差别只在「何时把结果交出去」，
        循环逻辑、上下文管理、韧性策略完全复用同一套代码。
        """
        messages = list(history or [])
        messages.append({"role": "user", "content": user_input})

        records: list[ToolCallRecord] = []
        context_stats: list[ContextStats] = []
        total_in = total_out = 0
        tools_schema = self._all_schemas()

        try:
            for round_index in range(1, self.max_rounds + 1):
                yield ("round_start", {"round": round_index})

                system, payload, stats = self.context.build(
                    SYSTEM_PROMPT, messages, tools_schema
                )
                context_stats.append(stats)
                if stats.saved_tokens > 0:
                    yield (
                        "context_compressed",
                        {
                            "original": stats.original_tokens,
                            "final": stats.final_tokens,
                            "saved_pct": round(stats.saved_pct, 1),
                            "compressed_results": stats.compressed_results,
                            "dropped_turns": stats.dropped_turns,
                        },
                    )

                response = None
                async for kind, payload_evt in self.llm.stream_message(
                    system=system, messages=payload, tools=tools_schema
                ):
                    if kind == "text_delta":
                        yield ("text_delta", {"text": payload_evt})
                    elif kind == "final":
                        response = payload_evt

                if response is None:
                    yield ("error", {"message": "模型未返回结果"})
                    return

                total_in += response.input_tokens
                total_out += response.output_tokens
                messages.append({"role": "assistant", "content": response.raw_content})

                if not response.tool_calls:
                    yield ("round_end", {"round": round_index, "tool_calls": []})
                    yield (
                        "done",
                        AgentResult(
                            answer=response.text,
                            tool_calls=records,
                            rounds=round_index,
                            input_tokens=total_in,
                            output_tokens=total_out,
                            context_stats=context_stats,
                        ),
                    )
                    return

                for call in response.tool_calls:
                    yield (
                        "tool_start",
                        {"name": call.name, "arguments": call.arguments},
                    )

                # 同一轮内并发执行，但结果按原顺序回灌
                new_records = await asyncio.gather(
                    *(
                        self._run_one_tool_call(call, round_index)
                        for call in response.tool_calls
                    )
                )
                records.extend(new_records)

                tool_results = []
                for call, rec in zip(response.tool_calls, new_records):
                    yield (
                        "tool_end",
                        {
                            "name": call.name,
                            "succeeded": rec.succeeded,
                            "attempts": rec.attempts,
                            "elapsed_ms": round(rec.elapsed_ms, 1),
                            "preview": _preview(rec.result, 80),
                        },
                    )
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": call.id,
                            "content": rec.result,
                        }
                    )

                messages.append({"role": "user", "content": tool_results})
                yield (
                    "round_end",
                    {
                        "round": round_index,
                        "tool_calls": [c.name for c in response.tool_calls],
                    },
                )

            yield (
                "done",
                AgentResult(
                    answer="已达到最大工具调用轮次，未能完成任务。请把问题拆得更具体一些。",
                    tool_calls=records,
                    rounds=self.max_rounds,
                    input_tokens=total_in,
                    output_tokens=total_out,
                    context_stats=context_stats,
                ),
            )
        except Exception as e:  # noqa: BLE001 — 流式接口必须把异常转成事件
            yield ("error", {"message": f"{type(e).__name__}: {e}"})


def _preview(text: str, limit: int = 60) -> str:
    flat = text.replace("\n", " ")
    return flat if len(flat) <= limit else flat[:limit] + "..."
