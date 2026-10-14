"""MCP 客户端适配器：让 Agent 通过标准协议调用远端工具。

架构上的关键点（也是踩过的坑）：

MCP 的 stdio 传输是**单向**的——客户端拉起 Server 子进程，
通信依赖创建 ClientSession 的那个 asyncio 任务，且必须在该任务的
上下文管理器内完成所有调用。这意味着 MCP 工具**无法**和本地工具
简单混进同一个 asyncio.gather，否则会抛
「Attempted to exit cancel scope in a different task」。

因此适配器的形态是：在独立的 asyncio 任务里跑一个会话维护协程，
通过队列把请求送进去、把结果取出来。对 Agent 而言，
它只是另一个「工具来源」，接口与本地工具保持一致。
"""

import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.resilience import FailureKind, ToolOutcome

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class MCPToolInfo:
    name: str
    description: str
    input_schema: dict


class MCPClient:
    """维护一条常驻 MCP 会话，对外提供与本地工具一致的调用接口。"""

    def __init__(self, server_script: str = "app/mcp_server.py") -> None:
        self.server_script = server_script
        self.tools: dict[str, MCPToolInfo] = {}
        self._task: asyncio.Task | None = None
        self._requests: asyncio.Queue | None = None
        self._ready = asyncio.Event()
        self._error: BaseException | None = None

    # ---- 生命周期 ------------------------------------------------------

    async def start(self, timeout: float = 30.0) -> None:
        """拉起 Server 子进程并完成 MCP 握手。"""
        self._requests = asyncio.Queue()
        self._task = asyncio.create_task(self._session_loop())

        try:
            await asyncio.wait_for(self._ready.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(f"MCP Server 启动超时（{timeout}s）") from None

        if self._error is not None:
            raise RuntimeError(f"MCP Server 启动失败：{self._error}") from self._error

    async def stop(self) -> None:
        if self._requests is not None:
            await self._requests.put(None)  # 发送停止信号
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()

    async def __aenter__(self) -> "MCPClient":
        await self.start()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.stop()

    # ---- 会话维护（在独立任务里跑） ------------------------------------

    async def _session_loop(self) -> None:
        """整个会话都在这个任务内完成，保证 cancel scope 不跨任务。"""
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "app.mcp_server"],
            cwd=str(_PROJECT_ROOT),
        )

        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()

                    # 握手后拉取工具清单，缓存下来供 Agent 每轮使用
                    listed = await session.list_tools()
                    for t in listed.tools:
                        self.tools[t.name] = MCPToolInfo(
                            name=t.name,
                            description=t.description or "",
                            input_schema=t.input_schema or {"type": "object", "properties": {}},
                        )

                    self._ready.set()

                    while True:
                        item = await self._requests.get()
                        if item is None:  # 停止信号
                            break
                        name, arguments, future = item
                        try:
                            result = await session.call_tool(name, arguments)
                            future.set_result(result)
                        except BaseException as exc:  # noqa: BLE001
                            if not future.done():
                                future.set_exception(exc)
        except BaseException as exc:  # noqa: BLE001
            self._error = exc
            self._ready.set()
            # 唤醒所有等待中的请求，避免死等
            if self._requests is not None:
                while not self._requests.empty():
                    item = self._requests.get_nowait()
                    if item is not None and not item[2].done():
                        item[2].set_exception(exc)

    # ---- 对外接口（与 ToolRegistry 保持一致） --------------------------

    def all_schemas(self) -> list[dict]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self.tools.values()
        ]

    def names(self) -> list[str]:
        return list(self.tools)

    async def execute(self, name: str, arguments: dict) -> ToolOutcome:
        if name not in self.tools:
            return ToolOutcome(
                text=f"错误：MCP 工具 `{name}` 不存在。可用工具：{', '.join(self.names())}",
                attempts=0,
                succeeded=False,
                kind=FailureKind.PERMANENT,
            )

        if self._requests is None or self._task is None or self._task.done():
            return ToolOutcome(
                text="错误：MCP 会话不可用，请检查 MCP Server 是否正常运行。",
                attempts=1,
                succeeded=False,
                kind=FailureKind.TRANSIENT,
            )

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        await self._requests.put((name, arguments, future))

        try:
            result = await future
        except BaseException as exc:  # noqa: BLE001
            return ToolOutcome(
                text=f"错误：MCP 调用失败（{type(exc).__name__}: {exc}）",
                attempts=1,
                succeeded=False,
                kind=FailureKind.TRANSIENT,
            )

        return ToolOutcome(
            text=_render_result(result),
            attempts=1,
            succeeded=_is_success(result),
        )


def _is_success(result: Any) -> bool:
    """判断这次 MCP 调用是否真的成功。

    注意：不能只看协议层的 is_error。我们在 Server 侧把异常转成了文本返回，
    以免 MCP 用「Error executing tool xxx」把具体原因吞掉——
    代价是这类失败在协议层表现为「成功」。因此还要看内容是不是我们
    自己生成的错误文案，否则失败率统计会全是 100%。
    """
    if getattr(result, "is_error", False):
        return False
    text = _render_result(result)
    return not text.startswith("错误：")


def _render_result(result: Any) -> str:
    """把 MCP 的 CallToolResult 转成模型可读的纯文本。"""
    if getattr(result, "is_error", False):
        parts = [_block_text(b) for b in getattr(result, "content", [])]
        return "错误：" + " ".join(p for p in parts if p)

    # 优先用结构化输出，它比文本块更稳定
    structured = getattr(result, "structured_content", None)
    if structured:
        # 单键且看起来是返回值时直接取，避免给模型塞一层无意义的 JSON 包装
        if isinstance(structured, dict) and len(structured) == 1:
            only = next(iter(structured.values()))
            if isinstance(only, str):
                return only
        return json.dumps(structured, ensure_ascii=False)

    parts = [_block_text(b) for b in getattr(result, "content", [])]
    text = "\n".join(p for p in parts if p)
    return text or "(工具返回了空结果)"


def _block_text(block: Any) -> str:
    if getattr(block, "type", None) == "text":
        return getattr(block, "text", "") or ""
    return ""
