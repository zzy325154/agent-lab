"""工具注册中心。

设计目标：新增一个工具 = 写一个带类型注解和 docstring 的普通函数 + 打一个装饰器。
Agent 核心循环不感知任何具体工具，这就是简历里说的「工具热插拔」。
"""

import asyncio
import inspect
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.resilience import (
    CircuitBreaker,
    RetryPolicy,
    ToolOutcome,
    execute_with_resilience,
)
from app.schema import build_parameters_schema


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable
    parameters: dict
    is_async: bool = field(default=False)
    # 单工具的超时覆盖。RAG 检索首次调用要构建索引（实测约 47s），
    # 用全局的 10s 会直接把冷启动判成失败。
    timeout: float | None = None

    def to_api_schema(self) -> dict:
        """转换成模型 API 需要的 function 描述格式。"""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


class ToolRegistry:
    def __init__(
        self,
        timeout: float = 10.0,
        policy: RetryPolicy | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self.timeout = timeout
        self.policy = policy or RetryPolicy()
        self.breaker = breaker or CircuitBreaker()
        # 允许测试注入时钟
        self.clock: Callable[[], float] = time.monotonic

    def register(self, fn: Callable, name: str | None = None, timeout: float | None = None) -> None:
        tool_name = name or fn.__name__
        if tool_name in self._tools:
            raise ValueError(f"工具名重复: {tool_name}")

        # 工具说明取 docstring 第一段（遇到 Args: 就截断）
        doc = inspect.getdoc(fn) or ""
        summary = doc.split("Args:")[0].strip() if "Args:" in doc else doc
        summary = summary or f"工具 {tool_name}"

        self._tools[tool_name] = Tool(
            name=tool_name,
            description=summary,
            fn=fn,
            parameters=build_parameters_schema(fn),
            is_async=inspect.iscoroutinefunction(fn),
            timeout=timeout,
        )

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def all_schemas(self) -> list[dict]:
        return [t.to_api_schema() for t in self._tools.values()]

    def names(self) -> list[str]:
        return list(self._tools)

    async def execute(self, name: str, arguments: dict) -> str:
        """执行工具，把任何异常都转成给模型看的文本，而不是让进程崩掉。

        这一点很关键：模型经常会给出幻觉工具名或非法参数，
        直接抛异常会让整个 Agent 循环中断，而把错误回灌给模型，
        它下一轮往往能自己纠正过来。

        超时 / 重试 / 熔断在这一层统一处理，工具函数本身不必关心。
        """
        tool = self.get(name)
        if tool is None:
            available = ", ".join(self.names())
            return f"错误：工具 `{name}` 不存在。可用工具：{available}"

        outcome = await self.execute_tool(tool, arguments)
        return outcome.text

    async def execute_tool(self, tool: Tool, arguments: dict) -> ToolOutcome:
        """带韧性策略的执行，返回含 attempts 的完整结果，供统计成功率使用。"""

        async def _call(**kwargs):
            if tool.is_async:
                return await tool.fn(**kwargs)
            # 同步工具丢到线程池，避免阻塞事件循环
            return await asyncio.to_thread(tool.fn, **kwargs)

        # 参数签名不匹配属于永久错误，先挡掉，别浪费重试次数
        try:
            inspect.signature(tool.fn).bind(**arguments)
        except TypeError as e:
            return ToolOutcome(
                text=f"错误：参数不合法（{e}）。请检查参数名和类型后重试。",
                attempts=0,
                succeeded=False,
            )

        return await execute_with_resilience(
            _call,
            arguments,
            is_async=True,
            # 工具级别的超时优先于全局值
            timeout=tool.timeout if tool.timeout is not None else self.timeout,
            policy=self.policy,
            breaker=self.breaker,
            tool_name=tool.name,
            clock=self.clock,
        )


registry = ToolRegistry()

#: 需要比全局超时更宽松的工具。
#: RAG 检索首次调用要下载/加载模型并构建索引（实测约 47s），
#: 用默认 10s 会把正常的冷启动误判为故障，进而触发重试与熔断。
_TOOL_TIMEOUTS = {"search_policy": 120.0}


def tool(fn: Callable) -> Callable:
    """把一个普通函数注册成 Agent 可调用的工具。

    函数自身的类型注解决定 JSON Schema，docstring 决定工具说明，
    因此写完函数不用再维护任何额外的描述文件。
    """
    registry.register(fn, timeout=_TOOL_TIMEOUTS.get(fn.__name__))
    return fn
