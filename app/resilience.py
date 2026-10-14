"""工具体系的韧性层：超时、选择性重试、熔断、降级。

核心判断是「哪些错值得重试」：
  - 瞬时故障（超时、连接中断、5xx）——重试有意义，外部服务可能只是抖了一下
  - 永久故障（参数不合法、工具不存在、业务规则拒绝）——重试纯属浪费，
    同样的输入必然得到同样的结果，重试只会拖长响应并消耗配额

一个新手常犯的错是给所有失败都加重试，结果把一个必然失败的调用
重试三次，用户多等三倍时间还是失败。
"""

import asyncio
import random
from dataclasses import dataclass, field
from enum import Enum


class FailureKind(Enum):
    TRANSIENT = "transient"    # 重试可能成功
    PERMANENT = "permanent"    # 重试没有意义
    UNKNOWN = "unknown"


# 按异常类型判断是否值得重试
_TRANSIENT_EXCEPTIONS = (
    asyncio.TimeoutError,
    ConnectionError,
    ConnectionResetError,
    ConnectionRefusedError,
    BrokenPipeError,
    TimeoutError,
)


class ToolTimeout(Exception):
    """工具执行超时。"""


class ToolUnavailable(Exception):
    """工具被熔断，暂时不可用。"""


def classify_error(exc: BaseException) -> FailureKind:
    if isinstance(exc, _TRANSIENT_EXCEPTIONS):
        return FailureKind.TRANSIENT
    if isinstance(exc, (TypeError, ValueError, KeyError, AttributeError, ToolUnavailable)):
        return FailureKind.PERMANENT

    # 按异常名兜底：各家 SDK 的超时异常类名各不相同
    name = type(exc).__name__.lower()
    if any(k in name for k in ("timeout", "connection", "unavailable", "ratelimit", "overloaded")):
        return FailureKind.TRANSIENT
    if any(k in name for k in ("authentication", "permission", "notfound", "badrequest")):
        return FailureKind.PERMANENT
    return FailureKind.UNKNOWN


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.1     # 秒
    max_delay: float = 2.0
    jitter: float = 0.3         # 抖动比例，避免多个调用同时重试形成尖峰

    def delay_for(self, attempt: int) -> float:
        """指数退避 + 抖动。attempt 从 1 开始。"""
        raw = min(self.base_delay * (2 ** (attempt - 1)), self.max_delay)
        return raw * (1 + random.uniform(-self.jitter, self.jitter))


@dataclass
class CircuitBreaker:
    """连续失败到阈值就熔断，避免持续击打已经挂掉的下游。

    这解决的是「重试放大故障」的问题：上游一挂，
    所有 Agent 实例都在重试，反而把下游彻底压死。
    """

    threshold: int = 5
    cooldown: float = 30.0
    _failures: dict[str, int] = field(default_factory=dict)
    _opened_at: dict[str, float] = field(default_factory=dict)

    def is_open(self, tool_name: str, now: float) -> bool:
        opened = self._opened_at.get(tool_name)
        if opened is None:
            return False
        if now - opened >= self.cooldown:
            # 冷却结束，放行一次探测
            self._opened_at.pop(tool_name, None)
            self._failures[tool_name] = 0
            return False
        return True

    def record_success(self, tool_name: str) -> None:
        self._failures.pop(tool_name, None)
        self._opened_at.pop(tool_name, None)

    def record_failure(self, tool_name: str, now: float) -> bool:
        """记录一次失败。返回是否因此触发熔断。"""
        self._failures[tool_name] = self._failures.get(tool_name, 0) + 1
        if self._failures[tool_name] >= self.threshold:
            self._opened_at[tool_name] = now
            return True
        return False

    def state(self) -> dict[str, int]:
        return dict(self._failures)


@dataclass
class ToolOutcome:
    text: str
    attempts: int
    succeeded: bool
    kind: FailureKind | None = None


async def execute_with_resilience(
    fn,
    arguments: dict,
    *,
    is_async: bool,
    timeout: float,
    policy: RetryPolicy,
    breaker: CircuitBreaker,
    tool_name: str,
    clock,
) -> ToolOutcome:
    """带超时、选择性重试与熔断的工具执行。

    clock 是一个返回当前时间的可调用对象，便于测试注入。
    """
    if breaker.is_open(tool_name, clock()):
        return ToolOutcome(
            text=f"错误：工具 `{tool_name}` 连续失败已暂时熔断，请改用其他方式或稍后再试。",
            attempts=0,
            succeeded=False,
            kind=FailureKind.PERMANENT,
        )

    last_kind: FailureKind | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            if is_async:
                result = await asyncio.wait_for(fn(**arguments), timeout=timeout)
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(fn, **arguments), timeout=timeout
                )
        except BaseException as exc:  # noqa: BLE001 — 需要覆盖超时与各类 SDK 异常
            kind = classify_error(exc) if not isinstance(exc, asyncio.TimeoutError) \
                else FailureKind.TRANSIENT

            # 超时与瞬时故障才重试；参数错误之类重试无意义
            if kind is FailureKind.PERMANENT or attempt >= policy.max_attempts:
                broke = breaker.record_failure(tool_name, clock())
                note = "（已触发熔断）" if broke else ""
                return ToolOutcome(
                    text=f"错误：工具 `{tool_name}` 执行失败"
                         f"（{type(exc).__name__}: {exc}），重试 {attempt} 次后放弃{note}。",
                    attempts=attempt,
                    succeeded=False,
                    kind=kind,
                )

            last_kind = kind
            backoff = policy.delay_for(attempt)
            await asyncio.sleep(backoff)
            continue

        breaker.record_success(tool_name)
        return ToolOutcome(text=str(result), attempts=attempt, succeeded=True)

    return ToolOutcome(
        text=f"错误：工具 `{tool_name}` 重试 {policy.max_attempts} 次后仍失败。",
        attempts=policy.max_attempts,
        succeeded=False,
        kind=last_kind,
    )
