"""韧性层测试：重试、超时、熔断、错误分类。

重点验证「选择性重试」这个判断——给必然失败的操作加重试是新手常见错误，
这里用调用次数直接断言：永久错误只允许调用一次。

运行：python tests/test_resilience.py
"""

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.registry import ToolRegistry
from app.resilience import (
    CircuitBreaker,
    FailureKind,
    RetryPolicy,
    classify_error,
)
from tests.fault_tools import Fault, make_broken_tool, make_flaky_tool, make_slow_tool


async def test_transient_retry_then_success():
    """下游抖动 2 次后恢复，重试应救回来。"""
    fault = Fault()
    fault.fail_times = 2

    reg = ToolRegistry(
        timeout=5.0,
        policy=RetryPolicy(max_attempts=3, base_delay=0.01, jitter=0.0),
    )
    reg.register(make_flaky_tool(fault))

    outcome = await reg.execute_tool(reg.get("flaky_tool"), {"query": "abc"})

    assert outcome.succeeded, f"应重试成功，实际失败: {outcome.text}"
    assert outcome.attempts == 3, f"应尝试 3 次，实际 {outcome.attempts}"
    assert fault.calls == 3, f"工具应被调用 3 次，实际 {fault.calls}"
    print(f"  [PASS] 瞬时故障重试成功（尝试 {outcome.attempts} 次）")


async def test_permanent_error_not_retried():
    """永久性错误必须立即放弃，重试是纯浪费。"""
    fault = Fault()

    reg = ToolRegistry(
        timeout=5.0,
        policy=RetryPolicy(max_attempts=3, base_delay=0.01, jitter=0.0),
    )
    reg.register(make_broken_tool(fault))

    outcome = await reg.execute_tool(reg.get("broken_tool"), {"query": "abc"})

    assert not outcome.succeeded
    assert outcome.kind is FailureKind.PERMANENT, f"应判定为永久错误，实际 {outcome.kind}"
    assert fault.calls == 1, f"永久错误只该调用 1 次，实际调用了 {fault.calls} 次"
    print("  [PASS] 永久错误不重试（仅调用 1 次）")


async def test_timeout_is_retried():
    """超时属于瞬时故障，应重试。"""
    fault = Fault()
    fault.sleep_seconds = 0.5

    reg = ToolRegistry(
        timeout=0.1,
        policy=RetryPolicy(max_attempts=3, base_delay=0.01, jitter=0.0),
    )
    reg.register(make_slow_tool(fault))

    outcome = await reg.execute_tool(reg.get("slow_tool"), {"query": "abc"})

    assert not outcome.succeeded, "不该成功"
    assert outcome.kind is FailureKind.TRANSIENT, f"超时应归为瞬时故障，实际 {outcome.kind}"
    assert fault.calls == 3, f"超时应重试满 3 次，实际 {fault.calls}"
    print("  [PASS] 超时被正确判为瞬时故障并重试")


async def test_slow_eventually_succeeds():
    """超时阈值放宽后，慢服务应成功。"""
    fault = Fault()
    fault.sleep_seconds = 0.15

    reg = ToolRegistry(timeout=2.0, policy=RetryPolicy(max_attempts=2, base_delay=0.01))
    reg.register(make_slow_tool(fault))

    outcome = await reg.execute_tool(reg.get("slow_tool"), {"query": "abc"})

    assert outcome.succeeded, outcome.text
    assert outcome.attempts == 1, "一次就该成功，不该重试"
    print("  [PASS] 超时阈值内慢服务正常成功")


async def test_backoff_grows_exponentially():
    """退避时间必须递增，否则重试会变成对下游的连续冲击。"""
    policy = RetryPolicy(base_delay=0.1, max_delay=10.0, jitter=0.0)
    delays = [policy.delay_for(i) for i in range(1, 5)]

    assert delays == [0.1, 0.2, 0.4, 0.8], f"退避序列错误: {delays}"

    # 有抖动时仍应保持递增趋势（取多次采样的均值）
    jittery = RetryPolicy(base_delay=0.1, max_delay=10.0, jitter=0.3)
    avg = [sum(jittery.delay_for(i) for _ in range(200)) / 200 for i in range(1, 5)]
    assert avg[0] < avg[1] < avg[2] < avg[3], f"抖动后仍未递增: {avg}"
    print(f"  [PASS] 退避递增（无抖动 {delays}，有抖动均值 "
          f"{[round(a, 3) for a in avg]}）")


async def test_backoff_caps_at_max():
    """退避必须有上限，否则重试间隔会指数爆炸。"""
    policy = RetryPolicy(base_delay=0.1, max_delay=1.0, jitter=0.0)
    assert policy.delay_for(10) == 1.0, "未在 max_delay 处封顶"
    print("  [PASS] 退避在 max_delay 处封顶")


async def test_circuit_breaker_opens():
    """连续失败到阈值应熔断，之后不再击打下游。"""
    fault = Fault()
    broken = make_broken_tool(fault)

    breaker = CircuitBreaker(threshold=3, cooldown=60.0)
    reg = ToolRegistry(
        timeout=5.0,
        policy=RetryPolicy(max_attempts=1, base_delay=0.01),
        breaker=breaker,
    )
    reg.register(broken)

    # 前 3 次各触发一次失败记录
    for _ in range(3):
        await reg.execute_tool(reg.get("broken_tool"), {"query": "x"})

    assert breaker.state().get("broken_tool") == 3, f"失败计数错误: {breaker.state()}"
    assert breaker.is_open("broken_tool", time.monotonic()), "达到阈值后应熔断"

    calls_before = fault.calls
    outcome = await reg.execute_tool(reg.get("broken_tool"), {"query": "x"})

    assert not outcome.succeeded
    assert "熔断" in outcome.text, f"应返回熔断提示，实际: {outcome.text}"
    assert fault.calls == calls_before, "熔断后不该再调用下游"
    print(f"  [PASS] 连续失败 {3} 次后熔断并停止调用下游")


async def test_circuit_breaker_recovers_after_cooldown():
    """冷却结束后应放行探测，不能永久熔断。"""
    breaker = CircuitBreaker(threshold=2, cooldown=0.3)

    now = [1000.0]  # 可控时钟
    breaker.record_failure("t", now[0])
    breaker.record_failure("t", now[0])
    assert breaker.is_open("t", now[0]), "应已熔断"

    assert breaker.is_open("t", now[0] + 0.1), "冷却期内不该放行"

    assert not breaker.is_open("t", now[0] + 0.4), "冷却结束后应放行"
    assert not breaker.is_open("t", now[0] + 0.4), "放行后应回到关闭状态"
    print("  [PASS] 熔断在冷却期后自动恢复")


async def test_success_resets_failure_count():
    """成功一次就该清零，避免偶发失败累积成误熔断。"""
    breaker = CircuitBreaker(threshold=3, cooldown=60.0)
    now = time.monotonic()

    breaker.record_failure("t", now)
    breaker.record_failure("t", now)
    assert breaker.state()["t"] == 2

    breaker.record_success("t")
    assert "t" not in breaker.state(), "成功后失败计数应清零"
    assert not breaker.is_open("t", now)
    print("  [PASS] 成功后失败计数清零")


def test_error_classification():
    """错误分类是选择性重试的依据，必须准。"""
    cases = [
        (asyncio.TimeoutError(), FailureKind.TRANSIENT),
        (ConnectionError("reset"), FailureKind.TRANSIENT),
        (ValueError("bad arg"), FailureKind.PERMANENT),
        (TypeError("wrong type"), FailureKind.PERMANENT),
        (KeyError("missing"), FailureKind.PERMANENT),
    ]
    for exc, expected in cases:
        got = classify_error(exc)
        assert got is expected, f"{type(exc).__name__} 应判为 {expected}，实际 {got}"

    # 按类名兜底：模拟各家 SDK 的超时异常
    class APITimeoutError(Exception):
        pass

    class RateLimitError(Exception):
        pass

    assert classify_error(APITimeoutError()) is FailureKind.TRANSIENT
    assert classify_error(RateLimitError()) is FailureKind.TRANSIENT
    print("  [PASS] 错误分类正确（含 SDK 类名兜底）")


async def test_bad_arguments_short_circuit():
    """参数不匹配应在调用前拦下，不占用重试次数。"""
    fault = Fault()
    reg = ToolRegistry(timeout=5.0, policy=RetryPolicy(max_attempts=3, base_delay=0.01))
    reg.register(make_flaky_tool(fault))

    outcome = await reg.execute_tool(reg.get("flaky_tool"), {"wrong_param": "x"})

    assert not outcome.succeeded
    assert outcome.attempts == 0, f"参数错误不该产生尝试，实际 {outcome.attempts}"
    assert fault.calls == 0, "参数不合法不该真的调用工具"
    assert "参数不合法" in outcome.text
    print("  [PASS] 参数不合法在调用前拦截（工具未被调用）")


async def main() -> int:
    tests = [
        test_transient_retry_then_success,
        test_permanent_error_not_retried,
        test_timeout_is_retried,
        test_slow_eventually_succeeds,
        test_backoff_grows_exponentially,
        test_backoff_caps_at_max,
        test_circuit_breaker_opens,
        test_circuit_breaker_recovers_after_cooldown,
        test_success_resets_failure_count,
        test_error_classification,
        test_bad_arguments_short_circuit,
    ]
    print("韧性层测试：")
    for t in tests:
        result = t()
        if asyncio.iscoroutine(result):
            await result
    print(f"\n全部 {len(tests)} 项通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
