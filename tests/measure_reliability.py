"""工具成功率测量：对比关闭/开启韧性层。

方法：注入固定故障率的外呼工具，跑 N 次调用，统计最终成功率。
「关闭」= 单次尝试无重试；「开启」= 指数退避重试 3 次 + 超时 + 熔断。

这里全部是本地调用，不消耗 API 额度，因此可以跑大样本。

运行：python tests/measure_reliability.py
"""

import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.registry import ToolRegistry
from app.resilience import CircuitBreaker, RetryPolicy, ToolOutcome, execute_with_resilience

# 固定种子保证结果可复现 —— 简历数字必须能被重复验证
SEED = 20260930
FLAKY_RATE = 0.3      # 30% 的调用会瞬时失败
TIMEOUT_RATE = 0.05   # 5% 的调用会超时
CALLS = 1000          # 样本量
TIMEOUT_SECONDS = 0.05


def make_unstable_tool(rng: random.Random, counters: dict):
    """按设定概率随机抛瞬时错误或超时。"""

    def unstable_api(query: str) -> str:
        """模拟一个不稳定的外部接口。

        Args:
            query: 查询内容
        """
        counters["calls"] += 1
        roll = rng.random()
        if roll < TIMEOUT_RATE:
            # 超时：睡够超过阈值
            import time

            time.sleep(TIMEOUT_SECONDS * 2)
            return ""
        if roll < TIMEOUT_RATE + FLAKY_RATE:
            raise ConnectionError("模拟下游抖动")
        return f"ok:{query}"

    return unstable_api


async def run_batch(use_resilience: bool, calls: int) -> dict:
    """跑一批调用，返回成功率统计。"""
    rng = random.Random(SEED)
    counters = {"calls": 0}

    if use_resilience:
        policy = RetryPolicy(max_attempts=3, base_delay=0.01, max_delay=0.05, jitter=0.2)
        # 成功率测量不该被熔断干扰，阈值设高
        breaker = CircuitBreaker(threshold=10_000, cooldown=60.0)
        reg = ToolRegistry(timeout=TIMEOUT_SECONDS, policy=policy, breaker=breaker)
    else:
        # 单次尝试、长超时：等价于「不做任何韧性处理」
        reg = ToolRegistry(
            timeout=10.0,
            policy=RetryPolicy(max_attempts=1),
            breaker=CircuitBreaker(threshold=10_000_000, cooldown=0.0),
        )

    reg.register(make_unstable_tool(rng, counters))

    succeeded = 0
    total_attempts = 0
    transient_seen = 0

    for _ in range(calls):
        outcome: ToolOutcome = await reg.execute_tool(
            reg.get("unstable_api"), {"query": "q"}
        )
        if outcome.succeeded:
            succeeded += 1
        else:
            transient_seen += 1
        total_attempts += outcome.attempts

    return {
        "success": succeeded,
        "rate": succeeded / calls * 100,
        "attempts": total_attempts,
        "tool_invocations": counters["calls"],
        "calls": calls,
    }


async def main() -> int:
    print(f"工具成功率测量（样本 {CALLS} 次，故障率 {FLAKY_RATE:.0%} + 超时率 {TIMEOUT_RATE:.0%}）")
    print(f"随机种子固定为 {SEED}，结果可复现")
    print("=" * 62)

    off = await run_batch(use_resilience=False, calls=CALLS)
    on = await run_batch(use_resilience=True, calls=CALLS)

    print(f"{'':<14}{'无韧性层':>14}{'有韧性层':>14}")
    print("-" * 62)
    print(f"{'成功次数':<14}{off['success']:>14}{on['success']:>14}")
    print(f"{'成功率':<14}{off['rate']:>13.1f}%{on['rate']:>13.1f}%")
    print(f"{'工具实际调用':<12}{off['tool_invocations']:>14}{on['tool_invocations']:>14}")
    print(f"{'总尝试次数':<13}{off['attempts']:>14}{on['attempts']:>14}")

    print()
    gain = on["rate"] - off["rate"]
    print(f"成功率：{off['rate']:.1f}% -> {on['rate']:.1f}%，提升 {gain:.1f} 个百分点")
    print(f"失败次数：{off['calls'] - off['success']} -> {on['calls'] - on['success']}"
          f"（减少 {(1 - (on['calls'] - on['success']) / max(1, off['calls'] - off['success'])) * 100:.0f}%）")
    print(f"代价：调用量增加 {on['tool_invocations'] - off['tool_invocations']} 次"
          f"（{on['tool_invocations'] / off['tool_invocations']:.2f}x），"
          f"用于换取失败率下降")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
