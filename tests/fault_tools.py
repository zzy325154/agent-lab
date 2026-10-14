"""故障注入工具，仅用于验证韧性层。

这些工具不注册进主程序，测试时手动装进一个独立的 ToolRegistry。
让「工具故障」变成可控参数，才能测出成功率、重试次数、熔断行为。
"""

import asyncio
import time


class Fault:
    """可配置的故障行为。"""

    def __init__(self) -> None:
        self.fail_times = 0        # 前 N 次失败，之后成功
        self.mode = "exception"    # exception | timeout | sleep
        self.sleep_seconds = 0.0
        self.calls = 0
        self.timestamps: list[float] = []

    def reset(self) -> None:
        self.calls = 0
        self.timestamps.clear()


def make_flaky_tool(fault: Fault):
    """前 fail_times 次调用抛异常，之后正常返回。"""

    def flaky_tool(query: str) -> str:
        """查询一个不稳定的外部服务。

        Args:
            query: 查询内容
        """
        fault.calls += 1
        fault.timestamps.append(time.monotonic())
        if fault.calls <= fault.fail_times:
            if fault.mode == "timeout":
                time.sleep(fault.sleep_seconds)
                return "不该走到这里"
            raise ConnectionError(f"模拟下游服务抖动（第 {fault.calls} 次调用）")
        return f"查询「{query}」成功，返回 3 条结果"

    return flaky_tool


def make_slow_tool(fault: Fault):
    """总是慢于超时阈值，用来验证超时判定。"""

    def slow_tool(query: str) -> str:
        """一个响应很慢的服务。

        Args:
            query: 查询内容
        """
        fault.calls += 1
        fault.timestamps.append(time.monotonic())
        time.sleep(fault.sleep_seconds)
        return f"查询「{query}」成功（耗时 {fault.sleep_seconds}s）"

    return slow_tool


def make_broken_tool(fault: Fault):
    """参数一旦不合法就永久失败，用来验证「不该重试的错误不重试」。"""

    def broken_tool(query: str) -> str:
        """一个总在参数上失败的服务。

        Args:
            query: 查询内容
        """
        fault.calls += 1
        fault.timestamps.append(time.monotonic())
        raise ValueError("业务规则拒绝：该查询关键字被禁用")

    return broken_tool
