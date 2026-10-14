"""MCP Server：把 core.py 的工具能力通过 Model Context Protocol 暴露。

MCP 的价值在这里落地：工具不再硬编码在 Agent 内部，而是作为独立进程
通过标准协议对外服务。任何支持 MCP 的客户端（Claude Desktop、其他
Agent 框架、IDE 插件）都能直接接入，Agent 侧零改动。

对照 builtin.py 可以看到：同一个 core.ALL_TOOLS，一边注册成本地工具，
一边暴露成 MCP 工具——**能力定义只有一份，接入方式是两套**。
这就是「工具热插拔」的实际形态。

传输方式：
    stdio           —— 本地子进程，客户端拉起（本文件默认）
    streamable-http —— 远程服务，换 server.run(transport=...) 即可

单独调试：python -m app.mcp_server
"""

import inspect
from functools import wraps

from mcp.server.mcpserver import MCPServer

from app.tools import core

server = MCPServer(
    name="agent-lab-tools",
    version="1.0.0",
    instructions="提供数学计算、时间查询、销售数据查询与规章制度检索能力。",
)


def _as_text_error(fn):
    """把工具抛出的异常转成模型可读的错误文本。

    不做这层包装的话，MCP Server 会把异常统一吞成
    「Error executing tool xxx」并把原始信息打到 stderr——
    模型拿到这种无信息量的错误就没法自我纠正，
    而我们整套错误恢复机制正是建立在「把结构化错误回灌给模型」之上的。

    必须区分同步与异步工具：工具集里既有普通函数，也有 async 函数
    （如 RAG 检索，它要 await 编码计算）。异步工具若被同步包装，
    返回的会是 coroutine 对象而非结果——这个 bug 在 MCP 链路上
    表现为模型收到一串无意义的对象表示。

    functools.wraps 保留 __doc__ 与 __annotations__，
    MCPServer 因此仍能从原始签名推导出正确的 input_schema。
    """
    if inspect.iscoroutinefunction(fn):

        @wraps(fn)
        async def async_wrapper(*args, **kwargs):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:  # noqa: BLE001 — 边界层需要兜住一切
                return f"错误：{type(e).__name__}: {e}"

        return async_wrapper

    @wraps(fn)
    def sync_wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            return f"错误：{type(e).__name__}: {e}"

    return sync_wrapper


# 把 core 里的实现挂上去。MCPServer 从同一套类型注解与 docstring
# 推导 input_schema，因此描述与本地链路完全一致，不会出现两份文档。
for _fn in core.ALL_TOOLS:
    server.add_tool(_as_text_error(_fn))


if __name__ == "__main__":
    # stdio 传输：由客户端以子进程方式拉起，走标准输入输出通信
    server.run(transport="stdio")
