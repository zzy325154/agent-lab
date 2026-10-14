"""把 core.py 里的工具实现注册到本地 ToolRegistry。

这是一层薄适配：只负责打 @tool 装饰器，不含任何业务逻辑。
实现本体在 core.py，MCP Server 用的是同一份。

新增工具时：在 core.py 写实现并加进 ALL_TOOLS，这里自动生效。
"""

from app.registry import tool
from app.tools import core

# 逐个注册，保持函数名即工具名
for _fn in core.ALL_TOOLS:
    tool(_fn)

# 显式引用，避免 linter 认为导入未使用
__all__ = ["core"]
