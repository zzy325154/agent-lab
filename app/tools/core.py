"""工具的唯一实现来源。

这里的函数是纯业务逻辑，不含任何框架依赖——不认识 MCP、也不认识
本地的 ToolRegistry。两种接入方式各自把它包一层：

    core.py (实现)  ──┬── builtin.py  (本地注册，供工具调用 Agent 使用)
                      └── mcp_server.py (MCP Server 暴露，供任意 MCP 客户端使用)

这样新增一个工具只需写一次实现，两条链路自动都有。
如果实现写两份，迟早会漂移——那正是「热插拔」想解决的问题的反面。
"""

import ast
import operator
import time
from datetime import datetime
from typing import Annotated

from pydantic import Field

# 制度文档已迁至 app/rag/corpus.py，作为 RAG 语料

_SALES_DB: dict[str, list[tuple[str, int]]] = {
    "成都": [
        ("春熙路店", 1284000),
        ("天府三街店", 963000),
        ("金融城店", 871500),
        ("建设路店", 702300),
        ("犀浦店", 588100),
    ],
    "杭州": [
        ("武林广场店", 1472000),
        ("未来科技城店", 1198000),
        ("滨江宝龙城店", 934700),
        ("城西银泰店", 815400),
        ("萧山万象汇店", 623900),
    ],
}


# --------------------------------------------------------------------------
# 工具实现（纯业务逻辑，无框架依赖）
# --------------------------------------------------------------------------


def calculator(expression: Annotated[str, Field(description='待计算的表达式，例如 (12 + 8) * 3 / 5')]) -> str:
    """计算一个数学表达式，支持 + - * / ** % 和括号。

    涉及数字运算时优先使用本工具，不要自己心算，避免算错。

    Args:
        expression: 待计算的表达式，例如 "(12 + 8) * 3 / 5"
    """
    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("只支持数字常量")
        if isinstance(node, ast.BinOp):
            op = allowed.get(type(node.op))
            if op is None:
                raise ValueError(f"不支持的运算符: {type(node.op).__name__}")
            return op(_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            op = allowed.get(type(node.op))
            if op is None:
                raise ValueError("不支持的一元运算符")
            return op(_eval(node.operand))
        raise ValueError(f"不支持的语法: {type(node).__name__}")

    # 用 AST 求值而不是 eval()，杜绝表达式注入
    tree = ast.parse(expression, mode="eval")
    result = _eval(tree.body)

    # 浮点误差会让 100*1.13 得到 112.99999999999999，
    # 这种脏数字喂给模型纯属噪音，先按十进制精度修掉
    if isinstance(result, float):
        result = round(result, 10)
        if result.is_integer():
            result = int(result)
    return f"{expression} = {result}"


def get_current_time(
    timezone: Annotated[str, Field(description='时区名称')] = "Asia/Shanghai",
) -> str:
    """获取当前日期和时间。

    当问题涉及「今天」「现在」「几天后」等时间概念时使用。

    Args:
        timezone: 时区名称，默认 Asia/Shanghai
    """
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return (
        f"当前时间：{now.strftime('%Y-%m-%d %H:%M:%S')} "
        f"{weekdays[now.weekday()]}（{timezone}）"
    )


def query_sales(
    city: Annotated[str, Field(description='城市名称，如「成都」「杭州」')],
    top_n: Annotated[int, Field(description='返回前几名')] = 5,
) -> str:
    """查询指定城市的销售排行数据。

    数据是演示用的内置样例，用于模拟「查数据库」这类真实工具调用。

    Args:
        city: 城市名称，如「成都」「杭州」
        top_n: 返回前几名，默认 5
    """
    time.sleep(0.2)  # 模拟一次数据库查询的耗时，便于在日志里观察

    rows = _SALES_DB.get(city)
    if rows is None:
        return f"暂无「{city}」的数据。当前仅支持：{'、'.join(_SALES_DB)}"

    lines = [f"{city} 销售排行 Top{min(top_n, len(rows))}："]
    for i, (store, amount) in enumerate(rows[:top_n], 1):
        lines.append(f"  {i}. {store}：{amount:,} 元")
    return "\n".join(lines)


async def search_policy(
    keyword: Annotated[
        str,
        Field(description="检索问题，用自然语言描述，如「出差住宿能报多少」「年假可以结转吗」"),
    ],
) -> str:
    """检索公司规章制度（报销、年假、出差）。

    这是真正的语义检索：支持口语化提问，不必包含文档里的原词。
    问「出差花钱怎么报」也能命中「差旅费报销标准」。

    Args:
        keyword: 检索问题，例如「出差住宿标准」「年假多少天」
    """
    from app.rag.service import format_outcome, search

    outcome = await search(keyword, top_k=5)
    return format_outcome(outcome)


#: 需要暴露给 Agent 的工具清单。新增工具时在这里追加一行即可。
ALL_TOOLS = [calculator, get_current_time, query_sales, search_policy]
