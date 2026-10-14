"""上下文管理的多轮对话效果测量。

不调用 API —— 基于 tests/calibrate_tokens.py 校准过的估算器（误差 1.3%）计算，
因此可在没有余额时复现。构造 10 轮对话，每轮都触发一次长文档检索，
对比「不管理」与「开管理」的累积输入 token。

运行：python tests/measure_context.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app.tools.builtin  # noqa: F401
from app.agent import SYSTEM_PROMPT
from app.context import ContextManager, estimate_messages, estimate_tools, estimate_tokens
from app.registry import registry

TOOLS = registry.all_schemas()
FIXED = estimate_tokens(SYSTEM_PROMPT) + estimate_tools(TOOLS)

# 每轮都问制度类问题，确保触发 search_policy 的长返回
QUESTIONS = [
    ("帮我查一下公司的报销制度", "报销"),
    ("那年假具体有多少天", "年假"),
    ("出差的话补贴标准是什么", "出差"),
    ("再确认一次报销的审批流程", "报销"),
    ("年假可以结转到明年吗", "年假"),
    ("出差能坐飞机吗", "出差"),
    ("报销单要多久内提交", "报销"),
    ("年假需要提前几天申请", "年假"),
    ("出差当天的交通费怎么算", "出差"),
    ("超过五千块的报销要谁批", "报销"),
]


def build_history(turns: int) -> list[dict]:
    """构造 turns 轮完整历史：提问 -> 工具调用 -> 长工具结果 -> 回答。"""
    from app.registry import registry

    msgs: list[dict] = []
    for i, (q, kw) in enumerate(QUESTIONS[:turns]):
        result = registry.get("search_policy").fn(keyword=kw)
        msgs.extend(
            [
                {"role": "user", "content": q},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "我来查询相关制度"},
                        {
                            "type": "tool_use",
                            "id": f"call_{i}",
                            "name": "search_policy",
                            "input": {"keyword": kw},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": f"call_{i}",
                            "content": result,
                        }
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": f"这是第 {i} 轮的回答"}]},
            ]
        )
    return msgs


def main() -> int:
    print("多轮对话上下文管理效果（基于校准估算器，误差 ±1.3%）")
    print(f"固定开销：system {estimate_tokens(SYSTEM_PROMPT)} + 工具 {estimate_tools(TOOLS)}"
          f" = {FIXED} token/轮")
    print("=" * 66)

    # 不管理：每轮的完整历史
    raw_per_turn = []
    for n in range(1, len(QUESTIONS) + 1):
        raw_per_turn.append(FIXED + estimate_messages(build_history(n)))

    # 开管理：预算 2000，保留最近 2 轮，工具结果压到 300 token
    cm = ContextManager(max_tokens=2000, keep_recent_turns=2, tool_result_max_tokens=300)
    managed_per_turn = []
    for n in range(1, len(QUESTIONS) + 1):
        _, payload, _ = cm.build(SYSTEM_PROMPT, build_history(n), TOOLS)
        managed_per_turn.append(estimate_tokens(SYSTEM_PROMPT) + estimate_tools(TOOLS)
                                + estimate_messages(payload))

    print(f"{'轮次':<6}{'不管理':>10}{'开管理':>10}{'节省':>10}")
    print("-" * 66)
    for i, (r, m) in enumerate(zip(raw_per_turn, managed_per_turn), 1):
        pct = (r - m) / r * 100 if r else 0
        print(f"第{i:>2}轮{r:>10}{m:>10}{pct:>9.0f}%")

    raw_total = sum(raw_per_turn)
    managed_total = sum(managed_per_turn)
    saved = raw_total - managed_total
    pct = saved / raw_total * 100

    print("-" * 66)
    print(f"{'合计':<6}{raw_total:>10}{managed_total:>10}{pct:>9.0f}%")
    print()
    print(f"10 轮对话输入 token：{raw_total} -> {managed_total}")
    print(f"节省 {saved} token，降低 {pct:.1f}%")
    print()
    print(f"末轮膨胀对比：第 10 轮 不管理 {raw_per_turn[-1]} vs 开管理 {managed_per_turn[-1]}"
          f"（{(raw_per_turn[-1]-managed_per_turn[-1])/raw_per_turn[-1]*100:.0f}%）")

    # 单轮长工具结果的压缩率
    from app.registry import registry as reg

    sample = reg.get("search_policy").fn(keyword="报销")
    compressed = cm._compress_text(sample)
    print(f"单个长工具结果压缩：{len(sample)} 字符 -> {len(compressed)} 字符"
          f"（压缩 {100 - len(compressed)/len(sample)*100:.0f}%）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
