"""上下文管理测试。

最关键的一条：无论怎么裁剪，tool_use 和它的 tool_result 都必须成对出现，
否则真实 API 会直接返回 400。这类 bug 在单元测试里不抓出来，
上线后只会在「对话到第 N 轮」时突然报错。

运行：python tests/test_context.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.context import ContextManager, estimate_messages, estimate_tokens


def _tool_round(round_id: int, result_text: str) -> list[dict]:
    """构造一轮完整的「提问 -> 调用工具 -> 工具结果」。"""
    return [
        {"role": "user", "content": f"这是第 {round_id} 个问题，请帮我查询一下相关信息"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "我这就查"},
                {
                    "type": "tool_use",
                    "id": f"call_{round_id}",
                    "name": "query_sales",
                    "input": {"city": "成都"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": f"call_{round_id}",
                    "content": result_text,
                }
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": f"第 {round_id} 轮的回答"}],
        },
    ]


def _check_pairing(messages: list[dict]) -> None:
    """断言每个 tool_use 都有配对的 tool_result，反之亦然。"""
    used_ids: set[str] = set()
    result_ids: set[str] = set()
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                used_ids.add(block["id"])
            elif block.get("type") == "tool_result":
                result_ids.add(block["tool_use_id"])

    orphan_uses = used_ids - result_ids
    orphan_results = result_ids - used_ids
    assert not orphan_uses, f"这些 tool_use 没有配对结果: {orphan_uses}"
    assert not orphan_results, f"这些 tool_result 没有配对调用: {orphan_results}"


def test_short_context_untouched():
    """短对话不该被改动，否则无谓地损失信息。"""
    cm = ContextManager(max_tokens=100000)
    msgs = _tool_round(1, "短结果")
    system, out, stats = cm.build("系统提示", msgs)

    assert len(out) == len(msgs), "短对话被误裁剪"
    assert stats.compressed_results == 0
    assert stats.dropped_turns == 0
    assert stats.saved_tokens == 0
    assert "已省略" not in system
    print("  [PASS] 短对话不被改动")


def test_tool_result_compression():
    """超长工具结果应被压缩，首尾内容保留。"""
    cm = ContextManager(max_tokens=100000, tool_result_max_tokens=100)
    long_text = "开头很重要" + "中间废话" * 500 + "结尾有错误码 E502"
    msgs = _tool_round(1, long_text)

    _, out, stats = cm.build("系统提示", msgs)

    assert stats.compressed_results == 1, "长结果未被压缩"
    compressed = out[2]["content"][0]["content"]
    assert "开头很重要" in compressed, "首部内容丢失"
    assert "E502" in compressed, "尾部内容丢失"
    assert "已省略" in compressed, "没有标记省略位置"
    assert len(compressed) < len(long_text), "压缩后反而变长"
    _check_pairing(out)
    print(f"  [PASS] 长工具结果被压缩（{len(long_text)} -> {len(compressed)} 字符）")


def test_turn_trimming_preserves_pairing():
    """核心用例：裁剪历史轮次时绝不能拆散工具调用对。"""
    cm = ContextManager(max_tokens=300, keep_recent_turns=2)
    msgs = []
    for i in range(8):
        msgs.extend(_tool_round(i, "结果" * 50))

    _, out, stats = cm.build("系统提示", msgs)

    assert stats.dropped_turns > 0, "超长历史未被裁剪"
    _check_pairing(out)
    assert estimate_messages(out) <= 300 + 200, f"裁剪后仍超预算: {estimate_messages(out)}"
    # 最近的轮次必须保留
    tail_texts = [b.get("text", "") for m in out if isinstance(m["content"], list) for b in m["content"] if isinstance(b, dict)]
    assert any("第 7 轮" in t for t in tail_texts), "最近的轮次被误删"
    print(f"  [PASS] 裁剪 {stats.dropped_turns} 轮历史且工具调用对完整")


def test_keep_recent_turns_floor():
    """即使超预算，也必须保留最近 N 轮，不能裁到只剩一句提问。"""
    cm = ContextManager(max_tokens=10, keep_recent_turns=3)
    msgs = []
    for i in range(10):
        msgs.extend(_tool_round(i, "结果" * 100))

    _, out, stats = cm.build("系统提示", msgs)

    assert stats.dropped_turns == 7, f"应丢弃 7 轮保留 3 轮，实际 {stats.dropped_turns}"
    _check_pairing(out)
    print("  [PASS] 预算极小时仍保留最近 3 轮")


def test_system_gets_omission_notice():
    """有历史被丢弃时，system 里必须留痕，否则模型会误判对话长度。"""
    cm = ContextManager(max_tokens=300, keep_recent_turns=2)
    msgs = []
    for i in range(8):
        msgs.extend(_tool_round(i, "结果" * 50))

    system, _, stats = cm.build("原始系统提示", msgs)

    assert stats.dropped_turns > 0
    assert "已省略" in system, "system 未追加省略说明"
    assert "原始系统提示" in system, "原始 system 内容丢失"
    print("  [PASS] system 中标注了历史省略")


def test_estimate_calibration_sanity():
    """估算函数的合理性。

    这里刻意不假设「中文比英文省 token」——实测下来，
    表达同样信息量时中英文的 token 数大致相当（52 vs 50）。
    真正踩过的坑是另一种：JSON、代码这类符号密集的内容
    曾被严重低估（工具 Schema 估 350 实际 600），
    因为它们既不算中文也不算英文字母。
    """
    from app.context import estimate_tools

    assert estimate_tokens("") == 0, "空字符串应为 0"

    # 符号密集内容必须被计入，不能只统计中英文
    json_blob = '{"name":"query_sales","properties":{"city":{"type":"string"}}}'
    assert estimate_tokens(json_blob) > 15, (
        f"符号密集内容被低估: {estimate_tokens(json_blob)}"
    )

    # 工具定义的开销必须包含结构成本，否则工具一多预算就失真
    fake_tool = {
        "name": "demo",
        "description": "演示工具",
        "input_schema": {"type": "object", "properties": {}},
    }
    cost = estimate_tools([fake_tool])
    assert cost >= 63, f"工具结构开销未计入: {cost}"

    print(f"  [PASS] 符号内容与工具开销均被正确计入（单个工具 {cost} token）")


def main() -> int:
    tests = [
        test_short_context_untouched,
        test_tool_result_compression,
        test_turn_trimming_preserves_pairing,
        test_keep_recent_turns_floor,
        test_system_gets_omission_notice,
        test_estimate_calibration_sanity,
    ]
    print("上下文管理测试：")
    for t in tests:
        t()
    print(f"\n全部 {len(tests)} 项通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
