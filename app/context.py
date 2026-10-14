"""上下文管理：让多轮 Agent 对话的 token 消耗不随轮数线性膨胀。

手段按代价从低到高依次使用：
  1. 压缩工具结果 —— 长结果保留首尾、砍掉中段。工具输出通常是 token 大头。
  2. 裁剪历史轮次 —— 只保留最近 N 轮完整对话，更早的整轮丢弃。
  3. 标记省略而非静默截断 —— 模型需要知道「有些历史我看不到了」。

一条必须守住的约束：tool_use 与 tool_result 必须成对出现，
只删其一会让 API 直接报 400 错误。因此所有裁剪点都落在
「普通用户消息」（原始提问）上，绝不落在工具往返的中间。
"""

import copy
from dataclasses import dataclass

# 以下系数由 tests/calibrate_tokens.py 用真实 API usage 反推得出，勿凭直觉修改。
# 踩过的坑：最初把英文按 0.3 token/字符估算，比真实值低 3 倍；
# 且完全没给数字和标点计数，导致 JSON 工具 Schema 的 token 被严重低估。
_TOKEN_PER_CJK = 0.516        # 中文字符
_TOKEN_PER_ALPHA = 0.195      # 英文字母
_TOKEN_PER_DIGIT = 0.457      # 数字与标点符号
_TOKEN_PER_SPACE = 0.151      # 空白字符

# 每个工具除了自身 JSON 之外，API 还会包一层 {"type":"function","function":{...}}
# 结构，实测约 63 token/工具。不加上这个，工具越多的 Agent 预算就越失真。
_TOOL_STRUCT_OVERHEAD = 63
# 一次请求的基础开销（角色标记、消息分隔等）
_REQUEST_OVERHEAD = 8


def _is_cjk(ch: str) -> bool:
    o = ord(ch)
    return (
        0x4E00 <= o <= 0x9FFF      # 基本汉字
        or 0x3400 <= o <= 0x4DBF   # 扩展 A
        or 0x3000 <= o <= 0x303F   # 中文标点
        or 0xFF00 <= o <= 0xFFEF   # 全角字符
    )


def estimate_tokens(text: str) -> int:
    """按字符类别分桶估算 token 数。

    不用「中/英」二分法，是因为英文标点和数字的 token 成本
    接近中文而非字母——JSON、代码、URL 这类内容占比很高时，
    二分法会严重低估。
    """
    if not text:
        return 0

    cjk = alpha = digit = space = 0
    for ch in text:
        if _is_cjk(ch):
            cjk += 1
        elif ch.isalpha():
            alpha += 1
        elif ch.isspace():
            space += 1
        else:
            digit += 1

    total = (
        cjk * _TOKEN_PER_CJK
        + alpha * _TOKEN_PER_ALPHA
        + digit * _TOKEN_PER_DIGIT
        + space * _TOKEN_PER_SPACE
    )
    # 每条内容至少占 1 个 token
    return max(1, round(total))


def estimate_messages(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    total += estimate_tokens(block.get("text", ""))
                elif block.get("type") == "tool_result":
                    total += estimate_tokens(str(block.get("content", "")))
                elif block.get("type") == "tool_use":
                    total += estimate_tokens(block.get("name", ""))
                    # 参数按 JSON 估算
                    total += estimate_tokens(str(block.get("input", "")))
        # 每条消息有固定的角色/分隔开销
        total += 4
    return total


def estimate_tools(tools: list[dict] | None) -> int:
    """估算工具定义占用的 token。

    这一步很容易被忽略：工具 Schema 是每轮请求都要重发的固定开销，
    工具数量多时能占到总输入的一大半。
    """
    if not tools:
        return 0
    import json

    total = _REQUEST_OVERHEAD
    for t in tools:
        visible = {
            "name": t.get("name", ""),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {}),
        }
        total += estimate_tokens(json.dumps(visible, ensure_ascii=False))
        total += _TOOL_STRUCT_OVERHEAD
    return total


@dataclass
class ContextStats:
    original_tokens: int
    final_tokens: int
    compressed_results: int = 0
    dropped_turns: int = 0

    @property
    def saved_tokens(self) -> int:
        return max(0, self.original_tokens - self.final_tokens)

    @property
    def saved_pct(self) -> float:
        if self.original_tokens == 0:
            return 0.0
        return self.saved_tokens / self.original_tokens * 100


class ContextManager:
    def __init__(
        self,
        max_tokens: int = 8000,
        keep_recent_turns: int = 3,
        tool_result_max_tokens: int = 400,
    ) -> None:
        self.max_tokens = max_tokens
        self.keep_recent_turns = keep_recent_turns
        self.tool_result_max_tokens = tool_result_max_tokens

    # ---- 对外的唯一入口 -------------------------------------------------

    def build(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> tuple[str, list[dict], ContextStats]:
        """返回 (可能被追加说明的 system, 裁剪后的 messages, 统计)。

        tools 必须传进来：工具 Schema 是每轮固定开销，
        不把它计入预算，预算就是假的——实测 4 个工具就占 600 token。
        """
        tools_tokens = estimate_tools(tools)
        overhead = estimate_tokens(system) + tools_tokens
        original_tokens = overhead + estimate_messages(messages)

        msgs = copy.deepcopy(messages)
        compressed = self._compress_tool_results(msgs)
        dropped = self._trim_turns(msgs, overhead)

        final_tokens = overhead + estimate_messages(msgs)
        stats = ContextStats(
            original_tokens=original_tokens,
            final_tokens=final_tokens,
            compressed_results=compressed,
            dropped_turns=dropped,
        )

        if dropped:
            # 让模型明确知道有历史被省略，避免它以为对话只有这么长
            system = (
                f"{system}\n\n[注：更早的 {dropped} 轮对话因上下文长度限制已省略，"
                f"如需相关信息请重新询问用户]"
            )

        return system, msgs, stats

    # ---- 第一层：压缩工具结果 -------------------------------------------

    def _compress_tool_results(self, messages: list[dict]) -> int:
        count = 0
        for msg in messages:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                text = block.get("content", "")
                if not isinstance(text, str):
                    continue
                if estimate_tokens(text) > self.tool_result_max_tokens:
                    block["content"] = self._compress_text(text)
                    count += 1
        return count

    def _compress_text(self, text: str) -> str:
        """保留首尾、砍中段。

        首部通常是结论/表头，尾部往往是错误信息或汇总，
        中间才是可牺牲的明细——所以两头都保。
        """
        # 按中文字符成本保守换算字符预算
        budget_chars = int(self.tool_result_max_tokens / _TOKEN_PER_CJK)
        head_chars = int(budget_chars * 0.6)
        tail_chars = budget_chars - head_chars

        if len(text) <= head_chars + tail_chars:
            return text

        omitted = len(text) - head_chars - tail_chars
        return (
            text[:head_chars]
            + f"\n...[中间 {omitted} 个字符已省略]...\n"
            + text[-tail_chars:]
        )

    # ---- 第二层：裁剪历史轮次 -------------------------------------------

    @staticmethod
    def _turn_starts(messages: list[dict]) -> list[int]:
        """找出所有「轮次起点」的下标。

        轮次起点定义为：role=user 且 content 是纯文本的消息，
        也就是用户最初的提问。工具结果消息的 content 是列表，
        不会被误判为起点——这正是保证 tool_use/tool_result 不被拆散的关键。
        """
        return [
            i
            for i, m in enumerate(messages)
            if m.get("role") == "user" and isinstance(m.get("content"), str)
        ]

    def _trim_turns(self, messages: list[dict], overhead: int = 0) -> int:
        dropped = 0
        while True:
            starts = self._turn_starts(messages)
            # 至少要保留 keep_recent_turns 轮，且第一轮是当前提问
            if len(starts) <= self.keep_recent_turns:
                break
            if overhead + estimate_messages(messages) <= self.max_tokens:
                break

            # 从最老的一轮开始丢。切点必须在轮次起点，
            # 这样 tool_use 和它的 tool_result 会一起被移除。
            cut = starts[1]
            del messages[0:cut]
            dropped += 1

        return dropped
