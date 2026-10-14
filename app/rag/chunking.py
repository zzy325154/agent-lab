"""文档分块。

分块策略是 RAG 里最容易做错、又最影响效果的一环。
固定字数切分会把一条完整的条款劈成两半，「单笔金额超过 5000 元」
和「需额外经部门负责人审批」被拆到不同块里，检索出来就是残缺信息。

这里的做法是**按文档自身的语义结构切**，而不是按字数切：

  第X条 (条款标题)          ← 一个 chunk 的边界
    1.1 具体规定……          ← 行内序号，保持在同一条内
    1.2 具体规定……

每个 chunk 都带上所属章节标题（「报销 > 第二条 报销流程」），
这样即使检索到的只是片段，模型也能知道它属于哪一条。
"""

import re
from dataclasses import dataclass, field

# 「第一条」「第三条」这类中文条款标题
_ARTICLE_RE = re.compile(r"^第[一二三四五六七八九十百]+条\s*(.*)$")
# 「1.1」「2.3.1」这类行内序号
_NUMBERED_RE = re.compile(r"^\d+(\.\d+)*[\s、.]")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    doc_title: str
    section: str          # 所属章节，如「第二条 报销流程」
    metadata: dict = field(default_factory=dict)

    @property
    def heading(self) -> str:
        """用于展示与检索上下文的全路径标题。"""
        if self.section:
            return f"{self.doc_title} > {self.section}"
        return self.doc_title

    def to_text(self, with_heading: bool = True) -> str:
        """给模型看的文本。带上标题路径，避免片段失去上下文。"""
        if with_heading:
            return f"【{self.heading}】\n{self.text}"
        return self.text


def chunk_documents(
    documents: dict[str, str],
    max_chars: int = 600,
) -> list[Chunk]:
    """把 {文档标题: 全文} 切成带章节归属的块。

    Args:
        documents: 文档标题 -> 正文（正文按行给出，条款以「第X条」开头）
        max_chars: 单块字符上限，超过则在同一章节内继续拆分
    """
    chunks: list[Chunk] = []
    for doc_title, body in documents.items():
        chunks.extend(_chunk_one(doc_title, body, max_chars))
    return chunks


def _chunk_one(doc_title: str, body: str, max_chars: int) -> list[Chunk]:
    lines = [ln.rstrip() for ln in body.split("\n") if ln.strip()]

    # 先按「第X条」把文档切成章节，章节标题独立成行
    sections: list[tuple[str, list[str]]] = []
    current_title = ""
    current_lines: list[str] = []

    for line in lines:
        m = _ARTICLE_RE.match(line)
        if m:
            if current_lines or current_title:
                sections.append((current_title, current_lines))
            current_title = line.strip()
            current_lines = []
        else:
            current_lines.append(line.strip())
    if current_lines or current_title:
        sections.append((current_title, current_lines))

    chunks: list[Chunk] = []
    index = 0

    for section_title, section_lines in sections:
        if not section_lines:
            # 只有标题没有内容，仍然保留，否则这条制度就消失了
            text = section_title
            chunks.append(
                Chunk(
                    chunk_id=f"{_slug(doc_title)}-{index}",
                    text=text,
                    doc_title=doc_title,
                    section=section_title,
                )
            )
            index += 1
            continue

        # 行内序号是天然的次级边界：1.1 / 1.2 ……
        # 尽量把连续的序号行放在同一块里，只有超长才拆
        buffer: list[str] = []
        size = 0
        for line in section_lines:
            if buffer and size + len(line) + 1 > max_chars:
                chunks.append(
                    Chunk(
                        chunk_id=f"{_slug(doc_title)}-{index}",
                        text="\n".join(buffer),
                        doc_title=doc_title,
                        section=section_title,
                    )
                )
                index += 1
                buffer, size = [], 0
            buffer.append(line)
            size += len(line) + 1

        if buffer:
            chunks.append(
                Chunk(
                    chunk_id=f"{_slug(doc_title)}-{index}",
                    text="\n".join(buffer),
                    doc_title=doc_title,
                    section=section_title,
                )
            )
            index += 1

    return chunks


def _slug(title: str) -> str:
    """把中文标题转成可读的 ASCII 短标识，用于 chunk_id 可读性。"""
    mapping = {"报销": "reimbursement", "年假": "annual-leave", "出差": "business-trip"}
    return mapping.get(title, f"doc{abs(hash(title)) % 10000}")
