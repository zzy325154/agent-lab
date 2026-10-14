"""混合检索：向量语义召回 + BM25 关键词召回，用 RRF 融合。

为什么不用单一检索：
  - 纯向量：对「1.3」「5000 元」「增值税发票」这类关键词不敏感，
    而制度问答恰恰要求数字和条款号不能错。
  - 纯 BM25：抓不住口语化提问。用户问「出差花钱怎么报」，
    文档里写的是「差旅费报销标准」，没有字面重叠，BM25 直接漏掉。

RRF（Reciprocal Rank Fusion）用排名倒数融合两路结果：
    score(d) = Σ 1 / (k + rank_i(d))
它不需要两路分数可比（余弦相似度与 BM25 分数量纲完全不同），
因此比手工调权重更稳，也是工业界的常见默认选择。
"""

from dataclasses import dataclass
from typing import Any

from app.rag.chunking import Chunk
from app.rag.embedding import EmbeddingModel
from app.rag.vector_store import VectorStore


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None

    @property
    def source(self) -> str:
        """命中的来源，便于排查「为什么召回了这条」。"""
        if self.vector_rank is not None and self.bm25_rank is not None:
            return "both"
        return "vector" if self.vector_rank is not None else "bm25"


def _tokenize_zh(text: str) -> list[str]:
    """中文分词。

    BM25 依赖词边界，而中文没有空格分隔，因此必须分词。
    jieba 未安装时退化为字符 bigram——不理想，但不会让检索直接失效。
    """
    try:
        import jieba

        return [t for t in jieba.lcut(text) if t.strip()]
    except ImportError:
        # 退化为字符二元组：中文里相邻字组成的词最稳定
        return [text[i : i + 2] for i in range(len(text) - 1)] or [text]


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedding: EmbeddingModel | None = None,
        rrf_k: int = 60,
    ) -> None:
        if not chunks:
            raise ValueError("检索器需要至少一个 chunk")

        self.chunks = chunks
        self.embedding = embedding or EmbeddingModel()
        self.rrf_k = rrf_k
        self._store = VectorStore()
        self._bm25: Any = None
        self._tokenized: list[list[str]] = []

    # ---- 建索引 --------------------------------------------------------

    def build(self, show_progress: bool = False) -> None:
        """编码所有 chunk 并建立向量索引与 BM25 索引。"""
        # 编码时带上标题路径，让「第二条 报销流程」这类信息也进入向量
        texts = [c.to_text(with_heading=True) for c in self.chunks]
        vectors = self.embedding.encode(texts, show_progress=show_progress)
        self._store.add(vectors)

        self._tokenized = [_tokenize_zh(t) for t in texts]
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self._tokenized)
        except ImportError:
            self._bm25 = None

    @property
    def is_built(self) -> bool:
        return self._store.size > 0

    # ---- 检索 ----------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3, pool: int = 10) -> list[RetrievalResult]:
        """混合检索，返回融合后的 top_k。"""
        if not self.is_built:
            raise RuntimeError("请先调用 build() 建立索引")

        # 两路各自多召回一些，再融合，避免某一路的正确答案在融合前就被截掉
        pool = max(pool, top_k)

        vector_ranks = self._vector_search(query, pool)
        bm25_ranks = self._bm25_search(query, pool)

        # RRF 融合
        scores: dict[int, float] = {}
        for idx, rank in vector_ranks.items():
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)
        for idx, rank in bm25_ranks.items():
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            RetrievalResult(
                chunk=self.chunks[idx],
                score=round(score, 6),
                vector_rank=vector_ranks.get(idx),
                bm25_rank=bm25_ranks.get(idx),
            )
            for idx, score in ordered
        ]

    def _vector_search(self, query: str, pool: int) -> dict[int, int]:
        """返回 {chunk 下标: 排名(从 1 开始)}。"""
        qv = self.embedding.encode_query(query)
        hits = self._store.search(qv, top_k=pool)
        return {h.index: rank for rank, h in enumerate(hits, 1)}

    def _bm25_search(self, query: str, pool: int) -> dict[int, int]:
        if self._bm25 is None:
            return {}
        tokens = _tokenize_zh(query)
        scores = self._bm25.get_scores(tokens)

        import numpy as np

        # 只保留有实际命中的（分数 > 0），否则会把一堆 0 分文档排进来
        nonzero = [i for i, s in enumerate(scores) if s > 0]
        if not nonzero:
            return {}
        ordered = sorted(nonzero, key=lambda i: -scores[i])[:pool]
        return {idx: rank for rank, idx in enumerate(ordered, 1)}

    # ---- 调试辅助 ------------------------------------------------------

    def explain(self, query: str, top_k: int = 3) -> str:
        """返回可读的检索过程，用于排查召回问题。"""
        results = self.retrieve(query, top_k=top_k)
        lines = [f"查询：{query}", f"（RRF k={self.rrf_k}）", ""]
        for i, r in enumerate(results, 1):
            lines.append(
                f"{i}. score={r.score:.5f} 来源={r.source} "
                f"向量排名={r.vector_rank} BM25排名={r.bm25_rank}"
            )
            lines.append(f"   {r.chunk.heading}")
            lines.append(f"   {r.chunk.text[:70]}...")
        return "\n".join(lines)


def build_retriever(
    documents: dict[str, str],
    embedding: EmbeddingModel | None = None,
    max_chars: int = 600,
    show_progress: bool = False,
) -> HybridRetriever:
    """从原始文档一步构建检索器。"""
    from app.rag.chunking import chunk_documents

    chunks = chunk_documents(documents, max_chars=max_chars)
    retriever = HybridRetriever(chunks, embedding=embedding)
    retriever.build(show_progress=show_progress)
    return retriever
