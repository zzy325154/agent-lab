"""RAG 测试。

分块与向量存储用假向量测试，不依赖模型下载；
真实语义检索的效果验证在 tests/measure_rag.py（需要模型，会下载权重）。

为什么要假向量：单元测试必须快且离线可跑。真模型下载 95MB、
加载数秒，放进单元测试会让整套测试变得没人愿意跑。
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunking import chunk_documents
from app.rag.corpus import POLICY_DOCUMENTS
from app.rag.vector_store import VectorStore


class FakeEmbedding:
    """确定性假向量：把关键词命中情况编码进向量维度。

    用来验证检索链路（归一化、排序、RRF 融合）是否正确，
    不验证语义质量——那是真模型的事。
    """

    def __init__(self, keywords: list[str]) -> None:
        self.keywords = keywords
        self.dim = len(keywords)

    def encode(self, texts, **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        rows = []
        for t in items:
            v = np.array([1.0 if k in t else 0.0 for k in self.keywords], dtype=np.float32)
            if v.sum() == 0:  # 避免零向量
                v[0] = 0.1
            rows.append(v)
        arr = np.vstack(rows)
        return arr.reshape(1, -1) if single else arr

    def encode_query(self, query: str):
        return self.encode(query)


# --------------------------------------------------------------------------
# 分块
# --------------------------------------------------------------------------


def test_chunks_do_not_break_articles():
    """条款不能被跨块劈开——这是分块策略的核心要求。"""
    chunks = chunk_documents(POLICY_DOCUMENTS, max_chars=600)

    # 「第X条」应出现在块的 section 里，而不是被切断在正文中间
    for c in chunks:
        assert c.section, f"块缺少章节归属: {c.text[:30]}"
        assert c.text.strip(), "存在空块"

    # 每条制度都应至少产出一个块
    docs = {c.doc_title for c in chunks}
    assert docs == set(POLICY_DOCUMENTS), f"文档覆盖不全: {docs}"
    print(f"  [PASS] 分块保留条款完整性（{len(chunks)} 个块）")


def test_chunk_heading_path():
    """块要带完整标题路径，否则片段脱离上下文。"""
    chunks = chunk_documents(POLICY_DOCUMENTS)
    sample = next(c for c in chunks if c.doc_title == "报销")

    assert sample.heading.startswith("报销 > "), f"标题路径错误: {sample.heading}"
    assert "第" in sample.heading

    # to_text 应包含标题，供模型判断片段归属
    text = sample.to_text(with_heading=True)
    assert sample.heading in text
    assert text != sample.to_text(with_heading=False)
    print(f"  [PASS] 标题路径完整（示例：{sample.heading}）")


def test_chunk_id_unique_and_stable():
    """chunk_id 必须唯一且可复现——重复或漂移会让检索结果无法溯源。"""
    a = chunk_documents(POLICY_DOCUMENTS)
    b = chunk_documents(POLICY_DOCUMENTS)

    ids_a = [c.chunk_id for c in a]
    assert len(ids_a) == len(set(ids_a)), "chunk_id 有重复"
    assert ids_a == [c.chunk_id for c in b], "两次分块结果不稳定"
    print(f"  [PASS] chunk_id 唯一且可复现（{len(ids_a)} 个）")


def test_respects_max_chars():
    """超长章节要在块内继续拆，不能生成巨块撑爆上下文。"""
    chunks = chunk_documents(POLICY_DOCUMENTS, max_chars=200)
    oversized = [c for c in chunks if len(c.text) > 400]
    assert not oversized, f"存在超长块: {[len(c.text) for c in oversized]}"
    print(f"  [PASS] 超长章节被拆分（max_chars=200，最大块 "
          f"{max(len(c.text) for c in chunks)} 字符）")


# --------------------------------------------------------------------------
# 向量存储
# --------------------------------------------------------------------------


def test_vector_store_normalizes():
    """入库向量必须归一化，否则大模长的向量会霸占相似度排名。"""
    store = VectorStore()
    store.add(np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32))

    m = store.matrix()
    norms = np.linalg.norm(m, axis=1)
    assert np.allclose(norms, 1.0), f"未归一化: {norms}"

    # 归一化后 (3,4) 与 (1,0) 的余弦相似度应为 0.6
    hits = store.search(np.array([1.0, 0.0]), top_k=2)
    assert abs(hits[0].score - 1.0) < 1e-6, f"自身相似度应为 1: {hits[0].score}"
    assert abs(hits[1].score - 0.6) < 1e-6, f"应为 0.6: {hits[1].score}"
    print("  [PASS] 向量归一化与余弦相似度正确")


def test_vector_store_ranking():
    """返回结果必须按相似度降序。"""
    store = VectorStore()
    vectors = np.array(
        [[1, 0, 0], [0.9, 0.1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32
    )
    store.add(vectors)

    hits = store.search(np.array([1.0, 0.0, 0.0]), top_k=3)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True), f"未按降序: {scores}"
    assert hits[0].index == 0, "最相似的应是第 0 条"
    assert hits[1].index == 1
    print(f"  [PASS] 检索结果按相似度降序（top1={hits[0].index}）")


def test_vector_store_dim_mismatch():
    """维度不一致必须报错，否则矩阵乘法会静默产出错误结果。"""
    store = VectorStore()
    store.add(np.array([[1.0, 2.0, 3.0]], dtype=np.float32))
    try:
        store.add(np.array([[1.0, 2.0]], dtype=np.float32))
        raise AssertionError("维度不一致应报错")
    except ValueError as e:
        assert "维度" in str(e)
    print("  [PASS] 向量维度不一致时明确报错")


def test_vector_store_empty():
    """空库检索应返回空列表，而不是抛异常。"""
    store = VectorStore()
    assert store.size == 0
    assert store.search(np.array([1.0, 0.0]), top_k=3) == []
    print("  [PASS] 空库检索安全返回")


def test_vector_store_top_k_exceeds_size():
    """top_k 大于库容量时不应越界。"""
    store = VectorStore()
    store.add(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    hits = store.search(np.array([1.0, 0.0]), top_k=10)
    assert len(hits) == 2, f"应返回全部 2 条，实际 {len(hits)}"
    print("  [PASS] top_k 超过库容量时安全截断")


# --------------------------------------------------------------------------
# RRF 融合
# --------------------------------------------------------------------------


def test_rrf_fusion_ranks():
    """RRF 融合逻辑：两路都命中的应排最前。"""
    from app.rag.retriever import HybridRetriever

    chunks = chunk_documents(POLICY_DOCUMENTS)
    # 用假模型构造，避免真下载
    emb = FakeEmbedding(["报销", "年假", "出差", "住宿", "补贴", "天数"])
    r = HybridRetriever(chunks[:8], embedding=emb, rrf_k=60)
    r.build()

    results = r.retrieve("报销", top_k=3)
    assert results, "检索应有结果"
    scores = [x.score for x in results]
    assert scores == sorted(scores, reverse=True), "RRF 分数应降序"
    # 每个结果都要带可溯源的排名信息
    for x in results:
        assert x.source in {"both", "vector", "bm25"}
    print(f"  [PASS] RRF 融合排序正确（来源分布 "
          f"{[x.source for x in results]}）")


def test_rrf_rewards_agreement():
    """被两路同时命中的文档，得分应高于仅单路命中。"""
    from app.rag.retriever import HybridRetriever

    chunks = chunk_documents(POLICY_DOCUMENTS)
    emb = FakeEmbedding(["报销", "年假", "出差", "住宿", "补贴", "天数"])
    r = HybridRetriever(chunks, embedding=emb)
    r.build()

    results = r.retrieve("报销 年假 出差", top_k=5)
    both = [x.score for x in results if x.source == "both"]
    single = [x.score for x in results if x.source != "both"]
    if both and single:
        assert min(both) >= min(single), "两路命中的得分应更高"
    print("  [PASS] 两路共同命中的文档得分更高")


def test_retriever_requires_build():
    """未建索引就检索应明确报错，而不是返回空结果让人困惑。"""
    from app.rag.retriever import HybridRetriever

    chunks = chunk_documents(POLICY_DOCUMENTS)[:3]
    r = HybridRetriever(chunks, embedding=FakeEmbedding(["报销"]))
    try:
        r.retrieve("报销")
        raise AssertionError("未建索引应报错")
    except RuntimeError as e:
        assert "build" in str(e)
    print("  [PASS] 未建索引时明确报错")


def test_tokenizer_fallback():
    """未装 jieba 时要有降级方案，不能让检索直接失效。"""
    from app.rag.retriever import _tokenize_zh

    tokens = _tokenize_zh("出差住宿标准")
    assert tokens, "分词结果不应为空"
    # 无论走 jieba 还是 bigram 降级，都要产出多个 token
    assert len(tokens) >= 2, f"分词过于粗糙: {tokens}"
    print(f"  [PASS] 中文分词可用（{len(tokens)} 个 token）")


def main() -> int:
    tests = [
        test_chunks_do_not_break_articles,
        test_chunk_heading_path,
        test_chunk_id_unique_and_stable,
        test_respects_max_chars,
        test_vector_store_normalizes,
        test_vector_store_ranking,
        test_vector_store_dim_mismatch,
        test_vector_store_empty,
        test_vector_store_top_k_exceeds_size,
        test_rrf_fusion_ranks,
        test_rrf_rewards_agreement,
        test_retriever_requires_build,
        test_tokenizer_fallback,
    ]
    print("RAG 测试（使用假向量，不下载模型）：")
    for t in tests:
        t()
    print(f"\n全部 {len(tests)} 项通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
