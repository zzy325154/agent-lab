"""RAG 检索效果测量。

核心对比是「词汇检索 vs 语义检索」，用同一组测试问题打两路：

  A. 关键词硬匹配（改造前的做法）：问题里必须出现文档原词才能命中
  B. 混合检索（向量 + BM25 + RRF）：支持口语化提问

之所以要设计这种对比，是因为「检索变准了」本身不可证伪——
必须构造一组「不含文档原词、但人类知道该命中哪条」的问题，
让旧方案暴露失败，新方案的成功才有说服力。

运行：python tests/measure_rag.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.rag.chunking import chunk_documents
from app.rag.corpus import POLICY_DOCUMENTS
from app.rag.retriever import HybridRetriever

# (提问, 期望命中的章节关键词)。
# 这些提问刻意不含文档原词，模拟真实用户的口语化问法。
CASES: list[tuple[str, str]] = [
    ("出差住宿一晚上最多能报多少钱", "住宿费"),
    ("报销单最晚什么时候交", "30 个自然日"),
    ("超过多少钱要总监签字", "20000"),
    ("年假能攒到明年吗", "结转"),
    ("坐飞机出差有什么限制", "经济舱"),
    ("自己开车去出差怎么算钱", "自驾"),
    ("休年假期间扣不扣工资", "工资照常发放"),
    ("发票要原件还是复印件", "增值税发票原件"),
    ("什么情况不给报", "个人消费"),
    ("出差每天吃饭补多少", "餐补"),
    ("离职时没休完的假怎么办", "折算工资"),
    ("去国外出差补贴多少", "美元"),
]


def legacy_keyword_search(query: str) -> list[str]:
    """复现改造前的做法：关键词直接匹配文档标题。

    这是当时 search_policy 的真实实现——只要提问里出现
    「报销」「年假」「出差」就命中文档，否则完全找不到。
    """
    for key in POLICY_DOCUMENTS:
        if key in query:
            return [key]
    return []


def main() -> int:
    print("构建混合检索索引（首次会下载 embedding 模型）...")
    t0 = time.perf_counter()
    chunks = chunk_documents(POLICY_DOCUMENTS)
    retriever = HybridRetriever(chunks)
    retriever.build()
    build_s = time.perf_counter() - t0

    print(f"索引完成：{len(chunks)} 个块，耗时 {build_s:.1f}s")
    print(f"（模型：{retriever.embedding.model_name}，维度 {retriever.embedding.dim}）")
    print("=" * 72)

    legacy_hits = 0
    hybrid_hits = 0
    legacy_miss: list[str] = []
    hybrid_miss: list[str] = []

    print(f"{'提问':<26}{'旧(词汇)':<12}{'新(混合)':<12}{'命中来源'}")
    print("-" * 72)

    for query, expected in CASES:
        # 旧方案：只能返回整个文档，检查该文档是否含期望内容
        old_docs = legacy_keyword_search(query)
        old_ok = any(expected in POLICY_DOCUMENTS[d] for d in old_docs)

        # 新方案：看 top-3 里是否命中期望内容
        results = retriever.retrieve(query, top_k=3)
        new_ok = any(expected in r.chunk.text for r in results)
        sources = ",".join(r.source for r in results[:3])

        legacy_hits += old_ok
        hybrid_hits += new_ok
        if not old_ok:
            legacy_miss.append(query)
        if not new_ok:
            hybrid_miss.append(query)

        mark = lambda ok: "命中" if ok else "未命中"  # noqa: E731
        print(f"{query[:24]:<26}{mark(old_ok):<12}{mark(new_ok):<12}{sources}")

    n = len(CASES)
    print("-" * 72)
    print()
    print(f"命中率：旧(词汇匹配) {legacy_hits}/{n} = {legacy_hits/n*100:.1f}%"
          f"  ->  新(混合检索) {hybrid_hits}/{n} = {hybrid_hits/n*100:.1f}%")
    print(f"提升 {(hybrid_hits-legacy_hits)/n*100:.1f} 个百分点")

    if legacy_miss:
        print(f"\n旧方案失败但新方案成功的提问（共 {n - legacy_hits} 个）：")
        for q in legacy_miss:
            new_ok = q not in hybrid_miss
            print(f"  {'[新方案救回]' if new_ok else '[仍未命中]'} {q}")

    if hybrid_miss:
        print(f"\n新方案仍未命中（共 {len(hybrid_miss)} 个，需排查）：")
        for q in hybrid_miss:
            print(f"  {q}")
            print(retriever.explain(q, top_k=2))

    # 检索延迟
    latencies = []
    for query, _ in CASES:
        t = time.perf_counter()
        retriever.retrieve(query, top_k=3)
        latencies.append((time.perf_counter() - t) * 1000)
    latencies.sort()
    print(f"\n检索延迟：平均 {sum(latencies)/len(latencies):.1f}ms，"
          f"P50 {latencies[len(latencies)//2]:.1f}ms，"
          f"最大 {latencies[-1]:.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
