"""RAG 模块：把关键词硬匹配的假检索，换成真正的检索增强生成。

检索链路：

    原始文档 ──分块──> chunks ──┬── 向量检索（语义相似）
                                └── BM25 检索（关键词精确匹配）
                                          │
                                     RRF 融合排序
                                          │
                                       Top-K 结果

为什么必须混合检索：纯向量检索对「1.3」「5000 元」这类
数字和专有名词不敏感，而这恰是制度问答的关键；
纯 BM25 又抓不住「出差花钱怎么报」这种口语化提问。
两者互补，用 RRF 融合比手工调权重更稳。
"""

from app.rag.chunking import Chunk, chunk_documents
from app.rag.embedding import EmbeddingModel, get_embedding_model
from app.rag.retriever import HybridRetriever, RetrievalResult
from app.rag.vector_store import VectorStore

__all__ = [
    "Chunk",
    "chunk_documents",
    "EmbeddingModel",
    "get_embedding_model",
    "HybridRetriever",
    "RetrievalResult",
    "VectorStore",
]
