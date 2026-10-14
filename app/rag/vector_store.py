"""向量存储与相似度检索。

刻意用 numpy 手写而不是引入 FAISS / Chroma：
文档量在千级以内时，一次矩阵乘法（N×D 与 D 做内积）耗时不到 1ms，
引入一个重量级依赖换不来实际收益。

关键实现细节是**把归一化后的向量堆成一个矩阵常驻内存**，
查询时只需一次矩阵乘法，而不是在 Python 里循环算 N 次余弦相似度——
后者在文档量上来后会慢两个数量级。
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SearchHit:
    index: int
    score: float


class VectorStore:
    def __init__(self, dim: int | None = None) -> None:
        self._matrix: np.ndarray | None = None   # (N, D)，已 L2 归一化
        self._dim = dim

    @property
    def size(self) -> int:
        return 0 if self._matrix is None else self._matrix.shape[0]

    def add(self, vectors: np.ndarray) -> None:
        """加入一批向量。若已有数据则追加。

        入参会在内部做 L2 归一化，归一化后余弦相似度等价于点积，
        查询时就不必再除以模长。
        """
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        normalized = _l2_normalize(vectors)
        if self._matrix is None:
            self._matrix = normalized
            self._dim = normalized.shape[1]
        else:
            if normalized.shape[1] != self._dim:
                raise ValueError(
                    f"向量维度不一致：已有 {self._dim}，新来 {normalized.shape[1]}"
                )
            self._matrix = np.vstack([self._matrix, normalized])

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[SearchHit]:
        """返回相似度最高的 top_k 个下标。"""
        if self._matrix is None or self.size == 0:
            return []

        q = _l2_normalize(query_vector.reshape(1, -1))
        # 一次矩阵乘法算出所有相似度，比 Python 循环快得多
        scores = (self._matrix @ q.T).ravel()

        k = min(top_k, self.size)
        # argpartition 只做部分排序，比全量 argsort 快
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [SearchHit(index=int(i), score=float(scores[i])) for i in top_idx]

    def matrix(self) -> np.ndarray | None:
        """只读访问底层矩阵，供可视化等场景使用。"""
        return None if self._matrix is None else self._matrix.copy()


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # 零向量会导致除零，用 where 兜住
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)
