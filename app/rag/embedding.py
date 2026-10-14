"""Embedding 模型封装。

为什么用本地模型而不是调用 API：
DeepSeek 没有提供 embedding 接口（实测 /embeddings 返回 404），
所以只能本地跑。选的 bge-small-zh-v1.5 是中文小模型，
约 95MB，CPU 上编码速度够用，效果在中文检索任务上明显好于
多语言大模型的同尺寸版本。

模型延迟加载：首次调用 get_embedding_model() 才下载权重，
避免 import 时就卡住（进程启动、单元测试都不该触发下载）。
"""

import os
from typing import Any

import numpy as np

#: 中文检索效果与体积的平衡点。换 bge-base-zh-v1.5 精度更高但慢约 3 倍。
DEFAULT_MODEL = "BAAI/bge-small-zh-v1.5"

#: 国内访问 huggingface.co 不通，默认走镜像
DEFAULT_ENDPOINT = "https://hf-mirror.com"


class EmbeddingModel:
    """薄封装，屏蔽 sentence-transformers 的细节。

    对外只暴露 encode()，且保证返回的是 float32 的 (N, D) 数组。
    """

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._dim: int | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        # 没显式配过 endpoint 就默认用镜像，否则国内下载会直接超时
        os.environ.setdefault("HF_ENDPOINT", DEFAULT_ENDPOINT)

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        # 从模型自身读出维度，避免把 512 之类的值硬编码错
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        self._ensure_loaded()
        assert self._dim is not None
        return self._dim

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def encode(
        self,
        texts: list[str] | str,
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """把文本编码成向量，形状 (N, D)，float32。"""
        self._ensure_loaded()
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)

        vectors = self._model.encode(
            items,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=False,  # 归一到 VectorStore 里统一做
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        return vectors.reshape(1, -1) if single else vectors

    def encode_query(self, query: str) -> np.ndarray:
        """编码检索问题。

        bge 系列在检索任务上要求给「查询」加指令前缀，
        而「文档」不加——这是该模型的训练约定，漏掉会明显损失召回率。
        """
        prefixed = f"为这个句子生成表示以用于检索相关文章：{query}"
        return self.encode(prefixed)


_model_cache: dict[str, EmbeddingModel] = {}


def get_embedding_model(model_name: str = DEFAULT_MODEL) -> EmbeddingModel:
    """按模型名缓存，避免重复加载权重。"""
    if model_name not in _model_cache:
        _model_cache[model_name] = EmbeddingModel(model_name)
    return _model_cache[model_name]
