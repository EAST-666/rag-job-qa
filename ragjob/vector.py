# -*- coding: utf-8 -*-
"""向量检索：fastembed + bge-small-zh-v1.5，CPU 就能跑。

为什么用 fastembed 而不是 sentence-transformers：前者底层是 onnxruntime，装完约 120MB；
后者要拉 torch，光依赖就 2GB 上下。这个项目只有 1820 条短文本，onnxruntime 足够。
"""
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from . import config

# 国内直连 huggingface 基本下不动，用镜像；模型缓存放 D 盘，避免 %TEMP% 被清理后重下
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("FASTEMBED_CACHE_PATH", r"D:\bangong\fastembed-cache")

MODEL_NAME = "BAAI/bge-small-zh-v1.5"
CACHE_PATH = config.ROOT / "data" / "vectors_bge_small_zh.npy"


class VectorIndex:
    """一键编码 + 缓存；接口形状与 BM25Index.search 保持一致，方便融合。"""

    def __init__(self, chunks: List[Dict], model_name: str = MODEL_NAME,
                 cache_path: Path = None):
        from fastembed import TextEmbedding

        self.chunks = chunks
        self.model = TextEmbedding(model_name=model_name)
        cache = Path(cache_path or CACHE_PATH)

        emb = None
        if cache.exists():
            saved = np.load(cache)
            if saved.shape[0] == len(chunks):
                emb = saved
        if emb is None:
            texts = [c["text"] for c in chunks]
            emb = np.array(list(self.model.embed(texts, batch_size=64)), dtype="float32")
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache, emb)

        # 归一化之后，内积就等于余弦相似度
        norms = np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
        self.emb = emb / norms

    def encode(self, text: str) -> np.ndarray:
        vec = np.array(list(self.model.embed([text]))[0], dtype="float32")
        return vec / max(float(np.linalg.norm(vec)), 1e-8)

    def scores(self, query: str) -> np.ndarray:
        return self.emb @ self.encode(query)

    def search(self, query: str, k: int = None, max_per_job: int = None) -> List[Tuple[float, int]]:
        """返回 [(score, chunk_index)]，同一岗位最多 max_per_job 段。"""
        k = config.TOP_K if k is None else k
        max_per_job = config.MAX_PER_JOB if max_per_job is None else max_per_job
        scores = self.scores(query)
        order = np.argsort(-scores)
        picked, per_title = [], Counter()
        for idx in order:
            if scores[idx] <= 0:
                break
            title = self.chunks[idx]["title"]
            if per_title[title] >= max_per_job:
                continue
            per_title[title] += 1
            picked.append((float(scores[idx]), int(idx)))
            if len(picked) >= k:
                break
        return picked
