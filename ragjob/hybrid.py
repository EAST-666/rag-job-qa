# -*- coding: utf-8 -*-
"""混合召回：BM25 与向量两路结果用加权 RRF 合并。

为什么是 RRF 而不是把分数相加：BM25 的分数量级是十几到几十，余弦是 0-1，
直接相加等于让 BM25 单方面决定排序；而加权求和要先调权重，20 条样本上
「调出来好看」的权重换一批数据就失效。RRF 只看名次，不用调参。

为什么权重是 2:1 而不是 1:1：两路在原题上表现接近（BM25 召回略高），
但在改写题上向量明显更强；给 BM25 略高的权重，是为了在关键词型问题上
不把它原本第 1 的结果压到第 3（实测 1:1 时 MRR 从 0.72 掉到 0.67）。
"""
from collections import Counter
from typing import Dict, List, Sequence, Tuple

from . import config
from .bm25 import tokenize
from .bm25 import tokenize


def rrf_fuse(rank_lists: Sequence[List[Tuple[float, int]]],
             weights: Sequence[float] = None,
             k: int = 60, top_n: int = 5,
             max_per_job: int = None, chunks: List[Dict] = None) -> List[Tuple[float, int]]:
    """把多路 [(score, idx)] 融合成一路；weights 缺省时等权。"""
    if weights is None:
        weights = [1.0] * len(rank_lists)

    agg: Dict[int, float] = {}
    for weight, ranked in zip(weights, rank_lists):
        for rank, (_score, idx) in enumerate(ranked, start=1):
            agg[idx] = agg.get(idx, 0.0) + weight * (1.0 / (k + rank))

    fused = sorted(agg.items(), key=lambda kv: -kv[1])
    if max_per_job is None or chunks is None:
        return [(score, idx) for idx, score in fused[:top_n]]

    picked, per_title = [], Counter()
    for idx, score in fused:
        title = chunks[idx]["title"]
        if per_title[title] >= max_per_job:
            continue
        per_title[title] += 1
        picked.append((score, idx))
        if len(picked) >= top_n:
            break
    return picked


class HybridRetriever:
    """对外只暴露 search，内部两路各取一批再融合。"""

    def __init__(self, bm25, vector, k_rrf: int = 60,
                 recall_multiplier: int = 4, bm25_weight: float = 2.0, vector_weight: float = 1.0):
        self.bm25 = bm25
        self.vector = vector
        self.k_rrf = k_rrf
        self.recall_multiplier = recall_multiplier
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

    def search(self, query: str, k: int = None, max_per_job: int = None) -> List[Tuple[float, int]]:
        from . import config

        k = config.TOP_K if k is None else k
        max_per_job = config.MAX_PER_JOB if max_per_job is None else max_per_job
        wide = max(k * self.recall_multiplier, 20)

        # 统计类问题靠词频排不上来（高频词 IDF 接近 0），BM25 这一路用带 boost 的版本
        bm25_hits = self.bm25.search_with_stats_boost(query, wide, max_per_job)
        vec_hits = self.vector.search(query, wide, max_per_job)
        fused = rrf_fuse([bm25_hits, vec_hits],
                         weights=[self.bm25_weight, self.vector_weight],
                         k=self.k_rrf, top_n=k,
                         max_per_job=max_per_job, chunks=self.bm25.chunks)

        # 统计类问题（多少条/几条/占比…）：向量那一路容易把相近的 JD 段顶上来，
        # 把统计段挤下去。这里按查询类型做一次路由：命中的统计段强制排到最前。
        if config.AGGREGATE_ROUTING and any(h in query for h in config.AGGREGATE_HINTS):
            stats = [(sc, idx) for sc, idx in fused if self.bm25.chunks[idx].get("src") == "统计"]
            if stats:
                # 多个统计段命中时，挑与问题词面重合最多的那条，而不是随便第一条
                q_terms = set(tokenize(query))
                def overlap(pair):
                    idx = pair[1]
                    c = self.bm25.chunks[idx]
                    return len(q_terms & set(tokenize(c.get("title", "") + c.get("text", "")[:80])))
                best = max(stats, key=overlap)
                rest = [(sc, idx) for sc, idx in fused if idx != best[1]]
                fused = ([best] + rest)[:k]
        return fused



