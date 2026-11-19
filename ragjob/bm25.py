# -*- coding: utf-8 -*-
"""中文 BM25 检索。

为什么用 BM25 而不是向量：这版要能解释每一分从哪来，BM25 的分数是词频与逆文档频率
的显式函数，出问题能查到具体是哪个词被淹没了（见 README「已知问题」第 1 条）。
"""
import re
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from . import config


def tokenize(text: str) -> List[str]:
    """中文按两字一组切，英文数字按整词。"""
    text = text.lower()
    latin = re.findall(r"[a-z0-9+#\.]+", text)
    han = re.sub(r"[^\u4e00-\u9fff]", "", text)
    return latin + [han[i:i + 2] for i in range(len(han) - 1)]


def query_terms(query: str) -> List[str]:
    """查询词：去掉疑问虚词，去重保序。"""
    terms = [t for t in tokenize(query) if not all(ch in config.STOP_TERMS for ch in t)]
    return list(dict.fromkeys(terms))


class BM25Index:
    """建一次索引，反复查询。"""

    def __init__(self, chunks: List[Dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b

        doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.vocab: Dict[str, int] = {}
        for tokens in doc_tokens:
            for t in tokens:
                self.vocab.setdefault(t, len(self.vocab))

        n = len(chunks)
        df = np.zeros(len(self.vocab))
        for tokens in doc_tokens:
            for t in set(tokens):
                df[self.vocab[t]] += 1
        self.idf = np.log(1 + (n - df + 0.5) / (df + 0.5))

        self.avg_len = sum(len(t) for t in doc_tokens) / max(1, n)
        rows, cols, vals, lens = [], [], [], []
        for i, tokens in enumerate(doc_tokens):
            tf = Counter(tokens)
            length = len(tokens)
            lens.append(length)
            for t, c in tf.items():
                score = self.idf[self.vocab[t]] * (c * (k1 + 1)) / (
                    c + k1 * (1 - b + b * length / self.avg_len))
                rows.append(i)
                cols.append(self.vocab[t])
                vals.append(score)
        self.rows = np.array(rows)
        self.cols = np.array(cols)
        self.vals = np.array(vals)
        self.lens = np.array(lens)

        # 列索引 → 非零位置，查询时只扫相关列，不用遍历整个稀疏矩阵
        self.by_col: Dict[int, np.ndarray] = {}
        order = np.argsort(self.cols, kind="stable")
        sorted_cols = self.cols[order]
        bounds = np.searchsorted(sorted_cols, np.arange(len(self.vocab) + 1))
        for col in range(len(self.vocab)):
            lo, hi = bounds[col], bounds[col + 1]
            if hi > lo:
                self.by_col[col] = order[lo:hi]

    def scores(self, query: str) -> Tuple[np.ndarray, List[str]]:
        scores = np.zeros(len(self.chunks))
        terms = [t for t in query_terms(query) if t in self.vocab]
        for t in terms:
            idx = self.by_col.get(self.vocab[t])
            if idx is not None:
                np.add.at(scores, self.rows[idx], self.vals[idx])
        return scores, terms

    def search(self, query: str, k: int = None, max_per_job: int = None) -> List[Tuple[float, int]]:
        """返回 [(score, chunk_index)]，同一岗位最多 max_per_job 段。"""
        k = config.TOP_K if k is None else k
        max_per_job = config.MAX_PER_JOB if max_per_job is None else max_per_job
        scores, _ = self.scores(query)
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

    def search_with_stats_boost(self, query: str, k: int = None,
                                max_per_job: int = None) -> List[Tuple[float, int]]:
        """统计类问题优先给汇总文档名额。

        原因：高频词（Python、Agent）在汇总行里只出现一次，IDF 接近 0，
        靠 BM25 打分永远排不上来 —— 只能按「条目名」直接对词。
        """
        k = config.TOP_K if k is None else k
        if not any(w in query for w in config.AGGREGATE_HINTS):
            return self.search(query, k, max_per_job)

        wide = self.search(query, max(k * 5, 25), max_per_job)
        q_terms = set(query_terms(query))
        stats = []
        others = []
        for score, idx in wide:
            chunk = self.chunks[idx]
            if chunk["src"] == "统计":
                name_terms = set(tokenize(chunk["title"].split("·")[-1]))
                stats.append((len(q_terms & name_terms), score, idx))
            else:
                others.append((score, idx))
        stats.sort(reverse=True)
        picked = [(s, i) for _, s, i in stats[:3]] + others
        return picked[:k]
