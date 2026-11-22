# -*- coding: utf-8 -*-
"""召回评测：同一批问题，对比 BM25 / 向量 / RRF 混合三种召回的 Recall@5 与 MRR。

两组题分开跑，因为它们考的东西不一样：
- 原题（关键词型）：题目用词和 JD 原文高度重合，BM25 天生占便宜；
- 改写题（同义型）：题目换成口语说法，JD 里那几个词一个都不出现，考的是语义匹配。

gold 怎么定：A 类问题的答案都落在「岗位画像统计汇总」那组段里，
所以每题标注一条「应该被召回」的统计段标题；提到举例的题，额外允许岗位段命中。
"""
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ragjob import config
from ragjob.bm25 import BM25Index
from ragjob.corpus import build_chunks
from ragjob.hybrid import HybridRetriever
from ragjob.vector import VectorIndex

# 原题（A 类关键词型）：题号 → 应该被召回的统计段标题关键字
GOLD_TITLES: Dict[int, List[str]] = {
    1: ["1. 出现频率最高的五项技能", "需求分析 / 沟通", "算法 / 模型训练",
        "Python", "Agent / 智能体", "知识库 / RAG"],
    2: ["要写代码（型 B"],
    3: ["Coze / 扣子"],
    4: ["模型评测 / badcase"],
    5: ["Python"],
    6: ["明确为实习岗"],
    7: ["提到硕士"],
    8: ["样本"],
    9: ["知识库 / RAG"],
    10: ["非 Coding（型 A"],
}

# 原题里要求「举一个具体岗位」的，岗位段命中也算对
EXTRA_JOB_KEYWORDS: Dict[int, List[str]] = {9: ["知识库", "RAG"]}


def gold_indices(chunks, titles: List[str], job_keywords: List[str]) -> set:
    gold = set()
    for key in titles:
        for i, c in enumerate(chunks):
            if c.get("src") == "统计" and key in c.get("title", ""):
                gold.add(i)
    for kw in job_keywords:
        for i, c in enumerate(chunks):
            if c.get("src") != "统计" and kw in c.get("text", ""):
                gold.add(i)
    return gold


def recall_at_k(hits, gold, k: int) -> int:
    return 1 if any(idx in gold for _s, idx in hits[:k]) else 0


def mrr(hits, gold) -> float:
    for rank, (_s, idx) in enumerate(hits, start=1):
        if idx in gold:
            return 1.0 / rank
    return 0.0


def run_set(name: str, items, chunks, modes) -> List[dict]:
    rows = []
    for mode_name, fn in modes.items():
        hit_sum, mrr_sum, detail = 0, 0.0, []
        for item in items:
            hits = fn(item["question"])
            gold = gold_indices(chunks, item["gold_titles"], item["gold_job_keywords"])
            r5 = recall_at_k(hits, gold, 5)
            m = mrr(hits, gold)
            hit_sum += r5
            mrr_sum += m
            detail.append((item["num"], r5, round(m, 2)))
        n = len(items)
        rows.append({
            "set": name, "mode": mode_name,
            "recall@5": round(hit_sum / n, 3), "mrr": round(mrr_sum / n, 3),
            "detail": detail,
        })
    return rows


def main() -> None:
    questions = json.loads((config.ROOT / "data" / "questions.json").read_text(encoding="utf-8"))
    a_items = []
    for q in questions:
        if q.get("category") != "A":
            continue
        a_items.append({
            "num": q["num"], "question": q["question"],
            "gold_titles": GOLD_TITLES.get(q["num"], []),
            "gold_job_keywords": EXTRA_JOB_KEYWORDS.get(q["num"], []),
        })
    paraphrase = json.loads((config.ROOT / "data" / "questions_paraphrase.json").read_text(encoding="utf-8"))

    chunks = build_chunks()
    bm25 = BM25Index(chunks)
    vector = VectorIndex(chunks)
    hybrid = HybridRetriever(bm25, vector)

    modes = {
        "BM25": lambda q: bm25.search(q, config.TOP_K),
        "向量": lambda q: vector.search(q, config.TOP_K),
        "RRF混合": lambda q: hybrid.search(q, config.TOP_K),
    }

    rows = run_set("原题-关键词型", a_items, chunks, modes)
    rows += run_set("改写题-同义型", paraphrase, chunks, modes)

    print(f"{'题组':<14}{'策略':<10}{'Recall@5':<12}{'MRR'}")
    for r in rows:
        print(f"{r['set']:<14}{r['mode']:<10}{r['recall@5']:<12}{r['mrr']}")
    print("\n逐题（题号, Recall@5, MRR）：")
    for r in rows:
        print(f"  [{r['set']}] {r['mode']}: {r['detail']}")

    out = config.ROOT / "data" / "recall_result.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
