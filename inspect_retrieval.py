# -*- coding: utf-8 -*-
"""排查工具：看一条查询到底为什么被排到这些段。

    python inspect_retrieval.py --question "明确写 Python 的岗位有多少条" --wide 25
"""
import argparse

from ragjob.bm25 import BM25Index, query_terms
from ragjob.config import AGGREGATE_HINTS
from ragjob.corpus import build_chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default="明确写 Python 的岗位有多少条")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--wide", type=int, default=25)
    ap.add_argument("--chunk", type=int, default=300)
    args = ap.parse_args()

    chunks = build_chunks(args.chunk, 50)
    index = BM25Index(chunks)

    print(f"问题：{args.question}")
    print(f"是否统计类查询：{any(h in args.question for h in AGGREGATE_HINTS)}")
    print(f"词典命中词：{query_terms(args.question)}")

    print(f"\n── 常规检索 top{args.topk} ──")
    for rank, (score, idx) in enumerate(index.search(args.question, args.topk), start=1):
        c = chunks[idx]
        print(f"[{rank}] {score:7.2f} [{c['src']}] {c['title']} · {c['company']}")

    print(f"\n── 宽召回 top{args.wide}（看统计段有没有进池）──")
    stats_hit = 0
    for rank, (score, idx) in enumerate(index.scores(args.question, args.wide), start=1):
        c = chunks[idx]
        is_stats = c["title"].startswith("岗位画像统计汇总")
        stats_hit += is_stats
        flag = "  ← 统计段" if is_stats else ""
        print(f"[{rank:2d}] {score:7.2f} {c['title'][:28]:28s}{flag}")
    print(f"\n宽召回里统计段出现 {stats_hit} 次（0 次说明 IDF 已坍塌，靠打分永远排不上）")

    print(f"\n── 带 boost 的结果 top{args.topk} ──")
    for rank, (score, idx) in enumerate(index.search_with_stats_boost(args.question, args.topk), start=1):
        c = chunks[idx]
        print(f"[{rank}] {score:7.2f} {c['title'][:40]}")


if __name__ == "__main__":
    main()
