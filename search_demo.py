# -*- coding: utf-8 -*-
"""检索演示：不调 API、不花钱，只看能不能捞到相关岗位。

    python search_demo.py
    python search_demo.py --chunk 300 --topk 5
"""
import argparse

from ragjob.bm25 import BM25Index
from ragjob.corpus import build_chunks

QUERIES = [
    "哪类岗位更强调知识库和 RAG？",
    "知识库 检索增强 向量数据库 召回",
    "数据标注 质检 模型评测",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=300, help="切段长度，0=不切")
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    chunks = build_chunks(args.chunk, args.overlap)
    print(f"语料：{len(chunks)} 段（切段 {args.chunk or '不切'}，重叠 {args.overlap}）")
    index = BM25Index(chunks)
    print(f"索引完成：词表 {len(index.vocab)}\n")

    for q in QUERIES:
        print("=" * 66)
        print(f"查询：{q}")
        print("=" * 66)
        for rank, (score, idx) in enumerate(index.search_with_stats_boost(q, args.topk), start=1):
            c = chunks[idx]
            print(f"[{rank}] 分数 {score:.2f}　[{c['src']}] {c['title']}　{c['company']}")
        print()


if __name__ == "__main__":
    main()
