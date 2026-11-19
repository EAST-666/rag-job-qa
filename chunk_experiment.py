# -*- coding: utf-8 -*-
"""切段实验：同一查询、只改切段长度，看 token / 耗时 / 答案怎么变。

    python chunk_experiment.py --dry                 只看 token 变化（不花钱）
    python chunk_experiment.py                       真跑三组
    python chunk_experiment.py --sizes 0 300 500 800 自定义档位
"""
import argparse
import json
import time

from ragjob.bm25 import BM25Index
from ragjob.corpus import build_chunks
from ragjob.generate import build_context, call_llm, load_api_key

QUERY = "哪类岗位更强调知识库和 RAG？这些岗位一般要求什么技能？"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[0, 300, 500])
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    key = None if args.dry else load_api_key()
    rows = []

    for size in args.sizes:
        chunks = build_chunks(size, args.overlap)
        index = BM25Index(chunks)

        t0 = time.perf_counter()
        hits = index.search_with_stats_boost(QUERY, args.topk)
        ms_retrieval = (time.perf_counter() - t0) * 1000
        context = build_context(chunks, hits)

        record = {"chunk": size, "overlap": args.overlap, "topk": args.topk,
                  "context_chars": len(context), "ms_retrieval": round(ms_retrieval),
                  "cited": [chunks[i]["title"] for _, i in hits]}

        if args.dry:
            print(f"切段 {size or '不切'}：上下文 {len(context)} 字，检索 {ms_retrieval:.0f} ms")
            rows.append(record)
            continue

        t1 = time.perf_counter()
        answer, usage = call_llm(QUERY, context, key)
        ms_gen = (time.perf_counter() - t1) * 1000
        record.update({"answer": answer, "ms_gen": round(ms_gen), "usage": usage})
        rows.append(record)

        print(f"切段 {size or '不切'}：上下文 {len(context)} 字｜检索 {ms_retrieval:.0f} ms"
              f"｜生成 {ms_gen:.0f} ms｜输入 {usage.get('prompt_tokens')} token")
        name = f"result_chunk{size}_ov{args.overlap}_k{args.topk}.json"
        with open(name, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=1)
        print(f"  → {name}")

    if args.dry:
        print("\n空跑完成（没花钱）")


if __name__ == "__main__":
    main()
