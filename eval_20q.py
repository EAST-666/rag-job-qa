# -*- coding: utf-8 -*-
"""20 题自动评测：跑一遍、判一遍、算三类分数。

    python eval_20q.py                 全量 20 题（约 40 秒）
    python eval_20q.py --only A        只跑 A 类
    python eval_20q.py --limit 3       只跑前 3 题（试用）
    python eval_20q.py --dry           只检索不调 API
"""
import argparse
import json
import time

from ragjob.bm25 import BM25Index
from ragjob.corpus import build_chunks
from ragjob.generate import build_context, call_llm, load_api_key
from ragjob.judge import judge, load_questions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="", help="只跑某一类：A / B / C")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out", default="eval_result.json")
    args = ap.parse_args()

    chunks = build_chunks(args.chunk, args.overlap)
    index = BM25Index(chunks)
    print(f"[配置] 切段={args.chunk} 重叠={args.overlap} topk={args.topk} ｜ {len(chunks)} 段")

    items = load_questions()
    if args.only:
        items = [q for q in items if q["category"] == args.only.upper()]
    if args.limit:
        items = items[:args.limit]

    key = None if args.dry else load_api_key()
    records, t_start = [], time.perf_counter()

    for item in items:
        question = item["question"]
        t0 = time.perf_counter()
        hits = index.search_with_stats_boost(question, args.topk)
        context = build_context(chunks, hits)
        ms_retrieval = (time.perf_counter() - t0) * 1000

        if args.dry:
            print(f"{item['num']:2d} [{item['category']}] 检索 {ms_retrieval:4.0f} ms"
                  f" 上下文 {len(context):5d} 字")
            continue

        t1 = time.perf_counter()
        answer, usage = call_llm(question, context, key)
        ms_gen = (time.perf_counter() - t1) * 1000
        ok, detail = judge(answer, item["rule"], question, context, item["category"])

        records.append({"num": item["num"], "category": item["category"],
                        "question": question, "standard": item["standard"],
                        "answer": answer, "passed": ok, "detail": detail,
                        "ms_retrieval": round(ms_retrieval), "ms_generate": round(ms_gen),
                        "input_tokens": usage.get("prompt_tokens")})

        mark = "PASS" if ok else ("REFUSE-MISS" if item["category"] == "B" else "FAIL")
        print(f"{item['num']:2d} [{item['category']}] {mark:11s} {detail[:36]:36s}"
              f" 检索{ms_retrieval:4.0f}ms 生成{ms_gen:4.0f}ms")

    if args.dry:
        print(f"\n空跑完成，耗时 {time.perf_counter() - t_start:.1f}s（没花钱）")
        return

    summary = {}
    for cat in ("A", "B", "C"):
        sub = [r for r in records if r["category"] == cat]
        if sub:
            summary[cat] = (sum(r["passed"] for r in sub), len(sub))
            print(f"\n{cat} 类：{summary[cat][0]}/{summary[cat][1]}")
    total_pass = sum(r["passed"] for r in records)
    print(f"总计：{total_pass}/{len(records)}")
    if records:
        avg_in = sum(r["input_tokens"] or 0 for r in records) / len(records)
        print(f"平均输入 token：{avg_in:.0f}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"config": {"chunk": args.chunk, "overlap": args.overlap, "topk": args.topk},
                   "summary": {k: {"passed": v[0], "total": v[1]} for k, v in summary.items()},
                   "total": {"passed": total_pass, "n": len(records)},
                   "records": records}, f, ensure_ascii=False, indent=1)
    print(f"已存 {args.out}")


if __name__ == "__main__":
    main()
