# -*- coding: utf-8 -*-
"""单条问答演示：检索 → 拼上下文 → 调模型，打印三段耗时。

    python gen_demo.py                    真跑一次（花几厘钱）
    python gen_demo.py --dry              只看检索到哪几段
    python gen_demo.py --question "..."   换问题
"""
import argparse
import time

from ragjob.bm25 import BM25Index
from ragjob.corpus import build_chunks
from ragjob.generate import build_context, call_llm, load_api_key

DEFAULT_Q = "哪类岗位更强调知识库和 RAG？这些岗位一般要求什么技能？"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=300)
    ap.add_argument("--overlap", type=int, default=50)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--question", default=DEFAULT_Q)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    chunks = build_chunks(args.chunk, args.overlap)
    index = BM25Index(chunks)

    t0 = time.perf_counter()
    hits = index.search_with_stats_boost(args.question, args.topk)
    ms_retrieval = (time.perf_counter() - t0) * 1000
    context = build_context(chunks, hits)

    print(f"[检索] {len(chunks)} 段里取前 {args.topk} 段，耗时 {ms_retrieval:.0f} ms")
    for rank, (score, idx) in enumerate(hits, start=1):
        print(f"  第{rank}段 {score:6.2f}  {chunks[idx]['title']}")

    if args.dry:
        print("\n[空跑] 不调 API。上下文前 500 字：\n")
        print(context[:500])
        return

    key = load_api_key()
    t1 = time.perf_counter()
    answer, usage = call_llm(args.question, context, key)
    ms_gen = (time.perf_counter() - t1) * 1000

    print("\n" + "=" * 66 + "\n答案：\n" + "=" * 66)
    print(answer)
    print("\n耗时拆段")
    print(f"  检索 {ms_retrieval:6.0f} ms")
    print(f"  生成 {ms_gen:6.0f} ms  ← 通常是瓶颈")
    print(f"  合计 {ms_retrieval + ms_gen:6.0f} ms")
    print(f"token：输入 {usage.get('prompt_tokens')} 输出 {usage.get('completion_tokens')}")


if __name__ == "__main__":
    main()
