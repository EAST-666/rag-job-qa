# -*- coding: utf-8 -*-
"""语料加载、清洗、切段。

三件事：
1. clean_text —— 招聘站点抓下来的正文里混着 JS/JSON 碎片，不洗会污染检索排序
2. load_corpus —— 读 JD 语料
3. load_stats_doc —— 统计汇总单独入库，且必须切短（长文档在 BM25 里会被长度惩罚）
"""
import json
import re
from pathlib import Path
from typing import List, Dict

from . import config


def clean_text(text: str) -> str:
    """去掉页面残留的 JSON 片段与代码痕迹。"""
    text = re.sub(r'"(?:[A-Za-z_]+)"\s*:\s*"(?:[^"\\]|\\.){0,120}"', " ", text)
    text = re.sub(r'"(?:[A-Za-z_]+)"\s*:\s*\[[^\]]{0,200}\]', " ", text)
    text = re.sub(r'\\+"', " ", text)
    text = re.sub(r"[{}]", " ", text)
    text = re.sub(r"[A-Za-z_ ]{3,}\s*=\s*(?:true|false)", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()


def load_corpus(path: Path = None) -> List[Dict]:
    """读 JD 语料，返回 [{title, company, src, text}]。"""
    path = Path(path or config.CORPUS)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = []
    for job in data["jobs"]:
        text = clean_text((job.get("text") or "").strip())
        if len(text) < config.MIN_DOC_LEN:
            continue
        docs.append({"title": job.get("title") or "",
                     "company": job.get("company") or "",
                     "src": job.get("src") or "",
                     "text": text})
    return docs


def split_chunks(docs: List[Dict], size: int = None, overlap: int = None) -> List[Dict]:
    """把每篇文档切成 size 字的段，相邻段重叠 overlap 字；size<=0 表示不切。"""
    size = config.CHUNK_SIZE if size is None else size
    overlap = config.CHUNK_OVERLAP if overlap is None else overlap
    chunks, seen = [], set()
    for doc in docs:
        text = doc["text"]
        if size <= 0 or len(text) <= size:
            pieces = [(0, text)]
        else:
            step = max(1, size - overlap)
            pieces = []
            for part, start in enumerate(range(0, len(text), step)):
                piece = text[start:start + size]
                if len(piece) < 60:
                    break
                pieces.append((part, piece))
                if start + size >= len(text):
                    break
        for part, piece in pieces:
            key = (doc["title"], doc["company"], part, piece[:80])
            if key in seen:
                continue
            seen.add(key)
            chunks.append({"title": doc["title"], "company": doc["company"],
                           "src": doc["src"], "text": piece, "part": part})
    return chunks


def load_stats_doc(path: Path = None) -> List[Dict]:
    """统计汇总按「一行一条」切成短段。

    为什么不整节入库：BM25 有长度惩罚，1100 字的一整节永远排不过 300 字的 JD 段，
    「多少条/占比」这类问题就会全部回答不了。
    """
    path = Path(path or config.STATS_DOC)
    if not path.exists():
        print(f"[warn] 统计汇总不存在：{path}（数字类问题会答不出）")
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    out, section = [], ""
    for line in lines:
        s = line.strip()
        if not s:
            continue
        if s.startswith("## ") or s.startswith("# "):
            section = s.lstrip("# ").strip()
            continue
        body = s[2:].strip() if s.startswith("- ") else s
        if len(body) < 6:
            continue
        name = body.split("：")[0].split(":")[0][:18]
        out.append({"title": f"岗位画像统计汇总·{name}", "company": "本次语料统计",
                    "src": "统计", "text": f"{section}：{body}", "part": len(out)})
    return out


def build_chunks(size: int = None, overlap: int = None, with_stats: bool = True) -> List[Dict]:
    """一步到位：JD 切段 + 统计汇总入库。"""
    chunks = split_chunks(load_corpus(), size, overlap)
    if with_stats:
        chunks += load_stats_doc()
    return chunks
