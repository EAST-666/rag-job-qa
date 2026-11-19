# -*- coding: utf-8 -*-
"""拼上下文 + 调 DeepSeek API。"""
from typing import Dict, List, Tuple

import requests

from . import config


def build_context(chunks: List[Dict], hits: List[Tuple[float, int]]) -> str:
    """把召回的段拼成带编号的资料，编号要和提示词里的【来源：第N段】对上。"""
    blocks = []
    for rank, (_, idx) in enumerate(hits, start=1):
        c = chunks[idx]
        blocks.append(f"第{rank}段｜{c['title']}｜{c['company']}\n{c['text']}")
    return "\n\n".join(blocks)


def load_api_key(path=None) -> str:
    path = path or config.KEY_FILE
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def call_llm(question: str, context: str, api_key: str,
             model: str = None, temperature: float = None) -> Tuple[str, Dict]:
    """调一次对话接口，返回 (答案, usage)。"""
    model = model or config.MODEL
    temperature = config.TEMPERATURE if temperature is None else temperature
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": config.SYSTEM_PROMPT.format(knowledge=context)},
            {"role": "user", "content": question},
        ],
        "temperature": temperature,
        "max_tokens": config.MAX_TOKENS,
    }
    resp = requests.post(config.API_URL,
                         headers={"Authorization": f"Bearer {api_key}",
                                  "Content-Type": "application/json"},
                         json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip(), data.get("usage", {})
