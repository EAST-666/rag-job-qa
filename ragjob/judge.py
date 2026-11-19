# -*- coding: utf-8 -*-
"""规则判分器。

为什么不用 LLM 判分：LLM 判分自己也要花钱且不稳定，这一版先用可复现的规则，
把「判分标准本身有偏差」明确写进 README，而不是假装 19/20 就是真值。
"""
import json
import re

from . import config

REFUSE_WORDS = ("没有提", "未提及", "没有相关", "无法确定", "不能确定", "不确定",
                "资料中未", "没有这个", "查不到", "没有给出")


def load_questions(path=None):
    """读题库：每条含 num / category / question / rule。"""
    path = path or config.QUESTIONS
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def judge(answer: str, rule: dict, question: str = "", context: str = "",
          category: str = "") -> tuple:
    """返回 (是否通过, 判定说明)。

    三类题走三条不同规则：
    - A 类：答案里必须出现标准答案的关键词
    - B 类：必须拒答，且不能出现资料里没有的数字
    - C 类：承认不确定即算合理
    """
    low = answer.lower()

    if category == "C" and any(w in answer for w in REFUSE_WORDS):
        return True, "承认不确定（C 类合理）"

    if rule.get("refuse"):
        hit = any(w in answer for w in REFUSE_WORDS)
        asked = set(re.findall(r"\d+", question)) | set(re.findall(r"\d+", context))
        made_up = [n for n in re.findall(r"\d{2,}", answer) if n not in asked]
        if hit and not made_up:
            return True, "正确拒答"
        if hit and made_up:
            return False, f"拒答但夹带了编造的数字 {made_up}"
        return False, "没有拒答（可能编造）"

    ok, detail = True, []
    for w in rule.get("all", []):
        if w.lower() not in low:
            ok = False
            detail.append(f"缺 {w}")
    need = rule.get("any", [])
    if need:
        hits = [w for w in need if w.lower() in low]
        min_hits = rule.get("min_hits", 1)
        if len(hits) < min_hits:
            ok = False
            detail.append(f"只命中 {len(hits)}/{min_hits} 个关键词")
        else:
            detail.append(f"命中 {len(hits)} 个")
    if rule.get("need_example") and "岗位" not in answer:
        ok = False
        detail.append("没举具体岗位")
    return ok, "；".join(detail) or "通过"
