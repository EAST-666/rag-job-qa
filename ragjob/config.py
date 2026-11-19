# -*- coding: utf-8 -*-
"""全局配置：路径、模型、默认参数。

所有相对路径都以仓库根目录为基准，换机器只需改 CORPUS / STATS_DOC 两处
（或用环境变量 RAG_CORPUS / RAG_STATS_DOC / RAG_API_KEY 覆盖）。
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------- 数据 ----------
CORPUS = Path(os.getenv("RAG_CORPUS", r"D:\bangong\DeepSeek Harness\resume_work\jobscraper\out\corpus_v3.json"))
STATS_DOC = Path(os.getenv("RAG_STATS_DOC", r"D:\bangong\求职\W2\01_知识库文档\10_岗位画像统计汇总.txt"))
QUESTIONS = ROOT / "data" / "questions.json"
KEY_FILE = Path(os.getenv("RAG_API_KEY_FILE", ROOT / "key.txt"))

# ---------- 模型 ----------
API_URL = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-chat"
TEMPERATURE = 0.3
MAX_TOKENS = 700

# ---------- 检索默认参数 ----------
CHUNK_SIZE = 300      # 切段长度，0 = 不切
CHUNK_OVERLAP = 50    # 相邻段重叠字数
TOP_K = 5             # 每次召回段数
MAX_PER_JOB = 1       # 同一岗位最多占几个召回名额
MIN_DOC_LEN = 80      # 正文短于此的文档直接丢弃

SYSTEM_PROMPT = """# 角色
你是求职资料助手，服务对象是一个计算机专业应届生，正在找杭州的 AI 实习。
你的知识只来自下面提供的「资料」，资料是真实岗位 JD 的片段。

# 资料
{knowledge}

# 任务
根据资料回答用户的问题。答案要具体，能直接拿去用。

# 规则
1. 资料里没有的内容，直接回答「资料中没有提到这个，我不能确定」，不要用常识补。
2. 每个结论后面用 【来源：第N段】 标出依据；没有依据的句子不要写。
3. 涉及数字（条数、比例）必须照资料里写，不要估算。
4. 回答控制在 300 字以内，用短句，不要客套话。
5. 不要输出代码，不要输出 JSON。
6. 不要用 markdown 加粗（不要出现 ** 号）。

# 输出格式
直接给答案，不要复述问题。"""

# 疑问词、虚词：查询里出现也当没出现
STOP_TERMS = set("哪类哪些什么怎么为什么如何可以吗呢的是了在有和与及或更强调要求岗位")

# 统计型问题（多少条 / 占比 / 频率）要优先给统计汇总加权
AGGREGATE_HINTS = ("多少", "几条", "条数", "占比", "比例", "频率", "占多少", "统计")
