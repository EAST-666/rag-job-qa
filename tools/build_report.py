# -*- coding: utf-8 -*-
"""把「平台版 vs 代码版」和「三组切段实验」做成一张对照表（xlsx，可直接截图进简历/面试用）。"""
import json
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

BASE = r"D:\bangong\求职\W3_代码版RAG"
OUT = r"D:\bangong\DeepSeek Harness\resume_work\W3-平台版vs代码版对照表.xlsx"

NAVY = "1F3F66"
HEAD = PatternFill("solid", fgColor=NAVY)
LIGHT = PatternFill("solid", fgColor="EEF2F7")
GOOD = PatternFill("solid", fgColor="E3F4E8")
BAD = PatternFill("solid", fgColor="FBE6E6")
THIN = Side(style="thin", color="C9D2DD")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def head_row(ws, row, values, widths=None):
    for i, v in enumerate(values, start=1):
        c = ws.cell(row=row, column=i, value=v)
        c.font = Font(name="微软雅黑", size=10, bold=True, color="FFFFFF")
        c.fill = HEAD
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = BORDER
    ws.row_dimensions[row].height = 26
    if widths:
        for i, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(i)].width = w


def fill_row(ws, row, values, height=30, fills=None, bold_first=False, size=9.5):
    for i, v in enumerate(values, start=1):
        c = ws.cell(row=row, column=i, value=v)
        c.font = Font(name="微软雅黑", size=size, bold=(bold_first and i == 1))
        c.alignment = Alignment(vertical="center", wrap_text=True,
                                horizontal="center" if i > 2 else "left")
        c.border = BORDER
        if fills and i in fills:
            c.fill = fills[i]
    ws.row_dimensions[row].height = height


wb = Workbook()
ws = wb.active
ws.title = "平台版vs代码版"

ws["A1"] = "同一个实验，两种实现：Coze 平台版 vs Python 代码版"
ws["A1"].font = Font(name="微软雅黑", size=13, bold=True, color=NAVY)
ws.merge_cells("A1:D1")
ws["A2"] = "同一批 410 条 JD、同一套 20 题（A10/B5/C5 含标准答案）、同一个模型 deepseek-chat(temp 0.3)"
ws["A2"].font = Font(name="微软雅黑", size=9.5, color="555555")
ws.merge_cells("A2:D2")

head_row(ws, 4, ["维度", "Coze 平台版", "Python 代码版", "说明"], [16, 34, 34, 40])

rows = [
    ("实现方式", "拖节点：知识库检索 → 大模型 → 结束", "BM25 检索 + 拼上下文 + 调 API（约 300 行）", "平台给的是“能用”，代码给的是“能改”"),
    ("语料", "10 份 txt 上传知识库（410 条 JD + 统计汇总）", "corpus_v3.json 直读 + 统计汇总单独入库", "代码版能看清每段是怎么被选中的"),
    ("切段", "500 字 / 重叠 20%（平台固定参数）", "300 字 / 重叠 50（可任意改，一行参数）", "切段在平台是黑盒，在代码里是一个数字"),
    ("召回", "top-k 默认 → 6 → 8（三轮手工试）", "top-k 5（一次跑完，可脚本扫参数）", "平台调参靠手动点，代码靠循环"),
    ("A 类答对", "6/10 → 8/10 → 9/10", "9/10", "两边最终同一水平；代码版一轮到位"),
    ("B 类拒答", "5/5（全程未退化）", "5/5", "防幻觉靠提示词约束，与实现无关"),
    ("C 类合理", "3/5 → 未单独测", "5/5（承认不确定即合理）", "C 类无标准答案，判分口径要写清"),
    ("平均输入 token", "1629（平台默认 top-k）", "870", "代码版可压到一半，直接省成本"),
    ("检索耗时", "295 ms（平台内部检索）", "2-4 ms（本地 BM25）", "本地检索是纯 CPU，比向量库轻"),
    ("生成耗时", "约 2.0-2.4 s", "0.9-2.3 s", "瓶颈都在模型侧，不在检索"),
    ("唯一失败题", "Q9 举具体岗位例子（跨文档）", "Q5 Python 条数（高频词）", "两边失败点不同，说明各自有短板"),
    ("能改的地方", "只能改提示词和 top-k", "换检索算法、加权重、改判分、批量扫参", "这就是“会 Coze”与“会 RAG”的分界线"),
]
r = 5
for row in rows:
    fill_row(ws, r, row, height=34, bold_first=True)
    r += 1

ws.cell(row=r + 1, column=1, value="一句话结论（可写进简历）").font = Font(name="微软雅黑", size=10, bold=True, color=NAVY)
ws.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=4)
concl = ("用同一批 410 条岗位语料和同一套 20 题评测集，在 Coze 平台版与 Python 代码版上各跑一遍："
         "两边 A 类答对率都到 9/10，但代码版把平均输入 token 从 1629 压到 870、检索从 295ms 降到 2-4ms，"
         "并且能定位到平台版查不出的问题——高频词（如 Python 出现 193 条）在统计型问题上召回失效。")
c = ws.cell(row=r + 2, column=1, value=concl)
c.font = Font(name="微软雅黑", size=9.5)
c.alignment = Alignment(wrap_text=True, vertical="center")
ws.merge_cells(start_row=r + 2, start_column=1, end_row=r + 2, end_column=4)
ws.row_dimensions[r + 2].height = 48

# ---------------- 第二张表：切段实验 ----------------
ws2 = wb.create_sheet("切段实验")
ws2["A1"] = "切段实验：只改切段长度，其他全部不变（top-k=5，同一查询）"
ws2["A1"].font = Font(name="微软雅黑", size=13, bold=True, color=NAVY)
ws2.merge_cells("A1:F1")
ws2["A2"] = "查询：哪类岗位更强调知识库和 RAG？这些岗位一般要求什么技能？"
ws2["A2"].font = Font(name="微软雅黑", size=9.5, color="555555")
ws2.merge_cells("A2:F2")

head_row(ws2, 4, ["切段设置", "上下文长度", "输入 token", "检索耗时", "生成耗时", "答案质量"], [16, 14, 12, 12, 12, 46])

data = [
    ("不切（整条 JD）", "4951 字", 3046, "192 ms", "2181 ms", "命中养生堂等大段 JD，广告话术混进答案；信息全但噪声多"),
    ("切 300 字", "1586 字", 1138, "287 ms", "2219 ms", "最好：命中链坊的 RAG 链路细节、Agent 工程师的解析/切片/重排"),
    ("切 500 字", "1851 字", 1310, "223 ms", "2259 ms", "同一岗位多片段占位，答案偏“技能清单”，丢岗位对比"),
]
r = 5
for row in data:
    fill_row(ws2, r, row, height=38, fills={3: (GOOD if row[2] == 1138 else LIGHT)})
    r += 1

ws2.cell(row=r + 1, column=1, value="读出来的三条").font = Font(name="微软雅黑", size=10, bold=True, color=NAVY)
ws2.merge_cells(start_row=r + 1, start_column=1, end_row=r + 1, end_column=6)
notes = [
    "① 切到 300 字后输入 token 降 63%（3046→1138），生成耗时几乎不变（2.2 s 恒定）⇒ 瓶颈是模型本身，不是上下文长度。",
    "② 不切段反而更差：token 多 3 倍、噪声更多，答案里混进招聘广告话术。",
    "③ 500 字会出现同一岗位多片段占位（一次检索吃掉多个名额），已经用“每岗位最多取 1 段”修掉。",
]
r += 2
for n in notes:
    c = ws2.cell(row=r, column=1, value=n)
    c.font = Font(name="微软雅黑", size=9.5)
    c.alignment = Alignment(wrap_text=True, vertical="center")
    ws2.merge_cells(start_row=r, start_column=1, end_row=r, end_column=6)
    ws2.row_dimensions[r].height = 30
    r += 1

for sheet in (ws, ws2):
    sheet.page_setup.orientation = "landscape"
    sheet.sheet_properties.pageSetUpPr.fitToPage = True
    sheet.page_setup.fitToWidth = 1
    sheet.page_setup.fitToHeight = 0
    sheet.page_margins.left = sheet.page_margins.right = 0.4
    sheet.page_margins.top = sheet.page_margins.bottom = 0.5

wb.save(OUT)
print("saved ->", OUT)
