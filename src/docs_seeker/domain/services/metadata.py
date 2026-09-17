"""docs-seeker - 问题结构化元数据解析（领域规则）

动机（2026-09-16）：语料的结构化字段（chapter/section/article 覆盖率
98%/89%/80%）此前完全没参与检索——「第三十六条是什么内容？」这类**指名道姓**
的问法只能靠语义相似度碰运气，检索窗口覆盖不到目标条款（评估集 T015 三轮恒为
0 分，而本机实测按条号过滤能直接取到含期望关键词的原文）。

本模块只负责**解析**：从问题里解析「第X章/节/条/款/项」，返回
``{字段: [取值前缀, ...]}``。把解析结果翻译成具体存储的过滤形式（Milvus 表达式 /
BM25 谓词）属于基础设施适配，见 ``infra/retrieval/metadata_filter.py``。

字段映射按本语料 doc-kit 入库的实际形态：章→chapter、节→section，
**条/款/项 都落在 article 字段**（实测 `第三章 article='第1款'`、
`第五章 article='第一项'`）。解析不出结构词时返回空 dict —— 调用方据此保持
旧行为（不过滤），保证改动可逆。

匹配一律用**前缀**（Milvus `like "第三十六条%"` / Python `startswith`）：
语料里 article 的实际取值是「第三十六条 领导水平」这类带标题的串，而 chapter
是「第四章」这类精确值，前缀对两者都成立。
"""

from __future__ import annotations

import re

# 中文数字 → 数值（两 = 2 的异体写法，语料中「第两」不会出现，但代价为零）
_CN_DIGITS = {
    "零": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100}

# 「第X章/节/条/款/项」：数字支持中文与阿拉伯（第36条），中间允许空格
_STRUCT_PATTERN = re.compile(r"第\s*([0-9]{1,3}|[零一二两三四五六七八九十百]{1,5})\s*([章节条款项])")

# 单位 → 语料字段（见模块 docstring：款/项 也存 article）
_UNIT_FIELD = {"章": "chapter", "节": "section", "条": "article", "款": "article", "项": "article"}


def _cn_to_int(text: str) -> int | None:
    """中文/阿拉伯数字串 → int；无法解析返回 None。

    支持 十 / 十X / X十 / X十Y / 一百零八 等写法。
    """
    text = text.strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    total = 0
    section = 0
    number = 0
    for char in text:
        if char in _CN_DIGITS:
            number = _CN_DIGITS[char]
        elif char in _CN_UNITS:
            unit = _CN_UNITS[char]
            if number == 0:
                number = 1  # 「十」= 10，「十六」= 16
            section += number * unit
            number = 0
        else:
            return None
    return total + section + number


def _int_to_cn(value: int) -> str:
    """int → 中文数字（1~999；语料条号不会超过三位）"""
    digits = "零一二三四五六七八九"
    if value <= 0:
        return str(value)
    if value < 10:
        return digits[value]
    if value < 20:
        return "十" + (digits[value - 10] if value > 10 else "")
    if value < 100:
        return digits[value // 10] + "十" + (digits[value % 10] if value % 10 else "")
    hundred, rest = divmod(value, 100)
    head = digits[hundred] + "百"
    if rest == 0:
        return head
    if rest < 10:
        return head + "零" + digits[rest]
    return head + _int_to_cn(rest)


def parse_question_metadata(question: str) -> dict[str, list[str]]:
    """解析问题中的「第X章/节/条/款/项」，返回 ``{字段: [取值前缀, ...]}``。

    - 同一字段多次引用会合并（如「第三十六条和第三十九条」→ article 两个前缀）；
    - 每个引用同时给出中文与阿拉伯两种写法（语料混用：正文条号是中文数字，
      部分章节的款/项是阿拉伯数字「第1款」）；
    - 解析不出结构词 → ``{}``（调用方据此走旧行为）。
    """
    found: dict[str, list[str]] = {}
    for raw_number, unit in _STRUCT_PATTERN.findall(question or ""):
        value = _cn_to_int(raw_number)
        if value is None or value <= 0:
            continue
        field = _UNIT_FIELD[unit]
        bucket = found.setdefault(field, [])
        for prefix in (f"第{_int_to_cn(value)}{unit}", f"第{value}{unit}"):
            if prefix not in bucket:
                bucket.append(prefix)
    return found
