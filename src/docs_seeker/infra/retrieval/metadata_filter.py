"""docs-seeker - 结构化元数据的存储过滤适配。

把 ``domain.services.metadata.parse_question_metadata`` 的解析结果翻译成具体
存储能用的过滤形式：

1. ``build_milvus_expr``：→ Milvus 过滤表达式（dense 路用）；
2. ``matches_metadata``：→ Python 谓词（BM25 进程内索引用）。

解析（从问题里认出「第X章/节/条/款/项」）是领域规则，在 domain 侧；
本模块只做「领域解析结果 → 存储查询语法」的翻译，无 IO、可单测。

匹配一律用**前缀**（Milvus `like "第三十六条%"` / Python `startswith`）：
语料里 article 的实际取值是「第三十六条 领导水平」这类带标题的串，而 chapter
是「第四章」这类精确值，前缀对两者都成立。
"""

from __future__ import annotations


def build_milvus_expr(meta: dict[str, list[str]] | None) -> str:
    """``{字段: [前缀, ...]}`` → Milvus 过滤表达式（空 dict → ""，即不过滤）。

    字段与取值都由 domain 的解析器生成（字段取自常量表、取值只含数字与中文字符），
    不接受任意用户输入拼接，无注入面。
    """
    if not meta:
        return ""
    clauses = []
    for field, prefixes in meta.items():
        alternatives = " or ".join(f'$meta["{field}"] like "{prefix}%"' for prefix in prefixes)
        clauses.append(f"({alternatives})")
    return " and ".join(clauses)


def matches_metadata(doc: dict, meta: dict[str, list[str]] | None) -> bool:
    """BM25 进程内索引的谓词版本（空 meta → True，即不过滤）"""
    if not meta:
        return True
    for field, prefixes in meta.items():
        value = str(doc.get(field) or "")
        if not any(value.startswith(prefix) for prefix in prefixes):
            return False
    return True
