"""
docs-seeker - BM25 路由
判断查询是否适合走 BM25 检索（关键词匹配型查询）
"""


class HybridRouter:
    """BM25 路由器

    判断逻辑：
    - 包含专有名词/代码/标识符 → 适合 BM25
    - 纯语义/理解型问题 → 不适合 BM25
    """

    # 适合 BM25 的信号词
    BM25_SIGNALS = [
        "定义",
        "什么是",
        "是指",
        "缩写",
        "全称",
        "简称",
        "第几",
        "多少",
        "哪些",
        "列举",
        "列表",
        "配置",
        "参数",
        "选项",
        "设置",
        "命令",
        "语法",
    ]

    # 不适合 BM25 的信号词（纯语义理解型）
    SEMANTIC_SIGNALS = [
        "为什么",
        "如何理解",
        "解释",
        "分析",
        "对比",
        "区别",
        "优缺点",
        "建议",
        "思路",
        "原理",
    ]

    def should_use_bm25(self, question: str) -> tuple[bool, str]:
        """判断是否应该使用 BM25 检索

        Returns:
            (是否使用 BM25, 原因)
        """
        # 检查 BM25 信号
        bm25_hits = sum(1 for s in self.BM25_SIGNALS if s in question)
        semantic_hits = sum(1 for s in self.SEMANTIC_SIGNALS if s in question)

        # 包含英文/代码/数字 → 更适合 BM25
        has_code = any(c.isalpha() and ord(c) < 128 and c.isupper() for c in question)
        has_number = any(c.isdigit() for c in question)

        if bm25_hits > semantic_hits:
            return True, f"关键词匹配型查询（BM25 信号 {bm25_hits}）"
        if has_code or has_number:
            return True, "包含标识符/数字，适合精确匹配"
        if semantic_hits > 0:
            return False, f"语义理解型查询（语义信号 {semantic_hits}）"

        # 默认：短查询走 BM25，长查询走语义
        if len(question) < 15:
            return True, "短查询，适合 BM25"
        return False, "默认语义检索"
