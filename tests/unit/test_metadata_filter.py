"""结构化元数据的存储过滤适配（infra）单元测试

覆盖：Milvus 表达式构造（空值、字段内 or / 字段间 and）、BM25 谓词前缀语义。
纯函数，不依赖 Milvus / LLM / Redis。
"""

from docs_seeker.infra.retrieval.metadata_filter import build_milvus_expr, matches_metadata


def test_build_milvus_expr_empty_is_blank():
    assert build_milvus_expr(None) == ""
    assert build_milvus_expr({}) == ""


def test_build_milvus_expr_or_within_field_and_across_fields():
    expr = build_milvus_expr({"chapter": ["第四章", "第4章"], "article": ["第三十六条", "第36条"]})
    assert expr == (
        '($meta["chapter"] like "第四章%" or $meta["chapter"] like "第4章%")'
        ' and ($meta["article"] like "第三十六条%" or $meta["article"] like "第36条%")'
    )


def test_matches_metadata_prefix_semantics():
    meta = {"article": ["第三十六条", "第36条"]}
    assert matches_metadata({"article": "第三十六条 领导水平"}, meta) is True
    assert matches_metadata({"article": "第三十六条之一"}, meta) is True
    assert matches_metadata({"article": "第36条 领导水平"}, meta) is True
    assert matches_metadata({"article": "第三十九条 辞退"}, meta) is False
    assert matches_metadata({"article": ""}, meta) is False


def test_matches_metadata_across_fields_is_anded():
    meta = {"chapter": ["第四章"], "article": ["第三十六条"]}
    assert matches_metadata({"chapter": "第四章", "article": "第三十六条 领导水平"}, meta) is True
    assert matches_metadata({"chapter": "第五章", "article": "第三十六条 领导水平"}, meta) is False


def test_matches_metadata_empty_meta_is_no_filter():
    assert matches_metadata({"article": "任意"}, None) is True
    assert matches_metadata({"article": "任意"}, {}) is True
