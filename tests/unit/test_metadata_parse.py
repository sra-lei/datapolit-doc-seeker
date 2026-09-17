"""问题结构化元数据解析（domain 领域规则）单元测试

覆盖：中文/阿拉伯数字解析、字段映射、多次引用合并、无结构词不误伤。
纯函数，不依赖 Milvus / LLM / Redis。
"""

from docs_seeker.domain.services.metadata import parse_question_metadata


def test_parse_article_chinese_numeral():
    assert parse_question_metadata("第三十六条是什么内容？") == {"article": ["第三十六条", "第36条"]}


def test_parse_chapter_keeps_unit():
    assert parse_question_metadata("第三章讲了什么？") == {"chapter": ["第三章", "第3章"]}


def test_parse_chapter_and_article_are_anded_fields():
    meta = parse_question_metadata("第四章第三十六条讲了什么？")
    assert meta == {
        "chapter": ["第四章", "第4章"],
        "article": ["第三十六条", "第36条"],
    }


def test_parse_arabic_numeral_normalizes_to_chinese_first():
    assert parse_question_metadata("第10条的内容？") == {"article": ["第十条", "第10条"]}


def test_parse_kuan_and_xiang_map_to_article_field():
    """款/项 在本语料存在 article 字段（实测 第三章 article='第1款'）"""
    assert parse_question_metadata("第五章第3款写了什么？")["article"] == ["第三款", "第3款"]
    assert parse_question_metadata("第六章第二项")["article"] == ["第二项", "第2项"]


def test_parse_section_maps_to_section_field():
    assert parse_question_metadata("第二节基本素质是什么？") == {"section": ["第二节", "第2节"]}


def test_parse_multiple_references_same_field_merged():
    meta = parse_question_metadata("第三十六条和第三十九条分别是什么？")
    assert meta["article"] == ["第三十六条", "第36条", "第三十九条", "第39条"]


def test_no_structure_word_returns_empty():
    """无「第X章/节/条/款/项」的问题必须解析为空 —— 否则会误伤本就正常的检索"""
    for question in (
        "公司的十六字方针是什么？",
        "公司对员工有哪些要求？",
        "实事求是是什么意思？",
        "第一次入职要带什么材料？",
        "第三方供应商如何管理？",
        "公司是中国房地产企业多少强？",
    ):
        assert parse_question_metadata(question) == {}
