"""docs-seeker - LLM 层错误类型"""


class AllModelsFailedError(Exception):
    """主模型与备用模型都失败（网关对上层暴露的统一失败类型）"""
