"""
docs-seeker - 核心配置（pydantic-settings 环境变量 + yaml 资源加载）
所有配置通过环境变量读取，禁止硬编码；prompts/retrieval 从同目录 yaml 加载。
"""

from pathlib import Path

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    # DeepSeek (Chat)
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    llm_model: str = "deepseek-chat"

    # 阿里百炼 (Embedding)
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    # 与 doc-kit 入库模型一致（text-embedding-v4 = 1024 维）；库已用 v4 重建
    embedding_model: str = "text-embedding-v4"

    # Milvus
    milvus_uri: str = "http://localhost:19530"
    milvus_token: str = ""
    collection_name: str = "chartermate_docs"
    summary_collection_name: str = "chartermate_summaries"

    # Redis（语义缓存）
    redis_url: str = "redis://localhost:6379/0"
    semantic_cache_enabled: bool = True
    similarity_threshold: float = 0.92
    cache_ttl_hours: int = 24

    # 应用配置
    log_level: str = "INFO"
    # 运行环境：development / production（production 下禁用 /docs、/redoc、/openapi.json）
    environment: str = "development"

    # 热门问题 Top10（P4）：记录 + 预热语义缓存（ChatWidget 欢迎语用）
    top_warmup_enabled: bool = True
    top_warmup_interval_hours: int = 6
    top_warmup_size: int = 10

    # BM25 索引
    # 新鲜度检查间隔（秒）：每隔该时长用 Milvus row count 探测文档数是否变化，
    # 变化才重建索引；0 表示关闭自动刷新（只靠启动预热 + 手动 refresh()）
    bm25_refresh_seconds: int = 300
    # 建索引单次拉取文档上限（与 MilvusStore.get_all_documents 的 limit 对齐）
    bm25_max_docs: int = 10000

    # LLM 输出预算（token）
    # ⚠️ 推理模型（如 deepseek-v4-flash）会先消耗 reasoning token，预算过小会
    # 出现 finish_reason=length 且正文为空——生成与查询改写两处都必须留足预算。
    # 实测：泛化汇总类问题（"公司对员工有哪些要求"）单是 reasoning 就 ~8000 token。
    llm_generate_max_tokens: int = 8000  # 答案生成预算
    llm_decompose_max_tokens: int = 2000  # 查询改写预算
    # 截断（finish_reason=length）且正文为空时的放大预算重试（每处至多一次）；
    # 小于等于当前预算时视为关闭该重试
    llm_retry_max_tokens: int = 16000
    # 单次 LLM 调用的 HTTP 超时（秒）。推理模型的响应时间随题目波动很大
    # （长推理问题实测 20~60s），沿用普通模型的 15s 会直接把长答案打成超时失败。
    llm_timeout_seconds: float = 120.0
    # 判断类调用专用短超时（agent 循环内的逐步判断等）：120s × 重试最坏 ≈6 分钟
    # 会卡死循环，判断宁可快速失败走回退，也不要阻塞主流程
    llm_judge_timeout_seconds: float = 20.0
    # Transport middleware 链（逗号分隔，顺序 = 外层到内层；空 = 默认链）。
    # 可选：observability（观测参数）/ fallback（主备降级）/
    #       circuit_breaker（熔断）/ budget_guard（截断预算兜底）/ retry（退避重试）。
    # 例：LLM_TRANSPORT_MIDDLEWARES=observability,retry 可临时关掉降级与熔断。
    llm_transport_middlewares: str = ""
    # 熔断阈值（连续失败多少次打开熔断）与冷却时长（秒，冷却后半开试探一次）。
    # ⚠️ 未配 FALLBACK_* 时，熔断打开期间请求会直接失败（不再打后端），
    # 这是熔断的预期行为；若希望故障期继续对外服务，请配备用 provider。
    llm_circuit_failure_threshold: int = 5
    llm_circuit_recovery_seconds: int = 60
    # 应用级护栏链（逗号分隔，顺序 = 执行顺序；空 = 全部内置）。
    # 可选：injection_guard（提示注入）/ topic_policy（话题白名单）/ pii_redaction（输出脱敏）。
    # 同一份配置同时用于 pipeline 边界（可拒答/可改写）与 gateway 内（仅告警）两处挂载。
    llm_guards: str = ""

    # 采样温度。0 = 确定性输出（**评估与 A/B 必须用 0**：同代码同语料下单轮
    # 22 题的分数摆幅曾达 0.75，噪声大于待测改动的量级）；>0 = 保留多样性。
    # 默认沿用历史口径 0.3，评估跑批用环境变量 LLM_TEMPERATURE=0 覆盖。
    llm_temperature: float = 0.3
    # 查询改写温度（原本硬编码 0.1；可配置以便评估时一并固定在 0）
    llm_decompose_temperature: float = 0.1

    # 生成层模型覆盖（空 = 用 LLM_MODEL，即旧行为）。分层模型路由：答案生成用
    # 非推理模型（如 deepseek-chat）降延迟/降成本，而「判断/改写」仍用 LLM_MODEL
    # 指定的推理模型——本语料是抽取式的，推理模型在生成环节的增益尚未证明。
    llm_generate_model: str = ""

    # ---- Agentic RAG（M1：默认关闭，旧单轮管线零改动；异常自动回退旧管线）----
    agent_enabled: bool = False
    agent_max_steps: int = 4  # 单问最大「思考→工具」步数（不含最终成文）
    # 循环内判断调用短超时复用 llm_judge_timeout_seconds（见上）
    agent_judge_max_tokens: int = 500  # 思考/动作 JSON 的生成长度
    agent_evidence_char_budget: int = 12000  # 成文前累计证据的总字符预算
    # agent 路径失败时是否回退旧单轮管线：
    #   None（默认）= 按 environment 推导 —— production 回退、其他环境不回退；
    #   显式 true/false 可覆盖（环境变量 AGENT_FALLBACK_ENABLED）。
    # 生产以可用性优先（回退保证有答案）；开发期默认不回退，让 agent 的失败**显式暴露**，
    # 而不是被旧管线的成功悄悄掩盖（否则 agentic 的 A/B 根本归因不了）。
    agent_fallback_enabled: bool | None = None

    @property
    def agent_fallback_to_pipeline(self) -> bool:
        """解析后的「是否回退旧管线」（显式配置优先，否则按 environment 推导）"""
        if self.agent_fallback_enabled is not None:
            return self.agent_fallback_enabled
        return self.environment.lower() == "production"


settings = Settings()

_CONFIG_DIR = Path(__file__).resolve().parent


def _load_yaml(name: str) -> dict:
    """读取 core 目录下的 yaml 配置，缺失/解析失败时返回空 dict"""
    path = _CONFIG_DIR / name
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# Prompt 模板（generator/query_decomposer 等使用）
prompts = _load_yaml("prompts.yaml")

# 检索策略配置（RRF 权重/k、单路召回参数等）
retrieval_config = _load_yaml("retrieval.yaml")

__all__ = ["Settings", "settings", "prompts", "retrieval_config"]
