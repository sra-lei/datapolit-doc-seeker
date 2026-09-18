"""单题延迟分解探针：同一个问题分别走「单轮管线」与「AgentRunner」，按调用点打印耗时。

用途：回答「agent 到底为什么更快 / 慢」——是**调用次数**少了，还是**单次调用**更快？
只跑 1 题（默认 T01），用于延迟归因，不做质量评估（质量看 run_agent_eval.py）。

⚠️ 必须先预热检索：首次检索要现场构建 BM25 索引（jieba 分词 1055 篇文档，几十秒级），
   冷启动成本会全额算到先跑的那条路径上 —— 不预热就会得出完全错误的归因。

用法：
    LANGFUSE_TRACING_ENABLED=false .venv/bin/python scripts/eval/probe_latency_breakdown.py
    # 换题：--question "..."；换 top_k：--top-k 10
"""

from __future__ import annotations

import argparse
import time

from docs_seeker.api.deps import get_llm_client
from docs_seeker.config.settings import settings
from docs_seeker.domain.models.llm import LLMRequest
from docs_seeker.domain.services.generator import Generator
from docs_seeker.domain.services.query_decomposer import QueryDecomposer
from docs_seeker.domain.services.rag_pipeline import RAGPipeline

DEFAULT_Q = "我准备注册亚马逊卖家账户，身份验证时对上传的营业执照和法人身份证照片有什么具体要求？"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=DEFAULT_Q)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    client = get_llm_client()
    retriever = _retriever()

    llm_calls: list[dict] = []
    search_calls: list[float] = []

    # ---- 预热：把 BM25 建索引 / 连接建立等一次性成本挡在计时之外 ----
    t0 = time.time()
    retriever.search("预热", top_k=1)
    print(f"预热（首次检索，含 BM25 建索引）：{time.time() - t0:.1f}s —— 不计入对照\n")

    original_generate = client.generate

    def timed_generate(request: LLMRequest):
        t = time.time()
        try:
            return original_generate(request)
        finally:
            llm_calls.append(
                {
                    "name": request.name,
                    "model": request.model or f"(={settings.llm_model})",
                    "max_tokens": request.max_tokens,
                    "timeout": request.timeout if request.timeout is not None else settings.llm_timeout_seconds,
                    "elapsed": round(time.time() - t, 2),
                }
            )

    original_search = retriever.search

    def timed_search(*a, **kw):
        t = time.time()
        try:
            return original_search(*a, **kw)
        finally:
            search_calls.append(round(time.time() - t, 2))

    client.generate = timed_generate
    retriever.search = timed_search  # ⚠️ 别只定义不挂：上一版就是漏了这行，检索耗时全被算成「其余」

    def _reset() -> None:
        llm_calls.clear()
        search_calls.clear()

    def _report(tag: str, total: float, extra: str = "") -> None:
        llm_sum = sum(c["elapsed"] for c in llm_calls)
        search_sum = sum(search_calls)
        print(
            f"【{tag}】总 {total:.1f}s | LLM {len(llm_calls)} 次 小计 {llm_sum:.1f}s | "
            f"检索 {len(search_calls)} 次 小计 {search_sum:.1f}s | 其余 {total - llm_sum - search_sum:.1f}s{extra}"
        )
        for c in llm_calls:
            print(
                f"   · llm  {c['name']:20} {c['model']:16} max_tokens={c['max_tokens']:>5} timeout={c['timeout']:>5} {c['elapsed']:5.1f}s"
            )
        for i, s in enumerate(search_calls, 1):
            print(f"   · 检索 #{i}  {s:5.1f}s")

    print(f"问题：{args.question}\n")

    # ---- 1) 单轮管线（热）----
    _reset()
    pipeline = RAGPipeline(
        retriever=retriever,
        decomposer=QueryDecomposer(llm=client),
        generator=Generator(llm=client),
    )
    t0 = time.time()
    _answer, _conf, _chunks, sub_questions = pipeline.run(args.question, top_k=args.top_k)
    single_total = time.time() - t0
    _report("单轮管线", single_total, f" | 子问题 {len(sub_questions)} 个")

    # ---- 2) AgentRunner（热）----
    from docs_seeker.agent.runner import AgentRunner

    _reset()
    runner = AgentRunner(llm=client, retriever=retriever)
    t0 = time.time()
    result = runner.run(args.question, top_k=args.top_k)
    agent_total = time.time() - t0
    _report("AgentRunner", agent_total, f" | 步数 {len(result.steps)} 证据 {len(result.evidence)}")

    print(f"\n差值：{single_total - agent_total:+.1f}s（正数 = agent 更快）")
    return 0


def _retriever():
    """复用服务同一份检索器（含 BM25 索引）"""
    from docs_seeker.api.deps import get_composite_retriever

    return get_composite_retriever()


if __name__ == "__main__":
    raise SystemExit(main())
