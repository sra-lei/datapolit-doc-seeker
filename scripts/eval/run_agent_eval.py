"""用 AgentRunner（M1 Agentic RAG）跑评估集，产出与 run_local_baseline.py 同构的结果 JSON。

区别只在「谁回答问题」：基线脚本走 ChatService（旧单轮管线，或 agent_enabled 时的 agent 分支），
本脚本**直接驱动 AgentRunner**（ReAct 循环 + 检索工具 + 两段式充分性裁决），并把 agent 特有指标
一起落盘，供 M1 的分层验收使用：

- 收全率：关键词覆盖（与基线同判分口径，可直接对比）
- 拒答正确率：拒答题命中率（判分复用基线脚本的启发式；同时记录 agent 自己的 sufficient 判定）
- 成本/延迟：步数分布、P50/P95 耗时、证据条数

判分**复用 run_local_baseline.py 的 judge / judge_abstain**（同一份实现，避免两套口径漂移），
所以结果 JSON 可以直接喂给 compare_eval_runs.py 做跨轮/跨管线对照。

用法（在仓库根，走本地 venv）：
    LANGFUSE_TRACING_ENABLED=false .venv/bin/python scripts/eval/run_agent_eval.py \
        --cases-file scripts/eval/cases/eval-set-v2.json \
        --out-dir /opt/datapilot-backups/eval-runs --label agent-m1 --workers 2

⚠️ 口径提醒：同口径对比要求 `LLM_TEMPERATURE=0 LLM_DECOMPOSE_TEMPERATURE=0
LLM_GENERATE_MODEL=deepseek-chat`；结果 JSON 的 params 里会写出生效值，对照前先核对。
⚠️ agent 路径每题的 LLM 调用次数是单轮的多倍（决策 + 成文 + 可能的核实），跑批按 workers=2 起，
   别按单轮的并发开。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_CASES = Path(__file__).resolve().parent / "cases" / "eval-set-v2.json"
DEFAULT_OUT = Path("eval-runs")


def _pct(values: list[float], q: float) -> float:
    """分位数（线性取位；对拍批规模足够）"""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cases-file",
        default=str(DEFAULT_CASES),
        help=f"用例 JSON，默认 {DEFAULT_CASES.name}",
    )
    ap.add_argument("--limit", type=int, default=-1, help="限制用例数，默认 -1（全量）")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2, help="agent 每题多轮 LLM 调用，别按单轮的并发开")
    ap.add_argument("--max-steps", type=int, default=None, help="覆盖 settings.agent_max_steps")
    ap.add_argument("--label", default="agent")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    return ap.parse_args()


def load_cases(path: str, limit: int) -> list[dict]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = raw.get("cases") if isinstance(raw, dict) else raw
    assert cases, f"用例为空：检查 {path}"
    return cases[:limit] if limit > 0 else cases


def main() -> int:
    args = parse_args()
    cases = load_cases(args.cases_file, args.limit)

    # 判分复用基线脚本（同一份实现）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_local_baseline as judge_lib  # noqa: E402

    from docs_seeker.agent.runner import AgentRunner  # noqa: E402
    from docs_seeker.api.deps import get_composite_retriever, get_llm_client  # noqa: E402
    from docs_seeker.core.config import settings  # noqa: E402

    if args.max_steps is not None:
        settings.agent_max_steps = args.max_steps

    runner = AgentRunner(llm=get_llm_client(), retriever=get_composite_retriever())

    params = {
        "runner": "AgentRunner(in-process)",
        "cases_file": str(args.cases_file),
        "top_k": args.top_k,
        "workers": args.workers,
        "use_cache": False,
        "judge": "core-eval-replica+abstain-ext",
        "agent_max_steps": settings.agent_max_steps,
        "agent_judge_max_tokens": settings.agent_judge_max_tokens,
        "agent_evidence_char_budget": settings.agent_evidence_char_budget,
        "llm_model": settings.llm_model,
        "llm_generate_model": settings.llm_generate_model or "(=llm_model)",
        "llm_temperature": settings.llm_temperature,
        "llm_decompose_temperature": settings.llm_decompose_temperature,
        "llm_judge_timeout_seconds": settings.llm_judge_timeout_seconds,
        "collection_name": settings.collection_name,
    }
    print(f"用例数: {len(cases)} | top_k={args.top_k} workers={args.workers} max_steps={settings.agent_max_steps}")
    print("LLM/集合参数:", params, flush=True)

    def run_case(case: dict) -> dict:
        t0 = time.time()
        base = {
            "case_id": case["case_id"],
            "question": case["question"],
            "category": case.get("category"),
            "expected_answer_type": "abstain" if case.get("expected_answer_type") == "abstain" else "answer",
        }
        try:
            ar = runner.run(case["question"], top_k=args.top_k)
        except Exception as exc:  # 单题失败不拖垮整批
            return {
                **base,
                "error": f"{type(exc).__name__}: {exc}",
                "answer": "",
                "score": 0.0,
                "passed": False,
                "elapsed": round(time.time() - t0, 2),
            }

        answer = ar.answer or ""
        sources = [c.to_dict() for c in ar.evidence]
        is_abstain = base["expected_answer_type"] == "abstain"
        if is_abstain:
            score, abstain_detail = judge_lib.judge_abstain(answer)
            found, chapter_match = [], None
        else:
            found, score, chapter_match = judge_lib.judge(
                answer, sources, case.get("expected_keywords") or [], case.get("expected_chapter")
            )
        return {
            **base,
            "expected_keywords": case.get("expected_keywords") or [],
            "expected_chapter": case.get("expected_chapter"),
            "keywords_found": found,
            "keyword_count": len(found),
            "chapter_match": chapter_match,
            "abstain_detail": abstain_detail if is_abstain else None,
            "score": score,
            "passed": score >= 0.8,
            "elapsed": round(time.time() - t0, 2),
            "answer": answer,
            "confidence": ar.confidence,
            "source_count": len(sources),
            "agent_steps": len(ar.steps),
            "agent_actions": [s.action for s in ar.steps],
            "agent_sufficient": ar.sufficient,
            "agent_parse_errors": sum(1 for s in ar.steps if s.action == "parse_error"),
            "sources_head": [{"source": s.get("source"), "chapter": s.get("chapter")} for s in sources[:3]],
        }

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = sorted(pool.map(run_case, cases), key=lambda r: r["case_id"])

    passed = sum(1 for r in results if r["passed"])
    avg = sum(r["score"] for r in results) / len(results)
    by_cat: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r.get("category") or "?", []).append(r["score"])
    abst = [r for r in results if r.get("expected_answer_type") == "abstain"]
    steps = [float(r.get("agent_steps") or 0) for r in results]
    elapses = [float(r["elapsed"]) for r in results]

    print(f"\n===== 结果（{len(results)} 题，{time.time() - t_start:.0f}s）")
    print(f"通过 {passed}/{len(results)} = {passed / len(results) * 100:.1f}% | 均分 {avg:.3f}")
    print(
        f"平均耗时 {sum(elapses) / len(elapses):.1f}s | P50 {_pct(elapses, 0.5):.1f}s | "
        f"P95 {_pct(elapses, 0.95):.1f}s | 空答案 {sum(1 for r in results if not r['answer'])} 条"
    )
    print(
        f"平均步数 {sum(steps) / len(steps):.1f} | 动作解析失败题 {sum(1 for r in results if r.get('agent_parse_errors'))} 条"
    )
    if abst:
        ok = sum(1 for r in abst if r["passed"])
        print(f"拒答正确率: {ok}/{len(abst)}（判分口径同基线）")
        agent_ok = sum(1 for r in abst if r.get("agent_sufficient") is False)
        print(f"  其中 agent 自己判 insufficient: {agent_ok}/{len(abst)}（两段式裁决的终态）")
    for cat, scores in sorted(by_cat.items()):
        print(f"  {cat}: n={len(scores)} 均分={sum(scores) / len(scores):.3f}")
    for r in results:
        if r.get("expected_answer_type") == "abstain":
            tag = f"abstain={r.get('abstain_detail') or 'HALLUCINATED'} suff={r.get('agent_sufficient')}"
        else:
            tag = f"kw={r.get('keyword_count')}/{len(r.get('expected_keywords') or [])} chap={r.get('chapter_match')}"
        print(
            f"  {'✅' if r['passed'] else '❌'} {r['case_id']} {r['score']:.2f} {tag} "
            f"steps={r.get('agent_steps')} src={r.get('source_count')} {r['elapsed']:5.1f}s | {r['question'][:24]}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(
        json.dumps(
            {
                "label": args.label,
                "params": params,
                "total": len(results),
                "passed": passed,
                "avg_score": round(avg, 4),
                "abstain_correct": sum(1 for r in abst if r["passed"]),
                "abstain_total": len(abst),
                "category_stats": {c: {"n": len(s), "avg": round(sum(s) / len(s), 4)} for c, s in by_cat.items()},
                "agent_stats": {
                    "avg_steps": round(sum(steps) / len(steps), 2),
                    "p50_elapsed": round(_pct(elapses, 0.5), 2),
                    "p95_elapsed": round(_pct(elapses, 0.95), 2),
                    "empty_answers": sum(1 for r in results if not r["answer"]),
                    "errors": sum(1 for r in results if r.get("error")),
                },
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\n已保存:", path)
    print("提醒：单轮噪声大，结论只取跨轮稳定方向；与单轮管线对照用 compare_eval_runs.py。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
