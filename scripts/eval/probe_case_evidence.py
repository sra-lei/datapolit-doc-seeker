"""单题证据链探针：定位「关键词没进答案」断在哪一环。

链路四环（按顺序查）：
  ① 全量证据里有没有该词（检索召回层）
  ② **成文输入**里有没有该词（`_format_evidence` 的 12000 字预算会截断证据！）
  ③ 最终答案里有没有该词（成文模型层）
  ④ agent 的决策步都检索了什么（少了哪些检索）

判读：
  - ①无 → 检索/决策没召回（agent 少检索的直接后果）
  - ①有 ②无 → **证据被字符预算挤掉**（成文输入裁剪问题，跟检索无关）
  - ①②有 ③无 → 成文模型没写出来（提示/覆盖面问题）

用法：
    LANGFUSE_TRACING_ENABLED=false .venv/bin/python scripts/eval/probe_case_evidence.py --case-id T15
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from docs_seeker.core.config import settings

DEFAULT_CASES = Path(__file__).resolve().parent / "cases" / "eval-set-v2.json"


def _locate(answer: str, keyword: str) -> str:
    """在答案里找该关键词的**最长可见片段**，返回上下文。

    用来区分两种丢分：**真没写**（连两字片段都找不到）vs **换了说法**（片段在但没连成完整子串）。
    """
    for length in range(len(keyword), 1, -1):
        for start in range(0, len(keyword) - length + 1):
            frag = keyword[start : start + length]
            pos = answer.find(frag)
            if pos >= 0:
                return f"可见片段「{frag}」｜原文：{answer[max(0, pos - 70) : pos + 170]}"
    return "答案里找不到该词的任何两字以上片段 → 真的没写"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", default="T15")
    ap.add_argument("--cases-file", default=str(DEFAULT_CASES))
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--repeat", type=int, default=1, help="agent 路径重复跑几次（同一进程，共享热索引）")
    args = ap.parse_args()

    raw = json.loads(Path(args.cases_file).read_text(encoding="utf-8"))
    cases = raw.get("cases") if isinstance(raw, dict) else raw
    case = next((c for c in cases if c["case_id"] == args.case_id), None)
    assert case, f"用例里没有 {args.case_id}"

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import run_local_baseline as judge_lib  # 与判分同一套归一化

    from docs_seeker.agent.runner import AgentRunner
    from docs_seeker.api.deps import get_composite_retriever
    from docs_seeker.domain.services.generator import Generator
    from docs_seeker.domain.services.query_decomposer import QueryDecomposer
    from docs_seeker.domain.services.rag_pipeline import RAGPipeline
    from docs_seeker.infra.llm.client import get_llm_client

    norm = judge_lib._norm
    keyword_list = case.get("expected_keywords") or []
    is_abstain = case.get("expected_answer_type") == "abstain"
    question = case["question"]

    gw = get_llm_client()
    retriever = get_composite_retriever()
    retriever.search("预热", top_k=1)  # 挡掉 BM25 建索引的冷启动

    print(f"用例 {case['case_id']}：{question}")
    print(f"期望关键词：{keyword_list}\n")

    def check(tag: str, evidence_texts: list[str], compose_input: str, answer: str) -> None:
        print(f"【{tag}】")
        if is_abstain:
            score, detail = judge_lib.judge_abstain(answer)
            print(
                f"   拒答判定：{'✅ 正确拒答' if score >= 0.8 else '❌ 硬编了（该拒答却作答）'} | detail={detail} | "
                f"答案 {len(answer)} 字"
            )
            print(f"   答案开头：{answer[:180]!r}")
            print(f"   证据 {len(evidence_texts)} 条 | 成文输入 {len(compose_input)} 字")
            return
        for kw in keyword_list:
            in_ev = any(norm(kw) in norm(t) for t in evidence_texts)
            in_ci = norm(kw) in norm(compose_input)
            in_an = norm(kw) in norm(answer)
            verdict = (
                "✅ 进答案"
                if in_an
                else (
                    "⚠️ 证据被成文预算挤掉"
                    if (in_ev and not in_ci)
                    else ("⚠️ 成文没写出来" if in_ci else "❌ 压根没召回")
                )
            )
            print(
                f"   {kw[:18]:20} ①证据 {'有' if in_ev else '无'} | ②成文输入 {'有' if in_ci else '无'} | "
                f"③答案 {'有' if in_an else '无'}  -> {verdict}"
            )
        print(f"   答案长度 {len(answer)} 字")

    # ---- 单轮管线 ----
    t0 = time.time()
    st_answer, _conf, st_chunks, st_subs = RAGPipeline(
        retriever=retriever,
        decomposer=QueryDecomposer(llm=gw),
        generator=Generator(llm=gw),
    ).run(question, top_k=args.top_k)
    st_elapsed = time.time() - t0
    st_texts = [c.text for c in st_chunks]
    # 单轮的成文输入等价于 generator._build_messages 里的拼接（不截断）
    st_compose = "\n".join(st_texts)
    print(f"（单轮 {st_elapsed:.1f}s，子问题 {len(st_subs)} 个，去重后证据 {len(st_chunks)} 条）")
    check("单轮管线", st_texts, st_compose, st_answer)

    # ---- AgentRunner ----
    t0 = time.time()
    ar = AgentRunner(llm=gw, retriever=retriever).run(question, top_k=args.top_k)
    agent_elapsed = time.time() - t0
    ag_texts = [c.text for c in ar.evidence]
    # 关键：成文输入是 _format_evidence 的结果（受 agent_evidence_char_budget 截断）
    ag_compose = AgentRunner._format_evidence(list(ar.evidence))
    print(
        f"\n（agent {agent_elapsed:.1f}s，步数 {len(ar.steps)}，证据 {len(ar.evidence)} 条，成文输入 {len(ag_compose)} 字 / 预算 {settings.agent_evidence_char_budget}）"
    )
    print("   决策轨迹：")
    for s in ar.steps:
        print(
            f"     #{s.idx} {s.action:9} {str(s.action_input.get('query', ''))[:46]!r} obs={len(s.observation or '')}字"
        )
    check("AgentRunner", ag_texts, ag_compose, ar.answer)

    # 被预算挤掉的证据（②无但①有）逐条定位
    dropped = [
        c
        for c in ar.evidence
        if any(norm(kw) in norm(c.text) for kw in keyword_list) and norm(c.text) not in norm(ag_compose)
    ]
    if dropped:
        print(
            f"\n⚠️ 含答案词但没进成文输入的证据 {len(dropped)} 条（被 {settings.agent_evidence_char_budget} 字预算丢弃）："
        )
        for c in dropped[:3]:
            print(f"   - {c.chapter}/{c.article} {str(c.source)[:40]} | {c.text[:90]}...")

    print("\n---- 答案原文（前 700 字）----")
    print("单轮:", st_answer[:700])
    print("\nagent:", ar.answer[:700])

    # ---- 重复跑：量化单题波动（复现性检查）----
    if args.repeat > 1:
        print(f"\n---- agent 路径重复跑 {args.repeat} 次（同一进程，热索引）----")
        hits = 0
        for i in range(args.repeat):
            t0 = time.time()
            r = AgentRunner(llm=gw, retriever=retriever).run(question, top_k=args.top_k)
            el = time.time() - t0
            compose_input = AgentRunner._format_evidence(list(r.evidence))
            if is_abstain:
                score, detail = judge_lib.judge_abstain(r.answer)
                ok = score >= 0.8
                print(
                    f"   第 {i + 1} 次: {'✅ 正确拒答' if ok else '❌ 硬编了'} | 步数 {len(r.steps)} 证据 {len(r.evidence)} | "
                    f"{el:4.1f}s | sufficient={r.sufficient} | detail={detail}"
                )
                if not ok:
                    print(f"      ↳ 硬编答案开头：{r.answer[:220]!r}")
                hits += ok
                continue
            per_kw = {kw: norm(kw) in norm(r.answer) for kw in keyword_list}
            ok = all(per_kw.values())
            hits += ok
            # 逐词定位断点：证据 → 成文输入（受预算截断）→ 答案
            where = {}
            for kw in keyword_list:
                if not per_kw[kw]:
                    in_ev = any(norm(kw) in norm(c.text) for c in r.evidence)
                    in_ci = norm(kw) in norm(compose_input)
                    where[kw] = "没召回" if not in_ev else ("成文预算截断" if not in_ci else "成文没写")
            print(
                f"   第 {i + 1} 次: {'✅ 全中' if ok else '❌ 漏词'} | 步数 {len(r.steps)} 证据 {len(r.evidence)} | {el:4.1f}s | "
                f"答案 {sum(per_kw.values())}/{len(per_kw)}" + ("" if ok else f" | 断点: {where}")
            )
            if not ok:
                # 失败样本的措辞：分辨「真没写」vs「换了说法」（判分是纯子串，后者同样丢分）
                for kw in keyword_list:
                    if not per_kw[kw]:
                        print(f"      ↳ 漏「{kw}」：{_locate(r.answer, kw)}")
        print(f"   → 全中 {hits}/{args.repeat}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
