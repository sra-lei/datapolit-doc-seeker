"""本机跑 InsightForge(docs-seeker) 评估集基线——复刻 core 的判分口径。

用途：本机改了代码但没重建线上镜像时，用它出数据（core 的「跑评估」打的是线上服务，测不出本地改动）。

用法（在 docs-seeker 仓库根）：
    PYTHONPATH=src LANGFUSE_TRACING_ENABLED=false \
        .venv/bin/python <skill:datapilot-rag-platform>/scripts/run_local_baseline.py \
        [--set-id 1] [--top-k 10] [--workers 4] [--label local]

⚠️ 温度必须置 0（`LLM_TEMPERATURE=0 LLM_DECOMPOSE_TEMPERATURE=0`）：默认 0.3 时，
同代码同语料的单题分数摆幅达 0.75，噪声大于待测改动的量级
（见 references/eval-noise-and-baseline.md）。生效的 LLM 参数会打印并写进结果 JSON
的 params 字段，跨轮对照时可核对口径。

前置：项目根 `.env` 就位（Milvus 可达）；本地建议 `SEMANTIC_CACHE_ENABLED=false`，别把跑批流量引进生产 Redis。

判分口径逐行复刻 core `dist/modules/eval/service.js`（不要「改进」它，否则无法与历史 run 对照）：
    found = [kw for kw in expected_keywords if kw in answer]      # 纯子串，无归一化
    score = len(found) / max(1, len(expected_keywords))
    if expected_chapter and not (expected_chapter in answer or
            any(expected_chapter in (s.get('chapter') or '') for s in sources)):
        score *= 0.7
    if not answer: score = 0
    passed = score >= 0.8

请求口径与 core 一致：`{question, use_cache: false, stream: false}`（不传 top_k → 默认 10）。
结果落盘 /opt/datapilot-backups/eval-runs/<label>-<时间戳>.json，供跨轮对照（本文件同目录 compare_eval_runs.py）。
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CORE = 'http://127.0.0.1:3002/core/eval'
# 默认输出到仓库下 eval-runs/（跨平台、随仓库走）；
# 云端跑批习惯落到仓库外：--out-dir /opt/datapilot-backups/eval-runs
DEFAULT_OUT = Path('eval-runs')


def judge(answer: str, sources: list[dict], expected_kw: list[str], expected_chapter: str | None):
    found = [kw for kw in expected_kw if kw in answer]
    score = len(found) / max(1, len(expected_kw))
    chapter_match = None
    if expected_chapter:
        chapter_match = expected_chapter in answer or any(
            expected_chapter in str(s.get('chapter') or '') for s in sources
        )
        if not chapter_match:
            score *= 0.7
    if not answer:
        score = 0.0
    return found, round(score, 4), chapter_match


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--set-id', type=int, default=1)
    ap.add_argument('--cases-file', default=None,
                    help='本地用例 JSON（不依赖 core API；离线/本机跑批用）。格式：[{case_id,question,expected_keywords,expected_chapter}, ...]')
    ap.add_argument('--top-k', type=int, default=10)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--label', default='local')
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT), help='结果目录，默认 ./eval-runs（云端习惯 --out-dir /opt/datapilot-backups/eval-runs）')
    args = ap.parse_args()

    if args.cases_file:
        raw = json.loads(Path(args.cases_file).read_text(encoding='utf-8'))
        cases = raw.get('cases') if isinstance(raw, dict) else raw
        if isinstance(raw, dict) and not cases and isinstance(raw.get('data'), dict):
            cases = raw['data'].get('cases')
    else:
        import requests  # 只在走 core API 时需要
        cases = requests.get(f'{CORE}/sets/{args.set_id}', timeout=20).json()['data']['cases']
    assert cases, '用例为空：检查 --cases-file 或 core 服务'
    print(f'用例数: {len(cases)} | top_k={args.top_k} workers={args.workers}', flush=True)

    from docs_seeker.api.deps import get_chat_service  # 延迟 import：需 PYTHONPATH=src
    from docs_seeker.core.config import settings

    service = get_chat_service()

    # 把生效的 LLM 参数写进结果 JSON：A/B 对比时可核对口径是否一致
    # （温度默认 0.3；评估必须 LLM_TEMPERATURE=0 LLM_DECOMPOSE_TEMPERATURE=0）
    llm_params = {
        'llm_model': settings.llm_model,
        'llm_generate_model': settings.llm_generate_model or '(=llm_model)',
        'llm_temperature': settings.llm_temperature,
        'llm_decompose_temperature': settings.llm_decompose_temperature,
        'llm_generate_max_tokens': settings.llm_generate_max_tokens,
        'llm_decompose_max_tokens': settings.llm_decompose_max_tokens,
        'llm_timeout_seconds': settings.llm_timeout_seconds,
        'collection_name': settings.collection_name,
    }
    print('LLM/集合参数:', llm_params, flush=True)

    def run_case(case: dict) -> dict:
        t0 = time.time()
        try:
            result = service.chat(case['question'], top_k=args.top_k, use_cache=False)
            answer, sources = result.answer or '', result.sources or []
        except Exception as exc:  # 单题失败不拖垮整批
            return {'case_id': case['case_id'], 'question': case['question'], 'category': case.get('category'),
                    'error': f'{type(exc).__name__}: {exc}', 'answer': '', 'sources': [],
                    'score': 0.0, 'passed': False, 'elapsed': round(time.time() - t0, 2)}
        found, score, chapter_match = judge(
            answer, sources, case.get('expected_keywords') or [], case.get('expected_chapter')
        )
        return {
            'case_id': case['case_id'], 'question': case['question'], 'category': case.get('category'),
            'expected_keywords': case.get('expected_keywords') or [],
            'expected_chapter': case.get('expected_chapter'),
            'keywords_found': found, 'keyword_count': len(found), 'chapter_match': chapter_match,
            'source_count': len(sources), 'score': score, 'passed': score >= 0.8,
            'elapsed': round(time.time() - t0, 2), 'answer': answer,
            'sources_head': [{'source': s.get('source'), 'chapter': s.get('chapter')} for s in sources[:3]],
            'sub_questions': result.query_decomposed,
        }

    t_start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = sorted(pool.map(run_case, cases), key=lambda r: r['case_id'])

    passed = sum(1 for r in results if r['passed'])
    avg = sum(r['score'] for r in results) / len(results)
    by_cat: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r.get('category') or '?', []).append(r['score'])

    print(f'\n===== 结果（{len(results)} 题，{time.time() - t_start:.0f}s）')
    print(f'通过 {passed}/{len(results)} = {passed / len(results) * 100:.1f}% | 均分 {avg:.3f}')
    print(f'平均耗时 {sum(r["elapsed"] for r in results) / len(results):.1f}s | '
          f'空答案 {sum(1 for r in results if not r["answer"])} 条')
    for cat, scores in sorted(by_cat.items()):
        print(f'  {cat}: n={len(scores)} 均分={sum(scores) / len(scores):.3f}')
    for r in results:
        print(f"  {'✅' if r['passed'] else '❌'} {r['case_id']} {r['score']:.2f} "
              f"kw={r.get('keyword_count')}/{len(r.get('expected_keywords') or [])} "
              f"chap={r.get('chapter_match')} src={r.get('source_count')} {r['elapsed']:5.1f}s | {r['question'][:26]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.label}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps({
        'label': args.label,
        'params': {'set_id': args.set_id, 'top_k': args.top_k, 'workers': args.workers,
                   'use_cache': False, 'stream': False, 'judge': 'core-eval-replica',
                   **llm_params},
        'total': len(results), 'passed': passed, 'avg_score': round(avg, 4),
        'category_stats': {c: {'n': len(s), 'avg': round(sum(s) / len(s), 4)} for c, s in by_cat.items()},
        'results': results,
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n已保存:', path)
    print('提醒：单轮噪声大（见 references/eval-noise-and-baseline.md），结论只取跨轮稳定方向。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
