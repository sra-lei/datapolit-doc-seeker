"""跨轮对照 InsightForge 评估结果（本地跑批 JSON 与 core 的历史 run JSON 都吃）。

用法：
    .venv/bin/python <本脚本> <run_a.json> <run_b.json> [<run_c.json> ...]
    # 本地跑批产物：/opt/datapilot-backups/eval-runs/*.json（含答案全文）
    # 线上历史 run：curl -s http://127.0.0.1:3002/core/eval/runs/<id> > run7.json

输出：逐题分数矩阵 + 摆幅标记 + 各轮总分/通过数/均分。
判读规则（见 references/eval-noise-and-baseline.md）：**摆幅 ≥0.4 的题视为抖动，不用于结论**；
单轮总分差 < 1 题（≈4.5pp）时不下结论。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load_scores(path: Path) -> tuple[str, dict[str, float]]:
    """吃两种格式：本脚本同目录 run_local_baseline.py 的产物 / core 的 /core/eval/runs/<id> 响应。"""
    data = json.loads(path.read_text())
    if isinstance(data, dict) and 'results' in data:  # 本地跑批
        return data.get('label') or path.stem, {r['case_id']: float(r.get('score') or 0) for r in data['results']}
    run = data.get('data', data)  # core run 响应
    return f"{run.get('set_name', 'run')}#{run.get('id')}", {
        c['case_id']: float(c.get('score') or 0) for c in run['cases']
    }


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    series = [(name, scores) for name, scores in (load_scores(Path(p)) for p in sys.argv[1:])]
    ids = sorted({cid for _, s in series for cid in s})

    head = f"{'题号':8s}" + ''.join(f'{name[:18]:>20s}' for name, _ in series) + f"{'摆幅':>8s}"
    print(head)
    volatile = []
    for cid in ids:
        row = [s.get(cid) for _, s in series]
        vals = [v for v in row if v is not None]
        span = max(vals) - min(vals) if vals else 0
        if span >= 0.4:
            volatile.append(cid)
        cells = ''.join(f'{v:20.2f}' if v is not None else f"{'—':>20s}" for v in row)
        print(f'{cid:8s}{cells}{span:8.2f}' + ('  ⚠️' if span >= 0.4 else ''))

    print()
    for name, scores in series:
        n = len(scores)
        print(f'{name[:24]:26s} 通过={sum(1 for v in scores.values() if v >= 0.8)}/{n} '
              f'均分={sum(scores.values()) / n:.3f}')
    print(f'\n抖动题（≥0.4，不用于结论）: {volatile or "无"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
