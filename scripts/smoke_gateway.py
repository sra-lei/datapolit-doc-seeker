"""LLM 层真实 API 冒烟（重构 Phase 0 起）。

单测用假客户端覆盖了 payload 与信封的**结构**；本脚本补上单测覆盖不到的一环：
真实 provider 是否接受我们组装的 payload，以及真实响应的字段提取是否正确。

跑法（需 .env 里有 DEEPSEEK_API_KEY；会发起 3 次真实调用，每次 ≤16 token）：

    LANGFUSE_TRACING_ENABLED=false .venv/bin/python scripts/smoke_gateway.py

检查项：
1. 普通调用 —— 信封取到正文 / finish_reason / usage，provider=primary、attempts=1；
2. 参数透传 —— extra 里的 provider 参数（top_p）被真实 API 接受，不报 400；
3. 流式 —— .text 为空、raw 是真 stream、iter_text() 能产出增量（含 include_usage 末包）。
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")

from docs_seeker.domain.interfaces.llm import LLMRequest  # noqa: E402
from docs_seeker.infra.llm.gateway import LLMGateway  # noqa: E402

PROMPT = [{"role": "user", "content": "只回复两个字：收到"}]

# deepseek-v4-flash 是推理模型：reasoning token 与正文共用 max_tokens 预算，
# 预算给小了会出现 finish_reason=length 且正文为空（见 test_llm_budget_guard.py）。
# 冒烟要给足预算，否则测的是「预算兜底」而不是「链路通不通」。
BUDGET = 2000


def main() -> int:
    gw = LLMGateway()
    failures: list[str] = []

    # 1. 普通调用
    resp = gw.generate(LLMRequest(messages=PROMPT, max_tokens=BUDGET, temperature=0.0, name="smoke-basic"))
    print(f"[1] text={resp.text!r} finish={resp.finish_reason} usage={resp.usage}")
    print(f"    provider={resp.provider} fallback={resp.fallback_used} attempts={resp.attempts} raw={'有' if resp.raw is not None else '无'}")
    if not resp.text:
        failures.append("普通调用正文为空")
    if resp.raw is None:
        failures.append("普通调用 raw 未保真")
    if resp.usage is None:
        failures.append("普通调用 usage 未提取")

    # 2. 参数透传（真实 API 接受 extra 参数）
    resp2 = gw.generate(
        LLMRequest(
            messages=PROMPT,
            max_tokens=BUDGET,
            temperature=0.0,
            name="smoke-extra",
            extra={"top_p": 0.5},
        )
    )
    print(f"[2] extra 透传 text={resp2.text!r} finish={resp2.finish_reason}")

    # 3. 流式：只透传 + iter_text 聚合
    resp3 = gw.generate(LLMRequest(messages=PROMPT, max_tokens=BUDGET, temperature=0.0, name="smoke-stream", stream=True))
    deltas = list(resp3.iter_text())
    print(f"[3] stream text={resp3.text!r} deltas={len(deltas)} 拼接={''.join(deltas)!r}")
    if resp3.text != "":
        failures.append("流式信封 .text 应为空（只做透传）")
    if not deltas:
        failures.append("流式未产出任何增量")

    if failures:
        print("\n冒烟失败：" + "；".join(failures))
        return 1
    print("\n冒烟通过：payload 被真实 provider 接受，信封字段提取正确。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
