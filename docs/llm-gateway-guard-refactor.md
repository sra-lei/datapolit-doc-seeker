# LLM Gateway 重构方案（透明传输 + 可插拔 Middleware）

> 状态：**已定稿**（2026-09-17，用户评审通过）。
> 范围：`services/docs-seeker` 的 LLM 调用层（`infra/llm/`、`domain/interfaces/llm.py`）与安全护栏（`core/security.py`、`domain/services/chat_service.py`）。
>
> **评审已决（2026-09-17）**：
> 1. 先落本方案文档并推送 → 评审通过后再开工（本文即交付物）；
> 2. `LLMResponse` 信封**直接启用，一步到位**，不设 `LLM_RETURN_ENVELOPE` 兼容开关（接受改动面扩散，换代码干净）；
> 3. 检索文档正文命中注入指令时**仅记录日志告警，不干预答案**（可用性优先）。

## 0. 一句话

现有 `LLMGateway` 把「传输」和「策略」揉在一个类里，结果同时违反三条底线：
**① 限制了能传给 LLM 的参数；② 降维/吞掉了 LLM 的真实返回；③ 上层被迫感知 OpenAI 结构与网关策略。**
重构目标 = **薄透明传输层（参数原样透传、返回原样保真）+ 洋葱式 Middleware 链（重试/熔断/降级/预算/护栏全部变成可插拔插件）**。

---

## 1. 现状与问题

### 1.1 限制了传递给 LLM 的参数

| # | 位置 | 问题 |
|---|---|---|
| P1 | `infra/llm/gateway.py:132-143` | `call_kwargs` 是**硬编码白名单**，只有 `model / messages / max_tokens / temperature / stream / timeout / name` 7 个键。`top_p`、`seed`、`stop`、`response_format`、`tools` / `tool_choice`、`logprobs`、`reasoning_effort`、`extra_body` …… 一个都传不进去。 |
| P2 | `domain/interfaces/llm.py:9-23` | `LLMRequest` 字段封闭，没有逃生舱。**每加一个参数要改 dataclass + 所有鸭子类型测试替身**（`ScriptedLLM` / `RecordingLLM` / `_ScriptedGenerator`），历史上加一个 `model=` 就红了 7 项用例。 |
| P3 | `gateway.py:142` | `"name": request.name` 被塞进 `chat.completions.create(**kwargs)` —— 这是 `langfuse.openai` 的封装参数，**框架参数和 provider 参数混在同一个命名空间**，换掉 Langfuse 包装就会 400。 |
| P4 | `gateway.py:144-147` | `stream_options={"include_usage": True}` 被**无条件注入**且仅流式注入，调用方无法关闭、也无法在非流式下指定。 |
| P5 | `gateway.py:132-143` | `max_tokens` / `temperature` / `stream` **恒被显式下发**，调用方无法表达「用服务端默认值」。`None` 应该 = 不下发该字段。 |

### 1.2 吞掉了 LLM 的真实返回

| # | 位置 | 问题 |
|---|---|---|
| R1 | `generator.py:146-157` | `_extract_answer` 只取 `choices[0].message.content` 和 `finish_reason`，`usage` / `reasoning_content` / `tool_calls` / `id` / `model` **全部丢弃**。 |
| R2 | `generator.py:94-122` | 到应用层只剩 `(answer, confidence)` 元组 —— 真实响应在「网关 → Generator」这一跳就被降维成字符串。 |
| R3 | `gateway.py:105-115` | **降级是静默的**：主模型失败改用备用模型时，上层完全不知道这次答案是备用模型产出的（`stats.fallback_calls` 只有聚合计数，单次调用无标记）。RAG 场景下这直接让评估无法归因。 |
| R4 | `gateway.py:158` | `raise last_error from None` —— **主动抹掉异常链**；`AllModelsFailedError("主模型和备用模型均失败")` 里主模型的真实异常对象只进了日志。 |
| R5 | `gateway.py:124-158` | 重试**不区分错误类型**：401/400（鉴权 / 参数错）也重试 3 次 + 指数退避，白等 7 秒才失败。 |
| R6 | `gateway.py:83-115` | 熔断状态手工维护（`self.circuit_breaker.failure_count = 0` 直接改字段、绕过锁），与 `CircuitBreaker` 类自身语义不一致。 |

### 1.3 对上层不是无感的

| # | 位置 | 问题 |
|---|---|---|
| T1 | `llm.py:30` | 抽象接口返回类型是 `Any` —— 契约空洞，替身各写各的。 |
| T2 | `generator.py:152`、`query_decomposer.py:40`、`agent/runner.py:98` | 上层直接解 `response.choices[0].message.content`，**绑定 OpenAI 结构**。换 SDK / 加一层 provider 适配要改所有上层。 |
| T3 | `gateway.py:150` | `stream=True` 时直接返回原生 stream 对象，应用层靠 `request.stream` 分支自行判断拿到的是 response 还是迭代器。 |
| T4 | `gateway.py:175-182` | 全局单例 `get_llm_gateway()` 在构造时读死 `settings`，测试隔离困难。 |
| T5 | `generator.py:68-92` + `query_decomposer.py:53-62` | **同一段「截断空返回 → 放大预算重试一次」逻辑抄了两遍**（约 15 行 ×2），agent runner 里还有第三套超时/预算口径。正交关注点散落在四处。 |

### 1.4 Guard 不是 Middleware

| # | 位置 | 问题 |
|---|---|---|
| G1 | `chat_service.py:74,122,171,229` + `top_warmup.py:80` | `check_injection` / `sanitize_output` 在**三个地方手工复制调用**（chat / chat_stream / warmup）。加一条护栏要人肉记得改所有入口；`top_warmup.py:8-9` 的注释「与 chat 路径保持一致的格式」本身就是坏味道。 |
| G2 | `core/security.py:35` | 护栏**只看原始 question**。RAG 里真正的注入通道是**检索回来的文档内容**——文档里写「忽略上述指令」现在完全不会被拦。 |
| G3 | `core/security.py:9-53` | 规则表硬编码为模块级列表，无法按环境/租户配置、无法关闭、无法排序。 |
| G4 | — | 无命中记录、无耗时、无短路语义、无 `enabled` 开关，观测上完全黑盒。 |
| G5 | `gateway.py:64` | `CircuitBreaker.call()` 定义完整却**零调用点**（死代码）。 |

---

## 2. 目标契约

### 2.1 请求：强类型常用字段 + 逃生舱

```python
@dataclass
class LLMRequest:
    messages: list[dict]

    # —— provider 参数（常用项保留强类型，便于发现与校验）——
    model: str | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False
    timeout: float | None = None

    # —— 逃生舱：任意 provider 参数原样透传，不做白名单 ——
    # 例：{"top_p": 0.9, "seed": 42, "response_format": {"type": "json_object"}}
    extra: dict[str, Any] = field(default_factory=dict)

    # —— 框架参数：绝不进入 provider payload ——
    # 例：{"name": "generate-response", "tags": ["eval"], "guard_policy": "strict"}
    meta: dict[str, Any] = field(default_factory=dict)
```

**透传规则（决定性的一条）：**

```
payload = {k: v for k, v in (model, messages, max_tokens, temperature, stream, timeout) if v is not None}
payload.update(extra)                      # 逃生舱可覆盖常用字段（显式覆盖，非静默）
payload.update(provider_specific_meta)     # 由 middleware 决定，如 langfuse 的 name/stream_options
```

- **不在白名单里的参数原样到达 SDK**；新参数不需要改 dataclass。
- `None` = 不下发该字段（让 provider 用默认值）。
- `meta` 里的键**永不**进入 payload —— 框架参数与 provider 参数彻底分离（修 P3）。
- `stream_options` 不再硬编码，改由 `ObservabilityMiddleware` 在 `meta` 明确要求时注入（修 P4）。

### 2.2 返回：保真信封，不降维

```python
@dataclass
class LLMResponse:
    text: str                    # 便捷字段（choices[0] 正文）
    raw: Any                     # ★ 原始 provider 响应对象，原样保留，绝不丢
    finish_reason: str | None
    usage: dict | None
    model: str | None
    reasoning: str | None
    tool_calls: list | None
    # —— 调用元信息（策略可见性）——
    provider: str                # 实际服务的 provider
    fallback_used: bool          # ★ 是否走了降级
    attempts: int
    latency_ms: int
    applied_middlewares: list[str]   # ★ 本次调用实际生效的 middleware，可审计
    trace_id: str | None
```

- 上层默认用 `.text` / `.finish_reason`；**要什么有什么**，`raw` 兜底。
- 流式：`generate()` 同样返回 `LLMResponse`，`.raw` 是原生 stream 迭代器，**`.text` 置空、只做透传**（评审已决：不在信封层聚合全文，聚合交给调用方）。
- **降级可见**（修 R3）：`fallback_used=True` + `provider` 如实上报，评估脚本可据此剔除/分组。
- **启用方式**：信封**直接生效，不设兼容开关**（评审已决）；所有调用点与鸭子类型替身一次性迁移到 `LLMResponse`。

### 2.3 错误模型

```python
class LLMError(Exception):
    attempts: int
    provider_errors: list[tuple[str, Exception]]   # (provider, 原始异常)
    fallback_attempted: bool
    retryable: bool
```

- 用 `raise ... from last_error` **保留异常链**（修 R4）。
- 错误分类：`RateLimitError` / `TimeoutError` / `ConnectionError` → 可重试；`AuthError` / `BadRequestError` → **立即失败不重试**（修 R5）。

### 2.4 Middleware 协议

```python
class LLMMiddleware(Protocol):
    name: str
    def before(self, req: LLMRequest, ctx: CallContext) -> LLMRequest | None: ...
    def after(self, resp: LLMResponse, ctx: CallContext) -> LLMResponse | None: ...
    def on_error(self, err: Exception, ctx: CallContext) -> LLMResponse | None: ...
```

- **洋葱模型**：`before` 正序执行、`after` 逆序返回（与 Web 框架直觉一致）。
- `before` 返回 `None` = **短路**（护栏拦截时用，直接把 `meta["blocked"]` 结果冒泡）。
- 两个插槽，共用一套协议：

| 插槽 | 位置 | 职责 | 默认链 |
|---|---|---|---|
| **Transport 级** | gateway 内部，每次 LLM 调用 | 观测 / 熔断 / 重试 / 降级 / 预算 / provider 参数适配 | `observability → circuit_breaker → retry → fallback → budget_guard` |
| **Application 级** | **pipeline 边界 + gateway 内部两处**（评审已决） | 注入检测 / 话题白名单 / PII 脱敏 / 输出校验 | `injection_guard → topic_policy → pii_redaction` |

**两处挂载的语义分工**（避免一刀切误杀）：

- **pipeline 边界那处**：作用对象 = **用户原始 question** / 最终 answer。可短路拒答、可改写输出。
- **gateway 内部那处**：作用对象 = **送到 provider 的完整 messages**（含 system prompt、多轮历史、检索到的证据正文）。让 agent 循环内的每一次 LLM 调用（决策 / 成文 / 判断）都受保护，而不是只有最外层用户问答。
- **短路策略按插槽区分**：边界处可拒答；gateway 内那处**默认只检测/告警不短路**——与「文档正文命中仅告警」同一口径（证据文本被误判的成本高于收益）。
- 脱敏类 middleware 幂等，两处都跑无副作用；检测类在 gateway 内只记一次调用级日志，避免重复刷屏。

- 配置驱动，可关、可排序、可换：
```yaml
llm:
  transport_middlewares:
    - {name: observability, enabled: true}
    - {name: circuit_breaker, enabled: true, failure_threshold: 5}
    - {name: retry, enabled: true, max_retries: 3}
    - {name: fallback, enabled: true}
    - {name: budget_guard, enabled: true}
  app_middlewares:
    mount: [pipeline_boundary, gateway_inner]   # 评审已决：两处挂载
    chain:
      - {name: injection_guard, enabled: true, scan_documents: true, block_on: [pipeline_boundary]}
      - {name: topic_policy, enabled: true, block_on: [pipeline_boundary]}
      - {name: pii_redaction, enabled: true}
```

---

## 3. Middleware 清单（把现状收编进来）

| Middleware | 收编自 | 行为 |
|---|---|---|
| `ObservabilityMiddleware` | `gateway.py` 内联的 Langfuse `name` + `stream_options` | 注入 `name`/`tags`/`stream_options`，记录 generation；**框架参数走 meta，不污染 payload** |
| `CircuitBreakerMiddleware` | `gateway.py:29-61`（含死代码） | 用真正的 `CircuitBreaker.call()`，加锁、状态机单点维护 |
| `RetryMiddleware` | `gateway.py:148-158` | 错误分类 + 退避；不可重试错误立即抛 |
| `FallbackMiddleware` | `gateway.py:105-115` | 主备切换，**打 `fallback_used` 标记** |
| `BudgetGuardMiddleware` | `generator.py:68-92` + `query_decomposer.py:53-62`（重复两份） | 「截断 + 空正文 → 放大预算重试一次」，一处实现，两处调用 |
| `InjectionGuardMiddleware` | `core/security.py:35` | 输入侧检测，**扩展为同时扫描 messages + 检索到的文档正文**（修 G2）；用户输入命中 → 拒答；**文档正文命中 → 仅告警不干预**（评审已决） |
| `PIIRedactionMiddleware` | `core/security.py:56-69` | 输出侧脱敏，规则可配置 |
| `TopicPolicyMiddleware` | `core/security.py:26-32` | 话题白名单（现 off-topic 表） |
| `ResponseValidationMiddleware` | `agent/runner.py` 的 `parse_llm_response` / JSON 解析 | 输出可解析性校验、空答案判定，统一口径 |

---

## 4. 目标目录结构

```
infra/llm/
  gateway.py          # 只剩：构造 payload → 调 SDK → 包 LLMResponse → 过 middleware 链
  middleware/
    base.py           # LLMMiddleware 协议 + CallContext + 链式执行器
    transport.py      # observability / circuit_breaker / retry / fallback / budget_guard
    payload.py        # provider 参数适配（langfuse 等框架参数隔离）
  errors.py           # LLMError 家族
domain/interfaces/
  llm.py              # LLMRequest / LLMResponse / LLMProvider（返回类型不再是 Any）
domain/services/
  guards/             # 应用级 middleware（injection_guard / topic_policy / pii_redaction）
```

---

## 5. 迁移路径（每阶段独立可回退）

> 遵循既有约定：**先跑 `ruff check` → `pytest tests/unit` → 再提交**；每阶段一个 commit，行为等价优先。
> Phase 0 的信封迁移按评审决定**一步到位**（不设兼容开关）；Phase 1-3 的策略插件化仍走**保留原路径 + 开关切换**，便于做行为等价对照。

| Phase | 内容 | 验收 | 回退 |
|---|---|---|---|
| **0** | `LLMRequest` 加 `extra` / `meta`；`None` 不下发；`LLMResponse` 信封**直接启用**（无兼容开关），调用点 + 鸭子替身一次性迁移 | 新单测：`extra` 原样到达 SDK、`None` 字段不出现、`meta` 不进 payload、`raw` 与原始响应同一对象 | 单 commit 回滚（`git revert`） |
| **1** | Middleware 骨架 + `Transport` 级把 retry/circuit/fallback/observability 从 gateway 内联逻辑搬成插件 | 现有 `test_llm_gateway.py` 3 项全绿 + 新增「fallback_used 标记」用例 | **一步到位**（评审已决）：无内联回退路径，回退靠 `git revert`；运行期可用 `LLM_TRANSPORT_MIDDLEWARES` 裁剪链路 |
| **2** | `BudgetGuardMiddleware` 收编 generator / decomposer 的重复兜底 | `test_llm_budget_guard.py` 全部用例的调用次数与预算序列不变 + 新增「按请求启用」用例 | 回退靠 `git revert`；运行期可用 `LLM_TRANSPORT_MIDDLEWARES` 去掉 `budget_guard` |
| **3** | Guard 全部 middleware 化并按评审**挂两处**（pipeline 边界 + gateway 内）；`chat_service` / `top_warmup` 只留一行链式调用；**新增文档正文注入扫描（仅告警）** | `test_guard.py` 全绿 + 新用例「文档内含注入指令 → 有告警日志、答案不变」+「agent 内部 LLM 调用经过 guard 链」 | 回退靠 `git revert`；运行期可用 `LLM_GUARDS` 裁剪护栏集合 |
| **4** | 清理死代码（`CircuitBreaker.call` 改为真用）、单例改 deps 注入、错误链修复 | 全量单测 + 端到端一问（`/v1/chat`） | — |
| **5** | 文档 + 评估（口径不变：`LLM_TEMPERATURE=0` + `LLM_GENERATE_MODEL=deepseek-chat`） | 22 题均分不低于当前 21/22 基线 | — |

**关键约束**：不动 `LLMProvider.generate(request)` 的方法签名（只扩 `LLMRequest` 字段），否则所有鸭子类型替身会一起红——这是历史上踩过的坑。

---

## 6. 风险与取舍

| 风险 | 处置 |
|---|---|
| 洋葱链让 debug 变难 | 每次调用记录 `applied_middlewares` 进 response + Langfuse trace，可审计 |
| **流式 + `after` 语义冲突**：已发出的 token 无法改写 | 约定：流式只在 `before` 阶段生效；`after` 只做「最终 chunk 校验」，不做改写 |
| 引入 async 改造会扩散到全仓 | 保持**同步**实现，middleware 用普通函数，不引入 async |
| 插件化后行为漂移 | Phase 1-2 要求「行为等价」验收：同输入下调用次数、预算序列、错误类型逐一比对 |

---

## 7. 决策记录

### 已决（2026-09-17 评审）

| # | 议题 | 决定 |
|---|---|---|
| 1 | 方案交付方式 | **先落 `docs/` 正式方案文档并推送**，评审通过后再开工 |
| 2 | `LLMResponse` 信封开关 | **直接启用、一步到位**，不设兼容开关 |
| 3 | 检索文档正文命中注入 | **仅记录日志告警，不干预答案**（可用性优先） |
| 4 | 应用级 guard 插槽位置 | **挂两处**：pipeline 边界（可短路）+ gateway 内部（默认只告警），agent 循环内调用同样受保护 |
| 5 | 流式 response 的 `.text` 语义 | **只做透传**，`.text` 置空，聚合交给调用方 |

> 至此设计项全部关闭，可开工。

## 8. Phase 0 建议切片

1. `domain/interfaces/llm.py`：`LLMRequest` 加 `extra` / `meta`；新增 `LLMResponse`；`LLMProvider.generate` 返回类型 `Any` → `LLMResponse`。
2. `infra/llm/gateway.py`：`call_kwargs` 硬编码白名单 → 「常用字段过滤 `None` + `extra` 合并 + `meta` 剥离」；返回包 `LLMResponse`。
3. 同步迁移调用点：`Generator` / `QueryDecomposer` / `AgentRunner` / `agent_loop` + 全部鸭子类型替身（`ScriptedLLM` / `RecordingLLM` / `_ScriptedGenerator`）。
4. 新增单测：`extra` 透传、`None` 不下发、`meta` 不进 payload、`raw` 保真、`fallback_used` 标记。

---

## 9. 落地记录

### Phase 0（参数透传 + 返回信封）✅ 2026-09-17

- **改动**：
  - `LLMRequest` 增 `extra`（逃生舱，任意 provider 参数原样透传）/ `meta`（框架参数，**永不进 payload**）；`None` 字段不下发；
  - 新增 `LLMResponse` 信封：`text` / `raw`（原始响应保真）/ `finish_reason` / `usage` / `reasoning` / `tool_calls` / `provider` / `fallback_used` / `attempts` / `latency_ms` / `applied_middlewares` / `trace_id`；流式 `.text` 置空 + `iter_text()`；
  - `LLMProvider.generate` 返回类型 `Any` → `LLMResponse`；
  - **vendor 结构解析收敛**到 `LLMResponse.from_raw` —— 全仓仅 `domain/interfaces/llm.py` 解 `.choices`（原散落在 generator / decomposer / adapter 三处）；
  - 网关 `call_kwargs` 硬编码白名单 → 「常用字段过滤 `None` + `extra` 合并 + `_provider_extras` 注入框架参数」；
  - 降级不再静默：`fallback_used` / `provider` 进信封，Generator 侧记 warning。
- **迁移调用点**：`Generator` / `QueryDecomposer` / `AgentRunner`（经 `adapter.parse_llm_response`）/ `agent_loop` + 3 个假 LLM 替身（`ScriptedLLM` / `RecordingLLM` / `FakeLLM`）。
- **验证**：
  - `ruff check src tests` 通过；
  - `pytest tests/unit` **107 passed**（基线 95 + 新增 12，含 `tests/unit/test_llm_passthrough.py`）；
  - `scripts/smoke_gateway.py` 真实 API 冒烟三条全过：普通调用（正文/usage/raw 均取到）、`extra={"top_p": 0.5}` 被真实 provider 接受、流式 `.text` 为空 + `iter_text()` 拼接正确。
- **冒烟副产品**：预算给小（16 token）时推理模型正文为空、`finish_reason=length` —— 真实调用复现了单测里的场景，信封如实上报而非静默吞掉。
- **未做**（后续 Phase）：middleware 骨架 / guard 双插槽 / 死代码清理 / 错误链修复 / 评估回归。

### Phase 1（middleware 骨架 + transport 插件化）✅ 2026-09-17

- **新增文件**：`infra/llm/middleware/base.py`（洋葱链 + `LLMCallContext` + `LLMMiddleware` 协议）、`middleware/transport.py`（observability / fallback / circuit_breaker / retry）、`infra/llm/circuit_breaker.py`（熔断器从 gateway 拆出）、`infra/llm/errors.py`。
  - （2026-09-18 注：`transport.py` 已按 middleware 拆为 `observability.py` / `fallback.py` / `circuit_breaker.py` / `budget_guard.py` / `retry.py`（含 `is_retryable`）五个文件，包 `__init__` 的公共导入面不变；同日熔断器状态机（`CircuitBreaker` / `CircuitBreakerOpenError` / `CircuitState`）也从 `infra/llm/circuit_breaker.py` 内聚进 `middleware/circuit_breaker.py`（与 middleware 同文件），经 middleware 包导出、`client.py` 再导出的公共面不变。下文 Phase 记录中 `middleware/transport.py` / `infra/llm/circuit_breaker.py` 路径均指变动前状态。）
- **网关收敛**：`generate()` 只剩「起 ctx → 跑链路 → 盖章到信封」；`_invoke()` = 选 client/模型 → 组 payload → 调 SDK → 包信封。策略零内联。
- **链路顺序**（外层→内层）：`observability → fallback → circuit_breaker → retry → terminal`
  - `fallback` 在熔断外层：熔断打开时由它接管走备用（与旧行为一致）；
  - `circuit_breaker` 在重试外层：**一个逻辑调用只记一次成败**，内部重试不会加速熔断。
- **可插拔**：`LLM_TRANSPORT_MIDDLEWARES=observability,retry`（逗号分隔，空 = 默认链）可运行期裁剪；也可 `LLMGateway(middlewares=[...])` 显式注入。`/v1/stats` 新增 `llm.middlewares` 暴露当前链路。
- **行为变更（重要，非等价处）**：
  1. **熔断器从死代码变成真生效**：旧实现 `failure_count` 只被重置、从未累加，`state` 从未置 OPEN —— 「熔断」从未发生过（文档与 stats 都在展示一个恒 closed 的摆设）。现在连续失败到阈值（默认 5）会真的拒绝请求，冷却 60s 后半开试探。**无备用模型时表现为快速失败**，不再反复重试打后端。
  2. **重试加了错误分类**：4xx（除 429 限流）不再退避重试，立即失败（旧实现一律重试 3 次、白等 7s）。
  3. `stream_options` 注入从网关硬编码挪到 `observability` middleware（写 `request.extra`，它本就是 provider 参数）。
- **验证**：`ruff check src tests` 通过；`pytest tests/unit` **116 passed**（107 + 新增 `test_llm_middleware.py` 9 项：重试成功 / 不可重试快失败 / 熔断打开后不打 SDK / 半开恢复 / 降级标记与模型 / 默认链路顺序 / 链路裁剪 / 显式注入 / stream_options 条件注入）；`scripts/smoke_gateway.py` 真实 API 三条全过（与 Phase 0 结果一致）。
- **未做**：`budget_guard` 收编（Phase 2）/ guard 双插槽（Phase 3）/ 错误模型与死代码收尾（Phase 4）/ 评估回归（Phase 5）。

### Phase 2（预算兜底收编成 `BudgetGuardMiddleware`）✅ 2026-09-17

- **新增 middleware**：`BudgetGuardMiddleware`（`middleware/transport.py`）。默认链变为
  `observability → fallback → circuit_breaker → budget_guard → retry`
  （`budget_guard` 在 `retry` **外层**：放大预算那次调用仍享受错误重试）。
- **按请求启用**：`request.meta["budget_guard"]=True`。由调用方决定预算语义 —— 生成走
  `LLM_GENERATE_MAX_TOKENS` + 兜底；显式传 `max_tokens` 的调用不带该标记，保持
  「单次调用、不放大」的旧口径（`test_explicit_max_tokens_keeps_single_call` 仍锁着）。
- **收编重复**：generator / query_decomposer 里各自那份约 15 行「截断空返回 → 放大预算
  重试一次」删除，两处只保留应用层决策（仍为空 → 记 warning / 回退原问题单路检索）。
- **测试重构（诚实记录）**：`test_llm_budget_guard.py` 中原先注入裸的假 LLM
  （`ScriptedLLM`），而机制搬进 middleware 后裸替身**测不到链路** → 改为「真实网关 +
  假 OpenAI 客户端」驱动；**断言口径不变**（调用次数与预算序列逐条一致），并新增 2 项
  「按请求启用」用例（不带标记不放大 / 放大预算不大于原预算时不重试）。
- **验证**：`ruff` 通过；`pytest` **119 passed**；真实 API 冒烟新增第 4 条端到端证明 ——
  16 token 打真实推理模型，**SDK 调用预算序列 `[16, 16000]`**、正文 `'收到'`，同时打出
  middleware 的「正文为空且被截断…重试一次」告警。只看正文会有歧义（小预算也可能碰巧
  吐出正文），所以该用例以**实际 SDK 调用序列**判定。

### Phase 3（guard middleware 化 + 双插槽）✅ 2026-09-17

- **新增 `domain/services/guards/`**：
  - `base.py`：`Guard` 协议（`inspect(text, ctx) -> GuardVerdict`）、`GuardContext`（挂载点
    `pipeline_boundary` / `gateway_inner` × 作用对象 `user_input` / `document` / `answer` /
    `llm_messages` × `can_block`）、`GuardChain`（顺序执行、拦截短路、改写逐级传递）；
  - `builtin.py`：`InjectionGuard` / `TopicPolicyGuard` / `PIIRedactionGuard` +
    `build_guard_chain()`（配置 `LLM_GUARDS`，空 = 全部内置）+ `get_guard_chain()` 单例。
    模式表仍留在 `core/security.py`（新增细分的 `check_injection_patterns` / `check_off_topic`，
    `check_injection` 作为组合入口保留，兼容既有调用方与 `test_guard.py`）；
  - `llm_guard.py`：`LLMGuardMiddleware`（gateway 内挂载，扫**非 system** 消息，仅告警）。
- **两处挂载**：
  1. **pipeline 边界**（`ChatService.chat` / `chat_stream` / `top_warmup`）：用户输入命中 → 拒答；
     最终答案 → 脱敏改写；**检索文档正文 → 仅告警、答案不变**；
  2. **gateway 内**（transport 链最外层 `guard`）：agent 循环内每次 LLM 调用都过护栏，
     **只告警不短路**。
- **协议实现说明（对方案 §2.4 草图的偏离，需知悉）**：草图写的是 `before / after / on_error`
  三段式；实现改为**洋葱式 `__call__(request, ctx, call_next)`** —— 重试/降级需要「自己决定
  调用几次、换不换 provider」，三段式 hook 表达不了。guard 复用同一协议（它天然是「调用前
  检查」），避免出现两套中间件抽象。
- **调用点收敛**：`chat_service` / `top_warmup` 里手写的 `check_injection` / `sanitize_output`
  （原先 3 个入口各一份）全部换成 `self.guards.inspect(...)` 单行调用；`api/deps.py` 显式注入。
- **验证**：`ruff` 通过；`pytest` **136 passed**（119 + 新增 `test_guards_chain.py` 17 项：边界
  拒答 / 话题 / 文档仅告警 / messages 不短路 / 答案脱敏 / 输入不脱敏 / 链路配置与短路 /
  ChatService 边界与文档扫描 / gateway 内挂载（含「system 消息不扫」）/ `applied_middlewares`
  含 guard）；真实 API 冒烟 4 条仍全过（guard 已是最外层，正常提示无误报、调用不受影响）；
  `api.deps` / `top_warmup` / guards 导入链自检通过。
- **未做**：错误模型与死代码收尾（Phase 4）/ 评估回归（Phase 5）。

### Phase 4（失败可归因：错误模型 + 错误链）✅ 2026-09-17

- `infra/llm/errors.py`：`LLMError` 基类（带 `attempts` / `provider_errors` /
  `fallback_attempted` / `retryable`，`__str__` 自动附 provider 明细）；`AllModelsFailedError`
  继承之 —— 失败不再是一句「都失败了」。
- `LLMCallContext` 增 `provider_errors`：`RetryMiddleware` 在**最终失败 / 不可重试**时记录
  `(provider, 原始异常)`；`FallbackMiddleware` 在「无备用」与「备用也失败」两条路径上组装带
  上下文的 `AllModelsFailedError`，并 `from e` 保留直接起因。
- **错误链**：Phase 1 重写时已消除 `raise ... from None`（重试中间件用 bare `raise`），本次用
  测试锁死（`__cause__` 必须是原始异常）。
- **死代码**：`CircuitBreaker.call` 已在 Phase 1 删除（改为 `before_call` /
  `record_success` / `record_failure` 真用），本次复查全仓无残留。
- **单例注入（有意保留）**：`api/deps.py` 已显式注入网关与护栏；`Generator` /
  `QueryDecomposer` / `ChatService` 保留 `or get_llm_gateway()` 兜底，供 top_warmup 的
  「未注入时的独立使用」路径 —— 强删会破坏该路径且无收益，故不动。
- **验证**：`ruff` 通过；`pytest` **142 passed**（+6 项 `test_llm_errors.py`：主挂可归因 /
  异常链保留 / 消息含 provider 明细 / 401 不可重试 / 主备都挂列两条 / 熔断打开被标记）。
- **未做**：端到端一问（`/v1/chat`）归入 Phase 5（需重建镜像）。

### Phase 5（部署 + v2 评估回归）✅ 2026-09-17

- **部署**：线上容器此前处于 **crash loop**（见 `fix(logging)` `564ea0a`），修复后重建上线。
  - ⚠️ 正规 `docker build`（`FROM python:3.12-slim`）卡在 deb.debian.org 的 apt 阶段
    （9.6MB 拉 7 分钟无进展）→ 改用 **overlay 镜像**（`FROM <旧镜像>` + `COPY src/`）。因
    `uv sync` 是 editable 安装（源码即 `/app/src`），对纯代码改动等价且秒级完成；旧镜像保留
    `:pre-refactor-rollback` tag 可回滚。**正规全量镜像需在 CNB CI 或网络更好的环境重建。**
  - 上线验证三项：`/v1/health` ok（milvus + redis）；启动日志 `BM25 索引构建完成: docs=1055`；
    `/v1/stats` 的 `llm.middlewares` = `[guard, observability, fallback, circuit_breaker,
    budget_guard, retry]`（新链路确实生效）；`/v1/chat` 端到端一问命中 T01 期望关键词；
    注入问题被边界护栏拒答（`检测到提示注入模式，请求已拒绝`）。
- **评估口径**（用户确认）：`LLM_TEMPERATURE=0` + `LLM_DECOMPOSE_TEMPERATURE=0` +
  `LLM_GENERATE_MODEL=deepseek-chat` + `use_cache=false` + `top_k=10`；集合
  `chartermate_docs_insightforge`；30 题（`scripts/eval/cases/eval-set-v2.json`）。
- **结果（同配置两轮）**：run1 **29/30（0.9833）**、run2 **28/30（0.9722）**；拒答 **4/4** 两轮全对。
- **对照基线**：v2-c 26/30（0.9389）。⚠️ 但历史三次 v2 基线**自身不一致** —— v2-a 4/0.185、
  v2-b 21/0.828、v2-c 26/0.939（同为 09-16 15:30–15:42、同参数），说明其中有被污染/中途失败的
  轮次，基线数字只能当参考上限，不能当精确对照。
- **逐题对账**：run1 相对 v2-c **零题下降**、3 题上升（T07 0.667→1.0、T10 0.5→1.0、T17 0.5→1.0）；
  run2 仅 T19 波动（1.0→0.667），同样 3 题上升。
- **结论（按实验纪律记账）**：本轮真正能下的结论是 **无回归** —— 该重构在成功路径上行为等价
  （retry 分类 / 熔断 / 错误模型只影响失败路径；预算兜底与护栏在改造前后行为一致）。T07/T10/T17
  的稳定提升**不归因于本次重构**（这三条路径没有行为改动），更可能是运行间/服务状态差异
  （case 级 ±1-2 例属噪声）。若要主张收益，需另做单变量实验或同口径多轮对照。
- **未做**：正规全量镜像重建（留给 CNB CI）。

### Phase 6（client 去默认参数：配置读取与初始化收归组装点）✅ 2026-09-18

**动机**：`LLMClient.__init__` 自读 `settings` / 环境变量并自建 SDK 客户端 ——
「默认值悄悄生效」会把「配置没接上」变成静默失败（例如 `FALLBACK_*` 从 `os.getenv`
裸读，`.env` 里配了但字段名不一致就静默无备用）。本次把**配置读取与客户端初始化
全部收归组装点** `api/deps.py::build_llm_client`，客户端构造参数**一律无默认值**。

- **`client.py` 变成纯组件**：移除 `settings` / `os` / `OpenAI` 运行时导入（SDK 类型只在
  `TYPE_CHECKING` 下引用）；`LLMClient.__init__` 改为 7 个**必填关键字参数**：
  `primary_client` / `primary_model` / `fallback_client` / `fallback_model` /
  `circuit_breaker` / `default_timeout` / `middlewares`。漏传直接 `TypeError`。
- **默认超时成为构造参数**：原来 `_build_payload` 现读 `settings.llm_timeout_seconds`，
  现在由组装点注入为 `default_timeout`（`request.timeout` 覆盖语义不变）。
- **middleware 组装函数变纯函数**：`default_transport_middlewares(client)` 改为
  `build_transport_middlewares(*, guard_chain, breaker, has_fallback, names)` ——
  不读 `settings`，逗号解析/未知名告警等**策略仍留在 infra**，配置值由组装点传入。
- **备用 provider 进 `Settings`**：新增 `fallback_api_key` / `fallback_base_url` /
  `fallback_model`（env 名不变：`FALLBACK_API_KEY` / `FALLBACK_BASE_URL` / `FALLBACK_MODEL`），
  不再从 `os.getenv` 裸读；**api_key 与 base_url 都配才启用备用**（显式化，含注释）。
- **`get_llm_client` 单例移到 `api/deps.py`**：导入点同步改为 deps ——
  `api/routes/stats.py`、`agent/agent_loop.py`、`scripts/eval/{probe_latency_breakdown,
  probe_case_evidence,run_agent_eval}.py`、`scripts/smoke_gateway.py`（改用 `build_llm_client()`）。
  无循环依赖：`deps → agent.runner → infra`，编排层不反向依赖组装点。
- **测试改造**（假 SDK 直接注入，不再 monkeypatch `client_module.OpenAI` / 操作 `FALLBACK_*` env）：
  新增 `tests/conftest.py::make_llm_client` 夹具（组装等价于生产、SDK 可替换，并统一 patch
  `retry.time.sleep` 跳过退避）；新增 `tests/unit/test_llm_wiring.py`（8 项）专门锁
  「settings → 组装」这条接线：熔断参数 / 默认超时 / 主模型与 SDK 配置 / 备用 provider
  「只配一个 = 没配」/ 链裁剪 / 空配置 = 默认链 / 单例 / **漏传参数必须报错**。
- **验证**：`ruff check src tests scripts` 通过；`pytest tests/unit` **169 passed**
  （重构前 157，净增 12：wiring 8 + 链相关 4）。导入冒烟：`build_llm_client()` 读到
  `chain=[guard, observability, fallback, circuit_breaker, budget_guard, retry]`、
  `default_timeout=120.0`、`breaker=5/60`、备用未配 = `None`；`api.main` 正常导入。
- **未做**：真实 API 冒烟（`scripts/smoke_gateway.py`，需 key）与线上镜像重建（留给 CNB CI）。


