# cognis-trace — Design Spec

**Date:** 2026-05-06
**Status:** Approved (brainstorming complete)
**Crate:** `cognis-trace`

## 1. Purpose

A pluggable observability layer for Cognis. Bridges the existing
`cognis_core::CallbackHandler` event stream to one or more external
observability backends (Langfuse, LangSmith, any OTLP-compatible system),
plus the supporting non-trace surfaces those platforms expose: prompt
management (pull) and evaluation scores (push).

The crate is purely additive. Apart from two small non-breaking fields on
`cognis_core::RunnableConfig`, no other workspace crate changes API shape.

## 2. Scope (v1)

In scope:

1. Trace ingestion — chain / LLM / tool / node / retriever spans, with
   parent linkage so traces are trees.
2. Sessions and threads — `session_id` / `user_id` / tags propagated to
   the trace root.
3. Cost tracking — token usage → USD, structured by token kind (input,
   output, cache_read, cache_write).
4. Prompt management — pull versioned prompts from a `PromptStore`, with
   automatic stamping of `prompt_name` / `prompt_version` on the
   generation that uses them.
5. Evaluation scores — numeric / categorical / boolean scores attached to
   a run, pushed via a `ScoreSink`.
6. Metrics — OTel histograms / counters for token usage, latency,
   error rate when the OTel exporter is active. No bespoke metrics API in
   v1; users get standard `gen_ai.*` semconv instruments.

Out of scope (deferred to a later release):

- Custom evaluator runners.
- Dataset experimentation.
- Replay / time-travel debugging.

## 3. Architecture

### 3.1 Crate layout

New crate `crates/cognis-trace`, added to the workspace as a sibling of
`cognis-core`, `cognis`, `cognisgraph`, `cognis-llm`, `cognis-rag`,
`cognis-macros`. It depends on `cognis-core` only — never on `cognis`,
`cognisgraph`, `cognis-llm`, or `cognis-rag`. This keeps it usable from
any layer of the stack.

```
crates/cognis-trace/
├── Cargo.toml
└── src/
    ├── lib.rs                 # public re-exports, crate-level docs
    ├── span.rs                # Span, Generation, ScoreRecord types
    ├── exporter.rs            # TraceExporter trait, ExporterStats
    ├── handler.rs             # TracingHandler: CallbackHandler bridge
    ├── meta.rs                # TraceMeta helpers
    ├── parent.rs              # task-local fallback for parent_run_id
    ├── batch.rs               # Batcher<T>: bounded queue + flush task
    ├── cost.rs                # PriceTable, default_pricing_2026_05
    ├── prompts.rs             # PromptStore trait, Prompt
    ├── scores.rs              # ScoreSink trait, ScoreValue
    ├── metrics.rs             # OTel instruments (gated on `otel`)
    ├── error.rs               # TraceError
    └── exporters/
        ├── mod.rs
        ├── stdout.rs          # always on
        ├── mock.rs            # always on; in-memory, for tests
        ├── langfuse/          # feature = "langfuse"
        │   ├── mod.rs
        │   ├── config.rs
        │   ├── client.rs
        │   ├── exporter.rs
        │   ├── prompts.rs
        │   └── scores.rs
        ├── langsmith/         # feature = "langsmith"
        │   └── ...
        └── otel/              # feature = "otel"
            ├── mod.rs
            └── exporter.rs
```

### 3.2 Workspace-level changes

Two small additions to `cognis_core::RunnableConfig`. Both have
default-initialized values so existing callers using
`RunnableConfig::default()` are unaffected.

```rust
pub struct RunnableConfig {
    // ...existing fields...
    pub parent_run_id: Option<Uuid>,
    pub metadata: HashMap<String, serde_json::Value>,
}
```

Composition sites that construct child configs must propagate
`parent_run_id`. The implementation plan audits these:

- `cognis_core::compose::Pipe`
- `cognis_core::compose::Sequence`
- `cognis_core::compose::Branch`
- `cognisgraph` Pregel engine, when invoking a node
- any other `Runnable::invoke_with_config` call site that builds a fresh
  config for a sub-runnable

A `tokio::task_local!` span stack in `cognis-trace::parent` serves as a
fallback for any composition site not yet plumbed. Phase 2 (post-v1)
removes the fallback after the audit is complete.

### 3.3 Bridge

```rust
pub struct TracingHandler {
    exporters: Vec<Arc<dyn TraceExporter>>,
    inflight: DashMap<Uuid, SpanBuilder>,
    span_batchers: Vec<Batcher<Span>>,        // one per exporter
    score_batchers: Vec<Batcher<ScoreRecord>>,
    pricing: Arc<PriceTable>,
}

impl CallbackHandler for TracingHandler {
    fn on_chain_start(&self, runnable, input, run_id) { ... }
    fn on_chain_end(&self, runnable, output, run_id) { ... }
    fn on_chain_error(&self, runnable, error, run_id) { ... }
    fn on_llm_start(&self, model, prompt, run_id) { ... }
    fn on_llm_end(&self, model, output, run_id) { ... }   // see §4.3
    fn on_llm_error(&self, model, error, run_id) { ... }
    fn on_tool_start(&self, tool, args, run_id) { ... }
    fn on_tool_end(&self, tool, result, run_id) { ... }
    fn on_tool_error(&self, tool, error, run_id) { ... }
    fn on_node_start(&self, node, step, run_id) { ... }
    fn on_node_end(&self, node, step, output, run_id) { ... }
    fn on_checkpoint(&self, step, run_id) { ... }
    fn on_custom(&self, kind, payload, run_id) { ... }
}
```

`TracingHandler` is one handler that fans out to N exporters, each with
its own `Batcher`, so a slow Langfuse never blocks OTel.

Lifecycle of a span inside the bridge:

1. `on_*_start` — read `parent_run_id` from `RunnableConfig` if
   present, else from the task-local stack. Compute `trace_id`: equal to
   `run_id` when there is no parent, inherited otherwise. Insert a
   `SpanBuilder` keyed by `run_id` into `inflight`. Push `run_id` onto
   the task-local stack.
2. `on_*_end` — remove the builder. For LLM ends, parse the structured
   payload (§4.3) into a `Generation`; compute `cost` from
   `PriceTable`. Construct a `Span`. Send to every exporter's batcher.
   Pop the task-local stack.
3. If this span is the trace root (no parent), the Langfuse exporter
   additionally emits a `trace-create` event populated from
   `RunnableConfig::metadata` (`session_id`, `user_id`, `tags`,
   `release`, `version`, `environment`, `public`).

## 4. Data model

### 4.1 Span

```rust
pub enum SpanKind {
    Span,
    Generation,
    Event,
    Agent,
    Tool,
    Chain,
    Retriever,
    Embedding,
    Guardrail,
}

pub enum ObservationLevel { Default, Debug, Warning, Error }

pub struct Span {
    pub run_id: Uuid,
    pub parent_run_id: Option<Uuid>,
    pub trace_id: Uuid,                    // root run_id of the tree
    pub kind: SpanKind,
    pub name: String,
    pub started_at: SystemTime,
    pub ended_at: Option<SystemTime>,
    pub level: ObservationLevel,
    pub status_message: Option<String>,    // Some when level == Error/Warning
    pub input: Option<serde_json::Value>,
    pub output: Option<serde_json::Value>,
    pub session_id: Option<String>,        // populated only on trace root
    pub user_id: Option<String>,           // populated only on trace root
    pub tags: Vec<String>,                 // populated only on trace root
    pub metadata: HashMap<String, serde_json::Value>,
    pub generation: Option<Generation>,    // Some iff kind == Generation
}
```

`SpanKind` aligns with Langfuse's `ObservationType`. The previous brainstorm
variants `Llm` and `Node` map to `Generation` and `Span` respectively.
`Custom` events go through the `Event` variant with `metadata.kind` carrying
the original kind string from `Observer::Custom`.

### 4.2 Generation

```rust
pub struct Generation {
    pub model: String,                     // "gpt-4o-2024-08-06"
    pub provider: String,                  // "openai"
    pub model_parameters: HashMap<String, serde_json::Value>,
    pub usage: TokenUsage,
    pub cost: Option<CostDetails>,
    pub completion_start_time: Option<SystemTime>,  // TTFT
    pub finish_reason: Option<String>,
    pub prompt_name: Option<String>,
    pub prompt_version: Option<u32>,
}

pub struct TokenUsage {
    pub input: u32,
    pub output: u32,
    pub cache_read: u32,
    pub cache_write: u32,
}

pub struct CostDetails {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
    pub total: f64,
}
```

### 4.3 LLM payload schema (provider contract)

To avoid extending `cognis_core::CallbackHandler`, the bridge parses
token usage out of the JSON payload that providers hand to
`on_llm_end`. Each provider in `cognis-llm` must serialize its response
to this schema:

```json
{
  "content": "...",
  "model": "gpt-4o-2024-08-06",
  "provider": "openai",
  "finish_reason": "stop",
  "usage": {
    "input_tokens": 123,
    "output_tokens": 45,
    "cache_read_tokens": 0,
    "cache_creation_tokens": 0
  },
  "model_parameters": { "temperature": 0.7, "max_tokens": 1024 },
  "completion_start_time": "2026-05-06T12:34:56.789Z",
  "prompt_name": "greeting",
  "prompt_version": 3
}
```

Fields beyond `content` and `model` are optional. Missing usage means
`Generation.usage = TokenUsage::default()` and cost stays `None`.
`prompt_name` / `prompt_version` are populated automatically when the
generation was driven by a `Prompt` fetched from a `PromptStore`
(see §6.4).

This schema is documented in `cognis-trace::span::doc_payload_schema`
and verified by a contract test that runs against every provider's
`MockChatModel` adapter.

### 4.4 Score

```rust
pub enum ScoreValue {
    Numeric(f64),
    Categorical(String),
    Boolean(bool),
}

pub struct ScoreRecord {
    pub run_id: Uuid,                      // observation id
    pub trace_id: Option<Uuid>,            // optional explicit trace pointer
    pub session_id: Option<String>,
    pub name: String,
    pub value: ScoreValue,
    pub comment: Option<String>,
}
```

## 5. Exporter trait

```rust
#[async_trait]
pub trait TraceExporter: Send + Sync {
    async fn export_spans(&self, spans: Vec<Span>) -> Result<(), TraceError>;
    async fn export_scores(&self, _scores: Vec<ScoreRecord>) -> Result<(), TraceError> {
        Err(TraceError::Unsupported("scores"))
    }
    async fn shutdown(&self) -> Result<(), TraceError> { Ok(()) }
    fn name(&self) -> &str;
}
```

Each exporter is wrapped by a per-exporter `Batcher<Span>` (and
`Batcher<ScoreRecord>` if it supports scores) that runs a background
`tokio::spawn` flush task. Exporters never block the bridge's `on_*_end`
callbacks; they receive batches from the flush task.

## 6. Backends

### 6.1 Langfuse

Native ingestion path. `LangfuseExporter` posts to
`POST {host}/api/public/ingestion` with HTTP Basic auth
(`pk-lf-...:sk-lf-...` base64). Although Langfuse marks this endpoint as
"legacy" in favor of OTel, it remains supported and gives lossless
mapping to our types without forcing the OTel dependency tree on
Langfuse-only users.

Mapping table (our type → Langfuse field):

| Cognis | Langfuse |
|---|---|
| `Span.run_id` | `id` |
| `Span.parent_run_id` | `parentObservationId` |
| `Span.trace_id` | `traceId` |
| `SpanKind` | `type` (SPAN, GENERATION, EVENT, AGENT, TOOL, CHAIN, RETRIEVER, EMBEDDING, GUARDRAIL) |
| `Span.level`, `status_message` | `level`, `statusMessage` |
| `Span.input` / `output` / `metadata` | `input` / `output` / `metadata` |
| `Generation.model` | `model` |
| `Generation.usage` | `usageDetails` (keys: `input`, `output`, `cache_read_input`, `cache_creation_input`) |
| `Generation.cost` | `costDetails` (keys: `input`, `output`, `cache_read_input`, `cache_creation_input`, `total`) |
| `Generation.completion_start_time` | `completionStartTime` |
| `Generation.prompt_name` / `prompt_version` | `promptName` / `promptVersion` |
| Trace-root only: `session_id`, `user_id`, `tags`, `release`, `version`, `environment`, `public` | corresponding fields on `trace-create` event |

Batches are sent as `{ "batch": [...events] }`. Langfuse returns 207
with `successes` and `errors`; per-event errors are logged once per
batch via `tracing::warn!` and counted in `ExporterStats.failed`. Batch
size is bounded to under 3.5 MB (Langfuse's documented limit) — the
batcher chunks larger queues automatically.

`LangfusePromptClient` and `LangfuseScorer` are separate structs that
share `LangfuseConfig` but are not coupled to the exporter:

- `LangfusePromptClient::get(name, version_or_label) -> Prompt`
  - `GET /api/public/v2/prompts/{name}` and
    `GET /api/public/v2/prompts/{name}/versions/{version}`
  - Local in-memory cache with configurable TTL (default 60s).
  - Returns a `Prompt { name, version, template, config, labels }`,
    where `template` is a `cognis_core::PromptTemplate` for `text` type
    or `cognis_core::ChatPromptTemplate` for `chat` type.
- `LangfuseScorer::score(record)` — `POST /api/public/scores`.

### 6.2 LangSmith

`LangSmithExporter` posts run trees to LangSmith's runs endpoint with
`x-api-key` auth. Uses LangSmith's `dotted_order` field to express
parent linkage. Maps `SpanKind` → LangSmith's run type strings (`llm`,
`tool`, `chain`, `retriever`). Cost is sent as
`extra.invocation_params.cost`.

### 6.3 OTel

`OtelExporter` rides on the official `opentelemetry`,
`opentelemetry-otlp`, `opentelemetry_sdk`, and
`opentelemetry-semantic-conventions` crates. We do not reimplement OTLP.

Span attributes emit both `gen_ai.*` (vendor-neutral) and `langfuse.*`
prefixes so the same exporter works for Honeycomb, Datadog, Tempo, and
Langfuse's OTel ingest:

- `gen_ai.request.model`, `gen_ai.system` (provider name),
  `gen_ai.request.temperature`, `gen_ai.request.max_tokens`
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`
- `gen_ai.response.finish_reasons`
- `langfuse.session.id`, `langfuse.user.id`, `langfuse.trace.tags`
- `langfuse.observation.prompt.name`, `langfuse.observation.prompt.version`
- `langfuse.observation.cost_details` (JSON-encoded)
- `langfuse.observation.usage_details` (JSON-encoded)

A convenience constructor
`OtelExporter::langfuse_preset(public_key, secret_key)` configures URL,
HTTP Basic auth, `x-langfuse-ingestion-version: 4` header, and protocol
to `http/protobuf`.

Metrics (item 6 in scope) are recorded by the OTel exporter using
standard `gen_ai.*` instruments:

- `gen_ai.client.token.usage` (histogram, by `gen_ai.token.type`)
- `gen_ai.client.operation.duration` (histogram)
- `gen_ai.server.request.errors` (counter)

These fire on every `on_llm_end` / `on_llm_error`. Backends that
support metrics (Tempo + Mimir, Datadog, Honeycomb) get them
automatically; pure-trace backends (Jaeger) ignore them.

## 7. Configuration

```rust
pub struct LangfuseConfig {
    pub host: String,                      // default: https://cloud.langfuse.com
    pub public_key: String,
    pub secret_key: SecretString,
    pub environment: Option<String>,
    pub release: Option<String>,
    pub flush_interval: Duration,          // default 1s
    pub max_batch: usize,                  // default 100
    pub queue_capacity: usize,             // default 10_000
    pub timeout: Duration,                 // default 10s
    pub max_retries: u32,                  // default 3
}
```

Each backend config implements `from_env()`:

| Backend | Env vars |
|---|---|
| Langfuse | `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_ENVIRONMENT`, `LANGFUSE_RELEASE`, `LANGFUSE_FLUSH_INTERVAL_MS`, `LANGFUSE_MAX_BATCH`, `LANGFUSE_QUEUE_CAPACITY`, `LANGFUSE_TIMEOUT_MS`, `LANGFUSE_MAX_RETRIES` |
| LangSmith | `LANGSMITH_API_KEY`, `LANGSMITH_ENDPOINT`, `LANGSMITH_PROJECT` |
| OTel | standard `OTEL_EXPORTER_OTLP_*` env vars (handled by `opentelemetry-otlp`) |

Per the project's secret policy, secrets are loaded via `direnv` +
`envchain`; the crate never reads `.env` files.

`TracingHandler::from_env()` is a unified entry point that auto-detects
which backends have credentials in the environment and wires them up.

```rust
let handler = TracingHandler::from_env()?;

// or explicit
let handler = TracingHandler::builder()
    .with_exporter(LangfuseExporter::new(LangfuseConfig::from_env()?))
    .with_exporter(StdoutExporter::pretty())
    .with_default_pricing()
    .override_price("gpt-4o", ModelPrice { input: 2.50, output: 10.00, cache_read: 1.25, cache_write: 3.75 })
    .build();

let cfg = RunnableConfig::default()
    .with_callback(Arc::new(handler))
    .with_metadata([
        TraceMeta::session("session-abc"),
        TraceMeta::user("user-123"),
        TraceMeta::tags(["prod", "checkout"]),
    ]);

chain.invoke(input, &cfg).await?;
```

## 8. Cost computation

`PriceTable` is a `HashMap<String, ModelPrice>` of model id to per-token
prices. The crate ships `default_pricing_2026_05` — a dated snapshot for
the major providers. Users can override or add models:

```rust
/// USD per 1M tokens for each token category.
pub struct ModelPrice {
    pub input: f64,
    pub output: f64,
    pub cache_read: f64,
    pub cache_write: f64,
}
```

The bridge computes cost in `cost.rs` after `on_llm_end` and writes
`Generation.cost`. Exporters consume `cost` as-is — no exporter
implements pricing logic. Unknown models log a one-time
`tracing::debug!` and produce `cost = None`; the span is still
exported.

## 9. Error handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum TraceError {
    #[error("missing required env var: {0}")]
    MissingEnvVar(String),
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    #[error("network error sending to {backend}: {source}")]
    Network { backend: &'static str, #[source] source: reqwest::Error },
    #[error("backend {backend} returned {status}: {body}")]
    BackendStatus { backend: &'static str, status: u16, body: String },
    #[error("queue overflowed; {dropped} events dropped (backend: {backend})")]
    QueueOverflow { backend: &'static str, dropped: usize },
    #[error("backend does not support: {0}")]
    Unsupported(&'static str),
    #[error("serialization: {0}")]
    Serde(#[from] serde_json::Error),
}
```

A failing exporter never panics or propagates errors into the user's
main code path. Errors are logged and counted in
`ExporterStats { sent, dropped, failed }`, retrievable via
`handler.stats(exporter_name)`.

| Situation | Behavior |
|---|---|
| Backend returns 4xx (auth, schema) | Log error once, mark exporter degraded, drop batch, continue |
| Backend returns 5xx / network error | Retry with exponential backoff up to `max_retries`, then drop |
| Queue full | Drop oldest events, increment `dropped`, rate-limited `tracing::warn!` |
| Process exits | `Drop` on handler runs blocking flush with 5s deadline |
| `handler.shutdown().await` | Graceful flush, await in-flight requests, close pool |

## 10. Cargo features

```toml
[features]
default = ["stdout"]
stdout = []
langfuse = ["reqwest", "secrecy", "base64"]
langsmith = ["reqwest", "secrecy"]
otel = [
    "opentelemetry",
    "opentelemetry-otlp",
    "opentelemetry_sdk",
    "opentelemetry-semantic-conventions",
]
all = ["langfuse", "langsmith", "otel"]
integration_tests = []
```

The `default` feature includes `stdout` so the dev experience without any
backend is "just see your traces in the terminal."

## 11. Public API surface

```rust
// crates/cognis-trace/src/lib.rs
pub use exporter::{TraceExporter, ExporterStats};
pub use handler::{TracingHandler, TracingHandlerBuilder};
pub use span::{
    Span, SpanKind, ObservationLevel,
    Generation, TokenUsage, CostDetails,
    ScoreRecord, ScoreValue,
};
pub use meta::TraceMeta;
pub use cost::{PriceTable, ModelPrice, default_pricing_2026_05};
pub use prompts::{PromptStore, Prompt};
pub use scores::ScoreSink;
pub use error::TraceError;
pub use exporters::stdout::StdoutExporter;
pub use exporters::mock::MockExporter;

#[cfg(feature = "langfuse")]
pub use exporters::langfuse::{
    LangfuseConfig, LangfuseExporter, LangfusePromptClient, LangfuseScorer,
};

#[cfg(feature = "langsmith")]
pub use exporters::langsmith::{LangSmithConfig, LangSmithExporter};

#[cfg(feature = "otel")]
pub use exporters::otel::{OtelConfig, OtelExporter};
```

## 12. Testing strategy

- Unit tests per module — `span`, `cost`, `batch`, `parent`, `meta` are
  pure logic and get full coverage.
- `MockExporter` (always built) collects spans into a `Vec<Span>` for
  assertions. Used by other crates' integration tests to verify their
  callbacks fire correctly.
- `wiremock`-based tests for the Langfuse and LangSmith exporters: stand
  up a fake HTTP server, assert request bodies match the documented
  schema. No real backends in CI.
- OTel exporter tests use `opentelemetry_sdk`'s in-memory exporter to
  assert attributes are emitted with the right keys.
- A contract test asserts every `cognis-llm` provider's `on_llm_end`
  payload conforms to the schema in §4.3.
- Integration test under `crates/examples`: a small chain
  `prompt | model | parser` with `MockChatModel` + `TracingHandler` +
  `MockExporter`, asserting tree shape, costs, session_id propagation.
- Real-backend tests behind `#[cfg(feature = "integration_tests")]`,
  off by default, run manually before releases against a Langfuse
  account configured via `envchain --set cognis LANGFUSE_*`.

## 13. Out of scope (deferred)

- Custom evaluator runners (running graders inside the bridge).
- Dataset experimentation (creating datasets, running experiments).
- Replay / time-travel debugging.
- Migrating Langfuse to OTel-only — keep both paths available; revisit
  when Langfuse formally deprecates ingestion.
- A `Middleware` `cognisagent::Middleware` adapter — once the
  `cognisagent` crate exists per the v2 architecture, add a thin shim
  that mounts `TracingHandler` automatically.

## 14. Open questions to revisit during implementation

- Should `Span.input` / `Span.output` be redacted by default? Some users
  send PII in prompts. Likely add a `Redactor` trait in v1.1 with
  no-op default.
- Whether to expose a `Drop`-based blocking flush on
  `TracingHandler` itself or require explicit `shutdown().await`. We
  ship both; the docs recommend `shutdown` for graceful exit.
- Whether `parent_run_id` should be made required on `RunnableConfig`
  (no `Option`) once Phase 2 plumbing is complete. Likely yes.
