# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

Cognis is a Rust-native LLM framework. It translates the conceptual surface of LangChain / LangGraph / DeepAgents (Runnable, agent loop, graph state, RAG pipeline, observability) into idiomatic Rust — typed traits, ownership-aware APIs, compile-time guarantees. It is not a line-by-line port.

The workspace is organized as a foundation crate, four sibling capability crates, a proc-macro crate, an umbrella, and an examples crate.

## Workspace

```
crates/
├── cognis-core    # Foundation. Zero internal-crate deps.
│                  # Runnable<I,O>, Message, ContentPart, prompts, output
│                  # parsers (JsonParser, StructuredOutputParser,
│                  # OutputFixingParser, RetryParser, …), callbacks +
│                  # Observer + Event, wrappers (Cache/Retry/Timeout/
│                  # Fallback/Bind/Configurable/Listeners), security,
│                  # compose (lambda/pipe/Branch/Parallel/Each).
├── cognis-llm     # LLM client + provider abstractions. Provider impls
│                  # (OpenAI/Anthropic/Google/Ollama/Azure/OpenRouter)
│                  # are feature-gated. Tool trait, ToolInput/Output,
│                  # SchemaBasedTool, factory, registry, streaming,
│                  # structured output.
├── cognis-rag     # RAG primitives: embeddings (Fake, OpenAI, Google,
│                  # Ollama, Voyage, Cached, Batched, Router), vector
│                  # stores (in-memory, FAISS, Chroma, Qdrant, Pinecone,
│                  # Weaviate), retrievers, splitters, loaders,
│                  # IndexingPipeline + RecordManager, transformers
│                  # (Dedup, Enrichment, MetadataTransformer,
│                  # LongContextReorder).
├── cognisgraph    # Crate name `cognis-graph`. Stateful Graph<S>,
│                  # Pregel-style engine, reducers, channels,
│                  # checkpointers (in-memory/SQLite/Postgres),
│                  # interrupts, time-travel, all 7 stream modes,
│                  # viz (DOT/Mermaid/ASCII).
├── cognis-trace   # Pluggable observability: bridges CallbackHandler
│                  # events to Langfuse, LangSmith, OpenTelemetry.
├── cognis-macros  # Proc macros: #[tool], #[derive(GraphState)].
├── cognis         # Umbrella + agent layer. AgentBuilder,
│                  # MultiAgentOrchestrator (Sequential / Supervisor /
│                  # ParallelVote / RoundRobin), AgentBus pub-sub,
│                  # memory variants (Buffer / Window / TokenBuffer /
│                  # SummaryBuffer / Vector / Entity / KnowledgeGraph),
│                  # middleware (rate limiters, model retry, fallback,
│                  # PII, prompt-caching, planning, summarization, todo,
│                  # …), built-in tools, ToolOrchestrator. Re-exports
│                  # core/llm/rag/graph at the top level.
└── examples       # Non-publishable. Demos under examples/<category>/
                   # registered here with [[example]] entries.
```

### Crate dependency rules (strict)

- `cognis-core` has **zero** internal-crate dependencies.
- `cognis-llm`, `cognis-rag`, `cognis-graph`, `cognis-trace` depend only on `cognis-core` (and `cognis-macros` where they need a derive).
- `cognis` is the only crate that depends on the four siblings together.
- `cognis-macros` is proc-macro-only; consumed by everything that wants the derives.

### Where new code goes

| Concept | Crate |
|---|---|
| LLM client, provider, chat options, tool dispatch, structured output | `cognis-llm` |
| Embeddings, vector stores, retrievers, splitters, loaders, indexing | `cognis-rag` |
| Graph<S>, nodes, channels, reducers, checkpoints, viz | `cognisgraph` |
| Tracing / observability adapter (Langfuse, OTel, …) | `cognis-trace` |
| Agent loop, multi-agent, agent bus, middleware, built-in tools, memory variants, eval | `cognis` |
| Pure trait/type everyone needs (Runnable, prompts, output parsers, error, Message) | `cognis-core` |
| Anything that does I/O against an external service | feature-gated, opt-in |

If a feature could plausibly live in two places, default to the smaller crate (closer to `cognis-core`).

## Build & test

```bash
cargo build --workspace
cargo build -p cognis --features all-providers
cargo test --workspace
cargo test -p cognis --lib agent::memory                # one module
cargo test -p cognis-core --lib output_parsers::fixing  # one file

cargo run -p cognis-examples --example tools_orchestrator
COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.2:1b \
  cargo run -p cognis-examples --example agents_react_agent
```

Workspace metadata is hoisted: `version`, `edition`, `license`, `authors` live in `[workspace.package]`. Member crates use `version.workspace = true` etc.

## Feature flags

- `cognis`: providers (`openai`, `anthropic`, `google`, `ollama`, `azure`, `openrouter`, `all-providers`); loaders (`pdf`, `yaml`, `toml-loader`); storage (`cache-sqlite`); HTTP tools (`tools-http`).
- `cognisgraph`: `sqlite`, `postgres` for checkpointer backends.
- `cognis-rag`: per-vector-store features (`faiss`, `chroma`, …) and per-embedding features (`openai`, `voyage`, `google`).
- `cognis-core` and `cognis-macros` compile with no network features ever.

## Worktrees

```bash
git worktree add .worktrees/feature-foo -b feature/foo
git worktree list
git worktree remove .worktrees/feature-foo
```

`.worktrees/` is project-local and globally gitignored.

---

# Design rules

## Generic, type-safe Runnable

`Runnable<I, O>` is generic — never `Value`-typed at trait boundaries. Use `serde_json::Value` only at system boundaries (user input, persistence, wire serialization). For heterogeneous composition, use type-erased wrappers (`DynRunnable`) and convert at the edge, not throughout.

## Traits, not classes

Each capability is a trait with a small required surface and generous provided defaults. Examples in the codebase:

- `Runnable<I, O>` — required: `invoke`. Provided: `name`, `stream_events`, `pipe`, `with_retry`, …
- `Memory` — required: `read`, `write`, `clear`. Provided: `seed`.
- `Tool` — required: `name`, `description`, `args_schema`, `_run`. Provided: `return_direct`.
- `RateLimiter` — required: `acquire`. Period.
- `OutputParser<T>` — required: `parse`. Provided: `format_instructions`.

When adding a new variant of an existing trait, mirror the builder methods and naming of its siblings. A new `Memory` impl ships `with_system(...)` because the others do. A new `RateLimiter` impl reuses the `acquire(estimated_tokens)` shape.

## Builders

`with_X(self, x) -> Self` chain. Returns `Self`, not `&mut Self`. Constructors don't take optional args — promote them to builder methods.

```rust
let bucket = TokenBucket::new(rate, burst);
let parser = RetryParser::with_retries(inner, fixer, 5);
let plan = ExecutionPlan::new()
    .step(ToolStep::new("a", "fetch", args))
    .step(ToolStep::new("b", "merge", args2).after(["a"]));
```

## Enums for closed sets

Where Python uses union types or magic strings, Rust uses enums with exhaustive matching. In the codebase: `Message`, `ContentPart`, `ToolChoice`, `StreamMode`, `Goto`, `Durability`, `SubscribeError`.

## Error handling

Each crate below the umbrella defines its own error type via `thiserror`. Cross-crate errors use `From` conversions — `cognis_graph::Error` wraps `cognis_core::CognisError`.

The umbrella `cognis` crate does **not** depend on `thiserror` directly. When adding a new error type there, hand-roll `Display` and `Error`:

```rust
#[derive(Debug)]
pub enum SubscribeError { Closed, Lagged(u64), Empty }
impl std::fmt::Display for SubscribeError { /* ... */ }
impl std::error::Error for SubscribeError {}
```

Errors must be **actionable**. Callers should be able to match on the variant and decide what to do.

## Ownership

- `Arc<dyn Trait>` for shared trait objects (models, tools, fixers, limiters).
- `Arc<dyn Fn(...) + Send + Sync>` for closures stored on structs (entity extractors, dedup key fns, etc.).
- `&self` is the default. `&mut self` only when interior mutability would be misleading. Most trait methods are `&self` and use `RwLock` / `Mutex` / `AtomicUsize` for state.
- `PhantomData<fn() -> T>` for unused type parameters that influence the type signature only.
- Avoid `Clone` on large data — pass references or `Arc`. Clone small config structs freely.

## Async

- I/O-bound trait methods are async via `#[async_trait]`.
- `tokio` runtime, `futures::Stream` for streaming.
- Concurrent fan-out: `FuturesUnordered<BoxFuture<'static, _>>`. The `Box::pin` is required because each spawned async block has its own anonymous type.
- Concurrency cap on streams: `buffer_unordered(n)` honoring `RunnableConfig::max_concurrency`.
- Sync wrappers are not provided — callers can use `tokio::runtime::Handle::block_on`.

## Feature flags for optional integrations

Every external integration (HTTP providers, DB checkpointers, vector store clients, observability exporters) sits behind a feature flag. Core crates compile with zero network features. New integration → new feature flag, opt-in, additive only.

## Macros

Use `cognis-macros` for boilerplate elimination. Today: `#[tool]`, `#[derive(GraphState)]`. Generated code must be readable when expanded (`cargo expand`) — no hidden magic.

For JSON Schema derivation, use `schemars` directly via the `cognis_core::schemars::JsonSchema` re-export. The legacy hand-rolled `JsonSchema` / `ToolSchema` derives were retired.

---

# Conventions

## Documentation

- File-level `//!` doc is **mandatory** on every module. State the module's role and when to reach for it. 1–3 lines is plenty.
- Public types, traits, fns get `///` doc. Explain the *why* and the *when*.
- For new types that contrast with existing siblings, lead with the contrast:
  > Different from `super::TokenBucket` (fixed rate). Use this when the upstream API enforces a rolling window.
- Cross-references via `[`Type`]` syntax, not bare names.
- Don't write WHAT comments. Identifiers and types already say WHAT. Comment only the WHY (a hidden constraint, an invariant, a workaround).

## Code style

- No `unwrap()` / `expect()` in library code — `?` propagation. Tests are the exception.
- No `println!` / `eprintln!` in library code — use `tracing::{info,debug,warn}!`. Examples are allowed `println!`.
- No commented-out code, no `// removed` placeholders, no backwards-compat shims unless the user explicitly asks.
- `cargo fmt` before every commit.
- `cargo clippy --workspace --all-targets -- -D warnings` must pass.

## Re-exports

- Each sub-crate `pub use`s its public types from `lib.rs` so users don't import sub-modules.
- The umbrella `cognis` crate re-exports everything frequently used (`pub use cognis_core::{...}`, etc.) so callers typically only need `use cognis::prelude::*;` plus a few specific imports.
- New public type → first add it to its sub-crate's `lib.rs`; then bubble up to `cognis`'s top-level re-exports if it's commonly used.

## Testing

- Inline `#[cfg(test)] mod tests` per file. Tests live next to the code, not in a separate test folder.
- Use `cognis_core::compose::lambda` to build fake `Runnable`s. Use scripted/canned test impls (`Tagged`, `ScriptedTool`, `CannedProvider`) for trait objects with state. Concurrency-sensitive tests use `Arc<AtomicUsize>` to count invocations.
- `#[tokio::test]` for async tests.
- Integration tests that need API keys go behind `#[cfg(feature = "integration_tests")]`.
- Test names: `verb_does_X_when_Y`. Snake_case, verbose.
- Prefer assertion messages where useful: `assert!(..., "got: {got:?}")`.
- Cover happy path + at least one error/edge case per public method.

## Examples

- One example per public-facing feature, registered in `crates/examples/Cargo.toml` with `[[example]] name = "<category>_<feature>"`.
- File-level `//!` doc explains the demo in 1–3 lines.
- Prefer offline. For offline LLM demos use a `Tagged` / `Canned` provider built from `lambda`. Set the `COGNIS_PROVIDER=ollama` opt-in only when the demo specifically illustrates a real LLM call.
- Output via `println!` is fine in examples (they aren't library code).
- Smoke-test each example before claiming done: `cargo run -p cognis-examples --example <name>`.
- Show one canonical usage; don't fully generalize. Examples are tutorials, not API references.

## Commits

- Conventional Commits prefix: `feat(scope):`, `fix(scope):`, `chore:`, `refactor:`, `docs:`, `test:`. Subject ≤ 70 chars.
- Body wraps at ~72 cols. Explains the *why* and *what changed* at a behavioral level. References parity gaps / issue numbers if relevant.
- One concept per commit. A multi-step task lands as a series of small commits, each green.
- Never add `Co-Authored-By: Claude` (or any AI attribution).
- Never `--no-verify`, never `--no-gpg-sign`, never `git add -f`.
- Build (`cargo build --workspace`) and tests (`cargo test --workspace`) must be green before each commit.
- The user owns commits; don't push without explicit instruction.

## Workflow

- For multi-task work, order tasks topologically: smallest / most-contained first. Each lands as its own commit so the workspace stays buildable between commits.
- Per task: write the code → tests → run tests → run the example → commit.
- Use `TaskCreate` / `TaskUpdate` to track multi-step work; mark each task `in_progress` before starting, `completed` when the commit lands.

---

# What NOT to port from Python

| Python | Cognis approach |
|---|---|
| `Chain` base class | Compose `Runnable`s via `lambda \| pipe \| Branch \| Parallel`. |
| Pydantic runtime validation | `serde` + boundary validation. |
| `@classmethod` constructors | Associated fns or builders. |
| Mixin classes | Trait composition. |
| `**kwargs` pass-through | Explicit config structs / builder methods. |
| Global callback manager | Per-call `RunnableConfig.observers`. |
| `asyncio.to_thread()` | Async-only API; callers `block_on()` if they need sync. |
| Dynamic `getattr` dispatch | Enum variants or trait objects. |
| `langchain_classic.chains` | Don't port. Build equivalents as `Runnable` compositions. |

---

# Pre-Push Checklist

Run in order before `git push`:

```bash
cargo fmt --all
cargo clippy --workspace --features all-providers -- -D warnings
cargo test --workspace
cubic review
```

All must pass. Fix `cubic review` findings locally before pushing — don't defer them to the PR cycle.

---

# Known gotchas

- **`cognis` lacks a direct `thiserror` dep.** When adding a new error type in that crate, hand-roll `Display` / `Error` impls. Sub-crates have `thiserror` and should use it.
- **`#[tool]` macro paths.** The macro defaults `crate_path = "cognis_core"` but the runtime `Tool` trait now lives in `cognis_llm::tools`. For now, prefer hand-written `impl Tool` blocks; the macro will be reconciled in a future change.
- **`LinearBuilder` (cognis-graph).** Auto-names stages `"0"`, `"1"`, … Linear edges between stages exist, but a node's returned `Goto` overrides them — there is no fall-through. If you need linear routing, return `Goto::node("<next-index>")` from each stage or use `Graph::new()` with explicit edges.
- **`FuturesUnordered` requires homogeneous future types.** Push spawned futures via `Box::pin(async move { ... })` and type the collection as `FuturesUnordered<BoxFuture<'static, _>>`.
- **Workspace dep keys must match crate `name`.** The dir is `crates/cognisgraph` but the crate's `name = "cognis-graph"`. Workspace deps key on the crate name (`cognis-graph = { path = "crates/cognisgraph", … }`).
- **Providers ≠ middleware ≠ orchestration.** Provider implementations live in `cognis-llm`; middleware (rate limit, retry, PII, prompt caching, summarization, …) lives in `cognis::middleware`; agent orchestration (multi-agent, agent_bus, memory variants, the agent loop) lives directly in `cognis`. When in doubt, prefer the lower crate.
- **Memory `seed()` may prepend synthesized system messages.** `EntityMemory` and `KnowledgeGraphMemory` both inject a "Known entities:" / "Knowledge:" preamble into `seed()`. Callers should not assume `seed().len() == read().len()`.
