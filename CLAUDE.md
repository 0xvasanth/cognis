# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Cognis is a **Rust-native LLM framework** that takes the conceptual foundations of the Python LangChain/LangGraph/DeepAgents ecosystem and reimagines them with idiomatic Rust design patterns. It is not a line-by-line port — it translates Python's *intent* into Rust's *strengths*: type safety, zero-cost abstractions, ownership semantics, and compile-time guarantees.

The workspace contains five library crates and one example crate.

## Build & Test Commands

```bash
# Build entire workspace
cargo build

# Build a single crate
cargo build -p cognis-core

# Build with provider features (required for chat model code)
cargo build -p cognis --features openai,anthropic
cargo build -p cognis --features all-providers

# Run all tests
cargo test

# Run tests for a single crate
cargo test -p cognis-core
cargo test -p cognisgraph

# Run a single test by name
cargo test -p cognis-core test_name

# Run an example (examples live in examples/ but are registered under crates/examples)
cargo run -p cognis-examples --example simple_chain

# Check without building
cargo check --workspace
```

## Workspace Architecture

```
crates/
├── cognis-core     # Zero workspace deps. Base traits, type-safe Runnable pipeline,
│                   # Message enum, output parsers, prompts, vectorstores, embeddings.
│                   # This is the foundation — everything depends on it, it depends on nothing.
├── cognis          # Depends on cognis-core. Concrete implementations: chat model providers
│                   # (behind feature flags), memory, document loaders, text splitters,
│                   # tools, embeddings, retrievers. Re-exports core as `cognis::core`.
├── cognisgraph     # Depends on cognis-core. StateGraph builder, Pregel execution engine,
│                   # channels, checkpointing (sqlite/postgres behind features), streaming,
│                   # Command/Send flow control, prebuilt agents. Exports START/END constants.
├── cognisagent     # Depends on all three above. High-level agent factory (create_deep_agent),
│                   # middleware system, backends (file I/O + state), tool registry,
│                   # planning, multi-agent orchestration.
├── cognis-macros   # Proc macros for derive(JsonSchema), derive(ToolSchema).
└── examples        # Non-publishable crate that registers all examples from examples/ dir
```

## Dependency Rules (strict, enforced)

- `cognis-core` has **zero** workspace crate dependencies
- `cognisgraph` depends only on `cognis-core`
- `cognis` depends only on `cognis-core`
- `cognisagent` depends on `cognis-core`, `cognis`, and `cognisgraph`
- Agent-level middleware (filesystem, memory, subagent, planning, skills) belongs **only** in `cognisagent`, never in `cognis`
- `cognis` contains **provider implementations, data utilities, and composable building blocks** — not agent orchestration

## Feature Flags

**cognis crate:** LLM providers are feature-gated. Each provider (`openai`, `anthropic`, `google`, `ollama`, `azure`) pulls in `reqwest` and `secrecy`. Use `all-providers` for everything. Loader features: `pdf`, `yaml`, `toml-loader`. Storage: `sqlite`.

**cognisgraph crate:** `sqlite` and `postgres` features gate checkpoint persistence via `sqlx`.

## Worktrees

Worktree directory: `.worktrees/` (project-local, globally gitignored)

```bash
git worktree add .worktrees/my-feature -b feature/my-feature
git worktree list
git worktree remove .worktrees/my-feature
```

---

# Migration Priorities & Rust Design Principles

The sections below define **how** Python concepts translate to Rust, **what** to prioritize, and **what rules to follow** when implementing features from the original LangChain/LangGraph/DeepAgents Python codebase.

## Priority Tiers

Work on Cognis follows a strict priority order. Higher tiers must be solid before lower tiers are expanded. Within each tier, items are ordered by importance.

### P0 — Structural Correctness (must be right before anything else)

1. **Generic `Runnable<I, O>` trait** — The `Runnable` trait must be generic over input and output types, not `Value`-typed. This is the single most important design decision. Python uses `Runnable[Input, Output]` with generic type propagation through chains. Rust must do the same — losing compile-time type safety defeats the purpose of the port. Use `serde_json::Value` only at system boundaries (user input, serialization) and provide a `DynRunnable` type-erased wrapper for heterogeneous composition when needed.

2. **Crate boundary enforcement** — Agent-specific middleware (filesystem ops, memory injection, subagent spawning, summarization, skills, planning) must live in `cognisagent`, not in `cognis`. The `cognis` crate provides building blocks (providers, loaders, splitters, retrievers, memory stores). If it orchestrates agent behavior, it belongs in `cognisagent`.

3. **Graph `Command` abstraction** — The `Command` type (state update + goto + resume) is essential for production graph workflows. It enables cross-graph communication, interrupt resumption, and dynamic routing. This must exist in `cognisgraph::types`.

4. **Graph state inspection** — `CompiledStateGraph` must expose `get_state()`, `update_state()`, and `get_state_history()` for human-in-the-loop workflows, debugging, and time-travel. Without these, the graph engine is not production-usable.

### P1 — Core API Completeness

5. **Structured event streaming** — Implement `stream_events()` on Runnable, matching Python's `astream_events`. This emits typed `StreamEvent` variants (on_chain_start, on_llm_token, on_tool_end, etc.) for observability across nested runnables. Critical for production monitoring.

6. **Runtime configurability** — `configurable_fields()` and `configurable_alternatives()` on Runnable, allowing runtime parameter swaps (e.g., swap model or prompt without rebuilding the chain). Use Rust's type system: a `Configurable<R: Runnable>` wrapper with a config key registry.

7. **Backend protocol alignment** — `cognisagent::Backend` must expose **file operations** (read, write, edit, ls, glob, grep) matching the Python `BackendProtocol`, not just state persistence (save/load). The agent's workspace is a file abstraction. State persistence is a separate concern handled by the checkpointer.

8. **All stream modes** — `cognisgraph` must support all 7 stream modes: `values`, `updates`, `messages`, `tasks`, `checkpoints`, `debug`, `custom`. The `custom` mode uses a `StreamWriter` callback that nodes can write arbitrary data to.

9. **Schema introspection** — Runnables should expose `input_schema()` and `output_schema()` returning JSON Schema values. This enables runtime validation, API serving, and documentation generation.

### P2 — Ergonomics & Completeness

10. **Operator-based composition** — Implement `BitOr` for composing runnables: `let chain = prompt | model | parser;`. This is the signature DX of LCEL. In Rust, use a `Pipe<A, B>` struct returned by `impl BitOr`.

11. **State schema with reducers** — Provide a `#[derive(GraphState)]` macro that generates per-field reducers from attributes, e.g. `#[reducer(append)]`, `#[reducer(last_value)]`, `#[reducer(merge)]`. This replaces Python's `Annotated[list, operator.add]` pattern.

12. **Durability modes** — Implement `Durability::Sync`, `Durability::Async`, `Durability::Exit` on graph compilation to control checkpoint timing relative to step execution.

13. **Deprecate standalone Chain types** — Python has deprecated `Chain` in favor of LCEL composition. Do not add new chain types to `cognis`. Express multi-step workflows as `Runnable` compositions. Existing chain types may remain for backward compatibility but should not be the recommended API.

14. **TodoList and PromptCaching middleware** — Port `TodoListMiddleware` and `AnthropicPromptCachingMiddleware` into `cognisagent`. These are part of the default DeepAgents middleware stack.

### P3 — Polish

15. **SubAgent specification parity** — Match Python's `SubAgent` and `CompiledSubAgent` TypedDict formats. Support `name`, `description`, `system_prompt`, `tools`, `model` (provider:model string), `middleware`, `interrupt_on`, `skills`.

16. **Subgraph namespace isolation** — Ensure nested graphs use `checkpoint_ns` for checkpoint isolation, matching Python's namespace semantics.

17. **Provider completeness** — Verify each provider (OpenAI, Anthropic, Google, Ollama, Azure) correctly handles: tool calling serialization, streaming with tool calls, structured output, rate limit errors, and auth failures.

---

## Rust Design Patterns (mandatory)

These patterns define how Python idioms translate to idiomatic Rust. Follow these when implementing any feature.

### Traits, Not Classes

Python's inheritance hierarchy (`BaseLLM → BaseLanguageModel → BaseChatModel`) becomes a flat trait hierarchy in Rust. Prefer composition over deep trait inheritance.

```
Python                          Rust
──────                          ────
class BaseLLM(ABC)            → trait Llm: Send + Sync
class BaseChatModel(BaseLLM)  → trait ChatModel: Llm
class BaseTool(ABC)           → trait Tool: Send + Sync
class BaseRetriever(ABC)      → trait Retriever: Send + Sync
class Runnable[I, O](ABC)    → trait Runnable<I, O>: Send + Sync
```

Traits should have **minimal required methods** and generous **provided methods** (defaults). A provider implementing `ChatModel` should only need to implement `generate()` and `model_id()` — everything else (`stream`, `invoke`, `bind_tools`, `with_structured_output`) has sensible defaults.

### Typestate and Builder Patterns

Use the **builder pattern** for complex configuration, and **typestate** where construction must follow a specific order.

```rust
// Builder — flexible, order-independent
let model = ChatOpenAI::builder()
    .model("gpt-4o")
    .temperature(0.7)
    .api_key_from_env("OPENAI_API_KEY")
    .build()?;

// Typestate — enforced construction order for graphs
let graph = StateGraph::new()
    .add_node("agent", agent_fn)       // returns StateGraph<HasNodes>
    .add_edge(START, "agent")          // returns StateGraph<HasEdges>
    .compile(checkpointer)?;           // returns CompiledStateGraph
```

### Enums Over Polymorphism

Where Python uses union types or subclass dispatch, Rust uses enums with exhaustive matching.

```rust
// Message — exhaustive, no "unknown type" at runtime
pub enum Message {
    Human(HumanMessage),
    Ai(AiMessage),
    System(SystemMessage),
    Tool(ToolMessage),
}

// ToolChoice — closed set, compile-time complete
pub enum ToolChoice {
    Auto,
    Any,
    Named(String),
    None,
}

// StreamMode — all variants known at compile time
pub enum StreamMode {
    Values,
    Updates,
    Messages,
    Tasks,
    Checkpoints,
    Debug,
    Custom,
}
```

### Error Handling

Each crate defines its own error enum via `thiserror`. Errors must be **actionable** — the caller should be able to match on the variant and decide what to do.

```rust
// cognis-core
#[derive(Debug, thiserror::Error)]
pub enum CognisError {
    #[error("LLM provider error: {provider}: {message}")]
    ProviderError { provider: String, message: String },
    #[error("rate limit exceeded, retry after {retry_after_ms}ms")]
    RateLimited { retry_after_ms: u64 },
    #[error("tool `{name}` failed: {reason}")]
    ToolError { name: String, reason: String },
    // ...
}
```

Cross-crate errors use `From` conversions — `cognisgraph::Error` can wrap `cognis_core::CognisError` transparently.

### Ownership and Lifetimes

- **`Arc<dyn Trait>`** for shared ownership of trait objects (models, tools, middleware)
- **`&self`** for stateless operations (most trait methods)
- **`&mut self`** only when interior mutation is required and `&self` + interior mutability would be misleading
- **Avoid `Clone` on large data** — pass references or `Arc`; clone only small config structs
- **Prefer borrowing over cloning** in function signatures: `fn process(messages: &[Message])` not `fn process(messages: Vec<Message>)`

### Async Conventions

- All I/O-bound trait methods are **async** via `#[async_trait]`
- Sync wrappers are not required — Rust's async ecosystem is mature enough
- Use `tokio` as the runtime, `futures::Stream` for streaming
- Concurrency control via `buffer_unordered(n)` on streams, respecting `RunnableConfig::max_concurrency`

### Feature Flags for Optional Dependencies

Every external integration (HTTP providers, database checkpointers, vector store clients) must be behind a feature flag. The core crates compile with zero network dependencies.

```toml
[features]
default = []
openai = ["reqwest", "secrecy"]
anthropic = ["reqwest", "secrecy"]
sqlite = ["sqlx/sqlite"]
postgres = ["sqlx/postgres"]
all-providers = ["openai", "anthropic", "google", "ollama", "azure"]
```

### Macros for Boilerplate Reduction

Use `cognis-macros` for derive macros that eliminate boilerplate:

- `#[derive(JsonSchema)]` — generate JSON Schema from struct fields
- `#[derive(ToolSchema)]` — generate tool metadata (name, description, args_schema)
- `#[derive(GraphState)]` — generate state reducers from field attributes

Macros must produce code that is **readable when expanded** (`cargo expand`). No hidden magic.

### Testing Patterns

- Every trait must have a `Fake*` or `Mock*` implementation in the crate for testing
- Use `#[tokio::test]` for async tests
- Provider tests that require API keys go behind `#[cfg(feature = "integration_tests")]`
- Graph tests should verify both the happy path and interrupt/resume cycles

---

## What NOT to Port Directly

Some Python patterns should be **deliberately skipped or redesigned** in Rust:

| Python Pattern | Rust Approach |
|---------------|---------------|
| `Chain` base class (deprecated) | Compose with `Runnable` trait + `\|` operator |
| Pydantic runtime validation | `serde` for deserialization + custom validators at boundaries |
| `@classmethod` constructors | Associated functions or builder pattern |
| Mixin classes | Trait composition, no mixins |
| `**kwargs` pass-through | Explicit config structs or builder methods |
| Global callback manager | Thread-local or config-carried `CallbackManager` |
| `asyncio.to_thread()` for sync compat | Async-only API; callers use `block_on()` if needed |
| Dynamic attribute access (`getattr`) | Enum dispatch or trait objects |
| `langchain_classic` chains module | Do not port; build equivalent as Runnable compositions |

---

## Code Quality Rules

- All public types, traits, and functions must have `///` doc comments
- No `unwrap()` or `expect()` in library code — propagate errors with `?`
- No `println!` in library code — use `tracing` crate macros (`tracing::info!`, `tracing::debug!`)
- Clippy must pass: `cargo clippy --workspace --all-targets -- -D warnings`
- Format with `cargo fmt` before every commit
- Every new public API needs at least one unit test and one doc-test

## Pre-Push Checklist

Before pushing to remote or creating a PR, run these checks locally **in order**:

```bash
# 1. Format
cargo fmt --all

# 2. Clippy (matches CI exactly)
cargo clippy --workspace --features all-providers -- -D warnings

# 3. Tests
cargo test --workspace

# 4. Code review with Cubic
cubic review
```

All four must pass before pushing. If `cubic review` flags issues, fix them before pushing — do not defer review findings to the PR cycle.
