<div align="center">

# cognisagent

**Zero-boilerplate agents with pluggable middleware and backends.**

[![crates.io](https://img.shields.io/crates/v/cognisagent.svg)](https://crates.io/crates/cognisagent)
[![docs.rs](https://docs.rs/cognisagent/badge.svg)](https://docs.rs/cognisagent)
[![MIT](https://img.shields.io/crates/l/cognisagent.svg)](https://opensource.org/licenses/MIT)

[Workspace](https://github.com/0xvasanth/cognis) | [API Docs](https://docs.rs/cognisagent)

</div>

---

`cognisagent` is the application layer of the [Cognis](https://github.com/0xvasanth/cognis) framework. It wraps `cognis` and `cognisgraph` into a batteries-included agent factory — configure once, get a production-ready agent with middleware hooks, storage backends, planning, and multi-agent collaboration.

## Quick Start

```toml
[dependencies]
cognisagent = "0.1"
tokio = { version = "1", features = ["full"] }
serde_json = "1"
```

```rust,ignore
use cognisagent::config::DeepAgentConfig;
use cognisagent::create_deep_agent;

let config = DeepAgentConfig::default();
let graph = create_deep_agent(config)?;
let result = graph.invoke(serde_json::json!({"messages": []})).await?;
```

## Middleware Pipeline

The `Middleware` trait provides four hooks around every model and tool call:

```text
before_model  ->  LLM call  ->  after_model
before_tool   ->  Tool exec  ->  after_tool
```

Built-in middleware:

| Middleware | What it does |
|-----------|-------------|
| `FilesystemMiddleware` | File read, write, list, glob, grep |
| `MemoryMiddleware` | Inject persistent memory into context |
| `SubAgentMiddleware` | Delegate tasks to isolated sub-agents |
| `SummarizationMiddleware` | Auto-summarize when context window fills |
| `PlanningMiddleware` | Plan-then-execute with step tracking |
| `ContextMiddleware` | Dynamic context injection |
| `RateLimiterMiddleware` | Token bucket rate limiting |
| `LoggingMiddleware` | Structured logging with PII redaction |
| `PatchToolCallsMiddleware` | Auto-correct malformed tool calls |
| `SkillsMiddleware` | Load and invoke custom skills |

## Storage Backends

| Backend | Description |
|---------|-------------|
| `StateBackend` | In-memory (default, zero setup) |
| `FilesystemBackend` | Persist sessions as JSON on disk |
| `SandboxBackend` | Isolated execution with resource limits |

## More Features

**Tool Registry** — Centralized tool management with permission levels, call counting, enable/disable, and filtering.

**Planning** — Plan-then-execute pattern with step-by-step tracking and replanning.

**Multi-Agent** — Orchestrate multiple agents with shared state and message passing.

**Presets** — Ready-made agent configurations via a preset registry for common agent types.

**Evaluation** — Built-in evaluation framework for testing agent quality.

**Health Monitoring** — Diagnostics, profiling, and alerting for production deployments.

## Part of the Cognis Workspace

| Crate | Role |
|-------|------|
| [cognis-core](https://crates.io/crates/cognis-core) | Foundation traits and types |
| [cognis](https://crates.io/crates/cognis) | LLM providers, chains, memory, tools |
| [cognisgraph](https://crates.io/crates/cognisgraph) | State graph orchestration engine |
| **cognisagent** | High-level agent framework (you are here) |
