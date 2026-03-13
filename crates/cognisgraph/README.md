<div align="center">

# cognisgraph

**Stateful, multi-actor agent workflows as executable graphs.**

[![crates.io](https://img.shields.io/crates/v/cognisgraph.svg)](https://crates.io/crates/cognisgraph)
[![docs.rs](https://docs.rs/cognisgraph/badge.svg)](https://docs.rs/cognisgraph)
[![MIT](https://img.shields.io/crates/l/cognisgraph.svg)](https://opensource.org/licenses/MIT)

[Workspace](https://github.com/0xvasanth/cognis) | [API Docs](https://docs.rs/cognisgraph)

</div>

---

`cognisgraph` is the orchestration layer of the [Cognis](https://github.com/0xvasanth/cognis) framework. Build agent workflows as directed graphs with conditional branching, checkpointing, streaming, human-in-the-loop interrupts, and subgraph composition. Inspired by Pregel and Apache Beam.

## Quick Start

```toml
[dependencies]
cognisgraph = "0.1"
tokio = { version = "1", features = ["full"] }
serde_json = "1"
```

```rust,ignore
use std::sync::Arc;
use cognisgraph::graph::state::{AsyncNodeAction, StateGraph};
use serde_json::{json, Value};

let classify: AsyncNodeAction = Arc::new(|state: Value| {
    Box::pin(async move {
        let input = state["input"].as_str().unwrap_or("");
        let category = if input.contains("error") { "issue" } else { "general" };
        Ok(json!({ "category": category }))
    })
});

let respond: AsyncNodeAction = Arc::new(|state: Value| {
    Box::pin(async move {
        let cat = state["category"].as_str().unwrap_or("unknown");
        Ok(json!({ "response": format!("Handling as: {cat}") }))
    })
});

let graph = StateGraph::new()
    .add_node("classify", classify)
    .add_node("respond", respond)
    .add_edge("__start__", "classify")
    .add_edge("classify", "respond")
    .add_edge("respond", "__end__")
    .compile()?;

let result = graph.invoke(json!({ "input": "There is an error" })).await?;
```

## Capabilities

**State Graphs** — Define nodes as async functions, connect them with edges, add conditional routing with `add_conditional_edges`.

**Pregel Execution** — Superstep-based execution engine that processes nodes in parallel where the graph allows.

**Checkpointing** — Save and resume graph execution. SQLite and Postgres backends available behind feature flags.

**Streaming** — Stream execution updates with `StreamMode::Values`, `Updates`, or `Debug`.

**Human-in-the-Loop** — Pause execution at any node with `InterruptType::Before` or `After` for approval workflows.

**Subgraphs** — Compose graphs within graphs for modular workflow design.

**Prebuilt Agents** — `create_react_agent` gives you a tool-calling ReAct loop out of the box.

**Retry & Caching** — Per-node `RetryPolicy` and `CachePolicy` for resilient execution.

## Feature Flags

```toml
cognisgraph = { version = "0.1", features = ["sqlite"] }    # SQLite checkpoints
cognisgraph = { version = "0.1", features = ["postgres"] }   # Postgres checkpoints
```

## Part of the Cognis Workspace

| Crate | Role |
|-------|------|
| [cognis-core](https://crates.io/crates/cognis-core) | Foundation traits and types |
| [cognis](https://crates.io/crates/cognis) | LLM providers, chains, memory, tools |
| **cognisgraph** | State graph orchestration engine (you are here) |
| [cognisagent](https://crates.io/crates/cognisagent) | High-level agent framework |
