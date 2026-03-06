# langgraph

Orchestration layer for building stateful, multi-actor agent workflows as directed
graphs. Inspired by Pregel and Apache Beam, this crate provides a `StateGraph` builder
that compiles into an executable graph with checkpointing, streaming, human-in-the-loop
interrupts, and subgraph composition.

## Key Types

| Type | Module | Description |
|------|--------|-------------|
| `StateGraph` | `graph::state` | Builder for defining nodes, edges, and branches |
| `CompiledStateGraph` | `graph::state` | Executable graph produced by `StateGraph::compile` |
| `CheckpointSaver` | `checkpoint` | Trait for persisting execution state |
| `Runtime` | `runtime` | Tokio-based async runtime utilities |
| `StreamUpdate` | `types` | Updates emitted during streaming execution |

## Features

- **Conditional routing** -- `add_conditional_edges` with router functions
- **Streaming** -- `stream` method with `StreamMode::Values`, `Updates`, or `Debug`
- **Checkpointing** -- Save and resume graph execution (SQLite backend via `sqlite` feature)
- **Human-in-the-loop** -- `InterruptType::Before` / `After` for approval workflows
- **Subgraphs** -- Compose graphs within graphs
- **Prebuilt agents** -- `create_react_agent` for tool-calling ReAct loops
- **Retry and caching** -- Per-node `RetryPolicy` and `CachePolicy`

## Usage

```toml
[dependencies]
langgraph = { path = "../langgraph" }
# For SQLite checkpoints:
# langgraph = { path = "../langgraph", features = ["sqlite"] }
```

```rust,ignore
use langgraph::graph::state::StateGraph;
use langgraph::{START, END};
use serde_json::{json, Value};

let mut graph = StateGraph::new();
graph.add_node("greet", |state: Value| Ok(json!({"response": "Hello!"})));
graph.add_edge(START, "greet");
graph.add_edge("greet", END);

let compiled = graph.compile(None).unwrap();
let result = compiled.invoke(json!({})).await.unwrap();
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `sqlite` | SQLite checkpoint persistence via `sqlx` |
