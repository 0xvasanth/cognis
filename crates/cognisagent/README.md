# cognisagent

Batteries-included, high-level agent framework built on `cognis` and `cognisgraph`.
Provides zero-boilerplate agent creation with pluggable middleware and storage backends.

## Key Types

| Type | Module | Description |
|------|--------|-------------|
| `create_deep_agent` | `agent` | Factory that builds a compiled CognisGraph from config |
| `DeepAgentConfig` | `config` | Configuration for model, tools, middleware, backend |
| `Middleware` | `middleware` | Trait for before/after hooks on model and tool calls |
| `Backend` | `backends` | Trait for session state persistence |

## Middleware

The `Middleware` trait provides four hooks:

- `before_model` -- mutate state before the LLM is called (e.g., inject context)
- `after_model` -- inspect or modify the model response
- `before_tool` -- run logic before a tool executes
- `after_tool` -- run logic after a tool completes

Built-in middleware:

- **FilesystemMiddleware** -- file read, write, list, glob, grep operations
- **MemoryMiddleware** -- inject persistent memory into the agent context

## Backends

- **StateBackend** -- in-memory state storage (default)
- **FilesystemBackend** -- persist sessions to local disk as JSON files

## Usage

```toml
[dependencies]
cognisagent = { path = "../cognisagent" }
```

```rust,ignore
use cognisagent::config::DeepAgentConfig;
use cognisagent::create_deep_agent;

let config = DeepAgentConfig::default();
let graph = create_deep_agent(config).unwrap();
let result = graph.invoke(serde_json::json!({"messages": []})).await.unwrap();
```
