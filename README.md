# RustChain -- LLM Application Framework in Rust

RustChain is a modular framework for building LLM-powered applications in Rust.
It provides composable abstractions for chat models, tool calling, agent orchestration,
and stateful multi-step workflows, with async-first design built on tokio.

## Architecture

```
+-----------------------------------------------------------+
|               deepagents  (Application Layer)              |
|  create_deep_agent(), middleware hooks, storage backends   |
+--------------------------+--------------------------------+
                           | uses
+--------------------------v--------------------------------+
|               langgraph  (Orchestration Layer)             |
|  StateGraph, Pregel engine, checkpoints, streaming,        |
|  ReAct agent, human-in-the-loop, subgraph composition      |
+--------------------------+--------------------------------+
                           | uses
+--------------------------v--------------------------------+
|               rustchain  (Implementation Layer)            |
|  Chat models (Anthropic, OpenAI, Google, Ollama, Azure),   |
|  embeddings, agents, chains, memory, document loaders,     |
|  text splitters, tools                                     |
+--------------------------+--------------------------------+
                           | uses
+--------------------------v--------------------------------+
|            rustchain-core  (Foundation Layer)               |
|  Base traits: BaseChatModel, BaseTool, Runnable, Message   |
|  Prompts, output parsers, callbacks, vector stores         |
+-----------------------------------------------------------+
```

## Quick Start

Add `rustchain-core` to your `Cargo.toml`:

```toml
[dependencies]
rustchain-core = { path = "crates/rustchain-core" }
```

Build a simple chain with the fake chat model:

```rust
use rustchain_core::language_models::FakeListChatModel;
use rustchain_core::messages::Message;
use rustchain_core::runnables::Runnable;
use serde_json::json;

#[tokio::main]
async fn main() {
    let model = FakeListChatModel::new(vec!["Hello! How can I help?".into()]);
    let input = json!({
        "messages": [Message::human("Hi there")]
    });
    let result = model.invoke(input, None).await.unwrap();
    println!("{:?}", result);
}
```

## Features by Crate

### rustchain-core

- **Messages** -- `Message` enum with Human, AI, System, Tool, and Function variants
- **Language models** -- `BaseChatModel` and `BaseLLM` traits, plus fake/testing models
- **Runnables** -- Composable `Runnable` trait with sequence, parallel, branch, lambda, retry, fallback
- **Tools** -- `BaseTool` trait and toolkit interface for agent tool calling
- **Prompts** -- Chat prompt templates, few-shot selectors, structured prompts
- **Output parsers** -- JSON, string, list, XML, and tool-call parsers
- **Callbacks** -- Extensible callback system with run managers and tracers
- **Vector stores** -- In-memory vector store with embeddings interface
- **Indexing** -- Document indexing with record managers

### rustchain

- **Chat models** -- Anthropic Claude, OpenAI GPT, Google Gemini, Ollama, Azure OpenAI
- **Embeddings** -- OpenAI and Ollama embedding providers
- **Agents** -- Agent executor with middleware pipeline (retry, PII redaction, summarization, etc.)
- **Chains** -- LLM chain, conversation chain, sequential chain
- **Memory** -- Buffer, window, and summary memory strategies
- **Document loaders** -- Text, CSV, JSON, and directory loaders
- **Text splitters** -- Character, recursive, markdown, HTML, JSON, code, and token splitters
- **Tools** -- Calculator, shell command, and JSON query tools

### langgraph

- **State graphs** -- `StateGraph` builder with sync/async node actions and conditional routing
- **Pregel engine** -- Execution engine inspired by Pregel/Apache Beam
- **Channels** -- LastValue, BinaryOp, Topic, AnyValue, NamedBarrier, EphemeralValue
- **Checkpointing** -- `CheckpointSaver` trait with SQLite backend
- **Streaming** -- Stream graph execution via `StreamMode` (values, updates, debug)
- **Prebuilt agents** -- `create_react_agent` for tool-calling ReAct loops
- **Human-in-the-loop** -- Interrupt support for approval workflows
- **Subgraph composition** -- Nested graph execution

### deepagents

- **Agent factory** -- `create_deep_agent()` builds a compiled graph with middleware
- **Middleware** -- `Middleware` trait with before/after hooks for model and tool calls
  - `FilesystemMiddleware` -- file read/write/list/glob/grep
  - `MemoryMiddleware` -- persistent memory injection
- **Backends** -- `Backend` trait for session state persistence
  - `StateBackend` -- in-memory storage
  - `FilesystemBackend` -- local disk storage

## Feature Flags

| Crate | Feature | Description |
|-------|---------|-------------|
| `rustchain` | `anthropic` | Anthropic Claude chat model |
| `rustchain` | `openai` | OpenAI GPT chat model and embeddings |
| `rustchain` | `google` | Google Gemini chat model |
| `rustchain` | `ollama` | Ollama local chat model and embeddings |
| `rustchain` | `azure` | Azure OpenAI chat model |
| `rustchain` | `all-providers` | Enable all provider integrations |
| `langgraph` | `sqlite` | SQLite checkpoint persistence via sqlx |

## Installation

Add the crates you need to your `Cargo.toml`:

```toml
[dependencies]
rustchain-core = { git = "https://github.com/user/rustchain" }
rustchain = { git = "https://github.com/user/rustchain", features = ["anthropic"] }
langgraph = { git = "https://github.com/user/rustchain" }
deepagents = { git = "https://github.com/user/rustchain" }
```

Or reference workspace crates locally:

```toml
[dependencies]
rustchain-core = { path = "../rustchain/crates/rustchain-core" }
```

## Workspace Structure

```
rustchain/
  Cargo.toml                  # Workspace root
  crates/
    rustchain-core/           # Base traits and types
    rustchain/                # Provider implementations and agent framework
    langgraph/                # State graph orchestration engine
    deepagents/               # High-level agent factory
```

## License

MIT
