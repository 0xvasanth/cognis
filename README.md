<div align="center">

# RustChain

**A comprehensive Rust implementation of the LangChain ecosystem — fast, type-safe, and composable.**

[![Build Status](https://img.shields.io/github/actions/workflow/status/0xvasanth/rustchain/ci.yml?branch=main&label=build)](https://github.com/0xvasanth/rustchain/actions)
[![Crate Version](https://img.shields.io/badge/crates.io-v0.1.0-blue)](https://crates.io/crates/rustchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![Tests](https://img.shields.io/badge/tests-2%2C738-green)](.)
[![Lines of Code](https://img.shields.io/badge/lines-112K-informational)](.)

[Getting Started](#getting-started) · [Examples](#examples) · [Architecture](#architecture) · [Feature Flags](#feature-flags) · [Contributing](#contributing)

</div>

---

## Overview

RustChain is a Rust-native LLM application framework that ports the Python [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph), and [DeepAgents](https://github.com/langchain-ai/deepagents) ecosystem to Rust. It provides composable abstractions for building chains, agents, RAG pipelines, and stateful graph workflows — all with Rust's performance, type safety, and fearless concurrency.

The project spans **4 crates**, **363 source files**, **~112K lines of Rust**, and **~2,738 tests**.

### Why Rust?

- **Type-safe by design** — Catch tool schema mismatches, invalid state transitions, and message type errors at compile time.
- **Zero-cost abstractions** — Traits and generics instead of runtime reflection. No garbage collector overhead.
- **Async-first** — Built on `tokio` with native streaming support via `futures::Stream`.
- **Modular** — LLM providers are behind feature flags so you don't pay for what you don't use.
- **Composable** — Chain prompts, models, parsers, and tools using the LCEL-inspired `chain!` macro and `Runnable` trait.
- **Production patterns** — Circuit breakers, rate limiting, retry with backoff, PII redaction, and human-in-the-loop approval built into the agent middleware pipeline.

---

## Architecture

RustChain is organized as a Cargo workspace with four crates, each with clear responsibility and strict dependency boundaries:

```
+-----------------------------------------------------------+
|               deepagents  (Application Layer)              |
|  create_deep_agent(), middleware hooks, storage backends   |
+--------------------------+--------------------------------+
                           | depends on
+--------------------------v--------------------------------+
|               langgraph  (Orchestration Layer)             |
|  StateGraph, Pregel engine, checkpoints, streaming,        |
|  ReAct agent, human-in-the-loop, subgraph composition      |
+--------------------------+--------------------------------+
                           | depends on
+--------------------------v--------------------------------+
|               rustchain  (Implementation Layer)            |
|  Chat models (5 providers), chains (12 types), memory,     |
|  document loaders, text splitters, agents, tools           |
+--------------------------+--------------------------------+
                           | depends on
+--------------------------v--------------------------------+
|            rustchain-core  (Foundation Layer)               |
|  Base traits: BaseChatModel, BaseTool, Runnable, Message   |
|  Prompts, output parsers, callbacks, vector stores         |
+-----------------------------------------------------------+
```

### Dependency Rules

| Crate | Allowed Dependencies |
|---|---|
| `rustchain-core` | Zero workspace dependencies |
| `rustchain` | `rustchain-core` only |
| `langgraph` | `rustchain-core` only |
| `deepagents` | `rustchain-core`, `rustchain`, `langgraph` |

---

## Features by Crate

### rustchain-core — Foundation Traits and Types

| Module | Description |
|---|---|
| `messages` | `Message` enum (Human, AI, System, Tool, Function) with streaming chunks, merge, and trim utilities |
| `language_models` | `BaseChatModel` and `BaseLLM` traits, plus fake/testing models |
| `runnables` | `Runnable` trait with sequence, parallel, branch, lambda, retry, fallback, and 30+ combinators |
| `tools` | `BaseTool` trait and toolkit interface for agent tool calling |
| `prompts` | Chat prompt templates, few-shot selectors, structured prompts, image prompts |
| `output_parsers` | JSON, string, list, XML, and tool-call parsers |
| `callbacks` | Extensible callback system with run managers and tracers |
| `vectorstores` | `VectorStore` trait and `InMemoryVectorStore` |
| `embeddings` | `Embeddings` trait for vector embedding providers |
| `documents` | `Document` type used across loaders, splitters, and retrievers |
| `retrievers` | `BaseRetriever` trait for document retrieval |
| `indexing` | Document indexing with record managers |
| `tracers` | stdout, OpenTelemetry, event/log stream tracers |
| `caches` | Caching infrastructure for LLM responses |
| `chat_history` | Chat history management interfaces |
| `stores` | Key-value store abstractions |

### rustchain — Implementations and Provider Integrations

| Module | Description |
|---|---|
| `chat_models` | **5 providers**: Anthropic Claude, OpenAI GPT, Google Gemini, Ollama, Azure OpenAI. **Wrappers**: cached, circuit breaker, rate limited, retrying, structured, token counting, graceful |
| `chains` | **12 types**: LLM, conversation, conversation retrieval, sequential, retrieval QA, map-reduce, refine, router, structured output, summarize, SQL, API |
| `memory` | **6 types**: buffer, window, summary, vector, chat history, hybrid |
| `retrievers` | **6 types**: contextual compression, docstore, ensemble, multi-vector, parent document, self-query |
| `document_loaders` | **8 types**: text, CSV, JSON, HTML, Markdown, PDF (optional), directory, web |
| `text_splitter` | **8 types**: character, recursive, markdown, HTML, JSON, code, token, token-aware |
| `embeddings` | **5 providers**: Anthropic, OpenAI, Google, Ollama + cached wrapper |
| `tools` | Calculator, shell command, JSON query, web search, Wikipedia, OpenAPI, cached wrapper |
| `agents` | `AgentExecutor` with tool calling and structured output. **18 middleware types**: retry, model retry, model fallback, tool retry, tool call limit, model call limit, human-in-the-loop, PII redaction, summarization, context editing, file search, shell tool, tool emulator, tool selection, todo, redaction, execution |
| `evaluation` | LLM output evaluation framework |
| `indexing` | Document indexing pipeline |
| `cache` | SQLite-backed LLM response cache |

### langgraph — State Graph Orchestration

| Module | Description |
|---|---|
| `graph::state` | `StateGraph` builder, `CompiledStateGraph`, conditional branching |
| `graph::persistent` | `PersistentGraph` with automatic checkpoint save/restore and fork |
| `graph::subgraph` | Subgraph composition for modular workflows |
| `graph::human_in_loop` | Human-in-the-loop interrupt and approval patterns |
| `graph::time_travel` | State history navigation and replay |
| `graph::send` | Send API for dynamic fan-out patterns |
| `graph::mermaid` | Export graphs as Mermaid diagrams |
| `graph::stream_events` | Streaming events from graph execution |
| `graph::annotations` | State annotations for reducer functions |
| `graph::message` | Message-based graph construction |
| `pregel` | Pregel-style superstep execution engine |
| `channels` | **8 types**: LastValue, BinaryOp, Topic, AnyValue, NamedBarrier, EphemeralValue, Untracked, reducers |
| `checkpoint` | `CheckpointSaver` trait with in-memory, SQLite, and PostgreSQL backends |
| `prebuilt` | `create_react_agent` and `create_tool_agent` factories |
| `types` | StreamMode (Values, Updates, Debug), InterruptType, RetryPolicy, CachePolicy |

### deepagents — High-Level Agent Factory (Beta)

| Module | Description |
|---|---|
| `agent` | `create_deep_agent()` factory returning a compiled graph with middleware |
| `middleware` | **6 types**: Filesystem (read, write, edit, ls, glob, grep), Memory, SubAgent, Summarization, Skills, PatchToolCalls |
| `backends` | **2 types**: `StateBackend` (in-memory), `FilesystemBackend` (local disk) |
| `config` | `DeepAgentConfig` for model, tools, middleware, and backend configuration |

---

## Quick Start

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) 1.70 or later
- An API key for your chosen LLM provider (optional -- all examples work with fake/mock models)

### Installation

Add RustChain to your `Cargo.toml`. Enable only the providers you need:

```toml
[dependencies]
rustchain = { git = "https://github.com/0xvasanth/rustchain", features = ["openai"] }
rustchain-core = { git = "https://github.com/0xvasanth/rustchain" }
tokio = { version = "1", features = ["full"] }
serde_json = "1"
```

### Composable Chain

Build a prompt -> model -> parser chain in a few lines:

```rust
use std::sync::Arc;
use serde_json::json;
use rustchain_core::chain;
use rustchain_core::language_models::{ChatModelRunnable, FakeListChatModel};
use rustchain_core::output_parsers::StrOutputParser;
use rustchain_core::prompts::ChatPromptTemplate;
use rustchain_core::runnables::Runnable;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = ChatPromptTemplate::from_messages(vec![
        ("system", "You are a helpful assistant."),
        ("human", "Explain {topic} in one sentence."),
    ])?;

    let model = FakeListChatModel::new(vec![
        "Rust is a systems programming language focused on safety and speed.".into(),
    ]);

    let chain = chain!(
        prompt,
        ChatModelRunnable::new(Arc::new(model)),
        StrOutputParser
    )?;

    let result = chain.invoke(json!({"topic": "Rust"}), None).await?;
    println!("{}", result.as_str().unwrap());
    Ok(())
}
```

> Replace `FakeListChatModel` with `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenAI`, or any other provider for real LLM calls.

### Stateful Graph Workflow

```rust
use std::sync::Arc;
use serde_json::{json, Value};
use langgraph::graph::state::{AsyncNodeAction, StateGraph};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
    println!("{}", result);
    Ok(())
}
```

### RAG Pipeline

```rust
use std::sync::Arc;
use rustchain::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};
use rustchain::document_loaders::text::TextLoader;
use rustchain_core::document_loaders::BaseLoader;
use rustchain_core::vectorstores::in_memory::InMemoryVectorStore;
use rustchain_core::vectorstores::base::VectorStore;

// 1. Load documents
let docs = TextLoader::new("data.txt").load().await?;

// 2. Split into chunks
let splitter = RecursiveCharacterTextSplitter::new()
    .with_chunk_size(500)
    .with_chunk_overlap(50);
let chunks = splitter.split_documents(&docs);

// 3. Store in vector database
let store = Arc::new(InMemoryVectorStore::new(embedding_model));
store.add_documents(chunks, None).await?;

// 4. Retrieve relevant context
let results = store.similarity_search("your question", 3).await?;
```

### Streaming

```rust
use futures::StreamExt;
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{HumanMessage, Message};

let messages = vec![Message::Human(HumanMessage::new("Tell me a story"))];
let mut stream = model._stream(&messages, None).await?;

while let Some(chunk) = stream.next().await {
    let chunk = chunk?;
    print!("{}", chunk.message.base.content.text());
}
```

---

## Feature Flags

| Crate | Flag | Description |
|---|---|---|
| `rustchain` | `openai` | OpenAI GPT chat models and embeddings |
| `rustchain` | `anthropic` | Anthropic Claude chat models and embeddings |
| `rustchain` | `google` | Google Gemini chat models and embeddings |
| `rustchain` | `ollama` | Ollama local inference models and embeddings |
| `rustchain` | `azure` | Azure OpenAI chat models |
| `rustchain` | `all-providers` | All five providers above |
| `rustchain` | `pdf` | PDF document loader (via `pdf-extract`) |
| `rustchain` | `sqlite` | SQLite-backed LLM response cache |
| `langgraph` | `sqlite` | SQLite checkpoint persistence |
| `langgraph` | `postgres` | PostgreSQL checkpoint persistence |

For graph orchestration, add `langgraph`:

```toml
[dependencies]
langgraph = { git = "https://github.com/0xvasanth/rustchain", features = ["sqlite"] }
```

---

## Examples

The `examples/` directory contains **15 runnable examples** -- all work without API keys using fake/mock models:

| Example | Description | Run Command |
|---|---|---|
| `simple_chain` | LCEL chain composition (prompt -> model -> parser) | `cargo run -p rustchain-examples --example simple_chain` |
| `tool_agent` | AgentExecutor with calculator tool calling | `cargo run -p rustchain-examples --example tool_agent` |
| `tool_calling_agent` | Advanced tool calling with multiple tools | `cargo run -p rustchain-examples --example tool_calling_agent` |
| `langgraph_agent` | ReAct agent with LangGraph state graph | `cargo run -p rustchain-examples --example langgraph_agent` |
| `rag_pipeline` | Full RAG: load -> split -> embed -> retrieve | `cargo run -p rustchain-examples --example rag_pipeline` |
| `rag_with_vectorstore` | Semantic similarity search | `cargo run -p rustchain-examples --example rag_with_vectorstore` |
| `indexing_rag` | Document indexing with record management | `cargo run -p rustchain-examples --example indexing_rag` |
| `conversational_agent` | Multi-turn conversation with memory | `cargo run -p rustchain-examples --example conversational_agent` |
| `graph_with_checkpoints` | Persistent graph with checkpoint save/resume/fork | `cargo run -p rustchain-examples --example graph_with_checkpoints` |
| `streaming` | Character-level and token-level streaming | `cargo run -p rustchain-examples --example streaming` |
| `streaming_chat` | Interactive streaming chat session | `cargo run -p rustchain-examples --example streaming_chat` |
| `semantic_router` | Dynamic routing based on semantic similarity | `cargo run -p rustchain-examples --example semantic_router` |
| `multi_agent_collaboration` | Multiple agents collaborating on tasks | `cargo run -p rustchain-examples --example multi_agent_collaboration` |
| `structured_extraction` | Structured data extraction from text | `cargo run -p rustchain-examples --example structured_extraction` |
| `evaluation_pipeline` | LLM output evaluation and scoring | `cargo run -p rustchain-examples --example evaluation_pipeline` |

Try one now:

```bash
git clone https://github.com/0xvasanth/rustchain.git
cd rustchain
cargo run -p rustchain-examples --example simple_chain
```

---

## Migration Status

This project migrates the Python LangChain ecosystem to Rust. Here is the mapping:

| Python Module | Rust Equivalent | Status |
|---|---|---|
| `langchain-core` (BaseLLM, BaseTool, Runnable, Message) | `rustchain-core` | Done |
| `langchain.chat_models` (OpenAI, Anthropic, Google, Ollama) | `rustchain::chat_models` | Done |
| `langchain.chains` (LLM, Sequential, RetrievalQA, etc.) | `rustchain::chains` | Done |
| `langchain.agents` (AgentExecutor, middleware) | `rustchain::agents` | Done |
| `langchain.memory` (Buffer, Window, Summary, Vector) | `rustchain::memory` | Done |
| `langchain.document_loaders` (Text, CSV, JSON, HTML, PDF) | `rustchain::document_loaders` | Done |
| `langchain.text_splitter` (Recursive, Markdown, Code) | `rustchain::text_splitter` | Done |
| `langchain.embeddings` (OpenAI, Anthropic, Google) | `rustchain::embeddings` | Done |
| `langchain.tools` (Calculator, Shell, Search) | `rustchain::tools` | Done |
| `langchain.vectorstores` (InMemory) | `rustchain_core::vectorstores` | Done |
| `langchain.retrievers` (MultiVector, SelfQuery, Ensemble) | `rustchain::retrievers` | Done |
| `langchain.evaluation` | `rustchain::evaluation` | Done |
| `langgraph.graph` (StateGraph, MessageGraph) | `langgraph::graph` | Done |
| `langgraph.pregel` (Execution engine) | `langgraph::pregel` | Done |
| `langgraph.channels` (LastValue, Topic, BinaryOp) | `langgraph::channels` | Done |
| `langgraph.checkpoint` (Memory, SQLite, Postgres) | `langgraph::checkpoint` | Done |
| `langgraph.prebuilt` (ReAct agent, Tool agent) | `langgraph::prebuilt` | Done |
| `deepagents.graph` (create_deep_agent) | `deepagents::agent` | Done |
| `deepagents.middleware` (Filesystem, Memory, SubAgent) | `deepagents::middleware` | Done |
| `deepagents.backends` (State, Filesystem) | `deepagents::backends` | Done |

---

## Core Traits

These are the foundational abstractions that power the framework:

```rust
/// Language model abstraction -- implement this to add a new LLM provider.
pub trait BaseChatModel: Send + Sync {
    async fn _generate(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatResult>;
    async fn stream(&self, messages: &[Message]) -> Result<BoxStream<ChatGenerationChunk>>;
    fn llm_type(&self) -> &str;
}

/// Composable computation unit (LCEL) -- the building block of chains.
pub trait Runnable: Send + Sync {
    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value>;
    async fn batch(&self, inputs: Vec<Value>, config: Option<&RunnableConfig>) -> Result<Vec<Value>>;
    async fn stream(&self, input: Value, config: Option<&RunnableConfig>) -> Result<RunnableStream>;
}

/// Tool abstraction for agents -- implement this to give agents new capabilities.
pub trait BaseTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput>;
}

/// Vector storage for RAG pipelines.
pub trait VectorStore: Send + Sync {
    async fn add_documents(&self, docs: Vec<Document>) -> Result<Vec<String>>;
    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<Document>>;
}

/// Embedding provider for converting text to vectors.
pub trait Embeddings: Send + Sync {
    async fn embed_documents(&self, texts: Vec<&str>) -> Result<Vec<Vec<f64>>>;
    async fn embed_query(&self, text: &str) -> Result<Vec<f64>>;
}
```

---

## Workspace Structure

```
rustchain/
  Cargo.toml                  # Workspace root
  examples/                   # 15 runnable example programs
  crates/
    rustchain-core/           # Base traits and types (zero workspace deps)
    rustchain/                # Provider implementations and agent framework
    langgraph/                # State graph orchestration engine
    deepagents/               # High-level agent factory
    examples/                 # Example runner crate
```

---

## Tech Stack

| Purpose | Library |
|---|---|
| Serialization | `serde`, `serde_json` |
| Async runtime | `tokio` |
| HTTP client | `reqwest` |
| Streaming | `futures`, `tokio-stream` |
| Database | `sqlx` (SQLite, PostgreSQL) |
| Error handling | `thiserror` |
| Secrets | `secrecy` |
| Regex | `regex` |
| CSV parsing | `csv` |
| File globbing | `glob` |
| PDF extraction | `pdf-extract` (optional) |

---

## Contributing

Contributions are welcome! Whether it's a bug fix, a new LLM provider, better documentation, or an entirely new feature.

### Development Setup

```bash
git clone https://github.com/0xvasanth/rustchain.git
cd rustchain

# Build all crates
cargo build --workspace

# Run all tests (~2,738 tests)
cargo test --workspace

# Run a specific example
cargo run -p rustchain-examples --example simple_chain

# Build with all LLM providers enabled
cargo build -p rustchain --features all-providers

# Check for warnings and clippy lints
cargo clippy --workspace
```

### How to Contribute

1. **Fork** the repository and clone it locally
2. **Create a branch** for your feature or fix: `git checkout -b feat/my-feature`
3. **Make your changes** following existing code style and conventions
4. **Add tests** for new functionality
5. **Run the test suite**: `cargo test --workspace`
6. **Submit a Pull Request** with a clear description of what you changed and why

### Project Conventions

| Rule | Details |
|---|---|
| Dependency boundaries | `rustchain-core` has zero workspace dependencies. `langgraph` depends only on `rustchain-core`. |
| Feature flags | LLM providers must be gated behind feature flags |
| Async runtime | All async code uses `tokio` |
| Error handling | Per-crate error types using `thiserror` |
| Documentation | All public APIs should have `///` doc comments |
| Testing | New features require tests |

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">

**Built with Rust. Inspired by LangChain.**

[Report a Bug](https://github.com/0xvasanth/rustchain/issues) · [Request a Feature](https://github.com/0xvasanth/rustchain/issues) · [Start a Discussion](https://github.com/0xvasanth/rustchain/discussions)

</div>
