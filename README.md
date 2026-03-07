<div align="center">

# RustChain

**A comprehensive Rust implementation of the LangChain ecosystem -- fast, type-safe, and composable.**

[![Build Status](https://img.shields.io/github/actions/workflow/status/0xvasanth/rustchain/ci.yml?branch=main&label=build)](https://github.com/0xvasanth/rustchain/actions)
[![Crate Version](https://img.shields.io/badge/crates.io-v0.1.0-blue)](https://crates.io/crates/rustchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)
[![Tests](https://img.shields.io/badge/tests-4%2C579%2B-green)](.)
[![Lines of Code](https://img.shields.io/badge/lines-178K-informational)](.)

[Getting Started](#getting-started) · [Examples](#examples) · [Architecture](#architecture) · [Feature Flags](#feature-flags) · [Contributing](#contributing)

</div>

---

## Overview

RustChain is a Rust-native LLM application framework that ports the Python [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph), and [DeepAgents](https://github.com/langchain-ai/deepagents) ecosystem to Rust. It provides composable abstractions for building chains, agents, RAG pipelines, and stateful graph workflows -- all with Rust's performance, type safety, and fearless concurrency.

The project spans **4 crates**, **430 source files**, **~178K lines of Rust**, and **~4,579 tests**.

### Why Rust?

- **Type-safe by design** -- Catch tool schema mismatches, invalid state transitions, and message type errors at compile time.
- **Zero-cost abstractions** -- Traits and generics instead of runtime reflection. No garbage collector overhead.
- **Async-first** -- Built on `tokio` with native streaming support via `futures::Stream`.
- **Modular** -- LLM providers are behind feature flags so you don't pay for what you don't use.
- **Composable** -- Chain prompts, models, parsers, and tools using the LCEL-inspired `chain!` macro and `Runnable` trait.
- **Production patterns** -- Circuit breakers, rate limiting, retry with backoff, PII redaction, and human-in-the-loop approval built into the agent middleware pipeline.

---

## Architecture

RustChain is organized as a Cargo workspace with four crates, each with clear responsibility and strict dependency boundaries:

```
+-----------------------------------------------------------+
|               deepagents  (Application Layer)              |
|  create_deep_agent(), middleware hooks, storage backends,  |
|  tool registry, planning, presets, events, conversations,  |
|  health monitoring, workflow engine                        |
+--------------------------+--------------------------------+
                           | depends on
+--------------------------v--------------------------------+
|               langgraph  (Orchestration Layer)             |
|  StateGraph, Pregel engine, checkpoints, streaming,        |
|  ReAct agent, human-in-the-loop, subgraph composition,     |
|  runner, timeout/cancellation, state reducers, event bus,  |
|  execution hooks, breakpoints, graph validator             |
+--------------------------+--------------------------------+
                           | depends on
+--------------------------v--------------------------------+
|               rustchain  (Implementation Layer)            |
|  Chat models (5 providers + factory + load balancer),       |
|  chains (19 types), memory (9 types), retrievers (11),     |
|  tools (14), agents (plan-and-execute, ReAct, tool-calling),|
|  document transformers, vectorstore filters                |
+--------------------------+--------------------------------+
                           | depends on
+--------------------------v--------------------------------+
|            rustchain-core  (Foundation Layer)               |
|  Base traits: BaseChatModel, BaseTool, Runnable, Message   |
|  Prompts, output parsers, callbacks, vector stores,        |
|  runnables (timeout/deadline, rate limiter, 30+ combinators)|
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

### rustchain-core -- Foundation Traits and Types

| Module | Description |
|---|---|
| `messages` | `Message` enum (Human, AI, System, Tool, Function) with streaming chunks, merge, and trim utilities |
| `language_models` | `BaseChatModel` and `BaseLLM` traits, plus fake/testing models |
| `runnables` | `Runnable` trait with sequence, parallel, branch, lambda, retry, fallback, scoped callbacks, **rate limiter/throttle wrappers**, **timeout/deadline** (configurable timeout behavior with fallback and default-value strategies), and 30+ combinators |
| `tools` | `BaseTool` trait, toolkit interface, structured tools, schema generation, and retriever adapter |
| `prompts` | Chat prompt templates, few-shot selectors, structured prompts, image prompts |
| `output_parsers` | JSON, string, list, XML, and tool-call parsers |
| `callbacks` | Extensible callback system with run managers and tracers |
| `vectorstores` | `VectorStore` trait and `InMemoryVectorStore` |
| `embeddings` | `Embeddings` trait for vector embedding providers |
| `documents` | `Document` type used across loaders, splitters, and retrievers |
| `retrievers` | `BaseRetriever` trait for document retrieval |
| `indexing` | Document indexing with record managers |
| `tracers` | stdout, OpenTelemetry, LangSmith-compatible, event/log stream tracers |
| `caches` | Caching infrastructure for LLM responses |
| `chat_history` | Chat history management interfaces |
| `stores` | Key-value store abstractions |

### rustchain -- Implementations and Provider Integrations

| Module | Description |
|---|---|
| `chat_models` | **5 providers**: Anthropic Claude, OpenAI GPT, Google Gemini, Ollama, Azure OpenAI. **Factory**: `init_chat_model` for dynamic provider creation via `ChatModelFactory` and global `ModelRegistry`. **Wrappers**: cached, circuit breaker, rate limited, retrying, structured, token counting, graceful, interceptor, **load balancer** (health-tracked round-robin across multiple models) |
| `chains` | **19 types**: LLM, sequential, conversation, conversation retrieval, retrieval QA, map-reduce, refine, router, structured output, summarize, SQL, API, streaming, extraction, **transform** (sync/async transforms with pipeline composition), **conditional/branch/switch** (routing chains with pluggable conditions), **stuff documents** (document combination with formatting strategies), **QA with citations** (retrieval QA with source citation tracking) |
| `memory` | **9 types**: buffer, window, summary, vector, chat history, hybrid, entity (named entity tracking with regex extraction), token buffer (token-count-aware trimming with pluggable `TokenCounter`), **knowledge graph** (subject-predicate-object triple extraction and storage for relational knowledge in prompts) |
| `vectorstores` | **6 backends**: InMemory, Qdrant, Pinecone, Weaviate, ChromaDB, FAISS (with Flat/IVF/HNSW indexes) |
| `retrievers` | **11 types**: contextual compression, compressor pipeline, docstore, ensemble, multi-vector, parent document, self-query, caching (TTL + LRU eviction), time-weighted (recency + relevance scoring), query translator (rule-based and LLM-powered structured filter generation), **multi-query** (query variation generation with reciprocal rank fusion) |
| `document_loaders` | **10 types**: text, CSV, JSON, HTML, Markdown, PDF, web/crawler, YAML, TOML, directory |
| `text_splitter` | **10 types**: character, recursive character (with language presets), Markdown, HTML, JSON, code, token, token-aware, sentence-aware (with abbreviation handling) |
| `embeddings` | **8 providers/utilities**: OpenAI, Anthropic, Google, Ollama, cached wrapper, distance metrics, router, **batch processor** (concurrent embedding with rate limiting and chunked processing) |
| `tools` | **14 types**: calculator, shell, web search, Wikipedia, JSON query, OpenAPI, cached, validation/auto-correction, Python REPL, retriever adapter, **human input** (interactive prompts with approval wrapper), **HTTP requests** (GET/POST with domain allowlisting and mock client), **file management** (read, write, copy, move, list, delete with path safety and toolkit grouping) |
| `agents` | `AgentExecutor` with tool calling and structured output. **Plan-and-execute agent** (plan creation, step-by-step execution, replanning on failure). **Output parsers**: ReAct (`Thought/Action/Action Input`), JSON, XML, and tool-call format parsers. **18 middleware types**: retry, model retry, model fallback, tool retry, tool call limit, model call limit, human-in-the-loop, PII redaction, summarization, context editing, file search, shell tool, tool emulator, tool selection, todo, redaction, execution |
| `chat_sessions` | `SessionManager` for managing multiple concurrent chat sessions with lifecycle states (Active, Archived, Deleted), pluggable persistence (`InMemorySessionStorage`, `FileSessionStorage`), and builder configuration |
| `stores` | **4 KV store implementations**: `InMemoryStore` (with optional TTL), `FileStore` (filesystem-backed), `NamespacedStore` (key prefix isolation), `LayeredStore` (read-through cache across multiple stores) |
| `prompts` | Prompt registry with versioning, built-in templates, **chat prompt templates** (message-level templates with variable extraction and placeholder support), and **few-shot selectors** (length-based and semantic similarity) |
| `evaluation` | LLM output evaluation framework |
| `indexing` | Document indexing pipeline |
| `cache` | SQLite-backed LLM response cache |
| `document_transformers` | **3 types**: metadata enrichment (word count, char count, language, hash), deduplication (embedding-based redundancy filter), enrichment pipeline (chain multiple transformers in sequence). Plus `LLMDocumentTransformer` for model-driven transforms |
| `vectorstores::filters` | Unified filter system with composable expressions (`Eq`, `Ne`, `Gt`, `Lt`, `In`, `Contains`) combinable via `And`/`Or`/`Not` for cross-backend metadata filtering |

### langgraph -- State Graph Orchestration

| Module | Description |
|---|---|
| `graph::state` | `StateGraph` builder, `CompiledStateGraph`, conditional branching |
| `graph::persistent` | `PersistentGraph` with automatic checkpoint save/restore and fork |
| `graph::subgraph` | Subgraph composition for modular workflows |
| `graph::human_in_loop` | Human-in-the-loop interrupt and approval patterns |
| `graph::time_travel` | State history navigation and replay |
| `graph::send` | Send API for dynamic fan-out patterns |
| `graph::runner` | `GraphRunner` with configurable step limits, timeouts, lifecycle hooks (`StepHook` trait), and event collection for step-level observability |
| `graph::hooks` | **Execution hooks** -- flexible lifecycle system for intercepting graph execution at before/after node, edge, and graph phases. Built-in hooks: `LoggingHook`, `StateValidationHook`, `TimingHook`, `StateSnapshotHook` |
| `graph::breakpoint` | **Breakpoint manager** -- conditional and per-node breakpoints for pausing execution with handler dispatch and history tracking |
| `graph::validator` | **Graph validator** -- comprehensive structural validation (connectivity, termination, cycles, edge integrity, node names) with severity levels and error codes |
| `graph::serialize` | **Graph serialization** -- `GraphDefinition` for persisting graph topology to JSON, plus `GraphRegistry` for named graph storage |
| `graph::stream_writer` | `StreamWriter`/`StreamReader` for producing and consuming `StreamChunk` values through bounded async channels, with `StreamCollector` for aggregation and `FilteredStream` for selective consumption |
| `graph::ascii` | ASCII art graph visualization for terminal display |
| `graph::mermaid` | Export graphs as Mermaid diagrams |
| `graph::snapshot` | Graph state snapshot and restore with pluggable storage |
| `graph::audit` | Execution audit log with trail tracing |
| `graph::stream_events` | Streaming events from graph execution |
| `graph::annotations` | State annotations for reducer functions |
| `graph::message` | Message-based graph construction |
| `graph::ui` | UI message types and reducer for rendering UI components |
| `managed` | Shared managed values with versioning and history: `SharedValue`, `SharedCounter`, `SharedAccumulator`, `SharedMap`, plus `IsLastStepManager` and `RemainingStepsManager` for Pregel context |
| `pregel` | Pregel-style superstep execution engine |
| `channels` | **10 types**: LastValue, BinaryOp, Topic, AnyValue, NamedBarrier, EphemeralValue, Untracked, Broadcast (pub/sub with topic filtering), reducers, **state reducers** (schema-validated reducers with append, merge, last-value, and binary-op strategies) |
| `checkpoint` | `CheckpointSaver` trait with **in-memory** (including lightweight in-memory store for testing), SQLite, and PostgreSQL backends. Serialization formats with diff support |
| `prebuilt` | `create_react_agent`, `create_tool_agent`, and `ChatAgent` (prebuilt tool-calling agent with streaming) |
| `utils` | Configuration utilities, execution profiler with bottleneck detection, and **node timeout/cancellation** (per-node timeouts, cancellation tokens, and execution budget management) |
| `types` | StreamMode (Values, Updates, Debug), InterruptType, RetryPolicy, CachePolicy |

### deepagents -- High-Level Agent Factory (Beta)

| Module | Description |
|---|---|
| `agent` | `create_deep_agent()` factory returning a compiled graph with middleware |
| `middleware` | **11 types**: Filesystem (read, write, edit, ls, glob, grep), Memory, SubAgent, Summarization, Skills, PatchToolCalls, Rate Limiter (token bucket with cost tracking), Logging (structured with redaction), Context Manager (dynamic context injection), Planning (plan-then-execute with step tracking, dependencies, and status injection) |
| `backends` | **3 types**: `StateBackend` (in-memory), `FilesystemBackend` (local disk), `SandboxBackend` (isolated execution with resource limits and permissions) |
| `tool_registry` | `ToolRegistry` for centralized tool management with permission levels, call counting, enable/disable control, and filtering |
| `config` | `DeepAgentConfig` with **builder pattern**, configuration **loader** (from file/env), and expanded **middleware config** for fine-grained control |
| `events` | **Event bus system** with typed event handlers, lifecycle tracking, and pub/sub dispatch for agent execution events |
| `conversation` | **Conversation manager** with context windowing, message history management, and export/serialization support |
| `presets` | **Agent presets** with a preset registry, customizer pattern, and ready-made configurations for common agent types |
| `health` | **Health monitoring** -- `HealthMonitor` aggregating multiple `HealthCheck` implementations (disk space, connectivity, middleware status) into unified `HealthReport` with builder configuration |
| `workflow` | **Workflow engine** -- multi-step workflow orchestration with dependency resolution, conditional execution, retries, timeouts, and `WorkflowBuilder`/`WorkflowExecutor` API |

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
| `rustchain` | `all-providers` | All providers plus `yaml` and `toml-loader` |
| `rustchain` | `pdf` | PDF document loader (via `pdf-extract`) |
| `rustchain` | `yaml` | YAML document loader (via `serde_yaml`) |
| `rustchain` | `toml-loader` | TOML document loader (via `toml`) |
| `rustchain` | `sqlite` | SQLite-backed LLM response cache |
| `rustchain` | `qdrant` | Qdrant vector store backend |
| `rustchain` | `pinecone` | Pinecone vector store backend |
| `rustchain` | `weaviate` | Weaviate vector store backend |
| `rustchain` | `chroma` | ChromaDB vector store backend |
| `rustchain` | `faiss` | FAISS-compatible vector store (pure-Rust Flat/IVF/HNSW indexes) |
| `langgraph` | `sqlite` | SQLite checkpoint persistence |
| `langgraph` | `postgres` | PostgreSQL checkpoint persistence |

For graph orchestration, add `langgraph`:

```toml
[dependencies]
langgraph = { git = "https://github.com/0xvasanth/rustchain", features = ["sqlite"] }
```

---

## Examples

The `examples/` directory contains **27 runnable examples** -- all work without API keys using fake/mock models:

| Example | Description | Run Command |
|---|---|---|
| `simple_chain` | LCEL chain composition (prompt -> model -> parser) | `cargo run --example simple_chain` |
| `tool_agent` | AgentExecutor with calculator tool calling | `cargo run --example tool_agent` |
| `tool_calling_agent` | Advanced tool calling with multiple tools | `cargo run --example tool_calling_agent` |
| `react_agent` | ReAct agent with reasoning traces | `cargo run --example react_agent` |
| `langgraph_agent` | ReAct agent with LangGraph state graph | `cargo run --example langgraph_agent` |
| `rag_pipeline` | Full RAG: load -> split -> embed -> retrieve | `cargo run --example rag_pipeline` |
| `rag_with_vectorstore` | Semantic similarity search | `cargo run --example rag_with_vectorstore` |
| `indexing_rag` | Document indexing with record management | `cargo run --example indexing_rag` |
| `conversational_agent` | Multi-turn conversation with memory | `cargo run --example conversational_agent` |
| `graph_with_checkpoints` | Persistent graph with checkpoint save/resume/fork | `cargo run --example graph_with_checkpoints` |
| `streaming` | Character-level and token-level streaming | `cargo run --example streaming` |
| `streaming_chat` | Interactive streaming chat session | `cargo run --example streaming_chat` |
| `semantic_router` | Dynamic routing based on semantic similarity | `cargo run --example semantic_router` |
| `multi_agent_collaboration` | Multiple agents collaborating on tasks | `cargo run --example multi_agent_collaboration` |
| `structured_extraction` | Structured data extraction from text | `cargo run --example structured_extraction` |
| `evaluation_pipeline` | LLM output evaluation and scoring | `cargo run --example evaluation_pipeline` |
| `vector_store_search` | Vector store similarity search operations | `cargo run --example vector_store_search` |
| `text_splitting` | Text splitting strategies and language presets | `cargo run --example text_splitting` |
| `extraction_chain` | Extraction chain for pulling structured data | `cargo run --example extraction_chain` |
| `graph_visualization` | ASCII and Mermaid graph visualization | `cargo run --example graph_visualization` |
| `memory_types` | Buffer, window, summary, entity, knowledge graph, and token buffer memory | `cargo run --example memory_types` |
| `agent_output_parsing` | ReAct, JSON, XML, and tool-call output parsing | `cargo run --example agent_output_parsing` |
| `caching_retriever` | Caching retriever with TTL and LRU eviction | `cargo run --example caching_retriever` |
| `planning_middleware` | Plan-then-execute with planning middleware | `cargo run --example planning_middleware` |
| `conditional_chains` | Conditional, branch, and switch chain routing | `cargo run --example conditional_chains` |
| `file_tools` | File management tools with path safety | `cargo run --example file_tools` |
| `qa_chain` | QA chain with citation tracking | `cargo run --example qa_chain` |

Try one now:

```bash
git clone https://github.com/0xvasanth/rustchain.git
cd rustchain
cargo run --example simple_chain
```

---

## Migration Status

This project migrates the Python LangChain ecosystem to Rust. Here is the mapping:

| Python Module | Rust Equivalent | Status |
|---|---|---|
| `langchain-core` (BaseLLM, BaseTool, Runnable, Message) | `rustchain-core` | Done |
| `langchain.chat_models` (OpenAI, Anthropic, Google, Ollama) | `rustchain::chat_models` | Done |
| `langchain.chat_models` (init_chat_model factory) | `rustchain::chat_models::factory` | Done |
| `langchain.chains` (LLM, Sequential, RetrievalQA, etc.) | `rustchain::chains` | Done |
| `langchain.agents` (AgentExecutor, middleware) | `rustchain::agents` | Done |
| `langchain.agents` (Plan-and-Execute) | `rustchain::agents::plan_and_execute` | Done |
| `langchain.agents` (output parsers: ReAct, JSON, XML, ToolCall) | `rustchain::agents::output_parser` | Done |
| `langchain.memory` (Buffer, Window, Summary, Vector, Entity, TokenBuffer) | `rustchain::memory` | Done |
| `langchain.memory` (Knowledge Graph) | `rustchain::memory::knowledge_graph` | Done |
| `langchain.document_loaders` (Text, CSV, JSON, HTML, PDF, YAML, TOML) | `rustchain::document_loaders` | Done |
| `langchain.text_splitter` (Recursive, Markdown, Code, Sentence) | `rustchain::text_splitter` | Done |
| `langchain.embeddings` (OpenAI, Anthropic, Google, Router) | `rustchain::embeddings` | Done |
| `langchain.tools` (Calculator, Shell, Search, Python REPL) | `rustchain::tools` | Done |
| `langchain.vectorstores` (InMemory, Qdrant, Pinecone, Weaviate, Chroma, FAISS) | `rustchain::vectorstores` | Done |
| `langchain.retrievers` (MultiVector, SelfQuery, Ensemble, Caching, TimeWeighted, QueryTranslator) | `rustchain::retrievers` | Done |
| `langchain.evaluation` | `rustchain::evaluation` | Done |
| `langchain.stores` (KV stores) | `rustchain::stores` | Done |
| `langchain.chat_sessions` (session management) | `rustchain::chat_sessions` | Done |
| `langchain_core.runnables` (rate limiter, throttle) | `rustchain_core::runnables` | Done |
| `langchain_core.runnables` (timeout, deadline) | `rustchain_core::runnables::timeout` | Done |
| `langgraph.graph` (StateGraph, MessageGraph) | `langgraph::graph` | Done |
| `langgraph.graph` (runner, stream writer/reader) | `langgraph::graph::runner`, `langgraph::graph::stream_writer` | Done |
| `langgraph.graph` (execution hooks) | `langgraph::graph::hooks` | Done |
| `langgraph.graph` (breakpoints) | `langgraph::graph::breakpoint` | Done |
| `langgraph.graph` (validator) | `langgraph::graph::validator` | Done |
| `langgraph.graph` (serialization, registry) | `langgraph::graph::serialize` | Done |
| `langgraph.graph` (UI messages) | `langgraph::graph::ui` | Done |
| `langgraph.pregel` (Execution engine) | `langgraph::pregel` | Done |
| `langgraph.channels` (LastValue, Topic, BinaryOp, Broadcast) | `langgraph::channels` | Done |
| `langgraph.checkpoint` (Memory, SQLite, Postgres, Serialization) | `langgraph::checkpoint` | Done |
| `langgraph.prebuilt` (ReAct agent, Tool agent, ChatAgent) | `langgraph::prebuilt` | Done |
| `langgraph.managed` (shared values, step managers) | `langgraph::managed` | Done |
| `deepagents.graph` (create_deep_agent) | `deepagents::agent` | Done |
| `deepagents.middleware` (Filesystem, Memory, SubAgent, RateLimiter, Logging, Context, Planning) | `deepagents::middleware` | Done |
| `deepagents.backends` (State, Filesystem, Sandbox) | `deepagents::backends` | Done |
| `deepagents.tool_registry` (tool management with permissions) | `deepagents::tool_registry` | Done |
| `deepagents` (health monitoring) | `deepagents::health` | Done |
| `deepagents` (workflow engine) | `deepagents::workflow` | Done |
| `langchain.chains` (StuffDocuments, QA with citations) | `rustchain::chains::documents`, `rustchain::chains::qa` | Done |
| `langchain.chains` (Conditional, Branch, Switch) | `rustchain::chains::conditional` | Done |
| `langchain.chains` (Transform, TransformPipeline) | `rustchain::chains::transform` | Done |
| `langchain.tools` (HumanInput, HTTP requests, file management) | `rustchain::tools::human`, `rustchain::tools::requests`, `rustchain::tools::file_management` | Done |
| `langchain.chat_models` (load-balanced model) | `rustchain::chat_models::load_balancer` | Done |
| `langchain.embeddings` (batch processor) | `rustchain::embeddings::batch` | Done |
| `langchain.document_transformers` (metadata, dedup, enrichment) | `rustchain::document_transformers` | Done |
| `langchain.vectorstores` (filter system) | `rustchain::vectorstores::filters` | Done |
| `langchain.retrievers` (multi-query with RRF) | `rustchain::retrievers::multi_query` | Done |
| `langgraph.channels` (state reducers with schema) | `langgraph::channels::state_reducers` | Done |
| `langgraph.checkpoint` (in-memory store) | `langgraph::checkpoint` | Done |
| `langgraph.utils` (node timeout/cancellation) | `langgraph::utils::timeout` | Done |
| `deepagents` (event bus system) | `deepagents::events` | Done |
| `deepagents` (conversation manager) | `deepagents::conversation` | Done |
| `deepagents` (agent presets) | `deepagents::presets` | Done |
| `deepagents.config` (expanded builder, loader, middleware config) | `deepagents::config` | Done |

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
  examples/                   # 27 runnable example programs
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
| YAML | `serde_yaml` (optional) |
| TOML | `toml` (optional) |

---

## Contributing

Contributions are welcome! Whether it's a bug fix, a new LLM provider, better documentation, or an entirely new feature.

### Development Setup

```bash
git clone https://github.com/0xvasanth/rustchain.git
cd rustchain

# Build all crates
cargo build --workspace

# Run all tests (~4,579 tests)
cargo test --workspace

# Run a specific example
cargo run --example simple_chain

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
