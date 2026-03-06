<div align="center">

# RustChain

**Build powerful LLM applications in Rust — fast, safe, and composable.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

RustChain is a Rust-native LLM application framework inspired by [LangChain](https://github.com/langchain-ai/langchain). It provides composable abstractions for building chains, agents, RAG pipelines, and stateful graph workflows — all with Rust's performance, type safety, and fearless concurrency.

[Getting Started](#getting-started) · [Examples](#examples) · [Architecture](#architecture) · [Contributing](#contributing)

</div>

---

## Why RustChain?

- **Type-safe by design** — Catch tool schema mismatches, invalid state transitions, and message type errors at compile time, not in production.
- **Zero-cost abstractions** — Traits and generics instead of runtime reflection. No garbage collector overhead.
- **Async-first** — Built on `tokio` with native streaming support via `futures::Stream`.
- **Modular** — Pick only what you need. LLM providers are behind feature flags so you don't pay for what you don't use.
- **Composable** — Chain prompts, models, parsers, and tools together using the LCEL-inspired `chain!` macro and `Runnable` trait.
- **Production patterns** — Circuit breakers, rate limiting, retry with backoff, PII redaction, and human-in-the-loop approval are built into the agent middleware pipeline.

## Features at a Glance

| Category | What's included |
|---|---|
| **LLM Providers** | OpenAI, Anthropic Claude, Google Gemini, Azure OpenAI, Ollama (local) |
| **Chains** | LLM, sequential, conversation, retrieval QA, map-reduce, refine, router, structured output, summarization |
| **Agents** | ReAct pattern, tool calling, 20+ middleware types (retry, fallback, circuit breaker, PII redaction, human approval, and more) |
| **Graph Orchestration** | StateGraph, Pregel execution engine, conditional routing, subgraphs, streaming, human-in-the-loop interrupts |
| **RAG** | 8 document loaders, 8 text splitters, in-memory vector store, 4 embedding providers, 5 retriever strategies |
| **Memory** | Buffer, sliding window, summary, vector-backed conversation memory |
| **Checkpointing** | In-memory, SQLite, PostgreSQL backends for durable graph execution |
| **Tools** | Calculator, shell, JSON query, web search, Wikipedia — plus an easy `BaseTool` trait for custom tools |
| **Streaming** | Character-level, token-level, and chunk-level streaming from any model or chain |

---

## Getting Started

### Prerequisites

- [Rust](https://www.rust-lang.org/tools/install) 1.70 or later
- An API key for your chosen LLM provider (optional — all examples work with fake/mock models)

### Installation

Add RustChain to your `Cargo.toml`. Enable only the providers you need:

```toml
[dependencies]
rustchain = { git = "https://github.com/0xvasanth/rustchain", features = ["openai"] }
rustchain-core = { git = "https://github.com/0xvasanth/rustchain" }
tokio = { version = "1", features = ["full"] }
serde_json = "1"
```

**Available feature flags:**

| Crate | Flag | Description |
|---|---|---|
| `rustchain` | `openai` | OpenAI GPT models |
| `rustchain` | `anthropic` | Anthropic Claude models |
| `rustchain` | `google` | Google Gemini models |
| `rustchain` | `ollama` | Ollama local inference |
| `rustchain` | `azure` | Azure OpenAI |
| `rustchain` | `all-providers` | All of the above |
| `rustchain` | `pdf` | PDF document loader |
| `rustchain` | `sqlite` | SQLite cache backend |
| `langgraph` | `sqlite` | SQLite checkpoint persistence |
| `langgraph` | `postgres` | PostgreSQL checkpoint persistence |

For graph orchestration, add `langgraph`:

```toml
[dependencies]
langgraph = { git = "https://github.com/0xvasanth/rustchain", features = ["sqlite"] }
```

### Quick Start — Composable Chain

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

### Quick Start — Tool-Calling Agent

```rust
use std::sync::Arc;
use rustchain::agents::AgentExecutor;
use rustchain::tools::calculator::CalculatorTool;
use rustchain_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = /* your ChatModel here */;
    let calculator = Arc::new(CalculatorTool);

    let executor = AgentExecutor::builder()
        .model(model)
        .tool(calculator)
        .max_iterations(5)
        .build();

    let result = executor.run(&[Message::human("What is (2 + 3) * 4?")]).await?;
    println!("{}", result.output);
    Ok(())
}
```

The `AgentExecutor` runs a ReAct loop: the model reasons, decides to call tools, receives results, and continues until it produces a final answer or hits `max_iterations`.

### Quick Start — Stateful Graph Workflow

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

Graphs support conditional routing, checkpointing (persist and resume), subgraph composition, streaming, and human-in-the-loop interrupts.

### Quick Start — RAG Pipeline

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

### Quick Start — Streaming

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

## Examples

The `examples/` directory contains **9 runnable examples** — all work without API keys using fake/mock models:

| Example | What it demonstrates | Run command |
|---|---|---|
| `simple_chain` | LCEL chain composition (prompt -> model -> parser) | `cargo run -p rustchain-examples --example simple_chain` |
| `tool_agent` | AgentExecutor with calculator tool calling | `cargo run -p rustchain-examples --example tool_agent` |
| `langgraph_agent` | ReAct agent with LangGraph state graph | `cargo run -p rustchain-examples --example langgraph_agent` |
| `rag_pipeline` | Full RAG: load -> split -> embed -> retrieve | `cargo run -p rustchain-examples --example rag_pipeline` |
| `rag_with_vectorstore` | Semantic similarity search | `cargo run -p rustchain-examples --example rag_with_vectorstore` |
| `conversational_agent` | Multi-turn conversation with memory | `cargo run -p rustchain-examples --example conversational_agent` |
| `graph_with_checkpoints` | Persistent graph with checkpoint save/resume/fork | `cargo run -p rustchain-examples --example graph_with_checkpoints` |
| `streaming` | Character-level and token-level streaming | `cargo run -p rustchain-examples --example streaming` |
| `semantic_router` | Dynamic routing based on semantic similarity | `cargo run -p rustchain-examples --example semantic_router` |

Try one now:

```bash
git clone https://github.com/0xvasanth/rustchain.git
cd rustchain
cargo run -p rustchain-examples --example simple_chain
```

---

## Architecture

RustChain is organized as a Cargo workspace with four crates, each with a clear responsibility and strict dependency boundaries:

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
|  Chat models (Anthropic, OpenAI, Google, Ollama, Azure),   |
|  embeddings, agents, chains, memory, document loaders,     |
|  text splitters, tools                                     |
+--------------------------+--------------------------------+
                           | depends on
+--------------------------v--------------------------------+
|            rustchain-core  (Foundation Layer)               |
|  Base traits: BaseChatModel, BaseTool, Runnable, Message   |
|  Prompts, output parsers, callbacks, vector stores         |
+-----------------------------------------------------------+
```

### Crate Breakdown

<details>
<summary><strong>rustchain-core</strong> — Foundation traits and types (zero workspace dependencies)</summary>

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

</details>

<details>
<summary><strong>rustchain</strong> — Concrete implementations and provider integrations</summary>

| Module | Description |
|---|---|
| `chat_models` | Anthropic Claude, OpenAI GPT, Google Gemini, Ollama, Azure OpenAI + wrappers (cached, circuit breaker, rate limited, retrying, structured) |
| `embeddings` | Anthropic, Google, Ollama, OpenAI embedding providers + cached wrapper |
| `agents` | `AgentExecutor` with 20+ middleware types (retry, fallback, PII redaction, human approval, tool limits, summarization, and more) |
| `chains` | LLMChain, ConversationChain, SequentialChain, RetrievalQAChain, MapReduceChain, RefineChain, RouterChain, StructuredOutputChain |
| `memory` | ConversationBufferMemory, WindowMemory, SummaryMemory, VectorStoreMemory |
| `document_loaders` | Text, CSV, JSON, HTML, Markdown, PDF (optional), directory, web |
| `text_splitter` | Character, recursive, markdown, HTML, JSON, code, token, token-aware |
| `tools` | Calculator, shell command, JSON query, web search, Wikipedia |
| `vectorstores` | In-memory vector store |

</details>

<details>
<summary><strong>langgraph</strong> — Stateful graph orchestration engine</summary>

| Module | Description |
|---|---|
| `graph` | `StateGraph` builder, `CompiledStateGraph`, conditional branching, subgraphs, Mermaid diagram export |
| `graph::persistent` | `PersistentGraph` with automatic checkpoint save/restore and fork |
| `pregel` | Pregel-style execution engine with superstep processing |
| `channels` | LastValue, BinaryOp, Topic, AnyValue, NamedBarrier, EphemeralValue, Untracked |
| `checkpoint` | `CheckpointSaver` trait, `InMemoryCheckpointSaver`, SQLite and PostgreSQL backends |
| `prebuilt` | `create_react_agent` and `create_tool_agent` factories |
| `types` | StreamMode (Values, Updates, Debug), InterruptType, RetryPolicy, CachePolicy |
| `runtime` | Tokio-based async runtime utilities |

</details>

<details>
<summary><strong>deepagents</strong> — High-level agent factory (beta)</summary>

| Module | Description |
|---|---|
| `agent` | `create_deep_agent()` factory returning a compiled graph with middleware |
| `middleware` | `Middleware` trait with before/after hooks — Filesystem, Memory, SubAgent, Summarization, Skills, PatchToolCalls |
| `backends` | `Backend` trait for session state — `StateBackend` (in-memory), `FilesystemBackend` (local disk) |
| `config` | `DeepAgentConfig` for model, tools, middleware, and backend configuration |

</details>

### Core Traits

These are the foundational abstractions that power the framework:

```rust
/// Language model abstraction — implement this to add a new LLM provider.
pub trait BaseChatModel: Send + Sync {
    async fn _generate(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatResult>;
    async fn stream(&self, messages: &[Message]) -> Result<BoxStream<ChatGenerationChunk>>;
    fn llm_type(&self) -> &str;
}

/// Composable computation unit (LCEL) — the building block of chains.
pub trait Runnable: Send + Sync {
    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value>;
    async fn batch(&self, inputs: Vec<Value>, config: Option<&RunnableConfig>) -> Result<Vec<Value>>;
    async fn stream(&self, input: Value, config: Option<&RunnableConfig>) -> Result<RunnableStream>;
}

/// Tool abstraction for agents — implement this to give agents new capabilities.
pub trait BaseTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn args_schema(&self) -> Option<Value>;
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput>;
}

/// Vector storage for RAG pipelines.
pub trait VectorStore: Send + Sync {
    async fn add_documents(&self, docs: Vec<Document>) -> Result<Vec<String>>;
    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<Document>>;
    async fn similarity_search_with_score(&self, query: &str, k: usize) -> Result<Vec<(Document, f64)>>;
}

/// Embedding provider for converting text to vectors.
pub trait Embeddings: Send + Sync {
    async fn embed_documents(&self, texts: Vec<&str>) -> Result<Vec<Vec<f64>>>;
    async fn embed_query(&self, text: &str) -> Result<Vec<f64>>;
}
```

---

## Writing Custom Tools

Implement the `BaseTool` trait to give your agent any capability:

```rust
use async_trait::async_trait;
use rustchain_core::tools::{BaseTool, types::{ToolInput, ToolOutput}};
use rustchain_core::error::Result;

struct WeatherTool;

#[async_trait]
impl BaseTool for WeatherTool {
    fn name(&self) -> &str { "weather" }
    fn description(&self) -> &str { "Get current weather for a city" }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let city = match &input {
            ToolInput::Text(s) => s.clone(),
            ToolInput::Structured(map) => {
                map.get("city").and_then(|v| v.as_str()).unwrap_or("unknown").to_string()
            }
            ToolInput::ToolCall(tc) => {
                tc.args.get("city").and_then(|v| v.as_str()).unwrap_or("unknown").to_string()
            }
        };
        // Call a real weather API here
        Ok(ToolOutput::Content(serde_json::json!(format!("72F and sunny in {city}"))))
    }
}
```

Then register it with an agent:

```rust
let executor = AgentExecutor::builder()
    .model(model)
    .tool(Arc::new(WeatherTool))
    .build();
```

---

## Workspace Structure

```
rustchain/
  Cargo.toml                  # Workspace root
  examples/                   # 9 runnable example programs
  docs/
    plans/                    # Design and planning documents
  crates/
    rustchain-core/           # Base traits and types (zero workspace deps)
    rustchain/                # Provider implementations and agent framework
    langgraph/                # State graph orchestration engine
    deepagents/               # High-level agent factory
    examples/                 # Example runner crate
```

---

## Roadmap

We're actively building RustChain. Here's what's coming next:

- [ ] Publish to [crates.io](https://crates.io)
- [ ] CI/CD pipeline with GitHub Actions
- [ ] More vector store backends (Qdrant, Pinecone, Weaviate, ChromaDB)
- [ ] Advanced RAG strategies (parent document retrieval, multi-vector, hybrid search)
- [ ] LangSmith-compatible observability and tracing
- [ ] `mdBook` documentation site with guides and tutorials
- [ ] WebSocket and SSE streaming adapters
- [ ] Plugin / extension system

---

## Contributing

Contributions are welcome and appreciated! Whether it's a bug fix, a new LLM provider, better documentation, or an entirely new feature — we'd love your help.

### How to Contribute

1. **Fork** the repository and clone it locally
2. **Create a branch** for your feature or fix:
   ```bash
   git checkout -b feat/my-feature
   ```
3. **Make your changes** — follow the existing code style and conventions
4. **Add tests** for new functionality
5. **Run the test suite** to make sure everything passes:
   ```bash
   cargo test --workspace
   ```
6. **Submit a Pull Request** with a clear description of what you changed and why

### Development Setup

```bash
# Clone the repo
git clone https://github.com/0xvasanth/rustchain.git
cd rustchain

# Build all crates
cargo build --workspace

# Run all tests
cargo test --workspace

# Run a specific example
cargo run -p rustchain-examples --example simple_chain

# Build with all LLM providers enabled
cargo build -p rustchain --features all-providers

# Check for warnings and clippy lints
cargo clippy --workspace
```

### Good First Issues

New to the project? Here are some great places to start:

- **Add a document loader** — YAML, TOML, XML, or another format
- **Add a text splitter** — for a specific programming language
- **Build a new tool** — HTTP client, date/time utilities, regex search
- **Improve error messages** — make errors more descriptive and actionable
- **Add doc comments** — help other developers understand the API
- **Write tests** — increase coverage for edge cases
- **Add a new example** — demonstrate a use case not yet covered

### Project Conventions

| Rule | Details |
|---|---|
| **Dependency boundaries** | `rustchain-core` has zero workspace dependencies. `langgraph` depends only on `rustchain-core`. |
| **Feature flags** | LLM providers must be gated behind feature flags |
| **Async runtime** | All async code uses `tokio` |
| **Error handling** | Per-crate error types using `thiserror` |
| **Documentation** | All public APIs should have `///` doc comments |
| **Testing** | New features require tests. Use `cargo test --workspace` to verify. |

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

---

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

---

<div align="center">

**Built with Rust. Inspired by LangChain.**

If you find RustChain useful, please consider giving it a star! It helps others discover the project.

[Report a Bug](https://github.com/0xvasanth/rustchain/issues) · [Request a Feature](https://github.com/0xvasanth/rustchain/issues) · [Start a Discussion](https://github.com/0xvasanth/rustchain/discussions)

</div>
