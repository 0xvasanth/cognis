<div align="center">

# RustChain

**Build LLM apps in Rust. Fast, type-safe, composable.**

[![CI](https://img.shields.io/github/actions/workflow/status/0xvasanth/rustchain/ci.yml?branch=main)](https://github.com/0xvasanth/rustchain/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

</div>

---

RustChain is a Rust-native framework for building LLM-powered applications — chains, agents, RAG pipelines, and stateful workflows. If you've used LangChain in Python, this is the same mental model with Rust's performance and compile-time guarantees.

## Why RustChain?

- **Compile-time safety** — Tool schemas, message types, and state transitions are checked before your code runs. No more runtime surprises.
- **Pay only for what you use** — LLM providers are behind feature flags. Your binary doesn't include OpenAI code if you only use Anthropic.
- **Async-native streaming** — Built on `tokio` with `futures::Stream`. Stream tokens from any provider with the same API.
- **Production patterns included** — Circuit breakers, retry with backoff, rate limiting, PII redaction, and human-in-the-loop are built into the middleware pipeline.
- **One workspace, full stack** — Chains, agents, RAG, graph workflows, and high-level agent orchestration all live together with strict dependency boundaries.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
rustchain = { git = "https://github.com/0xvasanth/rustchain", features = ["openai"] }
rustchain-core = { git = "https://github.com/0xvasanth/rustchain" }
tokio = { version = "1", features = ["full"] }
serde_json = "1"
```

### Chain: Prompt → Model → Parser

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
        "Rust is a systems language focused on safety and speed.".into(),
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

> Swap `FakeListChatModel` for `ChatOpenAI`, `ChatAnthropic`, `ChatGoogleGenAI`, or `ChatOllama` for real LLM calls.

### Stateful Graph Workflow

Build multi-step workflows with conditional branching, checkpointing, and human-in-the-loop:

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

### RAG in 10 Lines

```rust
use rustchain::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};
use rustchain::document_loaders::text::TextLoader;
use rustchain_core::document_loaders::BaseLoader;
use rustchain_core::vectorstores::{in_memory::InMemoryVectorStore, base::VectorStore};

let docs = TextLoader::new("data.txt").load().await?;

let splitter = RecursiveCharacterTextSplitter::new()
    .with_chunk_size(500)
    .with_chunk_overlap(50);
let chunks = splitter.split_documents(&docs);

let store = InMemoryVectorStore::new(embedding_model);
store.add_documents(chunks, None).await?;

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
    print!("{}", chunk?.message.base.content.text());
}
```

## What's Included

| Layer | Crate | What it does |
|---|---|---|
| **Foundation** | `rustchain-core` | Base traits (`ChatModel`, `Tool`, `Runnable`, `VectorStore`), message types, prompt templates, output parsers, callbacks |
| **Implementation** | `rustchain` | 5 LLM providers, 19 chain types, 14 retrievers, 11 memory types, 6 vector stores, document loaders, text splitters, tools |
| **Orchestration** | `langgraph` | State graphs, Pregel execution engine, checkpointing (SQLite/Postgres), streaming, human-in-the-loop, subgraph composition |
| **Application** | `deepagents` | Zero-boilerplate agent factory, middleware pipeline, sandboxed execution, planning, plugins, workflow engine |

## Providers

Enable only what you need via feature flags:

```toml
# Pick your providers
rustchain = { git = "https://github.com/0xvasanth/rustchain", features = ["anthropic", "openai"] }

# Or enable everything
rustchain = { git = "https://github.com/0xvasanth/rustchain", features = ["all-providers"] }

# Graph workflows with persistence
langgraph = { git = "https://github.com/0xvasanth/rustchain", features = ["sqlite"] }
```

| Flag | Provider |
|---|---|
| `openai` | OpenAI GPT models + embeddings |
| `anthropic` | Anthropic Claude models + embeddings |
| `google` | Google Gemini models + embeddings |
| `ollama` | Ollama local models + embeddings |
| `azure` | Azure OpenAI |
| `qdrant` / `pinecone` / `weaviate` / `chroma` / `faiss` | Vector store backends |
| `sqlite` / `postgres` | Checkpoint persistence (langgraph) |

## Examples

All examples work without API keys using mock models:

```bash
git clone https://github.com/0xvasanth/rustchain.git
cd rustchain

cargo run --example simple_chain           # Basic chain composition
cargo run --example tool_agent             # Agent with tool calling
cargo run --example rag_pipeline           # Full RAG pipeline
cargo run --example langgraph_agent        # Stateful graph agent
cargo run --example streaming              # Token streaming
cargo run --example graph_with_checkpoints # Persistent graph workflows
cargo run --example memory_types           # Conversation memory
cargo run --example plan_and_execute       # Planning agent
```

See the [`examples/`](examples/) directory for 35+ runnable demos.

## Contributing

```bash
cargo build --workspace          # Build
cargo test --workspace           # Test
cargo clippy --workspace         # Lint
cargo fmt --all                  # Format
```

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines, project structure, and conventions.

## License

MIT

---

<div align="center">

[Report a Bug](https://github.com/0xvasanth/rustchain/issues) · [Request a Feature](https://github.com/0xvasanth/rustchain/issues) · [Discussions](https://github.com/0xvasanth/rustchain/discussions)

</div>
