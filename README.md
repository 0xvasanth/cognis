# RustChain -- LLM Application Framework in Rust

RustChain is a modular framework for building LLM-powered applications in Rust.
It provides composable abstractions for chat models, tool calling, agent orchestration,
and stateful multi-step workflows, with async-first design built on tokio.

## Current Status

- **4 crates** -- rustchain-core, rustchain, langgraph, deepagents
- **302+ source files** with **2100+ tests**
- **5 LLM providers** -- Anthropic, OpenAI, Google Gemini, Ollama, Azure OpenAI
- **9 runnable examples** that work without API keys (fake models)

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

Add the crates you need to your `Cargo.toml`:

```toml
[dependencies]
rustchain-core = { path = "crates/rustchain-core" }
rustchain = { path = "crates/rustchain", features = ["anthropic"] }
langgraph = { path = "crates/langgraph" }
```

Build a chain with the LCEL pattern (no API key needed):

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

    let result = chain.invoke(json!({ "topic": "Rust" }), None).await?;
    println!("{}", result);
    Ok(())
}
```

## Features by Crate

### rustchain-core (Foundation)

| Module | Description |
|--------|-------------|
| `messages` | `Message` enum (Human, AI, System, Tool, Function) with merge/trim utilities |
| `language_models` | `BaseChatModel`, `BaseLLM` traits, fake/testing models |
| `runnables` | `Runnable` trait with sequence, parallel, branch, lambda, retry, fallback |
| `tools` | `BaseTool` trait and toolkit interface for agent tool calling |
| `prompts` | Chat prompt templates, few-shot selectors, structured prompts |
| `output_parsers` | JSON, string, list, XML, and tool-call parsers |
| `callbacks` | Extensible callback system with run managers and tracers |
| `vectorstores` | `VectorStore` trait, `InMemoryVectorStore`, similarity search |
| `embeddings` | `Embeddings` trait for vector embedding providers |
| `documents` | `Document` type used across loaders, splitters, and retrievers |
| `retrievers` | `BaseRetriever` trait for document retrieval |
| `indexing` | Document indexing with record managers |

### rustchain (Implementation)

| Module | Description |
|--------|-------------|
| `chat_models` | Anthropic Claude, OpenAI GPT, Google Gemini, Ollama, Azure OpenAI |
| `embeddings` | OpenAI and Ollama embedding providers |
| `agents` | `AgentExecutor` with middleware pipeline (retry, PII redaction, summarization) |
| `chains` | LLMChain, ConversationChain, SequentialChain, RetrievalQAChain, MapReduceChain, RefineChain, RouterChain |
| `memory` | ConversationBufferMemory, ConversationWindowMemory, ConversationSummaryMemory, VectorStoreMemory |
| `document_loaders` | Text, CSV, JSON, and directory loaders |
| `text_splitter` | Character, recursive, markdown, HTML, JSON, code, and token splitters |
| `tools` | Calculator, shell command, and JSON query tools |
| `vectorstores` | Vector store integrations |

### langgraph (Orchestration)

| Module | Description |
|--------|-------------|
| `graph` | `StateGraph` builder, `CompiledStateGraph`, conditional branching, subgraphs |
| `graph::persistent` | `PersistentGraph` with automatic checkpoint save/restore and fork |
| `pregel` | Pregel-style execution engine with superstep processing |
| `channels` | LastValue, BinaryOp, Topic, AnyValue, NamedBarrier, EphemeralValue |
| `checkpoint` | `CheckpointSaver` trait, `InMemoryCheckpointSaver`, SQLite backend |
| `prebuilt` | `create_react_agent` for tool-calling ReAct loops |
| `types` | StreamMode, InterruptType, RetryPolicy, CachePolicy |
| `runtime` | Tokio-based async runtime utilities |

### deepagents (Application)

| Module | Description |
|--------|-------------|
| `graph` | `create_deep_agent()` factory returning a compiled graph with middleware |
| `middleware` | `Middleware` trait with before/after hooks (filesystem, memory, sub-agent, summarization) |
| `backends` | `Backend` trait for session state (StateBackend, FilesystemBackend) |

## Examples

All examples run without API keys using fake/mock models.

```sh
# LCEL chain composition (prompt -> model -> parser)
cargo run -p rustchain-examples --example simple_chain

# Full RAG pipeline (load -> split -> embed -> store -> retrieve -> answer)
cargo run -p rustchain-examples --example rag_pipeline
cargo run -p rustchain-examples --example rag_with_vectorstore

# Tool-calling agent with AgentExecutor
cargo run -p rustchain-examples --example tool_agent

# ReAct agent with LangGraph
cargo run -p rustchain-examples --example langgraph_agent

# Streaming responses (character-level and word-level)
cargo run -p rustchain-examples --example streaming

# Multi-turn conversation with memory
cargo run -p rustchain-examples --example conversational_agent

# Persistent graph execution with checkpointing and fork
cargo run -p rustchain-examples --example graph_with_checkpoints

# Semantic routing to different prompt chains
cargo run -p rustchain-examples --example semantic_router
```

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
  examples/                   # Runnable example programs
  crates/
    rustchain-core/           # Base traits and types
    rustchain/                # Provider implementations and agent framework
    langgraph/                # State graph orchestration engine
    deepagents/               # High-level agent factory
    examples/                 # Example runner crate (registers examples)
```

## License

MIT
