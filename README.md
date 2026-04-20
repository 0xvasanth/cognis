<div align="center">

# Cognis

**Build LLM apps in Rust. Fast, type-safe, composable.**

[![crates.io](https://img.shields.io/crates/v/cognis.svg)](https://crates.io/crates/cognis)
[![docs.rs](https://docs.rs/cognis/badge.svg)](https://docs.rs/cognis)
[![CI](https://img.shields.io/github/actions/workflow/status/0xvasanth/cognis/ci.yml?branch=main)](https://github.com/0xvasanth/cognis/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org)

</div>

---

Cognis is a Rust-native framework for building LLM-powered applications — chains, agents, RAG pipelines, and stateful workflows. If you've used LangChain in Python, this is the same mental model with Rust's performance and compile-time guarantees.

## Why Cognis?

- **Compile-time safety** — Tool schemas, message types, and state transitions are checked before your code runs. No more runtime surprises.
- **Pay only for what you use** — LLM providers are behind feature flags. Your binary doesn't include OpenAI code if you only use Anthropic.
- **Async-native streaming** — Built on `tokio` with `futures::Stream`. Stream tokens from any provider with the same API.
- **Production patterns included** — Circuit breakers, retry with backoff, rate limiting, PII redaction, and human-in-the-loop are built into the middleware pipeline.
- **One workspace, full stack** — Chains, agents, RAG, graph workflows, and high-level agent orchestration all live together with strict dependency boundaries.

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
cognis = { version = "0.1", features = ["openai"] }
cognis-core = "0.1"
tokio = { version = "1", features = ["full"] }
serde_json = "1"
```

### Chain: Prompt → Model → Parser

```rust
use std::sync::Arc;
use serde_json::json;
use cognis_core::chain;
use cognis_core::language_models::{ChatModelRunnable, FakeListChatModel};
use cognis_core::output_parsers::StrOutputParser;
use cognis_core::prompts::ChatPromptTemplate;
use cognis_core::runnables::Runnable;

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
use cognisgraph::graph::state::{AsyncNodeAction, StateGraph};

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
use cognis::text_splitter::{RecursiveCharacterTextSplitter, TextSplitter};
use cognis::document_loaders::text::TextLoader;
use cognis_core::document_loaders::BaseLoader;
use cognis_core::vectorstores::{in_memory::InMemoryVectorStore, base::VectorStore};

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
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{HumanMessage, Message};

let messages = vec![Message::Human(HumanMessage::new("Tell me a story"))];
let mut stream = model._stream(&messages, None).await?;

while let Some(chunk) = stream.next().await {
    print!("{}", chunk?.message.base.content.text());
}
```

### Building an Agent

Chains answer questions. *Agents* loop over a model, call tools, and decide what to do next. Don't hand-roll the loop — `AgentExecutor` drives it for you:

```rust
use std::sync::Arc;
use cognis::agents::AgentExecutor;
use cognis_core::messages::Message;
use cognis_core::tools::base::BaseTool;
use cognis_core::tools::simple::SimpleTool;

let search = SimpleTool::new(
    "search",
    "Search for information about a topic",
    |query: &str| Ok(format!("Results for '{query}': ...")),
);

let executor = AgentExecutor::builder()
    .model(model)
    .tool(Arc::new(search) as Arc<dyn BaseTool>)
    .max_iterations(10)
    .handle_parsing_errors(true)
    .build();

let result = executor.run(&[Message::human("What is Rust?")]).await?;
println!("{}", result.output);
```

No JSON parser, no scratchpad formatting, no per-provider tool-call handling. The executor appends `ToolMessage`s with correct `tool_call_id`s, stops on `max_iterations`, and supports callbacks for streaming intermediate steps.

For the full walk-through — `SimpleTool` vs `StructuredTool` vs custom `BaseTool`, `with_structured_output`, `OutputFixingParser`, `CallbackHandler`, and the graph-based `create_react_agent` — see **[docs/building-an-agent.md](docs/building-an-agent.md)**.

## What's Included

| Layer              | Crate         | What it does                                                                                                               |
| ------------------ | ------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Foundation**     | `cognis-core` | Base traits (`ChatModel`, `Tool`, `Runnable`, `VectorStore`), message types, prompt templates, output parsers, callbacks   |
| **Implementation** | `cognis`      | 5 LLM providers, 19 chain types, 14 retrievers, 11 memory types, 6 vector stores, document loaders, text splitters, tools  |
| **Orchestration**  | `cognisgraph` | State graphs, Pregel execution engine, checkpointing (SQLite/Postgres), streaming, human-in-the-loop, subgraph composition |
| **Application**    | `cognisagent` | Zero-boilerplate agent factory, middleware pipeline, sandboxed execution, planning, plugins, workflow engine               |

## Providers

Enable only what you need via feature flags:

```toml
# Pick your providers
cognis = { version = "0.1", features = ["anthropic", "openai"] }

# Or enable everything
cognis = { version = "0.1", features = ["all-providers"] }

# Graph workflows with persistence
cognisgraph = { version = "0.1", features = ["sqlite"] }
```

| Flag                                                    | Provider                             |
| ------------------------------------------------------- | ------------------------------------ |
| `openai`                                                | OpenAI GPT models + embeddings       |
| `anthropic`                                             | Anthropic Claude models + embeddings |
| `google`                                                | Google Gemini models + embeddings    |
| `ollama`                                                | Ollama local models + embeddings     |
| `azure`                                                 | Azure OpenAI                         |
| `qdrant` / `pinecone` / `weaviate` / `chroma` / `faiss` | Vector store backends                |
| `sqlite` / `postgres`                                   | Checkpoint persistence (cognisgraph) |

## Examples

All examples work without API keys using mock models:

```bash
git clone https://github.com/0xvasanth/cognis.git
cd cognis

cargo run --example simple_chain           # Basic chain composition
cargo run --example tool_agent             # Agent with tool calling
cargo run --example rag_pipeline           # Full RAG pipeline
cargo run --example cognisgraph_agent        # Stateful graph agent
cargo run --example streaming              # Token streaming
cargo run --example graph_with_checkpoints # Persistent graph workflows
cargo run --example memory_types           # Conversation memory
cargo run --example plan_and_execute       # Planning agent
```

See the [`examples/`](examples/) directory for 35+ runnable demos.

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines, project structure, and conventions.

## Acknowledgments

Cognis is heavily inspired by the [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph), and [DeepAgents](https://github.com/langchain-ai/deepagents) Python ecosystem. Huge thanks to the LangChain team for pioneering the composable LLM framework paradigm — their design patterns, abstractions, and developer experience were the foundation that made this Rust port possible.

## License

MIT

---

<div align="center">

[Report a Bug](https://github.com/0xvasanth/cognis/issues) · [Request a Feature](https://github.com/0xvasanth/cognis/issues) · [Discussions](https://github.com/0xvasanth/cognis/discussions)

</div>
