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

Cognis is a Rust-native framework for building LLM-powered applications — agents, RAG pipelines, and stateful graph workflows. If you've used LangChain / LangGraph / DeepAgents in Python, this is the same conceptual surface translated into idiomatic Rust: typed `Runnable<I, O>`, stateful `Graph<S>`, and a small agent loop you compose with builders rather than configure with strings.

## Why Cognis?

- **Compile-time type safety.** `Runnable<I, O>` carries types end-to-end — tool schemas, message variants, graph state transitions all checked at compile time. No runtime surprises.
- **Pay only for what you use.** Every external integration (providers, vector stores, checkpoint backends, observability exporters) is feature-gated. Your binary doesn't include OpenAI code if you only use Anthropic.
- **Async-native streaming.** Built on `tokio` and `futures::Stream`. Stream tokens, events, or graph state updates with a single API.
- **Production patterns built in.** Retry with backoff, circuit breakers, sliding-window / cost-based / token-bucket rate limiters, PII redaction, prompt caching, summarization, planning, and human-in-the-loop ship as composable middleware.
- **Stateful graph engine.** `cognis-graph` provides Pregel-style supersteps, per-field reducers, all 7 stream modes, interrupts, and time-travel via SQLite/Postgres checkpointers.
- **One umbrella, full stack.** `cognis` re-exports the foundation, LLM, RAG, and graph layers. Most apps need a single `use cognis::prelude::*;` and a few specific imports.

## Quick start

```toml
[dependencies]
cognis = { version = "0.2", features = ["ollama"] }   # or openai / anthropic / google / azure / all-providers
tokio = { version = "1", features = ["full"] }
```

### A 5-line agent

```rust
use std::sync::Arc;
use cognis::prelude::*;
use cognis::{AgentBuilder, Calculator};
use cognis_llm::Client;

#[tokio::main]
async fn main() -> Result<()> {
    // Reads COGNIS_PROVIDER + COGNIS_*_MODEL / API_KEY from env.
    let client = Client::from_env()?;

    let mut agent = AgentBuilder::new()
        .with_llm(client)
        .with_tool(Arc::new(Calculator::new()))
        .with_system_prompt("Use the calculator for any arithmetic. Always state the final answer.")
        .with_max_iterations(4)
        .build()?;

    let resp = agent.run(Message::human("What is 47 * 23?")).await?;
    println!("{}", resp.content);
    Ok(())
}
```

```bash
COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.2:1b cargo run
```

### Stateful graph workflow

```rust
use cognis::prelude::*;

#[derive(Default, Clone, Debug)]
struct State { count: u32 }

#[derive(Default, Clone)]
struct Update { count: u32 }

impl GraphState for State {
    type Update = Update;
    fn apply(&mut self, u: Update) { self.count += u.count; }
}

#[tokio::main]
async fn main() -> Result<()> {
    let tick = node_fn::<State, _, _>("tick", |s, _| {
        let cur = s.count;
        async move {
            if cur >= 5 {
                Ok(NodeOut { update: Update { count: 0 }, goto: Goto::end() })
            } else {
                Ok(NodeOut { update: Update { count: 1 }, goto: Goto::node("tick") })
            }
        }
    });

    let graph = Graph::<State>::new()
        .node("tick", tick)
        .start_at("tick")
        .compile()?;

    let final_state = graph.invoke(State::default(), Default::default()).await?;
    println!("final count: {}", final_state.count);
    Ok(())
}
```

Add a `Checkpointer` to get time-travel and interrupt-resume for free; see [`examples/graphs/graph_with_checkpoints.rs`](examples/graphs/graph_with_checkpoints.rs).

### RAG pipeline

```rust
use std::sync::Arc;
use cognis::prelude::*;
use cognis_llm::Client;
use cognis_rag::{
    Document, Embeddings, FakeEmbeddings, InMemoryVectorStore, RecursiveCharSplitter,
    TextSplitter, VectorStore,
};

#[tokio::main]
async fn main() -> Result<()> {
    let docs = vec![
        Document::new("Cognis is a Rust LLM framework."),
        Document::new("cognis-graph offers a stateful graph engine."),
        Document::new("cognis-rag bundles embeddings, vector stores, and retrievers."),
    ];
    let chunks = RecursiveCharSplitter::new().with_chunk_size(120).split_all(&docs);

    let emb: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(32));
    let mut store = InMemoryVectorStore::new(emb);
    store.add_texts(chunks.iter().map(|c| c.content.clone()).collect(), None).await?;

    let hits = store.similarity_search("What does cognis-rag include?", 2).await?;
    let context = hits.iter().map(|h| format!("- {}", h.text)).collect::<Vec<_>>().join("\n");

    let client = Client::from_env()?;
    let prompt = format!("Answer using only:\n{context}\n\nQ: What does cognis-rag include?\nA:");
    let resp = client.invoke(vec![Message::human(prompt)]).await?;
    println!("{}", resp.content());
    Ok(())
}
```

Swap `FakeEmbeddings` for `OpenAIEmbeddings`, `OllamaEmbeddings`, `GoogleEmbeddings`, or `VoyageEmbeddings`. Swap `InMemoryVectorStore` for FAISS, Chroma, Qdrant, Pinecone, or Weaviate (each behind a feature flag).

### Streaming

```rust
use cognis::prelude::*;
use cognis_llm::Client;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;
    let mut s = client.stream(vec![Message::human("Tell me a one-line joke.")]).await?;
    while let Some(chunk) = s.next().await {
        print!("{}", chunk?.content);
    }
    println!();
    Ok(())
}
```

### Multi-agent orchestration

```rust
use cognis::{AgentBuilder, MultiAgentOrchestrator, Sequential};
use cognis::prelude::*;
use cognis_llm::Client;

#[tokio::main]
async fn main() -> Result<()> {
    let planner = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_system_prompt("Break the request into 3 numbered steps. No explanations.")
        .build()?;
    let executor = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_system_prompt("Receive a numbered plan; reply with one paragraph on how you'd carry it out.")
        .build()?;

    let orch = MultiAgentOrchestrator::new(Sequential)
        .add("planner", planner)
        .add("executor", executor);

    let resp = orch.run("Help me prepare for a 5-minute team standup.").await?;
    println!("{}", resp.content);
    Ok(())
}
```

Other strategies: `Supervisor`, `ParallelVote`, `RoundRobin`. Plug in a custom `HandoffStrategy` for full control. For pub/sub broadcast across agents, see `cognis::AgentBus`.

### Tool orchestration with dependencies

```rust
use cognis::{ExecutionPlan, ToolOrchestrator, ToolStep};
use cognis_llm::tools::ToolInput;

let orch = ToolOrchestrator::new()
    .register(fetch_a)
    .register(fetch_b)
    .register(merge)
    .with_max_concurrency(4);

let plan = ExecutionPlan::new()
    .step(ToolStep::new("a", "fetch_a", ToolInput::Text("doc-1".into())))
    .step(ToolStep::new("b", "fetch_b", ToolInput::Text("doc-2".into())))
    .step(ToolStep::new("m", "merge", ToolInput::Text("combine".into())).after(["a", "b"]));

let result = orch.run(plan).await?;
```

The orchestrator topo-sorts the DAG, runs independent steps concurrently, and skips downstream steps whose ancestors errored.

## Workspace layout

```
crates/
├── cognis-core    # Foundation: Runnable<I,O>, Message, prompts, output
│                  # parsers, callbacks/Observer/Event, wrappers, compose.
├── cognis-llm     # LLM client + providers (OpenAI/Anthropic/Google/
│                  # Ollama/Azure/OpenRouter), Tool trait, streaming.
├── cognis-rag     # Embeddings, vector stores, retrievers, splitters,
│                  # loaders, IndexingPipeline, document transformers.
├── cognisgraph    # Crate name `cognis-graph`. Stateful Graph<S>,
│                  # checkpointers, interrupts, time-travel, viz.
├── cognis-trace   # Pluggable observability adapters (Langfuse,
│                  # LangSmith, OpenTelemetry).
├── cognis-macros  # Proc macros: #[tool], #[derive(GraphState)].
├── cognis         # Umbrella + agent layer: AgentBuilder,
│                  # MultiAgentOrchestrator, AgentBus, memory variants,
│                  # middleware, built-in tools, ToolOrchestrator.
└── examples       # 100+ runnable demos under examples/<category>/.
```

| Layer            | Crate           | What it provides                                                                                                              |
| ---------------- | --------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| **Foundation**   | `cognis-core`   | `Runnable<I, O>`, `Message`, `ContentPart`, prompts, output parsers (incl. `OutputFixingParser` / `RetryParser`), callbacks   |
| **LLM**          | `cognis-llm`    | `Client`, providers, `Tool` trait, streaming, structured output                                                               |
| **RAG**          | `cognis-rag`    | Embeddings, vector stores (6), retrievers (9+), splitters, loaders, indexing, transformers                                    |
| **Graph**        | `cognis-graph`  | `Graph<S>`, Pregel engine, reducers, channels, checkpointers (in-memory/SQLite/Postgres), 7 stream modes, viz (DOT/Mermaid)  |
| **Tracing**      | `cognis-trace`  | Langfuse / LangSmith / OpenTelemetry exporters                                                                                |
| **Agent**        | `cognis`        | `AgentBuilder`, multi-agent (Sequential/Supervisor/ParallelVote/RoundRobin), `AgentBus`, 7 memory types, middleware, tools    |

`cognis-core` has zero internal-crate dependencies. Sibling crates depend only on `cognis-core` (and macros where needed). `cognis` is the only crate that depends on the siblings together.

## Feature flags

```toml
# Pick providers
cognis = { version = "0.2", features = ["openai", "anthropic"] }
# Or take everything
cognis = { version = "0.2", features = ["all-providers"] }

# Graph workflows with persistence
cognis-graph = { version = "0.2", features = ["sqlite"] }   # or "postgres"

# Vector stores (each opt-in)
cognis-rag = { version = "0.2", features = ["faiss", "openai"] }
```

| Crate          | Flags                                                                                            |
| -------------- | ------------------------------------------------------------------------------------------------ |
| `cognis`       | `openai`, `anthropic`, `google`, `ollama`, `azure`, `openrouter`, `all-providers`; `pdf`, `yaml`, `toml-loader`; `cache-sqlite`; `tools-http` |
| `cognis-graph` | `sqlite`, `postgres` (checkpointers)                                                             |
| `cognis-rag`   | `openai`, `google`, `voyage`, `ollama` (embeddings); `faiss`, `chroma`, `qdrant`, `pinecone`, `weaviate` (vector stores) |
| `cognis-trace` | `stdout` (default), `langfuse`, `langsmith`, `otel`                                              |

## Examples

```bash
git clone https://github.com/0xvasanth/cognis.git
cd cognis

# Offline demos (no API keys needed)
cargo run -p cognis-examples --example chains_pipe_operator
cargo run -p cognis-examples --example tools_orchestrator
cargo run -p cognis-examples --example agents_round_robin
cargo run -p cognis-examples --example agents_bus_pubsub
cargo run -p cognis-examples --example memory_entity
cargo run -p cognis-examples --example memory_knowledge_graph
cargo run -p cognis-examples --example retrieval_document_transformers
cargo run -p cognis-examples --example graphs_state_machine
cargo run -p cognis-examples --example graphs_dot_export
cargo run -p cognis-examples --example resilience_advanced_rate_limiters
cargo run -p cognis-examples --example parsers_fixing
cargo run -p cognis-examples --example parsers_retry

# Provider-backed demos (need a running LLM)
COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.2:1b \
  cargo run -p cognis-examples --example agents_react_agent
COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.2:1b \
  cargo run -p cognis-examples --example retrieval_rag_pipeline
```

The full demo set lives under [`examples/`](examples/) — organized by `chains/`, `agents/`, `memory/`, `models/`, `tools/`, `retrieval/`, `graphs/`, `observability/`, `resilience/`, and `parsers/`. Every example is registered in [`crates/examples/Cargo.toml`](crates/examples/Cargo.toml).

## Build & test

```bash
cargo build --workspace
cargo build -p cognis --features all-providers
cargo test --workspace
cargo test -p cognis --lib agent::memory          # one module
cargo clippy --workspace --all-targets -- -D warnings
```

## Documentation
- API docs: [docs.rs/cognis](https://docs.rs/cognis), per-crate.

## Contributing

See [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines, project structure, and conventions. Workflow rules and design patterns are codified in [CLAUDE.md](CLAUDE.md).

## Acknowledgments

Cognis is heavily inspired by the [LangChain](https://github.com/langchain-ai/langchain), [LangGraph](https://github.com/langchain-ai/langgraph), and [DeepAgents](https://github.com/langchain-ai/deepagents) Python ecosystem. Thanks to the LangChain team for pioneering the composable LLM framework paradigm — their abstractions and developer experience were the foundation that made the Rust port possible.

## License

MIT

---

<div align="center">

[Report a Bug](https://github.com/0xvasanth/cognis/issues) · [Request a Feature](https://github.com/0xvasanth/cognis/issues) · [Discussions](https://github.com/0xvasanth/cognis/discussions)

</div>
