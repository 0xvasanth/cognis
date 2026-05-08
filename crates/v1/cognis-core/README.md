<div align="center">

# cognis-core

**The foundation traits and types for the Cognis LLM framework.**

[![crates.io](https://img.shields.io/crates/v/cognis-core.svg)](https://crates.io/crates/cognis-core)
[![docs.rs](https://docs.rs/cognis-core/badge.svg)](https://docs.rs/cognis-core)
[![MIT](https://img.shields.io/crates/l/cognis-core.svg)](https://opensource.org/licenses/MIT)

[Workspace](https://github.com/0xvasanth/cognis) | [API Docs](https://docs.rs/cognis-core)

</div>

---

`cognis-core` defines the trait interfaces and types that every crate in the [Cognis](https://github.com/0xvasanth/cognis) workspace builds on. It has **zero** workspace dependencies — everything starts here.

## Core Abstractions

```text
BaseChatModel  ─  Chat model providers (Anthropic, OpenAI, ...)
BaseTool       ─  Tools that agents can call
Runnable       ─  Composable async computation unit (the LCEL equivalent)
Message        ─  Human | AI | System | Tool message types
Embeddings     ─  Vector embedding providers
VectorStore    ─  Similarity search interface
Document       ─  Text + metadata, used across loaders and retrievers
```

## Composable Chains with `chain!`

The `Runnable` trait and its combinators let you compose pipelines:

```rust
use std::sync::Arc;
use cognis_core::chain;
use cognis_core::language_models::{ChatModelRunnable, FakeListChatModel};
use cognis_core::output_parsers::StrOutputParser;
use cognis_core::prompts::ChatPromptTemplate;
use cognis_core::runnables::Runnable;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = ChatPromptTemplate::from_messages(vec![
        ("system", "You are a helpful assistant."),
        ("human", "Explain {topic} in one sentence."),
    ])?;

    let model = FakeListChatModel::new(vec![
        "Rust ensures memory safety without garbage collection.".into(),
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

## Runnable Combinators

| Combinator | What it does |
|------------|-------------|
| `chain!` / `RunnableSequence` | Pipeline: A then B then C |
| `RunnableParallel` | Fan-out: run A, B, C concurrently |
| `RunnableBranch` | Conditional routing based on input |
| `RunnableLambda` | Wrap any closure as a Runnable |
| `RunnableWithFallbacks` | Try A, fall back to B on error |
| `RunnableRetry` | Retry with configurable backoff |

## Testing

Fake model implementations are included for testing without API keys or network:

- `FakeListChatModel` — cycles through predefined string responses
- `FakeMessagesListChatModel` — cycles through predefined `Message` responses
- `GenericFakeChatModel` — word-level streaming from predefined messages
- `FakeListLLM` — completion-style fake model
- `DeterministicFakeEmbedding` — produces reproducible embeddings

## Part of the Cognis Workspace

| Crate | Role |
|-------|------|
| **cognis-core** | Foundation traits and types (you are here) |
| [cognis](https://crates.io/crates/cognis) | LLM providers, chains, memory, tools |
| [cognisgraph](https://crates.io/crates/cognisgraph) | State graph orchestration engine |
| [cognisagent](https://crates.io/crates/cognisagent) | High-level agent framework |
