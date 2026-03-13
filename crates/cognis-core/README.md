# cognis-core

Foundation layer for the Cognis LLM framework. Defines the base traits, types, and
abstractions that all other workspace crates depend on. This crate has **zero** dependencies
on other workspace crates.

## Key Traits and Types

| Trait / Type | Module | Description |
|-------------|--------|-------------|
| `BaseChatModel` | `language_models` | Interface for chat model providers |
| `BaseLLM` | `language_models` | Interface for completion-style LLMs |
| `Runnable` | `runnables` | Composable unit of async computation |
| `BaseTool` | `tools` | Interface for agent-callable tools |
| `Embeddings` | `embeddings` | Interface for vector embedding providers |
| `Message` | `messages` | Enum: Human, AI, System, Tool, Function |
| `Document` | `documents` | Text document with metadata |
| `VectorStore` | `vectorstores` | Interface for similarity search stores |

## Runnables

The `Runnable` trait and its combinators form the LCEL (LangChain Expression Language) equivalent:

- `RunnableSequence` -- chain runnables with the `chain!` macro
- `RunnableParallel` -- fan-out to multiple runnables
- `RunnableBranch` -- conditional routing
- `RunnableLambda` -- wrap a closure as a runnable
- `RunnableWithFallbacks` -- automatic fallback on error
- `RunnableRetry` -- retry with configurable policy

## Usage

```toml
[dependencies]
cognis-core = { path = "../cognis-core" }
```

```rust
use cognis_core::messages::Message;
use cognis_core::language_models::FakeListChatModel;

let model = FakeListChatModel::new(vec!["Paris".into()]);
let msg = Message::human("What is the capital of France?");
```

## Testing

Fake model implementations (`FakeListChatModel`, `FakeListLLM`, `ParrotFakeChatModel`, etc.)
are provided for testing without network calls.
