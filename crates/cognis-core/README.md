# cognis-core

The foundation crate for Cognis v2, providing the core traits and primitives for building composable AI workflows.

## Purpose
`cognis-core` defines the typed `Runnable<I, O>` trait, which is the building block of all operations in Cognis. It also provides the essential data structures for chat messages, prompts, output parsing, and functional composition.

## Key Features
- **Runnable Trait**: A standardized interface for any unit of work that can be invoked, batched, or streamed.
- **Composition**: Primitives like `pipe`, `Parallel`, and `Branch` to build complex chains from simple runnables.
- **Messages**: Typed chat messages (`SystemMessage`, `HumanMessage`, `AiMessage`, `ToolMessage`) with support for multi-modal content.
- **Prompts**: Template systems for generating formatted strings or message sequences.
- **Output Parsers**: Tools to transform raw LLM output into structured data (JSON, XML, lists, etc.).
- **Wrappers**: Higher-order runnables for `Retry`, `Fallback`, `Timeout`, and `Cache`.

## Usage
Add this to your `Cargo.toml`:
```toml
[dependencies]
cognis-core = "0.1.0"
```

### Basic Example: Defining and Piping Runnables
```rust
use cognis_core::prelude::*;
use async_trait::async_trait;

struct Doubler;

#[async_trait]
impl Runnable<u32, u32> for Doubler {
    async fn invoke(&self, input: u32, _: RunnableConfig) -> Result<u32> {
        Ok(input * 2)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let chain = pipe(Doubler, lambda(|x: u32| async move { Ok(x + 1) }));
    
    let result = chain.invoke(5, RunnableConfig::default()).await?;
    assert_eq!(result, 11); // (5 * 2) + 1
    Ok(())
}
```
