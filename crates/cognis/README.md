# cognis

The high-level umbrella crate for the Cognis v2 framework, providing advanced agentic capabilities, middleware, and a unified API.

## Purpose
`cognis` is the primary entry point for developers building AI-powered applications with Cognis. It re-exports all core sub-crates (`core`, `llm`, `rag`, `graph`) and adds high-level abstractions like ReAct agents, composable middleware, and execution backends.

## Key Features
- **ReAct Agents**: Out-of-the-box support for "Reason + Act" agent loops with flexible planning and tool usage.
- **Middleware Pipeline**: A powerful hook system for adding cross-cutting concerns like caching, rate limiting, human-in-the-loop approvals, and PII redaction.
- **Backends**: Unified interface for managing agent workspaces, supporting both in-memory and sandboxed filesystem storage.
- **Telemetry & Observability**: Integrated tracing and event logging to monitor agent behavior and performance.
- **Unified Prelude**: A single `prelude` module that brings in the most commonly used types and traits from across the entire framework.

## Usage
Add this to your `Cargo.toml`:
```toml
[dependencies]
cognis = "0.1.0"
```

### Basic Example: A Five-Line ReAct Agent
```rust
use cognis::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;
    let agent = Agent::builder()
        .client(client)
        .build();

    let response = agent.invoke("Explain quantum entanglement in one sentence.").await?;
    println!("Agent: {}", response.content);
    Ok(())
}
```

### Advanced Example: Agent with Middleware and Tools
```rust
use cognis::prelude::*;
use cognis_macros::tool;

#[tool]
/// Multiply two numbers.
async fn multiply(a: i32, b: i32) -> Result<i32> {
    Ok(a * b)
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?
        .with_middleware(ModelRetry::default())
        .with_middleware(PromptCaching::default());

    let agent = Agent::builder()
        .client(client)
        .with_tool(multiply)
        .build();

    let response = agent.invoke("What is 15 * 42?").await?;
    println!("Agent: {}", response.content);
    Ok(())
}
```
