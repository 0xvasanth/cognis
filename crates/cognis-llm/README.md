# cognis-llm

A unified client for Large Language Models (LLMs) with built-in support for tool calling and multiple providers.

## Purpose
`cognis-llm` provides a standard `Client` and `Provider` abstraction to interact with various LLM APIs (OpenAI, Ollama, Anthropic, etc.). It simplifies the process of sending chat messages, receiving responses, and handling complex tool-calling workflows.

## Key Features
- **Multi-Provider Support**: Switch between OpenAI, Ollama, Anthropic, Google, and Azure with minimal configuration.
- **Unified Client**: A single `Client` interface that implements `Runnable<Vec<Message>, AiMessage>`.
- **Tool Ergonomics**: Simplified 5-tier tool system for defining and executing functions that LLMs can call.
- **Structured Output**: Support for enforcing JSON schemas on model responses.
- **Resilience**: Built-in `CircuitBreaker`, `LoadBalancer`, and `Retryable` provider wrappers.

## Usage
Add this to your `Cargo.toml`:
```toml
[dependencies]
cognis-llm = "0.1.0"
```

### Basic Example: Simple Chat
```rust
use cognis_llm::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Requires COGNIS_OPENAI_API_KEY environment variable
    let client = Client::from_env()?;
    
    let response = client
        .invoke(vec![Message::human("What is the capital of France?")])
        .await?;
        
    println!("Response: {}", response.content);
    Ok(())
}
```

### Tool Calling Example
```rust
use cognis_llm::prelude::*;
use cognis_macros::tool;

#[tool]
/// Get the current weather for a location.
async fn get_weather(location: String) -> Result<String> {
    Ok(format!("The weather in {} is sunny.", location))
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?.with_tool(get_weather);
    
    let response = client
        .invoke(vec![Message::human("What is the weather in Paris?")])
        .await?;
        
    println!("Response: {:?}", response.tool_calls);
    Ok(())
}
```
