//! Ollama + tool calling. Builds an agent with a Calculator tool and
//! runs it against a local Ollama model. Requires Ollama to be running
//! with a model that supports function calling (e.g. `llama3.1`,
//! `qwen2.5`). Smaller `llama3.2:1b` may not always emit tool calls.
//!
//! Usage:
//! ```bash
//! COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=qwen2.5:3b \
//!   cargo run --example 06_ollama_tool_calling -p cognis-examples
//! ```

use std::sync::Arc;

use cognis::prelude::*;
use cognis::{AgentBuilder, Calculator};
use cognis_llm::Client;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        // Default to ollama for this example so it Just Works locally
        // when an Ollama daemon is up.
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let client = Client::from_env()?;
    let mut agent = AgentBuilder::new()
        .with_llm(client)
        .with_tool(Arc::new(Calculator::new()))
        .with_system_prompt(
            "You are a math assistant. When the user asks a calculation, \
             use the `calculator` tool. Always show the final answer.",
        )
        .with_max_iterations(4)
        .build()?;

    let resp = agent.run(Message::human("What is 23 * 17 + 4?")).await?;
    println!("---");
    println!("{}", resp.content);
    println!("---");
    println!("messages exchanged: {}", resp.messages.len());
    Ok(())
}
