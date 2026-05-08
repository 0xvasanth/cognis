//! What you'll learn:
//!   How to register a tool with `AgentBuilder` and watch the agent
//!   loop pick it, call it, and feed the result back to the model.
//!
//! Why this matters:
//!   Tool calling is the single biggest force-multiplier for an LLM:
//!   it's how the model reaches calculators, databases, and APIs
//!   instead of guessing. The agent loop, the tool dispatch, and the
//!   message wiring are all handled — you just register the tool.
//!
//! Scenario:
//!   The user asks "What is 23 * 17 + 4?". The agent decides to call
//!   the `calculator` tool, observes the result, and replies with the
//!   final number — the textbook ReAct loop wired in five lines.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=qwen2.5:3b \
//!     cargo run -p cognis-examples --example 06_ollama_tool_calling
//!
//! Sample output (against ollama / llama3.1):
//!   ---
//!   The result of the calculation 23 * 17 + 4 is 395.
//!   ---
//!   messages exchanged: 3
//!
//! Note: needs a model with native function-calling (e.g. `llama3.1`,
//! `qwen2.5`). Smaller `llama3.2:1b` may not always emit tool calls.

use std::sync::Arc;

use cognis::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
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
