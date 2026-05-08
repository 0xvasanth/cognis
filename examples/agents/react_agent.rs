//! What you'll learn:
//!   How to wire up the textbook ReAct loop — model reasons, calls a
//!   tool, observes the result, then writes the final answer — with
//!   `AgentBuilder`.
//!
//! Why this matters:
//!   ReAct is the default control flow for tool-using LLM agents.
//!   Cognis ships it as the standard agent shape; once you understand
//!   `with_tool` + `with_max_iterations`, the same code drives anything
//!   from a calculator to a multi-tool research agent.
//!
//! Scenario:
//!   The user asks a multi-step arithmetic question. The agent reasons
//!   about it, dispatches the `calculator` tool, observes the result,
//!   and writes the final answer — the canonical ReAct trace.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example agents_react_agent
//!
//! Sample output (against ollama / llama3.1):
//!   The result of multiplying 12 by 8 is 96.
//!   (iterations: 3 messages)

use std::sync::Arc;

use cognis::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let mut agent = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_tool(Arc::new(Calculator::new()))
        .with_system_prompt("You are a helpful assistant. Use the calculator for any arithmetic.")
        .with_max_iterations(4)
        .build()?;

    let resp = agent.run(Message::human("What is 12 * 8?")).await?;
    println!("{}", resp.content);
    println!("(iterations: {} messages)", resp.messages.len());
    Ok(())
}
