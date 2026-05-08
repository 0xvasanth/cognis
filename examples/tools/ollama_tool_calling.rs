//! End-to-end Ollama tool calling via the V2 ReAct agent. Wires the
//! Calculator tool into a `cognis::AgentBuilder` and lets the model
//! decide when to invoke it.

use std::sync::Arc;

use cognis::prelude::*;
use cognis::{AgentBuilder, Calculator};
use cognis_llm::Client;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let mut agent = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_tool(Arc::new(Calculator::new()))
        .with_system_prompt("Use `calculator` for arithmetic. Show the final answer.")
        .with_max_iterations(4)
        .build()?;

    let resp = agent
        .run(Message::human("compute 12 * 8 and explain."))
        .await?;
    println!("{}", resp.content);
    println!("(messages: {})", resp.messages.len());
    Ok(())
}
