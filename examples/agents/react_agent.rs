//! ReAct-style agent: AgentBuilder + tools + Ollama.
//!
//! Run with: COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.2:1b cargo run -p cognis-examples --example agents_react_agent

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
        .with_system_prompt("You are a helpful assistant. Use the calculator for any arithmetic.")
        .with_max_iterations(4)
        .build()?;

    let resp = agent.run(Message::human("What is 12 * 8?")).await?;
    println!("{}", resp.content);
    println!("(iterations: {} messages)", resp.messages.len());
    Ok(())
}
