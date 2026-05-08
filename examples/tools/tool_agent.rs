//! AgentBuilder + Calculator tool. Agent loops until the model produces
//! a final answer or hits `max_iterations`.

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
        .with_system_prompt("Use the calculator for any arithmetic. Always state the final answer.")
        .with_max_iterations(4)
        .build()?;

    let resp = agent.run(Message::human("What is 47 * 23?")).await?;
    println!("{}", resp.content);
    println!("(messages: {})", resp.messages.len());
    Ok(())
}
