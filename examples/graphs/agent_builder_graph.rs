//! AgentBuilder builds the standard tool-calling graph for you. This is
//! the high-level surface; `examples/graphs/message_graph.rs` shows the
//! hand-built equivalent.

use std::sync::Arc;

use cognis::prelude::*;
use cognis::{AgentBuilder, Calculator};
use cognis_llm::Client;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let mut a = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_tool(Arc::new(Calculator::new()))
        .with_max_iterations(3)
        .build()?;
    let r = a.run(Message::human("what is 7 + 5?")).await?;
    println!("{}", r.content);
    Ok(())
}
