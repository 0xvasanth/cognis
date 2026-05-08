//! Streaming structured events from an agent run via Agent::stream.

use cognis::prelude::*;
use cognis::AgentBuilder;
use cognis_llm::Client;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let mut agent = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_system_prompt("Answer in one sentence.")
        .build()?;

    let mut s = agent
        .stream(Message::human("What is the capital of France?"))
        .await?;
    while let Some(ev) = s.next().await {
        match ev {
            Event::OnNodeStart { node, step, .. } => println!("[start] step={step} node={node}"),
            Event::OnNodeEnd { node, step, .. } => println!("[end]   step={step} node={node}"),
            Event::OnEnd { .. } => println!("[done]"),
            _ => {}
        }
    }
    Ok(())
}
