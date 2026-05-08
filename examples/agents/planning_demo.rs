//! Planning middleware — V2 has cognis::Planning that prepends a
//! plan-then-act system fragment.

use cognis::prelude::*;
use cognis::{AgentBuilder, MiddlewarePipeline, Planning};
use cognis_llm::Client;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let raw = Client::from_env()?;
    let pipe = MiddlewarePipeline::new().push(Planning::new()).build(raw);
    println!("pipelined client name: {}", pipe.client().provider().name());

    let mut agent = AgentBuilder::new()
        .with_llm(pipe.client().clone())
        .build()?;
    let r = agent.run(Message::human("Plan how to make tea.")).await?;
    println!("{}", r.content);
    Ok(())
}
