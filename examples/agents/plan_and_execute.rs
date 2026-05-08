//! Plan-then-execute = Sequential multi-agent (planner → executor).
//! V2 expresses this directly with MultiAgentOrchestrator + Sequential.

use cognis::prelude::*;
use cognis::{AgentBuilder, MultiAgentOrchestrator, Sequential};
use cognis_llm::Client;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let planner = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_system_prompt("Break the user task into 3 numbered steps. Steps only, no prose.")
        .build()?;
    let executor = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_system_prompt("You are given a numbered plan. Reply with one sentence summarizing how you'd execute it.")
        .build()?;

    let orch = MultiAgentOrchestrator::new(Sequential)
        .add("planner", planner)
        .add("executor", executor);

    let r = orch.run("Write a short blog post about Rust.").await?;
    println!("{}", r.content);
    Ok(())
}
