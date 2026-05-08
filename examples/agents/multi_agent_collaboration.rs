//! Multi-agent collaboration — V2's MultiAgentOrchestrator with
//! Sequential, Supervisor, and ParallelVote handoff strategies.

use cognis::prelude::*;
use cognis::{AgentBuilder, MultiAgentOrchestrator, ParallelVote, Sequential};
use cognis_llm::Client;

fn make_agent(prompt: &str) -> Result<cognis::Agent> {
    AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_system_prompt(prompt)
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }

    println!("--- Sequential ---");
    let orch = MultiAgentOrchestrator::new(Sequential)
        .add(
            "brainstormer",
            make_agent("Generate 3 ideas for a kids' birthday party.")?,
        )
        .add(
            "editor",
            make_agent("Pick the best idea and explain why in one sentence.")?,
        );
    println!("{}\n", orch.run("Plan a kids' party.").await?.content);

    println!("--- ParallelVote ---");
    let orch2 = MultiAgentOrchestrator::new(ParallelVote)
        .add("a", make_agent("Reply with the single word: yes.")?)
        .add("b", make_agent("Reply with the single word: yes.")?)
        .add("c", make_agent("Reply with the single word: no.")?);
    println!(
        "majority answer: {}",
        orch2.run("Pick yes or no.").await?.content
    );
    Ok(())
}
