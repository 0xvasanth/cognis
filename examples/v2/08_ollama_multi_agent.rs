//! Multi-agent handoff with Ollama. Builds two agents — a "planner"
//! and an "executor" — and runs them with the `Sequential` strategy.
//! The planner's reply becomes the executor's input.
//!
//! Usage:
//! ```bash
//! COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.2:1b \
//!   cargo run --example 08_ollama_multi_agent -p cognis-examples
//! ```

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
        .with_system_prompt(
            "You are a planner. Break the user's request into 3 short, \
             numbered steps. No explanations — just the steps.",
        )
        .build()?;

    let executor = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_system_prompt(
            "You are an executor. You receive a numbered plan. Reply with \
             a one-paragraph summary of how you would carry it out.",
        )
        .build()?;

    let orch = MultiAgentOrchestrator::new(Sequential)
        .add("planner", planner)
        .add("executor", executor);

    let resp = orch
        .run("Help me prepare for a 5-minute team standup tomorrow.")
        .await?;
    println!("--- final agent reply ---");
    println!("{}", resp.content);
    Ok(())
}
