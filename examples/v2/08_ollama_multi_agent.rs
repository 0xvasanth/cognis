//! What you'll learn:
//!   How to chain two specialised agents — a planner and an executor —
//!   so the planner's reply becomes the executor's input.
//!
//! Why this matters:
//!   Real agent products almost never use a single prompt; they use a
//!   small team of focused agents. `MultiAgentOrchestrator` gives you
//!   `Sequential`, `ParallelVote`, and `Supervisor` strategies without
//!   you hand-rolling the message plumbing each time.
//!
//! Scenario:
//!   Help a user prep for tomorrow's standup. A planner agent breaks
//!   the request into 3 numbered steps, then hands off to an executor
//!   agent that turns the plan into a one-paragraph action summary.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example 08_ollama_multi_agent
//!
//! Sample output (against ollama / llama3.1):
//!   --- final agent reply ---
//!   To execute these steps, I would first spend some time reviewing my task list from last week, checking off completed tasks and making note of any unfinished ones that need attention. This will help me get a clear picture of what needs to be carried over into the current week. Next, I would take a few minutes to review each ongoing project, jotting down any updates or questions that have come up since last week's check-in. These notes will serve as a starting point for discussions with team members and stakeholders in the upcoming day/week. Finally, I would dedicate some time to reflecting on goals or objectives for the next day/week, breaking them down into concrete tasks and priorities where possible, and making sure they align with overall project objectives.

use cognis::prelude::*;
use cognis::{MultiAgentOrchestrator, Sequential};

#[tokio::main]
async fn main() -> Result<()> {
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
