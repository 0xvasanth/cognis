//! What you'll learn:
//!   How to combine several specialised agents under
//!   `MultiAgentOrchestrator` with two different strategies:
//!   `Sequential` (handoff) and `ParallelVote` (majority answer).
//!
//! Why this matters:
//!   Most production agent systems are a small team, not a single
//!   prompt — a brainstormer hands to an editor, three classifiers
//!   vote on a label. The orchestrator strategies abstract the
//!   message-routing so swapping topologies is a one-line change.
//!
//! Scenario:
//!   First, plan a kids' birthday party with a brainstormer-then-editor
//!   handoff (Sequential). Then, ask three classifier agents the same
//!   yes/no question and take the majority answer (ParallelVote).
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example agents_multi_agent
//!
//! Sample output (against ollama / llama3.1):
//!   --- Sequential ---
//!   The best idea is **Superhero Training Sessions**, where kids will participate in training sessions led by a "superhero instructor" (a party helper) and learn various superhero skills like jumping over obstacles, throwing foam balls, and using "super strength" to move heavy objects. This activity allows the kids to be actively engaged and creative while developing their physical and teamwork skills, making it an unforgettable experience for them.
//!
//!   --- ParallelVote ---
//!   majority answer: Yes.

use cognis::prelude::*;
use cognis::{MultiAgentOrchestrator, ParallelVote, Sequential};

fn make_agent(prompt: &str) -> Result<cognis::Agent> {
    AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_system_prompt(prompt)
        .build()
}

#[tokio::main]
async fn main() -> Result<()> {
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
