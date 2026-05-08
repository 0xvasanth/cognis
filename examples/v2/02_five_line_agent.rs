//! What you'll learn:
//!   How to spin up a working LLM agent in five lines: pick a client,
//!   feed it to `AgentBuilder`, and call `run`.
//!
//! Why this matters:
//!   This is the smallest possible agent — no tools, no memory, no graph
//!   wiring. It's the shape your code takes when you want a single LLM
//!   round-trip behind the same `Agent` API you'll later layer tools and
//!   middleware onto.
//!
//! Scenario:
//!   Greet the user. The shortest possible agent — pure LLM round-trip
//!   via `AgentBuilder`, no tools, no memory. The shape you start from
//!   when wiring an agent into a new app.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example 02_five_line_agent
//!
//! Sample output (against ollama / llama3.1):
//!   Hello, how can I assist you today?

use cognis::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;
    let mut agent = AgentBuilder::new().with_llm(client).build()?;
    let resp = agent
        .run(Message::human("Say hello in one sentence."))
        .await?;
    println!("{}", resp.content);
    Ok(())
}
