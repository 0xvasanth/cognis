//! What you'll learn:
//!   How to keep an agent's history across turns by attaching a
//!   `Buffer` memory and flipping it into stateful mode.
//!
//! Why this matters:
//!   By default agents are stateless — every `run` is a clean slate.
//!   For a chat product you want the opposite: each turn sees the
//!   prior turns. `with_memory(...).stateful()` is the one-liner that
//!   gives you that, and the same `Memory` trait scales up to
//!   token-budgeted and summary-folding variants.
//!
//! Scenario:
//!   A user introduces themselves on turn 1 ("My name is Sam"), then
//!   on turn 2 asks "what's my name?". A stateful agent answers
//!   correctly; a fresh one wouldn't.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example agents_conversational
//!
//! Sample output (against ollama / llama3.1):
//!   user: Hi! My name is Sam.
//!   ai:   Nice to meet you, Sam! I'm happy to be chatting with you today. What's on your mind? Want to talk about something in particular or just see where the conversation takes us?
//!
//!   user: What's my name?
//!   ai:   Your name is Sam. How are you doing today, Sam?

use cognis::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let mut agent = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_memory(Buffer::new().with_system("You are a friendly chatbot."))
        .stateful()
        .build()?;

    for prompt in ["Hi! My name is Sam.", "What's my name?"] {
        let r = agent.run(Message::human(prompt)).await?;
        println!("user: {prompt}");
        println!("ai:   {}\n", r.content);
    }
    Ok(())
}
