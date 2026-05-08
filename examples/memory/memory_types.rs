//! What you'll learn:
//!   How `Buffer`, `Window`, and `TokenBufferMemory` shape what an agent
//!   sees when a conversation grows long, and the trade-off each one
//!   makes between recall, cost, and latency.
//!
//! Why this matters:
//!   You're building a chatbot. As the user keeps talking, the prompt
//!   you send to the model grows turn by turn. Eventually you need to
//!   decide: keep everything (best recall, runaway cost), keep the last
//!   N (bounded cost, drops old facts), or fit a token budget (bounded
//!   cost, smarter eviction). All three implement `Memory` and slot
//!   into `AgentBuilder::with_memory` — pick the one that matches your
//!   tolerance for forgotten context.
//!
//! Scenario:
//!   A user introduces themselves, plans a trip, then asks the agent
//!   to recall their name and destination from the very first turn.
//!   We run the *same* conversation through three agents — one per
//!   memory variant — and watch which ones still remember.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example memory_types
//!
//! Sample output (against ollama / llama3.1):
//!   (one block per memory variant; first 4 turns elided for brevity)
//!
//!   === Buffer (keeps everything) ===
//!   USER: Quick check — what's my name and where am I going?
//!   AI:   Your name is Maya and you're heading to Lisbon, Portugal for a
//!         five-day food adventure!
//!
//!   === Window(2) (last 2 turns only) ===
//!   USER: Quick check — what's my name and where am I going?
//!   AI:   You're heading to Lisbon, Portugal!
//!         (knows the destination from turn 4, lost the name from turn 1)
//!
//!   === TokenBufferMemory(200) (token-budget eviction) ===
//!   USER: Quick check — what's my name and where am I going?
//!   AI:   I don't have any information about your name or destination yet,
//!         but I'd be happy to help you plan a trip if you tell me more!
//!         (older turns evicted to fit the token budget)

use cognis::prelude::*;
use cognis::{AgentBuilder, Buffer, TokenBufferMemory, Window};

/// Five-turn conversation. The last user message is a recall test —
/// only memories that retained turn 1 can answer it correctly.
const TURNS: &[&str] = &[
    "Hi! My name is Maya, and I'm planning a trip.",
    "I'm going to Lisbon next month.",
    "I'll be there for five days, mostly to eat.",
    "Any neighborhoods I should focus on?",
    "Quick check — what's my name and where am I going?",
];

async fn replay(label: &str, memory: impl Memory + 'static) -> Result<()> {
    let mut agent = AgentBuilder::new()
        .with_llm(Client::from_env()?)
        .with_memory(memory)
        .stateful()
        .build()?;

    println!("\n=== {label} ===");
    for turn in TURNS {
        let resp = agent.run(Message::human(*turn)).await?;
        println!("USER: {turn}");
        println!("AI:   {}\n", resp.content.trim());
    }
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let prompt = "You are a friendly travel assistant. Reply in one sentence.";

    // Buffer keeps every message. Recall is perfect; cost grows with
    // conversation length. Use this for short sessions or when you've
    // already capped turns elsewhere.
    replay(
        "Buffer (keeps everything)",
        Buffer::new().with_system(prompt),
    )
    .await?;

    // Window(2) keeps only the last 2 turns. Cheap and predictable, but
    // anything older than two turns ago is gone — the recall test will
    // fail. Use this for FAQ-style bots where each question is
    // self-contained.
    replay(
        "Window(2) (last 2 turns only)",
        Window::new(2).with_system(prompt),
    )
    .await?;

    // TokenBufferMemory trims by token budget instead of message count,
    // so a few short turns stay in scope while a single huge turn might
    // get evicted. Use this when message lengths vary (a code paste vs.
    // "yes please") and you want to maximize what fits in your budget.
    replay(
        "TokenBufferMemory(200) (token-budget eviction)",
        TokenBufferMemory::new(200).with_system(prompt),
    )
    .await?;

    Ok(())
}
