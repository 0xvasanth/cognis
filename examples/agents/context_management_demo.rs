//! Token-budgeted context window via TokenBufferMemory.

use cognis::prelude::*;
use cognis::TokenBufferMemory;
use cognis_core::message::TrimStrategy;

#[tokio::main]
async fn main() -> Result<()> {
    let mut mem = TokenBufferMemory::new(40)
        .with_system("You are a helpful assistant.")
        .with_strategy(TrimStrategy::First);

    for s in [
        "hi",
        "what's the weather",
        "tell me a long story please",
        "no actually shorter",
    ] {
        mem.write(Message::human(s.to_string()));
    }
    let seed = mem.seed();
    println!("kept {} messages within 40-char budget:", seed.len());
    for m in &seed {
        println!("  - {}: {}", role_of(m), m.content());
    }
    Ok(())
}

fn role_of(m: &Message) -> &'static str {
    match m {
        Message::Human(_) => "human",
        Message::Ai(_) => "ai",
        Message::System(_) => "system",
        Message::Tool(_) => "tool",
    }
}
