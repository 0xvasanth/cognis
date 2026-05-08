//! V2's Buffer (= V1's ConversationBufferMemory) keeps every message
//! in order; .seed() returns the full history (incl. pinned system).

use cognis::prelude::*;
use cognis::{Buffer, Memory};

fn main() {
    let mut mem = Buffer::new().with_system("Be terse.");
    mem.write(Message::human("hi"));
    mem.write(Message::ai("hello"));
    mem.write(Message::human("how are you"));

    let seed = mem.seed();
    println!("history ({} messages):", seed.len());
    for m in &seed {
        println!("  {}: {}", role(m), m.content());
    }
}

fn role(m: &Message) -> &'static str {
    match m {
        Message::Human(_) => "human",
        Message::Ai(_) => "ai",
        Message::System(_) => "system",
        Message::Tool(_) => "tool",
    }
}
