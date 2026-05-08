//! EntityMemory — buffers messages and extracts entities + their
//! contexts so the agent's seed surfaces a "Known entities:" preamble.

use cognis::prelude::*;
use cognis::EntityMemory;

fn main() {
    let mut mem = EntityMemory::new();
    mem.write(Message::human("Ada writes Rust at Mozilla."));
    mem.write(Message::human("Bob reviews Ada's PRs at Cloudflare."));

    println!("=== entities ===");
    let mut keys: Vec<_> = mem.entities().keys().collect();
    keys.sort();
    for k in keys {
        println!("  {k}: {} mention(s)", mem.entities()[k].len());
    }

    println!("\n=== seed (what the agent sees on next turn) ===");
    for m in mem.seed() {
        let role = match &m {
            Message::System(_) => "system",
            Message::Human(_) => "human",
            Message::Ai(_) => "ai",
            Message::Tool(_) => "tool",
        };
        println!("[{role}] {}", m.content().replace('\n', " | "));
    }
}
