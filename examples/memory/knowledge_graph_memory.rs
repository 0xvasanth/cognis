//! KnowledgeGraphMemory — extracts (subject, predicate, object) triples
//! from messages and surfaces them as a "Knowledge:" system preamble.

use cognis::prelude::*;
use cognis::KnowledgeGraphMemory;

fn main() {
    let mut mem = KnowledgeGraphMemory::new();
    mem.write(Message::human(
        "Cognis is a Rust framework. Tokio is async.",
    ));
    mem.write(Message::human("cognis-rag has embeddings."));
    mem.write(Message::human("Cognis is a Rust framework.")); // dup → no-op

    println!("=== triples (deduped) ===");
    for (s, p, o) in mem.triples() {
        println!("  ({s}, {p}, {o})");
    }

    println!("\n=== seed ===");
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
