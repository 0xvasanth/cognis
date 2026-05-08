//! What you'll learn:
//!   How `KnowledgeGraphMemory` extracts (subject, predicate, object)
//!   triples from messages and surfaces a deduped "Knowledge:"
//!   preamble the agent can read on every turn.
//!
//! Why this matters:
//!   For agents that need to remember *facts* (not just transcripts),
//!   triples give you a structured, queryable substrate that grows
//!   without bloating the context window — and survives memory
//!   compaction unchanged. Useful any time the user is teaching the
//!   bot about a domain (their projects, their team, their codebase).
//!
//! Scenario:
//!   A bot that learns about an internal project as the user
//!   describes it ("Project Atlas is led by Maya. Maya started in
//!   2024."). The triples accumulate; later, the user asks "Who leads
//!   Project Atlas?" and the agent answers from the knowledge graph
//!   it built during the earlier turns.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example memory_knowledge_graph
//!
//! Sample output (against ollama / llama3.1):
//!   === triples extracted ===
//!     (Project Atlas, is, led by Maya)
//!     (Maya, is, a staff engineer)
//!     (Atlas, is, written in Rust)
//!     (Rust, is, a systems language)
//!
//!   Q: Who leads Project Atlas?
//!   A: According to the knowledge base, Project Atlas is led by Maya. 
//!
//!   Answer: Maya.

use cognis::prelude::*;
use cognis::{KnowledgeGraphMemory, Memory};

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;
    let mut mem = KnowledgeGraphMemory::new()
        .with_system("You are a project knowledge base. Answer from known facts only.");

    // Teaching turns. The default extractor handles "X is Y" / "X are Y"
    // patterns — good enough to demo the storage shape without an LLM
    // extractor. Plug in `with_extractor(...)` for production NER.
    for fact in [
        "Project Atlas is led by Maya.",
        "Maya is a staff engineer.",
        "Atlas is written in Rust.",
        "Rust is a systems language.",
    ] {
        mem.write(Message::human(fact.to_string()));
    }

    println!("=== triples extracted ===");
    for (s, p, o) in mem.triples() {
        println!("  ({s}, {p}, {o})");
    }

    // Recall turn: build the agent's seed and append the question. The
    // KG memory injects a "Knowledge:" system message containing every
    // triple, so the model has the facts it needs without the original
    // transcript being in context.
    let mut seed = mem.seed();
    seed.push(Message::human("Who leads Project Atlas?"));

    let reply = client.invoke(seed).await?;
    println!("\nQ: Who leads Project Atlas?");
    println!("A: {}", reply.content().trim());
    Ok(())
}
