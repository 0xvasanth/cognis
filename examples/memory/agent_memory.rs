//! Agent Memory Example
//!
//! Demonstrates storing and retrieving memories across conversations using
//! the MemoryManager, which combines short-term (bounded, LRU) and long-term
//! (persistent, importance-filtered) memory tiers.
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example agent_memory`

#[path = "../shared.rs"]
mod shared;

use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::Message;
use cognisagent::memory::{MemoryCategory, MemoryManager};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Agent Memory Example ===\n");

    // Create a memory manager with short-term capacity of 5
    let mut memory = MemoryManager::new(5);

    // -- Conversation 1: Learn user preferences --
    println!("-- Conversation 1: Learning about the user --");

    memory.remember(
        "user_name",
        serde_json::json!("Alice"),
        MemoryCategory::Fact,
        0.9,
    );
    memory.remember(
        "language",
        serde_json::json!("Rust"),
        MemoryCategory::Preference,
        0.8,
    );
    memory.remember(
        "task",
        serde_json::json!("build an LLM framework"),
        MemoryCategory::Context,
        0.6,
    );
    memory.remember(
        "style",
        serde_json::json!("concise"),
        MemoryCategory::Preference,
        0.4,
    );

    println!(
        "Stored 4 memories (short-term: {}, long-term: {})",
        memory.short_term_len(),
        memory.long_term_len()
    );
    println!("  High-importance entries auto-promoted to long-term memory\n");

    // -- Conversation 2: Recall what we know --
    println!("-- Conversation 2: Recalling user context --");

    if let Some(entry) = memory.recall("user_name") {
        println!("  Name: {}", entry.value);
    }
    if let Some(entry) = memory.recall("language") {
        println!("  Preferred language: {}", entry.value);
    }
    if let Some(entry) = memory.recall("task") {
        println!("  Current task: {}", entry.value);
    }

    // Search across all memory tiers
    let results = memory.search("rust");
    println!("  Search 'rust': {} match(es)", results.len());

    // -- Promote and forget --
    println!("\n-- Managing memory lifecycle --");

    // Promote important context to long-term
    if memory.promote("task").is_ok() {
        println!("  Promoted 'task' to long-term memory");
    }

    // Forget ephemeral entries
    memory.forget("style");
    println!("  Forgot 'style' (low-importance, no longer needed)");

    // Check stats
    let stats = memory.stats();
    println!(
        "  Final state: {} short-term, {} long-term, {} total accesses",
        stats.short_term_count, stats.long_term_count, stats.total_accesses
    );

    // -- LLM-powered memory --
    println!("\n-- Storing an LLM-generated fact --");

    let model = shared::get_chat_model(vec![
        "Rust was first released in 2010 by Graydon Hoare at Mozilla.".into(),
    ]);

    let messages = vec![Message::human(
        "Tell me a key fact about Rust's history in one sentence.",
    )];
    let result = model._generate(&messages, None).await?;

    if let Some(gen) = result.generations.first() {
        let fact_text = gen.message.content().text();
        println!("  LLM fact: {}", fact_text);

        memory.remember(
            "rust_history",
            serde_json::json!(fact_text),
            MemoryCategory::Fact,
            0.85,
        );

        if let Some(entry) = memory.recall("rust_history") {
            println!("  Recalled from memory: {}", entry.value);
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
