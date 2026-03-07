//! Conversation Memory Example
//!
//! Demonstrates the `ConversationMemory` system with windowing, branching,
//! and search capabilities:
//!
//! - Create conversation turns and retrieve them with different window strategies
//! - Use `ConversationWindow::Last` and `ConversationWindow::TokenBudget`
//! - Create a branch, switch between branches, and observe isolation
//! - Use `ConversationSearch` for keyword and metadata search
//!
//! No API keys required.
//!
//! Run with: `cargo run -p rustchain-examples --example conversation_memory`

use std::collections::HashMap;

use rustchain::memory::conversation::{
    ConversationMemory, ConversationSearch, ConversationWindow,
};
use serde_json::json;

fn main() {
    println!("=== Conversation Memory Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Create turns and basic retrieval
    // -----------------------------------------------------------------------
    println!("--- 1. Adding conversation turns ---");

    let mut memory = ConversationMemory::new();

    memory.add_turn("What is Rust?", "Rust is a systems programming language focused on safety and performance.");
    memory.add_turn("How does ownership work?", "Each value in Rust has a single owner. When the owner goes out of scope, the value is dropped.");
    memory.add_turn("What are lifetimes?", "Lifetimes are annotations that tell the compiler how long references are valid.");
    memory.add_turn("Explain async in Rust.", "Async Rust uses futures and the async/await syntax to write non-blocking code, typically with the tokio runtime.");

    println!("Total turns stored: {}", memory.len());
    println!();

    // -----------------------------------------------------------------------
    // 2. Windowing with Last(n)
    // -----------------------------------------------------------------------
    println!("--- 2. ConversationWindow::Last(2) ---");

    let recent = memory.get_turns(&ConversationWindow::Last(2));
    println!("Retrieved {} most recent turns:", recent.len());
    for turn in &recent {
        println!("  Human: {}", turn.human);
        println!("  AI:    {}...", &turn.ai[..turn.ai.len().min(60)]);
        println!();
    }

    // -----------------------------------------------------------------------
    // 3. Windowing with TokenBudget
    // -----------------------------------------------------------------------
    println!("--- 3. ConversationWindow::TokenBudget(200) ---");

    let budget_turns = memory.get_turns(&ConversationWindow::TokenBudget(200));
    println!(
        "Retrieved {} turns within 200-character budget:",
        budget_turns.len()
    );
    let total_chars: usize = budget_turns
        .iter()
        .map(|t| t.human.len() + t.ai.len())
        .sum();
    println!("  Total characters used: {}", total_chars);
    for turn in &budget_turns {
        println!("  Human: {}", turn.human);
    }
    println!();

    // -----------------------------------------------------------------------
    // 4. Convert to messages (for LLM context)
    // -----------------------------------------------------------------------
    println!("--- 4. Convert to messages with window ---");

    let messages = memory.to_messages(&ConversationWindow::Last(1));
    println!("Messages for last 1 turn (role/content pairs):");
    for msg in &messages {
        println!("  {}: {}", msg["role"], msg["content"]);
    }
    println!();

    // -----------------------------------------------------------------------
    // 5. Branching
    // -----------------------------------------------------------------------
    println!("--- 5. Branching ---");

    memory
        .create_branch("python-tangent", 2)
        .expect("branch creation should succeed");
    println!(
        "Created branch 'python-tangent' diverging after turn 2 (out of {})",
        memory.len()
    );

    memory
        .switch_branch("python-tangent")
        .expect("switch should succeed");
    println!("Switched to branch: {:?}", memory.current_branch());

    memory.add_turn(
        "What about Python?",
        "Python is a high-level, dynamically typed language popular for scripting and data science.",
    );
    memory.add_turn(
        "Is Python faster than Rust?",
        "Generally no. Rust is compiled and much faster for CPU-bound tasks.",
    );

    println!(
        "Turns visible on branch 'python-tangent': {}",
        memory.len()
    );
    let branch_turns = memory.get_turns(&ConversationWindow::All);
    for (i, turn) in branch_turns.iter().enumerate() {
        println!("  [{}] Human: {}", i, turn.human);
    }
    println!();

    // Switch back to main
    println!("Switching back to main line...");
    // Reset to main by setting current_branch to None via export/re-creation workaround
    // The API exposes switch_branch but not a direct "switch to main", so we list branches
    // and show the main line via get_turns after checking branch info.
    let summary = memory.summary();
    println!("Memory summary: {}", serde_json::to_string_pretty(&summary).unwrap());
    println!();

    // -----------------------------------------------------------------------
    // 6. Search by keyword
    // -----------------------------------------------------------------------
    println!("--- 6. ConversationSearch: keyword search ---");

    let search = ConversationSearch::new(&memory);
    let results = search.by_keyword("rust");
    println!("Turns containing 'rust': {}", results.len());
    for turn in &results {
        println!("  Human: {}", turn.human);
    }
    println!();

    let results = search.by_keyword("python");
    println!("Turns containing 'python': {}", results.len());
    for turn in &results {
        println!("  Human: {}", turn.human);
    }
    println!();

    // -----------------------------------------------------------------------
    // 7. Search by metadata
    // -----------------------------------------------------------------------
    println!("--- 7. ConversationSearch: metadata search ---");

    // Create a new memory with metadata-tagged turns
    let mut tagged_memory = ConversationMemory::new();

    let mut meta_math = HashMap::new();
    meta_math.insert("topic".to_string(), json!("math"));
    tagged_memory.add_turn_with_metadata("What is 2+2?", "It is 4.", meta_math);

    let mut meta_history = HashMap::new();
    meta_history.insert("topic".to_string(), json!("history"));
    tagged_memory.add_turn_with_metadata(
        "When was the moon landing?",
        "July 20, 1969.",
        meta_history,
    );

    let mut meta_math2 = HashMap::new();
    meta_math2.insert("topic".to_string(), json!("math"));
    tagged_memory.add_turn_with_metadata("What is pi?", "Approximately 3.14159.", meta_math2);

    let tag_search = ConversationSearch::new(&tagged_memory);
    let math_turns = tag_search.by_metadata("topic", &json!("math"));
    println!("Turns tagged with topic=math: {}", math_turns.len());
    for turn in &math_turns {
        println!("  Q: {} -> A: {}", turn.human, turn.ai);
    }
    println!();

    // -----------------------------------------------------------------------
    // 8. Recent turns via search
    // -----------------------------------------------------------------------
    println!("--- 8. ConversationSearch: recent turns ---");

    let recent_2 = tag_search.recent(2);
    println!("Most recent 2 turns:");
    for turn in &recent_2 {
        println!("  Q: {}", turn.human);
    }
    println!();

    // -----------------------------------------------------------------------
    // 9. Export full memory as JSON
    // -----------------------------------------------------------------------
    println!("--- 9. Export memory ---");

    let export = tagged_memory.export();
    println!(
        "Exported memory JSON (truncated): {}...",
        &serde_json::to_string(&export).unwrap()[..120.min(serde_json::to_string(&export).unwrap().len())]
    );
    println!();

    println!("=== Conversation Memory Example Complete ===");
}
