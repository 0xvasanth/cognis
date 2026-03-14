//! Agent Memory Example
//!
//! Demonstrates the DeepAgents memory system for managing agent memory across
//! different timescales and categories. The memory system provides short-term
//! (bounded, LRU-evicting), long-term (persistent-style), and indexed memory
//! with a unified manager interface.
//!
//! Features shown:
//! - MemoryEntry creation with categories and importance
//! - ShortTermMemory with capacity and eviction
//! - LongTermMemory storage and by_importance filtering
//! - MemoryIndex for fast text search
//! - MemoryManager unified interface
//! - Auto-promotion of high-importance entries
//! - Recall across memory tiers
//! - MemoryStats computation
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example agent_memory`

mod shared;
use cognis_core::messages::Message;
use cognisagent::memory::{
    LongTermMemory, MemoryCategory, MemoryEntry, MemoryIndex, MemoryManager, ShortTermMemory,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Agent Memory Example ===\n");

    // -----------------------------------------------------------------------
    // 1. MemoryEntry Creation with Categories and Importance
    // -----------------------------------------------------------------------
    println!("--- 1. MemoryEntry Creation ---");
    println!("Memory entries hold a key, value, category, importance, and access metadata.\n");

    let fact = MemoryEntry::builder("user_name", serde_json::json!("Alice"))
        .category(MemoryCategory::Fact)
        .importance(0.9)
        .build();

    println!(
        "  Entry: key={}, category={}, importance={:.1}",
        fact.key, fact.category, fact.importance
    );
    println!("  Value: {}", fact.value);
    println!(
        "  Access count: {}, created_at: {}",
        fact.access_count, fact.created_at
    );

    let pref = MemoryEntry::builder("color_pref", serde_json::json!("blue"))
        .category(MemoryCategory::Preference)
        .importance(0.4)
        .build();

    println!(
        "\n  Preference entry: key={}, category={}, importance={:.1}",
        pref.key, pref.category, pref.importance
    );

    // Default category is Context, default importance is 0.5
    let default_entry = MemoryEntry::builder("temp", serde_json::json!("data")).build();
    println!(
        "  Default entry: category={}, importance={:.1}",
        default_entry.category, default_entry.importance
    );

    // Importance is clamped to [0.0, 1.0]
    let clamped = MemoryEntry::builder("high", serde_json::json!(true))
        .importance(5.0)
        .build();
    println!("  Clamped importance (5.0 -> {:.1})", clamped.importance);

    // Custom category
    let custom = MemoryEntry::builder("task_id", serde_json::json!("T-123"))
        .category(MemoryCategory::Custom("project".to_string()))
        .build();
    println!("  Custom category: {}", custom.category);

    // Serialize to JSON
    let json = fact.to_json();
    println!("\n  Fact entry as JSON:");
    println!("  {}", serde_json::to_string_pretty(&json).unwrap());

    // -----------------------------------------------------------------------
    // 2. ShortTermMemory with Capacity and Eviction
    // -----------------------------------------------------------------------
    println!("\n--- 2. ShortTermMemory ---");
    println!("Bounded memory store that evicts least-recently-accessed entries.\n");

    let mut stm = ShortTermMemory::new(3);
    println!("  Created ShortTermMemory with capacity=3");

    stm.store(
        "fact_1",
        serde_json::json!("The sky is blue"),
        MemoryCategory::Fact,
    );
    stm.store(
        "fact_2",
        serde_json::json!("Water is H2O"),
        MemoryCategory::Fact,
    );
    stm.store(
        "pref_1",
        serde_json::json!("dark mode"),
        MemoryCategory::Preference,
    );
    println!("  Stored 3 entries: len={}", stm.len());

    // Access fact_1 to increase its access count
    if let Some(entry) = stm.get_mut("fact_1") {
        entry.touch();
        println!("  Touched 'fact_1': access_count={}", entry.access_count);
    }

    // Store a 4th entry, triggering eviction of least-accessed
    stm.store(
        "ctx_1",
        serde_json::json!("current task: demo"),
        MemoryCategory::Context,
    );
    println!("  Stored 4th entry (triggers eviction): len={}", stm.len());

    // fact_1 was touched, so it survived; one of the untouched entries was evicted
    println!("  'fact_1' survived: {}", stm.get("fact_1").is_some());

    // Filter by category
    let facts = stm.by_category(&MemoryCategory::Fact);
    println!("  Fact entries remaining: {}", facts.len());

    // Search
    let results = stm.search("sky");
    println!("  Search 'sky': {} result(s)", results.len());
    for r in &results {
        println!("    key={}, value={}", r.key, r.value);
    }

    // -----------------------------------------------------------------------
    // 3. LongTermMemory Storage and Filtering
    // -----------------------------------------------------------------------
    println!("\n--- 3. LongTermMemory ---");
    println!("Unbounded persistent-style memory with importance-based filtering.\n");

    let mut ltm = LongTermMemory::new();

    ltm.store(
        MemoryEntry::builder("user_name", serde_json::json!("Alice"))
            .category(MemoryCategory::Fact)
            .importance(0.9)
            .build(),
    );
    ltm.store(
        MemoryEntry::builder("api_key_hint", serde_json::json!("sk-...xyz"))
            .category(MemoryCategory::Instruction)
            .importance(0.8)
            .build(),
    );
    ltm.store(
        MemoryEntry::builder("greeting_style", serde_json::json!("casual"))
            .category(MemoryCategory::Preference)
            .importance(0.3)
            .build(),
    );
    ltm.store(
        MemoryEntry::builder(
            "project_goal",
            serde_json::json!("Build an LLM framework in Rust"),
        )
        .category(MemoryCategory::Context)
        .importance(0.95)
        .build(),
    );

    println!("  Stored {} entries in long-term memory", ltm.len());

    // Filter by importance
    let high_importance = ltm.by_importance(0.8);
    println!(
        "  Entries with importance >= 0.8: {}",
        high_importance.len()
    );
    for entry in &high_importance {
        println!(
            "    key={}, importance={:.2}, category={}",
            entry.key, entry.importance, entry.category
        );
    }

    let low_importance = ltm.by_importance(0.0);
    println!(
        "  Entries with importance >= 0.0 (all): {}",
        low_importance.len()
    );

    // Search
    let results = ltm.search("rust");
    println!("  Search 'rust': {} result(s)", results.len());
    for r in &results {
        println!("    key={}, value={}", r.key, r.value);
    }

    // Filter by category
    let instructions = ltm.by_category(&MemoryCategory::Instruction);
    println!("  Instruction entries: {}", instructions.len());

    // Serialize to JSON
    let ltm_json = ltm.to_json();
    println!("  LTM JSON count field: {}", ltm_json["count"]);

    // -----------------------------------------------------------------------
    // 4. MemoryIndex for Fast Text Search
    // -----------------------------------------------------------------------
    println!("\n--- 4. MemoryIndex ---");
    println!("Inverted index for term-frequency-based text search across memory keys.\n");

    let mut index = MemoryIndex::new();

    index.add("doc1", "Rust is a systems programming language");
    index.add("doc2", "Python is great for machine learning");
    index.add("doc3", "Rust and Python are both popular languages");
    index.add("doc4", "The Rust borrow checker ensures memory safety");

    println!(
        "  Indexed 4 documents, {} distinct terms",
        index.term_count()
    );

    // Search for terms
    let results = index.search("Rust");
    println!("  Search 'Rust': {:?}", results);

    let results = index.search("Python learning");
    println!("  Search 'Python learning': {:?}", results);

    let results = index.search("nonexistent");
    println!("  Search 'nonexistent': {:?} (empty)", results);

    // Remove a document and search again
    index.remove("doc1");
    let results = index.search("systems programming");
    println!(
        "  After removing 'doc1', search 'systems programming': {:?}",
        results
    );

    // -----------------------------------------------------------------------
    // 5. MemoryManager Unified Interface
    // -----------------------------------------------------------------------
    println!("\n--- 5. MemoryManager ---");
    println!("Unified interface combining short-term and long-term memory.\n");

    let mut manager = MemoryManager::new(5); // short-term capacity of 5
    println!("  Created MemoryManager with short-term capacity=5");

    // Store entries with varying importance
    manager.remember(
        "user_name",
        serde_json::json!("Bob"),
        MemoryCategory::Fact,
        0.9,
    );
    manager.remember(
        "session_id",
        serde_json::json!("sess_abc"),
        MemoryCategory::Context,
        0.3,
    );
    manager.remember(
        "preference",
        serde_json::json!("verbose output"),
        MemoryCategory::Preference,
        0.5,
    );

    println!("  Stored 3 entries:");
    println!("    Short-term: {}", manager.short_term_len());
    println!("    Long-term: {}", manager.long_term_len());

    // -----------------------------------------------------------------------
    // 6. Auto-Promotion of High-Importance Entries
    // -----------------------------------------------------------------------
    println!("\n--- 6. Auto-Promotion ---");
    println!("Entries with importance >= 0.7 are automatically stored in long-term memory.\n");

    // 'user_name' had importance 0.9, so it should be in both stores
    println!(
        "  'user_name' (importance=0.9) in long-term: {}",
        manager.long_term_len() > 0
    );

    // Store another high-importance entry
    manager.remember(
        "critical_instruction",
        serde_json::json!("Always confirm before deleting"),
        MemoryCategory::Instruction,
        0.85,
    );
    println!("  After storing 'critical_instruction' (importance=0.85):");
    println!("    Short-term: {}", manager.short_term_len());
    println!("    Long-term: {}", manager.long_term_len());

    // Low importance entry stays only in short-term
    manager.remember(
        "temp_note",
        serde_json::json!("check later"),
        MemoryCategory::Context,
        0.2,
    );
    println!("  After storing 'temp_note' (importance=0.2):");
    println!("    Short-term: {}", manager.short_term_len());
    println!("    Long-term: {} (unchanged)", manager.long_term_len());

    // -----------------------------------------------------------------------
    // 7. Recall Across Memory Tiers
    // -----------------------------------------------------------------------
    println!("\n--- 7. Recall Across Tiers ---");
    println!("Recall checks short-term first, then falls back to long-term.\n");

    // Recall from short-term
    if let Some(entry) = manager.recall("session_id") {
        println!(
            "  Recalled 'session_id': value={}, category={}",
            entry.value, entry.category
        );
    }

    // Recall from long-term (user_name is in both, but short-term is checked first)
    if let Some(entry) = manager.recall("user_name") {
        println!(
            "  Recalled 'user_name': value={}, category={}",
            entry.value, entry.category
        );
    }

    // Search across both tiers
    let results = manager.search("bob");
    println!("  Search 'bob': {} result(s)", results.len());
    for r in &results {
        println!("    key={}, value={}", r.key, r.value);
    }

    // Explicit promotion from short-term to long-term
    match manager.promote("session_id") {
        Ok(()) => println!("  Promoted 'session_id' to long-term"),
        Err(e) => println!("  Promotion error: {}", e),
    }
    println!(
        "    Short-term: {}, Long-term: {}",
        manager.short_term_len(),
        manager.long_term_len()
    );

    // Promoting again fails (it was moved out of short-term)
    match manager.promote("session_id") {
        Ok(()) => println!("  Unexpected success"),
        Err(e) => println!("  Re-promote error (expected): {}", e),
    }

    // Still recallable from long-term
    if let Some(entry) = manager.recall("session_id") {
        println!(
            "  'session_id' still recallable from long-term: {}",
            entry.value
        );
    }

    // Forget an entry from all tiers
    manager.forget("temp_note");
    println!(
        "  Forgot 'temp_note': recall={:?}",
        manager.recall("temp_note").map(|e| &e.key)
    );

    // -----------------------------------------------------------------------
    // 8. MemoryStats Computation
    // -----------------------------------------------------------------------
    println!("\n--- 8. MemoryStats ---");
    println!("Compute aggregate statistics about the memory system.\n");

    let stats = manager.stats();
    println!("  Short-term count: {}", stats.short_term_count);
    println!("  Long-term count: {}", stats.long_term_count);
    println!("  Total accesses: {}", stats.total_accesses);
    println!("  Categories:");
    let mut cats: Vec<_> = stats.categories.iter().collect();
    cats.sort_by_key(|(k, _)| (*k).clone());
    for (cat, count) in &cats {
        println!("    {}: {}", cat, count);
    }

    // Serialize stats to JSON
    let stats_json = stats.to_json();
    println!("\n  Stats as JSON:");
    println!("  {}", serde_json::to_string_pretty(&stats_json).unwrap());

    // --- Real LLM Demo ---
    println!("\n--- Real LLM Demo ---\n");
    println!("Using an LLM to generate a memory-worthy fact, then storing it.\n");

    let model = shared::get_chat_model(vec![
        "The Rust programming language was first released in 2010 by Graydon Hoare at Mozilla.".into(),
    ]);
    let messages = vec![Message::human("Tell me a key fact about Rust's history in one sentence.")];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        let fact_text = gen.message.content().text();
        println!("LLM generated fact: {}", fact_text);

        // Store the LLM-generated fact in the memory manager
        manager.remember(
            "llm_rust_fact",
            serde_json::json!(fact_text),
            MemoryCategory::Fact,
            0.85,
        );
        println!("Stored in memory manager as 'llm_rust_fact'");

        if let Some(entry) = manager.recall("llm_rust_fact") {
            println!("Recalled: key={}, value={}", entry.key, entry.value);
        }
    }

    println!("\n=== Agent Memory Example Complete ===");
    Ok(())
}
