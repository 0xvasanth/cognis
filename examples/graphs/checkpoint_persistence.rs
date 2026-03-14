//! Checkpoint Persistence Example
//!
//! Demonstrates saving, restoring, and rolling back graph execution state
//! using the in-memory checkpoint system.
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example checkpoint_persistence`

#[path = "../shared.rs"]
mod shared;

use std::collections::HashMap;

use cognisgraph::checkpoint::persistence::{
    CheckpointFilter, CheckpointId, CheckpointManager, InMemoryPersistentSaver,
    PersistentCheckpoint,
};
use serde_json::json;

fn main() {
    println!("=== Checkpoint Persistence Example ===\n");

    // --- Set up a CheckpointManager with in-memory storage ---
    let saver = InMemoryPersistentSaver::new();
    let mut manager = CheckpointManager::new(Box::new(saver));

    // --- Create a chain of checkpoints simulating a conversation ---
    let mut meta = HashMap::new();
    meta.insert("node".to_string(), json!("__start__"));
    let cp0_id = manager
        .create_checkpoint("conv-1", json!({"messages": []}), None, meta)
        .unwrap();
    println!("Created root checkpoint: {}", cp0_id);

    let mut meta1 = HashMap::new();
    meta1.insert("node".to_string(), json!("agent"));
    let cp1_id = manager
        .create_checkpoint(
            "conv-1",
            json!({"messages": ["Hello, how can I help?"]}),
            Some(cp0_id.clone()),
            meta1,
        )
        .unwrap();
    println!("Created step 1: {}", cp1_id);

    let mut meta2 = HashMap::new();
    meta2.insert("node".to_string(), json!("tools"));
    let cp2_id = manager
        .create_checkpoint(
            "conv-1",
            json!({"messages": ["Hello, how can I help?", "Search results: ..."]}),
            Some(cp1_id.clone()),
            meta2,
        )
        .unwrap();
    println!("Created step 2: {}", cp2_id);

    // --- Restore the latest checkpoint ---
    let latest = manager.restore_latest("conv-1").unwrap().unwrap();
    println!(
        "\nLatest checkpoint: step={}, state={}",
        latest.step, latest.state
    );

    // --- Restore a specific checkpoint by ID ---
    let restored = manager.restore(&cp1_id).unwrap().unwrap();
    println!("Restored step 1:  state={}", restored.state);

    // --- View lineage (ancestry chain) ---
    let lineage = manager.history().get_lineage(&cp2_id);
    println!("\nLineage to step 2 ({} ancestors):", lineage.len());
    for id in &lineage {
        println!("  {}", id);
    }

    // --- Query with filters ---
    let filter = CheckpointFilter::new()
        .with_thread("conv-1")
        .with_step_range(0, 1);
    let results = manager.query(&filter).unwrap();
    println!("\nCheckpoints in steps 0..=1: {}", results.len());

    // --- Rollback to an earlier step ---
    let rolled_back = manager.rollback("conv-1", 0).unwrap().unwrap();
    println!("Rolled back to step 0: state={}", rolled_back.state);

    // --- Summary stats ---
    let stats = manager.stats();
    println!(
        "\nStats: {} checkpoints, {} threads, {:.1} avg steps/thread",
        stats.total_checkpoints, stats.threads, stats.avg_steps_per_thread
    );

    println!("\n=== Done ===");
}
