//! Checkpoint Persistence Example
//!
//! Demonstrates the LangGraph checkpoint persistence layer for saving,
//! restoring, querying, and rolling back graph execution state.
//!
//! Features shown:
//! - CheckpointId creation and uniqueness
//! - PersistentCheckpoint building with metadata
//! - InMemoryPersistentSaver CRUD operations
//! - CheckpointHistory lineage tracking
//! - CheckpointFilter querying
//! - CheckpointManager create/restore/rollback
//! - Multi-thread checkpointing
//! - CheckpointStats summary
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example checkpoint_persistence`

mod shared;
use std::collections::HashMap;

use cognisgraph::checkpoint::persistence::{
    CheckpointFilter, CheckpointHistory, CheckpointId, CheckpointManager, InMemoryPersistentSaver,
    PersistentCheckpoint, PersistentCheckpointSaver,
};
use serde_json::json;

fn main() {
    println!("=== Checkpoint Persistence Example ===\n");

    // -----------------------------------------------------------------------
    // 1. CheckpointId Creation
    // -----------------------------------------------------------------------
    println!("--- 1. CheckpointId Creation ---");
    println!("Each checkpoint gets a unique UUID-based identifier.\n");

    let id_a = CheckpointId::new();
    let id_b = CheckpointId::new();
    println!("  id_a: {}", id_a);
    println!("  id_b: {}", id_b);
    println!("  Unique? {}", id_a != id_b);

    let known_id = CheckpointId::from_str("step-0-root");
    println!("  From string: {}", known_id);
    println!("  As str:      {}", known_id.as_str());

    let cloned = known_id.clone();
    println!("  Clone equal?  {}", known_id == cloned);

    // -----------------------------------------------------------------------
    // 2. PersistentCheckpoint Building with Metadata
    // -----------------------------------------------------------------------
    println!("\n--- 2. Building Checkpoints ---");
    println!("Use the builder pattern to construct checkpoints with metadata.\n");

    let root_id = CheckpointId::from_str("root-001");
    let root_cp = PersistentCheckpoint::builder("thread-main", json!({"counter": 0}))
        .id(root_id.clone())
        .step(0)
        .metadata("source", json!("init"))
        .metadata("node", json!("__start__"))
        .build();

    println!("  Thread:    {}", root_cp.thread_id);
    println!("  Step:      {}", root_cp.step);
    println!("  Is root?   {}", root_cp.is_root());
    println!("  State:     {}", root_cp.state);
    println!("  Metadata:  {:?}", root_cp.metadata);
    println!("  Created:   {}", root_cp.created_at);

    // Child checkpoint with parent linkage.
    let child_id = CheckpointId::from_str("child-001");
    let child_cp = PersistentCheckpoint::builder("thread-main", json!({"counter": 1}))
        .id(child_id.clone())
        .parent_id(root_id.clone())
        .step(1)
        .metadata("source", json!("agent_step"))
        .build();

    println!("\n  Child checkpoint:");
    println!(
        "    Parent:  {:?}",
        child_cp.parent_id.as_ref().map(|p| p.as_str())
    );
    println!("    Is root? {}", child_cp.is_root());

    // Serialize to JSON.
    let cp_json = child_cp.to_json();
    println!(
        "    JSON:    {}",
        serde_json::to_string_pretty(&cp_json).unwrap()
    );

    // -----------------------------------------------------------------------
    // 3. InMemoryPersistentSaver CRUD Operations
    // -----------------------------------------------------------------------
    println!("\n--- 3. InMemoryPersistentSaver CRUD ---");
    println!("Save, load, list, and delete checkpoints in memory.\n");

    let mut saver = InMemoryPersistentSaver::new();
    println!(
        "  Empty saver: len={}, is_empty={}",
        saver.len(),
        saver.is_empty()
    );

    // Save checkpoints.
    let cp0 = PersistentCheckpoint::builder("session-1", json!({"step": "init"}))
        .id(CheckpointId::from_str("s1-cp0"))
        .step(0)
        .build();
    let cp1 = PersistentCheckpoint::builder("session-1", json!({"step": "processed"}))
        .id(CheckpointId::from_str("s1-cp1"))
        .parent_id(CheckpointId::from_str("s1-cp0"))
        .step(1)
        .metadata("important", json!(true))
        .build();
    let cp2 = PersistentCheckpoint::builder("session-1", json!({"step": "final"}))
        .id(CheckpointId::from_str("s1-cp2"))
        .parent_id(CheckpointId::from_str("s1-cp1"))
        .step(2)
        .build();

    saver.save(cp0).unwrap();
    saver.save(cp1).unwrap();
    saver.save(cp2).unwrap();
    println!("  Saved 3 checkpoints. len={}", saver.len());

    // Load by ID.
    let loaded = saver.load(&CheckpointId::from_str("s1-cp1")).unwrap();
    println!("  Load 's1-cp1': state={}", loaded.as_ref().unwrap().state);

    // Load latest for thread.
    let latest = saver.load_latest("session-1").unwrap().unwrap();
    println!(
        "  Latest 'session-1': step={}, state={}",
        latest.step, latest.state
    );

    // List all for thread (ordered by step).
    let all = saver.list("session-1").unwrap();
    println!("  List 'session-1': {} checkpoints", all.len());
    for cp in &all {
        println!("    step={} id={}", cp.step, cp.id);
    }

    // Delete a checkpoint.
    let deleted = saver.delete(&CheckpointId::from_str("s1-cp1")).unwrap();
    println!("\n  Deleted 's1-cp1': {}", deleted);
    println!("  Saver len after delete: {}", saver.len());
    let missing = saver.load(&CheckpointId::from_str("s1-cp1")).unwrap();
    println!("  Load deleted: {:?}", missing.is_none());

    // Thread IDs.
    println!("  Thread IDs: {:?}", saver.thread_ids());

    // -----------------------------------------------------------------------
    // 4. CheckpointHistory Lineage Tracking
    // -----------------------------------------------------------------------
    println!("\n--- 4. CheckpointHistory Lineage ---");
    println!("Track parent-child relationships and query lineage.\n");

    let mut history = CheckpointHistory::new();

    // Build a chain: root -> step1 -> step2 -> step3.
    let h_root = PersistentCheckpoint::builder("t1", json!(null))
        .id(CheckpointId::from_str("h-root"))
        .step(0)
        .build();
    let h_step1 = PersistentCheckpoint::builder("t1", json!(null))
        .id(CheckpointId::from_str("h-step1"))
        .parent_id(CheckpointId::from_str("h-root"))
        .step(1)
        .build();
    let h_step2 = PersistentCheckpoint::builder("t1", json!(null))
        .id(CheckpointId::from_str("h-step2"))
        .parent_id(CheckpointId::from_str("h-step1"))
        .step(2)
        .build();
    let h_step3 = PersistentCheckpoint::builder("t1", json!(null))
        .id(CheckpointId::from_str("h-step3"))
        .parent_id(CheckpointId::from_str("h-step2"))
        .step(3)
        .build();

    history.add(&h_root);
    history.add(&h_step1);
    history.add(&h_step2);
    history.add(&h_step3);

    // Get lineage from root to h-step3.
    let lineage = history.get_lineage(&CheckpointId::from_str("h-step3"));
    println!("  Lineage to 'h-step3':");
    for (i, id) in lineage.iter().enumerate() {
        println!("    {}: {}", i, id);
    }

    // Depth.
    println!(
        "\n  Depth of 'h-step3': {}",
        history.depth(&CheckpointId::from_str("h-step3"))
    );
    println!(
        "  Depth of 'h-root':  {}",
        history.depth(&CheckpointId::from_str("h-root"))
    );

    // Children.
    let children = history.get_children(&CheckpointId::from_str("h-root"));
    println!(
        "  Children of 'h-root': {:?}",
        children.iter().map(|c| c.as_str()).collect::<Vec<_>>()
    );

    // Branching: add a second child off h-step1.
    let h_branch = PersistentCheckpoint::builder("t1", json!(null))
        .id(CheckpointId::from_str("h-branch"))
        .parent_id(CheckpointId::from_str("h-step1"))
        .step(2)
        .build();
    history.add(&h_branch);

    let step1_children = history.get_children(&CheckpointId::from_str("h-step1"));
    println!(
        "  Children of 'h-step1' (branched): {:?}",
        step1_children
            .iter()
            .map(|c| c.as_str())
            .collect::<Vec<_>>()
    );

    // -----------------------------------------------------------------------
    // 5. CheckpointFilter Querying
    // -----------------------------------------------------------------------
    println!("\n--- 5. CheckpointFilter Querying ---");
    println!("Build filters to query checkpoints by thread, step range, or metadata.\n");

    // Recreate a saver with varied checkpoints for filtering.
    let mut filter_saver = InMemoryPersistentSaver::new();
    for step in 0..5 {
        let mut builder = PersistentCheckpoint::builder("workflow-A", json!({"step": step}))
            .id(CheckpointId::from_str(&format!("wA-{}", step)))
            .step(step);
        if step % 2 == 0 {
            builder = builder.metadata("tagged", json!(true));
        }
        filter_saver.save(builder.build()).unwrap();
    }
    for step in 0..3 {
        let cp = PersistentCheckpoint::builder("workflow-B", json!({"step": step}))
            .id(CheckpointId::from_str(&format!("wB-{}", step)))
            .step(step)
            .build();
        filter_saver.save(cp).unwrap();
    }
    println!("  Created 8 checkpoints across 2 threads.");

    // Filter by thread.
    let thread_filter = CheckpointFilter::new().with_thread("workflow-A");
    let all_a = filter_saver.list("workflow-A").unwrap();
    let matched: Vec<_> = all_a
        .iter()
        .filter(|cp| thread_filter.matches(cp))
        .collect();
    println!("  Filter thread='workflow-A': {} matches", matched.len());

    // Filter by step range.
    let range_filter = CheckpointFilter::new()
        .with_thread("workflow-A")
        .with_step_range(1, 3);
    let range_matched: Vec<_> = all_a.iter().filter(|cp| range_filter.matches(cp)).collect();
    println!("  Filter steps 1..=3: {} matches", range_matched.len());
    for cp in &range_matched {
        println!("    step={} id={}", cp.step, cp.id);
    }

    // Filter by metadata key.
    let meta_filter = CheckpointFilter::new()
        .with_thread("workflow-A")
        .with_metadata_key("tagged");
    let meta_matched: Vec<_> = all_a.iter().filter(|cp| meta_filter.matches(cp)).collect();
    println!(
        "  Filter metadata_key='tagged': {} matches (steps 0, 2, 4)",
        meta_matched.len()
    );

    // -----------------------------------------------------------------------
    // 6. CheckpointManager Create/Restore/Rollback
    // -----------------------------------------------------------------------
    println!("\n--- 6. CheckpointManager ---");
    println!("High-level manager combining saver, history, and filtering.\n");

    let saver = InMemoryPersistentSaver::new();
    let mut manager = CheckpointManager::new(Box::new(saver));

    // Create a chain of checkpoints.
    let mut meta = HashMap::new();
    meta.insert("node".to_string(), json!("__start__"));
    let cp0_id = manager
        .create_checkpoint("conv-1", json!({"messages": []}), None, meta)
        .unwrap();
    println!("  Created root: {}", cp0_id);

    let mut meta1 = HashMap::new();
    meta1.insert("node".to_string(), json!("agent"));
    let cp1_id = manager
        .create_checkpoint(
            "conv-1",
            json!({"messages": ["hello"]}),
            Some(cp0_id.clone()),
            meta1,
        )
        .unwrap();
    println!("  Created step 1: {}", cp1_id);

    let mut meta2 = HashMap::new();
    meta2.insert("node".to_string(), json!("tools"));
    let cp2_id = manager
        .create_checkpoint(
            "conv-1",
            json!({"messages": ["hello", "world"]}),
            Some(cp1_id.clone()),
            meta2,
        )
        .unwrap();
    println!("  Created step 2: {}", cp2_id);

    // Restore by ID.
    let restored = manager.restore(&cp1_id).unwrap().unwrap();
    println!(
        "\n  Restored cp1: step={}, state={}",
        restored.step, restored.state
    );

    // Restore latest.
    let latest = manager.restore_latest("conv-1").unwrap().unwrap();
    println!("  Latest 'conv-1': step={}", latest.step);

    // Rollback to step 0.
    let rolled_back = manager.rollback("conv-1", 0).unwrap().unwrap();
    println!("  Rollback to step 0: state={}", rolled_back.state);

    // Query with filter.
    let filter = CheckpointFilter::new()
        .with_thread("conv-1")
        .with_step_range(0, 1);
    let query_results = manager.query(&filter).unwrap();
    println!("  Query steps 0..=1: {} results", query_results.len());

    // Lineage via history.
    let lineage = manager.history().get_lineage(&cp2_id);
    println!("  Lineage to cp2 ({} ancestors):", lineage.len());
    for id in &lineage {
        println!("    {}", id);
    }

    // -----------------------------------------------------------------------
    // 7. Multi-Thread Checkpointing
    // -----------------------------------------------------------------------
    println!("\n--- 7. Multi-Thread Checkpointing ---");
    println!("Manage checkpoints across independent execution threads.\n");

    let saver = InMemoryPersistentSaver::new();
    let mut manager = CheckpointManager::new(Box::new(saver));

    // Thread A: 3 steps.
    let mut prev_id: Option<CheckpointId> = None;
    for step in 0..3 {
        let mut meta = HashMap::new();
        meta.insert("thread".to_string(), json!("A"));
        let id = manager
            .create_checkpoint(
                "thread-A",
                json!({"progress": format!("A-step-{}", step)}),
                prev_id,
                meta,
            )
            .unwrap();
        prev_id = Some(id);
    }
    println!("  Thread A: 3 checkpoints created");

    // Thread B: 4 steps.
    let mut prev_id: Option<CheckpointId> = None;
    for step in 0..4 {
        let mut meta = HashMap::new();
        meta.insert("thread".to_string(), json!("B"));
        let id = manager
            .create_checkpoint(
                "thread-B",
                json!({"progress": format!("B-step-{}", step)}),
                prev_id,
                meta,
            )
            .unwrap();
        prev_id = Some(id);
    }
    println!("  Thread B: 4 checkpoints created");

    // Thread C: 2 steps.
    let mut prev_id: Option<CheckpointId> = None;
    for step in 0..2 {
        let id = manager
            .create_checkpoint(
                "thread-C",
                json!({"progress": format!("C-step-{}", step)}),
                prev_id,
                HashMap::new(),
            )
            .unwrap();
        prev_id = Some(id);
    }
    println!("  Thread C: 2 checkpoints created");

    // Query across threads.
    let all_filter = CheckpointFilter::new().with_thread("thread-B");
    let thread_b = manager.query(&all_filter).unwrap();
    println!("\n  Thread B checkpoints: {}", thread_b.len());
    for cp in &thread_b {
        println!("    step={} state={}", cp.step, cp.state);
    }

    // Restore latest per thread.
    for tid in &["thread-A", "thread-B", "thread-C"] {
        let latest = manager.restore_latest(tid).unwrap().unwrap();
        println!("  Latest {}: step={}", tid, latest.step);
    }

    // -----------------------------------------------------------------------
    // 8. CheckpointStats
    // -----------------------------------------------------------------------
    println!("\n--- 8. CheckpointStats ---");
    println!("Get summary statistics about managed checkpoints.\n");

    let stats = manager.stats();
    println!("  Total checkpoints:     {}", stats.total_checkpoints);
    println!("  Threads:               {}", stats.threads);
    println!("  Avg steps per thread:  {:.2}", stats.avg_steps_per_thread);

    let stats_json = stats.to_json();
    println!(
        "  Stats JSON: {}",
        serde_json::to_string_pretty(&stats_json).unwrap()
    );

    // -----------------------------------------------------------------------
    // 9. Real LLM Demo — Generate a summary of checkpoint state
    // -----------------------------------------------------------------------
    println!("\n--- 9. Real LLM Demo ---");
    println!("Use an LLM to summarize the current checkpoint state.\n");

    let model = shared::get_chat_model(vec![
        "The checkpoint system has 9 checkpoints across 3 threads (A, B, C). Thread B has the most checkpoints (4). All threads are progressing normally.".into(),
    ]);

    let messages = vec![
        cognis_core::messages::Message::System(cognis_core::messages::SystemMessage::new(
            "You are a concise system status reporter.",
        )),
        cognis_core::messages::Message::Human(cognis_core::messages::HumanMessage::new(
            &format!(
                "Summarize this checkpoint state in 2-3 sentences: total_checkpoints={}, threads={}, avg_steps_per_thread={:.1}",
                stats.total_checkpoints, stats.threads, stats.avg_steps_per_thread
            ),
        )),
    ];

    let rt = tokio::runtime::Runtime::new().unwrap();
    let ai_msg = rt.block_on(async {
        model.invoke_messages(&messages, None).await
    });

    match ai_msg {
        Ok(response) => println!("  LLM summary: {}", response.base.content.text()),
        Err(e) => println!("  LLM error: {}", e),
    }

    println!("\n=== Checkpoint Persistence Example Complete ===");
}
