//! State Snapshot Example
//!
//! Demonstrates LangGraph state snapshots and time-travel debugging:
//!
//! - Create a `SnapshotStore` and save snapshots at different execution steps
//! - Use `TimeTravelDebugger` to navigate history (goto_step, rewind)
//! - Use `SnapshotComparator` to compute structural diffs between snapshots
//! - Query snapshots by step and node name
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example state_snapshot`

mod shared;
use cognisgraph::graph::{
    SnapshotComparator, SnapshotDiff, SnapshotId, SnapshotStore, StateSnapshot, TimeTravelDebugger,
};
use serde_json::json;

fn main() {
    println!("=== State Snapshot Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Create a SnapshotStore and save snapshots
    // -----------------------------------------------------------------------
    println!("--- 1. Creating SnapshotStore and saving snapshots ---");

    let mut store = SnapshotStore::with_max_snapshots(10);

    let snap0 = StateSnapshot::new(
        json!({"messages": [], "step_count": 0, "status": "initialized"}),
        "init",
        0,
    )
    .with_metadata("description", json!("Graph initialized"));

    let snap0_id = snap0.id.clone();
    store.save(snap0);
    println!("  Saved snapshot at step 0 (init)");

    let snap1 = StateSnapshot::new(
        json!({"messages": ["Hello"], "step_count": 1, "status": "processing"}),
        "router",
        1,
    )
    .with_parent(snap0_id.clone())
    .with_metadata("description", json!("Routing user input"));

    let snap1_id = snap1.id.clone();
    store.save(snap1);
    println!("  Saved snapshot at step 1 (router)");

    let snap2 = StateSnapshot::new(
        json!({"messages": ["Hello", "Hi there!"], "step_count": 2, "status": "processing", "tool_calls": 0}),
        "agent",
        2,
    )
    .with_parent(snap1_id.clone())
    .with_metadata("description", json!("Agent generated response"));

    let snap2_id = snap2.id.clone();
    store.save(snap2);
    println!("  Saved snapshot at step 2 (agent)");

    let snap3 = StateSnapshot::new(
        json!({"messages": ["Hello", "Hi there!", "Use search tool"], "step_count": 3, "status": "tool_call", "tool_calls": 1}),
        "tool_executor",
        3,
    )
    .with_parent(snap2_id.clone())
    .with_metadata("description", json!("Executing tool call"));

    store.save(snap3);
    println!("  Saved snapshot at step 3 (tool_executor)");

    let snap4 = StateSnapshot::new(
        json!({"messages": ["Hello", "Hi there!", "Use search tool", "Search results: ..."], "step_count": 4, "status": "complete", "tool_calls": 1}),
        "output",
        4,
    )
    .with_metadata("description", json!("Final output"));

    store.save(snap4);
    println!("  Saved snapshot at step 4 (output)");

    println!("  Store size: {} snapshots", store.len());
    println!();

    // -----------------------------------------------------------------------
    // 2. Query snapshots
    // -----------------------------------------------------------------------
    println!("--- 2. Querying snapshots ---");

    let by_step = store.get_by_step(2);
    println!("  Snapshots at step 2: {}", by_step.len());
    if let Some(s) = by_step.first() {
        println!("    Node: {}, State: {}", s.node_name, s.state);
    }

    let by_node = store.get_by_node("agent");
    println!("  Snapshots for node 'agent': {}", by_node.len());

    let latest = store.latest().expect("store is not empty");
    println!(
        "  Latest snapshot: step={}, node={}",
        latest.step, latest.node_name
    );
    println!();

    // -----------------------------------------------------------------------
    // 3. TimeTravelDebugger: navigation
    // -----------------------------------------------------------------------
    println!("--- 3. TimeTravelDebugger: navigation ---");

    let debugger = TimeTravelDebugger::new(&store);

    // goto_step
    if let Some(snap) = debugger.goto_step(1) {
        println!(
            "  goto_step(1): node={}, status={}",
            snap.node_name, snap.state["status"]
        );
    }

    // rewind from latest
    if let Some(snap) = debugger.rewind(0) {
        println!(
            "  rewind(0) [latest]: node={}, step={}",
            snap.node_name, snap.step
        );
    }
    if let Some(snap) = debugger.rewind(2) {
        println!(
            "  rewind(2) [2 back]: node={}, step={}",
            snap.node_name, snap.step
        );
    }
    if let Some(snap) = debugger.rewind(4) {
        println!(
            "  rewind(4) [4 back]: node={}, step={}",
            snap.node_name, snap.step
        );
    }

    // goto_snapshot by ID
    if let Some(snap) = debugger.goto_snapshot(&snap0_id) {
        println!("  goto_snapshot(snap0_id): node={}", snap.node_name);
    }

    // Missing step
    let missing = debugger.goto_step(99);
    println!("  goto_step(99): {:?}", missing.map(|s| &s.node_name));
    println!();

    // -----------------------------------------------------------------------
    // 4. Execution path
    // -----------------------------------------------------------------------
    println!("--- 4. Execution path ---");

    let path = debugger.execution_path();
    println!("  Full execution path:");
    for (step, node) in &path {
        println!("    step {} -> {}", step, node);
    }
    println!();

    // -----------------------------------------------------------------------
    // 5. SnapshotComparator: diffs
    // -----------------------------------------------------------------------
    println!("--- 5. SnapshotComparator: diffs ---");

    // Diff between step 0 and step 4
    let s0 = debugger.goto_step(0).expect("step 0 exists");
    let s4 = debugger.goto_step(4).expect("step 4 exists");

    let diff = SnapshotComparator::diff(s0, s4);
    println!("  Diff between step 0 (init) and step 4 (output):");
    println!("    Added keys:   {:?}", diff.added);
    println!("    Removed keys: {:?}", diff.removed);
    println!("    Changed keys:");
    for (key, old, new) in &diff.changed {
        println!("      {}: {} -> {}", key, old, new);
    }
    println!("    Summary: {}", diff.summary());
    println!();

    // Diff between step 2 and step 3 (tool_calls added/changed)
    let s2 = debugger.goto_step(2).expect("step 2 exists");
    let s3 = debugger.goto_step(3).expect("step 3 exists");

    let diff_2_3 = SnapshotComparator::diff(s2, s3);
    println!("  Diff between step 2 (agent) and step 3 (tool_executor):");
    println!("    Summary: {}", diff_2_3.summary());
    println!("    JSON: {}", diff_2_3.to_json());
    println!();

    // -----------------------------------------------------------------------
    // 6. diff_steps shortcut
    // -----------------------------------------------------------------------
    println!("--- 6. diff_steps shortcut ---");

    if let Some(diff_json) = debugger.diff_steps(1, 4) {
        println!(
            "  diff_steps(1, 4): {}",
            serde_json::to_string_pretty(&diff_json).unwrap()
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 7. Equivalence check
    // -----------------------------------------------------------------------
    println!("--- 7. Equivalence check ---");

    let equiv = SnapshotComparator::is_equivalent(s0, s4);
    println!("  step 0 equivalent to step 4? {}", equiv);

    // Create two snapshots with identical state
    let a = StateSnapshot::new(json!({"x": 1}), "node_a", 100);
    let b = StateSnapshot::new(json!({"x": 1}), "node_b", 200);
    let equiv2 = SnapshotComparator::is_equivalent(&a, &b);
    println!(
        "  Two snapshots with same state but different node/step equivalent? {}",
        equiv2
    );
    println!();

    // -----------------------------------------------------------------------
    // 8. Snapshot serialization
    // -----------------------------------------------------------------------
    println!("--- 8. Snapshot JSON serialization ---");

    let snap_json = s2.to_json();
    println!(
        "  Snapshot at step 2 as JSON:\n  {}",
        serde_json::to_string_pretty(&snap_json).unwrap()
    );
    println!();

    // -----------------------------------------------------------------------
    // 9. Capacity eviction
    // -----------------------------------------------------------------------
    println!("--- 9. Capacity eviction ---");

    let mut small_store = SnapshotStore::with_max_snapshots(3);
    for i in 0..5 {
        small_store.save(StateSnapshot::new(
            json!({"i": i}),
            &format!("step_{}", i),
            i,
        ));
    }
    println!(
        "  Stored 5 snapshots in a store with max_snapshots=3, current size: {}",
        small_store.len()
    );
    let history: Vec<_> = small_store.history().iter().map(|s| s.step).collect();
    println!("  Remaining steps: {:?}", history);
    println!();

    println!("=== State Snapshot Example Complete ===");
}
