//! State Snapshot Example
//!
//! Demonstrates state snapshots and time-travel debugging:
//! SnapshotStore, TimeTravelDebugger, and SnapshotComparator.
//!
//! Run with: `cargo run -p cognis-examples --example state_snapshot`

#[path = "../shared.rs"]
mod shared;
use cognis_core::messages::Message;
use cognisgraph::graph::{SnapshotComparator, SnapshotStore, StateSnapshot, TimeTravelDebugger};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create snapshots simulating a graph execution
    let mut store = SnapshotStore::with_max_snapshots(10);

    let snap0 = StateSnapshot::new(
        json!({"messages": [], "step_count": 0, "status": "initialized"}),
        "init",
        0,
    )
    .with_metadata("description", json!("Graph initialized"));
    let snap0_id = snap0.id.clone();
    store.save(snap0);

    let snap1 = StateSnapshot::new(
        json!({"messages": ["Hello"], "step_count": 1, "status": "processing"}),
        "router",
        1,
    )
    .with_parent(snap0_id.clone());
    let snap1_id = snap1.id.clone();
    store.save(snap1);

    let snap2 = StateSnapshot::new(
        json!({"messages": ["Hello", "Hi there!"], "step_count": 2, "status": "processing", "tool_calls": 0}),
        "agent", 2,
    ).with_parent(snap1_id.clone());
    store.save(snap2);

    let snap3 = StateSnapshot::new(
        json!({"messages": ["Hello", "Hi there!", "Use search tool"], "step_count": 3, "status": "tool_call", "tool_calls": 1}),
        "tool_executor",
        3,
    );
    store.save(snap3);

    store.save(StateSnapshot::new(
        json!({"messages": ["Hello", "Hi there!", "Use search tool", "Search results: ..."], "step_count": 4, "status": "complete", "tool_calls": 1}),
        "output", 4,
    ));

    println!("Store: {} snapshots", store.len());

    // Query snapshots
    let latest = store.latest().unwrap();
    println!("Latest: step={}, node={}", latest.step, latest.node_name);

    // Time-travel navigation
    let debugger = TimeTravelDebugger::new(&store);
    if let Some(snap) = debugger.goto_step(1) {
        println!(
            "goto_step(1): node={}, status={}",
            snap.node_name, snap.state["status"]
        );
    }
    if let Some(snap) = debugger.rewind(2) {
        println!("rewind(2): node={}, step={}", snap.node_name, snap.step);
    }

    // Execution path
    let path = debugger.execution_path();
    println!(
        "Execution path: {:?}",
        path.iter()
            .map(|(s, n)| format!("{}:{}", s, n))
            .collect::<Vec<_>>()
    );

    // Diffs between snapshots
    let s0 = debugger.goto_step(0).unwrap();
    let s4 = debugger.goto_step(4).unwrap();
    let diff = SnapshotComparator::diff(s0, s4);
    println!("Diff step 0->4: {}", diff.summary());

    // Capacity eviction
    let mut small_store = SnapshotStore::with_max_snapshots(3);
    for i in 0..5 {
        small_store.save(StateSnapshot::new(
            json!({"i": i}),
            &format!("step_{}", i),
            i,
        ));
    }
    println!("Stored 5 in max-3 store, remaining: {}", small_store.len());

    // LLM demo
    let model = shared::get_chat_model(vec![
        "The graph processed a user message through routing, agent response, tool execution, and completion. Key changes: messages grew from 0 to 4, status went from initialized to complete.".into(),
    ]);
    let messages = vec![
        Message::system("Analyze state snapshot diffs concisely."),
        Message::human(&format!(
            "Diff between step 0 and step 4:\n{}",
            diff.summary()
        )),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM: {}", gen.message.content().text());
    }

    Ok(())
}
