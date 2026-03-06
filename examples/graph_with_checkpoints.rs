//! Persistent Graph Execution with Checkpointing
//!
//! Demonstrates how to use PersistentGraph with InMemoryCheckpointSaver to:
//! - Automatically save graph state after each invocation
//! - Resume execution from a saved checkpoint
//! - Fork execution into a separate thread
//! - Inspect checkpoint history
//!
//! No API keys required -- uses pure state manipulation.
//!
//! Run with: cargo run -p rustchain-examples --example graph_with_checkpoints

use std::sync::Arc;

use serde_json::{json, Value};

use langgraph::checkpoint::InMemoryCheckpointSaver;
use langgraph::graph::state::{AsyncNodeAction, StateGraph};
use langgraph::graph::PersistentGraph;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Persistent Graph with Checkpointing ===\n");

    // Step 1: Build a simple graph with three nodes.
    //
    // The graph pipeline:
    //   START -> classify -> process -> summarize -> END
    //
    // Each node reads and updates state fields.

    let classify: AsyncNodeAction = Arc::new(|state: Value| {
        Box::pin(async move {
            let input = state
                .get("input")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let category = if input.contains("error") || input.contains("bug") {
                "issue"
            } else if input.contains("feature") || input.contains("add") {
                "feature_request"
            } else {
                "general"
            };
            println!("  [classify] Input: \"{input}\" -> Category: \"{category}\"");
            Ok(json!({
                "category": category,
                "classified": true,
            }))
        })
    });

    let process: AsyncNodeAction = Arc::new(|state: Value| {
        Box::pin(async move {
            let category = state
                .get("category")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let priority = match category {
                "issue" => "high",
                "feature_request" => "medium",
                _ => "low",
            };
            println!("  [process] Category: \"{category}\" -> Priority: \"{priority}\"");
            Ok(json!({
                "priority": priority,
                "processed": true,
            }))
        })
    });

    let summarize: AsyncNodeAction = Arc::new(|state: Value| {
        Box::pin(async move {
            let input = state.get("input").and_then(|v| v.as_str()).unwrap_or("");
            let category = state.get("category").and_then(|v| v.as_str()).unwrap_or("");
            let priority = state.get("priority").and_then(|v| v.as_str()).unwrap_or("");
            let summary = format!(
                "Processed '{}' as {} (priority: {})",
                input, category, priority
            );
            println!("  [summarize] {summary}");
            Ok(json!({
                "summary": summary,
                "complete": true,
            }))
        })
    });

    let graph = StateGraph::new()
        .add_node("classify", classify)
        .add_node("process", process)
        .add_node("summarize", summarize)
        .add_edge("__start__", "classify")
        .add_edge("classify", "process")
        .add_edge("process", "summarize")
        .add_edge("summarize", "__end__")
        .compile()?;

    println!("Built graph: START -> classify -> process -> summarize -> END\n");

    // Step 2: Create a PersistentGraph with in-memory checkpoint storage.
    let saver = Arc::new(InMemoryCheckpointSaver::new());
    let persistent = PersistentGraph::new(graph, saver.clone(), "ticket-thread-1");

    println!("--- First Invocation (thread: {}) ---\n", persistent.thread_id());

    let result = persistent
        .invoke(json!({ "input": "There is a bug in the login page" }))
        .await?;

    println!("\n  Result: {}\n", serde_json::to_string_pretty(&result)?);

    // Step 3: Check the saved state.
    let state = persistent.get_state().await?;
    println!("--- Saved State ---");
    if let Some(s) = &state {
        println!("  {}\n", serde_json::to_string_pretty(s)?);
    }

    // Step 4: Invoke again -- the checkpoint is automatically loaded and merged.
    println!("--- Second Invocation (resumes from checkpoint) ---\n");

    let result2 = persistent
        .invoke(json!({ "input": "Please add a dark mode feature" }))
        .await?;

    println!("\n  Result: {}\n", serde_json::to_string_pretty(&result2)?);

    // Step 5: Inspect checkpoint history.
    println!("--- Checkpoint History ---");
    let history = persistent.get_history(None).await?;
    println!("  Total checkpoints: {}", history.len());
    for (i, cp) in history.iter().enumerate() {
        let summary = cp
            .channel_values
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("(none)");
        println!("  [{}] id={} summary={}", i + 1, &cp.id[..8], summary);
    }
    println!();

    // Step 6: Fork into a new thread from the first checkpoint.
    println!("--- Forking from first checkpoint ---");
    let first_checkpoint_id = &history.last().unwrap().id;
    let forked = persistent
        .fork(first_checkpoint_id, "ticket-thread-2")
        .await?;

    println!("  Forked to thread: {}", forked.thread_id());

    let forked_state = forked.get_state().await?.unwrap();
    println!(
        "  Forked state summary: {}",
        forked_state
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("(none)")
    );

    // The forked thread continues independently.
    println!("\n--- Invoking on forked thread ---\n");
    let forked_result = forked
        .invoke(json!({ "input": "error in payment processing" }))
        .await?;
    println!("\n  Forked result: {}\n", serde_json::to_string_pretty(&forked_result)?);

    // Verify original thread is unaffected.
    let original_state = persistent.get_state().await?.unwrap();
    println!("--- Original thread state (unchanged) ---");
    println!(
        "  Summary: {}",
        original_state
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("(none)")
    );

    println!("\nDone!");
    Ok(())
}
