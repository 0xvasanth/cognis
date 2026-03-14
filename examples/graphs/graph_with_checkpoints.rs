//! Persistent Graph with Checkpointing
//!
//! Demonstrates PersistentGraph with InMemoryCheckpointSaver: automatic state
//! saving, resuming from checkpoints, forking into separate threads, and
//! inspecting checkpoint history.

#[path = "../shared.rs"]
mod shared;

use serde_json::{json, Value};
use std::sync::Arc;

use cognisgraph::checkpoint::InMemoryCheckpointSaver;
use cognisgraph::graph::state::{AsyncNodeAction, StateGraph};
use cognisgraph::graph::PersistentGraph;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Persistent Graph with Checkpointing ===\n");

    // Build pipeline: START -> classify -> process -> summarize -> END
    let classify: AsyncNodeAction = Arc::new(|state: Value| {
        Box::pin(async move {
            let input = state.get("input").and_then(|v| v.as_str()).unwrap_or("");
            let category = if input.contains("error") || input.contains("bug") {
                "issue"
            } else if input.contains("feature") || input.contains("add") {
                "feature_request"
            } else {
                "general"
            };
            println!("  [classify] '{}' -> {}", input, category);
            Ok(json!({"category": category, "classified": true}))
        })
    });

    let process: AsyncNodeAction = Arc::new(|state: Value| {
        Box::pin(async move {
            let cat = state
                .get("category")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown");
            let priority = match cat {
                "issue" => "high",
                "feature_request" => "medium",
                _ => "low",
            };
            println!("  [process] {} -> priority {}", cat, priority);
            Ok(json!({"priority": priority, "processed": true}))
        })
    });

    let summarize: AsyncNodeAction = Arc::new(|state: Value| {
        Box::pin(async move {
            let input = state.get("input").and_then(|v| v.as_str()).unwrap_or("");
            let cat = state.get("category").and_then(|v| v.as_str()).unwrap_or("");
            let pri = state.get("priority").and_then(|v| v.as_str()).unwrap_or("");
            let summary = format!("'{}' -> {} ({})", input, cat, pri);
            println!("  [summarize] {}", summary);
            Ok(json!({"summary": summary, "complete": true}))
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

    let saver = Arc::new(InMemoryCheckpointSaver::new());
    let persistent = PersistentGraph::new(graph, saver.clone(), "thread-1");

    // First invocation
    println!("--- Invocation 1 ---");
    let r1 = persistent
        .invoke(json!({"input": "There is a bug in login"}))
        .await?;
    println!("Result: {}\n", serde_json::to_string_pretty(&r1)?);

    // Second invocation (resumes from checkpoint)
    println!("--- Invocation 2 ---");
    let r2 = persistent
        .invoke(json!({"input": "Please add dark mode"}))
        .await?;
    println!("Result: {}\n", serde_json::to_string_pretty(&r2)?);

    // Checkpoint history
    let history = persistent.get_history(None).await?;
    println!("Checkpoints: {}", history.len());
    for (i, cp) in history.iter().enumerate() {
        let summary = cp
            .channel_values
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap_or("(none)");
        println!("  [{}] {}... -> {}", i + 1, &cp.id[..8], summary);
    }

    // Fork from first checkpoint
    let first_id = &history.last().unwrap().id;
    let forked = persistent.fork(first_id, "thread-2").await?;
    println!("\nForked to thread: {}", forked.thread_id());
    let fr = forked.invoke(json!({"input": "error in payments"})).await?;
    println!("Forked result: {}\n", serde_json::to_string_pretty(&fr)?);

    // Verify original is unchanged
    let orig = persistent.get_state().await?.unwrap();
    println!(
        "Original summary: {}",
        orig.get("summary").and_then(|v| v.as_str()).unwrap_or("?")
    );

    // LLM summary
    let model = shared::get_chat_model(vec![
        "Processed a login bug (high) and dark mode request (medium).".into(),
    ]);
    let result = model
        ._generate(
            &[cognis_core::messages::Message::human(
                "Summarize: login bug and dark mode feature request pipeline",
            )],
            None,
        )
        .await?;
    if let Some(gen) = result.generations.first() {
        println!("\nLLM: {}", gen.message.content().text());
    }

    Ok(())
}
