//! Graph State Management Example
//!
//! Demonstrates the full state lifecycle in cognisgraph:
//!
//! 1. **get_state** — inspect the current graph state via checkpoints
//! 2. **update_state** — inject state changes from outside the graph
//! 3. **get_state_history** — browse the checkpoint timeline
//! 4. **interrupt / resume** — human-in-the-loop review workflow
//! 5. **replay_from / fork_from** — time-travel and branching
//!
//! Uses Ollama when available, otherwise falls back to a fake model.
//!
//! Run with: `cargo run -p cognis-examples --example graph_state_management`

#[path = "../shared.rs"]
mod shared;

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};

use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::Message;
use cognisgraph::checkpoint::{CheckpointSaver, InMemoryCheckpointSaver};
use cognisgraph::graph::state::StateGraph;
use cognisgraph::{END, START};

/// Build a simple multi-step processing graph:
///
///   START → classify → process → review → END
///
/// - **classify**: uses the LLM to classify input text
/// - **process**: transforms the state based on classification
/// - **review**: produces a final summary
///
/// The "review" node is configured as an interrupt-after point so we can
/// inspect state before completion.
fn build_graph(
    model: Arc<dyn BaseChatModel>,
) -> Result<cognisgraph::graph::state::CompiledStateGraph, Box<dyn std::error::Error>> {
    let classify_model = model.clone();
    let review_model = model;

    let graph = StateGraph::new()
        .add_node("classify", Arc::new(move |state: Value| {
            let model = classify_model.clone();
            Box::pin(async move {
                let text = state.get("input")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");

                let messages = vec![
                    Message::system("Classify the following text into one of: question, statement, command. Reply with just the category name."),
                    Message::human(text),
                ];

                let response = model.invoke_messages(&messages, None).await
                    .map_err(|e| cognisgraph::errors::LangGraphError::Other(e.to_string()))?;
                let category = response.base.content.text().trim().to_lowercase();

                Ok(json!({
                    "category": category,
                    "step": "classified"
                }))
            })
        }))
        .add_node("process", Arc::new(|state: Value| {
            Box::pin(async move {
                let category = state.get("category")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");

                let processed = match category {
                    c if c.contains("question") => "This input requires an answer.",
                    c if c.contains("command") => "This input requires an action.",
                    _ => "This input is informational.",
                };

                Ok(json!({
                    "analysis": processed,
                    "step": "processed"
                }))
            })
        }))
        .add_node("review", Arc::new(move |state: Value| {
            let model = review_model.clone();
            Box::pin(async move {
                let input = state.get("input")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let category = state.get("category")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let analysis = state.get("analysis")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");

                let messages = vec![
                    Message::system("Summarize the analysis result in one sentence."),
                    Message::human(&format!(
                        "Input: {}\nCategory: {}\nAnalysis: {}",
                        input, category, analysis
                    )),
                ];

                let response = model.invoke_messages(&messages, None).await
                    .map_err(|e| cognisgraph::errors::LangGraphError::Other(e.to_string()))?;

                Ok(json!({
                    "summary": response.base.content.text().trim(),
                    "step": "reviewed"
                }))
            })
        }))
        .add_edge(START, "classify")
        .add_edge("classify", "process")
        .add_edge("process", "review")
        .add_edge("review", END)
        .interrupt_after(vec!["process"])
        .compile()?;

    Ok(graph)
}

/// Save the current state as a checkpoint.
async fn save_checkpoint(
    saver: &dyn CheckpointSaver,
    thread_id: &str,
    state: &Value,
    step: i64,
    node: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut config = HashMap::new();
    config.insert("thread_id".to_string(), json!(thread_id));

    let mut cp = cognisgraph::checkpoint::Checkpoint {
        v: 4,
        id: uuid::Uuid::now_v7().to_string(),
        ts: format!("step-{}", step),
        channel_values: HashMap::new(),
        channel_versions: HashMap::new(),
        versions_seen: HashMap::new(),
        updated_channels: None,
    };

    if let Value::Object(map) = state {
        for (k, v) in map {
            cp.channel_values.insert(k.clone(), v.clone());
            cp.channel_versions.insert(k.clone(), step as u64);
        }
    }

    let metadata = cognisgraph::checkpoint::CheckpointMetadata {
        source: "loop".to_string(),
        step,
        writes: Some({
            let mut m = HashMap::new();
            m.insert(node.to_string(), state.clone());
            m
        }),
        extra: HashMap::new(),
    };

    saver.put(&config, cp, metadata).await?;
    Ok(())
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Graph State Management Demo ===\n");

    // ── Setup ────────────────────────────────────────────────────────
    let model = shared::get_chat_model(vec![
        "question".into(),
        "The input \"What is Rust?\" was classified as a question and requires an answer.".into(),
    ]);

    let graph = build_graph(model)?;
    let saver = InMemoryCheckpointSaver::new();
    let thread_id = "demo-thread-1";

    println!("Graph nodes: {:?}", graph.node_names());
    println!("Interrupt points: after \"process\"\n");

    // ── 1. Run graph with interrupt ──────────────────────────────────
    println!("--- Step 1: Invoke with interrupt ---\n");

    let input = json!({
        "input": "What is Rust?"
    });

    let result = graph.invoke_with_interrupt(input.clone()).await?;

    match result {
        cognisgraph::InvokeResult::Interrupted(ref interrupted) => {
            println!(
                "Graph interrupted after: \"{}\"",
                interrupted.interrupted_at
            );
            println!(
                "Current state: {}",
                serde_json::to_string_pretty(&interrupted.state)?
            );
            println!("Next nodes: {:?}\n", interrupted.next_nodes);

            // Save checkpoint at the interrupt point.
            save_checkpoint(
                &saver,
                thread_id,
                &interrupted.state,
                1,
                &interrupted.interrupted_at,
            )
            .await?;
        }
        cognisgraph::InvokeResult::Complete(ref state) => {
            println!(
                "Graph completed (no interrupt hit): {}",
                serde_json::to_string_pretty(state)?
            );
            save_checkpoint(&saver, thread_id, state, 1, "complete").await?;
        }
    }

    // ── 2. get_state — inspect current state ─────────────────────────
    println!("--- Step 2: Inspect state via get_state ---\n");

    let snapshot = graph.get_state(thread_id, &saver).await?;
    match snapshot {
        Some(snap) => {
            println!(
                "State values: {}",
                serde_json::to_string_pretty(&snap.values)?
            );
            if let Some(ref meta) = snap.metadata {
                println!(
                    "Metadata: source={}, step={}",
                    meta.get("source").and_then(|v| v.as_str()).unwrap_or("?"),
                    meta.get("step").unwrap_or(&json!("?")),
                );
            }
            println!();
        }
        None => println!("No state found for thread.\n"),
    }

    // ── 3. update_state — inject external changes ────────────────────
    println!("--- Step 3: Update state externally ---\n");

    let update_values = json!({
        "human_note": "Approved by reviewer — proceed with detailed answer",
        "priority": "high"
    });
    println!(
        "Injecting: {}",
        serde_json::to_string_pretty(&update_values)?
    );

    let new_config = graph
        .update_state(thread_id, update_values, Some("human_reviewer"), &saver)
        .await?;
    println!(
        "New checkpoint: {}\n",
        new_config
            .get("checkpoint_id")
            .and_then(|v| v.as_str())
            .unwrap_or("?")
    );

    // Verify the update was applied.
    let updated_snap = graph.get_state(thread_id, &saver).await?;
    if let Some(snap) = updated_snap {
        let has_note = snap
            .values
            .get("human_note")
            .and_then(|v| v.as_str())
            .is_some();
        println!(
            "Verified: human_note present = {}, priority = {}",
            has_note,
            snap.values
                .get("priority")
                .and_then(|v| v.as_str())
                .unwrap_or("?")
        );
        println!();
    }

    // ── 4. get_state_history — browse timeline ───────────────────────
    println!("--- Step 4: Browse state history ---\n");

    let history = graph.get_state_history(thread_id, &saver).await?;
    println!("History entries: {}", history.len());
    for (i, entry) in history.iter().enumerate() {
        println!(
            "  [{}] checkpoint={}, node={}, keys={:?}",
            i,
            &entry.checkpoint_id[..8],
            entry.node_name,
            entry
                .state
                .as_object()
                .map(|m| m.keys().collect::<Vec<_>>())
                .unwrap_or_default(),
        );
    }
    println!();

    // ── 5. Resume from interrupt ─────────────────────────────────────
    println!("--- Step 5: Resume graph execution ---\n");

    if let cognisgraph::InvokeResult::Interrupted(interrupted) = result {
        // Resume with the additional human context merged in.
        let resume_update = json!({
            "human_note": "Approved — continue processing"
        });

        let final_result = graph.resume(interrupted, Some(resume_update)).await?;

        match final_result {
            cognisgraph::InvokeResult::Complete(state) => {
                println!("Graph completed after resume.");
                println!(
                    "Summary: {}",
                    state
                        .get("summary")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(no summary)")
                );
                println!(
                    "Human note preserved: {}",
                    state
                        .get("human_note")
                        .and_then(|v| v.as_str())
                        .unwrap_or("(none)")
                );

                // Save final checkpoint.
                save_checkpoint(&saver, thread_id, &state, 2, "review").await?;
            }
            cognisgraph::InvokeResult::Interrupted(int) => {
                println!("Graph interrupted again at: {}", int.interrupted_at);
            }
        }
    }
    println!();

    // ── 6. Fork from historical checkpoint ───────────────────────────
    println!("--- Step 6: Fork from history ---\n");

    let history = graph.get_state_history(thread_id, &saver).await?;
    if let Some(first) = history.first() {
        let fork_thread = "forked-thread";
        let forked_state = graph
            .fork_from(thread_id, &first.checkpoint_id, fork_thread, &saver)
            .await?;
        println!(
            "Forked from checkpoint {} into thread \"{}\"",
            &first.checkpoint_id[..8],
            fork_thread
        );
        println!(
            "Forked state keys: {:?}",
            forked_state
                .as_object()
                .map(|m| m.keys().collect::<Vec<_>>())
                .unwrap_or_default()
        );

        // Verify the fork exists.
        let fork_snap = graph.get_state(fork_thread, &saver).await?;
        println!("Fork state exists: {}\n", fork_snap.is_some());
    }

    // ── 7. Final history overview ────────────────────────────────────
    println!("--- Final: Complete history for thread ---\n");

    let final_history = graph.get_state_history(thread_id, &saver).await?;
    println!("Total checkpoints: {}", final_history.len());
    for (i, entry) in final_history.iter().enumerate() {
        let keys: Vec<&String> = entry
            .state
            .as_object()
            .map(|m| m.keys().collect())
            .unwrap_or_default();
        println!(
            "  [{}] id={}.. node={:<20} state_keys={:?}",
            i,
            &entry.checkpoint_id[..8],
            entry.node_name,
            keys,
        );
    }

    println!("\n=== Done ===");
    Ok(())
}
