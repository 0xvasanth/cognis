//! Execution Hooks Example
//!
//! Demonstrates the LangGraph hook system: LoggingHook, TimingHook,
//! StateSnapshotHook, and StateValidationHook dispatched through a HookRegistry.

#[path = "../shared.rs"]
mod shared;

use std::sync::Arc;

use cognisgraph::graph::hooks::{
    HookAction, HookContext, HookPhase, HookRegistry, LoggingHook, StateSnapshotHook,
    StateValidationHook, TimingHook,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Execution Hooks ===\n");

    // Set up registry with multiple hooks
    let timing = Arc::new(TimingHook::new());
    let snapshots = Arc::new(StateSnapshotHook::new());
    let mut registry = HookRegistry::new();
    registry.register(Arc::new(LoggingHook));
    registry.register(timing.clone());
    registry.register(snapshots.clone());

    // Simulate graph lifecycle: BeforeGraph -> node1 -> node2 -> AfterGraph
    registry
        .dispatch(&HookContext::new(
            HookPhase::BeforeGraph,
            json!({"input": "hello"}),
            0,
        ))
        .await?;

    // Node: process_input
    let ctx = HookContext::new(
        HookPhase::BeforeNode,
        json!({"input": "hello", "step": 1}),
        1,
    )
    .with_node("process_input");
    registry.dispatch(&ctx).await?;
    tokio::time::sleep(std::time::Duration::from_millis(15)).await;
    let ctx = HookContext::new(
        HookPhase::AfterNode,
        json!({"input": "hello", "processed": true}),
        1,
    )
    .with_node("process_input");
    registry.dispatch(&ctx).await?;

    // Edge
    registry
        .dispatch(
            &HookContext::new(HookPhase::BeforeEdge, json!({}), 2)
                .with_edge("process_input", "generate_response"),
        )
        .await?;

    // Node: generate_response
    let ctx = HookContext::new(
        HookPhase::BeforeNode,
        json!({"processed": true, "step": 2}),
        2,
    )
    .with_node("generate_response");
    registry.dispatch(&ctx).await?;
    tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    let ctx = HookContext::new(HookPhase::AfterNode, json!({"response": "Hi there!"}), 2)
        .with_node("generate_response");
    registry.dispatch(&ctx).await?;

    registry
        .dispatch(&HookContext::new(
            HookPhase::AfterGraph,
            json!({"response": "Hi there!"}),
            3,
        ))
        .await?;

    // Timing results
    println!("\nTiming:");
    for (node, total) in &timing.total_durations().await {
        println!("  {}: {:?}", node, total);
    }

    // Snapshots
    let snaps = snapshots.get_snapshots().await;
    println!("\nSnapshots: {} captured", snaps.len());
    for s in &snaps {
        println!("  '{}' {:?} step {}", s.node, s.phase, s.step);
    }

    // State validation
    let mut val_registry = HookRegistry::new();
    val_registry.register(Arc::new(StateValidationHook::new(vec![
        "input".into(),
        "session_id".into(),
    ])));

    let action = val_registry
        .dispatch(
            &HookContext::new(
                HookPhase::BeforeNode,
                json!({"input": "x", "session_id": "abc"}),
                0,
            )
            .with_node("n"),
        )
        .await?;
    println!("\nValid state: {:?}", action);

    let action = val_registry
        .dispatch(&HookContext::new(HookPhase::BeforeNode, json!({"input": "x"}), 0).with_node("n"))
        .await?;
    if let HookAction::Abort(reason) = &action {
        println!("Missing key: Abort - {}", reason);
    }

    // LLM demo with hooks
    let model = shared::get_chat_model(vec![
        "Execution hooks provide observability into graph workflows.".into(),
    ]);
    let messages = vec![
        cognis_core::messages::Message::system("Answer concisely."),
        cognis_core::messages::Message::human("What are execution hooks in graph workflows?"),
    ];
    match model.invoke_messages(&messages, None).await {
        Ok(r) => println!("\nLLM: {}", r.base.content.text()),
        Err(e) => println!("\nLLM error: {}", e),
    }

    Ok(())
}
