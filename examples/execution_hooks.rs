//! Execution Hooks Example
//!
//! Demonstrates the LangGraph execution hooks system:
//! - Creating a `HookRegistry` and registering hooks
//! - Using `LoggingHook` for lifecycle event logging
//! - Using `TimingHook` to track per-node execution time
//! - Using `StateSnapshotHook` to capture state at each node
//! - Using `StateValidationHook` to validate required state keys
//! - Creating `HookContext` objects and dispatching events
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example execution_hooks`

mod shared;
use std::sync::Arc;

use cognisgraph::graph::hooks::{
    HookAction, HookContext, HookPhase, HookRegistry, LoggingHook, StateSnapshotHook,
    StateValidationHook, TimingHook,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Execution Hooks Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Create a HookRegistry and register hooks
    // -----------------------------------------------------------------------
    println!("--- 1. HookRegistry Setup ---");
    println!("Register multiple hooks that listen to lifecycle events.\n");

    let timing_hook = Arc::new(TimingHook::new());
    let snapshot_hook = Arc::new(StateSnapshotHook::new());

    let mut registry = HookRegistry::new();
    registry.register(Arc::new(LoggingHook));
    registry.register(timing_hook.clone());
    registry.register(snapshot_hook.clone());

    println!("Registered {} hooks in the registry", registry.len());
    println!(
        "Hooks listening to BeforeNode: {}",
        registry.hooks_for_phase(&HookPhase::BeforeNode).len()
    );
    println!(
        "Hooks listening to BeforeGraph: {}",
        registry.hooks_for_phase(&HookPhase::BeforeGraph).len()
    );
    println!();

    // -----------------------------------------------------------------------
    // 2. Simulate graph execution lifecycle
    // -----------------------------------------------------------------------
    println!("--- 2. Simulating Graph Execution ---");
    println!("Dispatch lifecycle events through the registry.\n");

    // BeforeGraph
    println!(">> Dispatching BeforeGraph...");
    let ctx = HookContext::new(HookPhase::BeforeGraph, json!({"input": "hello"}), 0);
    let action = registry.dispatch(&ctx).await?;
    println!("   Action: {:?}\n", action);

    // BeforeNode: "process_input"
    println!(">> Dispatching BeforeNode for 'process_input'...");
    let ctx = HookContext::new(
        HookPhase::BeforeNode,
        json!({"input": "hello", "step": 1}),
        1,
    )
    .with_node("process_input")
    .with_metadata("source", json!("user"));
    let action = registry.dispatch(&ctx).await?;
    println!("   Action: {:?}", action);

    // Simulate some work
    tokio::time::sleep(std::time::Duration::from_millis(15)).await;

    // AfterNode: "process_input"
    println!(">> Dispatching AfterNode for 'process_input'...");
    let ctx = HookContext::new(
        HookPhase::AfterNode,
        json!({"input": "hello", "processed": true, "step": 1}),
        1,
    )
    .with_node("process_input");
    let action = registry.dispatch(&ctx).await?;
    println!("   Action: {:?}\n", action);

    // BeforeEdge
    println!(">> Dispatching BeforeEdge 'process_input' -> 'generate_response'...");
    let ctx = HookContext::new(HookPhase::BeforeEdge, json!({}), 2)
        .with_edge("process_input", "generate_response");
    let action = registry.dispatch(&ctx).await?;
    println!("   Action: {:?}\n", action);

    // BeforeNode: "generate_response"
    println!(">> Dispatching BeforeNode for 'generate_response'...");
    let ctx = HookContext::new(
        HookPhase::BeforeNode,
        json!({"input": "hello", "processed": true, "step": 2}),
        2,
    )
    .with_node("generate_response");
    registry.dispatch(&ctx).await?;

    tokio::time::sleep(std::time::Duration::from_millis(25)).await;

    // AfterNode: "generate_response"
    println!(">> Dispatching AfterNode for 'generate_response'...");
    let ctx = HookContext::new(
        HookPhase::AfterNode,
        json!({"input": "hello", "processed": true, "response": "Hi there!", "step": 2}),
        2,
    )
    .with_node("generate_response");
    registry.dispatch(&ctx).await?;
    println!();

    // AfterGraph
    println!(">> Dispatching AfterGraph...");
    let ctx = HookContext::new(HookPhase::AfterGraph, json!({"response": "Hi there!"}), 3);
    registry.dispatch(&ctx).await?;
    println!();

    // -----------------------------------------------------------------------
    // 3. Inspect timing results
    // -----------------------------------------------------------------------
    println!("--- 3. Timing Results ---");
    println!("TimingHook recorded per-node execution durations.\n");

    let timings = timing_hook.get_timings().await;
    for (node, durations) in &timings {
        println!("  Node '{}': {} invocation(s)", node, durations.len());
        for (i, d) in durations.iter().enumerate() {
            println!("    Run {}: {:?}", i, d);
        }
    }

    let totals = timing_hook.total_durations().await;
    println!("\n  Total durations:");
    for (node, total) in &totals {
        println!("    {}: {:?}", node, total);
    }
    println!();

    // -----------------------------------------------------------------------
    // 4. Inspect state snapshots
    // -----------------------------------------------------------------------
    println!("--- 4. State Snapshots ---");
    println!("StateSnapshotHook captured state at each node.\n");

    let snapshots = snapshot_hook.get_snapshots().await;
    println!("Total snapshots captured: {}", snapshots.len());
    for snap in &snapshots {
        println!(
            "  Node '{}' at {:?} (step {}): {}",
            snap.node,
            snap.phase,
            snap.step,
            serde_json::to_string(&snap.state).unwrap_or_default()
        );
    }

    let process_snaps = snapshot_hook.snapshots_for_node("process_input").await;
    println!(
        "\nSnapshots for 'process_input': {} (before + after)",
        process_snaps.len()
    );
    println!();

    // -----------------------------------------------------------------------
    // 5. StateValidationHook
    // -----------------------------------------------------------------------
    println!("--- 5. State Validation ---");
    println!("Validates that required keys are present in the state.\n");

    let mut validation_registry = HookRegistry::new();
    validation_registry.register(Arc::new(StateValidationHook::new(vec![
        "input".to_string(),
        "session_id".to_string(),
    ])));

    // Valid state
    let ctx = HookContext::new(
        HookPhase::BeforeNode,
        json!({"input": "test", "session_id": "abc123"}),
        0,
    )
    .with_node("my_node");
    let action = validation_registry.dispatch(&ctx).await?;
    println!("  Valid state: {:?}", action);

    // Missing key
    let ctx =
        HookContext::new(HookPhase::BeforeNode, json!({"input": "test"}), 0).with_node("my_node");
    let action = validation_registry.dispatch(&ctx).await?;
    match &action {
        HookAction::Abort(reason) => println!("  Missing key: Abort - {}", reason),
        other => println!("  Missing key: {:?}", other),
    }

    // Non-object state
    let ctx =
        HookContext::new(HookPhase::BeforeNode, json!("not an object"), 0).with_node("my_node");
    let action = validation_registry.dispatch(&ctx).await?;
    match &action {
        HookAction::Abort(reason) => println!("  Non-object state: Abort - {}", reason),
        other => println!("  Non-object state: {:?}", other),
    }

    // -----------------------------------------------------------------------
    // 6. Real LLM Demo — LLM call in graph execution with hooks
    // -----------------------------------------------------------------------
    println!("\n--- 6. Real LLM Demo (LLM Call with Execution Hooks) ---\n");

    let model = shared::get_chat_model(vec![
        "Execution hooks provide observability into graph workflows by capturing state before and after each node executes.".into(),
    ]);

    // Create hooks to monitor the LLM call
    let llm_timing = Arc::new(TimingHook::new());
    let llm_snapshot = Arc::new(StateSnapshotHook::new());
    let mut llm_registry = HookRegistry::new();
    llm_registry.register(Arc::new(LoggingHook));
    llm_registry.register(llm_timing.clone());
    llm_registry.register(llm_snapshot.clone());

    // Before LLM node
    let before_ctx = HookContext::new(
        HookPhase::BeforeNode,
        json!({"question": "What are execution hooks?"}),
        0,
    )
    .with_node("llm_call");
    llm_registry.dispatch(&before_ctx).await?;

    let messages = vec![
        cognis_core::messages::Message::System(cognis_core::messages::SystemMessage::new(
            "You are a helpful assistant. Answer concisely in 1-2 sentences.",
        )),
        cognis_core::messages::Message::Human(cognis_core::messages::HumanMessage::new(
            "What are execution hooks in graph-based workflows?",
        )),
    ];

    use cognis_core::language_models::chat_model::BaseChatModel;
    let result = model.invoke_messages(&messages, None).await;

    match result {
        Ok(response) => {
            // After LLM node
            let after_ctx = HookContext::new(
                HookPhase::AfterNode,
                json!({"question": "What are execution hooks?", "answer": response.content.clone()}),
                0,
            )
            .with_node("llm_call");
            llm_registry.dispatch(&after_ctx).await?;

            println!("  LLM response: {}", response.content);

            let snaps = llm_snapshot.snapshots_for_node("llm_call").await;
            println!("  Snapshots captured for 'llm_call': {}", snaps.len());
        }
        Err(e) => println!("  LLM error: {}", e),
    }

    println!("\n=== Execution Hooks Example Complete ===");
    Ok(())
}
