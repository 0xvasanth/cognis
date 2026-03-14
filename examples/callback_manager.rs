//! Callback Manager Example
//!
//! Demonstrates the centralized callback system for observing and instrumenting
//! the execution lifecycle of chains, LLMs, tools, and retrievers.
//!
//! Features shown:
//! - CallbackPhase variants (built-in and custom)
//! - CallbackData building with the builder pattern
//! - ConsoleCallbackHandler logging events
//! - MetricsCallbackHandler collecting timing data
//! - CallbackManager dispatching to multiple handlers
//! - CallbackScope RAII behavior (auto start/end events)
//! - FilteredHandler phase filtering
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example callback_manager`

mod shared;
use cognis::callbacks::manager::{
    CallbackDataBuilder, CallbackHandler, CallbackManager, CallbackPhase, CallbackScope,
    ConsoleCallbackHandler, FilteredHandler, MetricsCallbackHandler,
};
use cognis_core::messages::Message;
use serde_json::json;
use std::sync::Arc;

/// Wrapper to allow sharing a handler between the manager and our code.
/// The CallbackManager takes ownership via Box<dyn CallbackHandler>, so we
/// wrap an Arc to retain access for inspection after events are emitted.
struct SharedConsole(Arc<ConsoleCallbackHandler>);
impl CallbackHandler for SharedConsole {
    fn on_event(&self, data: &cognis::callbacks::manager::CallbackData) {
        self.0.on_event(data);
    }
    fn name(&self) -> &str {
        self.0.name()
    }
}

struct SharedMetrics(Arc<MetricsCallbackHandler>);
impl CallbackHandler for SharedMetrics {
    fn on_event(&self, data: &cognis::callbacks::manager::CallbackData) {
        self.0.on_event(data);
    }
    fn name(&self) -> &str {
        self.0.name()
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Callback Manager Example ===\n");

    // -----------------------------------------------------------------------
    // 1. CallbackPhase Variants
    // -----------------------------------------------------------------------
    println!("--- 1. CallbackPhase Variants ---");
    println!("Phases represent lifecycle stages of chains, LLMs, tools, and retrievers.\n");

    let phases = [
        CallbackPhase::ChainStart,
        CallbackPhase::ChainEnd,
        CallbackPhase::ChainError,
        CallbackPhase::LlmStart,
        CallbackPhase::LlmEnd,
        CallbackPhase::LlmError,
        CallbackPhase::ToolStart,
        CallbackPhase::ToolEnd,
        CallbackPhase::ToolError,
        CallbackPhase::RetrieverStart,
        CallbackPhase::RetrieverEnd,
        CallbackPhase::Custom("my_custom_phase".into()),
    ];

    println!("Built-in and custom phases:");
    for phase in &phases {
        println!("  {:?} => \"{}\"", phase, phase.as_str());
    }

    // Display trait
    println!("\nDisplay formatting: {}", CallbackPhase::ChainStart);

    // Equality and hashing
    let phase_a = CallbackPhase::LlmStart;
    let phase_b = CallbackPhase::LlmStart;
    assert_eq!(phase_a, phase_b);
    assert_ne!(CallbackPhase::LlmStart, CallbackPhase::LlmEnd);
    println!("LlmStart == LlmStart: true");
    println!("LlmStart == LlmEnd: false");

    // -----------------------------------------------------------------------
    // 2. CallbackData Building
    // -----------------------------------------------------------------------
    println!("\n--- 2. CallbackData Building ---");
    println!("Build structured event payloads with the builder pattern.\n");

    // Minimal event
    let run_id = CallbackManager::new_run_id();
    let minimal = CallbackDataBuilder::new(CallbackPhase::ChainStart, &run_id).build();
    println!("Minimal event:");
    println!("  phase: {}", minimal.phase);
    println!("  run_id: {}", minimal.run_id);
    println!("  parent_run_id: {:?}", minimal.parent_run_id);
    println!("  timestamp: {}", minimal.timestamp);

    // Full event with all fields
    let full = CallbackDataBuilder::new(CallbackPhase::LlmEnd, "run-42")
        .parent_run_id("parent-run-1")
        .data(json!({
            "model": "test-model",
            "tokens": 150,
            "response": "Hello from the LLM"
        }))
        .timestamp("2026-03-09T12:00:00Z")
        .build();

    println!("\nFull event:");
    println!(
        "  {}",
        serde_json::to_string_pretty(&full.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 3. ConsoleCallbackHandler
    // -----------------------------------------------------------------------
    println!("\n--- 3. ConsoleCallbackHandler ---");
    println!("Logs every event to an internal buffer for inspection.\n");

    let console = ConsoleCallbackHandler::new();
    println!("Handler name: '{}'", console.name());

    // Send a few events
    let event1 = CallbackDataBuilder::new(CallbackPhase::ChainStart, "r1")
        .data(json!({"chain": "qa_chain"}))
        .timestamp("t0")
        .build();
    let event2 = CallbackDataBuilder::new(CallbackPhase::LlmStart, "r2")
        .data(json!({"model": "gpt-test"}))
        .timestamp("t1")
        .build();
    let event3 = CallbackDataBuilder::new(CallbackPhase::LlmEnd, "r2")
        .data(json!({"tokens": 50}))
        .timestamp("t2")
        .build();

    console.on_event(&event1);
    console.on_event(&event2);
    console.on_event(&event3);

    println!("Logged {} events:", console.logs().len());
    for log_line in &console.logs() {
        println!("  {}", log_line);
    }

    // Clear logs
    console.clear();
    println!("\nAfter clear: {} logs", console.logs().len());

    // -----------------------------------------------------------------------
    // 4. MetricsCallbackHandler
    // -----------------------------------------------------------------------
    println!("\n--- 4. MetricsCallbackHandler ---");
    println!("Collects event counts and timing durations per phase.\n");

    let metrics = MetricsCallbackHandler::new();
    println!("Handler name: '{}'", metrics.name());

    // Simulate a chain run with start/end pairs
    let start_evt = CallbackDataBuilder::new(CallbackPhase::ChainStart, "run-a")
        .timestamp("t0")
        .build();
    metrics.on_event(&start_evt);

    let end_evt = CallbackDataBuilder::new(CallbackPhase::ChainEnd, "run-a")
        .timestamp("t1")
        .build();
    metrics.on_event(&end_evt);

    // Simulate an LLM call
    let llm_start = CallbackDataBuilder::new(CallbackPhase::LlmStart, "run-b")
        .timestamp("t2")
        .build();
    metrics.on_event(&llm_start);
    let llm_end = CallbackDataBuilder::new(CallbackPhase::LlmEnd, "run-b")
        .timestamp("t3")
        .build();
    metrics.on_event(&llm_end);

    // Simulate a tool error
    let tool_start = CallbackDataBuilder::new(CallbackPhase::ToolStart, "run-c")
        .timestamp("t4")
        .build();
    metrics.on_event(&tool_start);
    let tool_err = CallbackDataBuilder::new(CallbackPhase::ToolError, "run-c")
        .data(json!({"error": "tool not found"}))
        .timestamp("t5")
        .build();
    metrics.on_event(&tool_err);

    println!("Total events: {}", metrics.total_events());
    println!("Events by phase: {:?}", metrics.events_by_phase());

    if let Some(avg) = metrics.avg_duration_ms(&CallbackPhase::ChainEnd) {
        println!("Avg chain_end duration: {:.3}ms", avg);
    }
    if let Some(avg) = metrics.avg_duration_ms(&CallbackPhase::LlmEnd) {
        println!("Avg llm_end duration: {:.3}ms", avg);
    }
    if let Some(avg) = metrics.avg_duration_ms(&CallbackPhase::ToolError) {
        println!("Avg tool_error duration: {:.3}ms", avg);
    }

    println!("\nMetrics JSON:");
    println!(
        "  {}",
        serde_json::to_string_pretty(&metrics.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 5. CallbackManager Dispatching
    // -----------------------------------------------------------------------
    println!("\n--- 5. CallbackManager Dispatching ---");
    println!("The manager dispatches events to all registered handlers.\n");

    let manager = CallbackManager::new();
    println!("Initial handler count: {}", manager.handler_count());

    // Create shared handlers so we can inspect them after dispatch
    let shared_console = Arc::new(ConsoleCallbackHandler::new());
    let shared_metrics = Arc::new(MetricsCallbackHandler::new());

    manager.add_handler(Box::new(SharedConsole(shared_console.clone())));
    manager.add_handler(Box::new(SharedMetrics(shared_metrics.clone())));
    println!(
        "Handler count after adding two: {}",
        manager.handler_count()
    );

    // Emit events through the manager
    manager.emit_phase(
        CallbackPhase::ChainStart,
        "dispatch-run-1",
        json!({"chain": "example_chain"}),
    );
    manager.emit_phase(
        CallbackPhase::LlmStart,
        "dispatch-run-2",
        json!({"model": "test-llm"}),
    );
    manager.emit_phase(
        CallbackPhase::LlmEnd,
        "dispatch-run-2",
        json!({"tokens": 100}),
    );
    manager.emit_phase(
        CallbackPhase::ChainEnd,
        "dispatch-run-1",
        json!({"result": "success"}),
    );

    println!("\nConsole logs after dispatch:");
    for log in &shared_console.logs() {
        println!("  {}", log);
    }

    println!("\nMetrics after dispatch:");
    println!("  Total events: {}", shared_metrics.total_events());
    println!("  By phase: {:?}", shared_metrics.events_by_phase());

    // Unique run IDs
    let id1 = CallbackManager::new_run_id();
    let id2 = CallbackManager::new_run_id();
    println!("\nGenerated run IDs (unique):");
    println!("  {}", id1);
    println!("  {}", id2);
    assert_ne!(id1, id2);

    // -----------------------------------------------------------------------
    // 6. CallbackScope RAII Behavior
    // -----------------------------------------------------------------------
    println!("\n--- 6. CallbackScope RAII ---");
    println!("Scopes emit a start event on creation and an end event on drop.\n");

    let scope_console = Arc::new(ConsoleCallbackHandler::new());
    let scope_metrics = Arc::new(MetricsCallbackHandler::new());
    let scope_manager = CallbackManager::new();
    scope_manager.add_handler(Box::new(SharedConsole(scope_console.clone())));
    scope_manager.add_handler(Box::new(SharedMetrics(scope_metrics.clone())));

    // Normal scope: emits ChainStart on creation, ChainEnd on drop
    println!("Creating a chain scope...");
    {
        let _scope = CallbackScope::new(
            &scope_manager,
            CallbackPhase::ChainStart,
            CallbackPhase::ChainEnd,
            "scope-run-1".into(),
            json!({"chain": "my_chain"}),
        );
        println!(
            "  Inside scope: events so far = {}",
            scope_metrics.total_events()
        );
        // Simulating work inside the chain scope
        println!("  (doing work inside the chain...)");
    }
    println!(
        "  After scope dropped: events = {}",
        scope_metrics.total_events()
    );

    println!("\nScope logs:");
    for log in &scope_console.logs() {
        println!("  {}", log);
    }

    // Failed scope: emits start, then error (not end) on drop
    println!("\nCreating a failing LLM scope...");
    scope_console.clear();
    {
        let scope = CallbackScope::new(
            &scope_manager,
            CallbackPhase::LlmStart,
            CallbackPhase::LlmEnd,
            "scope-run-2".into(),
            json!({"model": "test-model"}),
        );
        println!("  Marking scope as failed...");
        scope.fail("connection timeout".into());
    }

    println!("  Logs after failed scope:");
    for log in &scope_console.logs() {
        println!("    {}", log);
    }
    // Verify the error phase was emitted instead of the end phase
    let logs = scope_console.logs();
    assert!(logs[0].contains("llm_start"));
    assert!(logs[1].contains("llm_error"));
    println!("  Confirmed: llm_error emitted instead of llm_end");

    // Tool scope failure
    println!("\nCreating a failing tool scope...");
    scope_console.clear();
    {
        let scope = CallbackScope::new(
            &scope_manager,
            CallbackPhase::ToolStart,
            CallbackPhase::ToolEnd,
            "scope-run-3".into(),
            json!({"tool": "calculator"}),
        );
        scope.fail("division by zero".into());
    }
    let tool_logs = scope_console.logs();
    assert!(tool_logs[1].contains("tool_error"));
    println!("  Confirmed: tool_error emitted on tool scope failure");

    // -----------------------------------------------------------------------
    // 7. FilteredHandler Phase Filtering
    // -----------------------------------------------------------------------
    println!("\n--- 7. FilteredHandler ---");
    println!("Wraps a handler to only receive events for specified phases.\n");

    let filter_console = Arc::new(ConsoleCallbackHandler::new());
    let filter_manager = CallbackManager::new();

    // Only allow LLM phases through the filter
    let filtered = FilteredHandler::new(
        Box::new(SharedConsole(filter_console.clone())),
        vec![
            CallbackPhase::LlmStart,
            CallbackPhase::LlmEnd,
            CallbackPhase::LlmError,
        ],
    );

    println!("Filter allows:");
    println!(
        "  LlmStart: {}",
        filtered.handles_phase(&CallbackPhase::LlmStart)
    );
    println!(
        "  LlmEnd: {}",
        filtered.handles_phase(&CallbackPhase::LlmEnd)
    );
    println!(
        "  ChainStart: {}",
        filtered.handles_phase(&CallbackPhase::ChainStart)
    );
    println!(
        "  ToolStart: {}",
        filtered.handles_phase(&CallbackPhase::ToolStart)
    );

    filter_manager.add_handler(Box::new(filtered));

    // Emit a mix of events
    filter_manager.emit_phase(CallbackPhase::ChainStart, "f1", json!({}));
    filter_manager.emit_phase(CallbackPhase::LlmStart, "f2", json!({"model": "filtered"}));
    filter_manager.emit_phase(CallbackPhase::ToolStart, "f3", json!({}));
    filter_manager.emit_phase(CallbackPhase::LlmEnd, "f2", json!({"tokens": 75}));
    filter_manager.emit_phase(CallbackPhase::ChainEnd, "f1", json!({}));

    println!("\nEmitted 5 events (ChainStart, LlmStart, ToolStart, LlmEnd, ChainEnd)");
    println!(
        "Filtered handler received {} events (only LLM phases):",
        filter_console.logs().len()
    );
    for log in &filter_console.logs() {
        println!("  {}", log);
    }

    // Empty filter blocks everything
    let empty_console = Arc::new(ConsoleCallbackHandler::new());
    let empty_filter = FilteredHandler::new(Box::new(SharedConsole(empty_console.clone())), vec![]);
    println!("\nEmpty filter blocks all phases:");
    println!(
        "  ChainStart: {}",
        empty_filter.handles_phase(&CallbackPhase::ChainStart)
    );
    println!(
        "  LlmStart: {}",
        empty_filter.handles_phase(&CallbackPhase::LlmStart)
    );

    // Custom phase filtering
    let custom_filter = FilteredHandler::new(
        Box::new(ConsoleCallbackHandler::new()),
        vec![CallbackPhase::Custom("audit_log".into())],
    );
    println!(
        "\nCustom phase filter accepts 'audit_log': {}",
        custom_filter.handles_phase(&CallbackPhase::Custom("audit_log".into()))
    );
    println!(
        "Custom phase filter rejects 'other': {}",
        custom_filter.handles_phase(&CallbackPhase::Custom("other".into()))
    );

    // --- Real LLM Demo ---
    println!("\n--- Real LLM Demo ---\n");
    println!("Using an LLM call with callback instrumentation.\n");

    let cb_console = Arc::new(ConsoleCallbackHandler::new());
    let cb_metrics = Arc::new(MetricsCallbackHandler::new());
    let cb_manager = CallbackManager::new();
    cb_manager.add_handler(Box::new(SharedConsole(cb_console.clone())));
    cb_manager.add_handler(Box::new(SharedMetrics(cb_metrics.clone())));

    let run_id = CallbackManager::new_run_id();
    cb_manager.emit_phase(
        CallbackPhase::LlmStart,
        &run_id,
        json!({"model": "shared_model"}),
    );

    let model = shared::get_chat_model(vec![
        "Callbacks are essential for observability in LLM applications.".into(),
    ]);
    let messages = vec![Message::human(
        "Why are callbacks important in LLM frameworks?",
    )];
    let result = model._generate(&messages, None).await?;

    if let Some(gen) = result.generations.first() {
        let response_text = gen.message.content().text();
        cb_manager.emit_phase(
            CallbackPhase::LlmEnd,
            &run_id,
            json!({"response": response_text}),
        );
        println!("LLM Response: {}", response_text);
    }

    println!("\nCallback logs from LLM call:");
    for log in &cb_console.logs() {
        println!("  {}", log);
    }
    println!("Total callback events: {}", cb_metrics.total_events());

    println!("\n=== Callback Manager Example Complete ===");
    Ok(())
}
