//! Event System Demo
//!
//! Demonstrates the event system from `cognis_core::events`:
//! - Creating Events with different EventTypes
//! - Using EventBus with subscribers
//! - Using EventFilter to filter events by type and source
//! - Using EventLog to record and query events
//! - Using EventHook for before/after transformations
//! - Using EventMetrics to track event processing
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example event_system_demo`

mod shared;
use cognis_core::events::{
    Event, EventBus, EventFilter, EventHook, EventLog, EventMetrics, EventType, HookRegistry,
};
use serde_json::json;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("=== Event System Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. EventTypes
    // -----------------------------------------------------------------------
    println!("--- 1. EventTypes ---\n");

    let types = vec![
        EventType::ChainStart,
        EventType::ChainEnd,
        EventType::LlmStart,
        EventType::LlmEnd,
        EventType::ToolStart,
        EventType::ToolEnd,
        EventType::RetrieverStart,
        EventType::RetrieverEnd,
        EventType::Error,
        EventType::Custom("my_custom_event".into()),
    ];

    for et in &types {
        println!(
            "  EventType: {:15} (name: {})",
            format!("{:?}", et),
            et.name()
        );
    }

    println!("\nEventType display: {}", EventType::LlmStart);
    let chain_start_a = EventType::ChainStart;
    let chain_start_b = EventType::ChainStart;
    println!(
        "EventType equality: ChainStart == ChainStart? {}",
        chain_start_a == chain_start_b
    );
    println!(
        "EventType equality: ChainStart == ChainEnd? {}",
        EventType::ChainStart == EventType::ChainEnd
    );

    // -----------------------------------------------------------------------
    // 2. Creating Events
    // -----------------------------------------------------------------------
    println!("\n--- 2. Creating Events ---\n");

    let chain_event = Event::new(
        EventType::ChainStart,
        json!({"chain_name": "qa_chain", "input": "What is Rust?"}),
    )
    .with_source("main_pipeline")
    .with_metadata("run_id", json!("run-001"))
    .with_metadata("environment", json!("production"));

    println!("Chain event:");
    println!("  ID:         {}", chain_event.id);
    println!("  Type:       {}", chain_event.event_type);
    println!("  Source:     {:?}", chain_event.source);
    println!("  Timestamp:  {}", chain_event.timestamp);
    println!("  Data:       {}", chain_event.data);
    println!("  Metadata:   {:?}", chain_event.metadata);

    let llm_event = Event::new(
        EventType::LlmStart,
        json!({"model": "claude-3", "temperature": 0.7}),
    )
    .with_source("llm_provider");

    let tool_event = Event::new(
        EventType::ToolStart,
        json!({"tool": "calculator", "input": "2 + 2"}),
    )
    .with_source("tool_executor");

    let error_event = Event::new(
        EventType::Error,
        json!({"message": "Rate limit exceeded", "code": 429}),
    )
    .with_source("api_client")
    .with_metadata("retry_after", json!(30));

    println!("\nLLM event JSON:\n  {}", llm_event.to_json());
    println!("\nUnique IDs: {} != {}", chain_event.id, llm_event.id);

    // -----------------------------------------------------------------------
    // 3. EventBus with Subscribers
    // -----------------------------------------------------------------------
    println!("\n--- 3. EventBus with Subscribers ---\n");

    let mut bus = EventBus::new();

    // Subscriber 1: Logs all chain events
    let chain_log = Rc::new(RefCell::new(Vec::<String>::new()));
    let chain_log_clone = chain_log.clone();
    bus.subscribe(
        "chain_logger".into(),
        vec![EventType::ChainStart, EventType::ChainEnd],
        Box::new(move |event| {
            let msg = format!("[CHAIN] {} - {}", event.event_type, event.data);
            chain_log_clone.borrow_mut().push(msg);
        }),
    );

    // Subscriber 2: Logs all LLM events
    let llm_log = Rc::new(RefCell::new(Vec::<String>::new()));
    let llm_log_clone = llm_log.clone();
    bus.subscribe(
        "llm_logger".into(),
        vec![EventType::LlmStart, EventType::LlmEnd],
        Box::new(move |event| {
            let msg = format!("[LLM] {} - {}", event.event_type, event.data);
            llm_log_clone.borrow_mut().push(msg);
        }),
    );

    // Subscriber 3: Counts errors
    let error_count = Rc::new(RefCell::new(0usize));
    let error_count_clone = error_count.clone();
    bus.subscribe(
        "error_counter".into(),
        vec![EventType::Error],
        Box::new(move |_| {
            *error_count_clone.borrow_mut() += 1;
        }),
    );

    println!("Subscribers registered: {}", bus.subscriber_count());

    // Publish events
    bus.publish(&chain_event);
    bus.publish(&llm_event);
    bus.publish(&tool_event); // No subscriber for ToolStart
    bus.publish(&error_event);
    bus.publish(&Event::new(EventType::ChainEnd, json!({"output": "done"})));
    bus.publish(&Event::new(EventType::LlmEnd, json!({"tokens": 150})));

    println!("Chain log entries: {:?}", *chain_log.borrow());
    println!("LLM log entries:   {:?}", *llm_log.borrow());
    println!("Error count:       {}", *error_count.borrow());

    // Unsubscribe
    bus.unsubscribe("error_counter");
    println!(
        "\nAfter unsubscribing error_counter: {} subscribers",
        bus.subscriber_count()
    );

    bus.publish(&Event::new(EventType::Error, json!("another error")));
    println!(
        "Error count after unsubscribe: {} (unchanged)",
        *error_count.borrow()
    );

    // -----------------------------------------------------------------------
    // 4. EventFilter
    // -----------------------------------------------------------------------
    println!("\n--- 4. EventFilter ---\n");

    let events = vec![
        Event::new(EventType::LlmStart, json!(null))
            .with_source("openai")
            .with_metadata("model", json!("gpt-4")),
        Event::new(EventType::LlmStart, json!(null))
            .with_source("anthropic")
            .with_metadata("model", json!("claude-3")),
        Event::new(EventType::LlmEnd, json!(null)).with_source("openai"),
        Event::new(EventType::ToolStart, json!(null)).with_source("tool_runner"),
        Event::new(EventType::Error, json!(null))
            .with_source("openai")
            .with_metadata("model", json!("gpt-4")),
    ];

    // Filter by type
    let type_filter = EventFilter::new().by_type(EventType::LlmStart);
    let matched: Vec<_> = events.iter().filter(|e| type_filter.matches(e)).collect();
    println!("Filter by LlmStart: {} matches", matched.len());

    // Filter by source
    let source_filter = EventFilter::new().by_source("openai");
    let matched: Vec<_> = events.iter().filter(|e| source_filter.matches(e)).collect();
    println!("Filter by source 'openai': {} matches", matched.len());

    // Filter by type AND source
    let combined_filter = EventFilter::new()
        .by_type(EventType::LlmStart)
        .by_source("anthropic");
    let matched: Vec<_> = events
        .iter()
        .filter(|e| combined_filter.matches(e))
        .collect();
    println!(
        "Filter by LlmStart + source 'anthropic': {} matches",
        matched.len()
    );

    // Filter by metadata
    let meta_filter = EventFilter::new().by_metadata("model", json!("gpt-4"));
    let matched: Vec<_> = events.iter().filter(|e| meta_filter.matches(e)).collect();
    println!("Filter by metadata model=gpt-4: {} matches", matched.len());

    // Multiple types (OR)
    let multi_type = EventFilter::new()
        .by_type(EventType::LlmStart)
        .by_type(EventType::LlmEnd);
    let matched: Vec<_> = events.iter().filter(|e| multi_type.matches(e)).collect();
    println!("Filter by LlmStart OR LlmEnd: {} matches", matched.len());

    // Empty filter matches everything
    let empty_filter = EventFilter::new();
    let matched: Vec<_> = events.iter().filter(|e| empty_filter.matches(e)).collect();
    println!("Empty filter (matches all): {} matches", matched.len());

    // -----------------------------------------------------------------------
    // 5. EventLog
    // -----------------------------------------------------------------------
    println!("\n--- 5. EventLog ---\n");

    let mut log = EventLog::new();
    println!(
        "Empty log: {} events, is_empty={}",
        log.len(),
        log.is_empty()
    );

    // Record events
    log.record(Event::new(EventType::ChainStart, json!({"chain": "qa"})).with_source("pipeline"));
    log.record(Event::new(EventType::LlmStart, json!({"model": "claude-3"})).with_source("llm"));
    log.record(Event::new(EventType::ToolStart, json!({"tool": "search"})).with_source("tools"));
    log.record(Event::new(EventType::LlmEnd, json!({"tokens": 200})).with_source("llm"));
    log.record(
        Event::new(EventType::ToolEnd, json!({"result": "found 5 docs"})).with_source("tools"),
    );
    log.record(
        Event::new(EventType::ChainEnd, json!({"output": "answer"})).with_source("pipeline"),
    );

    println!("Log has {} events", log.len());

    // Query by type
    let llm_events = log.events_by_type(&EventType::LlmStart);
    println!("LlmStart events: {}", llm_events.len());

    let tool_events = log.events_by_type(&EventType::ToolStart);
    println!("ToolStart events: {}", tool_events.len());

    // Filter the log
    let filter = EventFilter::new().by_source("llm");
    let llm_filtered = log.filter(&filter);
    println!("Events from 'llm' source: {}", llm_filtered.len());

    let filter = EventFilter::new()
        .by_type(EventType::ChainStart)
        .by_type(EventType::ChainEnd);
    let chain_events = log.filter(&filter);
    println!("Chain start/end events: {}", chain_events.len());

    // Access all events
    println!("\nAll events in log:");
    for event in log.events() {
        println!(
            "  [{}] {} from {:?}",
            event.timestamp, event.event_type, event.source
        );
    }

    // Serialize log to JSON
    let log_json = log.to_json();
    println!(
        "\nLog JSON array length: {}",
        log_json.as_array().map_or(0, |a| a.len())
    );

    // -----------------------------------------------------------------------
    // 6. EventHook for Before/After Transformations
    // -----------------------------------------------------------------------
    println!("\n--- 6. EventHook for Before/After Transformations ---\n");

    // Hook that adds a timestamp to input and wraps output
    let hook = EventHook::new("request_enricher".into())
        .before(Box::new(|input| {
            let mut enriched = input.clone();
            enriched["enriched"] = json!(true);
            enriched["request_id"] = json!("req-12345");
            enriched
        }))
        .after(Box::new(|output| {
            json!({
                "result": output.clone(),
                "processed": true,
                "status": "success"
            })
        }));

    println!("Hook: {}", hook.name());
    println!("Hook info: {}", hook.to_json());

    let input = json!({"query": "What is Rust?", "model": "gpt-4"});
    let before_result = hook.run_before(&input);
    println!("\nBefore hook:");
    println!("  Input:  {}", input);
    println!("  Output: {}", before_result);

    let raw_output = json!({"answer": "Rust is a systems language"});
    let after_result = hook.run_after(&raw_output);
    println!("\nAfter hook:");
    println!("  Input:  {}", raw_output);
    println!("  Output: {}", after_result);

    // Hook with no callbacks (pass-through)
    let noop_hook = EventHook::new("noop".into());
    let pass_through = noop_hook.run_before(&json!("unchanged"));
    println!("\nNo-op hook pass-through: {}", pass_through);

    // -----------------------------------------------------------------------
    // 7. HookRegistry
    // -----------------------------------------------------------------------
    println!("\n--- 7. HookRegistry ---\n");

    let mut registry = HookRegistry::new();
    println!("Empty registry: {} hooks", registry.len());

    registry.register(EventHook::new("add_context".into()).before(Box::new(|v| {
        let mut obj = v.clone();
        obj["context"] = json!("added by hook");
        obj
    })));

    registry.register(
        EventHook::new("validate".into())
            .before(Box::new(|v| {
                let mut obj = v.clone();
                obj["validated"] = json!(true);
                obj
            }))
            .after(Box::new(|v| {
                let mut obj = v.clone();
                obj["post_validated"] = json!(true);
                obj
            })),
    );

    registry.register(EventHook::new("log_output".into()).after(Box::new(|v| {
        let mut obj = v.clone();
        obj["logged"] = json!(true);
        obj
    })));

    println!("Registry has {} hooks", registry.len());

    // Run all before hooks in sequence
    let input = json!({"request": "test"});
    let processed = registry.run_all_before(&input);
    println!("\nrun_all_before:");
    println!("  Input:  {}", input);
    println!("  Output: {}", processed);

    // Run all after hooks in sequence
    let output = json!({"response": "result"});
    let processed = registry.run_all_after(&output);
    println!("\nrun_all_after:");
    println!("  Input:  {}", output);
    println!("  Output: {}", processed);

    // Look up a specific hook
    if let Some(hook) = registry.get("validate") {
        println!("\nFound hook '{}': {}", hook.name(), hook.to_json());
    }

    // Remove a hook
    registry.remove("log_output");
    println!("After removing 'log_output': {} hooks", registry.len());

    // -----------------------------------------------------------------------
    // 8. EventMetrics
    // -----------------------------------------------------------------------
    println!("\n--- 8. EventMetrics ---\n");

    let mut metrics = EventMetrics::new();

    // Simulate a pipeline execution with timing
    metrics.record_event(&EventType::ChainStart);
    metrics.record_processing_time(&EventType::ChainStart, 5);

    metrics.record_event(&EventType::LlmStart);
    metrics.record_processing_time(&EventType::LlmStart, 250);

    metrics.record_event(&EventType::LlmEnd);
    metrics.record_processing_time(&EventType::LlmEnd, 10);

    metrics.record_event(&EventType::ToolStart);
    metrics.record_processing_time(&EventType::ToolStart, 100);

    metrics.record_event(&EventType::ToolEnd);
    metrics.record_processing_time(&EventType::ToolEnd, 5);

    // Second LLM call
    metrics.record_event(&EventType::LlmStart);
    metrics.record_processing_time(&EventType::LlmStart, 350);

    metrics.record_event(&EventType::LlmEnd);
    metrics.record_processing_time(&EventType::LlmEnd, 8);

    metrics.record_event(&EventType::ChainEnd);
    metrics.record_processing_time(&EventType::ChainEnd, 3);

    // Custom event
    let custom_type = EventType::Custom("cache_lookup".into());
    metrics.record_event(&custom_type);
    metrics.record_processing_time(&custom_type, 2);

    println!("Total events recorded: {}", metrics.total_events());

    println!("\nEvents per type:");
    let per_type = metrics.events_per_type();
    let mut sorted_types: Vec<_> = per_type.iter().collect();
    sorted_types.sort_by_key(|(k, _)| (*k).clone());
    for (event_type, count) in &sorted_types {
        println!("  {}: {}", event_type, count);
    }

    println!("\nAverage processing times:");
    let check_types = vec![
        EventType::ChainStart,
        EventType::LlmStart,
        EventType::LlmEnd,
        EventType::ToolStart,
    ];
    for et in &check_types {
        if let Some(avg) = metrics.average_processing_time(et) {
            println!("  {}: {:.1}ms", et.name(), avg);
        }
    }

    println!("\nMetrics JSON: {}", metrics.to_json());

    // -----------------------------------------------------------------------
    // 9. Integration: Full Pipeline Simulation
    // -----------------------------------------------------------------------
    println!("\n--- 9. Integration: Full Pipeline Simulation ---\n");

    let mut bus = EventBus::new();
    let mut log = EventLog::new();
    let mut metrics = EventMetrics::new();

    // Use Rc<RefCell<>> to share the log and metrics with the bus callback
    let log_rc = Rc::new(RefCell::new(log));
    let metrics_rc = Rc::new(RefCell::new(metrics));

    let log_clone = log_rc.clone();
    let metrics_clone = metrics_rc.clone();

    // Subscribe a universal logger/metrics collector
    bus.subscribe(
        "universal_observer".into(),
        vec![
            EventType::ChainStart,
            EventType::ChainEnd,
            EventType::LlmStart,
            EventType::LlmEnd,
            EventType::ToolStart,
            EventType::ToolEnd,
            EventType::Error,
        ],
        Box::new(move |event| {
            log_clone.borrow_mut().record(event.clone());
            metrics_clone.borrow_mut().record_event(&event.event_type);
        }),
    );

    // Simulate a pipeline
    println!("Simulating agent pipeline...\n");

    bus.publish(
        &Event::new(
            EventType::ChainStart,
            json!({"input": "Summarize this article"}),
        )
        .with_source("agent"),
    );
    bus.publish(
        &Event::new(
            EventType::LlmStart,
            json!({"model": "claude-3", "prompt_tokens": 500}),
        )
        .with_source("llm"),
    );
    bus.publish(
        &Event::new(EventType::LlmEnd, json!({"completion_tokens": 200})).with_source("llm"),
    );
    bus.publish(
        &Event::new(
            EventType::ToolStart,
            json!({"tool": "web_search", "query": "article content"}),
        )
        .with_source("tool_executor"),
    );
    bus.publish(
        &Event::new(EventType::ToolEnd, json!({"results": 3})).with_source("tool_executor"),
    );
    bus.publish(
        &Event::new(
            EventType::LlmStart,
            json!({"model": "claude-3", "prompt_tokens": 1200}),
        )
        .with_source("llm"),
    );
    bus.publish(
        &Event::new(EventType::LlmEnd, json!({"completion_tokens": 500})).with_source("llm"),
    );
    bus.publish(
        &Event::new(EventType::ChainEnd, json!({"output": "Summary generated"}))
            .with_source("agent"),
    );

    // Access captured data
    log = log_rc.take();
    metrics = metrics_rc.take();

    println!("Pipeline complete.");
    println!("  Events logged: {}", log.len());
    println!("  Total events in metrics: {}", metrics.total_events());

    // Query the log
    let llm_filter = EventFilter::new()
        .by_type(EventType::LlmStart)
        .by_type(EventType::LlmEnd);
    let llm_events = log.filter(&llm_filter);
    println!("  LLM events: {}", llm_events.len());

    let tool_filter = EventFilter::new().by_source("tool_executor");
    let tool_events = log.filter(&tool_filter);
    println!("  Tool executor events: {}", tool_events.len());

    println!("\n  Events per type: {:?}", metrics.events_per_type());

    // -----------------------------------------------------------------------
    // 10. Real LLM Demo — LLM call with event tracking
    // -----------------------------------------------------------------------
    println!("\n--- 10. Real LLM Demo (LLM Call with Event Tracking) ---\n");

    let model = shared::get_chat_model(vec![
        "Event-driven architectures allow you to decouple components, track execution flow, and build observable LLM pipelines.".into(),
    ]);

    // Set up event tracking for the LLM call
    let mut llm_log = EventLog::new();
    let mut llm_metrics = EventMetrics::new();

    // Record LlmStart event
    let start_event = Event::new(
        EventType::LlmStart,
        json!({"model": "ollama/llama3.2", "question": "What are event-driven architectures?"}),
    )
    .with_source("real_llm_demo");
    llm_log.record(start_event);
    llm_metrics.record_event(&EventType::LlmStart);

    let start_time = std::time::Instant::now();
    let messages = vec![
        cognis_core::messages::Message::System(cognis_core::messages::SystemMessage::new(
            "You are a helpful assistant. Answer concisely in 1-2 sentences.",
        )),
        cognis_core::messages::Message::Human(cognis_core::messages::HumanMessage::new(
            "Why are event-driven architectures useful for LLM applications?",
        )),
    ];

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async { model.invoke_messages(&messages, None).await });
    let elapsed_ms = start_time.elapsed().as_millis() as u64;

    match result {
        Ok(response) => {
            // Record LlmEnd event
            let end_event = Event::new(
                EventType::LlmEnd,
                json!({"response_length": response.base.content.text().len()}),
            )
            .with_source("real_llm_demo");
            llm_log.record(end_event);
            llm_metrics.record_event(&EventType::LlmEnd);
            llm_metrics.record_processing_time(&EventType::LlmEnd, elapsed_ms);

            println!("  LLM response: {}", response.base.content.text());
            println!("  Duration: {}ms", elapsed_ms);
            println!("  Events logged: {}", llm_log.len());
            println!("  Total events tracked: {}", llm_metrics.total_events());
        }
        Err(e) => {
            let error_event = Event::new(EventType::Error, json!({"error": e.to_string()}))
                .with_source("real_llm_demo");
            llm_log.record(error_event);
            llm_metrics.record_event(&EventType::Error);
            println!("  LLM error: {}", e);
        }
    }

    println!("\n=== Event System Demo Complete ===");
}
