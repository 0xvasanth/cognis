//! Event System Demo
//!
//! Shows how to use the event bus to observe LLM pipeline activity:
//! create an event bus, register handlers, make an LLM call, and see events fire.
//!
//! Run with: `cargo run -p cognis-examples --example event_system_demo`

#[path = "../shared.rs"]
mod shared;

use cognis_core::events::{Event, EventBus, EventLog, EventMetrics, EventType};
use cognis_core::messages::{HumanMessage, Message, SystemMessage};
use serde_json::json;
use std::cell::RefCell;
use std::rc::Rc;

fn main() {
    println!("=== Event System Demo ===\n");

    // --- Set up observability infrastructure ---

    let log = Rc::new(RefCell::new(EventLog::new()));
    let metrics = Rc::new(RefCell::new(EventMetrics::new()));
    let mut bus = EventBus::new();

    // Register a handler that logs and tracks all LLM-related events
    let log_c = log.clone();
    let metrics_c = metrics.clone();
    bus.subscribe(
        "llm_observer".into(),
        vec![EventType::LlmStart, EventType::LlmEnd, EventType::Error],
        Box::new(move |event| {
            println!("  [event] {} from {:?}", event.event_type, event.source);
            log_c.borrow_mut().record(event.clone());
            metrics_c.borrow_mut().record_event(&event.event_type);
        }),
    );

    // --- Make an LLM call with event tracking ---

    let model = shared::get_chat_model(vec![
        "Event-driven architectures decouple components and enable observability in LLM pipelines."
            .into(),
    ]);

    let messages = vec![
        Message::System(SystemMessage::new(
            "You are a helpful assistant. Answer concisely in 1-2 sentences.",
        )),
        Message::Human(HumanMessage::new(
            "Why are event-driven architectures useful for LLM applications?",
        )),
    ];

    // Fire LlmStart
    bus.publish(
        &Event::new(
            EventType::LlmStart,
            json!({"question": "Why are event-driven architectures useful?"}),
        )
        .with_source("llm_pipeline"),
    );

    let start = std::time::Instant::now();
    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async { model.invoke_messages(&messages, None).await });
    let elapsed_ms = start.elapsed().as_millis() as u64;

    match result {
        Ok(response) => {
            bus.publish(
                &Event::new(
                    EventType::LlmEnd,
                    json!({"response_length": response.base.content.text().len()}),
                )
                .with_source("llm_pipeline"),
            );
            metrics
                .borrow_mut()
                .record_processing_time(&EventType::LlmEnd, elapsed_ms);
            println!("\nLLM response: {}", response.base.content.text());
        }
        Err(e) => {
            bus.publish(
                &Event::new(EventType::Error, json!({"error": e.to_string()}))
                    .with_source("llm_pipeline"),
            );
            println!("\nLLM error: {}", e);
        }
    }

    // --- Review captured events ---

    let log = log.borrow();
    let metrics = metrics.borrow();

    println!("\n--- Summary ---");
    println!("Events captured: {}", log.len());
    println!("Duration: {}ms", elapsed_ms);
    println!("Events per type: {:?}", metrics.events_per_type());
    if let Some(avg) = metrics.average_processing_time(&EventType::LlmEnd) {
        println!("Avg LLM processing time: {:.0}ms", avg);
    }

    println!("\n=== Done ===");
}
