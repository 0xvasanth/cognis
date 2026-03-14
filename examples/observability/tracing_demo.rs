//! Tracing Demo — Instrument an LLM call with spans and export the trace.
//!
//! Run with: `cargo run -p cognis-examples --example tracing_demo`

#[path = "../shared.rs"]
mod shared;

use cognis_core::messages::Message;
use cognis_core::tracing::{SpanStatus, TraceCollector, TraceExporter};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tracing Demo ===\n");

    let mut collector = TraceCollector::new();

    // --- Build the trace tree ---
    let trace_id = collector.start_trace("qa_pipeline");
    let root_id = collector
        .get_trace(&trace_id)
        .unwrap()
        .root_span()
        .unwrap()
        .span_id
        .clone();

    // 1. Retrieval span
    let retrieval_id = collector
        .start_child_span(&trace_id, &root_id, "document_retrieval")
        .unwrap();
    collector.set_span_attribute(&trace_id, &retrieval_id, "query", json!("What is Rust?"));
    collector.set_span_attribute(&trace_id, &retrieval_id, "top_k", json!(5));
    collector.finish_span(&trace_id, &retrieval_id);

    // 2. LLM generation span (with token-counting child)
    let llm_span_id = collector
        .start_child_span(&trace_id, &root_id, "llm_generation")
        .unwrap();
    collector.set_span_attribute(&trace_id, &llm_span_id, "model", json!("ollama/llama3.2"));

    let model = shared::get_chat_model(vec![
        "Rust is a systems programming language focused on safety, speed, and concurrency.".into(),
    ]);
    let messages = vec![Message::human("What is Rust? Answer in one sentence.")];
    let result = model._generate(&messages, None).await?;

    if let Some(gen) = result.generations.first() {
        let text = gen.message.content().text();
        collector.set_span_attribute(
            &trace_id,
            &llm_span_id,
            "response_preview",
            json!(text.chars().take(100).collect::<String>()),
        );
        println!("LLM response: {}\n", text);
    }

    // Token-counting child span
    let token_id = collector
        .start_child_span(&trace_id, &llm_span_id, "token_counting")
        .unwrap();
    collector.set_span_attribute(&trace_id, &token_id, "input_tokens", json!(12));
    collector.set_span_attribute(&trace_id, &token_id, "output_tokens", json!(18));
    collector.finish_span(&trace_id, &token_id);
    collector.finish_span(&trace_id, &llm_span_id);

    // 3. Post-processing span (simulate an error)
    let post_id = collector
        .start_child_span(&trace_id, &root_id, "post_processing")
        .unwrap();
    collector.set_span_status(
        &trace_id,
        &post_id,
        SpanStatus::Error("Output validation failed".into()),
    );
    collector.finish_span(&trace_id, &post_id);

    // Finish root
    collector.finish_span(&trace_id, &root_id);

    // --- Export ---
    let trace = collector.get_trace(&trace_id).unwrap();

    println!("--- Trace Summary ---");
    println!("{}", TraceExporter::to_summary(trace));

    println!("--- Trace JSON ---");
    let json_out = TraceExporter::to_json(trace);
    println!("{}", serde_json::to_string_pretty(&json_out)?);

    println!("\n=== Tracing Demo Complete ===");
    Ok(())
}
