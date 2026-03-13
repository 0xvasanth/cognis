//! Tracing Demo
//!
//! Demonstrates the tracing and observability system from `cognis_core::tracing`.
//!
//! Features shown:
//! - Creating a TraceCollector to manage traces
//! - Starting traces with root spans
//! - Adding child spans under a parent
//! - Setting attributes on spans
//! - Adding events to spans
//! - Finishing spans and traces
//! - Exporting traces to JSON with TraceExporter
//! - Generating human-readable summaries
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example tracing_demo`

use cognis_core::tracing::{
    Span, SpanEvent, SpanStatus, Trace, TraceCollector, TraceExporter, TraceId,
};
use serde_json::json;

fn main() {
    println!("=== Cognis Tracing Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Create a TraceCollector
    // -----------------------------------------------------------------------
    let mut collector = TraceCollector::new();
    println!("1. Created a TraceCollector\n");

    // -----------------------------------------------------------------------
    // 2. Start a trace (creates a root span automatically)
    // -----------------------------------------------------------------------
    let trace_id = collector.start_trace("llm_pipeline");
    println!("2. Started trace: {}\n", trace_id);

    // -----------------------------------------------------------------------
    // 3. Add a child span for the retrieval step
    // -----------------------------------------------------------------------
    // First, get the root span's ID so we can parent under it.
    let root_span_id = collector
        .get_trace(&trace_id)
        .unwrap()
        .root_span()
        .unwrap()
        .span_id
        .clone();

    let retrieval_span_id = collector
        .start_child_span(&trace_id, &root_span_id, "document_retrieval")
        .expect("trace should exist");
    println!(
        "3. Started child span 'document_retrieval': {}\n",
        retrieval_span_id
    );

    // -----------------------------------------------------------------------
    // 4. Set attributes on the retrieval span
    // -----------------------------------------------------------------------
    collector.set_span_attribute(
        &trace_id,
        &retrieval_span_id,
        "query",
        json!("What is Rust?"),
    );
    collector.set_span_attribute(&trace_id, &retrieval_span_id, "top_k", json!(5));
    println!("4. Set attributes on 'document_retrieval': query, top_k\n");

    // Finish the retrieval span
    collector.finish_span(&trace_id, &retrieval_span_id);
    println!("   Finished 'document_retrieval' span\n");

    // -----------------------------------------------------------------------
    // 5. Add another child span for the LLM call
    // -----------------------------------------------------------------------
    let llm_span_id = collector
        .start_child_span(&trace_id, &root_span_id, "llm_generation")
        .expect("trace should exist");
    println!("5. Started child span 'llm_generation': {}\n", llm_span_id);

    collector.set_span_attribute(&trace_id, &llm_span_id, "model", json!("claude-sonnet"));
    collector.set_span_attribute(&trace_id, &llm_span_id, "temperature", json!(0.7));
    collector.set_span_attribute(&trace_id, &llm_span_id, "max_tokens", json!(1024));

    // Add a grandchild span for token counting
    let token_span_id = collector
        .start_child_span(&trace_id, &llm_span_id, "token_counting")
        .expect("trace should exist");
    collector.set_span_attribute(&trace_id, &token_span_id, "input_tokens", json!(150));
    collector.set_span_attribute(&trace_id, &token_span_id, "output_tokens", json!(320));
    collector.finish_span(&trace_id, &token_span_id);
    println!("   Added grandchild span 'token_counting'\n");

    // Finish the LLM span
    collector.finish_span(&trace_id, &llm_span_id);

    // -----------------------------------------------------------------------
    // 6. Add a span with an error status
    // -----------------------------------------------------------------------
    let error_span_id = collector
        .start_child_span(&trace_id, &root_span_id, "post_processing")
        .expect("trace should exist");
    collector.set_span_status(
        &trace_id,
        &error_span_id,
        SpanStatus::Error("Output validation failed".to_string()),
    );
    collector.finish_span(&trace_id, &error_span_id);
    println!("6. Added span 'post_processing' with error status\n");

    // -----------------------------------------------------------------------
    // 7. Finish the root span
    // -----------------------------------------------------------------------
    collector.finish_span(&trace_id, &root_span_id);
    println!("7. Finished root span\n");

    // -----------------------------------------------------------------------
    // 8. Demonstrate adding events directly on a Span object
    // -----------------------------------------------------------------------
    println!("8. Demonstrating SpanEvent creation:\n");

    let trace_id_manual = TraceId::new();
    let mut span = Span::new(trace_id_manual.clone(), "manual_span_with_events");
    span.set_attribute("component", json!("demo"));

    let event1 = SpanEvent::new("cache_hit")
        .with_attribute("cache_key", json!("doc-abc123"))
        .with_attribute("hit", json!(true));
    span.add_event(event1);

    let event2 = SpanEvent::new("retry_attempt")
        .with_attribute("attempt", json!(2))
        .with_attribute("reason", json!("rate_limited"));
    span.add_event(event2);

    span.finish();
    println!("   Created span with 2 events: 'cache_hit', 'retry_attempt'");
    println!("   Span status: {}", span.status);
    println!("   Events count: {}\n", span.events.len());

    // Add this span to a separate trace for export
    let mut manual_trace = Trace::new();
    manual_trace.add_span(span);

    // -----------------------------------------------------------------------
    // 9. Export the collector's trace to JSON
    // -----------------------------------------------------------------------
    println!("9. Exporting traces:\n");

    let trace = collector.get_trace(&trace_id).expect("trace should exist");

    let json_output = TraceExporter::to_json(trace);
    println!("--- JSON Export (first trace) ---");
    println!("{}", serde_json::to_string_pretty(&json_output).unwrap());

    // -----------------------------------------------------------------------
    // 10. Generate a human-readable summary
    // -----------------------------------------------------------------------
    println!("\n10. Trace Summary:\n");

    let summary = TraceExporter::to_summary(trace);
    println!("{}", summary);

    // Also show summary for the manual trace with events
    println!("\n--- Manual trace summary ---");
    let manual_summary = TraceExporter::to_summary(&manual_trace);
    println!("{}", manual_summary);

    // -----------------------------------------------------------------------
    // 11. Show collector stats
    // -----------------------------------------------------------------------
    println!("\n11. Collector Stats:");
    println!(
        "    Total traces collected: {}",
        collector.all_traces().len()
    );
    println!("    Spans in main trace: {}", trace.span_count());
    println!(
        "    Root span: {}",
        trace.root_span().map(|s| s.name.as_str()).unwrap_or("none")
    );
    println!(
        "    Children of root: {}",
        trace.children_of(&root_span_id).len()
    );

    println!("\n=== Tracing Demo Complete ===");
}
