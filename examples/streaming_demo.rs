//! Streaming Utilities Demo
//!
//! Demonstrates the streaming utilities from `rustchain::streaming`:
//! - StreamEvent variants (Token, ToolCallStart, ToolCallArg, ToolCallEnd, Done)
//! - StreamBuffer for collecting events and extracting tokens/tool calls
//! - FilterTransformer for filtering events by type
//! - TokenAggregator for batching tokens into larger chunks
//! - StreamStats for tracking throughput and latency statistics
//!
//! No API keys required.
//!
//! Run with: `cargo run -p rustchain-examples --example streaming_demo`

use rustchain::streaming::{
    FilterTransformer, StreamBuffer, StreamEvent, StreamStats, StreamTransformer, TokenAggregator,
};

fn main() {
    println!("=== Streaming Utilities Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Creating StreamEvents
    // -----------------------------------------------------------------------
    println!("--- 1. Creating StreamEvents ---\n");

    let events = vec![
        StreamEvent::Token("Hello".into()),
        StreamEvent::Token(", ".into()),
        StreamEvent::Token("world".into()),
        StreamEvent::Token("! ".into()),
        StreamEvent::ToolCallStart {
            name: "search".into(),
            id: "call_001".into(),
        },
        StreamEvent::ToolCallArg("{\"query\":".into()),
        StreamEvent::ToolCallArg("\"Rust language\"}".into()),
        StreamEvent::ToolCallEnd,
        StreamEvent::Token("Here ".into()),
        StreamEvent::Token("are ".into()),
        StreamEvent::Token("the ".into()),
        StreamEvent::Token("results.".into()),
        StreamEvent::Done,
    ];

    for event in &events {
        println!("  Event: {event}");
    }

    println!("\nChecking terminal status:");
    println!(
        "  Token('Hello') is terminal: {}",
        StreamEvent::Token("Hello".into()).is_terminal()
    );
    println!("  Done is terminal: {}", StreamEvent::Done.is_terminal());
    println!(
        "  Error('timeout') is terminal: {}",
        StreamEvent::Error("timeout".into()).is_terminal()
    );

    println!("\nJSON serialization of a ToolCallStart event:");
    let tc_start = StreamEvent::ToolCallStart {
        name: "calculator".into(),
        id: "call_42".into(),
    };
    println!(
        "  {}",
        serde_json::to_string_pretty(&tc_start.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 2. StreamBuffer — collecting events
    // -----------------------------------------------------------------------
    println!("\n--- 2. StreamBuffer — Collecting Events ---\n");

    let mut buffer = StreamBuffer::new();

    for event in &events {
        buffer.push(event.clone());
    }

    println!("Accumulated tokens: \"{}\"", buffer.tokens());
    println!("Total events received: {}", buffer.event_count());
    println!("Stream complete: {}", buffer.is_complete());

    println!("\nTool calls collected:");
    for tc in buffer.tool_calls() {
        println!("  Tool: {} (id: {})", tc.name, tc.id);
        println!("  Arguments: {}", tc.arguments);
        println!("  Complete: {}", tc.is_complete);
        println!(
            "  JSON: {}",
            serde_json::to_string_pretty(&tc.to_json()).unwrap()
        );
    }

    // Demonstrate clear and reuse
    buffer.clear();
    println!("\nAfter clear — event count: {}", buffer.event_count());

    // -----------------------------------------------------------------------
    // 3. FilterTransformer — filtering by event type
    // -----------------------------------------------------------------------
    println!("\n--- 3. FilterTransformer — Filtering Events ---\n");

    // Filter that keeps only tokens (drops tool call events)
    let token_filter = FilterTransformer::new().keep_tokens();
    println!("Token-only filter:");
    for event in &events {
        if let Some(filtered) = token_filter.transform(event.clone()) {
            println!("  Kept: {filtered}");
        }
    }

    // Filter that keeps only tool call events
    let tool_filter = FilterTransformer::new().keep_tool_calls();
    println!("\nTool-call-only filter:");
    for event in &events {
        if let Some(filtered) = tool_filter.transform(event.clone()) {
            println!("  Kept: {filtered}");
        }
    }

    // Combined filter
    let combined_filter = FilterTransformer::new().keep_tokens().keep_tool_calls();
    println!("\nCombined (tokens + tool calls) filter:");
    let kept_count = events
        .iter()
        .filter(|e| combined_filter.transform((*e).clone()).is_some())
        .count();
    println!("  Kept {kept_count} out of {} events", events.len());

    // -----------------------------------------------------------------------
    // 4. TokenAggregator — batching tokens
    // -----------------------------------------------------------------------
    println!("\n--- 4. TokenAggregator — Batching Tokens ---\n");

    let tokens = vec![
        "The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog.",
    ];

    let mut aggregator = TokenAggregator::new(3);
    println!("Batching {} tokens with batch_size=3:", tokens.len());

    for token in &tokens {
        if let Some(batch) = aggregator.push(token) {
            println!("  Batch emitted: \"{batch}\"");
        }
    }

    // Flush any remaining tokens
    if let Some(remaining) = aggregator.flush() {
        println!("  Flush remaining: \"{remaining}\"");
    }

    // Demonstrate with batch_size=1 (immediate emission)
    println!("\nWith batch_size=1 (immediate):");
    let mut agg1 = TokenAggregator::new(1);
    for token in &tokens[..3] {
        if let Some(batch) = agg1.push(token) {
            println!("  Immediate: \"{batch}\"");
        }
    }

    // -----------------------------------------------------------------------
    // 5. StreamStats — tracking statistics
    // -----------------------------------------------------------------------
    println!("\n--- 5. StreamStats — Tracking Statistics ---\n");

    let mut stats = StreamStats::new();

    // Simulate a stream by recording events
    for event in &events {
        stats.record_event(event);
    }

    println!("Total tokens: {}", stats.total_tokens());
    println!("Total duration: {:?}", stats.total_duration());
    if let Some(latency) = stats.first_token_latency() {
        println!("First token latency: {:?}", latency);
    }
    println!("Tokens per second: {:.1}", stats.tokens_per_second());

    println!("\nStats as JSON:");
    println!(
        "  {}",
        serde_json::to_string_pretty(&stats.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 6. Putting it all together — end-to-end pipeline
    // -----------------------------------------------------------------------
    println!("\n--- 6. End-to-End Pipeline ---\n");

    let stream_events = vec![
        StreamEvent::Token("Let ".into()),
        StreamEvent::Token("me ".into()),
        StreamEvent::Token("search ".into()),
        StreamEvent::Token("for ".into()),
        StreamEvent::Token("that.".into()),
        StreamEvent::ToolCallStart {
            name: "web_search".into(),
            id: "ws_01".into(),
        },
        StreamEvent::ToolCallArg("{\"q\":\"Rust async\"}".into()),
        StreamEvent::ToolCallEnd,
        StreamEvent::Token("Found ".into()),
        StreamEvent::Token("3 ".into()),
        StreamEvent::Token("results.".into()),
        StreamEvent::Done,
    ];

    // Collect in buffer, track stats, and aggregate tokens
    let mut pipeline_buffer = StreamBuffer::new();
    let mut pipeline_stats = StreamStats::new();
    let mut pipeline_agg = TokenAggregator::new(3);
    let filter = FilterTransformer::new().keep_tokens();

    println!("Processing stream through pipeline...\n");

    for event in &stream_events {
        pipeline_stats.record_event(event);
        pipeline_buffer.push(event.clone());

        // Only aggregate tokens (using filter to check)
        if let Some(StreamEvent::Token(t)) = filter.transform(event.clone()) {
            if let Some(batch) = pipeline_agg.push(&t) {
                println!("  Aggregated batch: \"{batch}\"");
            }
        }
    }

    // Flush remaining tokens from aggregator
    if let Some(remaining) = pipeline_agg.flush() {
        println!("  Final batch: \"{remaining}\"");
    }

    println!("\nPipeline results:");
    println!("  Full text: \"{}\"", pipeline_buffer.tokens());
    println!("  Tool calls: {}", pipeline_buffer.tool_calls().len());
    println!("  Total tokens: {}", pipeline_stats.total_tokens());
    println!("  Stream complete: {}", pipeline_buffer.is_complete());

    println!("\n=== Done ===");
}
