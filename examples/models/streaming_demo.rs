//! Streaming Utilities Demo
//!
//! Demonstrates StreamEvent, StreamBuffer, FilterTransformer,
//! TokenAggregator, and StreamStats from `cognis::streaming`.

#[path = "../shared.rs"]
mod shared;

use cognis::streaming::{
    FilterTransformer, StreamBuffer, StreamEvent, StreamStats, StreamTransformer, TokenAggregator,
};
use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- StreamEvent creation ---
    let events = vec![
        StreamEvent::Token("Hello".into()),
        StreamEvent::Token(", world! ".into()),
        StreamEvent::ToolCallStart {
            name: "search".into(),
            id: "call_001".into(),
        },
        StreamEvent::ToolCallArg("{\"query\":\"Rust language\"}".into()),
        StreamEvent::ToolCallEnd,
        StreamEvent::Token("Here are the results.".into()),
        StreamEvent::Done,
    ];

    // --- StreamBuffer: collect events ---
    let mut buffer = StreamBuffer::new();
    for event in &events {
        buffer.push(event.clone());
    }
    println!(
        "Buffer: \"{}\" ({} events, complete: {})",
        buffer.tokens(),
        buffer.event_count(),
        buffer.is_complete()
    );
    for tc in buffer.tool_calls() {
        println!(
            "Tool call: {} (id: {}, args: {})",
            tc.name, tc.id, tc.arguments
        );
    }

    // --- FilterTransformer ---
    let token_filter = FilterTransformer::new().keep_tokens();
    let kept: Vec<_> = events
        .iter()
        .filter_map(|e| token_filter.transform(e.clone()))
        .collect();
    println!(
        "\nToken-only filter kept {} of {} events",
        kept.len(),
        events.len()
    );

    let tool_filter = FilterTransformer::new().keep_tool_calls();
    let kept: Vec<_> = events
        .iter()
        .filter_map(|e| tool_filter.transform(e.clone()))
        .collect();
    println!("Tool-call filter kept {} events", kept.len());

    // --- TokenAggregator: batch tokens ---
    let tokens = [
        "The ", "quick ", "brown ", "fox ", "jumps ", "over ", "the ", "lazy ", "dog.",
    ];
    let mut aggregator = TokenAggregator::new(3);
    print!("\nBatched (size=3): ");
    for token in &tokens {
        if let Some(batch) = aggregator.push(token) {
            print!("[{batch}] ");
        }
    }
    if let Some(remaining) = aggregator.flush() {
        print!("[{remaining}]");
    }
    println!();

    // --- StreamStats ---
    let mut stats = StreamStats::new();
    for event in &events {
        stats.record_event(event);
    }
    println!(
        "\nStats: {} tokens, {:.1} tok/s",
        stats.total_tokens(),
        stats.tokens_per_second()
    );

    // --- LLM integration ---
    let model = shared::get_chat_model(vec![
        "Streaming reduces perceived latency by sending tokens incrementally.".into(),
    ]);
    let result = model
        ._generate(&[Message::human("Why is streaming useful?")], None)
        .await?;
    if let Some(gen) = result.generations.first() {
        let text = gen.message.content().text();
        println!("\nLLM response: {text}");

        let mut llm_buffer = StreamBuffer::new();
        for word in text.split_whitespace() {
            llm_buffer.push(StreamEvent::Token(format!("{word} ")));
        }
        llm_buffer.push(StreamEvent::Done);
        println!("Reconstructed: \"{}\"", llm_buffer.tokens().trim());
    }

    Ok(())
}
