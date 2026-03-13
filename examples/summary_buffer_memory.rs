//! Summary Buffer Memory Example
//!
//! Demonstrates the SummaryBufferMemory system that combines a message buffer
//! with periodic summarization. When the buffer exceeds a configurable token
//! threshold, older messages are automatically summarized to keep the context
//! window manageable while preserving important information.
//!
//! Features shown:
//! - SimpleSummarizer (bullet-point concatenation)
//! - TemplateSummarizer (custom template with placeholders)
//! - Auto-summarization when buffer exceeds token threshold
//! - Context output with summary + recent messages
//! - Summary strategies (Oldest, First, Sliding)
//! - Builder pattern for configuration
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example summary_buffer_memory`

use cognis::memory::summary_buffer::{
    SimpleSummarizer, SummaryBufferMemory, SummaryStrategy, TemplateSummarizer,
};
use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Summary Buffer Memory Example ===\n");

    // -----------------------------------------------------------------------
    // 1. SimpleSummarizer — bullet-point summary without an LLM
    // -----------------------------------------------------------------------
    println!("--- 1. SimpleSummarizer ---");
    println!("Concatenates messages into a bullet-point summary.\n");

    let summarizer = SimpleSummarizer::new();

    let messages = vec![
        Message::human("What is the capital of France?"),
        Message::ai("The capital of France is Paris."),
        Message::human("And what about Germany?"),
        Message::ai("The capital of Germany is Berlin."),
    ];

    let summary =
        cognis::memory::summary_buffer::Summarizer::summarize(&summarizer, &messages, None)?;
    println!("Summary (no prior context):");
    println!("{}\n", summary);

    // With an existing summary to incorporate
    let new_messages = vec![
        Message::human("What about Spain?"),
        Message::ai("The capital of Spain is Madrid."),
    ];
    let summary_with_prior = cognis::memory::summary_buffer::Summarizer::summarize(
        &summarizer,
        &new_messages,
        Some(&summary),
    )?;
    println!("Summary (with prior context):");
    println!("{}\n", summary_with_prior);

    // -----------------------------------------------------------------------
    // 2. TemplateSummarizer — custom template with placeholders
    // -----------------------------------------------------------------------
    println!("--- 2. TemplateSummarizer ---");
    println!(
        "Uses a configurable template with {{messages}} and {{existing_summary}} placeholders.\n"
    );

    let template_summarizer = TemplateSummarizer::new(
        "=== Conversation Summary ===\nPrior: {existing_summary}\n\nRecent:\n{messages}",
    );

    let template_summary = cognis::memory::summary_buffer::Summarizer::summarize(
        &template_summarizer,
        &messages,
        Some("User asked about European capitals."),
    )?;
    println!("Template-based summary:");
    println!("{}\n", template_summary);

    // -----------------------------------------------------------------------
    // 3. Auto-summarization when buffer exceeds threshold
    // -----------------------------------------------------------------------
    println!("--- 3. Auto-Summarization ---");
    println!("Automatically summarizes when estimated token count exceeds the threshold.\n");

    // Use a small token threshold (30) to trigger summarization quickly
    let mut memory = SummaryBufferMemory::new(30, SimpleSummarizer::new());

    println!("Adding messages to buffer (max ~30 tokens)...");

    memory.add_message(Message::human(
        "Tell me about the history of the Roman Empire and its major conquests",
    ))?;
    println!(
        "  After msg 1: {} messages in buffer, has summary: {}",
        memory.message_count(),
        memory.has_summary()
    );

    memory.add_message(Message::ai(
        "The Roman Empire was one of the largest empires in ancient history spanning several centuries",
    ))?;
    println!(
        "  After msg 2: {} messages in buffer, has summary: {}",
        memory.message_count(),
        memory.has_summary()
    );

    memory.add_message(Message::human(
        "What were its greatest achievements in architecture and engineering",
    ))?;
    println!(
        "  After msg 3: {} messages in buffer, has summary: {}",
        memory.message_count(),
        memory.has_summary()
    );

    if memory.has_summary() {
        println!("\nSummarization was triggered!");
        println!("Current summary:");
        println!("  {}", memory.current_summary().unwrap());
        println!("Remaining messages in buffer: {}", memory.message_count());
    }

    println!("\nContext output (JSON):");
    let context = memory.get_context();
    println!("{}\n", serde_json::to_string_pretty(&context)?);

    // -----------------------------------------------------------------------
    // 4. Summary Strategies
    // -----------------------------------------------------------------------
    println!("--- 4. Summary Strategies ---");
    println!("Different strategies control which messages get summarized.\n");

    // Strategy: First (summarize only the oldest message)
    println!("Strategy: First (summarize only the oldest message)");
    let mut mem_first =
        SummaryBufferMemory::new(25, SimpleSummarizer::new()).with_strategy(SummaryStrategy::First);

    mem_first.add_message(Message::human(
        "First question about weather patterns around the world",
    ))?;
    mem_first.add_message(Message::ai(
        "Weather patterns vary greatly by region and season",
    ))?;
    mem_first.add_message(Message::human(
        "Tell me more about tropical storms and hurricanes",
    ))?;

    println!(
        "  Messages remaining: {}, Has summary: {}",
        mem_first.message_count(),
        mem_first.has_summary()
    );

    // Strategy: Sliding (keep only the N most recent messages)
    println!("\nStrategy: Sliding(2) (keep only the 2 most recent messages)");
    let mut mem_sliding = SummaryBufferMemory::new(15, SimpleSummarizer::new())
        .with_strategy(SummaryStrategy::Sliding(2));

    mem_sliding.add_message(Message::human(
        "Message one about artificial intelligence research",
    ))?;
    mem_sliding.add_message(Message::ai(
        "AI research has made significant advances recently",
    ))?;
    mem_sliding.add_message(Message::human("What about machine learning specifically"))?;
    mem_sliding.add_message(Message::ai(
        "Machine learning is a subset of AI focused on data",
    ))?;

    println!(
        "  Messages remaining: {}, Has summary: {}",
        mem_sliding.message_count(),
        mem_sliding.has_summary()
    );
    if mem_sliding.has_summary() {
        println!("  Summary: {}", mem_sliding.current_summary().unwrap());
    }

    // -----------------------------------------------------------------------
    // 5. Builder Pattern
    // -----------------------------------------------------------------------
    println!("\n--- 5. Builder Pattern ---");
    println!("Fluent builder for configuring SummaryBufferMemory.\n");

    let mut mem_built = SummaryBufferMemory::builder()
        .max_token_count(20)
        .summarizer(TemplateSummarizer::new("Summary: {messages}"))
        .human_prefix("User")
        .ai_prefix("Assistant")
        .memory_key("conversation")
        .strategy(SummaryStrategy::Oldest)
        .build();

    mem_built.add_message(Message::human("Hello, I need help with my Rust project"))?;
    mem_built.add_message(Message::ai(
        "Of course! I would be happy to help with your Rust project",
    ))?;
    mem_built.add_message(Message::human(
        "How do I use traits effectively in Rust programming",
    ))?;

    let built_context = mem_built.get_context();
    println!("Custom-configured context (key='conversation', prefixes='User'/'Assistant'):");
    println!("{}\n", serde_json::to_string_pretty(&built_context)?);

    // -----------------------------------------------------------------------
    // 6. Clear and Reset
    // -----------------------------------------------------------------------
    println!("--- 6. Clear and Reset ---");
    println!("Clearing the memory resets both buffer and summary.\n");

    println!(
        "Before clear: {} messages, has summary: {}",
        mem_built.message_count(),
        mem_built.has_summary()
    );
    mem_built.clear();
    println!(
        "After clear:  {} messages, has summary: {}",
        mem_built.message_count(),
        mem_built.has_summary()
    );

    println!("\n=== Summary Buffer Memory Example Complete ===");
    Ok(())
}
