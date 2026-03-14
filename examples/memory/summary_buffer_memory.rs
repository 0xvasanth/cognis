//! Summary Buffer Memory Example
//!
//! Demonstrates SummaryBufferMemory which combines a message buffer with
//! periodic summarization when the buffer exceeds a token threshold.

#[path = "../shared.rs"]
mod shared;

use cognis::memory::summary_buffer::{
    SimpleSummarizer, Summarizer, SummaryBufferMemory, SummaryStrategy, TemplateSummarizer,
};
use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- 1. SimpleSummarizer: bullet-point summary ---
    let summarizer = SimpleSummarizer::new();
    let messages = vec![
        Message::human("Capital of France?"),
        Message::ai("Paris."),
        Message::human("And Germany?"),
        Message::ai("Berlin."),
    ];
    let summary = Summarizer::summarize(&summarizer, &messages, None)?;
    println!("Simple summary:\n{summary}\n");

    let new_msgs = vec![Message::human("What about Spain?"), Message::ai("Madrid.")];
    let with_prior = Summarizer::summarize(&summarizer, &new_msgs, Some(&summary))?;
    println!("With prior context:\n{with_prior}\n");

    // --- 2. TemplateSummarizer: custom template ---
    let template_summarizer = TemplateSummarizer::new(
        "=== Summary ===\nPrior: {existing_summary}\n\nRecent:\n{messages}",
    );
    let template_summary = Summarizer::summarize(
        &template_summarizer,
        &messages,
        Some("User asked about European capitals."),
    )?;
    println!("Template summary:\n{template_summary}\n");

    // --- 3. Auto-summarization on threshold ---
    let mut memory = SummaryBufferMemory::new(30, SimpleSummarizer::new());
    memory.add_message(Message::human(
        "Tell me about the history of the Roman Empire",
    ))?;
    memory.add_message(Message::ai(
        "The Roman Empire was one of the largest empires in ancient history",
    ))?;
    memory.add_message(Message::human(
        "What were its greatest achievements in architecture",
    ))?;

    println!(
        "After 3 messages: {} in buffer, has summary: {}",
        memory.message_count(),
        memory.has_summary()
    );
    if memory.has_summary() {
        println!("Summary: {}", memory.current_summary().unwrap());
    }
    println!(
        "Context: {}\n",
        serde_json::to_string_pretty(&memory.get_context())?
    );

    // --- 4. Summary strategies ---
    let mut mem_first =
        SummaryBufferMemory::new(25, SimpleSummarizer::new()).with_strategy(SummaryStrategy::First);
    mem_first.add_message(Message::human("First question about weather patterns"))?;
    mem_first.add_message(Message::ai("Weather patterns vary by region and season"))?;
    mem_first.add_message(Message::human("Tell me more about tropical storms"))?;
    println!(
        "Strategy::First - {} msgs, summary: {}",
        mem_first.message_count(),
        mem_first.has_summary()
    );

    let mut mem_sliding = SummaryBufferMemory::new(15, SimpleSummarizer::new())
        .with_strategy(SummaryStrategy::Sliding(2));
    mem_sliding.add_message(Message::human("Message about AI research"))?;
    mem_sliding.add_message(Message::ai("AI research has advanced significantly"))?;
    mem_sliding.add_message(Message::human("What about machine learning?"))?;
    mem_sliding.add_message(Message::ai("ML is a subset of AI focused on data"))?;
    println!(
        "Strategy::Sliding(2) - {} msgs, summary: {}\n",
        mem_sliding.message_count(),
        mem_sliding.has_summary()
    );

    // --- 5. Builder pattern ---
    let mut mem_built = SummaryBufferMemory::builder()
        .max_token_count(20)
        .summarizer(TemplateSummarizer::new("Summary: {messages}"))
        .human_prefix("User")
        .ai_prefix("Assistant")
        .memory_key("conversation")
        .strategy(SummaryStrategy::Oldest)
        .build();

    mem_built.add_message(Message::human("Help with my Rust project"))?;
    mem_built.add_message(Message::ai("Of course! Happy to help with Rust"))?;
    mem_built.add_message(Message::human("How do I use traits effectively?"))?;
    println!(
        "Builder config:\n{}\n",
        serde_json::to_string_pretty(&mem_built.get_context())?
    );

    // --- 6. Clear and reset ---
    println!(
        "Before clear: {} msgs, summary: {}",
        mem_built.message_count(),
        mem_built.has_summary()
    );
    mem_built.clear();
    println!(
        "After clear: {} msgs, summary: {}\n",
        mem_built.message_count(),
        mem_built.has_summary()
    );

    // --- 7. LLM-generated summary ---
    let model = shared::get_chat_model(vec![
        "The user asked about European capitals: Paris (France), Berlin (Germany), Madrid (Spain)."
            .into(),
    ]);
    let llm_messages = vec![
        Message::system("Summarize the following conversation in one sentence."),
        Message::human("Human: Capital of France?\nAI: Paris.\nHuman: Germany?\nAI: Berlin.\nHuman: Spain?\nAI: Madrid."),
    ];
    let result = model._generate(&llm_messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM summary: {}", gen.message.content().text());
    }

    Ok(())
}
