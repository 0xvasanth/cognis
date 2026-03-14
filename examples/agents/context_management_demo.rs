//! Context Management Demo
//!
//! Shows how to manage a context window for long conversations:
//! messages accumulate, and when the token budget is exceeded,
//! the context is compressed to stay within limits.
//!
//! Run with: `cargo run -p cognis-examples --example context_management_demo`

#[path = "../shared.rs"]
mod shared;

use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::Message;
use cognisagent::context::{
    ContextCompressor, ContextEntry, ContextPolicy, ContextRole, ContextWindow,
};

/// Convert context window entries into LLM messages.
fn to_messages(window: &ContextWindow) -> Vec<Message> {
    window
        .entries()
        .iter()
        .map(|e| match &e.role {
            ContextRole::System => Message::system(&e.content),
            ContextRole::User => Message::human(&e.content),
            ContextRole::Assistant => Message::ai(&e.content),
            ContextRole::Tool(_) | ContextRole::Summary => Message::human(&e.content),
        })
        .collect()
}

/// Send the current context to the model and record the reply.
async fn chat(
    window: &mut ContextWindow,
    model: &dyn BaseChatModel,
    user_msg: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    window.push(ContextEntry::new(ContextRole::User, user_msg));
    let messages = to_messages(window);
    let result = model._generate(&messages, None).await?;
    let reply = result.generations.first().map_or_else(
        || String::from("(no response)"),
        |g| g.message.content().text(),
    );
    window.push(ContextEntry::new(ContextRole::Assistant, &reply));
    Ok(reply)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Context Management Demo ===\n");

    // Set up a small context window (200 tokens) to demonstrate compression.
    let mut window = ContextWindow::new(200);
    window.push(ContextEntry::new(
        ContextRole::System,
        "You are a concise coding assistant.",
    ));

    let model = shared::get_chat_model(vec![
        "Use `std::fs::read_to_string(\"path\")` to read a file.".into(),
        "Use `std::fs::write(\"path\", data)` for simple writes.".into(),
        "Use `serde_json::from_str` to parse JSON in Rust.".into(),
        "Wrap errors with `thiserror` or use `anyhow` for quick prototyping.".into(),
    ]);

    // Simulate a multi-turn conversation that fills the context window.
    let questions = [
        "How do I read a file in Rust?",
        "How do I write to a file?",
        "How do I parse JSON?",
        "What about error handling?",
    ];

    let policy = ContextPolicy::new(200).with_response_reserve(40);
    let compressor = ContextCompressor::new(policy.clone());

    for question in &questions {
        // Check if we need to compress before the next turn.
        if policy.needs_compression(&window) {
            let before = window.total_tokens();
            compressor.compress(&mut window);
            println!(
                "[Compressed context: {} -> {} tokens, {} entries remain]\n",
                before,
                window.total_tokens(),
                window.len(),
            );
        }

        let reply = chat(&mut window, model.as_ref(), question).await?;
        println!("User: {question}");
        println!("Assistant: {reply}");
        println!(
            "  (window: {} entries, {} / {} tokens, {:.0}% used)\n",
            window.len(),
            window.total_tokens(),
            window.max_tokens(),
            window.utilization() * 100.0,
        );
    }

    println!("=== Context Management Demo Complete ===");
    Ok(())
}
