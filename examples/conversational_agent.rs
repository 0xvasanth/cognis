//! Multi-Turn Conversational Agent with Memory
//!
//! Demonstrates how to use ConversationBufferMemory to maintain context
//! across multiple conversation turns. The memory automatically stores
//! each exchange and provides history to the model on subsequent turns.
//!
//! No API keys required -- uses FakeListChatModel.
//!
//! Run with: cargo run -p rustchain-examples --example conversational_agent

use rustchain::memory::{BaseMemory, ConversationBufferMemory};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::language_models::FakeListChatModel;
use rustchain_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Multi-Turn Conversational Agent ===\n");

    // Step 1: Create a fake model with predefined responses for each turn.
    //
    // In production, replace with ChatAnthropic, ChatOpenAI, etc.
    let model = FakeListChatModel::new(vec![
        "Hello! I'm your Rust assistant. How can I help you today?".into(),
        "Ownership is Rust's key memory management feature. Each value has exactly one owner, and when the owner goes out of scope, the value is dropped. This eliminates garbage collection.".into(),
        "Borrowing lets you reference a value without taking ownership. You can have either one mutable reference OR any number of immutable references at a time. The borrow checker enforces these rules at compile time.".into(),
        "Great question! Lifetimes are annotations that tell the compiler how long references are valid. They prevent dangling references. Most of the time, the compiler infers lifetimes automatically through lifetime elision rules.".into(),
    ]);

    // Step 2: Create conversation memory.
    //
    // ConversationBufferMemory stores every message exchanged.
    // with_return_messages(false) returns history as a formatted string
    // instead of structured messages.
    let memory = ConversationBufferMemory::new()
        .with_return_messages(false)
        .with_memory_key("history");

    println!("Created ConversationBufferMemory (memory_key=\"history\")\n");

    // Step 3: Simulate a multi-turn conversation.
    let user_messages = [
        "Hi! I'm learning Rust.",
        "Can you explain ownership?",
        "What about borrowing?",
        "How do lifetimes relate to borrowing?",
    ];

    for (turn, user_input) in user_messages.iter().enumerate() {
        println!("--- Turn {} ---", turn + 1);

        // Load current memory state.
        let vars = memory.load_memory_variables().await?;
        let history = vars
            .get("history")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        if !history.is_empty() {
            println!("  [Memory] Current history:\n    {}", history.replace('\n', "\n    "));
        } else {
            println!("  [Memory] No history yet.");
        }

        // Create the user message.
        let human_msg = Message::human(*user_input);
        println!("  User: {user_input}");

        // Call the model.
        // In a real app, you'd include the history in the prompt.
        let messages = vec![human_msg.clone()];
        let ai_response = model.invoke_messages(&messages, None).await?;
        let ai_text = ai_response.base.content.text();
        let ai_msg = Message::ai(&ai_text);

        println!("  Assistant: {ai_text}");

        // Save this turn to memory.
        memory.save_context(&human_msg, &ai_msg).await?;
        println!();
    }

    // Step 4: Show the final memory state.
    println!("=== Final Memory State ===\n");
    let final_vars = memory.load_memory_variables().await?;
    let final_history = final_vars
        .get("history")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    println!("{final_history}");

    // Step 5: Demonstrate clearing memory.
    println!("\n--- Clearing memory ---");
    memory.clear().await?;
    let cleared = memory.load_memory_variables().await?;
    let cleared_history = cleared
        .get("history")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    println!("  History after clear: \"{}\"", cleared_history);

    println!("\nDone!");
    Ok(())
}
