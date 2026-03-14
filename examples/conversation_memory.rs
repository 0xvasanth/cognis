//! Conversation Memory Example
//!
//! Demonstrates the conversation memory types from the `conversation` module:
//!
//! - `BufferMemory` — unbounded message storage
//! - `WindowMemory` — sliding window of recent message pairs
//! - `TokenBufferMemory` — token-budget-aware buffer
//! - `SummaryMemory` — running summary plus recent messages
//! - `ConversationStore` — multi-session storage
//! - `MemorySearch` — search through conversation history
//! - `MemoryStats` — aggregate statistics
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example conversation_memory`

mod shared;
use cognis::memory::conversation::{
    BufferMemory, ConversationMessage, ConversationStore, MemorySearch, MemoryStats, MessageRole,
    SummaryMemory, TokenBufferMemory, WindowMemory,
};
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::Message;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Conversation Memory Example ===\n");

    // -----------------------------------------------------------------------
    // 1. BufferMemory — unbounded
    // -----------------------------------------------------------------------
    println!("--- 1. BufferMemory ---");

    let mut buffer = BufferMemory::new();
    buffer.add_message(ConversationMessage::new(
        MessageRole::Human,
        "What is Rust?",
    ));
    buffer.add_message(ConversationMessage::new(
        MessageRole::Ai,
        "Rust is a systems programming language focused on safety and performance.",
    ));
    buffer.add_message(ConversationMessage::new(
        MessageRole::Human,
        "How does ownership work?",
    ));
    buffer.add_message(ConversationMessage::new(
        MessageRole::Ai,
        "Each value in Rust has a single owner.",
    ));

    println!("Buffer messages: {}", buffer.len());
    println!(
        "Prompt:\n{}\n",
        buffer.to_prompt_string("User", "Assistant")
    );

    // -----------------------------------------------------------------------
    // 2. WindowMemory — last N pairs
    // -----------------------------------------------------------------------
    println!("--- 2. WindowMemory (window_size=1) ---");

    let mut window = WindowMemory::new(1);
    window.add_message(ConversationMessage::new(MessageRole::Human, "first"));
    window.add_message(ConversationMessage::new(MessageRole::Ai, "first reply"));
    window.add_message(ConversationMessage::new(MessageRole::Human, "second"));
    window.add_message(ConversationMessage::new(MessageRole::Ai, "second reply"));

    println!("Window messages kept: {}", window.messages().len());
    for msg in window.messages() {
        println!("  {}: {}", msg.role, msg.content);
    }
    println!();

    // -----------------------------------------------------------------------
    // 3. TokenBufferMemory
    // -----------------------------------------------------------------------
    println!("--- 3. TokenBufferMemory (budget=10 tokens) ---");

    let mut tb = TokenBufferMemory::new(10);
    tb.add_message(ConversationMessage::new(
        MessageRole::Human,
        "one two three four",
    ));
    tb.add_message(ConversationMessage::new(MessageRole::Ai, "five six seven"));

    println!(
        "Token buffer: {} messages, {} tokens used, {} remaining",
        tb.messages().len(),
        tb.total_tokens(),
        tb.remaining_tokens()
    );
    println!();

    // -----------------------------------------------------------------------
    // 4. SummaryMemory
    // -----------------------------------------------------------------------
    println!("--- 4. SummaryMemory ---");

    let mut summary = SummaryMemory::new();
    summary.set_summary("The user has been asking about Rust programming.".to_string());
    summary.add_message(ConversationMessage::new(
        MessageRole::Human,
        "What about async?",
    ));
    summary.add_message(ConversationMessage::new(
        MessageRole::Ai,
        "Async Rust uses futures and tokio.",
    ));

    println!("Prompt:\n{}\n", summary.to_prompt_string());

    // -----------------------------------------------------------------------
    // 5. ConversationStore — multi-session
    // -----------------------------------------------------------------------
    println!("--- 5. ConversationStore ---");

    let mut store = ConversationStore::new();
    let s1 = store.create_session("Rust chat");
    let s2 = store.create_session("Python chat");
    store.add_to_session(
        &s1,
        ConversationMessage::new(MessageRole::Human, "Tell me about Rust"),
    );
    store.add_to_session(
        &s2,
        ConversationMessage::new(MessageRole::Human, "Tell me about Python"),
    );

    println!("Sessions: {}", store.session_count());
    for (id, name) in store.list_sessions() {
        let count = store.get_session(&id).map(|m| m.len()).unwrap_or(0);
        println!("  {} ({}): {} messages", name, &id[..8], count);
    }
    println!();

    // -----------------------------------------------------------------------
    // 6. MemorySearch
    // -----------------------------------------------------------------------
    println!("--- 6. MemorySearch ---");

    let search = MemorySearch::new();
    let results = search.search_by_content(buffer.messages(), "rust");
    println!("Messages containing 'rust': {}", results.len());

    let human_msgs = search.search_by_role(buffer.messages(), &MessageRole::Human);
    println!("Human messages: {}", human_msgs.len());
    println!();

    // -----------------------------------------------------------------------
    // 7. MemoryStats
    // -----------------------------------------------------------------------
    println!("--- 7. MemoryStats ---");

    let stats = MemoryStats::from_messages(buffer.messages());
    println!(
        "Stats: {}",
        serde_json::to_string_pretty(&stats.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 8. Metadata on messages
    // -----------------------------------------------------------------------
    println!("\n--- 8. Metadata ---");

    let msg = ConversationMessage::new(MessageRole::Human, "What is pi?")
        .with_metadata("topic", json!("math"))
        .with_metadata("confidence", json!(0.99));
    println!("Message JSON: {}", msg.to_json());

    // --- Real LLM Demo ---
    println!("\n--- Real LLM Demo ---\n");
    println!("Multi-turn conversation with memory.\n");

    let model = shared::get_chat_model(vec![
        "Rust is a systems programming language focused on safety, speed, and concurrency.".into(),
        "The borrow checker is Rust's compile-time mechanism that enforces ownership rules to prevent data races and memory errors.".into(),
    ]);

    let mut conv_buffer = BufferMemory::new();

    // Turn 1
    let q1 = "What is Rust?";
    conv_buffer.add_message(ConversationMessage::new(MessageRole::Human, q1));
    let messages = vec![Message::human(q1)];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        let reply = gen.message.content().text();
        conv_buffer.add_message(ConversationMessage::new(MessageRole::Ai, &reply));
        println!("Turn 1 - Q: {}", q1);
        println!("Turn 1 - A: {}\n", reply);
    }

    // Turn 2
    let q2 = "What is the borrow checker?";
    conv_buffer.add_message(ConversationMessage::new(MessageRole::Human, q2));
    let turn2_msgs: Vec<Message> = conv_buffer.messages().iter().map(|m| {
        match m.role {
            MessageRole::Human => Message::human(&m.content),
            MessageRole::Ai => Message::ai(&m.content),
            _ => Message::human(&m.content),
        }
    }).collect();
    // Remove last since it's the question we just added
    let result = model._generate(&turn2_msgs, None).await?;
    if let Some(gen) = result.generations.first() {
        let reply = gen.message.content().text();
        conv_buffer.add_message(ConversationMessage::new(MessageRole::Ai, &reply));
        println!("Turn 2 - Q: {}", q2);
        println!("Turn 2 - A: {}\n", reply);
    }

    println!("Conversation history ({} messages):", conv_buffer.len());
    println!("{}", conv_buffer.to_prompt_string("User", "Assistant"));

    println!("\n=== Conversation Memory Example Complete ===");
    Ok(())
}
