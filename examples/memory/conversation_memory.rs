//! Conversation Memory Example
//!
//! Demonstrates BufferMemory, WindowMemory, TokenBufferMemory, SummaryMemory,
//! ConversationStore, MemorySearch, and MemoryStats.
//!
//! Run with: `cargo run -p cognis-examples --example conversation_memory`

#[path = "../shared.rs"]
mod shared;
use cognis::memory::conversation::{
    BufferMemory, ConversationMessage, ConversationStore, MemorySearch, MemoryStats, MessageRole,
    SummaryMemory, TokenBufferMemory, WindowMemory,
};
use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // BufferMemory - unbounded
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
    println!("Buffer: {} messages", buffer.len());

    // WindowMemory - sliding window
    let mut window = WindowMemory::new(1);
    window.add_message(ConversationMessage::new(MessageRole::Human, "first"));
    window.add_message(ConversationMessage::new(MessageRole::Ai, "first reply"));
    window.add_message(ConversationMessage::new(MessageRole::Human, "second"));
    window.add_message(ConversationMessage::new(MessageRole::Ai, "second reply"));
    println!("Window(1): {} messages kept", window.messages().len());

    // TokenBufferMemory
    let mut tb = TokenBufferMemory::new(10);
    tb.add_message(ConversationMessage::new(
        MessageRole::Human,
        "one two three four",
    ));
    tb.add_message(ConversationMessage::new(MessageRole::Ai, "five six seven"));
    println!(
        "TokenBuffer: {} messages, {} tokens",
        tb.messages().len(),
        tb.total_tokens()
    );

    // SummaryMemory
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
    println!("Summary prompt:\n{}", summary.to_prompt_string());

    // ConversationStore - multi-session
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

    // MemorySearch
    let search = MemorySearch::new();
    let results = search.search_by_content(buffer.messages(), "rust");
    println!("Messages containing 'rust': {}", results.len());

    // MemoryStats
    let stats = MemoryStats::from_messages(buffer.messages());
    println!(
        "Stats: {}",
        serde_json::to_string(&stats.to_json()).unwrap()
    );

    // LLM demo - multi-turn conversation with memory
    let model = shared::get_chat_model(vec![
        "Rust is a systems programming language focused on safety, speed, and concurrency.".into(),
        "The borrow checker enforces ownership rules at compile time to prevent data races.".into(),
    ]);

    let mut conv_buffer = BufferMemory::new();

    let q1 = "What is Rust?";
    conv_buffer.add_message(ConversationMessage::new(MessageRole::Human, q1));
    let result = model._generate(&[Message::human(q1)], None).await?;
    if let Some(gen) = result.generations.first() {
        let reply = gen.message.content().text();
        conv_buffer.add_message(ConversationMessage::new(MessageRole::Ai, &reply));
        println!("Q: {} -> A: {}", q1, reply);
    }

    let q2 = "What is the borrow checker?";
    conv_buffer.add_message(ConversationMessage::new(MessageRole::Human, q2));
    let turn2_msgs: Vec<Message> = conv_buffer
        .messages()
        .iter()
        .map(|m| match m.role {
            MessageRole::Human => Message::human(&m.content),
            MessageRole::Ai => Message::ai(&m.content),
            _ => Message::human(&m.content),
        })
        .collect();
    let result = model._generate(&turn2_msgs, None).await?;
    if let Some(gen) = result.generations.first() {
        let reply = gen.message.content().text();
        conv_buffer.add_message(ConversationMessage::new(MessageRole::Ai, &reply));
        println!("Q: {} -> A: {}", q2, reply);
    }

    println!("Conversation: {} messages total", conv_buffer.len());
    Ok(())
}
