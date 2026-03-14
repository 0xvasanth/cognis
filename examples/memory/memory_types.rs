//! Memory Types Example
//!
//! Demonstrates ConversationBufferMemory, ConversationWindowMemory,
//! EntityMemory, and TokenBufferMemory.

#[path = "../shared.rs"]
mod shared;

use cognis::memory::token_buffer::{SimpleTokenCounter, TokenBufferMemory};
use cognis::memory::{
    BaseMemory, ConversationBufferMemory, ConversationWindowMemory, EntityMemory,
};
use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- 1. ConversationBufferMemory: stores all messages ---
    let buffer_mem = ConversationBufferMemory::new();
    buffer_mem
        .save_context(
            &Message::human("Hello! My name is Alice."),
            &Message::ai("Hi Alice!"),
        )
        .await?;
    buffer_mem
        .save_context(
            &Message::human("Capital of France?"),
            &Message::ai("Paris."),
        )
        .await?;
    buffer_mem
        .save_context(&Message::human("And Germany?"), &Message::ai("Berlin."))
        .await?;

    let vars = buffer_mem.load_memory_variables().await?;
    let history = vars.get("history").unwrap().as_array().unwrap();
    println!("BufferMemory: {} messages stored", history.len());

    buffer_mem.clear().await?;
    let cleared = buffer_mem.load_memory_variables().await?;
    println!(
        "After clear: {} messages\n",
        cleared.get("history").unwrap().as_array().unwrap().len()
    );

    // --- 2. ConversationWindowMemory: keeps last K turns ---
    let window_mem = ConversationWindowMemory::new(2);
    let turns = [
        ("What is Rust?", "A systems programming language."),
        ("Who created it?", "Graydon Hoare at Mozilla."),
        ("When was v1.0?", "May 15, 2015."),
        ("What about async?", "Stabilized in Rust 1.39."),
    ];
    for (human, ai) in &turns {
        window_mem
            .save_context(&Message::human(*human), &Message::ai(*ai))
            .await?;
    }
    let vars = window_mem.load_memory_variables().await?;
    let history = vars.get("history").unwrap().as_array().unwrap();
    println!(
        "WindowMemory(2): {} messages retained (turns 1-2 dropped)\n",
        history.len()
    );

    // --- 3. EntityMemory: tracks named entities ---
    let entity_mem = EntityMemory::new();
    entity_mem
        .save_context(
            &Message::human("Alice works at Google on the Rust compiler team."),
            &Message::ai("Exciting role for Alice at Google!"),
        )
        .await?;
    entity_mem
        .save_context(
            &Message::human("Bob is Alice's manager at Google."),
            &Message::ai("Great that Bob and Alice work together."),
        )
        .await?;

    println!(
        "EntityMemory: {} entities tracked: {:?}",
        entity_mem.entity_count().await,
        entity_mem.entity_names().await
    );
    let context = entity_mem.get_context("Tell me about Alice and Bob.").await;
    println!("Context:\n  {}\n", context.replace('\n', "\n  "));

    // --- 4. TokenBufferMemory: trims by token count ---
    let token_mem = TokenBufferMemory::new()
        .with_max_tokens(50)
        .with_counter(SimpleTokenCounter::new());
    token_mem
        .save_context(
            &Message::human("Tell me a long story about a brave knight."),
            &Message::ai("Once upon a time, a brave knight rode through forests and mountains."),
        )
        .await?;
    println!(
        "TokenBufferMemory: {} tokens, {} messages",
        token_mem.total_tokens().await,
        token_mem.get_messages().await.len()
    );

    token_mem
        .save_context(
            &Message::human("What happened next?"),
            &Message::ai("The knight discovered a hidden castle in an enchanted valley."),
        )
        .await?;
    println!(
        "After 2nd turn: {} tokens, {} messages (trimmed)\n",
        token_mem.total_tokens().await,
        token_mem.get_messages().await.len()
    );

    // --- 5. Memory with LLM ---
    let llm_mem = ConversationBufferMemory::new();
    let model = shared::get_chat_model(vec![
        "That's great! Rust is an excellent choice.".into(),
        "Your favorite programming language is Rust!".into(),
    ]);

    let user_msg_1 = Message::human("My favorite programming language is Rust");
    let ai_response_1 = model.invoke_messages(&[user_msg_1.clone()], None).await?;
    let ai_msg_1 = Message::ai(&ai_response_1.base.content.text());
    llm_mem.save_context(&user_msg_1, &ai_msg_1).await?;
    println!("Turn 1 - AI: {}", ai_response_1.base.content.text());

    let vars = llm_mem.load_memory_variables().await?;
    let mut messages: Vec<Message> = vars
        .get("history")
        .unwrap()
        .as_array()
        .unwrap()
        .iter()
        .map(|v| serde_json::from_value(v.clone()).unwrap())
        .collect();
    messages.push(Message::human("What's my favorite language?"));

    let ai_response_2 = model.invoke_messages(&messages, None).await?;
    println!("Turn 2 - AI: {}", ai_response_2.base.content.text());

    Ok(())
}
