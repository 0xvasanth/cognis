//! Streaming Example
//!
//! Demonstrates how to stream responses from a chat model character by
//! character using FakeListChatModel's _stream method.
//!
//! No API keys required -- uses fake/mock models.

use futures::StreamExt;

use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::language_models::{FakeListChatModel, GenericFakeChatModel};
use rustchain_core::messages::{AIMessage, HumanMessage, Message};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Streaming Example ===\n");

    // --- Part 1: Character-level streaming with FakeListChatModel ---
    //
    // FakeListChatModel streams each character of the response as a separate chunk.
    println!("--- Part 1: Character-level streaming (FakeListChatModel) ---\n");

    let model = FakeListChatModel::new(vec![
        "Hello! I am a streaming assistant.".into(),
    ]);

    let messages = vec![Message::Human(HumanMessage::new("Say hello"))];

    println!("Streaming response character by character:");
    print!("  ");

    let mut stream = model._stream(&messages, None).await?;
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        // Each chunk contains a single character from the response.
        let text = chunk.message.base.content.text();
        print!("{text}");
    }
    println!("\n");

    // --- Part 2: Word-level streaming with GenericFakeChatModel ---
    //
    // GenericFakeChatModel splits responses on whitespace boundaries,
    // yielding words and spaces as separate tokens.
    println!("--- Part 2: Word-level streaming (GenericFakeChatModel) ---\n");

    let model = GenericFakeChatModel::from_messages(vec![
        AIMessage::new("The quick brown fox jumps over the lazy dog"),
    ]);

    let messages = vec![Message::Human(HumanMessage::new("Tell me a sentence"))];

    println!("Streaming response token by token:");
    print!("  ");

    let mut stream = model._stream(&messages, None).await?;
    let mut token_count = 0;
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        let token = chunk.message.base.content.text();
        print!("{token}");
        token_count += 1;
    }
    println!("\n  (received {token_count} tokens)\n");

    // --- Part 3: Simulated streaming with sleep for visual effect ---
    //
    // FakeListChatModel supports an optional sleep delay between chunks.
    println!("--- Part 3: Streaming with simulated latency ---\n");

    let model = FakeListChatModel::new(vec![
        "Rust is blazingly fast!".into(),
    ])
    .with_sleep(30); // 30ms delay before starting (initial delay only)

    let messages = vec![Message::Human(HumanMessage::new("Tell me about Rust"))];

    println!("Streaming with initial delay:");
    print!("  ");

    let mut stream = model._stream(&messages, None).await?;
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        let text = chunk.message.base.content.text();
        print!("{text}");
    }
    println!("\n");

    // --- Part 4: Collecting a streamed response ---
    //
    // You can also collect all chunks into a final string.
    println!("--- Part 4: Collecting streamed chunks ---\n");

    let model = FakeListChatModel::new(vec![
        "Collected output from streaming.".into(),
    ]);

    let messages = vec![Message::Human(HumanMessage::new("Collect this"))];

    let stream = model._stream(&messages, None).await?;
    let chunks: Vec<_> = stream.collect().await;

    let full_text: String = chunks
        .into_iter()
        .filter_map(|r| r.ok())
        .map(|chunk| chunk.message.base.content.text())
        .collect();

    println!("  Collected text: \"{full_text}\"");
    println!("  Length: {} chars", full_text.len());

    println!("\nDone!");
    Ok(())
}
