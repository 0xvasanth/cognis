//! What you'll learn:
//!   How to consume a `Client` reply token-by-token instead of waiting
//!   for the whole completion.
//!
//! Why this matters:
//!   Streaming is what makes chat UIs feel responsive — and it's also
//!   how you cut perceived latency in long-form agent runs. Every
//!   provider in Cognis exposes the same `client.stream(...)` API, so
//!   the consumer code never has to change.
//!
//! Scenario:
//!   Stream a one-line joke and print tokens as they arrive. The shape
//!   your code takes when wiring an agent reply into a chat UI without
//!   buffering.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example 04_streaming_chat
//!
//! Sample output (against ollama / llama3.1):
//!   A man walked into a library and asked the librarian, "Do you have any books on Pavlov's dogs and Schrödinger's cat?" The librarian replied, "It rings a bell, but I'm not sure if it's here or not."

use cognis::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;
    let mut s = client
        .stream(vec![Message::human("Tell me a one-line joke.")])
        .await?;
    while let Some(chunk) = s.next().await {
        let chunk = chunk?;
        print!("{}", chunk.content);
    }
    println!();
    Ok(())
}
