//! What you'll learn:
//!   How `Client::stream` returns a stream of partial chunks you can
//!   print as they arrive — the building block under any chat UI's
//!   token-by-token render.
//!
//! Why this matters:
//!   Streaming is what makes chat feel responsive. Every provider in
//!   Cognis exposes the same `client.stream(...)` API, so the
//!   consumer side is identical whether you're hitting OpenAI or a
//!   local Ollama daemon. Drop this loop into a websocket handler and
//!   you've got a working chat UI back-end.
//!
//! Scenario:
//!   The user types "explain Rust ownership in 3 lines" into a CLI
//!   chat. We stream the model's reply and print each token the
//!   moment it lands — no buffering, no waiting for the final reply.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example models_streaming_chat
//!
//! Sample output (against ollama / llama3.1):
//!   USER> Explain Rust ownership in 3 lines.
//!   AI>   Here's a concise explanation of Rust ownership:
//!
//!   * Each value in Rust has an owner that is responsible for deallocating the value when it is no longer needed.
//!   * When a value's owner goes out of scope, the value is dropped and its resources are released.
//!   * Ownership can be transferred between variables using methods like `move` or `clone`, which changes the ownership relationship.

use std::io::Write;

use cognis::prelude::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;
    let prompt = "Explain Rust ownership in 3 lines.";

    println!("USER> {prompt}");
    print!("AI>   ");
    std::io::stdout().flush().ok();

    // The stream yields `StreamChunk`s — one per token (or sub-token,
    // depending on the provider). Flush after each so the terminal
    // shows them progressively rather than line-buffered.
    let mut s = client
        .stream(vec![Message::human(prompt.to_string())])
        .await?;
    while let Some(chunk) = s.next().await {
        let chunk = chunk?;
        print!("{}", chunk.content);
        std::io::stdout().flush().ok();
    }
    println!();
    Ok(())
}
