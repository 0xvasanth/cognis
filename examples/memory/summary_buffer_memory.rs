//! V2's SummaryBufferMemory: token-budgeted with LLM-summarized
//! overflow. Once the running cost exceeds max_tokens, the oldest
//! messages are folded into a running summary via the supplied client.

use cognis::prelude::*;
use cognis::{Memory, SummaryBufferMemory};
use cognis_llm::Client;

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let client = Client::from_env()?;
    let mut mem = SummaryBufferMemory::new(client, 50)
        .with_system("Be terse.");

    for s in [
        "Tell me a long fact about water.",
        "Tell me another long fact about ice.",
        "And one more about steam, please.",
    ] {
        mem.write(Message::human(s.to_string()));
    }

    println!("needs_compact? {}", mem.needs_compact());
    let folded = mem.compact().await?;
    println!("compacted {folded} messages into the running summary");
    println!("seed after compaction: {} messages", mem.seed().len());
    Ok(())
}
