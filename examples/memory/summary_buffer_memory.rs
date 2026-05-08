//! What you'll learn:
//!   How `SummaryBufferMemory` keeps recent turns verbatim but folds
//!   anything over the token budget into a running LLM-generated
//!   summary — so the agent can still answer questions about facts
//!   from way earlier in the conversation.
//!
//! Why this matters:
//!   Long conversations either bust the model's context window or
//!   make every call expensive. Summary-buffer memory is the standard
//!   answer: bounded context, but old turns survive as gist instead
//!   of being dropped entirely. This is what every long-lived support
//!   bot or pair-programming agent ends up using.
//!
//! Scenario:
//!   A customer opens with an order ID ("ORD-9117") and walks through
//!   ten turns of back-and-forth about a delayed shipment. After the
//!   transcript builds up, we ask the agent to recall the order ID —
//!   even though the early turns were folded into the summary, the
//!   gist still carries the number forward.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example memory_summary_buffer
//!
//! Sample output (against ollama / llama3.1):
//!   seed length BEFORE compact: 11
//!   needs_compact?              true
//!   compacted 8 oldest turns into the running summary
//!   seed length AFTER compact:  4
//!
//!   recall test -> The order number is ORD-9117.

use cognis::prelude::*;
use cognis::{Memory, SummaryBufferMemory};

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;

    // 200-char budget is intentionally tight: we want compaction to
    // kick in so the recall test exercises the running-summary path
    // rather than the verbatim-tail path.
    let mut mem = SummaryBufferMemory::new(client.clone(), 200)
        .with_system("You are a friendly support agent. Reply in one short sentence.");

    // Ten turns of a real-feeling support thread. The order ID sits in
    // turn 1 — that's the fact we want preserved through compaction.
    let transcript = [
        (
            "user",
            "Hi, my order ORD-9117 hasn't arrived and it's been two weeks.",
        ),
        (
            "agent",
            "I'm sorry to hear that — let me look into ORD-9117 right away.",
        ),
        (
            "user",
            "It was supposed to be the navy hiking boots, size 11.",
        ),
        (
            "agent",
            "Got it, navy hiking boots size 11, marked for two-day shipping.",
        ),
        ("user", "The tracking page just says 'label created'."),
        (
            "agent",
            "That usually means the carrier never picked it up.",
        ),
        ("user", "Can you reship to the same address?"),
        (
            "agent",
            "Yes — I can dispatch a replacement today via expedited shipping.",
        ),
        ("user", "Will I be charged again?"),
        (
            "agent",
            "No charge for the replacement; you'll get a tracking email by tonight.",
        ),
    ];
    for (role, content) in transcript {
        let m = match role {
            "user" => Message::human(content.to_string()),
            _ => Message::ai(content.to_string()),
        };
        mem.write(m);
    }

    println!("seed length BEFORE compact: {}", mem.seed().len());
    println!("needs_compact?              {}", mem.needs_compact());

    // Force compaction: oldest turns get summarised, newest stay
    // verbatim. In a real agent loop this happens between turns.
    let folded = mem.compact().await?;
    println!("compacted {folded} oldest turns into the running summary");
    println!("seed length AFTER compact:  {}", mem.seed().len());

    // The recall test: the agent's seed now contains a system summary
    // plus a few verbatim tail turns. Even though turn 1 ("ORD-9117")
    // was folded, the summary should still mention the order ID.
    let mut seed = mem.seed();
    seed.push(Message::human(
        "Quick check — what's the order number we've been discussing?",
    ));
    let resp = client.invoke(seed).await?;
    println!("\nrecall test -> {}", resp.content().trim());
    Ok(())
}
