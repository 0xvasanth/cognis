//! What you'll learn:
//!   How to drive an LLM call through a `.pipe()` chain that ends in
//!   `StructuredOutputParser<T>`, so a free-form blob of meeting notes
//!   comes out as a typed `Vec<ActionItem>` ready to write to your
//!   task tracker.
//!
//! Why this matters:
//!   Extraction (action items, line items, entities) is one of the
//!   most common production LLM jobs. Pairing the schema-aware
//!   format-instructions with `StructuredOutputParser` replaces the
//!   StructuredOutputChain wrapper from older frameworks with a
//!   single composable parser.
//!
//! Scenario:
//!   You're building a meeting bot. After the call, you paste the
//!   notes — "Maya owes the report by Friday; Tom will handle vendor
//!   outreach" — into the agent and it returns a typed list of
//!   `ActionItem { who, what, due }` you can hand straight to Linear.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example chains_structured_extraction
//!
//! Sample output (against ollama / llama3.1):
//!   --- raw model output ---
//!   ```json
//!   [
//!     {
//!       "who": "Maya",
//!       "what": "deliver Q2 report",
//!       "due": "2023-02-24"
//!     },
//!   ...
//!     [2023-02-24] Maya: deliver Q2 report
//!     [(no date)] Tom: handle vendor outreach
//!     [(no date)] someone: update runbook

use cognis::prelude::*;
use cognis_core::output_parsers::{OutputParser, StructuredOutputParser};
use cognis_core::schemars::{self, JsonSchema};
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct ActionItem {
    /// Person responsible.
    who: String,
    /// What they need to do.
    what: String,
    /// ISO date if mentioned, otherwise null.
    due: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = Client::from_env()?;
    let parser: StructuredOutputParser<Vec<ActionItem>> = StructuredOutputParser::new();

    // Embed the schema-aware format hint. This is the same hint your
    // production prompt-template would interpolate.
    let format_hint = OutputParser::format_instructions(&parser).unwrap_or_default();

    let notes = "Standup notes: Maya owes the Q2 report by Friday. Tom \
                 will handle vendor outreach this week. We still need \
                 someone to update the runbook.";

    let prompt = format!(
        "Extract every concrete action item from these meeting notes. \
         For each one, identify the owner, the task, and the due date \
         if mentioned (ISO format or null).\n\n\
         {format_hint}\n\nNotes:\n{notes}"
    );

    let reply = client.invoke(vec![Message::human(prompt)]).await?;
    let raw = reply.content().to_string();
    println!("--- raw model output ---\n{raw}\n");

    match parser.parse(&raw) {
        Ok(items) => {
            println!("--- parsed action items ---");
            for it in &items {
                let due = it.due.as_deref().unwrap_or("(no date)");
                println!("  [{due}] {}: {}", it.who, it.what);
            }
        }
        Err(e) => eprintln!("parse failed: {e}"),
    }
    Ok(())
}
