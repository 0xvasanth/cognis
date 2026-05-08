//! Typed structured extraction: free-form text → JSON → typed struct.
//! V2 uses `StructuredOutputParser<T>` directly (no separate
//! StructuredOutputChain wrapper).

use cognis::prelude::*;
use cognis_core::output_parsers::{OutputParser, StructuredOutputParser};
use cognis_core::schemars::{self, JsonSchema};
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct MeetingAction {
    /// Person responsible.
    owner: String,
    /// What needs to happen.
    action: String,
    /// ISO date if specified.
    due: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== V2 Structured Extraction ===\n");

    let parser: StructuredOutputParser<MeetingAction> = StructuredOutputParser::new();

    let raw = r#"
        Sure! Extracted action item:
        ```json
        {"owner": "Alice", "action": "ship the report", "due": "2026-05-15"}
        ```
    "#;
    let action: MeetingAction = parser.parse(raw)?;
    println!("Extracted: {action:?}");

    // Show the schema-aware prompt fragment users would embed.
    println!(
        "\n--- format hint to feed to the LLM ---\n{}",
        OutputParser::format_instructions(&parser).unwrap_or_default()
    );
    Ok(())
}
