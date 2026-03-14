//! Structured Output Parsing Demo
//!
//! Parses structured JSON from LLM responses, validates against a schema,
//! and repairs malformed output.
//!
//! Run with: `cargo run -p cognis-examples --example structured_parsing_demo`

#[path = "../shared.rs"]
mod shared;

use cognis::output_parsers::structured::{
    JsonType, OutputRepairer, SchemaEnforcer, StructuredParser,
};
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Structured Output Parsing Demo ===\n");

    let parser = StructuredParser::new();
    let repairer = OutputRepairer::new();

    // -- Schema: what we expect from the LLM --
    let schema = SchemaEnforcer::new()
        .require_field("language")
        .require_field("year")
        .require_type("language", JsonType::String)
        .require_type("year", JsonType::Number)
        .require_type("memory_safe", JsonType::Bool)
        .optional_field("paradigm", serde_json::json!("unknown"));

    // -- Ask the LLM for structured JSON --
    let fake_response = r#"```json
{"language": "Rust", "year": 2015, "paradigm": "systems", "memory_safe": true}
```"#;

    let model = shared::get_chat_model(vec![fake_response.into()]);
    let messages = vec![
        Message::system("Always respond with a JSON object inside a markdown code block."),
        Message::human(
            "Give me structured data about the Rust programming language: \
             name, first stable release year, paradigm, and whether it is memory safe.",
        ),
    ];

    let result = model._generate(&messages, None).await?;
    let raw = result
        .generations
        .first()
        .map(|g| g.message.content().text())
        .unwrap_or_default();

    println!("LLM response:\n{raw}\n");

    // -- Repair (handles trailing commas, single quotes, etc.) --
    let repaired = repairer.repair_json(&raw).unwrap_or_else(|_| raw.clone());

    // -- Extract JSON from markdown fences or inline text --
    let parsed = parser.parse_json_block(&repaired)?;
    println!("Parsed: {}", serde_json::to_string_pretty(&parsed)?);

    // -- Validate against schema --
    match schema.validate(&parsed) {
        Ok(validated) => {
            println!("\nValidation passed:");
            println!("{}", serde_json::to_string_pretty(&validated)?);
        }
        Err(violations) => {
            println!("\nValidation failed:");
            for v in &violations {
                println!("  - {v}");
            }
        }
    }

    Ok(())
}
