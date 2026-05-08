//! Schema-driven extraction via V2's `StructuredOutputParser<T>`. The
//! V1 `ExtractionChain` has been replaced by this leaner trait-based
//! parser that produces typed values directly.

use cognis::prelude::*;
use cognis_core::output_parsers::{OutputParser, StructuredOutputParser};
use cognis_core::schemars::{self, JsonSchema};
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Person {
    name: String,
    age: u32,
    occupation: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== V2 Structured Extraction ===\n");

    let parser: StructuredOutputParser<Person> = StructuredOutputParser::new();
    let format_hint = OutputParser::format_instructions(&parser).unwrap_or_default();
    println!("Schema fragment for the prompt:\n{format_hint}\n");

    // A model would normally produce this JSON; we hardcode for the demo.
    let raw_llm_output = r#"
        Sure, here is the extracted person:
        {"name": "Alice", "age": 30, "occupation": "engineer"}
    "#;

    let person: Person = parser.parse(raw_llm_output)?;
    println!("Extracted: {person:?}");
    Ok(())
}
