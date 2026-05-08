//! Structured-output parser + Ollama. Uses `StructuredOutputParser<T>`
//! to turn a free-form LLM reply into a typed struct.
//!
//! Usage:
//! ```bash
//! COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.2:1b \
//!   cargo run --example 07_ollama_structured_output -p cognis-examples
//! ```

use cognis::prelude::*;
use cognis_core::output_parsers::{OutputParser, StructuredOutputParser};
use cognis_core::schemars::{self, JsonSchema};
use cognis_llm::Client;
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Recipe {
    /// Recipe title.
    title: String,
    /// Ingredients (free-form list).
    ingredients: Vec<String>,
    /// Numbered steps.
    steps: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    if std::env::var("COGNIS_PROVIDER").is_err() {
        std::env::set_var("COGNIS_PROVIDER", "ollama");
    }
    let client = Client::from_env()?;

    let parser: StructuredOutputParser<Recipe> = StructuredOutputParser::new();
    let format_hint = OutputParser::format_instructions(&parser).unwrap_or_default();
    let prompt = format!("Give me a very simple recipe for scrambled eggs.\n\n{format_hint}");

    let reply = client.invoke(vec![Message::human(prompt)]).await?;
    let raw = reply.content().to_string();

    println!("--- raw model output ---");
    println!("{raw}");
    println!("--- parsed ---");
    match parser.parse(&raw) {
        Ok(recipe) => {
            println!("title: {}", recipe.title);
            println!("ingredients: {:?}", recipe.ingredients);
            println!("steps:");
            for (i, s) in recipe.steps.iter().enumerate() {
                println!("  {}. {s}", i + 1);
            }
        }
        Err(e) => {
            // Smaller models occasionally don't emit clean JSON. Surface
            // the parse error and the raw text rather than panicking.
            eprintln!("parse failed: {e}");
            eprintln!("(Smaller Ollama models sometimes wander off the JSON contract.)");
        }
    }
    Ok(())
}
