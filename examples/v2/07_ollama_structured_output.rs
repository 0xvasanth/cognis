//! What you'll learn:
//!   How to derive a JSON-schema-aware prompt from a Rust struct, then
//!   parse the model's reply back into that struct with
//!   `StructuredOutputParser<T>`.
//!
//! Why this matters:
//!   Free-form LLM text is a mess to consume from code. By describing
//!   the shape with `serde::Deserialize + JsonSchema`, you get a typed
//!   value out — and a parser that's robust to chatty models that wrap
//!   their JSON in prose or fenced code blocks.
//!
//! Scenario:
//!   Ask the model for a scrambled-eggs recipe, then parse the reply
//!   into a typed `Recipe { title, ingredients, steps }` struct ready
//!   to render in a UI or hand to a downstream system.
//!
//! Run with:
//!   COGNIS_PROVIDER=ollama COGNIS_OLLAMA_MODEL=llama3.1 \
//!     cargo run -p cognis-examples --example 07_ollama_structured_output
//!
//! Sample output (against ollama / llama3.1):
//!   --- raw model output ---
//!   ```json
//!   {
//!     "title": "Scrambled Eggs",
//!     "ingredients": [
//!       "2 eggs",
//!       "Salt and pepper to taste"
//!     ],
//!   ...
//!     2. Whisk them together with a fork.
//!     3. Heat a pan over medium heat.
//!     4. Add the egg mixture to the pan and stir until set.

use cognis::prelude::*;
use cognis_core::output_parsers::{OutputParser, StructuredOutputParser};
use cognis_core::schemars::{self, JsonSchema};
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
        }
    }
    Ok(())
}
