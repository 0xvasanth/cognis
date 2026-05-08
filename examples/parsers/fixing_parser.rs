//! What you'll learn:
//!   How `OutputFixingParser` tries the inner parser first, then on
//!   failure asks a "fixer" `Runnable` to rewrite the malformed text
//!   and re-parses the rewrite.
//!
//! Why this matters:
//!   Small models occasionally produce invalid JSON. `OutputFixingParser`
//!   re-prompts with the parse error attached and the model usually
//!   fixes it on the second try. The fixer is just another Runnable —
//!   in production it's `Client::invoke` wrapped as
//!   `Runnable<String, String>`, so the same parser swaps cleanly
//!   between offline tests and live model calls.
//!
//! Scenario:
//!   A small model emits "title: Scrambled eggs, ingredients: eggs and
//!   butter" — close to JSON but not parseable. The inner parser
//!   fails; the fixer rewrites the text into clean JSON and the second
//!   parse succeeds.
//!
//! Run with:
//!   cargo run -p cognis-examples --example parsers_fixing
//!
//! Sample output (against ollama / llama3.1):
//!   input (malformed): title: Scrambled eggs, ingredients: eggs and butter
//!   inner parse failed as expected: serialization error: json parse: expected ident at line 1 column 2
//!   repaired -> Scrambled eggs with 3 ingredients

use std::sync::Arc;

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::output_parsers::{JsonParser, OutputFixingParser, OutputParser};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Recipe {
    title: String,
    ingredients: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    // The fixer is `Runnable<String, String>` — in real code, that's
    // your `Client` wrapped to take the raw text and return cleaned
    // text. Here a lambda stands in so the demo is reproducible.
    let fixer: Arc<dyn Runnable<String, String>> = Arc::new(lambda(|_bad: String| async move {
        Ok::<_, CognisError>(
            r#"{"title":"Scrambled eggs","ingredients":["eggs","butter","salt"]}"#.into(),
        )
    }));

    let parser = OutputFixingParser::new(JsonParser::<Recipe>::new(), fixer);

    let bad = "title: Scrambled eggs, ingredients: eggs and butter";
    println!("input (malformed): {bad}");

    // Sync `.parse(...)` doesn't call the fixer — it falls through to inner.
    match parser.parse(bad) {
        Ok(_) => println!("(unexpected) inner parsed bad input"),
        Err(e) => println!("inner parse failed as expected: {e}"),
    }

    // The async path repairs and re-parses.
    let r = parser.parse_with_fix(bad).await?;
    println!(
        "repaired -> {} with {} ingredients",
        r.title,
        r.ingredients.len()
    );
    Ok(())
}
