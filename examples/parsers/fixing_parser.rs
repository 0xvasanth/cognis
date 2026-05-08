//! OutputFixingParser — try the inner parser, on failure ask an LLM-like
//! "fixer" Runnable to rewrite the malformed output, then parse the rewrite.
//!
//! In production the fixer is your `Client`; here we use a fake fixer
//! built from a `lambda` so the example runs offline.

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
    // The fake fixer: in real life this is your `Client::invoke` wrapped
    // as a Runnable<String, String>. Returning a clean JSON whatever the
    // input asks.
    let fixer: Arc<dyn Runnable<String, String>> = Arc::new(lambda(|_bad: String| async move {
        Ok::<_, CognisError>(
            r#"{"title":"Scrambled eggs","ingredients":["eggs","butter","salt"]}"#.into(),
        )
    }));

    let parser = OutputFixingParser::new(JsonParser::<Recipe>::new(), fixer);

    // Bad output the inner parser can't handle.
    let bad = "title: Scrambled eggs, ingredients: eggs and butter";
    println!("input (malformed): {bad}");

    // Sync `.parse(...)` doesn't call the fixer — it falls through to inner.
    match parser.parse(bad) {
        Ok(_) => println!("(unexpected) inner parsed bad input"),
        Err(e) => println!("inner parse failed as expected: {e}"),
    }

    // The async path repairs and re-parses.
    let r = parser.parse_with_fix(bad).await?;
    println!("repaired → {} with {} ingredients", r.title, r.ingredients.len());
    Ok(())
}
