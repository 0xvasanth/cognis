//! Robust JSON parsing — V2's `StructuredOutputParser` extracts the
//! first balanced JSON object/array out of prose, so chatty model
//! replies parse cleanly.

use cognis::prelude::*;
use cognis_core::output_parsers::{OutputParser, StructuredOutputParser};
use cognis_core::schemars::{self, JsonSchema};
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
struct Sentiment {
    label: String,
    confidence: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    let parser: StructuredOutputParser<Sentiment> = StructuredOutputParser::new();

    // Various dirty inputs the parser should recover from.
    let inputs = [
        r#"{"label": "positive", "confidence": 0.92}"#,
        r#"Sure! Here's the result: {"label": "negative", "confidence": 0.7} Hope that helps."#,
        r#"```json
{"label": "neutral", "confidence": 0.55}
```"#,
    ];

    for raw in inputs {
        match parser.parse(raw) {
            Ok(s) => println!("ok: {s:?}"),
            Err(e) => println!("err: {e}"),
        }
    }
    Ok(())
}
