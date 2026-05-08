//! What you'll learn:
//!   That `StructuredOutputParser` finds the first balanced JSON value
//!   inside chatty model output — bare JSON, JSON wrapped in prose, or
//!   JSON inside a fenced ```json block all parse identically.
//!
//! Why this matters:
//!   Real LLM replies almost never come back as "just JSON". The model
//!   adds "Sure!", apologises, wraps the payload in ```json fences. The
//!   parser handles all three transparently so your downstream code
//!   sees a clean typed value instead of a tangle of regex.
//!
//! Scenario:
//!   We replay three real-looking sentiment-analysis replies through
//!   the same parser: bare JSON, JSON buried in prose, and JSON inside
//!   a fenced code block. All three deserialise to `Sentiment`.
//!
//! Run with:
//!   cargo run -p cognis-examples --example chains_structured_parsing
//!
//! Sample output (against ollama / llama3.1):
//!   ok: Sentiment { label: "positive", confidence: 0.92 }
//!   ok: Sentiment { label: "negative", confidence: 0.7 }
//!   ok: Sentiment { label: "neutral", confidence: 0.55 }

use cognis::prelude::*;
use cognis_core::output_parsers::{OutputParser, StructuredOutputParser};
use cognis_core::schemars::{self, JsonSchema};
use serde::Deserialize;

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)] // the printout uses the Debug impl; explicit reads aren't needed.
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
