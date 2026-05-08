//! What you'll learn:
//!   How `RetryParser` loops the fixer + parse cycle up to N times,
//!   surfacing the last parse error if every attempt fails.
//!
//! Why this matters:
//!   Sometimes one rewrite isn't enough — the model returns
//!   different-but-still-broken JSON each pass. `RetryParser` re-runs
//!   the entire prompt N times if the parser fails. Reach for it when
//!   the prompt itself needs revision rather than just JSON repair.
//!
//! Scenario:
//!   A flaky fixer returns garbage on the first two attempts and
//!   well-formed `Person` JSON on the third. `RetryParser` keeps
//!   looping until the inner `JsonParser<Person>` finally succeeds.
//!
//! Run with:
//!   cargo run -p cognis-examples --example parsers_retry
//!
//! Sample output (against ollama / llama3.1):
//!   parsed: Eve (age 29) after 3 fix attempts

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::output_parsers::{JsonParser, RetryParser};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct Person {
    name: String,
    age: u32,
}

#[tokio::main]
async fn main() -> Result<()> {
    // The fixer stands in for `Client::invoke`. Returns garbage twice,
    // then valid JSON — the parser retries until the inner JsonParser
    // succeeds.
    let attempts = Arc::new(AtomicUsize::new(0));
    let attempts2 = attempts.clone();
    let fixer: Arc<dyn Runnable<String, String>> = Arc::new(lambda(move |_: String| {
        let a = attempts2.clone();
        async move {
            let n = a.fetch_add(1, Ordering::Relaxed);
            Ok::<_, CognisError>(if n < 2 {
                "still bad output".into()
            } else {
                r#"{"name":"Eve","age":29}"#.into()
            })
        }
    }));

    let parser = RetryParser::with_retries(JsonParser::<Person>::new(), fixer, 5);
    let p = parser.parse_with_retries("garbage").await?;
    println!(
        "parsed: {} (age {}) after {} fix attempts",
        p.name,
        p.age,
        attempts.load(Ordering::Relaxed)
    );
    Ok(())
}
