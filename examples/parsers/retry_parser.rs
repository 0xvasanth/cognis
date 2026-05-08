//! RetryParser — loops fixer + parse up to N times until the inner
//! parser succeeds, or surfaces the last parse error.

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
    // Fake fixer that returns garbage twice, then valid JSON.
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
