//! What you'll learn:
//!   How `Retry` wraps any `Runnable` to retry on failure with a
//!   `RetryPolicy` that controls attempt count, initial delay, and
//!   exponential backoff — so transient failures don't reach the
//!   user.
//!
//! Why this matters:
//!   You're calling a third-party endpoint that fails ~30% of the
//!   time with 5xx blips. A typed retry wrapper around the
//!   `Runnable` is the cleanest place to handle them, and it
//!   composes with everything else in the pipeline. User-facing
//!   requests still succeed; the retries happen quietly underneath.
//!
//! Scenario:
//!   A fake third-party "shorten this URL" service that fails on
//!   the first two attempts and succeeds on the third. With a
//!   `Retry` wrapper using exponential backoff (5ms, 10ms, 20ms),
//!   the third attempt lands well under any user-visible timeout.
//!
//! Run with:
//!   cargo run -p cognis-examples --example resilience_error_recovery
//!
//! Sample output (against ollama / llama3.1):
//!   shortened: https://shrt.example/abc123 (https://example.com/very/long/path)
//!   elapsed:   18.28875ms
//!   attempts:  3 (the user only ever saw success)

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::wrappers::{Retry, RetryPolicy};

#[tokio::main]
async fn main() -> Result<()> {
    let attempts = Arc::new(AtomicU32::new(0));
    let attempts2 = attempts.clone();

    // Pretend "POST https://shrt.example/shorten". Fails the first two
    // attempts with a transient 503; succeeds on the third.
    let shorten = lambda(move |url: String| {
        let attempts = attempts2.clone();
        async move {
            let n = attempts.fetch_add(1, Ordering::Relaxed) + 1;
            if n < 3 {
                Err(CognisError::Network {
                    status_code: Some(503),
                    message: format!("upstream blip on attempt {n}"),
                })
            } else {
                Ok::<_, CognisError>(format!("https://shrt.example/abc123 ({url})"))
            }
        }
    });

    // 5 max attempts, 5ms initial, 2x backoff (5 -> 10 -> 20 ...).
    let resilient = Retry::new(
        shorten,
        RetryPolicy::new(5)
            .with_initial_delay(Duration::from_millis(5))
            .with_backoff(2.0),
    );

    let t0 = Instant::now();
    let short = resilient
        .invoke("https://example.com/very/long/path".into(), Default::default())
        .await?;
    let elapsed = t0.elapsed();

    println!("shortened: {short}");
    println!("elapsed:   {elapsed:?}");
    println!("attempts:  {} (the user only ever saw success)", attempts.load(Ordering::Relaxed));
    Ok(())
}
