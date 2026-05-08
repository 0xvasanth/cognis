//! What you'll learn:
//!   How to install an `Observer` on a `RunnableConfig` and capture
//!   every `Event` the runnable emits — start, end, errors — without
//!   touching the runnable itself.
//!
//! Why this matters:
//!   You want to debug why your chain is slow but you don't want to
//!   thread logging calls into every stage. Observers are how Cognis
//!   delivers tracing, metrics, log aggregation, and cost accounting
//!   uniformly: every chain, agent, and graph emits the same `Event`
//!   enum, so the observer you write here works against all of them.
//!
//! Scenario:
//!   A two-stage chain that pretends to fetch and parse data. We
//!   attach a `TimedLogger` observer that prints each event with a
//!   millisecond timestamp — the kind of trace you'd save to disk
//!   while diagnosing a slow production run.
//!
//! Run with:
//!   cargo run -p cognis-examples --example obs_event_system
//!
//! Sample output (against ollama / llama3.1):
//!
//!   output: parsed: 70

use std::sync::Arc;
use std::time::Instant;

use cognis::prelude::*;
use cognis_core::compose::lambda;
use cognis_core::runnable_ext::RunnableExt;
use cognis_core::Observer;

/// Observer that prints every event with a timestamp relative to
/// when it was constructed. Drop one of these in the config and you
/// have a one-line trace of any chain.
struct TimedLogger {
    start: Instant,
}

impl Observer for TimedLogger {
    fn on_event(&self, e: &Event) {
        let elapsed = self.start.elapsed().as_millis();
        match e {
            Event::OnStart { .. } => println!("[{elapsed:>4}ms] start"),
            Event::OnEnd { .. } => println!("[{elapsed:>4}ms] end"),
            Event::OnError { error, .. } => println!("[{elapsed:>4}ms] error: {error}"),
            _ => println!("[{elapsed:>4}ms] {e:?}"),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let logger = Arc::new(TimedLogger {
        start: Instant::now(),
    });
    let mut cfg = RunnableConfig::default();
    cfg.observers.push(logger);

    // Two stages, each with a tiny artificial delay. The observer
    // shows you exactly when each one starts and ends.
    let fetch = lambda(|n: i32| async move {
        tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        Ok::<_, CognisError>(n * 10)
    });
    let parse = lambda(|n: i32| async move {
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        Ok::<_, CognisError>(format!("parsed: {n}"))
    });
    let pipeline = fetch.pipe(parse);

    let out = pipeline.invoke(7, cfg).await?;
    println!("\noutput: {out}");
    Ok(())
}
