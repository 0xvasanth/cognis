//! What you'll learn:
//!   How to implement the `Runnable<I, O>` trait once and get `invoke`,
//!   `batch`, and `stream` for free.
//!
//! Why this matters:
//!   `Runnable` is the type-safe atom every chain, agent, and graph in
//!   Cognis is built from. Anything you wrap in a Runnable composes with
//!   `.pipe()`, batches across inputs, and streams — without you writing
//!   any concurrency code.
//!
//! Scenario:
//!   A primitive that doubles a number — the smallest possible
//!   `Runnable<I, O>`, used to show the trait shape you'd implement for
//!   your own chain step.
//!
//! Run with:
//!   cargo run -p cognis-examples --example 01_hello_runnable
//!
//! Sample output (against ollama / llama3.1):
//!   invoke: 10
//!   batch: [2, 4, 6, 8]
//!   stream item: 14

use async_trait::async_trait;
use cognis::prelude::*;
use futures::StreamExt;

struct Doubler;

#[async_trait]
impl Runnable<u32, u32> for Doubler {
    async fn invoke(&self, input: u32, _: RunnableConfig) -> Result<u32> {
        Ok(input * 2)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let r = Doubler;

    println!("invoke: {}", r.invoke(5, RunnableConfig::default()).await?);

    let v = r.batch(vec![1, 2, 3, 4], RunnableConfig::default()).await?;
    println!("batch: {:?}", v);

    let mut s = r.stream(7, RunnableConfig::default()).await?;
    while let Some(item) = s.next().await {
        println!("stream item: {}", item?);
    }
    Ok(())
}
