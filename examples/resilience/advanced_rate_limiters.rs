//! What you'll learn:
//!   How `SlidingWindowLimiter`, `CostBasedLimiter`, and
//!   `CompositeLimiter` all implement the same `RateLimiter` trait —
//!   so you can stack different policies (requests-per-minute AND
//!   cost-per-minute) and they compose under one `acquire` call.
//!
//! Why this matters:
//!   Real APIs enforce more than one limit at once: requests per
//!   minute *and* tokens per minute, with a separate per-user cap.
//!   For OpenAI gpt-4o that translates into a real-money budget —
//!   you want a hard floor on $/minute. The composite pattern lets
//!   you encode the AND-stack of limits without writing custom
//!   orchestration each time.
//!
//! Scenario:
//!   You're spending money on `gpt-4o`. Build a cost-based limiter
//!   that caps spend at 100 "cost units per minute" (think: cents,
//!   tokens, whatever your unit is), then show three composition
//!   shapes: sliding-window only, cost only, and a composite that
//!   AND-stacks request-rate + cost.
//!
//! Run with:
//!   cargo run -p cognis-examples --example resilience_rate_limiters
//!
//! Sample output (against ollama / llama3.1):
//!   === SlidingWindowLimiter (request-rate only) ===
//!   third acquire blocked for 102.073208ms (window had to slide)
//!
//!   === CostBasedLimiter ($/minute style) ===
//!   after 2 calls: spent = 90 / 100
//!   after refund(20): spent = 70 / 100
//!   after reset:     spent = 0 / 100
//!
//!   === CompositeLimiter (rate AND cost) ===
//!   acquired 10 — both limiters had to permit before this returned

use std::sync::Arc;
use std::time::{Duration, Instant};

use cognis::middleware::{
    CompositeLimiter, CostBasedLimiter, RateLimiter, SlidingWindowLimiter, TokenBucket,
};

#[tokio::main]
async fn main() -> cognis::prelude::Result<()> {
    // 1. Sliding window: bound *requests*, not cost.
    //    "<= 10 permits per 100ms rolling window."
    println!("=== SlidingWindowLimiter (request-rate only) ===");
    let sw = SlidingWindowLimiter::new(10, Duration::from_millis(100));
    sw.acquire(5).await;
    sw.acquire(5).await;
    let t0 = Instant::now();
    sw.acquire(1).await;
    println!("third acquire blocked for {:?} (window had to slide)", t0.elapsed());

    // 2. Cost-based: bound the budget itself. Each `acquire(n)` debits
    //    n cost units; total spend caps at the configured ceiling.
    //    Refunds free up budget; reset wipes the meter.
    println!("\n=== CostBasedLimiter ($/minute style) ===");
    let cb = Arc::new(CostBasedLimiter::new(100));
    cb.acquire(60).await;
    cb.acquire(30).await;
    println!("after 2 calls: spent = {} / 100", cb.spent().await);
    cb.refund(20).await; // refund a partial usage estimate
    println!("after refund(20): spent = {} / 100", cb.spent().await);
    cb.reset().await;
    println!("after reset:     spent = {} / 100", cb.spent().await);

    // 3. Composite: AND-stack — every inner limiter must permit. This
    //    is what production code looks like: a token-bucket for
    //    request rate AND a cost limiter for spend.
    println!("\n=== CompositeLimiter (rate AND cost) ===");
    let comp = CompositeLimiter::new()
        .push(Arc::new(TokenBucket::new(1000.0, 100)) as Arc<dyn RateLimiter>)
        .push(Arc::new(CostBasedLimiter::new(500)) as Arc<dyn RateLimiter>);
    comp.acquire(10).await;
    println!("acquired 10 — both limiters had to permit before this returned");
    Ok(())
}
