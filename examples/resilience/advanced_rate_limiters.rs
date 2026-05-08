//! SlidingWindow / CostBased / Composite rate limiters — all implement
//! the same RateLimiter trait, so they compose under CompositeLimiter
//! and slot into RateLimit middleware identically.

use std::sync::Arc;
use std::time::{Duration, Instant};

use cognis::middleware::{
    CompositeLimiter, CostBasedLimiter, RateLimiter, SlidingWindowLimiter, TokenBucket,
};

#[tokio::main]
async fn main() -> cognis::prelude::Result<()> {
    // Sliding window: ≤ 10 permits per 100ms rolling window.
    println!("=== SlidingWindowLimiter ===");
    let sw = SlidingWindowLimiter::new(10, Duration::from_millis(100));
    sw.acquire(5).await;
    sw.acquire(5).await;
    let t0 = Instant::now();
    sw.acquire(1).await;
    println!(
        "third acquire blocked for {:?} (window had to slide)",
        t0.elapsed()
    );

    // Cost-based: ≤ 100 cost units total; refund/reset to release.
    println!("\n=== CostBasedLimiter ===");
    let cb = Arc::new(CostBasedLimiter::new(100));
    cb.acquire(60).await;
    cb.acquire(30).await;
    println!("after 2 acquires: spent = {}", cb.spent().await);
    cb.refund(20).await;
    println!("after refund(20): spent = {}", cb.spent().await);
    cb.reset().await;
    println!("after reset:     spent = {}", cb.spent().await);

    // Composite: AND-stack — every inner limiter must permit.
    println!("\n=== CompositeLimiter ===");
    let comp = CompositeLimiter::new()
        .push(Arc::new(TokenBucket::new(1000.0, 100)) as Arc<dyn RateLimiter>)
        .push(Arc::new(CostBasedLimiter::new(500)) as Arc<dyn RateLimiter>);
    comp.acquire(10).await;
    println!("acquired through both — only proceeds if every inner permits");
    Ok(())
}
