//! Proactive rate-limit middleware.
//!
//! Different from [`super::ModelRetry`] (which reacts to `RateLimited`
//! errors). This *prevents* hitting the limit by gating calls through a
//! [`RateLimiter`].
//!
//! The trait is the integration point: token-bucket, leaky-bucket, sliding
//! window, distributed Redis-backed — any impl plugs in. A simple in-process
//! `TokenBucket` ships.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::Mutex;

use cognis_core::Result;
use cognis_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

/// Pluggable rate limiter. The middleware calls `acquire(estimated_tokens)`
/// before delegating to the underlying client; impls may sleep until a
/// permit is available.
#[async_trait]
pub trait RateLimiter: Send + Sync {
    /// Block until the caller may issue a request that consumes
    /// approximately `estimated_tokens` tokens.
    async fn acquire(&self, estimated_tokens: u64);
}

/// Fixed-rate token bucket. Refills at `rate_per_sec` permits/second up to
/// `burst` permits. `acquire(n)` waits until `n` permits are available.
pub struct TokenBucket {
    inner: Mutex<TokenBucketState>,
}

struct TokenBucketState {
    permits: f64,
    capacity: f64,
    rate_per_sec: f64,
    last_refill: Instant,
}

impl TokenBucket {
    /// Build with `rate_per_sec` permits/second and a `burst` cap.
    pub fn new(rate_per_sec: f64, burst: u64) -> Self {
        Self {
            inner: Mutex::new(TokenBucketState {
                permits: burst as f64,
                capacity: burst as f64,
                rate_per_sec,
                last_refill: Instant::now(),
            }),
        }
    }
}

#[async_trait]
impl RateLimiter for TokenBucket {
    async fn acquire(&self, estimated_tokens: u64) {
        let needed = (estimated_tokens.max(1)) as f64;
        loop {
            let wait = {
                let mut s = self.inner.lock().await;
                let now = Instant::now();
                let elapsed = now.duration_since(s.last_refill).as_secs_f64();
                s.permits = (s.permits + elapsed * s.rate_per_sec).min(s.capacity);
                s.last_refill = now;
                if s.permits >= needed {
                    s.permits -= needed;
                    None
                } else {
                    let deficit = needed - s.permits;
                    Some(Duration::from_secs_f64(
                        (deficit / s.rate_per_sec).max(0.001),
                    ))
                }
            };
            match wait {
                None => return,
                Some(d) => tokio::time::sleep(d).await,
            }
        }
    }
}

/// Sliding-window limiter — caps total permits consumed in a rolling
/// time window. Useful for "≤ N requests per minute" / "≤ M tokens per
/// hour" style quotas where the upstream API enforces a hard window.
///
/// Maintains a deque of `(timestamp, permits)` events. On `acquire`,
/// evicts events older than the window, sums the rest, and sleeps until
/// enough headroom opens up.
pub struct SlidingWindowLimiter {
    inner: Mutex<SlidingWindowState>,
    capacity: u64,
    window: Duration,
}

struct SlidingWindowState {
    events: std::collections::VecDeque<(Instant, u64)>,
    used: u64,
}

impl SlidingWindowLimiter {
    /// Cap `capacity` permits across a rolling `window`.
    pub fn new(capacity: u64, window: Duration) -> Self {
        Self {
            inner: Mutex::new(SlidingWindowState {
                events: std::collections::VecDeque::new(),
                used: 0,
            }),
            capacity,
            window,
        }
    }
}

#[async_trait]
impl RateLimiter for SlidingWindowLimiter {
    async fn acquire(&self, estimated_tokens: u64) {
        let need = estimated_tokens.max(1);
        loop {
            let wait = {
                let mut s = self.inner.lock().await;
                let now = Instant::now();
                while let Some(&(t, n)) = s.events.front() {
                    if now.duration_since(t) >= self.window {
                        s.events.pop_front();
                        s.used = s.used.saturating_sub(n);
                    } else {
                        break;
                    }
                }
                if s.used + need <= self.capacity {
                    s.events.push_back((now, need));
                    s.used += need;
                    None
                } else {
                    let oldest = s.events.front().map(|(t, _)| *t);
                    oldest.map(|t| {
                        let elapsed = now.duration_since(t);
                        if elapsed >= self.window {
                            Duration::from_millis(1)
                        } else {
                            self.window - elapsed
                        }
                    })
                }
            };
            match wait {
                None => return,
                Some(d) => tokio::time::sleep(d).await,
            }
        }
    }
}

/// Cost-based limiter — budgets cumulative cost (in cents, dollars,
/// abstract "credits", whatever you put in) and rejects further
/// acquisitions once the cap is reached.
///
/// On overrun, [`acquire`] sleeps until a caller has called
/// [`refund`] (e.g. on cost reconciliation) or until [`reset`] resets
/// the counter for a new period.
pub struct CostBasedLimiter {
    inner: Mutex<CostState>,
    cap: u64,
}

struct CostState {
    spent: u64,
}

impl CostBasedLimiter {
    /// Cap total spend at `cap` units (units are caller-defined).
    pub fn new(cap: u64) -> Self {
        Self {
            inner: Mutex::new(CostState { spent: 0 }),
            cap,
        }
    }

    /// Drop the running total back to zero — call at the start of every
    /// period (e.g. once a day from a scheduled task).
    pub async fn reset(&self) {
        self.inner.lock().await.spent = 0;
    }

    /// Decrement spent by `units` — call after reconciling actual cost
    /// when the estimate over-counted.
    pub async fn refund(&self, units: u64) {
        let mut s = self.inner.lock().await;
        s.spent = s.spent.saturating_sub(units);
    }

    /// Current running total.
    pub async fn spent(&self) -> u64 {
        self.inner.lock().await.spent
    }
}

#[async_trait]
impl RateLimiter for CostBasedLimiter {
    async fn acquire(&self, estimated_tokens: u64) {
        let cost = estimated_tokens.max(1);
        loop {
            {
                let mut s = self.inner.lock().await;
                if s.spent + cost <= self.cap {
                    s.spent += cost;
                    return;
                }
            }
            // Over budget — wait for a refund or reset. Poll politely.
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

/// Composite limiter — every wrapped limiter must permit before the call
/// proceeds. Acquire runs them in declaration order; the slowest wins.
pub struct CompositeLimiter {
    limiters: Vec<Arc<dyn RateLimiter>>,
}

impl CompositeLimiter {
    /// Empty — add limiters with [`CompositeLimiter::push`].
    pub fn new() -> Self {
        Self {
            limiters: Vec::new(),
        }
    }

    /// Append a limiter. Builder-style.
    pub fn push(mut self, limiter: Arc<dyn RateLimiter>) -> Self {
        self.limiters.push(limiter);
        self
    }
}

impl Default for CompositeLimiter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RateLimiter for CompositeLimiter {
    async fn acquire(&self, estimated_tokens: u64) {
        for l in &self.limiters {
            l.acquire(estimated_tokens).await;
        }
    }
}

/// Middleware that calls `RateLimiter::acquire` before each LLM call.
pub struct RateLimit {
    limiter: Arc<dyn RateLimiter>,
    /// Estimator: takes the request payload and predicts token cost.
    /// Default: char-count of all message contents.
    estimator: Arc<dyn Fn(&MiddlewareCtx) -> u64 + Send + Sync>,
}

impl RateLimit {
    /// Build with a limiter and the default char-count estimator.
    pub fn new(limiter: Arc<dyn RateLimiter>) -> Self {
        Self {
            limiter,
            estimator: Arc::new(default_estimator),
        }
    }

    /// Override the cost estimator (e.g. plug in a real tokenizer).
    pub fn with_estimator<F>(mut self, f: F) -> Self
    where
        F: Fn(&MiddlewareCtx) -> u64 + Send + Sync + 'static,
    {
        self.estimator = Arc::new(f);
        self
    }
}

fn default_estimator(ctx: &MiddlewareCtx) -> u64 {
    ctx.messages
        .iter()
        .map(|m| m.content().chars().count() as u64)
        .sum()
}

#[async_trait]
impl Middleware for RateLimit {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let cost = (self.estimator)(&ctx);
        self.limiter.acquire(cost).await;
        next.invoke(ctx).await
    }
    fn name(&self) -> &str {
        "RateLimit"
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests_util::*;
    use super::*;
    use crate::middleware::MiddlewarePipeline;

    use cognis_core::Message;
    use cognis_llm::chat::ChatOptions;
    use cognis_llm::Client;

    #[tokio::test]
    async fn token_bucket_acquires_immediately_when_permits_available() {
        let b = TokenBucket::new(1000.0, 100);
        let start = Instant::now();
        b.acquire(10).await;
        assert!(start.elapsed() < Duration::from_millis(100));
    }

    #[tokio::test]
    async fn token_bucket_blocks_when_drained() {
        let b = TokenBucket::new(50.0, 10); // drain quickly
        b.acquire(10).await;
        let start = Instant::now();
        b.acquire(5).await;
        // Should have waited ~100ms (5 permits / 50/sec).
        assert!(start.elapsed() >= Duration::from_millis(50));
    }

    #[tokio::test]
    async fn middleware_passes_through_when_under_limit() {
        let provider = make_recording_provider("ok");
        let pipe = MiddlewarePipeline::new()
            .push(RateLimit::new(Arc::new(TokenBucket::new(100000.0, 100))))
            .build(Client::new(provider.clone()));
        let r = pipe
            .invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "ok");
    }

    #[tokio::test]
    async fn sliding_window_admits_until_cap_then_blocks_until_window_passes() {
        let l = SlidingWindowLimiter::new(10, Duration::from_millis(100));
        // Fill the window.
        l.acquire(5).await;
        l.acquire(5).await;
        // Next acquire should block ~100ms.
        let start = Instant::now();
        l.acquire(1).await;
        assert!(
            start.elapsed() >= Duration::from_millis(80),
            "expected wait, got {:?}",
            start.elapsed()
        );
    }

    #[tokio::test]
    async fn cost_based_limiter_admits_until_cap() {
        let l = CostBasedLimiter::new(100);
        l.acquire(40).await;
        l.acquire(40).await;
        assert_eq!(l.spent().await, 80);
    }

    #[tokio::test]
    async fn cost_based_limiter_blocks_then_unblocks_on_reset() {
        let l = Arc::new(CostBasedLimiter::new(50));
        l.acquire(50).await;
        let l2 = l.clone();
        let h = tokio::spawn(async move { l2.acquire(10).await });
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert!(!h.is_finished(), "should be blocked while over budget");
        l.reset().await;
        h.await.unwrap();
        assert_eq!(l.spent().await, 10);
    }

    #[tokio::test]
    async fn cost_based_limiter_refund_releases_capacity() {
        let l = CostBasedLimiter::new(100);
        l.acquire(80).await;
        l.refund(50).await;
        assert_eq!(l.spent().await, 30);
        l.acquire(60).await;
        assert_eq!(l.spent().await, 90);
    }

    #[tokio::test]
    async fn composite_limiter_runs_every_inner() {
        let token: Arc<dyn RateLimiter> = Arc::new(TokenBucket::new(1000.0, 100));
        let cost: Arc<dyn RateLimiter> = Arc::new(CostBasedLimiter::new(1000));
        let comp = CompositeLimiter::new().push(token).push(cost);
        comp.acquire(10).await;
        // No assertion on internal state — passing acquire without timeout
        // proves both inner limiters returned.
    }
}
