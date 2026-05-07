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

use cognis2_core::Result;
use cognis2_llm::chat::ChatResponse;

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

    use cognis2_core::Message;
    use cognis2_llm::chat::ChatOptions;
    use cognis2_llm::Client;

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
}
