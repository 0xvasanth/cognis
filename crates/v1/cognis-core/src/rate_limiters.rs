//! Rate limiter interfaces and in-memory token bucket implementation.
//!
//! Mirrors Python `langchain_core.rate_limiters`.

use std::sync::Mutex;
use std::time::Instant;

use async_trait::async_trait;

use crate::error::Result;

/// Abstract interface for rate limiters.
///
/// A rate limiter controls the rate at which requests are made to a resource.
/// Implementations can be blocking (wait until a token is available) or
/// non-blocking (return immediately with a boolean indicating availability).
#[async_trait]
pub trait BaseRateLimiter: Send + Sync {
    /// Acquire a token from the rate limiter.
    ///
    /// If `blocking` is true, waits until a token is available.
    /// Returns true if a token was acquired, false otherwise.
    fn acquire(&self, blocking: bool) -> Result<bool>;

    /// Async version of `acquire`.
    async fn aacquire(&self, blocking: bool) -> Result<bool>;
}

/// In-memory rate limiter based on the token bucket algorithm.
///
/// Tokens are replenished at `requests_per_second` rate, up to
/// `max_bucket_size`. Each `acquire` call consumes one token.
pub struct InMemoryRateLimiter {
    requests_per_second: f64,
    check_every_n_seconds: f64,
    max_bucket_size: f64,
    state: Mutex<TokenBucketState>,
}

struct TokenBucketState {
    available_tokens: f64,
    last_time: Instant,
}

impl InMemoryRateLimiter {
    /// Create a new in-memory rate limiter.
    ///
    /// # Arguments
    /// * `requests_per_second` - Rate at which tokens are replenished.
    /// * `check_every_n_seconds` - How often to poll when blocking.
    /// * `max_bucket_size` - Maximum number of tokens that can accumulate.
    pub fn new(requests_per_second: f64, check_every_n_seconds: f64, max_bucket_size: f64) -> Self {
        Self {
            requests_per_second,
            check_every_n_seconds,
            max_bucket_size,
            state: Mutex::new(TokenBucketState {
                available_tokens: 0.0,
                last_time: Instant::now(),
            }),
        }
    }

    /// Try to consume one token. Returns true if successful.
    fn consume(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        let now = Instant::now();
        let elapsed = now.duration_since(state.last_time).as_secs_f64();
        state.last_time = now;

        // Replenish tokens based on elapsed time, capped at max_bucket_size.
        state.available_tokens =
            (state.available_tokens + elapsed * self.requests_per_second).min(self.max_bucket_size);

        if state.available_tokens >= 1.0 {
            state.available_tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

impl Default for InMemoryRateLimiter {
    fn default() -> Self {
        Self::new(1.0, 0.1, 1.0)
    }
}

#[async_trait]
impl BaseRateLimiter for InMemoryRateLimiter {
    fn acquire(&self, blocking: bool) -> Result<bool> {
        if !blocking {
            return Ok(self.consume());
        }
        loop {
            if self.consume() {
                return Ok(true);
            }
            std::thread::sleep(std::time::Duration::from_secs_f64(
                self.check_every_n_seconds,
            ));
        }
    }

    async fn aacquire(&self, blocking: bool) -> Result<bool> {
        if !blocking {
            return Ok(self.consume());
        }
        loop {
            if self.consume() {
                return Ok(true);
            }
            tokio::time::sleep(std::time::Duration::from_secs_f64(
                self.check_every_n_seconds,
            ))
            .await;
        }
    }
}
