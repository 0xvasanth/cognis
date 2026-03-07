use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, RustChainError};

use super::base::Runnable;
use super::config::RunnableConfig;
use super::RunnableStream;

// ---------------------------------------------------------------------------
// RateLimitConfig
// ---------------------------------------------------------------------------

/// Configuration for token-bucket rate limiting.
///
/// # Example
/// ```ignore
/// use rustchain_core::runnables::rate_limit::RateLimitConfig;
///
/// let config = RateLimitConfig::new(10, 1000)   // 10 requests per 1 000 ms
///     .with_burst_limit(15);
/// ```
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum number of requests allowed within the time window.
    pub max_requests: usize,
    /// Duration of the time window in milliseconds.
    pub window_duration_ms: u64,
    /// Optional burst limit allowing temporary spikes above the steady rate.
    /// When set, the token bucket capacity is `burst_limit` instead of
    /// `max_requests`.
    pub burst_limit: Option<usize>,
}

impl RateLimitConfig {
    /// Create a new rate-limit configuration.
    ///
    /// # Arguments
    /// * `max_requests` — maximum requests allowed per window.
    /// * `window_duration_ms` — window duration in milliseconds.
    pub fn new(max_requests: usize, window_duration_ms: u64) -> Self {
        Self {
            max_requests,
            window_duration_ms,
            burst_limit: None,
        }
    }

    /// Set an optional burst limit.  Must be >= `max_requests` to be useful.
    pub fn with_burst_limit(mut self, burst_limit: usize) -> Self {
        self.burst_limit = Some(burst_limit);
        self
    }
}

// ---------------------------------------------------------------------------
// RateLimiter (token bucket)
// ---------------------------------------------------------------------------

/// Internal mutable state of the token bucket.
#[derive(Debug)]
struct TokenBucketState {
    tokens: usize,
    last_refill: Instant,
}

/// A thread-safe, token-bucket rate limiter.
///
/// Tokens are refilled proportionally based on elapsed time since the last
/// refill.  The bucket capacity equals `burst_limit` (if configured) or
/// `max_requests`.
#[derive(Debug)]
pub struct RateLimiter {
    config: RateLimitConfig,
    state: Mutex<TokenBucketState>,
}

impl RateLimiter {
    /// Create a new `RateLimiter` from the given configuration.
    pub fn new(config: RateLimitConfig) -> Self {
        let capacity = config.burst_limit.unwrap_or(config.max_requests);
        Self {
            state: Mutex::new(TokenBucketState {
                tokens: capacity,
                last_refill: Instant::now(),
            }),
            config,
        }
    }

    /// The maximum capacity of the token bucket.
    fn capacity(&self) -> usize {
        self.config.burst_limit.unwrap_or(self.config.max_requests)
    }

    /// Attempt to acquire a single token.
    ///
    /// Returns `true` if a token was acquired, `false` if the bucket is empty.
    pub fn try_acquire(&self) -> bool {
        let mut state = self.state.lock().unwrap();
        self.refill(&mut state);

        if state.tokens > 0 {
            state.tokens -= 1;
            true
        } else {
            false
        }
    }

    /// Returns the number of tokens currently available (after refill).
    pub fn available_tokens(&self) -> usize {
        let mut state = self.state.lock().unwrap();
        self.refill(&mut state);
        state.tokens
    }

    /// Reset the limiter to full capacity.
    pub fn reset(&self) {
        let mut state = self.state.lock().unwrap();
        state.tokens = self.capacity();
        state.last_refill = Instant::now();
    }

    /// Returns a reference to the underlying configuration.
    pub fn config(&self) -> &RateLimitConfig {
        &self.config
    }

    /// Refill tokens based on elapsed time.
    fn refill(&self, state: &mut TokenBucketState) {
        if self.config.window_duration_ms == 0 || self.config.max_requests == 0 {
            return;
        }

        let now = Instant::now();
        let elapsed_ms = now.duration_since(state.last_refill).as_millis() as u64;

        if elapsed_ms > 0 {
            let tokens_to_add = (elapsed_ms as u128 * self.config.max_requests as u128
                / self.config.window_duration_ms as u128) as usize;

            if tokens_to_add > 0 {
                let capacity = self.capacity();
                state.tokens = (state.tokens + tokens_to_add).min(capacity);
                state.last_refill = now;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SlidingWindowCounter
// ---------------------------------------------------------------------------

/// A sliding-window rate counter that tracks individual request timestamps.
///
/// Unlike the token bucket, this gives a precise count of requests within the
/// most recent window of duration `window_ms`.
#[derive(Debug)]
pub struct SlidingWindowCounter {
    max_requests: usize,
    window_ms: u64,
    timestamps: Mutex<VecDeque<Instant>>,
}

impl SlidingWindowCounter {
    /// Create a new sliding-window counter.
    ///
    /// # Arguments
    /// * `max_requests` — maximum requests allowed within the window.
    /// * `window_ms` — window duration in milliseconds.
    pub fn new(max_requests: usize, window_ms: u64) -> Self {
        Self {
            max_requests,
            window_ms,
            timestamps: Mutex::new(VecDeque::new()),
        }
    }

    /// Record a request.
    ///
    /// Returns `true` if the request is within the limit, `false` if the rate
    /// limit has been exceeded (the request is **not** recorded in that case).
    pub fn record(&self) -> bool {
        let mut timestamps = self.timestamps.lock().unwrap();
        let now = Instant::now();
        let window = Duration::from_millis(self.window_ms);

        // Evict expired entries.
        while let Some(&front) = timestamps.front() {
            if now.duration_since(front) > window {
                timestamps.pop_front();
            } else {
                break;
            }
        }

        if timestamps.len() < self.max_requests {
            timestamps.push_back(now);
            true
        } else {
            false
        }
    }

    /// Returns the number of requests recorded in the current window.
    pub fn current_count(&self) -> usize {
        let mut timestamps = self.timestamps.lock().unwrap();
        let now = Instant::now();
        let window = Duration::from_millis(self.window_ms);

        while let Some(&front) = timestamps.front() {
            if now.duration_since(front) > window {
                timestamps.pop_front();
            } else {
                break;
            }
        }

        timestamps.len()
    }

    /// Reset the counter, clearing all recorded timestamps.
    pub fn reset(&self) {
        self.timestamps.lock().unwrap().clear();
    }
}

// ---------------------------------------------------------------------------
// RunnableRateLimit
// ---------------------------------------------------------------------------

/// A runnable wrapper that enforces rate limiting via a token bucket.
///
/// If the rate limit is exceeded the invocation returns an error immediately
/// rather than waiting.
///
/// # Example
/// ```ignore
/// use rustchain_core::runnables::{RunnableRateLimit, RateLimitConfig};
///
/// let config = RateLimitConfig::new(10, 1000);
/// let limited = RunnableRateLimit::new(inner_runnable, config);
/// ```
pub struct RunnableRateLimit {
    inner: Arc<dyn Runnable>,
    limiter: RateLimiter,
}

impl RunnableRateLimit {
    /// Create a new `RunnableRateLimit` wrapping the given runnable.
    pub fn new(inner: Arc<dyn Runnable>, config: RateLimitConfig) -> Self {
        Self {
            inner,
            limiter: RateLimiter::new(config),
        }
    }

    /// Returns a reference to the underlying rate limiter.
    pub fn limiter(&self) -> &RateLimiter {
        &self.limiter
    }
}

#[async_trait]
impl Runnable for RunnableRateLimit {
    fn name(&self) -> &str {
        "RunnableRateLimit"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        if !self.limiter.try_acquire() {
            return Err(RustChainError::Other("Rate limit exceeded".to_string()));
        }
        self.inner.invoke(input, config).await
    }

    async fn batch(
        &self,
        inputs: Vec<Value>,
        config: Option<&RunnableConfig>,
    ) -> Result<Vec<Value>> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.invoke(input, config).await?);
        }
        Ok(results)
    }

    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        if !self.limiter.try_acquire() {
            return Err(RustChainError::Other("Rate limit exceeded".to_string()));
        }
        self.inner.stream(input, config).await
    }
}

// ---------------------------------------------------------------------------
// RunnableThrottle
// ---------------------------------------------------------------------------

/// A runnable wrapper that enforces a minimum delay between invocations.
///
/// If invoked before `min_interval_ms` has elapsed since the last invocation
/// it sleeps until the interval has passed.
///
/// # Example
/// ```ignore
/// use rustchain_core::runnables::RunnableThrottle;
///
/// // At most one invocation per 100 ms
/// let throttled = RunnableThrottle::new(inner, 100);
/// ```
pub struct RunnableThrottle {
    inner: Arc<dyn Runnable>,
    min_interval: Duration,
    last_invocation: Mutex<Option<Instant>>,
}

impl RunnableThrottle {
    /// Create a new `RunnableThrottle`.
    ///
    /// # Arguments
    /// * `inner` — the runnable to wrap.
    /// * `min_interval_ms` — minimum milliseconds between invocations.
    pub fn new(inner: Arc<dyn Runnable>, min_interval_ms: u64) -> Self {
        Self {
            inner,
            min_interval: Duration::from_millis(min_interval_ms),
            last_invocation: Mutex::new(None),
        }
    }

    /// Alternative constructor accepting a `Duration` directly.
    pub fn with_duration(inner: Arc<dyn Runnable>, min_interval: Duration) -> Self {
        Self {
            inner,
            min_interval,
            last_invocation: Mutex::new(None),
        }
    }

    /// Waits if necessary to enforce the minimum interval, then records the
    /// current time.
    async fn wait_if_needed(&self) {
        let sleep_duration = {
            let mut last = self.last_invocation.lock().unwrap();
            let now = Instant::now();

            let duration = if let Some(last_time) = *last {
                let elapsed = now.duration_since(last_time);
                if elapsed < self.min_interval {
                    Some(self.min_interval - elapsed)
                } else {
                    None
                }
            } else {
                None
            };

            *last = Some(now + duration.unwrap_or_default());
            duration
        };

        if let Some(d) = sleep_duration {
            tokio::time::sleep(d).await;
        }
    }
}

#[async_trait]
impl Runnable for RunnableThrottle {
    fn name(&self) -> &str {
        "RunnableThrottle"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        self.wait_if_needed().await;
        self.inner.invoke(input, config).await
    }

    async fn batch(
        &self,
        inputs: Vec<Value>,
        config: Option<&RunnableConfig>,
    ) -> Result<Vec<Value>> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.invoke(input, config).await?);
        }
        Ok(results)
    }

    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        self.wait_if_needed().await;
        self.inner.stream(input, config).await
    }
}

// ---------------------------------------------------------------------------
// ConcurrencyLimiter + ConcurrencyGuard
// ---------------------------------------------------------------------------

/// Internal mutable state for the concurrency limiter.
#[derive(Debug)]
struct ConcurrencyLimiterInner {
    active: usize,
}

/// Limits the number of concurrent executions.
///
/// When the limit is reached, [`try_acquire`](ConcurrencyLimiter::try_acquire)
/// returns `None`.  The returned [`ConcurrencyGuard`] is an RAII handle that
/// releases the slot on drop.
#[derive(Debug, Clone)]
pub struct ConcurrencyLimiter {
    inner: Arc<Mutex<ConcurrencyLimiterInner>>,
    max_concurrent: usize,
}

impl ConcurrencyLimiter {
    /// Create a new concurrency limiter.
    ///
    /// # Arguments
    /// * `max_concurrent` — maximum number of concurrent executions allowed.
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ConcurrencyLimiterInner { active: 0 })),
            max_concurrent,
        }
    }

    /// Try to acquire a concurrency slot.
    ///
    /// Returns `Some(ConcurrencyGuard)` if a slot is available, or `None` if
    /// the concurrency limit has been reached.
    pub fn try_acquire(&self) -> Option<ConcurrencyGuard> {
        let mut inner = self.inner.lock().unwrap();
        if inner.active < self.max_concurrent {
            inner.active += 1;
            Some(ConcurrencyGuard {
                inner: Arc::clone(&self.inner),
            })
        } else {
            None
        }
    }

    /// Returns the number of currently active executions.
    pub fn active_count(&self) -> usize {
        self.inner.lock().unwrap().active
    }

    /// Returns the configured maximum number of concurrent executions.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }
}

/// RAII guard that releases a concurrency slot when dropped.
///
/// Returned by [`ConcurrencyLimiter::try_acquire`].
#[derive(Debug)]
pub struct ConcurrencyGuard {
    inner: Arc<Mutex<ConcurrencyLimiterInner>>,
}

impl Drop for ConcurrencyGuard {
    fn drop(&mut self) {
        let mut inner = self.inner.lock().unwrap();
        inner.active -= 1;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // -- helpers --

    /// A simple runnable that returns its input unchanged.
    struct Echo;

    #[async_trait]
    impl Runnable for Echo {
        fn name(&self) -> &str {
            "Echo"
        }
        async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
            Ok(input)
        }
    }

    /// A runnable that counts how many times it has been invoked.
    struct Counter {
        count: AtomicUsize,
    }

    impl Counter {
        fn new() -> Self {
            Self {
                count: AtomicUsize::new(0),
            }
        }
        fn count(&self) -> usize {
            self.count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl Runnable for Counter {
        fn name(&self) -> &str {
            "Counter"
        }
        async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(input)
        }
    }

    // ==================== RateLimitConfig ====================

    #[test]
    fn test_config_new() {
        let c = RateLimitConfig::new(10, 1000);
        assert_eq!(c.max_requests, 10);
        assert_eq!(c.window_duration_ms, 1000);
        assert!(c.burst_limit.is_none());
    }

    #[test]
    fn test_config_with_burst_limit() {
        let c = RateLimitConfig::new(10, 1000).with_burst_limit(20);
        assert_eq!(c.burst_limit, Some(20));
    }

    #[test]
    fn test_config_builder_chain() {
        let c = RateLimitConfig::new(5, 500).with_burst_limit(10);
        assert_eq!(c.max_requests, 5);
        assert_eq!(c.window_duration_ms, 500);
        assert_eq!(c.burst_limit, Some(10));
    }

    #[test]
    fn test_config_clone() {
        let c = RateLimitConfig::new(10, 1000).with_burst_limit(15);
        let c2 = c.clone();
        assert_eq!(c2.max_requests, c.max_requests);
        assert_eq!(c2.window_duration_ms, c.window_duration_ms);
        assert_eq!(c2.burst_limit, c.burst_limit);
    }

    #[test]
    fn test_config_debug() {
        let c = RateLimitConfig::new(10, 1000).with_burst_limit(15);
        let s = format!("{:?}", c);
        assert!(s.contains("10"));
        assert!(s.contains("1000"));
        assert!(s.contains("15"));
    }

    // ==================== RateLimiter ====================

    #[test]
    fn test_limiter_acquire_within_limit() {
        let limiter = RateLimiter::new(RateLimitConfig::new(5, 60_000));
        for _ in 0..5 {
            assert!(limiter.try_acquire());
        }
    }

    #[test]
    fn test_limiter_acquire_exhaustion() {
        let limiter = RateLimiter::new(RateLimitConfig::new(3, 60_000));
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(!limiter.try_acquire());
        assert!(!limiter.try_acquire());
    }

    #[test]
    fn test_limiter_available_tokens() {
        let limiter = RateLimiter::new(RateLimitConfig::new(5, 60_000));
        assert_eq!(limiter.available_tokens(), 5);
        limiter.try_acquire();
        assert_eq!(limiter.available_tokens(), 4);
        limiter.try_acquire();
        limiter.try_acquire();
        assert_eq!(limiter.available_tokens(), 2);
    }

    #[test]
    fn test_limiter_reset() {
        let limiter = RateLimiter::new(RateLimitConfig::new(3, 60_000));
        limiter.try_acquire();
        limiter.try_acquire();
        limiter.try_acquire();
        assert_eq!(limiter.available_tokens(), 0);
        limiter.reset();
        assert_eq!(limiter.available_tokens(), 3);
    }

    #[test]
    fn test_limiter_config_accessor() {
        let limiter = RateLimiter::new(RateLimitConfig::new(10, 2000));
        assert_eq!(limiter.config().max_requests, 10);
        assert_eq!(limiter.config().window_duration_ms, 2000);
    }

    #[test]
    fn test_limiter_burst_limit() {
        let limiter = RateLimiter::new(RateLimitConfig::new(3, 60_000).with_burst_limit(5));
        assert_eq!(limiter.available_tokens(), 5);
        for _ in 0..5 {
            assert!(limiter.try_acquire());
        }
        assert!(!limiter.try_acquire());
    }

    #[tokio::test]
    async fn test_limiter_refill_over_time() {
        let limiter = RateLimiter::new(RateLimitConfig::new(10, 100));
        for _ in 0..10 {
            limiter.try_acquire();
        }
        assert_eq!(limiter.available_tokens(), 0);
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(limiter.available_tokens() > 0);
    }

    #[test]
    fn test_limiter_zero_max_requests() {
        let limiter = RateLimiter::new(RateLimitConfig::new(0, 1000));
        assert_eq!(limiter.available_tokens(), 0);
        assert!(!limiter.try_acquire());
    }

    #[test]
    fn test_limiter_zero_window() {
        let limiter = RateLimiter::new(RateLimitConfig::new(5, 0));
        assert_eq!(limiter.available_tokens(), 5);
        for _ in 0..5 {
            assert!(limiter.try_acquire());
        }
        assert!(!limiter.try_acquire());
    }

    #[test]
    fn test_limiter_immediate_reset_and_reuse() {
        let limiter = RateLimiter::new(RateLimitConfig::new(2, 60_000));
        limiter.try_acquire();
        limiter.try_acquire();
        assert!(!limiter.try_acquire());
        limiter.reset();
        assert!(limiter.try_acquire());
        assert!(limiter.try_acquire());
        assert!(!limiter.try_acquire());
    }

    #[test]
    fn test_burst_limit_capacity() {
        let limiter = RateLimiter::new(RateLimitConfig::new(5, 60_000).with_burst_limit(8));
        assert_eq!(limiter.available_tokens(), 8);
        for i in 0..8 {
            assert!(limiter.try_acquire(), "Token {} should be available", i);
        }
        assert!(!limiter.try_acquire());
    }

    // ==================== SlidingWindowCounter ====================

    #[test]
    fn test_sliding_window_within_limit() {
        let counter = SlidingWindowCounter::new(5, 60_000);
        for _ in 0..5 {
            assert!(counter.record());
        }
        assert_eq!(counter.current_count(), 5);
    }

    #[test]
    fn test_sliding_window_exceeds_limit() {
        let counter = SlidingWindowCounter::new(3, 60_000);
        assert!(counter.record());
        assert!(counter.record());
        assert!(counter.record());
        assert!(!counter.record());
        assert_eq!(counter.current_count(), 3);
    }

    #[test]
    fn test_sliding_window_reset() {
        let counter = SlidingWindowCounter::new(3, 60_000);
        counter.record();
        counter.record();
        counter.record();
        assert_eq!(counter.current_count(), 3);
        counter.reset();
        assert_eq!(counter.current_count(), 0);
        assert!(counter.record());
    }

    #[tokio::test]
    async fn test_sliding_window_expiry() {
        let counter = SlidingWindowCounter::new(2, 50);
        assert!(counter.record());
        assert!(counter.record());
        assert!(!counter.record());
        tokio::time::sleep(Duration::from_millis(70)).await;
        assert_eq!(counter.current_count(), 0);
        assert!(counter.record());
    }

    #[test]
    fn test_sliding_window_zero_limit() {
        let counter = SlidingWindowCounter::new(0, 1000);
        assert!(!counter.record());
        assert_eq!(counter.current_count(), 0);
    }

    #[test]
    fn test_sliding_window_very_large_window() {
        let counter = SlidingWindowCounter::new(100, 3_600_000);
        for _ in 0..100 {
            assert!(counter.record());
        }
        assert!(!counter.record());
        assert_eq!(counter.current_count(), 100);
    }

    // ==================== RunnableRateLimit ====================

    #[tokio::test]
    async fn test_runnable_rate_limit_allows_within_limit() {
        let inner = Arc::new(Echo) as Arc<dyn Runnable>;
        let config = RateLimitConfig::new(5, 60_000);
        let limited = RunnableRateLimit::new(inner, config);

        for i in 0..5 {
            let r = limited.invoke(json!(i), None).await;
            assert!(r.is_ok(), "Request {} should succeed", i);
            assert_eq!(r.unwrap(), json!(i));
        }
    }

    #[tokio::test]
    async fn test_runnable_rate_limit_rejects_over_limit() {
        let inner = Arc::new(Echo) as Arc<dyn Runnable>;
        let config = RateLimitConfig::new(2, 60_000);
        let limited = RunnableRateLimit::new(inner, config);

        assert!(limited.invoke(json!(1), None).await.is_ok());
        assert!(limited.invoke(json!(2), None).await.is_ok());

        let r = limited.invoke(json!(3), None).await;
        assert!(r.is_err());
        assert!(format!("{}", r.unwrap_err()).contains("Rate limit exceeded"));
    }

    #[tokio::test]
    async fn test_runnable_rate_limit_name() {
        let inner = Arc::new(Echo) as Arc<dyn Runnable>;
        let limited = RunnableRateLimit::new(inner, RateLimitConfig::new(5, 1000));
        assert_eq!(limited.name(), "RunnableRateLimit");
    }

    #[tokio::test]
    async fn test_runnable_rate_limit_batch_respects_limit() {
        let counter = Arc::new(Counter::new());
        let limited = RunnableRateLimit::new(
            counter.clone() as Arc<dyn Runnable>,
            RateLimitConfig::new(3, 60_000),
        );

        let inputs = vec![json!(1), json!(2), json!(3), json!(4), json!(5)];
        let r = limited.batch(inputs, None).await;
        assert!(r.is_err());
        assert_eq!(counter.count(), 3);
    }

    #[tokio::test]
    async fn test_runnable_rate_limit_stream_rejects() {
        let inner = Arc::new(Echo) as Arc<dyn Runnable>;
        let limited = RunnableRateLimit::new(inner, RateLimitConfig::new(1, 60_000));
        assert!(limited.stream(json!(1), None).await.is_ok());
        assert!(limited.stream(json!(2), None).await.is_err());
    }

    #[tokio::test]
    async fn test_runnable_rate_limit_limiter_accessor() {
        let inner = Arc::new(Echo) as Arc<dyn Runnable>;
        let limited = RunnableRateLimit::new(inner, RateLimitConfig::new(10, 1000));
        assert_eq!(limited.limiter().config().max_requests, 10);
    }

    #[tokio::test]
    async fn test_runnable_rate_limit_single_request() {
        let inner = Arc::new(Echo) as Arc<dyn Runnable>;
        let limited = RunnableRateLimit::new(inner, RateLimitConfig::new(1, 60_000));

        let r = limited.invoke(json!("only one"), None).await;
        assert!(r.is_ok());
        assert_eq!(r.unwrap(), json!("only one"));
        assert!(limited.invoke(json!("too many"), None).await.is_err());
    }

    // ==================== RunnableThrottle ====================

    #[tokio::test]
    async fn test_throttle_first_call_immediate() {
        let inner = Arc::new(Echo) as Arc<dyn Runnable>;
        let throttled = RunnableThrottle::new(inner, 200);

        let start = Instant::now();
        let r = throttled.invoke(json!("fast"), None).await;
        let elapsed = start.elapsed();

        assert!(r.is_ok());
        assert_eq!(r.unwrap(), json!("fast"));
        assert!(
            elapsed < Duration::from_millis(50),
            "First call should be immediate, took {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_throttle_enforces_min_interval() {
        let inner = Arc::new(Echo) as Arc<dyn Runnable>;
        let throttled = RunnableThrottle::new(inner, 80);

        throttled.invoke(json!(1), None).await.unwrap();

        let start = Instant::now();
        let r = throttled.invoke(json!(2), None).await;
        let elapsed = start.elapsed();

        assert!(r.is_ok());
        assert!(
            elapsed >= Duration::from_millis(60),
            "Expected throttle delay of ~80ms, got {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_throttle_no_delay_after_interval() {
        let inner = Arc::new(Echo) as Arc<dyn Runnable>;
        let throttled = RunnableThrottle::new(inner, 30);

        throttled.invoke(json!(1), None).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;

        let start = Instant::now();
        let r = throttled.invoke(json!(2), None).await;
        let elapsed = start.elapsed();

        assert!(r.is_ok());
        assert!(
            elapsed < Duration::from_millis(20),
            "Should not delay after interval has passed, took {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_throttle_name() {
        let inner = Arc::new(Echo) as Arc<dyn Runnable>;
        let throttled = RunnableThrottle::new(inner, 100);
        assert_eq!(throttled.name(), "RunnableThrottle");
    }

    #[tokio::test]
    async fn test_throttle_zero_interval() {
        let inner = Arc::new(Echo) as Arc<dyn Runnable>;
        let throttled = RunnableThrottle::new(inner, 0);

        for i in 0..10 {
            let r = throttled.invoke(json!(i), None).await;
            assert!(r.is_ok());
        }
    }

    #[tokio::test]
    async fn test_throttle_batch() {
        let counter = Arc::new(Counter::new());
        let throttled = RunnableThrottle::new(counter.clone() as Arc<dyn Runnable>, 20);

        let inputs = vec![json!(1), json!(2), json!(3)];
        let start = Instant::now();
        let results = throttled.batch(inputs, None).await;
        let elapsed = start.elapsed();

        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 3);
        assert_eq!(counter.count(), 3);
        assert!(
            elapsed >= Duration::from_millis(30),
            "Expected throttle delays, got {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_throttle_with_duration() {
        let inner = Arc::new(Echo) as Arc<dyn Runnable>;
        let throttled = RunnableThrottle::with_duration(inner, Duration::from_millis(50));

        throttled.invoke(json!(1), None).await.unwrap();
        let start = Instant::now();
        throttled.invoke(json!(2), None).await.unwrap();
        let elapsed = start.elapsed();

        assert!(
            elapsed >= Duration::from_millis(30),
            "Expected ~50ms delay, got {:?}",
            elapsed
        );
    }

    // ==================== ConcurrencyLimiter ====================

    #[test]
    fn test_concurrency_limiter_basic() {
        let limiter = ConcurrencyLimiter::new(2);
        assert_eq!(limiter.max_concurrent(), 2);
        assert_eq!(limiter.active_count(), 0);

        let _g1 = limiter.try_acquire().unwrap();
        assert_eq!(limiter.active_count(), 1);

        let _g2 = limiter.try_acquire().unwrap();
        assert_eq!(limiter.active_count(), 2);

        assert!(limiter.try_acquire().is_none());
    }

    #[test]
    fn test_concurrency_limiter_release_on_drop() {
        let limiter = ConcurrencyLimiter::new(1);

        {
            let _g = limiter.try_acquire().unwrap();
            assert_eq!(limiter.active_count(), 1);
            assert!(limiter.try_acquire().is_none());
        }
        assert_eq!(limiter.active_count(), 0);
        assert!(limiter.try_acquire().is_some());
    }

    #[test]
    fn test_concurrency_guard_raii() {
        let limiter = ConcurrencyLimiter::new(3);

        let g1 = limiter.try_acquire().unwrap();
        let g2 = limiter.try_acquire().unwrap();
        let g3 = limiter.try_acquire().unwrap();
        assert_eq!(limiter.active_count(), 3);
        assert!(limiter.try_acquire().is_none());

        drop(g2);
        assert_eq!(limiter.active_count(), 2);
        let _g4 = limiter.try_acquire().unwrap();
        assert_eq!(limiter.active_count(), 3);

        drop(g1);
        drop(g3);
        assert_eq!(limiter.active_count(), 1);
    }

    #[test]
    fn test_concurrency_limiter_zero() {
        let limiter = ConcurrencyLimiter::new(0);
        assert!(limiter.try_acquire().is_none());
        assert_eq!(limiter.active_count(), 0);
    }

    #[test]
    fn test_concurrency_limiter_clone() {
        let limiter = ConcurrencyLimiter::new(2);
        let cloned = limiter.clone();

        let _g = limiter.try_acquire().unwrap();
        assert_eq!(cloned.active_count(), 1);
    }

    #[test]
    fn test_concurrency_limiter_large_limit() {
        let limiter = ConcurrencyLimiter::new(1000);
        let mut guards = Vec::new();
        for _ in 0..1000 {
            guards.push(limiter.try_acquire().unwrap());
        }
        assert_eq!(limiter.active_count(), 1000);
        assert!(limiter.try_acquire().is_none());

        guards.clear();
        assert_eq!(limiter.active_count(), 0);
    }

    #[test]
    fn test_concurrency_guard_multiple_drop() {
        let limiter = ConcurrencyLimiter::new(5);
        let guards: Vec<_> = (0..5).map(|_| limiter.try_acquire().unwrap()).collect();
        assert_eq!(limiter.active_count(), 5);
        drop(guards);
        assert_eq!(limiter.active_count(), 0);
        // Can acquire again after all released.
        let _g = limiter.try_acquire().unwrap();
        assert_eq!(limiter.active_count(), 1);
    }
}
