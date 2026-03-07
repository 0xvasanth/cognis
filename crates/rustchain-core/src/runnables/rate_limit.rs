use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::error::{Result, RustChainError};

use super::base::Runnable;
use super::config::RunnableConfig;
use super::RunnableStream;

/// Configuration for rate limiting a runnable.
///
/// Uses a token bucket algorithm to control the rate of invocations.
///
/// # Example
/// ```ignore
/// use rustchain_core::runnables::rate_limit::RateLimitConfig;
///
/// let config = RateLimitConfig::new(10.0)  // 10 requests per second
///     .with_max_burst(5)
///     .with_wait_on_limit(true);
/// ```
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Maximum number of requests allowed per second.
    pub max_requests_per_second: f64,
    /// Maximum burst size (number of tokens the bucket can hold).
    pub max_burst: usize,
    /// If true, wait until a token is available. If false, return an error immediately.
    pub wait_on_limit: bool,
}

impl RateLimitConfig {
    /// Create a new rate limit configuration with the given requests per second.
    ///
    /// Defaults: max_burst=1, wait_on_limit=true.
    pub fn new(max_requests_per_second: f64) -> Self {
        Self {
            max_requests_per_second,
            max_burst: 1,
            wait_on_limit: true,
        }
    }

    /// Set the maximum burst size.
    pub fn with_max_burst(mut self, max_burst: usize) -> Self {
        self.max_burst = max_burst;
        self
    }

    /// Set whether to wait when the rate limit is exceeded.
    ///
    /// If false, an error is returned immediately when no tokens are available.
    pub fn with_wait_on_limit(mut self, wait_on_limit: bool) -> Self {
        self.wait_on_limit = wait_on_limit;
        self
    }
}

/// Internal token bucket implementation for rate limiting.
///
/// The bucket holds up to `max_tokens` tokens and refills at `refill_rate`
/// tokens per second. Each invocation consumes one token.
pub(crate) struct TokenBucket {
    /// Current number of available tokens.
    tokens: f64,
    /// Maximum number of tokens the bucket can hold.
    max_tokens: f64,
    /// Rate at which tokens are added (tokens per second).
    refill_rate: f64,
    /// Time of the last token refill.
    last_refill: Instant,
}

impl TokenBucket {
    /// Create a new token bucket.
    pub fn new(max_tokens: f64, refill_rate: f64) -> Self {
        Self {
            tokens: max_tokens,
            max_tokens,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Refill tokens based on elapsed time since last refill.
    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.max_tokens);
        self.last_refill = now;
    }

    /// Attempt to acquire a token. Returns true if successful.
    pub fn try_acquire(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Returns how long to wait before a token becomes available.
    ///
    /// If a token is already available, returns `Duration::ZERO`.
    pub fn wait_for_token(&mut self) -> Duration {
        self.refill();
        if self.tokens >= 1.0 {
            Duration::ZERO
        } else {
            let deficit = 1.0 - self.tokens;
            Duration::from_secs_f64(deficit / self.refill_rate)
        }
    }
}

/// A runnable wrapper that applies token-bucket rate limiting.
///
/// Wraps an inner `Runnable` and limits the rate of invocations using a
/// configurable token bucket. Thread-safe via `Arc<Mutex<TokenBucket>>`.
///
/// # Example
/// ```ignore
/// use rustchain_core::runnables::{RunnableLambda, RunnableExt};
/// use rustchain_core::runnables::rate_limit::RateLimitConfig;
///
/// let config = RateLimitConfig::new(5.0).with_max_burst(2);
/// let limited = my_runnable.with_rate_limit(config);
/// ```
pub struct RunnableRateLimit {
    /// The wrapped runnable.
    inner: Arc<dyn Runnable>,
    /// Thread-safe token bucket.
    bucket: Arc<Mutex<TokenBucket>>,
    /// Whether to wait or error when rate limited.
    wait_on_limit: bool,
}

impl RunnableRateLimit {
    /// Create a new rate-limited runnable.
    pub fn new(inner: Arc<dyn Runnable>, config: RateLimitConfig) -> Self {
        let bucket = TokenBucket::new(
            config.max_burst as f64,
            config.max_requests_per_second,
        );
        Self {
            inner,
            bucket: Arc::new(Mutex::new(bucket)),
            wait_on_limit: config.wait_on_limit,
        }
    }

    /// Acquire a token, waiting if configured to do so.
    async fn acquire_token(&self) -> Result<()> {
        loop {
            let wait_duration = {
                let mut bucket = self.bucket.lock().await;
                if bucket.try_acquire() {
                    return Ok(());
                }
                if !self.wait_on_limit {
                    return Err(RustChainError::Other(
                        "Rate limit exceeded".to_string(),
                    ));
                }
                bucket.wait_for_token()
            };
            tokio::time::sleep(wait_duration).await;
        }
    }
}

#[async_trait]
impl Runnable for RunnableRateLimit {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        self.acquire_token().await?;
        self.inner.invoke(input, config).await
    }

    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        self.acquire_token().await?;
        self.inner.stream(input, config).await
    }
}

/// A simpler throttle wrapper that enforces a minimum interval between invocations.
///
/// Unlike `RunnableRateLimit` which uses a token bucket, `RunnableThrottle`
/// simply ensures at least `min_interval` has elapsed since the last invocation.
///
/// # Example
/// ```ignore
/// use std::time::Duration;
/// use rustchain_core::runnables::{RunnableLambda, RunnableExt};
///
/// let throttled = my_runnable.with_throttle(Duration::from_millis(100));
/// ```
pub struct RunnableThrottle {
    /// The wrapped runnable.
    inner: Arc<dyn Runnable>,
    /// Minimum interval between invocations.
    min_interval: Duration,
    /// Timestamp of the last invocation.
    last_invocation: Arc<Mutex<Instant>>,
}

impl RunnableThrottle {
    /// Create a new throttled runnable with the given minimum interval.
    pub fn new(inner: Arc<dyn Runnable>, min_interval: Duration) -> Self {
        // Initialize last_invocation far enough in the past so the first call goes through.
        let past = Instant::now() - min_interval;
        Self {
            inner,
            min_interval,
            last_invocation: Arc::new(Mutex::new(past)),
        }
    }

    /// Wait until enough time has passed since the last invocation, then
    /// update the timestamp.
    async fn wait_for_interval(&self) {
        let sleep_duration = {
            let mut last = self.last_invocation.lock().await;
            let elapsed = last.elapsed();
            if elapsed >= self.min_interval {
                *last = Instant::now();
                Duration::ZERO
            } else {
                let wait = self.min_interval - elapsed;
                *last = Instant::now() + wait;
                wait
            }
        };
        if sleep_duration > Duration::ZERO {
            tokio::time::sleep(sleep_duration).await;
        }
    }
}

#[async_trait]
impl Runnable for RunnableThrottle {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        self.wait_for_interval().await;
        self.inner.invoke(input, config).await
    }

    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        self.wait_for_interval().await;
        self.inner.stream(input, config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::lambda::RunnableLambda;
    use serde_json::json;

    fn identity_runnable() -> RunnableLambda {
        RunnableLambda::new("identity", |v: Value| async move { Ok(v) })
    }

    fn doubler_runnable() -> RunnableLambda {
        RunnableLambda::new("doubler", |v: Value| async move {
            let n = v.as_i64().unwrap();
            Ok(json!(n * 2))
        })
    }

    // ==================== RateLimitConfig tests ====================

    #[test]
    fn test_rate_limit_config_defaults() {
        let config = RateLimitConfig::new(10.0);
        assert_eq!(config.max_requests_per_second, 10.0);
        assert_eq!(config.max_burst, 1);
        assert!(config.wait_on_limit);
    }

    #[test]
    fn test_rate_limit_config_builder() {
        let config = RateLimitConfig::new(5.0)
            .with_max_burst(10)
            .with_wait_on_limit(false);
        assert_eq!(config.max_requests_per_second, 5.0);
        assert_eq!(config.max_burst, 10);
        assert!(!config.wait_on_limit);
    }

    // ==================== TokenBucket tests ====================

    #[test]
    fn test_token_bucket_initial_acquire() {
        let mut bucket = TokenBucket::new(3.0, 10.0);
        assert!(bucket.try_acquire());
        assert!(bucket.try_acquire());
        assert!(bucket.try_acquire());
        // All 3 tokens consumed
        assert!(!bucket.try_acquire());
    }

    #[test]
    fn test_token_bucket_wait_for_token_when_available() {
        let mut bucket = TokenBucket::new(1.0, 10.0);
        let wait = bucket.wait_for_token();
        assert_eq!(wait, Duration::ZERO);
    }

    #[test]
    fn test_token_bucket_wait_for_token_when_empty() {
        let mut bucket = TokenBucket::new(1.0, 10.0);
        bucket.try_acquire(); // consume the token
        let wait = bucket.wait_for_token();
        // Should be approximately 0.1 seconds (1 token / 10 tokens per second)
        assert!(wait > Duration::ZERO);
        assert!(wait <= Duration::from_millis(150));
    }

    // ==================== RunnableRateLimit tests ====================

    #[tokio::test]
    async fn test_rate_limit_single_invoke() {
        let inner = identity_runnable();
        let config = RateLimitConfig::new(100.0).with_max_burst(1);
        let limited = RunnableRateLimit::new(Arc::new(inner), config);

        let result = limited.invoke(json!(42), None).await.unwrap();
        assert_eq!(result, json!(42));
    }

    #[tokio::test]
    async fn test_rate_limit_delegates_to_inner() {
        let inner = doubler_runnable();
        let config = RateLimitConfig::new(100.0).with_max_burst(5);
        let limited = RunnableRateLimit::new(Arc::new(inner), config);

        let result = limited.invoke(json!(7), None).await.unwrap();
        assert_eq!(result, json!(14));
    }

    #[tokio::test]
    async fn test_rate_limit_name_delegates() {
        let inner = identity_runnable();
        let config = RateLimitConfig::new(10.0);
        let limited = RunnableRateLimit::new(Arc::new(inner), config);

        assert_eq!(limited.name(), "identity");
    }

    #[tokio::test]
    async fn test_rate_limit_burst_allows_multiple() {
        let inner = identity_runnable();
        let config = RateLimitConfig::new(1.0).with_max_burst(5);
        let limited = RunnableRateLimit::new(Arc::new(inner), config);

        // Should allow 5 rapid invocations (burst size)
        for i in 0..5 {
            let result = limited.invoke(json!(i), None).await.unwrap();
            assert_eq!(result, json!(i));
        }
    }

    #[tokio::test]
    async fn test_rate_limit_error_when_no_wait() {
        let inner = identity_runnable();
        let config = RateLimitConfig::new(1.0)
            .with_max_burst(1)
            .with_wait_on_limit(false);
        let limited = RunnableRateLimit::new(Arc::new(inner), config);

        // First call should succeed (uses the initial token).
        let result = limited.invoke(json!(1), None).await;
        assert!(result.is_ok());

        // Second call should fail immediately (no waiting).
        let result = limited.invoke(json!(2), None).await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Rate limit exceeded"));
    }

    #[tokio::test]
    async fn test_rate_limit_waits_for_token() {
        let inner = identity_runnable();
        let config = RateLimitConfig::new(20.0) // 20 rps = 50ms per token
            .with_max_burst(1)
            .with_wait_on_limit(true);
        let limited = RunnableRateLimit::new(Arc::new(inner), config);

        // First call consumes the token.
        limited.invoke(json!(1), None).await.unwrap();

        // Second call must wait for a token refill.
        let start = Instant::now();
        let result = limited.invoke(json!(2), None).await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result, json!(2));
        // Should have waited roughly 50ms (1/20 seconds).
        assert!(
            elapsed >= Duration::from_millis(30),
            "Expected to wait for token, elapsed: {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_rate_limit_enforces_rate() {
        let inner = identity_runnable();
        // 10 rps with burst of 1 = one request per 100ms after the first.
        let config = RateLimitConfig::new(10.0)
            .with_max_burst(1)
            .with_wait_on_limit(true);
        let limited = RunnableRateLimit::new(Arc::new(inner), config);

        let start = Instant::now();
        for i in 0..4 {
            limited.invoke(json!(i), None).await.unwrap();
        }
        let elapsed = start.elapsed();

        // 4 calls at 10 rps with burst 1: first is free, then ~100ms each for 3 more = ~300ms.
        assert!(
            elapsed >= Duration::from_millis(250),
            "Expected at least ~300ms for 4 calls at 10rps, got {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_rate_limit_stream() {
        use futures::StreamExt;

        let inner = identity_runnable();
        let config = RateLimitConfig::new(100.0).with_max_burst(1);
        let limited = RunnableRateLimit::new(Arc::new(inner), config);

        let mut stream = limited.stream(json!("hello"), None).await.unwrap();
        let item = stream.next().await.unwrap().unwrap();
        assert_eq!(item, json!("hello"));
    }

    // ==================== RunnableThrottle tests ====================

    #[tokio::test]
    async fn test_throttle_single_invoke() {
        let inner = identity_runnable();
        let throttled = RunnableThrottle::new(
            Arc::new(inner),
            Duration::from_millis(10),
        );

        let result = throttled.invoke(json!("test"), None).await.unwrap();
        assert_eq!(result, json!("test"));
    }

    #[tokio::test]
    async fn test_throttle_name_delegates() {
        let inner = doubler_runnable();
        let throttled = RunnableThrottle::new(
            Arc::new(inner),
            Duration::from_millis(50),
        );

        assert_eq!(throttled.name(), "doubler");
    }

    #[tokio::test]
    async fn test_throttle_enforces_minimum_interval() {
        let inner = identity_runnable();
        let interval = Duration::from_millis(80);
        let throttled = RunnableThrottle::new(Arc::new(inner), interval);

        let start = Instant::now();
        for i in 0..4 {
            throttled.invoke(json!(i), None).await.unwrap();
        }
        let elapsed = start.elapsed();

        // First call is immediate, then 3 waits of ~80ms each = ~240ms minimum.
        assert!(
            elapsed >= Duration::from_millis(200),
            "Expected at least ~240ms for 4 throttled calls, got {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_throttle_first_call_immediate() {
        let inner = identity_runnable();
        let throttled = RunnableThrottle::new(
            Arc::new(inner),
            Duration::from_millis(1000), // very long interval
        );

        let start = Instant::now();
        let result = throttled.invoke(json!(1), None).await.unwrap();
        let elapsed = start.elapsed();

        assert_eq!(result, json!(1));
        // First call should be near-instant.
        assert!(
            elapsed < Duration::from_millis(50),
            "First call should be immediate, got {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_throttle_stream() {
        use futures::StreamExt;

        let inner = identity_runnable();
        let throttled = RunnableThrottle::new(
            Arc::new(inner),
            Duration::from_millis(10),
        );

        let mut stream = throttled.stream(json!(99), None).await.unwrap();
        let item = stream.next().await.unwrap().unwrap();
        assert_eq!(item, json!(99));
    }

    #[tokio::test]
    async fn test_throttle_delegates_computation() {
        let inner = doubler_runnable();
        let throttled = RunnableThrottle::new(
            Arc::new(inner),
            Duration::from_millis(10),
        );

        let result = throttled.invoke(json!(21), None).await.unwrap();
        assert_eq!(result, json!(42));
    }

    #[tokio::test]
    async fn test_rate_limit_thread_safety() {
        let inner = identity_runnable();
        let config = RateLimitConfig::new(100.0).with_max_burst(10);
        let limited = Arc::new(RunnableRateLimit::new(Arc::new(inner), config));

        let mut handles = vec![];
        for i in 0..5 {
            let limited = limited.clone();
            handles.push(tokio::spawn(async move {
                limited.invoke(json!(i), None).await.unwrap()
            }));
        }

        let mut results: Vec<i64> = vec![];
        for handle in handles {
            let val = handle.await.unwrap();
            results.push(val.as_i64().unwrap());
        }
        results.sort();
        assert_eq!(results, vec![0, 1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn test_throttle_thread_safety() {
        let inner = identity_runnable();
        let throttled = Arc::new(RunnableThrottle::new(
            Arc::new(inner),
            Duration::from_millis(5),
        ));

        let mut handles = vec![];
        for i in 0..3 {
            let throttled = throttled.clone();
            handles.push(tokio::spawn(async move {
                throttled.invoke(json!(i), None).await.unwrap()
            }));
        }

        for handle in handles {
            let val = handle.await.unwrap();
            assert!(val.is_number());
        }
    }
}
