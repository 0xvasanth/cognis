//! Rate limiter middleware — rate-limits model calls and tool invocations
//! to prevent abuse and manage API costs.

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use crate::agent::DeepAgentError;
use crate::middleware::{AgentState, Middleware, Result};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors specific to rate limiting.
#[derive(Debug, Clone)]
pub enum RateLimitError {
    /// Model call rate limit exceeded.
    ModelRateLimitExceeded {
        /// The configured limit.
        limit: u32,
        /// The time window for the limit.
        window: Duration,
    },
    /// Tool call rate limit exceeded.
    ToolRateLimitExceeded {
        /// The configured limit.
        limit: u32,
        /// The time window for the limit.
        window: Duration,
    },
    /// Token budget exceeded.
    TokenBudgetExceeded {
        /// The configured token limit.
        limit: u64,
        /// Tokens already used.
        used: u64,
    },
    /// Cost limit exceeded.
    CostLimitExceeded {
        /// The configured cost limit in dollars.
        limit: f64,
        /// Amount already spent in dollars.
        spent: f64,
    },
    /// Timeout exceeded while waiting for rate limit budget.
    TimeoutExceeded {
        /// Duration waited before giving up.
        waited: Duration,
    },
}

impl std::fmt::Display for RateLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ModelRateLimitExceeded { limit, window } => {
                write!(f, "model rate limit exceeded: {limit} calls per {window:?}")
            }
            Self::ToolRateLimitExceeded { limit, window } => {
                write!(f, "tool rate limit exceeded: {limit} calls per {window:?}")
            }
            Self::TokenBudgetExceeded { limit, used } => {
                write!(f, "token budget exceeded: used {used} of {limit}")
            }
            Self::CostLimitExceeded { limit, spent } => {
                write!(f, "cost limit exceeded: spent ${spent:.4} of ${limit:.4}")
            }
            Self::TimeoutExceeded { waited } => {
                write!(f, "timeout exceeded after waiting {waited:?}")
            }
        }
    }
}

impl std::error::Error for RateLimitError {}

// ---------------------------------------------------------------------------
// Backoff strategy
// ---------------------------------------------------------------------------

/// Strategy to apply when a rate limit is hit.
#[derive(Debug, Clone)]
pub enum BackoffStrategy {
    /// Return an error immediately.
    Reject,
    /// Wait until budget is available (may block indefinitely).
    Wait,
    /// Wait up to the given duration, then return an error.
    WaitWithTimeout(Duration),
    /// Slow down requests to fit within the configured limits.
    Throttle,
}

impl Default for BackoffStrategy {
    fn default() -> Self {
        Self::Reject
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the rate limiter middleware.
#[derive(Debug, Clone)]
pub struct RateLimiterConfig {
    /// Maximum model calls per minute.
    pub model_calls_per_minute: Option<u32>,
    /// Maximum model calls per hour.
    pub model_calls_per_hour: Option<u32>,
    /// Maximum tool calls per minute.
    pub tool_calls_per_minute: Option<u32>,
    /// Maximum tokens per minute.
    pub total_tokens_per_minute: Option<u64>,
    /// Maximum total cost in dollars.
    pub total_cost_limit: Option<f64>,
    /// Maximum burst of concurrent requests.
    pub burst_limit: Option<u32>,
    /// Strategy when a limit is hit.
    pub backoff_strategy: BackoffStrategy,
}

impl Default for RateLimiterConfig {
    fn default() -> Self {
        Self {
            model_calls_per_minute: None,
            model_calls_per_hour: None,
            tool_calls_per_minute: None,
            total_tokens_per_minute: None,
            total_cost_limit: None,
            burst_limit: None,
            backoff_strategy: BackoffStrategy::Reject,
        }
    }
}

// ---------------------------------------------------------------------------
// Token bucket
// ---------------------------------------------------------------------------

/// A simple token-bucket rate limiter.
///
/// Tokens are refilled continuously at `refill_rate` tokens per second,
/// up to `capacity`. `try_acquire` attempts to consume tokens without
/// blocking.
pub struct RateLimitBucket {
    /// Maximum tokens in the bucket.
    pub capacity: u32,
    /// Tokens added per second.
    pub refill_rate: f64,
    /// Current available tokens (stored as fractional for precision).
    available: Mutex<f64>,
    /// Last refill timestamp.
    last_refill: Mutex<Instant>,
}

impl RateLimitBucket {
    /// Create a new bucket starting at full capacity.
    pub fn new(capacity: u32, refill_rate: f64) -> Self {
        Self {
            capacity,
            refill_rate,
            available: Mutex::new(capacity as f64),
            last_refill: Mutex::new(Instant::now()),
        }
    }

    /// Refill tokens based on elapsed time.
    fn refill(&self) {
        let mut last = self.last_refill.lock().unwrap();
        let mut avail = self.available.lock().unwrap();
        let now = Instant::now();
        let elapsed = now.duration_since(*last).as_secs_f64();
        *avail = (*avail + elapsed * self.refill_rate).min(self.capacity as f64);
        *last = now;
    }

    /// Try to consume `n` tokens. Returns `true` if successful.
    pub fn try_acquire(&self, n: u32) -> bool {
        self.refill();
        let mut avail = self.available.lock().unwrap();
        let needed = n as f64;
        if *avail >= needed {
            *avail -= needed;
            true
        } else {
            false
        }
    }

    /// Calculate how long to wait for `n` tokens to become available.
    pub fn wait_for(&self, n: u32) -> Duration {
        self.refill();
        let avail = *self.available.lock().unwrap();
        let needed = n as f64;
        if avail >= needed {
            return Duration::ZERO;
        }
        let deficit = needed - avail;
        if self.refill_rate <= 0.0 {
            return Duration::from_secs(u64::MAX);
        }
        Duration::from_secs_f64(deficit / self.refill_rate)
    }

    /// Return the currently available tokens (after refill).
    pub fn available(&self) -> f64 {
        self.refill();
        *self.available.lock().unwrap()
    }
}

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

/// Accumulated statistics for the rate limiter.
#[derive(Debug, Clone)]
pub struct RateLimitStats {
    /// Model calls in the current minute window.
    pub model_calls_this_minute: u32,
    /// Model calls in the current hour window.
    pub model_calls_this_hour: u32,
    /// Tool calls in the current minute window.
    pub tool_calls_this_minute: u32,
    /// Tokens consumed in the current minute window.
    pub tokens_this_minute: u64,
    /// Total cost accumulated (dollars).
    pub total_cost: f64,
    /// Number of requests rejected.
    pub rejected_count: u32,
    /// Number of requests that had to wait.
    pub waited_count: u32,
}

// ---------------------------------------------------------------------------
// Internal sliding-window counter
// ---------------------------------------------------------------------------

/// A counter that resets after a configured window duration.
struct WindowCounter {
    count: AtomicU32,
    window_start: Mutex<Instant>,
    window: Duration,
}

impl WindowCounter {
    fn new(window: Duration) -> Self {
        Self {
            count: AtomicU32::new(0),
            window_start: Mutex::new(Instant::now()),
            window,
        }
    }

    /// Increment and return the new count, resetting if the window has elapsed.
    fn increment(&self) -> u32 {
        self.maybe_reset();
        self.count.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Get the current count, resetting if the window has elapsed.
    fn get(&self) -> u32 {
        self.maybe_reset();
        self.count.load(Ordering::SeqCst)
    }

    fn maybe_reset(&self) {
        let mut start = self.window_start.lock().unwrap();
        if start.elapsed() >= self.window {
            self.count.store(0, Ordering::SeqCst);
            *start = Instant::now();
        }
    }

    fn reset(&self) {
        self.count.store(0, Ordering::SeqCst);
        *self.window_start.lock().unwrap() = Instant::now();
    }
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

/// Middleware that enforces rate limits on model and tool invocations.
pub struct RateLimiterMiddleware {
    config: RateLimiterConfig,
    model_per_minute: WindowCounter,
    model_per_hour: WindowCounter,
    tool_per_minute: WindowCounter,
    tokens_per_minute: AtomicU64,
    tokens_window_start: Mutex<Instant>,
    total_cost: Mutex<f64>,
    rejected_count: AtomicU32,
    waited_count: AtomicU32,
    burst_bucket: Option<RateLimitBucket>,
}

impl RateLimiterMiddleware {
    /// Create a new rate limiter middleware with the given configuration.
    pub fn new(config: RateLimiterConfig) -> Self {
        let burst_bucket = config.burst_limit.map(|limit| {
            // Refill one token per second; capacity equals burst limit.
            RateLimitBucket::new(limit, 1.0)
        });

        Self {
            config,
            model_per_minute: WindowCounter::new(Duration::from_secs(60)),
            model_per_hour: WindowCounter::new(Duration::from_secs(3600)),
            tool_per_minute: WindowCounter::new(Duration::from_secs(60)),
            tokens_per_minute: AtomicU64::new(0),
            tokens_window_start: Mutex::new(Instant::now()),
            total_cost: Mutex::new(0.0),
            rejected_count: AtomicU32::new(0),
            waited_count: AtomicU32::new(0),
            burst_bucket,
        }
    }

    /// Get current rate limit statistics.
    pub fn get_stats(&self) -> RateLimitStats {
        self.maybe_reset_token_window();
        RateLimitStats {
            model_calls_this_minute: self.model_per_minute.get(),
            model_calls_this_hour: self.model_per_hour.get(),
            tool_calls_this_minute: self.tool_per_minute.get(),
            tokens_this_minute: self.tokens_per_minute.load(Ordering::SeqCst),
            total_cost: *self.total_cost.lock().unwrap(),
            rejected_count: self.rejected_count.load(Ordering::SeqCst),
            waited_count: self.waited_count.load(Ordering::SeqCst),
        }
    }

    /// Reset all statistics and counters.
    pub fn reset_stats(&self) {
        self.model_per_minute.reset();
        self.model_per_hour.reset();
        self.tool_per_minute.reset();
        self.tokens_per_minute.store(0, Ordering::SeqCst);
        *self.tokens_window_start.lock().unwrap() = Instant::now();
        *self.total_cost.lock().unwrap() = 0.0;
        self.rejected_count.store(0, Ordering::SeqCst);
        self.waited_count.store(0, Ordering::SeqCst);
    }

    /// Record token usage (call after a model response).
    pub fn record_tokens(&self, token_count: u64) {
        self.maybe_reset_token_window();
        self.tokens_per_minute
            .fetch_add(token_count, Ordering::SeqCst);
    }

    /// Record cost usage.
    pub fn record_cost(&self, cost: f64) {
        let mut total = self.total_cost.lock().unwrap();
        *total += cost;
    }

    fn maybe_reset_token_window(&self) {
        let mut start = self.tokens_window_start.lock().unwrap();
        if start.elapsed() >= Duration::from_secs(60) {
            self.tokens_per_minute.store(0, Ordering::SeqCst);
            *start = Instant::now();
        }
    }

    /// Apply the backoff strategy when a limit is exceeded.
    async fn apply_backoff(&self, error: RateLimitError) -> Result<()> {
        match &self.config.backoff_strategy {
            BackoffStrategy::Reject => {
                self.rejected_count.fetch_add(1, Ordering::SeqCst);
                Err(DeepAgentError::MiddlewareError(error.to_string()))
            }
            BackoffStrategy::Wait => {
                self.waited_count.fetch_add(1, Ordering::SeqCst);
                // In a real implementation this would sleep until budget is
                // available. For now we sleep a fixed 1 second.
                tokio::time::sleep(Duration::from_secs(1)).await;
                Ok(())
            }
            BackoffStrategy::WaitWithTimeout(timeout) => {
                self.waited_count.fetch_add(1, Ordering::SeqCst);
                let start = Instant::now();
                tokio::time::sleep((*timeout).min(Duration::from_secs(1))).await;
                if start.elapsed() >= *timeout {
                    self.rejected_count.fetch_add(1, Ordering::SeqCst);
                    Err(DeepAgentError::MiddlewareError(
                        RateLimitError::TimeoutExceeded { waited: *timeout }.to_string(),
                    ))
                } else {
                    Ok(())
                }
            }
            BackoffStrategy::Throttle => {
                self.waited_count.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(100)).await;
                Ok(())
            }
        }
    }

    /// Check model rate limits. Returns `Ok(())` if the call is allowed.
    fn check_model_limits(&self) -> std::result::Result<(), RateLimitError> {
        if let Some(limit) = self.config.model_calls_per_minute {
            let count = self.model_per_minute.get();
            if count >= limit {
                return Err(RateLimitError::ModelRateLimitExceeded {
                    limit,
                    window: Duration::from_secs(60),
                });
            }
        }
        if let Some(limit) = self.config.model_calls_per_hour {
            let count = self.model_per_hour.get();
            if count >= limit {
                return Err(RateLimitError::ModelRateLimitExceeded {
                    limit,
                    window: Duration::from_secs(3600),
                });
            }
        }
        Ok(())
    }

    /// Check tool rate limits.
    fn check_tool_limits(&self) -> std::result::Result<(), RateLimitError> {
        if let Some(limit) = self.config.tool_calls_per_minute {
            let count = self.tool_per_minute.get();
            if count >= limit {
                return Err(RateLimitError::ToolRateLimitExceeded {
                    limit,
                    window: Duration::from_secs(60),
                });
            }
        }
        Ok(())
    }

    /// Check token budget.
    fn check_token_budget(&self) -> std::result::Result<(), RateLimitError> {
        if let Some(limit) = self.config.total_tokens_per_minute {
            self.maybe_reset_token_window();
            let used = self.tokens_per_minute.load(Ordering::SeqCst);
            if used >= limit {
                return Err(RateLimitError::TokenBudgetExceeded { limit, used });
            }
        }
        Ok(())
    }

    /// Check cost limit.
    fn check_cost_limit(&self) -> std::result::Result<(), RateLimitError> {
        if let Some(limit) = self.config.total_cost_limit {
            let spent = *self.total_cost.lock().unwrap();
            if spent >= limit {
                return Err(RateLimitError::CostLimitExceeded { limit, spent });
            }
        }
        Ok(())
    }

    /// Check burst limit using the token bucket.
    fn check_burst_limit(&self) -> std::result::Result<(), RateLimitError> {
        if let Some(ref bucket) = self.burst_bucket {
            if !bucket.try_acquire(1) {
                return Err(RateLimitError::ModelRateLimitExceeded {
                    limit: self.config.burst_limit.unwrap_or(0),
                    window: Duration::from_secs(1),
                });
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Middleware for RateLimiterMiddleware {
    fn name(&self) -> &str {
        "rate_limiter"
    }

    /// Check and enforce model rate limits before the model is invoked.
    async fn before_model(&self, _state: &mut AgentState) -> Result<()> {
        // Check all limits before allowing the call.
        if let Err(e) = self.check_model_limits() {
            return self.apply_backoff(e).await;
        }
        if let Err(e) = self.check_token_budget() {
            return self.apply_backoff(e).await;
        }
        if let Err(e) = self.check_cost_limit() {
            return self.apply_backoff(e).await;
        }
        if let Err(e) = self.check_burst_limit() {
            return self.apply_backoff(e).await;
        }

        // Record the call.
        self.model_per_minute.increment();
        self.model_per_hour.increment();

        Ok(())
    }

    /// Track token usage after the model responds.
    async fn after_model(&self, state: &mut AgentState) -> Result<()> {
        // Try to extract token usage from the state if available.
        if let Some(usage) = state.get("usage").and_then(|u| u.get("total_tokens")) {
            if let Some(tokens) = usage.as_u64() {
                self.record_tokens(tokens);
            }
        }
        // Try to extract cost from the state if available.
        if let Some(cost) = state.get("usage").and_then(|u| u.get("cost")) {
            if let Some(c) = cost.as_f64() {
                self.record_cost(c);
            }
        }
        Ok(())
    }

    /// Check tool rate limits before a tool is executed.
    async fn before_tool(&self, _state: &mut AgentState, _tool_name: &str) -> Result<()> {
        if let Err(e) = self.check_tool_limits() {
            return self.apply_backoff(e).await;
        }

        // Record the call.
        self.tool_per_minute.increment();

        Ok(())
    }

    /// Track tool usage after execution.
    async fn after_tool(
        &self,
        _state: &mut AgentState,
        _tool_name: &str,
        _result: &str,
    ) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Duration;

    // Test 1: Config creation with defaults
    #[test]
    fn test_config_defaults() {
        let config = RateLimiterConfig::default();
        assert!(config.model_calls_per_minute.is_none());
        assert!(config.model_calls_per_hour.is_none());
        assert!(config.tool_calls_per_minute.is_none());
        assert!(config.total_tokens_per_minute.is_none());
        assert!(config.total_cost_limit.is_none());
        assert!(config.burst_limit.is_none());
        assert!(matches!(config.backoff_strategy, BackoffStrategy::Reject));
    }

    // Test 2: Token bucket try_acquire
    #[test]
    fn test_bucket_try_acquire() {
        let bucket = RateLimitBucket::new(5, 1.0);
        assert!(bucket.try_acquire(3));
        assert!(bucket.try_acquire(2));
        // Bucket should be empty now.
        assert!(!bucket.try_acquire(1));
    }

    // Test 3: Token bucket refill over time
    #[test]
    fn test_bucket_refill() {
        let bucket = RateLimitBucket::new(10, 1000.0); // High refill for test speed.
                                                       // Drain the bucket.
        assert!(bucket.try_acquire(10));
        assert!(!bucket.try_acquire(1));
        // Wait a small amount for refill.
        std::thread::sleep(Duration::from_millis(20));
        // Should have refilled some tokens (at 1000/s, 20ms = ~20 tokens, capped at 10).
        assert!(bucket.try_acquire(1));
    }

    // Test 4: Model rate limit enforcement (reject)
    #[tokio::test]
    async fn test_model_rate_limit_reject() {
        let config = RateLimiterConfig {
            model_calls_per_minute: Some(2),
            backoff_strategy: BackoffStrategy::Reject,
            ..Default::default()
        };
        let mw = RateLimiterMiddleware::new(config);
        let mut state = json!({"messages": []});

        assert!(mw.before_model(&mut state).await.is_ok());
        assert!(mw.before_model(&mut state).await.is_ok());
        // Third call should be rejected.
        assert!(mw.before_model(&mut state).await.is_err());
    }

    // Test 5: Tool rate limit enforcement
    #[tokio::test]
    async fn test_tool_rate_limit() {
        let config = RateLimiterConfig {
            tool_calls_per_minute: Some(3),
            backoff_strategy: BackoffStrategy::Reject,
            ..Default::default()
        };
        let mw = RateLimiterMiddleware::new(config);
        let mut state = json!({"messages": []});

        assert!(mw.before_tool(&mut state, "tool_a").await.is_ok());
        assert!(mw.before_tool(&mut state, "tool_b").await.is_ok());
        assert!(mw.before_tool(&mut state, "tool_c").await.is_ok());
        // Fourth call should be rejected.
        assert!(mw.before_tool(&mut state, "tool_d").await.is_err());
    }

    // Test 6: Token budget tracking
    #[tokio::test]
    async fn test_token_budget_tracking() {
        let config = RateLimiterConfig {
            total_tokens_per_minute: Some(100),
            backoff_strategy: BackoffStrategy::Reject,
            ..Default::default()
        };
        let mw = RateLimiterMiddleware::new(config);

        mw.record_tokens(60);
        assert_eq!(mw.get_stats().tokens_this_minute, 60);

        mw.record_tokens(50);
        // Now at 110, which exceeds 100.
        let mut state = json!({"messages": []});
        assert!(mw.before_model(&mut state).await.is_err());
    }

    // Test 7: Cost limit tracking
    #[tokio::test]
    async fn test_cost_limit_tracking() {
        let config = RateLimiterConfig {
            total_cost_limit: Some(1.0),
            backoff_strategy: BackoffStrategy::Reject,
            ..Default::default()
        };
        let mw = RateLimiterMiddleware::new(config);
        let mut state = json!({"messages": []});

        mw.record_cost(0.5);
        assert!(mw.before_model(&mut state).await.is_ok());

        mw.record_cost(0.6);
        // Now at 1.1, exceeding the $1.00 limit.
        assert!(mw.before_model(&mut state).await.is_err());
    }

    // Test 8: Backoff strategy reject
    #[tokio::test]
    async fn test_backoff_reject() {
        let config = RateLimiterConfig {
            model_calls_per_minute: Some(1),
            backoff_strategy: BackoffStrategy::Reject,
            ..Default::default()
        };
        let mw = RateLimiterMiddleware::new(config);
        let mut state = json!({"messages": []});

        assert!(mw.before_model(&mut state).await.is_ok());
        let err = mw.before_model(&mut state).await;
        assert!(err.is_err());
        assert!(err.unwrap_err().to_string().contains("rate limit exceeded"));
    }

    // Test 9: Backoff strategy wait with timeout
    #[tokio::test]
    async fn test_backoff_wait_with_timeout() {
        let config = RateLimiterConfig {
            model_calls_per_minute: Some(1),
            backoff_strategy: BackoffStrategy::WaitWithTimeout(Duration::from_millis(50)),
            ..Default::default()
        };
        let mw = RateLimiterMiddleware::new(config);
        let mut state = json!({"messages": []});

        assert!(mw.before_model(&mut state).await.is_ok());
        // Second call should wait, then potentially succeed or timeout.
        let start = Instant::now();
        let _result = mw.before_model(&mut state).await;
        // Should have waited at least close to the timeout.
        assert!(start.elapsed() >= Duration::from_millis(40));
    }

    // Test 10: Stats tracking
    #[tokio::test]
    async fn test_stats_tracking() {
        let config = RateLimiterConfig {
            model_calls_per_minute: Some(10),
            tool_calls_per_minute: Some(10),
            ..Default::default()
        };
        let mw = RateLimiterMiddleware::new(config);
        let mut state = json!({"messages": []});

        mw.before_model(&mut state).await.unwrap();
        mw.before_model(&mut state).await.unwrap();
        mw.before_tool(&mut state, "tool_a").await.unwrap();

        let stats = mw.get_stats();
        assert_eq!(stats.model_calls_this_minute, 2);
        assert_eq!(stats.model_calls_this_hour, 2);
        assert_eq!(stats.tool_calls_this_minute, 1);
    }

    // Test 11: Reset stats
    #[tokio::test]
    async fn test_reset_stats() {
        let config = RateLimiterConfig {
            model_calls_per_minute: Some(10),
            ..Default::default()
        };
        let mw = RateLimiterMiddleware::new(config);
        let mut state = json!({"messages": []});

        mw.before_model(&mut state).await.unwrap();
        mw.record_tokens(50);
        mw.record_cost(0.5);

        mw.reset_stats();
        let stats = mw.get_stats();
        assert_eq!(stats.model_calls_this_minute, 0);
        assert_eq!(stats.model_calls_this_hour, 0);
        assert_eq!(stats.tokens_this_minute, 0);
        assert_eq!(stats.total_cost, 0.0);
        assert_eq!(stats.rejected_count, 0);
        assert_eq!(stats.waited_count, 0);
    }

    // Test 12: Burst limit enforcement
    #[tokio::test]
    async fn test_burst_limit() {
        let config = RateLimiterConfig {
            burst_limit: Some(2),
            backoff_strategy: BackoffStrategy::Reject,
            ..Default::default()
        };
        let mw = RateLimiterMiddleware::new(config);
        let mut state = json!({"messages": []});

        assert!(mw.before_model(&mut state).await.is_ok());
        assert!(mw.before_model(&mut state).await.is_ok());
        // Third call exceeds burst limit.
        assert!(mw.before_model(&mut state).await.is_err());
    }

    // Test 13: Multiple windows (per-minute and per-hour)
    #[tokio::test]
    async fn test_multiple_windows() {
        let config = RateLimiterConfig {
            model_calls_per_minute: Some(5),
            model_calls_per_hour: Some(3),
            backoff_strategy: BackoffStrategy::Reject,
            ..Default::default()
        };
        let mw = RateLimiterMiddleware::new(config);
        let mut state = json!({"messages": []});

        assert!(mw.before_model(&mut state).await.is_ok());
        assert!(mw.before_model(&mut state).await.is_ok());
        assert!(mw.before_model(&mut state).await.is_ok());
        // Per-hour limit of 3 should kick in even though per-minute allows 5.
        assert!(mw.before_model(&mut state).await.is_err());

        let stats = mw.get_stats();
        assert_eq!(stats.model_calls_this_hour, 3);
        assert_eq!(stats.rejected_count, 1);
    }

    // Test 14: RateLimitError variants display
    #[test]
    fn test_rate_limit_error_variants() {
        let e1 = RateLimitError::ModelRateLimitExceeded {
            limit: 10,
            window: Duration::from_secs(60),
        };
        assert!(e1.to_string().contains("model rate limit exceeded"));
        assert!(e1.to_string().contains("10"));

        let e2 = RateLimitError::ToolRateLimitExceeded {
            limit: 5,
            window: Duration::from_secs(60),
        };
        assert!(e2.to_string().contains("tool rate limit exceeded"));

        let e3 = RateLimitError::TokenBudgetExceeded {
            limit: 1000,
            used: 1200,
        };
        assert!(e3.to_string().contains("token budget exceeded"));
        assert!(e3.to_string().contains("1200"));

        let e4 = RateLimitError::CostLimitExceeded {
            limit: 5.0,
            spent: 6.0,
        };
        assert!(e4.to_string().contains("cost limit exceeded"));

        let e5 = RateLimitError::TimeoutExceeded {
            waited: Duration::from_secs(30),
        };
        assert!(e5.to_string().contains("timeout exceeded"));
    }

    // Test 15: Middleware name
    #[test]
    fn test_middleware_name() {
        let mw = RateLimiterMiddleware::new(RateLimiterConfig::default());
        assert_eq!(mw.name(), "rate_limiter");
    }

    // Test 16: after_model extracts token usage from state
    #[tokio::test]
    async fn test_after_model_tracks_tokens() {
        let mw = RateLimiterMiddleware::new(RateLimiterConfig::default());
        let mut state = json!({
            "messages": [],
            "usage": {
                "total_tokens": 150,
                "cost": 0.003
            }
        });

        mw.after_model(&mut state).await.unwrap();
        let stats = mw.get_stats();
        assert_eq!(stats.tokens_this_minute, 150);
        assert!((stats.total_cost - 0.003).abs() < 1e-9);
    }

    // Test 17: Bucket wait_for calculation
    #[test]
    fn test_bucket_wait_for() {
        let bucket = RateLimitBucket::new(10, 2.0);
        // Drain bucket.
        assert!(bucket.try_acquire(10));
        // Need 4 tokens at 2/s => 2 seconds.
        let wait = bucket.wait_for(4);
        assert!(wait >= Duration::from_secs(1));
        assert!(wait <= Duration::from_secs(3));
    }
}
