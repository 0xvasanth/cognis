//! Resilience patterns for LLM API calls.
//!
//! This module provides retry strategies, circuit breakers, bulkheads, and
//! fallback chains to make LLM integrations robust against transient failures
//! and cascading outages.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// RetryStrategy
// ---------------------------------------------------------------------------

/// Describes how to compute the delay between retry attempts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum RetryStrategy {
    /// Constant delay between attempts.
    Fixed { delay_ms: u64 },
    /// Exponential back-off with a cap.
    Exponential {
        base_ms: u64,
        max_ms: u64,
        multiplier: f64,
    },
    /// Linearly increasing delay with a cap.
    Linear {
        initial_ms: u64,
        increment_ms: u64,
        max_ms: u64,
    },
    /// No delay / no retry.
    None,
}

impl RetryStrategy {
    /// Computes the delay in milliseconds for the given `attempt` (0-indexed).
    pub fn delay_for_attempt(&self, attempt: usize) -> u64 {
        match self {
            RetryStrategy::Fixed { delay_ms } => *delay_ms,
            RetryStrategy::Exponential {
                base_ms,
                max_ms,
                multiplier,
            } => {
                let delay = (*base_ms as f64) * multiplier.powi(attempt as i32);
                (delay as u64).min(*max_ms)
            }
            RetryStrategy::Linear {
                initial_ms,
                increment_ms,
                max_ms,
            } => {
                let delay = initial_ms + increment_ms * attempt as u64;
                delay.min(*max_ms)
            }
            RetryStrategy::None => 0,
        }
    }

    /// Serialises to a JSON `serde_json::Value`.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_default()
    }
}

impl fmt::Display for RetryStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RetryStrategy::Fixed { delay_ms } => write!(f, "Fixed({}ms)", delay_ms),
            RetryStrategy::Exponential {
                base_ms,
                max_ms,
                multiplier,
            } => write!(
                f,
                "Exponential(base={}ms, max={}ms, mult={})",
                base_ms, max_ms, multiplier
            ),
            RetryStrategy::Linear {
                initial_ms,
                increment_ms,
                max_ms,
            } => write!(
                f,
                "Linear(init={}ms, inc={}ms, max={}ms)",
                initial_ms, increment_ms, max_ms
            ),
            RetryStrategy::None => write!(f, "None"),
        }
    }
}

// ---------------------------------------------------------------------------
// RetryConfig
// ---------------------------------------------------------------------------

/// Configuration for retry behaviour around LLM API calls.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_attempts: usize,
    pub strategy: RetryStrategy,
    pub retryable_errors: Vec<String>,
}

impl RetryConfig {
    /// Creates a new config with the given limits and strategy.
    pub fn new(max_attempts: usize, strategy: RetryStrategy) -> Self {
        Self {
            max_attempts,
            strategy,
            retryable_errors: Vec::new(),
        }
    }

    /// Adds an error kind that should trigger a retry.
    pub fn with_retryable_error(mut self, error_kind: &str) -> Self {
        self.retryable_errors.push(error_kind.to_string());
        self
    }

    /// Returns `true` when the given attempt should be retried for `error`.
    ///
    /// Retries are allowed when:
    /// 1. `attempt < max_attempts`
    /// 2. Either the retryable list is empty (all errors retryable) **or** the
    ///    error string contains one of the registered patterns.
    pub fn should_retry(&self, attempt: usize, error: &str) -> bool {
        if attempt >= self.max_attempts {
            return false;
        }
        if self.retryable_errors.is_empty() {
            return true;
        }
        self.retryable_errors
            .iter()
            .any(|kind| error.contains(kind.as_str()))
    }

    /// Serialises to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_default()
    }
}

// ---------------------------------------------------------------------------
// CircuitState
// ---------------------------------------------------------------------------

/// State of a [`CircuitBreaker`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CircuitState {
    /// Normal operation — requests are forwarded.
    Closed,
    /// The breaker has tripped — requests are rejected.
    Open { since: String },
    /// The breaker is testing whether the downstream has recovered.
    HalfOpen,
}

impl fmt::Display for CircuitState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CircuitState::Closed => write!(f, "Closed"),
            CircuitState::Open { since } => write!(f, "Open(since={})", since),
            CircuitState::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

// ---------------------------------------------------------------------------
// CircuitBreaker
// ---------------------------------------------------------------------------

/// Prevents cascading failures by tracking consecutive errors and short-
/// circuiting calls when a threshold is exceeded.
#[derive(Debug)]
pub struct CircuitBreaker {
    failure_threshold: usize,
    recovery_timeout_ms: u64,
    inner: Mutex<CircuitBreakerInner>,
}

#[derive(Debug)]
struct CircuitBreakerInner {
    state: CircuitState,
    failure_count: usize,
    success_count: usize,
    last_failure_epoch_ms: u64,
}

impl CircuitBreaker {
    /// Creates a new breaker that opens after `failure_threshold` consecutive
    /// failures and attempts recovery after `recovery_timeout_ms`.
    pub fn new(failure_threshold: usize, recovery_timeout_ms: u64) -> Self {
        Self {
            failure_threshold,
            recovery_timeout_ms,
            inner: Mutex::new(CircuitBreakerInner {
                state: CircuitState::Closed,
                failure_count: 0,
                success_count: 0,
                last_failure_epoch_ms: 0,
            }),
        }
    }

    /// Records a successful call, resetting failure counters and closing the
    /// breaker if it was half-open.
    pub fn record_success(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.failure_count = 0;
        inner.success_count += 1;
        inner.state = CircuitState::Closed;
    }

    /// Records a failed call. If the failure threshold is reached the breaker
    /// transitions to Open.
    pub fn record_failure(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.failure_count += 1;
        if inner.failure_count >= self.failure_threshold {
            inner.state = CircuitState::Open {
                since: format!("epoch_ms_{}", self.now_ms()),
            };
            inner.last_failure_epoch_ms = self.now_ms();
        }
    }

    /// Returns `true` when a request may proceed.
    ///
    /// * **Closed** → always available.
    /// * **Open** → available only after the recovery timeout has elapsed, in
    ///   which case the state moves to HalfOpen.
    /// * **HalfOpen** → available (one trial request is allowed).
    pub fn is_available(&self) -> bool {
        let mut inner = self.inner.lock().unwrap();
        match &inner.state {
            CircuitState::Closed => true,
            CircuitState::Open { .. } => {
                let elapsed = self.now_ms().saturating_sub(inner.last_failure_epoch_ms);
                if elapsed >= self.recovery_timeout_ms {
                    inner.state = CircuitState::HalfOpen;
                    true
                } else {
                    false
                }
            }
            CircuitState::HalfOpen => true,
        }
    }

    /// Returns the current [`CircuitState`].
    pub fn state(&self) -> CircuitState {
        self.inner.lock().unwrap().state.clone()
    }

    /// Returns the running count of consecutive failures.
    pub fn failure_count(&self) -> usize {
        self.inner.lock().unwrap().failure_count
    }

    /// Resets the breaker to its initial closed state.
    pub fn reset(&self) {
        let mut inner = self.inner.lock().unwrap();
        inner.state = CircuitState::Closed;
        inner.failure_count = 0;
        inner.last_failure_epoch_ms = 0;
    }

    /// Serialises the breaker status to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        let inner = self.inner.lock().unwrap();
        serde_json::json!({
            "state": format!("{}", inner.state),
            "failure_count": inner.failure_count,
            "success_count": inner.success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_ms": self.recovery_timeout_ms,
        })
    }

    // Simple monotonic-ish timestamp helper (milliseconds).
    fn now_ms(&self) -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }
}

// ---------------------------------------------------------------------------
// Bulkhead
// ---------------------------------------------------------------------------

/// Limits concurrent access to a resource, preventing any single caller from
/// monopolising capacity.
#[derive(Debug)]
pub struct Bulkhead {
    max_concurrent: usize,
    available: Arc<AtomicUsize>,
}

impl Bulkhead {
    /// Creates a bulkhead that allows up to `max_concurrent` simultaneous
    /// permits.
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            available: Arc::new(AtomicUsize::new(max_concurrent)),
        }
    }

    /// Attempts to acquire a permit. Returns `None` if the bulkhead is
    /// saturated.
    pub fn try_acquire(&self) -> Option<BulkheadPermit> {
        loop {
            let current = self.available.load(Ordering::SeqCst);
            if current == 0 {
                return None;
            }
            if self
                .available
                .compare_exchange(current, current - 1, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return Some(BulkheadPermit {
                    available: Arc::clone(&self.available),
                });
            }
        }
    }

    /// Number of permits currently available.
    pub fn available(&self) -> usize {
        self.available.load(Ordering::SeqCst)
    }

    /// Number of permits currently in use.
    pub fn active(&self) -> usize {
        self.max_concurrent - self.available.load(Ordering::SeqCst)
    }

    /// Maximum concurrent permits this bulkhead allows.
    pub fn max_concurrent(&self) -> usize {
        self.max_concurrent
    }

    /// Serialises status to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "max_concurrent": self.max_concurrent,
            "available": self.available(),
            "active": self.active(),
        })
    }
}

// ---------------------------------------------------------------------------
// BulkheadPermit
// ---------------------------------------------------------------------------

/// RAII guard returned by [`Bulkhead::try_acquire`]. When dropped the permit
/// is returned to the bulkhead.
#[derive(Debug)]
pub struct BulkheadPermit {
    available: Arc<AtomicUsize>,
}

impl Drop for BulkheadPermit {
    fn drop(&mut self) {
        self.available.fetch_add(1, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// FallbackChain
// ---------------------------------------------------------------------------

/// Tries multiple providers / strategies in priority order, skipping entries
/// that have been marked as failed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FallbackChain {
    entries: Vec<FallbackEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FallbackEntry {
    name: String,
    priority: usize,
    failed: bool,
}

impl FallbackChain {
    /// Creates an empty fallback chain.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds a named fallback with the given priority (lower = tried first).
    pub fn add_fallback(&mut self, name: String, priority: usize) {
        self.entries.push(FallbackEntry {
            name,
            priority,
            failed: false,
        });
        self.entries.sort_by_key(|e| e.priority);
    }

    /// Returns the next available (non-failed) fallback name, or `None`.
    pub fn next_fallback(&self) -> Option<String> {
        self.entries
            .iter()
            .find(|e| !e.failed)
            .map(|e| e.name.clone())
    }

    /// Marks the given provider as failed so it is skipped in future lookups.
    pub fn mark_failed(&mut self, name: &str) {
        for entry in &mut self.entries {
            if entry.name == name {
                entry.failed = true;
            }
        }
    }

    /// Marks a previously-failed provider as recovered.
    pub fn mark_recovered(&mut self, name: &str) {
        for entry in &mut self.entries {
            if entry.name == name {
                entry.failed = false;
            }
        }
    }

    /// Number of providers that are currently available.
    pub fn available_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.failed).count()
    }

    /// Serialises to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or_default()
    }
}

impl Default for FallbackChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ResiliencePolicy
// ---------------------------------------------------------------------------

/// Combines retry, circuit-breaker, and bulkhead patterns into a single
/// policy that governs whether a request should proceed.
#[derive(Debug)]
pub struct ResiliencePolicy {
    pub name: String,
    pub retry_config: Option<RetryConfig>,
    pub circuit_breaker: Option<CircuitBreaker>,
    pub bulkhead: Option<Bulkhead>,
}

impl ResiliencePolicy {
    /// Creates a named policy with no patterns configured.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            retry_config: None,
            circuit_breaker: None,
            bulkhead: None,
        }
    }

    /// Attaches a retry configuration.
    pub fn with_retry(mut self, config: RetryConfig) -> Self {
        self.retry_config = Some(config);
        self
    }

    /// Attaches a circuit breaker.
    pub fn with_circuit_breaker(mut self, cb: CircuitBreaker) -> Self {
        self.circuit_breaker = Some(cb);
        self
    }

    /// Attaches a bulkhead.
    pub fn with_bulkhead(mut self, bh: Bulkhead) -> Self {
        self.bulkhead = Some(bh);
        self
    }

    /// Returns `true` when the policy allows execution (circuit breaker is
    /// available **and** a bulkhead permit could be obtained).
    pub fn can_execute(&self) -> bool {
        if let Some(ref cb) = self.circuit_breaker {
            if !cb.is_available() {
                return false;
            }
        }
        if let Some(ref bh) = self.bulkhead {
            if bh.available() == 0 {
                return false;
            }
        }
        true
    }

    /// Records a successful execution across all sub-patterns.
    pub fn record_success(&self) {
        if let Some(ref cb) = self.circuit_breaker {
            cb.record_success();
        }
    }

    /// Records a failed execution across all sub-patterns.
    pub fn record_failure(&self, _error: &str) {
        if let Some(ref cb) = self.circuit_breaker {
            cb.record_failure();
        }
    }

    /// Serialises to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "name": self.name,
            "retry_config": self.retry_config.as_ref().map(|r| r.to_json()),
            "circuit_breaker": self.circuit_breaker.as_ref().map(|c| c.to_json()),
            "bulkhead": self.bulkhead.as_ref().map(|b| b.to_json()),
        })
    }
}

// ---------------------------------------------------------------------------
// ResilienceMetrics
// ---------------------------------------------------------------------------

/// Tracks resilience-related events for observability.
#[derive(Debug)]
pub struct ResilienceMetrics {
    total_attempts: AtomicUsize,
    total_retries: AtomicUsize,
    total_successes: AtomicUsize,
    circuit_opens: AtomicUsize,
    bulkhead_rejects: AtomicUsize,
}

impl ResilienceMetrics {
    /// Creates a fresh metrics instance with all counters at zero.
    pub fn new() -> Self {
        Self {
            total_attempts: AtomicUsize::new(0),
            total_retries: AtomicUsize::new(0),
            total_successes: AtomicUsize::new(0),
            circuit_opens: AtomicUsize::new(0),
            bulkhead_rejects: AtomicUsize::new(0),
        }
    }

    /// Records an attempt (first try or retry).
    pub fn record_attempt(&self) {
        self.total_attempts.fetch_add(1, Ordering::SeqCst);
    }

    /// Records a retry for a given attempt number. Also increments
    /// `total_attempts`.
    pub fn record_retry(&self, _attempt: usize) {
        self.total_retries.fetch_add(1, Ordering::SeqCst);
        self.total_attempts.fetch_add(1, Ordering::SeqCst);
    }

    /// Records a successful outcome.
    pub fn record_success(&self) {
        self.total_successes.fetch_add(1, Ordering::SeqCst);
    }

    /// Records that a circuit breaker opened.
    pub fn record_circuit_open(&self) {
        self.circuit_opens.fetch_add(1, Ordering::SeqCst);
    }

    /// Records that a bulkhead rejected a request.
    pub fn record_bulkhead_reject(&self) {
        self.bulkhead_rejects.fetch_add(1, Ordering::SeqCst);
    }

    pub fn total_attempts(&self) -> usize {
        self.total_attempts.load(Ordering::SeqCst)
    }

    pub fn total_retries(&self) -> usize {
        self.total_retries.load(Ordering::SeqCst)
    }

    pub fn circuit_opens(&self) -> usize {
        self.circuit_opens.load(Ordering::SeqCst)
    }

    pub fn bulkhead_rejects(&self) -> usize {
        self.bulkhead_rejects.load(Ordering::SeqCst)
    }

    /// Success rate as a value in `[0.0, 1.0]`. Returns `0.0` when there are
    /// no attempts.
    pub fn success_rate(&self) -> f64 {
        let attempts = self.total_attempts() as f64;
        if attempts == 0.0 {
            return 0.0;
        }
        self.total_successes.load(Ordering::SeqCst) as f64 / attempts
    }

    /// Serialises to JSON.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "total_attempts": self.total_attempts(),
            "total_retries": self.total_retries(),
            "circuit_opens": self.circuit_opens(),
            "bulkhead_rejects": self.bulkhead_rejects(),
            "success_rate": self.success_rate(),
        })
    }
}

impl Default for ResilienceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // RetryStrategy
    // -----------------------------------------------------------------------

    #[test]
    fn fixed_delay_is_constant() {
        let s = RetryStrategy::Fixed { delay_ms: 100 };
        assert_eq!(s.delay_for_attempt(0), 100);
        assert_eq!(s.delay_for_attempt(5), 100);
        assert_eq!(s.delay_for_attempt(100), 100);
    }

    #[test]
    fn exponential_delay_grows() {
        let s = RetryStrategy::Exponential {
            base_ms: 100,
            max_ms: 10000,
            multiplier: 2.0,
        };
        assert_eq!(s.delay_for_attempt(0), 100); // 100 * 2^0
        assert_eq!(s.delay_for_attempt(1), 200); // 100 * 2^1
        assert_eq!(s.delay_for_attempt(2), 400); // 100 * 2^2
        assert_eq!(s.delay_for_attempt(3), 800); // 100 * 2^3
    }

    #[test]
    fn exponential_delay_caps_at_max() {
        let s = RetryStrategy::Exponential {
            base_ms: 100,
            max_ms: 500,
            multiplier: 2.0,
        };
        assert_eq!(s.delay_for_attempt(5), 500);
        assert_eq!(s.delay_for_attempt(10), 500);
    }

    #[test]
    fn linear_delay_increases() {
        let s = RetryStrategy::Linear {
            initial_ms: 100,
            increment_ms: 50,
            max_ms: 1000,
        };
        assert_eq!(s.delay_for_attempt(0), 100);
        assert_eq!(s.delay_for_attempt(1), 150);
        assert_eq!(s.delay_for_attempt(4), 300);
    }

    #[test]
    fn linear_delay_caps_at_max() {
        let s = RetryStrategy::Linear {
            initial_ms: 100,
            increment_ms: 200,
            max_ms: 500,
        };
        assert_eq!(s.delay_for_attempt(10), 500);
    }

    #[test]
    fn none_strategy_returns_zero() {
        let s = RetryStrategy::None;
        assert_eq!(s.delay_for_attempt(0), 0);
        assert_eq!(s.delay_for_attempt(99), 0);
    }

    #[test]
    fn retry_strategy_display_fixed() {
        let s = RetryStrategy::Fixed { delay_ms: 42 };
        assert_eq!(format!("{}", s), "Fixed(42ms)");
    }

    #[test]
    fn retry_strategy_display_exponential() {
        let s = RetryStrategy::Exponential {
            base_ms: 10,
            max_ms: 100,
            multiplier: 3.0,
        };
        assert!(format!("{}", s).contains("Exponential"));
    }

    #[test]
    fn retry_strategy_display_linear() {
        let s = RetryStrategy::Linear {
            initial_ms: 10,
            increment_ms: 5,
            max_ms: 100,
        };
        assert!(format!("{}", s).contains("Linear"));
    }

    #[test]
    fn retry_strategy_display_none() {
        assert_eq!(format!("{}", RetryStrategy::None), "None");
    }

    #[test]
    fn retry_strategy_to_json() {
        let s = RetryStrategy::Fixed { delay_ms: 50 };
        let j = s.to_json();
        assert_eq!(j["Fixed"]["delay_ms"], 50);
    }

    #[test]
    fn retry_strategy_serialization_roundtrip() {
        let original = RetryStrategy::Exponential {
            base_ms: 100,
            max_ms: 5000,
            multiplier: 2.5,
        };
        let json_str = serde_json::to_string(&original).unwrap();
        let restored: RetryStrategy = serde_json::from_str(&json_str).unwrap();
        assert_eq!(original, restored);
    }

    // -----------------------------------------------------------------------
    // RetryConfig
    // -----------------------------------------------------------------------

    #[test]
    fn retry_config_new() {
        let cfg = RetryConfig::new(3, RetryStrategy::None);
        assert_eq!(cfg.max_attempts, 3);
        assert!(cfg.retryable_errors.is_empty());
    }

    #[test]
    fn retry_config_with_retryable_error() {
        let cfg = RetryConfig::new(3, RetryStrategy::None)
            .with_retryable_error("timeout")
            .with_retryable_error("rate_limit");
        assert_eq!(cfg.retryable_errors.len(), 2);
    }

    #[test]
    fn should_retry_within_attempts_no_filter() {
        let cfg = RetryConfig::new(3, RetryStrategy::None);
        assert!(cfg.should_retry(0, "anything"));
        assert!(cfg.should_retry(2, "whatever"));
        assert!(!cfg.should_retry(3, "nope"));
    }

    #[test]
    fn should_retry_with_matching_error() {
        let cfg = RetryConfig::new(5, RetryStrategy::None).with_retryable_error("timeout");
        assert!(cfg.should_retry(0, "connection timeout"));
        assert!(!cfg.should_retry(0, "auth failure"));
    }

    #[test]
    fn should_retry_exceeds_max() {
        let cfg = RetryConfig::new(2, RetryStrategy::None);
        assert!(!cfg.should_retry(2, "err"));
        assert!(!cfg.should_retry(10, "err"));
    }

    #[test]
    fn retry_config_to_json() {
        let cfg = RetryConfig::new(3, RetryStrategy::Fixed { delay_ms: 100 })
            .with_retryable_error("timeout");
        let j = cfg.to_json();
        assert_eq!(j["max_attempts"], 3);
        assert!(j["retryable_errors"].is_array());
    }

    #[test]
    fn retry_config_serialization_roundtrip() {
        let cfg = RetryConfig::new(4, RetryStrategy::Fixed { delay_ms: 200 })
            .with_retryable_error("rate_limit");
        let json_str = serde_json::to_string(&cfg).unwrap();
        let restored: RetryConfig = serde_json::from_str(&json_str).unwrap();
        assert_eq!(restored.max_attempts, 4);
        assert_eq!(restored.retryable_errors, vec!["rate_limit".to_string()]);
    }

    // -----------------------------------------------------------------------
    // CircuitState
    // -----------------------------------------------------------------------

    #[test]
    fn circuit_state_display_closed() {
        assert_eq!(format!("{}", CircuitState::Closed), "Closed");
    }

    #[test]
    fn circuit_state_display_open() {
        let s = CircuitState::Open {
            since: "now".into(),
        };
        assert!(format!("{}", s).contains("Open"));
        assert!(format!("{}", s).contains("now"));
    }

    #[test]
    fn circuit_state_display_half_open() {
        assert_eq!(format!("{}", CircuitState::HalfOpen), "HalfOpen");
    }

    // -----------------------------------------------------------------------
    // CircuitBreaker
    // -----------------------------------------------------------------------

    #[test]
    fn circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new(3, 1000);
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_available());
        assert_eq!(cb.failure_count(), 0);
    }

    #[test]
    fn circuit_breaker_opens_on_threshold() {
        let cb = CircuitBreaker::new(2, 5000);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        match cb.state() {
            CircuitState::Open { .. } => {}
            other => panic!("expected Open, got {:?}", other),
        }
    }

    #[test]
    fn circuit_breaker_open_rejects_calls() {
        let cb = CircuitBreaker::new(1, 999_999);
        cb.record_failure();
        assert!(!cb.is_available());
    }

    #[test]
    fn circuit_breaker_success_resets() {
        let cb = CircuitBreaker::new(3, 1000);
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.failure_count(), 0);
    }

    #[test]
    fn circuit_breaker_reset() {
        let cb = CircuitBreaker::new(1, 1000);
        cb.record_failure();
        assert!(!cb.is_available());
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_available());
    }

    #[test]
    fn circuit_breaker_half_open_after_timeout() {
        let cb = CircuitBreaker::new(1, 0); // 0ms timeout = immediate recovery
        cb.record_failure();
        // With 0ms timeout, is_available should transition to HalfOpen
        assert!(cb.is_available());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn circuit_breaker_to_json() {
        let cb = CircuitBreaker::new(5, 3000);
        cb.record_failure();
        let j = cb.to_json();
        assert_eq!(j["failure_threshold"], 5);
        assert_eq!(j["failure_count"], 1);
        assert_eq!(j["recovery_timeout_ms"], 3000);
    }

    #[test]
    fn circuit_breaker_multiple_failures_then_success() {
        let cb = CircuitBreaker::new(3, 0);
        cb.record_failure();
        cb.record_failure();
        cb.record_failure();
        // Now open; with 0ms timeout it transitions to HalfOpen on check
        assert!(cb.is_available());
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn circuit_breaker_failure_count_increments() {
        let cb = CircuitBreaker::new(10, 1000);
        for i in 1..=5 {
            cb.record_failure();
            assert_eq!(cb.failure_count(), i);
        }
    }

    // -----------------------------------------------------------------------
    // Bulkhead
    // -----------------------------------------------------------------------

    #[test]
    fn bulkhead_new() {
        let bh = Bulkhead::new(5);
        assert_eq!(bh.max_concurrent(), 5);
        assert_eq!(bh.available(), 5);
        assert_eq!(bh.active(), 0);
    }

    #[test]
    fn bulkhead_acquire_reduces_available() {
        let bh = Bulkhead::new(3);
        let _p1 = bh.try_acquire().unwrap();
        assert_eq!(bh.available(), 2);
        assert_eq!(bh.active(), 1);
    }

    #[test]
    fn bulkhead_drop_permit_restores() {
        let bh = Bulkhead::new(2);
        {
            let _p = bh.try_acquire().unwrap();
            assert_eq!(bh.available(), 1);
        }
        assert_eq!(bh.available(), 2);
    }

    #[test]
    fn bulkhead_exhaustion() {
        let bh = Bulkhead::new(1);
        let _p1 = bh.try_acquire().unwrap();
        assert!(bh.try_acquire().is_none());
    }

    #[test]
    fn bulkhead_multiple_permits() {
        let bh = Bulkhead::new(3);
        let _p1 = bh.try_acquire().unwrap();
        let _p2 = bh.try_acquire().unwrap();
        let _p3 = bh.try_acquire().unwrap();
        assert!(bh.try_acquire().is_none());
        assert_eq!(bh.active(), 3);
    }

    #[test]
    fn bulkhead_release_and_reacquire() {
        let bh = Bulkhead::new(1);
        let p = bh.try_acquire().unwrap();
        drop(p);
        assert!(bh.try_acquire().is_some());
    }

    #[test]
    fn bulkhead_to_json() {
        let bh = Bulkhead::new(4);
        let _p = bh.try_acquire().unwrap();
        let j = bh.to_json();
        assert_eq!(j["max_concurrent"], 4);
        assert_eq!(j["available"], 3);
        assert_eq!(j["active"], 1);
    }

    // -----------------------------------------------------------------------
    // FallbackChain
    // -----------------------------------------------------------------------

    #[test]
    fn fallback_chain_empty() {
        let chain = FallbackChain::new();
        assert_eq!(chain.available_count(), 0);
        assert!(chain.next_fallback().is_none());
    }

    #[test]
    fn fallback_chain_priority_order() {
        let mut chain = FallbackChain::new();
        chain.add_fallback("secondary".into(), 2);
        chain.add_fallback("primary".into(), 1);
        assert_eq!(chain.next_fallback().unwrap(), "primary");
    }

    #[test]
    fn fallback_chain_mark_failed() {
        let mut chain = FallbackChain::new();
        chain.add_fallback("a".into(), 1);
        chain.add_fallback("b".into(), 2);
        chain.mark_failed("a");
        assert_eq!(chain.next_fallback().unwrap(), "b");
        assert_eq!(chain.available_count(), 1);
    }

    #[test]
    fn fallback_chain_all_failed() {
        let mut chain = FallbackChain::new();
        chain.add_fallback("a".into(), 1);
        chain.mark_failed("a");
        assert!(chain.next_fallback().is_none());
        assert_eq!(chain.available_count(), 0);
    }

    #[test]
    fn fallback_chain_mark_recovered() {
        let mut chain = FallbackChain::new();
        chain.add_fallback("a".into(), 1);
        chain.mark_failed("a");
        assert!(chain.next_fallback().is_none());
        chain.mark_recovered("a");
        assert_eq!(chain.next_fallback().unwrap(), "a");
    }

    #[test]
    fn fallback_chain_available_count() {
        let mut chain = FallbackChain::new();
        chain.add_fallback("x".into(), 1);
        chain.add_fallback("y".into(), 2);
        chain.add_fallback("z".into(), 3);
        assert_eq!(chain.available_count(), 3);
        chain.mark_failed("y");
        assert_eq!(chain.available_count(), 2);
    }

    #[test]
    fn fallback_chain_to_json() {
        let mut chain = FallbackChain::new();
        chain.add_fallback("provider_a".into(), 1);
        let j = chain.to_json();
        assert!(j["entries"].is_array());
    }

    #[test]
    fn fallback_chain_default() {
        let chain = FallbackChain::default();
        assert_eq!(chain.available_count(), 0);
    }

    // -----------------------------------------------------------------------
    // ResiliencePolicy
    // -----------------------------------------------------------------------

    #[test]
    fn policy_new_allows_execution() {
        let policy = ResiliencePolicy::new("test");
        assert_eq!(policy.name, "test");
        assert!(policy.can_execute());
    }

    #[test]
    fn policy_with_retry() {
        let policy = ResiliencePolicy::new("p")
            .with_retry(RetryConfig::new(3, RetryStrategy::Fixed { delay_ms: 100 }));
        assert!(policy.retry_config.is_some());
    }

    #[test]
    fn policy_with_circuit_breaker_closed() {
        let policy = ResiliencePolicy::new("p").with_circuit_breaker(CircuitBreaker::new(5, 1000));
        assert!(policy.can_execute());
    }

    #[test]
    fn policy_circuit_breaker_blocks() {
        let policy =
            ResiliencePolicy::new("p").with_circuit_breaker(CircuitBreaker::new(1, 999_999));
        policy.record_failure("err");
        assert!(!policy.can_execute());
    }

    #[test]
    fn policy_with_bulkhead() {
        let policy = ResiliencePolicy::new("p").with_bulkhead(Bulkhead::new(2));
        assert!(policy.can_execute());
    }

    #[test]
    fn policy_bulkhead_blocks_when_full() {
        let policy = ResiliencePolicy::new("p").with_bulkhead(Bulkhead::new(1));
        let _permit = policy.bulkhead.as_ref().unwrap().try_acquire().unwrap();
        assert!(!policy.can_execute());
    }

    #[test]
    fn policy_record_success_resets_cb() {
        let policy = ResiliencePolicy::new("p").with_circuit_breaker(CircuitBreaker::new(2, 1000));
        policy.record_failure("e");
        assert_eq!(policy.circuit_breaker.as_ref().unwrap().failure_count(), 1);
        policy.record_success();
        assert_eq!(policy.circuit_breaker.as_ref().unwrap().failure_count(), 0);
    }

    #[test]
    fn policy_to_json() {
        let policy = ResiliencePolicy::new("my_policy")
            .with_retry(RetryConfig::new(3, RetryStrategy::None))
            .with_circuit_breaker(CircuitBreaker::new(5, 2000))
            .with_bulkhead(Bulkhead::new(10));
        let j = policy.to_json();
        assert_eq!(j["name"], "my_policy");
        assert!(j["retry_config"].is_object());
        assert!(j["circuit_breaker"].is_object());
        assert!(j["bulkhead"].is_object());
    }

    #[test]
    fn policy_combined_cb_and_bulkhead() {
        let policy = ResiliencePolicy::new("combo")
            .with_circuit_breaker(CircuitBreaker::new(1, 999_999))
            .with_bulkhead(Bulkhead::new(1));
        assert!(policy.can_execute());
        policy.record_failure("err");
        assert!(!policy.can_execute());
    }

    // -----------------------------------------------------------------------
    // ResilienceMetrics
    // -----------------------------------------------------------------------

    #[test]
    fn metrics_new_zeroes() {
        let m = ResilienceMetrics::new();
        assert_eq!(m.total_attempts(), 0);
        assert_eq!(m.total_retries(), 0);
        assert_eq!(m.circuit_opens(), 0);
        assert_eq!(m.bulkhead_rejects(), 0);
        assert_eq!(m.success_rate(), 0.0);
    }

    #[test]
    fn metrics_record_attempt() {
        let m = ResilienceMetrics::new();
        m.record_attempt();
        m.record_attempt();
        assert_eq!(m.total_attempts(), 2);
    }

    #[test]
    fn metrics_record_retry_increments_both() {
        let m = ResilienceMetrics::new();
        m.record_retry(1);
        assert_eq!(m.total_retries(), 1);
        assert_eq!(m.total_attempts(), 1);
    }

    #[test]
    fn metrics_record_circuit_open() {
        let m = ResilienceMetrics::new();
        m.record_circuit_open();
        m.record_circuit_open();
        assert_eq!(m.circuit_opens(), 2);
    }

    #[test]
    fn metrics_record_bulkhead_reject() {
        let m = ResilienceMetrics::new();
        m.record_bulkhead_reject();
        assert_eq!(m.bulkhead_rejects(), 1);
    }

    #[test]
    fn metrics_success_rate_calculation() {
        let m = ResilienceMetrics::new();
        m.record_attempt();
        m.record_attempt();
        m.record_attempt();
        m.record_attempt();
        m.record_success();
        m.record_success();
        m.record_success();
        assert!((m.success_rate() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_success_rate_zero_attempts() {
        let m = ResilienceMetrics::new();
        assert_eq!(m.success_rate(), 0.0);
    }

    #[test]
    fn metrics_to_json() {
        let m = ResilienceMetrics::new();
        m.record_attempt();
        m.record_success();
        let j = m.to_json();
        assert_eq!(j["total_attempts"], 1);
        assert_eq!(j["total_retries"], 0);
    }

    #[test]
    fn metrics_default() {
        let m = ResilienceMetrics::default();
        assert_eq!(m.total_attempts(), 0);
    }
}
