//! Configurable retry policies with backoff strategies and circuit breaker.
//!
//! This module provides a rich retry policy framework including:
//! - [`RetryPolicy`] with configurable max retries, delays, and backoff
//! - [`BackoffStrategy`] enum supporting Fixed, Linear, Exponential, and Jitter modes
//! - [`RetryCondition`] trait for custom retry decisions
//! - [`RetryContext`] for tracking attempt metadata
//! - [`RetryBudget`] for shared retry budgets across operations
//! - [`CircuitBreaker`] for fail-fast after repeated failures
//! - [`RetryPolicyBuilder`] for fluent construction

use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// BackoffStrategy
// ---------------------------------------------------------------------------

/// Strategy for computing delays between retry attempts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub enum BackoffStrategy {
    /// Constant delay between retries.
    Fixed,
    /// Delay increases linearly: `initial_delay * attempt`.
    Linear,
    /// Delay doubles each attempt: `initial_delay * multiplier^(attempt-1)`.
    #[default]
    Exponential,
    /// Exponential with random jitter added (up to the computed delay).
    ExponentialWithJitter,
    /// Custom delay sequence. Falls back to the last value when exhausted.
    Custom(Vec<u64>),
}

// ---------------------------------------------------------------------------
// RetryCondition
// ---------------------------------------------------------------------------

/// Decides whether an error warrants a retry.
pub trait RetryCondition: Send + Sync {
    /// Return `true` if the operation should be retried for this error.
    fn should_retry(&self, error: &str) -> bool;
}

/// Always retry, regardless of the error.
pub struct AlwaysRetry;

impl RetryCondition for AlwaysRetry {
    fn should_retry(&self, _error: &str) -> bool {
        true
    }
}

/// Never retry.
pub struct NeverRetry;

impl RetryCondition for NeverRetry {
    fn should_retry(&self, _error: &str) -> bool {
        false
    }
}

/// Retry when the error message contains any of the given substrings.
pub struct ContainsRetry {
    pub patterns: Vec<String>,
}

impl ContainsRetry {
    pub fn new(patterns: Vec<String>) -> Self {
        Self { patterns }
    }
}

impl RetryCondition for ContainsRetry {
    fn should_retry(&self, error: &str) -> bool {
        let lower = error.to_lowercase();
        self.patterns
            .iter()
            .any(|p| lower.contains(&p.to_lowercase()))
    }
}

// ---------------------------------------------------------------------------
// RetryContext
// ---------------------------------------------------------------------------

/// Tracks metadata across retry attempts.
#[derive(Debug, Clone)]
pub struct RetryContext {
    /// Current attempt number (1-based).
    pub attempt: u32,
    /// When the first attempt started.
    pub started_at: Instant,
    /// The most recent error message, if any.
    pub last_error: Option<String>,
    /// All error messages collected across attempts.
    pub all_errors: Vec<String>,
}

impl RetryContext {
    /// Create a new context starting now.
    pub fn new() -> Self {
        Self {
            attempt: 0,
            started_at: Instant::now(),
            last_error: None,
            all_errors: Vec::new(),
        }
    }

    /// Total elapsed time since the first attempt.
    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    /// Record a failed attempt.
    pub fn record_error(&mut self, error: String) {
        self.attempt += 1;
        self.last_error = Some(error.clone());
        self.all_errors.push(error);
    }

    /// Record a successful attempt.
    pub fn record_success(&mut self) {
        self.attempt += 1;
    }
}

impl Default for RetryContext {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RetryOutcome
// ---------------------------------------------------------------------------

/// The outcome of a retry-managed operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RetryOutcome<T> {
    /// Operation succeeded with the given value.
    Success(T),
    /// All retry attempts exhausted.
    Exhausted {
        /// Total attempts made.
        attempts: u32,
        /// The last error message.
        last_error: String,
    },
    /// The circuit breaker is open; the operation was not attempted.
    CircuitOpen,
    /// The shared retry budget has been exhausted.
    BudgetExhausted,
}

impl<T> RetryOutcome<T> {
    /// Returns `true` if the outcome is [`RetryOutcome::Success`].
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success(_))
    }
}

// ---------------------------------------------------------------------------
// RetryBudget
// ---------------------------------------------------------------------------

/// A shared budget that limits the total number of retries across operations.
///
/// Thread-safe via atomic operations; can be cloned and shared.
#[derive(Debug, Clone)]
pub struct RetryBudget {
    remaining: Arc<AtomicU64>,
    total: u64,
}

impl RetryBudget {
    /// Create a new budget with the given total retry allowance.
    pub fn new(total: u64) -> Self {
        Self {
            remaining: Arc::new(AtomicU64::new(total)),
            total,
        }
    }

    /// Try to consume one retry from the budget. Returns `true` if successful.
    pub fn try_acquire(&self) -> bool {
        loop {
            let current = self.remaining.load(Ordering::SeqCst);
            if current == 0 {
                return false;
            }
            if self
                .remaining
                .compare_exchange(current, current - 1, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                return true;
            }
        }
    }

    /// Number of retries remaining.
    pub fn remaining(&self) -> u64 {
        self.remaining.load(Ordering::SeqCst)
    }

    /// The original total budget.
    pub fn total(&self) -> u64 {
        self.total
    }

    /// Reset the budget to its original total.
    pub fn reset(&self) {
        self.remaining.store(self.total, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// CircuitState / CircuitBreaker
// ---------------------------------------------------------------------------

/// State of a circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CircuitState {
    /// Normal operation; requests flow through.
    Closed,
    /// Too many failures; requests are rejected immediately.
    Open,
    /// A trial request is allowed to test recovery.
    HalfOpen,
}

impl fmt::Display for CircuitState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => write!(f, "Closed"),
            Self::Open => write!(f, "Open"),
            Self::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

/// A circuit breaker that trips after a threshold of consecutive failures.
///
/// After tripping, it enters the [`CircuitState::Open`] state for a
/// configurable cooldown duration, then transitions to [`CircuitState::HalfOpen`]
/// to allow a trial request.
#[derive(Debug, Clone)]
pub struct CircuitBreaker {
    /// Number of consecutive failures required to trip the breaker.
    pub failure_threshold: u32,
    /// How long the circuit stays open before transitioning to half-open.
    pub cooldown: Duration,
    consecutive_failures: u32,
    state: CircuitState,
    opened_at: Option<Instant>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    pub fn new(failure_threshold: u32, cooldown: Duration) -> Self {
        Self {
            failure_threshold,
            cooldown,
            consecutive_failures: 0,
            state: CircuitState::Closed,
            opened_at: None,
        }
    }

    /// Current state, accounting for cooldown expiry.
    pub fn state(&mut self) -> CircuitState {
        if self.state == CircuitState::Open {
            if let Some(opened) = self.opened_at {
                if opened.elapsed() >= self.cooldown {
                    self.state = CircuitState::HalfOpen;
                }
            }
        }
        self.state
    }

    /// Whether a request is currently allowed.
    pub fn is_allowed(&mut self) -> bool {
        match self.state() {
            CircuitState::Closed => true,
            CircuitState::HalfOpen => true,
            CircuitState::Open => false,
        }
    }

    /// Record a successful operation.
    pub fn record_success(&mut self) {
        self.consecutive_failures = 0;
        self.state = CircuitState::Closed;
        self.opened_at = None;
    }

    /// Record a failed operation.
    pub fn record_failure(&mut self) {
        self.consecutive_failures += 1;
        if self.consecutive_failures >= self.failure_threshold {
            self.state = CircuitState::Open;
            self.opened_at = Some(Instant::now());
        }
    }

    /// Manually reset the circuit breaker to closed.
    pub fn reset(&mut self) {
        self.consecutive_failures = 0;
        self.state = CircuitState::Closed;
        self.opened_at = None;
    }

    /// Number of consecutive failures recorded so far.
    pub fn consecutive_failures(&self) -> u32 {
        self.consecutive_failures
    }
}

// ---------------------------------------------------------------------------
// DelayCalculator
// ---------------------------------------------------------------------------

/// Computes the delay before the next retry attempt.
pub struct DelayCalculator;

impl DelayCalculator {
    /// Compute the delay for the given attempt number (1-based) and strategy.
    pub fn compute(
        strategy: &BackoffStrategy,
        attempt: u32,
        initial_delay: Duration,
        max_delay: Duration,
        backoff_multiplier: f64,
    ) -> Duration {
        let raw = match strategy {
            BackoffStrategy::Fixed => initial_delay,
            BackoffStrategy::Linear => initial_delay * attempt,
            BackoffStrategy::Exponential => {
                let factor = backoff_multiplier.powi(attempt.saturating_sub(1) as i32);
                initial_delay.mul_f64(factor)
            }
            BackoffStrategy::ExponentialWithJitter => {
                let factor = backoff_multiplier.powi(attempt.saturating_sub(1) as i32);
                let base = initial_delay.mul_f64(factor);
                // Deterministic pseudo-jitter based on attempt number.
                // We avoid pulling in `rand` by using a simple hash-like function.
                let jitter_frac = Self::pseudo_random_fraction(attempt);
                let jitter = base.mul_f64(jitter_frac);
                base.checked_sub(jitter).unwrap_or(initial_delay)
            }
            BackoffStrategy::Custom(delays) => {
                let idx = (attempt.saturating_sub(1) as usize).min(delays.len().saturating_sub(1));
                if delays.is_empty() {
                    initial_delay
                } else {
                    Duration::from_millis(delays[idx])
                }
            }
        };
        raw.min(max_delay)
    }

    /// A simple deterministic pseudo-random fraction in [0, 1) seeded by `n`.
    fn pseudo_random_fraction(n: u32) -> f64 {
        // Use a simple hash to produce a fraction.
        let hash = (n.wrapping_mul(2654435761)) as f64 / u32::MAX as f64;
        hash.abs().fract()
    }
}

// ---------------------------------------------------------------------------
// RetryPolicy
// ---------------------------------------------------------------------------

/// A configurable retry policy.
///
/// Combines a backoff strategy, retry condition, optional circuit breaker,
/// and optional shared budget to govern retry behavior.
#[derive(Clone)]
pub struct RetryPolicy {
    /// Maximum number of retry attempts (not counting the initial attempt).
    pub max_retries: u32,
    /// Delay before the first retry.
    pub initial_delay: Duration,
    /// Maximum delay between retries.
    pub max_delay: Duration,
    /// Multiplier applied per attempt for exponential backoff.
    pub backoff_multiplier: f64,
    /// The backoff strategy to use.
    pub strategy: BackoffStrategy,
    /// Optional shared retry budget.
    pub budget: Option<RetryBudget>,
}

impl RetryPolicy {
    /// Create a new policy with sensible defaults.
    pub fn new() -> Self {
        Self {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            strategy: BackoffStrategy::Exponential,
            budget: None,
        }
    }

    /// Start building a policy.
    pub fn builder() -> RetryPolicyBuilder {
        RetryPolicyBuilder::new()
    }

    /// Compute the delay for a given attempt number (1-based).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        DelayCalculator::compute(
            &self.strategy,
            attempt,
            self.initial_delay,
            self.max_delay,
            self.backoff_multiplier,
        )
    }

    /// Determine whether another retry should be attempted.
    ///
    /// Checks max retries and optional budget. Does NOT check circuit breaker
    /// (that is done externally since it requires `&mut`).
    pub fn should_retry(&self, ctx: &RetryContext, condition: &dyn RetryCondition) -> bool {
        if ctx.attempt >= self.max_retries {
            return false;
        }
        if let Some(ref budget) = self.budget {
            if budget.remaining() == 0 {
                return false;
            }
        }
        if let Some(ref err) = ctx.last_error {
            condition.should_retry(err)
        } else {
            false
        }
    }

    /// Execute a closure with this retry policy, an optional circuit breaker,
    /// and a retry condition. Returns a [`RetryOutcome`].
    ///
    /// The closure receives the current attempt number (1-based) and returns
    /// `Result<T, String>` where the error is a message string.
    pub fn execute<T, F>(
        &self,
        mut f: F,
        condition: &dyn RetryCondition,
        mut circuit_breaker: Option<&mut CircuitBreaker>,
    ) -> RetryOutcome<T>
    where
        F: FnMut(u32) -> Result<T, String>,
    {
        let mut ctx = RetryContext::new();

        // Check circuit breaker before starting.
        if let Some(cb) = circuit_breaker.as_deref_mut() {
            if !cb.is_allowed() {
                return RetryOutcome::CircuitOpen;
            }
        }

        // We need to reborrow the circuit breaker mutably for the loop,
        // but we already consumed it above. Use an Option trick.
        // Actually, let's restructure: take the Option and work with it.
        let cb = circuit_breaker;

        self.execute_inner(&mut ctx, &mut f, condition, cb)
    }

    fn execute_inner<T, F>(
        &self,
        ctx: &mut RetryContext,
        f: &mut F,
        condition: &dyn RetryCondition,
        mut cb: Option<&mut CircuitBreaker>,
    ) -> RetryOutcome<T>
    where
        F: FnMut(u32) -> Result<T, String>,
    {
        loop {
            let attempt = ctx.attempt + 1;
            match f(attempt) {
                Ok(value) => {
                    ctx.record_success();
                    if let Some(ref mut breaker) = cb {
                        breaker.record_success();
                    }
                    return RetryOutcome::Success(value);
                }
                Err(error) => {
                    ctx.record_error(error);
                    if let Some(ref mut breaker) = cb {
                        breaker.record_failure();
                        if !breaker.is_allowed() {
                            return RetryOutcome::CircuitOpen;
                        }
                    }

                    if ctx.attempt > self.max_retries {
                        return RetryOutcome::Exhausted {
                            attempts: ctx.attempt,
                            last_error: ctx.last_error.clone().unwrap_or_default(),
                        };
                    }

                    if !condition.should_retry(ctx.last_error.as_deref().unwrap_or("")) {
                        return RetryOutcome::Exhausted {
                            attempts: ctx.attempt,
                            last_error: ctx.last_error.clone().unwrap_or_default(),
                        };
                    }

                    if let Some(ref budget) = self.budget {
                        if !budget.try_acquire() {
                            return RetryOutcome::BudgetExhausted;
                        }
                    }

                    // In a real implementation we would sleep here for
                    // self.delay_for_attempt(ctx.attempt), but this is a
                    // synchronous execute that does not block.
                }
            }
        }
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for RetryPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RetryPolicy")
            .field("max_retries", &self.max_retries)
            .field("initial_delay", &self.initial_delay)
            .field("max_delay", &self.max_delay)
            .field("backoff_multiplier", &self.backoff_multiplier)
            .field("strategy", &self.strategy)
            .field("has_budget", &self.budget.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// RetryPolicyBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for [`RetryPolicy`].
#[derive(Debug, Clone)]
pub struct RetryPolicyBuilder {
    policy: RetryPolicy,
}

impl RetryPolicyBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            policy: RetryPolicy::new(),
        }
    }

    /// Set the maximum number of retries.
    pub fn max_retries(mut self, n: u32) -> Self {
        self.policy.max_retries = n;
        self
    }

    /// Set the initial delay between retries.
    pub fn initial_delay(mut self, d: Duration) -> Self {
        self.policy.initial_delay = d;
        self
    }

    /// Set the maximum delay cap.
    pub fn max_delay(mut self, d: Duration) -> Self {
        self.policy.max_delay = d;
        self
    }

    /// Set the backoff multiplier for exponential strategies.
    pub fn backoff_multiplier(mut self, m: f64) -> Self {
        self.policy.backoff_multiplier = m;
        self
    }

    /// Set the backoff strategy.
    pub fn strategy(mut self, s: BackoffStrategy) -> Self {
        self.policy.strategy = s;
        self
    }

    /// Use fixed backoff.
    pub fn fixed_backoff(self) -> Self {
        self.strategy(BackoffStrategy::Fixed)
    }

    /// Use linear backoff.
    pub fn linear_backoff(self) -> Self {
        self.strategy(BackoffStrategy::Linear)
    }

    /// Use exponential backoff.
    pub fn exponential_backoff(self) -> Self {
        self.strategy(BackoffStrategy::Exponential)
    }

    /// Use exponential backoff with jitter.
    pub fn exponential_with_jitter(self) -> Self {
        self.strategy(BackoffStrategy::ExponentialWithJitter)
    }

    /// Use a custom delay sequence (milliseconds).
    pub fn custom_delays(self, delays: Vec<u64>) -> Self {
        self.strategy(BackoffStrategy::Custom(delays))
    }

    /// Attach a shared retry budget.
    pub fn budget(mut self, budget: RetryBudget) -> Self {
        self.policy.budget = Some(budget);
        self
    }

    /// Set initial delay in milliseconds (convenience).
    pub fn initial_delay_ms(self, ms: u64) -> Self {
        self.initial_delay(Duration::from_millis(ms))
    }

    /// Set max delay in seconds (convenience).
    pub fn max_delay_secs(self, secs: u64) -> Self {
        self.max_delay(Duration::from_secs(secs))
    }

    /// Build the [`RetryPolicy`].
    pub fn build(self) -> RetryPolicy {
        self.policy
    }
}

impl Default for RetryPolicyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- BackoffStrategy --

    #[test]
    fn test_backoff_strategy_default_is_exponential() {
        assert_eq!(BackoffStrategy::default(), BackoffStrategy::Exponential);
    }

    #[test]
    fn test_backoff_strategy_serialize_deserialize() {
        let strategies = vec![
            BackoffStrategy::Fixed,
            BackoffStrategy::Linear,
            BackoffStrategy::Exponential,
            BackoffStrategy::ExponentialWithJitter,
            BackoffStrategy::Custom(vec![100, 200, 400]),
        ];
        for s in strategies {
            let json = serde_json::to_string(&s).unwrap();
            let de: BackoffStrategy = serde_json::from_str(&json).unwrap();
            assert_eq!(s, de);
        }
    }

    // -- DelayCalculator --

    #[test]
    fn test_fixed_delay() {
        let d = DelayCalculator::compute(
            &BackoffStrategy::Fixed,
            1,
            Duration::from_millis(100),
            Duration::from_secs(30),
            2.0,
        );
        assert_eq!(d, Duration::from_millis(100));

        let d3 = DelayCalculator::compute(
            &BackoffStrategy::Fixed,
            3,
            Duration::from_millis(100),
            Duration::from_secs(30),
            2.0,
        );
        assert_eq!(d3, Duration::from_millis(100));
    }

    #[test]
    fn test_linear_delay() {
        let d1 = DelayCalculator::compute(
            &BackoffStrategy::Linear,
            1,
            Duration::from_millis(100),
            Duration::from_secs(30),
            2.0,
        );
        assert_eq!(d1, Duration::from_millis(100));

        let d3 = DelayCalculator::compute(
            &BackoffStrategy::Linear,
            3,
            Duration::from_millis(100),
            Duration::from_secs(30),
            2.0,
        );
        assert_eq!(d3, Duration::from_millis(300));
    }

    #[test]
    fn test_exponential_delay() {
        let init = Duration::from_millis(100);
        let max = Duration::from_secs(30);
        let mult = 2.0;

        let d1 = DelayCalculator::compute(&BackoffStrategy::Exponential, 1, init, max, mult);
        assert_eq!(d1, Duration::from_millis(100));

        let d2 = DelayCalculator::compute(&BackoffStrategy::Exponential, 2, init, max, mult);
        assert_eq!(d2, Duration::from_millis(200));

        let d3 = DelayCalculator::compute(&BackoffStrategy::Exponential, 3, init, max, mult);
        assert_eq!(d3, Duration::from_millis(400));

        let d4 = DelayCalculator::compute(&BackoffStrategy::Exponential, 4, init, max, mult);
        assert_eq!(d4, Duration::from_millis(800));
    }

    #[test]
    fn test_exponential_delay_respects_max() {
        let d = DelayCalculator::compute(
            &BackoffStrategy::Exponential,
            20,
            Duration::from_millis(100),
            Duration::from_secs(5),
            2.0,
        );
        assert_eq!(d, Duration::from_secs(5));
    }

    #[test]
    fn test_exponential_with_jitter_is_bounded() {
        let init = Duration::from_millis(100);
        let max = Duration::from_secs(30);
        for attempt in 1..=10 {
            let d = DelayCalculator::compute(
                &BackoffStrategy::ExponentialWithJitter,
                attempt,
                init,
                max,
                2.0,
            );
            assert!(d <= max);
        }
    }

    #[test]
    fn test_custom_delays() {
        let custom = BackoffStrategy::Custom(vec![50, 150, 500]);
        let d1 = DelayCalculator::compute(&custom, 1, Duration::ZERO, Duration::from_secs(30), 1.0);
        assert_eq!(d1, Duration::from_millis(50));

        let d2 = DelayCalculator::compute(&custom, 2, Duration::ZERO, Duration::from_secs(30), 1.0);
        assert_eq!(d2, Duration::from_millis(150));

        let d3 = DelayCalculator::compute(&custom, 3, Duration::ZERO, Duration::from_secs(30), 1.0);
        assert_eq!(d3, Duration::from_millis(500));

        // Beyond the list: stays at last value.
        let d4 = DelayCalculator::compute(&custom, 4, Duration::ZERO, Duration::from_secs(30), 1.0);
        assert_eq!(d4, Duration::from_millis(500));
    }

    #[test]
    fn test_custom_delays_empty_falls_back() {
        let custom = BackoffStrategy::Custom(vec![]);
        let d = DelayCalculator::compute(
            &custom,
            1,
            Duration::from_millis(100),
            Duration::from_secs(30),
            1.0,
        );
        assert_eq!(d, Duration::from_millis(100));
    }

    #[test]
    fn test_linear_delay_respects_max() {
        let d = DelayCalculator::compute(
            &BackoffStrategy::Linear,
            1000,
            Duration::from_millis(100),
            Duration::from_secs(1),
            1.0,
        );
        assert_eq!(d, Duration::from_secs(1));
    }

    // -- RetryCondition --

    #[test]
    fn test_always_retry() {
        let c = AlwaysRetry;
        assert!(c.should_retry("any error"));
        assert!(c.should_retry(""));
    }

    #[test]
    fn test_never_retry() {
        let c = NeverRetry;
        assert!(!c.should_retry("any error"));
        assert!(!c.should_retry("timeout"));
    }

    #[test]
    fn test_contains_retry_matching() {
        let c = ContainsRetry::new(vec!["timeout".into(), "rate limit".into()]);
        assert!(c.should_retry("request timeout"));
        assert!(c.should_retry("Rate Limit exceeded"));
        assert!(!c.should_retry("not found"));
    }

    #[test]
    fn test_contains_retry_case_insensitive() {
        let c = ContainsRetry::new(vec!["TIMEOUT".into()]);
        assert!(c.should_retry("Timeout error"));
        assert!(c.should_retry("timeout"));
        assert!(c.should_retry("TIMEOUT"));
    }

    #[test]
    fn test_contains_retry_empty_patterns() {
        let c = ContainsRetry::new(vec![]);
        assert!(!c.should_retry("anything"));
    }

    // -- RetryContext --

    #[test]
    fn test_retry_context_new() {
        let ctx = RetryContext::new();
        assert_eq!(ctx.attempt, 0);
        assert!(ctx.last_error.is_none());
        assert!(ctx.all_errors.is_empty());
    }

    #[test]
    fn test_retry_context_record_error() {
        let mut ctx = RetryContext::new();
        ctx.record_error("error 1".into());
        assert_eq!(ctx.attempt, 1);
        assert_eq!(ctx.last_error.as_deref(), Some("error 1"));
        assert_eq!(ctx.all_errors.len(), 1);

        ctx.record_error("error 2".into());
        assert_eq!(ctx.attempt, 2);
        assert_eq!(ctx.last_error.as_deref(), Some("error 2"));
        assert_eq!(ctx.all_errors.len(), 2);
        assert_eq!(ctx.all_errors[0], "error 1");
        assert_eq!(ctx.all_errors[1], "error 2");
    }

    #[test]
    fn test_retry_context_record_success() {
        let mut ctx = RetryContext::new();
        ctx.record_success();
        assert_eq!(ctx.attempt, 1);
        assert!(ctx.last_error.is_none());
    }

    #[test]
    fn test_retry_context_elapsed() {
        let ctx = RetryContext::new();
        // Elapsed should be non-negative (essentially instant).
        assert!(ctx.elapsed() < Duration::from_secs(1));
    }

    #[test]
    fn test_retry_context_default() {
        let ctx = RetryContext::default();
        assert_eq!(ctx.attempt, 0);
    }

    // -- RetryBudget --

    #[test]
    fn test_budget_basic() {
        let budget = RetryBudget::new(3);
        assert_eq!(budget.remaining(), 3);
        assert_eq!(budget.total(), 3);
    }

    #[test]
    fn test_budget_acquire() {
        let budget = RetryBudget::new(2);
        assert!(budget.try_acquire());
        assert_eq!(budget.remaining(), 1);
        assert!(budget.try_acquire());
        assert_eq!(budget.remaining(), 0);
        assert!(!budget.try_acquire());
    }

    #[test]
    fn test_budget_reset() {
        let budget = RetryBudget::new(3);
        budget.try_acquire();
        budget.try_acquire();
        assert_eq!(budget.remaining(), 1);
        budget.reset();
        assert_eq!(budget.remaining(), 3);
    }

    #[test]
    fn test_budget_shared() {
        let budget = RetryBudget::new(2);
        let budget_clone = budget.clone();
        assert!(budget.try_acquire());
        assert_eq!(budget_clone.remaining(), 1);
        assert!(budget_clone.try_acquire());
        assert!(!budget.try_acquire());
    }

    #[test]
    fn test_budget_zero() {
        let budget = RetryBudget::new(0);
        assert!(!budget.try_acquire());
    }

    // -- CircuitState --

    #[test]
    fn test_circuit_state_display() {
        assert_eq!(CircuitState::Closed.to_string(), "Closed");
        assert_eq!(CircuitState::Open.to_string(), "Open");
        assert_eq!(CircuitState::HalfOpen.to_string(), "HalfOpen");
    }

    #[test]
    fn test_circuit_state_serialize() {
        let json = serde_json::to_string(&CircuitState::Open).unwrap();
        let de: CircuitState = serde_json::from_str(&json).unwrap();
        assert_eq!(de, CircuitState::Open);
    }

    // -- CircuitBreaker --

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(5));
        assert_eq!(cb.state(), CircuitState::Closed);
        assert!(cb.is_allowed());
    }

    #[test]
    fn test_circuit_breaker_trips_after_threshold() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Closed);
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        assert!(!cb.is_allowed());
    }

    #[test]
    fn test_circuit_breaker_resets_on_success() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert_eq!(cb.consecutive_failures(), 0);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_manual_reset() {
        let mut cb = CircuitBreaker::new(2, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
        assert_eq!(cb.consecutive_failures(), 0);
    }

    #[test]
    fn test_circuit_breaker_half_open_after_cooldown() {
        let mut cb = CircuitBreaker::new(1, Duration::from_millis(0));
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::HalfOpen);
        assert!(cb.is_allowed());
    }

    #[test]
    fn test_circuit_breaker_consecutive_failures() {
        let mut cb = CircuitBreaker::new(5, Duration::from_secs(60));
        cb.record_failure();
        assert_eq!(cb.consecutive_failures(), 1);
        cb.record_failure();
        assert_eq!(cb.consecutive_failures(), 2);
        cb.record_success();
        assert_eq!(cb.consecutive_failures(), 0);
    }

    // -- RetryPolicy --

    #[test]
    fn test_policy_default() {
        let policy = RetryPolicy::new();
        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.initial_delay, Duration::from_millis(100));
        assert_eq!(policy.max_delay, Duration::from_secs(30));
        assert_eq!(policy.backoff_multiplier, 2.0);
        assert_eq!(policy.strategy, BackoffStrategy::Exponential);
        assert!(policy.budget.is_none());
    }

    #[test]
    fn test_policy_delay_for_attempt() {
        let policy = RetryPolicy::new();
        assert_eq!(policy.delay_for_attempt(1), Duration::from_millis(100));
        assert_eq!(policy.delay_for_attempt(2), Duration::from_millis(200));
        assert_eq!(policy.delay_for_attempt(3), Duration::from_millis(400));
    }

    #[test]
    fn test_policy_should_retry() {
        let policy = RetryPolicy::new();
        let cond = AlwaysRetry;
        let mut ctx = RetryContext::new();
        ctx.record_error("error".into());
        assert!(policy.should_retry(&ctx, &cond));

        // Exhaust attempts
        ctx.record_error("error".into());
        ctx.record_error("error".into());
        assert!(!policy.should_retry(&ctx, &cond));
    }

    #[test]
    fn test_policy_should_retry_respects_condition() {
        let policy = RetryPolicy::new();
        let cond = NeverRetry;
        let mut ctx = RetryContext::new();
        ctx.record_error("error".into());
        assert!(!policy.should_retry(&ctx, &cond));
    }

    #[test]
    fn test_policy_should_retry_with_budget() {
        let budget = RetryBudget::new(1);
        budget.try_acquire(); // exhaust
        let policy = RetryPolicy::builder().budget(budget).build();
        let cond = AlwaysRetry;
        let mut ctx = RetryContext::new();
        ctx.record_error("error".into());
        assert!(!policy.should_retry(&ctx, &cond));
    }

    #[test]
    fn test_policy_debug() {
        let policy = RetryPolicy::new();
        let debug = format!("{:?}", policy);
        assert!(debug.contains("RetryPolicy"));
        assert!(debug.contains("max_retries"));
    }

    // -- RetryPolicyBuilder --

    #[test]
    fn test_builder_defaults() {
        let policy = RetryPolicyBuilder::new().build();
        assert_eq!(policy.max_retries, 3);
        assert_eq!(policy.strategy, BackoffStrategy::Exponential);
    }

    #[test]
    fn test_builder_max_retries() {
        let policy = RetryPolicy::builder().max_retries(5).build();
        assert_eq!(policy.max_retries, 5);
    }

    #[test]
    fn test_builder_initial_delay() {
        let policy = RetryPolicy::builder()
            .initial_delay(Duration::from_secs(1))
            .build();
        assert_eq!(policy.initial_delay, Duration::from_secs(1));
    }

    #[test]
    fn test_builder_initial_delay_ms() {
        let policy = RetryPolicy::builder().initial_delay_ms(250).build();
        assert_eq!(policy.initial_delay, Duration::from_millis(250));
    }

    #[test]
    fn test_builder_max_delay() {
        let policy = RetryPolicy::builder()
            .max_delay(Duration::from_secs(60))
            .build();
        assert_eq!(policy.max_delay, Duration::from_secs(60));
    }

    #[test]
    fn test_builder_max_delay_secs() {
        let policy = RetryPolicy::builder().max_delay_secs(10).build();
        assert_eq!(policy.max_delay, Duration::from_secs(10));
    }

    #[test]
    fn test_builder_backoff_multiplier() {
        let policy = RetryPolicy::builder().backoff_multiplier(3.0).build();
        assert_eq!(policy.backoff_multiplier, 3.0);
    }

    #[test]
    fn test_builder_fixed_backoff() {
        let policy = RetryPolicy::builder().fixed_backoff().build();
        assert_eq!(policy.strategy, BackoffStrategy::Fixed);
    }

    #[test]
    fn test_builder_linear_backoff() {
        let policy = RetryPolicy::builder().linear_backoff().build();
        assert_eq!(policy.strategy, BackoffStrategy::Linear);
    }

    #[test]
    fn test_builder_exponential_backoff() {
        let policy = RetryPolicy::builder().exponential_backoff().build();
        assert_eq!(policy.strategy, BackoffStrategy::Exponential);
    }

    #[test]
    fn test_builder_exponential_with_jitter() {
        let policy = RetryPolicy::builder().exponential_with_jitter().build();
        assert_eq!(policy.strategy, BackoffStrategy::ExponentialWithJitter);
    }

    #[test]
    fn test_builder_custom_delays() {
        let policy = RetryPolicy::builder()
            .custom_delays(vec![10, 20, 30])
            .build();
        assert_eq!(policy.strategy, BackoffStrategy::Custom(vec![10, 20, 30]));
    }

    #[test]
    fn test_builder_with_budget() {
        let budget = RetryBudget::new(10);
        let policy = RetryPolicy::builder().budget(budget).build();
        assert!(policy.budget.is_some());
        assert_eq!(policy.budget.as_ref().unwrap().total(), 10);
    }

    #[test]
    fn test_builder_chaining() {
        let policy = RetryPolicy::builder()
            .max_retries(5)
            .initial_delay_ms(200)
            .max_delay_secs(60)
            .backoff_multiplier(3.0)
            .exponential_backoff()
            .build();
        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.initial_delay, Duration::from_millis(200));
        assert_eq!(policy.max_delay, Duration::from_secs(60));
        assert_eq!(policy.backoff_multiplier, 3.0);
        assert_eq!(policy.strategy, BackoffStrategy::Exponential);
    }

    #[test]
    fn test_builder_default_trait() {
        let builder = RetryPolicyBuilder::default();
        let policy = builder.build();
        assert_eq!(policy.max_retries, 3);
    }

    // -- RetryOutcome --

    #[test]
    fn test_outcome_is_success() {
        let s: RetryOutcome<i32> = RetryOutcome::Success(42);
        assert!(s.is_success());

        let e: RetryOutcome<i32> = RetryOutcome::Exhausted {
            attempts: 3,
            last_error: "err".into(),
        };
        assert!(!e.is_success());

        let co: RetryOutcome<i32> = RetryOutcome::CircuitOpen;
        assert!(!co.is_success());

        let be: RetryOutcome<i32> = RetryOutcome::BudgetExhausted;
        assert!(!be.is_success());
    }

    #[test]
    fn test_outcome_eq() {
        let a: RetryOutcome<i32> = RetryOutcome::Success(1);
        let b: RetryOutcome<i32> = RetryOutcome::Success(1);
        assert_eq!(a, b);

        let c: RetryOutcome<i32> = RetryOutcome::CircuitOpen;
        let d: RetryOutcome<i32> = RetryOutcome::CircuitOpen;
        assert_eq!(c, d);
    }

    // -- execute --

    #[test]
    fn test_execute_success_first_try() {
        let policy = RetryPolicy::new();
        let result = policy.execute(|_| Ok::<_, String>(42), &AlwaysRetry, None);
        assert_eq!(result, RetryOutcome::Success(42));
    }

    #[test]
    fn test_execute_success_after_retries() {
        let policy = RetryPolicy::builder().max_retries(5).build();
        let mut call_count = 0;
        let result = policy.execute(
            |_attempt| {
                call_count += 1;
                if call_count < 3 {
                    Err("transient".into())
                } else {
                    Ok(99)
                }
            },
            &AlwaysRetry,
            None,
        );
        assert_eq!(result, RetryOutcome::Success(99));
        assert_eq!(call_count, 3);
    }

    #[test]
    fn test_execute_exhausted() {
        let policy = RetryPolicy::builder().max_retries(2).build();
        let result: RetryOutcome<i32> = policy.execute(|_| Err("fail".into()), &AlwaysRetry, None);
        match result {
            RetryOutcome::Exhausted {
                attempts,
                last_error,
            } => {
                assert_eq!(attempts, 3); // 1 initial + 2 retries
                assert_eq!(last_error, "fail");
            }
            other => panic!("Expected Exhausted, got {:?}", other),
        }
    }

    #[test]
    fn test_execute_condition_stops_retry() {
        let policy = RetryPolicy::builder().max_retries(5).build();
        let result: RetryOutcome<i32> =
            policy.execute(|_| Err("fatal error".into()), &NeverRetry, None);
        match result {
            RetryOutcome::Exhausted { attempts, .. } => {
                assert_eq!(attempts, 1);
            }
            other => panic!("Expected Exhausted, got {:?}", other),
        }
    }

    #[test]
    fn test_execute_circuit_open() {
        let policy = RetryPolicy::new();
        let mut cb = CircuitBreaker::new(1, Duration::from_secs(60));
        cb.record_failure(); // trip the breaker
        let result: RetryOutcome<i32> = policy.execute(|_| Ok(1), &AlwaysRetry, Some(&mut cb));
        assert_eq!(result, RetryOutcome::CircuitOpen);
    }

    #[test]
    fn test_execute_circuit_trips_during_retries() {
        let policy = RetryPolicy::builder().max_retries(10).build();
        let mut cb = CircuitBreaker::new(2, Duration::from_secs(60));
        let result: RetryOutcome<i32> =
            policy.execute(|_| Err("fail".into()), &AlwaysRetry, Some(&mut cb));
        assert_eq!(result, RetryOutcome::CircuitOpen);
    }

    #[test]
    fn test_execute_budget_exhausted() {
        let budget = RetryBudget::new(1);
        let policy = RetryPolicy::builder().max_retries(5).budget(budget).build();
        let mut call_count = 0;
        let result: RetryOutcome<i32> = policy.execute(
            |_| {
                call_count += 1;
                Err("fail".into())
            },
            &AlwaysRetry,
            None,
        );
        assert_eq!(result, RetryOutcome::BudgetExhausted);
        // First call fails, budget allows 1 retry, second call fails, budget exhausted.
        assert_eq!(call_count, 2);
    }

    #[test]
    fn test_execute_zero_retries() {
        let policy = RetryPolicy::builder().max_retries(0).build();
        let result: RetryOutcome<i32> = policy.execute(|_| Err("fail".into()), &AlwaysRetry, None);
        match result {
            RetryOutcome::Exhausted { attempts, .. } => assert_eq!(attempts, 1),
            other => panic!("Expected Exhausted, got {:?}", other),
        }
    }

    #[test]
    fn test_execute_with_contains_condition() {
        let policy = RetryPolicy::builder().max_retries(5).build();
        let cond = ContainsRetry::new(vec!["timeout".into()]);
        let mut call_count = 0;
        let result: RetryOutcome<i32> = policy.execute(
            |attempt| {
                call_count += 1;
                if attempt == 1 {
                    Err("timeout error".into())
                } else {
                    Err("permanent error".into())
                }
            },
            &cond,
            None,
        );
        // First call: "timeout error" -> retryable, retry.
        // Second call: "permanent error" -> not retryable, stop.
        assert_eq!(call_count, 2);
        match result {
            RetryOutcome::Exhausted {
                attempts,
                last_error,
            } => {
                assert_eq!(attempts, 2);
                assert_eq!(last_error, "permanent error");
            }
            other => panic!("Expected Exhausted, got {:?}", other),
        }
    }

    #[test]
    fn test_pseudo_random_fraction_bounded() {
        for n in 0..100 {
            let f = DelayCalculator::pseudo_random_fraction(n);
            assert!(
                f >= 0.0 && f < 1.0,
                "fraction {} out of range for n={}",
                f,
                n
            );
        }
    }

    #[test]
    fn test_policy_clone() {
        let policy = RetryPolicy::builder()
            .max_retries(5)
            .fixed_backoff()
            .build();
        let cloned = policy.clone();
        assert_eq!(cloned.max_retries, 5);
        assert_eq!(cloned.strategy, BackoffStrategy::Fixed);
    }
}
