//! Error recovery system for agents.
//!
//! Provides error classification, backoff strategies, recovery policies, and
//! a [`RecoveryManager`] that orchestrates automatic recovery from common
//! failures such as rate limits, timeouts, and model overload.
//!
//! # Example
//!
//! ```rust
//! use cognisagent::recovery::{RecoveryManager, RecoveryPolicy, ErrorClassifier};
//!
//! let policy = RecoveryPolicy::with_default_retry(3);
//! let mut manager = RecoveryManager::new(policy);
//! let action = manager.handle_error("rate limit exceeded");
//! assert!(action.should_retry());
//! ```

use regex::Regex;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt;
use std::time::Duration;

// ---------------------------------------------------------------------------
// ErrorCategory
// ---------------------------------------------------------------------------

/// Classifies an error into a high-level category for recovery decisions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// The request was throttled by rate limiting.
    RateLimit,
    /// Authentication or authorization failure.
    AuthFailure,
    /// The model backend is overloaded or unavailable.
    ModelOverloaded,
    /// A tool returned an error during execution.
    ToolError,
    /// Failed to parse a model response or tool output.
    ParseError,
    /// Network-level failure (DNS, connection reset, etc.).
    NetworkError,
    /// The operation timed out.
    Timeout,
    /// A resource limit was exhausted (memory, disk, tokens).
    ResourceExhausted,
    /// An error that does not match any known category.
    Unknown(String),
}

impl ErrorCategory {
    /// Returns `true` if errors in this category are generally safe to retry.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            ErrorCategory::RateLimit
                | ErrorCategory::ModelOverloaded
                | ErrorCategory::NetworkError
                | ErrorCategory::Timeout
        )
    }

    /// Returns a suggested delay before retrying for this category.
    pub fn suggested_delay(&self) -> Duration {
        match self {
            ErrorCategory::RateLimit => Duration::from_secs(30),
            ErrorCategory::ModelOverloaded => Duration::from_secs(10),
            ErrorCategory::NetworkError => Duration::from_secs(2),
            ErrorCategory::Timeout => Duration::from_secs(5),
            ErrorCategory::AuthFailure => Duration::from_secs(0),
            ErrorCategory::ToolError => Duration::from_millis(500),
            ErrorCategory::ParseError => Duration::from_millis(100),
            ErrorCategory::ResourceExhausted => Duration::from_secs(60),
            ErrorCategory::Unknown(_) => Duration::from_secs(1),
        }
    }
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorCategory::RateLimit => write!(f, "RateLimit"),
            ErrorCategory::AuthFailure => write!(f, "AuthFailure"),
            ErrorCategory::ModelOverloaded => write!(f, "ModelOverloaded"),
            ErrorCategory::ToolError => write!(f, "ToolError"),
            ErrorCategory::ParseError => write!(f, "ParseError"),
            ErrorCategory::NetworkError => write!(f, "NetworkError"),
            ErrorCategory::Timeout => write!(f, "Timeout"),
            ErrorCategory::ResourceExhausted => write!(f, "ResourceExhausted"),
            ErrorCategory::Unknown(msg) => write!(f, "Unknown({msg})"),
        }
    }
}

// ---------------------------------------------------------------------------
// ErrorClassifier
// ---------------------------------------------------------------------------

/// Classifies raw error strings into [`ErrorCategory`] values by pattern matching.
pub struct ErrorClassifier {
    /// Custom patterns added via [`add_pattern`](ErrorClassifier::add_pattern).
    custom_patterns: Vec<(Regex, ErrorCategory)>,
}

impl ErrorClassifier {
    /// Create a new classifier with built-in patterns.
    pub fn new() -> Self {
        Self {
            custom_patterns: Vec::new(),
        }
    }

    /// Classify an error message into an [`ErrorCategory`].
    ///
    /// Custom patterns are checked first, then built-in patterns.
    pub fn classify(&self, error: &str) -> ErrorCategory {
        let lower = error.to_lowercase();

        // Check custom patterns first.
        for (re, cat) in &self.custom_patterns {
            if re.is_match(&lower) {
                return cat.clone();
            }
        }

        // Built-in patterns.
        if lower.contains("rate limit")
            || lower.contains("rate_limit")
            || lower.contains("too many requests")
            || lower.contains("429")
            || lower.contains("throttl")
        {
            return ErrorCategory::RateLimit;
        }

        if lower.contains("auth")
            || lower.contains("unauthorized")
            || lower.contains("forbidden")
            || lower.contains("401")
            || lower.contains("403")
            || lower.contains("invalid api key")
            || lower.contains("invalid_api_key")
            || lower.contains("permission denied")
        {
            return ErrorCategory::AuthFailure;
        }

        if lower.contains("overloaded")
            || lower.contains("capacity")
            || lower.contains("503")
            || lower.contains("service unavailable")
            || lower.contains("server busy")
        {
            return ErrorCategory::ModelOverloaded;
        }

        if lower.contains("timeout")
            || lower.contains("timed out")
            || lower.contains("deadline exceeded")
        {
            return ErrorCategory::Timeout;
        }

        if lower.contains("network")
            || lower.contains("connection refused")
            || lower.contains("connection reset")
            || lower.contains("dns")
            || lower.contains("socket")
            || lower.contains("econnrefused")
        {
            return ErrorCategory::NetworkError;
        }

        if lower.contains("tool error")
            || lower.contains("tool execution failed")
            || lower.contains("tool not found")
        {
            return ErrorCategory::ToolError;
        }

        if lower.contains("parse")
            || lower.contains("invalid json")
            || lower.contains("deserialization")
            || lower.contains("expected")
        {
            return ErrorCategory::ParseError;
        }

        if lower.contains("resource exhausted")
            || lower.contains("out of memory")
            || lower.contains("disk full")
            || lower.contains("token limit")
            || lower.contains("context length")
        {
            return ErrorCategory::ResourceExhausted;
        }

        ErrorCategory::Unknown(error.to_string())
    }

    /// Register a custom regex pattern that maps to a category.
    ///
    /// Custom patterns take priority over built-in heuristics.
    pub fn add_pattern(&mut self, pattern: &str, category: ErrorCategory) {
        if let Ok(re) = Regex::new(&pattern.to_lowercase()) {
            self.custom_patterns.push((re, category));
        }
    }
}

impl Default for ErrorClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// BackoffStrategy
// ---------------------------------------------------------------------------

/// Determines how long to wait between retry attempts.
#[derive(Debug, Clone, PartialEq)]
pub enum BackoffStrategy {
    /// Wait a fixed duration every attempt.
    Fixed(Duration),
    /// Increase linearly: `initial + increment * attempt`.
    Linear {
        initial: Duration,
        increment: Duration,
    },
    /// Increase exponentially: `initial * multiplier^attempt`, capped at `max`.
    Exponential {
        initial: Duration,
        multiplier: f64,
        max: Duration,
    },
}

impl BackoffStrategy {
    /// Compute the delay for the given attempt number (0-indexed).
    pub fn delay_for_attempt(&self, attempt: u32) -> Duration {
        match self {
            BackoffStrategy::Fixed(d) => *d,
            BackoffStrategy::Linear { initial, increment } => *initial + *increment * attempt,
            BackoffStrategy::Exponential {
                initial,
                multiplier,
                max,
            } => {
                let factor = multiplier.powi(attempt as i32);
                let nanos = (initial.as_secs_f64() * factor * 1_000_000_000.0) as u128;
                let computed = Duration::from_nanos(nanos.min(u64::MAX as u128) as u64);
                if computed > *max {
                    *max
                } else {
                    computed
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RecoveryStrategy
// ---------------------------------------------------------------------------

/// Describes what action to take in response to an error.
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStrategy {
    /// Retry the failed operation with the given backoff.
    Retry {
        max_attempts: u32,
        backoff: BackoffStrategy,
    },
    /// Fall back to an alternative (e.g., a different model or tool).
    Fallback(String),
    /// Skip the failed step and continue.
    Skip,
    /// Abort the entire operation.
    Abort,
    /// A custom recovery action identified by name.
    Custom(String),
}

// ---------------------------------------------------------------------------
// RecoveryPolicy
// ---------------------------------------------------------------------------

/// Maps [`ErrorCategory`] values to [`RecoveryStrategy`] instances.
pub struct RecoveryPolicy {
    strategies: HashMap<String, RecoveryStrategy>,
    default_strategy: RecoveryStrategy,
}

impl RecoveryPolicy {
    /// Create a policy with sensible defaults for every category.
    pub fn new() -> Self {
        let mut strategies = HashMap::new();

        strategies.insert(
            "RateLimit".to_string(),
            RecoveryStrategy::Retry {
                max_attempts: 5,
                backoff: BackoffStrategy::Exponential {
                    initial: Duration::from_secs(1),
                    multiplier: 2.0,
                    max: Duration::from_secs(60),
                },
            },
        );
        strategies.insert(
            "ModelOverloaded".to_string(),
            RecoveryStrategy::Retry {
                max_attempts: 3,
                backoff: BackoffStrategy::Exponential {
                    initial: Duration::from_secs(2),
                    multiplier: 2.0,
                    max: Duration::from_secs(30),
                },
            },
        );
        strategies.insert(
            "NetworkError".to_string(),
            RecoveryStrategy::Retry {
                max_attempts: 3,
                backoff: BackoffStrategy::Linear {
                    initial: Duration::from_secs(1),
                    increment: Duration::from_secs(1),
                },
            },
        );
        strategies.insert(
            "Timeout".to_string(),
            RecoveryStrategy::Retry {
                max_attempts: 2,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(5)),
            },
        );
        strategies.insert("AuthFailure".to_string(), RecoveryStrategy::Abort);
        strategies.insert("ToolError".to_string(), RecoveryStrategy::Skip);
        strategies.insert("ParseError".to_string(), RecoveryStrategy::Skip);
        strategies.insert("ResourceExhausted".to_string(), RecoveryStrategy::Abort);

        Self {
            strategies,
            default_strategy: RecoveryStrategy::Abort,
        }
    }

    /// Set the strategy for a specific error category.
    pub fn set_strategy(&mut self, category: ErrorCategory, strategy: RecoveryStrategy) {
        let key = category_key(&category);
        self.strategies.insert(key, strategy);
    }

    /// Look up the strategy for the given error category.
    pub fn get_strategy(&self, category: &ErrorCategory) -> &RecoveryStrategy {
        let key = category_key(category);
        self.strategies.get(&key).unwrap_or(&self.default_strategy)
    }

    /// Build a policy where every retryable category uses exponential backoff.
    pub fn with_default_retry(max: u32) -> Self {
        let mut policy = Self::new();
        let retryable = [
            ErrorCategory::RateLimit,
            ErrorCategory::ModelOverloaded,
            ErrorCategory::NetworkError,
            ErrorCategory::Timeout,
        ];
        for cat in retryable {
            policy.set_strategy(
                cat,
                RecoveryStrategy::Retry {
                    max_attempts: max,
                    backoff: BackoffStrategy::Exponential {
                        initial: Duration::from_secs(1),
                        multiplier: 2.0,
                        max: Duration::from_secs(60),
                    },
                },
            );
        }
        policy
    }
}

impl Default for RecoveryPolicy {
    fn default() -> Self {
        Self::new()
    }
}

/// Produce a string key for an ErrorCategory (ignoring Unknown payload for lookup).
fn category_key(cat: &ErrorCategory) -> String {
    match cat {
        ErrorCategory::Unknown(_) => "Unknown".to_string(),
        other => format!("{other}"),
    }
}

// ---------------------------------------------------------------------------
// RecoveryAttempt
// ---------------------------------------------------------------------------

/// Records the outcome of a single recovery attempt.
#[derive(Debug, Clone)]
pub struct RecoveryAttempt {
    /// The classified error category.
    pub category: ErrorCategory,
    /// The strategy that was applied.
    pub strategy: RecoveryStrategy,
    /// The attempt number (1-indexed).
    pub attempt: u32,
    /// Whether the recovery succeeded.
    pub succeeded: bool,
    /// How long the recovery attempt took.
    pub duration: Duration,
}

impl RecoveryAttempt {
    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "category": self.category.to_string(),
            "strategy": format!("{:?}", self.strategy),
            "attempt": self.attempt,
            "succeeded": self.succeeded,
            "duration_ms": self.duration.as_millis() as u64,
        })
    }
}

// ---------------------------------------------------------------------------
// RecoveryLog
// ---------------------------------------------------------------------------

/// Accumulates [`RecoveryAttempt`] records for analysis.
pub struct RecoveryLog {
    attempts: Vec<RecoveryAttempt>,
}

impl RecoveryLog {
    /// Create an empty log.
    pub fn new() -> Self {
        Self {
            attempts: Vec::new(),
        }
    }

    /// Record a recovery attempt.
    pub fn record(&mut self, attempt: RecoveryAttempt) {
        self.attempts.push(attempt);
    }

    /// All recorded attempts.
    pub fn attempts(&self) -> &[RecoveryAttempt] {
        &self.attempts
    }

    /// Fraction of attempts that succeeded (0.0 if none recorded).
    pub fn success_rate(&self) -> f64 {
        if self.attempts.is_empty() {
            return 0.0;
        }
        let successes = self.attempts.iter().filter(|a| a.succeeded).count();
        successes as f64 / self.attempts.len() as f64
    }

    /// Filter attempts by category.
    pub fn attempts_by_category(&self, cat: &ErrorCategory) -> Vec<&RecoveryAttempt> {
        self.attempts
            .iter()
            .filter(|a| category_key(&a.category) == category_key(cat))
            .collect()
    }

    /// Total recovery time across all attempts.
    pub fn total_recovery_time(&self) -> Duration {
        self.attempts.iter().map(|a| a.duration).sum()
    }

    /// Number of recorded attempts.
    pub fn len(&self) -> usize {
        self.attempts.len()
    }

    /// Returns `true` if no attempts have been recorded.
    pub fn is_empty(&self) -> bool {
        self.attempts.is_empty()
    }

    /// Remove all recorded attempts.
    pub fn clear(&mut self) {
        self.attempts.clear();
    }

    /// Serialize the entire log to JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "total_attempts": self.len(),
            "success_rate": self.success_rate(),
            "total_recovery_time_ms": self.total_recovery_time().as_millis() as u64,
            "attempts": self.attempts.iter().map(|a| a.to_json()).collect::<Vec<_>>(),
        })
    }
}

impl Default for RecoveryLog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// RecoveryAction
// ---------------------------------------------------------------------------

/// The concrete action to take after classifying an error.
#[derive(Debug, Clone)]
pub struct RecoveryAction {
    /// The recovery strategy to execute.
    pub action: RecoveryStrategy,
    /// How long to wait before executing the action.
    pub delay: Duration,
    /// The current attempt number (1-indexed).
    pub attempt: u32,
    /// The classified error category.
    pub category: ErrorCategory,
}

impl RecoveryAction {
    /// Returns `true` if the action is a retry and the attempt limit has not
    /// been reached.
    pub fn should_retry(&self) -> bool {
        match &self.action {
            RecoveryStrategy::Retry { max_attempts, .. } => self.attempt <= *max_attempts,
            _ => false,
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "action": format!("{:?}", self.action),
            "delay_ms": self.delay.as_millis() as u64,
            "attempt": self.attempt,
            "category": self.category.to_string(),
            "should_retry": self.should_retry(),
        })
    }
}

// ---------------------------------------------------------------------------
// RecoveryManager
// ---------------------------------------------------------------------------

/// Orchestrates error classification, strategy selection, and logging.
pub struct RecoveryManager {
    policy: RecoveryPolicy,
    classifier: ErrorClassifier,
    log: RecoveryLog,
    /// Tracks consecutive attempts per category key.
    attempt_counters: HashMap<String, u32>,
}

impl RecoveryManager {
    /// Create a manager with the given policy and a default classifier.
    pub fn new(policy: RecoveryPolicy) -> Self {
        Self {
            policy,
            classifier: ErrorClassifier::new(),
            log: RecoveryLog::new(),
            attempt_counters: HashMap::new(),
        }
    }

    /// Create a manager with a custom classifier.
    pub fn with_classifier(classifier: ErrorClassifier) -> Self {
        Self {
            policy: RecoveryPolicy::new(),
            classifier,
            log: RecoveryLog::new(),
            attempt_counters: HashMap::new(),
        }
    }

    /// Classify the error, select a strategy, record the attempt, and return
    /// a [`RecoveryAction`].
    pub fn handle_error(&mut self, error: &str) -> RecoveryAction {
        let category = self.classifier.classify(error);
        let strategy = self.policy.get_strategy(&category).clone();
        let key = category_key(&category);

        let counter = self.attempt_counters.entry(key).or_insert(0);
        *counter += 1;
        let attempt = *counter;

        let delay = match &strategy {
            RecoveryStrategy::Retry { backoff, .. } => {
                backoff.delay_for_attempt(attempt.saturating_sub(1))
            }
            _ => Duration::ZERO,
        };

        let should_succeed = match &strategy {
            RecoveryStrategy::Retry { max_attempts, .. } => attempt <= *max_attempts,
            _ => false,
        };

        self.log.record(RecoveryAttempt {
            category: category.clone(),
            strategy: strategy.clone(),
            attempt,
            succeeded: should_succeed,
            duration: delay,
        });

        RecoveryAction {
            action: strategy,
            delay,
            attempt,
            category,
        }
    }

    /// Access the recovery log.
    pub fn log(&self) -> &RecoveryLog {
        &self.log
    }

    /// Reset all attempt counters and clear the log.
    pub fn reset(&mut self) {
        self.attempt_counters.clear();
        self.log.clear();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // === ErrorCategory tests ===============================================

    #[test]
    fn test_rate_limit_is_retryable() {
        assert!(ErrorCategory::RateLimit.is_retryable());
    }

    #[test]
    fn test_model_overloaded_is_retryable() {
        assert!(ErrorCategory::ModelOverloaded.is_retryable());
    }

    #[test]
    fn test_network_error_is_retryable() {
        assert!(ErrorCategory::NetworkError.is_retryable());
    }

    #[test]
    fn test_timeout_is_retryable() {
        assert!(ErrorCategory::Timeout.is_retryable());
    }

    #[test]
    fn test_auth_failure_not_retryable() {
        assert!(!ErrorCategory::AuthFailure.is_retryable());
    }

    #[test]
    fn test_tool_error_not_retryable() {
        assert!(!ErrorCategory::ToolError.is_retryable());
    }

    #[test]
    fn test_parse_error_not_retryable() {
        assert!(!ErrorCategory::ParseError.is_retryable());
    }

    #[test]
    fn test_resource_exhausted_not_retryable() {
        assert!(!ErrorCategory::ResourceExhausted.is_retryable());
    }

    #[test]
    fn test_unknown_not_retryable() {
        assert!(!ErrorCategory::Unknown("something".into()).is_retryable());
    }

    #[test]
    fn test_suggested_delays_differ() {
        let rate = ErrorCategory::RateLimit.suggested_delay();
        let net = ErrorCategory::NetworkError.suggested_delay();
        assert!(rate > net);
    }

    #[test]
    fn test_display_impl() {
        assert_eq!(ErrorCategory::RateLimit.to_string(), "RateLimit");
        assert_eq!(
            ErrorCategory::Unknown("oops".into()).to_string(),
            "Unknown(oops)"
        );
    }

    // === ErrorClassifier tests =============================================

    #[test]
    fn test_classify_rate_limit() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("rate limit exceeded"), ErrorCategory::RateLimit);
        assert_eq!(c.classify("Too Many Requests"), ErrorCategory::RateLimit);
        assert_eq!(c.classify("HTTP 429 error"), ErrorCategory::RateLimit);
        assert_eq!(c.classify("request throttled"), ErrorCategory::RateLimit);
    }

    #[test]
    fn test_classify_auth() {
        let c = ErrorClassifier::new();
        assert_eq!(
            c.classify("unauthorized access"),
            ErrorCategory::AuthFailure
        );
        assert_eq!(c.classify("HTTP 401"), ErrorCategory::AuthFailure);
        assert_eq!(c.classify("Forbidden"), ErrorCategory::AuthFailure);
        assert_eq!(c.classify("invalid api key"), ErrorCategory::AuthFailure);
        assert_eq!(c.classify("permission denied"), ErrorCategory::AuthFailure);
    }

    #[test]
    fn test_classify_model_overloaded() {
        let c = ErrorClassifier::new();
        assert_eq!(
            c.classify("model is overloaded"),
            ErrorCategory::ModelOverloaded
        );
        assert_eq!(
            c.classify("503 service unavailable"),
            ErrorCategory::ModelOverloaded
        );
        assert_eq!(c.classify("server busy"), ErrorCategory::ModelOverloaded);
    }

    #[test]
    fn test_classify_timeout() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("request timeout"), ErrorCategory::Timeout);
        assert_eq!(c.classify("operation timed out"), ErrorCategory::Timeout);
        assert_eq!(c.classify("deadline exceeded"), ErrorCategory::Timeout);
    }

    #[test]
    fn test_classify_network() {
        let c = ErrorClassifier::new();
        assert_eq!(
            c.classify("network error occurred"),
            ErrorCategory::NetworkError
        );
        assert_eq!(
            c.classify("connection refused"),
            ErrorCategory::NetworkError
        );
        assert_eq!(
            c.classify("dns resolution failed"),
            ErrorCategory::NetworkError
        );
    }

    #[test]
    fn test_classify_tool_error() {
        let c = ErrorClassifier::new();
        assert_eq!(c.classify("tool error: failed"), ErrorCategory::ToolError);
        assert_eq!(
            c.classify("tool execution failed"),
            ErrorCategory::ToolError
        );
    }

    #[test]
    fn test_classify_parse_error() {
        let c = ErrorClassifier::new();
        assert_eq!(
            c.classify("parse error at line 5"),
            ErrorCategory::ParseError
        );
        assert_eq!(c.classify("invalid json input"), ErrorCategory::ParseError);
    }

    #[test]
    fn test_classify_resource_exhausted() {
        let c = ErrorClassifier::new();
        assert_eq!(
            c.classify("resource exhausted"),
            ErrorCategory::ResourceExhausted
        );
        assert_eq!(
            c.classify("out of memory"),
            ErrorCategory::ResourceExhausted
        );
        assert_eq!(
            c.classify("token limit reached"),
            ErrorCategory::ResourceExhausted
        );
    }

    #[test]
    fn test_classify_unknown() {
        let c = ErrorClassifier::new();
        let cat = c.classify("an odd glitch occurred");
        assert!(matches!(cat, ErrorCategory::Unknown(_)));
    }

    #[test]
    fn test_custom_pattern() {
        let mut c = ErrorClassifier::new();
        c.add_pattern("my_custom_error", ErrorCategory::ToolError);
        assert_eq!(
            c.classify("got a my_custom_error here"),
            ErrorCategory::ToolError
        );
    }

    #[test]
    fn test_custom_pattern_takes_priority() {
        let mut c = ErrorClassifier::new();
        // "timeout" would normally be Timeout, but custom says RateLimit
        c.add_pattern("timeout", ErrorCategory::RateLimit);
        assert_eq!(c.classify("request timeout"), ErrorCategory::RateLimit);
    }

    // === BackoffStrategy tests =============================================

    #[test]
    fn test_fixed_backoff() {
        let b = BackoffStrategy::Fixed(Duration::from_secs(3));
        assert_eq!(b.delay_for_attempt(0), Duration::from_secs(3));
        assert_eq!(b.delay_for_attempt(1), Duration::from_secs(3));
        assert_eq!(b.delay_for_attempt(10), Duration::from_secs(3));
    }

    #[test]
    fn test_linear_backoff() {
        let b = BackoffStrategy::Linear {
            initial: Duration::from_secs(1),
            increment: Duration::from_secs(2),
        };
        assert_eq!(b.delay_for_attempt(0), Duration::from_secs(1));
        assert_eq!(b.delay_for_attempt(1), Duration::from_secs(3));
        assert_eq!(b.delay_for_attempt(2), Duration::from_secs(5));
        assert_eq!(b.delay_for_attempt(3), Duration::from_secs(7));
    }

    #[test]
    fn test_exponential_backoff() {
        let b = BackoffStrategy::Exponential {
            initial: Duration::from_secs(1),
            multiplier: 2.0,
            max: Duration::from_secs(60),
        };
        assert_eq!(b.delay_for_attempt(0), Duration::from_secs(1));
        assert_eq!(b.delay_for_attempt(1), Duration::from_secs(2));
        assert_eq!(b.delay_for_attempt(2), Duration::from_secs(4));
        assert_eq!(b.delay_for_attempt(3), Duration::from_secs(8));
    }

    #[test]
    fn test_exponential_backoff_capped() {
        let b = BackoffStrategy::Exponential {
            initial: Duration::from_secs(1),
            multiplier: 10.0,
            max: Duration::from_secs(30),
        };
        // attempt 0 = 1s, attempt 1 = 10s, attempt 2 = 100s -> capped at 30s
        assert_eq!(b.delay_for_attempt(2), Duration::from_secs(30));
    }

    // === RecoveryPolicy tests ==============================================

    #[test]
    fn test_default_policy_retries_rate_limit() {
        let policy = RecoveryPolicy::new();
        let strategy = policy.get_strategy(&ErrorCategory::RateLimit);
        assert!(matches!(strategy, RecoveryStrategy::Retry { .. }));
    }

    #[test]
    fn test_default_policy_aborts_auth() {
        let policy = RecoveryPolicy::new();
        let strategy = policy.get_strategy(&ErrorCategory::AuthFailure);
        assert_eq!(*strategy, RecoveryStrategy::Abort);
    }

    #[test]
    fn test_custom_strategy() {
        let mut policy = RecoveryPolicy::new();
        policy.set_strategy(
            ErrorCategory::AuthFailure,
            RecoveryStrategy::Fallback("backup-key".into()),
        );
        let strategy = policy.get_strategy(&ErrorCategory::AuthFailure);
        assert_eq!(*strategy, RecoveryStrategy::Fallback("backup-key".into()));
    }

    #[test]
    fn test_with_default_retry() {
        let policy = RecoveryPolicy::with_default_retry(5);
        for cat in &[
            ErrorCategory::RateLimit,
            ErrorCategory::ModelOverloaded,
            ErrorCategory::NetworkError,
            ErrorCategory::Timeout,
        ] {
            match policy.get_strategy(cat) {
                RecoveryStrategy::Retry { max_attempts, .. } => {
                    assert_eq!(*max_attempts, 5);
                }
                other => panic!("Expected Retry for {cat}, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_unknown_category_uses_default() {
        let policy = RecoveryPolicy::new();
        let strategy = policy.get_strategy(&ErrorCategory::Unknown("weird".into()));
        // Default strategy is Abort
        assert_eq!(*strategy, RecoveryStrategy::Abort);
    }

    // === RecoveryLog tests =================================================

    #[test]
    fn test_empty_log() {
        let log = RecoveryLog::new();
        assert_eq!(log.len(), 0);
        assert!(log.is_empty());
        assert_eq!(log.success_rate(), 0.0);
        assert_eq!(log.total_recovery_time(), Duration::ZERO);
    }

    #[test]
    fn test_log_record_and_len() {
        let mut log = RecoveryLog::new();
        log.record(RecoveryAttempt {
            category: ErrorCategory::RateLimit,
            strategy: RecoveryStrategy::Skip,
            attempt: 1,
            succeeded: true,
            duration: Duration::from_secs(1),
        });
        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_log_success_rate() {
        let mut log = RecoveryLog::new();
        log.record(RecoveryAttempt {
            category: ErrorCategory::RateLimit,
            strategy: RecoveryStrategy::Skip,
            attempt: 1,
            succeeded: true,
            duration: Duration::from_secs(1),
        });
        log.record(RecoveryAttempt {
            category: ErrorCategory::Timeout,
            strategy: RecoveryStrategy::Skip,
            attempt: 1,
            succeeded: false,
            duration: Duration::from_secs(2),
        });
        assert!((log.success_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_log_total_recovery_time() {
        let mut log = RecoveryLog::new();
        log.record(RecoveryAttempt {
            category: ErrorCategory::RateLimit,
            strategy: RecoveryStrategy::Skip,
            attempt: 1,
            succeeded: true,
            duration: Duration::from_secs(3),
        });
        log.record(RecoveryAttempt {
            category: ErrorCategory::Timeout,
            strategy: RecoveryStrategy::Skip,
            attempt: 1,
            succeeded: true,
            duration: Duration::from_secs(7),
        });
        assert_eq!(log.total_recovery_time(), Duration::from_secs(10));
    }

    #[test]
    fn test_log_attempts_by_category() {
        let mut log = RecoveryLog::new();
        log.record(RecoveryAttempt {
            category: ErrorCategory::RateLimit,
            strategy: RecoveryStrategy::Skip,
            attempt: 1,
            succeeded: true,
            duration: Duration::ZERO,
        });
        log.record(RecoveryAttempt {
            category: ErrorCategory::Timeout,
            strategy: RecoveryStrategy::Skip,
            attempt: 1,
            succeeded: true,
            duration: Duration::ZERO,
        });
        log.record(RecoveryAttempt {
            category: ErrorCategory::RateLimit,
            strategy: RecoveryStrategy::Skip,
            attempt: 2,
            succeeded: false,
            duration: Duration::ZERO,
        });
        let rate_limit_attempts = log.attempts_by_category(&ErrorCategory::RateLimit);
        assert_eq!(rate_limit_attempts.len(), 2);
    }

    #[test]
    fn test_log_clear() {
        let mut log = RecoveryLog::new();
        log.record(RecoveryAttempt {
            category: ErrorCategory::RateLimit,
            strategy: RecoveryStrategy::Skip,
            attempt: 1,
            succeeded: true,
            duration: Duration::ZERO,
        });
        assert_eq!(log.len(), 1);
        log.clear();
        assert_eq!(log.len(), 0);
        assert!(log.is_empty());
    }

    #[test]
    fn test_log_to_json() {
        let mut log = RecoveryLog::new();
        log.record(RecoveryAttempt {
            category: ErrorCategory::RateLimit,
            strategy: RecoveryStrategy::Skip,
            attempt: 1,
            succeeded: true,
            duration: Duration::from_millis(100),
        });
        let j = log.to_json();
        assert_eq!(j["total_attempts"], 1);
        assert_eq!(j["success_rate"], 1.0);
    }

    // === RecoveryAction tests ==============================================

    #[test]
    fn test_recovery_action_should_retry_true() {
        let action = RecoveryAction {
            action: RecoveryStrategy::Retry {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            },
            delay: Duration::from_secs(1),
            attempt: 2,
            category: ErrorCategory::RateLimit,
        };
        assert!(action.should_retry());
    }

    #[test]
    fn test_recovery_action_should_retry_false_exceeded() {
        let action = RecoveryAction {
            action: RecoveryStrategy::Retry {
                max_attempts: 3,
                backoff: BackoffStrategy::Fixed(Duration::from_secs(1)),
            },
            delay: Duration::from_secs(1),
            attempt: 4,
            category: ErrorCategory::RateLimit,
        };
        assert!(!action.should_retry());
    }

    #[test]
    fn test_recovery_action_should_retry_false_not_retry() {
        let action = RecoveryAction {
            action: RecoveryStrategy::Abort,
            delay: Duration::ZERO,
            attempt: 1,
            category: ErrorCategory::AuthFailure,
        };
        assert!(!action.should_retry());
    }

    #[test]
    fn test_recovery_action_to_json() {
        let action = RecoveryAction {
            action: RecoveryStrategy::Skip,
            delay: Duration::from_millis(50),
            attempt: 1,
            category: ErrorCategory::ToolError,
        };
        let j = action.to_json();
        assert_eq!(j["attempt"], 1);
        assert_eq!(j["category"], "ToolError");
        assert_eq!(j["should_retry"], false);
    }

    // === RecoveryAttempt tests =============================================

    #[test]
    fn test_recovery_attempt_to_json() {
        let a = RecoveryAttempt {
            category: ErrorCategory::Timeout,
            strategy: RecoveryStrategy::Abort,
            attempt: 3,
            succeeded: false,
            duration: Duration::from_millis(250),
        };
        let j = a.to_json();
        assert_eq!(j["category"], "Timeout");
        assert_eq!(j["attempt"], 3);
        assert_eq!(j["succeeded"], false);
        assert_eq!(j["duration_ms"], 250);
    }

    // === RecoveryManager end-to-end tests ==================================

    #[test]
    fn test_manager_handle_retryable_error() {
        let policy = RecoveryPolicy::with_default_retry(3);
        let mut mgr = RecoveryManager::new(policy);
        let action = mgr.handle_error("rate limit exceeded");
        assert_eq!(action.category, ErrorCategory::RateLimit);
        assert!(action.should_retry());
        assert_eq!(action.attempt, 1);
    }

    #[test]
    fn test_manager_handle_non_retryable_error() {
        let mut mgr = RecoveryManager::new(RecoveryPolicy::new());
        let action = mgr.handle_error("unauthorized access denied");
        assert_eq!(action.category, ErrorCategory::AuthFailure);
        assert!(!action.should_retry());
    }

    #[test]
    fn test_manager_multiple_attempts_increment() {
        let policy = RecoveryPolicy::with_default_retry(5);
        let mut mgr = RecoveryManager::new(policy);

        let a1 = mgr.handle_error("rate limit exceeded");
        assert_eq!(a1.attempt, 1);

        let a2 = mgr.handle_error("rate limit hit again");
        assert_eq!(a2.attempt, 2);

        let a3 = mgr.handle_error("still rate limited");
        assert_eq!(a3.attempt, 3);
    }

    #[test]
    fn test_manager_exhausts_retries() {
        let policy = RecoveryPolicy::with_default_retry(2);
        let mut mgr = RecoveryManager::new(policy);

        let a1 = mgr.handle_error("rate limit");
        assert!(a1.should_retry());

        let a2 = mgr.handle_error("rate limit");
        assert!(a2.should_retry());

        let a3 = mgr.handle_error("rate limit");
        assert!(!a3.should_retry());
    }

    #[test]
    fn test_manager_log_is_populated() {
        let mut mgr = RecoveryManager::new(RecoveryPolicy::new());
        mgr.handle_error("timeout occurred");
        mgr.handle_error("unauthorized");
        assert_eq!(mgr.log().len(), 2);
    }

    #[test]
    fn test_manager_reset_clears_state() {
        let mut mgr = RecoveryManager::new(RecoveryPolicy::with_default_retry(3));
        mgr.handle_error("rate limit");
        mgr.handle_error("rate limit");
        assert_eq!(mgr.log().len(), 2);

        mgr.reset();
        assert_eq!(mgr.log().len(), 0);

        // Attempt counter should also reset.
        let action = mgr.handle_error("rate limit");
        assert_eq!(action.attempt, 1);
    }

    #[test]
    fn test_manager_with_classifier() {
        let mut classifier = ErrorClassifier::new();
        classifier.add_pattern("custom_overload", ErrorCategory::ModelOverloaded);
        let mut mgr = RecoveryManager::with_classifier(classifier);
        let action = mgr.handle_error("custom_overload detected");
        assert_eq!(action.category, ErrorCategory::ModelOverloaded);
    }

    #[test]
    fn test_manager_backoff_increases_for_exponential() {
        let policy = RecoveryPolicy::with_default_retry(5);
        let mut mgr = RecoveryManager::new(policy);

        let a1 = mgr.handle_error("rate limit");
        let a2 = mgr.handle_error("rate limit");
        let a3 = mgr.handle_error("rate limit");

        // Exponential: 1s, 2s, 4s
        assert!(a2.delay > a1.delay);
        assert!(a3.delay > a2.delay);
    }

    #[test]
    fn test_zero_attempt_exponential_backoff() {
        let b = BackoffStrategy::Exponential {
            initial: Duration::from_secs(1),
            multiplier: 2.0,
            max: Duration::from_secs(60),
        };
        assert_eq!(b.delay_for_attempt(0), Duration::from_secs(1));
    }
}
