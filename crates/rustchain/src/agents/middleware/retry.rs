//! Shared retry utilities for middleware.
//!
//! Mirrors Python `langchain.agents.middleware._retry`.

use std::sync::Arc;
use std::time::Duration;

use rustchain_core::error::RustChainError;

/// Strategy for handling failures after retries are exhausted.
#[derive(Debug, Clone)]
pub enum OnFailure {
    /// Continue with an error message injected into the conversation.
    Continue,
    /// Re-raise the error.
    Error,
}

/// Condition under which a model call should be retried.
#[derive(Default)]
pub enum RetryCondition {
    /// Retry on any error.
    #[default]
    AnyError,
    /// Retry only when the error message contains one of the given substrings.
    ErrorContains(Vec<String>),
    /// Custom predicate for deciding whether to retry.
    Custom(Arc<dyn Fn(&RustChainError) -> bool + Send + Sync>),
}

impl std::fmt::Debug for RetryCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RetryCondition::AnyError => write!(f, "AnyError"),
            RetryCondition::ErrorContains(v) => write!(f, "ErrorContains({:?})", v),
            RetryCondition::Custom(_) => write!(f, "Custom(<fn>)"),
        }
    }
}

impl Clone for RetryCondition {
    fn clone(&self) -> Self {
        match self {
            RetryCondition::AnyError => RetryCondition::AnyError,
            RetryCondition::ErrorContains(v) => RetryCondition::ErrorContains(v.clone()),
            RetryCondition::Custom(f) => RetryCondition::Custom(Arc::clone(f)),
        }
    }
}

impl RetryCondition {
    /// Check if the given error matches this retry condition.
    pub fn matches(&self, error: &RustChainError) -> bool {
        match self {
            RetryCondition::AnyError => true,
            RetryCondition::ErrorContains(substrings) => {
                let msg = error.to_string();
                substrings.iter().any(|s| msg.contains(s))
            }
            RetryCondition::Custom(f) => f(error),
        }
    }
}

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts.
    pub max_retries: usize,
    /// Initial delay between retries in milliseconds.
    pub initial_delay_ms: u64,
    /// Backoff multiplier (1.0 = constant, 2.0 = exponential doubling).
    /// A value of 0.0 is treated as constant delay (equivalent to 1.0).
    pub backoff_multiplier: f64,
    /// Maximum delay cap in milliseconds.
    pub max_delay_ms: u64,
    /// Whether to add jitter to the delay.
    pub jitter: bool,
    /// Strategy when all retries are exhausted.
    pub on_failure: OnFailure,
    /// Condition under which to retry (default: any error).
    pub retry_on: RetryCondition,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            initial_delay_ms: 1000,
            backoff_multiplier: 2.0,
            max_delay_ms: 60_000,
            jitter: true,
            on_failure: OnFailure::Continue,
            retry_on: RetryCondition::default(),
        }
    }
}

impl RetryConfig {
    pub fn new(max_retries: usize) -> Self {
        Self {
            max_retries,
            ..Default::default()
        }
    }

    /// Calculate the delay for a given attempt number (0-indexed).
    pub fn calculate_delay(&self, attempt: usize) -> Duration {
        // Treat backoff_multiplier of 0.0 as constant delay.
        let effective_multiplier = if self.backoff_multiplier == 0.0 {
            1.0
        } else {
            self.backoff_multiplier
        };

        let base_ms = self.initial_delay_ms as f64 * effective_multiplier.powi(attempt as i32);
        let capped_ms = base_ms.min(self.max_delay_ms as f64);

        let final_ms = if self.jitter {
            // Use system time nanos for simple pseudo-randomness to vary jitter.
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos();
            // Map nanos to a factor in [0.5, 1.0)
            let jitter_factor = 0.5 + (nanos as f64 % 1_000_000.0) / 2_000_000.0;
            capped_ms * jitter_factor
        } else {
            capped_ms
        };

        Duration::from_millis(final_ms.max(1.0) as u64)
    }
}

/// Check if an error should trigger a retry based on a `RetryCondition`.
///
/// This replaces the old hardcoded `should_retry` function.
pub fn should_retry(error: &RustChainError, condition: &RetryCondition) -> bool {
    condition.matches(error)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_retry_config_default() {
        let config = RetryConfig::default();
        assert_eq!(config.max_retries, 2);
        assert_eq!(config.initial_delay_ms, 1000);
        assert_eq!(config.backoff_multiplier, 2.0);
        assert!(matches!(config.on_failure, OnFailure::Continue));
        assert!(matches!(config.retry_on, RetryCondition::AnyError));
    }

    #[test]
    fn test_calculate_delay_exponential() {
        let config = RetryConfig {
            initial_delay_ms: 100,
            backoff_multiplier: 2.0,
            jitter: false,
            max_delay_ms: 10_000,
            ..Default::default()
        };
        assert_eq!(config.calculate_delay(0).as_millis(), 100);
        assert_eq!(config.calculate_delay(1).as_millis(), 200);
        assert_eq!(config.calculate_delay(2).as_millis(), 400);
    }

    #[test]
    fn test_calculate_delay_capped() {
        let config = RetryConfig {
            initial_delay_ms: 5000,
            backoff_multiplier: 3.0,
            jitter: false,
            max_delay_ms: 10_000,
            ..Default::default()
        };
        // 5000 * 3^1 = 15000, capped at 10000
        assert_eq!(config.calculate_delay(1).as_millis(), 10_000);
    }

    #[test]
    fn test_calculate_delay_zero_multiplier() {
        let config = RetryConfig {
            initial_delay_ms: 500,
            backoff_multiplier: 0.0,
            jitter: false,
            max_delay_ms: 60_000,
            ..Default::default()
        };
        // backoff_multiplier of 0.0 should be treated as constant delay
        assert_eq!(config.calculate_delay(0).as_millis(), 500);
        assert_eq!(config.calculate_delay(1).as_millis(), 500);
        assert_eq!(config.calculate_delay(5).as_millis(), 500);
    }

    #[test]
    fn test_calculate_delay_with_jitter() {
        let config = RetryConfig {
            initial_delay_ms: 1000,
            backoff_multiplier: 1.0,
            jitter: true,
            max_delay_ms: 60_000,
            ..Default::default()
        };
        let delay = config.calculate_delay(0);
        // Jitter should produce a value between 500ms and 1000ms
        assert!(delay.as_millis() >= 500);
        assert!(delay.as_millis() <= 1000);
    }

    #[test]
    fn test_should_retry_any_error() {
        let cond = RetryCondition::AnyError;
        assert!(should_retry(
            &RustChainError::Other("timeout".into()),
            &cond
        ));
        assert!(should_retry(
            &RustChainError::ToolException("bad".into()),
            &cond
        ));
    }

    #[test]
    fn test_should_retry_error_contains() {
        let cond = RetryCondition::ErrorContains(vec!["timeout".into()]);
        assert!(should_retry(
            &RustChainError::Other("connection timeout".into()),
            &cond
        ));
        assert!(!should_retry(
            &RustChainError::Other("bad input".into()),
            &cond
        ));
    }

    #[test]
    fn test_should_retry_custom() {
        let cond =
            RetryCondition::Custom(Arc::new(|e| matches!(e, RustChainError::HttpError { .. })));
        assert!(should_retry(
            &RustChainError::HttpError {
                status: 500,
                body: "err".into()
            },
            &cond
        ));
        assert!(!should_retry(&RustChainError::Other("nope".into()), &cond));
    }

    #[test]
    fn test_retry_condition_clone() {
        let cond = RetryCondition::ErrorContains(vec!["test".into()]);
        let cloned = cond.clone();
        assert!(matches!(cloned, RetryCondition::ErrorContains(_)));
    }
}
