//! Tool retry middleware — automatic retry on tool execution failures.
//!
//! Mirrors Python `langchain.agents.middleware.tool_retry`.

use std::collections::HashSet;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use cognis_core::error::{CognisError, Result};
use cognis_core::tools::base::BaseTool;

use super::types::AgentMiddleware;

/// Behavior when all retries are exhausted.
#[derive(Debug, Clone, Default)]
pub enum OnToolFailure {
    /// Re-raise the error (current behavior).
    #[default]
    Error,
    /// Return a JSON value with the error message so the LLM can recover.
    Continue,
}

/// Backoff configuration for tool retries.
#[derive(Debug, Clone)]
pub struct ToolRetryBackoff {
    /// Initial delay between retries in milliseconds.
    pub initial_delay_ms: u64,
    /// Backoff multiplier (1.0 = constant, 2.0 = exponential doubling).
    pub multiplier: f64,
    /// Maximum delay cap in milliseconds.
    pub max_delay_ms: u64,
    /// Whether to add random jitter to the delay.
    pub jitter: bool,
}

impl Default for ToolRetryBackoff {
    fn default() -> Self {
        Self {
            initial_delay_ms: 500,
            multiplier: 2.0,
            max_delay_ms: 30_000,
            jitter: true,
        }
    }
}

impl ToolRetryBackoff {
    /// Calculate the delay for a given attempt number (0-indexed).
    ///
    /// When jitter is enabled, adds pseudo-random variation (up to +/- 25%)
    /// using a simple hash-based approach (no cryptographic randomness needed).
    pub fn calculate_delay(&self, attempt: usize) -> Duration {
        let base = self.initial_delay_ms as f64 * self.multiplier.powi(attempt as i32);
        let capped = base.min(self.max_delay_ms as f64);

        if self.jitter {
            // Simple pseudo-random jitter using hash of attempt + current time
            let now_nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos() as u64;
            // Simple hash: mix attempt and time
            let hash = now_nanos
                .wrapping_mul(6364136223846793005)
                .wrapping_add(attempt as u64);
            // Map to jitter factor in range [0.75, 1.25]
            let jitter_factor = 0.75 + (hash % 1000) as f64 / 2000.0;
            let jittered = (capped * jitter_factor).min(self.max_delay_ms as f64);
            Duration::from_millis(jittered.max(1.0) as u64)
        } else {
            Duration::from_millis(capped as u64)
        }
    }

    /// Calculate the delay without jitter (for deterministic testing).
    pub fn calculate_base_delay(&self, attempt: usize) -> Duration {
        let base = self.initial_delay_ms as f64 * self.multiplier.powi(attempt as i32);
        let capped = base.min(self.max_delay_ms as f64);
        Duration::from_millis(capped as u64)
    }
}

/// Conditions under which a tool call should be retried.
#[derive(Debug, Clone, Default)]
pub enum RetryOn {
    /// Retry on any error.
    #[default]
    AnyError,
    /// Retry only on specific error messages (substring match).
    ErrorContains(Vec<String>),
    /// Retry on specific error types.
    ErrorTypes(Vec<String>),
}

impl RetryOn {
    /// Check if an error matches this retry condition.
    pub fn matches(&self, error: &CognisError) -> bool {
        match self {
            RetryOn::AnyError => true,
            RetryOn::ErrorContains(substrings) => {
                let msg = error.to_string();
                substrings.iter().any(|s| msg.contains(s))
            }
            RetryOn::ErrorTypes(types) => {
                let error_type = match error {
                    CognisError::ToolException(_) => "ToolException",
                    CognisError::ToolValidationError(_) => "ToolValidationError",
                    CognisError::IoError(_) => "IoError",
                    CognisError::HttpError { .. } => "HttpError",
                    CognisError::Other(_) => "Other",
                    _ => "Unknown",
                };
                types.iter().any(|t| t == error_type)
            }
        }
    }
}

/// Tool filter: which tools this retry middleware applies to.
#[derive(Debug, Clone, Default)]
pub enum ToolFilter {
    /// Apply to all tools.
    #[default]
    All,
    /// Apply only to tools with these names.
    Only(HashSet<String>),
    /// Apply to all tools except those with these names.
    Except(HashSet<String>),
}

impl ToolFilter {
    /// Check if a tool name matches this filter.
    pub fn matches(&self, tool_name: &str) -> bool {
        match self {
            ToolFilter::All => true,
            ToolFilter::Only(names) => names.contains(tool_name),
            ToolFilter::Except(names) => !names.contains(tool_name),
        }
    }
}

/// Middleware that retries failed tool calls with configurable backoff.
pub struct ToolRetryMiddleware {
    /// Maximum number of retries per tool call.
    pub max_retries: usize,
    /// Which tools to apply retry logic to.
    pub tool_filter: ToolFilter,
    /// Conditions under which to retry.
    pub retry_on: RetryOn,
    /// Backoff configuration.
    pub backoff: ToolRetryBackoff,
    /// Behavior when all retries are exhausted.
    pub on_failure: OnToolFailure,
}

impl ToolRetryMiddleware {
    pub fn new(max_retries: usize) -> Self {
        Self {
            max_retries,
            tool_filter: ToolFilter::default(),
            retry_on: RetryOn::default(),
            backoff: ToolRetryBackoff::default(),
            on_failure: OnToolFailure::default(),
        }
    }

    pub fn with_tool_filter(mut self, filter: ToolFilter) -> Self {
        self.tool_filter = filter;
        self
    }

    pub fn with_retry_on(mut self, retry_on: RetryOn) -> Self {
        self.retry_on = retry_on;
        self
    }

    pub fn with_backoff(mut self, backoff: ToolRetryBackoff) -> Self {
        self.backoff = backoff;
        self
    }

    pub fn with_on_failure(mut self, on_failure: OnToolFailure) -> Self {
        self.on_failure = on_failure;
        self
    }
}

#[async_trait]
impl AgentMiddleware for ToolRetryMiddleware {
    fn name(&self) -> &str {
        "ToolRetryMiddleware"
    }

    async fn wrap_tool_call(
        &self,
        tool: &dyn BaseTool,
        input: &Value,
        handler: &(dyn for<'a, 'b> Fn(&'a dyn BaseTool, &'b Value) -> Result<Value> + Send + Sync),
    ) -> Result<Value> {
        if !self.tool_filter.matches(tool.name()) {
            return handler(tool, input);
        }

        let mut last_error: Option<CognisError> = None;
        for attempt in 0..=self.max_retries {
            match handler(tool, input) {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if !self.retry_on.matches(&e) || attempt == self.max_retries {
                        // All retries exhausted or non-retryable error
                        return match &self.on_failure {
                            OnToolFailure::Error => Err(e),
                            OnToolFailure::Continue => Ok(serde_json::json!({
                                "error": true,
                                "message": format!(
                                    "Tool '{}' failed after {} retries: {}",
                                    tool.name(),
                                    self.max_retries,
                                    e
                                ),
                                "tool": tool.name()
                            })),
                        };
                    }
                    last_error = Some(e);
                    let delay = self.backoff.calculate_delay(attempt);
                    tokio::time::sleep(delay).await;
                }
            }
        }

        match &self.on_failure {
            OnToolFailure::Error => {
                Err(last_error
                    .unwrap_or_else(|| CognisError::Other("Unknown tool retry error".into())))
            }
            OnToolFailure::Continue => {
                let error = last_error
                    .unwrap_or_else(|| CognisError::Other("Unknown tool retry error".into()));
                Ok(serde_json::json!({
                    "error": true,
                    "message": format!(
                        "Tool '{}' failed after {} retries: {}",
                        tool.name(),
                        self.max_retries,
                        error
                    ),
                    "tool": tool.name()
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tool_retry_new() {
        let mw = ToolRetryMiddleware::new(3);
        assert_eq!(mw.max_retries, 3);
        assert_eq!(mw.name(), "ToolRetryMiddleware");
    }

    #[test]
    fn test_tool_retry_backoff_calculation() {
        let backoff = ToolRetryBackoff {
            initial_delay_ms: 100,
            multiplier: 2.0,
            max_delay_ms: 10_000,
            jitter: false,
        };
        assert_eq!(backoff.calculate_delay(0).as_millis(), 100);
        assert_eq!(backoff.calculate_delay(1).as_millis(), 200);
        assert_eq!(backoff.calculate_delay(2).as_millis(), 400);
    }

    #[test]
    fn test_tool_retry_backoff_capped() {
        let backoff = ToolRetryBackoff {
            initial_delay_ms: 5000,
            multiplier: 3.0,
            max_delay_ms: 10_000,
            jitter: false,
        };
        assert_eq!(backoff.calculate_delay(2).as_millis(), 10_000);
    }

    #[test]
    fn test_tool_retry_backoff_base_delay() {
        let backoff = ToolRetryBackoff {
            initial_delay_ms: 100,
            multiplier: 2.0,
            max_delay_ms: 10_000,
            jitter: true,
        };
        // base_delay ignores jitter
        assert_eq!(backoff.calculate_base_delay(0).as_millis(), 100);
        assert_eq!(backoff.calculate_base_delay(1).as_millis(), 200);
    }

    #[test]
    fn test_tool_retry_backoff_jitter() {
        let backoff = ToolRetryBackoff {
            initial_delay_ms: 1000,
            multiplier: 1.0,
            max_delay_ms: 10_000,
            jitter: true,
        };
        // With jitter, delay should vary but stay within [750, 1250]
        let delay = backoff.calculate_delay(0).as_millis();
        assert!(
            delay >= 750 && delay <= 1250,
            "Jittered delay {} out of expected range",
            delay
        );
    }

    #[test]
    fn test_tool_filter_all() {
        let filter = ToolFilter::All;
        assert!(filter.matches("any_tool"));
    }

    #[test]
    fn test_tool_filter_only() {
        let mut names = HashSet::new();
        names.insert("search".into());
        let filter = ToolFilter::Only(names);
        assert!(filter.matches("search"));
        assert!(!filter.matches("other"));
    }

    #[test]
    fn test_tool_filter_except() {
        let mut names = HashSet::new();
        names.insert("dangerous".into());
        let filter = ToolFilter::Except(names);
        assert!(filter.matches("safe"));
        assert!(!filter.matches("dangerous"));
    }

    #[test]
    fn test_retry_on_any_error() {
        let cond = RetryOn::AnyError;
        assert!(cond.matches(&CognisError::Other("test".into())));
    }

    #[test]
    fn test_retry_on_error_contains() {
        let cond = RetryOn::ErrorContains(vec!["timeout".into()]);
        assert!(cond.matches(&CognisError::Other("connection timeout".into())));
        assert!(!cond.matches(&CognisError::Other("bad input".into())));
    }

    #[test]
    fn test_retry_on_error_types() {
        let cond = RetryOn::ErrorTypes(vec!["ToolException".into()]);
        assert!(cond.matches(&CognisError::ToolException("fail".into())));
        assert!(!cond.matches(&CognisError::Other("fail".into())));
    }

    #[test]
    fn test_tool_retry_builder() {
        let mw = ToolRetryMiddleware::new(5)
            .with_tool_filter(ToolFilter::All)
            .with_retry_on(RetryOn::AnyError)
            .with_backoff(ToolRetryBackoff::default());
        assert_eq!(mw.max_retries, 5);
    }

    #[test]
    fn test_tool_retry_builder_with_on_failure() {
        let mw = ToolRetryMiddleware::new(3).with_on_failure(OnToolFailure::Continue);
        assert!(matches!(mw.on_failure, OnToolFailure::Continue));
    }

    #[test]
    fn test_on_tool_failure_default() {
        let mw = ToolRetryMiddleware::new(3);
        assert!(matches!(mw.on_failure, OnToolFailure::Error));
    }
}
