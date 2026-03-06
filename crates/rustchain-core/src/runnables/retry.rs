use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, RustChainError};

use super::base::Runnable;
use super::config::RunnableConfig;
use super::RunnableStream;

/// A runnable that retries the wrapped runnable on failure with exponential backoff.
///
/// Wraps any `Runnable` and automatically retries failed invocations using
/// configurable exponential backoff. An optional error filter allows retrying
/// only specific error types.
///
/// # Example
/// ```ignore
/// use rustchain_core::runnables::{RunnableLambda, RunnableExt};
///
/// let chain = my_runnable.with_retry(3, 500);
/// ```
pub struct RunnableRetry {
    /// The inner runnable to retry on failure.
    inner: Arc<dyn Runnable>,
    /// Maximum number of retry attempts (total calls = max_retries + 1).
    max_retries: u32,
    /// Initial delay in milliseconds before the first retry.
    initial_delay_ms: u64,
    /// Multiply delay by this factor after each retry.
    backoff_factor: f64,
    /// Maximum delay cap in milliseconds.
    max_delay_ms: u64,
    /// Optional filter: if provided, only retry when this returns `true` for the error.
    retry_on: Option<Box<dyn Fn(&RustChainError) -> bool + Send + Sync>>,
}

impl RunnableRetry {
    /// Create a new `RunnableRetry` wrapping the given runnable.
    ///
    /// Defaults: max_retries=3, initial_delay_ms=500, backoff_factor=2.0, max_delay_ms=30000.
    pub fn new(inner: Arc<dyn Runnable>, max_retries: u32) -> Self {
        Self {
            inner,
            max_retries,
            initial_delay_ms: 500,
            backoff_factor: 2.0,
            max_delay_ms: 30_000,
            retry_on: None,
        }
    }

    /// Set the initial delay in milliseconds before the first retry.
    pub fn with_initial_delay(mut self, delay_ms: u64) -> Self {
        self.initial_delay_ms = delay_ms;
        self
    }

    /// Set the backoff multiplier applied after each retry.
    pub fn with_backoff_factor(mut self, factor: f64) -> Self {
        self.backoff_factor = factor;
        self
    }

    /// Set the maximum delay cap in milliseconds.
    pub fn with_max_delay(mut self, max_delay_ms: u64) -> Self {
        self.max_delay_ms = max_delay_ms;
        self
    }

    /// Configure the wait times for exponential backoff (legacy convenience method).
    pub fn with_wait(mut self, initial_ms: u64, max_ms: u64) -> Self {
        self.initial_delay_ms = initial_ms;
        self.max_delay_ms = max_ms;
        self
    }

    /// Set an error filter function. Only errors for which the filter returns
    /// `true` will be retried; others are returned immediately.
    pub fn with_retry_on<F>(mut self, filter: F) -> Self
    where
        F: Fn(&RustChainError) -> bool + Send + Sync + 'static,
    {
        self.retry_on = Some(Box::new(filter));
        self
    }
}

#[async_trait]
impl Runnable for RunnableRetry {
    fn name(&self) -> &str {
        "RunnableRetry"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let mut last_err = None;
        let mut delay = self.initial_delay_ms;

        for attempt in 0..=self.max_retries {
            match self.inner.invoke(input.clone(), config).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    // If a filter is set and the error doesn't match, fail immediately.
                    if let Some(ref filter) = self.retry_on {
                        if !filter(&e) {
                            return Err(e);
                        }
                    }
                    last_err = Some(e);
                    if attempt < self.max_retries {
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                        delay =
                            (delay as f64 * self.backoff_factor).min(self.max_delay_ms as f64) as u64;
                    }
                }
            }
        }

        Err(last_err.unwrap())
    }

    /// Batch invocation: retries each input independently.
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

    /// Stream invocation: retries the entire stream call on failure.
    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        let mut last_err = None;
        let mut delay = self.initial_delay_ms;

        for attempt in 0..=self.max_retries {
            match self.inner.stream(input.clone(), config).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    if let Some(ref filter) = self.retry_on {
                        if !filter(&e) {
                            return Err(e);
                        }
                    }
                    last_err = Some(e);
                    if attempt < self.max_retries {
                        tokio::time::sleep(Duration::from_millis(delay)).await;
                        delay =
                            (delay as f64 * self.backoff_factor).min(self.max_delay_ms as f64) as u64;
                    }
                }
            }
        }

        Err(last_err.unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A test runnable that fails a specified number of times before succeeding.
    struct FailNTimes {
        fail_count: u32,
        attempts: AtomicU32,
    }

    impl FailNTimes {
        fn new(fail_count: u32) -> Self {
            Self {
                fail_count,
                attempts: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl Runnable for FailNTimes {
        fn name(&self) -> &str {
            "FailNTimes"
        }

        async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            if attempt < self.fail_count {
                Err(RustChainError::Other(format!("attempt {} failed", attempt)))
            } else {
                Ok(input)
            }
        }
    }

    /// A test runnable that always fails.
    struct AlwaysFails;

    #[async_trait]
    impl Runnable for AlwaysFails {
        fn name(&self) -> &str {
            "AlwaysFails"
        }

        async fn invoke(&self, _input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
            Err(RustChainError::Other("always fails".into()))
        }
    }

    /// A test runnable that always fails with a specific error variant.
    struct FailsWithToolError;

    #[async_trait]
    impl Runnable for FailsWithToolError {
        fn name(&self) -> &str {
            "FailsWithToolError"
        }

        async fn invoke(&self, _input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
            Err(RustChainError::ToolException("tool broke".into()))
        }
    }

    #[tokio::test]
    async fn test_retry_succeeds_first_try() {
        let inner = Arc::new(FailNTimes::new(0)); // never fails
        let retry = RunnableRetry::new(inner, 3);

        let result = retry.invoke(serde_json::json!(42), None).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), serde_json::json!(42));
    }

    #[tokio::test]
    async fn test_retry_succeeds_after_failures() {
        let inner = Arc::new(FailNTimes::new(2)); // fails twice, succeeds on 3rd
        let retry = RunnableRetry::new(inner, 3)
            .with_initial_delay(1) // minimal delay for tests
            .with_max_delay(10);

        let result = retry.invoke(serde_json::json!("hello"), None).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), serde_json::json!("hello"));
    }

    #[tokio::test]
    async fn test_retry_exhausts_attempts() {
        let inner = Arc::new(AlwaysFails);
        let retry = RunnableRetry::new(inner, 2)
            .with_initial_delay(1)
            .with_max_delay(5);

        let result = retry.invoke(serde_json::json!("test"), None).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{}", err).contains("always fails"));
    }

    #[tokio::test]
    async fn test_retry_with_filter() {
        // Only retry ToolException errors, not Other errors.
        let inner = Arc::new(AlwaysFails); // produces Other error
        let retry = RunnableRetry::new(inner, 3)
            .with_initial_delay(1)
            .with_retry_on(|e| matches!(e, RustChainError::ToolException(_)));

        let result = retry.invoke(serde_json::json!("test"), None).await;
        // Should fail immediately without retrying because the error is Other, not ToolException.
        assert!(result.is_err());

        // Now test with a ToolException error - it should retry and eventually exhaust attempts.
        let inner2 = Arc::new(FailsWithToolError);
        let retry2 = RunnableRetry::new(inner2, 1)
            .with_initial_delay(1)
            .with_retry_on(|e| matches!(e, RustChainError::ToolException(_)));

        let result2 = retry2.invoke(serde_json::json!("test"), None).await;
        assert!(result2.is_err());
        assert!(format!("{}", result2.unwrap_err()).contains("tool broke"));
    }

    #[tokio::test]
    async fn test_retry_backoff_delay() {
        let inner = Arc::new(AlwaysFails);
        let retry = RunnableRetry::new(inner, 2)
            .with_initial_delay(50)
            .with_backoff_factor(2.0)
            .with_max_delay(30_000);

        let start = tokio::time::Instant::now();
        let _ = retry.invoke(serde_json::json!("test"), None).await;
        let elapsed = start.elapsed();

        // With 2 retries: delay after attempt 0 = 50ms, delay after attempt 1 = 100ms.
        // Total minimum delay = 150ms. Allow some slack.
        assert!(
            elapsed.as_millis() >= 140,
            "Expected at least ~150ms of backoff delay, got {}ms",
            elapsed.as_millis()
        );
    }
}
