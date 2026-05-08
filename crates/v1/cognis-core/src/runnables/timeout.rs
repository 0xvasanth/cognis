//! Timeout and deadline wrappers for runnables.
//!
//! Provides [`Runnable`] wrappers that enforce time limits on execution.
//! [`RunnableTimeout`] uses a relative duration from invocation, while
//! [`RunnableDeadline`] enforces an absolute point in time, which is useful
//! when multiple operations share a common deadline.
//!
//! - [`TimeoutBehavior`] — controls what happens on timeout (error, default value, or fallback runnable)
//! - [`TimeoutConfig`] — configuration combining a duration with a [`TimeoutBehavior`]
//! - [`RunnableTimeout`] — wrapper that enforces a relative timeout on each invocation
//! - [`RunnableDeadline`] — wrapper that enforces an absolute deadline across invocations

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use serde_json::Value;

use crate::error::{CognisError, Result};

use super::base::Runnable;
use super::config::RunnableConfig;
use super::RunnableStream;

/// Behavior when a timeout is reached.
///
/// Controls what happens when a runnable exceeds its allotted time.
pub enum TimeoutBehavior {
    /// Return a `CognisError::Other` with a timeout message.
    Error,
    /// Return a default `Value` instead of an error.
    Default(Value),
    /// Delegate to a fallback runnable on timeout.
    Fallback(Arc<dyn Runnable>),
}

/// Configuration for timeout behavior on a runnable.
///
/// # Example
/// ```ignore
/// use std::time::Duration;
/// use cognis_core::runnables::timeout::TimeoutConfig;
///
/// let config = TimeoutConfig::new(Duration::from_secs(5))
///     .with_behavior(TimeoutBehavior::Error);
/// ```
pub struct TimeoutConfig {
    /// Maximum duration allowed for the operation.
    pub duration: Duration,
    /// What to do when the timeout is reached.
    pub on_timeout: TimeoutBehavior,
}

impl TimeoutConfig {
    /// Create a new timeout configuration with the given duration.
    ///
    /// Defaults to `TimeoutBehavior::Error`.
    pub fn new(duration: Duration) -> Self {
        Self {
            duration,
            on_timeout: TimeoutBehavior::Error,
        }
    }

    /// Set the timeout behavior.
    pub fn with_behavior(mut self, behavior: TimeoutBehavior) -> Self {
        self.on_timeout = behavior;
        self
    }

    /// Convenience: set `TimeoutBehavior::Default` with the given value.
    pub fn with_default_value(self, value: Value) -> Self {
        self.with_behavior(TimeoutBehavior::Default(value))
    }

    /// Convenience: set `TimeoutBehavior::Fallback` with the given runnable.
    pub fn with_fallback(self, fallback: Arc<dyn Runnable>) -> Self {
        self.with_behavior(TimeoutBehavior::Fallback(fallback))
    }
}

/// A runnable wrapper that enforces a relative timeout on the inner runnable.
///
/// If the inner runnable does not complete within the configured duration,
/// the behavior is determined by `TimeoutConfig::on_timeout`.
///
/// # Example
/// ```ignore
/// use std::time::Duration;
/// use cognis_core::runnables::{RunnableLambda, RunnableExt};
///
/// let with_timeout = my_runnable.with_timeout(Duration::from_secs(5));
/// ```
pub struct RunnableTimeout {
    /// The wrapped runnable.
    inner: Arc<dyn Runnable>,
    /// Timeout configuration.
    config: TimeoutConfig,
}

impl RunnableTimeout {
    /// Create a new timeout wrapper with the given configuration.
    pub fn new(inner: Arc<dyn Runnable>, config: TimeoutConfig) -> Self {
        Self { inner, config }
    }

    /// Handle a timeout according to the configured behavior.
    async fn handle_timeout(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        match &self.config.on_timeout {
            TimeoutBehavior::Error => Err(CognisError::Other(format!(
                "Timeout: {} exceeded {:?}",
                self.inner.name(),
                self.config.duration
            ))),
            TimeoutBehavior::Default(val) => Ok(val.clone()),
            TimeoutBehavior::Fallback(fallback) => fallback.invoke(input, config).await,
        }
    }
}

#[async_trait]
impl Runnable for RunnableTimeout {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        match tokio::time::timeout(
            self.config.duration,
            self.inner.invoke(input.clone(), config),
        )
        .await
        {
            Ok(result) => result,
            Err(_elapsed) => self.handle_timeout(input, config).await,
        }
    }

    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        // Apply timeout to getting the first chunk from the stream.
        let duration = self.config.duration;
        match tokio::time::timeout(duration, self.inner.stream(input.clone(), config)).await {
            Ok(Ok(inner_stream)) => {
                // Wrap the stream so that each item is subject to the timeout as well.
                // We use a simple approach: timeout on the first chunk.
                let deadline = tokio::time::Instant::now() + duration;
                let wrapped = inner_stream.take_while(move |_| {
                    let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
                    async move { remaining > Duration::ZERO }
                });
                Ok(Box::pin(wrapped))
            }
            Ok(Err(e)) => Err(e),
            Err(_elapsed) => {
                // Timeout getting the stream itself.
                match &self.config.on_timeout {
                    TimeoutBehavior::Error => Err(CognisError::Other(format!(
                        "Timeout: {} stream exceeded {:?}",
                        self.inner.name(),
                        self.config.duration
                    ))),
                    TimeoutBehavior::Default(val) => {
                        let val = val.clone();
                        Ok(Box::pin(stream::once(async move { Ok(val) })))
                    }
                    TimeoutBehavior::Fallback(fallback) => fallback.stream(input, config).await,
                }
            }
        }
    }
}

/// A runnable wrapper that enforces an absolute deadline on the inner runnable.
///
/// Unlike `RunnableTimeout` which uses a relative duration from the time of
/// invocation, `RunnableDeadline` uses an absolute `Instant`. This is useful
/// when multiple operations share a common deadline.
///
/// # Example
/// ```ignore
/// use std::time::{Duration, Instant};
/// use cognis_core::runnables::timeout::RunnableDeadline;
///
/// let deadline = Instant::now() + Duration::from_secs(10);
/// let with_deadline = RunnableDeadline::new(Arc::new(my_runnable), deadline);
/// ```
pub struct RunnableDeadline {
    /// The wrapped runnable.
    inner: Arc<dyn Runnable>,
    /// Absolute deadline.
    deadline: Instant,
}

impl RunnableDeadline {
    /// Create a new deadline wrapper with the given absolute deadline.
    pub fn new(inner: Arc<dyn Runnable>, deadline: Instant) -> Self {
        Self { inner, deadline }
    }

    /// Calculate remaining duration until the deadline. Returns zero if past.
    fn remaining(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }
}

#[async_trait]
impl Runnable for RunnableDeadline {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let remaining = self.remaining();
        if remaining.is_zero() {
            return Err(CognisError::Other(format!(
                "Deadline exceeded for {}",
                self.inner.name()
            )));
        }
        match tokio::time::timeout(remaining, self.inner.invoke(input, config)).await {
            Ok(result) => result,
            Err(_elapsed) => Err(CognisError::Other(format!(
                "Deadline exceeded for {}",
                self.inner.name()
            ))),
        }
    }

    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        let remaining = self.remaining();
        if remaining.is_zero() {
            return Err(CognisError::Other(format!(
                "Deadline exceeded for {} stream",
                self.inner.name()
            )));
        }
        match tokio::time::timeout(remaining, self.inner.stream(input, config)).await {
            Ok(Ok(inner_stream)) => {
                let deadline = self.deadline;
                let wrapped = inner_stream.take_while(move |_| {
                    let now = Instant::now();
                    async move { now < deadline }
                });
                Ok(Box::pin(wrapped))
            }
            Ok(Err(e)) => Err(e),
            Err(_elapsed) => Err(CognisError::Other(format!(
                "Deadline exceeded for {} stream",
                self.inner.name()
            ))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::lambda::RunnableLambda;
    use futures::StreamExt;
    use serde_json::json;

    /// Helper: fast runnable that returns input doubled.
    fn fast_double() -> RunnableLambda {
        RunnableLambda::new("fast_double", |v: Value| async move {
            let n = v.as_i64().unwrap();
            Ok(json!(n * 2))
        })
    }

    /// Helper: slow runnable that sleeps for `delay_ms` then returns input.
    fn slow_identity(delay_ms: u64) -> RunnableLambda {
        RunnableLambda::new("slow_identity", move |v: Value| async move {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            Ok(v)
        })
    }

    /// Helper: slow runnable that sleeps then returns input doubled.
    fn slow_double(delay_ms: u64) -> RunnableLambda {
        RunnableLambda::new("slow_double", move |v: Value| async move {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            let n = v.as_i64().unwrap();
            Ok(json!(n * 2))
        })
    }

    /// Helper: runnable that always errors.
    fn error_runnable() -> RunnableLambda {
        RunnableLambda::new("error", |_v: Value| async move {
            Err(CognisError::Other("inner error".to_string()))
        })
    }

    // ==================== TimeoutConfig tests ====================

    #[test]
    fn test_timeout_config_defaults_to_error_behavior() {
        let config = TimeoutConfig::new(Duration::from_secs(5));
        assert_eq!(config.duration, Duration::from_secs(5));
        assert!(matches!(config.on_timeout, TimeoutBehavior::Error));
    }

    #[test]
    fn test_timeout_config_with_default_value() {
        let config =
            TimeoutConfig::new(Duration::from_secs(1)).with_default_value(json!("fallback"));
        assert!(matches!(config.on_timeout, TimeoutBehavior::Default(_)));
        if let TimeoutBehavior::Default(v) = &config.on_timeout {
            assert_eq!(v, &json!("fallback"));
        }
    }

    #[test]
    fn test_timeout_config_with_fallback() {
        let fb = Arc::new(fast_double()) as Arc<dyn Runnable>;
        let config = TimeoutConfig::new(Duration::from_secs(1)).with_fallback(fb);
        assert!(matches!(config.on_timeout, TimeoutBehavior::Fallback(_)));
    }

    // ==================== RunnableTimeout invoke tests ====================

    #[tokio::test]
    async fn test_timeout_fast_operation_succeeds() {
        let inner = fast_double();
        let config = TimeoutConfig::new(Duration::from_secs(1));
        let timed = RunnableTimeout::new(Arc::new(inner), config);

        let result = timed.invoke(json!(5), None).await.unwrap();
        assert_eq!(result, json!(10));
    }

    #[tokio::test]
    async fn test_timeout_slow_operation_errors() {
        let inner = slow_identity(200);
        let config = TimeoutConfig::new(Duration::from_millis(50));
        let timed = RunnableTimeout::new(Arc::new(inner), config);

        let result = timed.invoke(json!(1), None).await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Timeout"));
    }

    #[tokio::test]
    async fn test_timeout_slow_operation_returns_default() {
        let inner = slow_identity(200);
        let config =
            TimeoutConfig::new(Duration::from_millis(50)).with_default_value(json!("default"));
        let timed = RunnableTimeout::new(Arc::new(inner), config);

        let result = timed.invoke(json!(1), None).await.unwrap();
        assert_eq!(result, json!("default"));
    }

    #[tokio::test]
    async fn test_timeout_slow_operation_uses_fallback() {
        let inner = slow_identity(200);
        let fallback = Arc::new(fast_double()) as Arc<dyn Runnable>;
        let config = TimeoutConfig::new(Duration::from_millis(50)).with_fallback(fallback);
        let timed = RunnableTimeout::new(Arc::new(inner), config);

        let result = timed.invoke(json!(7), None).await.unwrap();
        assert_eq!(result, json!(14)); // fallback doubles the input
    }

    #[tokio::test]
    async fn test_timeout_name_delegates() {
        let inner = fast_double();
        let config = TimeoutConfig::new(Duration::from_secs(1));
        let timed = RunnableTimeout::new(Arc::new(inner), config);
        assert_eq!(timed.name(), "fast_double");
    }

    #[tokio::test]
    async fn test_timeout_inner_error_propagated() {
        let inner = error_runnable();
        let config = TimeoutConfig::new(Duration::from_secs(1));
        let timed = RunnableTimeout::new(Arc::new(inner), config);

        let result = timed.invoke(json!(1), None).await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("inner error"));
    }

    #[tokio::test]
    async fn test_timeout_just_within_limit() {
        // Operation takes 30ms, timeout is 100ms -- should succeed.
        let inner = slow_double(30);
        let config = TimeoutConfig::new(Duration::from_millis(100));
        let timed = RunnableTimeout::new(Arc::new(inner), config);

        let result = timed.invoke(json!(3), None).await.unwrap();
        assert_eq!(result, json!(6));
    }

    #[tokio::test]
    async fn test_timeout_completes_quickly_when_fast() {
        let inner = fast_double();
        let config = TimeoutConfig::new(Duration::from_secs(10));
        let timed = RunnableTimeout::new(Arc::new(inner), config);

        let start = Instant::now();
        let _result = timed.invoke(json!(1), None).await.unwrap();
        let elapsed = start.elapsed();

        // Should complete well under the 10s timeout.
        assert!(
            elapsed < Duration::from_millis(100),
            "Fast operation should complete quickly, took {:?}",
            elapsed
        );
    }

    // ==================== RunnableTimeout stream tests ====================

    #[tokio::test]
    async fn test_timeout_stream_fast_succeeds() {
        let inner = fast_double();
        let config = TimeoutConfig::new(Duration::from_secs(1));
        let timed = RunnableTimeout::new(Arc::new(inner), config);

        let mut stream = timed.stream(json!(4), None).await.unwrap();
        let item = stream.next().await.unwrap().unwrap();
        assert_eq!(item, json!(8));
    }

    #[tokio::test]
    async fn test_timeout_stream_slow_errors() {
        let inner = slow_identity(200);
        let config = TimeoutConfig::new(Duration::from_millis(50));
        let timed = RunnableTimeout::new(Arc::new(inner), config);

        // The stream creation itself should timeout.
        let result = timed.stream(json!(1), None).await;
        assert!(result.is_err());
    }

    // ==================== RunnableDeadline tests ====================

    #[tokio::test]
    async fn test_deadline_fast_operation_succeeds() {
        let inner = fast_double();
        let deadline = Instant::now() + Duration::from_secs(1);
        let dl = RunnableDeadline::new(Arc::new(inner), deadline);

        let result = dl.invoke(json!(5), None).await.unwrap();
        assert_eq!(result, json!(10));
    }

    #[tokio::test]
    async fn test_deadline_slow_operation_errors() {
        let inner = slow_identity(200);
        let deadline = Instant::now() + Duration::from_millis(50);
        let dl = RunnableDeadline::new(Arc::new(inner), deadline);

        let result = dl.invoke(json!(1), None).await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Deadline exceeded"));
    }

    #[tokio::test]
    async fn test_deadline_already_past() {
        let inner = fast_double();
        // Deadline in the past.
        let deadline = Instant::now() - Duration::from_secs(1);
        let dl = RunnableDeadline::new(Arc::new(inner), deadline);

        let result = dl.invoke(json!(5), None).await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("Deadline exceeded"));
    }

    #[tokio::test]
    async fn test_deadline_name_delegates() {
        let inner = fast_double();
        let deadline = Instant::now() + Duration::from_secs(1);
        let dl = RunnableDeadline::new(Arc::new(inner), deadline);
        assert_eq!(dl.name(), "fast_double");
    }

    #[tokio::test]
    async fn test_deadline_stream_succeeds() {
        let inner = fast_double();
        let deadline = Instant::now() + Duration::from_secs(1);
        let dl = RunnableDeadline::new(Arc::new(inner), deadline);

        let mut stream = dl.stream(json!(6), None).await.unwrap();
        let item = stream.next().await.unwrap().unwrap();
        assert_eq!(item, json!(12));
    }

    #[tokio::test]
    async fn test_deadline_stream_past_deadline_errors() {
        let inner = fast_double();
        let deadline = Instant::now() - Duration::from_secs(1);
        let dl = RunnableDeadline::new(Arc::new(inner), deadline);

        let result = dl.stream(json!(1), None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_deadline_shared_across_operations() {
        // Two operations sharing a common deadline of 150ms.
        // Each takes 50ms, so both should succeed.
        let deadline = Instant::now() + Duration::from_millis(150);

        let dl1 = RunnableDeadline::new(Arc::new(slow_double(50)), deadline);
        let dl2 = RunnableDeadline::new(Arc::new(slow_double(50)), deadline);

        let r1 = dl1.invoke(json!(2), None).await.unwrap();
        let r2 = dl2.invoke(json!(3), None).await.unwrap();
        assert_eq!(r1, json!(4));
        assert_eq!(r2, json!(6));
    }

    // ==================== Extension method tests ====================

    #[tokio::test]
    async fn test_ext_with_timeout() {
        use crate::runnables::ext::RunnableExt;

        let timed = fast_double().with_timeout(Duration::from_secs(1));
        let result = timed.invoke(json!(10), None).await.unwrap();
        assert_eq!(result, json!(20));
    }

    #[tokio::test]
    async fn test_ext_with_timeout_config() {
        use crate::runnables::ext::RunnableExt;

        let config =
            TimeoutConfig::new(Duration::from_millis(50)).with_default_value(json!("timed_out"));
        let timed = slow_identity(200).with_timeout_config(config);
        let result = timed.invoke(json!(1), None).await.unwrap();
        assert_eq!(result, json!("timed_out"));
    }
}
