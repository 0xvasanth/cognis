//! Retry wrapper with exponential backoff.

use std::marker::PhantomData;
use std::time::Duration;

use async_trait::async_trait;

use crate::runnable::{Runnable, RunnableConfig};
use crate::{CognisError, Result};

/// Policy controlling retry timing.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum total attempts (including the first). `1` means no retries.
    pub max_attempts: u32,
    /// Initial delay before the first retry.
    pub initial_delay: Duration,
    /// Multiplier applied to the delay after every failed attempt.
    pub backoff_multiplier: f64,
    /// Cap for the per-attempt delay.
    pub max_delay: Duration,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            backoff_multiplier: 2.0,
            max_delay: Duration::from_secs(30),
        }
    }
}

impl RetryPolicy {
    /// Build a policy with the given max attempts and otherwise default settings.
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            ..Default::default()
        }
    }
    /// Override the initial delay.
    pub fn with_initial_delay(mut self, d: Duration) -> Self {
        self.initial_delay = d;
        self
    }
    /// Override the backoff multiplier.
    pub fn with_backoff(mut self, factor: f64) -> Self {
        self.backoff_multiplier = factor;
        self
    }
    /// Override the per-attempt delay cap.
    pub fn with_max_delay(mut self, d: Duration) -> Self {
        self.max_delay = d;
        self
    }
}

/// Retries `inner.invoke` on retryable errors with exponential backoff.
///
/// Retryability is decided by [`CognisError::is_retryable`]. An error's
/// own `retry_delay()` (e.g. a `RateLimited`'s `Retry-After`) takes
/// precedence over the policy's computed delay for that attempt.
pub struct Retry<R, I, O> {
    inner: R,
    policy: RetryPolicy,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<R, I, O> Retry<R, I, O>
where
    R: Runnable<I, O>,
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    /// Wrap a runnable with the given retry policy.
    pub fn new(inner: R, policy: RetryPolicy) -> Self {
        Self {
            inner,
            policy,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<R, I, O> Runnable<I, O> for Retry<R, I, O>
where
    R: Runnable<I, O>,
    I: Clone + Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, config: RunnableConfig) -> Result<O> {
        let mut delay = self.policy.initial_delay;
        let mut last_err: Option<CognisError> = None;
        for attempt in 0..self.policy.max_attempts {
            match self.inner.invoke(input.clone(), config.clone()).await {
                Ok(v) => return Ok(v),
                Err(e) if !e.is_retryable() => return Err(e),
                Err(e) => {
                    let suggested = e.retry_delay().unwrap_or(delay);
                    last_err = Some(e);
                    if attempt + 1 >= self.policy.max_attempts {
                        break;
                    }
                    let sleep_for = suggested.min(self.policy.max_delay);
                    tokio::time::sleep(sleep_for).await;
                    delay = Duration::from_secs_f64(
                        (delay.as_secs_f64() * self.policy.backoff_multiplier)
                            .min(self.policy.max_delay.as_secs_f64()),
                    );
                }
            }
        }
        Err(last_err.unwrap_or_else(|| {
            CognisError::Internal("retry exhausted with no error captured".into())
        }))
    }
    fn name(&self) -> &str {
        "Retry"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Arc;

    struct FlakyTwice {
        attempts: Arc<AtomicU32>,
    }

    #[async_trait]
    impl Runnable<u32, u32> for FlakyTwice {
        async fn invoke(&self, input: u32, _: RunnableConfig) -> Result<u32> {
            let n = self.attempts.fetch_add(1, Ordering::SeqCst);
            if n < 2 {
                Err(CognisError::Network {
                    status_code: Some(503),
                    message: "boom".into(),
                })
            } else {
                Ok(input)
            }
        }
    }

    struct AlwaysAuth;

    #[async_trait]
    impl Runnable<u32, u32> for AlwaysAuth {
        async fn invoke(&self, _: u32, _: RunnableConfig) -> Result<u32> {
            Err(CognisError::AuthenticationFailed("bad key".into()))
        }
    }

    #[tokio::test]
    async fn retries_until_success() {
        let attempts = Arc::new(AtomicU32::new(0));
        let r = Retry::new(
            FlakyTwice {
                attempts: attempts.clone(),
            },
            RetryPolicy::new(5).with_initial_delay(Duration::from_millis(1)),
        );
        let out = r.invoke(7, RunnableConfig::default()).await.unwrap();
        assert_eq!(out, 7);
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn non_retryable_short_circuits() {
        let r = Retry::new(
            AlwaysAuth,
            RetryPolicy::new(5).with_initial_delay(Duration::from_millis(1)),
        );
        let err = r.invoke(0, RunnableConfig::default()).await.unwrap_err();
        assert!(matches!(err, CognisError::AuthenticationFailed(_)));
    }
}
