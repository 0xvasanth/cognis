//! Timeout wrapper — bound a runnable's execution time.

use std::marker::PhantomData;
use std::time::Duration;

use async_trait::async_trait;

use crate::runnable::{Runnable, RunnableConfig};
use crate::{CognisError, Result};

/// Bounds the inner runnable's execution time. Errors with
/// `CognisError::Timeout` if the deadline is exceeded.
pub struct Timeout<R, I, O> {
    inner: R,
    duration: Duration,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<R, I, O> Timeout<R, I, O>
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Send + 'static,
{
    /// Wrap a runnable with the given timeout.
    pub fn new(inner: R, duration: Duration) -> Self {
        Self {
            inner,
            duration,
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<R, I, O> Runnable<I, O> for Timeout<R, I, O>
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, input: I, config: RunnableConfig) -> Result<O> {
        let op_name = self.inner.name().to_string();
        let dur = self.duration;
        match tokio::time::timeout(dur, self.inner.invoke(input, config)).await {
            Ok(r) => r,
            Err(_) => Err(CognisError::Timeout {
                operation: op_name,
                timeout_ms: dur.as_millis() as u64,
            }),
        }
    }
    fn name(&self) -> &str {
        "Timeout"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Slow;

    #[async_trait]
    impl Runnable<u32, u32> for Slow {
        async fn invoke(&self, input: u32, _: RunnableConfig) -> Result<u32> {
            tokio::time::sleep(Duration::from_millis(50)).await;
            Ok(input)
        }
    }

    #[tokio::test]
    async fn errors_when_exceeded() {
        let t = Timeout::new(Slow, Duration::from_millis(5));
        let err = t.invoke(1, RunnableConfig::default()).await.unwrap_err();
        assert!(matches!(err, CognisError::Timeout { .. }));
    }

    #[tokio::test]
    async fn succeeds_within_budget() {
        let t = Timeout::new(Slow, Duration::from_millis(500));
        assert_eq!(t.invoke(7, RunnableConfig::default()).await.unwrap(), 7);
    }
}
