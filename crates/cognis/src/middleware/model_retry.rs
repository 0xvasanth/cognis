//! Retry the LLM call with exponential backoff on retryable errors.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use cognis_core::Result;
use cognis_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

/// Retry the underlying LLM call when it fails with a retryable
/// `CognisError` (rate-limited, network, timeout, transient provider error).
pub struct ModelRetry {
    max_attempts: u32,
    initial_delay: Duration,
    multiplier: f64,
    max_delay: Duration,
}

impl ModelRetry {
    /// Build with default backoff: 100ms initial, 2x multiplier, 30s cap.
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts,
            initial_delay: Duration::from_millis(100),
            multiplier: 2.0,
            max_delay: Duration::from_secs(30),
        }
    }

    /// Override the initial delay.
    pub fn with_initial_delay(mut self, d: Duration) -> Self {
        self.initial_delay = d;
        self
    }

    /// Override the backoff multiplier.
    pub fn with_multiplier(mut self, m: f64) -> Self {
        self.multiplier = m;
        self
    }

    /// Override the cap on per-attempt delay.
    pub fn with_max_delay(mut self, d: Duration) -> Self {
        self.max_delay = d;
        self
    }
}

#[async_trait]
impl Middleware for ModelRetry {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let mut delay = self.initial_delay;
        let mut last_err = None;
        for attempt in 0..self.max_attempts {
            match next.invoke(ctx.clone()).await {
                Ok(r) => return Ok(r),
                Err(e) if !e.is_retryable() => return Err(e),
                Err(e) => {
                    let suggested = e.retry_delay().unwrap_or(delay);
                    last_err = Some(e);
                    if attempt + 1 >= self.max_attempts {
                        break;
                    }
                    let sleep_for = suggested.min(self.max_delay);
                    tokio::time::sleep(sleep_for).await;
                    delay = Duration::from_secs_f64(
                        (delay.as_secs_f64() * self.multiplier).min(self.max_delay.as_secs_f64()),
                    );
                }
            }
        }
        Err(last_err.unwrap_or_else(|| {
            cognis_core::CognisError::Internal("retry exhausted with no error".into())
        }))
    }

    fn name(&self) -> &str {
        "ModelRetry"
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests_util::*;
    use super::*;
    use crate::middleware::MiddlewarePipeline;

    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use cognis_core::{CognisError, Message};
    use cognis_llm::chat::ChatOptions;
    use cognis_llm::Client;

    #[tokio::test]
    async fn retries_until_success() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let attempts_for_provider = attempts.clone();
        let provider = make_flaky_provider(move |i| {
            attempts_for_provider.store(i + 1, Ordering::SeqCst);
            if i < 2 {
                Err(CognisError::Network {
                    status_code: Some(503),
                    message: "boom".into(),
                })
            } else {
                Ok("ok".into())
            }
        });
        let client = Client::new(provider);
        let pipe = MiddlewarePipeline::new()
            .push(ModelRetry::new(5).with_initial_delay(Duration::from_millis(1)))
            .build(client);
        let r = pipe
            .invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "ok");
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn non_retryable_propagates() {
        let provider =
            make_flaky_provider(|_| Err(CognisError::AuthenticationFailed("nope".into())));
        let client = Client::new(provider);
        let pipe = MiddlewarePipeline::new()
            .push(ModelRetry::new(5).with_initial_delay(Duration::from_millis(1)))
            .build(client);
        let err = pipe
            .invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, CognisError::AuthenticationFailed(_)));
    }
}
