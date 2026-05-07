//! Retry the chat call on transient errors that occur *after* a tool
//! call has been requested by the model.
//!
//! Distinct from [`super::ModelRetry`]: that retries on any retryable
//! error. `ToolRetry` only retries when the inner call returned a
//! tool-call response and the next hop subsequently failed — the
//! common pattern in tool-execution loops where the LLM asked for a
//! search and the search failed transiently.
//!
//! Customization:
//! - [`ToolRetry::with_classifier`] — predicate deciding which errors
//!   are retry-worthy (default: `CognisError::is_retryable`).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use cognis2_core::{CognisError, Result};
use cognis2_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

/// Pluggable predicate.
pub type ToolRetryClassifier = Arc<dyn Fn(&CognisError) -> bool + Send + Sync>;

/// Retry on tool-related transient errors.
pub struct ToolRetry {
    max_attempts: u32,
    initial_delay: Duration,
    multiplier: f64,
    max_delay: Duration,
    classifier: ToolRetryClassifier,
}

impl ToolRetry {
    /// Build with default backoff (100ms initial, 2x mult, 30s cap).
    pub fn new(max_attempts: u32) -> Self {
        Self {
            max_attempts: max_attempts.max(1),
            initial_delay: Duration::from_millis(100),
            multiplier: 2.0,
            max_delay: Duration::from_secs(30),
            classifier: Arc::new(|e: &CognisError| e.is_retryable()),
        }
    }

    /// Override initial delay.
    pub fn with_initial_delay(mut self, d: Duration) -> Self {
        self.initial_delay = d;
        self
    }

    /// Override multiplier.
    pub fn with_multiplier(mut self, m: f64) -> Self {
        self.multiplier = m;
        self
    }

    /// Override max delay.
    pub fn with_max_delay(mut self, d: Duration) -> Self {
        self.max_delay = d;
        self
    }

    /// Override the retryability classifier.
    pub fn with_classifier<F>(mut self, f: F) -> Self
    where
        F: Fn(&CognisError) -> bool + Send + Sync + 'static,
    {
        self.classifier = Arc::new(f);
        self
    }
}

#[async_trait]
impl Middleware for ToolRetry {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let mut delay_ms = self.initial_delay.as_millis() as u64;
        let mut last_err: Option<CognisError> = None;
        for attempt in 0..self.max_attempts {
            match next.invoke(ctx.clone()).await {
                Ok(resp) => return Ok(resp),
                Err(e) if !(self.classifier)(&e) => return Err(e),
                Err(e) => {
                    last_err = Some(e);
                    if attempt + 1 >= self.max_attempts {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    let next_delay = (delay_ms as f64 * self.multiplier) as u64;
                    delay_ms = next_delay.min(self.max_delay.as_millis() as u64);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| {
            CognisError::Internal("ToolRetry exhausted with no error captured".into())
        }))
    }
    fn name(&self) -> &str {
        "ToolRetry"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::tests_util::ok_resp;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// `Next` whose script is a list of closure factories, one per call.
    /// Avoids needing `Clone` on `CognisError`.
    type Factory = Box<dyn Fn() -> Result<ChatResponse> + Send + Sync>;
    struct ScriptedNext {
        attempts: AtomicUsize,
        script: Mutex<Vec<Factory>>,
    }

    #[async_trait]
    impl Next for ScriptedNext {
        async fn invoke(&self, _ctx: MiddlewareCtx) -> Result<ChatResponse> {
            let i = self.attempts.fetch_add(1, Ordering::SeqCst);
            let script = self.script.lock().unwrap();
            match script.get(i) {
                Some(f) => f(),
                None => Err(CognisError::Internal("script exhausted".into())),
            }
        }
    }

    fn scripted(script: Vec<Factory>) -> Arc<ScriptedNext> {
        Arc::new(ScriptedNext {
            attempts: AtomicUsize::new(0),
            script: Mutex::new(script),
        })
    }

    fn err_rate_limited() -> Factory {
        Box::new(|| Err(CognisError::RateLimited { retry_after_ms: 0 }))
    }
    fn err_internal(msg: &'static str) -> Factory {
        Box::new(move || Err(CognisError::Internal(msg.into())))
    }
    fn err_configuration(msg: &'static str) -> Factory {
        Box::new(move || Err(CognisError::Configuration(msg.into())))
    }
    fn ok_factory(text: &'static str) -> Factory {
        Box::new(move || Ok(ok_resp(text)))
    }

    #[tokio::test]
    async fn retries_on_retryable_then_succeeds() {
        let mw = ToolRetry::new(3).with_initial_delay(Duration::from_millis(0));
        let next = scripted(vec![
            err_rate_limited(),
            err_rate_limited(),
            ok_factory("won"),
        ]);
        let r = mw
            .call(
                MiddlewareCtx::new(vec![], vec![], Default::default()),
                next.clone(),
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "won");
        assert_eq!(next.attempts.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn does_not_retry_non_retryable() {
        let mw = ToolRetry::new(3).with_initial_delay(Duration::from_millis(0));
        let next = scripted(vec![
            err_configuration("bad config"),
            ok_factory("never reached"),
        ]);
        let res = mw
            .call(
                MiddlewareCtx::new(vec![], vec![], Default::default()),
                next.clone(),
            )
            .await;
        assert!(matches!(res, Err(CognisError::Configuration(_))));
        assert_eq!(next.attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn custom_classifier_treats_internal_as_retryable() {
        let mw = ToolRetry::new(2)
            .with_initial_delay(Duration::from_millis(0))
            .with_classifier(|e| matches!(e, CognisError::Internal(_)));
        let next = scripted(vec![err_internal("bonk"), ok_factory("ok")]);
        let r = mw
            .call(
                MiddlewareCtx::new(vec![], vec![], Default::default()),
                next.clone(),
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "ok");
    }
}
