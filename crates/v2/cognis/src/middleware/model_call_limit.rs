//! Cap the number of LLM calls per `PipelinedClient` instance.
//!
//! Crosses the chain limit: the middleware short-circuits with a
//! [`CognisError::Configuration`] once `max` calls have run.
//!
//! Customization:
//! - [`ModelCallLimit::with_message`] — override the rejection message.
//! - [`ModelCallLimit::with_callback`] — fire a closure on the rejection
//!   so callers can record metrics or trigger fallbacks externally.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{CognisError, Result};
use cognis2_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

type RejectCallback = Arc<dyn Fn(u64) + Send + Sync>;

/// Hard cap on chat-completion calls.
pub struct ModelCallLimit {
    max: u64,
    counter: AtomicU64,
    message: String,
    on_reject: Option<RejectCallback>,
}

impl ModelCallLimit {
    /// Build with a maximum.
    pub fn new(max: u64) -> Self {
        Self {
            max,
            counter: AtomicU64::new(0),
            message: format!("model call limit reached ({max})"),
            on_reject: None,
        }
    }

    /// Override the rejection error message.
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = msg.into();
        self
    }

    /// Register a callback fired on rejection. Receives the cap value.
    pub fn with_callback<F>(mut self, f: F) -> Self
    where
        F: Fn(u64) + Send + Sync + 'static,
    {
        self.on_reject = Some(Arc::new(f));
        self
    }

    /// Current call count.
    pub fn count(&self) -> u64 {
        self.counter.load(Ordering::Relaxed)
    }

    /// Reset the counter.
    pub fn reset(&self) {
        self.counter.store(0, Ordering::Relaxed);
    }
}

#[async_trait]
impl Middleware for ModelCallLimit {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let n = self.counter.fetch_add(1, Ordering::Relaxed) + 1;
        if n > self.max {
            if let Some(cb) = &self.on_reject {
                cb(self.max);
            }
            return Err(CognisError::Configuration(self.message.clone()));
        }
        next.invoke(ctx).await
    }
    fn name(&self) -> &str {
        "ModelCallLimit"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::tests_util::{ok_resp, FixedNext};

    #[tokio::test]
    async fn first_n_calls_succeed_then_reject() {
        let mw = ModelCallLimit::new(2);
        let next: Arc<dyn Next> = Arc::new(FixedNext(ok_resp("ok")));
        let ctx = || MiddlewareCtx::new(vec![], vec![], Default::default());
        assert!(mw.call(ctx(), next.clone()).await.is_ok());
        assert!(mw.call(ctx(), next.clone()).await.is_ok());
        let err = mw.call(ctx(), next.clone()).await.unwrap_err();
        assert!(matches!(err, CognisError::Configuration(_)));
    }

    #[tokio::test]
    async fn callback_fires_on_reject() {
        use std::sync::atomic::AtomicUsize;
        let cap_seen = Arc::new(AtomicUsize::new(0));
        let cs = cap_seen.clone();
        let mw = ModelCallLimit::new(0).with_callback(move |max| {
            cs.store(max as usize, Ordering::SeqCst);
        });
        let next: Arc<dyn Next> = Arc::new(FixedNext(ok_resp("ok")));
        let _ = mw
            .call(MiddlewareCtx::new(vec![], vec![], Default::default()), next)
            .await;
        assert_eq!(cap_seen.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn reset_zeroes_counter() {
        let mw = ModelCallLimit::new(1);
        let next: Arc<dyn Next> = Arc::new(FixedNext(ok_resp("ok")));
        let ctx = || MiddlewareCtx::new(vec![], vec![], Default::default());
        let _ = mw.call(ctx(), next.clone()).await;
        let _ = mw.call(ctx(), next.clone()).await; // rejected
        mw.reset();
        assert!(mw.call(ctx(), next.clone()).await.is_ok());
    }
}
