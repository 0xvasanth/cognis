//! Recovery middleware — substitute a graceful response for LLM errors so
//! the agent loop can decide what to do with it instead of bailing.
//!
//! Differs from [`super::ModelFallback`]:
//! - `ModelFallback` retries the call against a different `Client`.
//! - `Recovery` short-circuits the call with a synthetic [`ChatResponse`]
//!   so the agent's `ThinkNode` keeps running — useful when no fallback
//!   is available and you'd rather degrade than crash.

use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{CognisError, Message, Result};
use cognis2_llm::chat::{ChatResponse, Usage};

use super::{Middleware, MiddlewareCtx, Next};

/// Decides how to handle an LLM-call error.
#[async_trait]
pub trait RecoveryStrategy: Send + Sync {
    /// Examine the error and return a recovery message, or `None` to
    /// re-raise the original error.
    async fn recover(&self, err: &CognisError) -> Option<String>;
}

/// Fixed-text strategy. Always recovers with the same string.
pub struct FixedRecovery {
    /// Synthetic message body produced on every recoverable error.
    pub message: String,
}

impl FixedRecovery {
    /// Build with a fixed message.
    pub fn new(msg: impl Into<String>) -> Self {
        Self {
            message: msg.into(),
        }
    }
}

#[async_trait]
impl RecoveryStrategy for FixedRecovery {
    async fn recover(&self, _: &CognisError) -> Option<String> {
        Some(self.message.clone())
    }
}

/// Closure-backed strategy.
pub struct FnRecovery<F: Fn(&CognisError) -> Option<String> + Send + Sync>(pub F);

#[async_trait]
impl<F: Fn(&CognisError) -> Option<String> + Send + Sync> RecoveryStrategy for FnRecovery<F> {
    async fn recover(&self, err: &CognisError) -> Option<String> {
        (self.0)(err)
    }
}

/// Middleware that intercepts LLM-call errors and substitutes a synthetic
/// response.
pub struct Recovery {
    strategy: Arc<dyn RecoveryStrategy>,
    /// If true, only recover for retryable errors (rate-limit, network,
    /// timeout, transient provider). If false, recover for ALL errors
    /// except `Cancelled`.
    only_retryable: bool,
}

impl Recovery {
    /// Build with a recovery strategy.
    pub fn new(strategy: Arc<dyn RecoveryStrategy>) -> Self {
        Self {
            strategy,
            only_retryable: true,
        }
    }

    /// Recover from any error, not just retryable ones (still excludes `Cancelled`).
    pub fn for_all_errors(mut self) -> Self {
        self.only_retryable = false;
        self
    }
}

#[async_trait]
impl Middleware for Recovery {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        match next.invoke(ctx).await {
            Ok(r) => Ok(r),
            Err(e) => {
                // Never swallow cancellations.
                if matches!(e, CognisError::Cancelled) {
                    return Err(e);
                }
                if self.only_retryable && !e.is_retryable() {
                    return Err(e);
                }
                match self.strategy.recover(&e).await {
                    Some(text) => Ok(ChatResponse {
                        message: Message::ai(text),
                        usage: Some(Usage::default()),
                        finish_reason: "recovery".into(),
                        model: "recovery".into(),
                    }),
                    None => Err(e),
                }
            }
        }
    }

    fn name(&self) -> &str {
        "Recovery"
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests_util::*;
    use super::*;
    use crate::middleware::MiddlewarePipeline;

    use cognis2_llm::chat::ChatOptions;
    use cognis2_llm::Client;

    #[tokio::test]
    async fn substitutes_message_on_retryable_error() {
        let provider = make_flaky_provider(|_| {
            Err(CognisError::Network {
                status_code: Some(503),
                message: "boom".into(),
            })
        });
        let pipe = MiddlewarePipeline::new()
            .push(Recovery::new(Arc::new(FixedRecovery::new(
                "the model is unavailable; please try again later",
            ))))
            .build(Client::new(provider));
        let r = pipe
            .invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert!(r.message.content().contains("unavailable"));
        assert_eq!(r.finish_reason, "recovery");
    }

    #[tokio::test]
    async fn skips_non_retryable_when_only_retryable() {
        let provider =
            make_flaky_provider(|_| Err(CognisError::AuthenticationFailed("nope".into())));
        let pipe = MiddlewarePipeline::new()
            .push(Recovery::new(Arc::new(FixedRecovery::new("recovered"))))
            .build(Client::new(provider));
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

    #[tokio::test]
    async fn for_all_errors_recovers_from_non_retryable() {
        let provider =
            make_flaky_provider(|_| Err(CognisError::AuthenticationFailed("nope".into())));
        let pipe = MiddlewarePipeline::new()
            .push(Recovery::new(Arc::new(FixedRecovery::new("recovered"))).for_all_errors())
            .build(Client::new(provider));
        let r = pipe
            .invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "recovered");
    }

    #[tokio::test]
    async fn never_swallows_cancellation() {
        let provider = make_flaky_provider(|_| Err(CognisError::Cancelled));
        let pipe = MiddlewarePipeline::new()
            .push(Recovery::new(Arc::new(FixedRecovery::new("recovered"))).for_all_errors())
            .build(Client::new(provider));
        let err = pipe
            .invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap_err();
        assert!(matches!(err, CognisError::Cancelled));
    }

    #[tokio::test]
    async fn fn_recovery_strategy_can_examine_error() {
        let provider = make_flaky_provider(|_| {
            Err(CognisError::RateLimited {
                retry_after_ms: 5000,
            })
        });
        let strat = FnRecovery(|err: &CognisError| match err {
            CognisError::RateLimited { retry_after_ms } => {
                Some(format!("rate limited; retry in {retry_after_ms}ms"))
            }
            _ => None,
        });
        let pipe = MiddlewarePipeline::new()
            .push(Recovery::new(Arc::new(strat)))
            .build(Client::new(provider));
        let r = pipe
            .invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        assert!(r.message.content().contains("5000ms"));
    }
}
