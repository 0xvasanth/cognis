//! Model retry middleware — automatic retry on model failures.
//!
//! Mirrors Python `langchain.agents.middleware.model_retry`.

use async_trait::async_trait;

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::messages::Message;

use super::retry::{should_retry, OnFailure, RetryCondition, RetryConfig};
use super::types::{
    AgentMiddleware, AsyncModelHandler, ModelCallResult, ModelRequest, ModelResponse,
};

/// Middleware that retries model calls on failure with exponential backoff.
#[derive(Default)]
pub struct ModelRetryMiddleware {
    config: RetryConfig,
}

impl ModelRetryMiddleware {
    pub fn new(config: RetryConfig) -> Self {
        Self { config }
    }

    /// Create with default config and specified max retries.
    pub fn with_max_retries(max_retries: usize) -> Self {
        Self {
            config: RetryConfig::new(max_retries),
        }
    }

    /// Set the retry condition.
    pub fn with_retry_on(mut self, retry_on: RetryCondition) -> Self {
        self.config.retry_on = retry_on;
        self
    }

    /// Set the on_failure strategy.
    pub fn with_on_failure(mut self, on_failure: OnFailure) -> Self {
        self.config.on_failure = on_failure;
        self
    }
}

#[async_trait]
impl AgentMiddleware for ModelRetryMiddleware {
    fn name(&self) -> &str {
        "ModelRetryMiddleware"
    }

    async fn wrap_model_call(
        &self,
        request: &ModelRequest,
        handler: &AsyncModelHandler,
    ) -> Result<ModelCallResult> {
        let mut last_error: Option<RustChainError> = None;

        for attempt in 0..=self.config.max_retries {
            match handler(request).await {
                Ok(response) => return Ok(ModelCallResult::Response(response)),
                Err(e) => {
                    if !should_retry(&e, &self.config.retry_on)
                        || attempt == self.config.max_retries
                    {
                        last_error = Some(e);
                        break;
                    }
                    last_error = Some(e);
                    let delay = self.config.calculate_delay(attempt);
                    tokio::time::sleep(delay).await;
                }
            }
        }

        let error = last_error.unwrap_or_else(|| RustChainError::Other("Unknown error".into()));

        match &self.config.on_failure {
            OnFailure::Error => Err(error),
            OnFailure::Continue => {
                let error_msg = Message::ai(format!(
                    "Model call failed after {} retries: {}",
                    self.config.max_retries, error
                ));
                Ok(ModelCallResult::Response(ModelResponse::new(vec![
                    error_msg,
                ])))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_retry_default() {
        let mw = ModelRetryMiddleware::default();
        assert_eq!(mw.name(), "ModelRetryMiddleware");
        assert_eq!(mw.config.max_retries, 2);
        assert!(matches!(mw.config.on_failure, OnFailure::Continue));
        assert!(matches!(mw.config.retry_on, RetryCondition::AnyError));
    }

    #[test]
    fn test_model_retry_with_max_retries() {
        let mw = ModelRetryMiddleware::with_max_retries(5);
        assert_eq!(mw.config.max_retries, 5);
    }

    #[test]
    fn test_model_retry_with_retry_on() {
        let mw = ModelRetryMiddleware::default()
            .with_retry_on(RetryCondition::ErrorContains(vec!["timeout".into()]));
        assert!(matches!(
            mw.config.retry_on,
            RetryCondition::ErrorContains(_)
        ));
    }

    #[test]
    fn test_model_retry_with_on_failure() {
        let mw = ModelRetryMiddleware::default().with_on_failure(OnFailure::Error);
        assert!(matches!(mw.config.on_failure, OnFailure::Error));
    }
}
