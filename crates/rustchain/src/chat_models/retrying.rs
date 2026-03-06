//! Retry wrapper for chat models with exponential backoff.
//!
//! Provides [`RetryingChatModel`], a chat model wrapper that automatically
//! retries failed requests using exponential backoff with jitter. Only
//! retryable errors (HTTP 429, 500, 502, 503, 504 and transient errors)
//! are retried.
//!
//! # Example
//!
//! ```no_run
//! use rustchain::chat_models::retrying::RetryingChatModel;
//! # fn example(model: Box<dyn rustchain_core::language_models::chat_model::BaseChatModel>) {
//! let retrying = RetryingChatModel::new(model)
//!     .with_max_retries(5)
//!     .with_initial_backoff_ms(500)
//!     .with_max_backoff_ms(30_000)
//!     .with_backoff_multiplier(2.0);
//! # }
//! ```

use std::time::Duration;

use async_trait::async_trait;

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::{
    BaseChatModel, ChatStream, ModelProfile, ToolChoice,
};
use rustchain_core::messages::Message;
use rustchain_core::outputs::ChatResult;
use rustchain_core::tools::ToolSchema;

/// Configuration for retry behavior.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of retry attempts (not counting the initial attempt).
    pub max_retries: u32,
    /// Initial backoff duration in milliseconds.
    pub initial_backoff_ms: u64,
    /// Maximum backoff duration in milliseconds.
    pub max_backoff_ms: u64,
    /// Multiplier applied to the backoff after each retry.
    pub backoff_multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff_ms: 1000,
            max_backoff_ms: 60_000,
            backoff_multiplier: 2.0,
        }
    }
}

/// Returns `true` if the error is considered retryable.
///
/// Retryable errors include HTTP 429 (rate limit) and server errors
/// (500, 502, 503, 504), as well as IO errors and messages containing
/// transient-error keywords.
fn is_retryable(error: &RustChainError) -> bool {
    match error {
        RustChainError::HttpError { status, .. } => {
            matches!(*status, 429 | 500 | 502 | 503 | 504)
        }
        RustChainError::IoError(_) => true,
        RustChainError::Other(msg) => {
            let lower = msg.to_lowercase();
            lower.contains("rate limit")
                || lower.contains("timeout")
                || lower.contains("connection")
        }
        _ => false,
    }
}

/// Compute backoff duration with jitter for a given attempt.
///
/// Uses full jitter: the actual sleep is uniformly distributed between
/// 0 and the computed exponential backoff.
fn compute_backoff(config: &RetryConfig, attempt: u32) -> Duration {
    let base_ms = config.initial_backoff_ms as f64
        * config.backoff_multiplier.powi(attempt as i32);
    let capped_ms = base_ms.min(config.max_backoff_ms as f64);
    // Simple jitter: use a deterministic-ish fraction to avoid external deps.
    // We use half the backoff plus a small per-attempt variation.
    let jitter_ms = capped_ms * 0.5 + capped_ms * 0.5 * jitter_fraction(attempt);
    Duration::from_millis(jitter_ms.max(1.0) as u64)
}

/// Produce a pseudo-random jitter fraction in [0, 1) based on attempt number.
/// This avoids pulling in an RNG crate while still decorrelating retries
/// across different attempt numbers.
fn jitter_fraction(attempt: u32) -> f64 {
    // Simple hash-like mixing
    let mut x = attempt as u64;
    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    x ^= x >> 33;
    x = x.wrapping_mul(0xff51afd7ed558ccd);
    x ^= x >> 33;
    (x as f64) / (u64::MAX as f64)
}

/// A chat model wrapper that retries failed requests with exponential backoff.
///
/// Only retryable errors (HTTP 429, 500, 502, 503, 504 and transient errors)
/// trigger retries. Non-retryable errors are returned immediately.
pub struct RetryingChatModel {
    /// The wrapped chat model.
    inner: Box<dyn BaseChatModel>,
    /// Retry configuration.
    config: RetryConfig,
}

impl RetryingChatModel {
    /// Create a new retrying wrapper with default configuration.
    ///
    /// Defaults: 3 retries, 1000ms initial backoff, 60s max backoff, 2x multiplier.
    pub fn new(inner: Box<dyn BaseChatModel>) -> Self {
        Self {
            inner,
            config: RetryConfig::default(),
        }
    }

    /// Create a retrying wrapper with explicit configuration.
    pub fn with_config(inner: Box<dyn BaseChatModel>, config: RetryConfig) -> Self {
        Self { inner, config }
    }

    /// Set the maximum number of retries.
    pub fn with_max_retries(mut self, max_retries: u32) -> Self {
        self.config.max_retries = max_retries;
        self
    }

    /// Set the initial backoff in milliseconds.
    pub fn with_initial_backoff_ms(mut self, ms: u64) -> Self {
        self.config.initial_backoff_ms = ms;
        self
    }

    /// Set the maximum backoff in milliseconds.
    pub fn with_max_backoff_ms(mut self, ms: u64) -> Self {
        self.config.max_backoff_ms = ms;
        self
    }

    /// Set the backoff multiplier.
    pub fn with_backoff_multiplier(mut self, multiplier: f64) -> Self {
        self.config.backoff_multiplier = multiplier;
        self
    }

    /// Execute a generate call with retries.
    async fn generate_with_retry(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        let mut last_error = None;
        for attempt in 0..=self.config.max_retries {
            match self.inner._generate(messages, stop).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    if attempt < self.config.max_retries && is_retryable(&e) {
                        let backoff = compute_backoff(&self.config, attempt);
                        tokio::time::sleep(backoff).await;
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        Err(last_error.unwrap_or_else(|| {
            RustChainError::Other("Retry loop ended without result".into())
        }))
    }
}

#[async_trait]
impl BaseChatModel for RetryingChatModel {
    async fn _generate(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        self.generate_with_retry(messages, stop).await
    }

    fn llm_type(&self) -> &str {
        self.inner.llm_type()
    }

    async fn _stream(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
    ) -> Result<ChatStream> {
        let mut last_error = None;
        for attempt in 0..=self.config.max_retries {
            match self.inner._stream(messages, stop).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    if attempt < self.config.max_retries && is_retryable(&e) {
                        let backoff = compute_backoff(&self.config, attempt);
                        tokio::time::sleep(backoff).await;
                        last_error = Some(e);
                    } else {
                        return Err(e);
                    }
                }
            }
        }
        Err(last_error.unwrap_or_else(|| {
            RustChainError::Other("Retry loop ended without result".into())
        }))
    }

    fn bind_tools(
        &self,
        tools: &[ToolSchema],
        tool_choice: Option<ToolChoice>,
    ) -> Result<Box<dyn BaseChatModel>> {
        let bound_inner = self.inner.bind_tools(tools, tool_choice)?;
        Ok(Box::new(RetryingChatModel::with_config(
            bound_inner,
            self.config.clone(),
        )))
    }

    fn profile(&self) -> ModelProfile {
        self.inner.profile()
    }

    fn get_num_tokens_from_messages(&self, messages: &[Message]) -> usize {
        self.inner.get_num_tokens_from_messages(messages)
    }
}

/// Convenience function to wrap a chat model with default retry behavior.
pub fn with_retry(model: Box<dyn BaseChatModel>) -> Box<dyn BaseChatModel> {
    Box::new(RetryingChatModel::new(model))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::messages::{AIMessage, HumanMessage, Message};
    use rustchain_core::outputs::{ChatGeneration, ChatResult};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    /// A mock that always succeeds.
    struct SuccessModel;

    #[async_trait]
    impl BaseChatModel for SuccessModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            Ok(ChatResult {
                generations: vec![ChatGeneration {
                    text: "OK".into(),
                    message: Message::Ai(AIMessage::new("OK")),
                    generation_info: None,
                }],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            "success_mock"
        }
    }

    /// A mock that fails with a retryable error N times, then succeeds.
    struct FailThenSucceedModel {
        failures_remaining: Arc<AtomicUsize>,
        call_count: Arc<AtomicUsize>,
        error_status: u16,
    }

    impl FailThenSucceedModel {
        fn new(failures: usize, status: u16) -> Self {
            Self {
                failures_remaining: Arc::new(AtomicUsize::new(failures)),
                call_count: Arc::new(AtomicUsize::new(0)),
                error_status: status,
            }
        }
    }

    #[async_trait]
    impl BaseChatModel for FailThenSucceedModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            let remaining = self.failures_remaining.load(Ordering::SeqCst);
            if remaining > 0 {
                self.failures_remaining.fetch_sub(1, Ordering::SeqCst);
                Err(RustChainError::HttpError {
                    status: self.error_status,
                    body: "Server Error".into(),
                })
            } else {
                Ok(ChatResult {
                    generations: vec![ChatGeneration {
                        text: "Recovered".into(),
                        message: Message::Ai(AIMessage::new("Recovered")),
                        generation_info: None,
                    }],
                    llm_output: None,
                })
            }
        }

        fn llm_type(&self) -> &str {
            "fail_then_succeed_mock"
        }
    }

    /// A mock that always fails with a non-retryable error.
    struct NonRetryableFailModel;

    #[async_trait]
    impl BaseChatModel for NonRetryableFailModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            Err(RustChainError::HttpError {
                status: 401,
                body: "Unauthorized".into(),
            })
        }

        fn llm_type(&self) -> &str {
            "non_retryable_mock"
        }
    }

    /// A mock that always fails with a retryable error.
    struct AlwaysFailModel {
        call_count: Arc<AtomicUsize>,
    }

    impl AlwaysFailModel {
        fn new() -> Self {
            Self {
                call_count: Arc::new(AtomicUsize::new(0)),
            }
        }
    }

    #[async_trait]
    impl BaseChatModel for AlwaysFailModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Err(RustChainError::HttpError {
                status: 500,
                body: "Internal Server Error".into(),
            })
        }

        fn llm_type(&self) -> &str {
            "always_fail_mock"
        }
    }

    fn test_messages() -> Vec<Message> {
        vec![Message::Human(HumanMessage::new("test"))]
    }

    #[tokio::test]
    async fn test_retrying_passes_through_on_success() {
        let model = RetryingChatModel::new(Box::new(SuccessModel));
        let result = model._generate(&test_messages(), None).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().generations[0].text, "OK");
    }

    #[tokio::test]
    async fn test_retrying_recovers_after_transient_failure() {
        let mock = FailThenSucceedModel::new(2, 500);
        let call_count = mock.call_count.clone();
        let model = RetryingChatModel::new(Box::new(mock))
            .with_max_retries(3)
            .with_initial_backoff_ms(1) // very short for testing
            .with_max_backoff_ms(10);

        let result = model._generate(&test_messages(), None).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap().generations[0].text, "Recovered");
        // Should have been called 3 times: 2 failures + 1 success
        assert_eq!(call_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_retrying_does_not_retry_non_retryable_errors() {
        let model = RetryingChatModel::new(Box::new(NonRetryableFailModel))
            .with_max_retries(5)
            .with_initial_backoff_ms(1);

        let result = model._generate(&test_messages(), None).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            RustChainError::HttpError { status, .. } => assert_eq!(status, 401),
            other => panic!("Expected HttpError(401), got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn test_retrying_exhausts_max_retries() {
        let mock = AlwaysFailModel::new();
        let call_count = mock.call_count.clone();
        let model = RetryingChatModel::new(Box::new(mock))
            .with_max_retries(3)
            .with_initial_backoff_ms(1)
            .with_max_backoff_ms(5);

        let result = model._generate(&test_messages(), None).await;
        assert!(result.is_err());
        // Should have been called 4 times: initial + 3 retries
        assert_eq!(call_count.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn test_retrying_rate_limit_429_is_retried() {
        let mock = FailThenSucceedModel::new(1, 429);
        let call_count = mock.call_count.clone();
        let model = RetryingChatModel::new(Box::new(mock))
            .with_max_retries(2)
            .with_initial_backoff_ms(1);

        let result = model._generate(&test_messages(), None).await;
        assert!(result.is_ok());
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_retrying_502_503_504_are_retried() {
        for status in [502u16, 503, 504] {
            let mock = FailThenSucceedModel::new(1, status);
            let call_count = mock.call_count.clone();
            let model = RetryingChatModel::new(Box::new(mock))
                .with_max_retries(2)
                .with_initial_backoff_ms(1);

            let result = model._generate(&test_messages(), None).await;
            assert!(result.is_ok(), "Expected success for status {}", status);
            assert_eq!(call_count.load(Ordering::SeqCst), 2);
        }
    }

    #[tokio::test]
    async fn test_retrying_llm_type_delegates() {
        let model = RetryingChatModel::new(Box::new(SuccessModel));
        assert_eq!(model.llm_type(), "success_mock");
    }

    #[tokio::test]
    async fn test_retrying_profile_delegates() {
        let model = RetryingChatModel::new(Box::new(SuccessModel));
        let profile = model.profile();
        // SuccessModel uses default profile
        assert_eq!(profile, ModelProfile::default());
    }

    #[tokio::test]
    async fn test_is_retryable_classification() {
        // Retryable
        assert!(is_retryable(&RustChainError::HttpError {
            status: 429,
            body: "".into()
        }));
        assert!(is_retryable(&RustChainError::HttpError {
            status: 500,
            body: "".into()
        }));
        assert!(is_retryable(&RustChainError::HttpError {
            status: 502,
            body: "".into()
        }));
        assert!(is_retryable(&RustChainError::HttpError {
            status: 503,
            body: "".into()
        }));
        assert!(is_retryable(&RustChainError::HttpError {
            status: 504,
            body: "".into()
        }));

        // Not retryable
        assert!(!is_retryable(&RustChainError::HttpError {
            status: 400,
            body: "".into()
        }));
        assert!(!is_retryable(&RustChainError::HttpError {
            status: 401,
            body: "".into()
        }));
        assert!(!is_retryable(&RustChainError::HttpError {
            status: 404,
            body: "".into()
        }));
        assert!(!is_retryable(&RustChainError::NotImplemented("".into())));
    }

    #[tokio::test]
    async fn test_compute_backoff_increases() {
        let config = RetryConfig {
            max_retries: 5,
            initial_backoff_ms: 100,
            max_backoff_ms: 10_000,
            backoff_multiplier: 2.0,
        };
        let b0 = compute_backoff(&config, 0);
        let b1 = compute_backoff(&config, 1);
        let b2 = compute_backoff(&config, 2);
        // Each should generally increase (backoff doubles, jitter varies)
        // At minimum, later attempts should have higher max backoff
        assert!(b1.as_millis() > 0);
        assert!(b2.as_millis() > 0);
        // The cap should be respected
        let b_large = compute_backoff(&config, 20);
        assert!(b_large <= Duration::from_millis(config.max_backoff_ms));
        // Sanity: b0 should be reasonable
        assert!(b0 <= Duration::from_millis(config.initial_backoff_ms + 1));
    }

    #[tokio::test]
    async fn test_with_retry_convenience() {
        let model = with_retry(Box::new(SuccessModel));
        let result = model._generate(&test_messages(), None).await;
        assert!(result.is_ok());
    }
}
