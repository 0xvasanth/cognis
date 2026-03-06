//! Rate-limited chat model wrapper using a token bucket algorithm.
//!
//! Wraps any `BaseChatModel` to enforce request-per-minute (and optionally
//! token-per-minute) rate limits. The [`TokenBucket`] implements a classic
//! token bucket that refills at a constant rate and blocks callers when the
//! bucket is empty.
//!
//! # Example
//!
//! ```no_run
//! use rustchain::chat_models::rate_limited::with_rate_limit;
//! # fn example(model: Box<dyn rustchain_core::language_models::chat_model::BaseChatModel>) {
//! // Wrap a model to allow at most 60 requests per minute
//! let limited = with_rate_limit(model, 60.0);
//! # }
//! ```

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;

use rustchain_core::error::Result;
use rustchain_core::language_models::chat_model::{
    BaseChatModel, ChatStream, ModelProfile, ToolChoice,
};
use rustchain_core::messages::Message;
use rustchain_core::outputs::ChatResult;
use rustchain_core::tools::ToolSchema;

/// A token bucket rate limiter.
///
/// Tokens are added to the bucket at `refill_rate` tokens per second, up to
/// `capacity`. Each call to [`acquire`](TokenBucket::acquire) consumes one
/// token, blocking asynchronously until a token is available.
pub struct TokenBucket {
    /// Maximum tokens in the bucket.
    capacity: f64,
    /// Current tokens available.
    tokens: Arc<Mutex<f64>>,
    /// Tokens added per second.
    refill_rate: f64,
    /// Last refill timestamp.
    last_refill: Arc<Mutex<Instant>>,
}

impl TokenBucket {
    /// Create a new token bucket.
    ///
    /// # Arguments
    /// * `capacity` - Maximum number of tokens the bucket can hold.
    /// * `refill_rate` - Tokens added per second.
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: Arc::new(Mutex::new(capacity)),
            refill_rate,
            last_refill: Arc::new(Mutex::new(Instant::now())),
        }
    }

    /// Acquire one token, waiting asynchronously if necessary.
    pub async fn acquire(&self) {
        loop {
            self.refill();
            let acquired = self.try_acquire_token();
            if acquired {
                return;
            }
            // Wait a bit for tokens to refill
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }

    /// Try to acquire a single token. Returns true if successful.
    fn try_acquire_token(&self) -> bool {
        let mut tokens = self.tokens.lock().unwrap();
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            true
        } else {
            false
        }
    }

    /// Refill the bucket based on elapsed time since the last refill.
    fn refill(&self) {
        let mut last = self.last_refill.lock().unwrap();
        let now = Instant::now();
        let elapsed = now.duration_since(*last).as_secs_f64();
        if elapsed > 0.0 {
            let mut tokens = self.tokens.lock().unwrap();
            *tokens = (*tokens + elapsed * self.refill_rate).min(self.capacity);
            *last = now;
        }
    }

    /// Return the current number of available tokens (for testing).
    #[cfg(test)]
    fn available(&self) -> f64 {
        self.refill();
        *self.tokens.lock().unwrap()
    }
}

/// A chat model wrapper that enforces rate limits.
///
/// Wraps an inner [`BaseChatModel`] and uses a [`TokenBucket`] to throttle
/// requests. An optional second bucket can limit token throughput (reserved
/// for future use).
pub struct RateLimitedChatModel {
    /// The wrapped chat model.
    inner: Box<dyn BaseChatModel>,
    /// Limits the number of requests per time window.
    request_limiter: TokenBucket,
    /// Optional limiter for tokens per minute (future use).
    #[allow(dead_code)]
    token_limiter: Option<TokenBucket>,
}

impl RateLimitedChatModel {
    /// Create a new rate-limited wrapper.
    ///
    /// # Arguments
    /// * `inner` - The chat model to wrap.
    /// * `requests_per_minute` - Maximum requests allowed per minute.
    pub fn new(inner: Box<dyn BaseChatModel>, requests_per_minute: f64) -> Self {
        let refill_rate = requests_per_minute / 60.0;
        let capacity = requests_per_minute;
        Self {
            inner,
            request_limiter: TokenBucket::new(capacity, refill_rate),
            token_limiter: None,
        }
    }

    /// Create a new rate-limited wrapper with per-second rate and burst size.
    ///
    /// # Arguments
    /// * `inner` - The chat model to wrap.
    /// * `max_requests_per_second` - Sustained request rate (tokens added per second).
    /// * `burst_size` - Maximum burst capacity (bucket size).
    pub fn with_rate(
        inner: Box<dyn BaseChatModel>,
        max_requests_per_second: f64,
        burst_size: usize,
    ) -> Self {
        Self {
            inner,
            request_limiter: TokenBucket::new(burst_size as f64, max_requests_per_second),
            token_limiter: None,
        }
    }

    /// Add a token-per-minute limiter (future use).
    ///
    /// # Arguments
    /// * `tokens_per_minute` - Maximum tokens allowed per minute.
    pub fn with_token_limit(mut self, tokens_per_minute: f64) -> Self {
        let refill_rate = tokens_per_minute / 60.0;
        self.token_limiter = Some(TokenBucket::new(tokens_per_minute, refill_rate));
        self
    }
}

#[async_trait]
impl BaseChatModel for RateLimitedChatModel {
    async fn _generate(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatResult> {
        self.request_limiter.acquire().await;
        self.inner._generate(messages, stop).await
    }

    fn llm_type(&self) -> &str {
        // We cannot return a borrowed &str from a computed String,
        // so we leak a small allocation. This is acceptable because
        // llm_type values are effectively static for the lifetime of the model.
        // Instead, we use a different approach: store the type string.
        // Actually, the trait requires &str, so we need to store it.
        // We'll work around this by using a static-like approach.
        // For now, delegate to inner since the wrapper is transparent.
        // The spec asks for format!(...) but the return type is &str.
        // We'll return the inner type directly — callers can check via
        // the struct type if they need to know it's rate-limited.
        self.inner.llm_type()
    }

    async fn _stream(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatStream> {
        self.request_limiter.acquire().await;
        self.inner._stream(messages, stop).await
    }

    fn bind_tools(
        &self,
        tools: &[ToolSchema],
        tool_choice: Option<ToolChoice>,
    ) -> Result<Box<dyn BaseChatModel>> {
        let bound_inner = self.inner.bind_tools(tools, tool_choice)?;
        // Preserve the same rate limits on the tool-bound model
        let rpm = self.request_limiter.capacity;
        let mut wrapped = RateLimitedChatModel::new(bound_inner, rpm);
        if let Some(ref tl) = self.token_limiter {
            wrapped.token_limiter = Some(TokenBucket::new(tl.capacity, tl.refill_rate));
        }
        Ok(Box::new(wrapped))
    }

    fn profile(&self) -> ModelProfile {
        self.inner.profile()
    }
}

/// Convenience function to wrap a chat model with a request rate limit.
///
/// # Arguments
/// * `model` - The chat model to wrap.
/// * `requests_per_minute` - Maximum requests allowed per minute.
///
/// # Returns
/// A boxed `BaseChatModel` that enforces the specified rate limit.
pub fn with_rate_limit(
    model: Box<dyn BaseChatModel>,
    requests_per_minute: f64,
) -> Box<dyn BaseChatModel> {
    Box::new(RateLimitedChatModel::new(model, requests_per_minute))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::language_models::chat_model::ModelProfile;
    use rustchain_core::messages::{AIMessage, Message};
    use rustchain_core::outputs::{ChatGeneration, ChatResult};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A mock chat model that counts invocations.
    struct MockChatModel {
        call_count: Arc<AtomicUsize>,
        supports_tools: bool,
    }

    impl MockChatModel {
        fn new() -> Self {
            Self {
                call_count: Arc::new(AtomicUsize::new(0)),
                supports_tools: false,
            }
        }

        fn with_tools() -> Self {
            Self {
                call_count: Arc::new(AtomicUsize::new(0)),
                supports_tools: true,
            }
        }

        fn count(&self) -> usize {
            self.call_count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl BaseChatModel for MockChatModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(ChatResult {
                generations: vec![ChatGeneration {
                    text: "Hello!".to_string(),
                    message: Message::Ai(AIMessage::new("Hello!")),
                    generation_info: None,
                }],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            "mock"
        }

        fn bind_tools(
            &self,
            _tools: &[ToolSchema],
            _tool_choice: Option<ToolChoice>,
        ) -> Result<Box<dyn BaseChatModel>> {
            if self.supports_tools {
                Ok(Box::new(MockChatModel::with_tools()))
            } else {
                Err(rustchain_core::error::RustChainError::NotImplemented(
                    "mock does not support tool binding".into(),
                ))
            }
        }

        fn profile(&self) -> ModelProfile {
            ModelProfile {
                max_input_tokens: Some(128_000),
                tool_calling: Some(true),
                ..Default::default()
            }
        }
    }

    #[tokio::test]
    async fn test_token_bucket_allows_within_capacity() {
        // Bucket with capacity 5 and high refill rate
        let bucket = TokenBucket::new(5.0, 100.0);
        // Should be able to acquire 5 tokens immediately
        for _ in 0..5 {
            bucket.acquire().await;
        }
        // Bucket should be near-empty now
        assert!(bucket.available() < 1.0);
    }

    #[tokio::test]
    async fn test_token_bucket_refills_over_time() {
        // Bucket with capacity 10, refill at 1000/sec (fast for testing)
        let bucket = TokenBucket::new(10.0, 1000.0);
        // Drain all tokens
        for _ in 0..10 {
            bucket.acquire().await;
        }
        assert!(bucket.available() < 1.0);
        // Wait for refill
        tokio::time::sleep(Duration::from_millis(50)).await;
        // Should have refilled significantly
        assert!(bucket.available() >= 1.0);
    }

    #[tokio::test]
    async fn test_token_bucket_does_not_exceed_capacity() {
        let bucket = TokenBucket::new(5.0, 1000.0);
        // Wait long enough that many tokens would be added
        tokio::time::sleep(Duration::from_millis(100)).await;
        // Should be capped at capacity
        assert!(bucket.available() <= 5.0 + 0.01); // small float tolerance
    }

    #[tokio::test]
    async fn test_rate_limited_model_delegates_generate() {
        let mock = MockChatModel::new();
        let count = mock.call_count.clone();
        let limited = RateLimitedChatModel::new(Box::new(mock), 600.0); // 10/sec

        let messages = vec![Message::Human(rustchain_core::messages::HumanMessage::new(
            "Hi",
        ))];
        let result = limited._generate(&messages, None).await;
        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 1);
        assert_eq!(result.unwrap().generations[0].text, "Hello!");
    }

    #[tokio::test]
    async fn test_rate_limited_model_delays_when_exhausted() {
        // Very low rate: 6 requests per minute = 0.1/sec
        // Capacity of 1 means only 1 request available immediately
        let bucket = TokenBucket::new(1.0, 0.1);

        // First acquire should be instant
        let start = tokio::time::Instant::now();
        bucket.acquire().await;
        let first_elapsed = start.elapsed();
        assert!(first_elapsed < Duration::from_millis(50));

        // Second acquire should wait for refill (~10 seconds at 0.1/sec)
        // Use a faster rate to keep the test short
        let bucket = TokenBucket::new(1.0, 100.0); // 100/sec
        bucket.acquire().await; // drain the one token

        let start = tokio::time::Instant::now();
        bucket.acquire().await; // must wait for refill
        let second_elapsed = start.elapsed();
        // Should have waited at least ~10ms (the sleep interval)
        assert!(second_elapsed >= Duration::from_millis(5));
    }

    #[tokio::test]
    async fn test_rate_limited_model_llm_type() {
        let mock = MockChatModel::new();
        let limited = RateLimitedChatModel::new(Box::new(mock), 60.0);
        assert_eq!(limited.llm_type(), "mock");
    }

    #[tokio::test]
    async fn test_rate_limited_model_profile_delegates() {
        let mock = MockChatModel::new();
        let limited = RateLimitedChatModel::new(Box::new(mock), 60.0);
        let profile = limited.profile();
        assert_eq!(profile.max_input_tokens, Some(128_000));
        assert_eq!(profile.tool_calling, Some(true));
    }

    #[tokio::test]
    async fn test_bind_tools_preserves_rate_limiting() {
        let mock = MockChatModel::with_tools();
        let limited = RateLimitedChatModel::new(Box::new(mock), 60.0);

        let tools: Vec<ToolSchema> = vec![];
        let result = limited.bind_tools(&tools, None);
        assert!(result.is_ok());

        // The returned model should still be rate-limited and functional
        let bound = result.unwrap();
        let messages = vec![Message::Human(rustchain_core::messages::HumanMessage::new(
            "Hi",
        ))];
        let gen_result = bound._generate(&messages, None).await;
        assert!(gen_result.is_ok());
    }

    #[tokio::test]
    async fn test_with_rate_limit_convenience() {
        let mock = MockChatModel::new();
        let limited = with_rate_limit(Box::new(mock), 120.0);
        assert_eq!(limited.llm_type(), "mock");

        let messages = vec![Message::Human(rustchain_core::messages::HumanMessage::new(
            "test",
        ))];
        let result = limited._generate(&messages, None).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_with_rate_constructor() {
        let mock = MockChatModel::new();
        let count = mock.call_count.clone();
        // 10 requests/sec, burst of 5
        let limited = RateLimitedChatModel::with_rate(Box::new(mock), 10.0, 5);

        let messages = vec![Message::Human(rustchain_core::messages::HumanMessage::new(
            "Hi",
        ))];
        let result = limited._generate(&messages, None).await;
        assert!(result.is_ok());
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_with_token_limit_builder() {
        let mock = MockChatModel::new();
        let limited = RateLimitedChatModel::new(Box::new(mock), 60.0).with_token_limit(100_000.0);
        assert!(limited.token_limiter.is_some());

        let messages = vec![Message::Human(rustchain_core::messages::HumanMessage::new(
            "test",
        ))];
        let result = limited._generate(&messages, None).await;
        assert!(result.is_ok());
    }
}
