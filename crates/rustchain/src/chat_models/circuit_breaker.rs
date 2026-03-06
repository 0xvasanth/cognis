//! Circuit breaker pattern for chat model calls.
//!
//! Provides a [`CircuitBreaker`] that tracks failures and prevents cascading
//! errors by short-circuiting requests when a failure threshold is exceeded.
//! Also provides [`CircuitBreakerChatModel`], a wrapper that applies circuit
//! breaker logic to any [`BaseChatModel`].

use std::future::Future;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::Mutex;

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::{
    BaseChatModel, ChatStream, ModelProfile, ToolChoice,
};
use rustchain_core::messages::Message;
use rustchain_core::outputs::ChatResult;
use rustchain_core::tools::ToolSchema;

/// The operational state of a circuit breaker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation: requests pass through.
    Closed,
    /// Tripped: requests immediately fail without calling the inner service.
    Open,
    /// Testing recovery: one request is allowed through to test if the service
    /// has recovered.
    HalfOpen,
}

/// A circuit breaker that tracks consecutive failures and prevents cascading
/// errors by short-circuiting requests when a failure threshold is exceeded.
///
/// ## State Machine
///
/// ```text
/// ┌────────┐  failure >= threshold   ┌──────┐
/// │ Closed ├────────────────────────►│ Open │
/// └────┬───┘                         └──┬───┘
///      │                                │
///      │  success                       │ reset_timeout elapsed
///      │                                ▼
///      │                          ┌──────────┐
///      └──────────────────────────┤ HalfOpen │
///              success            └────┬─────┘
///                                     │ failure
///                                     ▼
///                                  ┌──────┐
///                                  │ Open │
///                                  └──────┘
/// ```
pub struct CircuitBreaker {
    state: Arc<Mutex<CircuitState>>,
    failure_threshold: u32,
    reset_timeout: Duration,
    consecutive_failures: Arc<AtomicU32>,
    last_failure_time: Arc<Mutex<Option<Instant>>>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    ///
    /// # Arguments
    /// * `failure_threshold` - Number of consecutive failures before the circuit opens.
    /// * `reset_timeout` - Duration to wait in the open state before transitioning to half-open.
    pub fn new(failure_threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            state: Arc::new(Mutex::new(CircuitState::Closed)),
            failure_threshold,
            reset_timeout,
            consecutive_failures: Arc::new(AtomicU32::new(0)),
            last_failure_time: Arc::new(Mutex::new(None)),
        }
    }

    /// Returns the current state of the circuit breaker.
    pub async fn state(&self) -> CircuitState {
        let mut state = self.state.lock().await;
        // Check if we should transition from Open to HalfOpen
        if *state == CircuitState::Open {
            let last_failure = self.last_failure_time.lock().await;
            if let Some(t) = *last_failure {
                if t.elapsed() >= self.reset_timeout {
                    *state = CircuitState::HalfOpen;
                }
            }
        }
        *state
    }

    /// Reset the circuit breaker to the closed state.
    pub async fn reset(&self) {
        let mut state = self.state.lock().await;
        *state = CircuitState::Closed;
        self.consecutive_failures.store(0, Ordering::SeqCst);
        let mut last = self.last_failure_time.lock().await;
        *last = None;
    }

    /// Execute a future through the circuit breaker.
    ///
    /// - If the circuit is **closed**, the future is executed normally.
    /// - If the circuit is **open** and the reset timeout has not elapsed,
    ///   an error is returned immediately.
    /// - If the circuit is **half-open**, one request is allowed through.
    ///   On success the circuit closes; on failure it re-opens.
    pub async fn call<F, Fut, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        let current_state = self.state().await;

        match current_state {
            CircuitState::Open => {
                return Err(RustChainError::Other(
                    "Circuit breaker is open: too many consecutive failures".into(),
                ));
            }
            CircuitState::HalfOpen | CircuitState::Closed => {
                match f().await {
                    Ok(result) => {
                        // Success: reset failures, close circuit
                        self.consecutive_failures.store(0, Ordering::SeqCst);
                        let mut state = self.state.lock().await;
                        *state = CircuitState::Closed;
                        Ok(result)
                    }
                    Err(e) => {
                        let failures = self.consecutive_failures.fetch_add(1, Ordering::SeqCst) + 1;
                        let mut last = self.last_failure_time.lock().await;
                        *last = Some(Instant::now());

                        if current_state == CircuitState::HalfOpen
                            || failures >= self.failure_threshold
                        {
                            let mut state = self.state.lock().await;
                            *state = CircuitState::Open;
                        }
                        Err(e)
                    }
                }
            }
        }
    }
}

/// A chat model wrapper that applies circuit breaker logic.
///
/// When the circuit is open, calls to `_generate` and `_stream` fail
/// immediately with a descriptive error instead of hitting the downstream
/// service.
pub struct CircuitBreakerChatModel {
    inner: Box<dyn BaseChatModel>,
    breaker: CircuitBreaker,
}

impl CircuitBreakerChatModel {
    /// Wrap a chat model with a circuit breaker.
    ///
    /// # Arguments
    /// * `inner` - The chat model to wrap.
    /// * `failure_threshold` - Consecutive failures before the circuit opens (default: 5).
    /// * `reset_timeout` - Time to wait before testing recovery (default: 60s).
    pub fn new(
        inner: Box<dyn BaseChatModel>,
        failure_threshold: u32,
        reset_timeout: Duration,
    ) -> Self {
        Self {
            inner,
            breaker: CircuitBreaker::new(failure_threshold, reset_timeout),
        }
    }

    /// Returns the current circuit state.
    pub async fn circuit_state(&self) -> CircuitState {
        self.breaker.state().await
    }

    /// Manually reset the circuit breaker.
    pub async fn reset(&self) {
        self.breaker.reset().await;
    }
}

#[async_trait]
impl BaseChatModel for CircuitBreakerChatModel {
    async fn _generate(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatResult> {
        // We need to capture references for the closure
        let inner = &self.inner;
        self.breaker
            .call(|| async move { inner._generate(messages, stop).await })
            .await
    }

    fn llm_type(&self) -> &str {
        self.inner.llm_type()
    }

    async fn _stream(&self, messages: &[Message], stop: Option<&[String]>) -> Result<ChatStream> {
        let inner = &self.inner;
        self.breaker
            .call(|| async move { inner._stream(messages, stop).await })
            .await
    }

    fn bind_tools(
        &self,
        tools: &[ToolSchema],
        tool_choice: Option<ToolChoice>,
    ) -> Result<Box<dyn BaseChatModel>> {
        self.inner.bind_tools(tools, tool_choice)
    }

    fn profile(&self) -> ModelProfile {
        self.inner.profile()
    }

    fn get_num_tokens_from_messages(&self, messages: &[Message]) -> usize {
        self.inner.get_num_tokens_from_messages(messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::messages::{AIMessage, HumanMessage};
    use rustchain_core::outputs::ChatGeneration;
    use std::sync::atomic::AtomicU32;

    /// A mock chat model that fails a configurable number of times then succeeds.
    struct MockChatModel {
        fail_count: u32,
        attempts: AtomicU32,
    }

    impl MockChatModel {
        fn always_fails() -> Self {
            Self {
                fail_count: u32::MAX,
                attempts: AtomicU32::new(0),
            }
        }

        fn fails_n_times(n: u32) -> Self {
            Self {
                fail_count: n,
                attempts: AtomicU32::new(0),
            }
        }
    }

    #[async_trait]
    impl BaseChatModel for MockChatModel {
        async fn _generate(
            &self,
            _messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            let attempt = self.attempts.fetch_add(1, Ordering::SeqCst);
            if attempt < self.fail_count {
                Err(RustChainError::HttpError {
                    status: 500,
                    body: "Internal Server Error".into(),
                })
            } else {
                Ok(ChatResult {
                    generations: vec![ChatGeneration {
                        text: "OK".into(),
                        message: Message::Ai(AIMessage::new("OK")),
                        generation_info: None,
                    }],
                    llm_output: None,
                })
            }
        }

        fn llm_type(&self) -> &str {
            "mock"
        }
    }

    #[tokio::test]
    async fn test_circuit_breaker_closed_on_success() {
        let model = CircuitBreakerChatModel::new(
            Box::new(MockChatModel::fails_n_times(0)),
            3,
            Duration::from_secs(60),
        );

        let msgs = vec![Message::Human(HumanMessage::new("hi"))];
        let result = model._generate(&msgs, None).await;
        assert!(result.is_ok());
        assert_eq!(model.circuit_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_opens_after_threshold() {
        let model = CircuitBreakerChatModel::new(
            Box::new(MockChatModel::always_fails()),
            3,
            Duration::from_secs(60),
        );

        let msgs = vec![Message::Human(HumanMessage::new("hi"))];

        // Fail 3 times to trip the breaker
        for _ in 0..3 {
            let _ = model._generate(&msgs, None).await;
        }

        assert_eq!(model.circuit_state().await, CircuitState::Open);

        // Next call should fail immediately with circuit breaker error
        let result = model._generate(&msgs, None).await;
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("Circuit breaker is open"),
            "Expected circuit breaker error, got: {}",
            err
        );
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_to_closed() {
        let model = CircuitBreakerChatModel::new(
            Box::new(MockChatModel::fails_n_times(3)), // fails 3 times, then succeeds
            3,
            Duration::from_millis(50), // short timeout for test
        );

        let msgs = vec![Message::Human(HumanMessage::new("hi"))];

        // Trip the breaker
        for _ in 0..3 {
            let _ = model._generate(&msgs, None).await;
        }
        assert_eq!(model.circuit_state().await, CircuitState::Open);

        // Wait for reset timeout
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Should be half-open now
        assert_eq!(model.circuit_state().await, CircuitState::HalfOpen);

        // Successful call should close the circuit
        let result = model._generate(&msgs, None).await;
        assert!(result.is_ok());
        assert_eq!(model.circuit_state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn test_circuit_breaker_half_open_to_open() {
        let model = CircuitBreakerChatModel::new(
            Box::new(MockChatModel::always_fails()),
            3,
            Duration::from_millis(50),
        );

        let msgs = vec![Message::Human(HumanMessage::new("hi"))];

        // Trip the breaker
        for _ in 0..3 {
            let _ = model._generate(&msgs, None).await;
        }
        assert_eq!(model.circuit_state().await, CircuitState::Open);

        // Wait for reset timeout
        tokio::time::sleep(Duration::from_millis(60)).await;
        assert_eq!(model.circuit_state().await, CircuitState::HalfOpen);

        // Failure in half-open should re-open the circuit
        let result = model._generate(&msgs, None).await;
        assert!(result.is_err());
        assert_eq!(model.circuit_state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn test_circuit_breaker_reset() {
        let model = CircuitBreakerChatModel::new(
            Box::new(MockChatModel::always_fails()),
            2,
            Duration::from_secs(60),
        );

        let msgs = vec![Message::Human(HumanMessage::new("hi"))];

        // Trip the breaker
        for _ in 0..2 {
            let _ = model._generate(&msgs, None).await;
        }
        assert_eq!(model.circuit_state().await, CircuitState::Open);

        // Manual reset
        model.reset().await;
        assert_eq!(model.circuit_state().await, CircuitState::Closed);
    }
}
