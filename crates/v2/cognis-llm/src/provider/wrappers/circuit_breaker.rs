//! Circuit breaker — drops calls to an unhealthy provider until it has
//! a chance to recover.
//!
//! State transitions:
//!
//! - **Closed** (normal): every call passes through. Consecutive
//!   failures are counted; after `failure_threshold` the breaker opens.
//! - **Open**: every call is rejected immediately with a
//!   [`CognisError::Configuration`]. After `cooldown` elapses, the
//!   breaker transitions to half-open on the next call.
//! - **HalfOpen**: a single probe call is allowed through. Success
//!   closes the breaker; failure re-opens it.
//!
//! Customization:
//! - [`FailureClassifier`] — pluggable predicate that decides which
//!   errors count toward the failure budget. Defaults to "all errors
//!   count".
//! - [`RetryableClassifier`] — only retryable errors trip the breaker
//!   (network/transient errors, rate limits). Useful when you don't
//!   want a `Configuration`-class error to pop the breaker.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use async_trait::async_trait;

use cognis2_core::{CognisError, Result, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk};
use crate::provider::{LLMProvider, Provider};
use crate::tools::ToolDefinition;
use crate::Message;

/// Decides whether a given error counts as a failure for the circuit.
pub trait FailureClassifier: Send + Sync {
    /// True if `err` should increment the failure counter.
    fn is_failure(&self, err: &CognisError) -> bool;
}

/// Default classifier — every error counts.
#[derive(Debug, Default, Clone, Copy)]
pub struct AllErrorsAreFailures;

impl FailureClassifier for AllErrorsAreFailures {
    fn is_failure(&self, _err: &CognisError) -> bool {
        true
    }
}

/// Only retryable errors (network/transient/rate-limited) count toward
/// the failure budget.
#[derive(Debug, Default, Clone, Copy)]
pub struct RetryableClassifier;

impl FailureClassifier for RetryableClassifier {
    fn is_failure(&self, err: &CognisError) -> bool {
        err.is_retryable()
    }
}

/// Closure-based classifier.
impl<F> FailureClassifier for F
where
    F: Fn(&CognisError) -> bool + Send + Sync,
{
    fn is_failure(&self, err: &CognisError) -> bool {
        (self)(err)
    }
}

/// Circuit state visible via [`CircuitBreakerProvider::stats`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation; calls pass through.
    Closed,
    /// Circuit is open; calls are rejected.
    Open,
    /// One probe call permitted to test recovery.
    HalfOpen,
}

/// Snapshot of circuit-breaker counters.
#[derive(Debug, Clone, Copy)]
pub struct CircuitStats {
    /// Current state.
    pub state: CircuitState,
    /// Consecutive failures since the last success.
    pub consecutive_failures: u64,
    /// Total calls observed by the breaker.
    pub total_calls: u64,
    /// Total trips (state transitioned Closed → Open).
    pub trips: u64,
}

struct InnerState {
    state: CircuitState,
    opened_at: Option<Instant>,
}

/// Circuit-breaker wrapper.
pub struct CircuitBreakerProvider {
    inner: std::sync::Arc<dyn LLMProvider>,
    classifier: Box<dyn FailureClassifier>,
    failure_threshold: u64,
    cooldown: Duration,
    consecutive_failures: AtomicU64,
    total_calls: AtomicU64,
    trips: AtomicU64,
    state: Mutex<InnerState>,
}

impl CircuitBreakerProvider {
    /// Wrap `inner` with a circuit breaker. Defaults: threshold 5
    /// failures, cooldown 30s, classifier = "all errors are failures".
    pub fn new(inner: std::sync::Arc<dyn LLMProvider>) -> Self {
        Self {
            inner,
            classifier: Box::new(AllErrorsAreFailures),
            failure_threshold: 5,
            cooldown: Duration::from_secs(30),
            consecutive_failures: AtomicU64::new(0),
            total_calls: AtomicU64::new(0),
            trips: AtomicU64::new(0),
            state: Mutex::new(InnerState {
                state: CircuitState::Closed,
                opened_at: None,
            }),
        }
    }

    /// Override the failure threshold (consecutive failures that trip
    /// the breaker).
    pub fn with_failure_threshold(mut self, n: u64) -> Self {
        self.failure_threshold = n.max(1);
        self
    }

    /// Override the cooldown before the breaker transitions to half-open.
    pub fn with_cooldown(mut self, d: Duration) -> Self {
        self.cooldown = d;
        self
    }

    /// Plug in a custom failure classifier.
    pub fn with_classifier<C>(mut self, c: C) -> Self
    where
        C: FailureClassifier + 'static,
    {
        self.classifier = Box::new(c);
        self
    }

    /// Snapshot the breaker's counters and current state.
    pub fn stats(&self) -> CircuitStats {
        let state_lock = self.state.lock().expect("circuit state mutex poisoned");
        CircuitStats {
            state: state_lock.state,
            consecutive_failures: self.consecutive_failures.load(Ordering::Relaxed),
            total_calls: self.total_calls.load(Ordering::Relaxed),
            trips: self.trips.load(Ordering::Relaxed),
        }
    }

    /// Force-close the breaker (test/admin helper).
    pub fn reset(&self) {
        let mut s = self.state.lock().expect("circuit state mutex poisoned");
        s.state = CircuitState::Closed;
        s.opened_at = None;
        self.consecutive_failures.store(0, Ordering::Relaxed);
    }

    /// Check whether a call may proceed; transitions Open → HalfOpen on
    /// cooldown expiry. Returns the gate decision.
    fn try_acquire(&self) -> Result<()> {
        let mut s = self.state.lock().expect("circuit state mutex poisoned");
        match s.state {
            CircuitState::Closed | CircuitState::HalfOpen => Ok(()),
            CircuitState::Open => {
                let elapsed = s.opened_at.map(|t| t.elapsed()).unwrap_or(Duration::ZERO);
                if elapsed >= self.cooldown {
                    s.state = CircuitState::HalfOpen;
                    Ok(())
                } else {
                    Err(CognisError::Configuration(format!(
                        "circuit breaker open for `{}` (cooldown {}ms remaining)",
                        self.inner.name(),
                        (self.cooldown.saturating_sub(elapsed)).as_millis()
                    )))
                }
            }
        }
    }

    fn on_success(&self) {
        self.consecutive_failures.store(0, Ordering::Relaxed);
        let mut s = self.state.lock().expect("circuit state mutex poisoned");
        s.state = CircuitState::Closed;
        s.opened_at = None;
    }

    fn on_failure(&self, err: &CognisError) {
        if !self.classifier.is_failure(err) {
            return;
        }
        let n = self.consecutive_failures.fetch_add(1, Ordering::Relaxed) + 1;
        if n >= self.failure_threshold {
            let mut s = self.state.lock().expect("circuit state mutex poisoned");
            if !matches!(s.state, CircuitState::Open) {
                self.trips.fetch_add(1, Ordering::Relaxed);
            }
            s.state = CircuitState::Open;
            s.opened_at = Some(Instant::now());
        }
    }
}

#[async_trait]
impl LLMProvider for CircuitBreakerProvider {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn provider_type(&self) -> Provider {
        self.inner.provider_type()
    }

    async fn chat_completion(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        self.try_acquire()?;
        let res = self.inner.chat_completion(messages, opts).await;
        match &res {
            Ok(_) => self.on_success(),
            Err(e) => self.on_failure(e),
        }
        res
    }

    async fn chat_completion_stream(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        self.try_acquire()?;
        let res = self.inner.chat_completion_stream(messages, opts).await;
        match &res {
            Ok(_) => self.on_success(),
            Err(e) => self.on_failure(e),
        }
        res
    }

    async fn chat_completion_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        self.total_calls.fetch_add(1, Ordering::Relaxed);
        self.try_acquire()?;
        let res = self
            .inner
            .chat_completion_with_tools(messages, tools, opts)
            .await;
        match &res {
            Ok(_) => self.on_success(),
            Err(e) => self.on_failure(e),
        }
        res
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        self.inner.health_check().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    /// Test provider that lets us script success/failure per call.
    struct Scripted {
        outcomes: Mutex<Vec<bool>>, // pop_front per call: true = ok, false = err
    }

    impl Scripted {
        fn new(outcomes: Vec<bool>) -> Self {
            Self {
                outcomes: Mutex::new(outcomes.into_iter().rev().collect()),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for Scripted {
        fn name(&self) -> &str {
            "scripted"
        }
        fn provider_type(&self) -> Provider {
            Provider::OpenAI
        }
        async fn chat_completion(
            &self,
            _messages: Vec<Message>,
            _opts: ChatOptions,
        ) -> Result<ChatResponse> {
            let next = self.outcomes.lock().unwrap().pop().unwrap_or(true);
            if next {
                Ok(ChatResponse {
                    message: Message::ai("ok"),
                    usage: None,
                    finish_reason: "stop".into(),
                    model: "test".into(),
                })
            } else {
                Err(CognisError::Internal("scripted failure".into()))
            }
        }
        async fn chat_completion_stream(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unimplemented!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    #[tokio::test]
    async fn closed_state_passes_through() {
        let inner = Arc::new(Scripted::new(vec![true, true, true]));
        let cb = CircuitBreakerProvider::new(inner);
        for _ in 0..3 {
            assert!(cb
                .chat_completion(vec![], ChatOptions::default())
                .await
                .is_ok());
        }
        assert_eq!(cb.stats().state, CircuitState::Closed);
        assert_eq!(cb.stats().trips, 0);
    }

    #[tokio::test]
    async fn opens_after_threshold_failures() {
        let inner = Arc::new(Scripted::new(vec![false, false, false]));
        let cb = CircuitBreakerProvider::new(inner).with_failure_threshold(3);
        for _ in 0..3 {
            let _ = cb.chat_completion(vec![], ChatOptions::default()).await;
        }
        let stats = cb.stats();
        assert_eq!(stats.state, CircuitState::Open);
        assert_eq!(stats.trips, 1);
        // Subsequent call rejected without hitting inner.
        let err = cb
            .chat_completion(vec![], ChatOptions::default())
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("circuit breaker open"));
    }

    #[tokio::test]
    async fn half_opens_after_cooldown_and_closes_on_success() {
        let inner = Arc::new(Scripted::new(vec![false, false, true]));
        let cb = CircuitBreakerProvider::new(inner)
            .with_failure_threshold(2)
            .with_cooldown(Duration::from_millis(20));
        let _ = cb.chat_completion(vec![], ChatOptions::default()).await;
        let _ = cb.chat_completion(vec![], ChatOptions::default()).await;
        assert_eq!(cb.stats().state, CircuitState::Open);

        tokio::time::sleep(Duration::from_millis(30)).await;
        // Probe call: scripted returns Ok → breaker closes.
        assert!(cb
            .chat_completion(vec![], ChatOptions::default())
            .await
            .is_ok());
        assert_eq!(cb.stats().state, CircuitState::Closed);
    }

    #[tokio::test]
    async fn classifier_skips_non_retryable_errors() {
        let inner = Arc::new(Scripted::new(vec![false, false, false]));
        let cb = CircuitBreakerProvider::new(inner)
            .with_failure_threshold(2)
            // Internal errors are *not* retryable, so they don't count.
            .with_classifier(RetryableClassifier);
        for _ in 0..3 {
            let _ = cb.chat_completion(vec![], ChatOptions::default()).await;
        }
        assert_eq!(cb.stats().state, CircuitState::Closed);
        assert_eq!(cb.stats().trips, 0);
    }

    #[tokio::test]
    async fn reset_clears_state() {
        let inner = Arc::new(Scripted::new(vec![false, false]));
        let cb = CircuitBreakerProvider::new(inner).with_failure_threshold(2);
        let _ = cb.chat_completion(vec![], ChatOptions::default()).await;
        let _ = cb.chat_completion(vec![], ChatOptions::default()).await;
        assert_eq!(cb.stats().state, CircuitState::Open);
        cb.reset();
        assert_eq!(cb.stats().state, CircuitState::Closed);
    }

    #[tokio::test]
    async fn closure_classifier_works() {
        let inner = Arc::new(Scripted::new(vec![false, false, false]));
        let cb = CircuitBreakerProvider::new(inner)
            .with_failure_threshold(2)
            .with_classifier(|_e: &CognisError| false); // never count
        for _ in 0..3 {
            let _ = cb.chat_completion(vec![], ChatOptions::default()).await;
        }
        assert_eq!(cb.stats().state, CircuitState::Closed);
    }
}
