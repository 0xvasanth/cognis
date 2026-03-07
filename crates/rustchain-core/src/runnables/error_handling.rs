use std::fmt;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// ErrorKind
// ---------------------------------------------------------------------------

/// Classifies the nature of an error for routing decisions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    /// A transient error that may resolve on retry (e.g. network blip).
    Transient,
    /// A permanent error that will not resolve on retry (e.g. bad input).
    Permanent,
    /// The operation exceeded its time budget.
    Timeout,
    /// The upstream service is rate-limiting requests.
    RateLimit,
    /// The input failed validation before reaching the upstream service.
    Validation,
    /// The error could not be classified.
    Unknown,
}

impl ErrorKind {
    /// Returns `true` for error kinds that may succeed on a subsequent attempt.
    pub fn is_retryable(&self) -> bool {
        matches!(self, Self::Transient | Self::Timeout | Self::RateLimit)
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Transient => write!(f, "Transient"),
            Self::Permanent => write!(f, "Permanent"),
            Self::Timeout => write!(f, "Timeout"),
            Self::RateLimit => write!(f, "RateLimit"),
            Self::Validation => write!(f, "Validation"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

// ---------------------------------------------------------------------------
// ClassifiedError
// ---------------------------------------------------------------------------

/// An error enriched with classification metadata.
#[derive(Debug, Clone)]
pub struct ClassifiedError {
    /// The kind of error.
    pub kind: ErrorKind,
    /// Human-readable error message.
    pub message: String,
    /// Optional originating source / component name.
    pub source: Option<String>,
    /// When the error was recorded.
    pub timestamp: Instant,
}

impl ClassifiedError {
    /// Create a new `ClassifiedError` with the given kind and message.
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            source: None,
            timestamp: Instant::now(),
        }
    }

    /// Builder method: attach an originating source name.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Serialize the error to a JSON `Value`.
    ///
    /// Note: `timestamp` is represented as elapsed seconds since creation
    /// (relative to *now*) because `Instant` has no absolute calendar meaning.
    pub fn to_json(&self) -> Value {
        json!({
            "kind": self.kind.to_string(),
            "message": self.message,
            "source": self.source,
            "elapsed_secs": self.timestamp.elapsed().as_secs_f64(),
        })
    }
}

// ---------------------------------------------------------------------------
// ErrorClassifier trait + PatternErrorClassifier
// ---------------------------------------------------------------------------

/// Classifies a raw error string into an [`ErrorKind`].
pub trait ErrorClassifier: Send + Sync {
    /// Inspect the error message and return the appropriate kind.
    fn classify(&self, error: &str) -> ErrorKind;
}

/// Classifies errors by matching substrings in the error message.
///
/// Ships with sensible defaults for timeout, rate-limit, and validation
/// keywords. Custom patterns can be added via [`add_pattern`](Self::add_pattern).
pub struct PatternErrorClassifier {
    patterns: Vec<(String, ErrorKind)>,
}

impl PatternErrorClassifier {
    /// Create a classifier pre-loaded with common patterns.
    pub fn new() -> Self {
        let patterns = vec![
            ("timeout".to_string(), ErrorKind::Timeout),
            ("timed out".to_string(), ErrorKind::Timeout),
            ("deadline exceeded".to_string(), ErrorKind::Timeout),
            ("rate limit".to_string(), ErrorKind::RateLimit),
            ("rate_limit".to_string(), ErrorKind::RateLimit),
            ("too many requests".to_string(), ErrorKind::RateLimit),
            ("429".to_string(), ErrorKind::RateLimit),
            ("throttl".to_string(), ErrorKind::RateLimit),
            ("validation".to_string(), ErrorKind::Validation),
            ("invalid input".to_string(), ErrorKind::Validation),
            ("invalid argument".to_string(), ErrorKind::Validation),
            ("schema".to_string(), ErrorKind::Validation),
            ("connection refused".to_string(), ErrorKind::Transient),
            ("connection reset".to_string(), ErrorKind::Transient),
            ("temporary".to_string(), ErrorKind::Transient),
            ("503".to_string(), ErrorKind::Transient),
            ("unauthorized".to_string(), ErrorKind::Permanent),
            ("forbidden".to_string(), ErrorKind::Permanent),
            ("not found".to_string(), ErrorKind::Permanent),
            ("404".to_string(), ErrorKind::Permanent),
        ];
        Self { patterns }
    }

    /// Register an additional pattern.
    pub fn add_pattern(&mut self, pattern: &str, kind: ErrorKind) {
        self.patterns.push((pattern.to_lowercase(), kind));
    }
}

impl Default for PatternErrorClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorClassifier for PatternErrorClassifier {
    fn classify(&self, error: &str) -> ErrorKind {
        let lower = error.to_lowercase();
        for (pattern, kind) in &self.patterns {
            if lower.contains(pattern) {
                return kind.clone();
            }
        }
        ErrorKind::Unknown
    }
}

// ---------------------------------------------------------------------------
// ErrorAction + ErrorHandler trait
// ---------------------------------------------------------------------------

/// The action an [`ErrorHandler`] decides to take.
#[derive(Debug, Clone, PartialEq)]
pub enum ErrorAction {
    /// Let the error bubble up unchanged.
    Propagate,
    /// Replace the error with a recovery value.
    Recover(Value),
    /// Signal that the operation should be retried.
    Retry,
    /// Replace the result with a fallback value.
    Fallback(Value),
    /// Open the circuit breaker to stop further attempts.
    CircuitBreak,
}

/// Decides what to do with a classified error and the original input.
pub trait ErrorHandler: Send + Sync {
    /// Inspect the error (and optionally the input) and choose an action.
    fn handle(&self, error: &ClassifiedError, input: &Value) -> ErrorAction;
}

// ---------------------------------------------------------------------------
// RecoveryHandler
// ---------------------------------------------------------------------------

/// An [`ErrorHandler`] that always returns a default recovery value.
pub struct RecoveryHandler {
    default: Value,
}

impl RecoveryHandler {
    /// Create a handler that recovers with `default`.
    pub fn new(default: Value) -> Self {
        Self { default }
    }
}

impl ErrorHandler for RecoveryHandler {
    fn handle(&self, _error: &ClassifiedError, _input: &Value) -> ErrorAction {
        ErrorAction::Recover(self.default.clone())
    }
}

// ---------------------------------------------------------------------------
// MapErrorHandler
// ---------------------------------------------------------------------------

/// An [`ErrorHandler`] that transforms the error message and propagates.
pub struct MapErrorHandler {
    mapper: fn(&str) -> String,
}

impl MapErrorHandler {
    /// Create a handler with a mapping function applied to the error message.
    pub fn new(mapper: fn(&str) -> String) -> Self {
        Self { mapper }
    }

    /// Apply the mapper to the given message.
    pub fn map(&self, message: &str) -> String {
        (self.mapper)(message)
    }
}

impl ErrorHandler for MapErrorHandler {
    fn handle(&self, error: &ClassifiedError, _input: &Value) -> ErrorAction {
        let mapped = (self.mapper)(&error.message);
        // Propagate with the mapped message — callers can inspect the action
        // and re-create the error with the new message if desired.
        ErrorAction::Fallback(json!({ "error": mapped }))
    }
}

// ---------------------------------------------------------------------------
// CircuitState + CircuitBreaker
// ---------------------------------------------------------------------------

/// The state of a [`CircuitBreaker`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitState {
    /// Normal operation — requests are allowed.
    Closed,
    /// Too many failures — requests are blocked.
    Open,
    /// The reset timeout has elapsed — one probe request is allowed.
    HalfOpen,
}

impl fmt::Display for CircuitState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => write!(f, "Closed"),
            Self::Open => write!(f, "Open"),
            Self::HalfOpen => write!(f, "HalfOpen"),
        }
    }
}

/// A circuit breaker that tracks consecutive failures and stops sending
/// requests once a threshold is reached, resuming after a timeout.
pub struct CircuitBreaker {
    failure_threshold: u32,
    reset_timeout: Duration,
    failure_count: u32,
    success_count: u32,
    last_failure_time: Option<Instant>,
}

impl CircuitBreaker {
    /// Create a new circuit breaker.
    ///
    /// * `failure_threshold` — number of consecutive failures before the
    ///   circuit opens.
    /// * `reset_timeout` — how long the circuit stays open before transitioning
    ///   to half-open.
    pub fn new(failure_threshold: u32, reset_timeout: Duration) -> Self {
        Self {
            failure_threshold,
            reset_timeout,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
        }
    }

    /// Record a successful call, resetting the failure count and closing the
    /// circuit.
    pub fn record_success(&mut self) {
        self.failure_count = 0;
        self.success_count += 1;
        self.last_failure_time = None;
    }

    /// Record a failed call.
    pub fn record_failure(&mut self) {
        self.failure_count += 1;
        self.success_count = 0;
        self.last_failure_time = Some(Instant::now());
    }

    /// Returns `true` when the circuit is open (failures >= threshold and
    /// reset timeout has **not** yet elapsed).
    pub fn is_open(&self) -> bool {
        self.state() == CircuitState::Open
    }

    /// Returns `true` when the circuit is closed (fewer failures than the
    /// threshold).
    pub fn is_closed(&self) -> bool {
        self.state() == CircuitState::Closed
    }

    /// Returns `true` when the circuit is half-open (open but the reset
    /// timeout has elapsed).
    pub fn is_half_open(&self) -> bool {
        self.state() == CircuitState::HalfOpen
    }

    /// Returns `true` when a request is allowed (circuit closed or half-open).
    pub fn allow_request(&mut self) -> bool {
        let state = self.state();
        matches!(state, CircuitState::Closed | CircuitState::HalfOpen)
    }

    /// The current number of consecutive failures.
    pub fn failure_count(&self) -> u32 {
        self.failure_count
    }

    /// The current number of consecutive successes.
    pub fn success_count(&self) -> u32 {
        self.success_count
    }

    /// Reset the breaker to its initial closed state.
    pub fn reset(&mut self) {
        self.failure_count = 0;
        self.success_count = 0;
        self.last_failure_time = None;
    }

    /// Compute the current state.
    pub fn state(&self) -> CircuitState {
        if self.failure_count < self.failure_threshold {
            return CircuitState::Closed;
        }
        // Threshold reached — check if reset timeout has elapsed.
        match self.last_failure_time {
            Some(t) if t.elapsed() >= self.reset_timeout => CircuitState::HalfOpen,
            _ => CircuitState::Open,
        }
    }
}

// ---------------------------------------------------------------------------
// ErrorChain
// ---------------------------------------------------------------------------

/// Collects a sequence of [`ClassifiedError`]s through a processing pipeline.
#[derive(Debug, Clone)]
pub struct ErrorChain {
    errors: Vec<ClassifiedError>,
}

impl ErrorChain {
    /// Create an empty error chain.
    pub fn new() -> Self {
        Self { errors: Vec::new() }
    }

    /// Append an error to the chain.
    pub fn add(&mut self, error: ClassifiedError) {
        self.errors.push(error);
    }

    /// Return a reference to all collected errors.
    pub fn errors(&self) -> &[ClassifiedError] {
        &self.errors
    }

    /// Number of errors in the chain.
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Returns `true` if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }

    /// Return the most recently added error, if any.
    pub fn latest(&self) -> Option<&ClassifiedError> {
        self.errors.last()
    }

    /// Returns `true` if any error in the chain is [`ErrorKind::Permanent`].
    pub fn has_permanent(&self) -> bool {
        self.errors.iter().any(|e| e.kind == ErrorKind::Permanent)
    }

    /// Count the number of retryable errors in the chain.
    pub fn retry_count(&self) -> usize {
        self.errors.iter().filter(|e| e.kind.is_retryable()).count()
    }

    /// Serialize the entire chain to a JSON `Value`.
    pub fn to_json(&self) -> Value {
        let items: Vec<Value> = self.errors.iter().map(|e| e.to_json()).collect();
        json!({
            "count": self.errors.len(),
            "errors": items,
            "has_permanent": self.has_permanent(),
            "retry_count": self.retry_count(),
        })
    }
}

impl Default for ErrorChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::thread;
    use std::time::Duration;

    // ── ErrorKind ────────────────────────────────────────────────────

    #[test]
    fn test_transient_is_retryable() {
        assert!(ErrorKind::Transient.is_retryable());
    }

    #[test]
    fn test_timeout_is_retryable() {
        assert!(ErrorKind::Timeout.is_retryable());
    }

    #[test]
    fn test_rate_limit_is_retryable() {
        assert!(ErrorKind::RateLimit.is_retryable());
    }

    #[test]
    fn test_permanent_is_not_retryable() {
        assert!(!ErrorKind::Permanent.is_retryable());
    }

    #[test]
    fn test_validation_is_not_retryable() {
        assert!(!ErrorKind::Validation.is_retryable());
    }

    #[test]
    fn test_unknown_is_not_retryable() {
        assert!(!ErrorKind::Unknown.is_retryable());
    }

    #[test]
    fn test_error_kind_display() {
        assert_eq!(ErrorKind::Transient.to_string(), "Transient");
        assert_eq!(ErrorKind::Permanent.to_string(), "Permanent");
        assert_eq!(ErrorKind::Timeout.to_string(), "Timeout");
        assert_eq!(ErrorKind::RateLimit.to_string(), "RateLimit");
        assert_eq!(ErrorKind::Validation.to_string(), "Validation");
        assert_eq!(ErrorKind::Unknown.to_string(), "Unknown");
    }

    // ── ClassifiedError ──────────────────────────────────────────────

    #[test]
    fn test_classified_error_new() {
        let err = ClassifiedError::new(ErrorKind::Transient, "connection lost");
        assert_eq!(err.kind, ErrorKind::Transient);
        assert_eq!(err.message, "connection lost");
        assert!(err.source.is_none());
    }

    #[test]
    fn test_classified_error_with_source() {
        let err =
            ClassifiedError::new(ErrorKind::Permanent, "bad request").with_source("http_client");
        assert_eq!(err.source.as_deref(), Some("http_client"));
    }

    #[test]
    fn test_classified_error_to_json() {
        let err = ClassifiedError::new(ErrorKind::Timeout, "timed out").with_source("llm");
        let j = err.to_json();
        assert_eq!(j["kind"], "Timeout");
        assert_eq!(j["message"], "timed out");
        assert_eq!(j["source"], "llm");
        assert!(j["elapsed_secs"].as_f64().unwrap() >= 0.0);
    }

    #[test]
    fn test_classified_error_json_without_source() {
        let err = ClassifiedError::new(ErrorKind::Unknown, "mystery");
        let j = err.to_json();
        assert!(j["source"].is_null());
    }

    // ── PatternErrorClassifier ───────────────────────────────────────

    #[test]
    fn test_pattern_classifier_timeout() {
        let c = PatternErrorClassifier::new();
        assert_eq!(
            c.classify("request timed out after 30s"),
            ErrorKind::Timeout
        );
    }

    #[test]
    fn test_pattern_classifier_rate_limit() {
        let c = PatternErrorClassifier::new();
        assert_eq!(c.classify("rate limit exceeded"), ErrorKind::RateLimit);
    }

    #[test]
    fn test_pattern_classifier_rate_limit_429() {
        let c = PatternErrorClassifier::new();
        assert_eq!(
            c.classify("HTTP 429 Too Many Requests"),
            ErrorKind::RateLimit
        );
    }

    #[test]
    fn test_pattern_classifier_validation() {
        let c = PatternErrorClassifier::new();
        assert_eq!(
            c.classify("validation failed for field X"),
            ErrorKind::Validation
        );
    }

    #[test]
    fn test_pattern_classifier_transient() {
        let c = PatternErrorClassifier::new();
        assert_eq!(c.classify("connection refused"), ErrorKind::Transient);
    }

    #[test]
    fn test_pattern_classifier_permanent() {
        let c = PatternErrorClassifier::new();
        assert_eq!(c.classify("401 unauthorized"), ErrorKind::Permanent);
    }

    #[test]
    fn test_pattern_classifier_unknown() {
        let c = PatternErrorClassifier::new();
        assert_eq!(c.classify("something completely novel"), ErrorKind::Unknown);
    }

    #[test]
    fn test_pattern_classifier_custom_pattern() {
        let mut c = PatternErrorClassifier::new();
        c.add_pattern("flux capacitor", ErrorKind::Permanent);
        assert_eq!(
            c.classify("flux capacitor overloaded"),
            ErrorKind::Permanent
        );
    }

    #[test]
    fn test_pattern_classifier_case_insensitive() {
        let c = PatternErrorClassifier::new();
        assert_eq!(c.classify("TIMEOUT occurred"), ErrorKind::Timeout);
    }

    #[test]
    fn test_pattern_classifier_no_patterns_match() {
        let c = PatternErrorClassifier { patterns: vec![] };
        assert_eq!(c.classify("anything"), ErrorKind::Unknown);
    }

    // ── RecoveryHandler ──────────────────────────────────────────────

    #[test]
    fn test_recovery_handler_returns_default() {
        let handler = RecoveryHandler::new(json!({"fallback": true}));
        let err = ClassifiedError::new(ErrorKind::Transient, "oops");
        let action = handler.handle(&err, &json!("input"));
        assert_eq!(action, ErrorAction::Recover(json!({"fallback": true})));
    }

    // ── MapErrorHandler ──────────────────────────────────────────────

    #[test]
    fn test_map_error_handler_transforms_message() {
        let handler = MapErrorHandler::new(|msg| format!("wrapped: {}", msg));
        assert_eq!(handler.map("original"), "wrapped: original");
    }

    #[test]
    fn test_map_error_handler_handle_returns_fallback() {
        let handler = MapErrorHandler::new(|msg| format!("[ERR] {}", msg));
        let err = ClassifiedError::new(ErrorKind::Permanent, "bad");
        let action = handler.handle(&err, &json!(null));
        assert_eq!(action, ErrorAction::Fallback(json!({"error": "[ERR] bad"})));
    }

    // ── ErrorAction variants ─────────────────────────────────────────

    #[test]
    fn test_error_action_propagate() {
        let a = ErrorAction::Propagate;
        assert_eq!(a, ErrorAction::Propagate);
    }

    #[test]
    fn test_error_action_recover() {
        let a = ErrorAction::Recover(json!(42));
        assert_eq!(a, ErrorAction::Recover(json!(42)));
    }

    #[test]
    fn test_error_action_retry() {
        let a = ErrorAction::Retry;
        assert_eq!(a, ErrorAction::Retry);
    }

    #[test]
    fn test_error_action_circuit_break() {
        let a = ErrorAction::CircuitBreak;
        assert_eq!(a, ErrorAction::CircuitBreak);
    }

    // ── CircuitBreaker ───────────────────────────────────────────────

    #[test]
    fn test_circuit_breaker_starts_closed() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(5));
        assert!(cb.is_closed());
        assert!(!cb.is_open());
        assert!(!cb.is_half_open());
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[test]
    fn test_circuit_breaker_opens_after_threshold() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_closed()); // 2 < 3
        cb.record_failure();
        assert!(cb.is_open()); // 3 >= 3
        assert_eq!(cb.failure_count(), 3);
    }

    #[test]
    fn test_circuit_breaker_success_resets_failures() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(5));
        cb.record_failure();
        cb.record_failure();
        cb.record_success();
        assert!(cb.is_closed());
        assert_eq!(cb.failure_count(), 0);
        assert_eq!(cb.success_count(), 1);
    }

    #[test]
    fn test_circuit_breaker_half_open_after_timeout() {
        let mut cb = CircuitBreaker::new(2, Duration::from_millis(50));
        cb.record_failure();
        cb.record_failure();
        assert!(cb.is_open());

        // Wait for reset timeout
        thread::sleep(Duration::from_millis(60));
        assert!(cb.is_half_open());
        assert_eq!(cb.state(), CircuitState::HalfOpen);
    }

    #[test]
    fn test_circuit_breaker_allow_request_closed() {
        let mut cb = CircuitBreaker::new(3, Duration::from_secs(5));
        assert!(cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_allow_request_open() {
        let mut cb = CircuitBreaker::new(2, Duration::from_secs(60));
        cb.record_failure();
        cb.record_failure();
        assert!(!cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_allow_request_half_open() {
        let mut cb = CircuitBreaker::new(1, Duration::from_millis(50));
        cb.record_failure();
        assert!(!cb.allow_request());

        thread::sleep(Duration::from_millis(60));
        assert!(cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_reset() {
        let mut cb = CircuitBreaker::new(1, Duration::from_secs(5));
        cb.record_failure();
        assert!(cb.is_open());
        cb.reset();
        assert!(cb.is_closed());
        assert_eq!(cb.failure_count(), 0);
        assert_eq!(cb.success_count(), 0);
    }

    #[test]
    fn test_circuit_breaker_zero_threshold() {
        // With threshold 0, the circuit is never open (0 < 0 is false... but
        // 0 >= 0 means it opens immediately with zero failures).
        let mut cb = CircuitBreaker::new(0, Duration::from_secs(60));
        // 0 failures >= 0 threshold → open
        assert!(cb.is_open());
        assert!(!cb.allow_request());
    }

    #[test]
    fn test_circuit_breaker_state_display() {
        assert_eq!(CircuitState::Closed.to_string(), "Closed");
        assert_eq!(CircuitState::Open.to_string(), "Open");
        assert_eq!(CircuitState::HalfOpen.to_string(), "HalfOpen");
    }

    #[test]
    fn test_circuit_breaker_closed_to_open_to_half_open_to_closed() {
        let mut cb = CircuitBreaker::new(2, Duration::from_millis(50));

        // Closed
        assert_eq!(cb.state(), CircuitState::Closed);

        // -> Open
        cb.record_failure();
        cb.record_failure();
        assert_eq!(cb.state(), CircuitState::Open);

        // -> HalfOpen
        thread::sleep(Duration::from_millis(60));
        assert_eq!(cb.state(), CircuitState::HalfOpen);

        // -> Closed (success in half-open resets)
        cb.record_success();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    // ── ErrorChain ───────────────────────────────────────────────────

    #[test]
    fn test_error_chain_new_is_empty() {
        let chain = ErrorChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        assert!(chain.latest().is_none());
    }

    #[test]
    fn test_error_chain_add_and_len() {
        let mut chain = ErrorChain::new();
        chain.add(ClassifiedError::new(ErrorKind::Transient, "a"));
        chain.add(ClassifiedError::new(ErrorKind::Permanent, "b"));
        assert_eq!(chain.len(), 2);
        assert!(!chain.is_empty());
    }

    #[test]
    fn test_error_chain_latest() {
        let mut chain = ErrorChain::new();
        chain.add(ClassifiedError::new(ErrorKind::Transient, "first"));
        chain.add(ClassifiedError::new(ErrorKind::Timeout, "second"));
        let latest = chain.latest().unwrap();
        assert_eq!(latest.message, "second");
        assert_eq!(latest.kind, ErrorKind::Timeout);
    }

    #[test]
    fn test_error_chain_has_permanent() {
        let mut chain = ErrorChain::new();
        chain.add(ClassifiedError::new(ErrorKind::Transient, "a"));
        assert!(!chain.has_permanent());
        chain.add(ClassifiedError::new(ErrorKind::Permanent, "b"));
        assert!(chain.has_permanent());
    }

    #[test]
    fn test_error_chain_retry_count() {
        let mut chain = ErrorChain::new();
        chain.add(ClassifiedError::new(ErrorKind::Transient, "a"));
        chain.add(ClassifiedError::new(ErrorKind::Permanent, "b"));
        chain.add(ClassifiedError::new(ErrorKind::Timeout, "c"));
        chain.add(ClassifiedError::new(ErrorKind::RateLimit, "d"));
        chain.add(ClassifiedError::new(ErrorKind::Unknown, "e"));
        assert_eq!(chain.retry_count(), 3); // Transient + Timeout + RateLimit
    }

    #[test]
    fn test_error_chain_errors_slice() {
        let mut chain = ErrorChain::new();
        chain.add(ClassifiedError::new(ErrorKind::Validation, "v1"));
        chain.add(ClassifiedError::new(ErrorKind::Validation, "v2"));
        let errors = chain.errors();
        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].message, "v1");
        assert_eq!(errors[1].message, "v2");
    }

    #[test]
    fn test_error_chain_to_json() {
        let mut chain = ErrorChain::new();
        chain.add(ClassifiedError::new(ErrorKind::Transient, "err1"));
        chain.add(ClassifiedError::new(ErrorKind::Permanent, "err2"));
        let j = chain.to_json();
        assert_eq!(j["count"], 2);
        assert_eq!(j["has_permanent"], true);
        assert_eq!(j["retry_count"], 1);
        assert!(j["errors"].is_array());
        assert_eq!(j["errors"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_error_chain_default() {
        let chain = ErrorChain::default();
        assert!(chain.is_empty());
    }
}
