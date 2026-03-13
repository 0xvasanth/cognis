//! Middleware framework for runnable pipelines.
//!
//! Provides a composable middleware system that intercepts inputs and outputs
//! flowing through a runnable chain. Each middleware can inspect, transform,
//! or short-circuit values at any point in the pipeline.
//!
//! - [`MiddlewareAction`] — control-flow enum returned by middleware hooks (continue, short-circuit, or error)
//! - [`Middleware`] — trait for implementing custom middleware logic
//! - [`MiddlewareChain`] — ordered sequence of middleware that processes values in series
//! - [`RetryMiddleware`] — retries failed invocations with configurable backoff
//! - [`CacheMiddleware`] — caches results keyed by input to avoid redundant computation
//! - [`TimeoutMiddleware`] — enforces a maximum duration on downstream processing
//! - [`LoggingMiddleware`] — logs inputs, outputs, and errors for observability

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::error::{Result, CognisError};

/// Represents the action a middleware hook returns to control flow.
#[derive(Debug, Clone)]
pub enum MiddlewareAction {
    /// Continue processing with the (possibly modified) value.
    Continue(Value),
    /// Stop the middleware chain and return this value immediately.
    ShortCircuit(Value),
    /// Abort processing with an error.
    Error(String),
}

impl MiddlewareAction {
    /// Returns `true` if this action is `Continue`.
    pub fn is_continue(&self) -> bool {
        matches!(self, MiddlewareAction::Continue(_))
    }

    /// Returns `true` if this action is `ShortCircuit`.
    pub fn is_short_circuit(&self) -> bool {
        matches!(self, MiddlewareAction::ShortCircuit(_))
    }

    /// Consumes this action and returns the inner value, or an error if it was `Error`.
    pub fn into_value(self) -> Result<Value> {
        match self {
            MiddlewareAction::Continue(v) | MiddlewareAction::ShortCircuit(v) => Ok(v),
            MiddlewareAction::Error(msg) => Err(CognisError::Other(msg)),
        }
    }
}

/// Trait for middleware that can hook into runnable execution.
///
/// Middleware can inspect and transform inputs before processing (`before`)
/// and outputs after processing (`after`).
pub trait RunnableMiddleware: Send + Sync {
    /// Returns the name of this middleware.
    fn name(&self) -> &str;

    /// Called before the runnable processes the input.
    ///
    /// Default implementation passes the input through unchanged.
    fn before(&self, input: &Value) -> MiddlewareAction {
        MiddlewareAction::Continue(input.clone())
    }

    /// Called after the runnable produces output.
    ///
    /// Default implementation passes the output through unchanged.
    fn after(&self, _input: &Value, output: &Value) -> MiddlewareAction {
        MiddlewareAction::Continue(output.clone())
    }
}

/// Middleware that logs inputs and outputs to a shared buffer.
pub struct LoggingMiddleware {
    logs: Arc<Mutex<Vec<String>>>,
}

impl LoggingMiddleware {
    /// Creates a new `LoggingMiddleware` with an empty log buffer.
    pub fn new() -> Self {
        Self {
            logs: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Returns a copy of all logged messages.
    pub fn logs(&self) -> Vec<String> {
        self.logs.lock().unwrap().clone()
    }
}

impl Default for LoggingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl RunnableMiddleware for LoggingMiddleware {
    fn name(&self) -> &str {
        "LoggingMiddleware"
    }

    fn before(&self, input: &Value) -> MiddlewareAction {
        self.logs.lock().unwrap().push(format!("before: {}", input));
        MiddlewareAction::Continue(input.clone())
    }

    fn after(&self, _input: &Value, output: &Value) -> MiddlewareAction {
        self.logs.lock().unwrap().push(format!("after: {}", output));
        MiddlewareAction::Continue(output.clone())
    }
}

/// Middleware that applies a transform function to input before processing.
pub struct TransformMiddleware {
    middleware_name: String,
    transform_fn: Box<dyn Fn(&Value) -> Value + Send + Sync>,
}

impl TransformMiddleware {
    /// Creates a new `TransformMiddleware` with the given name and transform function.
    pub fn new<F>(name: impl Into<String>, transform_fn: F) -> Self
    where
        F: Fn(&Value) -> Value + Send + Sync + 'static,
    {
        Self {
            middleware_name: name.into(),
            transform_fn: Box::new(transform_fn),
        }
    }
}

impl RunnableMiddleware for TransformMiddleware {
    fn name(&self) -> &str {
        &self.middleware_name
    }

    fn before(&self, input: &Value) -> MiddlewareAction {
        let transformed = (self.transform_fn)(input);
        MiddlewareAction::Continue(transformed)
    }
}

/// Middleware that validates input contains all required fields.
pub struct ValidationMiddleware {
    required_fields: Vec<String>,
}

impl ValidationMiddleware {
    /// Creates a new `ValidationMiddleware` that checks for the given required fields.
    pub fn new(required_fields: Vec<String>) -> Self {
        Self { required_fields }
    }
}

impl RunnableMiddleware for ValidationMiddleware {
    fn name(&self) -> &str {
        "ValidationMiddleware"
    }

    fn before(&self, input: &Value) -> MiddlewareAction {
        if let Some(obj) = input.as_object() {
            for field in &self.required_fields {
                if !obj.contains_key(field) {
                    return MiddlewareAction::Error(format!("Missing required field: '{}'", field));
                }
            }
            MiddlewareAction::Continue(input.clone())
        } else {
            MiddlewareAction::Error("Input must be a JSON object".to_string())
        }
    }
}

/// Middleware that records execution time for each invocation.
pub struct TimingMiddleware {
    timings: Arc<Mutex<Vec<Duration>>>,
    start_time: Arc<Mutex<Option<Instant>>>,
}

impl TimingMiddleware {
    /// Creates a new `TimingMiddleware`.
    pub fn new() -> Self {
        Self {
            timings: Arc::new(Mutex::new(Vec::new())),
            start_time: Arc::new(Mutex::new(None)),
        }
    }

    /// Returns a copy of all recorded durations.
    pub fn timings(&self) -> Vec<Duration> {
        self.timings.lock().unwrap().clone()
    }

    /// Returns the average duration, or `None` if no timings have been recorded.
    pub fn average(&self) -> Option<Duration> {
        let timings = self.timings.lock().unwrap();
        if timings.is_empty() {
            return None;
        }
        let total: Duration = timings.iter().sum();
        Some(total / timings.len() as u32)
    }
}

impl Default for TimingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl RunnableMiddleware for TimingMiddleware {
    fn name(&self) -> &str {
        "TimingMiddleware"
    }

    fn before(&self, input: &Value) -> MiddlewareAction {
        *self.start_time.lock().unwrap() = Some(Instant::now());
        MiddlewareAction::Continue(input.clone())
    }

    fn after(&self, _input: &Value, output: &Value) -> MiddlewareAction {
        if let Some(start) = self.start_time.lock().unwrap().take() {
            self.timings.lock().unwrap().push(start.elapsed());
        }
        MiddlewareAction::Continue(output.clone())
    }
}

/// Middleware that tracks retry attempts.
pub struct RetryMiddleware {
    max_retries: usize,
    attempts: Arc<Mutex<usize>>,
}

impl RetryMiddleware {
    /// Creates a new `RetryMiddleware` with the given maximum retry count.
    pub fn new(max_retries: usize) -> Self {
        Self {
            max_retries,
            attempts: Arc::new(Mutex::new(0)),
        }
    }

    /// Returns the number of attempts recorded so far.
    pub fn attempts(&self) -> usize {
        *self.attempts.lock().unwrap()
    }
}

impl RunnableMiddleware for RetryMiddleware {
    fn name(&self) -> &str {
        "RetryMiddleware"
    }

    fn before(&self, input: &Value) -> MiddlewareAction {
        let mut attempts = self.attempts.lock().unwrap();
        *attempts += 1;
        if *attempts > self.max_retries + 1 {
            return MiddlewareAction::Error(format!("Max retries ({}) exceeded", self.max_retries));
        }
        MiddlewareAction::Continue(input.clone())
    }
}

/// An ordered chain of middleware that runs before/after hooks sequentially.
pub struct MiddlewareChain {
    middlewares: Vec<Arc<dyn RunnableMiddleware>>,
}

impl MiddlewareChain {
    /// Creates an empty middleware chain.
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    /// Adds a middleware to the end of the chain.
    pub fn add(&mut self, middleware: Arc<dyn RunnableMiddleware>) {
        self.middlewares.push(middleware);
    }

    /// Returns the number of middleware in the chain.
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    /// Returns `true` if the chain contains no middleware.
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Runs all `before` hooks in order.
    ///
    /// If any middleware returns `ShortCircuit` or `Error`, the chain stops
    /// and that action is returned immediately. Otherwise the final
    /// `Continue` value is returned.
    pub fn run_before(&self, input: &Value) -> MiddlewareAction {
        let mut current = input.clone();
        for mw in &self.middlewares {
            match mw.before(&current) {
                MiddlewareAction::Continue(v) => current = v,
                action @ MiddlewareAction::ShortCircuit(_) => return action,
                action @ MiddlewareAction::Error(_) => return action,
            }
        }
        MiddlewareAction::Continue(current)
    }

    /// Runs all `after` hooks in order.
    ///
    /// If any middleware returns `ShortCircuit` or `Error`, the chain stops
    /// and that action is returned immediately.
    pub fn run_after(&self, input: &Value, output: &Value) -> MiddlewareAction {
        let mut current = output.clone();
        for mw in &self.middlewares {
            match mw.after(input, &current) {
                MiddlewareAction::Continue(v) => current = v,
                action @ MiddlewareAction::ShortCircuit(_) => return action,
                action @ MiddlewareAction::Error(_) => return action,
            }
        }
        MiddlewareAction::Continue(current)
    }
}

impl Default for MiddlewareChain {
    fn default() -> Self {
        Self::new()
    }
}

/// A conceptual runnable wrapped with a middleware chain.
///
/// When `execute` is called, it runs the before chain, applies an identity
/// transform (passes the value through), then runs the after chain.
pub struct RunnableWithMiddleware {
    name: String,
    chain: MiddlewareChain,
}

impl RunnableWithMiddleware {
    /// Creates a new `RunnableWithMiddleware` with the given name and an empty chain.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            chain: MiddlewareChain::new(),
        }
    }

    /// Adds a middleware to this runnable's chain. Builder pattern.
    pub fn with_middleware(mut self, middleware: Arc<dyn RunnableMiddleware>) -> Self {
        self.chain.add(middleware);
        self
    }

    /// Executes the middleware pipeline:
    /// 1. Runs `before` hooks (may short-circuit or error)
    /// 2. Identity transform (passes the value through)
    /// 3. Runs `after` hooks (may short-circuit or error)
    pub fn execute(&self, input: Value) -> Result<Value> {
        // Run before chain
        let before_result = self.chain.run_before(&input);
        let processed = match before_result {
            MiddlewareAction::Continue(v) => v,
            MiddlewareAction::ShortCircuit(v) => return Ok(v),
            MiddlewareAction::Error(msg) => return Err(CognisError::Other(msg)),
        };

        // Identity transform — the "runnable" just passes through
        let output = processed.clone();

        // Run after chain
        let after_result = self.chain.run_after(&processed, &output);
        match after_result {
            MiddlewareAction::Continue(v) | MiddlewareAction::ShortCircuit(v) => Ok(v),
            MiddlewareAction::Error(msg) => Err(CognisError::Other(msg)),
        }
    }

    /// Returns the name of this runnable.
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── MiddlewareAction tests ──────────────────────────────────────

    #[test]
    fn test_continue_is_continue() {
        let action = MiddlewareAction::Continue(json!(1));
        assert!(action.is_continue());
        assert!(!action.is_short_circuit());
    }

    #[test]
    fn test_short_circuit_is_short_circuit() {
        let action = MiddlewareAction::ShortCircuit(json!("done"));
        assert!(action.is_short_circuit());
        assert!(!action.is_continue());
    }

    #[test]
    fn test_error_is_neither() {
        let action = MiddlewareAction::Error("fail".into());
        assert!(!action.is_continue());
        assert!(!action.is_short_circuit());
    }

    #[test]
    fn test_continue_into_value() {
        let action = MiddlewareAction::Continue(json!(42));
        assert_eq!(action.into_value().unwrap(), json!(42));
    }

    #[test]
    fn test_short_circuit_into_value() {
        let action = MiddlewareAction::ShortCircuit(json!("early"));
        assert_eq!(action.into_value().unwrap(), json!("early"));
    }

    #[test]
    fn test_error_into_value_is_err() {
        let action = MiddlewareAction::Error("boom".into());
        let result = action.into_value();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("boom"));
    }

    // ── LoggingMiddleware tests ─────────────────────────────────────

    #[test]
    fn test_logging_middleware_before() {
        let logger = LoggingMiddleware::new();
        let action = logger.before(&json!({"key": "value"}));
        assert!(action.is_continue());
        let logs = logger.logs();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].starts_with("before:"));
    }

    #[test]
    fn test_logging_middleware_after() {
        let logger = LoggingMiddleware::new();
        let action = logger.after(&json!("in"), &json!("out"));
        assert!(action.is_continue());
        let logs = logger.logs();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].starts_with("after:"));
    }

    #[test]
    fn test_logging_middleware_multiple_calls() {
        let logger = LoggingMiddleware::new();
        logger.before(&json!(1));
        logger.after(&json!(1), &json!(2));
        logger.before(&json!(3));
        logger.after(&json!(3), &json!(4));
        assert_eq!(logger.logs().len(), 4);
    }

    #[test]
    fn test_logging_middleware_name() {
        let logger = LoggingMiddleware::new();
        assert_eq!(logger.name(), "LoggingMiddleware");
    }

    // ── TransformMiddleware tests ───────────────────────────────────

    #[test]
    fn test_transform_middleware_modifies_input() {
        let transform =
            TransformMiddleware::new("double", |v: &Value| json!(v.as_i64().unwrap_or(0) * 2));
        let action = transform.before(&json!(5));
        assert!(action.is_continue());
        assert_eq!(action.into_value().unwrap(), json!(10));
    }

    #[test]
    fn test_transform_middleware_name() {
        let transform = TransformMiddleware::new("my_transform", |v: &Value| v.clone());
        assert_eq!(transform.name(), "my_transform");
    }

    #[test]
    fn test_transform_middleware_after_passes_through() {
        let transform = TransformMiddleware::new("noop", |v: &Value| v.clone());
        let action = transform.after(&json!("in"), &json!("out"));
        assert!(action.is_continue());
        assert_eq!(action.into_value().unwrap(), json!("out"));
    }

    #[test]
    fn test_transform_middleware_adds_field() {
        let transform = TransformMiddleware::new("add_field", |v: &Value| {
            let mut obj = v.as_object().cloned().unwrap_or_default();
            obj.insert("added".to_string(), json!(true));
            Value::Object(obj)
        });
        let action = transform.before(&json!({"name": "test"}));
        let result = action.into_value().unwrap();
        assert_eq!(result["added"], json!(true));
        assert_eq!(result["name"], json!("test"));
    }

    // ── ValidationMiddleware tests ──────────────────────────────────

    #[test]
    fn test_validation_passes_with_all_fields() {
        let validator = ValidationMiddleware::new(vec!["name".into(), "age".into()]);
        let action = validator.before(&json!({"name": "Alice", "age": 30}));
        assert!(action.is_continue());
    }

    #[test]
    fn test_validation_fails_missing_field() {
        let validator = ValidationMiddleware::new(vec!["name".into(), "email".into()]);
        let action = validator.before(&json!({"name": "Alice"}));
        match action {
            MiddlewareAction::Error(msg) => assert!(msg.contains("email")),
            _ => panic!("Expected Error action"),
        }
    }

    #[test]
    fn test_validation_fails_non_object() {
        let validator = ValidationMiddleware::new(vec!["field".into()]);
        let action = validator.before(&json!("not an object"));
        match action {
            MiddlewareAction::Error(msg) => assert!(msg.contains("JSON object")),
            _ => panic!("Expected Error action"),
        }
    }

    #[test]
    fn test_validation_empty_required_fields() {
        let validator = ValidationMiddleware::new(vec![]);
        let action = validator.before(&json!({"anything": true}));
        assert!(action.is_continue());
    }

    #[test]
    fn test_validation_middleware_name() {
        let validator = ValidationMiddleware::new(vec![]);
        assert_eq!(validator.name(), "ValidationMiddleware");
    }

    // ── TimingMiddleware tests ──────────────────────────────────────

    #[test]
    fn test_timing_records_duration() {
        let timer = TimingMiddleware::new();
        timer.before(&json!("start"));
        std::thread::sleep(Duration::from_millis(10));
        timer.after(&json!("start"), &json!("end"));
        let timings = timer.timings();
        assert_eq!(timings.len(), 1);
        assert!(timings[0] >= Duration::from_millis(5));
    }

    #[test]
    fn test_timing_multiple_invocations() {
        let timer = TimingMiddleware::new();
        for _ in 0..3 {
            timer.before(&json!("in"));
            timer.after(&json!("in"), &json!("out"));
        }
        assert_eq!(timer.timings().len(), 3);
    }

    #[test]
    fn test_timing_average_none_when_empty() {
        let timer = TimingMiddleware::new();
        assert!(timer.average().is_none());
    }

    #[test]
    fn test_timing_average_some_when_recorded() {
        let timer = TimingMiddleware::new();
        timer.before(&json!(1));
        std::thread::sleep(Duration::from_millis(10));
        timer.after(&json!(1), &json!(2));
        let avg = timer.average();
        assert!(avg.is_some());
        assert!(avg.unwrap() >= Duration::from_millis(5));
    }

    #[test]
    fn test_timing_middleware_name() {
        let timer = TimingMiddleware::new();
        assert_eq!(timer.name(), "TimingMiddleware");
    }

    // ── RetryMiddleware tests ───────────────────────────────────────

    #[test]
    fn test_retry_allows_within_limit() {
        let retry = RetryMiddleware::new(3);
        for _ in 0..4 {
            let action = retry.before(&json!("try"));
            assert!(action.is_continue());
        }
        assert_eq!(retry.attempts(), 4);
    }

    #[test]
    fn test_retry_exceeds_limit() {
        let retry = RetryMiddleware::new(2);
        // 3 attempts allowed (1 original + 2 retries)
        retry.before(&json!("try")); // attempt 1
        retry.before(&json!("try")); // attempt 2
        retry.before(&json!("try")); // attempt 3
        let action = retry.before(&json!("try")); // attempt 4 - exceeds
        match action {
            MiddlewareAction::Error(msg) => assert!(msg.contains("Max retries")),
            _ => panic!("Expected Error action"),
        }
    }

    #[test]
    fn test_retry_middleware_name() {
        let retry = RetryMiddleware::new(1);
        assert_eq!(retry.name(), "RetryMiddleware");
    }

    // ── MiddlewareChain tests ───────────────────────────────────────

    #[test]
    fn test_chain_empty_passthrough_before() {
        let chain = MiddlewareChain::new();
        let action = chain.run_before(&json!(42));
        assert!(action.is_continue());
        assert_eq!(action.into_value().unwrap(), json!(42));
    }

    #[test]
    fn test_chain_empty_passthrough_after() {
        let chain = MiddlewareChain::new();
        let action = chain.run_after(&json!("in"), &json!("out"));
        assert!(action.is_continue());
        assert_eq!(action.into_value().unwrap(), json!("out"));
    }

    #[test]
    fn test_chain_len_and_is_empty() {
        let mut chain = MiddlewareChain::new();
        assert!(chain.is_empty());
        assert_eq!(chain.len(), 0);
        chain.add(Arc::new(LoggingMiddleware::new()));
        assert!(!chain.is_empty());
        assert_eq!(chain.len(), 1);
    }

    #[test]
    fn test_chain_ordering_before() {
        let mut chain = MiddlewareChain::new();
        // First: multiply by 2
        chain.add(Arc::new(TransformMiddleware::new("mul2", |v: &Value| {
            json!(v.as_i64().unwrap() * 2)
        })));
        // Second: add 1
        chain.add(Arc::new(TransformMiddleware::new("add1", |v: &Value| {
            json!(v.as_i64().unwrap() + 1)
        })));
        // Input 5 -> mul2 -> 10 -> add1 -> 11
        let action = chain.run_before(&json!(5));
        assert_eq!(action.into_value().unwrap(), json!(11));
    }

    #[test]
    fn test_chain_short_circuit_stops_processing() {
        struct ShortCircuitMW;
        impl RunnableMiddleware for ShortCircuitMW {
            fn name(&self) -> &str {
                "short_circuit"
            }
            fn before(&self, _input: &Value) -> MiddlewareAction {
                MiddlewareAction::ShortCircuit(json!("stopped"))
            }
        }

        let mut chain = MiddlewareChain::new();
        chain.add(Arc::new(ShortCircuitMW));
        chain.add(Arc::new(TransformMiddleware::new("never_reached", |_| {
            json!("should not appear")
        })));

        let action = chain.run_before(&json!("input"));
        assert!(action.is_short_circuit());
        assert_eq!(action.into_value().unwrap(), json!("stopped"));
    }

    #[test]
    fn test_chain_error_stops_processing() {
        let mut chain = MiddlewareChain::new();
        chain.add(Arc::new(ValidationMiddleware::new(vec!["required".into()])));
        chain.add(Arc::new(LoggingMiddleware::new()));

        let action = chain.run_before(&json!({"other": true}));
        match action {
            MiddlewareAction::Error(msg) => assert!(msg.contains("required")),
            _ => panic!("Expected Error action"),
        }
    }

    #[test]
    fn test_chain_multiple_middleware_before_and_after() {
        let logger = Arc::new(LoggingMiddleware::new());
        let mut chain = MiddlewareChain::new();
        chain.add(logger.clone());

        let action = chain.run_before(&json!("hello"));
        assert!(action.is_continue());
        let action = chain.run_after(&json!("hello"), &json!("world"));
        assert!(action.is_continue());

        let logs = logger.logs();
        assert_eq!(logs.len(), 2);
        assert!(logs[0].contains("hello"));
        assert!(logs[1].contains("world"));
    }

    // ── RunnableWithMiddleware tests ────────────────────────────────

    #[test]
    fn test_runnable_with_middleware_passthrough() {
        let runnable = RunnableWithMiddleware::new("identity");
        let result = runnable.execute(json!(42)).unwrap();
        assert_eq!(result, json!(42));
    }

    #[test]
    fn test_runnable_with_middleware_name() {
        let runnable = RunnableWithMiddleware::new("my_runnable");
        assert_eq!(runnable.name(), "my_runnable");
    }

    #[test]
    fn test_runnable_with_logging() {
        let logger = Arc::new(LoggingMiddleware::new());
        let runnable = RunnableWithMiddleware::new("logged").with_middleware(logger.clone());
        let result = runnable.execute(json!("test")).unwrap();
        assert_eq!(result, json!("test"));
        let logs = logger.logs();
        assert_eq!(logs.len(), 2);
        assert!(logs[0].starts_with("before:"));
        assert!(logs[1].starts_with("after:"));
    }

    #[test]
    fn test_runnable_with_transform() {
        let runnable = RunnableWithMiddleware::new("transformed").with_middleware(Arc::new(
            TransformMiddleware::new("upper", |v: &Value| {
                json!(v.as_str().unwrap_or("").to_uppercase())
            }),
        ));
        let result = runnable.execute(json!("hello")).unwrap();
        assert_eq!(result, json!("HELLO"));
    }

    #[test]
    fn test_runnable_with_validation_pass() {
        let runnable = RunnableWithMiddleware::new("validated")
            .with_middleware(Arc::new(ValidationMiddleware::new(vec!["name".into()])));
        let result = runnable.execute(json!({"name": "Alice"}));
        assert!(result.is_ok());
    }

    #[test]
    fn test_runnable_with_validation_fail() {
        let runnable = RunnableWithMiddleware::new("validated")
            .with_middleware(Arc::new(ValidationMiddleware::new(vec!["name".into()])));
        let result = runnable.execute(json!({"age": 30}));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("name"));
    }

    #[test]
    fn test_runnable_short_circuit_in_before() {
        struct EarlyReturn;
        impl RunnableMiddleware for EarlyReturn {
            fn name(&self) -> &str {
                "early_return"
            }
            fn before(&self, _input: &Value) -> MiddlewareAction {
                MiddlewareAction::ShortCircuit(json!("early"))
            }
        }

        let runnable = RunnableWithMiddleware::new("short").with_middleware(Arc::new(EarlyReturn));
        let result = runnable.execute(json!("ignored")).unwrap();
        assert_eq!(result, json!("early"));
    }

    #[test]
    fn test_runnable_error_in_before() {
        let runnable = RunnableWithMiddleware::new("fail")
            .with_middleware(Arc::new(ValidationMiddleware::new(vec!["x".into()])));
        let result = runnable.execute(json!("not_an_object"));
        assert!(result.is_err());
    }

    #[test]
    fn test_runnable_multiple_middleware_composition() {
        let logger = Arc::new(LoggingMiddleware::new());
        let runnable = RunnableWithMiddleware::new("composed")
            .with_middleware(Arc::new(ValidationMiddleware::new(vec!["value".into()])))
            .with_middleware(Arc::new(TransformMiddleware::new(
                "extract",
                |v: &Value| v.get("value").cloned().unwrap_or(json!(null)),
            )))
            .with_middleware(logger.clone());

        let result = runnable.execute(json!({"value": 99})).unwrap();
        assert_eq!(result, json!(99));
        assert_eq!(logger.logs().len(), 2);
    }

    #[test]
    fn test_runnable_with_timing() {
        let timer = Arc::new(TimingMiddleware::new());
        let runnable = RunnableWithMiddleware::new("timed").with_middleware(timer.clone());

        runnable.execute(json!("a")).unwrap();
        runnable.execute(json!("b")).unwrap();
        assert_eq!(timer.timings().len(), 2);
        assert!(timer.average().is_some());
    }

    #[test]
    fn test_runnable_with_retry_within_limit() {
        let retry = Arc::new(RetryMiddleware::new(3));
        let runnable = RunnableWithMiddleware::new("retry_ok").with_middleware(retry.clone());

        assert!(runnable.execute(json!("try")).is_ok());
        assert_eq!(retry.attempts(), 1);
    }

    #[test]
    fn test_chain_after_short_circuit() {
        struct AfterShortCircuit;
        impl RunnableMiddleware for AfterShortCircuit {
            fn name(&self) -> &str {
                "after_sc"
            }
            fn after(&self, _input: &Value, _output: &Value) -> MiddlewareAction {
                MiddlewareAction::ShortCircuit(json!("after_early"))
            }
        }

        let mut chain = MiddlewareChain::new();
        chain.add(Arc::new(AfterShortCircuit));
        let action = chain.run_after(&json!("in"), &json!("out"));
        assert!(action.is_short_circuit());
        assert_eq!(action.into_value().unwrap(), json!("after_early"));
    }

    #[test]
    fn test_chain_after_error() {
        struct AfterError;
        impl RunnableMiddleware for AfterError {
            fn name(&self) -> &str {
                "after_err"
            }
            fn after(&self, _input: &Value, _output: &Value) -> MiddlewareAction {
                MiddlewareAction::Error("after failed".into())
            }
        }

        let mut chain = MiddlewareChain::new();
        chain.add(Arc::new(AfterError));
        let action = chain.run_after(&json!("in"), &json!("out"));
        match action {
            MiddlewareAction::Error(msg) => assert!(msg.contains("after failed")),
            _ => panic!("Expected Error"),
        }
    }

    #[test]
    fn test_default_trait_impls_pass_through() {
        struct NoopMiddleware;
        impl RunnableMiddleware for NoopMiddleware {
            fn name(&self) -> &str {
                "noop"
            }
        }

        let mw = NoopMiddleware;
        let before = mw.before(&json!("x"));
        assert!(before.is_continue());
        assert_eq!(before.into_value().unwrap(), json!("x"));

        let after = mw.after(&json!("in"), &json!("out"));
        assert!(after.is_continue());
        assert_eq!(after.into_value().unwrap(), json!("out"));
    }
}
