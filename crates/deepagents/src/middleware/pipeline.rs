//! Middleware pipeline system for DeepAgents.
//!
//! Provides a composable pipeline of middleware that can intercept, validate,
//! transform, and rate-limit requests flowing through the agent system.
//!
//! # Built-in middleware
//!
//! - [`LoggingMiddleware`] -- records each request/response to an internal log
//! - [`ValidationMiddleware`] -- ensures required fields are present in requests
//! - [`TransformMiddleware`] -- applies arbitrary transformations to request values
//! - [`RateLimitMiddleware`] -- simple call-count-based rate limiting
//!
//! # Pipeline execution
//!
//! [`MiddlewarePipeline`] runs middleware in priority order (lowest priority value
//! first). Execution halts early if any middleware returns [`MiddlewareResult::Halt`]
//! or [`MiddlewareResult::Error`].
//!
//! # Middleware stacks
//!
//! [`MiddlewareStack`] provides named collections of pipelines, allowing different
//! pipelines to be registered and executed by name.

use std::cell::RefCell;
use std::collections::HashMap;

use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// MiddlewareContext
// ---------------------------------------------------------------------------

/// Context passed through the middleware pipeline.
///
/// Carries the request payload, arbitrary metadata, an agent identifier, and
/// a timestamp string.
#[derive(Debug, Clone)]
pub struct MiddlewareContext {
    /// The request payload.
    pub request: Value,
    /// Arbitrary metadata associated with this context.
    pub metadata: HashMap<String, Value>,
    /// Identifier of the agent that originated the request.
    pub agent_id: String,
    /// Timestamp string (ISO 8601 or similar).
    pub timestamp: String,
}

impl MiddlewareContext {
    /// Create a new context with the given request, agent id, and timestamp.
    pub fn new(request: Value, agent_id: impl Into<String>, timestamp: impl Into<String>) -> Self {
        Self {
            request,
            metadata: HashMap::new(),
            agent_id: agent_id.into(),
            timestamp: timestamp.into(),
        }
    }

    /// Retrieve a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&Value> {
        self.metadata.get(key)
    }

    /// Insert or update a metadata value.
    pub fn set_metadata(&mut self, key: impl Into<String>, value: Value) {
        self.metadata.insert(key.into(), value);
    }

    /// Serialize the entire context to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "request": self.request,
            "metadata": self.metadata,
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
        })
    }
}

// ---------------------------------------------------------------------------
// MiddlewareResult
// ---------------------------------------------------------------------------

/// Outcome of a single pipeline middleware invocation.
#[derive(Debug, Clone)]
pub enum MiddlewareResult {
    /// Processing should continue with the (possibly modified) value.
    Continue(Value),
    /// Processing should stop; the value is the final response.
    Halt(Value),
    /// An error occurred.
    Error(String),
}

impl MiddlewareResult {
    /// Returns `true` if this is [`MiddlewareResult::Continue`].
    pub fn is_continue(&self) -> bool {
        matches!(self, Self::Continue(_))
    }

    /// Returns `true` if this is [`MiddlewareResult::Halt`].
    pub fn is_halt(&self) -> bool {
        matches!(self, Self::Halt(_))
    }

    /// Returns `true` if this is [`MiddlewareResult::Error`].
    pub fn is_error(&self) -> bool {
        matches!(self, Self::Error(_))
    }

    /// Consume the result and return the inner value.
    ///
    /// For [`MiddlewareResult::Continue`] and [`MiddlewareResult::Halt`] the
    /// inner [`Value`] is returned. For [`MiddlewareResult::Error`] the error
    /// string is wrapped in a JSON string value.
    pub fn into_value(self) -> Value {
        match self {
            Self::Continue(v) | Self::Halt(v) => v,
            Self::Error(e) => Value::String(e),
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineMiddleware trait
// ---------------------------------------------------------------------------

/// Trait for middleware that participates in a [`MiddlewarePipeline`].
///
/// This is a synchronous, request-oriented middleware pattern complementing
/// the async [`super::Middleware`] trait which hooks into model/tool lifecycle
/// events.
///
/// Pipeline middleware is processed in priority order (lowest value first).
/// The [`process`](PipelineMiddleware::process) method receives a mutable
/// context and returns a [`MiddlewareResult`] that determines whether the
/// pipeline continues, halts, or errors.
pub trait PipelineMiddleware {
    /// Returns the name of this middleware.
    fn name(&self) -> &str;

    /// Returns the priority of this middleware. Lower values run first.
    /// The default priority is `0`.
    fn priority(&self) -> i32 {
        0
    }

    /// Process a request through this middleware.
    fn process(&self, ctx: &mut MiddlewareContext) -> MiddlewareResult;

    /// Called when an error occurs downstream of this middleware.
    ///
    /// The default implementation returns `MiddlewareResult::Error` with the
    /// same error string.
    fn on_error(&self, _ctx: &MiddlewareContext, error: &str) -> MiddlewareResult {
        MiddlewareResult::Error(error.to_string())
    }
}

// ---------------------------------------------------------------------------
// MiddlewarePipeline
// ---------------------------------------------------------------------------

/// An ordered chain of middleware executed in priority order.
pub struct MiddlewarePipeline {
    middlewares: Vec<Box<dyn PipelineMiddleware>>,
}

impl MiddlewarePipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            middlewares: Vec::new(),
        }
    }

    /// Add a middleware to the pipeline. The pipeline will be re-sorted by
    /// priority on the next [`execute`](Self::execute) call.
    pub fn add(&mut self, middleware: Box<dyn PipelineMiddleware>) {
        self.middlewares.push(middleware);
    }

    /// Execute the pipeline against the given context.
    ///
    /// Middleware is executed in ascending priority order. If any middleware
    /// returns [`MiddlewareResult::Halt`] or [`MiddlewareResult::Error`],
    /// execution stops immediately and that result is returned. If a
    /// middleware returns [`MiddlewareResult::Error`], the `on_error` callback
    /// of all previously-executed middleware is invoked (in reverse order).
    ///
    /// If all middleware return [`MiddlewareResult::Continue`], the last
    /// continue value is returned. For an empty pipeline,
    /// `Continue(Value::Null)` is returned.
    pub fn execute(&self, ctx: &mut MiddlewareContext) -> MiddlewareResult {
        if self.middlewares.is_empty() {
            return MiddlewareResult::Continue(Value::Null);
        }

        // Build a sorted index by priority.
        let mut indices: Vec<usize> = (0..self.middlewares.len()).collect();
        indices.sort_by_key(|&i| self.middlewares[i].priority());

        let mut last_value = Value::Null;
        let mut executed: Vec<usize> = Vec::new();

        for &idx in &indices {
            let mw = &self.middlewares[idx];
            let result = mw.process(ctx);
            match result {
                MiddlewareResult::Continue(v) => {
                    last_value = v;
                    executed.push(idx);
                }
                MiddlewareResult::Halt(v) => {
                    return MiddlewareResult::Halt(v);
                }
                MiddlewareResult::Error(ref e) => {
                    // Notify previously executed middleware of the error.
                    let err_str = e.clone();
                    for &prev_idx in executed.iter().rev() {
                        let _ = self.middlewares[prev_idx].on_error(ctx, &err_str);
                    }
                    return MiddlewareResult::Error(err_str);
                }
            }
        }

        MiddlewareResult::Continue(last_value)
    }

    /// Return the number of middleware in the pipeline.
    pub fn len(&self) -> usize {
        self.middlewares.len()
    }

    /// Return `true` if the pipeline contains no middleware.
    pub fn is_empty(&self) -> bool {
        self.middlewares.is_empty()
    }

    /// Return the names of all middleware in the pipeline (insertion order).
    pub fn middleware_names(&self) -> Vec<&str> {
        self.middlewares.iter().map(|m| m.name()).collect()
    }
}

impl Default for MiddlewarePipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LoggingMiddleware
// ---------------------------------------------------------------------------

/// Pipeline middleware that logs each request to an internal buffer.
///
/// Uses interior mutability ([`RefCell`]) to allow logging from an immutable
/// reference, which is required by the [`PipelineMiddleware`] trait.
pub struct LoggingMiddleware {
    name: String,
    logs: RefCell<Vec<String>>,
}

impl LoggingMiddleware {
    /// Create a new logging middleware with the name `"logging"`.
    pub fn new() -> Self {
        Self {
            name: "logging".to_string(),
            logs: RefCell::new(Vec::new()),
        }
    }

    /// Return a copy of the internal log entries.
    pub fn logs(&self) -> Vec<String> {
        self.logs.borrow().clone()
    }

    /// Clear all recorded log entries.
    pub fn clear(&self) {
        self.logs.borrow_mut().clear();
    }
}

impl Default for LoggingMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineMiddleware for LoggingMiddleware {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&self, ctx: &mut MiddlewareContext) -> MiddlewareResult {
        let entry = format!(
            "[{}] agent={} request={}",
            ctx.timestamp, ctx.agent_id, ctx.request
        );
        self.logs.borrow_mut().push(entry);
        MiddlewareResult::Continue(ctx.request.clone())
    }

    fn on_error(&self, ctx: &MiddlewareContext, error: &str) -> MiddlewareResult {
        let entry = format!(
            "[{}] agent={} ERROR: {}",
            ctx.timestamp, ctx.agent_id, error
        );
        self.logs.borrow_mut().push(entry);
        MiddlewareResult::Error(error.to_string())
    }
}

// ---------------------------------------------------------------------------
// ValidationMiddleware
// ---------------------------------------------------------------------------

/// Pipeline middleware that validates required fields are present in the request.
///
/// If the request is a JSON object, each required field is checked. If any
/// required field is missing, [`MiddlewareResult::Error`] is returned.
pub struct ValidationMiddleware {
    name: String,
    required_fields: Vec<String>,
}

impl ValidationMiddleware {
    /// Create a new validation middleware with no required fields.
    pub fn new() -> Self {
        Self {
            name: "validation".to_string(),
            required_fields: Vec::new(),
        }
    }

    /// Add a required field.
    pub fn require_field(mut self, name: impl Into<String>) -> Self {
        self.required_fields.push(name.into());
        self
    }
}

impl Default for ValidationMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineMiddleware for ValidationMiddleware {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&self, ctx: &mut MiddlewareContext) -> MiddlewareResult {
        if let Some(obj) = ctx.request.as_object() {
            for field in &self.required_fields {
                if !obj.contains_key(field) {
                    return MiddlewareResult::Error(format!(
                        "missing required field: '{}'",
                        field
                    ));
                }
            }
            MiddlewareResult::Continue(ctx.request.clone())
        } else if self.required_fields.is_empty() {
            MiddlewareResult::Continue(ctx.request.clone())
        } else {
            MiddlewareResult::Error(
                "request is not a JSON object; cannot validate fields".to_string(),
            )
        }
    }
}

// ---------------------------------------------------------------------------
// TransformMiddleware
// ---------------------------------------------------------------------------

/// Pipeline middleware that applies a transformation function to the request value.
pub struct TransformMiddleware {
    name: String,
    transform: Box<dyn Fn(Value) -> Value>,
}

impl TransformMiddleware {
    /// Create a new transform middleware with an identity transform.
    pub fn new() -> Self {
        Self {
            name: "transform".to_string(),
            transform: Box::new(|v| v),
        }
    }

    /// Set the transformation function.
    pub fn with_transform(mut self, f: Box<dyn Fn(Value) -> Value>) -> Self {
        self.transform = f;
        self
    }
}

impl Default for TransformMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineMiddleware for TransformMiddleware {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&self, ctx: &mut MiddlewareContext) -> MiddlewareResult {
        let transformed = (self.transform)(ctx.request.clone());
        ctx.request = transformed.clone();
        MiddlewareResult::Continue(transformed)
    }
}

// ---------------------------------------------------------------------------
// RateLimitMiddleware
// ---------------------------------------------------------------------------

/// Simple call-count-based rate limiter.
///
/// Tracks the number of calls and returns [`MiddlewareResult::Halt`] once the
/// maximum number of calls has been reached.
pub struct RateLimitMiddleware {
    name: String,
    max_calls: usize,
    call_count: RefCell<usize>,
}

impl RateLimitMiddleware {
    /// Create a new rate limiter allowing up to `max_calls` invocations.
    pub fn new(max_calls: usize) -> Self {
        Self {
            name: "rate_limit".to_string(),
            max_calls,
            call_count: RefCell::new(0),
        }
    }

    /// Return the number of remaining calls before the limit is reached.
    pub fn remaining(&self) -> usize {
        let count = *self.call_count.borrow();
        self.max_calls.saturating_sub(count)
    }

    /// Reset the call counter to zero.
    pub fn reset(&self) {
        *self.call_count.borrow_mut() = 0;
    }
}

impl PipelineMiddleware for RateLimitMiddleware {
    fn name(&self) -> &str {
        &self.name
    }

    fn process(&self, ctx: &mut MiddlewareContext) -> MiddlewareResult {
        let mut count = self.call_count.borrow_mut();
        if *count >= self.max_calls {
            return MiddlewareResult::Halt(json!({
                "error": "rate limit exceeded",
                "max_calls": self.max_calls,
            }));
        }
        *count += 1;
        MiddlewareResult::Continue(ctx.request.clone())
    }
}

// ---------------------------------------------------------------------------
// MiddlewareStack
// ---------------------------------------------------------------------------

/// Named collection of middleware pipelines.
///
/// Allows registering multiple pipelines under distinct names and executing
/// them independently.
pub struct MiddlewareStack {
    pipelines: HashMap<String, MiddlewarePipeline>,
}

impl MiddlewareStack {
    /// Create an empty stack.
    pub fn new() -> Self {
        Self {
            pipelines: HashMap::new(),
        }
    }

    /// Register a pipeline under the given name, replacing any existing pipeline
    /// with the same name.
    pub fn register(&mut self, name: impl Into<String>, pipeline: MiddlewarePipeline) {
        self.pipelines.insert(name.into(), pipeline);
    }

    /// Get a reference to the pipeline with the given name.
    pub fn get(&self, name: &str) -> Option<&MiddlewarePipeline> {
        self.pipelines.get(name)
    }

    /// Execute the named pipeline against the given context.
    ///
    /// Returns `MiddlewareResult::Error` if no pipeline with the given name
    /// exists.
    pub fn execute(&self, name: &str, ctx: &mut MiddlewareContext) -> MiddlewareResult {
        match self.pipelines.get(name) {
            Some(pipeline) => pipeline.execute(ctx),
            None => MiddlewareResult::Error(format!("pipeline '{}' not found", name)),
        }
    }

    /// Return the names of all registered pipelines.
    pub fn pipeline_names(&self) -> Vec<&str> {
        self.pipelines.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for MiddlewareStack {
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

    // -- Helper middleware for testing --

    /// A simple pass-through middleware with configurable priority.
    struct PassThrough {
        name: String,
        priority: i32,
    }

    impl PassThrough {
        fn new(name: &str, priority: i32) -> Self {
            Self {
                name: name.to_string(),
                priority,
            }
        }
    }

    impl PipelineMiddleware for PassThrough {
        fn name(&self) -> &str {
            &self.name
        }

        fn priority(&self) -> i32 {
            self.priority
        }

        fn process(&self, ctx: &mut MiddlewareContext) -> MiddlewareResult {
            // Append our name to metadata to track execution order.
            let mut order: Vec<String> = ctx
                .get_metadata("order")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            order.push(self.name.clone());
            ctx.set_metadata("order", json!(order));
            MiddlewareResult::Continue(ctx.request.clone())
        }
    }

    /// A middleware that always halts.
    struct AlwaysHalt {
        name: String,
        priority: i32,
    }

    impl AlwaysHalt {
        fn new(name: &str, priority: i32) -> Self {
            Self {
                name: name.to_string(),
                priority,
            }
        }
    }

    impl PipelineMiddleware for AlwaysHalt {
        fn name(&self) -> &str {
            &self.name
        }

        fn priority(&self) -> i32 {
            self.priority
        }

        fn process(&self, _ctx: &mut MiddlewareContext) -> MiddlewareResult {
            MiddlewareResult::Halt(json!({"halted_by": self.name}))
        }
    }

    /// A middleware that always errors.
    struct AlwaysError {
        name: String,
        priority: i32,
    }

    impl AlwaysError {
        fn new(name: &str, priority: i32) -> Self {
            Self {
                name: name.to_string(),
                priority,
            }
        }
    }

    impl PipelineMiddleware for AlwaysError {
        fn name(&self) -> &str {
            &self.name
        }

        fn priority(&self) -> i32 {
            self.priority
        }

        fn process(&self, _ctx: &mut MiddlewareContext) -> MiddlewareResult {
            MiddlewareResult::Error(format!("error from {}", self.name))
        }
    }

    /// A middleware that tracks on_error calls via RefCell.
    struct ErrorTracker {
        name: String,
        priority: i32,
        error_calls: RefCell<Vec<String>>,
    }

    impl ErrorTracker {
        fn new(name: &str, priority: i32) -> Self {
            Self {
                name: name.to_string(),
                priority,
                error_calls: RefCell::new(Vec::new()),
            }
        }
    }

    impl PipelineMiddleware for ErrorTracker {
        fn name(&self) -> &str {
            &self.name
        }

        fn priority(&self) -> i32 {
            self.priority
        }

        fn process(&self, ctx: &mut MiddlewareContext) -> MiddlewareResult {
            let mut order: Vec<String> = ctx
                .get_metadata("order")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();
            order.push(self.name.clone());
            ctx.set_metadata("order", json!(order));
            MiddlewareResult::Continue(ctx.request.clone())
        }

        fn on_error(&self, _ctx: &MiddlewareContext, error: &str) -> MiddlewareResult {
            self.error_calls.borrow_mut().push(error.to_string());
            MiddlewareResult::Error(error.to_string())
        }
    }

    fn make_ctx() -> MiddlewareContext {
        MiddlewareContext::new(json!({"key": "value"}), "agent-1", "2026-01-01T00:00:00Z")
    }

    // ===================================================================
    // MiddlewareContext tests
    // ===================================================================

    #[test]
    fn test_context_new() {
        let ctx = make_ctx();
        assert_eq!(ctx.agent_id, "agent-1");
        assert_eq!(ctx.timestamp, "2026-01-01T00:00:00Z");
        assert_eq!(ctx.request, json!({"key": "value"}));
        assert!(ctx.metadata.is_empty());
    }

    #[test]
    fn test_context_metadata_get_set() {
        let mut ctx = make_ctx();
        assert!(ctx.get_metadata("foo").is_none());
        ctx.set_metadata("foo", json!(42));
        assert_eq!(ctx.get_metadata("foo"), Some(&json!(42)));
    }

    #[test]
    fn test_context_metadata_overwrite() {
        let mut ctx = make_ctx();
        ctx.set_metadata("k", json!("a"));
        ctx.set_metadata("k", json!("b"));
        assert_eq!(ctx.get_metadata("k"), Some(&json!("b")));
    }

    #[test]
    fn test_context_to_json() {
        let mut ctx = make_ctx();
        ctx.set_metadata("m", json!(true));
        let j = ctx.to_json();
        assert_eq!(j["agent_id"], "agent-1");
        assert_eq!(j["timestamp"], "2026-01-01T00:00:00Z");
        assert_eq!(j["request"]["key"], "value");
        assert_eq!(j["metadata"]["m"], true);
    }

    #[test]
    fn test_context_clone() {
        let mut ctx = make_ctx();
        ctx.set_metadata("x", json!(1));
        let ctx2 = ctx.clone();
        assert_eq!(ctx2.agent_id, ctx.agent_id);
        assert_eq!(ctx2.get_metadata("x"), Some(&json!(1)));
    }

    // ===================================================================
    // MiddlewareResult tests
    // ===================================================================

    #[test]
    fn test_result_continue() {
        let r = MiddlewareResult::Continue(json!(1));
        assert!(r.is_continue());
        assert!(!r.is_halt());
        assert!(!r.is_error());
    }

    #[test]
    fn test_result_halt() {
        let r = MiddlewareResult::Halt(json!(2));
        assert!(!r.is_continue());
        assert!(r.is_halt());
        assert!(!r.is_error());
    }

    #[test]
    fn test_result_error() {
        let r = MiddlewareResult::Error("oops".into());
        assert!(!r.is_continue());
        assert!(!r.is_halt());
        assert!(r.is_error());
    }

    #[test]
    fn test_result_into_value_continue() {
        let r = MiddlewareResult::Continue(json!({"a": 1}));
        assert_eq!(r.into_value(), json!({"a": 1}));
    }

    #[test]
    fn test_result_into_value_halt() {
        let r = MiddlewareResult::Halt(json!("stopped"));
        assert_eq!(r.into_value(), json!("stopped"));
    }

    #[test]
    fn test_result_into_value_error() {
        let r = MiddlewareResult::Error("bad".into());
        assert_eq!(r.into_value(), json!("bad"));
    }

    // ===================================================================
    // MiddlewarePipeline tests
    // ===================================================================

    #[test]
    fn test_pipeline_new_is_empty() {
        let p = MiddlewarePipeline::new();
        assert!(p.is_empty());
        assert_eq!(p.len(), 0);
    }

    #[test]
    fn test_pipeline_default() {
        let p = MiddlewarePipeline::default();
        assert!(p.is_empty());
    }

    #[test]
    fn test_pipeline_add_and_len() {
        let mut p = MiddlewarePipeline::new();
        p.add(Box::new(PassThrough::new("a", 0)));
        p.add(Box::new(PassThrough::new("b", 0)));
        assert_eq!(p.len(), 2);
        assert!(!p.is_empty());
    }

    #[test]
    fn test_pipeline_middleware_names() {
        let mut p = MiddlewarePipeline::new();
        p.add(Box::new(PassThrough::new("alpha", 0)));
        p.add(Box::new(PassThrough::new("beta", 0)));
        let names = p.middleware_names();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_pipeline_empty_execute() {
        let p = MiddlewarePipeline::new();
        let mut ctx = make_ctx();
        let result = p.execute(&mut ctx);
        assert!(result.is_continue());
        assert_eq!(result.into_value(), Value::Null);
    }

    #[test]
    fn test_pipeline_execution_order_by_priority() {
        let mut p = MiddlewarePipeline::new();
        p.add(Box::new(PassThrough::new("third", 30)));
        p.add(Box::new(PassThrough::new("first", 10)));
        p.add(Box::new(PassThrough::new("second", 20)));
        let mut ctx = make_ctx();
        let result = p.execute(&mut ctx);
        assert!(result.is_continue());
        let order: Vec<String> =
            serde_json::from_value(ctx.get_metadata("order").unwrap().clone()).unwrap();
        assert_eq!(order, vec!["first", "second", "third"]);
    }

    #[test]
    fn test_pipeline_halt_stops_execution() {
        let mut p = MiddlewarePipeline::new();
        p.add(Box::new(PassThrough::new("before", 1)));
        p.add(Box::new(AlwaysHalt::new("halter", 2)));
        p.add(Box::new(PassThrough::new("after", 3)));
        let mut ctx = make_ctx();
        let result = p.execute(&mut ctx);
        assert!(result.is_halt());
        // "after" should not have executed.
        let order: Vec<String> =
            serde_json::from_value(ctx.get_metadata("order").unwrap().clone()).unwrap();
        assert_eq!(order, vec!["before"]);
    }

    #[test]
    fn test_pipeline_error_stops_execution() {
        let mut p = MiddlewarePipeline::new();
        p.add(Box::new(PassThrough::new("before", 1)));
        p.add(Box::new(AlwaysError::new("errorer", 2)));
        p.add(Box::new(PassThrough::new("after", 3)));
        let mut ctx = make_ctx();
        let result = p.execute(&mut ctx);
        assert!(result.is_error());
        let order: Vec<String> =
            serde_json::from_value(ctx.get_metadata("order").unwrap().clone()).unwrap();
        assert_eq!(order, vec!["before"]);
    }

    #[test]
    fn test_pipeline_all_halt() {
        let mut p = MiddlewarePipeline::new();
        p.add(Box::new(AlwaysHalt::new("h1", 1)));
        p.add(Box::new(AlwaysHalt::new("h2", 2)));
        let mut ctx = make_ctx();
        let result = p.execute(&mut ctx);
        assert!(result.is_halt());
        assert_eq!(result.into_value()["halted_by"], "h1");
    }

    #[test]
    fn test_pipeline_all_error() {
        let mut p = MiddlewarePipeline::new();
        p.add(Box::new(AlwaysError::new("e1", 1)));
        p.add(Box::new(AlwaysError::new("e2", 2)));
        let mut ctx = make_ctx();
        let result = p.execute(&mut ctx);
        assert!(result.is_error());
    }

    #[test]
    fn test_pipeline_on_error_callback_invoked() {
        let mut p = MiddlewarePipeline::new();
        p.add(Box::new(ErrorTracker::new("tracker", 1)));
        p.add(Box::new(AlwaysError::new("err", 2)));
        let mut ctx = make_ctx();
        let result = p.execute(&mut ctx);
        assert!(result.is_error());
        // ErrorTracker's on_error was called by the pipeline. We verify
        // execution order shows tracker ran before the error.
        let order: Vec<String> =
            serde_json::from_value(ctx.get_metadata("order").unwrap().clone()).unwrap();
        assert_eq!(order, vec!["tracker"]);
    }

    #[test]
    fn test_pipeline_single_middleware() {
        let mut p = MiddlewarePipeline::new();
        p.add(Box::new(PassThrough::new("only", 0)));
        let mut ctx = make_ctx();
        let result = p.execute(&mut ctx);
        assert!(result.is_continue());
    }

    // ===================================================================
    // LoggingMiddleware tests
    // ===================================================================

    #[test]
    fn test_logging_new() {
        let lm = LoggingMiddleware::new();
        assert_eq!(lm.name(), "logging");
        assert!(lm.logs().is_empty());
    }

    #[test]
    fn test_logging_records_request() {
        let lm = LoggingMiddleware::new();
        let mut ctx = make_ctx();
        let result = lm.process(&mut ctx);
        assert!(result.is_continue());
        let logs = lm.logs();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("agent-1"));
        assert!(logs[0].contains("2026-01-01T00:00:00Z"));
    }

    #[test]
    fn test_logging_multiple_requests() {
        let lm = LoggingMiddleware::new();
        let mut ctx = make_ctx();
        lm.process(&mut ctx);
        lm.process(&mut ctx);
        lm.process(&mut ctx);
        assert_eq!(lm.logs().len(), 3);
    }

    #[test]
    fn test_logging_clear() {
        let lm = LoggingMiddleware::new();
        let mut ctx = make_ctx();
        lm.process(&mut ctx);
        assert_eq!(lm.logs().len(), 1);
        lm.clear();
        assert!(lm.logs().is_empty());
    }

    #[test]
    fn test_logging_on_error_records() {
        let lm = LoggingMiddleware::new();
        let ctx = make_ctx();
        lm.on_error(&ctx, "something went wrong");
        let logs = lm.logs();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("ERROR"));
        assert!(logs[0].contains("something went wrong"));
    }

    #[test]
    fn test_logging_in_pipeline() {
        let mut p = MiddlewarePipeline::new();
        let lm = LoggingMiddleware::new();
        p.add(Box::new(lm));
        let mut ctx = make_ctx();
        let result = p.execute(&mut ctx);
        assert!(result.is_continue());
    }

    // ===================================================================
    // ValidationMiddleware tests
    // ===================================================================

    #[test]
    fn test_validation_no_fields_passes() {
        let vm = ValidationMiddleware::new();
        let mut ctx = make_ctx();
        let result = vm.process(&mut ctx);
        assert!(result.is_continue());
    }

    #[test]
    fn test_validation_required_field_present() {
        let vm = ValidationMiddleware::new().require_field("key");
        let mut ctx = make_ctx();
        let result = vm.process(&mut ctx);
        assert!(result.is_continue());
    }

    #[test]
    fn test_validation_required_field_missing() {
        let vm = ValidationMiddleware::new().require_field("missing_field");
        let mut ctx = make_ctx();
        let result = vm.process(&mut ctx);
        assert!(result.is_error());
        if let MiddlewareResult::Error(e) = result {
            assert!(e.contains("missing_field"));
        }
    }

    #[test]
    fn test_validation_multiple_required_fields() {
        let vm = ValidationMiddleware::new()
            .require_field("key")
            .require_field("other");
        let mut ctx = make_ctx();
        let result = vm.process(&mut ctx);
        assert!(result.is_error());
        if let MiddlewareResult::Error(e) = result {
            assert!(e.contains("other"));
        }
    }

    #[test]
    fn test_validation_non_object_with_fields() {
        let vm = ValidationMiddleware::new().require_field("x");
        let mut ctx = MiddlewareContext::new(json!("string_value"), "a", "t");
        let result = vm.process(&mut ctx);
        assert!(result.is_error());
    }

    #[test]
    fn test_validation_non_object_no_fields() {
        let vm = ValidationMiddleware::new();
        let mut ctx = MiddlewareContext::new(json!(42), "a", "t");
        let result = vm.process(&mut ctx);
        assert!(result.is_continue());
    }

    #[test]
    fn test_validation_all_fields_present() {
        let vm = ValidationMiddleware::new()
            .require_field("a")
            .require_field("b");
        let mut ctx = MiddlewareContext::new(json!({"a": 1, "b": 2, "c": 3}), "ag", "ts");
        let result = vm.process(&mut ctx);
        assert!(result.is_continue());
    }

    // ===================================================================
    // TransformMiddleware tests
    // ===================================================================

    #[test]
    fn test_transform_identity() {
        let tm = TransformMiddleware::new();
        let mut ctx = make_ctx();
        let result = tm.process(&mut ctx);
        assert!(result.is_continue());
        assert_eq!(result.into_value(), json!({"key": "value"}));
    }

    #[test]
    fn test_transform_custom_function() {
        let tm = TransformMiddleware::new().with_transform(Box::new(|mut v| {
            if let Some(obj) = v.as_object_mut() {
                obj.insert("added".to_string(), json!(true));
            }
            v
        }));
        let mut ctx = make_ctx();
        let result = tm.process(&mut ctx);
        assert!(result.is_continue());
        let val = result.into_value();
        assert_eq!(val["added"], true);
        assert_eq!(val["key"], "value");
    }

    #[test]
    fn test_transform_modifies_context_request() {
        let tm = TransformMiddleware::new().with_transform(Box::new(|_| json!({"new": "data"})));
        let mut ctx = make_ctx();
        tm.process(&mut ctx);
        assert_eq!(ctx.request, json!({"new": "data"}));
    }

    #[test]
    fn test_transform_chained_in_pipeline() {
        let mut p = MiddlewarePipeline::new();
        let t1 = TransformMiddleware {
            name: "t1".to_string(),
            transform: Box::new(|mut v| {
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("step1".to_string(), json!(true));
                }
                v
            }),
        };
        let t2 = TransformMiddleware {
            name: "t2".to_string(),
            transform: Box::new(|mut v| {
                if let Some(obj) = v.as_object_mut() {
                    obj.insert("step2".to_string(), json!(true));
                }
                v
            }),
        };
        p.add(Box::new(t1));
        p.add(Box::new(t2));
        let mut ctx = make_ctx();
        let result = p.execute(&mut ctx);
        assert!(result.is_continue());
        let val = result.into_value();
        assert_eq!(val["step1"], true);
        assert_eq!(val["step2"], true);
        assert_eq!(val["key"], "value");
    }

    // ===================================================================
    // RateLimitMiddleware tests
    // ===================================================================

    #[test]
    fn test_rate_limit_allows_under_limit() {
        let rl = RateLimitMiddleware::new(3);
        let mut ctx = make_ctx();
        assert!(rl.process(&mut ctx).is_continue());
        assert!(rl.process(&mut ctx).is_continue());
        assert!(rl.process(&mut ctx).is_continue());
        assert_eq!(rl.remaining(), 0);
    }

    #[test]
    fn test_rate_limit_halts_at_limit() {
        let rl = RateLimitMiddleware::new(2);
        let mut ctx = make_ctx();
        rl.process(&mut ctx);
        rl.process(&mut ctx);
        let result = rl.process(&mut ctx);
        assert!(result.is_halt());
    }

    #[test]
    fn test_rate_limit_remaining() {
        let rl = RateLimitMiddleware::new(5);
        assert_eq!(rl.remaining(), 5);
        let mut ctx = make_ctx();
        rl.process(&mut ctx);
        assert_eq!(rl.remaining(), 4);
        rl.process(&mut ctx);
        assert_eq!(rl.remaining(), 3);
    }

    #[test]
    fn test_rate_limit_reset() {
        let rl = RateLimitMiddleware::new(2);
        let mut ctx = make_ctx();
        rl.process(&mut ctx);
        rl.process(&mut ctx);
        assert_eq!(rl.remaining(), 0);
        rl.reset();
        assert_eq!(rl.remaining(), 2);
        assert!(rl.process(&mut ctx).is_continue());
    }

    #[test]
    fn test_rate_limit_zero_max() {
        let rl = RateLimitMiddleware::new(0);
        let mut ctx = make_ctx();
        let result = rl.process(&mut ctx);
        assert!(result.is_halt());
    }

    #[test]
    fn test_rate_limit_halt_value_contains_info() {
        let rl = RateLimitMiddleware::new(1);
        let mut ctx = make_ctx();
        rl.process(&mut ctx);
        let result = rl.process(&mut ctx);
        let val = result.into_value();
        assert_eq!(val["max_calls"], 1);
    }

    // ===================================================================
    // MiddlewareStack tests
    // ===================================================================

    #[test]
    fn test_stack_new_is_empty() {
        let s = MiddlewareStack::new();
        assert!(s.pipeline_names().is_empty());
    }

    #[test]
    fn test_stack_default() {
        let s = MiddlewareStack::default();
        assert!(s.pipeline_names().is_empty());
    }

    #[test]
    fn test_stack_register_and_get() {
        let mut s = MiddlewareStack::new();
        let p = MiddlewarePipeline::new();
        s.register("test", p);
        assert!(s.get("test").is_some());
        assert!(s.get("other").is_none());
    }

    #[test]
    fn test_stack_pipeline_names() {
        let mut s = MiddlewareStack::new();
        s.register("alpha", MiddlewarePipeline::new());
        s.register("beta", MiddlewarePipeline::new());
        let mut names = s.pipeline_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_stack_execute_existing() {
        let mut s = MiddlewareStack::new();
        let mut p = MiddlewarePipeline::new();
        p.add(Box::new(PassThrough::new("pt", 0)));
        s.register("main", p);
        let mut ctx = make_ctx();
        let result = s.execute("main", &mut ctx);
        assert!(result.is_continue());
    }

    #[test]
    fn test_stack_execute_missing() {
        let s = MiddlewareStack::new();
        let mut ctx = make_ctx();
        let result = s.execute("nonexistent", &mut ctx);
        assert!(result.is_error());
    }

    #[test]
    fn test_stack_replace_pipeline() {
        let mut s = MiddlewareStack::new();
        let p1 = MiddlewarePipeline::new();
        s.register("x", p1);
        assert_eq!(s.get("x").unwrap().len(), 0);

        let mut p2 = MiddlewarePipeline::new();
        p2.add(Box::new(PassThrough::new("new", 0)));
        s.register("x", p2);
        assert_eq!(s.get("x").unwrap().len(), 1);
    }

    // ===================================================================
    // Integration / edge-case tests
    // ===================================================================

    #[test]
    fn test_mixed_pipeline_validation_then_transform() {
        let mut p = MiddlewarePipeline::new();
        let vm = ValidationMiddleware::new().require_field("key");
        let tm = TransformMiddleware::new().with_transform(Box::new(|mut v| {
            if let Some(obj) = v.as_object_mut() {
                obj.insert("validated".to_string(), json!(true));
            }
            v
        }));
        p.add(Box::new(vm));
        p.add(Box::new(tm));
        let mut ctx = make_ctx();
        let result = p.execute(&mut ctx);
        assert!(result.is_continue());
        assert_eq!(result.into_value()["validated"], true);
    }

    #[test]
    fn test_mixed_pipeline_validation_fails_before_transform() {
        let mut p = MiddlewarePipeline::new();
        let vm = ValidationMiddleware::new().require_field("nonexistent");
        let tm = TransformMiddleware::new().with_transform(Box::new(|_| {
            panic!("should not reach transform");
        }));
        p.add(Box::new(vm));
        p.add(Box::new(tm));
        let mut ctx = make_ctx();
        let result = p.execute(&mut ctx);
        assert!(result.is_error());
    }

    #[test]
    fn test_rate_limit_in_pipeline() {
        let mut p = MiddlewarePipeline::new();
        p.add(Box::new(RateLimitMiddleware::new(2)));
        p.add(Box::new(PassThrough::new("pt", 1)));
        let mut ctx = make_ctx();
        assert!(p.execute(&mut ctx).is_continue());
        assert!(p.execute(&mut ctx).is_continue());
        assert!(p.execute(&mut ctx).is_halt());
    }

    #[test]
    fn test_on_error_default_returns_error() {
        let pt = PassThrough::new("test", 0);
        let ctx = make_ctx();
        let result = pt.on_error(&ctx, "some error");
        assert!(result.is_error());
        if let MiddlewareResult::Error(e) = result {
            assert_eq!(e, "some error");
        }
    }

    #[test]
    fn test_middleware_priority_default_is_zero() {
        let vm = ValidationMiddleware::new();
        assert_eq!(vm.priority(), 0);
    }

    #[test]
    fn test_context_empty_request() {
        let ctx = MiddlewareContext::new(json!(null), "a", "t");
        assert_eq!(ctx.request, Value::Null);
    }

    #[test]
    fn test_pipeline_same_priority_preserves_insertion_order() {
        let mut p = MiddlewarePipeline::new();
        p.add(Box::new(PassThrough::new("a", 0)));
        p.add(Box::new(PassThrough::new("b", 0)));
        p.add(Box::new(PassThrough::new("c", 0)));
        let mut ctx = make_ctx();
        p.execute(&mut ctx);
        let order: Vec<String> =
            serde_json::from_value(ctx.get_metadata("order").unwrap().clone()).unwrap();
        assert_eq!(order, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_pipeline_negative_priority_runs_first() {
        let mut p = MiddlewarePipeline::new();
        p.add(Box::new(PassThrough::new("normal", 0)));
        p.add(Box::new(PassThrough::new("early", -10)));
        p.add(Box::new(PassThrough::new("late", 10)));
        let mut ctx = make_ctx();
        p.execute(&mut ctx);
        let order: Vec<String> =
            serde_json::from_value(ctx.get_metadata("order").unwrap().clone()).unwrap();
        assert_eq!(order, vec!["early", "normal", "late"]);
    }

    #[test]
    fn test_logging_default() {
        let lm = LoggingMiddleware::default();
        assert_eq!(lm.name(), "logging");
    }

    #[test]
    fn test_validation_default() {
        let vm = ValidationMiddleware::default();
        assert_eq!(vm.name(), "validation");
    }

    #[test]
    fn test_transform_default() {
        let tm = TransformMiddleware::default();
        assert_eq!(tm.name(), "transform");
    }
}
