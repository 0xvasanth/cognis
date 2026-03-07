//! Scoped callback dispatching for runnables.
//!
//! This module provides a callback system that mirrors LangChain's callback
//! manager, enabling observation and instrumentation of runnable execution.
//! Callbacks are dispatched within scopes that track parent/child relationships,
//! and [`ScopeGuard`] provides RAII-based automatic start/end event dispatching.

use std::sync::Arc;

use serde_json::Value;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// CallbackEvent
// ---------------------------------------------------------------------------

/// An event that can be dispatched to callback handlers during runnable
/// execution.
#[derive(Debug, Clone)]
pub enum CallbackEvent {
    /// The runnable has started executing.
    OnStart {
        /// The input value passed to the runnable.
        input: Value,
    },
    /// The runnable has finished executing successfully.
    OnEnd {
        /// The output value produced by the runnable.
        output: Value,
    },
    /// The runnable encountered an error.
    OnError {
        /// A description of the error that occurred.
        error: String,
    },
    /// The runnable produced a text chunk (e.g. streaming).
    OnText {
        /// The text content emitted.
        text: String,
    },
    /// The runnable is retrying after a failure.
    OnRetry {
        /// The retry attempt number.
        attempt: u32,
    },
    /// A custom, user-defined event.
    Custom {
        /// The name of the custom event.
        name: String,
        /// Arbitrary data associated with the event.
        data: Value,
    },
}

impl CallbackEvent {
    /// Returns a string label identifying the type of this event.
    pub fn event_type(&self) -> &str {
        match self {
            CallbackEvent::OnStart { .. } => "on_start",
            CallbackEvent::OnEnd { .. } => "on_end",
            CallbackEvent::OnError { .. } => "on_error",
            CallbackEvent::OnText { .. } => "on_text",
            CallbackEvent::OnRetry { .. } => "on_retry",
            CallbackEvent::Custom { .. } => "custom",
        }
    }

    /// Serialize this event to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        match self {
            CallbackEvent::OnStart { input } => {
                serde_json::json!({ "event_type": "on_start", "input": input })
            }
            CallbackEvent::OnEnd { output } => {
                serde_json::json!({ "event_type": "on_end", "output": output })
            }
            CallbackEvent::OnError { error } => {
                serde_json::json!({ "event_type": "on_error", "error": error })
            }
            CallbackEvent::OnText { text } => {
                serde_json::json!({ "event_type": "on_text", "text": text })
            }
            CallbackEvent::OnRetry { attempt } => {
                serde_json::json!({ "event_type": "on_retry", "attempt": attempt })
            }
            CallbackEvent::Custom { name, data } => {
                serde_json::json!({ "event_type": "custom", "name": name, "data": data })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CallbackScope
// ---------------------------------------------------------------------------

/// Identifies the scope in which callback events are dispatched.
///
/// Scopes form a tree: each scope has a unique `run_id` and an optional
/// `parent_run_id` linking it to its parent scope.
#[derive(Debug, Clone)]
pub struct CallbackScope {
    /// Unique identifier for this run.
    pub run_id: String,
    /// Identifier of the parent run, if any.
    pub parent_run_id: Option<String>,
    /// Human-readable name of this scope (e.g. the runnable name).
    pub name: String,
    /// The type of run (e.g. "chain", "tool", "llm").
    pub run_type: String,
}

impl CallbackScope {
    /// Create a new root scope with an auto-generated `run_id`.
    pub fn new(name: impl Into<String>, run_type: impl Into<String>) -> Self {
        Self {
            run_id: Uuid::new_v4().to_string(),
            parent_run_id: None,
            name: name.into(),
            run_type: run_type.into(),
        }
    }

    /// Create a child scope whose `parent_run_id` points to this scope's
    /// `run_id`.
    pub fn child(&self, name: impl Into<String>, run_type: impl Into<String>) -> Self {
        Self {
            run_id: Uuid::new_v4().to_string(),
            parent_run_id: Some(self.run_id.clone()),
            name: name.into(),
            run_type: run_type.into(),
        }
    }
}

// ---------------------------------------------------------------------------
// Handler type alias
// ---------------------------------------------------------------------------

/// A callback handler function that receives a scope and an event.
pub type CallbackHandlerFn = Box<dyn Fn(&CallbackScope, &CallbackEvent) + Send + Sync>;

// ---------------------------------------------------------------------------
// ScopedCallbackConfig
// ---------------------------------------------------------------------------

/// Configuration holding a list of callback handler functions.
///
/// Built using the builder pattern via [`ScopedCallbackConfig::new`] and
/// [`ScopedCallbackConfig::with_handler`].
pub struct ScopedCallbackConfig {
    handlers: Vec<Arc<CallbackHandlerFn>>,
}

impl ScopedCallbackConfig {
    /// Create a new, empty callback configuration.
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    /// Add a handler to this configuration, returning `self` for chaining.
    pub fn with_handler<F>(mut self, handler: F) -> Self
    where
        F: Fn(&CallbackScope, &CallbackEvent) + Send + Sync + 'static,
    {
        self.handlers.push(Arc::new(Box::new(handler)));
        self
    }

    /// Returns the number of handlers in this configuration.
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Returns a slice of the handler references.
    pub fn handlers(&self) -> &[Arc<CallbackHandlerFn>] {
        &self.handlers
    }
}

impl Default for ScopedCallbackConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ScopedCallbackConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScopedCallbackConfig")
            .field("handler_count", &self.handlers.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// CallbackDispatcher
// ---------------------------------------------------------------------------

/// Dispatches [`CallbackEvent`]s to all registered handlers within a
/// [`CallbackScope`].
pub struct CallbackDispatcher {
    handlers: Vec<Arc<CallbackHandlerFn>>,
}

impl CallbackDispatcher {
    /// Create a new dispatcher with no handlers.
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    /// Create a dispatcher from a [`ScopedCallbackConfig`].
    pub fn from_config(config: &ScopedCallbackConfig) -> Self {
        Self {
            handlers: config.handlers.clone(),
        }
    }

    /// Add a handler to this dispatcher.
    pub fn add_handler<F>(&mut self, handler: F)
    where
        F: Fn(&CallbackScope, &CallbackEvent) + Send + Sync + 'static,
    {
        self.handlers.push(Arc::new(Box::new(handler)));
    }

    /// Dispatch an event to all registered handlers.
    pub fn dispatch(&self, scope: &CallbackScope, event: &CallbackEvent) {
        for handler in &self.handlers {
            handler(scope, event);
        }
    }

    /// Returns the number of registered handlers.
    pub fn handler_count(&self) -> usize {
        self.handlers.len()
    }

    /// Enter a scope, returning a [`ScopeGuard`] that dispatches `OnStart`
    /// immediately and `OnEnd` or `OnError` when dropped.
    pub fn enter_scope(self: &Arc<Self>, scope: CallbackScope, input: Value) -> ScopeGuard {
        // Dispatch OnStart immediately.
        self.dispatch(
            &scope,
            &CallbackEvent::OnStart {
                input: input.clone(),
            },
        );

        ScopeGuard {
            dispatcher: Arc::clone(self),
            scope,
            result: None,
            _input: input,
        }
    }
}

impl Default for CallbackDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for CallbackDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CallbackDispatcher")
            .field("handler_count", &self.handlers.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ScopeGuard
// ---------------------------------------------------------------------------

/// RAII guard that dispatches `OnStart` on creation and `OnEnd` or `OnError`
/// on drop.
///
/// Created via [`CallbackDispatcher::enter_scope`]. Call [`ScopeGuard::complete`]
/// to signal successful completion, or [`ScopeGuard::fail`] to signal an error.
/// If neither is called before the guard is dropped, an `OnError` event with
/// a message indicating an unfinished scope is dispatched.
pub struct ScopeGuard {
    dispatcher: Arc<CallbackDispatcher>,
    scope: CallbackScope,
    result: Option<ScopeResult>,
    _input: Value,
}

enum ScopeResult {
    Success(Value),
    Error(String),
}

impl ScopeGuard {
    /// Mark this scope as successfully completed with the given output.
    pub fn complete(&mut self, output: Value) {
        self.result = Some(ScopeResult::Success(output));
    }

    /// Mark this scope as having failed with the given error message.
    pub fn fail(&mut self, error: String) {
        self.result = Some(ScopeResult::Error(error));
    }

    /// Returns a reference to the scope associated with this guard.
    pub fn scope(&self) -> &CallbackScope {
        &self.scope
    }
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        match self.result.take() {
            Some(ScopeResult::Success(output)) => {
                self.dispatcher
                    .dispatch(&self.scope, &CallbackEvent::OnEnd { output });
            }
            Some(ScopeResult::Error(error)) => {
                self.dispatcher
                    .dispatch(&self.scope, &CallbackEvent::OnError { error });
            }
            None => {
                // Scope was neither completed nor failed -- emit an error.
                self.dispatcher.dispatch(
                    &self.scope,
                    &CallbackEvent::OnError {
                        error: "scope dropped without completion".to_string(),
                    },
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// RunnableWithCallbacks
// ---------------------------------------------------------------------------

/// Wraps a [`Runnable`](super::base::Runnable) and dispatches callback events
/// around its execution.
///
/// This struct demonstrates the pattern for augmenting runnables with callback
/// support. It does not implement the `Runnable` trait directly; instead it
/// provides a convenience `invoke` method that dispatches events.
pub struct RunnableWithCallbacks {
    /// The inner runnable (trait object).
    pub inner: Arc<dyn super::base::Runnable>,
    /// The dispatcher used to send callback events.
    pub dispatcher: Arc<CallbackDispatcher>,
}

impl RunnableWithCallbacks {
    /// Create a new wrapper around the given runnable with the provided
    /// dispatcher.
    pub fn new(inner: Arc<dyn super::base::Runnable>, dispatcher: Arc<CallbackDispatcher>) -> Self {
        Self { inner, dispatcher }
    }

    /// Invoke the wrapped runnable, dispatching `OnStart`, `OnEnd`/`OnError`
    /// callback events around the call.
    pub async fn invoke(
        &self,
        input: Value,
        config: Option<&super::config::RunnableConfig>,
    ) -> crate::error::Result<Value> {
        let scope = CallbackScope::new(self.inner.name(), "chain");
        let mut guard = self.dispatcher.enter_scope(scope, input.clone());

        match self.inner.invoke(input, config).await {
            Ok(output) => {
                guard.complete(output.clone());
                Ok(output)
            }
            Err(e) => {
                guard.fail(e.to_string());
                Err(e)
            }
        }
    }
}

impl std::fmt::Debug for RunnableWithCallbacks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnableWithCallbacks")
            .field("inner_name", &self.inner.name())
            .field("dispatcher", &self.dispatcher)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Create a [`ScopedCallbackConfig`] from a list of handler functions.
/// Type alias for scoped callback handler functions.
type ScopedCallbackHandler = Box<dyn Fn(&CallbackScope, &CallbackEvent) + Send + Sync>;

pub fn with_callbacks(handlers: Vec<ScopedCallbackHandler>) -> ScopedCallbackConfig {
    let mut config = ScopedCallbackConfig::new();
    for handler in handlers {
        config.handlers.push(Arc::new(handler));
    }
    config
}

/// Merge two [`ScopedCallbackConfig`]s into a single config containing
/// all handlers from both, with `a`'s handlers appearing first.
pub fn merge_callback_configs(
    a: &ScopedCallbackConfig,
    b: &ScopedCallbackConfig,
) -> ScopedCallbackConfig {
    let mut handlers = a.handlers.clone();
    handlers.extend(b.handlers.iter().cloned());
    ScopedCallbackConfig { handlers }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::Mutex;

    // -----------------------------------------------------------------------
    // CallbackEvent -- variant event_type() tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_event_on_start_type() {
        let event = CallbackEvent::OnStart { input: json!(1) };
        assert_eq!(event.event_type(), "on_start");
    }

    #[test]
    fn test_event_on_end_type() {
        let event = CallbackEvent::OnEnd { output: json!(2) };
        assert_eq!(event.event_type(), "on_end");
    }

    #[test]
    fn test_event_on_error_type() {
        let event = CallbackEvent::OnError {
            error: "fail".into(),
        };
        assert_eq!(event.event_type(), "on_error");
    }

    #[test]
    fn test_event_on_text_type() {
        let event = CallbackEvent::OnText {
            text: "hello".into(),
        };
        assert_eq!(event.event_type(), "on_text");
    }

    #[test]
    fn test_event_on_retry_type() {
        let event = CallbackEvent::OnRetry { attempt: 3 };
        assert_eq!(event.event_type(), "on_retry");
    }

    #[test]
    fn test_event_custom_type() {
        let event = CallbackEvent::Custom {
            name: "my_event".into(),
            data: json!(null),
        };
        assert_eq!(event.event_type(), "custom");
    }

    // -----------------------------------------------------------------------
    // CallbackEvent -- to_json() serialization tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_event_on_start_to_json() {
        let event = CallbackEvent::OnStart {
            input: json!({"key": "value"}),
        };
        let j = event.to_json();
        assert_eq!(j["event_type"], "on_start");
        assert_eq!(j["input"]["key"], "value");
    }

    #[test]
    fn test_event_on_end_to_json() {
        let event = CallbackEvent::OnEnd {
            output: json!([1, 2, 3]),
        };
        let j = event.to_json();
        assert_eq!(j["event_type"], "on_end");
        assert_eq!(j["output"], json!([1, 2, 3]));
    }

    #[test]
    fn test_event_on_error_to_json() {
        let event = CallbackEvent::OnError {
            error: "oops".into(),
        };
        let j = event.to_json();
        assert_eq!(j["event_type"], "on_error");
        assert_eq!(j["error"], "oops");
    }

    #[test]
    fn test_event_on_text_to_json() {
        let event = CallbackEvent::OnText {
            text: "chunk".into(),
        };
        let j = event.to_json();
        assert_eq!(j["event_type"], "on_text");
        assert_eq!(j["text"], "chunk");
    }

    #[test]
    fn test_event_on_retry_to_json() {
        let event = CallbackEvent::OnRetry { attempt: 5 };
        let j = event.to_json();
        assert_eq!(j["event_type"], "on_retry");
        assert_eq!(j["attempt"], 5);
    }

    #[test]
    fn test_event_custom_to_json() {
        let event = CallbackEvent::Custom {
            name: "ping".into(),
            data: json!({"ts": 12345}),
        };
        let j = event.to_json();
        assert_eq!(j["event_type"], "custom");
        assert_eq!(j["name"], "ping");
        assert_eq!(j["data"]["ts"], 12345);
    }

    #[test]
    fn test_event_on_start_to_json_null_input() {
        let event = CallbackEvent::OnStart { input: json!(null) };
        let j = event.to_json();
        assert_eq!(j["input"], Value::Null);
    }

    #[test]
    fn test_event_clone() {
        let event = CallbackEvent::OnStart { input: json!(42) };
        let cloned = event.clone();
        assert_eq!(cloned.event_type(), "on_start");
        assert_eq!(cloned.to_json()["input"], 42);
    }

    #[test]
    fn test_event_debug() {
        let event = CallbackEvent::OnRetry { attempt: 7 };
        let debug = format!("{:?}", event);
        assert!(debug.contains("OnRetry"));
        assert!(debug.contains("7"));
    }

    // -----------------------------------------------------------------------
    // CallbackScope tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_scope_new_has_generated_run_id() {
        let scope = CallbackScope::new("my_runnable", "chain");
        assert!(!scope.run_id.is_empty());
        assert_eq!(scope.name, "my_runnable");
        assert_eq!(scope.run_type, "chain");
        assert!(scope.parent_run_id.is_none());
    }

    #[test]
    fn test_scope_new_run_ids_are_unique() {
        let s1 = CallbackScope::new("a", "chain");
        let s2 = CallbackScope::new("b", "chain");
        assert_ne!(s1.run_id, s2.run_id);
    }

    #[test]
    fn test_scope_child_has_parent() {
        let parent = CallbackScope::new("parent", "chain");
        let child = parent.child("child_tool", "tool");
        assert_eq!(child.parent_run_id, Some(parent.run_id.clone()));
        assert_eq!(child.name, "child_tool");
        assert_eq!(child.run_type, "tool");
        assert_ne!(child.run_id, parent.run_id);
    }

    #[test]
    fn test_scope_child_chain() {
        let root = CallbackScope::new("root", "chain");
        let mid = root.child("mid", "tool");
        let leaf = mid.child("leaf", "llm");
        assert_eq!(mid.parent_run_id, Some(root.run_id.clone()));
        assert_eq!(leaf.parent_run_id, Some(mid.run_id.clone()));
    }

    #[test]
    fn test_scope_clone() {
        let scope = CallbackScope::new("test", "chain");
        let cloned = scope.clone();
        assert_eq!(cloned.run_id, scope.run_id);
        assert_eq!(cloned.name, scope.name);
    }

    #[test]
    fn test_scope_debug() {
        let scope = CallbackScope::new("my_debug_scope", "llm");
        let debug = format!("{:?}", scope);
        assert!(debug.contains("my_debug_scope"));
        assert!(debug.contains("llm"));
    }

    // -----------------------------------------------------------------------
    // ScopedCallbackConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_config_new_is_empty() {
        let config = ScopedCallbackConfig::new();
        assert_eq!(config.handler_count(), 0);
    }

    #[test]
    fn test_config_default_is_empty() {
        let config = ScopedCallbackConfig::default();
        assert_eq!(config.handler_count(), 0);
    }

    #[test]
    fn test_config_with_handler_single() {
        let config = ScopedCallbackConfig::new().with_handler(|_scope, _event| {});
        assert_eq!(config.handler_count(), 1);
    }

    #[test]
    fn test_config_with_handler_chained() {
        let config = ScopedCallbackConfig::new()
            .with_handler(|_scope, _event| {})
            .with_handler(|_scope, _event| {})
            .with_handler(|_scope, _event| {});
        assert_eq!(config.handler_count(), 3);
    }

    #[test]
    fn test_config_handlers_accessor() {
        let config = ScopedCallbackConfig::new()
            .with_handler(|_, _| {})
            .with_handler(|_, _| {});
        assert_eq!(config.handlers().len(), 2);
    }

    #[test]
    fn test_config_debug() {
        let config = ScopedCallbackConfig::new().with_handler(|_, _| {});
        let debug = format!("{:?}", config);
        assert!(debug.contains("handler_count: 1"));
    }

    // -----------------------------------------------------------------------
    // CallbackDispatcher tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_dispatcher_new_empty() {
        let dispatcher = CallbackDispatcher::new();
        assert_eq!(dispatcher.handler_count(), 0);
    }

    #[test]
    fn test_dispatcher_default_empty() {
        let dispatcher = CallbackDispatcher::default();
        assert_eq!(dispatcher.handler_count(), 0);
    }

    #[test]
    fn test_dispatcher_add_handler() {
        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(|_, _| {});
        dispatcher.add_handler(|_, _| {});
        assert_eq!(dispatcher.handler_count(), 2);
    }

    #[test]
    fn test_dispatcher_dispatch_calls_all_handlers() {
        let counter = Arc::new(AtomicU32::new(0));
        let c1 = counter.clone();
        let c2 = counter.clone();

        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(move |_, _| {
            c1.fetch_add(1, Ordering::SeqCst);
        });
        dispatcher.add_handler(move |_, _| {
            c2.fetch_add(10, Ordering::SeqCst);
        });

        let scope = CallbackScope::new("test", "chain");
        let event = CallbackEvent::OnStart { input: json!(null) };
        dispatcher.dispatch(&scope, &event);

        assert_eq!(counter.load(Ordering::SeqCst), 11);
    }

    #[test]
    fn test_dispatcher_dispatch_receives_correct_event() {
        let received = Arc::new(Mutex::new(Vec::new()));
        let r = received.clone();

        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(move |_scope, event| {
            r.lock().unwrap().push(event.event_type().to_string());
        });

        let scope = CallbackScope::new("test", "chain");
        dispatcher.dispatch(&scope, &CallbackEvent::OnStart { input: json!(1) });
        dispatcher.dispatch(&scope, &CallbackEvent::OnEnd { output: json!(2) });
        dispatcher.dispatch(
            &scope,
            &CallbackEvent::OnError {
                error: "err".into(),
            },
        );

        let types = received.lock().unwrap();
        assert_eq!(*types, vec!["on_start", "on_end", "on_error"]);
    }

    #[test]
    fn test_dispatcher_dispatch_receives_correct_scope() {
        let received_name = Arc::new(Mutex::new(String::new()));
        let rn = received_name.clone();

        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(move |scope, _event| {
            *rn.lock().unwrap() = scope.name.clone();
        });

        let scope = CallbackScope::new("my_scope", "chain");
        dispatcher.dispatch(&scope, &CallbackEvent::OnStart { input: json!(null) });

        assert_eq!(*received_name.lock().unwrap(), "my_scope");
    }

    #[test]
    fn test_dispatcher_no_handlers_dispatch_does_nothing() {
        let dispatcher = CallbackDispatcher::new();
        let scope = CallbackScope::new("test", "chain");
        // Should not panic.
        dispatcher.dispatch(&scope, &CallbackEvent::OnStart { input: json!(null) });
    }

    #[test]
    fn test_dispatcher_from_config() {
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();

        let config = ScopedCallbackConfig::new().with_handler(move |_, _| {
            c.fetch_add(1, Ordering::SeqCst);
        });

        let dispatcher = CallbackDispatcher::from_config(&config);
        assert_eq!(dispatcher.handler_count(), 1);

        let scope = CallbackScope::new("test", "chain");
        dispatcher.dispatch(&scope, &CallbackEvent::OnStart { input: json!(null) });
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_dispatcher_debug() {
        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(|_, _| {});
        let debug = format!("{:?}", dispatcher);
        assert!(debug.contains("handler_count: 1"));
    }

    #[test]
    fn test_dispatcher_dispatch_all_event_variants() {
        let received = Arc::new(Mutex::new(Vec::new()));
        let r = received.clone();

        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(move |_, event| {
            r.lock().unwrap().push(event.event_type().to_string());
        });

        let scope = CallbackScope::new("test", "chain");
        dispatcher.dispatch(&scope, &CallbackEvent::OnStart { input: json!(null) });
        dispatcher.dispatch(
            &scope,
            &CallbackEvent::OnEnd {
                output: json!(null),
            },
        );
        dispatcher.dispatch(&scope, &CallbackEvent::OnError { error: "e".into() });
        dispatcher.dispatch(&scope, &CallbackEvent::OnText { text: "t".into() });
        dispatcher.dispatch(&scope, &CallbackEvent::OnRetry { attempt: 1 });
        dispatcher.dispatch(
            &scope,
            &CallbackEvent::Custom {
                name: "c".into(),
                data: json!(null),
            },
        );

        let types = received.lock().unwrap();
        assert_eq!(
            *types,
            vec!["on_start", "on_end", "on_error", "on_text", "on_retry", "custom"]
        );
    }

    // -----------------------------------------------------------------------
    // ScopeGuard tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_scope_guard_dispatches_on_start_and_on_end() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let e = events.clone();

        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(move |_scope, event| {
            e.lock().unwrap().push(event.event_type().to_string());
        });
        let dispatcher = Arc::new(dispatcher);

        let scope = CallbackScope::new("guarded", "chain");
        {
            let mut guard = dispatcher.enter_scope(scope, json!({"q": "hello"}));
            guard.complete(json!({"a": "world"}));
        } // guard dropped here

        let types = events.lock().unwrap();
        assert_eq!(*types, vec!["on_start", "on_end"]);
    }

    #[test]
    fn test_scope_guard_dispatches_on_start_and_on_error() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let e = events.clone();

        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(move |_scope, event| {
            e.lock().unwrap().push(event.event_type().to_string());
        });
        let dispatcher = Arc::new(dispatcher);

        let scope = CallbackScope::new("guarded", "chain");
        {
            let mut guard = dispatcher.enter_scope(scope, json!(null));
            guard.fail("something went wrong".into());
        }

        let types = events.lock().unwrap();
        assert_eq!(*types, vec!["on_start", "on_error"]);
    }

    #[test]
    fn test_scope_guard_dropped_without_result_emits_error() {
        let events = Arc::new(Mutex::new(Vec::new()));
        let e = events.clone();

        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(move |_scope, event| {
            e.lock().unwrap().push(event.event_type().to_string());
        });
        let dispatcher = Arc::new(dispatcher);

        let scope = CallbackScope::new("abandoned", "chain");
        {
            let _guard = dispatcher.enter_scope(scope, json!(null));
            // Intentionally not calling complete() or fail().
        }

        let types = events.lock().unwrap();
        assert_eq!(*types, vec!["on_start", "on_error"]);
    }

    #[test]
    fn test_scope_guard_dropped_without_result_error_message() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let er = errors.clone();

        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(move |_scope, event| {
            if let CallbackEvent::OnError { error } = event {
                er.lock().unwrap().push(error.clone());
            }
        });
        let dispatcher = Arc::new(dispatcher);

        {
            let _guard = dispatcher.enter_scope(CallbackScope::new("test", "chain"), json!(null));
        }

        let errs = errors.lock().unwrap();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0], "scope dropped without completion");
    }

    #[test]
    fn test_scope_guard_scope_accessor() {
        let dispatcher = Arc::new(CallbackDispatcher::new());
        let scope = CallbackScope::new("accessor_test", "tool");
        let mut guard = dispatcher.enter_scope(scope, json!(null));
        assert_eq!(guard.scope().name, "accessor_test");
        assert_eq!(guard.scope().run_type, "tool");
        guard.complete(json!(null));
    }

    #[test]
    fn test_scope_guard_end_event_carries_output() {
        let outputs = Arc::new(Mutex::new(Vec::new()));
        let o = outputs.clone();

        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(move |_scope, event| {
            if let CallbackEvent::OnEnd { output } = event {
                o.lock().unwrap().push(output.clone());
            }
        });
        let dispatcher = Arc::new(dispatcher);

        {
            let mut guard =
                dispatcher.enter_scope(CallbackScope::new("test", "chain"), json!(null));
            guard.complete(json!({"result": 42}));
        }

        let out = outputs.lock().unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], json!({"result": 42}));
    }

    #[test]
    fn test_scope_guard_error_event_carries_message() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let er = errors.clone();

        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(move |_scope, event| {
            if let CallbackEvent::OnError { error } = event {
                er.lock().unwrap().push(error.clone());
            }
        });
        let dispatcher = Arc::new(dispatcher);

        {
            let mut guard =
                dispatcher.enter_scope(CallbackScope::new("test", "chain"), json!(null));
            guard.fail("timeout".into());
        }

        let errs = errors.lock().unwrap();
        assert_eq!(errs.len(), 1);
        assert_eq!(errs[0], "timeout");
    }

    #[test]
    fn test_scope_guard_start_event_carries_input() {
        let inputs = Arc::new(Mutex::new(Vec::new()));
        let i = inputs.clone();

        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(move |_scope, event| {
            if let CallbackEvent::OnStart { input } = event {
                i.lock().unwrap().push(input.clone());
            }
        });
        let dispatcher = Arc::new(dispatcher);

        {
            let mut guard =
                dispatcher.enter_scope(CallbackScope::new("test", "chain"), json!({"x": 99}));
            guard.complete(json!(null));
        }

        let ins = inputs.lock().unwrap();
        assert_eq!(ins.len(), 1);
        assert_eq!(ins[0], json!({"x": 99}));
    }

    // -----------------------------------------------------------------------
    // with_callbacks tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_with_callbacks_empty() {
        let config = with_callbacks(vec![]);
        assert_eq!(config.handler_count(), 0);
    }

    #[test]
    fn test_with_callbacks_multiple() {
        let config = with_callbacks(vec![
            Box::new(|_, _| {}),
            Box::new(|_, _| {}),
            Box::new(|_, _| {}),
        ]);
        assert_eq!(config.handler_count(), 3);
    }

    #[test]
    fn test_with_callbacks_handlers_are_called() {
        let counter = Arc::new(AtomicU32::new(0));
        let c = counter.clone();

        let config = with_callbacks(vec![Box::new(move |_, _| {
            c.fetch_add(1, Ordering::SeqCst);
        })]);

        let dispatcher = CallbackDispatcher::from_config(&config);
        let scope = CallbackScope::new("test", "chain");
        dispatcher.dispatch(&scope, &CallbackEvent::OnStart { input: json!(null) });

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    // -----------------------------------------------------------------------
    // merge_callback_configs tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_merge_empty_configs() {
        let a = ScopedCallbackConfig::new();
        let b = ScopedCallbackConfig::new();
        let merged = merge_callback_configs(&a, &b);
        assert_eq!(merged.handler_count(), 0);
    }

    #[test]
    fn test_merge_configs_combines_handlers() {
        let a = ScopedCallbackConfig::new()
            .with_handler(|_, _| {})
            .with_handler(|_, _| {});
        let b = ScopedCallbackConfig::new().with_handler(|_, _| {});
        let merged = merge_callback_configs(&a, &b);
        assert_eq!(merged.handler_count(), 3);
    }

    #[test]
    fn test_merge_configs_preserves_order() {
        let order = Arc::new(Mutex::new(Vec::new()));
        let o1 = order.clone();
        let o2 = order.clone();

        let a = ScopedCallbackConfig::new().with_handler(move |_, _| {
            o1.lock().unwrap().push("a");
        });
        let b = ScopedCallbackConfig::new().with_handler(move |_, _| {
            o2.lock().unwrap().push("b");
        });

        let merged = merge_callback_configs(&a, &b);
        let dispatcher = CallbackDispatcher::from_config(&merged);
        let scope = CallbackScope::new("test", "chain");
        dispatcher.dispatch(&scope, &CallbackEvent::OnStart { input: json!(null) });

        let calls = order.lock().unwrap();
        assert_eq!(*calls, vec!["a", "b"]);
    }

    #[test]
    fn test_merge_one_empty_one_populated() {
        let empty = ScopedCallbackConfig::new();
        let populated = ScopedCallbackConfig::new()
            .with_handler(|_, _| {})
            .with_handler(|_, _| {});
        let merged = merge_callback_configs(&empty, &populated);
        assert_eq!(merged.handler_count(), 2);

        let merged2 = merge_callback_configs(&populated, &empty);
        assert_eq!(merged2.handler_count(), 2);
    }

    #[test]
    fn test_merge_configs_does_not_affect_originals() {
        let a = ScopedCallbackConfig::new().with_handler(|_, _| {});
        let b = ScopedCallbackConfig::new().with_handler(|_, _| {});
        let _merged = merge_callback_configs(&a, &b);
        // Original configs should still have their original handler counts.
        assert_eq!(a.handler_count(), 1);
        assert_eq!(b.handler_count(), 1);
    }

    // -----------------------------------------------------------------------
    // RunnableWithCallbacks tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_runnable_with_callbacks_dispatches_events() {
        use crate::runnables::RunnableLambda;

        let events = Arc::new(Mutex::new(Vec::new()));
        let e = events.clone();

        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(move |_scope, event| {
            e.lock().unwrap().push(event.event_type().to_string());
        });

        let inner = Arc::new(RunnableLambda::new("double", |v: Value| async move {
            let n = v.as_i64().unwrap();
            Ok(json!(n * 2))
        })) as Arc<dyn super::super::base::Runnable>;

        let wrapped = RunnableWithCallbacks::new(inner, Arc::new(dispatcher));
        let result = wrapped.invoke(json!(5), None).await.unwrap();
        assert_eq!(result, json!(10));

        let types = events.lock().unwrap();
        assert_eq!(*types, vec!["on_start", "on_end"]);
    }

    #[tokio::test]
    async fn test_runnable_with_callbacks_on_error() {
        use crate::runnables::RunnableLambda;

        let events = Arc::new(Mutex::new(Vec::new()));
        let e = events.clone();

        let mut dispatcher = CallbackDispatcher::new();
        dispatcher.add_handler(move |_scope, event| {
            e.lock().unwrap().push(event.event_type().to_string());
        });

        let inner = Arc::new(RunnableLambda::new("failing", |_v: Value| async move {
            Err(crate::error::RustChainError::Other("boom".into()))
        })) as Arc<dyn super::super::base::Runnable>;

        let wrapped = RunnableWithCallbacks::new(inner, Arc::new(dispatcher));
        let result = wrapped.invoke(json!(1), None).await;
        assert!(result.is_err());

        let types = events.lock().unwrap();
        assert_eq!(*types, vec!["on_start", "on_error"]);
    }

    #[test]
    fn test_runnable_with_callbacks_debug() {
        use crate::runnables::RunnableLambda;

        let inner = Arc::new(RunnableLambda::new(
            "test_fn",
            |v: Value| async move { Ok(v) },
        )) as Arc<dyn super::super::base::Runnable>;

        let wrapped = RunnableWithCallbacks::new(inner, Arc::new(CallbackDispatcher::new()));
        let debug = format!("{:?}", wrapped);
        assert!(debug.contains("test_fn"));
    }
}
