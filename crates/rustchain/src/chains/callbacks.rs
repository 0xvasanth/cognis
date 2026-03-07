//! Callback manager for chain execution lifecycle events.
//!
//! Provides hooks into chain start, end, error, and step-level events,
//! along with built-in callback implementations for logging, metrics
//! collection, execution tracing, and event filtering.
//!
//! # Key Types
//!
//! - [`ChainEvent`] -- Enum representing all chain lifecycle events.
//! - [`ChainCallback`] -- Trait for handling chain events.
//! - [`LoggingCallback`] -- Prints events to stderr with timestamps.
//! - [`MetricsCallback`] -- Tracks per-chain run/step/error counts and durations.
//! - [`TracingCallback`] -- Builds a trace tree of chain execution.
//! - [`FilterCallback`] -- Wraps another callback, forwarding only matching events.
//! - [`CallbackManager`] -- Manages and dispatches events to multiple callbacks.
//! - [`CallbackScope`] -- RAII guard that emits start/end events automatically.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime};

use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// ChainEvent
// ---------------------------------------------------------------------------

/// Represents a lifecycle event during chain execution.
#[derive(Debug, Clone)]
pub enum ChainEvent {
    /// Emitted when a chain begins execution.
    ChainStart {
        /// Name of the chain.
        name: String,
        /// Input value provided to the chain.
        input: Value,
    },
    /// Emitted when a chain completes successfully.
    ChainEnd {
        /// Name of the chain.
        name: String,
        /// Output value produced by the chain.
        output: Value,
    },
    /// Emitted when a chain encounters an error.
    ChainError {
        /// Name of the chain.
        name: String,
        /// Error description.
        error: String,
    },
    /// Emitted when a step within a chain begins.
    StepStart {
        /// Name of the parent chain.
        chain_name: String,
        /// Zero-based index of the step.
        step_index: usize,
        /// Name of the step.
        step_name: String,
    },
    /// Emitted when a step within a chain completes.
    StepEnd {
        /// Name of the parent chain.
        chain_name: String,
        /// Zero-based index of the step.
        step_index: usize,
        /// Output value produced by the step.
        output: Value,
    },
    /// Emitted when a step within a chain encounters an error.
    StepError {
        /// Name of the parent chain.
        chain_name: String,
        /// Zero-based index of the step.
        step_index: usize,
        /// Error description.
        error: String,
    },
}

impl ChainEvent {
    /// Returns the chain name associated with this event.
    pub fn name(&self) -> &str {
        match self {
            ChainEvent::ChainStart { name, .. } => name,
            ChainEvent::ChainEnd { name, .. } => name,
            ChainEvent::ChainError { name, .. } => name,
            ChainEvent::StepStart { chain_name, .. } => chain_name,
            ChainEvent::StepEnd { chain_name, .. } => chain_name,
            ChainEvent::StepError { chain_name, .. } => chain_name,
        }
    }

    /// Returns `true` if this event represents an error.
    pub fn is_error(&self) -> bool {
        matches!(
            self,
            ChainEvent::ChainError { .. } | ChainEvent::StepError { .. }
        )
    }

    /// Serializes this event to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        match self {
            ChainEvent::ChainStart { name, input } => json!({
                "event": "chain_start",
                "name": name,
                "input": input,
            }),
            ChainEvent::ChainEnd { name, output } => json!({
                "event": "chain_end",
                "name": name,
                "output": output,
            }),
            ChainEvent::ChainError { name, error } => json!({
                "event": "chain_error",
                "name": name,
                "error": error,
            }),
            ChainEvent::StepStart {
                chain_name,
                step_index,
                step_name,
            } => json!({
                "event": "step_start",
                "chain_name": chain_name,
                "step_index": step_index,
                "step_name": step_name,
            }),
            ChainEvent::StepEnd {
                chain_name,
                step_index,
                output,
            } => json!({
                "event": "step_end",
                "chain_name": chain_name,
                "step_index": step_index,
                "output": output,
            }),
            ChainEvent::StepError {
                chain_name,
                step_index,
                error,
            } => json!({
                "event": "step_error",
                "chain_name": chain_name,
                "step_index": step_index,
                "error": error,
            }),
        }
    }
}

// ---------------------------------------------------------------------------
// ChainCallback trait
// ---------------------------------------------------------------------------

/// Trait for handling chain lifecycle events.
///
/// Implementors receive every [`ChainEvent`] dispatched by a [`CallbackManager`].
pub trait ChainCallback: Send + Sync {
    /// Called when a chain event occurs.
    fn on_event(&self, event: &ChainEvent);

    /// Returns the name of this callback handler. Used for identification
    /// and removal via [`CallbackManager::remove_callback`].
    fn callback_name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

// ---------------------------------------------------------------------------
// LoggingCallback
// ---------------------------------------------------------------------------

/// A callback that prints chain events to stderr with timestamps.
pub struct LoggingCallback;

impl ChainCallback for LoggingCallback {
    fn on_event(&self, event: &ChainEvent) {
        let now = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        let secs = now.as_secs();
        let event_type = match event {
            ChainEvent::ChainStart { .. } => "CHAIN_START",
            ChainEvent::ChainEnd { .. } => "CHAIN_END",
            ChainEvent::ChainError { .. } => "CHAIN_ERROR",
            ChainEvent::StepStart { .. } => "STEP_START",
            ChainEvent::StepEnd { .. } => "STEP_END",
            ChainEvent::StepError { .. } => "STEP_ERROR",
        };
        eprintln!(
            "[{secs}] [{event_type}] chain={} {}",
            event.name(),
            serde_json::to_string(&event.to_json()).unwrap_or_default()
        );
    }

    fn callback_name(&self) -> &str {
        "LoggingCallback"
    }
}

// ---------------------------------------------------------------------------
// MetricsCallback
// ---------------------------------------------------------------------------

/// Per-chain metrics collected by [`MetricsCallback`].
#[derive(Debug, Clone, Default)]
pub struct ChainMetrics {
    /// Number of completed chain runs.
    pub runs: usize,
    /// Number of steps executed.
    pub steps: usize,
    /// Number of errors (chain-level and step-level).
    pub errors: usize,
    /// Cumulative wall-clock duration of all runs.
    pub total_duration: Duration,
}

/// A callback that tracks per-chain run counts, step counts, error counts,
/// and cumulative duration.
pub struct MetricsCallback {
    metrics: Mutex<HashMap<String, ChainMetrics>>,
    /// Tracks the start time of each active chain by name.
    active_starts: Mutex<HashMap<String, Instant>>,
}

impl MetricsCallback {
    /// Creates a new, empty `MetricsCallback`.
    pub fn new() -> Self {
        Self {
            metrics: Mutex::new(HashMap::new()),
            active_starts: Mutex::new(HashMap::new()),
        }
    }

    /// Returns a snapshot of the collected metrics for all chains.
    pub fn get_metrics(&self) -> HashMap<String, ChainMetrics> {
        self.metrics.lock().unwrap().clone()
    }
}

impl Default for MetricsCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl ChainCallback for MetricsCallback {
    fn on_event(&self, event: &ChainEvent) {
        match event {
            ChainEvent::ChainStart { name, .. } => {
                self.active_starts
                    .lock()
                    .unwrap()
                    .insert(name.clone(), Instant::now());
            }
            ChainEvent::ChainEnd { name, .. } => {
                let elapsed = self
                    .active_starts
                    .lock()
                    .unwrap()
                    .remove(name)
                    .map(|start| start.elapsed())
                    .unwrap_or_default();
                let mut metrics = self.metrics.lock().unwrap();
                let entry = metrics.entry(name.clone()).or_default();
                entry.runs += 1;
                entry.total_duration += elapsed;
            }
            ChainEvent::ChainError { name, .. } => {
                let elapsed = self
                    .active_starts
                    .lock()
                    .unwrap()
                    .remove(name)
                    .map(|start| start.elapsed())
                    .unwrap_or_default();
                let mut metrics = self.metrics.lock().unwrap();
                let entry = metrics.entry(name.clone()).or_default();
                entry.errors += 1;
                entry.total_duration += elapsed;
            }
            ChainEvent::StepStart { chain_name, .. } => {
                let mut metrics = self.metrics.lock().unwrap();
                let entry = metrics.entry(chain_name.clone()).or_default();
                entry.steps += 1;
            }
            ChainEvent::StepEnd { .. } => {}
            ChainEvent::StepError { chain_name, .. } => {
                let mut metrics = self.metrics.lock().unwrap();
                let entry = metrics.entry(chain_name.clone()).or_default();
                entry.errors += 1;
            }
        }
    }

    fn callback_name(&self) -> &str {
        "MetricsCallback"
    }
}

// ---------------------------------------------------------------------------
// TracingCallback
// ---------------------------------------------------------------------------

/// Represents a traced step within a chain execution.
#[derive(Debug, Clone)]
pub struct TracedStep {
    /// Zero-based index of the step.
    pub index: usize,
    /// Name of the step.
    pub name: String,
    /// Output of the step, if available.
    pub output: Option<Value>,
    /// Whether the step ended in an error.
    pub error: Option<String>,
}

/// A trace of a single chain execution.
#[derive(Debug, Clone)]
pub struct ChainTrace {
    /// Name of the chain.
    pub name: String,
    /// Duration of the chain execution.
    pub duration: Duration,
    /// Steps executed within the chain.
    pub steps: Vec<TracedStep>,
    /// Final status: "success", "error: <msg>".
    pub status: String,
}

/// Internal state for an in-progress trace.
struct ActiveTrace {
    name: String,
    start: Instant,
    steps: Vec<TracedStep>,
}

/// A callback that builds a trace tree of chain executions.
pub struct TracingCallback {
    traces: Mutex<Vec<ChainTrace>>,
    active: Mutex<HashMap<String, ActiveTrace>>,
}

impl TracingCallback {
    /// Creates a new, empty `TracingCallback`.
    pub fn new() -> Self {
        Self {
            traces: Mutex::new(Vec::new()),
            active: Mutex::new(HashMap::new()),
        }
    }

    /// Returns all completed traces.
    pub fn get_traces(&self) -> Vec<ChainTrace> {
        self.traces.lock().unwrap().clone()
    }
}

impl Default for TracingCallback {
    fn default() -> Self {
        Self::new()
    }
}

impl ChainCallback for TracingCallback {
    fn on_event(&self, event: &ChainEvent) {
        match event {
            ChainEvent::ChainStart { name, .. } => {
                let trace = ActiveTrace {
                    name: name.clone(),
                    start: Instant::now(),
                    steps: Vec::new(),
                };
                self.active.lock().unwrap().insert(name.clone(), trace);
            }
            ChainEvent::ChainEnd { name, .. } => {
                if let Some(active) = self.active.lock().unwrap().remove(name) {
                    let completed = ChainTrace {
                        name: active.name,
                        duration: active.start.elapsed(),
                        steps: active.steps,
                        status: "success".to_string(),
                    };
                    self.traces.lock().unwrap().push(completed);
                }
            }
            ChainEvent::ChainError { name, error } => {
                if let Some(active) = self.active.lock().unwrap().remove(name) {
                    let completed = ChainTrace {
                        name: active.name,
                        duration: active.start.elapsed(),
                        steps: active.steps,
                        status: format!("error: {}", error),
                    };
                    self.traces.lock().unwrap().push(completed);
                }
            }
            ChainEvent::StepStart {
                chain_name,
                step_index,
                step_name,
            } => {
                if let Some(active) = self.active.lock().unwrap().get_mut(chain_name) {
                    active.steps.push(TracedStep {
                        index: *step_index,
                        name: step_name.clone(),
                        output: None,
                        error: None,
                    });
                }
            }
            ChainEvent::StepEnd {
                chain_name,
                step_index,
                output,
            } => {
                if let Some(active) = self.active.lock().unwrap().get_mut(chain_name) {
                    if let Some(step) = active.steps.iter_mut().find(|s| s.index == *step_index) {
                        step.output = Some(output.clone());
                    }
                }
            }
            ChainEvent::StepError {
                chain_name,
                step_index,
                error,
            } => {
                if let Some(active) = self.active.lock().unwrap().get_mut(chain_name) {
                    if let Some(step) = active.steps.iter_mut().find(|s| s.index == *step_index) {
                        step.error = Some(error.clone());
                    }
                }
            }
        }
    }

    fn callback_name(&self) -> &str {
        "TracingCallback"
    }
}

// ---------------------------------------------------------------------------
// FilterCallback
// ---------------------------------------------------------------------------

/// A callback that wraps another callback, forwarding only events for which
/// the predicate returns `true`.
pub struct FilterCallback {
    inner: Box<dyn ChainCallback>,
    predicate: Box<dyn Fn(&ChainEvent) -> bool + Send + Sync>,
    name: String,
}

impl FilterCallback {
    /// Creates a new `FilterCallback` wrapping `inner`, forwarding only
    /// events where `predicate` returns `true`.
    pub fn new(
        inner: Box<dyn ChainCallback>,
        predicate: impl Fn(&ChainEvent) -> bool + Send + Sync + 'static,
    ) -> Self {
        let name = format!("FilterCallback({})", inner.callback_name());
        Self {
            inner,
            predicate: Box::new(predicate),
            name,
        }
    }
}

impl ChainCallback for FilterCallback {
    fn on_event(&self, event: &ChainEvent) {
        if (self.predicate)(event) {
            self.inner.on_event(event);
        }
    }

    fn callback_name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// CallbackManager
// ---------------------------------------------------------------------------

/// Manages multiple [`ChainCallback`] handlers and dispatches events to all of them.
pub struct CallbackManager {
    callbacks: Vec<Box<dyn ChainCallback>>,
}

impl CallbackManager {
    /// Creates a new, empty `CallbackManager`.
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Builder method: adds a callback and returns `self`.
    pub fn with_callback(mut self, callback: Box<dyn ChainCallback>) -> Self {
        self.callbacks.push(callback);
        self
    }

    /// Adds a callback to this manager.
    pub fn add_callback(&mut self, callback: Box<dyn ChainCallback>) {
        self.callbacks.push(callback);
    }

    /// Removes the first callback whose [`ChainCallback::callback_name`]
    /// matches the given `name`. Returns `true` if a callback was removed.
    pub fn remove_callback(&mut self, name: &str) -> bool {
        if let Some(pos) = self
            .callbacks
            .iter()
            .position(|c| c.callback_name() == name)
        {
            self.callbacks.remove(pos);
            true
        } else {
            false
        }
    }

    /// Dispatches an event to all registered callbacks.
    pub fn emit(&self, event: &ChainEvent) {
        for cb in &self.callbacks {
            cb.on_event(event);
        }
    }

    /// Returns the number of registered callbacks.
    pub fn len(&self) -> usize {
        self.callbacks.len()
    }

    /// Returns `true` if no callbacks are registered.
    pub fn is_empty(&self) -> bool {
        self.callbacks.is_empty()
    }
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CallbackScope (RAII guard)
// ---------------------------------------------------------------------------

/// RAII guard that emits [`ChainEvent::ChainStart`] on creation and
/// [`ChainEvent::ChainEnd`] or [`ChainEvent::ChainError`] on drop.
///
/// Use [`CallbackScope::finish`] to emit a successful end event.
/// If dropped without calling `finish`, a `ChainError` event is emitted
/// indicating the scope was dropped without completion.
pub struct CallbackScope<'a> {
    manager: &'a CallbackManager,
    chain_name: String,
    finished: bool,
}

impl<'a> CallbackScope<'a> {
    /// Creates a new `CallbackScope`, immediately emitting a [`ChainEvent::ChainStart`].
    pub fn new(manager: &'a CallbackManager, chain_name: impl Into<String>, input: Value) -> Self {
        let chain_name = chain_name.into();
        manager.emit(&ChainEvent::ChainStart {
            name: chain_name.clone(),
            input,
        });
        Self {
            manager,
            chain_name,
            finished: false,
        }
    }

    /// Emits a [`ChainEvent::StepStart`] for the given step.
    pub fn step(&self, index: usize, name: impl Into<String>) {
        self.manager.emit(&ChainEvent::StepStart {
            chain_name: self.chain_name.clone(),
            step_index: index,
            step_name: name.into(),
        });
    }

    /// Emits a [`ChainEvent::StepEnd`] for the given step.
    pub fn step_end(&self, index: usize, output: Value) {
        self.manager.emit(&ChainEvent::StepEnd {
            chain_name: self.chain_name.clone(),
            step_index: index,
            output,
        });
    }

    /// Emits a [`ChainEvent::StepError`] for the given step.
    pub fn step_error(&self, index: usize, error: impl Into<String>) {
        self.manager.emit(&ChainEvent::StepError {
            chain_name: self.chain_name.clone(),
            step_index: index,
            error: error.into(),
        });
    }

    /// Consumes this scope, emitting a [`ChainEvent::ChainEnd`] event.
    pub fn finish(mut self, output: Value) {
        self.finished = true;
        self.manager.emit(&ChainEvent::ChainEnd {
            name: self.chain_name.clone(),
            output,
        });
    }

    /// Consumes this scope, emitting a [`ChainEvent::ChainError`] event.
    pub fn error(mut self, error: impl Into<String>) {
        self.finished = true;
        self.manager.emit(&ChainEvent::ChainError {
            name: self.chain_name.clone(),
            error: error.into(),
        });
    }
}

impl<'a> Drop for CallbackScope<'a> {
    fn drop(&mut self) {
        if !self.finished {
            self.manager.emit(&ChainEvent::ChainError {
                name: self.chain_name.clone(),
                error: "scope dropped without finish".to_string(),
            });
        }
    }
}

// ---------------------------------------------------------------------------
// RecordingCallback (utility for tests and inspection)
// ---------------------------------------------------------------------------

/// A callback that records all events for later inspection.
/// Useful for testing and debugging.
pub struct RecordingCallback {
    events: Mutex<Vec<ChainEvent>>,
    name: String,
}

impl RecordingCallback {
    /// Creates a new `RecordingCallback` with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            name: name.into(),
        }
    }

    /// Returns a snapshot of all recorded events.
    pub fn events(&self) -> Vec<ChainEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Returns the number of recorded events.
    pub fn event_count(&self) -> usize {
        self.events.lock().unwrap().len()
    }
}

impl ChainCallback for RecordingCallback {
    fn on_event(&self, event: &ChainEvent) {
        self.events.lock().unwrap().push(event.clone());
    }

    fn callback_name(&self) -> &str {
        &self.name
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;

    // -----------------------------------------------------------------------
    // ChainEvent tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chain_start_event_name() {
        let event = ChainEvent::ChainStart {
            name: "my_chain".into(),
            input: json!({"q": "hello"}),
        };
        assert_eq!(event.name(), "my_chain");
    }

    #[test]
    fn test_chain_end_event_name() {
        let event = ChainEvent::ChainEnd {
            name: "my_chain".into(),
            output: json!("result"),
        };
        assert_eq!(event.name(), "my_chain");
    }

    #[test]
    fn test_chain_error_event_name() {
        let event = ChainEvent::ChainError {
            name: "my_chain".into(),
            error: "boom".into(),
        };
        assert_eq!(event.name(), "my_chain");
    }

    #[test]
    fn test_step_start_event_name() {
        let event = ChainEvent::StepStart {
            chain_name: "parent".into(),
            step_index: 0,
            step_name: "step_a".into(),
        };
        assert_eq!(event.name(), "parent");
    }

    #[test]
    fn test_step_end_event_name() {
        let event = ChainEvent::StepEnd {
            chain_name: "parent".into(),
            step_index: 1,
            output: json!(42),
        };
        assert_eq!(event.name(), "parent");
    }

    #[test]
    fn test_step_error_event_name() {
        let event = ChainEvent::StepError {
            chain_name: "parent".into(),
            step_index: 2,
            error: "step failed".into(),
        };
        assert_eq!(event.name(), "parent");
    }

    #[test]
    fn test_is_error_chain_error() {
        let event = ChainEvent::ChainError {
            name: "c".into(),
            error: "e".into(),
        };
        assert!(event.is_error());
    }

    #[test]
    fn test_is_error_step_error() {
        let event = ChainEvent::StepError {
            chain_name: "c".into(),
            step_index: 0,
            error: "e".into(),
        };
        assert!(event.is_error());
    }

    #[test]
    fn test_is_error_false_for_start() {
        let event = ChainEvent::ChainStart {
            name: "c".into(),
            input: json!(null),
        };
        assert!(!event.is_error());
    }

    #[test]
    fn test_is_error_false_for_end() {
        let event = ChainEvent::ChainEnd {
            name: "c".into(),
            output: json!(null),
        };
        assert!(!event.is_error());
    }

    #[test]
    fn test_is_error_false_for_step_start() {
        let event = ChainEvent::StepStart {
            chain_name: "c".into(),
            step_index: 0,
            step_name: "s".into(),
        };
        assert!(!event.is_error());
    }

    #[test]
    fn test_is_error_false_for_step_end() {
        let event = ChainEvent::StepEnd {
            chain_name: "c".into(),
            step_index: 0,
            output: json!(null),
        };
        assert!(!event.is_error());
    }

    #[test]
    fn test_chain_start_to_json() {
        let event = ChainEvent::ChainStart {
            name: "test".into(),
            input: json!({"key": "value"}),
        };
        let j = event.to_json();
        assert_eq!(j["event"], "chain_start");
        assert_eq!(j["name"], "test");
        assert_eq!(j["input"]["key"], "value");
    }

    #[test]
    fn test_chain_end_to_json() {
        let event = ChainEvent::ChainEnd {
            name: "test".into(),
            output: json!(42),
        };
        let j = event.to_json();
        assert_eq!(j["event"], "chain_end");
        assert_eq!(j["output"], 42);
    }

    #[test]
    fn test_chain_error_to_json() {
        let event = ChainEvent::ChainError {
            name: "test".into(),
            error: "something broke".into(),
        };
        let j = event.to_json();
        assert_eq!(j["event"], "chain_error");
        assert_eq!(j["error"], "something broke");
    }

    #[test]
    fn test_step_start_to_json() {
        let event = ChainEvent::StepStart {
            chain_name: "c".into(),
            step_index: 3,
            step_name: "transform".into(),
        };
        let j = event.to_json();
        assert_eq!(j["event"], "step_start");
        assert_eq!(j["chain_name"], "c");
        assert_eq!(j["step_index"], 3);
        assert_eq!(j["step_name"], "transform");
    }

    #[test]
    fn test_step_end_to_json() {
        let event = ChainEvent::StepEnd {
            chain_name: "c".into(),
            step_index: 1,
            output: json!("done"),
        };
        let j = event.to_json();
        assert_eq!(j["event"], "step_end");
        assert_eq!(j["output"], "done");
    }

    #[test]
    fn test_step_error_to_json() {
        let event = ChainEvent::StepError {
            chain_name: "c".into(),
            step_index: 0,
            error: "fail".into(),
        };
        let j = event.to_json();
        assert_eq!(j["event"], "step_error");
        assert_eq!(j["error"], "fail");
    }

    // -----------------------------------------------------------------------
    // LoggingCallback tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_logging_callback_does_not_panic_on_chain_start() {
        let cb = LoggingCallback;
        cb.on_event(&ChainEvent::ChainStart {
            name: "test".into(),
            input: json!(null),
        });
    }

    #[test]
    fn test_logging_callback_does_not_panic_on_chain_end() {
        let cb = LoggingCallback;
        cb.on_event(&ChainEvent::ChainEnd {
            name: "test".into(),
            output: json!("ok"),
        });
    }

    #[test]
    fn test_logging_callback_does_not_panic_on_error() {
        let cb = LoggingCallback;
        cb.on_event(&ChainEvent::ChainError {
            name: "test".into(),
            error: "err".into(),
        });
    }

    #[test]
    fn test_logging_callback_does_not_panic_on_step_events() {
        let cb = LoggingCallback;
        cb.on_event(&ChainEvent::StepStart {
            chain_name: "c".into(),
            step_index: 0,
            step_name: "s".into(),
        });
        cb.on_event(&ChainEvent::StepEnd {
            chain_name: "c".into(),
            step_index: 0,
            output: json!(1),
        });
        cb.on_event(&ChainEvent::StepError {
            chain_name: "c".into(),
            step_index: 0,
            error: "e".into(),
        });
    }

    // -----------------------------------------------------------------------
    // MetricsCallback tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_metrics_callback_counts_runs() {
        let cb = MetricsCallback::new();
        cb.on_event(&ChainEvent::ChainStart {
            name: "a".into(),
            input: json!(null),
        });
        cb.on_event(&ChainEvent::ChainEnd {
            name: "a".into(),
            output: json!(null),
        });
        cb.on_event(&ChainEvent::ChainStart {
            name: "a".into(),
            input: json!(null),
        });
        cb.on_event(&ChainEvent::ChainEnd {
            name: "a".into(),
            output: json!(null),
        });
        let m = cb.get_metrics();
        assert_eq!(m["a"].runs, 2);
    }

    #[test]
    fn test_metrics_callback_counts_steps() {
        let cb = MetricsCallback::new();
        cb.on_event(&ChainEvent::ChainStart {
            name: "b".into(),
            input: json!(null),
        });
        cb.on_event(&ChainEvent::StepStart {
            chain_name: "b".into(),
            step_index: 0,
            step_name: "s0".into(),
        });
        cb.on_event(&ChainEvent::StepStart {
            chain_name: "b".into(),
            step_index: 1,
            step_name: "s1".into(),
        });
        cb.on_event(&ChainEvent::ChainEnd {
            name: "b".into(),
            output: json!(null),
        });
        let m = cb.get_metrics();
        assert_eq!(m["b"].steps, 2);
    }

    #[test]
    fn test_metrics_callback_counts_errors() {
        let cb = MetricsCallback::new();
        cb.on_event(&ChainEvent::ChainStart {
            name: "c".into(),
            input: json!(null),
        });
        cb.on_event(&ChainEvent::StepError {
            chain_name: "c".into(),
            step_index: 0,
            error: "oops".into(),
        });
        cb.on_event(&ChainEvent::ChainError {
            name: "c".into(),
            error: "fatal".into(),
        });
        let m = cb.get_metrics();
        assert_eq!(m["c"].errors, 2);
    }

    #[test]
    fn test_metrics_callback_tracks_duration() {
        let cb = MetricsCallback::new();
        cb.on_event(&ChainEvent::ChainStart {
            name: "d".into(),
            input: json!(null),
        });
        std::thread::sleep(Duration::from_millis(10));
        cb.on_event(&ChainEvent::ChainEnd {
            name: "d".into(),
            output: json!(null),
        });
        let m = cb.get_metrics();
        assert!(m["d"].total_duration >= Duration::from_millis(5));
    }

    #[test]
    fn test_metrics_callback_multiple_chains() {
        let cb = MetricsCallback::new();
        cb.on_event(&ChainEvent::ChainStart {
            name: "x".into(),
            input: json!(null),
        });
        cb.on_event(&ChainEvent::ChainEnd {
            name: "x".into(),
            output: json!(null),
        });
        cb.on_event(&ChainEvent::ChainStart {
            name: "y".into(),
            input: json!(null),
        });
        cb.on_event(&ChainEvent::ChainEnd {
            name: "y".into(),
            output: json!(null),
        });
        let m = cb.get_metrics();
        assert_eq!(m.len(), 2);
        assert_eq!(m["x"].runs, 1);
        assert_eq!(m["y"].runs, 1);
    }

    // -----------------------------------------------------------------------
    // TracingCallback tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tracing_callback_basic_trace() {
        let cb = TracingCallback::new();
        cb.on_event(&ChainEvent::ChainStart {
            name: "t".into(),
            input: json!(null),
        });
        cb.on_event(&ChainEvent::ChainEnd {
            name: "t".into(),
            output: json!("ok"),
        });
        let traces = cb.get_traces();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].name, "t");
        assert_eq!(traces[0].status, "success");
    }

    #[test]
    fn test_tracing_callback_with_steps() {
        let cb = TracingCallback::new();
        cb.on_event(&ChainEvent::ChainStart {
            name: "t".into(),
            input: json!(null),
        });
        cb.on_event(&ChainEvent::StepStart {
            chain_name: "t".into(),
            step_index: 0,
            step_name: "prompt".into(),
        });
        cb.on_event(&ChainEvent::StepEnd {
            chain_name: "t".into(),
            step_index: 0,
            output: json!("formatted"),
        });
        cb.on_event(&ChainEvent::StepStart {
            chain_name: "t".into(),
            step_index: 1,
            step_name: "llm".into(),
        });
        cb.on_event(&ChainEvent::StepEnd {
            chain_name: "t".into(),
            step_index: 1,
            output: json!("answer"),
        });
        cb.on_event(&ChainEvent::ChainEnd {
            name: "t".into(),
            output: json!("answer"),
        });
        let traces = cb.get_traces();
        assert_eq!(traces[0].steps.len(), 2);
        assert_eq!(traces[0].steps[0].name, "prompt");
        assert_eq!(traces[0].steps[1].name, "llm");
        assert_eq!(traces[0].steps[0].output, Some(json!("formatted")));
    }

    #[test]
    fn test_tracing_callback_error_trace() {
        let cb = TracingCallback::new();
        cb.on_event(&ChainEvent::ChainStart {
            name: "e".into(),
            input: json!(null),
        });
        cb.on_event(&ChainEvent::ChainError {
            name: "e".into(),
            error: "timeout".into(),
        });
        let traces = cb.get_traces();
        assert_eq!(traces.len(), 1);
        assert!(traces[0].status.starts_with("error"));
    }

    #[test]
    fn test_tracing_callback_step_error() {
        let cb = TracingCallback::new();
        cb.on_event(&ChainEvent::ChainStart {
            name: "f".into(),
            input: json!(null),
        });
        cb.on_event(&ChainEvent::StepStart {
            chain_name: "f".into(),
            step_index: 0,
            step_name: "bad_step".into(),
        });
        cb.on_event(&ChainEvent::StepError {
            chain_name: "f".into(),
            step_index: 0,
            error: "step boom".into(),
        });
        cb.on_event(&ChainEvent::ChainError {
            name: "f".into(),
            error: "chain boom".into(),
        });
        let traces = cb.get_traces();
        assert_eq!(traces[0].steps[0].error, Some("step boom".to_string()));
    }

    #[test]
    fn test_tracing_callback_duration() {
        let cb = TracingCallback::new();
        cb.on_event(&ChainEvent::ChainStart {
            name: "g".into(),
            input: json!(null),
        });
        std::thread::sleep(Duration::from_millis(10));
        cb.on_event(&ChainEvent::ChainEnd {
            name: "g".into(),
            output: json!(null),
        });
        let traces = cb.get_traces();
        assert!(traces[0].duration >= Duration::from_millis(5));
    }

    // -----------------------------------------------------------------------
    // FilterCallback tests
    // -----------------------------------------------------------------------

    /// Wrapper to delegate to an `Arc<RecordingCallback>`.
    struct RecordingCallbackWrapper(Arc<RecordingCallback>);

    impl ChainCallback for RecordingCallbackWrapper {
        fn on_event(&self, event: &ChainEvent) {
            self.0.on_event(event);
        }

        fn callback_name(&self) -> &str {
            self.0.callback_name()
        }
    }

    #[test]
    fn test_filter_callback_forwards_matching_events() {
        let recorder = Arc::new(RecordingCallback::new("rec"));
        let recorder_clone = recorder.clone();
        let filter = FilterCallback::new(
            Box::new(RecordingCallbackWrapper(recorder_clone)),
            |event| event.is_error(),
        );
        filter.on_event(&ChainEvent::ChainStart {
            name: "a".into(),
            input: json!(null),
        });
        filter.on_event(&ChainEvent::ChainError {
            name: "a".into(),
            error: "err".into(),
        });
        assert_eq!(recorder.event_count(), 1);
    }

    #[test]
    fn test_filter_callback_blocks_non_matching() {
        let recorder = Arc::new(RecordingCallback::new("rec"));
        let recorder_clone = recorder.clone();
        let filter = FilterCallback::new(
            Box::new(RecordingCallbackWrapper(recorder_clone)),
            |event| matches!(event, ChainEvent::ChainStart { .. }),
        );
        filter.on_event(&ChainEvent::ChainEnd {
            name: "a".into(),
            output: json!(null),
        });
        assert_eq!(recorder.event_count(), 0);
    }

    #[test]
    fn test_filter_callback_allows_all_with_true_predicate() {
        let recorder = Arc::new(RecordingCallback::new("rec"));
        let recorder_clone = recorder.clone();
        let filter = FilterCallback::new(
            Box::new(RecordingCallbackWrapper(recorder_clone)),
            |_| true,
        );
        filter.on_event(&ChainEvent::ChainStart {
            name: "a".into(),
            input: json!(null),
        });
        filter.on_event(&ChainEvent::ChainEnd {
            name: "a".into(),
            output: json!(null),
        });
        assert_eq!(recorder.event_count(), 2);
    }

    // -----------------------------------------------------------------------
    // CallbackManager tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_callback_manager_new_is_empty() {
        let mgr = CallbackManager::new();
        assert!(mgr.is_empty());
        assert_eq!(mgr.len(), 0);
    }

    #[test]
    fn test_callback_manager_with_callback() {
        let mgr = CallbackManager::new().with_callback(Box::new(LoggingCallback));
        assert_eq!(mgr.len(), 1);
    }

    #[test]
    fn test_callback_manager_add_callback() {
        let mut mgr = CallbackManager::new();
        mgr.add_callback(Box::new(LoggingCallback));
        assert_eq!(mgr.len(), 1);
    }

    #[test]
    fn test_callback_manager_remove_callback() {
        let mut mgr = CallbackManager::new();
        mgr.add_callback(Box::new(RecordingCallback::new("r1")));
        mgr.add_callback(Box::new(RecordingCallback::new("r2")));
        assert_eq!(mgr.len(), 2);
        let removed = mgr.remove_callback("r1");
        assert!(removed);
        assert_eq!(mgr.len(), 1);
    }

    #[test]
    fn test_callback_manager_remove_nonexistent() {
        let mut mgr = CallbackManager::new();
        let removed = mgr.remove_callback("nope");
        assert!(!removed);
    }

    #[test]
    fn test_callback_manager_dispatches_to_multiple() {
        let r1 = Arc::new(RecordingCallback::new("r1"));
        let r2 = Arc::new(RecordingCallback::new("r2"));
        let mgr = CallbackManager::new()
            .with_callback(Box::new(RecordingCallbackWrapper(r1.clone())))
            .with_callback(Box::new(RecordingCallbackWrapper(r2.clone())));
        mgr.emit(&ChainEvent::ChainStart {
            name: "test".into(),
            input: json!(null),
        });
        assert_eq!(r1.event_count(), 1);
        assert_eq!(r2.event_count(), 1);
    }

    #[test]
    fn test_callback_manager_emit_empty_no_panic() {
        let mgr = CallbackManager::new();
        mgr.emit(&ChainEvent::ChainStart {
            name: "test".into(),
            input: json!(null),
        });
    }

    // -----------------------------------------------------------------------
    // CallbackScope tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_callback_scope_emits_start_and_finish() {
        let r = Arc::new(RecordingCallback::new("r"));
        let mgr =
            CallbackManager::new().with_callback(Box::new(RecordingCallbackWrapper(r.clone())));

        let scope = CallbackScope::new(&mgr, "my_chain", json!({"q": "hello"}));
        scope.finish(json!({"a": "world"}));

        let events = r.events();
        assert_eq!(events.len(), 2);
        assert!(
            matches!(&events[0], ChainEvent::ChainStart { name, .. } if name == "my_chain")
        );
        assert!(
            matches!(&events[1], ChainEvent::ChainEnd { name, .. } if name == "my_chain")
        );
    }

    #[test]
    fn test_callback_scope_emits_error_on_drop() {
        let r = Arc::new(RecordingCallback::new("r"));
        let mgr =
            CallbackManager::new().with_callback(Box::new(RecordingCallbackWrapper(r.clone())));

        {
            let _scope = CallbackScope::new(&mgr, "dropped_chain", json!(null));
            // scope dropped without finish
        }

        let events = r.events();
        assert_eq!(events.len(), 2);
        assert!(
            matches!(&events[1], ChainEvent::ChainError { name, error } if name == "dropped_chain" && error.contains("dropped"))
        );
    }

    #[test]
    fn test_callback_scope_explicit_error() {
        let r = Arc::new(RecordingCallback::new("r"));
        let mgr =
            CallbackManager::new().with_callback(Box::new(RecordingCallbackWrapper(r.clone())));

        let scope = CallbackScope::new(&mgr, "err_chain", json!(null));
        scope.error("explicit error");

        let events = r.events();
        assert_eq!(events.len(), 2);
        assert!(
            matches!(&events[1], ChainEvent::ChainError { error, .. } if error == "explicit error")
        );
    }

    #[test]
    fn test_callback_scope_with_steps() {
        let r = Arc::new(RecordingCallback::new("r"));
        let mgr =
            CallbackManager::new().with_callback(Box::new(RecordingCallbackWrapper(r.clone())));

        let scope = CallbackScope::new(&mgr, "step_chain", json!(null));
        scope.step(0, "format");
        scope.step_end(0, json!("formatted"));
        scope.step(1, "invoke");
        scope.step_end(1, json!("result"));
        scope.finish(json!("result"));

        let events = r.events();
        // start + step_start + step_end + step_start + step_end + end = 6
        assert_eq!(events.len(), 6);
    }

    #[test]
    fn test_callback_scope_step_error() {
        let r = Arc::new(RecordingCallback::new("r"));
        let mgr =
            CallbackManager::new().with_callback(Box::new(RecordingCallbackWrapper(r.clone())));

        let scope = CallbackScope::new(&mgr, "se_chain", json!(null));
        scope.step(0, "bad_step");
        scope.step_error(0, "step failed");
        scope.error("chain failed");

        let events = r.events();
        assert_eq!(events.len(), 4); // start + step_start + step_error + chain_error
        assert!(
            matches!(&events[2], ChainEvent::StepError { error, .. } if error == "step failed")
        );
    }

    // -----------------------------------------------------------------------
    // Integration: MetricsCallback + CallbackScope
    // -----------------------------------------------------------------------

    #[test]
    fn test_metrics_with_scope() {
        let metrics = Arc::new(MetricsCallback::new());
        struct MetricsWrapper(Arc<MetricsCallback>);
        impl ChainCallback for MetricsWrapper {
            fn on_event(&self, event: &ChainEvent) {
                self.0.on_event(event);
            }
            fn callback_name(&self) -> &str {
                "MetricsWrapper"
            }
        }

        let mgr =
            CallbackManager::new().with_callback(Box::new(MetricsWrapper(metrics.clone())));

        let scope = CallbackScope::new(&mgr, "integrated", json!(null));
        scope.step(0, "s0");
        scope.step_end(0, json!(null));
        scope.finish(json!(null));

        let m = metrics.get_metrics();
        assert_eq!(m["integrated"].runs, 1);
        assert_eq!(m["integrated"].steps, 1);
        assert_eq!(m["integrated"].errors, 0);
    }

    // -----------------------------------------------------------------------
    // Integration: TracingCallback + CallbackScope
    // -----------------------------------------------------------------------

    #[test]
    fn test_tracing_with_scope() {
        let tracing = Arc::new(TracingCallback::new());
        struct TracingWrapper(Arc<TracingCallback>);
        impl ChainCallback for TracingWrapper {
            fn on_event(&self, event: &ChainEvent) {
                self.0.on_event(event);
            }
            fn callback_name(&self) -> &str {
                "TracingWrapper"
            }
        }

        let mgr =
            CallbackManager::new().with_callback(Box::new(TracingWrapper(tracing.clone())));

        let scope = CallbackScope::new(&mgr, "traced", json!({"input": 1}));
        scope.step(0, "prompt");
        scope.step_end(0, json!("formatted"));
        scope.finish(json!("done"));

        let traces = tracing.get_traces();
        assert_eq!(traces.len(), 1);
        assert_eq!(traces[0].status, "success");
        assert_eq!(traces[0].steps.len(), 1);
        assert_eq!(traces[0].steps[0].name, "prompt");
    }

    // -----------------------------------------------------------------------
    // RecordingCallback tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_recording_callback_records_events() {
        let cb = RecordingCallback::new("test_rec");
        cb.on_event(&ChainEvent::ChainStart {
            name: "a".into(),
            input: json!(null),
        });
        cb.on_event(&ChainEvent::ChainEnd {
            name: "a".into(),
            output: json!(null),
        });
        assert_eq!(cb.event_count(), 2);
        assert_eq!(cb.callback_name(), "test_rec");
    }
}
