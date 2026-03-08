//! Centralized callback manager with phase-based handlers, metrics, and RAII scopes.
//!
//! This module implements a callback system that covers the full execution
//! lifecycle of chains, LLMs, tools, and retrievers. Events are described by
//! a [`CallbackPhase`] and carry arbitrary JSON [`CallbackData`]. Handlers
//! registered with the [`CallbackManager`] receive every emitted event (unless
//! filtered by [`FilteredHandler`]).
//!
//! [`CallbackScope`] provides an RAII guard that emits a start event on
//! creation and an end (or error) event when dropped.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::Instant;

use serde_json::{json, Value};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// CallbackPhase
// ---------------------------------------------------------------------------

/// Identifies the lifecycle phase of an event.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum CallbackPhase {
    /// A chain has started executing.
    ChainStart,
    /// A chain has finished executing.
    ChainEnd,
    /// A chain encountered an error.
    ChainError,
    /// An LLM call has started.
    LlmStart,
    /// An LLM call has finished.
    LlmEnd,
    /// An LLM call encountered an error.
    LlmError,
    /// A tool invocation has started.
    ToolStart,
    /// A tool invocation has finished.
    ToolEnd,
    /// A tool invocation encountered an error.
    ToolError,
    /// A retriever query has started.
    RetrieverStart,
    /// A retriever query has finished.
    RetrieverEnd,
    /// A user-defined phase.
    Custom(String),
}

impl CallbackPhase {
    /// Returns a string representation of this phase.
    pub fn as_str(&self) -> &str {
        match self {
            CallbackPhase::ChainStart => "chain_start",
            CallbackPhase::ChainEnd => "chain_end",
            CallbackPhase::ChainError => "chain_error",
            CallbackPhase::LlmStart => "llm_start",
            CallbackPhase::LlmEnd => "llm_end",
            CallbackPhase::LlmError => "llm_error",
            CallbackPhase::ToolStart => "tool_start",
            CallbackPhase::ToolEnd => "tool_end",
            CallbackPhase::ToolError => "tool_error",
            CallbackPhase::RetrieverStart => "retriever_start",
            CallbackPhase::RetrieverEnd => "retriever_end",
            CallbackPhase::Custom(s) => s.as_str(),
        }
    }
}

impl fmt::Display for CallbackPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// CallbackData
// ---------------------------------------------------------------------------

/// Structured event payload emitted through the callback system.
#[derive(Debug, Clone)]
pub struct CallbackData {
    /// The lifecycle phase this event belongs to.
    pub phase: CallbackPhase,
    /// Unique identifier for this run.
    pub run_id: String,
    /// Optional parent run id for nested runs.
    pub parent_run_id: Option<String>,
    /// Arbitrary JSON payload.
    pub data: Value,
    /// ISO-8601 formatted timestamp.
    pub timestamp: String,
}

impl CallbackData {
    /// Serializes this event to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "phase": self.phase.as_str(),
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "data": self.data,
            "timestamp": self.timestamp,
        })
    }
}

/// Builder for [`CallbackData`].
#[derive(Debug)]
pub struct CallbackDataBuilder {
    phase: CallbackPhase,
    run_id: String,
    parent_run_id: Option<String>,
    data: Value,
    timestamp: Option<String>,
}

impl CallbackDataBuilder {
    /// Creates a new builder with required fields.
    pub fn new(phase: CallbackPhase, run_id: impl Into<String>) -> Self {
        Self {
            phase,
            run_id: run_id.into(),
            parent_run_id: None,
            data: Value::Null,
            timestamp: None,
        }
    }

    /// Sets the parent run id.
    pub fn parent_run_id(mut self, id: impl Into<String>) -> Self {
        self.parent_run_id = Some(id.into());
        self
    }

    /// Sets the data payload.
    pub fn data(mut self, data: Value) -> Self {
        self.data = data;
        self
    }

    /// Sets the timestamp string.
    pub fn timestamp(mut self, ts: impl Into<String>) -> Self {
        self.timestamp = Some(ts.into());
        self
    }

    /// Builds the [`CallbackData`], defaulting the timestamp to now if unset.
    pub fn build(self) -> CallbackData {
        CallbackData {
            phase: self.phase,
            run_id: self.run_id,
            parent_run_id: self.parent_run_id,
            data: self.data,
            timestamp: self.timestamp.unwrap_or_else(now_iso8601),
        }
    }
}

fn now_iso8601() -> String {
    use std::time::SystemTime;
    let d = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}s", d.as_secs())
}

// ---------------------------------------------------------------------------
// CallbackHandler trait
// ---------------------------------------------------------------------------

/// Trait for types that receive callback events.
pub trait CallbackHandler: Send + Sync {
    /// Called when an event is emitted.
    fn on_event(&self, data: &CallbackData);

    /// Human-readable name of this handler.
    fn name(&self) -> &str;

    /// Returns `true` if this handler is interested in the given phase.
    /// Defaults to `true` for all phases.
    fn handles_phase(&self, _phase: &CallbackPhase) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// ConsoleCallbackHandler
// ---------------------------------------------------------------------------

/// A callback handler that logs events to an internal buffer.
pub struct ConsoleCallbackHandler {
    logs: Mutex<Vec<String>>,
}

impl ConsoleCallbackHandler {
    /// Creates a new handler with an empty log buffer.
    pub fn new() -> Self {
        Self {
            logs: Mutex::new(Vec::new()),
        }
    }

    /// Returns a snapshot of all log lines.
    pub fn logs(&self) -> Vec<String> {
        self.logs.lock().unwrap().clone()
    }

    /// Clears the log buffer.
    pub fn clear(&self) {
        self.logs.lock().unwrap().clear();
    }
}

impl Default for ConsoleCallbackHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CallbackHandler for ConsoleCallbackHandler {
    fn on_event(&self, data: &CallbackData) {
        let line = format!(
            "[{}] {} run={} data={}",
            data.timestamp,
            data.phase,
            data.run_id,
            data.data,
        );
        self.logs.lock().unwrap().push(line);
    }

    fn name(&self) -> &str {
        "ConsoleCallbackHandler"
    }
}

// ---------------------------------------------------------------------------
// MetricsCallbackHandler
// ---------------------------------------------------------------------------

/// Collects timing and count metrics for callback events.
pub struct MetricsCallbackHandler {
    inner: Mutex<MetricsInner>,
}

struct MetricsInner {
    counts: HashMap<String, usize>,
    durations: HashMap<String, Vec<f64>>,
    pending: HashMap<String, Instant>,
}

impl MetricsCallbackHandler {
    /// Creates a new metrics handler.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(MetricsInner {
                counts: HashMap::new(),
                durations: HashMap::new(),
                pending: HashMap::new(),
            }),
        }
    }

    /// Returns the total number of events received.
    pub fn total_events(&self) -> usize {
        let inner = self.inner.lock().unwrap();
        inner.counts.values().sum()
    }

    /// Returns a map of phase name to event count.
    pub fn events_by_phase(&self) -> HashMap<String, usize> {
        self.inner.lock().unwrap().counts.clone()
    }

    /// Returns the average duration in milliseconds for a given phase, if
    /// any duration samples have been recorded.
    pub fn avg_duration_ms(&self, phase: &CallbackPhase) -> Option<f64> {
        let inner = self.inner.lock().unwrap();
        let durations = inner.durations.get(phase.as_str())?;
        if durations.is_empty() {
            return None;
        }
        let sum: f64 = durations.iter().sum();
        Some(sum / durations.len() as f64)
    }

    /// Serializes the collected metrics to JSON.
    pub fn to_json(&self) -> Value {
        let inner = self.inner.lock().unwrap();
        let avg: HashMap<&str, f64> = inner
            .durations
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(k, v)| {
                let sum: f64 = v.iter().sum();
                (k.as_str(), sum / v.len() as f64)
            })
            .collect();
        json!({
            "total_events": inner.counts.values().sum::<usize>(),
            "events_by_phase": inner.counts,
            "avg_duration_ms": avg,
        })
    }
}

impl Default for MetricsCallbackHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl CallbackHandler for MetricsCallbackHandler {
    fn on_event(&self, data: &CallbackData) {
        let phase_str = data.phase.as_str().to_string();
        let mut inner = self.inner.lock().unwrap();

        *inner.counts.entry(phase_str.clone()).or_insert(0) += 1;

        // Track timing: start phases create a pending entry, end/error phases
        // complete it.
        let is_start = phase_str.ends_with("_start");
        let is_end = phase_str.ends_with("_end") || phase_str.ends_with("_error");

        if is_start {
            inner
                .pending
                .insert(data.run_id.clone(), Instant::now());
        } else if is_end {
            if let Some(start) = inner.pending.remove(&data.run_id) {
                let elapsed = start.elapsed().as_secs_f64() * 1000.0;
                inner
                    .durations
                    .entry(phase_str)
                    .or_default()
                    .push(elapsed);
            }
        }
    }

    fn name(&self) -> &str {
        "MetricsCallbackHandler"
    }
}

// ---------------------------------------------------------------------------
// CallbackManager
// ---------------------------------------------------------------------------

/// Manages multiple callback handlers and dispatches events to them.
pub struct CallbackManager {
    handlers: Mutex<Vec<Box<dyn CallbackHandler>>>,
}

impl CallbackManager {
    /// Creates a new empty manager.
    pub fn new() -> Self {
        Self {
            handlers: Mutex::new(Vec::new()),
        }
    }

    /// Registers a handler.
    pub fn add_handler(&self, handler: Box<dyn CallbackHandler>) {
        self.handlers.lock().unwrap().push(handler);
    }

    /// Emits a [`CallbackData`] event to all handlers that accept its phase.
    pub fn emit(&self, data: CallbackData) {
        let handlers = self.handlers.lock().unwrap();
        for h in handlers.iter() {
            if h.handles_phase(&data.phase) {
                h.on_event(&data);
            }
        }
    }

    /// Convenience method that constructs a [`CallbackData`] and emits it.
    pub fn emit_phase(&self, phase: CallbackPhase, run_id: impl Into<String>, data: Value) {
        let cb = CallbackDataBuilder::new(phase, run_id)
            .data(data)
            .build();
        self.emit(cb);
    }

    /// Returns the number of registered handlers.
    pub fn handler_count(&self) -> usize {
        self.handlers.lock().unwrap().len()
    }

    /// Generates a new unique run id.
    pub fn new_run_id() -> String {
        Uuid::new_v4().to_string()
    }
}

impl Default for CallbackManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CallbackScope
// ---------------------------------------------------------------------------

/// RAII guard that emits a start event on creation and an end (or error)
/// event when dropped.
pub struct CallbackScope<'a> {
    manager: &'a CallbackManager,
    phase_end: CallbackPhase,
    run_id: String,
    data: Value,
    failed: Mutex<Option<String>>,
}

impl<'a> CallbackScope<'a> {
    /// Creates a new scope and immediately emits the start event.
    pub fn new(
        manager: &'a CallbackManager,
        phase_start: CallbackPhase,
        phase_end: CallbackPhase,
        run_id: String,
        data: Value,
    ) -> Self {
        let start = CallbackDataBuilder::new(phase_start, &run_id)
            .data(data.clone())
            .build();
        manager.emit(start);

        Self {
            manager,
            phase_end,
            run_id,
            data,
            failed: Mutex::new(None),
        }
    }

    /// Marks this scope as failed. When dropped the error phase will be
    /// emitted instead of the normal end phase.
    pub fn fail(&self, error: String) {
        *self.failed.lock().unwrap() = Some(error);
    }
}

impl<'a> Drop for CallbackScope<'a> {
    fn drop(&mut self) {
        let failed = self.failed.lock().unwrap().take();
        if let Some(error) = failed {
            // Derive the error phase from the end phase.
            let error_phase = match &self.phase_end {
                CallbackPhase::ChainEnd => CallbackPhase::ChainError,
                CallbackPhase::LlmEnd => CallbackPhase::LlmError,
                CallbackPhase::ToolEnd => CallbackPhase::ToolError,
                other => other.clone(),
            };
            let cb = CallbackDataBuilder::new(error_phase, &self.run_id)
                .data(json!({ "error": error }))
                .build();
            self.manager.emit(cb);
        } else {
            let cb = CallbackDataBuilder::new(self.phase_end.clone(), &self.run_id)
                .data(self.data.clone())
                .build();
            self.manager.emit(cb);
        }
    }
}

// ---------------------------------------------------------------------------
// FilteredHandler
// ---------------------------------------------------------------------------

/// Wraps a [`CallbackHandler`] and only forwards events whose phase is in
/// the configured allow-list.
pub struct FilteredHandler {
    inner: Box<dyn CallbackHandler>,
    phases: Vec<CallbackPhase>,
}

impl FilteredHandler {
    /// Creates a new filtered handler.
    pub fn new(inner: Box<dyn CallbackHandler>, phases: Vec<CallbackPhase>) -> Self {
        Self { inner, phases }
    }
}

impl CallbackHandler for FilteredHandler {
    fn on_event(&self, data: &CallbackData) {
        self.inner.on_event(data);
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn handles_phase(&self, phase: &CallbackPhase) -> bool {
        self.phases.contains(phase)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;

    // ---- CallbackPhase tests ----

    #[test]
    fn phase_as_str_builtin() {
        assert_eq!(CallbackPhase::ChainStart.as_str(), "chain_start");
        assert_eq!(CallbackPhase::ChainEnd.as_str(), "chain_end");
        assert_eq!(CallbackPhase::ChainError.as_str(), "chain_error");
        assert_eq!(CallbackPhase::LlmStart.as_str(), "llm_start");
        assert_eq!(CallbackPhase::LlmEnd.as_str(), "llm_end");
        assert_eq!(CallbackPhase::LlmError.as_str(), "llm_error");
        assert_eq!(CallbackPhase::ToolStart.as_str(), "tool_start");
        assert_eq!(CallbackPhase::ToolEnd.as_str(), "tool_end");
        assert_eq!(CallbackPhase::ToolError.as_str(), "tool_error");
        assert_eq!(CallbackPhase::RetrieverStart.as_str(), "retriever_start");
        assert_eq!(CallbackPhase::RetrieverEnd.as_str(), "retriever_end");
    }

    #[test]
    fn phase_as_str_custom() {
        let p = CallbackPhase::Custom("my_phase".into());
        assert_eq!(p.as_str(), "my_phase");
    }

    #[test]
    fn phase_display() {
        assert_eq!(format!("{}", CallbackPhase::ChainStart), "chain_start");
        assert_eq!(
            format!("{}", CallbackPhase::Custom("x".into())),
            "x"
        );
    }

    #[test]
    fn phase_equality() {
        assert_eq!(CallbackPhase::ChainStart, CallbackPhase::ChainStart);
        assert_ne!(CallbackPhase::ChainStart, CallbackPhase::ChainEnd);
        assert_eq!(
            CallbackPhase::Custom("a".into()),
            CallbackPhase::Custom("a".into()),
        );
        assert_ne!(
            CallbackPhase::Custom("a".into()),
            CallbackPhase::Custom("b".into()),
        );
    }

    #[test]
    fn phase_clone() {
        let p = CallbackPhase::ToolStart;
        let p2 = p.clone();
        assert_eq!(p, p2);
    }

    #[test]
    fn phase_debug() {
        let dbg = format!("{:?}", CallbackPhase::LlmEnd);
        assert!(dbg.contains("LlmEnd"));
    }

    #[test]
    fn phase_hash() {
        let mut map = HashMap::new();
        map.insert(CallbackPhase::ChainStart, 1);
        map.insert(CallbackPhase::ChainEnd, 2);
        assert_eq!(map[&CallbackPhase::ChainStart], 1);
        assert_eq!(map[&CallbackPhase::ChainEnd], 2);
    }

    // ---- CallbackData & Builder tests ----

    #[test]
    fn builder_minimal() {
        let d = CallbackDataBuilder::new(CallbackPhase::ChainStart, "run-1")
            .build();
        assert_eq!(d.phase, CallbackPhase::ChainStart);
        assert_eq!(d.run_id, "run-1");
        assert!(d.parent_run_id.is_none());
        assert_eq!(d.data, Value::Null);
        assert!(!d.timestamp.is_empty());
    }

    #[test]
    fn builder_full() {
        let d = CallbackDataBuilder::new(CallbackPhase::LlmEnd, "run-2")
            .parent_run_id("parent-1")
            .data(json!({"result": 42}))
            .timestamp("2026-01-01T00:00:00Z")
            .build();
        assert_eq!(d.run_id, "run-2");
        assert_eq!(d.parent_run_id.as_deref(), Some("parent-1"));
        assert_eq!(d.data, json!({"result": 42}));
        assert_eq!(d.timestamp, "2026-01-01T00:00:00Z");
    }

    #[test]
    fn data_to_json() {
        let d = CallbackDataBuilder::new(CallbackPhase::ToolStart, "r")
            .data(json!("hello"))
            .timestamp("ts")
            .build();
        let j = d.to_json();
        assert_eq!(j["phase"], "tool_start");
        assert_eq!(j["run_id"], "r");
        assert_eq!(j["data"], "hello");
        assert_eq!(j["timestamp"], "ts");
        assert!(j["parent_run_id"].is_null());
    }

    #[test]
    fn data_to_json_with_parent() {
        let d = CallbackDataBuilder::new(CallbackPhase::ChainEnd, "r")
            .parent_run_id("p")
            .timestamp("ts")
            .build();
        let j = d.to_json();
        assert_eq!(j["parent_run_id"], "p");
    }

    // ---- ConsoleCallbackHandler tests ----

    #[test]
    fn console_handler_logs_events() {
        let h = ConsoleCallbackHandler::new();
        let d = CallbackDataBuilder::new(CallbackPhase::ChainStart, "r1")
            .data(json!("input"))
            .timestamp("t0")
            .build();
        h.on_event(&d);
        let logs = h.logs();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("chain_start"));
        assert!(logs[0].contains("r1"));
    }

    #[test]
    fn console_handler_clear() {
        let h = ConsoleCallbackHandler::new();
        let d = CallbackDataBuilder::new(CallbackPhase::ChainEnd, "r1")
            .timestamp("t0")
            .build();
        h.on_event(&d);
        assert_eq!(h.logs().len(), 1);
        h.clear();
        assert!(h.logs().is_empty());
    }

    #[test]
    fn console_handler_name() {
        let h = ConsoleCallbackHandler::new();
        assert_eq!(h.name(), "ConsoleCallbackHandler");
    }

    #[test]
    fn console_handler_default() {
        let h = ConsoleCallbackHandler::default();
        assert!(h.logs().is_empty());
    }

    #[test]
    fn console_handler_multiple_events() {
        let h = ConsoleCallbackHandler::new();
        for i in 0..5 {
            let d = CallbackDataBuilder::new(CallbackPhase::LlmStart, format!("r{}", i))
                .timestamp("t")
                .build();
            h.on_event(&d);
        }
        assert_eq!(h.logs().len(), 5);
    }

    // ---- MetricsCallbackHandler tests ----

    #[test]
    fn metrics_total_events() {
        let m = MetricsCallbackHandler::new();
        assert_eq!(m.total_events(), 0);
        let d = CallbackDataBuilder::new(CallbackPhase::ChainStart, "r1")
            .timestamp("t")
            .build();
        m.on_event(&d);
        assert_eq!(m.total_events(), 1);
    }

    #[test]
    fn metrics_events_by_phase() {
        let m = MetricsCallbackHandler::new();
        let d1 = CallbackDataBuilder::new(CallbackPhase::ChainStart, "r1")
            .timestamp("t")
            .build();
        let d2 = CallbackDataBuilder::new(CallbackPhase::ChainStart, "r2")
            .timestamp("t")
            .build();
        let d3 = CallbackDataBuilder::new(CallbackPhase::ChainEnd, "r1")
            .timestamp("t")
            .build();
        m.on_event(&d1);
        m.on_event(&d2);
        m.on_event(&d3);
        let by_phase = m.events_by_phase();
        assert_eq!(by_phase["chain_start"], 2);
        assert_eq!(by_phase["chain_end"], 1);
    }

    #[test]
    fn metrics_avg_duration() {
        let m = MetricsCallbackHandler::new();
        // Record start then end for same run_id to get a duration sample.
        let start = CallbackDataBuilder::new(CallbackPhase::LlmStart, "r1")
            .timestamp("t")
            .build();
        m.on_event(&start);
        // Small delay would be nice but we just test it records something >= 0.
        let end = CallbackDataBuilder::new(CallbackPhase::LlmEnd, "r1")
            .timestamp("t")
            .build();
        m.on_event(&end);
        let avg = m.avg_duration_ms(&CallbackPhase::LlmEnd);
        assert!(avg.is_some());
        assert!(avg.unwrap() >= 0.0);
    }

    #[test]
    fn metrics_avg_duration_no_data() {
        let m = MetricsCallbackHandler::new();
        assert!(m.avg_duration_ms(&CallbackPhase::ChainEnd).is_none());
    }

    #[test]
    fn metrics_to_json() {
        let m = MetricsCallbackHandler::new();
        let d = CallbackDataBuilder::new(CallbackPhase::ToolStart, "r1")
            .timestamp("t")
            .build();
        m.on_event(&d);
        let j = m.to_json();
        assert_eq!(j["total_events"], 1);
        assert_eq!(j["events_by_phase"]["tool_start"], 1);
    }

    #[test]
    fn metrics_name() {
        assert_eq!(MetricsCallbackHandler::new().name(), "MetricsCallbackHandler");
    }

    #[test]
    fn metrics_default() {
        let m = MetricsCallbackHandler::default();
        assert_eq!(m.total_events(), 0);
    }

    // ---- CallbackManager tests ----

    #[test]
    fn manager_new_empty() {
        let mgr = CallbackManager::new();
        assert_eq!(mgr.handler_count(), 0);
    }

    #[test]
    fn manager_add_handler() {
        let mgr = CallbackManager::new();
        mgr.add_handler(Box::new(ConsoleCallbackHandler::new()));
        assert_eq!(mgr.handler_count(), 1);
    }

    #[test]
    fn manager_emit_dispatches() {
        let mgr = CallbackManager::new();
        let metrics = Arc::new(MetricsCallbackHandler::new());

        // We can't add Arc<T> directly. Use a wrapper.
        struct ArcMetrics(Arc<MetricsCallbackHandler>);
        impl CallbackHandler for ArcMetrics {
            fn on_event(&self, data: &CallbackData) {
                self.0.on_event(data);
            }
            fn name(&self) -> &str {
                "ArcMetrics"
            }
        }

        mgr.add_handler(Box::new(ArcMetrics(metrics.clone())));
        mgr.emit_phase(CallbackPhase::ChainStart, "r1", json!({}));
        assert_eq!(metrics.total_events(), 1);
    }

    #[test]
    fn manager_emit_phase() {
        struct Counter(Mutex<usize>);
        impl CallbackHandler for Counter {
            fn on_event(&self, _data: &CallbackData) {
                *self.0.lock().unwrap() += 1;
            }
            fn name(&self) -> &str {
                "Counter"
            }
        }

        let mgr = CallbackManager::new();
        let c = Arc::new(Counter(Mutex::new(0)));
        struct ArcCounter(Arc<Counter>);
        impl CallbackHandler for ArcCounter {
            fn on_event(&self, d: &CallbackData) {
                self.0.on_event(d);
            }
            fn name(&self) -> &str {
                "ArcCounter"
            }
        }
        mgr.add_handler(Box::new(ArcCounter(c.clone())));
        mgr.emit_phase(CallbackPhase::LlmStart, "r", json!(null));
        mgr.emit_phase(CallbackPhase::LlmEnd, "r", json!(null));
        assert_eq!(*c.0.lock().unwrap(), 2);
    }

    #[test]
    fn manager_new_run_id_unique() {
        let id1 = CallbackManager::new_run_id();
        let id2 = CallbackManager::new_run_id();
        assert_ne!(id1, id2);
        assert!(!id1.is_empty());
    }

    #[test]
    fn manager_default() {
        let mgr = CallbackManager::default();
        assert_eq!(mgr.handler_count(), 0);
    }

    #[test]
    fn manager_multiple_handlers() {
        let mgr = CallbackManager::new();
        mgr.add_handler(Box::new(ConsoleCallbackHandler::new()));
        mgr.add_handler(Box::new(MetricsCallbackHandler::new()));
        assert_eq!(mgr.handler_count(), 2);
    }

    // ---- CallbackScope tests ----

    #[test]
    fn scope_emits_start_and_end() {
        let metrics = Arc::new(MetricsCallbackHandler::new());
        let mgr = CallbackManager::new();
        struct AM(Arc<MetricsCallbackHandler>);
        impl CallbackHandler for AM {
            fn on_event(&self, d: &CallbackData) { self.0.on_event(d); }
            fn name(&self) -> &str { "AM" }
        }
        mgr.add_handler(Box::new(AM(metrics.clone())));

        {
            let _scope = CallbackScope::new(
                &mgr,
                CallbackPhase::ChainStart,
                CallbackPhase::ChainEnd,
                "run-1".into(),
                json!({}),
            );
            // Start event emitted on creation.
            assert_eq!(metrics.total_events(), 1);
        }
        // End event emitted on drop.
        assert_eq!(metrics.total_events(), 2);
        let by_phase = metrics.events_by_phase();
        assert_eq!(by_phase["chain_start"], 1);
        assert_eq!(by_phase["chain_end"], 1);
    }

    #[test]
    fn scope_fail_emits_error() {
        let metrics = Arc::new(MetricsCallbackHandler::new());
        let mgr = CallbackManager::new();
        struct AM(Arc<MetricsCallbackHandler>);
        impl CallbackHandler for AM {
            fn on_event(&self, d: &CallbackData) { self.0.on_event(d); }
            fn name(&self) -> &str { "AM" }
        }
        mgr.add_handler(Box::new(AM(metrics.clone())));

        {
            let scope = CallbackScope::new(
                &mgr,
                CallbackPhase::LlmStart,
                CallbackPhase::LlmEnd,
                "run-2".into(),
                json!({}),
            );
            scope.fail("timeout".into());
        }
        let by_phase = metrics.events_by_phase();
        assert_eq!(by_phase.get("llm_start"), Some(&1));
        assert_eq!(by_phase.get("llm_error"), Some(&1));
        assert!(by_phase.get("llm_end").is_none());
    }

    #[test]
    fn scope_tool_fail_emits_tool_error() {
        let metrics = Arc::new(MetricsCallbackHandler::new());
        let mgr = CallbackManager::new();
        struct AM(Arc<MetricsCallbackHandler>);
        impl CallbackHandler for AM {
            fn on_event(&self, d: &CallbackData) { self.0.on_event(d); }
            fn name(&self) -> &str { "AM" }
        }
        mgr.add_handler(Box::new(AM(metrics.clone())));

        {
            let scope = CallbackScope::new(
                &mgr,
                CallbackPhase::ToolStart,
                CallbackPhase::ToolEnd,
                "run-t".into(),
                json!({}),
            );
            scope.fail("not found".into());
        }
        let by_phase = metrics.events_by_phase();
        assert_eq!(by_phase.get("tool_error"), Some(&1));
    }

    #[test]
    fn scope_chain_fail_emits_chain_error() {
        let console = Arc::new(ConsoleCallbackHandler::new());
        let mgr = CallbackManager::new();
        struct AC(Arc<ConsoleCallbackHandler>);
        impl CallbackHandler for AC {
            fn on_event(&self, d: &CallbackData) { self.0.on_event(d); }
            fn name(&self) -> &str { "AC" }
        }
        mgr.add_handler(Box::new(AC(console.clone())));

        {
            let scope = CallbackScope::new(
                &mgr,
                CallbackPhase::ChainStart,
                CallbackPhase::ChainEnd,
                "run-c".into(),
                json!({}),
            );
            scope.fail("bad input".into());
        }
        let logs = console.logs();
        assert_eq!(logs.len(), 2);
        assert!(logs[1].contains("chain_error"));
        assert!(!logs[1].contains("chain_end"));
    }

    // ---- FilteredHandler tests ----

    #[test]
    fn filtered_handler_allows_matching_phase() {
        let console = Arc::new(ConsoleCallbackHandler::new());
        struct AC(Arc<ConsoleCallbackHandler>);
        impl CallbackHandler for AC {
            fn on_event(&self, d: &CallbackData) { self.0.on_event(d); }
            fn name(&self) -> &str { "AC" }
        }
        let filtered = FilteredHandler::new(
            Box::new(AC(console.clone())),
            vec![CallbackPhase::ChainStart],
        );

        assert!(filtered.handles_phase(&CallbackPhase::ChainStart));
        assert!(!filtered.handles_phase(&CallbackPhase::ChainEnd));
    }

    #[test]
    fn filtered_handler_blocks_non_matching() {
        let mgr = CallbackManager::new();
        let console = Arc::new(ConsoleCallbackHandler::new());
        struct AC(Arc<ConsoleCallbackHandler>);
        impl CallbackHandler for AC {
            fn on_event(&self, d: &CallbackData) { self.0.on_event(d); }
            fn name(&self) -> &str { "AC" }
        }
        let filtered = FilteredHandler::new(
            Box::new(AC(console.clone())),
            vec![CallbackPhase::LlmStart],
        );
        mgr.add_handler(Box::new(filtered));

        mgr.emit_phase(CallbackPhase::ChainStart, "r1", json!({}));
        mgr.emit_phase(CallbackPhase::LlmStart, "r2", json!({}));

        // Only the LlmStart event should have been forwarded.
        assert_eq!(console.logs().len(), 1);
        assert!(console.logs()[0].contains("llm_start"));
    }

    #[test]
    fn filtered_handler_name_delegates() {
        struct Named;
        impl CallbackHandler for Named {
            fn on_event(&self, _: &CallbackData) {}
            fn name(&self) -> &str { "MyHandler" }
        }
        let f = FilteredHandler::new(Box::new(Named), vec![]);
        assert_eq!(f.name(), "MyHandler");
    }

    #[test]
    fn filtered_handler_multiple_phases() {
        let f = FilteredHandler::new(
            Box::new(ConsoleCallbackHandler::new()),
            vec![CallbackPhase::ChainStart, CallbackPhase::ChainEnd, CallbackPhase::ChainError],
        );
        assert!(f.handles_phase(&CallbackPhase::ChainStart));
        assert!(f.handles_phase(&CallbackPhase::ChainEnd));
        assert!(f.handles_phase(&CallbackPhase::ChainError));
        assert!(!f.handles_phase(&CallbackPhase::LlmStart));
    }

    #[test]
    fn filtered_handler_empty_phases_blocks_all() {
        let f = FilteredHandler::new(
            Box::new(ConsoleCallbackHandler::new()),
            vec![],
        );
        assert!(!f.handles_phase(&CallbackPhase::ChainStart));
        assert!(!f.handles_phase(&CallbackPhase::LlmEnd));
    }

    #[test]
    fn filtered_handler_custom_phase() {
        let f = FilteredHandler::new(
            Box::new(ConsoleCallbackHandler::new()),
            vec![CallbackPhase::Custom("my_event".into())],
        );
        assert!(f.handles_phase(&CallbackPhase::Custom("my_event".into())));
        assert!(!f.handles_phase(&CallbackPhase::Custom("other".into())));
    }

    // ---- Edge case tests ----

    #[test]
    fn emit_with_no_handlers() {
        let mgr = CallbackManager::new();
        // Should not panic.
        mgr.emit_phase(CallbackPhase::ChainStart, "r1", json!({}));
    }

    #[test]
    fn callback_data_clone() {
        let d = CallbackDataBuilder::new(CallbackPhase::ToolEnd, "r1")
            .data(json!({"key": "val"}))
            .timestamp("ts")
            .build();
        let d2 = d.clone();
        assert_eq!(d.run_id, d2.run_id);
        assert_eq!(d.data, d2.data);
    }

    #[test]
    fn callback_data_debug() {
        let d = CallbackDataBuilder::new(CallbackPhase::RetrieverStart, "r1")
            .timestamp("ts")
            .build();
        let dbg = format!("{:?}", d);
        assert!(dbg.contains("RetrieverStart"));
    }

    #[test]
    fn metrics_multiple_durations() {
        let m = MetricsCallbackHandler::new();
        for i in 0..3 {
            let rid = format!("r{}", i);
            m.on_event(
                &CallbackDataBuilder::new(CallbackPhase::ChainStart, &rid)
                    .timestamp("t")
                    .build(),
            );
            m.on_event(
                &CallbackDataBuilder::new(CallbackPhase::ChainEnd, &rid)
                    .timestamp("t")
                    .build(),
            );
        }
        let avg = m.avg_duration_ms(&CallbackPhase::ChainEnd);
        assert!(avg.is_some());
        assert_eq!(m.events_by_phase()["chain_start"], 3);
        assert_eq!(m.events_by_phase()["chain_end"], 3);
    }

    #[test]
    fn metrics_error_phase_records_duration() {
        let m = MetricsCallbackHandler::new();
        m.on_event(
            &CallbackDataBuilder::new(CallbackPhase::ToolStart, "r1")
                .timestamp("t")
                .build(),
        );
        m.on_event(
            &CallbackDataBuilder::new(CallbackPhase::ToolError, "r1")
                .timestamp("t")
                .build(),
        );
        assert!(m.avg_duration_ms(&CallbackPhase::ToolError).is_some());
    }

    #[test]
    fn scope_preserves_data_on_end() {
        let console = Arc::new(ConsoleCallbackHandler::new());
        let mgr = CallbackManager::new();
        struct AC(Arc<ConsoleCallbackHandler>);
        impl CallbackHandler for AC {
            fn on_event(&self, d: &CallbackData) { self.0.on_event(d); }
            fn name(&self) -> &str { "AC" }
        }
        mgr.add_handler(Box::new(AC(console.clone())));

        {
            let _scope = CallbackScope::new(
                &mgr,
                CallbackPhase::RetrieverStart,
                CallbackPhase::RetrieverEnd,
                "rr".into(),
                json!({"query": "test"}),
            );
        }
        let logs = console.logs();
        assert_eq!(logs.len(), 2);
        assert!(logs[0].contains("retriever_start"));
        assert!(logs[1].contains("retriever_end"));
    }

    #[test]
    fn builder_data_defaults_to_null() {
        let d = CallbackDataBuilder::new(CallbackPhase::ChainStart, "r")
            .timestamp("t")
            .build();
        assert_eq!(d.data, Value::Null);
    }

    #[test]
    fn custom_phase_in_manager() {
        let console = Arc::new(ConsoleCallbackHandler::new());
        let mgr = CallbackManager::new();
        struct AC(Arc<ConsoleCallbackHandler>);
        impl CallbackHandler for AC {
            fn on_event(&self, d: &CallbackData) { self.0.on_event(d); }
            fn name(&self) -> &str { "AC" }
        }
        mgr.add_handler(Box::new(AC(console.clone())));
        mgr.emit_phase(CallbackPhase::Custom("my_custom".into()), "r1", json!(42));
        let logs = console.logs();
        assert_eq!(logs.len(), 1);
        assert!(logs[0].contains("my_custom"));
    }
}
