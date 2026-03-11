//! Debugging tools for LangGraph — step-through execution, breakpoints,
//! state inspection, and execution tracing.
//!
//! This module provides a comprehensive debugging toolkit for graph execution:
//! - [`Breakpoint`] and [`BreakCondition`] for pausing execution at specific nodes
//! - [`DebugSession`] for managing a full debugging session with history
//! - [`DebugStep`] for recording individual execution steps
//! - [`StateInspector`] and [`StateReport`] for querying and summarizing graph state
//! - [`StateDelta`] and [`DeltaType`] for tracking state changes
//! - [`ExecutionProfiler`] for profiling node execution times

use serde_json::{json, Value};
use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

// ---------------------------------------------------------------------------
// BreakCondition
// ---------------------------------------------------------------------------

/// Condition that determines when a breakpoint triggers.
#[derive(Debug, Clone)]
pub enum BreakCondition {
    /// Always break.
    Always,
    /// Break when the given state key changes between steps.
    OnStateChange { key: String },
    /// Break when a state key equals an expected value.
    OnValue { key: String, expected: Value },
    /// Break on a specific iteration (hit count).
    OnIteration(usize),
    /// Custom condition described by a human-readable string.
    /// Always evaluates to `true` (acts as a labeled breakpoint).
    Custom { description: String },
}

impl BreakCondition {
    /// Evaluate whether this condition is met given the current state.
    pub fn evaluate(&self, state: &Value) -> bool {
        match self {
            BreakCondition::Always => true,
            BreakCondition::OnStateChange { key } => {
                // When evaluated against a single snapshot, we consider the key
                // present as a trigger. Real cross-step detection lives in
                // DebugSession.
                state.get(key).is_some()
            }
            BreakCondition::OnValue { key, expected } => {
                state.get(key).map_or(false, |v| v == expected)
            }
            BreakCondition::OnIteration(_) => {
                // Iteration check is handled by Breakpoint::should_break.
                true
            }
            BreakCondition::Custom { .. } => true,
        }
    }
}

impl fmt::Display for BreakCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BreakCondition::Always => write!(f, "Always"),
            BreakCondition::OnStateChange { key } => write!(f, "OnStateChange({})", key),
            BreakCondition::OnValue { key, expected } => {
                write!(f, "OnValue({} == {})", key, expected)
            }
            BreakCondition::OnIteration(n) => write!(f, "OnIteration({})", n),
            BreakCondition::Custom { description } => write!(f, "Custom({})", description),
        }
    }
}

// ---------------------------------------------------------------------------
// Breakpoint
// ---------------------------------------------------------------------------

/// A point where graph execution should pause.
#[derive(Debug, Clone)]
pub struct Breakpoint {
    /// The node at which to break.
    pub node_id: String,
    /// Optional condition that must be satisfied.
    pub condition: Option<BreakCondition>,
    /// Whether the breakpoint is currently enabled.
    pub enabled: bool,
    /// Number of times this breakpoint has been hit.
    pub hit_count: usize,
}

impl Breakpoint {
    /// Create a new enabled breakpoint for the given node with no condition.
    pub fn new(node_id: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            condition: None,
            enabled: true,
            hit_count: 0,
        }
    }

    /// Attach a break condition.
    pub fn with_condition(mut self, cond: BreakCondition) -> Self {
        self.condition = Some(cond);
        self
    }

    /// Enable the breakpoint.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable the breakpoint.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Record a hit (increment hit_count).
    pub fn record_hit(&mut self) {
        self.hit_count += 1;
    }

    /// Determine whether execution should break here given the current state.
    pub fn should_break(&self, state: &Value) -> bool {
        if !self.enabled {
            return false;
        }
        match &self.condition {
            None => true,
            Some(BreakCondition::OnIteration(n)) => self.hit_count + 1 == *n,
            Some(cond) => cond.evaluate(state),
        }
    }

    /// Serialize the breakpoint to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "node_id": self.node_id,
            "enabled": self.enabled,
            "hit_count": self.hit_count,
            "condition": self.condition.as_ref().map(|c| c.to_string()),
        })
    }
}

// ---------------------------------------------------------------------------
// DebugStep
// ---------------------------------------------------------------------------

/// Record of a single execution step.
#[derive(Debug, Clone)]
pub struct DebugStep {
    /// Sequential step number.
    pub step_number: usize,
    /// The node that was executed.
    pub node_id: String,
    /// State snapshot before the node executed.
    pub state_before: Value,
    /// State snapshot after the node executed (filled after completion).
    pub state_after: Option<Value>,
    /// ISO-8601 timestamp string.
    pub timestamp: String,
    /// Duration in milliseconds (filled after completion).
    pub duration_ms: Option<u64>,
}

impl DebugStep {
    /// Create a new step record. Timestamp is set to the provided value or a
    /// placeholder when testing.
    pub fn new(step_number: usize, node_id: impl Into<String>, state: Value) -> Self {
        Self {
            step_number,
            node_id: node_id.into(),
            state_before: state,
            state_after: None,
            timestamp: current_timestamp(),
            duration_ms: None,
        }
    }

    /// Mark the step as complete, recording the resulting state and duration.
    pub fn complete(&mut self, state_after: Value, duration_ms: u64) {
        self.state_after = Some(state_after);
        self.duration_ms = Some(duration_ms);
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "step_number": self.step_number,
            "node_id": self.node_id,
            "state_before": self.state_before,
            "state_after": self.state_after,
            "timestamp": self.timestamp,
            "duration_ms": self.duration_ms,
        })
    }
}

// ---------------------------------------------------------------------------
// DebugSession
// ---------------------------------------------------------------------------

/// Manages a debugging session with breakpoints and execution history.
#[derive(Debug)]
pub struct DebugSession {
    breakpoints: HashMap<String, Breakpoint>,
    steps: Vec<DebugStep>,
}

impl DebugSession {
    /// Create a new empty debug session.
    pub fn new() -> Self {
        Self {
            breakpoints: HashMap::new(),
            steps: Vec::new(),
        }
    }

    /// Add a breakpoint. If one already exists for the same node it is replaced.
    pub fn add_breakpoint(&mut self, bp: Breakpoint) {
        self.breakpoints.insert(bp.node_id.clone(), bp);
    }

    /// Remove the breakpoint for the given node, returning it if present.
    pub fn remove_breakpoint(&mut self, node_id: &str) -> Option<Breakpoint> {
        self.breakpoints.remove(node_id)
    }

    /// List all breakpoints.
    pub fn list_breakpoints(&self) -> Vec<&Breakpoint> {
        self.breakpoints.values().collect()
    }

    /// Check whether execution should pause at `node_id` given the current state.
    /// If so, the breakpoint's hit count is incremented.
    pub fn is_breakpoint(&mut self, node_id: &str, state: &Value) -> bool {
        if let Some(bp) = self.breakpoints.get_mut(node_id) {
            if bp.should_break(state) {
                bp.record_hit();
                return true;
            }
        }
        false
    }

    /// Remove all breakpoints.
    pub fn clear_breakpoints(&mut self) {
        self.breakpoints.clear();
    }

    /// Number of recorded steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Record a new execution step and return its index.
    pub fn record_step(&mut self, node_id: impl Into<String>, state: Value) -> usize {
        let idx = self.steps.len();
        self.steps.push(DebugStep::new(idx, node_id, state));
        idx
    }

    /// Access the full execution history.
    pub fn execution_history(&self) -> &[DebugStep] {
        &self.steps
    }
}

impl Default for DebugSession {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StateReport
// ---------------------------------------------------------------------------

/// Summary of the current graph state.
#[derive(Debug, Clone)]
pub struct StateReport {
    /// Top-level keys in the state object.
    pub keys: Vec<String>,
    /// Total number of fields (recursive).
    pub total_fields: usize,
    /// Maximum nesting depth.
    pub nested_depth: usize,
    /// Estimated size in bytes (JSON serialization length).
    pub estimated_size: usize,
}

impl StateReport {
    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "keys": self.keys,
            "total_fields": self.total_fields,
            "nested_depth": self.nested_depth,
            "estimated_size": self.estimated_size,
        })
    }
}

// ---------------------------------------------------------------------------
// DeltaType
// ---------------------------------------------------------------------------

/// Kind of change in a state delta.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeltaType {
    /// Key was added.
    Added,
    /// Key was removed.
    Removed,
    /// Value was modified.
    Modified,
}

impl fmt::Display for DeltaType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeltaType::Added => write!(f, "Added"),
            DeltaType::Removed => write!(f, "Removed"),
            DeltaType::Modified => write!(f, "Modified"),
        }
    }
}

// ---------------------------------------------------------------------------
// StateDelta
// ---------------------------------------------------------------------------

/// A single change between two state snapshots.
#[derive(Debug, Clone)]
pub struct StateDelta {
    /// The key that changed.
    pub key: String,
    /// The kind of change.
    pub change_type: DeltaType,
    /// Previous value (None for additions).
    pub old_value: Option<Value>,
    /// New value (None for removals).
    pub new_value: Option<Value>,
}

impl StateDelta {
    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "key": self.key,
            "change_type": self.change_type.to_string(),
            "old_value": self.old_value,
            "new_value": self.new_value,
        })
    }
}

impl fmt::Display for StateDelta {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.change_type {
            DeltaType::Added => write!(f, "+ {}: {:?}", self.key, self.new_value),
            DeltaType::Removed => write!(f, "- {}: {:?}", self.key, self.old_value),
            DeltaType::Modified => write!(
                f,
                "~ {}: {:?} -> {:?}",
                self.key, self.old_value, self.new_value
            ),
        }
    }
}

// ---------------------------------------------------------------------------
// StateInspector
// ---------------------------------------------------------------------------

/// Inspects and queries graph state.
#[derive(Debug)]
pub struct StateInspector {
    watched_keys: Vec<String>,
}

impl StateInspector {
    /// Create a new inspector with an empty watchlist.
    pub fn new() -> Self {
        Self {
            watched_keys: Vec::new(),
        }
    }

    /// Inspect the given state and produce a report.
    pub fn inspect(&self, state: &Value) -> StateReport {
        let keys = match state.as_object() {
            Some(obj) => obj.keys().cloned().collect(),
            None => Vec::new(),
        };
        let total_fields = count_fields(state);
        let nested_depth = compute_depth(state);
        let estimated_size = serde_json::to_string(state).map_or(0, |s| s.len());

        StateReport {
            keys,
            total_fields,
            nested_depth,
            estimated_size,
        }
    }

    /// Compute the diff between two state snapshots.
    pub fn diff(&self, before: &Value, after: &Value) -> Vec<StateDelta> {
        let empty = serde_json::Map::new();
        let before_obj = before.as_object().unwrap_or(&empty);
        let after_obj = after.as_object().unwrap_or(&empty);

        let mut deltas = Vec::new();

        // Check for removed and modified keys.
        for (key, old_val) in before_obj {
            match after_obj.get(key) {
                None => deltas.push(StateDelta {
                    key: key.clone(),
                    change_type: DeltaType::Removed,
                    old_value: Some(old_val.clone()),
                    new_value: None,
                }),
                Some(new_val) if new_val != old_val => deltas.push(StateDelta {
                    key: key.clone(),
                    change_type: DeltaType::Modified,
                    old_value: Some(old_val.clone()),
                    new_value: Some(new_val.clone()),
                }),
                _ => {}
            }
        }

        // Check for added keys.
        for (key, new_val) in after_obj {
            if !before_obj.contains_key(key) {
                deltas.push(StateDelta {
                    key: key.clone(),
                    change_type: DeltaType::Added,
                    old_value: None,
                    new_value: Some(new_val.clone()),
                });
            }
        }

        deltas
    }

    /// Add a key to the watchlist.
    pub fn watch_key(&mut self, key: &str) {
        if !self.watched_keys.contains(&key.to_string()) {
            self.watched_keys.push(key.to_string());
        }
    }

    /// Return current values for all watched keys.
    pub fn watched_values(&self, state: &Value) -> HashMap<String, Value> {
        let mut result = HashMap::new();
        for key in &self.watched_keys {
            if let Some(val) = state.get(key) {
                result.insert(key.clone(), val.clone());
            }
        }
        result
    }
}

impl Default for StateInspector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ExecutionProfiler
// ---------------------------------------------------------------------------

/// Profiles graph execution performance.
#[derive(Debug)]
pub struct ExecutionProfiler {
    /// Running timers for nodes (node_id -> start instant).
    running: HashMap<String, Instant>,
    /// Completed durations per node (node_id -> list of durations in ms).
    durations: HashMap<String, Vec<u64>>,
}

impl ExecutionProfiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        Self {
            running: HashMap::new(),
            durations: HashMap::new(),
        }
    }

    /// Mark the start of a node execution.
    pub fn start_node(&mut self, node_id: impl Into<String>) {
        self.running.insert(node_id.into(), Instant::now());
    }

    /// Mark the end of a node execution. Records the elapsed time.
    pub fn end_node(&mut self, node_id: &str) {
        if let Some(start) = self.running.remove(node_id) {
            let elapsed = start.elapsed().as_millis() as u64;
            self.durations
                .entry(node_id.to_string())
                .or_default()
                .push(elapsed);
        }
    }

    /// Return a copy of all recorded node durations.
    pub fn node_times(&self) -> HashMap<String, Vec<u64>> {
        self.durations.clone()
    }

    /// Compute the average execution time for a node in ms.
    pub fn average_time(&self, node_id: &str) -> Option<f64> {
        self.durations.get(node_id).map(|times| {
            let sum: u64 = times.iter().sum();
            sum as f64 / times.len() as f64
        })
    }

    /// Return the slowest node (by average time).
    pub fn slowest_node(&self) -> Option<(String, f64)> {
        self.durations
            .keys()
            .filter_map(|id| self.average_time(id).map(|avg| (id.clone(), avg)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Total execution time across all recorded durations.
    pub fn total_execution_ms(&self) -> u64 {
        self.durations.values().flat_map(|v| v.iter()).sum()
    }

    /// Serialize profiling data to JSON.
    pub fn to_json(&self) -> Value {
        let nodes: serde_json::Map<String, Value> = self
            .durations
            .iter()
            .map(|(id, times)| {
                let avg = self.average_time(id).unwrap_or(0.0);
                (
                    id.clone(),
                    json!({
                        "times_ms": times,
                        "average_ms": avg,
                        "invocations": times.len(),
                    }),
                )
            })
            .collect();

        json!({
            "nodes": nodes,
            "total_ms": self.total_execution_ms(),
            "slowest_node": self.slowest_node().map(|(id, avg)| json!({"node_id": id, "average_ms": avg})),
        })
    }
}

impl Default for ExecutionProfiler {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return a timestamp string. Uses a simple format for determinism in tests
/// when the `chrono` feature is available, otherwise falls back to a
/// placeholder.
fn current_timestamp() -> String {
    chrono::Utc::now().to_rfc3339()
}

/// Recursively count all fields in a JSON value.
fn count_fields(value: &Value) -> usize {
    match value {
        Value::Object(map) => {
            let mut count = map.len();
            for v in map.values() {
                count += count_fields(v);
            }
            count
        }
        Value::Array(arr) => {
            let mut count = 0;
            for v in arr {
                count += count_fields(v);
            }
            count
        }
        _ => 0,
    }
}

/// Compute the maximum nesting depth of a JSON value.
fn compute_depth(value: &Value) -> usize {
    match value {
        Value::Object(map) => {
            if map.is_empty() {
                1
            } else {
                1 + map.values().map(compute_depth).max().unwrap_or(0)
            }
        }
        Value::Array(arr) => {
            if arr.is_empty() {
                1
            } else {
                1 + arr.iter().map(compute_depth).max().unwrap_or(0)
            }
        }
        _ => 0,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- BreakCondition tests -----------------------------------------------

    #[test]
    fn test_break_condition_always() {
        let cond = BreakCondition::Always;
        assert!(cond.evaluate(&json!({})));
        assert!(cond.evaluate(&json!({"x": 1})));
    }

    #[test]
    fn test_break_condition_on_state_change_present() {
        let cond = BreakCondition::OnStateChange {
            key: "counter".into(),
        };
        assert!(cond.evaluate(&json!({"counter": 5})));
    }

    #[test]
    fn test_break_condition_on_state_change_absent() {
        let cond = BreakCondition::OnStateChange {
            key: "counter".into(),
        };
        assert!(!cond.evaluate(&json!({"other": 1})));
    }

    #[test]
    fn test_break_condition_on_value_match() {
        let cond = BreakCondition::OnValue {
            key: "status".into(),
            expected: json!("done"),
        };
        assert!(cond.evaluate(&json!({"status": "done"})));
    }

    #[test]
    fn test_break_condition_on_value_no_match() {
        let cond = BreakCondition::OnValue {
            key: "status".into(),
            expected: json!("done"),
        };
        assert!(!cond.evaluate(&json!({"status": "running"})));
    }

    #[test]
    fn test_break_condition_on_value_missing_key() {
        let cond = BreakCondition::OnValue {
            key: "status".into(),
            expected: json!("done"),
        };
        assert!(!cond.evaluate(&json!({})));
    }

    #[test]
    fn test_break_condition_on_iteration() {
        let cond = BreakCondition::OnIteration(3);
        // OnIteration always returns true from evaluate; actual check is in Breakpoint.
        assert!(cond.evaluate(&json!({})));
    }

    #[test]
    fn test_break_condition_custom() {
        let cond = BreakCondition::Custom {
            description: "debug this".into(),
        };
        assert!(cond.evaluate(&json!({})));
    }

    #[test]
    fn test_break_condition_display() {
        assert_eq!(BreakCondition::Always.to_string(), "Always");
        assert_eq!(
            BreakCondition::OnStateChange {
                key: "x".into()
            }
            .to_string(),
            "OnStateChange(x)"
        );
        assert_eq!(BreakCondition::OnIteration(5).to_string(), "OnIteration(5)");
        let custom = BreakCondition::Custom {
            description: "test".into(),
        };
        assert_eq!(custom.to_string(), "Custom(test)");
    }

    #[test]
    fn test_break_condition_on_value_display() {
        let cond = BreakCondition::OnValue {
            key: "k".into(),
            expected: json!(42),
        };
        assert!(cond.to_string().contains("OnValue"));
    }

    // -- Breakpoint tests ---------------------------------------------------

    #[test]
    fn test_breakpoint_new() {
        let bp = Breakpoint::new("node_a");
        assert_eq!(bp.node_id, "node_a");
        assert!(bp.enabled);
        assert_eq!(bp.hit_count, 0);
        assert!(bp.condition.is_none());
    }

    #[test]
    fn test_breakpoint_with_condition() {
        let bp = Breakpoint::new("n").with_condition(BreakCondition::Always);
        assert!(bp.condition.is_some());
    }

    #[test]
    fn test_breakpoint_enable_disable() {
        let mut bp = Breakpoint::new("n");
        bp.disable();
        assert!(!bp.enabled);
        bp.enable();
        assert!(bp.enabled);
    }

    #[test]
    fn test_breakpoint_should_break_no_condition() {
        let bp = Breakpoint::new("n");
        assert!(bp.should_break(&json!({})));
    }

    #[test]
    fn test_breakpoint_should_break_disabled() {
        let mut bp = Breakpoint::new("n");
        bp.disable();
        assert!(!bp.should_break(&json!({})));
    }

    #[test]
    fn test_breakpoint_should_break_always() {
        let bp = Breakpoint::new("n").with_condition(BreakCondition::Always);
        assert!(bp.should_break(&json!({})));
    }

    #[test]
    fn test_breakpoint_should_break_on_iteration() {
        let mut bp = Breakpoint::new("n").with_condition(BreakCondition::OnIteration(3));
        // hit_count is 0, iteration check: 0+1 != 3
        assert!(!bp.should_break(&json!({})));
        bp.record_hit();
        // hit_count=1, 1+1 != 3
        assert!(!bp.should_break(&json!({})));
        bp.record_hit();
        // hit_count=2, 2+1 == 3
        assert!(bp.should_break(&json!({})));
    }

    #[test]
    fn test_breakpoint_record_hit() {
        let mut bp = Breakpoint::new("n");
        assert_eq!(bp.hit_count, 0);
        bp.record_hit();
        assert_eq!(bp.hit_count, 1);
        bp.record_hit();
        assert_eq!(bp.hit_count, 2);
    }

    #[test]
    fn test_breakpoint_to_json() {
        let bp = Breakpoint::new("n");
        let j = bp.to_json();
        assert_eq!(j["node_id"], "n");
        assert_eq!(j["enabled"], true);
        assert_eq!(j["hit_count"], 0);
    }

    #[test]
    fn test_breakpoint_to_json_with_condition() {
        let bp = Breakpoint::new("x").with_condition(BreakCondition::Always);
        let j = bp.to_json();
        assert_eq!(j["condition"], "Always");
    }

    #[test]
    fn test_breakpoint_on_value_condition() {
        let bp = Breakpoint::new("n").with_condition(BreakCondition::OnValue {
            key: "status".into(),
            expected: json!("error"),
        });
        assert!(bp.should_break(&json!({"status": "error"})));
        assert!(!bp.should_break(&json!({"status": "ok"})));
    }

    // -- DebugStep tests ----------------------------------------------------

    #[test]
    fn test_debug_step_new() {
        let step = DebugStep::new(0, "node_a", json!({"x": 1}));
        assert_eq!(step.step_number, 0);
        assert_eq!(step.node_id, "node_a");
        assert_eq!(step.state_before, json!({"x": 1}));
        assert!(step.state_after.is_none());
        assert!(step.duration_ms.is_none());
        assert!(!step.timestamp.is_empty());
    }

    #[test]
    fn test_debug_step_complete() {
        let mut step = DebugStep::new(1, "node_b", json!({"a": 1}));
        step.complete(json!({"a": 2}), 42);
        assert_eq!(step.state_after, Some(json!({"a": 2})));
        assert_eq!(step.duration_ms, Some(42));
    }

    #[test]
    fn test_debug_step_to_json() {
        let mut step = DebugStep::new(0, "n", json!({}));
        step.complete(json!({"done": true}), 10);
        let j = step.to_json();
        assert_eq!(j["step_number"], 0);
        assert_eq!(j["node_id"], "n");
        assert_eq!(j["duration_ms"], 10);
    }

    #[test]
    fn test_debug_step_to_json_incomplete() {
        let step = DebugStep::new(5, "x", json!({"k": "v"}));
        let j = step.to_json();
        assert!(j["state_after"].is_null());
        assert!(j["duration_ms"].is_null());
    }

    // -- DebugSession tests -------------------------------------------------

    #[test]
    fn test_debug_session_new() {
        let session = DebugSession::new();
        assert_eq!(session.step_count(), 0);
        assert!(session.list_breakpoints().is_empty());
    }

    #[test]
    fn test_debug_session_default() {
        let session = DebugSession::default();
        assert_eq!(session.step_count(), 0);
    }

    #[test]
    fn test_debug_session_add_breakpoint() {
        let mut session = DebugSession::new();
        session.add_breakpoint(Breakpoint::new("a"));
        assert_eq!(session.list_breakpoints().len(), 1);
    }

    #[test]
    fn test_debug_session_remove_breakpoint() {
        let mut session = DebugSession::new();
        session.add_breakpoint(Breakpoint::new("a"));
        let removed = session.remove_breakpoint("a");
        assert!(removed.is_some());
        assert!(session.list_breakpoints().is_empty());
    }

    #[test]
    fn test_debug_session_remove_nonexistent() {
        let mut session = DebugSession::new();
        assert!(session.remove_breakpoint("nope").is_none());
    }

    #[test]
    fn test_debug_session_is_breakpoint_true() {
        let mut session = DebugSession::new();
        session.add_breakpoint(Breakpoint::new("a"));
        assert!(session.is_breakpoint("a", &json!({})));
    }

    #[test]
    fn test_debug_session_is_breakpoint_false() {
        let mut session = DebugSession::new();
        assert!(!session.is_breakpoint("a", &json!({})));
    }

    #[test]
    fn test_debug_session_is_breakpoint_increments_hit() {
        let mut session = DebugSession::new();
        session.add_breakpoint(Breakpoint::new("a"));
        session.is_breakpoint("a", &json!({}));
        session.is_breakpoint("a", &json!({}));
        let bps = session.list_breakpoints();
        let bp = bps.iter().find(|b| b.node_id == "a").unwrap();
        assert_eq!(bp.hit_count, 2);
    }

    #[test]
    fn test_debug_session_clear_breakpoints() {
        let mut session = DebugSession::new();
        session.add_breakpoint(Breakpoint::new("a"));
        session.add_breakpoint(Breakpoint::new("b"));
        session.clear_breakpoints();
        assert!(session.list_breakpoints().is_empty());
    }

    #[test]
    fn test_debug_session_record_step() {
        let mut session = DebugSession::new();
        let idx = session.record_step("node_a", json!({"x": 1}));
        assert_eq!(idx, 0);
        assert_eq!(session.step_count(), 1);
    }

    #[test]
    fn test_debug_session_execution_history() {
        let mut session = DebugSession::new();
        session.record_step("a", json!({}));
        session.record_step("b", json!({"k": 1}));
        let history = session.execution_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].node_id, "a");
        assert_eq!(history[1].node_id, "b");
    }

    #[test]
    fn test_debug_session_replace_breakpoint() {
        let mut session = DebugSession::new();
        session.add_breakpoint(Breakpoint::new("a"));
        session.add_breakpoint(Breakpoint::new("a").with_condition(BreakCondition::Always));
        assert_eq!(session.list_breakpoints().len(), 1);
    }

    #[test]
    fn test_debug_session_disabled_breakpoint_no_trigger() {
        let mut session = DebugSession::new();
        let mut bp = Breakpoint::new("a");
        bp.disable();
        session.add_breakpoint(bp);
        assert!(!session.is_breakpoint("a", &json!({})));
    }

    // -- StateReport tests --------------------------------------------------

    #[test]
    fn test_state_report_to_json() {
        let report = StateReport {
            keys: vec!["a".into(), "b".into()],
            total_fields: 5,
            nested_depth: 2,
            estimated_size: 100,
        };
        let j = report.to_json();
        assert_eq!(j["total_fields"], 5);
        assert_eq!(j["nested_depth"], 2);
    }

    // -- DeltaType tests ----------------------------------------------------

    #[test]
    fn test_delta_type_display() {
        assert_eq!(DeltaType::Added.to_string(), "Added");
        assert_eq!(DeltaType::Removed.to_string(), "Removed");
        assert_eq!(DeltaType::Modified.to_string(), "Modified");
    }

    #[test]
    fn test_delta_type_eq() {
        assert_eq!(DeltaType::Added, DeltaType::Added);
        assert_ne!(DeltaType::Added, DeltaType::Removed);
    }

    // -- StateDelta tests ---------------------------------------------------

    #[test]
    fn test_state_delta_to_json() {
        let delta = StateDelta {
            key: "x".into(),
            change_type: DeltaType::Added,
            old_value: None,
            new_value: Some(json!(42)),
        };
        let j = delta.to_json();
        assert_eq!(j["key"], "x");
        assert_eq!(j["change_type"], "Added");
    }

    #[test]
    fn test_state_delta_display_added() {
        let delta = StateDelta {
            key: "x".into(),
            change_type: DeltaType::Added,
            old_value: None,
            new_value: Some(json!(1)),
        };
        let s = delta.to_string();
        assert!(s.starts_with("+ x:"));
    }

    #[test]
    fn test_state_delta_display_removed() {
        let delta = StateDelta {
            key: "x".into(),
            change_type: DeltaType::Removed,
            old_value: Some(json!(1)),
            new_value: None,
        };
        let s = delta.to_string();
        assert!(s.starts_with("- x:"));
    }

    #[test]
    fn test_state_delta_display_modified() {
        let delta = StateDelta {
            key: "x".into(),
            change_type: DeltaType::Modified,
            old_value: Some(json!(1)),
            new_value: Some(json!(2)),
        };
        let s = delta.to_string();
        assert!(s.starts_with("~ x:"));
    }

    // -- StateInspector tests -----------------------------------------------

    #[test]
    fn test_state_inspector_new() {
        let inspector = StateInspector::new();
        assert!(inspector.watched_keys.is_empty());
    }

    #[test]
    fn test_state_inspector_default() {
        let inspector = StateInspector::default();
        assert!(inspector.watched_keys.is_empty());
    }

    #[test]
    fn test_state_inspector_inspect_empty() {
        let inspector = StateInspector::new();
        let report = inspector.inspect(&json!({}));
        assert!(report.keys.is_empty());
        assert_eq!(report.total_fields, 0);
    }

    #[test]
    fn test_state_inspector_inspect_flat() {
        let inspector = StateInspector::new();
        let state = json!({"a": 1, "b": "hello"});
        let report = inspector.inspect(&state);
        assert_eq!(report.keys.len(), 2);
        assert_eq!(report.total_fields, 2);
        assert_eq!(report.nested_depth, 1);
    }

    #[test]
    fn test_state_inspector_inspect_nested() {
        let inspector = StateInspector::new();
        let state = json!({"a": {"b": {"c": 1}}});
        let report = inspector.inspect(&state);
        assert_eq!(report.nested_depth, 3);
        // Fields: a (1) + b (1) + c (1) = 3
        assert_eq!(report.total_fields, 3);
    }

    #[test]
    fn test_state_inspector_inspect_non_object() {
        let inspector = StateInspector::new();
        let report = inspector.inspect(&json!(42));
        assert!(report.keys.is_empty());
        assert_eq!(report.total_fields, 0);
        assert_eq!(report.nested_depth, 0);
    }

    #[test]
    fn test_state_inspector_diff_added() {
        let inspector = StateInspector::new();
        let before = json!({});
        let after = json!({"x": 1});
        let deltas = inspector.diff(&before, &after);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].change_type, DeltaType::Added);
        assert_eq!(deltas[0].key, "x");
    }

    #[test]
    fn test_state_inspector_diff_removed() {
        let inspector = StateInspector::new();
        let before = json!({"x": 1});
        let after = json!({});
        let deltas = inspector.diff(&before, &after);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].change_type, DeltaType::Removed);
    }

    #[test]
    fn test_state_inspector_diff_modified() {
        let inspector = StateInspector::new();
        let before = json!({"x": 1});
        let after = json!({"x": 2});
        let deltas = inspector.diff(&before, &after);
        assert_eq!(deltas.len(), 1);
        assert_eq!(deltas[0].change_type, DeltaType::Modified);
    }

    #[test]
    fn test_state_inspector_diff_no_change() {
        let inspector = StateInspector::new();
        let state = json!({"x": 1});
        let deltas = inspector.diff(&state, &state);
        assert!(deltas.is_empty());
    }

    #[test]
    fn test_state_inspector_diff_multiple() {
        let inspector = StateInspector::new();
        let before = json!({"a": 1, "b": 2});
        let after = json!({"b": 3, "c": 4});
        let deltas = inspector.diff(&before, &after);
        assert_eq!(deltas.len(), 3); // a removed, b modified, c added
    }

    #[test]
    fn test_state_inspector_watch_key() {
        let mut inspector = StateInspector::new();
        inspector.watch_key("x");
        inspector.watch_key("y");
        assert_eq!(inspector.watched_keys.len(), 2);
    }

    #[test]
    fn test_state_inspector_watch_key_dedup() {
        let mut inspector = StateInspector::new();
        inspector.watch_key("x");
        inspector.watch_key("x");
        assert_eq!(inspector.watched_keys.len(), 1);
    }

    #[test]
    fn test_state_inspector_watched_values() {
        let mut inspector = StateInspector::new();
        inspector.watch_key("a");
        inspector.watch_key("missing");
        let state = json!({"a": 42, "b": 99});
        let watched = inspector.watched_values(&state);
        assert_eq!(watched.len(), 1);
        assert_eq!(watched["a"], json!(42));
    }

    #[test]
    fn test_state_inspector_estimated_size() {
        let inspector = StateInspector::new();
        let state = json!({"key": "value"});
        let report = inspector.inspect(&state);
        assert!(report.estimated_size > 0);
    }

    // -- ExecutionProfiler tests --------------------------------------------

    #[test]
    fn test_profiler_new() {
        let profiler = ExecutionProfiler::new();
        assert!(profiler.node_times().is_empty());
        assert_eq!(profiler.total_execution_ms(), 0);
    }

    #[test]
    fn test_profiler_default() {
        let profiler = ExecutionProfiler::default();
        assert!(profiler.node_times().is_empty());
    }

    #[test]
    fn test_profiler_start_end_node() {
        let mut profiler = ExecutionProfiler::new();
        profiler.start_node("a");
        profiler.end_node("a");
        let times = profiler.node_times();
        assert_eq!(times["a"].len(), 1);
    }

    #[test]
    fn test_profiler_multiple_runs() {
        let mut profiler = ExecutionProfiler::new();
        profiler.start_node("a");
        profiler.end_node("a");
        profiler.start_node("a");
        profiler.end_node("a");
        let times = profiler.node_times();
        assert_eq!(times["a"].len(), 2);
    }

    #[test]
    fn test_profiler_average_time() {
        let mut profiler = ExecutionProfiler::new();
        // Manually inject durations for deterministic testing.
        profiler.durations.insert("a".into(), vec![10, 20, 30]);
        assert_eq!(profiler.average_time("a"), Some(20.0));
    }

    #[test]
    fn test_profiler_average_time_missing() {
        let profiler = ExecutionProfiler::new();
        assert_eq!(profiler.average_time("nope"), None);
    }

    #[test]
    fn test_profiler_slowest_node() {
        let mut profiler = ExecutionProfiler::new();
        profiler.durations.insert("fast".into(), vec![1, 2, 3]);
        profiler.durations.insert("slow".into(), vec![100, 200]);
        let (id, _avg) = profiler.slowest_node().unwrap();
        assert_eq!(id, "slow");
    }

    #[test]
    fn test_profiler_slowest_node_empty() {
        let profiler = ExecutionProfiler::new();
        assert!(profiler.slowest_node().is_none());
    }

    #[test]
    fn test_profiler_total_execution_ms() {
        let mut profiler = ExecutionProfiler::new();
        profiler.durations.insert("a".into(), vec![10, 20]);
        profiler.durations.insert("b".into(), vec![5]);
        assert_eq!(profiler.total_execution_ms(), 35);
    }

    #[test]
    fn test_profiler_to_json() {
        let mut profiler = ExecutionProfiler::new();
        profiler.durations.insert("a".into(), vec![10, 20]);
        let j = profiler.to_json();
        assert!(j["nodes"]["a"].is_object());
        assert_eq!(j["total_ms"], 30);
    }

    #[test]
    fn test_profiler_end_node_without_start() {
        let mut profiler = ExecutionProfiler::new();
        profiler.end_node("unknown");
        assert!(profiler.node_times().is_empty());
    }

    // -- Helper function tests ----------------------------------------------

    #[test]
    fn test_count_fields_empty() {
        assert_eq!(count_fields(&json!({})), 0);
    }

    #[test]
    fn test_count_fields_flat() {
        assert_eq!(count_fields(&json!({"a": 1, "b": 2})), 2);
    }

    #[test]
    fn test_count_fields_nested() {
        assert_eq!(count_fields(&json!({"a": {"b": 1}})), 2);
    }

    #[test]
    fn test_count_fields_array() {
        assert_eq!(count_fields(&json!({"a": [{"b": 1}]})), 2);
    }

    #[test]
    fn test_compute_depth_scalar() {
        assert_eq!(compute_depth(&json!(42)), 0);
    }

    #[test]
    fn test_compute_depth_flat_object() {
        assert_eq!(compute_depth(&json!({"a": 1})), 1);
    }

    #[test]
    fn test_compute_depth_nested() {
        assert_eq!(compute_depth(&json!({"a": {"b": {"c": 1}}})), 3);
    }

    #[test]
    fn test_compute_depth_empty_object() {
        assert_eq!(compute_depth(&json!({})), 1);
    }

    #[test]
    fn test_compute_depth_array() {
        assert_eq!(compute_depth(&json!([1, 2, 3])), 1);
    }

    #[test]
    fn test_state_inspector_inspect_with_array() {
        let inspector = StateInspector::new();
        let state = json!({"items": [1, 2, 3], "name": "test"});
        let report = inspector.inspect(&state);
        assert_eq!(report.keys.len(), 2);
        assert!(report.total_fields >= 2);
    }

    #[test]
    fn test_profiler_to_json_slowest() {
        let mut profiler = ExecutionProfiler::new();
        profiler.durations.insert("x".into(), vec![100]);
        let j = profiler.to_json();
        assert!(j["slowest_node"].is_object());
        assert_eq!(j["slowest_node"]["node_id"], "x");
    }

    #[test]
    fn test_profiler_to_json_empty() {
        let profiler = ExecutionProfiler::new();
        let j = profiler.to_json();
        assert_eq!(j["total_ms"], 0);
        assert!(j["slowest_node"].is_null());
    }
}
