//! Pregel-style execution engine with superstep processing.
//!
//! This module implements the core Pregel execution model where graph nodes are
//! executed in discrete "supersteps". In each superstep, all nodes whose
//! dependencies have been satisfied are executed, their outputs are collected as
//! messages, and the process repeats until no more nodes are ready or the
//! maximum number of supersteps is reached.
//!
//! # Key Types
//!
//! - [`NodeFunction`] — the function signature for node computations
//! - [`ExecutionNode`] — a named node with a function and dependency list
//! - [`SuperStep`] — records the result of a single superstep
//! - [`ExecutionResult`] — the final output of a complete execution run
//! - [`ExecutionConfig`] — tuning knobs (max supersteps, retries, halt behavior)
//! - [`PregelExecutor`] — the main executor that orchestrates superstep processing
//! - [`ExecutionTrace`] — debug-friendly recording of the execution history
//!
//! # Example
//!
//! ```rust
//! use cognisgraph::pregel::executor::{ExecutionConfig, ExecutionNode, PregelExecutor};
//! use serde_json::json;
//!
//! let config = ExecutionConfig::new();
//! let mut executor = PregelExecutor::new(config);
//!
//! executor.add_node(ExecutionNode::new("greet", |state| {
//!     Ok(json!({"greeting": "hello"}))
//! }));
//!
//! let result = executor.execute(json!({})).unwrap();
//! assert!(result.final_state.get("greeting").is_some());
//! ```

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use serde_json::Value;

use crate::errors::LangGraphError;

/// Type alias for a node's computation function.
///
/// Accepts the current state as a [`Value`] and returns either an updated
/// (partial) state or an error. The function must be `Send + Sync` so that it
/// can be shared across threads.
pub type NodeFunction = Box<dyn Fn(Value) -> Result<Value, LangGraphError> + Send + Sync>;

/// A node registered in the Pregel execution graph.
///
/// Each node has a unique name, a computation function, and a list of
/// dependency names (other nodes that must complete before this node can run).
pub struct ExecutionNode {
    /// Unique name of this node.
    pub name: String,
    /// The computation to execute when this node runs.
    pub function: NodeFunction,
    /// Names of nodes that must complete before this node is eligible.
    pub dependencies: Vec<String>,
}

impl ExecutionNode {
    /// Create a new execution node with the given name and function.
    ///
    /// The node starts with no dependencies.
    pub fn new<F>(name: impl Into<String>, function: F) -> Self
    where
        F: Fn(Value) -> Result<Value, LangGraphError> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            function: Box::new(function),
            dependencies: Vec::new(),
        }
    }

    /// Add a dependency on another node.
    ///
    /// This node will not execute until the named dependency has completed.
    pub fn with_dependency(mut self, dep: impl Into<String>) -> Self {
        self.dependencies.push(dep.into());
        self
    }
}

impl std::fmt::Debug for ExecutionNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExecutionNode")
            .field("name", &self.name)
            .field("dependencies", &self.dependencies)
            .finish_non_exhaustive()
    }
}

/// Records the outcome of a single Pregel superstep.
///
/// Each superstep executes all nodes whose dependencies have been satisfied and
/// collects their output messages (partial state updates).
#[derive(Debug, Clone)]
pub struct SuperStep {
    /// The zero-based index of this superstep within the execution.
    pub step_number: usize,
    /// Names of nodes that were executed in this superstep.
    pub nodes_executed: Vec<String>,
    /// Messages (partial state updates) produced by each node, keyed by node name.
    pub messages: HashMap<String, Value>,
    /// Wall-clock duration of this superstep in milliseconds.
    pub duration_ms: u64,
}

impl SuperStep {
    /// Serialize this superstep to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "step_number": self.step_number,
            "nodes_executed": self.nodes_executed,
            "messages": self.messages,
            "duration_ms": self.duration_ms,
        })
    }
}

/// The result of a complete Pregel execution run.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// The merged final state after all supersteps have completed.
    pub final_state: Value,
    /// The sequence of supersteps that were executed.
    pub supersteps: Vec<SuperStep>,
    /// Total number of supersteps executed.
    pub total_steps: usize,
    /// Total wall-clock duration of the entire execution in milliseconds.
    pub total_duration_ms: u64,
}

impl ExecutionResult {
    /// Serialize this result to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "final_state": self.final_state,
            "supersteps": self.supersteps.iter().map(|s| s.to_json()).collect::<Vec<_>>(),
            "total_steps": self.total_steps,
            "total_duration_ms": self.total_duration_ms,
        })
    }
}

/// Configuration for the Pregel execution engine.
///
/// Controls limits, retry behavior, and error handling strategy.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Maximum number of supersteps before the executor halts.
    /// Default: 100.
    pub max_supersteps: usize,
    /// Maximum number of retries for a failing node within a single superstep.
    /// Default: 3.
    pub max_node_retries: u32,
    /// Whether to stop execution immediately when a node returns an error.
    /// If `false`, execution continues and the error is recorded in the state.
    /// Default: true.
    pub halt_on_error: bool,
}

impl ExecutionConfig {
    /// Create a new configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of supersteps.
    pub fn with_max_supersteps(mut self, max: usize) -> Self {
        self.max_supersteps = max;
        self
    }

    /// Set the maximum number of retries per node.
    pub fn with_max_node_retries(mut self, retries: u32) -> Self {
        self.max_node_retries = retries;
        self
    }

    /// Set the halt-on-error flag.
    pub fn with_halt_on_error(mut self, halt: bool) -> Self {
        self.halt_on_error = halt;
        self
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_supersteps: 100,
            max_node_retries: 3,
            halt_on_error: true,
        }
    }
}

/// The main Pregel-style execution engine.
///
/// Orchestrates graph node execution through a sequence of supersteps. In each
/// superstep every node whose dependencies have already completed is executed.
/// The process repeats until either no more nodes are eligible, all nodes have
/// completed, or the maximum superstep limit is reached.
pub struct PregelExecutor {
    /// Registered execution nodes, keyed by name.
    nodes: HashMap<String, ExecutionNode>,
    /// Execution configuration.
    config: ExecutionConfig,
}

impl PregelExecutor {
    /// Create a new executor with the given configuration.
    pub fn new(config: ExecutionConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            config,
        }
    }

    /// Register an execution node.
    ///
    /// If a node with the same name already exists it will be replaced.
    pub fn add_node(&mut self, node: ExecutionNode) {
        self.nodes.insert(node.name.clone(), node);
    }

    /// Return the number of registered nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return `true` if no nodes are registered.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Execute a single superstep.
    ///
    /// Runs all nodes whose dependencies are present in `completed`, merges
    /// their output into `state`, adds the newly completed node names to
    /// `completed`, and returns a [`SuperStep`] describing what happened.
    pub fn execute_step(
        &self,
        state: &mut Value,
        completed: &mut HashSet<String>,
        step_number: usize,
    ) -> Result<SuperStep, LangGraphError> {
        let start = Instant::now();
        let mut nodes_executed = Vec::new();
        let mut messages: HashMap<String, Value> = HashMap::new();

        // Collect ready nodes (those whose dependencies are all in `completed`).
        let ready: Vec<&String> = self
            .nodes
            .keys()
            .filter(|name| {
                !completed.contains(*name)
                    && self.nodes[*name]
                        .dependencies
                        .iter()
                        .all(|dep| completed.contains(dep))
            })
            .collect();

        for name in ready {
            let node = &self.nodes[name];
            let mut last_err: Option<LangGraphError> = None;
            let mut succeeded = false;

            for _attempt in 0..=self.config.max_node_retries {
                match (node.function)(state.clone()) {
                    Ok(output) => {
                        // Merge the partial output into the state.
                        if let Value::Object(map) = &output {
                            if let Value::Object(ref mut state_map) = state {
                                for (k, v) in map {
                                    state_map.insert(k.clone(), v.clone());
                                }
                            }
                        }
                        messages.insert(name.clone(), output);
                        nodes_executed.push(name.clone());
                        completed.insert(name.clone());
                        succeeded = true;
                        last_err = None;
                        break;
                    }
                    Err(e) => {
                        last_err = Some(e);
                    }
                }
            }

            if !succeeded {
                if self.config.halt_on_error {
                    return Err(last_err.unwrap_or_else(|| {
                        LangGraphError::Other(format!("Node '{}' failed", name))
                    }));
                }
                // Record the error in state and mark node as completed so
                // dependents can see it.
                let err_msg = last_err
                    .as_ref()
                    .map(|e| e.to_string())
                    .unwrap_or_else(|| "unknown error".to_string());
                if let Value::Object(ref mut state_map) = state {
                    state_map.insert(format!("{}_error", name), Value::String(err_msg.clone()));
                }
                messages.insert(name.clone(), serde_json::json!({"error": err_msg}));
                nodes_executed.push(name.clone());
                completed.insert(name.clone());
            }
        }

        let duration_ms = start.elapsed().as_millis() as u64;

        Ok(SuperStep {
            step_number,
            nodes_executed,
            messages,
            duration_ms,
        })
    }

    /// Run the full Pregel execution loop starting from `initial_state`.
    ///
    /// Repeatedly executes supersteps until:
    /// - All nodes have completed.
    /// - No more nodes are ready (e.g., unsatisfied or circular dependencies).
    /// - The maximum superstep limit is reached.
    ///
    /// Returns an [`ExecutionResult`] containing the final state and execution
    /// history.
    pub fn execute(&self, initial_state: Value) -> Result<ExecutionResult, LangGraphError> {
        let overall_start = Instant::now();
        let mut state = initial_state;
        let mut completed: HashSet<String> = HashSet::new();
        let mut supersteps: Vec<SuperStep> = Vec::new();

        for step_number in 0..self.config.max_supersteps {
            let superstep = self.execute_step(&mut state, &mut completed, step_number)?;

            let made_progress = !superstep.nodes_executed.is_empty();
            supersteps.push(superstep);

            if !made_progress {
                // No nodes could run — either all are done or remaining nodes
                // have unsatisfiable dependencies (e.g., circular).
                break;
            }

            if completed.len() == self.nodes.len() {
                // All nodes have completed.
                break;
            }
        }

        let total_duration_ms = overall_start.elapsed().as_millis() as u64;
        let total_steps = supersteps.len();

        Ok(ExecutionResult {
            final_state: state,
            supersteps,
            total_steps,
            total_duration_ms,
        })
    }
}

impl std::fmt::Debug for PregelExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PregelExecutor")
            .field("node_count", &self.nodes.len())
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("config", &self.config)
            .finish()
    }
}

/// Records a trace of the execution for debugging and introspection.
///
/// Each superstep is recorded as it completes, and the full trace can be
/// serialized to JSON.
#[derive(Debug, Clone, Default)]
pub struct ExecutionTrace {
    /// Recorded supersteps.
    steps: Vec<SuperStep>,
}

impl ExecutionTrace {
    /// Create a new empty trace.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Record a superstep in the trace.
    pub fn add_step(&mut self, step: SuperStep) {
        self.steps.push(step);
    }

    /// Return the number of recorded steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Serialize the full trace to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "steps": self.steps.iter().map(|s| s.to_json()).collect::<Vec<_>>(),
            "total_steps": self.steps.len(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── Single node execution ─────────────────────────────────────────

    #[test]
    fn test_single_node_execution() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());
        executor.add_node(ExecutionNode::new("a", |_state| Ok(json!({"result": 42}))));

        let result = executor.execute(json!({})).unwrap();
        assert_eq!(result.final_state["result"], 42);
        assert_eq!(result.total_steps, 1);
    }

    #[test]
    fn test_single_node_reads_state() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());
        executor.add_node(ExecutionNode::new("doubler", |state| {
            let x = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(json!({"doubled": x * 2}))
        }));

        let result = executor.execute(json!({"x": 5})).unwrap();
        assert_eq!(result.final_state["doubled"], 10);
    }

    // ── Linear dependency chains ──────────────────────────────────────

    #[test]
    fn test_linear_chain_a_b_c() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());

        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({"a_done": true}))));
        executor.add_node(
            ExecutionNode::new("b", |state| {
                assert!(state.get("a_done").is_some());
                Ok(json!({"b_done": true}))
            })
            .with_dependency("a"),
        );
        executor.add_node(
            ExecutionNode::new("c", |state| {
                assert!(state.get("b_done").is_some());
                Ok(json!({"c_done": true}))
            })
            .with_dependency("b"),
        );

        let result = executor.execute(json!({})).unwrap();
        assert_eq!(result.final_state["a_done"], true);
        assert_eq!(result.final_state["b_done"], true);
        assert_eq!(result.final_state["c_done"], true);
        // 3 supersteps (one node per step) + 1 final empty step to detect completion
        // Actually: a runs in step 0, b in step 1, c in step 2, then step 3 is empty -> breaks
        // But we break when completed.len() == nodes.len() after step 2.
        assert!(result.total_steps >= 3);
    }

    #[test]
    fn test_linear_chain_order() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());

        executor.add_node(ExecutionNode::new("first", |_| Ok(json!({"order": [1]}))));
        executor.add_node(
            ExecutionNode::new("second", |state| {
                let mut order = state
                    .get("order")
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default();
                order.push(json!(2));
                Ok(json!({"order": order}))
            })
            .with_dependency("first"),
        );
        executor.add_node(
            ExecutionNode::new("third", |state| {
                let mut order = state
                    .get("order")
                    .and_then(|v| v.as_array().cloned())
                    .unwrap_or_default();
                order.push(json!(3));
                Ok(json!({"order": order}))
            })
            .with_dependency("second"),
        );

        let result = executor.execute(json!({})).unwrap();
        let order = result.final_state["order"].as_array().unwrap();
        assert_eq!(order, &[json!(1), json!(2), json!(3)]);
    }

    // ── Parallel execution ────────────────────────────────────────────

    #[test]
    fn test_parallel_independent_nodes() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());

        executor.add_node(ExecutionNode::new("x", |_| Ok(json!({"x": 1}))));
        executor.add_node(ExecutionNode::new("y", |_| Ok(json!({"y": 2}))));
        executor.add_node(ExecutionNode::new("z", |_| Ok(json!({"z": 3}))));

        let result = executor.execute(json!({})).unwrap();
        assert_eq!(result.final_state["x"], 1);
        assert_eq!(result.final_state["y"], 2);
        assert_eq!(result.final_state["z"], 3);
        // All three should execute in the first superstep.
        assert_eq!(result.supersteps[0].nodes_executed.len(), 3);
    }

    #[test]
    fn test_parallel_nodes_same_superstep() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());

        executor.add_node(ExecutionNode::new("p1", |_| Ok(json!({"p1": true}))));
        executor.add_node(ExecutionNode::new("p2", |_| Ok(json!({"p2": true}))));

        let result = executor.execute(json!({})).unwrap();
        let step0 = &result.supersteps[0];
        assert!(step0.nodes_executed.contains(&"p1".to_string()));
        assert!(step0.nodes_executed.contains(&"p2".to_string()));
    }

    // ── Diamond dependencies ──────────────────────────────────────────

    #[test]
    fn test_diamond_a_bc_d() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());

        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({"a": 1}))));
        executor.add_node(ExecutionNode::new("b", |_| Ok(json!({"b": 2}))).with_dependency("a"));
        executor.add_node(ExecutionNode::new("c", |_| Ok(json!({"c": 3}))).with_dependency("a"));
        executor.add_node(
            ExecutionNode::new("d", |state| {
                let b = state.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
                let c = state.get("c").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({"d": b + c}))
            })
            .with_dependency("b")
            .with_dependency("c"),
        );

        let result = executor.execute(json!({})).unwrap();
        assert_eq!(result.final_state["a"], 1);
        assert_eq!(result.final_state["b"], 2);
        assert_eq!(result.final_state["c"], 3);
        assert_eq!(result.final_state["d"], 5);
    }

    #[test]
    fn test_diamond_superstep_structure() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());

        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({"a": 1}))));
        executor.add_node(ExecutionNode::new("b", |_| Ok(json!({"b": 2}))).with_dependency("a"));
        executor.add_node(ExecutionNode::new("c", |_| Ok(json!({"c": 3}))).with_dependency("a"));
        executor.add_node(
            ExecutionNode::new("d", |_| Ok(json!({"d": 4})))
                .with_dependency("b")
                .with_dependency("c"),
        );

        let result = executor.execute(json!({})).unwrap();

        // Step 0: a
        assert_eq!(result.supersteps[0].nodes_executed, vec!["a".to_string()]);
        // Step 1: b and c (parallel)
        let step1_names: HashSet<String> = result.supersteps[1]
            .nodes_executed
            .iter()
            .cloned()
            .collect();
        assert!(step1_names.contains("b"));
        assert!(step1_names.contains("c"));
        assert_eq!(step1_names.len(), 2);
        // Step 2: d
        assert_eq!(result.supersteps[2].nodes_executed, vec!["d".to_string()]);
    }

    // ── Max supersteps limit ──────────────────────────────────────────

    #[test]
    fn test_max_supersteps_limit() {
        let config = ExecutionConfig::new().with_max_supersteps(2);
        let mut executor = PregelExecutor::new(config);

        // Chain of 5 nodes — needs 5 supersteps but limit is 2.
        executor.add_node(ExecutionNode::new("n1", |_| Ok(json!({"n1": true}))));
        executor
            .add_node(ExecutionNode::new("n2", |_| Ok(json!({"n2": true}))).with_dependency("n1"));
        executor
            .add_node(ExecutionNode::new("n3", |_| Ok(json!({"n3": true}))).with_dependency("n2"));
        executor
            .add_node(ExecutionNode::new("n4", |_| Ok(json!({"n4": true}))).with_dependency("n3"));
        executor
            .add_node(ExecutionNode::new("n5", |_| Ok(json!({"n5": true}))).with_dependency("n4"));

        let result = executor.execute(json!({})).unwrap();
        assert_eq!(result.total_steps, 2);
        // Only n1 and n2 should have completed.
        assert_eq!(result.final_state.get("n1"), Some(&json!(true)));
        assert_eq!(result.final_state.get("n2"), Some(&json!(true)));
        assert!(result.final_state.get("n3").is_none());
    }

    #[test]
    fn test_max_supersteps_zero() {
        let config = ExecutionConfig::new().with_max_supersteps(0);
        let mut executor = PregelExecutor::new(config);
        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({"a": 1}))));

        let result = executor.execute(json!({})).unwrap();
        assert_eq!(result.total_steps, 0);
        assert!(result.final_state.get("a").is_none());
    }

    // ── Halt on error behavior ────────────────────────────────────────

    #[test]
    fn test_halt_on_error_true() {
        let config = ExecutionConfig::new()
            .with_halt_on_error(true)
            .with_max_node_retries(0);
        let mut executor = PregelExecutor::new(config);

        executor.add_node(ExecutionNode::new("fail", |_| {
            Err(LangGraphError::Other("boom".to_string()))
        }));

        let result = executor.execute(json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_halt_on_error_false_continues() {
        let config = ExecutionConfig::new()
            .with_halt_on_error(false)
            .with_max_node_retries(0);
        let mut executor = PregelExecutor::new(config);

        executor.add_node(ExecutionNode::new("fail", |_| {
            Err(LangGraphError::Other("boom".to_string()))
        }));
        executor.add_node(ExecutionNode::new("ok", |_| Ok(json!({"ok": true}))));

        let result = executor.execute(json!({})).unwrap();
        // The error node records an error in state.
        assert!(result.final_state.get("fail_error").is_some());
        assert_eq!(result.final_state["ok"], true);
    }

    #[test]
    fn test_halt_on_error_false_dependent_sees_error() {
        let config = ExecutionConfig::new()
            .with_halt_on_error(false)
            .with_max_node_retries(0);
        let mut executor = PregelExecutor::new(config);

        executor.add_node(ExecutionNode::new("fail", |_| {
            Err(LangGraphError::Other("boom".to_string()))
        }));
        executor.add_node(
            ExecutionNode::new("after_fail", |state| {
                let has_err = state.get("fail_error").is_some();
                Ok(json!({"saw_error": has_err}))
            })
            .with_dependency("fail"),
        );

        let result = executor.execute(json!({})).unwrap();
        assert_eq!(result.final_state["saw_error"], true);
    }

    // ── Node retry on failure ─────────────────────────────────────────

    #[test]
    fn test_node_retry_succeeds_on_second_attempt() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let config = ExecutionConfig::new().with_max_node_retries(3);
        let mut executor = PregelExecutor::new(config);

        executor.add_node(ExecutionNode::new("flaky", move |_| {
            let n = attempts_clone.fetch_add(1, Ordering::SeqCst);
            if n < 1 {
                Err(LangGraphError::Other("transient".to_string()))
            } else {
                Ok(json!({"flaky": "ok"}))
            }
        }));

        let result = executor.execute(json!({})).unwrap();
        assert_eq!(result.final_state["flaky"], "ok");
        assert!(attempts.load(Ordering::SeqCst) >= 2);
    }

    #[test]
    fn test_node_retry_exhausted() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let config = ExecutionConfig::new()
            .with_max_node_retries(2)
            .with_halt_on_error(true);
        let mut executor = PregelExecutor::new(config);

        executor.add_node(ExecutionNode::new("always_fail", move |_| {
            attempts_clone.fetch_add(1, Ordering::SeqCst);
            Err(LangGraphError::Other("persistent".to_string()))
        }));

        let result = executor.execute(json!({}));
        assert!(result.is_err());
        // 1 initial attempt + 2 retries = 3 total
        assert_eq!(attempts.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn test_retry_zero_means_single_attempt() {
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        let attempts = Arc::new(AtomicU32::new(0));
        let attempts_clone = attempts.clone();

        let config = ExecutionConfig::new()
            .with_max_node_retries(0)
            .with_halt_on_error(true);
        let mut executor = PregelExecutor::new(config);

        executor.add_node(ExecutionNode::new("once", move |_| {
            attempts_clone.fetch_add(1, Ordering::SeqCst);
            Err(LangGraphError::Other("fail".to_string()))
        }));

        let _ = executor.execute(json!({}));
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    // ── ExecutionConfig defaults and customization ────────────────────

    #[test]
    fn test_execution_config_defaults() {
        let config = ExecutionConfig::new();
        assert_eq!(config.max_supersteps, 100);
        assert_eq!(config.max_node_retries, 3);
        assert!(config.halt_on_error);
    }

    #[test]
    fn test_execution_config_default_trait() {
        let config = ExecutionConfig::default();
        assert_eq!(config.max_supersteps, 100);
        assert_eq!(config.max_node_retries, 3);
        assert!(config.halt_on_error);
    }

    #[test]
    fn test_execution_config_builder() {
        let config = ExecutionConfig::new()
            .with_max_supersteps(50)
            .with_max_node_retries(5)
            .with_halt_on_error(false);

        assert_eq!(config.max_supersteps, 50);
        assert_eq!(config.max_node_retries, 5);
        assert!(!config.halt_on_error);
    }

    // ── ExecutionResult and SuperStep serialization ───────────────────

    #[test]
    fn test_superstep_to_json() {
        let step = SuperStep {
            step_number: 0,
            nodes_executed: vec!["a".to_string(), "b".to_string()],
            messages: {
                let mut m = HashMap::new();
                m.insert("a".to_string(), json!({"x": 1}));
                m.insert("b".to_string(), json!({"y": 2}));
                m
            },
            duration_ms: 42,
        };

        let j = step.to_json();
        assert_eq!(j["step_number"], 0);
        assert_eq!(j["duration_ms"], 42);
        let executed = j["nodes_executed"].as_array().unwrap();
        assert_eq!(executed.len(), 2);
    }

    #[test]
    fn test_execution_result_to_json() {
        let result = ExecutionResult {
            final_state: json!({"done": true}),
            supersteps: vec![SuperStep {
                step_number: 0,
                nodes_executed: vec!["a".to_string()],
                messages: HashMap::new(),
                duration_ms: 10,
            }],
            total_steps: 1,
            total_duration_ms: 10,
        };

        let j = result.to_json();
        assert_eq!(j["total_steps"], 1);
        assert_eq!(j["total_duration_ms"], 10);
        assert_eq!(j["final_state"]["done"], true);
        let steps = j["supersteps"].as_array().unwrap();
        assert_eq!(steps.len(), 1);
    }

    #[test]
    fn test_execution_result_from_run() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());
        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({"v": 1}))));

        let result = executor.execute(json!({})).unwrap();
        let j = result.to_json();
        assert!(j.get("final_state").is_some());
        assert!(j.get("supersteps").is_some());
        assert!(j.get("total_steps").is_some());
        assert!(j.get("total_duration_ms").is_some());
    }

    // ── ExecutionTrace recording ──────────────────────────────────────

    #[test]
    fn test_execution_trace_new() {
        let trace = ExecutionTrace::new();
        assert_eq!(trace.step_count(), 0);
    }

    #[test]
    fn test_execution_trace_add_step() {
        let mut trace = ExecutionTrace::new();
        trace.add_step(SuperStep {
            step_number: 0,
            nodes_executed: vec!["a".to_string()],
            messages: HashMap::new(),
            duration_ms: 5,
        });
        trace.add_step(SuperStep {
            step_number: 1,
            nodes_executed: vec!["b".to_string()],
            messages: HashMap::new(),
            duration_ms: 3,
        });

        assert_eq!(trace.step_count(), 2);
    }

    #[test]
    fn test_execution_trace_to_json() {
        let mut trace = ExecutionTrace::new();
        trace.add_step(SuperStep {
            step_number: 0,
            nodes_executed: vec!["x".to_string()],
            messages: HashMap::new(),
            duration_ms: 7,
        });

        let j = trace.to_json();
        assert_eq!(j["total_steps"], 1);
        let steps = j["steps"].as_array().unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0]["step_number"], 0);
    }

    #[test]
    fn test_execution_trace_from_execution() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());
        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({"a": 1}))));
        executor.add_node(ExecutionNode::new("b", |_| Ok(json!({"b": 2}))).with_dependency("a"));

        let result = executor.execute(json!({})).unwrap();

        let mut trace = ExecutionTrace::new();
        for step in &result.supersteps {
            trace.add_step(step.clone());
        }

        assert!(trace.step_count() >= 2);
    }

    // ── Empty executor ────────────────────────────────────────────────

    #[test]
    fn test_empty_executor() {
        let executor = PregelExecutor::new(ExecutionConfig::new());
        assert!(executor.is_empty());
        assert_eq!(executor.node_count(), 0);
    }

    #[test]
    fn test_empty_executor_execute() {
        let executor = PregelExecutor::new(ExecutionConfig::new());
        let result = executor.execute(json!({"x": 1})).unwrap();
        // State passes through unchanged.
        assert_eq!(result.final_state["x"], 1);
        // One superstep that finds nothing to do.
        assert_eq!(result.total_steps, 1);
        assert!(result.supersteps[0].nodes_executed.is_empty());
    }

    #[test]
    fn test_executor_is_not_empty_after_add() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());
        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({}))));
        assert!(!executor.is_empty());
        assert_eq!(executor.node_count(), 1);
    }

    // ── Circular dependency handling ──────────────────────────────────

    #[test]
    fn test_circular_dependency_does_not_hang() {
        let config = ExecutionConfig::new().with_max_supersteps(10);
        let mut executor = PregelExecutor::new(config);

        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({"a": 1}))).with_dependency("b"));
        executor.add_node(ExecutionNode::new("b", |_| Ok(json!({"b": 2}))).with_dependency("a"));

        let result = executor.execute(json!({})).unwrap();
        // Neither node should have executed.
        assert!(result.final_state.get("a").is_none());
        assert!(result.final_state.get("b").is_none());
        // Should terminate quickly (first superstep finds nothing to do).
        assert_eq!(result.total_steps, 1);
    }

    #[test]
    fn test_three_node_cycle() {
        let config = ExecutionConfig::new().with_max_supersteps(5);
        let mut executor = PregelExecutor::new(config);

        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({"a": 1}))).with_dependency("c"));
        executor.add_node(ExecutionNode::new("b", |_| Ok(json!({"b": 2}))).with_dependency("a"));
        executor.add_node(ExecutionNode::new("c", |_| Ok(json!({"c": 3}))).with_dependency("b"));

        let result = executor.execute(json!({})).unwrap();
        assert!(result.final_state.get("a").is_none());
        assert!(result.final_state.get("b").is_none());
        assert!(result.final_state.get("c").is_none());
        assert_eq!(result.total_steps, 1);
    }

    #[test]
    fn test_partial_cycle_with_valid_nodes() {
        let config = ExecutionConfig::new().with_max_supersteps(10);
        let mut executor = PregelExecutor::new(config);

        // "ok" has no deps and should execute.
        executor.add_node(ExecutionNode::new("ok", |_| Ok(json!({"ok": true}))));
        // "a" and "b" form a cycle.
        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({"a": 1}))).with_dependency("b"));
        executor.add_node(ExecutionNode::new("b", |_| Ok(json!({"b": 2}))).with_dependency("a"));

        let result = executor.execute(json!({})).unwrap();
        assert_eq!(result.final_state["ok"], true);
        assert!(result.final_state.get("a").is_none());
        assert!(result.final_state.get("b").is_none());
    }

    // ── ExecutionNode builder ─────────────────────────────────────────

    #[test]
    fn test_execution_node_builder() {
        let node = ExecutionNode::new("test", |_| Ok(json!({})))
            .with_dependency("dep1")
            .with_dependency("dep2");

        assert_eq!(node.name, "test");
        assert_eq!(node.dependencies, vec!["dep1", "dep2"]);
    }

    #[test]
    fn test_execution_node_no_deps() {
        let node = ExecutionNode::new("solo", |_| Ok(json!({})));
        assert!(node.dependencies.is_empty());
    }

    // ── Additional edge cases ─────────────────────────────────────────

    #[test]
    fn test_node_overwrites_state_key() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());
        executor.add_node(ExecutionNode::new("writer", |_| {
            Ok(json!({"key": "new_value"}))
        }));

        let result = executor.execute(json!({"key": "old_value"})).unwrap();
        assert_eq!(result.final_state["key"], "new_value");
    }

    #[test]
    fn test_multiple_nodes_write_different_keys() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());
        executor.add_node(ExecutionNode::new("w1", |_| Ok(json!({"k1": 1}))));
        executor.add_node(ExecutionNode::new("w2", |_| Ok(json!({"k2": 2}))));

        let result = executor.execute(json!({})).unwrap();
        assert_eq!(result.final_state["k1"], 1);
        assert_eq!(result.final_state["k2"], 2);
    }

    #[test]
    fn test_node_returns_non_object() {
        // A node returning a non-object value should not crash — it just
        // doesn't merge anything into the state.
        let mut executor = PregelExecutor::new(ExecutionConfig::new());
        executor.add_node(ExecutionNode::new("scalar", |_| Ok(json!(42))));

        let result = executor.execute(json!({"existing": true})).unwrap();
        assert_eq!(result.final_state["existing"], true);
    }

    #[test]
    fn test_superstep_messages_populated() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());
        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({"v": 1}))));

        let result = executor.execute(json!({})).unwrap();
        let step = &result.supersteps[0];
        assert!(step.messages.contains_key("a"));
        assert_eq!(step.messages["a"]["v"], 1);
    }

    #[test]
    fn test_execution_node_debug() {
        let node = ExecutionNode::new("dbg_node", |_| Ok(json!({}))).with_dependency("x");
        let debug_str = format!("{:?}", node);
        assert!(debug_str.contains("dbg_node"));
        assert!(debug_str.contains("x"));
    }

    #[test]
    fn test_pregel_executor_debug() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());
        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({}))));
        let debug_str = format!("{:?}", executor);
        assert!(debug_str.contains("PregelExecutor"));
        assert!(debug_str.contains("node_count"));
    }

    #[test]
    fn test_deep_chain_10_nodes() {
        let config = ExecutionConfig::new().with_max_supersteps(20);
        let mut executor = PregelExecutor::new(config);

        executor.add_node(ExecutionNode::new("n0", |_| Ok(json!({"n0": true}))));
        for i in 1..10 {
            let prev = format!("n{}", i - 1);
            let name = format!("n{}", i);
            let key = name.clone();
            executor.add_node(
                ExecutionNode::new(name, move |_| Ok(json!({ key.clone(): true })))
                    .with_dependency(prev),
            );
        }

        let result = executor.execute(json!({})).unwrap();
        assert_eq!(result.final_state.get("n9").unwrap(), &json!(true));
        assert!(result.total_steps >= 10);
    }

    #[test]
    fn test_wide_fan_out() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());

        executor.add_node(ExecutionNode::new("root", |_| Ok(json!({"root": true}))));
        for i in 0..10 {
            let name = format!("leaf_{}", i);
            let key = name.clone();
            executor.add_node(
                ExecutionNode::new(name, move |_| Ok(json!({ key.clone(): true })))
                    .with_dependency("root"),
            );
        }

        let result = executor.execute(json!({})).unwrap();
        // Root in step 0, all 10 leaves in step 1.
        assert_eq!(result.supersteps[0].nodes_executed.len(), 1);
        assert_eq!(result.supersteps[1].nodes_executed.len(), 10);
    }

    #[test]
    fn test_replace_node_with_same_name() {
        let mut executor = PregelExecutor::new(ExecutionConfig::new());
        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({"a": "first"}))));
        executor.add_node(ExecutionNode::new("a", |_| Ok(json!({"a": "second"}))));

        assert_eq!(executor.node_count(), 1);
        let result = executor.execute(json!({})).unwrap();
        assert_eq!(result.final_state["a"], "second");
    }

    #[test]
    fn test_dependency_on_nonexistent_node() {
        let config = ExecutionConfig::new().with_max_supersteps(5);
        let mut executor = PregelExecutor::new(config);

        executor.add_node(
            ExecutionNode::new("blocked", |_| Ok(json!({"b": 1}))).with_dependency("nonexistent"),
        );

        let result = executor.execute(json!({})).unwrap();
        // The node never runs because its dependency never completes.
        assert!(result.final_state.get("b").is_none());
    }
}
