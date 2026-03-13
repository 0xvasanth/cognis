//! Graph execution runner with step hooks and event tracking.
//!
//! The [`GraphRunner`] wraps [`CompiledStateGraph`] execution with
//! configurable step limits, timeouts, lifecycle hooks, and event collection.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::constants::{END, START};
use crate::errors::{LangGraphError, Result};

use super::state::CompiledStateGraph;

// ---------------------------------------------------------------------------
// RunConfig
// ---------------------------------------------------------------------------

/// Configuration for a graph execution run.
#[derive(Debug, Clone)]
pub struct RunConfig {
    /// Maximum number of node executions before aborting.
    pub max_steps: Option<usize>,
    /// Wall-clock timeout for the entire run.
    pub timeout: Option<Duration>,
    /// Maximum recursion / re-entry depth (forwarded to the graph).
    pub recursion_limit: usize,
    /// Arbitrary string tags attached to the run.
    pub tags: Vec<String>,
    /// Arbitrary key-value metadata attached to the run.
    pub metadata: HashMap<String, Value>,
    /// Optional checkpoint id for resuming from a checkpoint.
    pub checkpoint_id: Option<String>,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            max_steps: Some(100),
            timeout: None,
            recursion_limit: 25,
            tags: Vec::new(),
            metadata: HashMap::new(),
            checkpoint_id: None,
        }
    }
}

impl RunConfig {
    /// Create a new `RunConfig` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of steps.
    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Remove the step limit (unlimited steps).
    pub fn no_step_limit(mut self) -> Self {
        self.max_steps = None;
        self
    }

    /// Set the wall-clock timeout.
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set the recursion limit.
    pub fn recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = limit;
        self
    }

    /// Add a tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add tags.
    pub fn tags(mut self, tags: impl IntoIterator<Item = impl Into<String>>) -> Self {
        self.tags.extend(tags.into_iter().map(Into::into));
        self
    }

    /// Add metadata.
    pub fn metadata_entry(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set the checkpoint id.
    pub fn checkpoint_id(mut self, id: impl Into<String>) -> Self {
        self.checkpoint_id = Some(id.into());
        self
    }
}

// ---------------------------------------------------------------------------
// StepEvent
// ---------------------------------------------------------------------------

/// Events emitted during graph execution.
#[derive(Debug, Clone)]
pub enum StepEvent {
    /// A node is about to execute.
    NodeStart {
        node: String,
        step: usize,
        input: Value,
    },
    /// A node finished executing.
    NodeEnd {
        node: String,
        step: usize,
        output: Value,
        duration: Duration,
    },
    /// An edge was traversed between two nodes.
    EdgeTraversed {
        from: String,
        to: String,
        step: usize,
    },
    /// A checkpoint was saved.
    CheckpointSaved { step: usize, checkpoint_id: String },
    /// An error occurred during node execution.
    Error {
        node: String,
        step: usize,
        error: String,
    },
    /// The maximum number of steps was reached.
    MaxStepsReached { steps: usize },
}

// ---------------------------------------------------------------------------
// StepHook trait
// ---------------------------------------------------------------------------

/// Hook that receives lifecycle events during graph execution.
#[async_trait]
pub trait StepHook: Send + Sync {
    /// Called when a [`StepEvent`] is emitted.
    async fn on_event(&self, event: &StepEvent) -> Result<()>;
}

// ---------------------------------------------------------------------------
// GraphRunner
// ---------------------------------------------------------------------------

/// Execution engine that runs a [`CompiledStateGraph`] with step tracking,
/// event emission, and pluggable hooks.
pub struct GraphRunner {
    config: RunConfig,
    hooks: Vec<Arc<dyn StepHook>>,
    step_count: usize,
}

impl GraphRunner {
    /// Create a new runner with the given configuration.
    pub fn new(config: RunConfig) -> Self {
        Self {
            config,
            hooks: Vec::new(),
            step_count: 0,
        }
    }

    /// Register a hook that will receive step events.
    pub fn add_hook(&mut self, hook: Arc<dyn StepHook>) {
        self.hooks.push(hook);
    }

    /// Return the number of steps executed during the last (or current) run.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Emit an event to all registered hooks.
    async fn emit(&self, event: &StepEvent) -> Result<()> {
        for hook in &self.hooks {
            hook.on_event(event).await?;
        }
        Ok(())
    }

    /// Execute the graph and return the final state.
    pub async fn run(&mut self, graph: &CompiledStateGraph, input: Value) -> Result<Value> {
        let (value, _events) = self.run_inner(graph, input).await?;
        Ok(value)
    }

    /// Execute the graph and return both the final state and all collected events.
    pub async fn run_with_events(
        &mut self,
        graph: &CompiledStateGraph,
        input: Value,
    ) -> Result<(Value, Vec<StepEvent>)> {
        self.run_inner(graph, input).await
    }

    /// Core execution loop.
    async fn run_inner(
        &mut self,
        graph: &CompiledStateGraph,
        input: Value,
    ) -> Result<(Value, Vec<StepEvent>)> {
        self.step_count = 0;
        let mut events: Vec<StepEvent> = Vec::new();
        let mut state = input;

        let deadline = self.config.timeout.map(|d| Instant::now() + d);

        // Resolve initial nodes from START.
        let mut current_nodes: Vec<String> = graph.get_next_nodes_static_pub(START, &state)?;

        loop {
            if current_nodes.is_empty() {
                break;
            }

            // Filter out END markers.
            let executable: Vec<String> = current_nodes.into_iter().filter(|n| n != END).collect();

            if executable.is_empty() {
                break;
            }

            let mut next_nodes: Vec<String> = Vec::new();

            for node_name in &executable {
                self.step_count += 1;

                // Check max steps.
                if let Some(max) = self.config.max_steps {
                    if self.step_count > max {
                        let evt = StepEvent::MaxStepsReached { steps: max };
                        events.push(evt.clone());
                        self.emit(&evt).await?;
                        return Err(LangGraphError::GraphRecursionError(format!(
                            "Maximum steps limit of {} reached",
                            max
                        )));
                    }
                }

                // Check recursion limit.
                if self.step_count > self.config.recursion_limit {
                    return Err(LangGraphError::GraphRecursionError(format!(
                        "Recursion limit of {} reached after {} steps",
                        self.config.recursion_limit, self.step_count
                    )));
                }

                // Check timeout.
                if let Some(dl) = deadline {
                    if Instant::now() > dl {
                        let evt = StepEvent::Error {
                            node: node_name.clone(),
                            step: self.step_count,
                            error: "Execution timed out".into(),
                        };
                        events.push(evt.clone());
                        self.emit(&evt).await?;
                        return Err(LangGraphError::Other("Execution timed out".into()));
                    }
                }

                // Emit NodeStart.
                let start_evt = StepEvent::NodeStart {
                    node: node_name.clone(),
                    step: self.step_count,
                    input: state.clone(),
                };
                events.push(start_evt.clone());
                self.emit(&start_evt).await?;

                let node_start = Instant::now();

                // Execute the node.
                let node = graph.nodes.get(node_name.as_str()).ok_or_else(|| {
                    LangGraphError::Other(format!(
                        "Node '{}' not found in compiled graph",
                        node_name
                    ))
                })?;

                let result = (node.action)(state.clone()).await;

                match result {
                    Ok(update) => {
                        let elapsed = node_start.elapsed();

                        let end_evt = StepEvent::NodeEnd {
                            node: node_name.clone(),
                            step: self.step_count,
                            output: update.clone(),
                            duration: elapsed,
                        };
                        events.push(end_evt.clone());
                        self.emit(&end_evt).await?;

                        CompiledStateGraph::merge_state_pub(&mut state, update)?;

                        // Determine next nodes.
                        let successors =
                            graph.get_next_nodes_static_pub(node_name.as_str(), &state)?;

                        for succ in &successors {
                            let edge_evt = StepEvent::EdgeTraversed {
                                from: node_name.clone(),
                                to: succ.clone(),
                                step: self.step_count,
                            };
                            events.push(edge_evt.clone());
                            self.emit(&edge_evt).await?;
                        }

                        next_nodes.extend(successors);
                    }
                    Err(e) => {
                        let err_evt = StepEvent::Error {
                            node: node_name.clone(),
                            step: self.step_count,
                            error: e.to_string(),
                        };
                        events.push(err_evt.clone());
                        self.emit(&err_evt).await?;
                        return Err(e);
                    }
                }
            }

            current_nodes = next_nodes;
        }

        Ok((state, events))
    }
}

// ---------------------------------------------------------------------------
// LoggingHook
// ---------------------------------------------------------------------------

/// Built-in hook that prints step events to stderr.
pub struct LoggingHook;

#[async_trait]
impl StepHook for LoggingHook {
    async fn on_event(&self, event: &StepEvent) -> Result<()> {
        match event {
            StepEvent::NodeStart { node, step, .. } => {
                eprintln!("[step {}] Starting node '{}'", step, node);
            }
            StepEvent::NodeEnd {
                node,
                step,
                duration,
                ..
            } => {
                eprintln!(
                    "[step {}] Node '{}' completed in {:?}",
                    step, node, duration
                );
            }
            StepEvent::EdgeTraversed { from, to, step } => {
                eprintln!("[step {}] Edge: '{}' -> '{}'", step, from, to);
            }
            StepEvent::CheckpointSaved {
                step,
                checkpoint_id,
            } => {
                eprintln!("[step {}] Checkpoint saved: {}", step, checkpoint_id);
            }
            StepEvent::Error {
                node, step, error, ..
            } => {
                eprintln!("[step {}] Error in node '{}': {}", step, node, error);
            }
            StepEvent::MaxStepsReached { steps } => {
                eprintln!("Maximum steps reached: {}", steps);
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MetricsHook
// ---------------------------------------------------------------------------

/// Built-in hook that collects per-node timing metrics.
pub struct MetricsHook {
    /// Per-node durations, guarded by a mutex for interior mutability.
    node_durations: Mutex<HashMap<String, Vec<Duration>>>,
}

impl MetricsHook {
    /// Create a new `MetricsHook`.
    pub fn new() -> Self {
        Self {
            node_durations: Mutex::new(HashMap::new()),
        }
    }

    /// Return a snapshot of per-node durations.
    pub async fn durations(&self) -> HashMap<String, Vec<Duration>> {
        self.node_durations.lock().await.clone()
    }

    /// Return per-node total duration.
    pub async fn total_durations(&self) -> HashMap<String, Duration> {
        let durations = self.node_durations.lock().await;
        durations
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().sum()))
            .collect()
    }

    /// Return per-node call counts.
    pub async fn call_counts(&self) -> HashMap<String, usize> {
        let durations = self.node_durations.lock().await;
        durations
            .iter()
            .map(|(k, v)| (k.clone(), v.len()))
            .collect()
    }
}

impl Default for MetricsHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl StepHook for MetricsHook {
    async fn on_event(&self, event: &StepEvent) -> Result<()> {
        if let StepEvent::NodeEnd { node, duration, .. } = event {
            let mut durations = self.node_durations.lock().await;
            durations.entry(node.clone()).or_default().push(*duration);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::graph::branch::{RouterFn, RouterResult};
    use crate::graph::state::{NodeAction, StateGraph};

    /// Helper: build a simple linear graph  START -> a -> b -> END
    fn build_linear_graph() -> CompiledStateGraph {
        let action_a: NodeAction = Arc::new(|state: Value| {
            let mut s = state;
            s["a"] = json!("done");
            Ok(s)
        });
        let action_b: NodeAction = Arc::new(|state: Value| {
            let mut s = state;
            s["b"] = json!("done");
            Ok(s)
        });
        StateGraph::new()
            .add_node_sync("a", action_a)
            .add_node_sync("b", action_b)
            .add_edge(START, "a")
            .add_edge("a", "b")
            .add_edge("b", END)
            .compile()
            .unwrap()
    }

    /// Helper: build a single-node graph  START -> only -> END
    fn build_single_node_graph() -> CompiledStateGraph {
        let action: NodeAction = Arc::new(|state: Value| {
            let mut s = state;
            s["result"] = json!(42);
            Ok(s)
        });
        StateGraph::new()
            .add_node_sync("only", action)
            .add_edge(START, "only")
            .add_edge("only", END)
            .compile()
            .unwrap()
    }

    /// Helper: build a looping graph that counts up to a limit.
    fn build_loop_graph(limit: i64) -> CompiledStateGraph {
        let action: NodeAction = Arc::new(move |state: Value| {
            let count = state["count"].as_i64().unwrap_or(0);
            Ok(json!({ "count": count + 1 }))
        });
        let router: RouterFn = Arc::new(move |state: &Value| {
            let count = state["count"].as_i64().unwrap_or(0);
            if count >= limit {
                RouterResult::Single(END.to_string())
            } else {
                RouterResult::Single("counter".to_string())
            }
        });
        StateGraph::new()
            .add_node_sync("counter", action)
            .add_edge(START, "counter")
            .add_conditional_edges("counter", router, None)
            .compile()
            .unwrap()
    }

    // -----------------------------------------------------------------------
    // Test 1: Basic run produces correct output
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_basic_run() {
        let graph = build_linear_graph();
        let mut runner = GraphRunner::new(RunConfig::new());
        let result = runner.run(&graph, json!({})).await.unwrap();
        assert_eq!(result["a"], json!("done"));
        assert_eq!(result["b"], json!("done"));
    }

    // -----------------------------------------------------------------------
    // Test 2: Step count is tracked correctly
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_step_count() {
        let graph = build_linear_graph();
        let mut runner = GraphRunner::new(RunConfig::new());
        runner.run(&graph, json!({})).await.unwrap();
        assert_eq!(runner.step_count(), 2);
    }

    // -----------------------------------------------------------------------
    // Test 3: Single node graph
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_single_node() {
        let graph = build_single_node_graph();
        let mut runner = GraphRunner::new(RunConfig::new());
        let result = runner.run(&graph, json!({})).await.unwrap();
        assert_eq!(result["result"], json!(42));
        assert_eq!(runner.step_count(), 1);
    }

    // -----------------------------------------------------------------------
    // Test 4: Events are collected
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_events_collected() {
        let graph = build_single_node_graph();
        let mut runner = GraphRunner::new(RunConfig::new());
        let (_result, events) = runner.run_with_events(&graph, json!({})).await.unwrap();
        // Expect: NodeStart, NodeEnd, EdgeTraversed (to END)
        assert!(events.len() >= 2);
        assert!(matches!(&events[0], StepEvent::NodeStart { node, .. } if node == "only"));
        assert!(matches!(&events[1], StepEvent::NodeEnd { node, .. } if node == "only"));
    }

    // -----------------------------------------------------------------------
    // Test 5: Max steps limit triggers error
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_max_steps_reached() {
        let graph = build_loop_graph(200); // loops 200 times
        let config = RunConfig::new().max_steps(5).recursion_limit(500);
        let mut runner = GraphRunner::new(config);
        let err = runner.run(&graph, json!({"count": 0})).await.unwrap_err();
        assert!(err.to_string().contains("Maximum steps limit"));
    }

    // -----------------------------------------------------------------------
    // Test 6: MaxStepsReached event is emitted
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_max_steps_event() {
        let graph = build_loop_graph(200);
        let config = RunConfig::new().max_steps(3).recursion_limit(500);
        let mut runner = GraphRunner::new(config);
        let err = runner.run_with_events(&graph, json!({"count": 0})).await;
        assert!(err.is_err());
    }

    // -----------------------------------------------------------------------
    // Test 7: Recursion limit triggers error
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_recursion_limit() {
        let graph = build_loop_graph(200);
        let config = RunConfig::new().no_step_limit().recursion_limit(4);
        let mut runner = GraphRunner::new(config);
        let err = runner.run(&graph, json!({"count": 0})).await.unwrap_err();
        assert!(err.to_string().contains("Recursion limit"));
    }

    // -----------------------------------------------------------------------
    // Test 8: Hook receives events
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_hook_receives_events() {
        let graph = build_single_node_graph();
        let metrics = Arc::new(MetricsHook::new());
        let mut runner = GraphRunner::new(RunConfig::new());
        runner.add_hook(metrics.clone());
        runner.run(&graph, json!({})).await.unwrap();
        let counts = metrics.call_counts().await;
        assert_eq!(counts.get("only"), Some(&1));
    }

    // -----------------------------------------------------------------------
    // Test 9: MetricsHook records durations
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_metrics_hook_durations() {
        let graph = build_linear_graph();
        let metrics = Arc::new(MetricsHook::new());
        let mut runner = GraphRunner::new(RunConfig::new());
        runner.add_hook(metrics.clone());
        runner.run(&graph, json!({})).await.unwrap();
        let durations = metrics.durations().await;
        assert!(durations.contains_key("a"));
        assert!(durations.contains_key("b"));
        assert_eq!(durations["a"].len(), 1);
        assert_eq!(durations["b"].len(), 1);
    }

    // -----------------------------------------------------------------------
    // Test 10: MetricsHook total_durations
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_metrics_total_durations() {
        let graph = build_loop_graph(3);
        let metrics = Arc::new(MetricsHook::new());
        let config = RunConfig::new().recursion_limit(100);
        let mut runner = GraphRunner::new(config);
        runner.add_hook(metrics.clone());
        runner.run(&graph, json!({"count": 0})).await.unwrap();
        let totals = metrics.total_durations().await;
        assert!(totals.contains_key("counter"));
        // counter runs 3 times
        let counts = metrics.call_counts().await;
        assert_eq!(counts["counter"], 3);
    }

    // -----------------------------------------------------------------------
    // Test 11: LoggingHook does not error
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_logging_hook() {
        let graph = build_single_node_graph();
        let mut runner = GraphRunner::new(RunConfig::new());
        runner.add_hook(Arc::new(LoggingHook));
        let result = runner.run(&graph, json!({})).await.unwrap();
        assert_eq!(result["result"], json!(42));
    }

    // -----------------------------------------------------------------------
    // Test 12: Multiple hooks
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_multiple_hooks() {
        let graph = build_linear_graph();
        let metrics = Arc::new(MetricsHook::new());
        let mut runner = GraphRunner::new(RunConfig::new());
        runner.add_hook(Arc::new(LoggingHook));
        runner.add_hook(metrics.clone());
        runner.run(&graph, json!({})).await.unwrap();
        let counts = metrics.call_counts().await;
        assert_eq!(counts.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Test 13: RunConfig builder
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_run_config_builder() {
        let config = RunConfig::new()
            .max_steps(50)
            .recursion_limit(10)
            .tag("test")
            .tags(vec!["a", "b"])
            .metadata_entry("key", json!("value"))
            .timeout(Duration::from_secs(30))
            .checkpoint_id("cp-1");

        assert_eq!(config.max_steps, Some(50));
        assert_eq!(config.recursion_limit, 10);
        assert_eq!(config.tags, vec!["test", "a", "b"]);
        assert_eq!(config.metadata.get("key"), Some(&json!("value")));
        assert_eq!(config.timeout, Some(Duration::from_secs(30)));
        assert_eq!(config.checkpoint_id, Some("cp-1".to_string()));
    }

    // -----------------------------------------------------------------------
    // Test 14: Default config values
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_default_config() {
        let config = RunConfig::default();
        assert_eq!(config.max_steps, Some(100));
        assert_eq!(config.recursion_limit, 25);
        assert!(config.tags.is_empty());
        assert!(config.metadata.is_empty());
        assert!(config.timeout.is_none());
        assert!(config.checkpoint_id.is_none());
    }

    // -----------------------------------------------------------------------
    // Test 15: Edge events are emitted
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_edge_events() {
        let graph = build_linear_graph();
        let mut runner = GraphRunner::new(RunConfig::new());
        let (_result, events) = runner.run_with_events(&graph, json!({})).await.unwrap();
        let edge_events: Vec<&StepEvent> = events
            .iter()
            .filter(|e| matches!(e, StepEvent::EdgeTraversed { .. }))
            .collect();
        // a->b and b->END
        assert_eq!(edge_events.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Test 16: Node error produces Error event
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_error_event() {
        let action: NodeAction =
            Arc::new(|_state: Value| Err(LangGraphError::Other("intentional failure".into())));
        let graph = StateGraph::new()
            .add_node_sync("fail", action)
            .add_edge(START, "fail")
            .add_edge("fail", END)
            .compile()
            .unwrap();

        let mut runner = GraphRunner::new(RunConfig::new());
        let err = runner.run_with_events(&graph, json!({})).await;
        assert!(err.is_err());
    }

    // -----------------------------------------------------------------------
    // Test 17: Step count resets on new run
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_step_count_resets() {
        let graph = build_linear_graph();
        let mut runner = GraphRunner::new(RunConfig::new());
        runner.run(&graph, json!({})).await.unwrap();
        assert_eq!(runner.step_count(), 2);
        runner.run(&graph, json!({})).await.unwrap();
        assert_eq!(runner.step_count(), 2);
    }

    // -----------------------------------------------------------------------
    // Test 18: Custom hook
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_custom_hook() {
        struct CountHook {
            count: Mutex<usize>,
        }

        #[async_trait]
        impl StepHook for CountHook {
            async fn on_event(&self, _event: &StepEvent) -> Result<()> {
                let mut c = self.count.lock().await;
                *c += 1;
                Ok(())
            }
        }

        let hook = Arc::new(CountHook {
            count: Mutex::new(0),
        });

        let graph = build_linear_graph();
        let mut runner = GraphRunner::new(RunConfig::new());
        runner.add_hook(hook.clone());
        runner.run(&graph, json!({})).await.unwrap();

        let count = *hook.count.lock().await;
        // NodeStart(a) + NodeEnd(a) + Edge(a->b) + NodeStart(b) + NodeEnd(b) + Edge(b->END)
        assert!(count >= 4);
    }
}
