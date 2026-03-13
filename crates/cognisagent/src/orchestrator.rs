//! Tool orchestrator for coordinating multiple tool executions with dependency
//! resolution, parallel execution, and result aggregation.
//!
//! The orchestrator accepts an [`ExecutionPlan`] consisting of [`ToolCall`]s with
//! declared dependencies. It resolves the calls into batches that respect the
//! dependency ordering, executes each batch (calls within a batch are independent
//! and may run in parallel), and collects the results into an [`OrchestratorResult`].
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use cognisagent::orchestrator::*;
//! use serde_json::json;
//!
//! let mut plan = ExecutionPlan::new();
//! let call_a = ToolCall::new("read_file", json!({"path": "a.txt"}));
//! let a_id = call_a.id.clone();
//! plan.add_call(call_a);
//! plan.add_call(ToolCall::new("read_file", json!({"path": "b.txt"})).with_dependency(&a_id));
//!
//! let executor = MockToolExecutor::new();
//! let orchestrator = Orchestrator::new(Box::new(executor));
//! let result = orchestrator.execute_plan(&plan).unwrap();
//! assert!(result.all_succeeded());
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::{Duration, Instant};

use serde_json::{json, Value};
use uuid::Uuid;

use crate::agent::DeepAgentError;

/// Result type alias for orchestrator operations.
pub type Result<T> = std::result::Result<T, DeepAgentError>;

// ---------------------------------------------------------------------------
// ToolCall
// ---------------------------------------------------------------------------

/// A single tool invocation request with optional dependencies on other calls.
#[derive(Debug, Clone)]
pub struct ToolCall {
    /// Unique identifier for this call.
    pub id: String,
    /// Name of the tool to invoke.
    pub tool_name: String,
    /// Arguments to pass to the tool.
    pub arguments: Value,
    /// IDs of calls that must complete before this one executes.
    pub dependencies: Vec<String>,
}

impl ToolCall {
    /// Create a new tool call with an auto-generated ID.
    pub fn new(tool_name: impl Into<String>, arguments: Value) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            tool_name: tool_name.into(),
            arguments,
            dependencies: Vec::new(),
        }
    }

    /// Add a dependency on another call (builder pattern).
    pub fn with_dependency(mut self, call_id: &str) -> Self {
        self.dependencies.push(call_id.to_string());
        self
    }

    /// Returns `true` if this call depends on at least one other call.
    pub fn has_dependencies(&self) -> bool {
        !self.dependencies.is_empty()
    }

    /// Serialize this call to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "dependencies": self.dependencies,
        })
    }
}

// ---------------------------------------------------------------------------
// CallStatus
// ---------------------------------------------------------------------------

/// Outcome status of a single tool call execution.
#[derive(Debug, Clone, PartialEq)]
pub enum CallStatus {
    /// The call completed successfully.
    Success,
    /// The call failed with the given error message.
    Error(String),
    /// The call was skipped (e.g. a dependency failed).
    Skipped(String),
    /// The call exceeded its timeout.
    Timeout,
}

impl fmt::Display for CallStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallStatus::Success => write!(f, "success"),
            CallStatus::Error(msg) => write!(f, "error: {}", msg),
            CallStatus::Skipped(reason) => write!(f, "skipped: {}", reason),
            CallStatus::Timeout => write!(f, "timeout"),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolCallResult
// ---------------------------------------------------------------------------

/// The result of executing a single tool call.
#[derive(Debug, Clone)]
pub struct ToolCallResult {
    /// ID of the call that produced this result.
    pub call_id: String,
    /// Name of the tool that was called.
    pub tool_name: String,
    /// Output value from the tool.
    pub output: Value,
    /// Wall-clock duration of the execution.
    pub duration: Duration,
    /// Outcome status.
    pub status: CallStatus,
}

impl ToolCallResult {
    /// Returns `true` if the call succeeded.
    pub fn is_success(&self) -> bool {
        self.status == CallStatus::Success
    }

    /// Serialize this result to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "output": self.output,
            "duration_ms": self.duration.as_millis() as u64,
            "status": self.status.to_string(),
        })
    }
}

// ---------------------------------------------------------------------------
// ExecutionPlan
// ---------------------------------------------------------------------------

/// An ordered collection of tool calls with dependency metadata.
///
/// Use [`resolve`](ExecutionPlan::resolve) to compute execution batches.
#[derive(Debug, Clone, Default)]
pub struct ExecutionPlan {
    calls: Vec<ToolCall>,
}

impl ExecutionPlan {
    /// Create an empty execution plan.
    pub fn new() -> Self {
        Self { calls: Vec::new() }
    }

    /// Add a tool call to the plan.
    pub fn add_call(&mut self, call: ToolCall) {
        self.calls.push(call);
    }

    /// Number of calls in the plan.
    pub fn call_count(&self) -> usize {
        self.calls.len()
    }

    /// Validate the plan: check for missing dependencies and cycles.
    pub fn validate(&self) -> Result<()> {
        let ids: HashSet<&str> = self.calls.iter().map(|c| c.id.as_str()).collect();

        // Check for missing dependencies.
        for call in &self.calls {
            for dep in &call.dependencies {
                if !ids.contains(dep.as_str()) {
                    return Err(DeepAgentError::Other(format!(
                        "call '{}' depends on unknown call '{}'",
                        call.id, dep
                    )));
                }
            }
        }

        // Check for cycles using Kahn's algorithm.
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
        for call in &self.calls {
            in_degree.entry(call.id.as_str()).or_insert(0);
            adjacency.entry(call.id.as_str()).or_default();
            for dep in &call.dependencies {
                adjacency
                    .entry(dep.as_str())
                    .or_default()
                    .push(call.id.as_str());
                *in_degree.entry(call.id.as_str()).or_insert(0) += 1;
            }
        }

        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        let mut visited = 0usize;

        while let Some(node) = queue.pop_front() {
            visited += 1;
            if let Some(neighbors) = adjacency.get(node) {
                for &neighbor in neighbors {
                    let deg = in_degree.get_mut(neighbor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        if visited != self.calls.len() {
            return Err(DeepAgentError::Other(
                "execution plan contains a dependency cycle".to_string(),
            ));
        }

        Ok(())
    }

    /// Resolve the plan into batches ordered by dependencies.
    ///
    /// Calls within the same batch are independent and may execute in parallel.
    /// Each subsequent batch depends on all prior batches completing.
    pub fn resolve(&self) -> Result<Vec<Vec<&ToolCall>>> {
        self.validate()?;

        if self.calls.is_empty() {
            return Ok(Vec::new());
        }

        let call_map: HashMap<&str, &ToolCall> =
            self.calls.iter().map(|c| (c.id.as_str(), c)).collect();

        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

        for call in &self.calls {
            in_degree.entry(call.id.as_str()).or_insert(0);
            for dep in &call.dependencies {
                dependents
                    .entry(dep.as_str())
                    .or_default()
                    .push(call.id.as_str());
                *in_degree.entry(call.id.as_str()).or_insert(0) += 1;
            }
        }

        let mut batches: Vec<Vec<&ToolCall>> = Vec::new();
        let mut ready: Vec<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();
        ready.sort(); // deterministic ordering

        while !ready.is_empty() {
            let mut batch: Vec<&ToolCall> = ready.iter().map(|&id| call_map[id]).collect();
            batch.sort_by(|a, b| a.id.cmp(&b.id));

            let mut next_ready: Vec<&str> = Vec::new();
            for &id in &ready {
                if let Some(deps) = dependents.get(id) {
                    for &dep_id in deps {
                        let deg = in_degree.get_mut(dep_id).unwrap();
                        *deg -= 1;
                        if *deg == 0 {
                            next_ready.push(dep_id);
                        }
                    }
                }
            }
            next_ready.sort();

            batches.push(batch);
            ready = next_ready;
        }

        Ok(batches)
    }

    /// Number of batches after resolution. Returns 0 if resolution fails.
    pub fn batch_count(&self) -> usize {
        self.resolve().map(|b| b.len()).unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// ToolExecutor trait
// ---------------------------------------------------------------------------

/// Trait for executing a tool by name with the given arguments.
pub trait ToolExecutor: Send + Sync {
    /// Execute a tool and return its output.
    fn execute(&self, tool_name: &str, arguments: &Value) -> Result<Value>;
}

// ---------------------------------------------------------------------------
// MockToolExecutor
// ---------------------------------------------------------------------------

/// A mock tool executor for testing. Returns arguments as output by default,
/// with configurable per-tool delays and failures.
#[derive(Debug, Default)]
pub struct MockToolExecutor {
    delays: HashMap<String, Duration>,
    failures: HashMap<String, String>,
}

impl MockToolExecutor {
    /// Create a new mock executor with no delays or failures.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configure a delay for a specific tool (builder pattern).
    pub fn with_delay(mut self, tool_name: impl Into<String>, duration: Duration) -> Self {
        self.delays.insert(tool_name.into(), duration);
        self
    }

    /// Configure a failure for a specific tool (builder pattern).
    pub fn with_failure(mut self, tool_name: impl Into<String>, error: impl Into<String>) -> Self {
        self.failures.insert(tool_name.into(), error.into());
        self
    }
}

impl ToolExecutor for MockToolExecutor {
    fn execute(&self, tool_name: &str, arguments: &Value) -> Result<Value> {
        // Simulate delay.
        if let Some(delay) = self.delays.get(tool_name) {
            std::thread::sleep(*delay);
        }

        // Simulate failure.
        if let Some(error) = self.failures.get(tool_name) {
            return Err(DeepAgentError::Other(error.clone()));
        }

        // Default: return arguments as output.
        Ok(arguments.clone())
    }
}

// ---------------------------------------------------------------------------
// Orchestrator
// ---------------------------------------------------------------------------

/// Coordinates tool execution according to an [`ExecutionPlan`].
pub struct Orchestrator {
    executor: Box<dyn ToolExecutor>,
    timeout: Option<Duration>,
    max_parallel: Option<usize>,
}

impl Orchestrator {
    /// Create a new orchestrator with the given executor.
    pub fn new(executor: Box<dyn ToolExecutor>) -> Self {
        Self {
            executor,
            timeout: None,
            max_parallel: None,
        }
    }

    /// Set a per-call timeout (builder pattern).
    pub fn with_timeout(mut self, duration: Duration) -> Self {
        self.timeout = Some(duration);
        self
    }

    /// Set the maximum number of parallel calls per batch (builder pattern).
    pub fn with_max_parallel(mut self, n: usize) -> Self {
        self.max_parallel = Some(n);
        self
    }

    /// Execute a single tool call and return its result.
    pub fn execute_single(&self, call: &ToolCall) -> ToolCallResult {
        let start = Instant::now();

        // Check timeout.
        if let Some(timeout) = self.timeout {
            // For synchronous execution we can only enforce timeout via thread::sleep
            // estimation in the mock. Real async would use tokio::time::timeout.
            // Here we do a simple wall-clock check after execution.
            let result = self.executor.execute(&call.tool_name, &call.arguments);
            let elapsed = start.elapsed();
            if elapsed > timeout {
                return ToolCallResult {
                    call_id: call.id.clone(),
                    tool_name: call.tool_name.clone(),
                    output: Value::Null,
                    duration: elapsed,
                    status: CallStatus::Timeout,
                };
            }
            match result {
                Ok(output) => ToolCallResult {
                    call_id: call.id.clone(),
                    tool_name: call.tool_name.clone(),
                    output,
                    duration: elapsed,
                    status: CallStatus::Success,
                },
                Err(e) => ToolCallResult {
                    call_id: call.id.clone(),
                    tool_name: call.tool_name.clone(),
                    output: Value::Null,
                    duration: elapsed,
                    status: CallStatus::Error(e.to_string()),
                },
            }
        } else {
            let result = self.executor.execute(&call.tool_name, &call.arguments);
            let elapsed = start.elapsed();
            match result {
                Ok(output) => ToolCallResult {
                    call_id: call.id.clone(),
                    tool_name: call.tool_name.clone(),
                    output,
                    duration: elapsed,
                    status: CallStatus::Success,
                },
                Err(e) => ToolCallResult {
                    call_id: call.id.clone(),
                    tool_name: call.tool_name.clone(),
                    output: Value::Null,
                    duration: elapsed,
                    status: CallStatus::Error(e.to_string()),
                },
            }
        }
    }

    /// Execute an entire plan, resolving dependencies and running batches sequentially.
    ///
    /// Calls within a batch are executed sequentially in this synchronous implementation;
    /// an async version could run them in parallel with `tokio::spawn`.
    pub fn execute_plan(&self, plan: &ExecutionPlan) -> Result<OrchestratorResult> {
        let batches = plan.resolve()?;
        let overall_start = Instant::now();
        let mut all_results: Vec<ToolCallResult> = Vec::new();
        let mut failed_ids: HashSet<String> = HashSet::new();
        let mut batches_executed = 0usize;

        for batch in &batches {
            let chunk_size = self.max_parallel.unwrap_or(batch.len());
            for chunk in batch.chunks(chunk_size) {
                for call in chunk {
                    // Skip if any dependency failed.
                    let skip_reason = call.dependencies.iter().find_map(|dep_id| {
                        if failed_ids.contains(dep_id) {
                            Some(format!("dependency '{}' failed", dep_id))
                        } else {
                            None
                        }
                    });

                    if let Some(reason) = skip_reason {
                        failed_ids.insert(call.id.clone());
                        all_results.push(ToolCallResult {
                            call_id: call.id.clone(),
                            tool_name: call.tool_name.clone(),
                            output: Value::Null,
                            duration: Duration::ZERO,
                            status: CallStatus::Skipped(reason),
                        });
                    } else {
                        let result = self.execute_single(call);
                        if !result.is_success() {
                            failed_ids.insert(call.id.clone());
                        }
                        all_results.push(result);
                    }
                }
            }
            batches_executed += 1;
        }

        Ok(OrchestratorResult {
            results: all_results,
            total_duration: overall_start.elapsed(),
            batches_executed,
        })
    }
}

// ---------------------------------------------------------------------------
// OrchestratorResult
// ---------------------------------------------------------------------------

/// Aggregated results from executing an [`ExecutionPlan`].
#[derive(Debug, Clone)]
pub struct OrchestratorResult {
    /// Individual results for each call.
    pub results: Vec<ToolCallResult>,
    /// Total wall-clock duration of the entire plan execution.
    pub total_duration: Duration,
    /// Number of batches that were executed.
    pub batches_executed: usize,
}

impl OrchestratorResult {
    /// Count of successful calls.
    pub fn success_count(&self) -> usize {
        self.results.iter().filter(|r| r.is_success()).count()
    }

    /// Count of failed (non-success) calls.
    pub fn failure_count(&self) -> usize {
        self.results.iter().filter(|r| !r.is_success()).count()
    }

    /// Look up a result by call ID.
    pub fn get_result(&self, call_id: &str) -> Option<&ToolCallResult> {
        self.results.iter().find(|r| r.call_id == call_id)
    }

    /// Returns `true` if every call succeeded.
    pub fn all_succeeded(&self) -> bool {
        self.results.iter().all(|r| r.is_success())
    }

    /// Serialize the full result to JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "results": self.results.iter().map(|r| r.to_json()).collect::<Vec<_>>(),
            "total_duration_ms": self.total_duration.as_millis() as u64,
            "batches_executed": self.batches_executed,
            "success_count": self.success_count(),
            "failure_count": self.failure_count(),
        })
    }
}

// ---------------------------------------------------------------------------
// CallGraph
// ---------------------------------------------------------------------------

/// Visualization helper for tool call dependency graphs.
#[derive(Debug)]
pub struct CallGraph {
    /// Nodes: (id, tool_name).
    nodes: Vec<(String, String)>,
    /// Edges: (from_id, to_id) where `from` must complete before `to`.
    edges: Vec<(String, String)>,
}

impl CallGraph {
    /// Build a call graph from an execution plan.
    pub fn from_plan(plan: &ExecutionPlan) -> Self {
        let nodes: Vec<(String, String)> = plan
            .calls
            .iter()
            .map(|c| (c.id.clone(), c.tool_name.clone()))
            .collect();

        let mut edges = Vec::new();
        for call in &plan.calls {
            for dep in &call.dependencies {
                edges.push((dep.clone(), call.id.clone()));
            }
        }

        Self { nodes, edges }
    }

    /// Render the call graph as a Mermaid diagram string.
    pub fn to_mermaid(&self) -> String {
        let mut lines = vec!["graph TD".to_string()];

        // Build a lookup for short labels.
        let label_map: HashMap<&str, String> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, (id, name))| (id.as_str(), format!("N{}[{}]", i, name)))
            .collect();

        let id_to_alias: HashMap<&str, String> = self
            .nodes
            .iter()
            .enumerate()
            .map(|(i, (id, _))| (id.as_str(), format!("N{}", i)))
            .collect();

        for (id, _) in &self.nodes {
            if let Some(label) = label_map.get(id.as_str()) {
                lines.push(format!("    {}", label));
            }
        }

        for (from, to) in &self.edges {
            if let (Some(from_alias), Some(to_alias)) =
                (id_to_alias.get(from.as_str()), id_to_alias.get(to.as_str()))
            {
                lines.push(format!("    {} --> {}", from_alias, to_alias));
            }
        }

        lines.join("\n")
    }

    /// Compute the critical path — the longest dependency chain by number of nodes.
    ///
    /// Returns tool names along the critical path. If the graph is empty, returns
    /// an empty vector.
    pub fn critical_path(&self) -> Vec<&str> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let id_to_name: HashMap<&str, &str> = self
            .nodes
            .iter()
            .map(|(id, name)| (id.as_str(), name.as_str()))
            .collect();

        // Build adjacency: dep -> list of dependents
        let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut in_degree: HashMap<&str, usize> = HashMap::new();

        for (id, _) in &self.nodes {
            adjacency.entry(id.as_str()).or_default();
            in_degree.entry(id.as_str()).or_insert(0);
        }

        for (from, to) in &self.edges {
            adjacency
                .entry(from.as_str())
                .or_default()
                .push(to.as_str());
            *in_degree.entry(to.as_str()).or_insert(0) += 1;
        }

        // Topological sort + longest path via DP.
        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        let mut dist: HashMap<&str, usize> = HashMap::new();
        let mut prev: HashMap<&str, &str> = HashMap::new();

        for &id in in_degree.keys() {
            dist.insert(id, 0);
        }

        // Process in topological order.
        let mut topo_order: Vec<&str> = Vec::new();
        let mut in_deg = in_degree.clone();
        while let Some(node) = queue.pop_front() {
            topo_order.push(node);
            if let Some(neighbors) = adjacency.get(node) {
                for &neighbor in neighbors {
                    let new_dist = dist[node] + 1;
                    if new_dist > dist[neighbor] {
                        dist.insert(neighbor, new_dist);
                        prev.insert(neighbor, node);
                    }
                    let deg = in_deg.get_mut(neighbor).unwrap();
                    *deg -= 1;
                    if *deg == 0 {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        // Find the node with the largest distance.
        let end_node = dist
            .iter()
            .max_by_key(|(_, &d)| d)
            .map(|(&id, _)| id)
            .unwrap();

        // Reconstruct path.
        let mut path = vec![end_node];
        let mut current = end_node;
        while let Some(&predecessor) = prev.get(current) {
            path.push(predecessor);
            current = predecessor;
        }
        path.reverse();

        path.iter()
            .map(|&id| *id_to_name.get(id).unwrap_or(&"unknown"))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- ToolCall tests --

    #[test]
    fn test_tool_call_new() {
        let call = ToolCall::new("read_file", json!({"path": "a.txt"}));
        assert_eq!(call.tool_name, "read_file");
        assert_eq!(call.arguments, json!({"path": "a.txt"}));
        assert!(!call.id.is_empty());
        assert!(!call.has_dependencies());
    }

    #[test]
    fn test_tool_call_unique_ids() {
        let a = ToolCall::new("t", json!({}));
        let b = ToolCall::new("t", json!({}));
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn test_tool_call_with_dependency() {
        let a = ToolCall::new("t1", json!({}));
        let b = ToolCall::new("t2", json!({})).with_dependency(&a.id);
        assert!(b.has_dependencies());
        assert_eq!(b.dependencies.len(), 1);
        assert_eq!(b.dependencies[0], a.id);
    }

    #[test]
    fn test_tool_call_multiple_dependencies() {
        let a = ToolCall::new("t1", json!({}));
        let b = ToolCall::new("t2", json!({}));
        let c = ToolCall::new("t3", json!({}))
            .with_dependency(&a.id)
            .with_dependency(&b.id);
        assert_eq!(c.dependencies.len(), 2);
    }

    #[test]
    fn test_tool_call_to_json() {
        let call = ToolCall::new("my_tool", json!({"key": "val"}));
        let j = call.to_json();
        assert_eq!(j["tool_name"], "my_tool");
        assert_eq!(j["arguments"]["key"], "val");
        assert_eq!(j["id"], call.id);
    }

    // -- CallStatus tests --

    #[test]
    fn test_call_status_display_success() {
        assert_eq!(CallStatus::Success.to_string(), "success");
    }

    #[test]
    fn test_call_status_display_error() {
        assert_eq!(CallStatus::Error("boom".into()).to_string(), "error: boom");
    }

    #[test]
    fn test_call_status_display_skipped() {
        assert_eq!(
            CallStatus::Skipped("dep failed".into()).to_string(),
            "skipped: dep failed"
        );
    }

    #[test]
    fn test_call_status_display_timeout() {
        assert_eq!(CallStatus::Timeout.to_string(), "timeout");
    }

    // -- ToolCallResult tests --

    #[test]
    fn test_tool_call_result_is_success() {
        let r = ToolCallResult {
            call_id: "1".into(),
            tool_name: "t".into(),
            output: json!({}),
            duration: Duration::ZERO,
            status: CallStatus::Success,
        };
        assert!(r.is_success());
    }

    #[test]
    fn test_tool_call_result_is_not_success() {
        let r = ToolCallResult {
            call_id: "1".into(),
            tool_name: "t".into(),
            output: Value::Null,
            duration: Duration::ZERO,
            status: CallStatus::Error("e".into()),
        };
        assert!(!r.is_success());
    }

    #[test]
    fn test_tool_call_result_to_json() {
        let r = ToolCallResult {
            call_id: "abc".into(),
            tool_name: "my_tool".into(),
            output: json!(42),
            duration: Duration::from_millis(100),
            status: CallStatus::Success,
        };
        let j = r.to_json();
        assert_eq!(j["call_id"], "abc");
        assert_eq!(j["tool_name"], "my_tool");
        assert_eq!(j["output"], 42);
        assert_eq!(j["duration_ms"], 100);
        assert_eq!(j["status"], "success");
    }

    // -- ExecutionPlan tests --

    #[test]
    fn test_empty_plan() {
        let plan = ExecutionPlan::new();
        assert_eq!(plan.call_count(), 0);
        let batches = plan.resolve().unwrap();
        assert!(batches.is_empty());
        assert_eq!(plan.batch_count(), 0);
    }

    #[test]
    fn test_plan_single_call() {
        let mut plan = ExecutionPlan::new();
        plan.add_call(ToolCall::new("t", json!({})));
        assert_eq!(plan.call_count(), 1);
        let batches = plan.resolve().unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 1);
    }

    #[test]
    fn test_plan_all_independent() {
        let mut plan = ExecutionPlan::new();
        plan.add_call(ToolCall::new("a", json!({})));
        plan.add_call(ToolCall::new("b", json!({})));
        plan.add_call(ToolCall::new("c", json!({})));
        let batches = plan.resolve().unwrap();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 3);
    }

    #[test]
    fn test_plan_all_sequential() {
        let mut plan = ExecutionPlan::new();
        let a = ToolCall::new("a", json!({}));
        let a_id = a.id.clone();
        plan.add_call(a);

        let b = ToolCall::new("b", json!({})).with_dependency(&a_id);
        let b_id = b.id.clone();
        plan.add_call(b);

        let c = ToolCall::new("c", json!({})).with_dependency(&b_id);
        plan.add_call(c);

        let batches = plan.resolve().unwrap();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 1);
        assert_eq!(batches[1].len(), 1);
        assert_eq!(batches[2].len(), 1);
        assert_eq!(batches[0][0].tool_name, "a");
        assert_eq!(batches[1][0].tool_name, "b");
        assert_eq!(batches[2][0].tool_name, "c");
    }

    #[test]
    fn test_plan_diamond_dependency() {
        //   A
        //  / \
        // B   C
        //  \ /
        //   D
        let mut plan = ExecutionPlan::new();
        let a = ToolCall::new("a", json!({}));
        let a_id = a.id.clone();
        plan.add_call(a);

        let b = ToolCall::new("b", json!({})).with_dependency(&a_id);
        let b_id = b.id.clone();
        plan.add_call(b);

        let c = ToolCall::new("c", json!({})).with_dependency(&a_id);
        let c_id = c.id.clone();
        plan.add_call(c);

        let d = ToolCall::new("d", json!({}))
            .with_dependency(&b_id)
            .with_dependency(&c_id);
        plan.add_call(d);

        let batches = plan.resolve().unwrap();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].len(), 1); // A
        assert_eq!(batches[1].len(), 2); // B, C
        assert_eq!(batches[2].len(), 1); // D
    }

    #[test]
    fn test_plan_validate_missing_dependency() {
        let mut plan = ExecutionPlan::new();
        plan.add_call(ToolCall::new("t", json!({})).with_dependency("nonexistent"));
        let err = plan.validate().unwrap_err();
        assert!(err.to_string().contains("unknown call"));
    }

    #[test]
    fn test_plan_validate_cycle() {
        let mut plan = ExecutionPlan::new();
        let mut a = ToolCall::new("a", json!({}));
        let mut b = ToolCall::new("b", json!({}));
        let a_id = a.id.clone();
        let b_id = b.id.clone();
        a.dependencies.push(b_id.clone());
        b.dependencies.push(a_id.clone());
        plan.add_call(a);
        plan.add_call(b);

        let err = plan.validate().unwrap_err();
        assert!(err.to_string().contains("cycle"));
    }

    #[test]
    fn test_plan_validate_self_cycle() {
        let mut plan = ExecutionPlan::new();
        let mut a = ToolCall::new("a", json!({}));
        a.dependencies.push(a.id.clone());
        plan.add_call(a);

        let err = plan.validate().unwrap_err();
        assert!(err.to_string().contains("cycle"));
    }

    #[test]
    fn test_plan_batch_count() {
        let mut plan = ExecutionPlan::new();
        let a = ToolCall::new("a", json!({}));
        let a_id = a.id.clone();
        plan.add_call(a);
        plan.add_call(ToolCall::new("b", json!({})).with_dependency(&a_id));
        assert_eq!(plan.batch_count(), 2);
    }

    // -- MockToolExecutor tests --

    #[test]
    fn test_mock_executor_default() {
        let exec = MockToolExecutor::new();
        let args = json!({"x": 1});
        let result = exec.execute("any_tool", &args).unwrap();
        assert_eq!(result, args);
    }

    #[test]
    fn test_mock_executor_failure() {
        let exec = MockToolExecutor::new().with_failure("bad_tool", "something broke");
        let result = exec.execute("bad_tool", &json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("something broke"));
    }

    #[test]
    fn test_mock_executor_delay() {
        let exec = MockToolExecutor::new().with_delay("slow_tool", Duration::from_millis(50));
        let start = Instant::now();
        let _ = exec.execute("slow_tool", &json!({})).unwrap();
        assert!(start.elapsed() >= Duration::from_millis(40));
    }

    #[test]
    fn test_mock_executor_no_delay_for_other_tool() {
        let exec = MockToolExecutor::new().with_delay("slow", Duration::from_secs(10));
        let start = Instant::now();
        let _ = exec.execute("fast", &json!({})).unwrap();
        assert!(start.elapsed() < Duration::from_secs(1));
    }

    // -- Orchestrator tests --

    #[test]
    fn test_orchestrator_execute_single_success() {
        let exec = MockToolExecutor::new();
        let orch = Orchestrator::new(Box::new(exec));
        let call = ToolCall::new("t", json!({"a": 1}));
        let result = orch.execute_single(&call);
        assert!(result.is_success());
        assert_eq!(result.output, json!({"a": 1}));
        assert_eq!(result.call_id, call.id);
    }

    #[test]
    fn test_orchestrator_execute_single_failure() {
        let exec = MockToolExecutor::new().with_failure("t", "fail");
        let orch = Orchestrator::new(Box::new(exec));
        let call = ToolCall::new("t", json!({}));
        let result = orch.execute_single(&call);
        assert!(!result.is_success());
        assert!(matches!(result.status, CallStatus::Error(_)));
    }

    #[test]
    fn test_orchestrator_execute_plan_empty() {
        let exec = MockToolExecutor::new();
        let orch = Orchestrator::new(Box::new(exec));
        let plan = ExecutionPlan::new();
        let result = orch.execute_plan(&plan).unwrap();
        assert!(result.all_succeeded());
        assert_eq!(result.results.len(), 0);
        assert_eq!(result.batches_executed, 0);
    }

    #[test]
    fn test_orchestrator_execute_plan_independent() {
        let exec = MockToolExecutor::new();
        let orch = Orchestrator::new(Box::new(exec));
        let mut plan = ExecutionPlan::new();
        plan.add_call(ToolCall::new("a", json!({"n": 1})));
        plan.add_call(ToolCall::new("b", json!({"n": 2})));

        let result = orch.execute_plan(&plan).unwrap();
        assert!(result.all_succeeded());
        assert_eq!(result.success_count(), 2);
        assert_eq!(result.failure_count(), 0);
        assert_eq!(result.batches_executed, 1);
    }

    #[test]
    fn test_orchestrator_execute_plan_sequential() {
        let exec = MockToolExecutor::new();
        let orch = Orchestrator::new(Box::new(exec));
        let mut plan = ExecutionPlan::new();
        let a = ToolCall::new("a", json!({}));
        let a_id = a.id.clone();
        plan.add_call(a);
        plan.add_call(ToolCall::new("b", json!({})).with_dependency(&a_id));

        let result = orch.execute_plan(&plan).unwrap();
        assert!(result.all_succeeded());
        assert_eq!(result.batches_executed, 2);
    }

    #[test]
    fn test_orchestrator_skips_on_failed_dependency() {
        let exec = MockToolExecutor::new().with_failure("a", "boom");
        let orch = Orchestrator::new(Box::new(exec));
        let mut plan = ExecutionPlan::new();
        let a = ToolCall::new("a", json!({}));
        let a_id = a.id.clone();
        plan.add_call(a);
        let b = ToolCall::new("b", json!({})).with_dependency(&a_id);
        let b_id = b.id.clone();
        plan.add_call(b);

        let result = orch.execute_plan(&plan).unwrap();
        assert!(!result.all_succeeded());
        assert_eq!(result.failure_count(), 2);
        let b_result = result.get_result(&b_id).unwrap();
        assert!(matches!(b_result.status, CallStatus::Skipped(_)));
    }

    #[test]
    fn test_orchestrator_with_max_parallel() {
        let exec = MockToolExecutor::new();
        let orch = Orchestrator::new(Box::new(exec)).with_max_parallel(1);
        let mut plan = ExecutionPlan::new();
        plan.add_call(ToolCall::new("a", json!({})));
        plan.add_call(ToolCall::new("b", json!({})));
        plan.add_call(ToolCall::new("c", json!({})));

        let result = orch.execute_plan(&plan).unwrap();
        assert!(result.all_succeeded());
        assert_eq!(result.success_count(), 3);
    }

    #[test]
    fn test_orchestrator_timeout() {
        let exec = MockToolExecutor::new().with_delay("slow", Duration::from_millis(200));
        let orch = Orchestrator::new(Box::new(exec)).with_timeout(Duration::from_millis(50));
        let call = ToolCall::new("slow", json!({}));
        let result = orch.execute_single(&call);
        assert_eq!(result.status, CallStatus::Timeout);
    }

    // -- OrchestratorResult tests --

    #[test]
    fn test_orchestrator_result_get_result() {
        let exec = MockToolExecutor::new();
        let orch = Orchestrator::new(Box::new(exec));
        let mut plan = ExecutionPlan::new();
        let call = ToolCall::new("t", json!({}));
        let call_id = call.id.clone();
        plan.add_call(call);

        let result = orch.execute_plan(&plan).unwrap();
        assert!(result.get_result(&call_id).is_some());
        assert!(result.get_result("nonexistent").is_none());
    }

    #[test]
    fn test_orchestrator_result_to_json() {
        let exec = MockToolExecutor::new();
        let orch = Orchestrator::new(Box::new(exec));
        let mut plan = ExecutionPlan::new();
        plan.add_call(ToolCall::new("t", json!({})));
        let result = orch.execute_plan(&plan).unwrap();

        let j = result.to_json();
        assert_eq!(j["success_count"], 1);
        assert_eq!(j["failure_count"], 0);
        assert_eq!(j["batches_executed"], 1);
        assert!(j["results"].is_array());
    }

    // -- CallGraph tests --

    #[test]
    fn test_call_graph_from_empty_plan() {
        let plan = ExecutionPlan::new();
        let graph = CallGraph::from_plan(&plan);
        assert!(graph.nodes.is_empty());
        assert!(graph.edges.is_empty());
    }

    #[test]
    fn test_call_graph_mermaid_no_edges() {
        let mut plan = ExecutionPlan::new();
        plan.add_call(ToolCall::new("a", json!({})));
        plan.add_call(ToolCall::new("b", json!({})));
        let graph = CallGraph::from_plan(&plan);
        let mermaid = graph.to_mermaid();
        assert!(mermaid.starts_with("graph TD"));
        assert!(mermaid.contains("[a]"));
        assert!(mermaid.contains("[b]"));
        assert!(!mermaid.contains("-->"));
    }

    #[test]
    fn test_call_graph_mermaid_with_edges() {
        let mut plan = ExecutionPlan::new();
        let a = ToolCall::new("a", json!({}));
        let a_id = a.id.clone();
        plan.add_call(a);
        plan.add_call(ToolCall::new("b", json!({})).with_dependency(&a_id));
        let graph = CallGraph::from_plan(&plan);
        let mermaid = graph.to_mermaid();
        assert!(mermaid.contains("-->"));
    }

    #[test]
    fn test_call_graph_critical_path_empty() {
        let plan = ExecutionPlan::new();
        let graph = CallGraph::from_plan(&plan);
        assert!(graph.critical_path().is_empty());
    }

    #[test]
    fn test_call_graph_critical_path_single() {
        let mut plan = ExecutionPlan::new();
        plan.add_call(ToolCall::new("only", json!({})));
        let graph = CallGraph::from_plan(&plan);
        let cp = graph.critical_path();
        assert_eq!(cp, vec!["only"]);
    }

    #[test]
    fn test_call_graph_critical_path_chain() {
        let mut plan = ExecutionPlan::new();
        let a = ToolCall::new("a", json!({}));
        let a_id = a.id.clone();
        plan.add_call(a);

        let b = ToolCall::new("b", json!({})).with_dependency(&a_id);
        let b_id = b.id.clone();
        plan.add_call(b);

        let c = ToolCall::new("c", json!({})).with_dependency(&b_id);
        plan.add_call(c);

        let graph = CallGraph::from_plan(&plan);
        let cp = graph.critical_path();
        assert_eq!(cp, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_call_graph_critical_path_diamond() {
        //   A
        //  / \
        // B   C
        //  \ /
        //   D
        let mut plan = ExecutionPlan::new();
        let a = ToolCall::new("a", json!({}));
        let a_id = a.id.clone();
        plan.add_call(a);

        let b = ToolCall::new("b", json!({})).with_dependency(&a_id);
        let b_id = b.id.clone();
        plan.add_call(b);

        let c = ToolCall::new("c", json!({})).with_dependency(&a_id);
        let c_id = c.id.clone();
        plan.add_call(c);

        let d = ToolCall::new("d", json!({}))
            .with_dependency(&b_id)
            .with_dependency(&c_id);
        plan.add_call(d);

        let graph = CallGraph::from_plan(&plan);
        let cp = graph.critical_path();
        // Critical path length should be 3 (A -> B/C -> D)
        assert_eq!(cp.len(), 3);
        assert_eq!(cp[0], "a");
        assert_eq!(cp[2], "d");
    }

    #[test]
    fn test_call_graph_critical_path_selects_longest() {
        // A -> B -> C -> D (length 4)
        // E (length 1)
        let mut plan = ExecutionPlan::new();
        let a = ToolCall::new("a", json!({}));
        let a_id = a.id.clone();
        plan.add_call(a);

        let b = ToolCall::new("b", json!({})).with_dependency(&a_id);
        let b_id = b.id.clone();
        plan.add_call(b);

        let c = ToolCall::new("c", json!({})).with_dependency(&b_id);
        let c_id = c.id.clone();
        plan.add_call(c);

        let d = ToolCall::new("d", json!({})).with_dependency(&c_id);
        plan.add_call(d);

        plan.add_call(ToolCall::new("e", json!({})));

        let graph = CallGraph::from_plan(&plan);
        let cp = graph.critical_path();
        assert_eq!(cp.len(), 4);
        assert_eq!(cp, vec!["a", "b", "c", "d"]);
    }
}
