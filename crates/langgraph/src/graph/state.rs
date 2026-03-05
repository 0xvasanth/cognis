//! StateGraph builder and CompiledStateGraph executor.
//!
//! The [`StateGraph`] provides a builder API for constructing directed graphs of
//! async node actions connected by edges (direct or conditional). Once built,
//! [`StateGraph::compile`] produces a [`CompiledStateGraph`] that can be invoked
//! to run the graph to completion.

use std::collections::HashMap;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use futures::Stream;

use crate::constants::{END, START};
use crate::errors::LangGraphError;
use crate::types::{CachePolicy, RetryPolicy, StreamMode, StreamUpdate};

use super::branch::{Branch, RouterFn};

/// Default recursion limit for graph execution.
const DEFAULT_RECURSION_LIMIT: usize = 25;

/// A synchronous node action: takes state, returns state update.
pub type NodeAction = Arc<dyn Fn(Value) -> Result<Value, LangGraphError> + Send + Sync>;

/// An asynchronous node action: takes state, returns a future that resolves to a state update.
pub type AsyncNodeAction = Arc<
    dyn Fn(Value) -> Pin<Box<dyn std::future::Future<Output = Result<Value, LangGraphError>> + Send>>
        + Send
        + Sync,
>;

/// Specification for a node in the graph.
pub struct NodeSpec {
    /// The name of this node.
    pub name: String,
    /// The async action to execute when this node runs.
    pub action: AsyncNodeAction,
    /// Optional metadata associated with this node.
    pub metadata: Option<HashMap<String, Value>>,
    /// Optional retry policy for this node.
    pub retry_policy: Option<RetryPolicy>,
    /// Optional cache policy for this node's results.
    pub cache_policy: Option<CachePolicy>,
    /// Optional mapping of end conditions for this node.
    /// Maps end names to target node names.
    pub ends: Option<HashMap<String, String>>,
    /// Whether this node's execution should be deferred.
    pub defer: bool,
}

impl fmt::Debug for NodeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NodeSpec")
            .field("name", &self.name)
            .field("metadata", &self.metadata)
            .field("retry_policy", &self.retry_policy)
            .field("cache_policy", &self.cache_policy)
            .field("ends", &self.ends)
            .field("defer", &self.defer)
            .field("action", &"<AsyncNodeAction>")
            .finish()
    }
}

/// Edge types in the graph.
#[derive(Clone)]
enum Edge {
    /// Direct edge from one node to another.
    Direct { from: String, to: String },
    /// Conditional edge with routing function.
    Conditional { from: String, branch: Branch },
}

impl fmt::Debug for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Edge::Direct { from, to } => f
                .debug_struct("Edge::Direct")
                .field("from", from)
                .field("to", to)
                .finish(),
            Edge::Conditional { from, branch } => f
                .debug_struct("Edge::Conditional")
                .field("from", from)
                .field("branch", branch)
                .finish(),
        }
    }
}

/// Reference to an outgoing edge from a node.
#[derive(Clone)]
enum EdgeRef {
    /// Direct target node.
    Direct(String),
    /// Conditional branch to resolve at runtime.
    Conditional(Branch),
}

impl fmt::Debug for EdgeRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeRef::Direct(target) => write!(f, "EdgeRef::Direct({target})"),
            EdgeRef::Conditional(branch) => {
                write!(f, "EdgeRef::Conditional({branch:?})")
            }
        }
    }
}

/// Builder for state graphs.
///
/// # Example
///
/// ```rust
/// use std::sync::Arc;
/// use serde_json::{json, Value};
/// use langgraph::graph::state::{StateGraph, AsyncNodeAction};
/// use langgraph::errors::LangGraphError;
///
/// let action: AsyncNodeAction = Arc::new(|state: Value| {
///     Box::pin(async move {
///         Ok(json!({"result": "done"}))
///     })
/// });
///
/// let graph = StateGraph::new()
///     .add_node("my_node", action)
///     .set_entry_point("my_node")
///     .set_finish_point("my_node")
///     .compile()
///     .unwrap();
/// ```
pub struct StateGraph {
    nodes: HashMap<String, NodeSpec>,
    edges: Vec<Edge>,
    entry_point: Option<String>,
    finish_points: Vec<String>,
    recursion_limit: usize,
    input_schema: Option<Value>,
    output_schema: Option<Value>,
}

impl StateGraph {
    /// Create a new empty state graph.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_point: None,
            finish_points: Vec::new(),
            recursion_limit: DEFAULT_RECURSION_LIMIT,
            input_schema: None,
            output_schema: None,
        }
    }

    /// Set the recursion limit for graph execution (default: 25).
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = limit;
        self
    }

    /// Add a node with an async action to the graph.
    pub fn add_node(mut self, name: &str, action: AsyncNodeAction) -> Self {
        if name == START || name == END {
            panic!(
                "Cannot add a node named '{}': reserved name",
                name
            );
        }
        self.nodes.insert(
            name.to_string(),
            NodeSpec {
                name: name.to_string(),
                action,
                metadata: None,
                retry_policy: None,
                cache_policy: None,
                ends: None,
                defer: false,
            },
        );
        self
    }

    /// Add a node with a synchronous action (wrapped in async).
    pub fn add_node_sync(self, name: &str, action: NodeAction) -> Self {
        let async_action: AsyncNodeAction = Arc::new(move |state: Value| {
            let action = action.clone();
            Box::pin(async move { (action)(state) })
        });
        self.add_node(name, async_action)
    }

    /// Add a node with metadata and optional retry policy.
    pub fn add_node_with_config(
        mut self,
        name: &str,
        action: AsyncNodeAction,
        metadata: Option<HashMap<String, Value>>,
        retry_policy: Option<RetryPolicy>,
    ) -> Self {
        if name == START || name == END {
            panic!(
                "Cannot add a node named '{}': reserved name",
                name
            );
        }
        self.nodes.insert(
            name.to_string(),
            NodeSpec {
                name: name.to_string(),
                action,
                metadata,
                retry_policy,
                cache_policy: None,
                ends: None,
                defer: false,
            },
        );
        self
    }

    /// Add a node with all configuration options including cache policy, ends, and defer.
    #[allow(clippy::too_many_arguments)]
    pub fn add_node_with_full_config(
        mut self,
        name: &str,
        action: AsyncNodeAction,
        metadata: Option<HashMap<String, Value>>,
        retry_policy: Option<RetryPolicy>,
        cache_policy: Option<CachePolicy>,
        ends: Option<HashMap<String, String>>,
        defer: bool,
    ) -> Self {
        if name == START || name == END {
            panic!(
                "Cannot add a node named '{}': reserved name",
                name
            );
        }
        self.nodes.insert(
            name.to_string(),
            NodeSpec {
                name: name.to_string(),
                action,
                metadata,
                retry_policy,
                cache_policy,
                ends,
                defer,
            },
        );
        self
    }

    /// Add a direct edge between two nodes.
    pub fn add_edge(mut self, from: &str, to: &str) -> Self {
        self.edges.push(Edge::Direct {
            from: from.to_string(),
            to: to.to_string(),
        });
        self
    }

    /// Add a conditional edge from a node using a routing function.
    pub fn add_conditional_edges(
        mut self,
        from: &str,
        path: RouterFn,
        path_map: Option<HashMap<String, String>>,
    ) -> Self {
        let mut branch = Branch::new(path);
        if let Some(map) = path_map {
            branch = branch.with_path_map(map);
        }
        self.edges.push(Edge::Conditional {
            from: from.to_string(),
            branch,
        });
        self
    }

    /// Set the entry point of the graph (equivalent to `add_edge(START, node)`).
    pub fn set_entry_point(mut self, node: &str) -> Self {
        self.entry_point = Some(node.to_string());
        self.edges.push(Edge::Direct {
            from: START.to_string(),
            to: node.to_string(),
        });
        self
    }

    /// Add a sequence of nodes connected by direct edges.
    ///
    /// Each node is connected to the next by a direct edge, creating a linear chain.
    /// This is equivalent to calling `add_node` and `add_edge` for each pair.
    pub fn add_sequence(mut self, nodes: Vec<(&str, AsyncNodeAction)>) -> Self {
        let names: Vec<String> = nodes.iter().map(|(name, _)| name.to_string()).collect();
        for (name, action) in nodes {
            self = self.add_node(name, action);
        }
        for i in 0..names.len().saturating_sub(1) {
            self.edges.push(Edge::Direct {
                from: names[i].clone(),
                to: names[i + 1].clone(),
            });
        }
        self
    }

    /// Set a conditional entry point that routes to different nodes based on input.
    ///
    /// This is equivalent to `add_conditional_edges(START, path, path_map)`.
    pub fn set_conditional_entry_point(
        self,
        path: RouterFn,
        path_map: Option<HashMap<String, String>>,
    ) -> Self {
        self.add_conditional_edges(START, path, path_map)
    }

    /// Add an edge from multiple source nodes to a single target.
    ///
    /// All source nodes must complete before the target node can execute.
    /// This creates a join point in the graph.
    pub fn add_edges(mut self, from_nodes: &[&str], to: &str) -> Self {
        for from in from_nodes {
            self.edges.push(Edge::Direct {
                from: from.to_string(),
                to: to.to_string(),
            });
        }
        self
    }

    /// Set the input schema for the graph.
    pub fn with_input_schema(mut self, schema: Value) -> Self {
        self.input_schema = Some(schema);
        self
    }

    /// Set the output schema for the graph.
    pub fn with_output_schema(mut self, schema: Value) -> Self {
        self.output_schema = Some(schema);
        self
    }

    /// Set a finish point (equivalent to `add_edge(node, END)`).
    pub fn set_finish_point(mut self, node: &str) -> Self {
        self.finish_points.push(node.to_string());
        self.edges.push(Edge::Direct {
            from: node.to_string(),
            to: END.to_string(),
        });
        self
    }

    /// Compile the graph into an executable form.
    ///
    /// Validates the graph structure and builds the adjacency index.
    pub fn compile(self) -> Result<CompiledStateGraph, LangGraphError> {
        // Validate that an entry point is defined.
        if self.entry_point.is_none() {
            // Check if there's a START edge
            let has_start_edge = self.edges.iter().any(|e| match e {
                Edge::Direct { from, .. } => from == START,
                Edge::Conditional { from, .. } => from == START,
            });
            if !has_start_edge {
                return Err(LangGraphError::Other(
                    "Graph has no entry point. Use set_entry_point() or add_edge(START, ...) to define one.".to_string(),
                ));
            }
        }

        // Validate that all edge targets reference existing nodes or special nodes.
        for edge in &self.edges {
            match edge {
                Edge::Direct { from, to } => {
                    if from != START && !self.nodes.contains_key(from) {
                        return Err(LangGraphError::Other(format!(
                            "Edge references unknown source node: '{from}'"
                        )));
                    }
                    if to != END && !self.nodes.contains_key(to) {
                        return Err(LangGraphError::Other(format!(
                            "Edge references unknown target node: '{to}'"
                        )));
                    }
                }
                Edge::Conditional { from, .. } => {
                    if from != START && !self.nodes.contains_key(from) {
                        return Err(LangGraphError::Other(format!(
                            "Conditional edge references unknown source node: '{from}'"
                        )));
                    }
                }
            }
        }

        // Build adjacency index.
        let mut outgoing: HashMap<String, Vec<EdgeRef>> = HashMap::new();
        for edge in &self.edges {
            match edge {
                Edge::Direct { from, to } => {
                    outgoing
                        .entry(from.clone())
                        .or_default()
                        .push(EdgeRef::Direct(to.clone()));
                }
                Edge::Conditional { from, branch } => {
                    outgoing
                        .entry(from.clone())
                        .or_default()
                        .push(EdgeRef::Conditional(branch.clone()));
                }
            }
        }

        Ok(CompiledStateGraph {
            nodes: self.nodes,
            edges: self.edges,
            outgoing,
            recursion_limit: self.recursion_limit,
            input_schema: self.input_schema,
            output_schema: self.output_schema,
        })
    }
}

impl Default for StateGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for StateGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StateGraph")
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("edges", &self.edges)
            .field("entry_point", &self.entry_point)
            .field("finish_points", &self.finish_points)
            .field("recursion_limit", &self.recursion_limit)
            .field("input_schema", &self.input_schema)
            .field("output_schema", &self.output_schema)
            .finish()
    }
}

/// A compiled state graph ready for execution.
///
/// Created by [`StateGraph::compile`]. Use [`invoke`](CompiledStateGraph::invoke)
/// to run the graph with an initial state.
pub struct CompiledStateGraph {
    /// The nodes in the graph.
    pub nodes: HashMap<String, NodeSpec>,
    /// All edges (kept for introspection).
    #[allow(dead_code)]
    edges: Vec<Edge>,
    /// Adjacency: node name -> list of outgoing edge references.
    outgoing: HashMap<String, Vec<EdgeRef>>,
    /// Maximum number of node executions before erroring.
    recursion_limit: usize,
    /// Optional input JSON schema.
    input_schema: Option<Value>,
    /// Optional output JSON schema.
    output_schema: Option<Value>,
}

impl CompiledStateGraph {
    /// Invoke the graph with initial state, running to completion.
    ///
    /// Execution starts from the START node, follows edges through the graph,
    /// and stops when reaching the END node or when no more nodes are reachable.
    /// Returns the final merged state.
    pub async fn invoke(&self, input: Value) -> Result<Value, LangGraphError> {
        let mut state = input;
        let mut step_count: usize = 0;

        // Resolve the initial nodes from START.
        let mut current_nodes = self.get_next_nodes(START, &state)?;

        loop {
            if current_nodes.is_empty() {
                break;
            }

            // Filter out END markers and collect real nodes to execute.
            let executable: Vec<String> = current_nodes
                .iter()
                .filter(|n| n.as_str() != END)
                .cloned()
                .collect();

            // If all targets are END, we're done.
            if executable.is_empty() {
                break;
            }

            // Execute each node in order (sequential execution for now).
            let mut next_nodes: Vec<String> = Vec::new();

            for node_name in &executable {
                step_count += 1;
                if step_count > self.recursion_limit {
                    return Err(LangGraphError::GraphRecursionError(format!(
                        "Recursion limit of {} reached after {} steps",
                        self.recursion_limit, step_count
                    )));
                }

                let update = self.execute_node(node_name, &state).await?;
                Self::merge_state(&mut state, update)?;

                // Determine the next nodes from this node.
                let mut successors = self.get_next_nodes(node_name, &state)?;
                next_nodes.append(&mut successors);
            }

            current_nodes = next_nodes;
        }

        Ok(state)
    }

    /// Stream state updates as the graph executes.
    ///
    /// Yields updates after each node executes, based on the stream mode:
    /// - [`StreamMode::Values`] — yield the full state after each node executes
    /// - [`StreamMode::Updates`] — yield only the delta (what the node returned)
    /// - [`StreamMode::Debug`] — yield both full state and delta plus metadata
    ///   (node name, step number, elapsed time in milliseconds)
    ///
    /// Other stream modes fall back to [`StreamMode::Updates`] behaviour.
    pub async fn stream(
        &self,
        input: Value,
        stream_mode: StreamMode,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamUpdate, LangGraphError>> + Send>>, LangGraphError>
    {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<StreamUpdate, LangGraphError>>(64);

        // Clone all the data we need to move into the spawned task.
        // CompiledStateGraph is not Send because of the Arc<dyn Fn> fields,
        // so we capture the pieces we need.
        let nodes: HashMap<String, AsyncNodeAction> = self
            .nodes
            .iter()
            .map(|(k, v)| (k.clone(), v.action.clone()))
            .collect();
        let outgoing = self.outgoing.clone();
        let recursion_limit = self.recursion_limit;

        tokio::spawn(async move {
            let start_time = std::time::Instant::now();
            let mut state = input;
            let mut step_count: usize = 0;

            // Resolve the initial nodes from START.
            let mut current_nodes = match Self::get_next_nodes_static(&outgoing, START, &state) {
                Ok(n) => n,
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            };

            loop {
                if current_nodes.is_empty() {
                    break;
                }

                let executable: Vec<String> = current_nodes
                    .iter()
                    .filter(|n| n.as_str() != END)
                    .cloned()
                    .collect();

                if executable.is_empty() {
                    break;
                }

                let mut next_nodes: Vec<String> = Vec::new();

                for node_name in &executable {
                    step_count += 1;
                    if step_count > recursion_limit {
                        let _ = tx
                            .send(Err(LangGraphError::GraphRecursionError(format!(
                                "Recursion limit of {} reached after {} steps",
                                recursion_limit, step_count
                            ))))
                            .await;
                        return;
                    }

                    let action = match nodes.get(node_name) {
                        Some(a) => a.clone(),
                        None => {
                            let _ = tx
                                .send(Err(LangGraphError::Other(format!(
                                    "Node '{}' not found in compiled graph",
                                    node_name
                                ))))
                                .await;
                            return;
                        }
                    };

                    let update = match action(state.clone()).await {
                        Ok(u) => u,
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                    };

                    // Merge update into state.
                    match Self::merge_state(&mut state, update.clone()) {
                        Ok(()) => {}
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                    }

                    // Build the stream update based on mode.
                    let stream_update = match stream_mode {
                        StreamMode::Values => StreamUpdate {
                            node: node_name.clone(),
                            data: state.clone(),
                            mode: StreamMode::Values,
                        },
                        StreamMode::Updates => StreamUpdate {
                            node: node_name.clone(),
                            data: update,
                            mode: StreamMode::Updates,
                        },
                        StreamMode::Debug => {
                            let elapsed_ms = start_time.elapsed().as_millis() as u64;
                            StreamUpdate {
                                node: node_name.clone(),
                                data: serde_json::json!({
                                    "step": step_count,
                                    "elapsed_ms": elapsed_ms,
                                    "update": update,
                                    "state": state,
                                }),
                                mode: StreamMode::Debug,
                            }
                        }
                        // Other modes fall back to Updates behaviour.
                        _ => StreamUpdate {
                            node: node_name.clone(),
                            data: update,
                            mode: stream_mode,
                        },
                    };

                    if tx.send(Ok(stream_update)).await.is_err() {
                        // Receiver dropped; stop.
                        return;
                    }

                    match Self::get_next_nodes_static(&outgoing, node_name, &state) {
                        Ok(mut successors) => next_nodes.append(&mut successors),
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                    }
                }

                current_nodes = next_nodes;
            }
            // tx is dropped here, closing the stream.
        });

        Ok(Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx)))
    }

    /// Static version of `get_next_nodes` that does not require `&self`.
    ///
    /// Used by the spawned streaming task.
    fn get_next_nodes_static(
        outgoing: &HashMap<String, Vec<EdgeRef>>,
        current: &str,
        state: &Value,
    ) -> Result<Vec<String>, LangGraphError> {
        let Some(edges) = outgoing.get(current) else {
            return Ok(Vec::new());
        };

        let mut targets = Vec::new();
        for edge_ref in edges {
            match edge_ref {
                EdgeRef::Direct(target) => {
                    targets.push(target.clone());
                }
                EdgeRef::Conditional(branch) => {
                    let mut resolved = branch.resolve(state)?;
                    targets.append(&mut resolved);
                }
            }
        }

        Ok(targets)
    }

    /// Get the next nodes to execute given the current node and state.
    fn get_next_nodes(
        &self,
        current: &str,
        state: &Value,
    ) -> Result<Vec<String>, LangGraphError> {
        let Some(edges) = self.outgoing.get(current) else {
            return Ok(Vec::new());
        };

        let mut targets = Vec::new();
        for edge_ref in edges {
            match edge_ref {
                EdgeRef::Direct(target) => {
                    targets.push(target.clone());
                }
                EdgeRef::Conditional(branch) => {
                    let mut resolved = branch.resolve(state)?;
                    targets.append(&mut resolved);
                }
            }
        }

        Ok(targets)
    }

    /// Execute a single node by name.
    async fn execute_node(
        &self,
        name: &str,
        state: &Value,
    ) -> Result<Value, LangGraphError> {
        let node = self.nodes.get(name).ok_or_else(|| {
            LangGraphError::Other(format!("Node '{name}' not found in compiled graph"))
        })?;

        (node.action)(state.clone()).await
    }

    /// Merge a node's output into the current state.
    ///
    /// For JSON objects, keys from `update` are merged into `current` (shallow merge).
    /// For other types, the update replaces the current state.
    fn merge_state(current: &mut Value, update: Value) -> Result<(), LangGraphError> {
        match (&mut *current, update) {
            (Value::Object(current_map), Value::Object(update_map)) => {
                for (key, value) in update_map {
                    current_map.insert(key, value);
                }
                Ok(())
            }
            (current_val, update) => {
                *current_val = update;
                Ok(())
            }
        }
    }

    /// Get the names of all nodes in the graph (excluding START and END).
    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.keys().map(|s| s.as_str()).collect()
    }

    /// Get the recursion limit.
    pub fn recursion_limit(&self) -> usize {
        self.recursion_limit
    }

    /// Get the input JSON schema, if specified.
    pub fn get_input_schema(&self) -> Option<&Value> {
        self.input_schema.as_ref()
    }

    /// Get the output JSON schema, if specified.
    pub fn get_output_schema(&self) -> Option<&Value> {
        self.output_schema.as_ref()
    }
}

impl fmt::Debug for CompiledStateGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledStateGraph")
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("edges", &self.edges)
            .field("outgoing_keys", &self.outgoing.keys().collect::<Vec<_>>())
            .field("recursion_limit", &self.recursion_limit)
            .field("input_schema", &self.input_schema)
            .field("output_schema", &self.output_schema)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use serde_json::json;

    /// Helper: create an async node action that sets a key in state.
    fn make_set_action(key: &str, value: Value) -> AsyncNodeAction {
        let key = key.to_string();
        Arc::new(move |_state: Value| {
            let key = key.clone();
            let value = value.clone();
            Box::pin(async move {
                let mut update = serde_json::Map::new();
                update.insert(key, value);
                Ok(Value::Object(update))
            })
        })
    }

    /// Helper: create an async node action that reads state and transforms it.
    fn make_transform_action<F>(f: F) -> AsyncNodeAction
    where
        F: Fn(Value) -> Result<Value, LangGraphError> + Send + Sync + 'static,
    {
        Arc::new(move |state: Value| {
            let result = f(state);
            Box::pin(async move { result })
        })
    }

    #[tokio::test]
    async fn test_linear_graph() {
        // A -> B -> C
        let graph = StateGraph::new()
            .add_node("a", make_set_action("step_a", json!(true)))
            .add_node("b", make_set_action("step_b", json!(true)))
            .add_node("c", make_set_action("step_c", json!(true)))
            .set_entry_point("a")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .set_finish_point("c")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({"input": "hello"})).await.unwrap();

        assert_eq!(result["input"], json!("hello"));
        assert_eq!(result["step_a"], json!(true));
        assert_eq!(result["step_b"], json!(true));
        assert_eq!(result["step_c"], json!(true));
    }

    #[tokio::test]
    async fn test_branching_graph() {
        // START -> router -> (left | right) -> END
        let graph = StateGraph::new()
            .add_node("router", make_set_action("routed", json!(true)))
            .add_node("left", make_set_action("branch", json!("left")))
            .add_node("right", make_set_action("branch", json!("right")))
            .set_entry_point("router")
            .add_conditional_edges(
                "router",
                Arc::new(|state: &Value| {
                    if state.get("go_left").and_then(|v| v.as_bool()).unwrap_or(false) {
                        super::super::branch::RouterResult::Single("left".to_string())
                    } else {
                        super::super::branch::RouterResult::Single("right".to_string())
                    }
                }),
                None,
            )
            .set_finish_point("left")
            .set_finish_point("right")
            .compile()
            .unwrap();

        // Test going left.
        let result = graph.invoke(json!({"go_left": true})).await.unwrap();
        assert_eq!(result["branch"], json!("left"));
        assert_eq!(result["routed"], json!(true));

        // Test going right.
        let result = graph.invoke(json!({"go_left": false})).await.unwrap();
        assert_eq!(result["branch"], json!("right"));
    }

    #[tokio::test]
    async fn test_cycle_with_recursion_limit() {
        // A node that loops back to itself until a counter reaches a threshold.
        let graph = StateGraph::new()
            .add_node(
                "counter",
                make_transform_action(|state: Value| {
                    let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                    Ok(json!({"count": count + 1}))
                }),
            )
            .set_entry_point("counter")
            .add_conditional_edges(
                "counter",
                Arc::new(|state: &Value| {
                    let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                    if count >= 5 {
                        super::super::branch::RouterResult::Single(END.to_string())
                    } else {
                        super::super::branch::RouterResult::Single("counter".to_string())
                    }
                }),
                None,
            )
            .compile()
            .unwrap();

        let result = graph.invoke(json!({"count": 0})).await.unwrap();
        assert_eq!(result["count"], json!(5));
    }

    #[tokio::test]
    async fn test_recursion_limit_exceeded() {
        // Infinite loop with a low recursion limit.
        let graph = StateGraph::new()
            .with_recursion_limit(3)
            .add_node("loop_node", make_set_action("x", json!(1)))
            .set_entry_point("loop_node")
            .add_edge("loop_node", "loop_node")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({})).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            LangGraphError::GraphRecursionError(msg) => {
                assert!(msg.contains("Recursion limit of 3"));
            }
            other => panic!("Expected GraphRecursionError, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_single_node_graph() {
        let graph = StateGraph::new()
            .add_node("only", make_set_action("done", json!(true)))
            .set_entry_point("only")
            .set_finish_point("only")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({})).await.unwrap();
        assert_eq!(result["done"], json!(true));
    }

    #[test]
    fn test_compile_no_entry_point() {
        let result = StateGraph::new()
            .add_node("a", make_set_action("x", json!(1)))
            .compile();

        assert!(result.is_err());
    }

    #[test]
    fn test_compile_unknown_edge_target() {
        let result = StateGraph::new()
            .add_node("a", make_set_action("x", json!(1)))
            .set_entry_point("a")
            .add_edge("a", "nonexistent")
            .compile();

        assert!(result.is_err());
    }

    #[test]
    fn test_compile_unknown_edge_source() {
        let result = StateGraph::new()
            .add_node("a", make_set_action("x", json!(1)))
            .set_entry_point("a")
            .add_edge("nonexistent", "a")
            .compile();

        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_state_merging() {
        // Each node adds a different key; all should be present at the end.
        let graph = StateGraph::new()
            .add_node("first", make_set_action("a", json!(1)))
            .add_node("second", make_set_action("b", json!(2)))
            .set_entry_point("first")
            .add_edge("first", "second")
            .set_finish_point("second")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({"initial": true})).await.unwrap();
        assert_eq!(result["initial"], json!(true));
        assert_eq!(result["a"], json!(1));
        assert_eq!(result["b"], json!(2));
    }

    #[tokio::test]
    async fn test_sync_node_action() {
        let action: NodeAction = Arc::new(|_state: Value| Ok(json!({"sync_result": 42})));

        let graph = StateGraph::new()
            .add_node_sync("sync_node", action)
            .set_entry_point("sync_node")
            .set_finish_point("sync_node")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({})).await.unwrap();
        assert_eq!(result["sync_result"], json!(42));
    }

    #[test]
    fn test_node_names() {
        let graph = StateGraph::new()
            .add_node("alpha", make_set_action("x", json!(1)))
            .add_node("beta", make_set_action("y", json!(2)))
            .set_entry_point("alpha")
            .add_edge("alpha", "beta")
            .set_finish_point("beta")
            .compile()
            .unwrap();

        let mut names = graph.node_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    #[should_panic(expected = "reserved name")]
    fn test_cannot_add_start_node() {
        let _ = StateGraph::new().add_node(START, make_set_action("x", json!(1)));
    }

    #[test]
    #[should_panic(expected = "reserved name")]
    fn test_cannot_add_end_node() {
        let _ = StateGraph::new().add_node(END, make_set_action("x", json!(1)));
    }

    #[tokio::test]
    async fn test_conditional_with_path_map() {
        let mut path_map = HashMap::new();
        path_map.insert("go_a".to_string(), "node_a".to_string());
        path_map.insert("go_b".to_string(), "node_b".to_string());

        let graph = StateGraph::new()
            .add_node("start_node", make_set_action("started", json!(true)))
            .add_node("node_a", make_set_action("result", json!("a")))
            .add_node("node_b", make_set_action("result", json!("b")))
            .set_entry_point("start_node")
            .add_conditional_edges(
                "start_node",
                Arc::new(|_state: &Value| {
                    super::super::branch::RouterResult::Single("go_a".to_string())
                }),
                Some(path_map),
            )
            .set_finish_point("node_a")
            .set_finish_point("node_b")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({})).await.unwrap();
        assert_eq!(result["result"], json!("a"));
    }

    #[tokio::test]
    async fn test_add_sequence() {
        let graph = StateGraph::new()
            .add_sequence(vec![
                ("a", make_set_action("step_a", json!(true))),
                ("b", make_set_action("step_b", json!(true))),
                ("c", make_set_action("step_c", json!(true))),
            ])
            .set_entry_point("a")
            .set_finish_point("c")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({})).await.unwrap();
        assert_eq!(result["step_a"], json!(true));
        assert_eq!(result["step_b"], json!(true));
        assert_eq!(result["step_c"], json!(true));
    }

    #[tokio::test]
    async fn test_set_conditional_entry_point() {
        let graph = StateGraph::new()
            .add_node("a", make_set_action("result", json!("a")))
            .add_node("b", make_set_action("result", json!("b")))
            .set_conditional_entry_point(
                Arc::new(|state: &Value| {
                    if state.get("pick_a").and_then(|v| v.as_bool()).unwrap_or(false) {
                        super::super::branch::RouterResult::Single("a".to_string())
                    } else {
                        super::super::branch::RouterResult::Single("b".to_string())
                    }
                }),
                None,
            )
            .set_finish_point("a")
            .set_finish_point("b")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({"pick_a": true})).await.unwrap();
        assert_eq!(result["result"], json!("a"));
    }

    #[tokio::test]
    async fn test_add_edges_join() {
        // a and b both lead to c
        let graph = StateGraph::new()
            .add_node("a", make_set_action("from_a", json!(true)))
            .add_node("b", make_set_action("from_b", json!(true)))
            .add_node("c", make_set_action("joined", json!(true)))
            .set_entry_point("a")
            .add_edge("a", "b")
            .add_edges(&["a", "b"], "c")
            .set_finish_point("c")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({})).await.unwrap();
        assert_eq!(result["joined"], json!(true));
    }

    #[test]
    fn test_schema_accessors() {
        let graph = StateGraph::new()
            .add_node("a", make_set_action("x", json!(1)))
            .set_entry_point("a")
            .set_finish_point("a")
            .with_input_schema(json!({"type": "object"}))
            .with_output_schema(json!({"type": "object"}))
            .compile()
            .unwrap();

        assert!(graph.get_input_schema().is_some());
        assert!(graph.get_output_schema().is_some());
    }

    #[test]
    fn test_node_spec_with_cache_policy() {
        let cache_policy = CachePolicy {
            ttl: Some(300),
            key_func: Some("my_key_func".to_string()),
        };
        let node = NodeSpec {
            name: "cached_node".to_string(),
            action: make_set_action("x", json!(1)),
            metadata: None,
            retry_policy: None,
            cache_policy: Some(cache_policy),
            ends: None,
            defer: false,
        };

        assert_eq!(node.name, "cached_node");
        assert!(node.cache_policy.is_some());
        let cp = node.cache_policy.unwrap();
        assert_eq!(cp.ttl, Some(300));
        assert_eq!(cp.key_func.as_deref(), Some("my_key_func"));
        assert!(node.ends.is_none());
        assert!(!node.defer);
    }

    #[test]
    fn test_node_spec_defer() {
        let node = NodeSpec {
            name: "deferred_node".to_string(),
            action: make_set_action("x", json!(1)),
            metadata: None,
            retry_policy: None,
            cache_policy: None,
            ends: None,
            defer: true,
        };

        assert_eq!(node.name, "deferred_node");
        assert!(node.defer);
        assert!(node.cache_policy.is_none());
        assert!(node.ends.is_none());
    }

    #[tokio::test]
    async fn test_add_node_with_full_config() {
        let cache_policy = CachePolicy {
            ttl: Some(60),
            key_func: None,
        };
        let mut ends = HashMap::new();
        ends.insert("success".to_string(), "next_node".to_string());
        ends.insert("failure".to_string(), "error_handler".to_string());

        let mut metadata = HashMap::new();
        metadata.insert("key".to_string(), json!("value"));

        let graph = StateGraph::new()
            .add_node_with_full_config(
                "full_node",
                make_set_action("result", json!("done")),
                Some(metadata),
                Some(RetryPolicy::default()),
                Some(cache_policy),
                Some(ends.clone()),
                true,
            )
            .add_node("next_node", make_set_action("next", json!(true)))
            .add_node("error_handler", make_set_action("error", json!(true)))
            .set_entry_point("full_node")
            .set_finish_point("full_node")
            .compile()
            .unwrap();

        let node = graph.nodes.get("full_node").unwrap();
        assert_eq!(node.name, "full_node");
        assert!(node.cache_policy.is_some());
        assert_eq!(node.cache_policy.as_ref().unwrap().ttl, Some(60));
        assert!(node.ends.is_some());
        assert_eq!(node.ends.as_ref().unwrap().len(), 2);
        assert!(node.defer);
        assert!(node.metadata.is_some());
        assert!(node.retry_policy.is_some());

        let result = graph.invoke(json!({})).await.unwrap();
        assert_eq!(result["result"], json!("done"));
    }

    // ── Streaming tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn test_stream_values_mode() {
        // Single node: stream should yield full state after execution.
        let graph = StateGraph::new()
            .add_node("only", make_set_action("done", json!(true)))
            .set_entry_point("only")
            .set_finish_point("only")
            .compile()
            .unwrap();

        let mut stream = graph
            .stream(json!({"input": "hello"}), StreamMode::Values)
            .await
            .unwrap();

        let update = stream.next().await.unwrap().unwrap();
        assert_eq!(update.node, "only");
        assert_eq!(update.mode, StreamMode::Values);
        // Full state should contain both the original input and the node's output.
        assert_eq!(update.data["input"], json!("hello"));
        assert_eq!(update.data["done"], json!(true));

        // No more updates.
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_stream_updates_mode() {
        // Single node: stream should yield only the delta.
        let graph = StateGraph::new()
            .add_node("only", make_set_action("done", json!(true)))
            .set_entry_point("only")
            .set_finish_point("only")
            .compile()
            .unwrap();

        let mut stream = graph
            .stream(json!({"input": "hello"}), StreamMode::Updates)
            .await
            .unwrap();

        let update = stream.next().await.unwrap().unwrap();
        assert_eq!(update.node, "only");
        assert_eq!(update.mode, StreamMode::Updates);
        // Delta should contain only the node's output, not the original input.
        assert_eq!(update.data["done"], json!(true));
        assert!(update.data.get("input").is_none());

        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_stream_two_node_graph() {
        // A -> B: should yield two updates in order.
        let graph = StateGraph::new()
            .add_node("a", make_set_action("step_a", json!(1)))
            .add_node("b", make_set_action("step_b", json!(2)))
            .set_entry_point("a")
            .add_edge("a", "b")
            .set_finish_point("b")
            .compile()
            .unwrap();

        let mut stream = graph
            .stream(json!({}), StreamMode::Updates)
            .await
            .unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.node, "a");
        assert_eq!(first.data["step_a"], json!(1));

        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(second.node, "b");
        assert_eq!(second.data["step_b"], json!(2));

        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_stream_conditional_routing() {
        // START -> router -> (left | right) -> END
        let graph = StateGraph::new()
            .add_node("router", make_set_action("routed", json!(true)))
            .add_node("left", make_set_action("branch", json!("left")))
            .add_node("right", make_set_action("branch", json!("right")))
            .set_entry_point("router")
            .add_conditional_edges(
                "router",
                Arc::new(|state: &Value| {
                    if state
                        .get("go_left")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                    {
                        super::super::branch::RouterResult::Single("left".to_string())
                    } else {
                        super::super::branch::RouterResult::Single("right".to_string())
                    }
                }),
                None,
            )
            .set_finish_point("left")
            .set_finish_point("right")
            .compile()
            .unwrap();

        // Test going left.
        let mut stream = graph
            .stream(json!({"go_left": true}), StreamMode::Values)
            .await
            .unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.node, "router");
        assert_eq!(first.data["routed"], json!(true));

        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(second.node, "left");
        assert_eq!(second.data["branch"], json!("left"));

        assert!(stream.next().await.is_none());

        // Test going right.
        let mut stream = graph
            .stream(json!({"go_left": false}), StreamMode::Updates)
            .await
            .unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.node, "router");

        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(second.node, "right");
        assert_eq!(second.data["branch"], json!("right"));

        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_stream_debug_mode() {
        // Debug mode should include step number, elapsed_ms, update, and state.
        let graph = StateGraph::new()
            .add_node("a", make_set_action("val", json!(42)))
            .set_entry_point("a")
            .set_finish_point("a")
            .compile()
            .unwrap();

        let mut stream = graph
            .stream(json!({}), StreamMode::Debug)
            .await
            .unwrap();

        let update = stream.next().await.unwrap().unwrap();
        assert_eq!(update.node, "a");
        assert_eq!(update.mode, StreamMode::Debug);
        assert_eq!(update.data["step"], json!(1));
        assert!(update.data.get("elapsed_ms").is_some());
        assert_eq!(update.data["update"]["val"], json!(42));
        assert_eq!(update.data["state"]["val"], json!(42));

        assert!(stream.next().await.is_none());
    }
}
