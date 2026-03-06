//! StateGraph builder and CompiledStateGraph executor.
//!
//! The [`StateGraph`] provides a builder API for constructing directed graphs of
//! async node actions connected by edges (direct or conditional). Once built,
//! [`StateGraph::compile`] produces a [`CompiledStateGraph`] that can be invoked
//! to run the graph to completion.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use serde_json::Value;

use futures::Stream;

use crate::constants::{END, START};
use crate::errors::LangGraphError;
use crate::types::{CachePolicy, InterruptType, InterruptedState, InvokeResult, RetryPolicy, Send as GraphSend, StreamMode, StreamUpdate};

use super::branch::{Branch, RouterFn, RouterResult};

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

/// A resolved next-node target. Either a plain node name (which receives
/// the current graph state) or a [`GraphSend`] instruction that carries a
/// custom input for the target node.
#[derive(Debug, Clone)]
enum NextNode {
    /// Execute the named node with the current graph state.
    Node(String),
    /// Execute the named node with a custom input value (fan-out / Send).
    Send(GraphSend),
}

impl NextNode {
    /// Return the target node name regardless of variant.
    fn name(&self) -> &str {
        match self {
            NextNode::Node(n) => n,
            NextNode::Send(s) => &s.node,
        }
    }
}

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

impl Clone for NodeSpec {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            action: Arc::clone(&self.action),
            metadata: self.metadata.clone(),
            retry_policy: self.retry_policy.clone(),
            cache_policy: self.cache_policy.clone(),
            ends: self.ends.clone(),
            defer: self.defer,
        }
    }
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
    interrupt_before: HashSet<String>,
    interrupt_after: HashSet<String>,
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
            interrupt_before: HashSet::new(),
            interrupt_after: HashSet::new(),
        }
    }

    /// Set the recursion limit for graph execution (default: 25).
    pub fn with_recursion_limit(mut self, limit: usize) -> Self {
        self.recursion_limit = limit;
        self
    }

    /// Configure nodes that should pause execution **before** they run.
    ///
    /// When the graph reaches one of these nodes, execution is interrupted and
    /// an [`InterruptedState`] is returned so a human can inspect the state,
    /// optionally modify it, and then call [`CompiledStateGraph::resume`] to
    /// continue.
    pub fn interrupt_before(mut self, nodes: Vec<&str>) -> Self {
        for n in nodes {
            self.interrupt_before.insert(n.to_string());
        }
        self
    }

    /// Configure nodes that should pause execution **after** they run.
    ///
    /// The node's action executes and its output is merged into the state, but
    /// execution pauses before continuing to successor nodes.
    pub fn interrupt_after(mut self, nodes: Vec<&str>) -> Self {
        for n in nodes {
            self.interrupt_after.insert(n.to_string());
        }
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

    /// Add a compiled subgraph as a node.
    ///
    /// The subgraph runs as a single step: the parent's current state is passed
    /// to the subgraph's [`invoke`](CompiledStateGraph::invoke), and the
    /// subgraph's final output is merged back into the parent's state.
    ///
    /// # Example
    ///
    /// ```rust
    /// use std::sync::Arc;
    /// use serde_json::{json, Value};
    /// use langgraph::graph::state::{StateGraph, AsyncNodeAction};
    /// use langgraph::errors::LangGraphError;
    ///
    /// let action: AsyncNodeAction = Arc::new(|_s: Value| {
    ///     Box::pin(async { Ok(json!({"from_inner": true})) })
    /// });
    ///
    /// let inner = StateGraph::new()
    ///     .add_node("a", action)
    ///     .set_entry_point("a")
    ///     .set_finish_point("a")
    ///     .compile()
    ///     .unwrap();
    ///
    /// let outer = StateGraph::new()
    ///     .add_subgraph("sub", inner)
    ///     .set_entry_point("sub")
    ///     .set_finish_point("sub")
    ///     .compile()
    ///     .unwrap();
    /// ```
    pub fn add_subgraph(self, name: &str, subgraph: CompiledStateGraph) -> Self {
        use super::subgraph::SubgraphNode;
        let action = SubgraphNode::new(subgraph).into_action();
        self.add_node(name, action)
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

    /// Add a conditional edge whose router may return [`RouterResult::Sends`]
    /// to fan out work to multiple node invocations with custom inputs.
    ///
    /// This is functionally identical to [`add_conditional_edges`](Self::add_conditional_edges)
    /// — the router can return any [`RouterResult`] variant — but serves as
    /// documentation that Send-based fan-out is the intended use case.
    pub fn add_conditional_edges_with_send(
        self,
        from: &str,
        router: RouterFn,
        path_map: Option<HashMap<String, String>>,
    ) -> Self {
        self.add_conditional_edges(from, router, path_map)
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
            interrupt_before: self.interrupt_before,
            interrupt_after: self.interrupt_after,
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
            .field("interrupt_before", &self.interrupt_before)
            .field("interrupt_after", &self.interrupt_after)
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
    /// Nodes that trigger an interrupt **before** execution.
    interrupt_before: HashSet<String>,
    /// Nodes that trigger an interrupt **after** execution.
    interrupt_after: HashSet<String>,
}

impl Clone for CompiledStateGraph {
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
            edges: self.edges.clone(),
            outgoing: self.outgoing.clone(),
            recursion_limit: self.recursion_limit,
            input_schema: self.input_schema.clone(),
            output_schema: self.output_schema.clone(),
            interrupt_before: self.interrupt_before.clone(),
            interrupt_after: self.interrupt_after.clone(),
        }
    }
}

impl CompiledStateGraph {
    /// Invoke the graph with initial state, running to completion.
    ///
    /// Execution starts from the START node, follows edges through the graph,
    /// and stops when reaching the END node or when no more nodes are reachable.
    /// Returns the final merged state.
    ///
    /// When a conditional edge returns [`RouterResult::Sends`], each
    /// [`Send`](crate::types::Send) instruction invokes the target node with its
    /// custom `arg` instead of the current graph state. All results are merged
    /// back into the state, enabling map-reduce / fan-out patterns.
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
            let executable: Vec<NextNode> = current_nodes
                .into_iter()
                .filter(|n| n.name() != END)
                .collect();

            // If all targets are END, we're done.
            if executable.is_empty() {
                break;
            }

            // Execute each node in order (sequential execution for now).
            let mut next_nodes: Vec<NextNode> = Vec::new();

            for next in &executable {
                step_count += 1;
                if step_count > self.recursion_limit {
                    return Err(LangGraphError::GraphRecursionError(format!(
                        "Recursion limit of {} reached after {} steps",
                        self.recursion_limit, step_count
                    )));
                }

                let node_name = next.name();

                // For Send instructions, invoke the node with the custom arg.
                // For plain nodes, invoke with the current state.
                let node_input = match next {
                    NextNode::Send(send) => send.arg.clone(),
                    NextNode::Node(_) => state.clone(),
                };

                let node = self.nodes.get(node_name).ok_or_else(|| {
                    LangGraphError::Other(format!(
                        "Node '{node_name}' not found in compiled graph"
                    ))
                })?;
                let update = (node.action)(node_input).await?;
                Self::merge_state(&mut state, update)?;

                // Determine the next nodes from this node.
                let mut successors = self.get_next_nodes(node_name, &state)?;
                next_nodes.append(&mut successors);
            }

            current_nodes = next_nodes;
        }

        Ok(state)
    }

    /// Invoke the graph with interrupt support for human-in-the-loop workflows.
    ///
    /// Behaves like [`invoke`](Self::invoke) but checks for configured interrupt
    /// points. When a node in `interrupt_before` or `interrupt_after` is
    /// encountered, execution pauses and an [`InvokeResult::Interrupted`] is
    /// returned containing the current state and metadata needed to resume.
    ///
    /// If no interrupts are configured (or none are triggered), the graph runs
    /// to completion and returns [`InvokeResult::Complete`].
    pub async fn invoke_with_interrupt(&self, input: Value) -> Result<InvokeResult, LangGraphError> {
        self.run_with_interrupt(input, None, None).await
    }

    /// Resume execution after an interrupt.
    ///
    /// Takes the state from an [`InterruptedState`] (possibly modified by a
    /// human) and an optional state update to merge before continuing. Execution
    /// picks up from the point where it was interrupted.
    ///
    /// * For `InterruptType::Before` interrupts the interrupted node is executed
    ///   first.
    /// * For `InterruptType::After` interrupts execution continues with the
    ///   successor nodes.
    pub async fn resume(
        &self,
        interrupted: InterruptedState,
        update: Option<Value>,
    ) -> Result<InvokeResult, LangGraphError> {
        let mut state = interrupted.state;

        // Apply the optional human-provided update.
        if let Some(upd) = update {
            Self::merge_state(&mut state, upd)?;
        }

        // Determine where to pick up.
        let (resume_nodes, skip_interrupt_for) = match interrupted.interrupt_type {
            InterruptType::Before => {
                // The node hasn't run yet — start from it, and skip its
                // interrupt_before check so we don't loop.
                (
                    vec![interrupted.interrupted_at.clone()],
                    Some(interrupted.interrupted_at),
                )
            }
            InterruptType::After => {
                // The node already ran — continue with its successors.
                (interrupted.next_nodes, None)
            }
        };

        self.run_with_interrupt(state, Some(resume_nodes), skip_interrupt_for)
            .await
    }

    /// Internal execution loop shared by `invoke_with_interrupt` and `resume`.
    ///
    /// When `resume_from` is `None` the graph starts from START. Otherwise it
    /// starts from the provided node list. `skip_interrupt_for` names a node
    /// whose interrupt check should be skipped once (used when resuming from a
    /// `Before` interrupt so the same node does not immediately re-interrupt).
    async fn run_with_interrupt(
        &self,
        input: Value,
        resume_from: Option<Vec<String>>,
        skip_interrupt_for: Option<String>,
    ) -> Result<InvokeResult, LangGraphError> {
        let mut state = input;
        let mut step_count: usize = 0;
        let mut skip_node = skip_interrupt_for;

        let mut current_nodes: Vec<NextNode> = match resume_from {
            Some(nodes) => nodes.into_iter().map(NextNode::Node).collect(),
            None => self.get_next_nodes(START, &state)?,
        };

        loop {
            if current_nodes.is_empty() {
                break;
            }

            let executable: Vec<NextNode> = current_nodes
                .into_iter()
                .filter(|n| n.name() != END)
                .collect();

            if executable.is_empty() {
                break;
            }

            let mut next_nodes: Vec<NextNode> = Vec::new();

            for next in &executable {
                step_count += 1;
                if step_count > self.recursion_limit {
                    return Err(LangGraphError::GraphRecursionError(format!(
                        "Recursion limit of {} reached after {} steps",
                        self.recursion_limit, step_count
                    )));
                }

                let node_name = next.name();

                // --- interrupt_before check ---
                let should_skip = skip_node.as_deref() == Some(node_name);
                if !should_skip && self.interrupt_before.contains(node_name) {
                    let successors = self.get_next_nodes(node_name, &state)?;
                    let successor_names = successors.iter().map(|n| n.name().to_string()).collect();
                    return Ok(InvokeResult::Interrupted(InterruptedState {
                        state,
                        interrupted_at: node_name.to_string(),
                        interrupt_type: InterruptType::Before,
                        next_nodes: successor_names,
                    }));
                }
                // Clear skip after the first node is processed.
                if should_skip {
                    skip_node = None;
                }

                // For Send instructions, invoke the node with the custom arg.
                let node_input = match next {
                    NextNode::Send(send) => send.arg.clone(),
                    NextNode::Node(_) => state.clone(),
                };

                let node = self.nodes.get(node_name).ok_or_else(|| {
                    LangGraphError::Other(format!(
                        "Node '{node_name}' not found in compiled graph"
                    ))
                })?;
                let update = (node.action)(node_input).await?;
                Self::merge_state(&mut state, update)?;

                // --- interrupt_after check ---
                if self.interrupt_after.contains(node_name) {
                    let successors = self.get_next_nodes(node_name, &state)?;
                    let successor_names = successors.iter().map(|n| n.name().to_string()).collect();
                    return Ok(InvokeResult::Interrupted(InterruptedState {
                        state,
                        interrupted_at: node_name.to_string(),
                        interrupt_type: InterruptType::After,
                        next_nodes: successor_names,
                    }));
                }

                let mut successors = self.get_next_nodes(node_name, &state)?;
                next_nodes.append(&mut successors);
            }

            current_nodes = next_nodes;
        }

        Ok(InvokeResult::Complete(state))
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
    /// Used by the spawned streaming task. Returns plain node names (Send
    /// information is lost — the streaming path does not yet support fan-out
    /// with custom inputs).
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
    ///
    /// Returns [`NextNode`] items that preserve [`GraphSend`] instructions
    /// so the execution loop can pass custom inputs to fan-out targets.
    fn get_next_nodes(
        &self,
        current: &str,
        state: &Value,
    ) -> Result<Vec<NextNode>, LangGraphError> {
        let Some(edges) = self.outgoing.get(current) else {
            return Ok(Vec::new());
        };

        let mut targets = Vec::new();
        for edge_ref in edges {
            match edge_ref {
                EdgeRef::Direct(target) => {
                    targets.push(NextNode::Node(target.clone()));
                }
                EdgeRef::Conditional(branch) => {
                    let raw = branch.resolve_raw(state)?;
                    match raw {
                        RouterResult::Single(node) => {
                            targets.push(NextNode::Node(node));
                        }
                        RouterResult::Multiple(nodes) => {
                            for n in nodes {
                                targets.push(NextNode::Node(n));
                            }
                        }
                        RouterResult::Sends(sends) => {
                            for s in sends {
                                targets.push(NextNode::Send(s));
                            }
                        }
                    }
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

    // ── Time-travel API ──────────────────────────────────────────────

    /// List the full checkpoint history for a thread.
    ///
    /// Returns a [`CheckpointEntry`] for every checkpoint stored under the
    /// given `thread_id`, ordered from oldest to newest. This is the
    /// primary entry-point for inspecting past states.
    pub async fn get_state_history(
        &self,
        thread_id: &str,
        saver: &dyn crate::checkpoint::CheckpointSaver,
    ) -> Result<Vec<crate::checkpoint::CheckpointEntry>, LangGraphError> {
        saver.list_checkpoints(thread_id).await
    }

    /// Replay execution from a historical checkpoint.
    ///
    /// Loads the state stored at `checkpoint_id` for the given `thread_id`
    /// and continues graph execution from that state. The original thread
    /// history is **not** modified; this simply re-runs the graph starting
    /// from the historical state.
    ///
    /// Returns the final state after execution completes.
    pub async fn replay_from(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
        saver: &dyn crate::checkpoint::CheckpointSaver,
    ) -> Result<Value, LangGraphError> {
        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), Value::String(thread_id.to_string()));
        config.insert(
            "checkpoint_id".to_string(),
            Value::String(checkpoint_id.to_string()),
        );

        let tuple = saver
            .get_tuple(&config)
            .await?
            .ok_or_else(|| {
                LangGraphError::Other(format!(
                    "Checkpoint '{}' not found for thread '{}'",
                    checkpoint_id, thread_id
                ))
            })?;

        // Build the state from channel_values.
        let state = Value::Object(
            tuple
                .checkpoint
                .channel_values
                .into_iter()
                .collect(),
        );

        self.invoke(state).await
    }

    /// Fork a new thread from a historical checkpoint.
    ///
    /// Loads the state stored at `checkpoint_id` for `thread_id` and saves
    /// it as the initial checkpoint of `new_thread_id`. The new thread is
    /// completely independent of the original.
    ///
    /// Returns the forked state.
    pub async fn fork_from(
        &self,
        thread_id: &str,
        checkpoint_id: &str,
        new_thread_id: &str,
        saver: &dyn crate::checkpoint::CheckpointSaver,
    ) -> Result<Value, LangGraphError> {
        let mut config = HashMap::new();
        config.insert("thread_id".to_string(), Value::String(thread_id.to_string()));
        config.insert(
            "checkpoint_id".to_string(),
            Value::String(checkpoint_id.to_string()),
        );

        let tuple = saver
            .get_tuple(&config)
            .await?
            .ok_or_else(|| {
                LangGraphError::Other(format!(
                    "Checkpoint '{}' not found for thread '{}'",
                    checkpoint_id, thread_id
                ))
            })?;

        // Save the checkpoint under the new thread.
        let mut new_config = HashMap::new();
        new_config.insert(
            "thread_id".to_string(),
            Value::String(new_thread_id.to_string()),
        );

        let metadata = tuple.metadata.unwrap_or_else(|| {
            crate::checkpoint::CheckpointMetadata {
                source: "fork".to_string(),
                step: 0,
                writes: None,
                extra: HashMap::new(),
            }
        });

        saver
            .put(&new_config, tuple.checkpoint.clone(), metadata)
            .await?;

        let state = Value::Object(
            tuple
                .checkpoint
                .channel_values
                .into_iter()
                .collect(),
        );

        Ok(state)
    }
}

impl CompiledStateGraph {
    /// Generate a Mermaid diagram string representing the graph structure.
    ///
    /// The output can be pasted into Mermaid-compatible renderers
    /// (GitHub markdown, mermaid.live, etc.).
    ///
    /// - Start/end nodes use rounded shape `([...])`
    /// - Direct edges use solid arrows `-->`
    /// - Conditional edges use dotted arrows `-. label .->`
    /// - Interrupt nodes are highlighted with a special style
    pub fn draw_mermaid(&self) -> String {
        let mut lines: Vec<String> = vec!["graph TD".to_string()];

        // Collect all node names. Always include __start__ and __end__.
        let mut node_names: Vec<String> = Vec::new();
        node_names.push(START.to_string());
        let mut sorted_keys: Vec<&String> = self.nodes.keys().collect();
        sorted_keys.sort();
        for name in &sorted_keys {
            node_names.push((*name).clone());
        }
        node_names.push(END.to_string());

        // Emit node declarations with rounded shape.
        for name in &node_names {
            lines.push(format!("    {}([{}])", name, name));
        }

        // Emit edges from the stored edge list.
        for edge in &self.edges {
            match edge {
                Edge::Direct { from, to } => {
                    lines.push(format!("    {} --> {}", from, to));
                }
                Edge::Conditional { from, branch } => {
                    if let Some(ref ends) = branch.ends {
                        let mut sorted_ends: Vec<(&String, &String)> = ends.iter().collect();
                        sorted_ends.sort_by_key(|(k, _)| (*k).clone());
                        for (label, target) in sorted_ends {
                            lines.push(format!("    {} -. {} .-> {}", from, label, target));
                        }
                    } else {
                        // No path_map: we cannot know the targets statically,
                        // so emit a single dotted edge with a generic label.
                        lines.push(format!("    {} -. condition .-> ???", from));
                    }
                }
            }
        }

        // Emit styles for interrupt nodes.
        let mut interrupt_nodes: Vec<&String> = self
            .interrupt_before
            .iter()
            .chain(self.interrupt_after.iter())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        interrupt_nodes.sort();
        for name in interrupt_nodes {
            lines.push(format!(
                "    style {} fill:#f9f,stroke:#333",
                name
            ));
        }

        lines.join("\n")
    }

    /// Return the Mermaid diagram wrapped in a markdown code block.
    ///
    /// This is useful for embedding in documentation or chat messages.
    pub fn draw_mermaid_png(&self) -> String {
        format!("```mermaid\n{}\n```", self.draw_mermaid())
    }

    /// Serialize the graph structure (nodes, edges) to a JSON [`Value`] for debugging.
    pub fn to_json(&self) -> Value {
        let node_list: Vec<Value> = {
            let mut sorted_keys: Vec<&String> = self.nodes.keys().collect();
            sorted_keys.sort();
            sorted_keys
                .iter()
                .map(|name| {
                    let node = &self.nodes[*name];
                    let mut obj = serde_json::Map::new();
                    obj.insert("name".into(), Value::String(node.name.clone()));
                    if let Some(ref meta) = node.metadata {
                        obj.insert(
                            "metadata".into(),
                            serde_json::to_value(meta).unwrap_or(Value::Null),
                        );
                    }
                    Value::Object(obj)
                })
                .collect()
        };

        let edge_list: Vec<Value> = self
            .edges
            .iter()
            .map(|edge| match edge {
                Edge::Direct { from, to } => {
                    serde_json::json!({
                        "type": "direct",
                        "from": from,
                        "to": to,
                    })
                }
                Edge::Conditional { from, branch } => {
                    let mut obj = serde_json::json!({
                        "type": "conditional",
                        "from": from,
                    });
                    if let Some(ref ends) = branch.ends {
                        obj["path_map"] = serde_json::to_value(ends).unwrap_or(Value::Null);
                    }
                    obj
                }
            })
            .collect();

        serde_json::json!({
            "nodes": node_list,
            "edges": edge_list,
            "entry_point": self.outgoing.contains_key(START).then(|| START),
            "interrupt_before": self.interrupt_before.iter().collect::<Vec<_>>(),
            "interrupt_after": self.interrupt_after.iter().collect::<Vec<_>>(),
        })
    }

    /// Export the graph topology as a serializable [`GraphDefinition`].
    ///
    /// This captures all structural information (nodes, edges, conditional
    /// routing targets, interrupts) but not the runtime closures.
    pub fn to_definition(&self) -> super::serialize::GraphDefinition {
        use super::serialize::{ConditionalEdgeDef, GraphDefinition};

        let mut node_names: Vec<String> = self.nodes.keys().cloned().collect();
        node_names.sort();

        let mut direct_edges: Vec<(String, String)> = Vec::new();
        let mut conditional_edges: Vec<ConditionalEdgeDef> = Vec::new();

        for edge in &self.edges {
            match edge {
                Edge::Direct { from, to } => {
                    direct_edges.push((from.clone(), to.clone()));
                }
                Edge::Conditional { from, branch } => {
                    let targets: Vec<String> = if let Some(ref ends) = branch.ends {
                        ends.values().cloned().collect()
                    } else {
                        Vec::new()
                    };

                    let labels = branch.ends.clone();

                    conditional_edges.push(ConditionalEdgeDef {
                        from: from.clone(),
                        targets,
                        labels,
                    });
                }
            }
        }

        // Determine entry point from the outgoing edges of START.
        let entry_point = direct_edges
            .iter()
            .find(|(from, _)| from == START)
            .map(|(_, to)| to.clone())
            .unwrap_or_else(|| {
                // Fall back: first node in sorted order.
                node_names.first().cloned().unwrap_or_default()
            });

        let mut interrupt_before: Vec<String> = self.interrupt_before.iter().cloned().collect();
        interrupt_before.sort();
        let mut interrupt_after: Vec<String> = self.interrupt_after.iter().cloned().collect();
        interrupt_after.sort();

        GraphDefinition {
            nodes: node_names,
            edges: direct_edges,
            conditional_edges,
            entry_point,
            interrupt_before,
            interrupt_after,
        }
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
            .field("interrupt_before", &self.interrupt_before)
            .field("interrupt_after", &self.interrupt_after)
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

    // ── Interrupt / Human-in-the-loop tests ─────────────────────────

    #[tokio::test]
    async fn test_interrupt_before() {
        // A -> B -> C, interrupt before B
        let graph = StateGraph::new()
            .add_node("a", make_set_action("step_a", json!(true)))
            .add_node("b", make_set_action("step_b", json!(true)))
            .add_node("c", make_set_action("step_c", json!(true)))
            .set_entry_point("a")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .set_finish_point("c")
            .interrupt_before(vec!["b"])
            .compile()
            .unwrap();

        let result = graph.invoke_with_interrupt(json!({"input": "hello"})).await.unwrap();

        match result {
            InvokeResult::Interrupted(interrupted) => {
                assert_eq!(interrupted.interrupted_at, "b");
                assert_eq!(interrupted.interrupt_type, InterruptType::Before);
                // Node A should have run, but not B.
                assert_eq!(interrupted.state["step_a"], json!(true));
                assert!(interrupted.state.get("step_b").is_none());
            }
            InvokeResult::Complete(_) => panic!("Expected interrupt, got completion"),
        }
    }

    #[tokio::test]
    async fn test_interrupt_after() {
        // A -> B -> C, interrupt after B
        let graph = StateGraph::new()
            .add_node("a", make_set_action("step_a", json!(true)))
            .add_node("b", make_set_action("step_b", json!(true)))
            .add_node("c", make_set_action("step_c", json!(true)))
            .set_entry_point("a")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .set_finish_point("c")
            .interrupt_after(vec!["b"])
            .compile()
            .unwrap();

        let result = graph.invoke_with_interrupt(json!({"input": "hello"})).await.unwrap();

        match result {
            InvokeResult::Interrupted(interrupted) => {
                assert_eq!(interrupted.interrupted_at, "b");
                assert_eq!(interrupted.interrupt_type, InterruptType::After);
                // Both A and B should have run.
                assert_eq!(interrupted.state["step_a"], json!(true));
                assert_eq!(interrupted.state["step_b"], json!(true));
                // C should not have run.
                assert!(interrupted.state.get("step_c").is_none());
                // Next nodes should include C.
                assert!(interrupted.next_nodes.contains(&"c".to_string()));
            }
            InvokeResult::Complete(_) => panic!("Expected interrupt, got completion"),
        }
    }

    #[tokio::test]
    async fn test_interrupt_resume_after_interrupt() {
        // A -> B -> C, interrupt before B, then resume.
        let graph = StateGraph::new()
            .add_node("a", make_set_action("step_a", json!(true)))
            .add_node("b", make_set_action("step_b", json!(true)))
            .add_node("c", make_set_action("step_c", json!(true)))
            .set_entry_point("a")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .set_finish_point("c")
            .interrupt_before(vec!["b"])
            .compile()
            .unwrap();

        // First invocation — should interrupt before B.
        let result = graph.invoke_with_interrupt(json!({"input": "hello"})).await.unwrap();
        let interrupted = match result {
            InvokeResult::Interrupted(i) => i,
            InvokeResult::Complete(_) => panic!("Expected interrupt"),
        };
        assert_eq!(interrupted.interrupted_at, "b");

        // Resume without update — should run B and C to completion.
        let result = graph.resume(interrupted, None).await.unwrap();
        match result {
            InvokeResult::Complete(state) => {
                assert_eq!(state["step_a"], json!(true));
                assert_eq!(state["step_b"], json!(true));
                assert_eq!(state["step_c"], json!(true));
            }
            InvokeResult::Interrupted(_) => panic!("Expected completion after resume"),
        }
    }

    #[tokio::test]
    async fn test_interrupt_resume_with_update() {
        // A -> B -> C, interrupt before B, human provides state update.
        let graph = StateGraph::new()
            .add_node("a", make_set_action("step_a", json!(true)))
            .add_node(
                "b",
                make_transform_action(|state: Value| {
                    let approved = state.get("human_approved").and_then(|v| v.as_bool()).unwrap_or(false);
                    Ok(json!({"step_b": true, "was_approved": approved}))
                }),
            )
            .add_node("c", make_set_action("step_c", json!(true)))
            .set_entry_point("a")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .set_finish_point("c")
            .interrupt_before(vec!["b"])
            .compile()
            .unwrap();

        let result = graph.invoke_with_interrupt(json!({})).await.unwrap();
        let interrupted = match result {
            InvokeResult::Interrupted(i) => i,
            InvokeResult::Complete(_) => panic!("Expected interrupt"),
        };

        // Human provides an update before resuming.
        let result = graph
            .resume(interrupted, Some(json!({"human_approved": true})))
            .await
            .unwrap();

        match result {
            InvokeResult::Complete(state) => {
                assert_eq!(state["step_a"], json!(true));
                assert_eq!(state["step_b"], json!(true));
                assert_eq!(state["was_approved"], json!(true));
                assert_eq!(state["step_c"], json!(true));
                assert_eq!(state["human_approved"], json!(true));
            }
            InvokeResult::Interrupted(_) => panic!("Expected completion"),
        }
    }

    #[tokio::test]
    async fn test_interrupt_no_interrupt_normal_flow() {
        // No interrupts configured — should complete normally.
        let graph = StateGraph::new()
            .add_node("a", make_set_action("step_a", json!(true)))
            .add_node("b", make_set_action("step_b", json!(true)))
            .set_entry_point("a")
            .add_edge("a", "b")
            .set_finish_point("b")
            .compile()
            .unwrap();

        let result = graph.invoke_with_interrupt(json!({"input": "hello"})).await.unwrap();
        match result {
            InvokeResult::Complete(state) => {
                assert_eq!(state["step_a"], json!(true));
                assert_eq!(state["step_b"], json!(true));
            }
            InvokeResult::Interrupted(_) => panic!("Expected completion, got interrupt"),
        }
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

    // ── Subgraph composition tests ──────────────────────────────────

    #[tokio::test]
    async fn test_subgraph_simple() {
        // Inner graph: inner_a -> inner_b
        let inner = StateGraph::new()
            .add_node("inner_a", make_set_action("inner_step_a", json!(true)))
            .add_node("inner_b", make_set_action("inner_step_b", json!(true)))
            .set_entry_point("inner_a")
            .add_edge("inner_a", "inner_b")
            .set_finish_point("inner_b")
            .compile()
            .unwrap();

        // Outer graph: setup -> subgraph -> finish
        let outer = StateGraph::new()
            .add_node("setup", make_set_action("setup_done", json!(true)))
            .add_subgraph("subgraph", inner)
            .add_node("finish", make_set_action("finish_done", json!(true)))
            .set_entry_point("setup")
            .add_edge("setup", "subgraph")
            .add_edge("subgraph", "finish")
            .set_finish_point("finish")
            .compile()
            .unwrap();

        let result = outer.invoke(json!({"input": "hello"})).await.unwrap();

        assert_eq!(result["input"], json!("hello"));
        assert_eq!(result["setup_done"], json!(true));
        assert_eq!(result["inner_step_a"], json!(true));
        assert_eq!(result["inner_step_b"], json!(true));
        assert_eq!(result["finish_done"], json!(true));
    }

    #[tokio::test]
    async fn test_subgraph_state_passthrough() {
        // Verify that state keys from the parent flow into the subgraph and
        // keys produced by the subgraph flow back to the parent.
        let inner = StateGraph::new()
            .add_node(
                "transform",
                make_transform_action(|state: Value| {
                    let x = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                    Ok(json!({"y": x * 2}))
                }),
            )
            .set_entry_point("transform")
            .set_finish_point("transform")
            .compile()
            .unwrap();

        let outer = StateGraph::new()
            .add_node("init", make_set_action("x", json!(5)))
            .add_subgraph("sub", inner)
            .add_node(
                "check",
                make_transform_action(|state: Value| {
                    let y = state.get("y").and_then(|v| v.as_i64()).unwrap_or(0);
                    Ok(json!({"z": y + 1}))
                }),
            )
            .set_entry_point("init")
            .add_edge("init", "sub")
            .add_edge("sub", "check")
            .set_finish_point("check")
            .compile()
            .unwrap();

        let result = outer.invoke(json!({})).await.unwrap();

        // init sets x=5, subgraph reads x=5 and sets y=10, check reads y=10 and sets z=11
        assert_eq!(result["x"], json!(5));
        assert_eq!(result["y"], json!(10));
        assert_eq!(result["z"], json!(11));
    }

    #[tokio::test]
    async fn test_subgraph_in_conditional_branch() {
        // Build two subgraphs for different branches.
        let left_sub = StateGraph::new()
            .add_node("left_inner", make_set_action("path", json!("left_subgraph")))
            .set_entry_point("left_inner")
            .set_finish_point("left_inner")
            .compile()
            .unwrap();

        let right_sub = StateGraph::new()
            .add_node("right_inner", make_set_action("path", json!("right_subgraph")))
            .set_entry_point("right_inner")
            .set_finish_point("right_inner")
            .compile()
            .unwrap();

        let outer = StateGraph::new()
            .add_node("router", make_set_action("routed", json!(true)))
            .add_subgraph("left_sub", left_sub)
            .add_subgraph("right_sub", right_sub)
            .set_entry_point("router")
            .add_conditional_edges(
                "router",
                Arc::new(|state: &Value| {
                    if state.get("go_left").and_then(|v| v.as_bool()).unwrap_or(false) {
                        super::super::branch::RouterResult::Single("left_sub".to_string())
                    } else {
                        super::super::branch::RouterResult::Single("right_sub".to_string())
                    }
                }),
                None,
            )
            .set_finish_point("left_sub")
            .set_finish_point("right_sub")
            .compile()
            .unwrap();

        // Test left branch
        let result = outer.invoke(json!({"go_left": true})).await.unwrap();
        assert_eq!(result["routed"], json!(true));
        assert_eq!(result["path"], json!("left_subgraph"));

        // Test right branch
        let result = outer.invoke(json!({"go_left": false})).await.unwrap();
        assert_eq!(result["routed"], json!(true));
        assert_eq!(result["path"], json!("right_subgraph"));
    }

    // ── Time-travel tests ────────────────────────────────────────────

    mod time_travel {
        use super::*;
        use crate::checkpoint::{
            CheckpointMetadata, CheckpointSaver, InMemoryCheckpointSaver,
        };
        use crate::pregel::checkpoint::empty_checkpoint;
        use std::sync::Arc;

        /// Helper: build a simple two-node graph (a -> b).
        fn two_node_graph() -> CompiledStateGraph {
            let action_a: AsyncNodeAction = Arc::new(|_state: Value| {
                Box::pin(async move { Ok(json!({"a_ran": true, "count": 1})) })
            });
            let action_b: AsyncNodeAction = Arc::new(|state: Value| {
                Box::pin(async move {
                    let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                    Ok(json!({"b_ran": true, "count": count + 10}))
                })
            });

            StateGraph::new()
                .add_node("a", action_a)
                .add_node("b", action_b)
                .set_entry_point("a")
                .add_edge("a", "b")
                .set_finish_point("b")
                .compile()
                .unwrap()
        }

        /// Populate the saver with checkpoints mimicking what a graph run would
        /// produce (one per node execution).
        async fn populate_checkpoints(
            saver: &InMemoryCheckpointSaver,
            thread_id: &str,
        ) -> Vec<String> {
            let mut config = std::collections::HashMap::new();
            config.insert("thread_id".to_string(), json!(thread_id));

            let mut ids = Vec::new();

            // Checkpoint after node "a"
            let mut cp_a = empty_checkpoint();
            cp_a.channel_values
                .insert("a_ran".to_string(), json!(true));
            cp_a.channel_values
                .insert("count".to_string(), json!(1));
            let meta_a = CheckpointMetadata {
                source: "loop".to_string(),
                step: 1,
                writes: Some({
                    let mut m = std::collections::HashMap::new();
                    m.insert("a".to_string(), json!({"a_ran": true, "count": 1}));
                    m
                }),
                extra: std::collections::HashMap::new(),
            };
            ids.push(cp_a.id.clone());
            let config = saver.put(&config, cp_a, meta_a).await.unwrap();

            // Checkpoint after node "b"
            let mut cp_b = empty_checkpoint();
            cp_b.channel_values
                .insert("a_ran".to_string(), json!(true));
            cp_b.channel_values
                .insert("b_ran".to_string(), json!(true));
            cp_b.channel_values
                .insert("count".to_string(), json!(11));
            let meta_b = CheckpointMetadata {
                source: "loop".to_string(),
                step: 2,
                writes: Some({
                    let mut m = std::collections::HashMap::new();
                    m.insert("b".to_string(), json!({"b_ran": true, "count": 11}));
                    m
                }),
                extra: std::collections::HashMap::new(),
            };
            ids.push(cp_b.id.clone());
            saver.put(&config, cp_b, meta_b).await.unwrap();

            ids
        }

        #[tokio::test]
        async fn test_get_state_history_lists_all_checkpoints() {
            let saver = InMemoryCheckpointSaver::new();
            let graph = two_node_graph();
            let ids = populate_checkpoints(&saver, "thread-1").await;

            let history = graph.get_state_history("thread-1", &saver).await.unwrap();

            assert_eq!(history.len(), 2, "should have exactly 2 checkpoints");
            assert_eq!(history[0].checkpoint_id, ids[0]);
            assert_eq!(history[1].checkpoint_id, ids[1]);
            assert_eq!(history[0].thread_id, "thread-1");
            assert_eq!(history[1].thread_id, "thread-1");
            // The first checkpoint's node_name should be "a" (from writes key).
            assert_eq!(history[0].node_name, "a");
            assert_eq!(history[1].node_name, "b");
        }

        #[tokio::test]
        async fn test_get_state_history_empty_for_unknown_thread() {
            let saver = InMemoryCheckpointSaver::new();
            let graph = two_node_graph();

            let history = graph
                .get_state_history("nonexistent", &saver)
                .await
                .unwrap();
            assert!(history.is_empty());
        }

        #[tokio::test]
        async fn test_replay_from_checkpoint() {
            let saver = InMemoryCheckpointSaver::new();
            let graph = two_node_graph();
            let ids = populate_checkpoints(&saver, "thread-1").await;

            // Replay from the first checkpoint (after node "a").
            // The state at that point is {a_ran: true, count: 1}.
            // Re-running the graph from that state goes through a -> b again.
            let result = graph
                .replay_from("thread-1", &ids[0], &saver)
                .await
                .unwrap();

            // After replaying: node "a" sets count=1, node "b" reads count
            // and sets count=count+10. But since we feed the *state* back as
            // input, node "a" will overwrite count to 1, then node "b" sets it
            // to 11.
            assert_eq!(result["b_ran"], json!(true));
            assert_eq!(result["a_ran"], json!(true));
            assert_eq!(result["count"], json!(11));
        }

        #[tokio::test]
        async fn test_replay_from_nonexistent_checkpoint_fails() {
            let saver = InMemoryCheckpointSaver::new();
            let graph = two_node_graph();

            let result = graph
                .replay_from("thread-1", "does-not-exist", &saver)
                .await;
            assert!(result.is_err());
        }

        #[tokio::test]
        async fn test_fork_creates_independent_thread() {
            let saver = InMemoryCheckpointSaver::new();
            let graph = two_node_graph();
            let ids = populate_checkpoints(&saver, "thread-1").await;

            // Fork from the first checkpoint into a new thread.
            let forked_state = graph
                .fork_from("thread-1", &ids[0], "thread-2", &saver)
                .await
                .unwrap();

            // The forked state should match checkpoint 0's state.
            assert_eq!(forked_state["a_ran"], json!(true));
            assert_eq!(forked_state["count"], json!(1));

            // The new thread should have its own checkpoint history.
            let history_new = graph
                .get_state_history("thread-2", &saver)
                .await
                .unwrap();
            assert_eq!(history_new.len(), 1);
            assert_eq!(history_new[0].thread_id, "thread-2");

            // The original thread should be unchanged.
            let history_orig = graph
                .get_state_history("thread-1", &saver)
                .await
                .unwrap();
            assert_eq!(history_orig.len(), 2);
        }

        #[tokio::test]
        async fn test_fork_from_nonexistent_checkpoint_fails() {
            let saver = InMemoryCheckpointSaver::new();
            let graph = two_node_graph();

            let result = graph
                .fork_from("thread-1", "nope", "thread-2", &saver)
                .await;
            assert!(result.is_err());
        }
    }

    // ── Send / Fan-out / Map-reduce tests ──────────────────────────────

    #[tokio::test]
    async fn test_send_fan_out_to_worker() {
        // Dispatcher fans out 3 tasks to "worker" with different inputs.
        // Worker reads "task" from its Send arg and writes to "results" array.
        let worker: AsyncNodeAction = Arc::new(|input: Value| {
            Box::pin(async move {
                let task_id = input.get("task").and_then(|v| v.as_i64()).unwrap_or(-1);
                Ok(json!({ "results": [task_id * 10] }))
            })
        });

        let graph = StateGraph::new()
            .add_node(
                "dispatcher",
                make_set_action("dispatched", json!(true)),
            )
            .add_node("worker", worker)
            .set_entry_point("dispatcher")
            .add_conditional_edges_with_send(
                "dispatcher",
                Arc::new(|_state: &Value| {
                    use crate::types::Send as S;
                    RouterResult::Sends(vec![
                        S { node: "worker".into(), arg: json!({"task": 1}) },
                        S { node: "worker".into(), arg: json!({"task": 2}) },
                        S { node: "worker".into(), arg: json!({"task": 3}) },
                    ])
                }),
                None,
            )
            .set_finish_point("worker")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({})).await.unwrap();

        // Dispatcher ran.
        assert_eq!(result["dispatched"], json!(true));
        // The last worker merge wins for the "results" key (shallow merge).
        // Each worker writes its own results array; the last one (task 3) overwrites.
        assert_eq!(result["results"], json!([30]));
    }

    #[tokio::test]
    async fn test_send_map_reduce() {
        // Full map-reduce: dispatcher → fan-out workers → reducer collects.
        //
        // Workers accumulate into a "worker_outputs" array in state. Because
        // merge_state does a shallow object merge, each worker appends via its
        // own unique key, then the reducer reads all of them.

        // Worker: reads Send arg, writes result under a unique key.
        let worker: AsyncNodeAction = Arc::new(|input: Value| {
            Box::pin(async move {
                let task_id = input.get("task").and_then(|v| v.as_i64()).unwrap_or(0);
                let key = format!("worker_result_{}", task_id);
                let mut map = serde_json::Map::new();
                map.insert(key, json!(task_id * 100));
                Ok(Value::Object(map))
            })
        });

        // Reducer: reads worker results and produces a summary.
        let reducer: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let mut total = 0i64;
                if let Value::Object(map) = &state {
                    for (k, v) in map {
                        if k.starts_with("worker_result_") {
                            total += v.as_i64().unwrap_or(0);
                        }
                    }
                }
                Ok(json!({ "total": total }))
            })
        });

        let graph = StateGraph::new()
            .add_node("dispatcher", make_set_action("dispatched", json!(true)))
            .add_node("worker", worker)
            .add_node("reducer", reducer)
            .set_entry_point("dispatcher")
            .add_conditional_edges_with_send(
                "dispatcher",
                Arc::new(|_state: &Value| {
                    use crate::types::Send as S;
                    RouterResult::Sends(vec![
                        S { node: "worker".into(), arg: json!({"task": 1}) },
                        S { node: "worker".into(), arg: json!({"task": 2}) },
                        S { node: "worker".into(), arg: json!({"task": 3}) },
                    ])
                }),
                None,
            )
            // All worker invocations lead to reducer.
            .add_edge("worker", "reducer")
            .set_finish_point("reducer")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({})).await.unwrap();

        assert_eq!(result["dispatched"], json!(true));
        // Workers wrote: worker_result_1=100, worker_result_2=200, worker_result_3=300
        assert_eq!(result["worker_result_1"], json!(100));
        assert_eq!(result["worker_result_2"], json!(200));
        assert_eq!(result["worker_result_3"], json!(300));
        // Reducer summed them: 100 + 200 + 300 = 600
        assert_eq!(result["total"], json!(600));
    }

    #[tokio::test]
    async fn test_send_with_different_inputs() {
        // Each Send targets a different node with a unique arg.
        let alpha: AsyncNodeAction = Arc::new(|input: Value| {
            Box::pin(async move {
                let x = input.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "alpha_result": x + 1 }))
            })
        });

        let beta: AsyncNodeAction = Arc::new(|input: Value| {
            Box::pin(async move {
                let y = input.get("y").and_then(|v| v.as_str()).unwrap_or("").to_uppercase();
                Ok(json!({ "beta_result": y }))
            })
        });

        let graph = StateGraph::new()
            .add_node("start_node", make_set_action("started", json!(true)))
            .add_node("alpha", alpha)
            .add_node("beta", beta)
            .set_entry_point("start_node")
            .add_conditional_edges(
                "start_node",
                Arc::new(|_state: &Value| {
                    use crate::types::Send as S;
                    RouterResult::Sends(vec![
                        S { node: "alpha".into(), arg: json!({"x": 41}) },
                        S { node: "beta".into(), arg: json!({"y": "hello"}) },
                    ])
                }),
                None,
            )
            .set_finish_point("alpha")
            .set_finish_point("beta")
            .compile()
            .unwrap();

        let result = graph.invoke(json!({})).await.unwrap();

        assert_eq!(result["started"], json!(true));
        assert_eq!(result["alpha_result"], json!(42));
        assert_eq!(result["beta_result"], json!("HELLO"));
    }
}

#[cfg(test)]
mod mermaid_tests {
    use super::*;
    use serde_json::json;

    /// Helper: create an async node action that sets a key in state.
    fn make_action(key: &str, value: Value) -> AsyncNodeAction {
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

    #[test]
    fn test_mermaid_simple_linear_graph() {
        let graph = StateGraph::new()
            .add_node("agent", make_action("x", json!(1)))
            .add_node("tools", make_action("y", json!(2)))
            .set_entry_point("agent")
            .add_edge("agent", "tools")
            .add_edge("tools", "agent")
            .set_finish_point("agent")
            .compile()
            .unwrap();

        let mermaid = graph.draw_mermaid();
        assert!(mermaid.contains("graph TD"));
        assert!(mermaid.contains("__start__([__start__])"));
        assert!(mermaid.contains("__end__([__end__])"));
        assert!(mermaid.contains("agent([agent])"));
        assert!(mermaid.contains("tools([tools])"));
        assert!(mermaid.contains("__start__ --> agent"));
        assert!(mermaid.contains("tools --> agent"));
        assert!(mermaid.contains("agent --> tools"));
        assert!(mermaid.contains("agent --> __end__"));
    }

    #[test]
    fn test_mermaid_conditional_edges_with_path_map() {
        let mut path_map = HashMap::new();
        path_map.insert("tool_calls".to_string(), "tools".to_string());
        path_map.insert("no_tool_calls".to_string(), END.to_string());

        let graph = StateGraph::new()
            .add_node("agent", make_action("x", json!(1)))
            .add_node("tools", make_action("y", json!(2)))
            .set_entry_point("agent")
            .add_conditional_edges(
                "agent",
                Arc::new(|_state: &Value| {
                    super::super::branch::RouterResult::Single("tool_calls".to_string())
                }),
                Some(path_map),
            )
            .add_edge("tools", "agent")
            .compile()
            .unwrap();

        let mermaid = graph.draw_mermaid();
        // Conditional edges should use dotted lines with labels.
        assert!(mermaid.contains("agent -. no_tool_calls .-> __end__"));
        assert!(mermaid.contains("agent -. tool_calls .-> tools"));
        // Direct edge.
        assert!(mermaid.contains("tools --> agent"));
    }

    #[test]
    fn test_mermaid_conditional_without_path_map() {
        let graph = StateGraph::new()
            .add_node("router", make_action("x", json!(1)))
            .add_node("a", make_action("y", json!(2)))
            .set_entry_point("router")
            .add_conditional_edges(
                "router",
                Arc::new(|_state: &Value| {
                    super::super::branch::RouterResult::Single("a".to_string())
                }),
                None,
            )
            .set_finish_point("a")
            .compile()
            .unwrap();

        let mermaid = graph.draw_mermaid();
        // Without a path_map, a generic "condition" label should be used.
        assert!(mermaid.contains("router -. condition .-> ???"));
    }

    #[test]
    fn test_mermaid_interrupt_style() {
        let graph = StateGraph::new()
            .add_node("agent", make_action("x", json!(1)))
            .add_node("human", make_action("y", json!(2)))
            .set_entry_point("agent")
            .add_edge("agent", "human")
            .set_finish_point("human")
            .interrupt_before(vec!["human"])
            .compile()
            .unwrap();

        let mermaid = graph.draw_mermaid();
        assert!(mermaid.contains("style human fill:#f9f,stroke:#333"));
    }

    #[test]
    fn test_mermaid_png_wrapping() {
        let graph = StateGraph::new()
            .add_node("a", make_action("x", json!(1)))
            .set_entry_point("a")
            .set_finish_point("a")
            .compile()
            .unwrap();

        let png = graph.draw_mermaid_png();
        assert!(png.starts_with("```mermaid\n"));
        assert!(png.ends_with("\n```"));
        assert!(png.contains("graph TD"));
    }

    #[test]
    fn test_to_json_structure() {
        let mut path_map = HashMap::new();
        path_map.insert("yes".to_string(), "b".to_string());
        path_map.insert("no".to_string(), END.to_string());

        let graph = StateGraph::new()
            .add_node("a", make_action("x", json!(1)))
            .add_node("b", make_action("y", json!(2)))
            .set_entry_point("a")
            .add_conditional_edges(
                "a",
                Arc::new(|_state: &Value| {
                    super::super::branch::RouterResult::Single("yes".to_string())
                }),
                Some(path_map),
            )
            .add_edge("b", "a")
            .set_finish_point("b")
            .compile()
            .unwrap();

        let json_val = graph.to_json();

        // Check nodes array.
        let nodes = json_val["nodes"].as_array().unwrap();
        assert_eq!(nodes.len(), 2);
        assert_eq!(nodes[0]["name"], "a");
        assert_eq!(nodes[1]["name"], "b");

        // Check edges array.
        let edges = json_val["edges"].as_array().unwrap();
        assert!(edges.len() >= 3); // start->a, a->conditional, b->a, b->end

        // Check there is a conditional edge with path_map.
        let conditional_edge = edges.iter().find(|e| e["type"] == "conditional").unwrap();
        assert_eq!(conditional_edge["from"], "a");
        assert!(conditional_edge.get("path_map").is_some());

        // Check entry_point.
        assert_eq!(json_val["entry_point"], "__start__");
    }
}
