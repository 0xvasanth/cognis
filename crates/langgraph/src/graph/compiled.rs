//! Compiled graph module — executable graph produced by compiling a graph definition.
//!
//! This module provides [`CompiledGraph`], the result of "compiling" a declarative
//! graph definition into an executable form. It mirrors Python LangGraph's
//! `CompiledGraph` concept, offering validation, execution, and Mermaid export.
//!
//! # Overview
//!
//! - [`CompiledNode`] — a named node with an executable handler function.
//! - [`CompiledEdge`] — a direct or conditional edge connecting nodes.
//! - [`CompiledGraph`] — the executable graph; runs from entry to finish.
//! - [`GraphDefinitionBuilder`] — declarative builder for graph structure.
//! - [`GraphCompiler`] — compiles a [`GraphDefinitionBuilder`] into a [`CompiledGraph`].
//! - [`ExecutionLog`] / [`LogEntry`] — records execution steps for observability.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::Instant;

use serde_json::Value;

use crate::errors::LangGraphError;

// ---------------------------------------------------------------------------
// CompiledNode
// ---------------------------------------------------------------------------

/// A compiled node that wraps a named handler function.
pub struct CompiledNode {
    /// The name of this node.
    name: String,
    /// The handler function that processes state.
    handler: Box<dyn Fn(Value) -> Result<Value, LangGraphError> + Send + Sync>,
}

impl CompiledNode {
    /// Create a new compiled node.
    pub fn new<F>(name: impl Into<String>, handler: F) -> Self
    where
        F: Fn(Value) -> Result<Value, LangGraphError> + Send + Sync + 'static,
    {
        Self {
            name: name.into(),
            handler: Box::new(handler),
        }
    }

    /// Execute this node with the given state.
    pub fn execute(&self, state: Value) -> Result<Value, LangGraphError> {
        (self.handler)(state)
    }

    /// Return the name of this node.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Debug for CompiledNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledNode")
            .field("name", &self.name)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// CompiledEdge
// ---------------------------------------------------------------------------

/// A compiled edge — either a direct connection or a conditional router.
pub enum CompiledEdge {
    /// A direct edge from one node to another.
    Direct {
        /// Source node name.
        from: String,
        /// Target node name.
        to: String,
    },
    /// A conditional edge whose target is determined at runtime.
    Conditional {
        /// Source node name.
        from: String,
        /// Router function that inspects state and returns a target node name.
        router: Box<dyn Fn(&Value) -> String + Send + Sync>,
        /// The set of possible target node names.
        targets: Vec<String>,
    },
}

impl CompiledEdge {
    /// Return the source node name.
    pub fn source(&self) -> &str {
        match self {
            CompiledEdge::Direct { from, .. } => from,
            CompiledEdge::Conditional { from, .. } => from,
        }
    }

    /// Whether this edge is conditional.
    pub fn is_conditional(&self) -> bool {
        matches!(self, CompiledEdge::Conditional { .. })
    }
}

impl fmt::Debug for CompiledEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompiledEdge::Direct { from, to } => f
                .debug_struct("Direct")
                .field("from", from)
                .field("to", to)
                .finish(),
            CompiledEdge::Conditional { from, targets, .. } => f
                .debug_struct("Conditional")
                .field("from", from)
                .field("targets", targets)
                .finish(),
        }
    }
}

// ---------------------------------------------------------------------------
// LogEntry / ExecutionLog
// ---------------------------------------------------------------------------

/// A single entry in the execution log.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// The name of the node that was executed.
    pub node_name: String,
    /// The input state passed to the node.
    pub input: Value,
    /// The output state returned by the node.
    pub output: Value,
    /// Wall-clock timestamp when execution started (seconds since log creation).
    pub timestamp: f64,
    /// Duration of node execution in milliseconds.
    pub duration_ms: f64,
}

/// Records execution steps for observability and debugging.
#[derive(Debug, Clone)]
pub struct ExecutionLog {
    entries: Vec<LogEntry>,
    start: Option<Instant>,
}

impl ExecutionLog {
    /// Create a new empty execution log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
            start: None,
        }
    }

    /// Record an executed node step with timing information.
    pub fn record(&mut self, node_name: &str, input: &Value, output: &Value) {
        let now = Instant::now();
        let start = *self.start.get_or_insert(now);
        let timestamp = now.duration_since(start).as_secs_f64();
        self.entries.push(LogEntry {
            node_name: node_name.to_string(),
            input: input.clone(),
            output: output.clone(),
            timestamp,
            duration_ms: 0.0,
        });
    }

    /// Record a step with explicit duration.
    pub fn record_with_duration(
        &mut self,
        node_name: &str,
        input: &Value,
        output: &Value,
        duration_ms: f64,
    ) {
        let now = Instant::now();
        let start = *self.start.get_or_insert(now);
        let timestamp = now.duration_since(start).as_secs_f64();
        self.entries.push(LogEntry {
            node_name: node_name.to_string(),
            input: input.clone(),
            output: output.clone(),
            timestamp,
            duration_ms,
        });
    }

    /// Return a slice of all log entries.
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Number of recorded entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize the log to a JSON value.
    pub fn to_json(&self) -> Value {
        let entries: Vec<Value> = self
            .entries
            .iter()
            .map(|e| {
                serde_json::json!({
                    "node_name": e.node_name,
                    "input": e.input,
                    "output": e.output,
                    "timestamp": e.timestamp,
                    "duration_ms": e.duration_ms,
                })
            })
            .collect();
        Value::Array(entries)
    }
}

impl Default for ExecutionLog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// CompiledGraph
// ---------------------------------------------------------------------------

/// An executable compiled graph.
///
/// Nodes are executed sequentially from the entry point, following edges until a
/// finish point is reached. State is threaded through nodes by deep-merging each
/// node's output into the running state.
pub struct CompiledGraph {
    nodes: HashMap<String, CompiledNode>,
    edges: Vec<CompiledEdge>,
    entry_point: String,
    finish_points: Vec<String>,
}

impl CompiledGraph {
    /// Create a new compiled graph with the given entry point.
    pub fn new(entry_point: impl Into<String>) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            entry_point: entry_point.into(),
            finish_points: Vec::new(),
        }
    }

    /// Add a compiled node to the graph.
    pub fn add_node(&mut self, node: CompiledNode) {
        self.nodes.insert(node.name.clone(), node);
    }

    /// Add a direct edge between two nodes.
    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>) {
        self.edges.push(CompiledEdge::Direct {
            from: from.into(),
            to: to.into(),
        });
    }

    /// Add a conditional edge with a router function and possible targets.
    pub fn add_conditional_edge<F>(
        &mut self,
        from: impl Into<String>,
        router: F,
        targets: Vec<String>,
    ) where
        F: Fn(&Value) -> String + Send + Sync + 'static,
    {
        self.edges.push(CompiledEdge::Conditional {
            from: from.into(),
            router: Box::new(router),
            targets,
        });
    }

    /// Mark a node as a finish point (execution stops when it completes).
    pub fn set_finish_point(&mut self, name: impl Into<String>) {
        let name = name.into();
        if !self.finish_points.contains(&name) {
            self.finish_points.push(name);
        }
    }

    /// Return the names of all nodes in the graph.
    pub fn node_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.nodes.keys().map(|s| s.as_str()).collect();
        names.sort();
        names
    }

    /// Return the total number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Return a reference to the entry point name.
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }

    /// Return the finish points.
    pub fn finish_points(&self) -> &[String] {
        &self.finish_points
    }

    /// Validate the graph structure.
    ///
    /// Checks:
    /// - Entry point node exists.
    /// - All edge targets reference existing nodes or finish points.
    /// - All finish points reference existing nodes.
    /// - At least one finish point is reachable from the entry point.
    pub fn validate(&self) -> Result<(), LangGraphError> {
        // Check entry point exists
        if !self.nodes.contains_key(&self.entry_point) {
            return Err(LangGraphError::Other(format!(
                "Entry point '{}' does not exist in the graph",
                self.entry_point
            )));
        }

        // Check all edge targets exist
        for edge in &self.edges {
            match edge {
                CompiledEdge::Direct { from, to } => {
                    if !self.nodes.contains_key(from) {
                        return Err(LangGraphError::Other(format!(
                            "Edge source '{}' does not exist in the graph",
                            from
                        )));
                    }
                    if !self.nodes.contains_key(to) && !self.finish_points.contains(to) {
                        return Err(LangGraphError::Other(format!(
                            "Edge target '{}' does not exist in the graph",
                            to
                        )));
                    }
                }
                CompiledEdge::Conditional { from, targets, .. } => {
                    if !self.nodes.contains_key(from) {
                        return Err(LangGraphError::Other(format!(
                            "Edge source '{}' does not exist in the graph",
                            from
                        )));
                    }
                    for target in targets {
                        if !self.nodes.contains_key(target) && !self.finish_points.contains(target)
                        {
                            return Err(LangGraphError::Other(format!(
                                "Conditional edge target '{}' does not exist in the graph",
                                target
                            )));
                        }
                    }
                }
            }
        }

        // Check finish points reference existing nodes
        for fp in &self.finish_points {
            if !self.nodes.contains_key(fp) {
                return Err(LangGraphError::Other(format!(
                    "Finish point '{}' does not exist in the graph",
                    fp
                )));
            }
        }

        // Check at least one finish point is reachable from entry
        if !self.finish_points.is_empty() {
            let reachable = self.reachable_from(&self.entry_point);
            let any_reachable = self
                .finish_points
                .iter()
                .any(|fp| reachable.contains(fp.as_str()));
            if !any_reachable {
                return Err(LangGraphError::Other(
                    "No finish point is reachable from the entry point".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Detect if the graph contains cycles reachable from the entry point.
    pub fn has_cycle(&self) -> bool {
        let mut visited = HashSet::new();
        let mut stack = HashSet::new();
        self.dfs_cycle(&self.entry_point, &mut visited, &mut stack)
    }

    fn dfs_cycle(
        &self,
        node: &str,
        visited: &mut HashSet<String>,
        stack: &mut HashSet<String>,
    ) -> bool {
        if stack.contains(node) {
            return true;
        }
        if visited.contains(node) {
            return false;
        }
        visited.insert(node.to_string());
        stack.insert(node.to_string());

        for edge in &self.edges {
            match edge {
                CompiledEdge::Direct { from, to } if from == node => {
                    if self.nodes.contains_key(to) && self.dfs_cycle(to, visited, stack) {
                        return true;
                    }
                }
                CompiledEdge::Conditional { from, targets, .. } if from == node => {
                    for target in targets {
                        if self.nodes.contains_key(target) && self.dfs_cycle(target, visited, stack)
                        {
                            return true;
                        }
                    }
                }
                _ => {}
            }
        }

        stack.remove(node);
        false
    }

    /// Return the set of node names reachable from `start`.
    fn reachable_from(&self, start: &str) -> HashSet<String> {
        let mut reachable = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(start.to_string());

        while let Some(current) = queue.pop_front() {
            if !reachable.insert(current.clone()) {
                continue;
            }
            for edge in &self.edges {
                match edge {
                    CompiledEdge::Direct { from, to } if from == &current => {
                        if !reachable.contains(to) {
                            queue.push_back(to.clone());
                        }
                    }
                    CompiledEdge::Conditional { from, targets, .. } if from == &current => {
                        for target in targets {
                            if !reachable.contains(target) {
                                queue.push_back(target.clone());
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        reachable
    }

    /// Find the next node(s) to execute after `current_node` given the current `state`.
    fn next_node(&self, current_node: &str, state: &Value) -> Option<String> {
        for edge in &self.edges {
            match edge {
                CompiledEdge::Direct { from, to } if from == current_node => {
                    return Some(to.clone());
                }
                CompiledEdge::Conditional {
                    from,
                    router,
                    targets,
                } if from == current_node => {
                    let target = router(state);
                    if targets.contains(&target) {
                        return Some(target);
                    } else {
                        return None;
                    }
                }
                _ => {}
            }
        }
        None
    }

    /// Execute the graph from entry to finish, threading state through each node.
    ///
    /// State is deep-merged: each node receives the accumulated state and its
    /// output is merged on top. Execution stops when a finish point is reached
    /// or there are no more edges to follow.
    pub fn invoke(&self, input: Value) -> Result<Value, LangGraphError> {
        self.invoke_with_log(input).map(|(state, _)| state)
    }

    /// Execute the graph and return both the final state and the execution log.
    pub fn invoke_with_log(&self, input: Value) -> Result<(Value, ExecutionLog), LangGraphError> {
        let mut state = input;
        let mut current = self.entry_point.clone();
        let mut log = ExecutionLog::new();
        let mut steps = 0;
        const MAX_STEPS: usize = 1000;

        loop {
            if steps >= MAX_STEPS {
                return Err(LangGraphError::GraphRecursionError(format!(
                    "Exceeded maximum step limit of {}",
                    MAX_STEPS
                )));
            }
            steps += 1;

            let node = self.nodes.get(&current).ok_or_else(|| {
                LangGraphError::Other(format!("Node '{}' not found during execution", current))
            })?;

            let start = Instant::now();
            let output = node.execute(state.clone())?;
            let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

            log.record_with_duration(&current, &state, &output, duration_ms);

            // Deep-merge node output into running state
            state = deep_merge(&state, &output);

            // Check if we reached a finish point
            if self.finish_points.contains(&current) {
                return Ok((state, log));
            }

            // Find next node
            match self.next_node(&current, &state) {
                Some(next) => {
                    // If the next target is a finish-point name that isn't a real
                    // node, we are done.
                    if !self.nodes.contains_key(&next) {
                        return Ok((state, log));
                    }
                    current = next;
                }
                None => return Ok((state, log)),
            }
        }
    }

    /// Generate a Mermaid diagram representation of the graph.
    pub fn to_mermaid(&self) -> String {
        let mut lines = Vec::new();
        lines.push("graph TD".to_string());

        // Add nodes
        let mut sorted_names: Vec<&str> = self.nodes.keys().map(|s| s.as_str()).collect();
        sorted_names.sort();
        for name in &sorted_names {
            let label = if *name == self.entry_point {
                format!("    {}[\"{}(entry)\"]", name, name)
            } else if self.finish_points.iter().any(|fp| fp == *name) {
                format!("    {}[\"{}(finish)\"]", name, name)
            } else {
                format!("    {}[\"{}\"]", name, name)
            };
            lines.push(label);
        }

        // Add edges
        for edge in &self.edges {
            match edge {
                CompiledEdge::Direct { from, to } => {
                    lines.push(format!("    {} --> {}", from, to));
                }
                CompiledEdge::Conditional { from, targets, .. } => {
                    for target in targets {
                        lines.push(format!("    {} -.-> {}", from, target));
                    }
                }
            }
        }

        lines.join("\n")
    }
}

impl fmt::Debug for CompiledGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompiledGraph")
            .field("entry_point", &self.entry_point)
            .field("finish_points", &self.finish_points)
            .field("node_count", &self.nodes.len())
            .field("edge_count", &self.edges.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Deep merge helper
// ---------------------------------------------------------------------------

/// Deep-merge two JSON values. Object fields in `b` override those in `a`;
/// non-object values in `b` replace those in `a`.
fn deep_merge(a: &Value, b: &Value) -> Value {
    match (a, b) {
        (Value::Object(a_map), Value::Object(b_map)) => {
            let mut merged = a_map.clone();
            for (key, b_val) in b_map {
                let merged_val = if let Some(a_val) = merged.get(key) {
                    deep_merge(a_val, b_val)
                } else {
                    b_val.clone()
                };
                merged.insert(key.clone(), merged_val);
            }
            Value::Object(merged)
        }
        _ => b.clone(),
    }
}

// ---------------------------------------------------------------------------
// GraphDefinitionBuilder
// ---------------------------------------------------------------------------

/// Node handler: takes a `Value` and returns a `Result<Value, LangGraphError>`.
type NodeHandler = Box<dyn Fn(Value) -> Result<Value, LangGraphError> + Send + Sync>;

/// Conditional edge: source node, routing function, and list of possible targets.
type ConditionalEdge = (
    String,
    Box<dyn Fn(&Value) -> String + Send + Sync>,
    Vec<String>,
);

/// A declarative builder for constructing graph definitions that can be compiled.
pub struct GraphDefinitionBuilder {
    name: String,
    nodes: Vec<(String, NodeHandler)>,
    edges: Vec<(String, String)>,
    conditional_edges: Vec<ConditionalEdge>,
    entry_point: Option<String>,
    finish_points: Vec<String>,
}

impl GraphDefinitionBuilder {
    /// Create a new graph definition builder with a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            nodes: Vec::new(),
            edges: Vec::new(),
            conditional_edges: Vec::new(),
            entry_point: None,
            finish_points: Vec::new(),
        }
    }

    /// Add a node with a handler.
    pub fn node<F>(mut self, name: impl Into<String>, handler: F) -> Self
    where
        F: Fn(Value) -> Result<Value, LangGraphError> + Send + Sync + 'static,
    {
        self.nodes.push((name.into(), Box::new(handler)));
        self
    }

    /// Add a direct edge.
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edges.push((from.into(), to.into()));
        self
    }

    /// Add a conditional edge.
    pub fn conditional_edge<F>(
        mut self,
        from: impl Into<String>,
        router: F,
        targets: Vec<String>,
    ) -> Self
    where
        F: Fn(&Value) -> String + Send + Sync + 'static,
    {
        self.conditional_edges
            .push((from.into(), Box::new(router), targets));
        self
    }

    /// Set the entry point.
    pub fn entry(mut self, name: impl Into<String>) -> Self {
        self.entry_point = Some(name.into());
        self
    }

    /// Set a finish point.
    pub fn finish(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        if !self.finish_points.contains(&name) {
            self.finish_points.push(name);
        }
        self
    }

    /// Return the builder's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Build and validate the graph definition, returning itself for compilation.
    pub fn build(self) -> Result<GraphDefinitionBuilder, LangGraphError> {
        if self.entry_point.is_none() {
            return Err(LangGraphError::Other(
                "No entry point specified".to_string(),
            ));
        }
        if self.nodes.is_empty() {
            return Err(LangGraphError::Other("Graph has no nodes".to_string()));
        }
        let node_names: HashSet<&str> = self.nodes.iter().map(|(n, _)| n.as_str()).collect();
        let entry = self.entry_point.as_deref().unwrap();
        if !node_names.contains(entry) {
            return Err(LangGraphError::Other(format!(
                "Entry point '{}' is not a defined node",
                entry
            )));
        }
        for fp in &self.finish_points {
            if !node_names.contains(fp.as_str()) {
                return Err(LangGraphError::Other(format!(
                    "Finish point '{}' is not a defined node",
                    fp
                )));
            }
        }
        Ok(self)
    }
}

impl fmt::Debug for GraphDefinitionBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphDefinitionBuilder")
            .field("name", &self.name)
            .field("node_count", &self.nodes.len())
            .field("edge_count", &self.edges.len())
            .field("entry_point", &self.entry_point)
            .field("finish_points", &self.finish_points)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// GraphCompiler
// ---------------------------------------------------------------------------

/// Compiles a [`GraphDefinitionBuilder`] into a [`CompiledGraph`].
pub struct GraphCompiler;

impl GraphCompiler {
    /// Create a new graph compiler.
    pub fn new() -> Self {
        Self
    }

    /// Compile a graph definition builder into an executable compiled graph.
    pub fn compile(
        &self,
        definition: GraphDefinitionBuilder,
    ) -> Result<CompiledGraph, LangGraphError> {
        let entry_point = definition
            .entry_point
            .ok_or_else(|| LangGraphError::Other("No entry point specified".to_string()))?;

        let mut graph = CompiledGraph::new(entry_point);

        for (name, handler) in definition.nodes {
            graph.add_node(CompiledNode {
                name: name.clone(),
                handler,
            });
        }

        for (from, to) in definition.edges {
            graph.add_edge(from, to);
        }

        for (from, router, targets) in definition.conditional_edges {
            graph.edges.push(CompiledEdge::Conditional {
                from,
                router,
                targets,
            });
        }

        for fp in definition.finish_points {
            graph.set_finish_point(fp);
        }

        graph.validate()?;

        Ok(graph)
    }
}

impl Default for GraphCompiler {
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

    // -- CompiledNode tests --

    #[test]
    fn test_compiled_node_new_and_name() {
        let node = CompiledNode::new("test_node", |state| Ok(state));
        assert_eq!(node.name(), "test_node");
    }

    #[test]
    fn test_compiled_node_execute_passthrough() {
        let node = CompiledNode::new("passthrough", |state| Ok(state));
        let input = json!({"key": "value"});
        let output = node.execute(input.clone()).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn test_compiled_node_execute_transform() {
        let node = CompiledNode::new("transform", |_state| Ok(json!({"transformed": true})));
        let output = node.execute(json!({})).unwrap();
        assert_eq!(output, json!({"transformed": true}));
    }

    #[test]
    fn test_compiled_node_execute_error() {
        let node = CompiledNode::new("error_node", |_state| {
            Err(LangGraphError::Other("test error".to_string()))
        });
        let result = node.execute(json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_compiled_node_debug() {
        let node = CompiledNode::new("debug_node", |s| Ok(s));
        let debug_str = format!("{:?}", node);
        assert!(debug_str.contains("debug_node"));
    }

    // -- CompiledEdge tests --

    #[test]
    fn test_compiled_edge_direct_source() {
        let edge = CompiledEdge::Direct {
            from: "A".to_string(),
            to: "B".to_string(),
        };
        assert_eq!(edge.source(), "A");
        assert!(!edge.is_conditional());
    }

    #[test]
    fn test_compiled_edge_conditional_source() {
        let edge = CompiledEdge::Conditional {
            from: "A".to_string(),
            router: Box::new(|_| "B".to_string()),
            targets: vec!["B".to_string(), "C".to_string()],
        };
        assert_eq!(edge.source(), "A");
        assert!(edge.is_conditional());
    }

    #[test]
    fn test_compiled_edge_debug() {
        let edge = CompiledEdge::Direct {
            from: "X".to_string(),
            to: "Y".to_string(),
        };
        let debug_str = format!("{:?}", edge);
        assert!(debug_str.contains("X"));
        assert!(debug_str.contains("Y"));
    }

    #[test]
    fn test_compiled_edge_conditional_debug() {
        let edge = CompiledEdge::Conditional {
            from: "A".to_string(),
            router: Box::new(|_| "B".to_string()),
            targets: vec!["B".to_string()],
        };
        let debug_str = format!("{:?}", edge);
        assert!(debug_str.contains("Conditional"));
    }

    // -- CompiledGraph construction tests --

    #[test]
    fn test_compiled_graph_new() {
        let graph = CompiledGraph::new("start");
        assert_eq!(graph.entry_point(), "start");
        assert!(graph.node_names().is_empty());
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_compiled_graph_add_node() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        assert_eq!(graph.node_names(), vec!["A"]);
    }

    #[test]
    fn test_compiled_graph_add_edge() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_node(CompiledNode::new("B", |s| Ok(s)));
        graph.add_edge("A", "B");
        assert_eq!(graph.edge_count(), 1);
    }

    #[test]
    fn test_compiled_graph_set_finish_point() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.set_finish_point("A");
        assert_eq!(graph.finish_points(), &["A".to_string()]);
    }

    #[test]
    fn test_compiled_graph_set_finish_point_dedup() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.set_finish_point("A");
        graph.set_finish_point("A");
        assert_eq!(graph.finish_points().len(), 1);
    }

    #[test]
    fn test_compiled_graph_node_names_sorted() {
        let mut graph = CompiledGraph::new("C");
        graph.add_node(CompiledNode::new("C", |s| Ok(s)));
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_node(CompiledNode::new("B", |s| Ok(s)));
        assert_eq!(graph.node_names(), vec!["A", "B", "C"]);
    }

    // -- Validation tests --

    #[test]
    fn test_validate_missing_entry_point() {
        let graph = CompiledGraph::new("nonexistent");
        let result = graph.validate();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("nonexistent"));
    }

    #[test]
    fn test_validate_missing_edge_source() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_edge("nonexistent", "A");
        let result = graph.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_missing_edge_target() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_edge("A", "nonexistent");
        let result = graph.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_missing_conditional_target() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_conditional_edge(
            "A",
            |_| "nonexistent".to_string(),
            vec!["nonexistent".to_string()],
        );
        let result = graph.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_missing_finish_point_node() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.set_finish_point("nonexistent");
        let result = graph.validate();
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_unreachable_finish_point() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_node(CompiledNode::new("B", |s| Ok(s)));
        // No edge from A to B, B is unreachable
        graph.set_finish_point("B");
        let result = graph.validate();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("reachable"));
    }

    #[test]
    fn test_validate_valid_graph() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_node(CompiledNode::new("B", |s| Ok(s)));
        graph.add_edge("A", "B");
        graph.set_finish_point("B");
        let result = graph.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_no_finish_points_ok() {
        // A graph with no explicit finish points is valid (runs until no more edges)
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        assert!(graph.validate().is_ok());
    }

    // -- Cycle detection tests --

    #[test]
    fn test_has_cycle_linear() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_node(CompiledNode::new("B", |s| Ok(s)));
        graph.add_edge("A", "B");
        assert!(!graph.has_cycle());
    }

    #[test]
    fn test_has_cycle_with_cycle() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_node(CompiledNode::new("B", |s| Ok(s)));
        graph.add_edge("A", "B");
        graph.add_edge("B", "A");
        assert!(graph.has_cycle());
    }

    #[test]
    fn test_has_cycle_self_loop() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_edge("A", "A");
        assert!(graph.has_cycle());
    }

    // -- Linear graph execution tests --

    #[test]
    fn test_invoke_single_node() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |_| Ok(json!({"result": "done"}))));
        graph.set_finish_point("A");
        let result = graph.invoke(json!({})).unwrap();
        assert_eq!(result, json!({"result": "done"}));
    }

    #[test]
    fn test_invoke_linear_a_b_c() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |_| Ok(json!({"a": 1}))));
        graph.add_node(CompiledNode::new("B", |_| Ok(json!({"b": 2}))));
        graph.add_node(CompiledNode::new("C", |_| Ok(json!({"c": 3}))));
        graph.add_edge("A", "B");
        graph.add_edge("B", "C");
        graph.set_finish_point("C");

        let result = graph.invoke(json!({})).unwrap();
        assert_eq!(result, json!({"a": 1, "b": 2, "c": 3}));
    }

    #[test]
    fn test_invoke_state_threading() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |_| Ok(json!({"count": 1}))));
        graph.add_node(CompiledNode::new("B", |state: Value| {
            let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(json!({"count": count + 1}))
        }));
        graph.add_edge("A", "B");
        graph.set_finish_point("B");

        let result = graph.invoke(json!({})).unwrap();
        assert_eq!(result["count"], 2);
    }

    #[test]
    fn test_invoke_preserves_input_fields() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |_| Ok(json!({"added": true}))));
        graph.set_finish_point("A");
        let result = graph.invoke(json!({"original": "data"})).unwrap();
        assert_eq!(result["original"], "data");
        assert_eq!(result["added"], true);
    }

    #[test]
    fn test_invoke_stops_at_no_edges() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |_| Ok(json!({"done": true}))));
        // No finish point set, no edges — should stop after A
        let result = graph.invoke(json!({})).unwrap();
        assert_eq!(result["done"], true);
    }

    // -- Conditional graph execution tests --

    #[test]
    fn test_invoke_conditional_edge_branch_a() {
        let mut graph = CompiledGraph::new("router");
        graph.add_node(CompiledNode::new("router", |_| {
            Ok(json!({"route": "left"}))
        }));
        graph.add_node(CompiledNode::new("left", |_| Ok(json!({"path": "left"}))));
        graph.add_node(CompiledNode::new("right", |_| Ok(json!({"path": "right"}))));
        graph.add_conditional_edge(
            "router",
            |state| {
                state
                    .get("route")
                    .and_then(|v| v.as_str())
                    .unwrap_or("left")
                    .to_string()
            },
            vec!["left".to_string(), "right".to_string()],
        );
        graph.set_finish_point("left");
        graph.set_finish_point("right");

        let result = graph.invoke(json!({})).unwrap();
        assert_eq!(result["path"], "left");
    }

    #[test]
    fn test_invoke_conditional_edge_branch_b() {
        let mut graph = CompiledGraph::new("router");
        graph.add_node(CompiledNode::new("router", |_| {
            Ok(json!({"route": "right"}))
        }));
        graph.add_node(CompiledNode::new("left", |_| Ok(json!({"path": "left"}))));
        graph.add_node(CompiledNode::new("right", |_| Ok(json!({"path": "right"}))));
        graph.add_conditional_edge(
            "router",
            |state| {
                state
                    .get("route")
                    .and_then(|v| v.as_str())
                    .unwrap_or("right")
                    .to_string()
            },
            vec!["left".to_string(), "right".to_string()],
        );
        graph.set_finish_point("left");
        graph.set_finish_point("right");

        let result = graph.invoke(json!({})).unwrap();
        assert_eq!(result["path"], "right");
    }

    #[test]
    fn test_invoke_conditional_unknown_target_stops() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |_| Ok(json!({"done": true}))));
        graph.add_conditional_edge(
            "A",
            |_| "unknown".to_string(),
            vec!["B".to_string()], // router returns "unknown" which is not in targets
        );
        // Should stop because router returned a target not in the list
        let result = graph.invoke(json!({})).unwrap();
        assert_eq!(result["done"], true);
    }

    // -- Multiple finish points --

    #[test]
    fn test_multiple_finish_points() {
        let mut graph = CompiledGraph::new("start");
        graph.add_node(CompiledNode::new("start", |_| {
            Ok(json!({"route": "end_a"}))
        }));
        graph.add_node(CompiledNode::new("end_a", |_| Ok(json!({"result": "A"}))));
        graph.add_node(CompiledNode::new("end_b", |_| Ok(json!({"result": "B"}))));
        graph.add_conditional_edge(
            "start",
            |state| {
                state
                    .get("route")
                    .and_then(|v| v.as_str())
                    .unwrap_or("end_a")
                    .to_string()
            },
            vec!["end_a".to_string(), "end_b".to_string()],
        );
        graph.set_finish_point("end_a");
        graph.set_finish_point("end_b");

        let result = graph.invoke(json!({})).unwrap();
        assert_eq!(result["result"], "A");
    }

    // -- invoke_with_log tests --

    #[test]
    fn test_invoke_with_log() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |_| Ok(json!({"a": 1}))));
        graph.add_node(CompiledNode::new("B", |_| Ok(json!({"b": 2}))));
        graph.add_edge("A", "B");
        graph.set_finish_point("B");

        let (state, log) = graph.invoke_with_log(json!({})).unwrap();
        assert_eq!(state, json!({"a": 1, "b": 2}));
        assert_eq!(log.len(), 2);
        assert_eq!(log.entries()[0].node_name, "A");
        assert_eq!(log.entries()[1].node_name, "B");
    }

    #[test]
    fn test_invoke_error_propagation() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |_| {
            Err(LangGraphError::Other("boom".to_string()))
        }));
        graph.set_finish_point("A");
        let result = graph.invoke(json!({}));
        assert!(result.is_err());
    }

    // -- Mermaid diagram tests --

    #[test]
    fn test_to_mermaid_basic() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_node(CompiledNode::new("B", |s| Ok(s)));
        graph.add_edge("A", "B");
        graph.set_finish_point("B");

        let mermaid = graph.to_mermaid();
        assert!(mermaid.contains("graph TD"));
        assert!(mermaid.contains("A --> B"));
        assert!(mermaid.contains("entry"));
        assert!(mermaid.contains("finish"));
    }

    #[test]
    fn test_to_mermaid_conditional() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_node(CompiledNode::new("B", |s| Ok(s)));
        graph.add_node(CompiledNode::new("C", |s| Ok(s)));
        graph.add_conditional_edge(
            "A",
            |_| "B".to_string(),
            vec!["B".to_string(), "C".to_string()],
        );

        let mermaid = graph.to_mermaid();
        assert!(mermaid.contains("A -.-> B"));
        assert!(mermaid.contains("A -.-> C"));
    }

    #[test]
    fn test_to_mermaid_no_edges() {
        let mut graph = CompiledGraph::new("solo");
        graph.add_node(CompiledNode::new("solo", |s| Ok(s)));
        let mermaid = graph.to_mermaid();
        assert!(mermaid.contains("graph TD"));
        assert!(mermaid.contains("solo"));
    }

    // -- GraphDefinitionBuilder tests --

    #[test]
    fn test_definition_builder_name() {
        let builder = GraphDefinitionBuilder::new("my_graph");
        assert_eq!(builder.name(), "my_graph");
    }

    #[test]
    fn test_definition_builder_build_no_entry() {
        let builder = GraphDefinitionBuilder::new("test").node("A", |s| Ok(s));
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_definition_builder_build_no_nodes() {
        let builder = GraphDefinitionBuilder::new("test").entry("A");
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_definition_builder_build_entry_not_a_node() {
        let builder = GraphDefinitionBuilder::new("test")
            .node("B", |s| Ok(s))
            .entry("A");
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_definition_builder_build_finish_not_a_node() {
        let builder = GraphDefinitionBuilder::new("test")
            .node("A", |s| Ok(s))
            .entry("A")
            .finish("nonexistent");
        let result = builder.build();
        assert!(result.is_err());
    }

    #[test]
    fn test_definition_builder_build_valid() {
        let builder = GraphDefinitionBuilder::new("test")
            .node("A", |s| Ok(s))
            .node("B", |s| Ok(s))
            .edge("A", "B")
            .entry("A")
            .finish("B");
        let result = builder.build();
        assert!(result.is_ok());
    }

    #[test]
    fn test_definition_builder_debug() {
        let builder = GraphDefinitionBuilder::new("dbg_test")
            .node("A", |s| Ok(s))
            .entry("A");
        let debug_str = format!("{:?}", builder);
        assert!(debug_str.contains("dbg_test"));
    }

    // -- GraphCompiler tests --

    #[test]
    fn test_compiler_compile_linear() {
        let compiler = GraphCompiler::new();
        let def = GraphDefinitionBuilder::new("linear")
            .node("A", |_| Ok(json!({"a": 1})))
            .node("B", |_| Ok(json!({"b": 2})))
            .edge("A", "B")
            .entry("A")
            .finish("B")
            .build()
            .unwrap();

        let graph = compiler.compile(def).unwrap();
        assert_eq!(graph.entry_point(), "A");
        assert_eq!(graph.node_names(), vec!["A", "B"]);
        assert_eq!(graph.edge_count(), 1);

        let result = graph.invoke(json!({})).unwrap();
        assert_eq!(result, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn test_compiler_compile_conditional() {
        let compiler = GraphCompiler::new();
        let def = GraphDefinitionBuilder::new("cond")
            .node("start", |_| Ok(json!({"choice": "yes"})))
            .node("yes_node", |_| Ok(json!({"answer": "yes"})))
            .node("no_node", |_| Ok(json!({"answer": "no"})))
            .conditional_edge(
                "start",
                |state| {
                    state
                        .get("choice")
                        .and_then(|v| v.as_str())
                        .unwrap_or("no")
                        .to_string()
                        + "_node"
                },
                vec!["yes_node".to_string(), "no_node".to_string()],
            )
            .entry("start")
            .finish("yes_node")
            .finish("no_node")
            .build()
            .unwrap();

        let graph = compiler.compile(def).unwrap();
        let result = graph.invoke(json!({})).unwrap();
        assert_eq!(result["answer"], "yes");
    }

    #[test]
    fn test_compiler_default() {
        let compiler = GraphCompiler::default();
        let def = GraphDefinitionBuilder::new("t")
            .node("A", |s| Ok(s))
            .entry("A")
            .build()
            .unwrap();
        assert!(compiler.compile(def).is_ok());
    }

    // -- ExecutionLog tests --

    #[test]
    fn test_execution_log_new_empty() {
        let log = ExecutionLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert!(log.entries().is_empty());
    }

    #[test]
    fn test_execution_log_record() {
        let mut log = ExecutionLog::new();
        let input = json!({"in": 1});
        let output = json!({"out": 2});
        log.record("node_a", &input, &output);
        assert_eq!(log.len(), 1);
        assert_eq!(log.entries()[0].node_name, "node_a");
        assert_eq!(log.entries()[0].input, input);
        assert_eq!(log.entries()[0].output, output);
    }

    #[test]
    fn test_execution_log_record_multiple() {
        let mut log = ExecutionLog::new();
        log.record("A", &json!({}), &json!({"a": 1}));
        log.record("B", &json!({"a": 1}), &json!({"b": 2}));
        log.record("C", &json!({"a": 1, "b": 2}), &json!({"c": 3}));
        assert_eq!(log.len(), 3);
    }

    #[test]
    fn test_execution_log_to_json() {
        let mut log = ExecutionLog::new();
        log.record("X", &json!({"k": "v"}), &json!({"r": 42}));
        let json_val = log.to_json();
        assert!(json_val.is_array());
        let arr = json_val.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["node_name"], "X");
    }

    #[test]
    fn test_execution_log_default() {
        let log = ExecutionLog::default();
        assert!(log.is_empty());
    }

    #[test]
    fn test_execution_log_with_duration() {
        let mut log = ExecutionLog::new();
        log.record_with_duration("fast", &json!({}), &json!({}), 1.5);
        assert_eq!(log.entries()[0].duration_ms, 1.5);
    }

    // -- Deep merge tests --

    #[test]
    fn test_deep_merge_objects() {
        let a = json!({"x": 1, "y": 2});
        let b = json!({"y": 3, "z": 4});
        let merged = deep_merge(&a, &b);
        assert_eq!(merged, json!({"x": 1, "y": 3, "z": 4}));
    }

    #[test]
    fn test_deep_merge_nested() {
        let a = json!({"nested": {"a": 1, "b": 2}});
        let b = json!({"nested": {"b": 3, "c": 4}});
        let merged = deep_merge(&a, &b);
        assert_eq!(merged, json!({"nested": {"a": 1, "b": 3, "c": 4}}));
    }

    #[test]
    fn test_deep_merge_non_object() {
        let a = json!(42);
        let b = json!("hello");
        let merged = deep_merge(&a, &b);
        assert_eq!(merged, json!("hello"));
    }

    // -- Reachability tests --

    #[test]
    fn test_reachable_from_linear() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_node(CompiledNode::new("B", |s| Ok(s)));
        graph.add_node(CompiledNode::new("C", |s| Ok(s)));
        graph.add_edge("A", "B");
        graph.add_edge("B", "C");

        let reachable = graph.reachable_from("A");
        assert!(reachable.contains("A"));
        assert!(reachable.contains("B"));
        assert!(reachable.contains("C"));
    }

    #[test]
    fn test_reachable_from_disconnected() {
        let mut graph = CompiledGraph::new("A");
        graph.add_node(CompiledNode::new("A", |s| Ok(s)));
        graph.add_node(CompiledNode::new("B", |s| Ok(s)));
        // No edge from A to B

        let reachable = graph.reachable_from("A");
        assert!(reachable.contains("A"));
        assert!(!reachable.contains("B"));
    }

    // -- CompiledGraph debug --

    #[test]
    fn test_compiled_graph_debug() {
        let mut graph = CompiledGraph::new("entry");
        graph.add_node(CompiledNode::new("entry", |s| Ok(s)));
        let debug_str = format!("{:?}", graph);
        assert!(debug_str.contains("entry"));
        assert!(debug_str.contains("CompiledGraph"));
    }
}
