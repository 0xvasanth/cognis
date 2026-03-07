//! Subgraph composition support.
//!
//! This module provides types for embedding compiled graphs as nodes within
//! parent graphs, with state mapping between parent and child scopes.
//!
//! Key types:
//! - [`StateMapping`] — maps fields between parent and child state
//! - [`SubgraphConfig`] — configuration for a subgraph node (name, namespace, mappings)
//! - [`SubgraphNode`] — wraps a compiled subgraph (or arbitrary function) as a graph node
//! - [`SubgraphRegistry`] — stores named subgraphs for reuse
//! - [`SubgraphBuilder`] — fluent builder for creating `SubgraphNode` instances
//! - [`NestedSubgraph`] — supports multiple levels of nesting with depth tracking

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::errors::{LangGraphError, Result};

use super::state::{AsyncNodeAction, CompiledStateGraph};

// ---------------------------------------------------------------------------
// StateMapping
// ---------------------------------------------------------------------------

/// Maps fields between parent and child state scopes.
///
/// Each mapping entry is a `(parent_key, child_key)` pair. During input
/// mapping the value at `parent_key` is placed at `child_key` in the child
/// state. During output mapping the value at `child_key` in the child output
/// is written back to `parent_key` in the parent state.
#[derive(Debug, Clone, Default)]
pub struct StateMapping {
    /// parent_key -> child_key
    mappings: HashMap<String, String>,
}

impl StateMapping {
    /// Create an empty `StateMapping`.
    pub fn new() -> Self {
        Self {
            mappings: HashMap::new(),
        }
    }

    /// Add a mapping from a parent field to a child field.
    pub fn map(mut self, parent_key: &str, child_key: &str) -> Self {
        self.mappings
            .insert(parent_key.to_string(), child_key.to_string());
        self
    }

    /// Map a set of keys to themselves (identity mapping).
    pub fn identity(mut self, keys: &[&str]) -> Self {
        for key in keys {
            self.mappings.insert(key.to_string(), key.to_string());
        }
        self
    }

    /// Returns `true` if no mappings are configured.
    pub fn is_empty(&self) -> bool {
        self.mappings.is_empty()
    }

    /// Returns the number of mappings.
    pub fn len(&self) -> usize {
        self.mappings.len()
    }

    /// Extract mapped fields from `parent_state` into a child state `Value`.
    ///
    /// For each `(parent_key, child_key)` pair, the value at `parent_key` in
    /// `parent_state` is placed at `child_key` in the returned object. Keys
    /// not present in `parent_state` are silently skipped.
    pub fn apply_input(&self, parent_state: &Value) -> Value {
        if self.mappings.is_empty() {
            return parent_state.clone();
        }
        let mut result = serde_json::Map::new();
        if let Value::Object(obj) = parent_state {
            for (parent_key, child_key) in &self.mappings {
                if let Some(value) = obj.get(parent_key) {
                    result.insert(child_key.clone(), value.clone());
                }
            }
        }
        Value::Object(result)
    }

    /// Merge child output fields back into `parent_state`.
    ///
    /// For each `(parent_key, child_key)` pair, the value at `child_key` in
    /// `child_state` is written to `parent_key` in `parent_state`. Keys not
    /// present in `child_state` are silently skipped.
    pub fn apply_output(&self, child_state: &Value, parent_state: &mut Value) {
        if self.mappings.is_empty() {
            // Pass-through: merge all child keys into parent.
            if let (Value::Object(parent), Value::Object(child)) = (parent_state, child_state) {
                for (k, v) in child {
                    parent.insert(k.clone(), v.clone());
                }
            }
            return;
        }
        if let (Value::Object(parent), Value::Object(child)) = (parent_state, child_state) {
            for (parent_key, child_key) in &self.mappings {
                if let Some(value) = child.get(child_key) {
                    parent.insert(parent_key.clone(), value.clone());
                }
            }
        }
    }

    /// Return the underlying mappings.
    pub fn mappings(&self) -> &HashMap<String, String> {
        &self.mappings
    }
}

// ---------------------------------------------------------------------------
// SubgraphConfig
// ---------------------------------------------------------------------------

/// Configuration for a subgraph node.
#[derive(Debug, Clone)]
pub struct SubgraphConfig {
    /// Human-readable name for this subgraph.
    pub name: String,
    /// Input mapping from parent state to child state.
    pub input_mapping: StateMapping,
    /// Output mapping from child state back to parent state.
    pub output_mapping: StateMapping,
    /// Optional namespace to isolate state keys.
    pub namespace: Option<String>,
}

impl SubgraphConfig {
    /// Create a new `SubgraphConfig` with the given name and no mappings.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            input_mapping: StateMapping::new(),
            output_mapping: StateMapping::new(),
            namespace: None,
        }
    }

    /// Set the name.
    pub fn with_name(mut self, name: &str) -> Self {
        self.name = name.to_string();
        self
    }

    /// Set the input mapping.
    pub fn with_input_mapping(mut self, mapping: StateMapping) -> Self {
        self.input_mapping = mapping;
        self
    }

    /// Set the output mapping.
    pub fn with_output_mapping(mut self, mapping: StateMapping) -> Self {
        self.output_mapping = mapping;
        self
    }

    /// Set the namespace for state key isolation.
    pub fn with_namespace(mut self, namespace: &str) -> Self {
        self.namespace = Some(namespace.to_string());
        self
    }
}

// ---------------------------------------------------------------------------
// SubgraphNode
// ---------------------------------------------------------------------------

/// Type alias for a synchronous graph function used by `SubgraphNode`.
pub type GraphFn = Arc<dyn Fn(Value) -> Result<Value> + Send + Sync>;

/// Internal representation: either a compiled graph (async) or a plain
/// synchronous function.
enum GraphImpl {
    /// Wraps a `CompiledStateGraph` and invokes it asynchronously.
    Compiled(Arc<CompiledStateGraph>),
    /// Wraps a synchronous function (useful for testing or simple transforms).
    SyncFn(GraphFn),
}

impl Clone for GraphImpl {
    fn clone(&self) -> Self {
        match self {
            Self::Compiled(g) => Self::Compiled(g.clone()),
            Self::SyncFn(f) => Self::SyncFn(f.clone()),
        }
    }
}

/// A wrapper that allows a compiled graph (or any compatible function) to be
/// used as a node action inside a parent [`StateGraph`].
///
/// When executed, the `SubgraphNode`:
/// 1. Extracts keys from the parent state via `input_mapping`
/// 2. Optionally applies namespace prefixing
/// 3. Invokes the inner graph function with the mapped state
/// 4. Maps output keys back via `output_mapping`
/// 5. Returns only the mapped output, ensuring state isolation
///
/// # Example
///
/// ```rust
/// use std::collections::HashMap;
/// use std::sync::Arc;
/// use serde_json::{json, Value};
/// use langgraph::graph::state::{StateGraph, AsyncNodeAction};
/// use langgraph::graph::subgraph::SubgraphNode;
/// use langgraph::errors::LangGraphError;
///
/// let action: AsyncNodeAction = Arc::new(|_s: Value| {
///     Box::pin(async { Ok(json!({"inner_done": true})) })
/// });
///
/// let inner = StateGraph::new()
///     .add_node("inner_a", action)
///     .set_entry_point("inner_a")
///     .set_finish_point("inner_a")
///     .compile()
///     .unwrap();
///
/// let outer = StateGraph::new()
///     .add_subgraph("my_subgraph", inner)
///     .set_entry_point("my_subgraph")
///     .set_finish_point("my_subgraph")
///     .compile()
///     .unwrap();
/// ```
pub struct SubgraphNode {
    /// The underlying graph implementation.
    graph_impl: GraphImpl,
    /// Configuration (name, mappings, namespace).
    config: SubgraphConfig,
}

impl SubgraphNode {
    /// Create a new `SubgraphNode` wrapping the given compiled graph.
    ///
    /// Without mappings the subgraph receives the full parent state and its
    /// full output is merged back (pass-through mode).
    pub fn new(graph: CompiledStateGraph) -> Self {
        Self {
            graph_impl: GraphImpl::Compiled(Arc::new(graph)),
            config: SubgraphConfig::new("subgraph"),
        }
    }

    /// Create a new `SubgraphNode` from a name, a synchronous graph function,
    /// and a configuration.
    pub fn from_fn(name: &str, graph_fn: GraphFn, config: SubgraphConfig) -> Self {
        Self {
            graph_impl: GraphImpl::SyncFn(graph_fn),
            config: SubgraphConfig {
                name: name.to_string(),
                ..config
            },
        }
    }

    /// Create a new `SubgraphNode` with explicit input and output state mappings.
    pub fn with_mapping(
        graph: CompiledStateGraph,
        input_mapping: HashMap<String, String>,
        output_mapping: HashMap<String, String>,
    ) -> Self {
        let mut node = Self::new(graph);
        // Convert HashMap-based mappings to StateMapping.
        let mut input = StateMapping::new();
        for (pk, ck) in input_mapping {
            input.mappings.insert(pk, ck);
        }
        let mut output = StateMapping::new();
        for (ck, pk) in output_mapping {
            output.mappings.insert(pk, ck);
        }
        node.config.input_mapping = input;
        node.config.output_mapping = output;
        node
    }

    /// Execute the subgraph synchronously, applying input/output mappings.
    ///
    /// This works only for `SubgraphNode` instances created via [`from_fn`].
    /// For compiled-graph-backed nodes, use [`into_action`] and invoke the
    /// resulting action within an async context instead.
    ///
    /// 1. `apply_input` extracts fields from `parent_state` into child input.
    /// 2. The inner graph function is invoked with the child input.
    /// 3. `apply_output` merges child results back into a copy of `parent_state`.
    /// 4. Returns the updated parent state.
    pub fn execute(&self, parent_state: &Value) -> Result<Value> {
        let child_input = self.build_child_input(parent_state);

        // Run the subgraph (sync path only).
        let child_output = match &self.graph_impl {
            GraphImpl::SyncFn(f) => (f)(child_input)?,
            GraphImpl::Compiled(_) => {
                return Err(LangGraphError::Other(
                    "Cannot call execute() synchronously on a compiled graph; \
                     use into_action() and invoke asynchronously"
                        .to_string(),
                ));
            }
        };

        self.build_parent_output(parent_state, &child_output)
    }

    /// Return the subgraph configuration.
    pub fn config(&self) -> &SubgraphConfig {
        &self.config
    }

    /// Return the name of this subgraph.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Convert this `SubgraphNode` into an [`AsyncNodeAction`] that can be
    /// added to a [`StateGraph`].
    pub fn into_action(self) -> AsyncNodeAction {
        let graph_impl = self.graph_impl;
        let config = self.config;

        Arc::new(move |state: Value| {
            let graph_impl = graph_impl.clone();
            let input_mapping = config.input_mapping.clone();
            let output_mapping = config.output_mapping.clone();

            Box::pin(async move {
                // Build subgraph input from parent state.
                let subgraph_input = if input_mapping.is_empty() {
                    state
                } else {
                    input_mapping.apply_input(&state)
                };

                // Invoke the subgraph.
                let subgraph_output = match &graph_impl {
                    GraphImpl::Compiled(g) => g.invoke(subgraph_input).await?,
                    GraphImpl::SyncFn(f) => (f)(subgraph_input)?,
                };

                // Build parent output from subgraph output.
                if output_mapping.is_empty() {
                    Ok(subgraph_output)
                } else {
                    let mut result = serde_json::Map::new();
                    for (parent_key, child_key) in output_mapping.mappings() {
                        if let Value::Object(obj) = &subgraph_output {
                            if let Some(v) = obj.get(child_key) {
                                result.insert(parent_key.clone(), v.clone());
                            }
                        }
                    }
                    Ok(Value::Object(result))
                }
            })
        })
    }

    // -- private helpers --

    fn build_child_input(&self, parent_state: &Value) -> Value {
        if self.config.namespace.is_some() {
            let namespaced = self.apply_namespace_input(parent_state);
            self.config.input_mapping.apply_input(&namespaced)
        } else {
            self.config.input_mapping.apply_input(parent_state)
        }
    }

    fn build_parent_output(&self, parent_state: &Value, child_output: &Value) -> Result<Value> {
        let mut result = parent_state.clone();
        if self.config.namespace.is_some() {
            let mapped = self.apply_namespace_output(child_output);
            self.config
                .output_mapping
                .apply_output(&mapped, &mut result);
        } else {
            self.config
                .output_mapping
                .apply_output(child_output, &mut result);
        }
        Ok(result)
    }

    fn apply_namespace_input(&self, state: &Value) -> Value {
        if let (Some(ns), Value::Object(obj)) = (&self.config.namespace, state) {
            let prefix = format!("{}.", ns);
            let mut result = serde_json::Map::new();
            for (k, v) in obj {
                if let Some(stripped) = k.strip_prefix(&prefix) {
                    result.insert(stripped.to_string(), v.clone());
                } else {
                    result.insert(k.clone(), v.clone());
                }
            }
            Value::Object(result)
        } else {
            state.clone()
        }
    }

    fn apply_namespace_output(&self, state: &Value) -> Value {
        if let (Some(ns), Value::Object(obj)) = (&self.config.namespace, state) {
            let prefix = format!("{}.", ns);
            let mut result = serde_json::Map::new();
            for (k, v) in obj {
                result.insert(format!("{}{}", prefix, k), v.clone());
            }
            Value::Object(result)
        } else {
            state.clone()
        }
    }
}

// ---------------------------------------------------------------------------
// SubgraphRegistry
// ---------------------------------------------------------------------------

/// Stores named `SubgraphNode` instances for reuse across multiple graphs.
pub struct SubgraphRegistry {
    subgraphs: HashMap<String, SubgraphNode>,
}

impl SubgraphRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            subgraphs: HashMap::new(),
        }
    }

    /// Register a subgraph under the given name.
    ///
    /// If a subgraph with the same name already exists it is replaced.
    pub fn register(&mut self, name: &str, node: SubgraphNode) {
        self.subgraphs.insert(name.to_string(), node);
    }

    /// Get a reference to a registered subgraph by name.
    pub fn get(&self, name: &str) -> Option<&SubgraphNode> {
        self.subgraphs.get(name)
    }

    /// Remove a subgraph by name, returning it if it existed.
    pub fn remove(&mut self, name: &str) -> Option<SubgraphNode> {
        self.subgraphs.remove(name)
    }

    /// List all registered subgraph names.
    pub fn list(&self) -> Vec<&str> {
        self.subgraphs.keys().map(|s| s.as_str()).collect()
    }

    /// Return the number of registered subgraphs.
    pub fn len(&self) -> usize {
        self.subgraphs.len()
    }

    /// Return `true` if the registry contains no subgraphs.
    pub fn is_empty(&self) -> bool {
        self.subgraphs.is_empty()
    }
}

impl Default for SubgraphRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SubgraphBuilder
// ---------------------------------------------------------------------------

/// Fluent builder for creating [`SubgraphNode`] instances.
///
/// # Example
///
/// ```ignore
/// let node = SubgraphBuilder::new("summarize")
///     .graph_fn(my_fn)
///     .input_mapping(StateMapping::new().map("text", "input_text"))
///     .output_mapping(StateMapping::new().map("summary", "result"))
///     .namespace("summarizer")
///     .build()
///     .unwrap();
/// ```
pub struct SubgraphBuilder {
    name: String,
    graph_fn: Option<GraphFn>,
    input_mapping: StateMapping,
    output_mapping: StateMapping,
    namespace: Option<String>,
}

impl SubgraphBuilder {
    /// Start building a subgraph node with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            graph_fn: None,
            input_mapping: StateMapping::new(),
            output_mapping: StateMapping::new(),
            namespace: None,
        }
    }

    /// Set the graph function.
    pub fn graph_fn(mut self, f: GraphFn) -> Self {
        self.graph_fn = Some(f);
        self
    }

    /// Set the input mapping.
    pub fn input_mapping(mut self, mapping: StateMapping) -> Self {
        self.input_mapping = mapping;
        self
    }

    /// Set the output mapping.
    pub fn output_mapping(mut self, mapping: StateMapping) -> Self {
        self.output_mapping = mapping;
        self
    }

    /// Set the namespace.
    pub fn namespace(mut self, ns: &str) -> Self {
        self.namespace = Some(ns.to_string());
        self
    }

    /// Build the `SubgraphNode`.
    ///
    /// Returns an error if no graph function was provided.
    pub fn build(self) -> Result<SubgraphNode> {
        let graph_fn = self.graph_fn.ok_or_else(|| {
            LangGraphError::Other("SubgraphBuilder: graph_fn is required".to_string())
        })?;
        let config = SubgraphConfig {
            name: self.name.clone(),
            input_mapping: self.input_mapping,
            output_mapping: self.output_mapping,
            namespace: self.namespace,
        };
        Ok(SubgraphNode {
            graph_impl: GraphImpl::SyncFn(graph_fn),
            config,
        })
    }
}

// ---------------------------------------------------------------------------
// NestedSubgraph
// ---------------------------------------------------------------------------

/// Supports multiple levels of subgraph nesting with depth tracking.
///
/// A `NestedSubgraph` wraps a `SubgraphNode` and optionally contains children,
/// enabling tree-structured compositions where a subgraph can itself contain
/// further subgraphs.
pub struct NestedSubgraph {
    /// The subgraph node at this level.
    node: SubgraphNode,
    /// Child subgraphs nested under this one.
    children: Vec<NestedSubgraph>,
}

impl NestedSubgraph {
    /// Create a new `NestedSubgraph` with no children.
    pub fn new(node: SubgraphNode) -> Self {
        Self {
            node,
            children: Vec::new(),
        }
    }

    /// Add a child nested subgraph.
    pub fn add_child(mut self, child: NestedSubgraph) -> Self {
        self.children.push(child);
        self
    }

    /// Return the depth of the deepest nesting path.
    ///
    /// A leaf node has depth 1. A node with children has depth
    /// `1 + max(child depths)`.
    pub fn depth(&self) -> usize {
        if self.children.is_empty() {
            1
        } else {
            1 + self.children.iter().map(|c| c.depth()).max().unwrap_or(0)
        }
    }

    /// Flatten the nested structure into a list of `(depth, name)` pairs
    /// in depth-first order.
    pub fn flatten(&self) -> Vec<(usize, String)> {
        let mut result = Vec::new();
        self.flatten_recursive(1, &mut result);
        result
    }

    fn flatten_recursive(&self, current_depth: usize, out: &mut Vec<(usize, String)>) {
        out.push((current_depth, self.node.name().to_string()));
        for child in &self.children {
            child.flatten_recursive(current_depth + 1, out);
        }
    }

    /// Execute this nested subgraph and all children in depth-first order.
    ///
    /// Children are executed after the parent node, and each child's output
    /// is folded into the running state.
    pub fn execute(&self, parent_state: &Value) -> Result<Value> {
        let mut state = self.node.execute(parent_state)?;
        for child in &self.children {
            state = child.execute(&state)?;
        }
        Ok(state)
    }

    /// Return a reference to the inner `SubgraphNode`.
    pub fn node(&self) -> &SubgraphNode {
        &self.node
    }

    /// Return the children.
    pub fn children(&self) -> &[NestedSubgraph] {
        &self.children
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::state::{AsyncNodeAction, StateGraph};
    use serde_json::json;

    // -- Helpers --

    /// Build a simple synchronous graph function from a closure.
    fn make_graph_fn<F>(f: F) -> GraphFn
    where
        F: Fn(Value) -> Result<Value> + Send + Sync + 'static,
    {
        Arc::new(f)
    }

    /// Helper: build a simple single-node subgraph that applies the given action.
    fn build_single_node_subgraph(action: AsyncNodeAction) -> CompiledStateGraph {
        StateGraph::new()
            .add_node("inner", action)
            .set_entry_point("inner")
            .set_finish_point("inner")
            .compile()
            .unwrap()
    }

    // -----------------------------------------------------------------------
    // StateMapping tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_mapping_new_is_empty() {
        let m = StateMapping::new();
        assert!(m.is_empty());
        assert_eq!(m.len(), 0);
    }

    #[test]
    fn test_state_mapping_map_adds_entries() {
        let m = StateMapping::new().map("a", "b").map("c", "d");
        assert_eq!(m.len(), 2);
        assert!(!m.is_empty());
    }

    #[test]
    fn test_state_mapping_identity() {
        let m = StateMapping::new().identity(&["x", "y", "z"]);
        assert_eq!(m.len(), 3);
        assert_eq!(m.mappings().get("x").unwrap(), "x");
        assert_eq!(m.mappings().get("y").unwrap(), "y");
    }

    #[test]
    fn test_state_mapping_apply_input_basic() {
        let m = StateMapping::new()
            .map("parent_a", "child_a")
            .map("parent_b", "child_b");
        let parent = json!({"parent_a": 10, "parent_b": "hello", "extra": true});
        let child = m.apply_input(&parent);
        assert_eq!(child["child_a"], 10);
        assert_eq!(child["child_b"], "hello");
        assert!(child.get("extra").is_none());
        assert!(child.get("parent_a").is_none());
    }

    #[test]
    fn test_state_mapping_apply_input_missing_keys() {
        let m = StateMapping::new().map("exists", "a").map("missing", "b");
        let parent = json!({"exists": 42});
        let child = m.apply_input(&parent);
        assert_eq!(child["a"], 42);
        assert!(child.get("b").is_none());
    }

    #[test]
    fn test_state_mapping_apply_input_empty_passthrough() {
        let m = StateMapping::new();
        let parent = json!({"x": 1, "y": 2});
        let child = m.apply_input(&parent);
        assert_eq!(child, parent);
    }

    #[test]
    fn test_state_mapping_apply_output_basic() {
        let m = StateMapping::new().map("parent_out", "child_out");
        let child = json!({"child_out": 99, "internal": "secret"});
        let mut parent = json!({"existing": true});
        m.apply_output(&child, &mut parent);
        assert_eq!(parent["parent_out"], 99);
        assert_eq!(parent["existing"], true);
        assert!(parent.get("internal").is_none());
        assert!(parent.get("child_out").is_none());
    }

    #[test]
    fn test_state_mapping_apply_output_empty_merges_all() {
        let m = StateMapping::new();
        let child = json!({"a": 1, "b": 2});
        let mut parent = json!({"c": 3});
        m.apply_output(&child, &mut parent);
        assert_eq!(parent["a"], 1);
        assert_eq!(parent["b"], 2);
        assert_eq!(parent["c"], 3);
    }

    #[test]
    fn test_state_mapping_apply_output_overwrites_existing() {
        let m = StateMapping::new().map("x", "x");
        let child = json!({"x": "new_value"});
        let mut parent = json!({"x": "old_value", "y": "preserved"});
        m.apply_output(&child, &mut parent);
        assert_eq!(parent["x"], "new_value");
        assert_eq!(parent["y"], "preserved");
    }

    #[test]
    fn test_state_mapping_apply_output_missing_child_key() {
        let m = StateMapping::new().map("out", "nonexistent");
        let child = json!({"other": 1});
        let mut parent = json!({"out": "original"});
        m.apply_output(&child, &mut parent);
        // "out" should remain unchanged because the child key doesn't exist.
        assert_eq!(parent["out"], "original");
    }

    #[test]
    fn test_state_mapping_identity_apply_roundtrip() {
        let m = StateMapping::new().identity(&["a", "b"]);
        let state = json!({"a": 1, "b": 2, "c": 3});
        let child_in = m.apply_input(&state);
        assert_eq!(child_in["a"], 1);
        assert_eq!(child_in["b"], 2);
        assert!(child_in.get("c").is_none());
    }

    // -----------------------------------------------------------------------
    // SubgraphConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_subgraph_config_builder_pattern() {
        let config = SubgraphConfig::new("my_sub")
            .with_name("renamed")
            .with_input_mapping(StateMapping::new().map("a", "b"))
            .with_output_mapping(StateMapping::new().map("c", "d"))
            .with_namespace("ns1");

        assert_eq!(config.name, "renamed");
        assert_eq!(config.input_mapping.len(), 1);
        assert_eq!(config.output_mapping.len(), 1);
        assert_eq!(config.namespace.as_deref(), Some("ns1"));
    }

    #[test]
    fn test_subgraph_config_defaults() {
        let config = SubgraphConfig::new("default");
        assert_eq!(config.name, "default");
        assert!(config.input_mapping.is_empty());
        assert!(config.output_mapping.is_empty());
        assert!(config.namespace.is_none());
    }

    // -----------------------------------------------------------------------
    // SubgraphNode tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_subgraph_node_execute_passthrough() {
        let gfn = make_graph_fn(|state: Value| {
            let x = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(json!({"x": x * 2}))
        });
        let node = SubgraphNode::from_fn("double", gfn, SubgraphConfig::new("double"));
        let result = node.execute(&json!({"x": 5})).unwrap();
        assert_eq!(result["x"], 10);
    }

    #[test]
    fn test_subgraph_node_execute_with_mappings() {
        let gfn = make_graph_fn(|state: Value| {
            let val = state.get("input").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(json!({"output": val + 100}))
        });
        let config = SubgraphConfig::new("adder")
            .with_input_mapping(StateMapping::new().map("parent_val", "input"))
            .with_output_mapping(StateMapping::new().map("parent_result", "output"));
        let node = SubgraphNode::from_fn("adder", gfn, config);

        let result = node
            .execute(&json!({"parent_val": 42, "other": "keep"}))
            .unwrap();
        assert_eq!(result["parent_result"], 142);
        assert_eq!(result["parent_val"], 42);
        assert_eq!(result["other"], "keep");
        // "output" from child should not leak.
        assert!(result.get("output").is_none());
    }

    #[test]
    fn test_subgraph_node_execute_error_propagation() {
        let gfn =
            make_graph_fn(|_state: Value| Err(LangGraphError::Other("child error".to_string())));
        let node = SubgraphNode::from_fn("fail", gfn, SubgraphConfig::new("fail"));
        let result = node.execute(&json!({"x": 1}));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(format!("{}", err).contains("child error"));
    }

    #[test]
    fn test_subgraph_node_name() {
        let gfn = make_graph_fn(|s: Value| Ok(s));
        let node = SubgraphNode::from_fn("test_name", gfn, SubgraphConfig::new("test_name"));
        assert_eq!(node.name(), "test_name");
    }

    #[test]
    fn test_subgraph_node_with_namespace() {
        // The subgraph function works with un-namespaced keys internally.
        let gfn = make_graph_fn(|state: Value| {
            let x = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(json!({"x": x + 1}))
        });
        // With namespace "child", output keys are prefixed as "child.x".
        // The output mapping maps from namespaced parent key to the child key.
        let config = SubgraphConfig::new("ns_test")
            .with_namespace("child")
            .with_input_mapping(StateMapping::new().identity(&["x"]))
            .with_output_mapping(StateMapping::new().map("result", "child.x"));
        let node = SubgraphNode::from_fn("ns_test", gfn, config);
        let result = node.execute(&json!({"x": 10})).unwrap();
        // The namespace prefixes output "x" -> "child.x", and the output
        // mapping maps "child.x" (child key) to "result" (parent key).
        assert_eq!(result["result"], 11);
        // Original key preserved.
        assert_eq!(result["x"], 10);
    }

    #[test]
    fn test_subgraph_node_empty_state() {
        let gfn = make_graph_fn(|_state: Value| Ok(json!({"result": "done"})));
        let node = SubgraphNode::from_fn("empty", gfn, SubgraphConfig::new("empty"));
        let result = node.execute(&json!({})).unwrap();
        assert_eq!(result["result"], "done");
    }

    // -----------------------------------------------------------------------
    // SubgraphRegistry tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_registry_new_is_empty() {
        let reg = SubgraphRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = SubgraphRegistry::new();
        let gfn = make_graph_fn(|s: Value| Ok(s));
        let node = SubgraphNode::from_fn("a", gfn, SubgraphConfig::new("a"));
        reg.register("a", node);
        assert_eq!(reg.len(), 1);
        assert!(reg.get("a").is_some());
        assert!(reg.get("b").is_none());
    }

    #[test]
    fn test_registry_remove() {
        let mut reg = SubgraphRegistry::new();
        let gfn = make_graph_fn(|s: Value| Ok(s));
        reg.register(
            "x",
            SubgraphNode::from_fn("x", gfn.clone(), SubgraphConfig::new("x")),
        );
        reg.register(
            "y",
            SubgraphNode::from_fn("y", gfn, SubgraphConfig::new("y")),
        );
        assert_eq!(reg.len(), 2);
        let removed = reg.remove("x");
        assert!(removed.is_some());
        assert_eq!(reg.len(), 1);
        assert!(reg.get("x").is_none());
    }

    #[test]
    fn test_registry_list() {
        let mut reg = SubgraphRegistry::new();
        let gfn = make_graph_fn(|s: Value| Ok(s));
        reg.register(
            "alpha",
            SubgraphNode::from_fn("alpha", gfn.clone(), SubgraphConfig::new("alpha")),
        );
        reg.register(
            "beta",
            SubgraphNode::from_fn("beta", gfn, SubgraphConfig::new("beta")),
        );
        let mut names = reg.list();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_registry_replace_existing() {
        let mut reg = SubgraphRegistry::new();
        let gfn1 = make_graph_fn(|_s: Value| Ok(json!({"v": 1})));
        let gfn2 = make_graph_fn(|_s: Value| Ok(json!({"v": 2})));
        reg.register(
            "a",
            SubgraphNode::from_fn("a", gfn1, SubgraphConfig::new("a")),
        );
        reg.register(
            "a",
            SubgraphNode::from_fn("a", gfn2, SubgraphConfig::new("a")),
        );
        assert_eq!(reg.len(), 1);
        let node = reg.get("a").unwrap();
        let result = node.execute(&json!({})).unwrap();
        assert_eq!(result["v"], 2);
    }

    #[test]
    fn test_registry_remove_nonexistent() {
        let mut reg = SubgraphRegistry::new();
        assert!(reg.remove("nope").is_none());
    }

    // -----------------------------------------------------------------------
    // SubgraphBuilder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_fluent_api() {
        let gfn = make_graph_fn(|s: Value| Ok(s));
        let node = SubgraphBuilder::new("test")
            .graph_fn(gfn)
            .input_mapping(StateMapping::new().map("a", "b"))
            .output_mapping(StateMapping::new().map("c", "d"))
            .namespace("ns")
            .build()
            .unwrap();

        assert_eq!(node.name(), "test");
        assert_eq!(node.config().namespace.as_deref(), Some("ns"));
        assert_eq!(node.config().input_mapping.len(), 1);
        assert_eq!(node.config().output_mapping.len(), 1);
    }

    #[test]
    fn test_builder_missing_graph_fn_errors() {
        let result = SubgraphBuilder::new("no_fn").build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_minimal() {
        let gfn = make_graph_fn(|s: Value| Ok(s));
        let node = SubgraphBuilder::new("minimal")
            .graph_fn(gfn)
            .build()
            .unwrap();
        assert_eq!(node.name(), "minimal");
        assert!(node.config().input_mapping.is_empty());
        assert!(node.config().output_mapping.is_empty());
        assert!(node.config().namespace.is_none());
    }

    // -----------------------------------------------------------------------
    // NestedSubgraph tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_nested_depth_leaf() {
        let gfn = make_graph_fn(|s: Value| Ok(s));
        let node = SubgraphNode::from_fn("leaf", gfn, SubgraphConfig::new("leaf"));
        let nested = NestedSubgraph::new(node);
        assert_eq!(nested.depth(), 1);
    }

    #[test]
    fn test_nested_depth_two_levels() {
        let gfn = make_graph_fn(|s: Value| Ok(s));
        let child = NestedSubgraph::new(SubgraphNode::from_fn(
            "child",
            gfn.clone(),
            SubgraphConfig::new("child"),
        ));
        let parent = NestedSubgraph::new(SubgraphNode::from_fn(
            "parent",
            gfn,
            SubgraphConfig::new("parent"),
        ))
        .add_child(child);
        assert_eq!(parent.depth(), 2);
    }

    #[test]
    fn test_nested_depth_three_levels() {
        let gfn = make_graph_fn(|s: Value| Ok(s));
        let grandchild = NestedSubgraph::new(SubgraphNode::from_fn(
            "gc",
            gfn.clone(),
            SubgraphConfig::new("gc"),
        ));
        let child = NestedSubgraph::new(SubgraphNode::from_fn(
            "child",
            gfn.clone(),
            SubgraphConfig::new("child"),
        ))
        .add_child(grandchild);
        let root = NestedSubgraph::new(SubgraphNode::from_fn(
            "root",
            gfn,
            SubgraphConfig::new("root"),
        ))
        .add_child(child);
        assert_eq!(root.depth(), 3);
    }

    #[test]
    fn test_nested_flatten() {
        let gfn = make_graph_fn(|s: Value| Ok(s));
        let gc = NestedSubgraph::new(SubgraphNode::from_fn(
            "gc",
            gfn.clone(),
            SubgraphConfig::new("gc"),
        ));
        let child = NestedSubgraph::new(SubgraphNode::from_fn(
            "child",
            gfn.clone(),
            SubgraphConfig::new("child"),
        ))
        .add_child(gc);
        let root = NestedSubgraph::new(SubgraphNode::from_fn(
            "root",
            gfn,
            SubgraphConfig::new("root"),
        ))
        .add_child(child);

        let flat = root.flatten();
        assert_eq!(flat.len(), 3);
        assert_eq!(flat[0], (1, "root".to_string()));
        assert_eq!(flat[1], (2, "child".to_string()));
        assert_eq!(flat[2], (3, "gc".to_string()));
    }

    #[test]
    fn test_nested_execute_chain() {
        let gfn_add = make_graph_fn(|state: Value| {
            let x = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(json!({"x": x + 1}))
        });
        let gfn_double = make_graph_fn(|state: Value| {
            let x = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(json!({"x": x * 2}))
        });

        let child = NestedSubgraph::new(SubgraphNode::from_fn(
            "double",
            gfn_double,
            SubgraphConfig::new("double"),
        ));
        let root = NestedSubgraph::new(SubgraphNode::from_fn(
            "add",
            gfn_add,
            SubgraphConfig::new("add"),
        ))
        .add_child(child);

        // x=5 -> add: 6 -> double: 12
        let result = root.execute(&json!({"x": 5})).unwrap();
        assert_eq!(result["x"], 12);
    }

    #[test]
    fn test_nested_error_propagation() {
        let gfn_ok = make_graph_fn(|s: Value| Ok(s));
        let gfn_err =
            make_graph_fn(|_s: Value| Err(LangGraphError::Other("nested failure".to_string())));

        let child = NestedSubgraph::new(SubgraphNode::from_fn(
            "fail",
            gfn_err,
            SubgraphConfig::new("fail"),
        ));
        let root = NestedSubgraph::new(SubgraphNode::from_fn(
            "ok",
            gfn_ok,
            SubgraphConfig::new("ok"),
        ))
        .add_child(child);

        let result = root.execute(&json!({"x": 1}));
        assert!(result.is_err());
    }

    #[test]
    fn test_nested_multiple_children() {
        let gfn_add = make_graph_fn(|state: Value| {
            let x = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(json!({"x": x + 10}))
        });
        let gfn_set_flag = make_graph_fn(|mut state: Value| {
            state["flag"] = json!(true);
            Ok(state)
        });

        let child1 = NestedSubgraph::new(SubgraphNode::from_fn(
            "add",
            gfn_add,
            SubgraphConfig::new("add"),
        ));
        let child2 = NestedSubgraph::new(SubgraphNode::from_fn(
            "flag",
            gfn_set_flag,
            SubgraphConfig::new("flag"),
        ));

        let gfn_id = make_graph_fn(|s: Value| Ok(s));
        let root = NestedSubgraph::new(SubgraphNode::from_fn(
            "root",
            gfn_id,
            SubgraphConfig::new("root"),
        ))
        .add_child(child1)
        .add_child(child2);

        let result = root.execute(&json!({"x": 5})).unwrap();
        // root: passthrough -> child1: x=15 -> child2: flag=true
        assert_eq!(result["x"], 15);
        assert_eq!(result["flag"], true);
        assert_eq!(root.depth(), 2);
    }

    // -----------------------------------------------------------------------
    // Integration tests using CompiledStateGraph
    // -----------------------------------------------------------------------

    /// Test: Simple subgraph that transforms a value.
    #[tokio::test]
    async fn test_simple_subgraph_transforms_value() {
        let inner_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "x": val * 2 }))
            })
        });

        let inner = build_single_node_subgraph(inner_action);

        let outer = StateGraph::new()
            .add_subgraph("sub", inner)
            .set_entry_point("sub")
            .set_finish_point("sub")
            .compile()
            .unwrap();

        let result = outer.invoke(json!({ "x": 5 })).await.unwrap();
        assert_eq!(result["x"], 10);
    }

    /// Test: Subgraph with input/output mapping.
    #[tokio::test]
    async fn test_subgraph_with_input_output_mapping() {
        let inner_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("input_val").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "output_val": val + 100 }))
            })
        });

        let inner = build_single_node_subgraph(inner_action);

        let mut input_map = HashMap::new();
        input_map.insert("parent_x".to_string(), "input_val".to_string());

        let mut output_map = HashMap::new();
        output_map.insert("output_val".to_string(), "parent_result".to_string());

        let sub_node = SubgraphNode::with_mapping(inner, input_map, output_map);
        let outer = StateGraph::new()
            .add_node("sub", sub_node.into_action())
            .set_entry_point("sub")
            .set_finish_point("sub")
            .compile()
            .unwrap();

        let result = outer
            .invoke(json!({ "parent_x": 42, "other_key": "hello" }))
            .await
            .unwrap();

        assert_eq!(result["parent_result"], 142);
        assert_eq!(result["parent_x"], 42);
        assert_eq!(result["other_key"], "hello");
    }

    /// Test: Nested subgraphs (subgraph within subgraph).
    #[tokio::test]
    async fn test_nested_subgraphs() {
        let leaf_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("v").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "v": val + 10 }))
            })
        });

        let innermost = build_single_node_subgraph(leaf_action);

        let middle = StateGraph::new()
            .add_subgraph("nested_inner", innermost)
            .set_entry_point("nested_inner")
            .set_finish_point("nested_inner")
            .compile()
            .unwrap();

        let outer = StateGraph::new()
            .add_subgraph("nested_middle", middle)
            .set_entry_point("nested_middle")
            .set_finish_point("nested_middle")
            .compile()
            .unwrap();

        let result = outer.invoke(json!({ "v": 5 })).await.unwrap();
        assert_eq!(result["v"], 15);
    }

    /// Test: State isolation — subgraph internal state does not leak to parent.
    #[tokio::test]
    async fn test_state_isolation() {
        let inner_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({
                    "a": val * 2,
                    "result": val * 2,
                    "internal_secret": "should_not_leak"
                }))
            })
        });

        let inner = build_single_node_subgraph(inner_action);

        let mut input_map = HashMap::new();
        input_map.insert("a".to_string(), "a".to_string());

        let mut output_map = HashMap::new();
        output_map.insert("result".to_string(), "result".to_string());

        let sub_node = SubgraphNode::with_mapping(inner, input_map, output_map);
        let outer = StateGraph::new()
            .add_node("sub", sub_node.into_action())
            .set_entry_point("sub")
            .set_finish_point("sub")
            .compile()
            .unwrap();

        let result = outer.invoke(json!({ "a": 7 })).await.unwrap();

        assert_eq!(result["result"], 14);
        assert!(result.get("internal_secret").is_none());
        assert_eq!(result["a"], 7);
    }

    /// Test: Subgraph in a larger pipeline with other nodes before and after.
    #[tokio::test]
    async fn test_subgraph_in_pipeline() {
        let pre_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "x": val + 1, "stage": "pre" }))
            })
        });

        let sub_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "x": val * 10 }))
            })
        });

        let post_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "x": val + 1000, "stage": "post" }))
            })
        });

        let inner = build_single_node_subgraph(sub_action);

        let outer = StateGraph::new()
            .add_node("pre", pre_action)
            .add_subgraph("sub", inner)
            .add_node("post", post_action)
            .set_entry_point("pre")
            .add_edge("pre", "sub")
            .add_edge("sub", "post")
            .set_finish_point("post")
            .compile()
            .unwrap();

        let result = outer.invoke(json!({ "x": 2 })).await.unwrap();
        assert_eq!(result["x"], 1030);
        assert_eq!(result["stage"], "post");
    }

    /// Test: add_subgraph_with_mapping convenience method on StateGraph.
    #[tokio::test]
    async fn test_add_subgraph_with_mapping_convenience() {
        let inner_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("sub_input").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "sub_output": val + 50 }))
            })
        });

        let inner = build_single_node_subgraph(inner_action);

        let mut input_map = HashMap::new();
        input_map.insert("parent_val".to_string(), "sub_input".to_string());

        let mut output_map = HashMap::new();
        output_map.insert("sub_output".to_string(), "parent_result".to_string());

        let outer = StateGraph::new()
            .add_subgraph_with_mapping("sub", inner, input_map, output_map)
            .set_entry_point("sub")
            .set_finish_point("sub")
            .compile()
            .unwrap();

        let result = outer.invoke(json!({ "parent_val": 25 })).await.unwrap();

        assert_eq!(result["parent_result"], 75);
        assert_eq!(result["parent_val"], 25);
    }
}
