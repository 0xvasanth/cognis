//! Graph serialization for saving, loading, and sharing graph definitions.
//!
//! This module provides rich serialization primitives beyond the basic
//! [`super::serialize::GraphDefinition`]: structured node/edge representations,
//! multiple output formats, reusable graph templates with parameter substitution,
//! and graph definition versioning with diff support.

use std::collections::HashMap;
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::{LangGraphError, Result};

// ---------------------------------------------------------------------------
// EdgeType
// ---------------------------------------------------------------------------

/// The type of an edge in a serialized graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeType {
    /// A direct, unconditional edge.
    Direct,
    /// A conditional edge whose target is chosen at runtime.
    Conditional,
    /// A default/fallback edge.
    Default,
}

impl EdgeType {
    /// Return a static string representation.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Direct => "direct",
            Self::Conditional => "conditional",
            Self::Default => "default",
        }
    }
}

impl fmt::Display for EdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// SerializedNode
// ---------------------------------------------------------------------------

/// A fully-serializable representation of a graph node.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerializedNode {
    /// The unique name of this node.
    pub name: String,
    /// A free-form type label (e.g. `"llm"`, `"tool"`, `"router"`).
    pub node_type: String,
    /// Arbitrary configuration for the node.
    pub config: HashMap<String, Value>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, Value>,
}

impl SerializedNode {
    /// Create a new node with the given name and type.
    pub fn new(name: impl Into<String>, node_type: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            node_type: node_type.into(),
            config: HashMap::new(),
            metadata: HashMap::new(),
        }
    }

    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).expect("SerializedNode is always serializable")
    }

    /// Deserialize from a JSON [`Value`].
    pub fn from_json(value: &Value) -> Result<Self> {
        serde_json::from_value(value.clone()).map_err(|e| {
            LangGraphError::Other(format!("Failed to deserialize SerializedNode: {e}"))
        })
    }
}

// ---------------------------------------------------------------------------
// SerializedEdge
// ---------------------------------------------------------------------------

/// A fully-serializable representation of a graph edge.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerializedEdge {
    /// Source node name.
    pub source: String,
    /// Target node name.
    pub target: String,
    /// Edge kind.
    pub edge_type: EdgeType,
    /// Optional condition expression (for conditional edges).
    pub condition: Option<String>,
    /// Optional human-readable label.
    pub label: Option<String>,
}

impl SerializedEdge {
    /// Create a new direct edge.
    pub fn direct(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            edge_type: EdgeType::Direct,
            condition: None,
            label: None,
        }
    }

    /// Create a new conditional edge.
    pub fn conditional(
        source: impl Into<String>,
        target: impl Into<String>,
        condition: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            edge_type: EdgeType::Conditional,
            condition: Some(condition.into()),
            label: None,
        }
    }

    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).expect("SerializedEdge is always serializable")
    }

    /// Deserialize from a JSON [`Value`].
    pub fn from_json(value: &Value) -> Result<Self> {
        serde_json::from_value(value.clone()).map_err(|e| {
            LangGraphError::Other(format!("Failed to deserialize SerializedEdge: {e}"))
        })
    }
}

// ---------------------------------------------------------------------------
// GraphDefinition (serialization module version)
// ---------------------------------------------------------------------------

/// A rich, fully-serializable graph definition.
///
/// Unlike [`super::serialize::GraphDefinition`] which captures only topology,
/// this struct carries per-node configuration, metadata, and version info —
/// making it suitable for export, sharing, and template instantiation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SerializableGraphDefinition {
    /// Graph name.
    pub name: String,
    /// Semantic version string.
    pub version: String,
    /// Optional description.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Nodes in the graph.
    pub nodes: Vec<SerializedNode>,
    /// Edges in the graph.
    pub edges: Vec<SerializedEdge>,
    /// The name of the entry-point node.
    pub entry_point: String,
    /// Nodes that terminate the graph.
    pub finish_points: Vec<String>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, Value>,
}

impl SerializableGraphDefinition {
    /// Start building a new graph definition.
    pub fn builder(name: impl Into<String>, version: impl Into<String>) -> GraphDefBuilder {
        GraphDefBuilder {
            name: name.into(),
            version: version.into(),
            description: None,
            nodes: Vec::new(),
            edges: Vec::new(),
            entry_point: String::new(),
            finish_points: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Serialize to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).expect("SerializableGraphDefinition is always serializable")
    }

    /// Deserialize from a JSON [`Value`].
    pub fn from_json(value: &Value) -> Result<Self> {
        serde_json::from_value(value.clone()).map_err(|e| {
            LangGraphError::Other(format!(
                "Failed to deserialize SerializableGraphDefinition: {e}"
            ))
        })
    }

    /// Serialize to a pretty-printed JSON string.
    pub fn to_pretty_json(&self) -> String {
        serde_json::to_string_pretty(self)
            .expect("SerializableGraphDefinition is always serializable")
    }
}

// ---------------------------------------------------------------------------
// GraphDefBuilder
// ---------------------------------------------------------------------------

/// Builder for [`SerializableGraphDefinition`].
pub struct GraphDefBuilder {
    name: String,
    version: String,
    description: Option<String>,
    nodes: Vec<SerializedNode>,
    edges: Vec<SerializedEdge>,
    entry_point: String,
    finish_points: Vec<String>,
    metadata: HashMap<String, Value>,
}

impl GraphDefBuilder {
    /// Set the description.
    pub fn description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Add a node.
    pub fn node(mut self, node: SerializedNode) -> Self {
        self.nodes.push(node);
        self
    }

    /// Add an edge.
    pub fn edge(mut self, edge: SerializedEdge) -> Self {
        self.edges.push(edge);
        self
    }

    /// Set the entry point.
    pub fn entry_point(mut self, name: impl Into<String>) -> Self {
        self.entry_point = name.into();
        self
    }

    /// Add a finish point.
    pub fn finish_point(mut self, name: impl Into<String>) -> Self {
        self.finish_points.push(name.into());
        self
    }

    /// Add a metadata key-value pair.
    pub fn metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Build the graph definition.
    pub fn build(self) -> SerializableGraphDefinition {
        SerializableGraphDefinition {
            name: self.name,
            version: self.version,
            description: self.description,
            nodes: self.nodes,
            edges: self.edges,
            entry_point: self.entry_point,
            finish_points: self.finish_points,
            metadata: self.metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// GraphFormat
// ---------------------------------------------------------------------------

/// Supported serialization formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphFormat {
    /// Pretty-printed JSON.
    Json,
    /// YAML (approximated via JSON-to-YAML-like output since `serde_yaml` is
    /// not in the dependency list).
    Yaml,
    /// Compact single-line JSON.
    Compact,
}

impl GraphFormat {
    /// Serialize a [`SerializableGraphDefinition`] to a string in this format.
    pub fn serialize(&self, def: &SerializableGraphDefinition) -> Result<String> {
        match self {
            Self::Json => Ok(def.to_pretty_json()),
            Self::Compact => serde_json::to_string(def)
                .map_err(|e| LangGraphError::Other(format!("Failed to serialize (compact): {e}"))),
            Self::Yaml => {
                // Minimal YAML-like representation built from JSON value.
                let json = def.to_json();
                Ok(value_to_yaml(&json, 0))
            }
        }
    }

    /// Deserialize a [`SerializableGraphDefinition`] from a string.
    ///
    /// For [`GraphFormat::Yaml`] this expects JSON input (true YAML parsing
    /// would require an extra dependency).
    pub fn deserialize(&self, input: &str) -> Result<SerializableGraphDefinition> {
        match self {
            Self::Json | Self::Compact | Self::Yaml => {
                let value: Value = serde_json::from_str(input)
                    .map_err(|e| LangGraphError::Other(format!("Failed to parse input: {e}")))?;
                SerializableGraphDefinition::from_json(&value)
            }
        }
    }
}

/// Very small YAML-like renderer (no external dep).
fn value_to_yaml(val: &Value, indent: usize) -> String {
    let pad = " ".repeat(indent);
    match val {
        Value::Null => format!("{pad}null\n"),
        Value::Bool(b) => format!("{pad}{b}\n"),
        Value::Number(n) => format!("{pad}{n}\n"),
        Value::String(s) => format!("{pad}\"{s}\"\n"),
        Value::Array(arr) => {
            let mut out = String::new();
            for item in arr {
                out.push_str(&format!("{pad}- "));
                let rendered = value_to_yaml(item, indent + 2);
                out.push_str(rendered.trim_start());
            }
            if arr.is_empty() {
                out.push_str(&format!("{pad}[]\n"));
            }
            out
        }
        Value::Object(map) => {
            let mut out = String::new();
            for (k, v) in map {
                match v {
                    Value::Object(_) | Value::Array(_) => {
                        out.push_str(&format!("{pad}{k}:\n"));
                        out.push_str(&value_to_yaml(v, indent + 2));
                    }
                    _ => {
                        let rendered = value_to_yaml(v, 0);
                        out.push_str(&format!("{pad}{k}: {}", rendered.trim_start()));
                    }
                }
            }
            if map.is_empty() {
                out.push_str(&format!("{pad}{{}}\n"));
            }
            out
        }
    }
}

// ---------------------------------------------------------------------------
// GraphTemplate
// ---------------------------------------------------------------------------

/// A reusable graph template with parameter placeholders.
///
/// Parameter placeholders use the `{{param_name}}` syntax inside node
/// configuration values (strings only). When instantiated, every occurrence
/// is replaced with the supplied (or default) value.
#[derive(Debug, Clone)]
pub struct GraphTemplate {
    /// Template name.
    pub name: String,
    /// Template description.
    pub description: String,
    /// The base graph definition.
    definition: SerializableGraphDefinition,
    /// Parameter defaults: name -> default JSON value.
    parameters: HashMap<String, Value>,
}

impl GraphTemplate {
    /// Create a new empty template.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            definition: SerializableGraphDefinition {
                name: String::new(),
                version: "0.1.0".to_string(),
                description: None,
                nodes: Vec::new(),
                edges: Vec::new(),
                entry_point: String::new(),
                finish_points: Vec::new(),
                metadata: HashMap::new(),
            },
            parameters: HashMap::new(),
        }
    }

    /// Build a template from an existing graph definition.
    pub fn from_definition(def: &SerializableGraphDefinition) -> Self {
        Self {
            name: def.name.clone(),
            description: def.description.clone().unwrap_or_default(),
            definition: def.clone(),
            parameters: HashMap::new(),
        }
    }

    /// Register a parameter with an optional default value.
    pub fn with_parameter(mut self, name: impl Into<String>, default_value: Value) -> Self {
        self.parameters.insert(name.into(), default_value);
        self
    }

    /// Instantiate the template, producing a concrete graph definition.
    ///
    /// Parameters in `params` override any defaults. Placeholders of the form
    /// `{{name}}` are replaced in every string value inside node configs.
    pub fn instantiate(
        &self,
        params: &HashMap<String, Value>,
    ) -> Result<SerializableGraphDefinition> {
        // Merge defaults with provided params (provided wins).
        let mut merged = self.parameters.clone();
        for (k, v) in params {
            merged.insert(k.clone(), v.clone());
        }

        let mut def = self.definition.clone();

        // Replace placeholders in node configs.
        for node in &mut def.nodes {
            for (_key, val) in node.config.iter_mut() {
                substitute_value(val, &merged);
            }
            for (_key, val) in node.metadata.iter_mut() {
                substitute_value(val, &merged);
            }
        }

        Ok(def)
    }
}

/// Recursively substitute `{{param}}` placeholders in JSON string values.
fn substitute_value(val: &mut Value, params: &HashMap<String, Value>) {
    match val {
        Value::String(s) => {
            let mut result = s.clone();
            for (name, replacement) in params {
                let placeholder = format!("{{{{{}}}}}", name);
                if result.contains(&placeholder) {
                    let replacement_str = match replacement {
                        Value::String(rs) => rs.clone(),
                        other => other.to_string(),
                    };
                    result = result.replace(&placeholder, &replacement_str);
                }
            }
            *s = result;
        }
        Value::Array(arr) => {
            for item in arr {
                substitute_value(item, params);
            }
        }
        Value::Object(map) => {
            for (_k, v) in map.iter_mut() {
                substitute_value(v, params);
            }
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// GraphChanges
// ---------------------------------------------------------------------------

/// A summary of differences between two graph definition versions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct GraphChanges {
    /// Node names added in the newer version.
    pub added_nodes: Vec<String>,
    /// Node names removed from the older version.
    pub removed_nodes: Vec<String>,
    /// Edge descriptions added.
    pub added_edges: Vec<String>,
    /// Edge descriptions removed.
    pub removed_edges: Vec<String>,
    /// Node names whose config or metadata changed.
    pub modified_nodes: Vec<String>,
}

impl GraphChanges {
    /// Returns `true` if there are any differences.
    pub fn has_changes(&self) -> bool {
        !self.added_nodes.is_empty()
            || !self.removed_nodes.is_empty()
            || !self.added_edges.is_empty()
            || !self.removed_edges.is_empty()
            || !self.modified_nodes.is_empty()
    }

    /// Serialize the changes to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).expect("GraphChanges is always serializable")
    }
}

// ---------------------------------------------------------------------------
// GraphVersioning
// ---------------------------------------------------------------------------

/// Tracks multiple versions of a graph definition.
#[derive(Debug, Clone, Default)]
pub struct GraphVersioning {
    /// Ordered list of (version, definition).
    versions: Vec<(String, SerializableGraphDefinition)>,
}

impl GraphVersioning {
    /// Create a new empty versioning store.
    pub fn new() -> Self {
        Self {
            versions: Vec::new(),
        }
    }

    /// Save a new version. Uses `def.version` as the version key.
    pub fn save_version(&mut self, def: SerializableGraphDefinition) {
        let version = def.version.clone();
        self.versions.push((version, def));
    }

    /// Retrieve a definition by version string.
    pub fn get_version(&self, version: &str) -> Option<&SerializableGraphDefinition> {
        self.versions
            .iter()
            .find(|(v, _)| v == version)
            .map(|(_, d)| d)
    }

    /// Get the latest (most recently saved) definition.
    pub fn latest(&self) -> Option<&SerializableGraphDefinition> {
        self.versions.last().map(|(_, d)| d)
    }

    /// List all version strings in order.
    pub fn version_list(&self) -> Vec<&str> {
        self.versions.iter().map(|(v, _)| v.as_str()).collect()
    }

    /// Compute the diff between two versions.
    pub fn diff(&self, v1: &str, v2: &str) -> GraphChanges {
        let def1 = self.get_version(v1);
        let def2 = self.get_version(v2);

        match (def1, def2) {
            (Some(d1), Some(d2)) => compute_diff(d1, d2),
            _ => GraphChanges {
                added_nodes: Vec::new(),
                removed_nodes: Vec::new(),
                added_edges: Vec::new(),
                removed_edges: Vec::new(),
                modified_nodes: Vec::new(),
            },
        }
    }
}

/// Compute the diff between two graph definitions.
fn compute_diff(
    old: &SerializableGraphDefinition,
    new: &SerializableGraphDefinition,
) -> GraphChanges {
    // --- Nodes ---
    let old_names: HashMap<&str, &SerializedNode> =
        old.nodes.iter().map(|n| (n.name.as_str(), n)).collect();
    let new_names: HashMap<&str, &SerializedNode> =
        new.nodes.iter().map(|n| (n.name.as_str(), n)).collect();

    let mut added_nodes: Vec<String> = Vec::new();
    let mut removed_nodes: Vec<String> = Vec::new();
    let mut modified_nodes: Vec<String> = Vec::new();

    for name in new_names.keys() {
        if !old_names.contains_key(name) {
            added_nodes.push(name.to_string());
        }
    }
    for name in old_names.keys() {
        if !new_names.contains_key(name) {
            removed_nodes.push(name.to_string());
        }
    }
    for (name, old_node) in &old_names {
        if let Some(new_node) = new_names.get(name) {
            if old_node != new_node {
                modified_nodes.push(name.to_string());
            }
        }
    }

    added_nodes.sort();
    removed_nodes.sort();
    modified_nodes.sort();

    // --- Edges ---
    let edge_key = |e: &SerializedEdge| format!("{}->{}({})", e.source, e.target, e.edge_type);

    let old_edges: Vec<String> = old.edges.iter().map(edge_key).collect();
    let new_edges: Vec<String> = new.edges.iter().map(edge_key).collect();

    let old_set: std::collections::HashSet<&str> = old_edges.iter().map(|s| s.as_str()).collect();
    let new_set: std::collections::HashSet<&str> = new_edges.iter().map(|s| s.as_str()).collect();

    let mut added_edges: Vec<String> = new_set
        .difference(&old_set)
        .map(|s| s.to_string())
        .collect();
    let mut removed_edges: Vec<String> = old_set
        .difference(&new_set)
        .map(|s| s.to_string())
        .collect();

    added_edges.sort();
    removed_edges.sort();

    GraphChanges {
        added_nodes,
        removed_nodes,
        added_edges,
        removed_edges,
        modified_nodes,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- helpers ----

    fn sample_node(name: &str) -> SerializedNode {
        let mut config = HashMap::new();
        config.insert("model".to_string(), json!("gpt-4"));
        SerializedNode {
            name: name.to_string(),
            node_type: "llm".to_string(),
            config,
            metadata: HashMap::new(),
        }
    }

    fn sample_edge() -> SerializedEdge {
        SerializedEdge::direct("a", "b")
    }

    fn sample_graph_def() -> SerializableGraphDefinition {
        SerializableGraphDefinition::builder("test_graph", "1.0.0")
            .description("A test graph")
            .node(sample_node("agent"))
            .node(sample_node("tools"))
            .edge(SerializedEdge::direct("__start__", "agent"))
            .edge(SerializedEdge::conditional(
                "agent",
                "tools",
                "has_tool_call",
            ))
            .edge(SerializedEdge::direct("tools", "agent"))
            .entry_point("agent")
            .finish_point("__end__")
            .metadata("author", json!("test"))
            .build()
    }

    // ================================================================
    // SerializedNode tests
    // ================================================================

    #[test]
    fn test_serialized_node_to_json() {
        let node = sample_node("agent");
        let json = node.to_json();
        assert_eq!(json["name"], "agent");
        assert_eq!(json["node_type"], "llm");
        assert_eq!(json["config"]["model"], "gpt-4");
    }

    #[test]
    fn test_serialized_node_from_json() {
        let val = json!({
            "name": "router",
            "node_type": "router",
            "config": {"max_retries": 3},
            "metadata": {"version": "1"}
        });
        let node = SerializedNode::from_json(&val).unwrap();
        assert_eq!(node.name, "router");
        assert_eq!(node.node_type, "router");
        assert_eq!(node.config["max_retries"], json!(3));
        assert_eq!(node.metadata["version"], json!("1"));
    }

    #[test]
    fn test_serialized_node_json_round_trip() {
        let node = sample_node("n1");
        let json = node.to_json();
        let restored = SerializedNode::from_json(&json).unwrap();
        assert_eq!(node, restored);
    }

    #[test]
    fn test_serialized_node_empty_config() {
        let node = SerializedNode::new("empty", "plain");
        let json = node.to_json();
        let restored = SerializedNode::from_json(&json).unwrap();
        assert_eq!(restored.config.len(), 0);
        assert_eq!(restored.metadata.len(), 0);
    }

    #[test]
    fn test_serialized_node_from_json_missing_field() {
        let val = json!({"name": "x"});
        let result = SerializedNode::from_json(&val);
        assert!(result.is_err());
    }

    // ================================================================
    // SerializedEdge tests
    // ================================================================

    #[test]
    fn test_serialized_edge_direct_to_json() {
        let edge = SerializedEdge::direct("a", "b");
        let json = edge.to_json();
        assert_eq!(json["source"], "a");
        assert_eq!(json["target"], "b");
        assert_eq!(json["edge_type"], "direct");
        assert!(json["condition"].is_null());
    }

    #[test]
    fn test_serialized_edge_conditional_to_json() {
        let edge = SerializedEdge::conditional("a", "b", "check");
        let json = edge.to_json();
        assert_eq!(json["edge_type"], "conditional");
        assert_eq!(json["condition"], "check");
    }

    #[test]
    fn test_serialized_edge_json_round_trip() {
        let edge = sample_edge();
        let json = edge.to_json();
        let restored = SerializedEdge::from_json(&json).unwrap();
        assert_eq!(edge, restored);
    }

    #[test]
    fn test_serialized_edge_with_label() {
        let edge = SerializedEdge {
            source: "x".into(),
            target: "y".into(),
            edge_type: EdgeType::Default,
            condition: None,
            label: Some("fallback".into()),
        };
        let json = edge.to_json();
        assert_eq!(json["label"], "fallback");
        let restored = SerializedEdge::from_json(&json).unwrap();
        assert_eq!(restored.label, Some("fallback".into()));
    }

    #[test]
    fn test_serialized_edge_from_json_missing_field() {
        let val = json!({"source": "a"});
        assert!(SerializedEdge::from_json(&val).is_err());
    }

    // ================================================================
    // EdgeType tests
    // ================================================================

    #[test]
    fn test_edge_type_as_str() {
        assert_eq!(EdgeType::Direct.as_str(), "direct");
        assert_eq!(EdgeType::Conditional.as_str(), "conditional");
        assert_eq!(EdgeType::Default.as_str(), "default");
    }

    #[test]
    fn test_edge_type_display() {
        assert_eq!(format!("{}", EdgeType::Direct), "direct");
        assert_eq!(format!("{}", EdgeType::Conditional), "conditional");
        assert_eq!(format!("{}", EdgeType::Default), "default");
    }

    // ================================================================
    // SerializableGraphDefinition tests
    // ================================================================

    #[test]
    fn test_graph_def_builder() {
        let def = sample_graph_def();
        assert_eq!(def.name, "test_graph");
        assert_eq!(def.version, "1.0.0");
        assert_eq!(def.description, Some("A test graph".to_string()));
        assert_eq!(def.nodes.len(), 2);
        assert_eq!(def.edges.len(), 3);
        assert_eq!(def.entry_point, "agent");
        assert_eq!(def.finish_points, vec!["__end__"]);
        assert_eq!(def.metadata["author"], json!("test"));
    }

    #[test]
    fn test_graph_def_to_json() {
        let def = sample_graph_def();
        let json = def.to_json();
        assert_eq!(json["name"], "test_graph");
        assert!(json["nodes"].is_array());
        assert_eq!(json["nodes"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_graph_def_from_json() {
        let def = sample_graph_def();
        let json = def.to_json();
        let restored = SerializableGraphDefinition::from_json(&json).unwrap();
        assert_eq!(def, restored);
    }

    #[test]
    fn test_graph_def_to_pretty_json() {
        let def = sample_graph_def();
        let pretty = def.to_pretty_json();
        assert!(pretty.contains('\n'));
        assert!(pretty.contains("test_graph"));
    }

    #[test]
    fn test_graph_def_json_round_trip() {
        let def = sample_graph_def();
        let pretty = def.to_pretty_json();
        let value: Value = serde_json::from_str(&pretty).unwrap();
        let restored = SerializableGraphDefinition::from_json(&value).unwrap();
        assert_eq!(def, restored);
    }

    #[test]
    fn test_graph_def_empty_graph() {
        let def = SerializableGraphDefinition::builder("empty", "0.0.1").build();
        assert!(def.nodes.is_empty());
        assert!(def.edges.is_empty());
        assert!(def.finish_points.is_empty());
        assert!(def.description.is_none());
        let json = def.to_json();
        let restored = SerializableGraphDefinition::from_json(&json).unwrap();
        assert_eq!(def, restored);
    }

    #[test]
    fn test_graph_def_from_json_invalid() {
        let val = json!({"name": "x"});
        assert!(SerializableGraphDefinition::from_json(&val).is_err());
    }

    #[test]
    fn test_graph_def_no_description_skipped() {
        let def = SerializableGraphDefinition::builder("g", "1.0.0").build();
        let json = def.to_json();
        assert!(json.get("description").is_none());
    }

    // ================================================================
    // GraphFormat tests
    // ================================================================

    #[test]
    fn test_format_json_serialize() {
        let def = sample_graph_def();
        let output = GraphFormat::Json.serialize(&def).unwrap();
        assert!(output.contains('\n'));
        assert!(output.contains("test_graph"));
    }

    #[test]
    fn test_format_compact_serialize() {
        let def = sample_graph_def();
        let output = GraphFormat::Compact.serialize(&def).unwrap();
        assert!(!output.contains('\n'));
        assert!(output.contains("test_graph"));
    }

    #[test]
    fn test_format_yaml_serialize() {
        let def = sample_graph_def();
        let output = GraphFormat::Yaml.serialize(&def).unwrap();
        assert!(output.contains("test_graph"));
    }

    #[test]
    fn test_format_json_deserialize() {
        let def = sample_graph_def();
        let json_str = GraphFormat::Json.serialize(&def).unwrap();
        let restored = GraphFormat::Json.deserialize(&json_str).unwrap();
        assert_eq!(def, restored);
    }

    #[test]
    fn test_format_compact_deserialize() {
        let def = sample_graph_def();
        let compact = GraphFormat::Compact.serialize(&def).unwrap();
        let restored = GraphFormat::Compact.deserialize(&compact).unwrap();
        assert_eq!(def, restored);
    }

    #[test]
    fn test_format_deserialize_invalid_json() {
        let result = GraphFormat::Json.deserialize("not json");
        assert!(result.is_err());
    }

    // ================================================================
    // GraphTemplate tests
    // ================================================================

    #[test]
    fn test_template_new() {
        let t = GraphTemplate::new("my_tmpl", "A template");
        assert_eq!(t.name, "my_tmpl");
        assert_eq!(t.description, "A template");
    }

    #[test]
    fn test_template_from_definition() {
        let def = sample_graph_def();
        let t = GraphTemplate::from_definition(&def);
        assert_eq!(t.name, "test_graph");
        assert_eq!(t.description, "A test graph");
    }

    #[test]
    fn test_template_with_parameter() {
        let t = GraphTemplate::new("t", "d").with_parameter("model", json!("gpt-4"));
        assert_eq!(t.parameters["model"], json!("gpt-4"));
    }

    #[test]
    fn test_template_instantiate_no_params() {
        let def = sample_graph_def();
        let t = GraphTemplate::from_definition(&def);
        let params: HashMap<String, Value> = HashMap::new();
        let result = t.instantiate(&params).unwrap();
        assert_eq!(result.name, def.name);
        assert_eq!(result.nodes.len(), def.nodes.len());
    }

    #[test]
    fn test_template_parameter_substitution() {
        let mut node = SerializedNode::new("agent", "llm");
        node.config
            .insert("model".to_string(), json!("{{model_name}}"));
        node.config
            .insert("temperature".to_string(), json!("{{temp}}"));

        let def = SerializableGraphDefinition::builder("tmpl", "1.0.0")
            .node(node)
            .entry_point("agent")
            .build();

        let tmpl = GraphTemplate::from_definition(&def)
            .with_parameter("model_name", json!("default-model"));

        let mut params = HashMap::new();
        params.insert("model_name".to_string(), json!("claude-4"));
        params.insert("temp".to_string(), json!("0.7"));

        let result = tmpl.instantiate(&params).unwrap();
        assert_eq!(result.nodes[0].config["model"], json!("claude-4"));
        assert_eq!(result.nodes[0].config["temperature"], json!("0.7"));
    }

    #[test]
    fn test_template_default_parameter_used() {
        let mut node = SerializedNode::new("agent", "llm");
        node.config
            .insert("model".to_string(), json!("{{model_name}}"));

        let def = SerializableGraphDefinition::builder("tmpl", "1.0.0")
            .node(node)
            .entry_point("agent")
            .build();

        let tmpl = GraphTemplate::from_definition(&def)
            .with_parameter("model_name", json!("fallback-model"));

        let params = HashMap::new();
        let result = tmpl.instantiate(&params).unwrap();
        assert_eq!(result.nodes[0].config["model"], json!("fallback-model"));
    }

    #[test]
    fn test_template_substitution_in_metadata() {
        let mut node = SerializedNode::new("n", "t");
        node.metadata
            .insert("owner".to_string(), json!("{{owner}}"));

        let def = SerializableGraphDefinition::builder("t", "1.0.0")
            .node(node)
            .build();

        let tmpl = GraphTemplate::from_definition(&def);
        let mut params = HashMap::new();
        params.insert("owner".to_string(), json!("alice"));

        let result = tmpl.instantiate(&params).unwrap();
        assert_eq!(result.nodes[0].metadata["owner"], json!("alice"));
    }

    #[test]
    fn test_template_no_placeholder_unchanged() {
        let mut node = SerializedNode::new("n", "t");
        node.config.insert("key".to_string(), json!("plain_value"));

        let def = SerializableGraphDefinition::builder("t", "1.0.0")
            .node(node)
            .build();

        let tmpl = GraphTemplate::from_definition(&def);
        let mut params = HashMap::new();
        params.insert("unrelated".to_string(), json!("val"));

        let result = tmpl.instantiate(&params).unwrap();
        assert_eq!(result.nodes[0].config["key"], json!("plain_value"));
    }

    #[test]
    fn test_template_numeric_substitution() {
        let mut node = SerializedNode::new("n", "t");
        node.config.insert("count".to_string(), json!("{{count}}"));

        let def = SerializableGraphDefinition::builder("t", "1.0.0")
            .node(node)
            .build();

        let tmpl = GraphTemplate::from_definition(&def);
        let mut params = HashMap::new();
        params.insert("count".to_string(), json!(42));

        let result = tmpl.instantiate(&params).unwrap();
        assert_eq!(result.nodes[0].config["count"], json!("42"));
    }

    // ================================================================
    // GraphVersioning tests
    // ================================================================

    #[test]
    fn test_versioning_new_is_empty() {
        let v = GraphVersioning::new();
        assert!(v.latest().is_none());
        assert!(v.version_list().is_empty());
    }

    #[test]
    fn test_versioning_save_and_get() {
        let mut v = GraphVersioning::new();
        let def = sample_graph_def();
        v.save_version(def.clone());
        let fetched = v.get_version("1.0.0").unwrap();
        assert_eq!(fetched.name, "test_graph");
    }

    #[test]
    fn test_versioning_latest() {
        let mut v = GraphVersioning::new();
        let def1 = SerializableGraphDefinition::builder("g", "1.0.0").build();
        let def2 = SerializableGraphDefinition::builder("g", "2.0.0").build();
        v.save_version(def1);
        v.save_version(def2);
        assert_eq!(v.latest().unwrap().version, "2.0.0");
    }

    #[test]
    fn test_versioning_version_list() {
        let mut v = GraphVersioning::new();
        v.save_version(SerializableGraphDefinition::builder("g", "0.1.0").build());
        v.save_version(SerializableGraphDefinition::builder("g", "0.2.0").build());
        v.save_version(SerializableGraphDefinition::builder("g", "1.0.0").build());
        assert_eq!(v.version_list(), vec!["0.1.0", "0.2.0", "1.0.0"]);
    }

    #[test]
    fn test_versioning_get_nonexistent() {
        let v = GraphVersioning::new();
        assert!(v.get_version("nope").is_none());
    }

    #[test]
    fn test_versioning_diff_added_node() {
        let mut v = GraphVersioning::new();
        let def1 = SerializableGraphDefinition::builder("g", "1.0.0")
            .node(sample_node("a"))
            .build();
        let def2 = SerializableGraphDefinition::builder("g", "2.0.0")
            .node(sample_node("a"))
            .node(sample_node("b"))
            .build();
        v.save_version(def1);
        v.save_version(def2);

        let changes = v.diff("1.0.0", "2.0.0");
        assert!(changes.has_changes());
        assert_eq!(changes.added_nodes, vec!["b"]);
        assert!(changes.removed_nodes.is_empty());
    }

    #[test]
    fn test_versioning_diff_removed_node() {
        let mut v = GraphVersioning::new();
        let def1 = SerializableGraphDefinition::builder("g", "1.0.0")
            .node(sample_node("a"))
            .node(sample_node("b"))
            .build();
        let def2 = SerializableGraphDefinition::builder("g", "2.0.0")
            .node(sample_node("a"))
            .build();
        v.save_version(def1);
        v.save_version(def2);

        let changes = v.diff("1.0.0", "2.0.0");
        assert!(changes.has_changes());
        assert_eq!(changes.removed_nodes, vec!["b"]);
    }

    #[test]
    fn test_versioning_diff_modified_node() {
        let mut v = GraphVersioning::new();
        let node1 = sample_node("a");
        let mut node2 = sample_node("a");
        node2.config.insert("model".to_string(), json!("gpt-5"));

        let def1 = SerializableGraphDefinition::builder("g", "1.0.0")
            .node(node1)
            .build();
        let def2 = SerializableGraphDefinition::builder("g", "2.0.0")
            .node(node2)
            .build();
        v.save_version(def1);
        v.save_version(def2);

        let changes = v.diff("1.0.0", "2.0.0");
        assert!(changes.has_changes());
        assert_eq!(changes.modified_nodes, vec!["a"]);
    }

    #[test]
    fn test_versioning_diff_added_edge() {
        let mut v = GraphVersioning::new();
        let def1 = SerializableGraphDefinition::builder("g", "1.0.0")
            .edge(SerializedEdge::direct("a", "b"))
            .build();
        let def2 = SerializableGraphDefinition::builder("g", "2.0.0")
            .edge(SerializedEdge::direct("a", "b"))
            .edge(SerializedEdge::direct("b", "c"))
            .build();
        v.save_version(def1);
        v.save_version(def2);

        let changes = v.diff("1.0.0", "2.0.0");
        assert!(changes.has_changes());
        assert_eq!(changes.added_edges.len(), 1);
        assert!(changes.added_edges[0].contains("b->c"));
    }

    #[test]
    fn test_versioning_diff_no_changes() {
        let mut v = GraphVersioning::new();
        let def1 = SerializableGraphDefinition::builder("g", "1.0.0")
            .node(sample_node("a"))
            .build();
        let def2 = SerializableGraphDefinition::builder("g", "2.0.0")
            .node(sample_node("a"))
            .build();
        v.save_version(def1);
        v.save_version(def2);

        let changes = v.diff("1.0.0", "2.0.0");
        assert!(!changes.has_changes());
    }

    #[test]
    fn test_versioning_diff_nonexistent_version() {
        let v = GraphVersioning::new();
        let changes = v.diff("1.0.0", "2.0.0");
        assert!(!changes.has_changes());
    }

    // ================================================================
    // GraphChanges tests
    // ================================================================

    #[test]
    fn test_graph_changes_has_changes_true() {
        let c = GraphChanges {
            added_nodes: vec!["x".into()],
            removed_nodes: vec![],
            added_edges: vec![],
            removed_edges: vec![],
            modified_nodes: vec![],
        };
        assert!(c.has_changes());
    }

    #[test]
    fn test_graph_changes_has_changes_false() {
        let c = GraphChanges {
            added_nodes: vec![],
            removed_nodes: vec![],
            added_edges: vec![],
            removed_edges: vec![],
            modified_nodes: vec![],
        };
        assert!(!c.has_changes());
    }

    #[test]
    fn test_graph_changes_to_json() {
        let c = GraphChanges {
            added_nodes: vec!["new_node".into()],
            removed_nodes: vec!["old_node".into()],
            added_edges: vec!["a->b".into()],
            removed_edges: vec![],
            modified_nodes: vec!["changed".into()],
        };
        let json = c.to_json();
        assert_eq!(json["added_nodes"][0], "new_node");
        assert_eq!(json["removed_nodes"][0], "old_node");
        assert_eq!(json["modified_nodes"][0], "changed");
    }

    #[test]
    fn test_graph_changes_has_changes_with_edges_only() {
        let c = GraphChanges {
            added_nodes: vec![],
            removed_nodes: vec![],
            added_edges: vec!["edge".into()],
            removed_edges: vec![],
            modified_nodes: vec![],
        };
        assert!(c.has_changes());
    }

    #[test]
    fn test_graph_changes_has_changes_with_modified_only() {
        let c = GraphChanges {
            added_nodes: vec![],
            removed_nodes: vec![],
            added_edges: vec![],
            removed_edges: vec![],
            modified_nodes: vec!["m".into()],
        };
        assert!(c.has_changes());
    }

    // ================================================================
    // Substitute value tests
    // ================================================================

    #[test]
    fn test_substitute_nested_array() {
        let mut val = json!(["{{x}}", "literal", "{{y}}"]);
        let mut params = HashMap::new();
        params.insert("x".to_string(), json!("replaced_x"));
        params.insert("y".to_string(), json!("replaced_y"));
        substitute_value(&mut val, &params);
        assert_eq!(val[0], "replaced_x");
        assert_eq!(val[1], "literal");
        assert_eq!(val[2], "replaced_y");
    }

    #[test]
    fn test_substitute_nested_object() {
        let mut val = json!({"inner": {"key": "{{v}}"}});
        let mut params = HashMap::new();
        params.insert("v".to_string(), json!("done"));
        substitute_value(&mut val, &params);
        assert_eq!(val["inner"]["key"], "done");
    }
}
