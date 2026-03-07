use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Result, RustChainError};

// ---------------------------------------------------------------------------
// Existing types (preserved)
// ---------------------------------------------------------------------------

/// A node in a runnable graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Node {
    /// Unique identifier for this node.
    pub id: String,
    /// Human-readable name of the node.
    pub name: String,
    /// Optional data payload associated with the node.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<NodeData>,
    /// Arbitrary metadata attached to the node.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, Value>,
}

impl Node {
    /// Create a new node with the given id and name.
    pub fn new(id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            data: None,
            metadata: HashMap::new(),
        }
    }

    /// Attach a data payload to this node.
    pub fn with_data(mut self, data: NodeData) -> Self {
        self.data = Some(data);
        self
    }

    /// Attach metadata to this node.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Data associated with a graph node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeData {
    /// A JSON schema describing the node's input or output.
    Schema(Value),
    /// The type name of the runnable this node represents.
    Runnable(String),
}

/// An edge connecting two nodes in the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    /// ID of the source node.
    pub source: String,
    /// ID of the target node.
    pub target: String,
    /// Optional label for the edge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<String>,
    /// Whether this edge represents a conditional branch.
    #[serde(default)]
    pub conditional: bool,
}

impl Edge {
    /// Create a new unconditional edge.
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            data: None,
            conditional: false,
        }
    }

    /// Create a new conditional edge.
    pub fn conditional(
        source: impl Into<String>,
        target: impl Into<String>,
        label: impl Into<String>,
    ) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            data: Some(label.into()),
            conditional: true,
        }
    }
}

/// A branch point in the graph with named targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Branch {
    /// Description or name of the condition function.
    pub condition: String,
    /// Mapping from condition output values to target node IDs.
    pub targets: HashMap<String, String>,
}

/// A directed graph representing a runnable's execution flow.
///
/// This is the Rust equivalent of `langchain_core.runnables.graph.Graph`.
/// It can represent linear chains, parallel branches, and conditional routing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Graph {
    /// Nodes in the graph, keyed by node ID.
    pub nodes: HashMap<String, Node>,
    /// Directed edges connecting nodes.
    pub edges: Vec<Edge>,
}

impl Graph {
    /// Create a new empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: Node) -> &mut Self {
        self.nodes.insert(node.id.clone(), node);
        self
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: Edge) -> &mut Self {
        self.edges.push(edge);
        self
    }

    /// Get the first (input) node -- a node with no incoming edges.
    pub fn first_node(&self) -> Option<&Node> {
        let targets: HashSet<&str> = self.edges.iter().map(|e| e.target.as_str()).collect();
        self.nodes
            .values()
            .find(|n| !targets.contains(n.id.as_str()))
    }

    /// Get the last (output) node -- a node with no outgoing edges.
    pub fn last_node(&self) -> Option<&Node> {
        let sources: HashSet<&str> = self.edges.iter().map(|e| e.source.as_str()).collect();
        self.nodes
            .values()
            .find(|n| !sources.contains(n.id.as_str()))
    }

    /// Perform a topological sort using Kahn's algorithm.
    ///
    /// Returns node IDs in topological order. If the graph contains a cycle,
    /// the returned list will be shorter than the number of nodes.
    pub fn topological_sort(&self) -> Vec<String> {
        // Build in-degree map and adjacency list.
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();

        for id in self.nodes.keys() {
            in_degree.entry(id.as_str()).or_insert(0);
            adjacency.entry(id.as_str()).or_default();
        }

        for edge in &self.edges {
            // Only count edges between known nodes.
            if self.nodes.contains_key(&edge.source) && self.nodes.contains_key(&edge.target) {
                *in_degree.entry(edge.target.as_str()).or_insert(0) += 1;
                adjacency
                    .entry(edge.source.as_str())
                    .or_default()
                    .push(edge.target.as_str());
            }
        }

        // Seed the queue with zero-in-degree nodes.
        let mut queue: VecDeque<&str> = in_degree
            .iter()
            .filter(|(_, &deg)| deg == 0)
            .map(|(&id, _)| id)
            .collect();

        // Sort the initial queue for deterministic output.
        let mut sorted_queue: Vec<&str> = queue.drain(..).collect();
        sorted_queue.sort();
        queue.extend(sorted_queue);

        let mut result: Vec<String> = Vec::with_capacity(self.nodes.len());

        while let Some(node_id) = queue.pop_front() {
            result.push(node_id.to_string());

            if let Some(neighbors) = adjacency.get(node_id) {
                // Sort neighbors for deterministic ordering.
                let mut sorted_neighbors: Vec<&str> = neighbors.clone();
                sorted_neighbors.sort();

                for &neighbor in &sorted_neighbors {
                    if let Some(deg) = in_degree.get_mut(neighbor) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }
        }

        result
    }

    /// Render the graph as simple ASCII art.
    ///
    /// This produces a linear top-to-bottom representation following
    /// topological order. Branching is not visually represented.
    pub fn draw_ascii(&self) -> String {
        let mut output = String::new();
        let sorted = self.topological_sort();

        for (i, node_id) in sorted.iter().enumerate() {
            if let Some(node) = self.nodes.get(node_id) {
                if i > 0 {
                    output.push_str("  |\n  v\n");
                }
                let border = "-".repeat(node.name.len() + 2);
                output.push_str(&format!("+{}+\n", border));
                output.push_str(&format!("| {} |\n", node.name));
                output.push_str(&format!("+{}+\n", border));
            }
        }

        output
    }

    /// Render the graph as a Mermaid diagram.
    ///
    /// The output can be pasted into Mermaid-compatible renderers
    /// (GitHub markdown, mermaid.live, etc.).
    pub fn draw_mermaid(&self) -> String {
        let mut lines = vec![
            "%%{init: {'flowchart': {'curve': 'linear'}}}%%".to_string(),
            "graph TD;".to_string(),
        ];

        // Emit nodes in topological order for consistency.
        let sorted = self.topological_sort();
        for node_id in &sorted {
            if let Some(node) = self.nodes.get(node_id) {
                lines.push(format!("  {}[\"{}\"];", node_id, node.name));
            }
        }

        // Emit edges.
        for edge in &self.edges {
            let arrow = if edge.conditional { "-.->" } else { "-->" };
            match &edge.data {
                Some(label) => {
                    lines.push(format!(
                        "  {} {}|\"{}\"| {};",
                        edge.source, arrow, label, edge.target
                    ));
                }
                None => {
                    lines.push(format!("  {} {} {};", edge.source, arrow, edge.target));
                }
            }
        }

        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// New types: extended graph visualization
// ---------------------------------------------------------------------------

/// Classification of a node within a runnable pipeline graph.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeType {
    /// A standard runnable step.
    Runnable,
    /// An input entry point.
    Input,
    /// An output exit point.
    Output,
    /// A conditional branch point.
    Branch,
    /// A parallel fan-out/fan-in node.
    Parallel,
    /// An inline lambda/closure.
    Lambda,
    /// A user-defined custom node type.
    Custom(String),
}

impl fmt::Display for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeType::Runnable => write!(f, "Runnable"),
            NodeType::Input => write!(f, "Input"),
            NodeType::Output => write!(f, "Output"),
            NodeType::Branch => write!(f, "Branch"),
            NodeType::Parallel => write!(f, "Parallel"),
            NodeType::Lambda => write!(f, "Lambda"),
            NodeType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// A richly-typed node in a `RunnableGraph`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique identifier.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// The semantic type of this node.
    pub node_type: NodeType,
    /// Arbitrary key-value metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, Value>,
}

impl GraphNode {
    /// Create a new graph node.
    pub fn new(id: impl Into<String>, name: impl Into<String>, node_type: NodeType) -> Self {
        Self {
            id: id.into(),
            name: name.into(),
            node_type,
            metadata: HashMap::new(),
        }
    }

    /// Add a metadata entry (builder pattern).
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Serialize this node to a JSON `Value`.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// A directed edge in a `RunnableGraph`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Source node ID.
    pub source: String,
    /// Target node ID.
    pub target: String,
    /// Optional human-readable label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    /// Optional condition expression that guards this edge.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub condition: Option<String>,
}

impl GraphEdge {
    /// Create a new edge between two nodes.
    pub fn new(source: impl Into<String>, target: impl Into<String>) -> Self {
        Self {
            source: source.into(),
            target: target.into(),
            label: None,
            condition: None,
        }
    }

    /// Set the label (builder pattern).
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Set the condition (builder pattern).
    pub fn with_condition(mut self, condition: impl Into<String>) -> Self {
        self.condition = Some(condition.into());
        self
    }

    /// Serialize this edge to a JSON `Value`.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// A DAG representing a runnable pipeline with rich node types and export capabilities.
#[derive(Debug, Clone, Default)]
pub struct RunnableGraph {
    nodes: Vec<GraphNode>,
    node_index: HashMap<String, usize>,
    edges: Vec<GraphEdge>,
}

impl RunnableGraph {
    /// Create a new, empty runnable graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a node to the graph.
    pub fn add_node(&mut self, node: GraphNode) {
        let idx = self.nodes.len();
        self.node_index.insert(node.id.clone(), idx);
        self.nodes.push(node);
    }

    /// Add an edge to the graph.
    pub fn add_edge(&mut self, edge: GraphEdge) {
        self.edges.push(edge);
    }

    /// Look up a node by its ID.
    pub fn get_node(&self, id: &str) -> Option<&GraphNode> {
        self.node_index.get(id).map(|&idx| &self.nodes[idx])
    }

    /// Return all edges originating from the given node.
    pub fn get_edges_from(&self, id: &str) -> Vec<&GraphEdge> {
        self.edges.iter().filter(|e| e.source == id).collect()
    }

    /// Iterator over all nodes.
    pub fn nodes(&self) -> &[GraphNode] {
        &self.nodes
    }

    /// Iterator over all edges.
    pub fn edges(&self) -> &[GraphEdge] {
        &self.edges
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Return root nodes (nodes with no incoming edges).
    pub fn roots(&self) -> Vec<&GraphNode> {
        let targets: HashSet<&str> = self.edges.iter().map(|e| e.target.as_str()).collect();
        self.nodes
            .iter()
            .filter(|n| !targets.contains(n.id.as_str()))
            .collect()
    }

    /// Return leaf nodes (nodes with no outgoing edges).
    pub fn leaves(&self) -> Vec<&GraphNode> {
        let sources: HashSet<&str> = self.edges.iter().map(|e| e.source.as_str()).collect();
        self.nodes
            .iter()
            .filter(|n| !sources.contains(n.id.as_str()))
            .collect()
    }

    /// Topological sort via Kahn's algorithm.
    ///
    /// Returns an error if the graph contains a cycle.
    pub fn topological_sort(&self) -> Result<Vec<&GraphNode>> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut adjacency: HashMap<&str, Vec<&str>> = HashMap::new();

        for node in &self.nodes {
            in_degree.entry(node.id.as_str()).or_insert(0);
            adjacency.entry(node.id.as_str()).or_default();
        }

        for edge in &self.edges {
            if self.node_index.contains_key(&edge.source)
                && self.node_index.contains_key(&edge.target)
            {
                *in_degree.entry(edge.target.as_str()).or_insert(0) += 1;
                adjacency
                    .entry(edge.source.as_str())
                    .or_default()
                    .push(edge.target.as_str());
            }
        }

        let mut queue: VecDeque<&str> = VecDeque::new();
        let mut initial: Vec<&str> = in_degree
            .iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        initial.sort();
        queue.extend(initial);

        let mut sorted_ids: Vec<&str> = Vec::with_capacity(self.nodes.len());

        while let Some(id) = queue.pop_front() {
            sorted_ids.push(id);
            if let Some(neighbors) = adjacency.get(id) {
                let mut sorted_n: Vec<&str> = neighbors.clone();
                sorted_n.sort();
                for &n in &sorted_n {
                    if let Some(d) = in_degree.get_mut(n) {
                        *d -= 1;
                        if *d == 0 {
                            queue.push_back(n);
                        }
                    }
                }
            }
        }

        if sorted_ids.len() != self.nodes.len() {
            return Err(RustChainError::Other(
                "Graph contains a cycle; topological sort is not possible".into(),
            ));
        }

        Ok(sorted_ids
            .iter()
            .map(|id| self.get_node(id).unwrap())
            .collect())
    }

    /// Serialize the full graph to a JSON `Value`.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "nodes": self.nodes.iter().map(|n| n.to_json()).collect::<Vec<_>>(),
            "edges": self.edges.iter().map(|e| e.to_json()).collect::<Vec<_>>(),
        })
    }
}

// ---------------------------------------------------------------------------
// Exporters
// ---------------------------------------------------------------------------

/// Exports a `RunnableGraph` as a Mermaid flowchart diagram.
pub struct MermaidExporter;

impl MermaidExporter {
    /// Export using the default top-down direction (`TD`).
    pub fn export(graph: &RunnableGraph) -> String {
        Self::export_with_direction(graph, "TD")
    }

    /// Export with a custom direction (`LR`, `TD`, `RL`, `BT`).
    pub fn export_with_direction(graph: &RunnableGraph, direction: &str) -> String {
        let sorted = match graph.topological_sort() {
            Ok(s) => s,
            Err(_) => {
                // Fall back to insertion order when there is a cycle.
                graph.nodes.iter().collect()
            }
        };

        let mut lines = vec![format!("graph {};", direction)];

        // Emit nodes with shapes based on type.
        for node in &sorted {
            let shape = match &node.node_type {
                NodeType::Input => format!("([\"{}\"]);", node.name),
                NodeType::Output => format!("([\"{}\"]);", node.name),
                NodeType::Branch => format!("{{\"{}\"}}; ", node.name),
                NodeType::Parallel => format!("[/\"{}\"\\];", node.name),
                NodeType::Lambda => format!(">\"{}\"]; ", node.name),
                NodeType::Custom(_) => format!("[\"{}\"];", node.name),
                NodeType::Runnable => format!("[\"{}\"];", node.name),
            };
            lines.push(format!("  {}{}", node.id, shape));
        }

        // Emit edges.
        for edge in &graph.edges {
            match (&edge.label, &edge.condition) {
                (Some(label), _) => {
                    lines.push(format!(
                        "  {} -->|\"{}\"| {};",
                        edge.source, label, edge.target
                    ));
                }
                (None, Some(cond)) => {
                    lines.push(format!(
                        "  {} -.->|\"{}\"| {};",
                        edge.source, cond, edge.target
                    ));
                }
                _ => {
                    lines.push(format!("  {} --> {};", edge.source, edge.target));
                }
            }
        }

        lines.join("\n")
    }
}

/// Options for the Graphviz DOT exporter.
#[derive(Debug, Clone)]
pub struct DotOptions {
    /// Graph rank direction (e.g. `"LR"`, `"TB"`).
    pub rankdir: String,
    /// Default node shape.
    pub node_shape: String,
    /// Default edge style.
    pub edge_style: String,
}

impl Default for DotOptions {
    fn default() -> Self {
        Self {
            rankdir: "TB".to_string(),
            node_shape: "box".to_string(),
            edge_style: "solid".to_string(),
        }
    }
}

/// Exports a `RunnableGraph` as Graphviz DOT format.
pub struct DotExporter;

impl DotExporter {
    /// Export with default options.
    pub fn export(graph: &RunnableGraph) -> String {
        Self::export_with_options(graph, DotOptions::default())
    }

    /// Export with custom options.
    pub fn export_with_options(graph: &RunnableGraph, options: DotOptions) -> String {
        let mut lines = vec![
            "digraph G {".to_string(),
            format!("  rankdir={};", options.rankdir),
            format!(
                "  node [shape={}, style=filled, fillcolor=lightyellow];",
                options.node_shape
            ),
            format!("  edge [style={}];", options.edge_style),
        ];

        let sorted = match graph.topological_sort() {
            Ok(s) => s,
            Err(_) => graph.nodes.iter().collect(),
        };

        for node in &sorted {
            let shape = match &node.node_type {
                NodeType::Input => "ellipse",
                NodeType::Output => "ellipse",
                NodeType::Branch => "diamond",
                NodeType::Parallel => "parallelogram",
                NodeType::Lambda => "oval",
                _ => &options.node_shape,
            };
            lines.push(format!(
                "  {} [label=\"{}\" shape={}];",
                node.id, node.name, shape
            ));
        }

        for edge in &graph.edges {
            let label_part = match (&edge.label, &edge.condition) {
                (Some(l), _) => format!(" [label=\"{}\"]", l),
                (None, Some(c)) => format!(" [label=\"{}\" style=dashed]", c),
                _ => String::new(),
            };
            lines.push(format!(
                "  {} -> {}{};",
                edge.source, edge.target, label_part
            ));
        }

        lines.push("}".to_string());
        lines.join("\n")
    }
}

/// Exports a `RunnableGraph` as a simple ASCII tree representation.
pub struct AsciiExporter;

impl AsciiExporter {
    /// Export the graph as ASCII art following topological order.
    pub fn export(graph: &RunnableGraph) -> String {
        let sorted = match graph.topological_sort() {
            Ok(s) => s,
            Err(_) => graph.nodes.iter().collect(),
        };

        if sorted.is_empty() {
            return String::new();
        }

        let mut output = String::new();

        for (i, node) in sorted.iter().enumerate() {
            if i > 0 {
                output.push_str("  |\n  v\n");
            }
            let type_tag = format!("[{}]", node.node_type);
            let label = format!("{} {}", node.name, type_tag);
            let border = "-".repeat(label.len() + 2);
            output.push_str(&format!("+{}+\n", border));
            output.push_str(&format!("| {} |\n", label));
            output.push_str(&format!("+{}+\n", border));
        }

        output
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- Existing tests (preserved) --

    fn sample_graph() -> Graph {
        let mut g = Graph::new();
        g.add_node(Node::new("a", "Input"));
        g.add_node(Node::new("b", "Process"));
        g.add_node(Node::new("c", "Output"));
        g.add_edge(Edge::new("a", "b"));
        g.add_edge(Edge::new("b", "c"));
        g
    }

    #[test]
    fn test_topological_sort_linear() {
        let g = sample_graph();
        assert_eq!(g.topological_sort(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_first_and_last_node() {
        let g = sample_graph();
        assert_eq!(g.first_node().unwrap().id, "a");
        assert_eq!(g.last_node().unwrap().id, "c");
    }

    #[test]
    fn test_topological_sort_diamond() {
        let mut g = Graph::new();
        g.add_node(Node::new("start", "Start"));
        g.add_node(Node::new("left", "Left"));
        g.add_node(Node::new("right", "Right"));
        g.add_node(Node::new("end", "End"));
        g.add_edge(Edge::new("start", "left"));
        g.add_edge(Edge::new("start", "right"));
        g.add_edge(Edge::new("left", "end"));
        g.add_edge(Edge::new("right", "end"));

        let sorted = g.topological_sort();
        assert_eq!(sorted.len(), 4);
        assert_eq!(sorted[0], "start");
        assert_eq!(sorted[3], "end");
        // left and right can be in either order, but we sort deterministically
        assert!(sorted[1] == "left" && sorted[2] == "right");
    }

    #[test]
    fn test_draw_ascii() {
        let g = sample_graph();
        let ascii = g.draw_ascii();
        assert!(ascii.contains("Input"));
        assert!(ascii.contains("Process"));
        assert!(ascii.contains("Output"));
    }

    #[test]
    fn test_draw_mermaid() {
        let g = sample_graph();
        let mermaid = g.draw_mermaid();
        assert!(mermaid.contains("graph TD;"));
        assert!(mermaid.contains("a --> b;"));
        assert!(mermaid.contains("b --> c;"));
    }

    #[test]
    fn test_conditional_edge_mermaid() {
        let mut g = Graph::new();
        g.add_node(Node::new("a", "Check"));
        g.add_node(Node::new("b", "Yes"));
        g.add_node(Node::new("c", "No"));
        g.add_edge(Edge::conditional("a", "b", "true"));
        g.add_edge(Edge::conditional("a", "c", "false"));

        let mermaid = g.draw_mermaid();
        assert!(mermaid.contains("-.->"));
        assert!(mermaid.contains("\"true\""));
        assert!(mermaid.contains("\"false\""));
    }

    #[test]
    fn test_empty_graph() {
        let g = Graph::new();
        assert!(g.topological_sort().is_empty());
        assert!(g.first_node().is_none());
        assert!(g.last_node().is_none());
        assert!(g.draw_ascii().is_empty());
    }

    // -- New tests for extended types --

    // NodeType Display

    #[test]
    fn test_node_type_display_runnable() {
        assert_eq!(NodeType::Runnable.to_string(), "Runnable");
    }

    #[test]
    fn test_node_type_display_input() {
        assert_eq!(NodeType::Input.to_string(), "Input");
    }

    #[test]
    fn test_node_type_display_output() {
        assert_eq!(NodeType::Output.to_string(), "Output");
    }

    #[test]
    fn test_node_type_display_branch() {
        assert_eq!(NodeType::Branch.to_string(), "Branch");
    }

    #[test]
    fn test_node_type_display_parallel() {
        assert_eq!(NodeType::Parallel.to_string(), "Parallel");
    }

    #[test]
    fn test_node_type_display_lambda() {
        assert_eq!(NodeType::Lambda.to_string(), "Lambda");
    }

    #[test]
    fn test_node_type_display_custom() {
        assert_eq!(
            NodeType::Custom("MyType".into()).to_string(),
            "Custom(MyType)"
        );
    }

    // GraphNode

    #[test]
    fn test_graph_node_creation() {
        let node = GraphNode::new("n1", "Step 1", NodeType::Runnable);
        assert_eq!(node.id, "n1");
        assert_eq!(node.name, "Step 1");
        assert_eq!(node.node_type, NodeType::Runnable);
        assert!(node.metadata.is_empty());
    }

    #[test]
    fn test_graph_node_with_metadata() {
        let node = GraphNode::new("n1", "Step 1", NodeType::Input)
            .with_metadata("version", json!(2))
            .with_metadata("author", json!("test"));
        assert_eq!(node.metadata.len(), 2);
        assert_eq!(node.metadata["version"], json!(2));
    }

    #[test]
    fn test_graph_node_to_json() {
        let node = GraphNode::new("n1", "Step 1", NodeType::Lambda);
        let j = node.to_json();
        assert_eq!(j["id"], "n1");
        assert_eq!(j["name"], "Step 1");
        assert_eq!(j["node_type"], "Lambda");
    }

    // GraphEdge

    #[test]
    fn test_graph_edge_creation() {
        let edge = GraphEdge::new("a", "b");
        assert_eq!(edge.source, "a");
        assert_eq!(edge.target, "b");
        assert!(edge.label.is_none());
        assert!(edge.condition.is_none());
    }

    #[test]
    fn test_graph_edge_with_label() {
        let edge = GraphEdge::new("a", "b").with_label("next");
        assert_eq!(edge.label.as_deref(), Some("next"));
    }

    #[test]
    fn test_graph_edge_with_condition() {
        let edge = GraphEdge::new("a", "b").with_condition("x > 0");
        assert_eq!(edge.condition.as_deref(), Some("x > 0"));
    }

    #[test]
    fn test_graph_edge_to_json() {
        let edge = GraphEdge::new("a", "b").with_label("go");
        let j = edge.to_json();
        assert_eq!(j["source"], "a");
        assert_eq!(j["target"], "b");
        assert_eq!(j["label"], "go");
    }

    // RunnableGraph basics

    fn sample_runnable_graph() -> RunnableGraph {
        let mut g = RunnableGraph::new();
        g.add_node(GraphNode::new("input", "Input", NodeType::Input));
        g.add_node(GraphNode::new("step1", "Process", NodeType::Runnable));
        g.add_node(GraphNode::new("output", "Output", NodeType::Output));
        g.add_edge(GraphEdge::new("input", "step1"));
        g.add_edge(GraphEdge::new("step1", "output"));
        g
    }

    #[test]
    fn test_runnable_graph_add_and_count() {
        let g = sample_runnable_graph();
        assert_eq!(g.node_count(), 3);
        assert_eq!(g.edge_count(), 2);
    }

    #[test]
    fn test_runnable_graph_get_node() {
        let g = sample_runnable_graph();
        assert_eq!(g.get_node("step1").unwrap().name, "Process");
        assert!(g.get_node("nonexistent").is_none());
    }

    #[test]
    fn test_runnable_graph_get_edges_from() {
        let g = sample_runnable_graph();
        let edges = g.get_edges_from("input");
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].target, "step1");
    }

    #[test]
    fn test_runnable_graph_nodes_and_edges() {
        let g = sample_runnable_graph();
        assert_eq!(g.nodes().len(), 3);
        assert_eq!(g.edges().len(), 2);
    }

    #[test]
    fn test_runnable_graph_roots() {
        let g = sample_runnable_graph();
        let roots = g.roots();
        assert_eq!(roots.len(), 1);
        assert_eq!(roots[0].id, "input");
    }

    #[test]
    fn test_runnable_graph_leaves() {
        let g = sample_runnable_graph();
        let leaves = g.leaves();
        assert_eq!(leaves.len(), 1);
        assert_eq!(leaves[0].id, "output");
    }

    // Topological sort

    #[test]
    fn test_runnable_graph_topological_sort_linear() {
        let g = sample_runnable_graph();
        let sorted = g.topological_sort().unwrap();
        let ids: Vec<&str> = sorted.iter().map(|n| n.id.as_str()).collect();
        // input must come before step1, step1 before output
        let pos = |id: &str| ids.iter().position(|&x| x == id).unwrap();
        assert!(pos("input") < pos("step1"));
        assert!(pos("step1") < pos("output"));
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_runnable_graph_topological_sort_branching() {
        let mut g = RunnableGraph::new();
        g.add_node(GraphNode::new("start", "Start", NodeType::Input));
        g.add_node(GraphNode::new("left", "Left", NodeType::Runnable));
        g.add_node(GraphNode::new("right", "Right", NodeType::Runnable));
        g.add_node(GraphNode::new("end", "End", NodeType::Output));
        g.add_edge(GraphEdge::new("start", "left"));
        g.add_edge(GraphEdge::new("start", "right"));
        g.add_edge(GraphEdge::new("left", "end"));
        g.add_edge(GraphEdge::new("right", "end"));

        let sorted = g.topological_sort().unwrap();
        let ids: Vec<&str> = sorted.iter().map(|n| n.id.as_str()).collect();
        assert_eq!(ids[0], "start");
        assert_eq!(ids[3], "end");
        // left and right in between (deterministic sort)
        assert!(ids.contains(&"left"));
        assert!(ids.contains(&"right"));
    }

    #[test]
    fn test_runnable_graph_topological_sort_cycle_detection() {
        let mut g = RunnableGraph::new();
        g.add_node(GraphNode::new("a", "A", NodeType::Runnable));
        g.add_node(GraphNode::new("b", "B", NodeType::Runnable));
        g.add_node(GraphNode::new("c", "C", NodeType::Runnable));
        g.add_edge(GraphEdge::new("a", "b"));
        g.add_edge(GraphEdge::new("b", "c"));
        g.add_edge(GraphEdge::new("c", "a")); // cycle

        let result = g.topological_sort();
        assert!(result.is_err());
    }

    #[test]
    fn test_runnable_graph_empty() {
        let g = RunnableGraph::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert!(g.roots().is_empty());
        assert!(g.leaves().is_empty());
        let sorted = g.topological_sort().unwrap();
        assert!(sorted.is_empty());
    }

    #[test]
    fn test_runnable_graph_single_node() {
        let mut g = RunnableGraph::new();
        g.add_node(GraphNode::new("only", "Alone", NodeType::Runnable));
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.roots().len(), 1);
        assert_eq!(g.leaves().len(), 1);
        assert_eq!(g.roots()[0].id, "only");
        assert_eq!(g.leaves()[0].id, "only");
    }

    #[test]
    fn test_runnable_graph_to_json() {
        let g = sample_runnable_graph();
        let j = g.to_json();
        assert!(j["nodes"].is_array());
        assert!(j["edges"].is_array());
        assert_eq!(j["nodes"].as_array().unwrap().len(), 3);
        assert_eq!(j["edges"].as_array().unwrap().len(), 2);
    }

    // MermaidExporter

    #[test]
    fn test_mermaid_export_default_direction() {
        let g = sample_runnable_graph();
        let output = MermaidExporter::export(&g);
        assert!(output.starts_with("graph TD;"));
        assert!(output.contains("input"));
        assert!(output.contains("step1"));
        assert!(output.contains("output"));
        assert!(output.contains("-->"));
    }

    #[test]
    fn test_mermaid_export_lr_direction() {
        let g = sample_runnable_graph();
        let output = MermaidExporter::export_with_direction(&g, "LR");
        assert!(output.starts_with("graph LR;"));
    }

    #[test]
    fn test_mermaid_export_with_label() {
        let mut g = RunnableGraph::new();
        g.add_node(GraphNode::new("a", "A", NodeType::Runnable));
        g.add_node(GraphNode::new("b", "B", NodeType::Runnable));
        g.add_edge(GraphEdge::new("a", "b").with_label("next"));
        let output = MermaidExporter::export(&g);
        assert!(output.contains("|\"next\"|"));
    }

    #[test]
    fn test_mermaid_export_with_condition() {
        let mut g = RunnableGraph::new();
        g.add_node(GraphNode::new("a", "A", NodeType::Branch));
        g.add_node(GraphNode::new("b", "B", NodeType::Runnable));
        g.add_edge(GraphEdge::new("a", "b").with_condition("x > 0"));
        let output = MermaidExporter::export(&g);
        assert!(output.contains("-.->"));
        assert!(output.contains("x > 0"));
    }

    #[test]
    fn test_mermaid_export_input_output_shapes() {
        let g = sample_runnable_graph();
        let output = MermaidExporter::export(&g);
        // Input nodes use stadium shape ([" "])
        assert!(output.contains("input([\"Input\"]);"));
        // Output nodes use stadium shape
        assert!(output.contains("output([\"Output\"]);"));
    }

    // DotExporter

    #[test]
    fn test_dot_export_default() {
        let g = sample_runnable_graph();
        let output = DotExporter::export(&g);
        assert!(output.starts_with("digraph G {"));
        assert!(output.ends_with("}"));
        assert!(output.contains("rankdir=TB;"));
        assert!(output.contains("input -> step1;"));
        assert!(output.contains("step1 -> output;"));
    }

    #[test]
    fn test_dot_export_custom_options() {
        let g = sample_runnable_graph();
        let opts = DotOptions {
            rankdir: "LR".into(),
            node_shape: "circle".into(),
            edge_style: "dashed".into(),
        };
        let output = DotExporter::export_with_options(&g, opts);
        assert!(output.contains("rankdir=LR;"));
        assert!(output.contains("style=dashed"));
    }

    #[test]
    fn test_dot_export_with_label() {
        let mut g = RunnableGraph::new();
        g.add_node(GraphNode::new("a", "A", NodeType::Runnable));
        g.add_node(GraphNode::new("b", "B", NodeType::Runnable));
        g.add_edge(GraphEdge::new("a", "b").with_label("process"));
        let output = DotExporter::export(&g);
        assert!(output.contains("[label=\"process\"]"));
    }

    #[test]
    fn test_dot_export_with_condition() {
        let mut g = RunnableGraph::new();
        g.add_node(GraphNode::new("a", "A", NodeType::Branch));
        g.add_node(GraphNode::new("b", "B", NodeType::Runnable));
        g.add_edge(GraphEdge::new("a", "b").with_condition("flag == true"));
        let output = DotExporter::export(&g);
        assert!(output.contains("style=dashed"));
        assert!(output.contains("flag == true"));
    }

    #[test]
    fn test_dot_export_node_shapes() {
        let mut g = RunnableGraph::new();
        g.add_node(GraphNode::new("i", "In", NodeType::Input));
        g.add_node(GraphNode::new("b", "Check", NodeType::Branch));
        g.add_node(GraphNode::new("o", "Out", NodeType::Output));
        g.add_edge(GraphEdge::new("i", "b"));
        g.add_edge(GraphEdge::new("b", "o"));
        let output = DotExporter::export(&g);
        assert!(output.contains("shape=ellipse"));
        assert!(output.contains("shape=diamond"));
    }

    // AsciiExporter

    #[test]
    fn test_ascii_export_linear() {
        let g = sample_runnable_graph();
        let output = AsciiExporter::export(&g);
        assert!(output.contains("Input"));
        assert!(output.contains("Process"));
        assert!(output.contains("Output"));
        assert!(output.contains("[Runnable]"));
        assert!(output.contains("[Input]"));
        assert!(output.contains("[Output]"));
        assert!(output.contains("  |\n  v\n"));
    }

    #[test]
    fn test_ascii_export_empty_graph() {
        let g = RunnableGraph::new();
        let output = AsciiExporter::export(&g);
        assert!(output.is_empty());
    }

    #[test]
    fn test_ascii_export_single_node() {
        let mut g = RunnableGraph::new();
        g.add_node(GraphNode::new("x", "Only", NodeType::Lambda));
        let output = AsciiExporter::export(&g);
        assert!(output.contains("Only"));
        assert!(output.contains("[Lambda]"));
        assert!(!output.contains("  |"));
    }

    // Complex graph

    #[test]
    fn test_complex_parallel_graph() {
        let mut g = RunnableGraph::new();
        g.add_node(GraphNode::new("start", "Start", NodeType::Input));
        g.add_node(GraphNode::new("fan_out", "FanOut", NodeType::Parallel));
        g.add_node(GraphNode::new("w1", "Worker1", NodeType::Runnable));
        g.add_node(GraphNode::new("w2", "Worker2", NodeType::Runnable));
        g.add_node(GraphNode::new("w3", "Worker3", NodeType::Runnable));
        g.add_node(GraphNode::new("merge", "Merge", NodeType::Parallel));
        g.add_node(GraphNode::new("end", "End", NodeType::Output));
        g.add_edge(GraphEdge::new("start", "fan_out"));
        g.add_edge(GraphEdge::new("fan_out", "w1"));
        g.add_edge(GraphEdge::new("fan_out", "w2"));
        g.add_edge(GraphEdge::new("fan_out", "w3"));
        g.add_edge(GraphEdge::new("w1", "merge"));
        g.add_edge(GraphEdge::new("w2", "merge"));
        g.add_edge(GraphEdge::new("w3", "merge"));
        g.add_edge(GraphEdge::new("merge", "end"));

        assert_eq!(g.node_count(), 7);
        assert_eq!(g.edge_count(), 8);
        assert_eq!(g.roots().len(), 1);
        assert_eq!(g.leaves().len(), 1);

        let sorted = g.topological_sort().unwrap();
        assert_eq!(sorted.len(), 7);
        assert_eq!(sorted[0].id, "start");
        assert_eq!(sorted.last().unwrap().id, "end");

        // All exporters should succeed
        let mermaid = MermaidExporter::export(&g);
        assert!(mermaid.contains("fan_out"));
        let dot = DotExporter::export(&g);
        assert!(dot.contains("fan_out -> w1"));
        let ascii = AsciiExporter::export(&g);
        assert!(ascii.contains("FanOut"));
    }

    #[test]
    fn test_complex_branch_graph() {
        let mut g = RunnableGraph::new();
        g.add_node(GraphNode::new("start", "Start", NodeType::Input));
        g.add_node(GraphNode::new("check", "Check", NodeType::Branch));
        g.add_node(GraphNode::new("path_a", "PathA", NodeType::Runnable));
        g.add_node(GraphNode::new("path_b", "PathB", NodeType::Runnable));
        g.add_node(GraphNode::new("end", "End", NodeType::Output));
        g.add_edge(GraphEdge::new("start", "check"));
        g.add_edge(GraphEdge::new("check", "path_a").with_condition("value > 10"));
        g.add_edge(GraphEdge::new("check", "path_b").with_condition("value <= 10"));
        g.add_edge(GraphEdge::new("path_a", "end"));
        g.add_edge(GraphEdge::new("path_b", "end"));

        let sorted = g.topological_sort().unwrap();
        assert_eq!(sorted.len(), 5);

        let mermaid = MermaidExporter::export(&g);
        assert!(mermaid.contains("-.->"));
        assert!(mermaid.contains("value > 10"));

        let dot = DotExporter::export(&g);
        assert!(dot.contains("shape=diamond"));
    }

    #[test]
    fn test_dot_options_default() {
        let opts = DotOptions::default();
        assert_eq!(opts.rankdir, "TB");
        assert_eq!(opts.node_shape, "box");
        assert_eq!(opts.edge_style, "solid");
    }

    #[test]
    fn test_node_type_equality() {
        assert_eq!(NodeType::Runnable, NodeType::Runnable);
        assert_ne!(NodeType::Input, NodeType::Output);
        assert_eq!(NodeType::Custom("x".into()), NodeType::Custom("x".into()));
        assert_ne!(NodeType::Custom("x".into()), NodeType::Custom("y".into()));
    }

    #[test]
    fn test_graph_edge_label_and_condition() {
        let edge = GraphEdge::new("a", "b")
            .with_label("step")
            .with_condition("if ready");
        assert_eq!(edge.label.as_deref(), Some("step"));
        assert_eq!(edge.condition.as_deref(), Some("if ready"));
        let j = edge.to_json();
        assert_eq!(j["label"], "step");
        assert_eq!(j["condition"], "if ready");
    }

    #[test]
    fn test_multiple_roots_and_leaves() {
        let mut g = RunnableGraph::new();
        g.add_node(GraphNode::new("r1", "Root1", NodeType::Input));
        g.add_node(GraphNode::new("r2", "Root2", NodeType::Input));
        g.add_node(GraphNode::new("mid", "Middle", NodeType::Runnable));
        g.add_node(GraphNode::new("l1", "Leaf1", NodeType::Output));
        g.add_node(GraphNode::new("l2", "Leaf2", NodeType::Output));
        g.add_edge(GraphEdge::new("r1", "mid"));
        g.add_edge(GraphEdge::new("r2", "mid"));
        g.add_edge(GraphEdge::new("mid", "l1"));
        g.add_edge(GraphEdge::new("mid", "l2"));

        let roots = g.roots();
        assert_eq!(roots.len(), 2);
        let leaves = g.leaves();
        assert_eq!(leaves.len(), 2);
    }
}
