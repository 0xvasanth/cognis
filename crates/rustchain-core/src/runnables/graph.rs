use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::Value;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
