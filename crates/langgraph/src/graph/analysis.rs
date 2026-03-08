//! Graph analysis utilities for inspecting and understanding LangGraph graphs.
//!
//! This module provides tools for analyzing graph structure, finding paths,
//! detecting cycles, computing metrics, comparing graphs, extracting subgraphs,
//! and generating visual representations.
//!
//! # Key Types
//!
//! - [`GraphTopology`] — core graph structure with nodes and edges
//! - [`PathFinder`] — finds paths between nodes (all, shortest, longest)
//! - [`GraphMetrics`] — computes structural metrics (density, degree, depth)
//! - [`CycleDetector`] — finds all cycles in a graph
//! - [`GraphDiff`] — compares two graph topologies
//! - [`SubgraphExtractor`] — extracts subgraphs by node set or reachability
//! - [`GraphVisualizer`] — generates DOT, Mermaid, and ASCII representations

use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use crate::errors::LangGraphError;

// ---------------------------------------------------------------------------
// GraphTopology
// ---------------------------------------------------------------------------

/// Analyzes and stores graph structure as nodes and directed edges.
#[derive(Debug, Clone)]
pub struct GraphTopology {
    /// Ordered set of node names.
    nodes: BTreeSet<String>,
    /// Directed edges as (from, to) pairs.
    edges: Vec<(String, String)>,
    /// Forward adjacency: node -> list of successors.
    forward: HashMap<String, Vec<String>>,
    /// Reverse adjacency: node -> list of predecessors.
    reverse: HashMap<String, Vec<String>>,
}

impl GraphTopology {
    /// Create a new empty graph topology.
    pub fn new() -> Self {
        Self {
            nodes: BTreeSet::new(),
            edges: Vec::new(),
            forward: HashMap::new(),
            reverse: HashMap::new(),
        }
    }

    /// Add a node to the graph. Duplicate names are ignored.
    pub fn add_node(&mut self, name: &str) {
        self.nodes.insert(name.to_string());
    }

    /// Add a directed edge from `from` to `to`.
    ///
    /// Both nodes are implicitly added if not already present.
    pub fn add_edge(&mut self, from: &str, to: &str) {
        self.nodes.insert(from.to_string());
        self.nodes.insert(to.to_string());
        self.edges.push((from.to_string(), to.to_string()));
        self.forward
            .entry(from.to_string())
            .or_default()
            .push(to.to_string());
        self.reverse
            .entry(to.to_string())
            .or_default()
            .push(from.to_string());
    }

    /// Return the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Return the number of edges in the graph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Return all node names as a sorted vector.
    pub fn node_names(&self) -> Vec<&str> {
        self.nodes.iter().map(|s| s.as_str()).collect()
    }

    /// Return all edges as (from, to) pairs.
    pub fn edge_list(&self) -> &[(String, String)] {
        &self.edges
    }

    /// Return the in-degree (number of incoming edges) for a node.
    ///
    /// Returns 0 if the node does not exist.
    pub fn in_degree(&self, node: &str) -> usize {
        self.reverse
            .get(node)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Return the out-degree (number of outgoing edges) for a node.
    ///
    /// Returns 0 if the node does not exist.
    pub fn out_degree(&self, node: &str) -> usize {
        self.forward
            .get(node)
            .map(|v| v.len())
            .unwrap_or(0)
    }

    /// Return source nodes (nodes with no incoming edges).
    pub fn sources(&self) -> Vec<&str> {
        self.nodes
            .iter()
            .filter(|n| self.in_degree(n) == 0)
            .map(|s| s.as_str())
            .collect()
    }

    /// Return sink nodes (nodes with no outgoing edges).
    pub fn sinks(&self) -> Vec<&str> {
        self.nodes
            .iter()
            .filter(|n| self.out_degree(n) == 0)
            .map(|s| s.as_str())
            .collect()
    }

    /// Check whether the graph is a directed acyclic graph (DAG).
    pub fn is_dag(&self) -> bool {
        self.topological_sort().is_ok()
    }

    /// Perform a topological sort using Kahn's algorithm.
    ///
    /// Returns `Err` if the graph contains a cycle.
    pub fn topological_sort(&self) -> crate::Result<Vec<String>> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        for node in &self.nodes {
            in_degree.insert(node.as_str(), 0);
        }
        for (_from, to) in &self.edges {
            *in_degree.entry(to.as_str()).or_insert(0) += 1;
        }

        let mut queue: VecDeque<&str> = VecDeque::new();
        for (&node, &deg) in &in_degree {
            if deg == 0 {
                queue.push_back(node);
            }
        }

        let mut result = Vec::new();
        while let Some(node) = queue.pop_front() {
            result.push(node.to_string());
            if let Some(neighbors) = self.forward.get(node) {
                for neighbor in neighbors {
                    if let Some(deg) = in_degree.get_mut(neighbor.as_str()) {
                        *deg -= 1;
                        if *deg == 0 {
                            queue.push_back(neighbor.as_str());
                        }
                    }
                }
            }
        }

        if result.len() != self.nodes.len() {
            Err(LangGraphError::Other(
                "Graph contains a cycle; topological sort is not possible".to_string(),
            ))
        } else {
            Ok(result)
        }
    }

    /// Return successors of a node.
    pub fn successors(&self, node: &str) -> Vec<&str> {
        self.forward
            .get(node)
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Return predecessors of a node.
    pub fn predecessors(&self, node: &str) -> Vec<&str> {
        self.reverse
            .get(node)
            .map(|v| v.iter().map(|s| s.as_str()).collect())
            .unwrap_or_default()
    }

    /// Check whether the graph contains a node with the given name.
    pub fn has_node(&self, name: &str) -> bool {
        self.nodes.contains(name)
    }

    /// Check whether the graph contains an edge from `from` to `to`.
    pub fn has_edge(&self, from: &str, to: &str) -> bool {
        self.edges.iter().any(|(f, t)| f == from && t == to)
    }
}

impl Default for GraphTopology {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PathFinder
// ---------------------------------------------------------------------------

/// Finds paths between nodes in a graph topology.
pub struct PathFinder<'a> {
    topology: &'a GraphTopology,
}

impl<'a> PathFinder<'a> {
    /// Create a new path finder for the given topology.
    pub fn new(topology: &'a GraphTopology) -> Self {
        Self { topology }
    }

    /// Find all paths from `from` to `to` using DFS.
    ///
    /// Warning: for graphs with many paths this can be expensive.
    pub fn all_paths(&self, from: &str, to: &str) -> Vec<Vec<String>> {
        let mut results = Vec::new();
        let mut path = vec![from.to_string()];
        let mut visited = HashSet::new();
        visited.insert(from.to_string());
        self.dfs_all_paths(from, to, &mut path, &mut visited, &mut results);
        results
    }

    fn dfs_all_paths(
        &self,
        current: &str,
        target: &str,
        path: &mut Vec<String>,
        visited: &mut HashSet<String>,
        results: &mut Vec<Vec<String>>,
    ) {
        if current == target {
            results.push(path.clone());
            return;
        }

        for neighbor in self.topology.successors(current) {
            if !visited.contains(neighbor) {
                visited.insert(neighbor.to_string());
                path.push(neighbor.to_string());
                self.dfs_all_paths(neighbor, target, path, visited, results);
                path.pop();
                visited.remove(neighbor);
            }
        }
    }

    /// Find the shortest path from `from` to `to` using BFS.
    ///
    /// Returns `None` if no path exists.
    pub fn shortest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        if from == to {
            return Some(vec![from.to_string()]);
        }
        if !self.topology.has_node(from) || !self.topology.has_node(to) {
            return None;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut parent: HashMap<String, String> = HashMap::new();

        visited.insert(from.to_string());
        queue.push_back(from.to_string());

        while let Some(current) = queue.pop_front() {
            for neighbor in self.topology.successors(&current) {
                if !visited.contains(neighbor) {
                    visited.insert(neighbor.to_string());
                    parent.insert(neighbor.to_string(), current.clone());
                    if neighbor == to {
                        return Some(self.reconstruct_path(&parent, from, to));
                    }
                    queue.push_back(neighbor.to_string());
                }
            }
        }

        None
    }

    fn reconstruct_path(
        &self,
        parent: &HashMap<String, String>,
        from: &str,
        to: &str,
    ) -> Vec<String> {
        let mut path = Vec::new();
        let mut current = to.to_string();
        while current != from {
            path.push(current.clone());
            current = parent[&current].clone();
        }
        path.push(from.to_string());
        path.reverse();
        path
    }

    /// Find the longest path from `from` to `to` (by number of edges).
    ///
    /// Returns `None` if no path exists.
    /// Uses DFS to enumerate all paths and returns the longest one.
    pub fn longest_path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        let paths = self.all_paths(from, to);
        paths.into_iter().max_by_key(|p| p.len())
    }

    /// Check whether `to` is reachable from `from`.
    pub fn is_reachable(&self, from: &str, to: &str) -> bool {
        if from == to {
            return true;
        }
        if !self.topology.has_node(from) || !self.topology.has_node(to) {
            return false;
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(from.to_string());
        queue.push_back(from.to_string());

        while let Some(current) = queue.pop_front() {
            for neighbor in self.topology.successors(&current) {
                if neighbor == to {
                    return true;
                }
                if visited.insert(neighbor.to_string()) {
                    queue.push_back(neighbor.to_string());
                }
            }
        }

        false
    }
}

// ---------------------------------------------------------------------------
// GraphMetrics
// ---------------------------------------------------------------------------

/// Computed structural metrics for a graph topology.
#[derive(Debug, Clone)]
pub struct GraphMetrics {
    /// Number of nodes.
    pub node_count: usize,
    /// Number of edges.
    pub edge_count: usize,
    /// Graph density: edges / (nodes * (nodes - 1)) for directed graphs.
    pub density: f64,
    /// Average in-degree across all nodes.
    pub avg_in_degree: f64,
    /// Average out-degree across all nodes.
    pub avg_out_degree: f64,
    /// Maximum depth from any source to any sink (longest path in DAG, 0 if cyclic or empty).
    pub max_depth: usize,
    /// Whether the graph is weakly connected (ignoring edge direction).
    pub is_connected: bool,
}

impl GraphMetrics {
    /// Compute metrics from a graph topology.
    pub fn from_topology(topology: &GraphTopology) -> Self {
        let node_count = topology.node_count();
        let edge_count = topology.edge_count();

        let density = if node_count > 1 {
            edge_count as f64 / (node_count as f64 * (node_count as f64 - 1.0))
        } else {
            0.0
        };

        let avg_in_degree = if node_count > 0 {
            edge_count as f64 / node_count as f64
        } else {
            0.0
        };

        // For directed graphs, avg_in_degree == avg_out_degree
        let avg_out_degree = avg_in_degree;

        let max_depth = Self::compute_max_depth(topology);
        let is_connected = Self::check_connected(topology);

        Self {
            node_count,
            edge_count,
            density,
            avg_in_degree,
            avg_out_degree,
            max_depth,
            is_connected,
        }
    }

    fn compute_max_depth(topology: &GraphTopology) -> usize {
        // Only meaningful for DAGs
        let sorted = match topology.topological_sort() {
            Ok(s) => s,
            Err(_) => return 0,
        };

        let mut dist: HashMap<&str, usize> = HashMap::new();
        for node in &sorted {
            dist.insert(node.as_str(), 0);
        }

        for node in &sorted {
            let current_dist = dist[node.as_str()];
            if let Some(neighbors) = topology.forward.get(node.as_str()) {
                for neighbor in neighbors {
                    let new_dist = current_dist + 1;
                    let entry = dist.entry(neighbor.as_str()).or_insert(0);
                    if new_dist > *entry {
                        *entry = new_dist;
                    }
                }
            }
        }

        dist.values().copied().max().unwrap_or(0)
    }

    fn check_connected(topology: &GraphTopology) -> bool {
        if topology.node_count() == 0 {
            return true;
        }

        // Build undirected adjacency
        let mut undirected: HashMap<&str, HashSet<&str>> = HashMap::new();
        for node in &topology.nodes {
            undirected.entry(node.as_str()).or_default();
        }
        for (from, to) in &topology.edges {
            undirected
                .entry(from.as_str())
                .or_default()
                .insert(to.as_str());
            undirected
                .entry(to.as_str())
                .or_default()
                .insert(from.as_str());
        }

        // BFS from first node
        let start = topology.nodes.iter().next().unwrap().as_str();
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(start);
        queue.push_back(start);

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = undirected.get(current) {
                for &neighbor in neighbors {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        visited.len() == topology.node_count()
    }

    /// Serialize metrics to a JSON string.
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"node_count":{},"edge_count":{},"density":{:.4},"avg_in_degree":{:.4},"avg_out_degree":{:.4},"max_depth":{},"is_connected":{}}}"#,
            self.node_count,
            self.edge_count,
            self.density,
            self.avg_in_degree,
            self.avg_out_degree,
            self.max_depth,
            self.is_connected,
        )
    }
}

// ---------------------------------------------------------------------------
// CycleDetector
// ---------------------------------------------------------------------------

/// Finds all cycles in a graph topology.
///
/// Note: This is distinct from [`super::validator::CycleDetector`] which only
/// detects the presence of a single cycle. This detector finds *all* elementary
/// cycles using Johnson's algorithm approach (simplified DFS).
pub struct CycleDetector;

impl CycleDetector {
    /// Detect all elementary cycles in the graph.
    ///
    /// Returns a list of cycles, where each cycle is a list of node names
    /// forming the cycle path (first and last elements are the same).
    pub fn detect(topology: &GraphTopology) -> Vec<Vec<String>> {
        let mut cycles = Vec::new();
        let nodes: Vec<&str> = topology.node_names();

        for &start in &nodes {
            let mut visited = HashSet::new();
            let mut path = vec![start.to_string()];
            visited.insert(start.to_string());
            Self::dfs_cycles(topology, start, start, &mut path, &mut visited, &mut cycles);
        }

        // Deduplicate cycles: normalize each cycle by rotating to smallest element
        let mut unique: Vec<Vec<String>> = Vec::new();
        let mut seen: HashSet<Vec<String>> = HashSet::new();

        for cycle in cycles {
            let normalized = Self::normalize_cycle(&cycle);
            if seen.insert(normalized.clone()) {
                unique.push(normalized);
            }
        }

        unique
    }

    fn dfs_cycles(
        topology: &GraphTopology,
        start: &str,
        current: &str,
        path: &mut Vec<String>,
        visited: &mut HashSet<String>,
        cycles: &mut Vec<Vec<String>>,
    ) {
        for neighbor in topology.successors(current) {
            if neighbor == start {
                // Found a cycle back to start (includes self-loops where current == start)
                let mut cycle = path.clone();
                cycle.push(start.to_string());
                cycles.push(cycle);
            } else if !visited.contains(neighbor) {
                visited.insert(neighbor.to_string());
                path.push(neighbor.to_string());
                Self::dfs_cycles(topology, start, neighbor, path, visited, cycles);
                path.pop();
                visited.remove(neighbor);
            }
        }
    }

    fn normalize_cycle(cycle: &[String]) -> Vec<String> {
        if cycle.len() <= 1 {
            return cycle.to_vec();
        }
        // Remove the repeated last element for rotation
        let body = &cycle[..cycle.len() - 1];
        // Find the minimum element's position
        let min_pos = body
            .iter()
            .enumerate()
            .min_by_key(|(_, v)| v.as_str())
            .map(|(i, _)| i)
            .unwrap_or(0);
        // Rotate so minimum is first
        let mut normalized: Vec<String> = body[min_pos..].to_vec();
        normalized.extend_from_slice(&body[..min_pos]);
        // Add the closing element
        normalized.push(normalized[0].clone());
        normalized
    }
}

// ---------------------------------------------------------------------------
// GraphDiff
// ---------------------------------------------------------------------------

/// Result of comparing two graph topologies.
#[derive(Debug, Clone)]
pub struct GraphDiff {
    /// Nodes present in `b` but not in `a`.
    pub added_nodes: Vec<String>,
    /// Nodes present in `a` but not in `b`.
    pub removed_nodes: Vec<String>,
    /// Edges present in `b` but not in `a`.
    pub added_edges: Vec<(String, String)>,
    /// Edges present in `a` but not in `b`.
    pub removed_edges: Vec<(String, String)>,
}

impl GraphDiff {
    /// Compare two graph topologies and produce a diff.
    pub fn diff(a: &GraphTopology, b: &GraphTopology) -> Self {
        let a_nodes: BTreeSet<&str> = a.nodes.iter().map(|s| s.as_str()).collect();
        let b_nodes: BTreeSet<&str> = b.nodes.iter().map(|s| s.as_str()).collect();

        let added_nodes: Vec<String> = b_nodes
            .difference(&a_nodes)
            .map(|s| s.to_string())
            .collect();
        let removed_nodes: Vec<String> = a_nodes
            .difference(&b_nodes)
            .map(|s| s.to_string())
            .collect();

        let a_edges: HashSet<(&str, &str)> = a
            .edges
            .iter()
            .map(|(f, t)| (f.as_str(), t.as_str()))
            .collect();
        let b_edges: HashSet<(&str, &str)> = b
            .edges
            .iter()
            .map(|(f, t)| (f.as_str(), t.as_str()))
            .collect();

        let added_edges: Vec<(String, String)> = b_edges
            .difference(&a_edges)
            .map(|(f, t)| (f.to_string(), t.to_string()))
            .collect();
        let removed_edges: Vec<(String, String)> = a_edges
            .difference(&b_edges)
            .map(|(f, t)| (f.to_string(), t.to_string()))
            .collect();

        Self {
            added_nodes,
            removed_nodes,
            added_edges,
            removed_edges,
        }
    }

    /// Returns `true` if any nodes or edges were added or removed.
    pub fn has_changes(&self) -> bool {
        !self.added_nodes.is_empty()
            || !self.removed_nodes.is_empty()
            || !self.added_edges.is_empty()
            || !self.removed_edges.is_empty()
    }

    /// Serialize the diff to a JSON string.
    pub fn to_json(&self) -> String {
        let added_nodes = self
            .added_nodes
            .iter()
            .map(|n| format!("\"{}\"", n))
            .collect::<Vec<_>>()
            .join(",");
        let removed_nodes = self
            .removed_nodes
            .iter()
            .map(|n| format!("\"{}\"", n))
            .collect::<Vec<_>>()
            .join(",");
        let added_edges = self
            .added_edges
            .iter()
            .map(|(f, t)| format!("[\"{}\",\"{}\"]", f, t))
            .collect::<Vec<_>>()
            .join(",");
        let removed_edges = self
            .removed_edges
            .iter()
            .map(|(f, t)| format!("[\"{}\",\"{}\"]", f, t))
            .collect::<Vec<_>>()
            .join(",");

        format!(
            r#"{{"added_nodes":[{}],"removed_nodes":[{}],"added_edges":[{}],"removed_edges":[{}]}}"#,
            added_nodes, removed_nodes, added_edges, removed_edges,
        )
    }
}

// ---------------------------------------------------------------------------
// SubgraphExtractor
// ---------------------------------------------------------------------------

/// Extracts subgraphs from a graph topology.
pub struct SubgraphExtractor;

impl SubgraphExtractor {
    /// Extract a subgraph containing only the specified nodes and edges between them.
    pub fn extract(topology: &GraphTopology, nodes: &[&str]) -> GraphTopology {
        let node_set: HashSet<&str> = nodes.iter().copied().collect();
        let mut sub = GraphTopology::new();

        for &node in nodes {
            if topology.has_node(node) {
                sub.add_node(node);
            }
        }

        for (from, to) in &topology.edges {
            if node_set.contains(from.as_str()) && node_set.contains(to.as_str()) {
                sub.add_edge(from, to);
            }
        }

        sub
    }

    /// Extract all nodes reachable from `node` (including the node itself) and edges between them.
    pub fn downstream(topology: &GraphTopology, node: &str) -> GraphTopology {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        if topology.has_node(node) {
            visited.insert(node.to_string());
            queue.push_back(node.to_string());
        }

        while let Some(current) = queue.pop_front() {
            for neighbor in topology.successors(&current) {
                if visited.insert(neighbor.to_string()) {
                    queue.push_back(neighbor.to_string());
                }
            }
        }

        let node_refs: Vec<&str> = visited.iter().map(|s| s.as_str()).collect();
        Self::extract(topology, &node_refs)
    }
}

// ---------------------------------------------------------------------------
// GraphVisualizer
// ---------------------------------------------------------------------------

/// Generates visual representations of a graph topology.
pub struct GraphVisualizer;

impl GraphVisualizer {
    /// Generate a DOT (Graphviz) representation of the graph.
    pub fn to_dot(topology: &GraphTopology) -> String {
        let mut lines = Vec::new();
        lines.push("digraph G {".to_string());
        lines.push("    rankdir=TB;".to_string());

        for node in topology.node_names() {
            lines.push(format!("    \"{}\";", node));
        }

        for (from, to) in &topology.edges {
            lines.push(format!("    \"{}\" -> \"{}\";", from, to));
        }

        lines.push("}".to_string());
        lines.join("\n")
    }

    /// Generate a Mermaid diagram representation of the graph.
    pub fn to_mermaid(topology: &GraphTopology) -> String {
        let mut lines = Vec::new();
        lines.push("graph TD".to_string());

        for node in topology.node_names() {
            lines.push(format!("    {}[{}]", node, node));
        }

        for (from, to) in &topology.edges {
            lines.push(format!("    {} --> {}", from, to));
        }

        lines.join("\n")
    }

    /// Generate a simple ASCII text representation of the graph.
    pub fn to_ascii(topology: &GraphTopology) -> String {
        let mut lines = Vec::new();
        lines.push("Graph:".to_string());
        lines.push(format!(
            "  Nodes ({}): {}",
            topology.node_count(),
            topology.node_names().join(", ")
        ));
        lines.push(format!("  Edges ({}):", topology.edge_count()));

        for (from, to) in &topology.edges {
            lines.push(format!("    {} -> {}", from, to));
        }

        let sources = topology.sources();
        if !sources.is_empty() {
            lines.push(format!("  Sources: {}", sources.join(", ")));
        }

        let sinks = topology.sinks();
        if !sinks.is_empty() {
            lines.push(format!("  Sinks: {}", sinks.join(", ")));
        }

        lines.join("\n")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- helpers --

    /// Build a simple linear graph: a -> b -> c
    fn linear_graph() -> GraphTopology {
        let mut g = GraphTopology::new();
        g.add_edge("a", "b");
        g.add_edge("b", "c");
        g
    }

    /// Build a diamond graph: a -> b, a -> c, b -> d, c -> d
    fn diamond_graph() -> GraphTopology {
        let mut g = GraphTopology::new();
        g.add_edge("a", "b");
        g.add_edge("a", "c");
        g.add_edge("b", "d");
        g.add_edge("c", "d");
        g
    }

    /// Build a graph with a cycle: a -> b -> c -> a
    fn cyclic_graph() -> GraphTopology {
        let mut g = GraphTopology::new();
        g.add_edge("a", "b");
        g.add_edge("b", "c");
        g.add_edge("c", "a");
        g
    }

    // -----------------------------------------------------------------------
    // GraphTopology — construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_empty_graph() {
        let g = GraphTopology::new();
        assert_eq!(g.node_count(), 0);
        assert_eq!(g.edge_count(), 0);
        assert!(g.sources().is_empty());
        assert!(g.sinks().is_empty());
        assert!(g.is_dag());
    }

    #[test]
    fn test_single_node() {
        let mut g = GraphTopology::new();
        g.add_node("lonely");
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
        assert_eq!(g.sources(), vec!["lonely"]);
        assert_eq!(g.sinks(), vec!["lonely"]);
        assert!(g.is_dag());
    }

    #[test]
    fn test_add_duplicate_node() {
        let mut g = GraphTopology::new();
        g.add_node("a");
        g.add_node("a");
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_add_edge_adds_nodes() {
        let mut g = GraphTopology::new();
        g.add_edge("x", "y");
        assert!(g.has_node("x"));
        assert!(g.has_node("y"));
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    // -----------------------------------------------------------------------
    // GraphTopology — in/out degree
    // -----------------------------------------------------------------------

    #[test]
    fn test_in_degree() {
        let g = diamond_graph();
        assert_eq!(g.in_degree("a"), 0);
        assert_eq!(g.in_degree("b"), 1);
        assert_eq!(g.in_degree("c"), 1);
        assert_eq!(g.in_degree("d"), 2);
    }

    #[test]
    fn test_out_degree() {
        let g = diamond_graph();
        assert_eq!(g.out_degree("a"), 2);
        assert_eq!(g.out_degree("b"), 1);
        assert_eq!(g.out_degree("c"), 1);
        assert_eq!(g.out_degree("d"), 0);
    }

    #[test]
    fn test_degree_nonexistent_node() {
        let g = GraphTopology::new();
        assert_eq!(g.in_degree("ghost"), 0);
        assert_eq!(g.out_degree("ghost"), 0);
    }

    // -----------------------------------------------------------------------
    // GraphTopology — sources and sinks
    // -----------------------------------------------------------------------

    #[test]
    fn test_sources_linear() {
        let g = linear_graph();
        assert_eq!(g.sources(), vec!["a"]);
    }

    #[test]
    fn test_sinks_linear() {
        let g = linear_graph();
        assert_eq!(g.sinks(), vec!["c"]);
    }

    #[test]
    fn test_sources_diamond() {
        let g = diamond_graph();
        assert_eq!(g.sources(), vec!["a"]);
    }

    #[test]
    fn test_sinks_diamond() {
        let g = diamond_graph();
        assert_eq!(g.sinks(), vec!["d"]);
    }

    #[test]
    fn test_cyclic_no_sources_no_sinks() {
        let g = cyclic_graph();
        // In a fully cyclic graph, every node has in-degree and out-degree > 0
        assert!(g.sources().is_empty());
        assert!(g.sinks().is_empty());
    }

    // -----------------------------------------------------------------------
    // GraphTopology — DAG detection
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_dag_linear() {
        assert!(linear_graph().is_dag());
    }

    #[test]
    fn test_is_dag_diamond() {
        assert!(diamond_graph().is_dag());
    }

    #[test]
    fn test_is_not_dag_cyclic() {
        assert!(!cyclic_graph().is_dag());
    }

    #[test]
    fn test_is_dag_empty() {
        assert!(GraphTopology::new().is_dag());
    }

    // -----------------------------------------------------------------------
    // GraphTopology — topological sort
    // -----------------------------------------------------------------------

    #[test]
    fn test_topological_sort_linear() {
        let g = linear_graph();
        let sorted = g.topological_sort().unwrap();
        assert_eq!(sorted.len(), 3);
        // a must come before b, b before c
        let pos_a = sorted.iter().position(|n| n == "a").unwrap();
        let pos_b = sorted.iter().position(|n| n == "b").unwrap();
        let pos_c = sorted.iter().position(|n| n == "c").unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_topological_sort_diamond() {
        let g = diamond_graph();
        let sorted = g.topological_sort().unwrap();
        assert_eq!(sorted.len(), 4);
        let pos_a = sorted.iter().position(|n| n == "a").unwrap();
        let pos_b = sorted.iter().position(|n| n == "b").unwrap();
        let pos_c = sorted.iter().position(|n| n == "c").unwrap();
        let pos_d = sorted.iter().position(|n| n == "d").unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    #[test]
    fn test_topological_sort_cyclic_error() {
        let g = cyclic_graph();
        assert!(g.topological_sort().is_err());
    }

    // -----------------------------------------------------------------------
    // PathFinder — all paths
    // -----------------------------------------------------------------------

    #[test]
    fn test_all_paths_linear() {
        let g = linear_graph();
        let pf = PathFinder::new(&g);
        let paths = pf.all_paths("a", "c");
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec!["a", "b", "c"]);
    }

    #[test]
    fn test_all_paths_diamond() {
        let g = diamond_graph();
        let pf = PathFinder::new(&g);
        let paths = pf.all_paths("a", "d");
        assert_eq!(paths.len(), 2);
        // Both paths: a->b->d and a->c->d
        let path_strs: HashSet<String> = paths
            .iter()
            .map(|p| p.join("->"))
            .collect();
        assert!(path_strs.contains("a->b->d"));
        assert!(path_strs.contains("a->c->d"));
    }

    #[test]
    fn test_all_paths_no_path() {
        let g = linear_graph();
        let pf = PathFinder::new(&g);
        let paths = pf.all_paths("c", "a");
        assert!(paths.is_empty());
    }

    #[test]
    fn test_all_paths_same_node() {
        let g = linear_graph();
        let pf = PathFinder::new(&g);
        let paths = pf.all_paths("a", "a");
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec!["a"]);
    }

    // -----------------------------------------------------------------------
    // PathFinder — shortest path
    // -----------------------------------------------------------------------

    #[test]
    fn test_shortest_path_linear() {
        let g = linear_graph();
        let pf = PathFinder::new(&g);
        let path = pf.shortest_path("a", "c").unwrap();
        assert_eq!(path, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_shortest_path_diamond() {
        let g = diamond_graph();
        let pf = PathFinder::new(&g);
        let path = pf.shortest_path("a", "d").unwrap();
        // Both a->b->d and a->c->d are length 3, either is valid
        assert_eq!(path.len(), 3);
        assert_eq!(path[0], "a");
        assert_eq!(path[2], "d");
    }

    #[test]
    fn test_shortest_path_no_path() {
        let g = linear_graph();
        let pf = PathFinder::new(&g);
        assert!(pf.shortest_path("c", "a").is_none());
    }

    #[test]
    fn test_shortest_path_same_node() {
        let g = linear_graph();
        let pf = PathFinder::new(&g);
        let path = pf.shortest_path("b", "b").unwrap();
        assert_eq!(path, vec!["b"]);
    }

    // -----------------------------------------------------------------------
    // PathFinder — longest path
    // -----------------------------------------------------------------------

    #[test]
    fn test_longest_path_linear() {
        let g = linear_graph();
        let pf = PathFinder::new(&g);
        let path = pf.longest_path("a", "c").unwrap();
        assert_eq!(path, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_longest_path_with_extra_path() {
        // a -> b -> d, a -> b -> c -> d
        let mut g = GraphTopology::new();
        g.add_edge("a", "b");
        g.add_edge("b", "c");
        g.add_edge("b", "d");
        g.add_edge("c", "d");
        let pf = PathFinder::new(&g);
        let path = pf.longest_path("a", "d").unwrap();
        assert_eq!(path, vec!["a", "b", "c", "d"]);
    }

    #[test]
    fn test_longest_path_no_path() {
        let g = linear_graph();
        let pf = PathFinder::new(&g);
        assert!(pf.longest_path("c", "a").is_none());
    }

    // -----------------------------------------------------------------------
    // PathFinder — reachability
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_reachable_linear() {
        let g = linear_graph();
        let pf = PathFinder::new(&g);
        assert!(pf.is_reachable("a", "c"));
        assert!(pf.is_reachable("a", "b"));
        assert!(!pf.is_reachable("c", "a"));
    }

    #[test]
    fn test_is_reachable_same_node() {
        let g = linear_graph();
        let pf = PathFinder::new(&g);
        assert!(pf.is_reachable("a", "a"));
    }

    #[test]
    fn test_is_reachable_nonexistent() {
        let g = linear_graph();
        let pf = PathFinder::new(&g);
        assert!(!pf.is_reachable("a", "ghost"));
        assert!(!pf.is_reachable("ghost", "a"));
    }

    // -----------------------------------------------------------------------
    // GraphMetrics
    // -----------------------------------------------------------------------

    #[test]
    fn test_metrics_empty_graph() {
        let g = GraphTopology::new();
        let m = GraphMetrics::from_topology(&g);
        assert_eq!(m.node_count, 0);
        assert_eq!(m.edge_count, 0);
        assert_eq!(m.density, 0.0);
        assert_eq!(m.avg_in_degree, 0.0);
        assert_eq!(m.max_depth, 0);
        assert!(m.is_connected);
    }

    #[test]
    fn test_metrics_linear() {
        let g = linear_graph();
        let m = GraphMetrics::from_topology(&g);
        assert_eq!(m.node_count, 3);
        assert_eq!(m.edge_count, 2);
        // density = 2 / (3 * 2) = 0.3333...
        assert!((m.density - 1.0 / 3.0).abs() < 1e-6);
        assert!((m.avg_in_degree - 2.0 / 3.0).abs() < 1e-6);
        assert_eq!(m.max_depth, 2);
        assert!(m.is_connected);
    }

    #[test]
    fn test_metrics_diamond() {
        let g = diamond_graph();
        let m = GraphMetrics::from_topology(&g);
        assert_eq!(m.node_count, 4);
        assert_eq!(m.edge_count, 4);
        // density = 4 / (4 * 3) = 1/3
        assert!((m.density - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(m.max_depth, 2);
        assert!(m.is_connected);
    }

    #[test]
    fn test_metrics_disconnected() {
        let mut g = GraphTopology::new();
        g.add_node("a");
        g.add_node("b");
        // No edges, two isolated nodes
        let m = GraphMetrics::from_topology(&g);
        assert!(!m.is_connected);
    }

    #[test]
    fn test_metrics_to_json() {
        let g = linear_graph();
        let m = GraphMetrics::from_topology(&g);
        let json = m.to_json();
        assert!(json.contains("\"node_count\":3"));
        assert!(json.contains("\"edge_count\":2"));
        assert!(json.contains("\"is_connected\":true"));
    }

    // -----------------------------------------------------------------------
    // CycleDetector
    // -----------------------------------------------------------------------

    #[test]
    fn test_cycle_detector_no_cycles() {
        let g = linear_graph();
        let cycles = CycleDetector::detect(&g);
        assert!(cycles.is_empty());
    }

    #[test]
    fn test_cycle_detector_simple_cycle() {
        let g = cyclic_graph();
        let cycles = CycleDetector::detect(&g);
        assert_eq!(cycles.len(), 1);
        // The cycle should contain a, b, c
        let cycle = &cycles[0];
        assert!(cycle.contains(&"a".to_string()));
        assert!(cycle.contains(&"b".to_string()));
        assert!(cycle.contains(&"c".to_string()));
        // First and last should be the same
        assert_eq!(cycle.first(), cycle.last());
    }

    #[test]
    fn test_cycle_detector_self_loop() {
        let mut g = GraphTopology::new();
        g.add_edge("a", "a");
        let cycles = CycleDetector::detect(&g);
        assert_eq!(cycles.len(), 1);
        assert_eq!(cycles[0], vec!["a", "a"]);
    }

    #[test]
    fn test_cycle_detector_multiple_cycles() {
        let mut g = GraphTopology::new();
        g.add_edge("a", "b");
        g.add_edge("b", "a");
        g.add_edge("c", "d");
        g.add_edge("d", "c");
        let cycles = CycleDetector::detect(&g);
        assert_eq!(cycles.len(), 2);
    }

    #[test]
    fn test_cycle_detector_dag() {
        let g = diamond_graph();
        let cycles = CycleDetector::detect(&g);
        assert!(cycles.is_empty());
    }

    // -----------------------------------------------------------------------
    // GraphDiff
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_identical_graphs() {
        let g = linear_graph();
        let diff = GraphDiff::diff(&g, &g);
        assert!(!diff.has_changes());
    }

    #[test]
    fn test_diff_added_node_and_edge() {
        let a = linear_graph();
        let mut b = linear_graph();
        b.add_edge("c", "d");

        let diff = GraphDiff::diff(&a, &b);
        assert!(diff.has_changes());
        assert!(diff.added_nodes.contains(&"d".to_string()));
        assert!(diff.removed_nodes.is_empty());
        assert_eq!(diff.added_edges.len(), 1);
        assert!(diff.removed_edges.is_empty());
    }

    #[test]
    fn test_diff_removed_node_and_edge() {
        let a = linear_graph();
        let mut b = GraphTopology::new();
        b.add_edge("a", "b");

        let diff = GraphDiff::diff(&a, &b);
        assert!(diff.has_changes());
        assert!(diff.removed_nodes.contains(&"c".to_string()));
        assert!(diff.removed_edges.len() == 1);
    }

    #[test]
    fn test_diff_to_json() {
        let a = linear_graph();
        let b = linear_graph();
        let diff = GraphDiff::diff(&a, &b);
        let json = diff.to_json();
        assert!(json.contains("\"added_nodes\":[]"));
        assert!(json.contains("\"removed_nodes\":[]"));
    }

    #[test]
    fn test_diff_empty_graphs() {
        let a = GraphTopology::new();
        let b = GraphTopology::new();
        let diff = GraphDiff::diff(&a, &b);
        assert!(!diff.has_changes());
    }

    // -----------------------------------------------------------------------
    // SubgraphExtractor
    // -----------------------------------------------------------------------

    #[test]
    fn test_extract_subgraph() {
        let g = diamond_graph();
        let sub = SubgraphExtractor::extract(&g, &["a", "b", "d"]);
        assert_eq!(sub.node_count(), 3);
        assert!(sub.has_node("a"));
        assert!(sub.has_node("b"));
        assert!(sub.has_node("d"));
        assert!(!sub.has_node("c"));
        // Edge a->b and b->d should be kept, a->c and c->d should be dropped
        assert!(sub.has_edge("a", "b"));
        assert!(sub.has_edge("b", "d"));
        assert!(!sub.has_edge("a", "c"));
        assert!(!sub.has_edge("c", "d"));
    }

    #[test]
    fn test_extract_subgraph_empty_nodes() {
        let g = linear_graph();
        let sub = SubgraphExtractor::extract(&g, &[]);
        assert_eq!(sub.node_count(), 0);
        assert_eq!(sub.edge_count(), 0);
    }

    #[test]
    fn test_extract_nonexistent_nodes() {
        let g = linear_graph();
        let sub = SubgraphExtractor::extract(&g, &["ghost"]);
        assert_eq!(sub.node_count(), 0);
    }

    #[test]
    fn test_downstream_from_root() {
        let g = diamond_graph();
        let sub = SubgraphExtractor::downstream(&g, "a");
        assert_eq!(sub.node_count(), 4);
        assert_eq!(sub.edge_count(), 4);
    }

    #[test]
    fn test_downstream_from_middle() {
        let g = diamond_graph();
        let sub = SubgraphExtractor::downstream(&g, "b");
        assert_eq!(sub.node_count(), 2);
        assert!(sub.has_node("b"));
        assert!(sub.has_node("d"));
        assert!(sub.has_edge("b", "d"));
    }

    #[test]
    fn test_downstream_from_leaf() {
        let g = diamond_graph();
        let sub = SubgraphExtractor::downstream(&g, "d");
        assert_eq!(sub.node_count(), 1);
        assert!(sub.has_node("d"));
        assert_eq!(sub.edge_count(), 0);
    }

    #[test]
    fn test_downstream_nonexistent() {
        let g = linear_graph();
        let sub = SubgraphExtractor::downstream(&g, "ghost");
        assert_eq!(sub.node_count(), 0);
    }

    // -----------------------------------------------------------------------
    // GraphVisualizer — DOT
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_dot_contains_digraph() {
        let g = linear_graph();
        let dot = GraphVisualizer::to_dot(&g);
        assert!(dot.contains("digraph G {"));
        assert!(dot.contains("}"));
    }

    #[test]
    fn test_to_dot_contains_nodes() {
        let g = linear_graph();
        let dot = GraphVisualizer::to_dot(&g);
        assert!(dot.contains("\"a\""));
        assert!(dot.contains("\"b\""));
        assert!(dot.contains("\"c\""));
    }

    #[test]
    fn test_to_dot_contains_edges() {
        let g = linear_graph();
        let dot = GraphVisualizer::to_dot(&g);
        assert!(dot.contains("\"a\" -> \"b\""));
        assert!(dot.contains("\"b\" -> \"c\""));
    }

    #[test]
    fn test_to_dot_empty_graph() {
        let g = GraphTopology::new();
        let dot = GraphVisualizer::to_dot(&g);
        assert!(dot.contains("digraph G {"));
        assert!(dot.contains("}"));
    }

    // -----------------------------------------------------------------------
    // GraphVisualizer — Mermaid
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_mermaid_header() {
        let g = linear_graph();
        let mermaid = GraphVisualizer::to_mermaid(&g);
        assert!(mermaid.starts_with("graph TD"));
    }

    #[test]
    fn test_to_mermaid_contains_edges() {
        let g = linear_graph();
        let mermaid = GraphVisualizer::to_mermaid(&g);
        assert!(mermaid.contains("a --> b"));
        assert!(mermaid.contains("b --> c"));
    }

    #[test]
    fn test_to_mermaid_contains_nodes() {
        let g = linear_graph();
        let mermaid = GraphVisualizer::to_mermaid(&g);
        assert!(mermaid.contains("a[a]"));
        assert!(mermaid.contains("b[b]"));
    }

    // -----------------------------------------------------------------------
    // GraphVisualizer — ASCII
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_ascii_header() {
        let g = linear_graph();
        let ascii = GraphVisualizer::to_ascii(&g);
        assert!(ascii.starts_with("Graph:"));
    }

    #[test]
    fn test_to_ascii_contains_node_count() {
        let g = linear_graph();
        let ascii = GraphVisualizer::to_ascii(&g);
        assert!(ascii.contains("Nodes (3)"));
    }

    #[test]
    fn test_to_ascii_contains_edges() {
        let g = linear_graph();
        let ascii = GraphVisualizer::to_ascii(&g);
        assert!(ascii.contains("a -> b"));
        assert!(ascii.contains("b -> c"));
    }

    #[test]
    fn test_to_ascii_contains_sources_and_sinks() {
        let g = linear_graph();
        let ascii = GraphVisualizer::to_ascii(&g);
        assert!(ascii.contains("Sources: a"));
        assert!(ascii.contains("Sinks: c"));
    }

    // -----------------------------------------------------------------------
    // Edge cases — disconnected components
    // -----------------------------------------------------------------------

    #[test]
    fn test_disconnected_components() {
        let mut g = GraphTopology::new();
        g.add_edge("a", "b");
        g.add_edge("c", "d");

        assert!(g.is_dag());
        assert_eq!(g.sources(), vec!["a", "c"]);
        assert_eq!(g.sinks(), vec!["b", "d"]);

        let m = GraphMetrics::from_topology(&g);
        assert!(!m.is_connected);

        let pf = PathFinder::new(&g);
        assert!(!pf.is_reachable("a", "c"));
        assert!(!pf.is_reachable("a", "d"));
    }

    // -----------------------------------------------------------------------
    // Edge cases — self-loops
    // -----------------------------------------------------------------------

    #[test]
    fn test_self_loop_not_dag() {
        let mut g = GraphTopology::new();
        g.add_edge("a", "a");
        assert!(!g.is_dag());
    }

    #[test]
    fn test_self_loop_in_out_degree() {
        let mut g = GraphTopology::new();
        g.add_edge("a", "a");
        assert_eq!(g.in_degree("a"), 1);
        assert_eq!(g.out_degree("a"), 1);
        assert!(g.sources().is_empty());
        assert!(g.sinks().is_empty());
    }
}
