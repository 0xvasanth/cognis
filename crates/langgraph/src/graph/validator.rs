//! Comprehensive graph-level validation for LangGraph state graphs.
//!
//! This module provides structured validation that produces detailed
//! [`ValidationResult`]s with severity levels, error codes, and affected nodes.
//! It validates connectivity, termination, cycles, edge integrity, node names,
//! and conditional edge targets before a graph is compiled.
//!
//! # Example
//!
//! ```rust,ignore
//! use langgraph::graph::validator::GraphValidator;
//! use langgraph::graph::serialize::GraphDefinition;
//! use langgraph::constants::{START, END};
//!
//! let def = GraphDefinition {
//!     nodes: vec!["agent".into(), "tools".into()],
//!     edges: vec![
//!         (START.into(), "agent".into()),
//!         ("tools".into(), "agent".into()),
//!         ("agent".into(), END.into()),
//!     ],
//!     conditional_edges: vec![],
//!     entry_point: "agent".into(),
//!     interrupt_before: vec![],
//!     interrupt_after: vec![],
//! };
//!
//! let result = GraphValidator::validate_all(&def);
//! assert!(result.is_valid());
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

use crate::constants::{END, START};
use crate::graph::serialize::GraphDefinition;

/// Reserved node names that users must not use.
const RESERVED_NAMES: &[&str] = &[
    "__start__",
    "__end__",
    "__interrupt__",
    "__pregel_tasks",
    "__pregel_resume",
];

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Severity of a validation issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidationSeverity {
    /// A fatal problem that prevents compilation.
    Error,
    /// A potential problem that may cause unexpected behaviour.
    Warning,
    /// An informational note.
    Info,
}

impl fmt::Display for ValidationSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
            Self::Info => write!(f, "INFO"),
        }
    }
}

/// A single validation issue.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// How severe the issue is.
    pub severity: ValidationSeverity,
    /// A short, unique code such as `"UNREACHABLE_NODE"` or `"MISSING_EDGE"`.
    pub code: String,
    /// Human-readable description of the problem.
    pub message: String,
    /// The node most closely associated with the issue, if any.
    pub node: Option<String>,
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ref node) = self.node {
            write!(
                f,
                "[{}] {} (node: {}): {}",
                self.severity, self.code, node, self.message
            )
        } else {
            write!(f, "[{}] {}: {}", self.severity, self.code, self.message)
        }
    }
}

/// Aggregated result of one or more validation passes.
#[derive(Debug, Clone, Default)]
pub struct ValidationResult {
    /// All issues found during validation.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationResult {
    /// Create an empty (valid) result.
    pub fn new() -> Self {
        Self { issues: Vec::new() }
    }

    /// Returns `true` if no `Error`-level issues exist.
    pub fn is_valid(&self) -> bool {
        !self
            .issues
            .iter()
            .any(|i| i.severity == ValidationSeverity::Error)
    }

    /// Return all `Error`-level issues.
    pub fn errors(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == ValidationSeverity::Error)
            .collect()
    }

    /// Return all `Warning`-level issues.
    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == ValidationSeverity::Warning)
            .collect()
    }

    /// Return all `Info`-level issues.
    pub fn infos(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.severity == ValidationSeverity::Info)
            .collect()
    }

    /// Check whether any issue carries the given code.
    pub fn has_error_code(&self, code: &str) -> bool {
        self.issues.iter().any(|i| i.code == code)
    }

    /// Merge another result into this one.
    fn merge(&mut self, other: ValidationResult) {
        self.issues.extend(other.issues);
    }

    /// Push a single issue.
    fn push(&mut self, issue: ValidationIssue) {
        self.issues.push(issue);
    }
}

impl fmt::Display for ValidationResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.issues.is_empty() {
            write!(f, "Validation passed (no issues)")
        } else {
            for issue in &self.issues {
                writeln!(f, "{}", issue)?;
            }
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// CycleDetector
// ---------------------------------------------------------------------------

/// DFS-based cycle detector that can return the cycle path.
pub struct CycleDetector;

impl CycleDetector {
    /// Detect whether the graph contains any cycle.
    ///
    /// Returns `Some(path)` with the cycle path if a cycle is found, or `None`.
    /// The `adjacency` map should include all edges (direct + conditional targets).
    pub fn detect(
        nodes: &[String],
        adjacency: &HashMap<String, Vec<String>>,
    ) -> Option<Vec<String>> {
        // Build the full set of nodes including START/END so traversal works.
        let all: HashSet<&str> = nodes
            .iter()
            .map(String::as_str)
            .chain([START, END])
            .collect();

        let mut visited: HashSet<String> = HashSet::new();
        let mut on_stack: HashSet<String> = HashSet::new();
        let mut parent: HashMap<String, String> = HashMap::new();

        for node in &all {
            let node = node.to_string();
            if !visited.contains(&node) {
                if let Some(cycle) =
                    Self::dfs(&node, adjacency, &mut visited, &mut on_stack, &mut parent)
                {
                    return Some(cycle);
                }
            }
        }
        None
    }

    fn dfs(
        node: &str,
        adjacency: &HashMap<String, Vec<String>>,
        visited: &mut HashSet<String>,
        on_stack: &mut HashSet<String>,
        parent: &mut HashMap<String, String>,
    ) -> Option<Vec<String>> {
        visited.insert(node.to_string());
        on_stack.insert(node.to_string());

        if let Some(neighbors) = adjacency.get(node) {
            for neighbor in neighbors {
                if !visited.contains(neighbor) {
                    parent.insert(neighbor.clone(), node.to_string());
                    if let Some(cycle) = Self::dfs(neighbor, adjacency, visited, on_stack, parent) {
                        return Some(cycle);
                    }
                } else if on_stack.contains(neighbor) {
                    // Found a cycle — reconstruct the path.
                    let mut path = vec![neighbor.clone()];
                    let mut current = node.to_string();
                    while current != *neighbor {
                        path.push(current.clone());
                        current = parent.get(&current).cloned().unwrap_or_default();
                    }
                    path.push(neighbor.clone());
                    path.reverse();
                    return Some(path);
                }
            }
        }

        on_stack.remove(node);
        None
    }
}

// ---------------------------------------------------------------------------
// ReachabilityAnalyzer
// ---------------------------------------------------------------------------

/// BFS-based reachability analysis.
pub struct ReachabilityAnalyzer;

impl ReachabilityAnalyzer {
    /// Return the set of nodes reachable from `start` via the given adjacency map.
    pub fn reachable_from(
        start: &str,
        adjacency: &HashMap<String, Vec<String>>,
    ) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        visited.insert(start.to_string());
        queue.push_back(start.to_string());

        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = adjacency.get(&current) {
                for neighbor in neighbors {
                    if visited.insert(neighbor.clone()) {
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }
        visited
    }

    /// Find nodes that are *not* reachable from `start`.
    pub fn unreachable_nodes(
        nodes: &[String],
        start: &str,
        adjacency: &HashMap<String, Vec<String>>,
    ) -> Vec<String> {
        let reachable = Self::reachable_from(start, adjacency);
        nodes
            .iter()
            .filter(|n| n.as_str() != START && n.as_str() != END && !reachable.contains(n.as_str()))
            .cloned()
            .collect()
    }

    /// Find user nodes that cannot reach `target` (dead-end nodes).
    pub fn dead_end_nodes(
        nodes: &[String],
        target: &str,
        adjacency: &HashMap<String, Vec<String>>,
    ) -> Vec<String> {
        // Build reverse adjacency.
        let mut reverse: HashMap<String, Vec<String>> = HashMap::new();
        for (from, tos) in adjacency {
            for to in tos {
                reverse.entry(to.clone()).or_default().push(from.clone());
            }
        }

        let can_reach_end = Self::reachable_from(target, &reverse);
        nodes
            .iter()
            .filter(|n| {
                n.as_str() != START && n.as_str() != END && !can_reach_end.contains(n.as_str())
            })
            .cloned()
            .collect()
    }
}

// ---------------------------------------------------------------------------
// GraphValidator
// ---------------------------------------------------------------------------

/// Comprehensive graph validator.
///
/// All methods are stateless and take the graph topology as parameters.
pub struct GraphValidator;

impl GraphValidator {
    // -- helpers --

    /// Build a combined adjacency map from direct edges and conditional edge targets.
    fn build_adjacency(def: &GraphDefinition) -> HashMap<String, Vec<String>> {
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for (from, to) in &def.edges {
            adj.entry(from.clone()).or_default().push(to.clone());
        }
        for cond in &def.conditional_edges {
            for target in &cond.targets {
                adj.entry(cond.from.clone())
                    .or_default()
                    .push(target.clone());
            }
        }
        adj
    }

    // -- individual validators --

    /// Validate that every user node is reachable from START.
    pub fn validate_connectivity(nodes: &[String], edges: &[(String, String)]) -> ValidationResult {
        let mut result = ValidationResult::new();
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for (from, to) in edges {
            adj.entry(from.clone()).or_default().push(to.clone());
        }

        let unreachable = ReachabilityAnalyzer::unreachable_nodes(nodes, START, &adj);
        for node in unreachable {
            result.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "UNREACHABLE_NODE".into(),
                message: format!("Node '{}' is not reachable from START", node),
                node: Some(node),
            });
        }
        result
    }

    /// Validate that every user node can reach END.
    pub fn validate_termination(nodes: &[String], edges: &[(String, String)]) -> ValidationResult {
        let mut result = ValidationResult::new();
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for (from, to) in edges {
            adj.entry(from.clone()).or_default().push(to.clone());
        }

        let dead_ends = ReachabilityAnalyzer::dead_end_nodes(nodes, END, &adj);
        for node in dead_ends {
            result.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "NO_PATH_TO_END".into(),
                message: format!("Node '{}' has no path to END", node),
                node: Some(node),
            });
        }
        result
    }

    /// Detect cycles in the graph.
    ///
    /// Cycles are reported as warnings because some graph topologies legitimately
    /// contain cycles (e.g. agent loops).
    pub fn validate_no_cycles(nodes: &[String], edges: &[(String, String)]) -> ValidationResult {
        let mut result = ValidationResult::new();
        let mut adj: HashMap<String, Vec<String>> = HashMap::new();
        for (from, to) in edges {
            adj.entry(from.clone()).or_default().push(to.clone());
        }

        if let Some(cycle_path) = CycleDetector::detect(nodes, &adj) {
            let cycle_str = cycle_path.join(" -> ");
            result.push(ValidationIssue {
                severity: ValidationSeverity::Warning,
                code: "CYCLE_DETECTED".into(),
                message: format!("Cycle detected: {}", cycle_str),
                node: cycle_path.first().cloned(),
            });
        }
        result
    }

    /// Validate that all edge endpoints reference existing nodes (or START/END).
    pub fn validate_edges(nodes: &[String], edges: &[(String, String)]) -> ValidationResult {
        let mut result = ValidationResult::new();
        let valid: HashSet<&str> = nodes
            .iter()
            .map(String::as_str)
            .chain([START, END])
            .collect();

        for (from, to) in edges {
            if !valid.contains(from.as_str()) {
                result.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "MISSING_EDGE_SOURCE".into(),
                    message: format!("Edge source '{}' does not exist", from),
                    node: Some(from.clone()),
                });
            }
            if !valid.contains(to.as_str()) {
                result.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "MISSING_EDGE_TARGET".into(),
                    message: format!("Edge target '{}' does not exist", to),
                    node: Some(to.clone()),
                });
            }
        }
        result
    }

    /// Validate node names: no reserved names, no duplicates, no empty names.
    pub fn validate_node_names(nodes: &[String]) -> ValidationResult {
        let mut result = ValidationResult::new();
        let mut seen: HashSet<&str> = HashSet::new();

        for name in nodes {
            if name.is_empty() {
                result.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "EMPTY_NODE_NAME".into(),
                    message: "Node name must not be empty".into(),
                    node: None,
                });
                continue;
            }

            if RESERVED_NAMES.contains(&name.as_str()) {
                result.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "RESERVED_NODE_NAME".into(),
                    message: format!("Node name '{}' is reserved", name),
                    node: Some(name.clone()),
                });
            }

            if !seen.insert(name.as_str()) {
                result.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "DUPLICATE_NODE_NAME".into(),
                    message: format!("Duplicate node name '{}'", name),
                    node: Some(name.clone()),
                });
            }
        }
        result
    }

    /// Validate that all conditional edge targets reference existing nodes.
    pub fn validate_conditional_edges(
        nodes: &[String],
        conditional_edges: &[(String, Vec<String>)],
    ) -> ValidationResult {
        let mut result = ValidationResult::new();
        let valid: HashSet<&str> = nodes
            .iter()
            .map(String::as_str)
            .chain([START, END])
            .collect();

        for (from, targets) in conditional_edges {
            if !valid.contains(from.as_str()) {
                result.push(ValidationIssue {
                    severity: ValidationSeverity::Error,
                    code: "INVALID_CONDITIONAL_SOURCE".into(),
                    message: format!("Conditional edge source '{}' does not exist", from),
                    node: Some(from.clone()),
                });
            }

            if targets.is_empty() {
                result.push(ValidationIssue {
                    severity: ValidationSeverity::Warning,
                    code: "EMPTY_CONDITIONAL_TARGETS".into(),
                    message: format!("Conditional edge from '{}' has no targets", from),
                    node: Some(from.clone()),
                });
            }

            for target in targets {
                if !valid.contains(target.as_str()) {
                    result.push(ValidationIssue {
                        severity: ValidationSeverity::Error,
                        code: "INVALID_CONDITIONAL_TARGET".into(),
                        message: format!(
                            "Conditional edge target '{}' (from '{}') does not exist",
                            target, from
                        ),
                        node: Some(target.clone()),
                    });
                }
            }
        }
        result
    }

    /// Run all validations against a [`GraphDefinition`].
    pub fn validate_all(def: &GraphDefinition) -> ValidationResult {
        let mut result = ValidationResult::new();

        // 1. Node names
        result.merge(Self::validate_node_names(&def.nodes));

        // 2. Edge endpoints
        result.merge(Self::validate_edges(&def.nodes, &def.edges));

        // 3. Conditional edges — convert from ConditionalEdgeDef
        let cond_edges: Vec<(String, Vec<String>)> = def
            .conditional_edges
            .iter()
            .map(|c| (c.from.clone(), c.targets.clone()))
            .collect();
        result.merge(Self::validate_conditional_edges(&def.nodes, &cond_edges));

        // 4. Entry point
        if !def.nodes.contains(&def.entry_point) {
            result.push(ValidationIssue {
                severity: ValidationSeverity::Error,
                code: "INVALID_ENTRY_POINT".into(),
                message: format!("Entry point '{}' is not in the node list", def.entry_point),
                node: Some(def.entry_point.clone()),
            });
        }

        // Build combined edges for connectivity / termination / cycle checks.
        let adj = Self::build_adjacency(def);
        let all_edges: Vec<(String, String)> = adj
            .iter()
            .flat_map(|(from, tos)| tos.iter().map(move |to| (from.clone(), to.clone())))
            .collect();

        // 5. Connectivity
        result.merge(Self::validate_connectivity(&def.nodes, &all_edges));

        // 6. Termination
        result.merge(Self::validate_termination(&def.nodes, &all_edges));

        // 7. Cycles (warning only)
        result.merge(Self::validate_no_cycles(&def.nodes, &all_edges));

        // 8. Info: node count
        result.push(ValidationIssue {
            severity: ValidationSeverity::Info,
            code: "NODE_COUNT".into(),
            message: format!("Graph has {} node(s)", def.nodes.len()),
            node: None,
        });

        result
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::serialize::ConditionalEdgeDef;

    /// Helper to build a simple GraphDefinition.
    fn make_def(nodes: Vec<&str>, edges: Vec<(&str, &str)>, entry: &str) -> GraphDefinition {
        GraphDefinition {
            nodes: nodes.into_iter().map(String::from).collect(),
            edges: edges
                .into_iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
            conditional_edges: vec![],
            entry_point: entry.to_string(),
            interrupt_before: vec![],
            interrupt_after: vec![],
        }
    }

    // -----------------------------------------------------------------------
    // validate_node_names
    // -----------------------------------------------------------------------

    #[test]
    fn test_node_names_valid() {
        let result = GraphValidator::validate_node_names(&["agent".into(), "tools".into()]);
        assert!(result.is_valid());
        assert!(result.errors().is_empty());
    }

    #[test]
    fn test_node_names_reserved() {
        let result = GraphValidator::validate_node_names(&["agent".into(), "__start__".into()]);
        assert!(!result.is_valid());
        assert!(result.has_error_code("RESERVED_NODE_NAME"));
    }

    #[test]
    fn test_node_names_duplicate() {
        let result = GraphValidator::validate_node_names(&["agent".into(), "agent".into()]);
        assert!(!result.is_valid());
        assert!(result.has_error_code("DUPLICATE_NODE_NAME"));
    }

    #[test]
    fn test_node_names_empty() {
        let result = GraphValidator::validate_node_names(&["".into()]);
        assert!(!result.is_valid());
        assert!(result.has_error_code("EMPTY_NODE_NAME"));
    }

    // -----------------------------------------------------------------------
    // validate_edges
    // -----------------------------------------------------------------------

    #[test]
    fn test_edges_valid() {
        let nodes = vec!["a".into(), "b".into()];
        let edges = vec![
            (START.to_string(), "a".to_string()),
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), END.to_string()),
        ];
        let result = GraphValidator::validate_edges(&nodes, &edges);
        assert!(result.is_valid());
    }

    #[test]
    fn test_edges_missing_source() {
        let nodes = vec!["a".into()];
        let edges = vec![("ghost".to_string(), "a".to_string())];
        let result = GraphValidator::validate_edges(&nodes, &edges);
        assert!(!result.is_valid());
        assert!(result.has_error_code("MISSING_EDGE_SOURCE"));
    }

    #[test]
    fn test_edges_missing_target() {
        let nodes = vec!["a".into()];
        let edges = vec![("a".to_string(), "ghost".to_string())];
        let result = GraphValidator::validate_edges(&nodes, &edges);
        assert!(!result.is_valid());
        assert!(result.has_error_code("MISSING_EDGE_TARGET"));
    }

    // -----------------------------------------------------------------------
    // validate_connectivity
    // -----------------------------------------------------------------------

    #[test]
    fn test_connectivity_all_reachable() {
        let nodes = vec!["a".into(), "b".into()];
        let edges = vec![
            (START.to_string(), "a".to_string()),
            ("a".to_string(), "b".to_string()),
        ];
        let result = GraphValidator::validate_connectivity(&nodes, &edges);
        assert!(result.is_valid());
    }

    #[test]
    fn test_connectivity_unreachable_node() {
        let nodes = vec!["a".into(), "isolated".into()];
        let edges = vec![(START.to_string(), "a".to_string())];
        let result = GraphValidator::validate_connectivity(&nodes, &edges);
        assert!(!result.is_valid());
        assert!(result.has_error_code("UNREACHABLE_NODE"));
        assert!(result.errors()[0].node.as_deref() == Some("isolated"));
    }

    // -----------------------------------------------------------------------
    // validate_termination
    // -----------------------------------------------------------------------

    #[test]
    fn test_termination_all_reach_end() {
        let nodes = vec!["a".into(), "b".into()];
        let edges = vec![
            (START.to_string(), "a".to_string()),
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), END.to_string()),
        ];
        let result = GraphValidator::validate_termination(&nodes, &edges);
        assert!(result.is_valid());
    }

    #[test]
    fn test_termination_dead_end() {
        let nodes = vec!["a".into(), "b".into()];
        let edges = vec![
            (START.to_string(), "a".to_string()),
            ("a".to_string(), "b".to_string()),
            // b does not go to END
        ];
        let result = GraphValidator::validate_termination(&nodes, &edges);
        assert!(!result.is_valid());
        assert!(result.has_error_code("NO_PATH_TO_END"));
    }

    // -----------------------------------------------------------------------
    // validate_no_cycles
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_cycles_acyclic() {
        let nodes = vec!["a".into(), "b".into()];
        let edges = vec![
            (START.to_string(), "a".to_string()),
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), END.to_string()),
        ];
        let result = GraphValidator::validate_no_cycles(&nodes, &edges);
        assert!(result.is_valid()); // no errors
        assert!(!result.has_error_code("CYCLE_DETECTED"));
    }

    #[test]
    fn test_no_cycles_with_cycle() {
        let nodes = vec!["a".into(), "b".into()];
        let edges = vec![
            (START.to_string(), "a".to_string()),
            ("a".to_string(), "b".to_string()),
            ("b".to_string(), "a".to_string()), // cycle
        ];
        let result = GraphValidator::validate_no_cycles(&nodes, &edges);
        assert!(result.has_error_code("CYCLE_DETECTED"));
        // Should be a warning, not an error
        assert!(result.is_valid());
        assert_eq!(result.warnings().len(), 1);
    }

    // -----------------------------------------------------------------------
    // validate_conditional_edges
    // -----------------------------------------------------------------------

    #[test]
    fn test_conditional_edges_valid() {
        let nodes = vec!["a".into(), "b".into(), "c".into()];
        let conds = vec![("a".to_string(), vec!["b".to_string(), "c".to_string()])];
        let result = GraphValidator::validate_conditional_edges(&nodes, &conds);
        assert!(result.is_valid());
    }

    #[test]
    fn test_conditional_edges_invalid_source() {
        let nodes = vec!["a".into()];
        let conds = vec![("ghost".to_string(), vec!["a".to_string()])];
        let result = GraphValidator::validate_conditional_edges(&nodes, &conds);
        assert!(!result.is_valid());
        assert!(result.has_error_code("INVALID_CONDITIONAL_SOURCE"));
    }

    #[test]
    fn test_conditional_edges_invalid_target() {
        let nodes = vec!["a".into()];
        let conds = vec![("a".to_string(), vec!["ghost".to_string()])];
        let result = GraphValidator::validate_conditional_edges(&nodes, &conds);
        assert!(!result.is_valid());
        assert!(result.has_error_code("INVALID_CONDITIONAL_TARGET"));
    }

    #[test]
    fn test_conditional_edges_empty_targets_warning() {
        let nodes = vec!["a".into()];
        let conds = vec![("a".to_string(), vec![])];
        let result = GraphValidator::validate_conditional_edges(&nodes, &conds);
        // Empty targets is a warning, not an error
        assert!(result.is_valid());
        assert!(result.has_error_code("EMPTY_CONDITIONAL_TARGETS"));
        assert_eq!(result.warnings().len(), 1);
    }

    #[test]
    fn test_conditional_edges_end_target_valid() {
        let nodes = vec!["a".into()];
        let conds = vec![("a".to_string(), vec![END.to_string()])];
        let result = GraphValidator::validate_conditional_edges(&nodes, &conds);
        assert!(result.is_valid());
    }

    // -----------------------------------------------------------------------
    // validate_all
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_all_valid_simple_graph() {
        let def = make_def(
            vec!["a", "b"],
            vec![(START, "a"), ("a", "b"), ("b", END)],
            "a",
        );
        let result = GraphValidator::validate_all(&def);
        assert!(result.is_valid());
        assert!(result.errors().is_empty());
        // Should have an info about node count
        assert!(result.has_error_code("NODE_COUNT"));
    }

    #[test]
    fn test_validate_all_invalid_entry_point() {
        let def = make_def(vec!["a"], vec![(START, "a"), ("a", END)], "nonexistent");
        let result = GraphValidator::validate_all(&def);
        assert!(!result.is_valid());
        assert!(result.has_error_code("INVALID_ENTRY_POINT"));
    }

    #[test]
    fn test_validate_all_with_conditional_edges() {
        let def = GraphDefinition {
            nodes: vec!["agent".into(), "tools".into()],
            edges: vec![
                (START.into(), "agent".into()),
                ("tools".into(), "agent".into()),
            ],
            conditional_edges: vec![ConditionalEdgeDef {
                from: "agent".into(),
                targets: vec!["tools".into(), END.into()],
                labels: None,
            }],
            entry_point: "agent".into(),
            interrupt_before: vec![],
            interrupt_after: vec![],
        };
        let result = GraphValidator::validate_all(&def);
        assert!(result.is_valid());
    }

    #[test]
    fn test_validate_all_with_invalid_conditional_target() {
        let def = GraphDefinition {
            nodes: vec!["agent".into()],
            edges: vec![(START.into(), "agent".into())],
            conditional_edges: vec![ConditionalEdgeDef {
                from: "agent".into(),
                targets: vec!["nonexistent".into()],
                labels: None,
            }],
            entry_point: "agent".into(),
            interrupt_before: vec![],
            interrupt_after: vec![],
        };
        let result = GraphValidator::validate_all(&def);
        assert!(!result.is_valid());
        assert!(result.has_error_code("INVALID_CONDITIONAL_TARGET"));
    }

    // -----------------------------------------------------------------------
    // CycleDetector
    // -----------------------------------------------------------------------

    #[test]
    fn test_cycle_detector_no_cycle() {
        let nodes = vec!["a".into(), "b".into()];
        let mut adj = HashMap::new();
        adj.insert(START.to_string(), vec!["a".to_string()]);
        adj.insert("a".to_string(), vec!["b".to_string()]);
        adj.insert("b".to_string(), vec![END.to_string()]);

        assert!(CycleDetector::detect(&nodes, &adj).is_none());
    }

    #[test]
    fn test_cycle_detector_with_cycle() {
        let nodes = vec!["a".into(), "b".into()];
        let mut adj = HashMap::new();
        adj.insert("a".to_string(), vec!["b".to_string()]);
        adj.insert("b".to_string(), vec!["a".to_string()]);

        let cycle = CycleDetector::detect(&nodes, &adj);
        assert!(cycle.is_some());
        let path = cycle.unwrap();
        // The cycle should contain both a and b
        assert!(path.contains(&"a".to_string()));
        assert!(path.contains(&"b".to_string()));
    }

    #[test]
    fn test_cycle_detector_self_loop() {
        let nodes = vec!["a".into()];
        let mut adj = HashMap::new();
        adj.insert("a".to_string(), vec!["a".to_string()]);

        let cycle = CycleDetector::detect(&nodes, &adj);
        assert!(cycle.is_some());
    }

    // -----------------------------------------------------------------------
    // ReachabilityAnalyzer
    // -----------------------------------------------------------------------

    #[test]
    fn test_reachability_simple() {
        let mut adj = HashMap::new();
        adj.insert(START.to_string(), vec!["a".to_string()]);
        adj.insert("a".to_string(), vec!["b".to_string()]);

        let reachable = ReachabilityAnalyzer::reachable_from(START, &adj);
        assert!(reachable.contains(START));
        assert!(reachable.contains("a"));
        assert!(reachable.contains("b"));
    }

    #[test]
    fn test_reachability_unreachable() {
        let nodes = vec!["a".into(), "b".into(), "isolated".into()];
        let mut adj = HashMap::new();
        adj.insert(START.to_string(), vec!["a".to_string()]);
        adj.insert("a".to_string(), vec!["b".to_string()]);

        let unreachable = ReachabilityAnalyzer::unreachable_nodes(&nodes, START, &adj);
        assert_eq!(unreachable, vec!["isolated".to_string()]);
    }

    #[test]
    fn test_dead_end_nodes() {
        let nodes = vec!["a".into(), "b".into(), "dead".into()];
        let mut adj = HashMap::new();
        adj.insert(START.to_string(), vec!["a".to_string(), "dead".to_string()]);
        adj.insert("a".to_string(), vec!["b".to_string()]);
        adj.insert("b".to_string(), vec![END.to_string()]);
        // "dead" has no outgoing edges → cannot reach END

        let dead_ends = ReachabilityAnalyzer::dead_end_nodes(&nodes, END, &adj);
        assert_eq!(dead_ends, vec!["dead".to_string()]);
    }

    // -----------------------------------------------------------------------
    // ValidationResult methods
    // -----------------------------------------------------------------------

    #[test]
    fn test_validation_result_is_valid_when_empty() {
        let result = ValidationResult::new();
        assert!(result.is_valid());
        assert!(result.errors().is_empty());
        assert!(result.warnings().is_empty());
    }

    #[test]
    fn test_validation_result_display() {
        let mut result = ValidationResult::new();
        result.push(ValidationIssue {
            severity: ValidationSeverity::Error,
            code: "TEST_CODE".into(),
            message: "something went wrong".into(),
            node: Some("my_node".into()),
        });
        let display = format!("{}", result);
        assert!(display.contains("TEST_CODE"));
        assert!(display.contains("my_node"));
    }

    #[test]
    fn test_validation_result_merge() {
        let mut r1 = ValidationResult::new();
        r1.push(ValidationIssue {
            severity: ValidationSeverity::Warning,
            code: "W1".into(),
            message: "warning one".into(),
            node: None,
        });

        let mut r2 = ValidationResult::new();
        r2.push(ValidationIssue {
            severity: ValidationSeverity::Error,
            code: "E1".into(),
            message: "error one".into(),
            node: None,
        });

        r1.merge(r2);
        assert_eq!(r1.issues.len(), 2);
        assert!(!r1.is_valid());
        assert_eq!(r1.warnings().len(), 1);
        assert_eq!(r1.errors().len(), 1);
    }
}
