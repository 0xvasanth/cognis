//! Graph versioning, migration, and compatibility checking.
//!
//! This module provides a versioning system for graph definitions, enabling:
//! - Semantic versioning of graph topologies
//! - Migration pipelines to transform graphs between versions
//! - Diffing to compare two graph versions
//! - Version history tracking with changelog
//! - Compatibility checking between versions
//! - Version constraints for dependency specification

use std::collections::HashSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::serialize::GraphDefinition;
use crate::errors::{LangGraphError, Result};

// ---------------------------------------------------------------------------
// GraphVersion
// ---------------------------------------------------------------------------

/// Semantic version with major.minor.patch components.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GraphVersion {
    pub major: u32,
    pub minor: u32,
    pub patch: u32,
}

impl GraphVersion {
    /// Create a new version.
    pub fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Parse a version string like "1.2.3".
    pub fn parse(s: &str) -> Result<Self> {
        let parts: Vec<&str> = s.split('.').collect();
        if parts.len() != 3 {
            return Err(LangGraphError::Other(format!(
                "Invalid version string '{}': expected major.minor.patch",
                s
            )));
        }
        let major = parts[0].parse::<u32>().map_err(|_| {
            LangGraphError::Other(format!("Invalid major version in '{}'", s))
        })?;
        let minor = parts[1].parse::<u32>().map_err(|_| {
            LangGraphError::Other(format!("Invalid minor version in '{}'", s))
        })?;
        let patch = parts[2].parse::<u32>().map_err(|_| {
            LangGraphError::Other(format!("Invalid patch version in '{}'", s))
        })?;
        Ok(Self {
            major,
            minor,
            patch,
        })
    }

    /// Check if this version is compatible with another (same major version).
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.major == other.major
    }

    /// Bump the major version (resets minor and patch).
    pub fn bump_major(&self) -> Self {
        Self::new(self.major + 1, 0, 0)
    }

    /// Bump the minor version (resets patch).
    pub fn bump_minor(&self) -> Self {
        Self::new(self.major, self.minor + 1, 0)
    }

    /// Bump the patch version.
    pub fn bump_patch(&self) -> Self {
        Self::new(self.major, self.minor, self.patch + 1)
    }
}

impl fmt::Display for GraphVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

impl PartialOrd for GraphVersion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GraphVersion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.major
            .cmp(&other.major)
            .then(self.minor.cmp(&other.minor))
            .then(self.patch.cmp(&other.patch))
    }
}

// ---------------------------------------------------------------------------
// VersionedGraph
// ---------------------------------------------------------------------------

/// A graph definition wrapped with version metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionedGraph {
    /// The version of this graph.
    pub version: GraphVersion,
    /// The graph definition.
    pub definition: GraphDefinition,
    /// ISO 8601 creation timestamp string.
    pub created_at: String,
    /// Author of this version.
    pub author: String,
    /// Optional description of this version.
    pub description: Option<String>,
}

impl VersionedGraph {
    /// Create a new versioned graph.
    pub fn new(
        version: GraphVersion,
        definition: GraphDefinition,
        author: impl Into<String>,
    ) -> Self {
        Self {
            version,
            definition,
            created_at: current_timestamp(),
            author: author.into(),
            description: None,
        }
    }

    /// Set the description.
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Set the creation timestamp.
    pub fn with_timestamp(mut self, ts: impl Into<String>) -> Self {
        self.created_at = ts.into();
        self
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            LangGraphError::Other(format!("Failed to serialize versioned graph: {e}"))
        })
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            LangGraphError::Other(format!("Failed to deserialize versioned graph: {e}"))
        })
    }
}

/// Get a simple timestamp string from system time.
fn current_timestamp() -> String {
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", dur.as_secs())
}

// ---------------------------------------------------------------------------
// GraphMigration trait
// ---------------------------------------------------------------------------

/// Trait for transforming a graph definition from one version to another.
pub trait GraphMigration {
    /// The source version this migration applies to.
    fn from_version(&self) -> &GraphVersion;

    /// The target version this migration produces.
    fn to_version(&self) -> &GraphVersion;

    /// Apply the migration to a graph definition.
    fn migrate(&self, definition: GraphDefinition) -> Result<GraphDefinition>;
}

// ---------------------------------------------------------------------------
// MigrationStep
// ---------------------------------------------------------------------------

/// A concrete migration step with a transform function.
pub struct MigrationStep {
    /// Source version.
    pub from: GraphVersion,
    /// Target version.
    pub to: GraphVersion,
    /// Transform function.
    transform: Box<dyn Fn(GraphDefinition) -> Result<GraphDefinition> + Send + Sync>,
}

impl MigrationStep {
    /// Create a new migration step.
    pub fn new(
        from: GraphVersion,
        to: GraphVersion,
        transform: impl Fn(GraphDefinition) -> Result<GraphDefinition> + Send + Sync + 'static,
    ) -> Self {
        Self {
            from,
            to,
            transform: Box::new(transform),
        }
    }
}

impl fmt::Debug for MigrationStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MigrationStep")
            .field("from", &self.from)
            .field("to", &self.to)
            .finish()
    }
}

impl GraphMigration for MigrationStep {
    fn from_version(&self) -> &GraphVersion {
        &self.from
    }

    fn to_version(&self) -> &GraphVersion {
        &self.to
    }

    fn migrate(&self, definition: GraphDefinition) -> Result<GraphDefinition> {
        (self.transform)(definition)
    }
}

// ---------------------------------------------------------------------------
// MigrationPipeline
// ---------------------------------------------------------------------------

/// A pipeline that chains multiple migration steps to go from any version
/// to any other version, automatically finding a path.
pub struct MigrationPipeline {
    steps: Vec<MigrationStep>,
}

impl fmt::Debug for MigrationPipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MigrationPipeline")
            .field("steps_count", &self.steps.len())
            .finish()
    }
}

impl MigrationPipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a migration step.
    pub fn add_step(&mut self, step: MigrationStep) {
        self.steps.push(step);
    }

    /// Find a migration path from `from` to `to` and apply it.
    ///
    /// Uses BFS to find the shortest path through available migration steps.
    pub fn migrate(
        &self,
        definition: GraphDefinition,
        from: &GraphVersion,
        to: &GraphVersion,
    ) -> Result<GraphDefinition> {
        if from == to {
            return Ok(definition);
        }

        let path = self.find_path(from, to)?;
        let mut current = definition;
        for step_idx in path {
            current = self.steps[step_idx].migrate(current)?;
        }
        Ok(current)
    }

    /// Find the sequence of step indices to get from `from` to `to`.
    fn find_path(&self, from: &GraphVersion, to: &GraphVersion) -> Result<Vec<usize>> {
        // BFS over versions using step indices as edges
        let mut queue: std::collections::VecDeque<(GraphVersion, Vec<usize>)> =
            std::collections::VecDeque::new();
        let mut visited: HashSet<GraphVersion> = HashSet::new();

        queue.push_back((from.clone(), Vec::new()));
        visited.insert(from.clone());

        while let Some((current_version, path)) = queue.pop_front() {
            for (idx, step) in self.steps.iter().enumerate() {
                if step.from == current_version && !visited.contains(&step.to) {
                    let mut new_path = path.clone();
                    new_path.push(idx);

                    if &step.to == to {
                        return Ok(new_path);
                    }

                    visited.insert(step.to.clone());
                    queue.push_back((step.to.clone(), new_path));
                }
            }
        }

        Err(LangGraphError::Other(format!(
            "No migration path found from version {} to {}",
            from, to
        )))
    }

    /// List all registered migration steps (from -> to).
    pub fn list_steps(&self) -> Vec<(GraphVersion, GraphVersion)> {
        self.steps
            .iter()
            .map(|s| (s.from.clone(), s.to.clone()))
            .collect()
    }

    /// Check if a migration path exists between two versions.
    pub fn has_path(&self, from: &GraphVersion, to: &GraphVersion) -> bool {
        if from == to {
            return true;
        }
        self.find_path(from, to).is_ok()
    }
}

impl Default for MigrationPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// GraphDiff
// ---------------------------------------------------------------------------

/// The type of modification applied to a node or edge.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModificationType {
    /// Node's edges changed.
    EdgesChanged,
    /// Node's interrupt configuration changed.
    InterruptChanged,
    /// Entry point changed.
    EntryPointChanged,
    /// Conditional edges changed.
    ConditionalEdgesChanged,
}

/// Represents the difference between two graph definitions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphDiff {
    /// Nodes added in the new version.
    pub added_nodes: Vec<String>,
    /// Nodes removed in the new version.
    pub removed_nodes: Vec<String>,
    /// Nodes that exist in both but have modifications.
    pub modified_nodes: Vec<(String, ModificationType)>,
    /// Edges added in the new version.
    pub added_edges: Vec<(String, String)>,
    /// Edges removed in the new version.
    pub removed_edges: Vec<(String, String)>,
    /// Whether the entry point changed.
    pub entry_point_changed: bool,
}

impl GraphDiff {
    /// Compute the diff between two graph definitions.
    pub fn compute(old: &GraphDefinition, new: &GraphDefinition) -> Self {
        let old_nodes: HashSet<&str> = old.nodes.iter().map(|s| s.as_str()).collect();
        let new_nodes: HashSet<&str> = new.nodes.iter().map(|s| s.as_str()).collect();

        let added_nodes: Vec<String> = new_nodes
            .difference(&old_nodes)
            .map(|s| s.to_string())
            .collect();
        let removed_nodes: Vec<String> = old_nodes
            .difference(&new_nodes)
            .map(|s| s.to_string())
            .collect();

        // Check edge changes per node
        let old_edges: HashSet<(&str, &str)> = old
            .edges
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();
        let new_edges: HashSet<(&str, &str)> = new
            .edges
            .iter()
            .map(|(a, b)| (a.as_str(), b.as_str()))
            .collect();

        let added_edges: Vec<(String, String)> = new_edges
            .difference(&old_edges)
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();
        let removed_edges: Vec<(String, String)> = old_edges
            .difference(&new_edges)
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect();

        // Detect modified nodes (exist in both, but edges or interrupts changed)
        let common_nodes: HashSet<&str> = old_nodes.intersection(&new_nodes).copied().collect();
        let mut modified_nodes: Vec<(String, ModificationType)> = Vec::new();

        for node in &common_nodes {
            // Check if edges involving this node changed
            let old_node_edges: HashSet<_> = old_edges
                .iter()
                .filter(|(a, b)| a == node || b == node)
                .collect();
            let new_node_edges: HashSet<_> = new_edges
                .iter()
                .filter(|(a, b)| a == node || b == node)
                .collect();
            if old_node_edges != new_node_edges {
                modified_nodes.push((node.to_string(), ModificationType::EdgesChanged));
                continue;
            }

            // Check interrupt changes
            let old_interrupt_before = old.interrupt_before.iter().any(|n| n.as_str() == *node);
            let new_interrupt_before = new.interrupt_before.iter().any(|n| n.as_str() == *node);
            let old_interrupt_after = old.interrupt_after.iter().any(|n| n.as_str() == *node);
            let new_interrupt_after = new.interrupt_after.iter().any(|n| n.as_str() == *node);
            if old_interrupt_before != new_interrupt_before
                || old_interrupt_after != new_interrupt_after
            {
                modified_nodes.push((node.to_string(), ModificationType::InterruptChanged));
                continue;
            }

            // Check conditional edge changes for this node
            let old_cond: Vec<_> = old
                .conditional_edges
                .iter()
                .filter(|c| c.from.as_str() == *node)
                .collect();
            let new_cond: Vec<_> = new
                .conditional_edges
                .iter()
                .filter(|c| c.from.as_str() == *node)
                .collect();
            let old_cond_targets: HashSet<&str> = old_cond
                .iter()
                .flat_map(|c| c.targets.iter().map(|t| t.as_str()))
                .collect();
            let new_cond_targets: HashSet<&str> = new_cond
                .iter()
                .flat_map(|c| c.targets.iter().map(|t| t.as_str()))
                .collect();
            if old_cond_targets != new_cond_targets {
                modified_nodes.push((
                    node.to_string(),
                    ModificationType::ConditionalEdgesChanged,
                ));
            }
        }

        let entry_point_changed = old.entry_point != new.entry_point;

        GraphDiff {
            added_nodes,
            removed_nodes,
            modified_nodes,
            added_edges,
            removed_edges,
            entry_point_changed,
        }
    }

    /// Returns true if the two graphs are identical (no diff).
    pub fn is_empty(&self) -> bool {
        self.added_nodes.is_empty()
            && self.removed_nodes.is_empty()
            && self.modified_nodes.is_empty()
            && self.added_edges.is_empty()
            && self.removed_edges.is_empty()
            && !self.entry_point_changed
    }

    /// Returns a human-readable summary of the diff.
    pub fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.added_nodes.is_empty() {
            parts.push(format!("added nodes: {}", self.added_nodes.join(", ")));
        }
        if !self.removed_nodes.is_empty() {
            parts.push(format!("removed nodes: {}", self.removed_nodes.join(", ")));
        }
        if !self.modified_nodes.is_empty() {
            let names: Vec<&str> = self.modified_nodes.iter().map(|(n, _)| n.as_str()).collect();
            parts.push(format!("modified nodes: {}", names.join(", ")));
        }
        if !self.added_edges.is_empty() {
            parts.push(format!("{} edges added", self.added_edges.len()));
        }
        if !self.removed_edges.is_empty() {
            parts.push(format!("{} edges removed", self.removed_edges.len()));
        }
        if self.entry_point_changed {
            parts.push("entry point changed".to_string());
        }
        if parts.is_empty() {
            "no changes".to_string()
        } else {
            parts.join("; ")
        }
    }
}

// ---------------------------------------------------------------------------
// VersionHistory
// ---------------------------------------------------------------------------

/// A changelog entry for a version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangelogEntry {
    /// The version this entry describes.
    pub version: GraphVersion,
    /// Timestamp of the change.
    pub timestamp: String,
    /// Author of the change.
    pub author: String,
    /// Description of changes.
    pub message: String,
    /// The diff from the previous version, if available.
    pub diff_summary: Option<String>,
}

/// Tracks all versions of a graph with changelog entries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionHistory {
    /// Name of the graph.
    pub graph_name: String,
    /// All versioned snapshots, ordered by version.
    pub versions: Vec<VersionedGraph>,
    /// Changelog entries.
    pub changelog: Vec<ChangelogEntry>,
}

impl VersionHistory {
    /// Create a new version history for a named graph.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            graph_name: name.into(),
            versions: Vec::new(),
            changelog: Vec::new(),
        }
    }

    /// Add a new version to the history.
    pub fn add_version(&mut self, graph: VersionedGraph, message: impl Into<String>) {
        let diff_summary = if let Some(last) = self.versions.last() {
            let diff = GraphDiff::compute(&last.definition, &graph.definition);
            Some(diff.summary())
        } else {
            None
        };

        let entry = ChangelogEntry {
            version: graph.version.clone(),
            timestamp: graph.created_at.clone(),
            author: graph.author.clone(),
            message: message.into(),
            diff_summary,
        };

        self.changelog.push(entry);
        self.versions.push(graph);
    }

    /// Get a specific version.
    pub fn get_version(&self, version: &GraphVersion) -> Option<&VersionedGraph> {
        self.versions.iter().find(|v| &v.version == version)
    }

    /// Get the latest version.
    pub fn latest(&self) -> Option<&VersionedGraph> {
        self.versions.last()
    }

    /// Get the number of versions tracked.
    pub fn len(&self) -> usize {
        self.versions.len()
    }

    /// Check if there are no versions.
    pub fn is_empty(&self) -> bool {
        self.versions.is_empty()
    }

    /// List all version numbers.
    pub fn list_versions(&self) -> Vec<&GraphVersion> {
        self.versions.iter().map(|v| &v.version).collect()
    }

    /// Get the diff between two versions in the history.
    pub fn diff_versions(
        &self,
        from: &GraphVersion,
        to: &GraphVersion,
    ) -> Result<GraphDiff> {
        let from_graph = self.get_version(from).ok_or_else(|| {
            LangGraphError::Other(format!("Version {} not found in history", from))
        })?;
        let to_graph = self.get_version(to).ok_or_else(|| {
            LangGraphError::Other(format!("Version {} not found in history", to))
        })?;
        Ok(GraphDiff::compute(
            &from_graph.definition,
            &to_graph.definition,
        ))
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string_pretty(self).map_err(|e| {
            LangGraphError::Other(format!("Failed to serialize version history: {e}"))
        })
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self> {
        serde_json::from_str(json).map_err(|e| {
            LangGraphError::Other(format!("Failed to deserialize version history: {e}"))
        })
    }
}

// ---------------------------------------------------------------------------
// CompatibilityChecker
// ---------------------------------------------------------------------------

/// Result of a compatibility check between two graph versions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompatibilityReport {
    /// Whether the two versions are compatible.
    pub compatible: bool,
    /// Reasons for incompatibility (empty if compatible).
    pub issues: Vec<String>,
    /// The diff between the two versions.
    pub diff: GraphDiff,
}

/// Checks whether two graph versions are compatible.
///
/// Two versions are compatible if they have the same set of node names
/// and the same entry point. Edge changes are allowed as long as the
/// node signatures (names) remain the same.
pub struct CompatibilityChecker;

impl CompatibilityChecker {
    /// Check compatibility between two graph definitions.
    pub fn check(old: &GraphDefinition, new: &GraphDefinition) -> CompatibilityReport {
        let diff = GraphDiff::compute(old, new);
        let mut issues = Vec::new();

        if !diff.added_nodes.is_empty() {
            issues.push(format!(
                "New nodes added: {}",
                diff.added_nodes.join(", ")
            ));
        }
        if !diff.removed_nodes.is_empty() {
            issues.push(format!(
                "Nodes removed: {}",
                diff.removed_nodes.join(", ")
            ));
        }
        if diff.entry_point_changed {
            issues.push(format!(
                "Entry point changed from '{}' to '{}'",
                old.entry_point, new.entry_point
            ));
        }

        CompatibilityReport {
            compatible: issues.is_empty(),
            issues,
            diff,
        }
    }

    /// Check if two versioned graphs are compatible.
    pub fn check_versioned(old: &VersionedGraph, new: &VersionedGraph) -> CompatibilityReport {
        Self::check(&old.definition, &new.definition)
    }

    /// Check if a version is backward-compatible (only additions, no removals).
    pub fn is_backward_compatible(old: &GraphDefinition, new: &GraphDefinition) -> bool {
        let diff = GraphDiff::compute(old, new);
        diff.removed_nodes.is_empty() && !diff.entry_point_changed
    }
}

// ---------------------------------------------------------------------------
// VersionConstraint
// ---------------------------------------------------------------------------

/// Specifies a version requirement.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum VersionConstraint {
    /// Exact version match.
    Exact(GraphVersion),
    /// Minimum version (>=).
    GreaterOrEqual(GraphVersion),
    /// Less than version (<).
    LessThan(GraphVersion),
    /// Version range [min, max).
    Range {
        min: GraphVersion,
        max: GraphVersion,
    },
    /// Compatible with (~=): same major version, >= specified.
    Compatible(GraphVersion),
    /// Any version.
    Any,
}

impl VersionConstraint {
    /// Parse a constraint string.
    ///
    /// Supported formats:
    /// - `"1.2.3"` → Exact
    /// - `">=1.2.3"` → GreaterOrEqual
    /// - `"<1.2.3"` → LessThan
    /// - `">=1.0.0,<2.0.0"` → Range
    /// - `"~=1.2.3"` → Compatible
    /// - `"*"` → Any
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim();
        if s == "*" {
            return Ok(Self::Any);
        }
        if let Some(rest) = s.strip_prefix("~=") {
            let v = GraphVersion::parse(rest.trim())?;
            return Ok(Self::Compatible(v));
        }
        if s.contains(',') {
            let parts: Vec<&str> = s.split(',').collect();
            if parts.len() != 2 {
                return Err(LangGraphError::Other(format!(
                    "Invalid range constraint '{}': expected exactly two parts",
                    s
                )));
            }
            let min_str = parts[0].trim();
            let max_str = parts[1].trim();
            let min = min_str
                .strip_prefix(">=")
                .ok_or_else(|| {
                    LangGraphError::Other(format!(
                        "Range min must start with '>=' in '{}'",
                        s
                    ))
                })?;
            let max = max_str
                .strip_prefix('<')
                .ok_or_else(|| {
                    LangGraphError::Other(format!(
                        "Range max must start with '<' in '{}'",
                        s
                    ))
                })?;
            return Ok(Self::Range {
                min: GraphVersion::parse(min.trim())?,
                max: GraphVersion::parse(max.trim())?,
            });
        }
        if let Some(rest) = s.strip_prefix(">=") {
            return Ok(Self::GreaterOrEqual(GraphVersion::parse(rest.trim())?));
        }
        if let Some(rest) = s.strip_prefix('<') {
            return Ok(Self::LessThan(GraphVersion::parse(rest.trim())?));
        }
        // Default: exact match
        Ok(Self::Exact(GraphVersion::parse(s)?))
    }

    /// Check if a version satisfies this constraint.
    pub fn matches(&self, version: &GraphVersion) -> bool {
        match self {
            Self::Exact(v) => version == v,
            Self::GreaterOrEqual(v) => version >= v,
            Self::LessThan(v) => version < v,
            Self::Range { min, max } => version >= min && version < max,
            Self::Compatible(v) => version.major == v.major && version >= v,
            Self::Any => true,
        }
    }
}

impl fmt::Display for VersionConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exact(v) => write!(f, "{}", v),
            Self::GreaterOrEqual(v) => write!(f, ">={}", v),
            Self::LessThan(v) => write!(f, "<{}", v),
            Self::Range { min, max } => write!(f, ">={},<{}", min, max),
            Self::Compatible(v) => write!(f, "~={}", v),
            Self::Any => write!(f, "*"),
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: create a sample graph definition for tests
// ---------------------------------------------------------------------------

fn _make_graph(
    nodes: Vec<&str>,
    edges: Vec<(&str, &str)>,
    entry: &str,
) -> GraphDefinition {
    GraphDefinition {
        nodes: nodes.into_iter().map(|s| s.to_string()).collect(),
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::{END, START};
    use crate::graph::serialize::ConditionalEdgeDef;

    fn simple_graph() -> GraphDefinition {
        _make_graph(
            vec!["agent", "tools"],
            vec![
                (START, "agent"),
                ("agent", "tools"),
                ("tools", "agent"),
                ("agent", END),
            ],
            "agent",
        )
    }

    fn extended_graph() -> GraphDefinition {
        _make_graph(
            vec!["agent", "tools", "review"],
            vec![
                (START, "agent"),
                ("agent", "tools"),
                ("tools", "review"),
                ("review", "agent"),
                ("agent", END),
            ],
            "agent",
        )
    }

    // === GraphVersion tests ===

    #[test]
    fn test_version_new() {
        let v = GraphVersion::new(1, 2, 3);
        assert_eq!(v.major, 1);
        assert_eq!(v.minor, 2);
        assert_eq!(v.patch, 3);
    }

    #[test]
    fn test_version_display() {
        let v = GraphVersion::new(1, 2, 3);
        assert_eq!(v.to_string(), "1.2.3");
    }

    #[test]
    fn test_version_parse() {
        let v = GraphVersion::parse("1.2.3").unwrap();
        assert_eq!(v, GraphVersion::new(1, 2, 3));
    }

    #[test]
    fn test_version_parse_invalid() {
        assert!(GraphVersion::parse("1.2").is_err());
        assert!(GraphVersion::parse("abc").is_err());
        assert!(GraphVersion::parse("1.2.x").is_err());
    }

    #[test]
    fn test_version_ordering() {
        let v1 = GraphVersion::new(1, 0, 0);
        let v2 = GraphVersion::new(1, 1, 0);
        let v3 = GraphVersion::new(2, 0, 0);
        assert!(v1 < v2);
        assert!(v2 < v3);
        assert!(v1 < v3);
    }

    #[test]
    fn test_version_ordering_patch() {
        let v1 = GraphVersion::new(1, 0, 0);
        let v2 = GraphVersion::new(1, 0, 1);
        assert!(v1 < v2);
    }

    #[test]
    fn test_version_equality() {
        let v1 = GraphVersion::new(1, 2, 3);
        let v2 = GraphVersion::new(1, 2, 3);
        assert_eq!(v1, v2);
    }

    #[test]
    fn test_version_compatible() {
        let v1 = GraphVersion::new(1, 0, 0);
        let v2 = GraphVersion::new(1, 5, 0);
        let v3 = GraphVersion::new(2, 0, 0);
        assert!(v1.is_compatible_with(&v2));
        assert!(!v1.is_compatible_with(&v3));
    }

    #[test]
    fn test_version_bump_major() {
        let v = GraphVersion::new(1, 2, 3);
        assert_eq!(v.bump_major(), GraphVersion::new(2, 0, 0));
    }

    #[test]
    fn test_version_bump_minor() {
        let v = GraphVersion::new(1, 2, 3);
        assert_eq!(v.bump_minor(), GraphVersion::new(1, 3, 0));
    }

    #[test]
    fn test_version_bump_patch() {
        let v = GraphVersion::new(1, 2, 3);
        assert_eq!(v.bump_patch(), GraphVersion::new(1, 2, 4));
    }

    #[test]
    fn test_version_serialize() {
        let v = GraphVersion::new(1, 2, 3);
        let json = serde_json::to_string(&v).unwrap();
        let restored: GraphVersion = serde_json::from_str(&json).unwrap();
        assert_eq!(v, restored);
    }

    // === VersionedGraph tests ===

    #[test]
    fn test_versioned_graph_new() {
        let graph = VersionedGraph::new(
            GraphVersion::new(1, 0, 0),
            simple_graph(),
            "alice",
        );
        assert_eq!(graph.version, GraphVersion::new(1, 0, 0));
        assert_eq!(graph.author, "alice");
        assert!(graph.description.is_none());
    }

    #[test]
    fn test_versioned_graph_with_description() {
        let graph = VersionedGraph::new(
            GraphVersion::new(1, 0, 0),
            simple_graph(),
            "alice",
        )
        .with_description("Initial version");
        assert_eq!(graph.description.as_deref(), Some("Initial version"));
    }

    #[test]
    fn test_versioned_graph_json_round_trip() {
        let graph = VersionedGraph::new(
            GraphVersion::new(1, 0, 0),
            simple_graph(),
            "alice",
        )
        .with_description("test")
        .with_timestamp("2025-01-01T00:00:00Z");

        let json = graph.to_json().unwrap();
        let restored = VersionedGraph::from_json(&json).unwrap();
        assert_eq!(restored.version, graph.version);
        assert_eq!(restored.author, "alice");
        assert_eq!(restored.description, graph.description);
        assert_eq!(restored.created_at, "2025-01-01T00:00:00Z");
    }

    #[test]
    fn test_versioned_graph_with_timestamp() {
        let graph = VersionedGraph::new(
            GraphVersion::new(1, 0, 0),
            simple_graph(),
            "bob",
        )
        .with_timestamp("2025-06-15T12:00:00Z");
        assert_eq!(graph.created_at, "2025-06-15T12:00:00Z");
    }

    // === MigrationStep tests ===

    #[test]
    fn test_migration_step_identity() {
        let step = MigrationStep::new(
            GraphVersion::new(1, 0, 0),
            GraphVersion::new(1, 1, 0),
            |def| Ok(def),
        );
        assert_eq!(*step.from_version(), GraphVersion::new(1, 0, 0));
        assert_eq!(*step.to_version(), GraphVersion::new(1, 1, 0));
    }

    #[test]
    fn test_migration_step_transform() {
        let step = MigrationStep::new(
            GraphVersion::new(1, 0, 0),
            GraphVersion::new(1, 1, 0),
            |mut def| {
                def.nodes.push("new_node".to_string());
                Ok(def)
            },
        );
        let result = step.migrate(simple_graph()).unwrap();
        assert!(result.nodes.contains(&"new_node".to_string()));
    }

    #[test]
    fn test_migration_step_debug() {
        let step = MigrationStep::new(
            GraphVersion::new(1, 0, 0),
            GraphVersion::new(2, 0, 0),
            |def| Ok(def),
        );
        let debug_str = format!("{:?}", step);
        assert!(debug_str.contains("MigrationStep"));
    }

    // === MigrationPipeline tests ===

    #[test]
    fn test_pipeline_same_version() {
        let pipeline = MigrationPipeline::new();
        let v = GraphVersion::new(1, 0, 0);
        let result = pipeline.migrate(simple_graph(), &v, &v).unwrap();
        assert_eq!(result.nodes, simple_graph().nodes);
    }

    #[test]
    fn test_pipeline_single_step() {
        let mut pipeline = MigrationPipeline::new();
        pipeline.add_step(MigrationStep::new(
            GraphVersion::new(1, 0, 0),
            GraphVersion::new(1, 1, 0),
            |mut def| {
                def.nodes.push("cache".to_string());
                Ok(def)
            },
        ));
        let result = pipeline
            .migrate(
                simple_graph(),
                &GraphVersion::new(1, 0, 0),
                &GraphVersion::new(1, 1, 0),
            )
            .unwrap();
        assert!(result.nodes.contains(&"cache".to_string()));
    }

    #[test]
    fn test_pipeline_multi_step() {
        let mut pipeline = MigrationPipeline::new();
        pipeline.add_step(MigrationStep::new(
            GraphVersion::new(1, 0, 0),
            GraphVersion::new(1, 1, 0),
            |mut def| {
                def.nodes.push("cache".to_string());
                Ok(def)
            },
        ));
        pipeline.add_step(MigrationStep::new(
            GraphVersion::new(1, 1, 0),
            GraphVersion::new(2, 0, 0),
            |mut def| {
                def.nodes.push("logger".to_string());
                Ok(def)
            },
        ));
        let result = pipeline
            .migrate(
                simple_graph(),
                &GraphVersion::new(1, 0, 0),
                &GraphVersion::new(2, 0, 0),
            )
            .unwrap();
        assert!(result.nodes.contains(&"cache".to_string()));
        assert!(result.nodes.contains(&"logger".to_string()));
    }

    #[test]
    fn test_pipeline_no_path() {
        let pipeline = MigrationPipeline::new();
        let result = pipeline.migrate(
            simple_graph(),
            &GraphVersion::new(1, 0, 0),
            &GraphVersion::new(3, 0, 0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_pipeline_has_path() {
        let mut pipeline = MigrationPipeline::new();
        pipeline.add_step(MigrationStep::new(
            GraphVersion::new(1, 0, 0),
            GraphVersion::new(2, 0, 0),
            |def| Ok(def),
        ));
        assert!(pipeline.has_path(&GraphVersion::new(1, 0, 0), &GraphVersion::new(2, 0, 0)));
        assert!(!pipeline.has_path(&GraphVersion::new(1, 0, 0), &GraphVersion::new(3, 0, 0)));
        assert!(pipeline.has_path(&GraphVersion::new(1, 0, 0), &GraphVersion::new(1, 0, 0)));
    }

    #[test]
    fn test_pipeline_list_steps() {
        let mut pipeline = MigrationPipeline::new();
        pipeline.add_step(MigrationStep::new(
            GraphVersion::new(1, 0, 0),
            GraphVersion::new(2, 0, 0),
            |def| Ok(def),
        ));
        let steps = pipeline.list_steps();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].0, GraphVersion::new(1, 0, 0));
        assert_eq!(steps[0].1, GraphVersion::new(2, 0, 0));
    }

    #[test]
    fn test_pipeline_default() {
        let pipeline = MigrationPipeline::default();
        assert!(pipeline.list_steps().is_empty());
    }

    // === GraphDiff tests ===

    #[test]
    fn test_diff_identical() {
        let g = simple_graph();
        let diff = GraphDiff::compute(&g, &g);
        assert!(diff.is_empty());
    }

    #[test]
    fn test_diff_added_nodes() {
        let old = simple_graph();
        let new = extended_graph();
        let diff = GraphDiff::compute(&old, &new);
        assert!(diff.added_nodes.contains(&"review".to_string()));
    }

    #[test]
    fn test_diff_removed_nodes() {
        let old = extended_graph();
        let new = simple_graph();
        let diff = GraphDiff::compute(&old, &new);
        assert!(diff.removed_nodes.contains(&"review".to_string()));
    }

    #[test]
    fn test_diff_entry_point_changed() {
        let old = simple_graph();
        let mut new = simple_graph();
        new.entry_point = "tools".to_string();
        let diff = GraphDiff::compute(&old, &new);
        assert!(diff.entry_point_changed);
    }

    #[test]
    fn test_diff_entry_point_unchanged() {
        let g = simple_graph();
        let diff = GraphDiff::compute(&g, &g);
        assert!(!diff.entry_point_changed);
    }

    #[test]
    fn test_diff_edges_added() {
        let old = simple_graph();
        let mut new = simple_graph();
        new.edges.push(("tools".to_string(), END.to_string()));
        let diff = GraphDiff::compute(&old, &new);
        assert!(!diff.added_edges.is_empty());
    }

    #[test]
    fn test_diff_edges_removed() {
        let old = simple_graph();
        let mut new = simple_graph();
        new.edges.retain(|e| e.0 != "tools");
        let diff = GraphDiff::compute(&old, &new);
        assert!(!diff.removed_edges.is_empty());
    }

    #[test]
    fn test_diff_modified_node_edges() {
        let old = simple_graph();
        let mut new = simple_graph();
        // Change an edge involving "tools"
        new.edges.retain(|e| !(e.0 == "tools" && e.1 == "agent"));
        new.edges.push(("tools".to_string(), END.to_string()));
        let diff = GraphDiff::compute(&old, &new);
        assert!(
            diff.modified_nodes.iter().any(|(n, _)| n == "tools")
                || diff.modified_nodes.iter().any(|(n, _)| n == "agent")
        );
    }

    #[test]
    fn test_diff_interrupt_changed() {
        let old = simple_graph();
        let mut new = simple_graph();
        new.interrupt_before.push("agent".to_string());
        let diff = GraphDiff::compute(&old, &new);
        assert!(diff
            .modified_nodes
            .iter()
            .any(|(n, t)| n == "agent" && *t == ModificationType::InterruptChanged));
    }

    #[test]
    fn test_diff_conditional_edges_changed() {
        let old = simple_graph();
        let mut new = simple_graph();
        new.conditional_edges.push(ConditionalEdgeDef {
            from: "agent".to_string(),
            targets: vec!["tools".to_string(), END.to_string()],
            labels: None,
        });
        let diff = GraphDiff::compute(&old, &new);
        assert!(diff
            .modified_nodes
            .iter()
            .any(|(n, t)| n == "agent"
                && *t == ModificationType::ConditionalEdgesChanged));
    }

    #[test]
    fn test_diff_summary_no_changes() {
        let g = simple_graph();
        let diff = GraphDiff::compute(&g, &g);
        assert_eq!(diff.summary(), "no changes");
    }

    #[test]
    fn test_diff_summary_with_changes() {
        let old = simple_graph();
        let new = extended_graph();
        let diff = GraphDiff::compute(&old, &new);
        let summary = diff.summary();
        assert!(summary.contains("added nodes") || summary.contains("edges"));
    }

    #[test]
    fn test_diff_is_empty_false() {
        let old = simple_graph();
        let new = extended_graph();
        let diff = GraphDiff::compute(&old, &new);
        assert!(!diff.is_empty());
    }

    #[test]
    fn test_diff_serialize() {
        let old = simple_graph();
        let new = extended_graph();
        let diff = GraphDiff::compute(&old, &new);
        let json = serde_json::to_string(&diff).unwrap();
        let restored: GraphDiff = serde_json::from_str(&json).unwrap();
        assert_eq!(diff.added_nodes, restored.added_nodes);
        assert_eq!(diff.removed_nodes, restored.removed_nodes);
    }

    // === VersionHistory tests ===

    #[test]
    fn test_history_new() {
        let history = VersionHistory::new("my_graph");
        assert_eq!(history.graph_name, "my_graph");
        assert!(history.is_empty());
        assert_eq!(history.len(), 0);
    }

    #[test]
    fn test_history_add_version() {
        let mut history = VersionHistory::new("test");
        let v1 = VersionedGraph::new(GraphVersion::new(1, 0, 0), simple_graph(), "alice");
        history.add_version(v1, "Initial version");
        assert_eq!(history.len(), 1);
        assert!(!history.is_empty());
    }

    #[test]
    fn test_history_get_version() {
        let mut history = VersionHistory::new("test");
        let v1 = VersionedGraph::new(GraphVersion::new(1, 0, 0), simple_graph(), "alice");
        history.add_version(v1, "Initial");
        assert!(history.get_version(&GraphVersion::new(1, 0, 0)).is_some());
        assert!(history.get_version(&GraphVersion::new(2, 0, 0)).is_none());
    }

    #[test]
    fn test_history_latest() {
        let mut history = VersionHistory::new("test");
        let v1 = VersionedGraph::new(GraphVersion::new(1, 0, 0), simple_graph(), "alice");
        let v2 = VersionedGraph::new(GraphVersion::new(2, 0, 0), extended_graph(), "bob");
        history.add_version(v1, "v1");
        history.add_version(v2, "v2");
        let latest = history.latest().unwrap();
        assert_eq!(latest.version, GraphVersion::new(2, 0, 0));
    }

    #[test]
    fn test_history_list_versions() {
        let mut history = VersionHistory::new("test");
        let v1 = VersionedGraph::new(GraphVersion::new(1, 0, 0), simple_graph(), "alice");
        let v2 = VersionedGraph::new(GraphVersion::new(1, 1, 0), simple_graph(), "bob");
        history.add_version(v1, "v1");
        history.add_version(v2, "v2");
        let versions = history.list_versions();
        assert_eq!(versions.len(), 2);
        assert_eq!(*versions[0], GraphVersion::new(1, 0, 0));
        assert_eq!(*versions[1], GraphVersion::new(1, 1, 0));
    }

    #[test]
    fn test_history_diff_versions() {
        let mut history = VersionHistory::new("test");
        let v1 = VersionedGraph::new(GraphVersion::new(1, 0, 0), simple_graph(), "alice");
        let v2 = VersionedGraph::new(GraphVersion::new(2, 0, 0), extended_graph(), "bob");
        history.add_version(v1, "v1");
        history.add_version(v2, "v2");
        let diff = history
            .diff_versions(&GraphVersion::new(1, 0, 0), &GraphVersion::new(2, 0, 0))
            .unwrap();
        assert!(diff.added_nodes.contains(&"review".to_string()));
    }

    #[test]
    fn test_history_diff_version_not_found() {
        let history = VersionHistory::new("test");
        let result = history.diff_versions(
            &GraphVersion::new(1, 0, 0),
            &GraphVersion::new(2, 0, 0),
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_history_changelog() {
        let mut history = VersionHistory::new("test");
        let v1 = VersionedGraph::new(GraphVersion::new(1, 0, 0), simple_graph(), "alice");
        let v2 = VersionedGraph::new(GraphVersion::new(2, 0, 0), extended_graph(), "bob");
        history.add_version(v1, "Initial version");
        history.add_version(v2, "Added review node");
        assert_eq!(history.changelog.len(), 2);
        assert_eq!(history.changelog[0].message, "Initial version");
        assert!(history.changelog[0].diff_summary.is_none());
        assert!(history.changelog[1].diff_summary.is_some());
    }

    #[test]
    fn test_history_json_round_trip() {
        let mut history = VersionHistory::new("test");
        let v1 = VersionedGraph::new(GraphVersion::new(1, 0, 0), simple_graph(), "alice")
            .with_timestamp("1000");
        history.add_version(v1, "Initial");
        let json = history.to_json().unwrap();
        let restored = VersionHistory::from_json(&json).unwrap();
        assert_eq!(restored.graph_name, "test");
        assert_eq!(restored.len(), 1);
    }

    // === CompatibilityChecker tests ===

    #[test]
    fn test_compatibility_same_graph() {
        let g = simple_graph();
        let report = CompatibilityChecker::check(&g, &g);
        assert!(report.compatible);
        assert!(report.issues.is_empty());
    }

    #[test]
    fn test_compatibility_added_nodes_incompatible() {
        let old = simple_graph();
        let new = extended_graph();
        let report = CompatibilityChecker::check(&old, &new);
        assert!(!report.compatible);
        assert!(report.issues.iter().any(|i| i.contains("added")));
    }

    #[test]
    fn test_compatibility_removed_nodes_incompatible() {
        let old = extended_graph();
        let new = simple_graph();
        let report = CompatibilityChecker::check(&old, &new);
        assert!(!report.compatible);
        assert!(report.issues.iter().any(|i| i.contains("removed")));
    }

    #[test]
    fn test_compatibility_entry_point_changed() {
        let old = simple_graph();
        let mut new = simple_graph();
        new.entry_point = "tools".to_string();
        let report = CompatibilityChecker::check(&old, &new);
        assert!(!report.compatible);
        assert!(report.issues.iter().any(|i| i.contains("Entry point")));
    }

    #[test]
    fn test_compatibility_edge_change_compatible() {
        let old = simple_graph();
        let mut new = simple_graph();
        // Just add a new edge between existing nodes
        new.edges.push(("tools".to_string(), END.to_string()));
        let report = CompatibilityChecker::check(&old, &new);
        assert!(report.compatible);
    }

    #[test]
    fn test_compatibility_versioned() {
        let v1 = VersionedGraph::new(GraphVersion::new(1, 0, 0), simple_graph(), "alice");
        let v2 = VersionedGraph::new(GraphVersion::new(1, 0, 0), simple_graph(), "bob");
        let report = CompatibilityChecker::check_versioned(&v1, &v2);
        assert!(report.compatible);
    }

    #[test]
    fn test_backward_compatible_additions() {
        let old = simple_graph();
        let new = extended_graph();
        // Adding nodes is backward compatible (no removals)
        assert!(CompatibilityChecker::is_backward_compatible(&old, &new));
    }

    #[test]
    fn test_not_backward_compatible_removals() {
        let old = extended_graph();
        let new = simple_graph();
        assert!(!CompatibilityChecker::is_backward_compatible(&old, &new));
    }

    #[test]
    fn test_not_backward_compatible_entry_change() {
        let old = simple_graph();
        let mut new = simple_graph();
        new.entry_point = "tools".to_string();
        assert!(!CompatibilityChecker::is_backward_compatible(&old, &new));
    }

    // === VersionConstraint tests ===

    #[test]
    fn test_constraint_exact() {
        let c = VersionConstraint::Exact(GraphVersion::new(1, 2, 3));
        assert!(c.matches(&GraphVersion::new(1, 2, 3)));
        assert!(!c.matches(&GraphVersion::new(1, 2, 4)));
    }

    #[test]
    fn test_constraint_greater_or_equal() {
        let c = VersionConstraint::GreaterOrEqual(GraphVersion::new(1, 0, 0));
        assert!(c.matches(&GraphVersion::new(1, 0, 0)));
        assert!(c.matches(&GraphVersion::new(2, 0, 0)));
        assert!(!c.matches(&GraphVersion::new(0, 9, 0)));
    }

    #[test]
    fn test_constraint_less_than() {
        let c = VersionConstraint::LessThan(GraphVersion::new(2, 0, 0));
        assert!(c.matches(&GraphVersion::new(1, 9, 9)));
        assert!(!c.matches(&GraphVersion::new(2, 0, 0)));
        assert!(!c.matches(&GraphVersion::new(3, 0, 0)));
    }

    #[test]
    fn test_constraint_range() {
        let c = VersionConstraint::Range {
            min: GraphVersion::new(1, 0, 0),
            max: GraphVersion::new(2, 0, 0),
        };
        assert!(c.matches(&GraphVersion::new(1, 0, 0)));
        assert!(c.matches(&GraphVersion::new(1, 5, 0)));
        assert!(!c.matches(&GraphVersion::new(2, 0, 0)));
        assert!(!c.matches(&GraphVersion::new(0, 9, 0)));
    }

    #[test]
    fn test_constraint_compatible() {
        let c = VersionConstraint::Compatible(GraphVersion::new(1, 2, 0));
        assert!(c.matches(&GraphVersion::new(1, 2, 0)));
        assert!(c.matches(&GraphVersion::new(1, 3, 0)));
        assert!(!c.matches(&GraphVersion::new(1, 1, 0)));
        assert!(!c.matches(&GraphVersion::new(2, 0, 0)));
    }

    #[test]
    fn test_constraint_any() {
        let c = VersionConstraint::Any;
        assert!(c.matches(&GraphVersion::new(0, 0, 0)));
        assert!(c.matches(&GraphVersion::new(99, 99, 99)));
    }

    #[test]
    fn test_constraint_parse_exact() {
        let c = VersionConstraint::parse("1.2.3").unwrap();
        assert_eq!(c, VersionConstraint::Exact(GraphVersion::new(1, 2, 3)));
    }

    #[test]
    fn test_constraint_parse_gte() {
        let c = VersionConstraint::parse(">=1.0.0").unwrap();
        assert_eq!(
            c,
            VersionConstraint::GreaterOrEqual(GraphVersion::new(1, 0, 0))
        );
    }

    #[test]
    fn test_constraint_parse_lt() {
        let c = VersionConstraint::parse("<2.0.0").unwrap();
        assert_eq!(
            c,
            VersionConstraint::LessThan(GraphVersion::new(2, 0, 0))
        );
    }

    #[test]
    fn test_constraint_parse_range() {
        let c = VersionConstraint::parse(">=1.0.0,<2.0.0").unwrap();
        assert_eq!(
            c,
            VersionConstraint::Range {
                min: GraphVersion::new(1, 0, 0),
                max: GraphVersion::new(2, 0, 0),
            }
        );
    }

    #[test]
    fn test_constraint_parse_compatible() {
        let c = VersionConstraint::parse("~=1.2.0").unwrap();
        assert_eq!(
            c,
            VersionConstraint::Compatible(GraphVersion::new(1, 2, 0))
        );
    }

    #[test]
    fn test_constraint_parse_any() {
        let c = VersionConstraint::parse("*").unwrap();
        assert_eq!(c, VersionConstraint::Any);
    }

    #[test]
    fn test_constraint_parse_invalid_range() {
        assert!(VersionConstraint::parse(">=1.0.0,<2.0.0,<3.0.0").is_err());
    }

    #[test]
    fn test_constraint_display() {
        assert_eq!(
            VersionConstraint::Exact(GraphVersion::new(1, 2, 3)).to_string(),
            "1.2.3"
        );
        assert_eq!(
            VersionConstraint::GreaterOrEqual(GraphVersion::new(1, 0, 0)).to_string(),
            ">=1.0.0"
        );
        assert_eq!(
            VersionConstraint::LessThan(GraphVersion::new(2, 0, 0)).to_string(),
            "<2.0.0"
        );
        assert_eq!(VersionConstraint::Any.to_string(), "*");
    }

    #[test]
    fn test_constraint_display_range() {
        let c = VersionConstraint::Range {
            min: GraphVersion::new(1, 0, 0),
            max: GraphVersion::new(2, 0, 0),
        };
        assert_eq!(c.to_string(), ">=1.0.0,<2.0.0");
    }

    #[test]
    fn test_constraint_display_compatible() {
        let c = VersionConstraint::Compatible(GraphVersion::new(1, 2, 0));
        assert_eq!(c.to_string(), "~=1.2.0");
    }

    // === Integration / edge case tests ===

    #[test]
    fn test_pipeline_three_step_chain() {
        let mut pipeline = MigrationPipeline::new();
        pipeline.add_step(MigrationStep::new(
            GraphVersion::new(1, 0, 0),
            GraphVersion::new(1, 1, 0),
            |mut def| {
                def.interrupt_before.push("agent".to_string());
                Ok(def)
            },
        ));
        pipeline.add_step(MigrationStep::new(
            GraphVersion::new(1, 1, 0),
            GraphVersion::new(1, 2, 0),
            |mut def| {
                def.nodes.push("cache".to_string());
                Ok(def)
            },
        ));
        pipeline.add_step(MigrationStep::new(
            GraphVersion::new(1, 2, 0),
            GraphVersion::new(2, 0, 0),
            |mut def| {
                def.nodes.push("monitor".to_string());
                Ok(def)
            },
        ));
        let result = pipeline
            .migrate(
                simple_graph(),
                &GraphVersion::new(1, 0, 0),
                &GraphVersion::new(2, 0, 0),
            )
            .unwrap();
        assert!(result.interrupt_before.contains(&"agent".to_string()));
        assert!(result.nodes.contains(&"cache".to_string()));
        assert!(result.nodes.contains(&"monitor".to_string()));
    }

    #[test]
    fn test_migration_step_error_propagation() {
        let step = MigrationStep::new(
            GraphVersion::new(1, 0, 0),
            GraphVersion::new(2, 0, 0),
            |_def| Err(LangGraphError::Other("migration failed".to_string())),
        );
        let result = step.migrate(simple_graph());
        assert!(result.is_err());
    }

    #[test]
    fn test_version_hash() {
        let mut set = HashSet::new();
        set.insert(GraphVersion::new(1, 0, 0));
        set.insert(GraphVersion::new(1, 0, 0));
        assert_eq!(set.len(), 1);
        set.insert(GraphVersion::new(2, 0, 0));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_version_zero() {
        let v = GraphVersion::new(0, 0, 0);
        assert_eq!(v.to_string(), "0.0.0");
        assert_eq!(v.bump_patch(), GraphVersion::new(0, 0, 1));
    }

    #[test]
    fn test_diff_summary_entry_point_change() {
        let old = simple_graph();
        let mut new = simple_graph();
        new.entry_point = "tools".to_string();
        let diff = GraphDiff::compute(&old, &new);
        assert!(diff.summary().contains("entry point changed"));
    }

    #[test]
    fn test_history_empty_latest() {
        let history = VersionHistory::new("empty");
        assert!(history.latest().is_none());
    }
}
