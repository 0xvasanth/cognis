//! Comprehensive graph validation for LangGraph.
//!
//! This module provides structural validation, state transition rules, schema
//! compatibility checking, validation caching, and fast pre-flight checks.
//!
//! # Overview
//!
//! - [`GraphValidator`] — validates graph topology (unreachable nodes, missing
//!   entry/finish, dangling edges, self-loops, duplicate edges, orphan nodes).
//! - [`StateValidator`] — validates state transitions against a set of [`StateRule`]s.
//! - [`SchemaValidator`] — validates that connected nodes have compatible
//!   input/output key schemas.
//! - [`ValidationCache`] — caches [`ValidationReport`]s keyed by graph hash.
//! - [`QuickCheck`] — fast boolean pre-flight checks (entry exists, finish
//!   exists, connected, cycles).

use serde_json::Value;
use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;

// ---------------------------------------------------------------------------
// ValidationLevel
// ---------------------------------------------------------------------------

/// Severity level for a validation finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValidationLevel {
    /// A fatal problem that prevents compilation.
    Error,
    /// A potential problem that may cause unexpected behaviour.
    Warning,
    /// An informational note.
    Info,
}

impl ValidationLevel {
    /// Numeric severity — higher means more severe.
    ///
    /// `Error = 2`, `Warning = 1`, `Info = 0`.
    pub fn severity(&self) -> u8 {
        match self {
            Self::Error => 2,
            Self::Warning => 1,
            Self::Info => 0,
        }
    }
}

impl fmt::Display for ValidationLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error => write!(f, "ERROR"),
            Self::Warning => write!(f, "WARNING"),
            Self::Info => write!(f, "INFO"),
        }
    }
}

impl PartialOrd for ValidationLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ValidationLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.severity().cmp(&other.severity())
    }
}

// ---------------------------------------------------------------------------
// ValidationIssue
// ---------------------------------------------------------------------------

/// A single validation finding with level, code, message, and optional context.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    /// Severity of this issue.
    pub level: ValidationLevel,
    /// Short, unique code such as `"UNREACHABLE_NODE"`.
    pub code: String,
    /// Human-readable description.
    pub message: String,
    /// The node most closely associated with this issue, if any.
    pub node_id: Option<String>,
    /// Actionable suggestion for fixing the issue.
    pub suggestion: Option<String>,
}

impl ValidationIssue {
    /// Create a new issue with the given level, code, and message.
    pub fn new(
        level: ValidationLevel,
        code: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            level,
            code: code.into(),
            message: message.into(),
            node_id: None,
            suggestion: None,
        }
    }

    /// Attach a node id to this issue (builder-style).
    pub fn for_node(mut self, node_id: impl Into<String>) -> Self {
        self.node_id = Some(node_id.into());
        self
    }

    /// Attach a suggestion to this issue (builder-style).
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }

    /// Serialize this issue to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("level".into(), Value::String(self.level.to_string()));
        map.insert("code".into(), Value::String(self.code.clone()));
        map.insert("message".into(), Value::String(self.message.clone()));
        if let Some(ref nid) = self.node_id {
            map.insert("node_id".into(), Value::String(nid.clone()));
        }
        if let Some(ref s) = self.suggestion {
            map.insert("suggestion".into(), Value::String(s.clone()));
        }
        Value::Object(map)
    }
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.level, self.code)?;
        if let Some(ref nid) = self.node_id {
            write!(f, " (node: {})", nid)?;
        }
        write!(f, ": {}", self.message)?;
        if let Some(ref s) = self.suggestion {
            write!(f, " — suggestion: {}", s)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ValidationReport
// ---------------------------------------------------------------------------

/// Collection of [`ValidationIssue`]s with filtering and summary helpers.
#[derive(Debug, Clone, Default)]
pub struct ValidationReport {
    issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Create an empty report.
    pub fn new() -> Self {
        Self { issues: Vec::new() }
    }

    /// Add an issue to the report.
    pub fn add_issue(&mut self, issue: ValidationIssue) {
        self.issues.push(issue);
    }

    /// Return all error-level issues.
    pub fn errors(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.level == ValidationLevel::Error)
            .collect()
    }

    /// Return all warning-level issues.
    pub fn warnings(&self) -> Vec<&ValidationIssue> {
        self.issues
            .iter()
            .filter(|i| i.level == ValidationLevel::Warning)
            .collect()
    }

    /// Whether the report contains any error-level issues.
    pub fn has_errors(&self) -> bool {
        self.issues
            .iter()
            .any(|i| i.level == ValidationLevel::Error)
    }

    /// Whether the report contains any warning-level issues.
    pub fn has_warnings(&self) -> bool {
        self.issues
            .iter()
            .any(|i| i.level == ValidationLevel::Warning)
    }

    /// Total number of issues.
    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }

    /// The report is valid when there are no error-level issues.
    pub fn is_valid(&self) -> bool {
        !self.has_errors()
    }

    /// Serialize the report to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let issues: Vec<Value> = self.issues.iter().map(|i| i.to_json()).collect();
        let mut map = serde_json::Map::new();
        map.insert("valid".into(), Value::Bool(self.is_valid()));
        map.insert(
            "issue_count".into(),
            Value::Number(self.issue_count().into()),
        );
        map.insert("issues".into(), Value::Array(issues));
        Value::Object(map)
    }

    /// Return a human-readable one-line summary.
    pub fn summary(&self) -> String {
        let errors = self.errors().len();
        let warnings = self.warnings().len();
        let infos = self.issue_count() - errors - warnings;
        if self.issues.is_empty() {
            "Validation passed: no issues found.".to_string()
        } else {
            format!(
                "Validation {}: {} error(s), {} warning(s), {} info(s).",
                if self.is_valid() { "passed" } else { "failed" },
                errors,
                warnings,
                infos,
            )
        }
    }

    /// Borrow the full list of issues.
    pub fn issues(&self) -> &[ValidationIssue] {
        &self.issues
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{}", self.summary())?;
        for issue in &self.issues {
            writeln!(f, "  {}", issue)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// GraphValidator (structural)
// ---------------------------------------------------------------------------

/// Validates graph topology: unreachable nodes, missing entry/finish, dangling
/// edges, self-loops, duplicate edges, and orphan nodes.
#[derive(Default)]
pub struct GraphValidator;

impl GraphValidator {
    /// Create a new `GraphValidator`. (Stateless — provided for API symmetry.)
    pub fn new() -> Self {
        Self
    }

    /// Run all structural checks on the given graph.
    ///
    /// * `nodes` — list of node identifiers.
    /// * `edges` — list of `(source, target)` directed edges.
    /// * `entry` — optional entry node name.
    /// * `finish` — optional finish/end node name.
    pub fn validate(
        &self,
        nodes: &[String],
        edges: &[(String, String)],
        entry: Option<&str>,
        finish: Option<&str>,
    ) -> ValidationReport {
        let mut report = ValidationReport::new();

        self.check_missing_entry(nodes, entry, &mut report);
        self.check_missing_finish(nodes, finish, &mut report);
        self.check_dangling_edges(nodes, edges, &mut report);
        self.check_self_loops(edges, &mut report);
        self.check_duplicate_edges(edges, &mut report);
        self.check_orphan_nodes(nodes, edges, &mut report);
        self.check_unreachable_nodes(nodes, edges, entry, &mut report);

        report
    }

    // -- individual checks --------------------------------------------------

    fn check_missing_entry(
        &self,
        nodes: &[String],
        entry: Option<&str>,
        report: &mut ValidationReport,
    ) {
        match entry {
            None => {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Error,
                        "MISSING_ENTRY",
                        "No entry node specified",
                    )
                    .with_suggestion("Specify an entry node for the graph"),
                );
            }
            Some(e) if !nodes.iter().any(|n| n == e) => {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Error,
                        "INVALID_ENTRY",
                        format!("Entry node '{}' does not exist in the graph", e),
                    )
                    .for_node(e)
                    .with_suggestion("Use an existing node as the entry point"),
                );
            }
            _ => {}
        }
    }

    fn check_missing_finish(
        &self,
        nodes: &[String],
        finish: Option<&str>,
        report: &mut ValidationReport,
    ) {
        match finish {
            None => {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Warning,
                        "MISSING_FINISH",
                        "No finish node specified",
                    )
                    .with_suggestion("Specify a finish node for the graph"),
                );
            }
            Some(f) if !nodes.iter().any(|n| n == f) => {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Error,
                        "INVALID_FINISH",
                        format!("Finish node '{}' does not exist in the graph", f),
                    )
                    .for_node(f)
                    .with_suggestion("Use an existing node as the finish point"),
                );
            }
            _ => {}
        }
    }

    fn check_dangling_edges(
        &self,
        nodes: &[String],
        edges: &[(String, String)],
        report: &mut ValidationReport,
    ) {
        let node_set: HashSet<&str> = nodes.iter().map(String::as_str).collect();
        for (from, to) in edges {
            if !node_set.contains(from.as_str()) {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Error,
                        "DANGLING_EDGE_SOURCE",
                        format!("Edge source '{}' is not a known node", from),
                    )
                    .for_node(from)
                    .with_suggestion("Add the missing node or remove the edge"),
                );
            }
            if !node_set.contains(to.as_str()) {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Error,
                        "DANGLING_EDGE_TARGET",
                        format!("Edge target '{}' is not a known node", to),
                    )
                    .for_node(to)
                    .with_suggestion("Add the missing node or remove the edge"),
                );
            }
        }
    }

    fn check_self_loops(&self, edges: &[(String, String)], report: &mut ValidationReport) {
        for (from, to) in edges {
            if from == to {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Warning,
                        "SELF_LOOP",
                        format!("Node '{}' has an edge to itself", from),
                    )
                    .for_node(from)
                    .with_suggestion("Remove the self-loop unless intentional"),
                );
            }
        }
    }

    fn check_duplicate_edges(&self, edges: &[(String, String)], report: &mut ValidationReport) {
        let mut seen: HashSet<(&str, &str)> = HashSet::new();
        for (from, to) in edges {
            if !seen.insert((from.as_str(), to.as_str())) {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Warning,
                        "DUPLICATE_EDGE",
                        format!("Duplicate edge from '{}' to '{}'", from, to),
                    )
                    .with_suggestion("Remove the duplicate edge"),
                );
            }
        }
    }

    fn check_orphan_nodes(
        &self,
        nodes: &[String],
        edges: &[(String, String)],
        report: &mut ValidationReport,
    ) {
        let mut referenced: HashSet<&str> = HashSet::new();
        for (from, to) in edges {
            referenced.insert(from.as_str());
            referenced.insert(to.as_str());
        }
        for node in nodes {
            if !referenced.contains(node.as_str()) {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Warning,
                        "ORPHAN_NODE",
                        format!("Node '{}' has no edges", node),
                    )
                    .for_node(node)
                    .with_suggestion("Connect the node or remove it"),
                );
            }
        }
    }

    fn check_unreachable_nodes(
        &self,
        nodes: &[String],
        edges: &[(String, String)],
        entry: Option<&str>,
        report: &mut ValidationReport,
    ) {
        let entry = match entry {
            Some(e) if nodes.iter().any(|n| n == e) => e,
            _ => return, // can't check reachability without a valid entry
        };

        // BFS from entry
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for (from, to) in edges {
            adj.entry(from.as_str()).or_default().push(to.as_str());
        }

        let mut visited: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<&str> = VecDeque::new();
        visited.insert(entry);
        queue.push_back(entry);
        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = adj.get(current) {
                for &neighbor in neighbors {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        for node in nodes {
            if node != entry && !visited.contains(node.as_str()) {
                report.add_issue(
                    ValidationIssue::new(
                        ValidationLevel::Error,
                        "UNREACHABLE_NODE",
                        format!("Node '{}' is not reachable from entry '{}'", node, entry),
                    )
                    .for_node(node)
                    .with_suggestion("Add an edge path from the entry node"),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StateRule
// ---------------------------------------------------------------------------

/// A rule for validating state transitions.
#[derive(Debug, Clone)]
pub enum StateRule {
    /// The given key must exist in the state.
    RequiredKey(String),
    /// The given key must not change between states.
    ImmutableKey(String),
    /// The value at the given key must match the expected JSON type name
    /// (`"string"`, `"number"`, `"boolean"`, `"array"`, `"object"`, `"null"`).
    TypeConstraint { key: String, expected: String },
    /// A custom rule placeholder with a name and description.
    Custom { name: String, description: String },
}

impl StateRule {
    /// Create a required-key rule.
    pub fn required_key(key: impl Into<String>) -> Self {
        Self::RequiredKey(key.into())
    }

    /// Create an immutable-key rule.
    pub fn immutable_key(key: impl Into<String>) -> Self {
        Self::ImmutableKey(key.into())
    }

    /// Create a type-constraint rule.
    pub fn type_constraint(key: impl Into<String>, expected: impl Into<String>) -> Self {
        Self::TypeConstraint {
            key: key.into(),
            expected: expected.into(),
        }
    }

    /// Create a custom rule placeholder.
    pub fn custom(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self::Custom {
            name: name.into(),
            description: description.into(),
        }
    }

    /// Human-readable name for this rule.
    pub fn name(&self) -> &str {
        match self {
            Self::RequiredKey(k) => k.as_str(),
            Self::ImmutableKey(k) => k.as_str(),
            Self::TypeConstraint { key, .. } => key.as_str(),
            Self::Custom { name, .. } => name.as_str(),
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        match self {
            Self::RequiredKey(k) => {
                map.insert("type".into(), Value::String("required_key".into()));
                map.insert("key".into(), Value::String(k.clone()));
            }
            Self::ImmutableKey(k) => {
                map.insert("type".into(), Value::String("immutable_key".into()));
                map.insert("key".into(), Value::String(k.clone()));
            }
            Self::TypeConstraint { key, expected } => {
                map.insert("type".into(), Value::String("type_constraint".into()));
                map.insert("key".into(), Value::String(key.clone()));
                map.insert("expected".into(), Value::String(expected.clone()));
            }
            Self::Custom { name, description } => {
                map.insert("type".into(), Value::String("custom".into()));
                map.insert("name".into(), Value::String(name.clone()));
                map.insert("description".into(), Value::String(description.clone()));
            }
        }
        Value::Object(map)
    }
}

// ---------------------------------------------------------------------------
// StateValidator
// ---------------------------------------------------------------------------

/// Validates state transitions against a collection of [`StateRule`]s.
#[derive(Default)]
pub struct StateValidator {
    rules: Vec<StateRule>,
}

impl StateValidator {
    /// Create a new, empty state validator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a rule.
    pub fn add_rule(&mut self, rule: StateRule) {
        self.rules.push(rule);
    }

    /// Number of registered rules.
    pub fn rule_count(&self) -> usize {
        self.rules.len()
    }

    /// Validate a state transition from `from` to `to`.
    ///
    /// Both values are expected to be JSON objects (maps). Rules are checked
    /// against the `to` state; immutable-key rules compare `from` with `to`.
    pub fn validate_transition(&self, from: &Value, to: &Value) -> ValidationReport {
        let mut report = ValidationReport::new();
        let to_obj = to.as_object();
        let from_obj = from.as_object();

        for rule in &self.rules {
            match rule {
                StateRule::RequiredKey(key) => {
                    if let Some(obj) = to_obj {
                        if !obj.contains_key(key) {
                            report.add_issue(
                                ValidationIssue::new(
                                    ValidationLevel::Error,
                                    "MISSING_REQUIRED_KEY",
                                    format!("Required key '{}' is missing from state", key),
                                )
                                .with_suggestion(format!(
                                    "Ensure '{}' is present in the state",
                                    key
                                )),
                            );
                        }
                    } else {
                        report.add_issue(ValidationIssue::new(
                            ValidationLevel::Error,
                            "INVALID_STATE_TYPE",
                            "State must be a JSON object",
                        ));
                    }
                }
                StateRule::ImmutableKey(key) => {
                    if let (Some(f), Some(t)) = (from_obj, to_obj) {
                        let fv = f.get(key);
                        let tv = t.get(key);
                        if fv.is_some() && tv.is_some() && fv != tv {
                            report.add_issue(
                                ValidationIssue::new(
                                    ValidationLevel::Error,
                                    "IMMUTABLE_KEY_CHANGED",
                                    format!("Immutable key '{}' was modified", key),
                                )
                                .with_suggestion(format!("Do not modify '{}'", key)),
                            );
                        }
                    }
                }
                StateRule::TypeConstraint { key, expected } => {
                    if let Some(obj) = to_obj {
                        if let Some(val) = obj.get(key) {
                            let actual = json_type_name(val);
                            if actual != expected.as_str() {
                                report.add_issue(
                                    ValidationIssue::new(
                                        ValidationLevel::Error,
                                        "TYPE_MISMATCH",
                                        format!(
                                            "Key '{}' expected type '{}' but got '{}'",
                                            key, expected, actual
                                        ),
                                    )
                                    .with_suggestion(format!(
                                        "Change '{}' to type '{}'",
                                        key, expected
                                    )),
                                );
                            }
                        }
                    }
                }
                StateRule::Custom { name, description } => {
                    // Custom rules are placeholders — emit an info.
                    report.add_issue(ValidationIssue::new(
                        ValidationLevel::Info,
                        "CUSTOM_RULE",
                        format!("Custom rule '{}': {}", name, description),
                    ));
                }
            }
        }

        report
    }
}

/// Return the JSON type name for a value.
fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

// ---------------------------------------------------------------------------
// SchemaValidator
// ---------------------------------------------------------------------------

/// Validates that connected nodes have compatible input/output key schemas.
#[derive(Default)]
pub struct SchemaValidator {
    /// node_id -> (input_keys, output_keys)
    schemas: HashMap<String, (Vec<String>, Vec<String>)>,
}

impl SchemaValidator {
    /// Create a new, empty schema validator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register expected input and output keys for a node.
    pub fn register_node_schema(
        &mut self,
        node_id: &str,
        input_keys: Vec<String>,
        output_keys: Vec<String>,
    ) {
        self.schemas
            .insert(node_id.to_string(), (input_keys, output_keys));
    }

    /// Number of registered node schemas.
    pub fn node_count(&self) -> usize {
        self.schemas.len()
    }

    /// Validate that for every edge `(source, target)`, the source's output
    /// keys are a superset of the target's input keys.
    pub fn validate_connections(&self, edges: &[(String, String)]) -> ValidationReport {
        let mut report = ValidationReport::new();

        for (source, target) in edges {
            let source_outputs = self.schemas.get(source).map(|(_, o)| o);
            let target_inputs = self.schemas.get(target).map(|(i, _)| i);

            match (source_outputs, target_inputs) {
                (Some(outputs), Some(inputs)) => {
                    let output_set: HashSet<&str> = outputs.iter().map(String::as_str).collect();
                    for input_key in inputs {
                        if !output_set.contains(input_key.as_str()) {
                            report.add_issue(
                                ValidationIssue::new(
                                    ValidationLevel::Error,
                                    "SCHEMA_MISMATCH",
                                    format!(
                                        "Node '{}' requires input key '{}' but '{}' does not produce it",
                                        target, input_key, source
                                    ),
                                )
                                .for_node(target)
                                .with_suggestion(format!(
                                    "Add '{}' to the output of '{}' or remove it from input of '{}'",
                                    input_key, source, target
                                )),
                            );
                        }
                    }
                }
                (None, Some(_)) => {
                    report.add_issue(
                        ValidationIssue::new(
                            ValidationLevel::Warning,
                            "MISSING_SOURCE_SCHEMA",
                            format!("No schema registered for source node '{}'", source),
                        )
                        .for_node(source),
                    );
                }
                (Some(_), None) => {
                    report.add_issue(
                        ValidationIssue::new(
                            ValidationLevel::Warning,
                            "MISSING_TARGET_SCHEMA",
                            format!("No schema registered for target node '{}'", target),
                        )
                        .for_node(target),
                    );
                }
                (None, None) => {
                    // Neither side has a schema — nothing to check.
                }
            }
        }

        report
    }
}

// ---------------------------------------------------------------------------
// ValidationCache
// ---------------------------------------------------------------------------

/// Caches [`ValidationReport`]s keyed by graph hash strings.
#[derive(Default)]
pub struct ValidationCache {
    cache: HashMap<String, ValidationReport>,
}

impl ValidationCache {
    /// Create a new, empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store a report for the given graph hash.
    pub fn store(&mut self, graph_hash: &str, report: ValidationReport) {
        self.cache.insert(graph_hash.to_string(), report);
    }

    /// Retrieve a cached report by graph hash.
    pub fn get(&self, graph_hash: &str) -> Option<&ValidationReport> {
        self.cache.get(graph_hash)
    }

    /// Remove a cached report.
    pub fn invalidate(&mut self, graph_hash: &str) {
        self.cache.remove(graph_hash);
    }

    /// Clear all cached reports.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Number of cached reports.
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

// ---------------------------------------------------------------------------
// QuickCheck
// ---------------------------------------------------------------------------

/// Fast boolean pre-flight checks for common graph properties.
pub struct QuickCheck;

impl QuickCheck {
    /// Returns `true` if the entry node exists in the node list.
    pub fn has_entry(nodes: &[String], entry: &str) -> bool {
        nodes.iter().any(|n| n == entry)
    }

    /// Returns `true` if the finish node exists in the node list.
    pub fn has_finish(nodes: &[String], finish: &str) -> bool {
        nodes.iter().any(|n| n == finish)
    }

    /// Returns `true` if every node is reachable from at least one other node
    /// or is the source of at least one edge (i.e., the graph is weakly connected).
    pub fn is_connected(nodes: &[String], edges: &[(String, String)]) -> bool {
        if nodes.is_empty() {
            return true;
        }
        if nodes.len() == 1 {
            return true;
        }

        // Build undirected adjacency
        let mut adj: HashMap<&str, HashSet<&str>> = HashMap::new();
        for node in nodes {
            adj.entry(node.as_str()).or_default();
        }
        for (from, to) in edges {
            adj.entry(from.as_str()).or_default().insert(to.as_str());
            adj.entry(to.as_str()).or_default().insert(from.as_str());
        }

        // BFS from first node
        let start = nodes[0].as_str();
        let mut visited: HashSet<&str> = HashSet::new();
        let mut queue: VecDeque<&str> = VecDeque::new();
        visited.insert(start);
        queue.push_back(start);
        while let Some(current) = queue.pop_front() {
            if let Some(neighbors) = adj.get(current) {
                for &neighbor in neighbors {
                    if visited.insert(neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        nodes.iter().all(|n| visited.contains(n.as_str()))
    }

    /// Returns `true` if the directed graph contains at least one cycle.
    pub fn has_cycles(nodes: &[String], edges: &[(String, String)]) -> bool {
        let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
        for (from, to) in edges {
            adj.entry(from.as_str()).or_default().push(to.as_str());
        }

        let mut visited: HashSet<&str> = HashSet::new();
        let mut on_stack: HashSet<&str> = HashSet::new();

        for node in nodes {
            if !visited.contains(node.as_str())
                && Self::dfs_has_cycle(node.as_str(), &adj, &mut visited, &mut on_stack)
            {
                return true;
            }
        }
        false
    }

    fn dfs_has_cycle<'a>(
        node: &'a str,
        adj: &HashMap<&'a str, Vec<&'a str>>,
        visited: &mut HashSet<&'a str>,
        on_stack: &mut HashSet<&'a str>,
    ) -> bool {
        visited.insert(node);
        on_stack.insert(node);

        if let Some(neighbors) = adj.get(node) {
            for &neighbor in neighbors {
                if !visited.contains(neighbor) {
                    if Self::dfs_has_cycle(neighbor, adj, visited, on_stack) {
                        return true;
                    }
                } else if on_stack.contains(neighbor) {
                    return true;
                }
            }
        }

        on_stack.remove(node);
        false
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- helpers -----------------------------------------------------------

    fn nodes(names: &[&str]) -> Vec<String> {
        names.iter().map(|s| s.to_string()).collect()
    }

    fn edges(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(a, b)| (a.to_string(), b.to_string()))
            .collect()
    }

    // =====================================================================
    // ValidationLevel
    // =====================================================================

    #[test]
    fn test_level_severity_ordering() {
        assert!(ValidationLevel::Error > ValidationLevel::Warning);
        assert!(ValidationLevel::Warning > ValidationLevel::Info);
    }

    #[test]
    fn test_level_severity_values() {
        assert_eq!(ValidationLevel::Error.severity(), 2);
        assert_eq!(ValidationLevel::Warning.severity(), 1);
        assert_eq!(ValidationLevel::Info.severity(), 0);
    }

    #[test]
    fn test_level_display() {
        assert_eq!(ValidationLevel::Error.to_string(), "ERROR");
        assert_eq!(ValidationLevel::Warning.to_string(), "WARNING");
        assert_eq!(ValidationLevel::Info.to_string(), "INFO");
    }

    #[test]
    fn test_level_equality() {
        assert_eq!(ValidationLevel::Error, ValidationLevel::Error);
        assert_ne!(ValidationLevel::Error, ValidationLevel::Warning);
    }

    // =====================================================================
    // ValidationIssue
    // =====================================================================

    #[test]
    fn test_issue_new() {
        let issue = ValidationIssue::new(ValidationLevel::Error, "CODE", "msg");
        assert_eq!(issue.level, ValidationLevel::Error);
        assert_eq!(issue.code, "CODE");
        assert_eq!(issue.message, "msg");
        assert!(issue.node_id.is_none());
        assert!(issue.suggestion.is_none());
    }

    #[test]
    fn test_issue_for_node() {
        let issue = ValidationIssue::new(ValidationLevel::Warning, "W1", "warn").for_node("n1");
        assert_eq!(issue.node_id.as_deref(), Some("n1"));
    }

    #[test]
    fn test_issue_with_suggestion() {
        let issue =
            ValidationIssue::new(ValidationLevel::Info, "I1", "info").with_suggestion("fix it");
        assert_eq!(issue.suggestion.as_deref(), Some("fix it"));
    }

    #[test]
    fn test_issue_builder_chain() {
        let issue = ValidationIssue::new(ValidationLevel::Error, "E1", "err")
            .for_node("node_a")
            .with_suggestion("do something");
        assert_eq!(issue.node_id.as_deref(), Some("node_a"));
        assert_eq!(issue.suggestion.as_deref(), Some("do something"));
    }

    #[test]
    fn test_issue_display_basic() {
        let issue = ValidationIssue::new(ValidationLevel::Error, "E1", "bad thing");
        let s = issue.to_string();
        assert!(s.contains("[ERROR]"));
        assert!(s.contains("E1"));
        assert!(s.contains("bad thing"));
    }

    #[test]
    fn test_issue_display_with_node() {
        let issue =
            ValidationIssue::new(ValidationLevel::Warning, "W1", "warn").for_node("my_node");
        let s = issue.to_string();
        assert!(s.contains("my_node"));
    }

    #[test]
    fn test_issue_display_with_suggestion() {
        let issue =
            ValidationIssue::new(ValidationLevel::Info, "I1", "note").with_suggestion("try this");
        let s = issue.to_string();
        assert!(s.contains("try this"));
    }

    #[test]
    fn test_issue_to_json() {
        let issue = ValidationIssue::new(ValidationLevel::Error, "E1", "msg")
            .for_node("n1")
            .with_suggestion("fix");
        let j = issue.to_json();
        assert_eq!(j["level"], "ERROR");
        assert_eq!(j["code"], "E1");
        assert_eq!(j["message"], "msg");
        assert_eq!(j["node_id"], "n1");
        assert_eq!(j["suggestion"], "fix");
    }

    #[test]
    fn test_issue_to_json_minimal() {
        let issue = ValidationIssue::new(ValidationLevel::Info, "I1", "info");
        let j = issue.to_json();
        assert!(j.get("node_id").is_none());
        assert!(j.get("suggestion").is_none());
    }

    // =====================================================================
    // ValidationReport
    // =====================================================================

    #[test]
    fn test_report_new_is_valid() {
        let report = ValidationReport::new();
        assert!(report.is_valid());
        assert!(!report.has_errors());
        assert!(!report.has_warnings());
        assert_eq!(report.issue_count(), 0);
    }

    #[test]
    fn test_report_add_issue() {
        let mut report = ValidationReport::new();
        report.add_issue(ValidationIssue::new(ValidationLevel::Error, "E1", "err"));
        assert_eq!(report.issue_count(), 1);
        assert!(report.has_errors());
        assert!(!report.is_valid());
    }

    #[test]
    fn test_report_errors_filter() {
        let mut report = ValidationReport::new();
        report.add_issue(ValidationIssue::new(ValidationLevel::Error, "E1", "err"));
        report.add_issue(ValidationIssue::new(ValidationLevel::Warning, "W1", "warn"));
        report.add_issue(ValidationIssue::new(ValidationLevel::Info, "I1", "info"));
        assert_eq!(report.errors().len(), 1);
        assert_eq!(report.warnings().len(), 1);
    }

    #[test]
    fn test_report_has_warnings() {
        let mut report = ValidationReport::new();
        report.add_issue(ValidationIssue::new(ValidationLevel::Warning, "W1", "warn"));
        assert!(report.has_warnings());
        assert!(report.is_valid()); // warnings don't make it invalid
    }

    #[test]
    fn test_report_summary_empty() {
        let report = ValidationReport::new();
        assert!(report.summary().contains("no issues"));
    }

    #[test]
    fn test_report_summary_with_errors() {
        let mut report = ValidationReport::new();
        report.add_issue(ValidationIssue::new(ValidationLevel::Error, "E1", "err"));
        let s = report.summary();
        assert!(s.contains("failed"));
        assert!(s.contains("1 error(s)"));
    }

    #[test]
    fn test_report_summary_with_warnings_only() {
        let mut report = ValidationReport::new();
        report.add_issue(ValidationIssue::new(ValidationLevel::Warning, "W1", "w"));
        let s = report.summary();
        assert!(s.contains("passed"));
        assert!(s.contains("1 warning(s)"));
    }

    #[test]
    fn test_report_to_json() {
        let mut report = ValidationReport::new();
        report.add_issue(ValidationIssue::new(ValidationLevel::Error, "E1", "e"));
        let j = report.to_json();
        assert_eq!(j["valid"], false);
        assert_eq!(j["issue_count"], 1);
        assert!(j["issues"].is_array());
    }

    #[test]
    fn test_report_display() {
        let mut report = ValidationReport::new();
        report.add_issue(ValidationIssue::new(ValidationLevel::Error, "E1", "err"));
        let s = report.to_string();
        assert!(s.contains("E1"));
    }

    #[test]
    fn test_report_issues_accessor() {
        let mut report = ValidationReport::new();
        report.add_issue(ValidationIssue::new(ValidationLevel::Info, "I1", "info"));
        assert_eq!(report.issues().len(), 1);
    }

    // =====================================================================
    // GraphValidator — structural checks
    // =====================================================================

    #[test]
    fn test_gv_valid_simple_graph() {
        let gv = GraphValidator::new();
        let n = nodes(&["a", "b", "c"]);
        let e = edges(&[("a", "b"), ("b", "c")]);
        let report = gv.validate(&n, &e, Some("a"), Some("c"));
        assert!(report.is_valid());
    }

    #[test]
    fn test_gv_missing_entry() {
        let gv = GraphValidator::new();
        let n = nodes(&["a"]);
        let e = edges(&[]);
        let report = gv.validate(&n, &e, None, Some("a"));
        assert!(!report.is_valid());
        assert!(report.errors().iter().any(|i| i.code == "MISSING_ENTRY"));
    }

    #[test]
    fn test_gv_invalid_entry() {
        let gv = GraphValidator::new();
        let n = nodes(&["a"]);
        let e = edges(&[]);
        let report = gv.validate(&n, &e, Some("ghost"), Some("a"));
        assert!(!report.is_valid());
        assert!(report.errors().iter().any(|i| i.code == "INVALID_ENTRY"));
    }

    #[test]
    fn test_gv_missing_finish() {
        let gv = GraphValidator::new();
        let n = nodes(&["a"]);
        let e = edges(&[]);
        let report = gv.validate(&n, &e, Some("a"), None);
        assert!(report.has_warnings());
        assert!(report.warnings().iter().any(|i| i.code == "MISSING_FINISH"));
    }

    #[test]
    fn test_gv_invalid_finish() {
        let gv = GraphValidator::new();
        let n = nodes(&["a"]);
        let e = edges(&[]);
        let report = gv.validate(&n, &e, Some("a"), Some("ghost"));
        assert!(!report.is_valid());
        assert!(report.errors().iter().any(|i| i.code == "INVALID_FINISH"));
    }

    #[test]
    fn test_gv_dangling_edge_source() {
        let gv = GraphValidator::new();
        let n = nodes(&["a"]);
        let e = edges(&[("ghost", "a")]);
        let report = gv.validate(&n, &e, Some("a"), Some("a"));
        assert!(!report.is_valid());
        assert!(report
            .errors()
            .iter()
            .any(|i| i.code == "DANGLING_EDGE_SOURCE"));
    }

    #[test]
    fn test_gv_dangling_edge_target() {
        let gv = GraphValidator::new();
        let n = nodes(&["a"]);
        let e = edges(&[("a", "ghost")]);
        let report = gv.validate(&n, &e, Some("a"), Some("a"));
        assert!(!report.is_valid());
        assert!(report
            .errors()
            .iter()
            .any(|i| i.code == "DANGLING_EDGE_TARGET"));
    }

    #[test]
    fn test_gv_self_loop() {
        let gv = GraphValidator::new();
        let n = nodes(&["a"]);
        let e = edges(&[("a", "a")]);
        let report = gv.validate(&n, &e, Some("a"), Some("a"));
        assert!(report.warnings().iter().any(|i| i.code == "SELF_LOOP"));
    }

    #[test]
    fn test_gv_duplicate_edges() {
        let gv = GraphValidator::new();
        let n = nodes(&["a", "b"]);
        let e = edges(&[("a", "b"), ("a", "b")]);
        let report = gv.validate(&n, &e, Some("a"), Some("b"));
        assert!(report.warnings().iter().any(|i| i.code == "DUPLICATE_EDGE"));
    }

    #[test]
    fn test_gv_orphan_node() {
        let gv = GraphValidator::new();
        let n = nodes(&["a", "b", "orphan"]);
        let e = edges(&[("a", "b")]);
        let report = gv.validate(&n, &e, Some("a"), Some("b"));
        assert!(report.warnings().iter().any(|i| i.code == "ORPHAN_NODE"));
    }

    #[test]
    fn test_gv_unreachable_node() {
        let gv = GraphValidator::new();
        let n = nodes(&["a", "b", "island"]);
        let e = edges(&[("a", "b"), ("island", "island")]);
        let report = gv.validate(&n, &e, Some("a"), Some("b"));
        assert!(report.errors().iter().any(|i| i.code == "UNREACHABLE_NODE"));
    }

    #[test]
    fn test_gv_all_reachable() {
        let gv = GraphValidator::new();
        let n = nodes(&["a", "b", "c"]);
        let e = edges(&[("a", "b"), ("b", "c")]);
        let report = gv.validate(&n, &e, Some("a"), Some("c"));
        assert!(!report.errors().iter().any(|i| i.code == "UNREACHABLE_NODE"));
    }

    #[test]
    fn test_gv_empty_graph() {
        let gv = GraphValidator::new();
        let n: Vec<String> = vec![];
        let e: Vec<(String, String)> = vec![];
        let report = gv.validate(&n, &e, None, None);
        // Should report missing entry (error) and missing finish (warning)
        assert!(report.has_errors());
    }

    // =====================================================================
    // StateValidator & StateRule
    // =====================================================================

    #[test]
    fn test_state_rule_required_key_pass() {
        let mut sv = StateValidator::new();
        sv.add_rule(StateRule::required_key("name"));
        let from = json!({});
        let to = json!({"name": "alice"});
        let report = sv.validate_transition(&from, &to);
        assert!(report.is_valid());
    }

    #[test]
    fn test_state_rule_required_key_fail() {
        let mut sv = StateValidator::new();
        sv.add_rule(StateRule::required_key("name"));
        let from = json!({});
        let to = json!({"age": 30});
        let report = sv.validate_transition(&from, &to);
        assert!(!report.is_valid());
        assert!(report
            .errors()
            .iter()
            .any(|i| i.code == "MISSING_REQUIRED_KEY"));
    }

    #[test]
    fn test_state_rule_immutable_key_pass() {
        let mut sv = StateValidator::new();
        sv.add_rule(StateRule::immutable_key("id"));
        let from = json!({"id": 1});
        let to = json!({"id": 1, "name": "new"});
        let report = sv.validate_transition(&from, &to);
        assert!(report.is_valid());
    }

    #[test]
    fn test_state_rule_immutable_key_fail() {
        let mut sv = StateValidator::new();
        sv.add_rule(StateRule::immutable_key("id"));
        let from = json!({"id": 1});
        let to = json!({"id": 2});
        let report = sv.validate_transition(&from, &to);
        assert!(!report.is_valid());
        assert!(report
            .errors()
            .iter()
            .any(|i| i.code == "IMMUTABLE_KEY_CHANGED"));
    }

    #[test]
    fn test_state_rule_immutable_key_missing_in_from() {
        let mut sv = StateValidator::new();
        sv.add_rule(StateRule::immutable_key("id"));
        let from = json!({});
        let to = json!({"id": 1});
        // key was not in `from`, so no violation
        let report = sv.validate_transition(&from, &to);
        assert!(report.is_valid());
    }

    #[test]
    fn test_state_rule_type_constraint_pass() {
        let mut sv = StateValidator::new();
        sv.add_rule(StateRule::type_constraint("count", "number"));
        let from = json!({});
        let to = json!({"count": 42});
        let report = sv.validate_transition(&from, &to);
        assert!(report.is_valid());
    }

    #[test]
    fn test_state_rule_type_constraint_fail() {
        let mut sv = StateValidator::new();
        sv.add_rule(StateRule::type_constraint("count", "number"));
        let from = json!({});
        let to = json!({"count": "not a number"});
        let report = sv.validate_transition(&from, &to);
        assert!(!report.is_valid());
        assert!(report.errors().iter().any(|i| i.code == "TYPE_MISMATCH"));
    }

    #[test]
    fn test_state_rule_type_constraint_key_absent() {
        let mut sv = StateValidator::new();
        sv.add_rule(StateRule::type_constraint("count", "number"));
        let from = json!({});
        let to = json!({"other": 1});
        // key not present — no type violation
        let report = sv.validate_transition(&from, &to);
        assert!(report.is_valid());
    }

    #[test]
    fn test_state_rule_custom() {
        let mut sv = StateValidator::new();
        sv.add_rule(StateRule::custom("my_rule", "does nothing"));
        let from = json!({});
        let to = json!({});
        let report = sv.validate_transition(&from, &to);
        // custom rules produce info, not errors
        assert!(report.is_valid());
        assert_eq!(report.issue_count(), 1);
    }

    #[test]
    fn test_state_rule_required_key_non_object_state() {
        let mut sv = StateValidator::new();
        sv.add_rule(StateRule::required_key("key"));
        let from = json!(null);
        let to = json!("not an object");
        let report = sv.validate_transition(&from, &to);
        assert!(!report.is_valid());
        assert!(report
            .errors()
            .iter()
            .any(|i| i.code == "INVALID_STATE_TYPE"));
    }

    #[test]
    fn test_state_validator_rule_count() {
        let mut sv = StateValidator::new();
        assert_eq!(sv.rule_count(), 0);
        sv.add_rule(StateRule::required_key("a"));
        sv.add_rule(StateRule::immutable_key("b"));
        assert_eq!(sv.rule_count(), 2);
    }

    #[test]
    fn test_state_rule_name() {
        assert_eq!(StateRule::required_key("x").name(), "x");
        assert_eq!(StateRule::immutable_key("y").name(), "y");
        assert_eq!(StateRule::type_constraint("z", "string").name(), "z");
        assert_eq!(StateRule::custom("my_rule", "desc").name(), "my_rule");
    }

    #[test]
    fn test_state_rule_to_json() {
        let j = StateRule::required_key("x").to_json();
        assert_eq!(j["type"], "required_key");
        assert_eq!(j["key"], "x");

        let j = StateRule::immutable_key("y").to_json();
        assert_eq!(j["type"], "immutable_key");

        let j = StateRule::type_constraint("z", "number").to_json();
        assert_eq!(j["type"], "type_constraint");
        assert_eq!(j["expected"], "number");

        let j = StateRule::custom("r", "d").to_json();
        assert_eq!(j["type"], "custom");
        assert_eq!(j["name"], "r");
        assert_eq!(j["description"], "d");
    }

    // =====================================================================
    // SchemaValidator
    // =====================================================================

    #[test]
    fn test_schema_validator_compatible() {
        let mut sv = SchemaValidator::new();
        sv.register_node_schema("a", vec![], vec!["x".into(), "y".into()]);
        sv.register_node_schema("b", vec!["x".into()], vec!["z".into()]);
        let e = edges(&[("a", "b")]);
        let report = sv.validate_connections(&e);
        assert!(report.is_valid());
    }

    #[test]
    fn test_schema_validator_mismatch() {
        let mut sv = SchemaValidator::new();
        sv.register_node_schema("a", vec![], vec!["x".into()]);
        sv.register_node_schema("b", vec!["x".into(), "y".into()], vec![]);
        let e = edges(&[("a", "b")]);
        let report = sv.validate_connections(&e);
        assert!(!report.is_valid());
        assert!(report.errors().iter().any(|i| i.code == "SCHEMA_MISMATCH"));
    }

    #[test]
    fn test_schema_validator_missing_source_schema() {
        let mut sv = SchemaValidator::new();
        sv.register_node_schema("b", vec!["x".into()], vec![]);
        let e = edges(&[("a", "b")]);
        let report = sv.validate_connections(&e);
        assert!(report.has_warnings());
        assert!(report
            .warnings()
            .iter()
            .any(|i| i.code == "MISSING_SOURCE_SCHEMA"));
    }

    #[test]
    fn test_schema_validator_missing_target_schema() {
        let mut sv = SchemaValidator::new();
        sv.register_node_schema("a", vec![], vec!["x".into()]);
        let e = edges(&[("a", "b")]);
        let report = sv.validate_connections(&e);
        assert!(report.has_warnings());
        assert!(report
            .warnings()
            .iter()
            .any(|i| i.code == "MISSING_TARGET_SCHEMA"));
    }

    #[test]
    fn test_schema_validator_no_schemas() {
        let sv = SchemaValidator::new();
        let e = edges(&[("a", "b")]);
        let report = sv.validate_connections(&e);
        assert!(report.is_valid());
        assert_eq!(report.issue_count(), 0);
    }

    #[test]
    fn test_schema_validator_node_count() {
        let mut sv = SchemaValidator::new();
        assert_eq!(sv.node_count(), 0);
        sv.register_node_schema("a", vec![], vec![]);
        assert_eq!(sv.node_count(), 1);
        sv.register_node_schema("b", vec![], vec![]);
        assert_eq!(sv.node_count(), 2);
    }

    // =====================================================================
    // ValidationCache
    // =====================================================================

    #[test]
    fn test_cache_store_and_get() {
        let mut cache = ValidationCache::new();
        let report = ValidationReport::new();
        cache.store("hash1", report);
        assert!(cache.get("hash1").is_some());
        assert!(cache.get("hash2").is_none());
    }

    #[test]
    fn test_cache_invalidate() {
        let mut cache = ValidationCache::new();
        cache.store("h1", ValidationReport::new());
        assert_eq!(cache.len(), 1);
        cache.invalidate("h1");
        assert_eq!(cache.len(), 0);
        assert!(cache.get("h1").is_none());
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = ValidationCache::new();
        cache.store("h1", ValidationReport::new());
        cache.store("h2", ValidationReport::new());
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_len() {
        let mut cache = ValidationCache::new();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        cache.store("h1", ValidationReport::new());
        assert_eq!(cache.len(), 1);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cache_overwrite() {
        let mut cache = ValidationCache::new();
        let mut r1 = ValidationReport::new();
        r1.add_issue(ValidationIssue::new(ValidationLevel::Error, "E1", "old"));
        cache.store("h1", r1);
        assert!(!cache.get("h1").unwrap().is_valid());

        cache.store("h1", ValidationReport::new());
        assert!(cache.get("h1").unwrap().is_valid());
        assert_eq!(cache.len(), 1);
    }

    // =====================================================================
    // QuickCheck
    // =====================================================================

    #[test]
    fn test_qc_has_entry_true() {
        assert!(QuickCheck::has_entry(&nodes(&["a", "b"]), "a"));
    }

    #[test]
    fn test_qc_has_entry_false() {
        assert!(!QuickCheck::has_entry(&nodes(&["a", "b"]), "c"));
    }

    #[test]
    fn test_qc_has_finish_true() {
        assert!(QuickCheck::has_finish(&nodes(&["a", "end"]), "end"));
    }

    #[test]
    fn test_qc_has_finish_false() {
        assert!(!QuickCheck::has_finish(&nodes(&["a"]), "end"));
    }

    #[test]
    fn test_qc_is_connected_true() {
        let n = nodes(&["a", "b", "c"]);
        let e = edges(&[("a", "b"), ("b", "c")]);
        assert!(QuickCheck::is_connected(&n, &e));
    }

    #[test]
    fn test_qc_is_connected_false() {
        let n = nodes(&["a", "b", "island"]);
        let e = edges(&[("a", "b")]);
        assert!(!QuickCheck::is_connected(&n, &e));
    }

    #[test]
    fn test_qc_is_connected_empty() {
        let n: Vec<String> = vec![];
        let e: Vec<(String, String)> = vec![];
        assert!(QuickCheck::is_connected(&n, &e));
    }

    #[test]
    fn test_qc_is_connected_single_node() {
        let n = nodes(&["a"]);
        let e: Vec<(String, String)> = vec![];
        assert!(QuickCheck::is_connected(&n, &e));
    }

    #[test]
    fn test_qc_has_cycles_true() {
        let n = nodes(&["a", "b"]);
        let e = edges(&[("a", "b"), ("b", "a")]);
        assert!(QuickCheck::has_cycles(&n, &e));
    }

    #[test]
    fn test_qc_has_cycles_false() {
        let n = nodes(&["a", "b", "c"]);
        let e = edges(&[("a", "b"), ("b", "c")]);
        assert!(!QuickCheck::has_cycles(&n, &e));
    }

    #[test]
    fn test_qc_has_cycles_self_loop() {
        let n = nodes(&["a"]);
        let e = edges(&[("a", "a")]);
        assert!(QuickCheck::has_cycles(&n, &e));
    }

    #[test]
    fn test_qc_has_cycles_no_edges() {
        let n = nodes(&["a", "b"]);
        let e: Vec<(String, String)> = vec![];
        assert!(!QuickCheck::has_cycles(&n, &e));
    }

    // =====================================================================
    // Multiple-rule state validation
    // =====================================================================

    #[test]
    fn test_state_multiple_rules_all_pass() {
        let mut sv = StateValidator::new();
        sv.add_rule(StateRule::required_key("name"));
        sv.add_rule(StateRule::immutable_key("id"));
        sv.add_rule(StateRule::type_constraint("count", "number"));
        let from = json!({"id": 1, "name": "old"});
        let to = json!({"id": 1, "name": "new", "count": 5});
        let report = sv.validate_transition(&from, &to);
        assert!(report.is_valid());
    }

    #[test]
    fn test_state_multiple_rules_multiple_fail() {
        let mut sv = StateValidator::new();
        sv.add_rule(StateRule::required_key("name"));
        sv.add_rule(StateRule::immutable_key("id"));
        let from = json!({"id": 1});
        let to = json!({"id": 2}); // id changed + name missing
        let report = sv.validate_transition(&from, &to);
        assert!(!report.is_valid());
        assert_eq!(report.errors().len(), 2);
    }

    // =====================================================================
    // Integration: GraphValidator with many issues
    // =====================================================================

    #[test]
    fn test_gv_multiple_issues() {
        let gv = GraphValidator::new();
        let n = nodes(&["a", "b", "orphan"]);
        // ghost -> a is dangling source, a -> a is self-loop, orphan has no edges
        let e = edges(&[("ghost", "a"), ("a", "a"), ("a", "b"), ("a", "b")]);
        let report = gv.validate(&n, &e, Some("a"), Some("b"));
        // Should find: dangling source, self-loop, duplicate edge, orphan, unreachable orphan
        assert!(report.issue_count() >= 4);
    }

    #[test]
    fn test_gv_linear_chain() {
        let gv = GraphValidator::new();
        let n = nodes(&["start", "middle", "end"]);
        let e = edges(&[("start", "middle"), ("middle", "end")]);
        let report = gv.validate(&n, &e, Some("start"), Some("end"));
        assert!(report.is_valid());
        assert!(!report.has_warnings());
    }
}
