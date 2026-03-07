//! Advanced conditional edge routing for LangGraph.
//!
//! This module provides types for dynamically selecting graph edges based on
//! state evaluation. It supports field-based routing, predicate routing,
//! numeric threshold routing, and multi-condition composition.

use std::collections::HashMap;
use std::fmt;

use chrono::Utc;
use serde_json::Value;

use crate::errors::{LangGraphError, Result};

// ---------------------------------------------------------------------------
// Condition trait
// ---------------------------------------------------------------------------

/// A condition that evaluates state and returns the name of a target node.
pub trait Condition: Send + Sync + fmt::Debug {
    /// Evaluate the condition against the given state.
    ///
    /// Returns the name of the target node or edge on success.
    fn evaluate(&self, state: &Value) -> Result<String>;
}

// ---------------------------------------------------------------------------
// FieldCondition
// ---------------------------------------------------------------------------

/// Routes based on a state field's value by looking it up in a mapping table.
#[derive(Debug, Clone)]
pub struct FieldCondition {
    field: String,
    mapping: HashMap<String, String>,
    default: Option<String>,
}

impl FieldCondition {
    /// Create a new `FieldCondition`.
    ///
    /// `field` is the JSON key to inspect in the state object.
    /// `mapping` maps string representations of the field value to target node names.
    pub fn new(field: &str, mapping: HashMap<String, String>) -> Self {
        Self {
            field: field.to_string(),
            mapping,
            default: None,
        }
    }

    /// Set a default target node to use when the field value has no mapping entry.
    pub fn with_default(mut self, target: &str) -> Self {
        self.default = Some(target.to_string());
        self
    }
}

impl Condition for FieldCondition {
    fn evaluate(&self, state: &Value) -> Result<String> {
        let field_value = state.get(&self.field).ok_or_else(|| {
            LangGraphError::Other(format!(
                "Field '{}' not found in state",
                self.field
            ))
        })?;

        let key = match field_value {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Null => "null".to_string(),
            _ => field_value.to_string(),
        };

        if let Some(target) = self.mapping.get(&key) {
            return Ok(target.clone());
        }

        if let Some(ref default) = self.default {
            return Ok(default.clone());
        }

        Err(LangGraphError::Other(format!(
            "No mapping found for field '{}' value '{}' and no default set",
            self.field, key
        )))
    }
}

// ---------------------------------------------------------------------------
// PredicateCondition
// ---------------------------------------------------------------------------

/// Routes based on a predicate function applied to the state.
pub struct PredicateCondition {
    predicate: Box<dyn Fn(&Value) -> bool + Send + Sync>,
    if_true: String,
    if_false: String,
}

impl PredicateCondition {
    /// Create a new `PredicateCondition`.
    ///
    /// `predicate` is called with the state; if it returns `true`, the edge
    /// routes to `if_true`, otherwise to `if_false`.
    pub fn new<F>(predicate: F, if_true: &str, if_false: &str) -> Self
    where
        F: Fn(&Value) -> bool + Send + Sync + 'static,
    {
        Self {
            predicate: Box::new(predicate),
            if_true: if_true.to_string(),
            if_false: if_false.to_string(),
        }
    }
}

impl fmt::Debug for PredicateCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PredicateCondition")
            .field("if_true", &self.if_true)
            .field("if_false", &self.if_false)
            .field("predicate", &"<Fn>")
            .finish()
    }
}

impl Condition for PredicateCondition {
    fn evaluate(&self, state: &Value) -> Result<String> {
        if (self.predicate)(state) {
            Ok(self.if_true.clone())
        } else {
            Ok(self.if_false.clone())
        }
    }
}

// ---------------------------------------------------------------------------
// ThresholdCondition
// ---------------------------------------------------------------------------

/// Routes based on numeric comparison of a state field against a threshold.
#[derive(Debug, Clone)]
pub struct ThresholdCondition {
    field: String,
    threshold: f64,
    above: String,
    below: String,
    equal: Option<String>,
}

impl ThresholdCondition {
    /// Create a new `ThresholdCondition`.
    ///
    /// When the state field value is above `threshold`, routes to `above`;
    /// when below, routes to `below`. By default, equality routes to `above`.
    pub fn new(field: &str, threshold: f64, above: &str, below: &str) -> Self {
        Self {
            field: field.to_string(),
            threshold,
            above: above.to_string(),
            below: below.to_string(),
            equal: None,
        }
    }

    /// Set a specific target for when the value equals the threshold exactly.
    pub fn with_equal(mut self, target: &str) -> Self {
        self.equal = Some(target.to_string());
        self
    }
}

impl Condition for ThresholdCondition {
    fn evaluate(&self, state: &Value) -> Result<String> {
        let field_value = state.get(&self.field).ok_or_else(|| {
            LangGraphError::Other(format!(
                "Field '{}' not found in state",
                self.field
            ))
        })?;

        let num = field_value.as_f64().ok_or_else(|| {
            LangGraphError::Other(format!(
                "Field '{}' is not a number",
                self.field
            ))
        })?;

        if (num - self.threshold).abs() < f64::EPSILON {
            if let Some(ref equal_target) = self.equal {
                return Ok(equal_target.clone());
            }
            return Ok(self.above.clone());
        }

        if num > self.threshold {
            Ok(self.above.clone())
        } else {
            Ok(self.below.clone())
        }
    }
}

// ---------------------------------------------------------------------------
// MultiCondition
// ---------------------------------------------------------------------------

/// Evaluates multiple conditions in order and uses the first match.
pub struct MultiCondition {
    conditions: Vec<(Box<dyn Condition>, String)>,
    default: Option<String>,
}

impl MultiCondition {
    /// Create an empty `MultiCondition`.
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
            default: None,
        }
    }

    /// Add a condition; if it evaluates successfully and returns `target`, that
    /// value is used. The `target` parameter here acts as a label so the caller
    /// can specify the expected outcome, but the actual routing target comes
    /// from `condition.evaluate()`.
    pub fn add_condition(&mut self, condition: Box<dyn Condition>, _target: &str) {
        self.conditions.push((condition, _target.to_string()));
    }

    /// Set a default target if no condition matches.
    pub fn with_default(mut self, target: &str) -> Self {
        self.default = Some(target.to_string());
        self
    }
}

impl Default for MultiCondition {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for MultiCondition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiCondition")
            .field("num_conditions", &self.conditions.len())
            .field("default", &self.default)
            .finish()
    }
}

impl Condition for MultiCondition {
    fn evaluate(&self, state: &Value) -> Result<String> {
        for (condition, _) in &self.conditions {
            match condition.evaluate(state) {
                Ok(target) => return Ok(target),
                Err(_) => continue,
            }
        }

        if let Some(ref default) = self.default {
            return Ok(default.clone());
        }

        Err(LangGraphError::Other(
            "No condition matched and no default set".to_string(),
        ))
    }
}

// ---------------------------------------------------------------------------
// ConditionalEdge
// ---------------------------------------------------------------------------

/// An edge in the graph whose target is determined by evaluating a condition.
pub struct ConditionalEdge {
    source: String,
    condition: Box<dyn Condition>,
}

impl ConditionalEdge {
    /// Create a new conditional edge from `source` with the given condition.
    pub fn new(source: &str, condition: Box<dyn Condition>) -> Self {
        Self {
            source: source.to_string(),
            condition,
        }
    }

    /// Evaluate the condition against state and return the target node name.
    pub fn evaluate(&self, state: &Value) -> Result<String> {
        self.condition.evaluate(state)
    }

    /// Return the source node name.
    pub fn source(&self) -> &str {
        &self.source
    }
}

impl fmt::Debug for ConditionalEdge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ConditionalEdge")
            .field("source", &self.source)
            .field("condition", &self.condition)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ConditionalRouter
// ---------------------------------------------------------------------------

/// Manages conditional routing for a graph by collecting conditional edges and
/// dispatching routing requests.
#[derive(Debug, Default)]
pub struct ConditionalRouter {
    edges: Vec<ConditionalEdge>,
}

impl ConditionalRouter {
    /// Create an empty router.
    pub fn new() -> Self {
        Self { edges: Vec::new() }
    }

    /// Add a conditional edge.
    pub fn add_edge(&mut self, edge: ConditionalEdge) {
        self.edges.push(edge);
    }

    /// Find the first conditional edge for `source` and evaluate it.
    pub fn route(&self, source: &str, state: &Value) -> Result<String> {
        let edge = self
            .edges
            .iter()
            .find(|e| e.source() == source)
            .ok_or_else(|| {
                LangGraphError::Other(format!("No conditional edge found for source '{}'", source))
            })?;

        edge.evaluate(state)
    }

    /// Return all conditional edges originating from `source`.
    pub fn edges_from(&self, source: &str) -> Vec<&ConditionalEdge> {
        self.edges.iter().filter(|e| e.source() == source).collect()
    }

    /// Check whether a conditional edge exists for `source`.
    pub fn has_route(&self, source: &str) -> bool {
        self.edges.iter().any(|e| e.source() == source)
    }

    /// Return a list of all distinct source node names.
    pub fn sources(&self) -> Vec<&str> {
        let mut seen = Vec::new();
        for edge in &self.edges {
            let s = edge.source();
            if !seen.contains(&s) {
                seen.push(s);
            }
        }
        seen
    }

    /// Return the total number of conditional edges.
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Return `true` if the router contains no edges.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

// ---------------------------------------------------------------------------
// RoutingDecision
// ---------------------------------------------------------------------------

/// Records a single routing decision for debugging purposes.
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// The source node that triggered the routing.
    pub source: String,
    /// The target node selected by the condition.
    pub target: String,
    /// Timestamp of when the decision was recorded.
    pub timestamp: chrono::DateTime<Utc>,
    /// Snapshot of the state at the time of the decision.
    pub state_snapshot: Value,
}

impl RoutingDecision {
    /// Serialize this decision to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "source": self.source,
            "target": self.target,
            "timestamp": self.timestamp.to_rfc3339(),
            "state_snapshot": self.state_snapshot,
        })
    }
}

// ---------------------------------------------------------------------------
// RoutingTable
// ---------------------------------------------------------------------------

/// Records routing decisions for debugging and introspection.
#[derive(Debug, Clone, Default)]
pub struct RoutingTable {
    decisions: Vec<RoutingDecision>,
}

impl RoutingTable {
    /// Create an empty routing table.
    pub fn new() -> Self {
        Self {
            decisions: Vec::new(),
        }
    }

    /// Record a routing decision.
    pub fn record(&mut self, source: &str, target: &str, state: &Value) {
        self.decisions.push(RoutingDecision {
            source: source.to_string(),
            target: target.to_string(),
            timestamp: Utc::now(),
            state_snapshot: state.clone(),
        });
    }

    /// Return a reference to all recorded decisions.
    pub fn decisions(&self) -> &[RoutingDecision] {
        &self.decisions
    }

    /// Return decisions originating from `source`.
    pub fn decisions_from(&self, source: &str) -> Vec<&RoutingDecision> {
        self.decisions
            .iter()
            .filter(|d| d.source == source)
            .collect()
    }

    /// Clear all recorded decisions.
    pub fn clear(&mut self) {
        self.decisions.clear();
    }

    /// Return the number of recorded decisions.
    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    /// Return `true` if no decisions have been recorded.
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ---- FieldCondition tests ----

    #[test]
    fn field_condition_exact_match() {
        let mut mapping = HashMap::new();
        mapping.insert("error".to_string(), "handle_error".to_string());
        mapping.insert("success".to_string(), "handle_success".to_string());
        let cond = FieldCondition::new("status", mapping);

        let state = json!({"status": "error"});
        assert_eq!(cond.evaluate(&state).unwrap(), "handle_error");
    }

    #[test]
    fn field_condition_second_match() {
        let mut mapping = HashMap::new();
        mapping.insert("error".to_string(), "handle_error".to_string());
        mapping.insert("success".to_string(), "handle_success".to_string());
        let cond = FieldCondition::new("status", mapping);

        let state = json!({"status": "success"});
        assert_eq!(cond.evaluate(&state).unwrap(), "handle_success");
    }

    #[test]
    fn field_condition_no_match_no_default() {
        let mut mapping = HashMap::new();
        mapping.insert("a".to_string(), "node_a".to_string());
        let cond = FieldCondition::new("status", mapping);

        let state = json!({"status": "unknown"});
        assert!(cond.evaluate(&state).is_err());
    }

    #[test]
    fn field_condition_no_match_with_default() {
        let mut mapping = HashMap::new();
        mapping.insert("a".to_string(), "node_a".to_string());
        let cond = FieldCondition::new("status", mapping).with_default("fallback");

        let state = json!({"status": "unknown"});
        assert_eq!(cond.evaluate(&state).unwrap(), "fallback");
    }

    #[test]
    fn field_condition_missing_field() {
        let mapping = HashMap::new();
        let cond = FieldCondition::new("missing", mapping);

        let state = json!({"other": "value"});
        assert!(cond.evaluate(&state).is_err());
    }

    #[test]
    fn field_condition_numeric_value() {
        let mut mapping = HashMap::new();
        mapping.insert("42".to_string(), "answer_node".to_string());
        let cond = FieldCondition::new("code", mapping);

        let state = json!({"code": 42});
        assert_eq!(cond.evaluate(&state).unwrap(), "answer_node");
    }

    #[test]
    fn field_condition_bool_value() {
        let mut mapping = HashMap::new();
        mapping.insert("true".to_string(), "yes_node".to_string());
        mapping.insert("false".to_string(), "no_node".to_string());
        let cond = FieldCondition::new("flag", mapping);

        let state = json!({"flag": true});
        assert_eq!(cond.evaluate(&state).unwrap(), "yes_node");
    }

    // ---- PredicateCondition tests ----

    #[test]
    fn predicate_condition_true_path() {
        let cond = PredicateCondition::new(
            |state| state.get("ready").and_then(|v| v.as_bool()).unwrap_or(false),
            "proceed",
            "wait",
        );

        let state = json!({"ready": true});
        assert_eq!(cond.evaluate(&state).unwrap(), "proceed");
    }

    #[test]
    fn predicate_condition_false_path() {
        let cond = PredicateCondition::new(
            |state| state.get("ready").and_then(|v| v.as_bool()).unwrap_or(false),
            "proceed",
            "wait",
        );

        let state = json!({"ready": false});
        assert_eq!(cond.evaluate(&state).unwrap(), "wait");
    }

    #[test]
    fn predicate_condition_missing_field_defaults_false() {
        let cond = PredicateCondition::new(
            |state| state.get("ready").and_then(|v| v.as_bool()).unwrap_or(false),
            "proceed",
            "wait",
        );

        let state = json!({});
        assert_eq!(cond.evaluate(&state).unwrap(), "wait");
    }

    #[test]
    fn predicate_condition_complex_logic() {
        let cond = PredicateCondition::new(
            |state| {
                let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                count > 5
            },
            "high",
            "low",
        );

        assert_eq!(cond.evaluate(&json!({"count": 10})).unwrap(), "high");
        assert_eq!(cond.evaluate(&json!({"count": 3})).unwrap(), "low");
    }

    // ---- ThresholdCondition tests ----

    #[test]
    fn threshold_condition_above() {
        let cond = ThresholdCondition::new("score", 0.5, "high", "low");

        let state = json!({"score": 0.8});
        assert_eq!(cond.evaluate(&state).unwrap(), "high");
    }

    #[test]
    fn threshold_condition_below() {
        let cond = ThresholdCondition::new("score", 0.5, "high", "low");

        let state = json!({"score": 0.2});
        assert_eq!(cond.evaluate(&state).unwrap(), "low");
    }

    #[test]
    fn threshold_condition_equal_no_equal_target() {
        let cond = ThresholdCondition::new("score", 0.5, "high", "low");

        let state = json!({"score": 0.5});
        // Without explicit equal target, equality routes to `above`.
        assert_eq!(cond.evaluate(&state).unwrap(), "high");
    }

    #[test]
    fn threshold_condition_equal_with_equal_target() {
        let cond = ThresholdCondition::new("score", 0.5, "high", "low").with_equal("exact");

        let state = json!({"score": 0.5});
        assert_eq!(cond.evaluate(&state).unwrap(), "exact");
    }

    #[test]
    fn threshold_condition_missing_field() {
        let cond = ThresholdCondition::new("score", 0.5, "high", "low");

        let state = json!({"other": 1});
        assert!(cond.evaluate(&state).is_err());
    }

    #[test]
    fn threshold_condition_non_numeric_field() {
        let cond = ThresholdCondition::new("score", 0.5, "high", "low");

        let state = json!({"score": "not_a_number"});
        assert!(cond.evaluate(&state).is_err());
    }

    #[test]
    fn threshold_condition_integer_value() {
        let cond = ThresholdCondition::new("count", 10.0, "above", "below");

        let state = json!({"count": 15});
        assert_eq!(cond.evaluate(&state).unwrap(), "above");

        let state = json!({"count": 5});
        assert_eq!(cond.evaluate(&state).unwrap(), "below");
    }

    // ---- MultiCondition tests ----

    #[test]
    fn multi_condition_first_match() {
        let mut multi = MultiCondition::new();

        let mut m1 = HashMap::new();
        m1.insert("a".to_string(), "node_a".to_string());
        multi.add_condition(Box::new(FieldCondition::new("status", m1)), "node_a");

        let mut m2 = HashMap::new();
        m2.insert("b".to_string(), "node_b".to_string());
        multi.add_condition(Box::new(FieldCondition::new("status", m2)), "node_b");

        let state = json!({"status": "a"});
        assert_eq!(multi.evaluate(&state).unwrap(), "node_a");
    }

    #[test]
    fn multi_condition_second_match() {
        let mut multi = MultiCondition::new();

        let mut m1 = HashMap::new();
        m1.insert("a".to_string(), "node_a".to_string());
        multi.add_condition(Box::new(FieldCondition::new("status", m1)), "node_a");

        let mut m2 = HashMap::new();
        m2.insert("b".to_string(), "node_b".to_string());
        multi.add_condition(Box::new(FieldCondition::new("status", m2)), "node_b");

        let state = json!({"status": "b"});
        assert_eq!(multi.evaluate(&state).unwrap(), "node_b");
    }

    #[test]
    fn multi_condition_no_match_no_default() {
        let multi = MultiCondition::new();

        let state = json!({"status": "x"});
        assert!(multi.evaluate(&state).is_err());
    }

    #[test]
    fn multi_condition_no_match_with_default() {
        let multi = MultiCondition::new().with_default("fallback");

        let state = json!({"status": "x"});
        assert_eq!(multi.evaluate(&state).unwrap(), "fallback");
    }

    #[test]
    fn multi_condition_first_match_semantics() {
        // Both conditions would match, but we should get the first one.
        let mut multi = MultiCondition::new();

        multi.add_condition(
            Box::new(PredicateCondition::new(|_| true, "first", "other")),
            "first",
        );
        multi.add_condition(
            Box::new(PredicateCondition::new(|_| true, "second", "other")),
            "second",
        );

        let state = json!({});
        assert_eq!(multi.evaluate(&state).unwrap(), "first");
    }

    // ---- ConditionalEdge tests ----

    #[test]
    fn conditional_edge_evaluate() {
        let mut mapping = HashMap::new();
        mapping.insert("go".to_string(), "next".to_string());
        let cond = FieldCondition::new("action", mapping);

        let edge = ConditionalEdge::new("start", Box::new(cond));

        assert_eq!(edge.source(), "start");

        let state = json!({"action": "go"});
        assert_eq!(edge.evaluate(&state).unwrap(), "next");
    }

    #[test]
    fn conditional_edge_source_getter() {
        let cond = PredicateCondition::new(|_| true, "a", "b");
        let edge = ConditionalEdge::new("my_source", Box::new(cond));
        assert_eq!(edge.source(), "my_source");
    }

    #[test]
    fn conditional_edge_error_propagation() {
        let mapping = HashMap::new();
        let cond = FieldCondition::new("missing", mapping);
        let edge = ConditionalEdge::new("src", Box::new(cond));

        let state = json!({"other": 1});
        assert!(edge.evaluate(&state).is_err());
    }

    // ---- ConditionalRouter tests ----

    #[test]
    fn router_route_basic() {
        let mut router = ConditionalRouter::new();

        let mut mapping = HashMap::new();
        mapping.insert("yes".to_string(), "approve".to_string());
        mapping.insert("no".to_string(), "reject".to_string());
        let cond = FieldCondition::new("decision", mapping);

        router.add_edge(ConditionalEdge::new("review", Box::new(cond)));

        let state = json!({"decision": "yes"});
        assert_eq!(router.route("review", &state).unwrap(), "approve");
    }

    #[test]
    fn router_route_no_edge_for_source() {
        let router = ConditionalRouter::new();

        let state = json!({});
        assert!(router.route("nonexistent", &state).is_err());
    }

    #[test]
    fn router_edges_from() {
        let mut router = ConditionalRouter::new();

        let cond1 = PredicateCondition::new(|_| true, "a", "b");
        let cond2 = PredicateCondition::new(|_| false, "c", "d");
        let cond3 = PredicateCondition::new(|_| true, "e", "f");

        router.add_edge(ConditionalEdge::new("src1", Box::new(cond1)));
        router.add_edge(ConditionalEdge::new("src1", Box::new(cond2)));
        router.add_edge(ConditionalEdge::new("src2", Box::new(cond3)));

        assert_eq!(router.edges_from("src1").len(), 2);
        assert_eq!(router.edges_from("src2").len(), 1);
        assert_eq!(router.edges_from("src3").len(), 0);
    }

    #[test]
    fn router_has_route() {
        let mut router = ConditionalRouter::new();

        let cond = PredicateCondition::new(|_| true, "a", "b");
        router.add_edge(ConditionalEdge::new("node_a", Box::new(cond)));

        assert!(router.has_route("node_a"));
        assert!(!router.has_route("node_b"));
    }

    #[test]
    fn router_sources() {
        let mut router = ConditionalRouter::new();

        let cond1 = PredicateCondition::new(|_| true, "a", "b");
        let cond2 = PredicateCondition::new(|_| true, "c", "d");
        let cond3 = PredicateCondition::new(|_| true, "e", "f");

        router.add_edge(ConditionalEdge::new("alpha", Box::new(cond1)));
        router.add_edge(ConditionalEdge::new("beta", Box::new(cond2)));
        router.add_edge(ConditionalEdge::new("alpha", Box::new(cond3)));

        let sources = router.sources();
        assert_eq!(sources.len(), 2);
        assert!(sources.contains(&"alpha"));
        assert!(sources.contains(&"beta"));
    }

    #[test]
    fn router_len_and_is_empty() {
        let mut router = ConditionalRouter::new();
        assert!(router.is_empty());
        assert_eq!(router.len(), 0);

        let cond = PredicateCondition::new(|_| true, "a", "b");
        router.add_edge(ConditionalEdge::new("x", Box::new(cond)));
        assert!(!router.is_empty());
        assert_eq!(router.len(), 1);
    }

    #[test]
    fn router_multiple_edges_same_source_uses_first() {
        let mut router = ConditionalRouter::new();

        let cond1 = PredicateCondition::new(|_| true, "first_target", "other");
        let cond2 = PredicateCondition::new(|_| true, "second_target", "other");

        router.add_edge(ConditionalEdge::new("src", Box::new(cond1)));
        router.add_edge(ConditionalEdge::new("src", Box::new(cond2)));

        // `route` finds the first edge for the source.
        let state = json!({});
        assert_eq!(router.route("src", &state).unwrap(), "first_target");
    }

    // ---- RoutingTable tests ----

    #[test]
    fn routing_table_record_and_query() {
        let mut table = RoutingTable::new();
        assert!(table.is_empty());

        let state = json!({"key": "value"});
        table.record("node_a", "node_b", &state);

        assert_eq!(table.len(), 1);
        assert!(!table.is_empty());

        let decisions = table.decisions();
        assert_eq!(decisions.len(), 1);
        assert_eq!(decisions[0].source, "node_a");
        assert_eq!(decisions[0].target, "node_b");
        assert_eq!(decisions[0].state_snapshot, state);
    }

    #[test]
    fn routing_table_decisions_from() {
        let mut table = RoutingTable::new();

        table.record("a", "b", &json!({}));
        table.record("a", "c", &json!({}));
        table.record("b", "d", &json!({}));

        let from_a = table.decisions_from("a");
        assert_eq!(from_a.len(), 2);

        let from_b = table.decisions_from("b");
        assert_eq!(from_b.len(), 1);

        let from_c = table.decisions_from("c");
        assert_eq!(from_c.len(), 0);
    }

    #[test]
    fn routing_table_clear() {
        let mut table = RoutingTable::new();
        table.record("a", "b", &json!({}));
        table.record("c", "d", &json!({}));
        assert_eq!(table.len(), 2);

        table.clear();
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn routing_decision_to_json() {
        let mut table = RoutingTable::new();
        let state = json!({"x": 42});
        table.record("src", "dst", &state);

        let decision = &table.decisions()[0];
        let json_val = decision.to_json();

        assert_eq!(json_val["source"], "src");
        assert_eq!(json_val["target"], "dst");
        assert_eq!(json_val["state_snapshot"]["x"], 42);
        assert!(json_val["timestamp"].is_string());
    }

    // ---- Empty router edge case ----

    #[test]
    fn empty_router_route_returns_error() {
        let router = ConditionalRouter::new();
        let state = json!({"anything": true});
        assert!(router.route("any_source", &state).is_err());
    }

    #[test]
    fn empty_router_has_no_sources() {
        let router = ConditionalRouter::new();
        assert!(router.sources().is_empty());
    }

    // ---- Integration-style tests ----

    #[test]
    fn full_routing_workflow() {
        // Build a router with multiple edge types.
        let mut router = ConditionalRouter::new();

        // Field-based routing from "classifier"
        let mut mapping = HashMap::new();
        mapping.insert("positive".to_string(), "celebrate".to_string());
        mapping.insert("negative".to_string(), "console".to_string());
        let cond = FieldCondition::new("sentiment", mapping).with_default("neutral_handler");
        router.add_edge(ConditionalEdge::new("classifier", Box::new(cond)));

        // Threshold routing from "scorer"
        let cond = ThresholdCondition::new("confidence", 0.8, "high_conf", "low_conf");
        router.add_edge(ConditionalEdge::new("scorer", Box::new(cond)));

        // Route from classifier
        let state = json!({"sentiment": "positive", "confidence": 0.9});
        let target = router.route("classifier", &state).unwrap();
        assert_eq!(target, "celebrate");

        // Route from scorer
        let target = router.route("scorer", &state).unwrap();
        assert_eq!(target, "high_conf");

        // Record decisions in a routing table
        let mut table = RoutingTable::new();
        table.record("classifier", "celebrate", &state);
        table.record("scorer", "high_conf", &state);
        assert_eq!(table.len(), 2);
    }

    #[test]
    fn field_condition_null_value() {
        let mut mapping = HashMap::new();
        mapping.insert("null".to_string(), "null_handler".to_string());
        let cond = FieldCondition::new("value", mapping);

        let state = json!({"value": null});
        assert_eq!(cond.evaluate(&state).unwrap(), "null_handler");
    }

    #[test]
    fn threshold_condition_negative_values() {
        let cond = ThresholdCondition::new("temp", 0.0, "warm", "cold");

        assert_eq!(cond.evaluate(&json!({"temp": -5.0})).unwrap(), "cold");
        assert_eq!(cond.evaluate(&json!({"temp": 5.0})).unwrap(), "warm");
    }
}
