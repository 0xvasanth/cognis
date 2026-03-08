//! Evaluation and benchmarking framework for DeepAgents.
//!
//! Provides traits and types for defining evaluation metrics, building test
//! suites, running evaluations, and generating reports.
//!
//! # Example
//!
//! ```rust,ignore
//! use deepagents::evaluation::{
//!     EvalCase, EvalSuite, ExactMatchMetric, ContainsMetric, EvalMetric,
//! };
//! use serde_json::json;
//!
//! let mut suite = EvalSuite::new("my_suite");
//! suite.add_case(
//!     EvalCase::builder("test_1")
//!         .input(json!("What is 2+2?"))
//!         .expected_output(json!("4"))
//!         .tag("math")
//!         .build(),
//! );
//!
//! let metrics: Vec<Box<dyn EvalMetric>> = vec![
//!     Box::new(ExactMatchMetric),
//!     Box::new(ContainsMetric::new(false)),
//! ];
//!
//! let report = suite.run(&|input| input.clone(), &metrics);
//! println!("passing rate: {:.1}%", report.passing_rate(0.5) * 100.0);
//! ```

use serde_json::Value;
use std::collections::HashMap;
use std::time::Instant;

// ---------------------------------------------------------------------------
// EvalMetric trait
// ---------------------------------------------------------------------------

/// A metric that can score how well an actual output matches an expected output.
pub trait EvalMetric {
    /// The name of this metric.
    fn name(&self) -> &str;

    /// A human-readable description of what this metric measures.
    fn description(&self) -> &str;

    /// Evaluate the actual output against the expected output and return a score.
    fn evaluate(&self, expected: &Value, actual: &Value) -> EvalScore;
}

// ---------------------------------------------------------------------------
// EvalScore
// ---------------------------------------------------------------------------

/// The result of evaluating a single metric on a single case.
#[derive(Debug, Clone)]
pub struct EvalScore {
    /// The name of the metric that produced this score.
    pub metric_name: String,
    /// A score between 0.0 (worst) and 1.0 (best).
    pub score: f64,
    /// Additional details about the evaluation.
    pub details: HashMap<String, Value>,
}

impl EvalScore {
    /// Create a new score.
    pub fn new(metric_name: impl Into<String>, score: f64) -> Self {
        Self {
            metric_name: metric_name.into(),
            score: score.clamp(0.0, 1.0),
            details: HashMap::new(),
        }
    }

    /// Add a detail entry (builder pattern).
    pub fn with_detail(mut self, key: impl Into<String>, value: Value) -> Self {
        self.details.insert(key.into(), value);
        self
    }

    /// Returns `true` if the score meets or exceeds the given threshold.
    pub fn is_passing(&self, threshold: f64) -> bool {
        self.score >= threshold
    }

    /// Serialize this score to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "metric_name": self.metric_name,
            "score": self.score,
            "details": self.details,
        })
    }
}

// ---------------------------------------------------------------------------
// ExactMatchMetric
// ---------------------------------------------------------------------------

/// Checks whether the actual output is exactly equal to the expected output.
#[derive(Debug, Clone, Default)]
pub struct ExactMatchMetric;

impl EvalMetric for ExactMatchMetric {
    fn name(&self) -> &str {
        "exact_match"
    }

    fn description(&self) -> &str {
        "Checks exact equality between expected and actual values"
    }

    fn evaluate(&self, expected: &Value, actual: &Value) -> EvalScore {
        let matched = expected == actual;
        EvalScore::new("exact_match", if matched { 1.0 } else { 0.0 })
            .with_detail("matched", serde_json::json!(matched))
    }
}

// ---------------------------------------------------------------------------
// ContainsMetric
// ---------------------------------------------------------------------------

/// Checks whether the actual output contains the expected value as a substring.
///
/// Both values are converted to strings via their JSON representation for
/// non-string types, or via the inner string for `Value::String`.
#[derive(Debug, Clone)]
pub struct ContainsMetric {
    /// Whether the comparison should be case-insensitive.
    pub case_insensitive: bool,
}

impl ContainsMetric {
    /// Create a new `ContainsMetric`.
    pub fn new(case_insensitive: bool) -> Self {
        Self { case_insensitive }
    }

    fn value_to_string(v: &Value) -> String {
        match v {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }
}

impl Default for ContainsMetric {
    fn default() -> Self {
        Self::new(false)
    }
}

impl EvalMetric for ContainsMetric {
    fn name(&self) -> &str {
        "contains"
    }

    fn description(&self) -> &str {
        "Checks if the actual output contains the expected value as a substring"
    }

    fn evaluate(&self, expected: &Value, actual: &Value) -> EvalScore {
        let expected_str = Self::value_to_string(expected);
        let actual_str = Self::value_to_string(actual);

        let contained = if self.case_insensitive {
            actual_str.to_lowercase().contains(&expected_str.to_lowercase())
        } else {
            actual_str.contains(&expected_str)
        };

        EvalScore::new("contains", if contained { 1.0 } else { 0.0 })
            .with_detail("contained", serde_json::json!(contained))
            .with_detail("case_insensitive", serde_json::json!(self.case_insensitive))
    }
}

// ---------------------------------------------------------------------------
// NumericDistanceMetric
// ---------------------------------------------------------------------------

/// Computes a normalized distance between two numeric values.
///
/// The score is `1.0 - (|expected - actual| / tolerance)`, clamped to `[0.0, 1.0]`.
/// Non-numeric values receive a score of 0.0.
#[derive(Debug, Clone)]
pub struct NumericDistanceMetric {
    /// The maximum acceptable distance. Values further apart than this
    /// receive a score of 0.0.
    pub tolerance: f64,
}

impl NumericDistanceMetric {
    /// Create a new metric with the given tolerance.
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance: tolerance.abs().max(f64::EPSILON),
        }
    }
}

impl EvalMetric for NumericDistanceMetric {
    fn name(&self) -> &str {
        "numeric_distance"
    }

    fn description(&self) -> &str {
        "Computes normalized distance between numeric values within a tolerance"
    }

    fn evaluate(&self, expected: &Value, actual: &Value) -> EvalScore {
        let expected_num = value_as_f64(expected);
        let actual_num = value_as_f64(actual);

        match (expected_num, actual_num) {
            (Some(e), Some(a)) => {
                let distance = (e - a).abs();
                let score = (1.0 - distance / self.tolerance).clamp(0.0, 1.0);
                EvalScore::new("numeric_distance", score)
                    .with_detail("expected", serde_json::json!(e))
                    .with_detail("actual", serde_json::json!(a))
                    .with_detail("distance", serde_json::json!(distance))
                    .with_detail("tolerance", serde_json::json!(self.tolerance))
            }
            _ => EvalScore::new("numeric_distance", 0.0)
                .with_detail("error", serde_json::json!("non-numeric value")),
        }
    }
}

/// Try to extract a f64 from a JSON value.
fn value_as_f64(v: &Value) -> Option<f64> {
    match v {
        Value::Number(n) => n.as_f64(),
        Value::String(s) => s.parse::<f64>().ok(),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// JsonSimilarityMetric
// ---------------------------------------------------------------------------

/// Compares two JSON structures and scores based on matching keys and values.
///
/// For objects, the score is the fraction of expected keys present in the actual
/// value with matching values. For arrays, it compares element-by-element.
/// For scalars, it falls back to exact equality.
#[derive(Debug, Clone, Default)]
pub struct JsonSimilarityMetric;

impl JsonSimilarityMetric {
    /// Recursively compute the similarity between two JSON values.
    ///
    /// Returns `(matched, total)` counts.
    fn similarity(expected: &Value, actual: &Value) -> (f64, f64) {
        match (expected, actual) {
            (Value::Object(exp_map), Value::Object(act_map)) => {
                if exp_map.is_empty() {
                    return if act_map.is_empty() {
                        (1.0, 1.0)
                    } else {
                        (0.0, 1.0)
                    };
                }
                let mut matched = 0.0;
                let total = exp_map.len() as f64;
                for (key, exp_val) in exp_map {
                    if let Some(act_val) = act_map.get(key) {
                        let (m, t) = Self::similarity(exp_val, act_val);
                        matched += m / t;
                    }
                }
                (matched, total)
            }
            (Value::Array(exp_arr), Value::Array(act_arr)) => {
                if exp_arr.is_empty() {
                    return if act_arr.is_empty() {
                        (1.0, 1.0)
                    } else {
                        (0.0, 1.0)
                    };
                }
                let total = exp_arr.len() as f64;
                let mut matched = 0.0;
                for (i, exp_val) in exp_arr.iter().enumerate() {
                    if let Some(act_val) = act_arr.get(i) {
                        let (m, t) = Self::similarity(exp_val, act_val);
                        matched += m / t;
                    }
                }
                (matched, total)
            }
            _ => {
                if expected == actual {
                    (1.0, 1.0)
                } else {
                    (0.0, 1.0)
                }
            }
        }
    }
}

impl EvalMetric for JsonSimilarityMetric {
    fn name(&self) -> &str {
        "json_similarity"
    }

    fn description(&self) -> &str {
        "Compares JSON structures and scores based on matching keys and values"
    }

    fn evaluate(&self, expected: &Value, actual: &Value) -> EvalScore {
        let (matched, total) = Self::similarity(expected, actual);
        let score = if total == 0.0 { 1.0 } else { matched / total };
        EvalScore::new("json_similarity", score)
            .with_detail("matched", serde_json::json!(matched))
            .with_detail("total", serde_json::json!(total))
    }
}

// ---------------------------------------------------------------------------
// EvalCase
// ---------------------------------------------------------------------------

/// A single evaluation test case.
#[derive(Debug, Clone)]
pub struct EvalCase {
    /// A descriptive name for this case.
    pub name: String,
    /// The input to pass to the evaluator function.
    pub input: Value,
    /// The expected output to compare against.
    pub expected_output: Value,
    /// Tags for grouping and filtering cases.
    pub tags: Vec<String>,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, Value>,
}

impl EvalCase {
    /// Create a new builder for an eval case with the given name.
    pub fn builder(name: impl Into<String>) -> EvalCaseBuilder {
        EvalCaseBuilder {
            name: name.into(),
            input: Value::Null,
            expected_output: Value::Null,
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Serialize this case to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "input": self.input,
            "expected_output": self.expected_output,
            "tags": self.tags,
            "metadata": self.metadata,
        })
    }
}

/// Builder for [`EvalCase`].
#[derive(Debug, Clone)]
pub struct EvalCaseBuilder {
    name: String,
    input: Value,
    expected_output: Value,
    tags: Vec<String>,
    metadata: HashMap<String, Value>,
}

impl EvalCaseBuilder {
    /// Set the input value.
    pub fn input(mut self, input: Value) -> Self {
        self.input = input;
        self
    }

    /// Set the expected output value.
    pub fn expected_output(mut self, expected: Value) -> Self {
        self.expected_output = expected;
        self
    }

    /// Add a tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Add metadata.
    pub fn metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Build the [`EvalCase`].
    pub fn build(self) -> EvalCase {
        EvalCase {
            name: self.name,
            input: self.input,
            expected_output: self.expected_output,
            tags: self.tags,
            metadata: self.metadata,
        }
    }
}

// ---------------------------------------------------------------------------
// EvalCaseResult
// ---------------------------------------------------------------------------

/// The result of evaluating a single case with one or more metrics.
#[derive(Debug, Clone)]
pub struct EvalCaseResult {
    /// The name of the case.
    pub case_name: String,
    /// Scores from each metric.
    pub scores: Vec<EvalScore>,
    /// The input that was provided.
    pub input: Value,
    /// The expected output.
    pub expected: Value,
    /// The actual output produced by the evaluator.
    pub actual: Value,
    /// Whether this case is considered passing (all metric averages >= threshold).
    pub passed: bool,
}

impl EvalCaseResult {
    /// Compute the average score across all metrics for this case.
    pub fn avg_score(&self) -> f64 {
        if self.scores.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.scores.iter().map(|s| s.score).sum();
        sum / self.scores.len() as f64
    }

    /// Serialize this result to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "case_name": self.case_name,
            "scores": self.scores.iter().map(|s| s.to_json()).collect::<Vec<_>>(),
            "input": self.input,
            "expected": self.expected,
            "actual": self.actual,
            "passed": self.passed,
            "avg_score": self.avg_score(),
        })
    }
}

// ---------------------------------------------------------------------------
// EvalSuite
// ---------------------------------------------------------------------------

/// A collection of evaluation cases that can be run against an evaluator.
#[derive(Debug, Clone)]
pub struct EvalSuite {
    /// The name of this suite.
    pub name: String,
    /// The cases in this suite.
    cases: Vec<EvalCase>,
}

impl EvalSuite {
    /// Create a new empty suite.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            cases: Vec::new(),
        }
    }

    /// Add a case to the suite.
    pub fn add_case(&mut self, case: EvalCase) {
        self.cases.push(case);
    }

    /// Return the number of cases.
    pub fn case_count(&self) -> usize {
        self.cases.len()
    }

    /// Return all cases that have the given tag.
    pub fn by_tag(&self, tag: &str) -> Vec<&EvalCase> {
        self.cases
            .iter()
            .filter(|c| c.tags.iter().any(|t| t == tag))
            .collect()
    }

    /// Run the suite using the provided evaluator function and metrics.
    ///
    /// The `evaluator` receives each case's input and must return an output
    /// value. Each metric is then applied to compare the expected and actual
    /// outputs.
    ///
    /// A default passing threshold of 0.5 is used to determine the `passed`
    /// field on each case result.
    pub fn run(
        &self,
        evaluator: &dyn Fn(&Value) -> Value,
        metrics: &[Box<dyn EvalMetric>],
    ) -> EvalReport {
        let start = Instant::now();
        let threshold = 0.5;

        let mut results = Vec::with_capacity(self.cases.len());
        for case in &self.cases {
            let actual = evaluator(&case.input);
            let scores: Vec<EvalScore> = metrics
                .iter()
                .map(|m| m.evaluate(&case.expected_output, &actual))
                .collect();

            let avg = if scores.is_empty() {
                0.0
            } else {
                scores.iter().map(|s| s.score).sum::<f64>() / scores.len() as f64
            };

            results.push(EvalCaseResult {
                case_name: case.name.clone(),
                scores,
                input: case.input.clone(),
                expected: case.expected_output.clone(),
                actual,
                passed: avg >= threshold,
            });
        }

        let overall_score = if results.is_empty() {
            0.0
        } else {
            results.iter().map(|r| r.avg_score()).sum::<f64>() / results.len() as f64
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        EvalReport {
            suite_name: self.name.clone(),
            results,
            overall_score,
            duration_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// EvalReport
// ---------------------------------------------------------------------------

/// The results of running an evaluation suite.
#[derive(Debug, Clone)]
pub struct EvalReport {
    /// The name of the suite that was run.
    pub suite_name: String,
    /// Results for each case.
    pub results: Vec<EvalCaseResult>,
    /// The overall average score across all cases and metrics.
    pub overall_score: f64,
    /// How long the evaluation took, in milliseconds.
    pub duration_ms: u64,
}

impl EvalReport {
    /// The fraction of cases whose average score meets or exceeds the threshold.
    pub fn passing_rate(&self, threshold: f64) -> f64 {
        if self.results.is_empty() {
            return 0.0;
        }
        let passing = self
            .results
            .iter()
            .filter(|r| r.avg_score() >= threshold)
            .count();
        passing as f64 / self.results.len() as f64
    }

    /// Return references to all case results that did not meet the threshold.
    pub fn failures(&self, threshold: f64) -> Vec<&EvalCaseResult> {
        self.results
            .iter()
            .filter(|r| r.avg_score() < threshold)
            .collect()
    }

    /// Serialize this report to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "suite_name": self.suite_name,
            "overall_score": self.overall_score,
            "duration_ms": self.duration_ms,
            "case_count": self.results.len(),
            "results": self.results.iter().map(|r| r.to_json()).collect::<Vec<_>>(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- EvalScore --

    #[test]
    fn test_eval_score_creation() {
        let score = EvalScore::new("test_metric", 0.75);
        assert_eq!(score.metric_name, "test_metric");
        assert!((score.score - 0.75).abs() < f64::EPSILON);
        assert!(score.details.is_empty());
    }

    #[test]
    fn test_eval_score_clamped_above() {
        let score = EvalScore::new("m", 1.5);
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_score_clamped_below() {
        let score = EvalScore::new("m", -0.5);
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_score_with_detail() {
        let score = EvalScore::new("m", 0.5)
            .with_detail("key", json!("value"));
        assert_eq!(score.details["key"], json!("value"));
    }

    #[test]
    fn test_eval_score_is_passing_true() {
        let score = EvalScore::new("m", 0.8);
        assert!(score.is_passing(0.7));
        assert!(score.is_passing(0.8));
    }

    #[test]
    fn test_eval_score_is_passing_false() {
        let score = EvalScore::new("m", 0.3);
        assert!(!score.is_passing(0.5));
    }

    #[test]
    fn test_eval_score_to_json() {
        let score = EvalScore::new("m", 0.9)
            .with_detail("info", json!("ok"));
        let j = score.to_json();
        assert_eq!(j["metric_name"], "m");
        assert_eq!(j["score"], 0.9);
        assert_eq!(j["details"]["info"], "ok");
    }

    // -- ExactMatchMetric --

    #[test]
    fn test_exact_match_equal_strings() {
        let m = ExactMatchMetric;
        let score = m.evaluate(&json!("hello"), &json!("hello"));
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_exact_match_not_equal_strings() {
        let m = ExactMatchMetric;
        let score = m.evaluate(&json!("hello"), &json!("world"));
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_exact_match_equal_numbers() {
        let m = ExactMatchMetric;
        let score = m.evaluate(&json!(42), &json!(42));
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_exact_match_different_types() {
        let m = ExactMatchMetric;
        let score = m.evaluate(&json!("42"), &json!(42));
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_exact_match_equal_objects() {
        let m = ExactMatchMetric;
        let score = m.evaluate(&json!({"a": 1}), &json!({"a": 1}));
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_exact_match_name() {
        let m = ExactMatchMetric;
        assert_eq!(m.name(), "exact_match");
        assert!(!m.description().is_empty());
    }

    // -- ContainsMetric --

    #[test]
    fn test_contains_substring_found() {
        let m = ContainsMetric::new(false);
        let score = m.evaluate(&json!("world"), &json!("hello world"));
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_contains_substring_not_found() {
        let m = ContainsMetric::new(false);
        let score = m.evaluate(&json!("xyz"), &json!("hello world"));
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_contains_case_sensitive_fail() {
        let m = ContainsMetric::new(false);
        let score = m.evaluate(&json!("HELLO"), &json!("hello world"));
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_contains_case_insensitive_pass() {
        let m = ContainsMetric::new(true);
        let score = m.evaluate(&json!("HELLO"), &json!("hello world"));
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_contains_non_string_values() {
        let m = ContainsMetric::new(false);
        // json!(42) -> "42", json!([1,42,3]) -> "[1,42,3]"
        let score = m.evaluate(&json!("42"), &json!("[1,42,3]"));
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_contains_name() {
        let m = ContainsMetric::new(false);
        assert_eq!(m.name(), "contains");
    }

    // -- NumericDistanceMetric --

    #[test]
    fn test_numeric_distance_exact() {
        let m = NumericDistanceMetric::new(10.0);
        let score = m.evaluate(&json!(5.0), &json!(5.0));
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_numeric_distance_within_tolerance() {
        let m = NumericDistanceMetric::new(10.0);
        let score = m.evaluate(&json!(10.0), &json!(15.0));
        // distance = 5, score = 1 - 5/10 = 0.5
        assert!((score.score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_numeric_distance_at_tolerance() {
        let m = NumericDistanceMetric::new(10.0);
        let score = m.evaluate(&json!(0.0), &json!(10.0));
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_numeric_distance_beyond_tolerance() {
        let m = NumericDistanceMetric::new(10.0);
        let score = m.evaluate(&json!(0.0), &json!(20.0));
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_numeric_distance_non_numeric() {
        let m = NumericDistanceMetric::new(10.0);
        let score = m.evaluate(&json!("abc"), &json!(5.0));
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_numeric_distance_string_numbers() {
        let m = NumericDistanceMetric::new(10.0);
        let score = m.evaluate(&json!("5.0"), &json!("5.0"));
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_numeric_distance_name() {
        let m = NumericDistanceMetric::new(1.0);
        assert_eq!(m.name(), "numeric_distance");
    }

    #[test]
    fn test_numeric_distance_zero_tolerance() {
        // Should use EPSILON to avoid division by zero
        let m = NumericDistanceMetric::new(0.0);
        assert!(m.tolerance > 0.0);
    }

    // -- JsonSimilarityMetric --

    #[test]
    fn test_json_similarity_identical_objects() {
        let m = JsonSimilarityMetric;
        let v = json!({"a": 1, "b": "hello"});
        let score = m.evaluate(&v, &v);
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_similarity_partial_match() {
        let m = JsonSimilarityMetric;
        let expected = json!({"a": 1, "b": 2});
        let actual = json!({"a": 1, "b": 99});
        let score = m.evaluate(&expected, &actual);
        // a matches (1/2), b doesn't (0/2) => 0.5
        assert!((score.score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_similarity_missing_keys() {
        let m = JsonSimilarityMetric;
        let expected = json!({"a": 1, "b": 2});
        let actual = json!({"a": 1});
        let score = m.evaluate(&expected, &actual);
        // only a matches => 0.5
        assert!((score.score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_similarity_arrays() {
        let m = JsonSimilarityMetric;
        let expected = json!([1, 2, 3]);
        let actual = json!([1, 2, 99]);
        let score = m.evaluate(&expected, &actual);
        // 2 out of 3 match
        assert!((score.score - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_json_similarity_scalar_match() {
        let m = JsonSimilarityMetric;
        let score = m.evaluate(&json!(42), &json!(42));
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_similarity_scalar_mismatch() {
        let m = JsonSimilarityMetric;
        let score = m.evaluate(&json!(42), &json!(99));
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_similarity_nested_objects() {
        let m = JsonSimilarityMetric;
        let expected = json!({"a": {"x": 1, "y": 2}});
        let actual = json!({"a": {"x": 1, "y": 99}});
        let score = m.evaluate(&expected, &actual);
        // inner: x matches, y doesn't => 0.5 for "a" => overall 0.5
        assert!((score.score - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_similarity_empty_objects() {
        let m = JsonSimilarityMetric;
        let score = m.evaluate(&json!({}), &json!({}));
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_similarity_name() {
        let m = JsonSimilarityMetric;
        assert_eq!(m.name(), "json_similarity");
    }

    // -- EvalCase building --

    #[test]
    fn test_eval_case_builder_basic() {
        let case = EvalCase::builder("test_case")
            .input(json!("input"))
            .expected_output(json!("output"))
            .build();
        assert_eq!(case.name, "test_case");
        assert_eq!(case.input, json!("input"));
        assert_eq!(case.expected_output, json!("output"));
        assert!(case.tags.is_empty());
        assert!(case.metadata.is_empty());
    }

    #[test]
    fn test_eval_case_builder_with_tags() {
        let case = EvalCase::builder("c")
            .input(json!(null))
            .expected_output(json!(null))
            .tag("math")
            .tag("easy")
            .build();
        assert_eq!(case.tags, vec!["math", "easy"]);
    }

    #[test]
    fn test_eval_case_builder_with_metadata() {
        let case = EvalCase::builder("c")
            .input(json!(null))
            .expected_output(json!(null))
            .metadata("difficulty", json!("hard"))
            .build();
        assert_eq!(case.metadata["difficulty"], json!("hard"));
    }

    #[test]
    fn test_eval_case_to_json() {
        let case = EvalCase::builder("c")
            .input(json!("in"))
            .expected_output(json!("out"))
            .tag("t")
            .build();
        let j = case.to_json();
        assert_eq!(j["name"], "c");
        assert_eq!(j["input"], "in");
        assert_eq!(j["expected_output"], "out");
        assert_eq!(j["tags"][0], "t");
    }

    // -- EvalSuite --

    #[test]
    fn test_eval_suite_new() {
        let suite = EvalSuite::new("test_suite");
        assert_eq!(suite.name, "test_suite");
        assert_eq!(suite.case_count(), 0);
    }

    #[test]
    fn test_eval_suite_add_case() {
        let mut suite = EvalSuite::new("s");
        suite.add_case(EvalCase::builder("c1").input(json!(1)).expected_output(json!(1)).build());
        suite.add_case(EvalCase::builder("c2").input(json!(2)).expected_output(json!(2)).build());
        assert_eq!(suite.case_count(), 2);
    }

    #[test]
    fn test_eval_suite_by_tag() {
        let mut suite = EvalSuite::new("s");
        suite.add_case(EvalCase::builder("c1").tag("math").input(json!(1)).expected_output(json!(1)).build());
        suite.add_case(EvalCase::builder("c2").tag("text").input(json!(2)).expected_output(json!(2)).build());
        suite.add_case(EvalCase::builder("c3").tag("math").input(json!(3)).expected_output(json!(3)).build());

        let math_cases = suite.by_tag("math");
        assert_eq!(math_cases.len(), 2);
        assert_eq!(math_cases[0].name, "c1");
        assert_eq!(math_cases[1].name, "c3");
    }

    #[test]
    fn test_eval_suite_by_tag_no_match() {
        let suite = EvalSuite::new("s");
        assert!(suite.by_tag("nonexistent").is_empty());
    }

    #[test]
    fn test_eval_suite_run_all_pass() {
        let mut suite = EvalSuite::new("s");
        suite.add_case(EvalCase::builder("c1").input(json!("a")).expected_output(json!("a")).build());
        suite.add_case(EvalCase::builder("c2").input(json!(42)).expected_output(json!(42)).build());

        let metrics: Vec<Box<dyn EvalMetric>> = vec![Box::new(ExactMatchMetric)];
        let report = suite.run(&|input| input.clone(), &metrics);

        assert_eq!(report.results.len(), 2);
        assert!((report.overall_score - 1.0).abs() < f64::EPSILON);
        assert!(report.results.iter().all(|r| r.passed));
    }

    #[test]
    fn test_eval_suite_run_all_fail() {
        let mut suite = EvalSuite::new("s");
        suite.add_case(EvalCase::builder("c1").input(json!("a")).expected_output(json!("b")).build());

        let metrics: Vec<Box<dyn EvalMetric>> = vec![Box::new(ExactMatchMetric)];
        let report = suite.run(&|input| input.clone(), &metrics);

        assert!((report.overall_score - 0.0).abs() < f64::EPSILON);
        assert!(!report.results[0].passed);
    }

    #[test]
    fn test_eval_suite_run_mixed() {
        let mut suite = EvalSuite::new("s");
        suite.add_case(EvalCase::builder("pass").input(json!("x")).expected_output(json!("x")).build());
        suite.add_case(EvalCase::builder("fail").input(json!("x")).expected_output(json!("y")).build());

        let metrics: Vec<Box<dyn EvalMetric>> = vec![Box::new(ExactMatchMetric)];
        let report = suite.run(&|input| input.clone(), &metrics);

        assert!((report.overall_score - 0.5).abs() < f64::EPSILON);
        assert!(report.results[0].passed);
        assert!(!report.results[1].passed);
    }

    #[test]
    fn test_eval_suite_run_multiple_metrics() {
        let mut suite = EvalSuite::new("s");
        suite.add_case(
            EvalCase::builder("c1")
                .input(json!("hello world"))
                .expected_output(json!("hello world"))
                .build(),
        );

        let metrics: Vec<Box<dyn EvalMetric>> = vec![
            Box::new(ExactMatchMetric),
            Box::new(ContainsMetric::new(false)),
        ];
        let report = suite.run(&|input| input.clone(), &metrics);

        assert_eq!(report.results[0].scores.len(), 2);
        assert!((report.results[0].avg_score() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_suite_run_empty() {
        let suite = EvalSuite::new("empty");
        let metrics: Vec<Box<dyn EvalMetric>> = vec![Box::new(ExactMatchMetric)];
        let report = suite.run(&|input| input.clone(), &metrics);

        assert_eq!(report.results.len(), 0);
        assert!((report.overall_score - 0.0).abs() < f64::EPSILON);
    }

    // -- EvalReport --

    #[test]
    fn test_eval_report_passing_rate() {
        let mut suite = EvalSuite::new("s");
        for i in 0..10 {
            let expected = if i < 7 { json!(i) } else { json!("wrong") };
            suite.add_case(
                EvalCase::builder(format!("c{}", i))
                    .input(json!(i))
                    .expected_output(expected)
                    .build(),
            );
        }
        let metrics: Vec<Box<dyn EvalMetric>> = vec![Box::new(ExactMatchMetric)];
        let report = suite.run(&|input| input.clone(), &metrics);

        assert!((report.passing_rate(0.5) - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_report_passing_rate_empty() {
        let suite = EvalSuite::new("s");
        let metrics: Vec<Box<dyn EvalMetric>> = vec![Box::new(ExactMatchMetric)];
        let report = suite.run(&|_| json!(null), &metrics);
        assert!((report.passing_rate(0.5) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_report_failures() {
        let mut suite = EvalSuite::new("s");
        suite.add_case(EvalCase::builder("pass").input(json!(1)).expected_output(json!(1)).build());
        suite.add_case(EvalCase::builder("fail").input(json!(1)).expected_output(json!(2)).build());

        let metrics: Vec<Box<dyn EvalMetric>> = vec![Box::new(ExactMatchMetric)];
        let report = suite.run(&|input| input.clone(), &metrics);

        let failures = report.failures(0.5);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].case_name, "fail");
    }

    #[test]
    fn test_eval_report_to_json() {
        let mut suite = EvalSuite::new("my_suite");
        suite.add_case(EvalCase::builder("c").input(json!(1)).expected_output(json!(1)).build());

        let metrics: Vec<Box<dyn EvalMetric>> = vec![Box::new(ExactMatchMetric)];
        let report = suite.run(&|input| input.clone(), &metrics);

        let j = report.to_json();
        assert_eq!(j["suite_name"], "my_suite");
        assert_eq!(j["case_count"], 1);
        assert!(j["overall_score"].is_number());
        assert!(j["duration_ms"].is_number());
        assert!(j["results"].is_array());
    }

    #[test]
    fn test_eval_report_duration() {
        let mut suite = EvalSuite::new("s");
        suite.add_case(EvalCase::builder("c").input(json!(1)).expected_output(json!(1)).build());
        let metrics: Vec<Box<dyn EvalMetric>> = vec![Box::new(ExactMatchMetric)];
        let report = suite.run(&|input| input.clone(), &metrics);
        // Duration should be non-negative
        assert!(report.duration_ms < 10_000);
    }

    // -- EvalCaseResult --

    #[test]
    fn test_eval_case_result_avg_score() {
        let result = EvalCaseResult {
            case_name: "c".into(),
            scores: vec![
                EvalScore::new("m1", 1.0),
                EvalScore::new("m2", 0.5),
                EvalScore::new("m3", 0.0),
            ],
            input: json!(null),
            expected: json!(null),
            actual: json!(null),
            passed: false,
        };
        assert!((result.avg_score() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_case_result_avg_score_empty() {
        let result = EvalCaseResult {
            case_name: "c".into(),
            scores: vec![],
            input: json!(null),
            expected: json!(null),
            actual: json!(null),
            passed: false,
        };
        assert!((result.avg_score() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_case_result_to_json() {
        let result = EvalCaseResult {
            case_name: "c".into(),
            scores: vec![EvalScore::new("m", 0.8)],
            input: json!("in"),
            expected: json!("exp"),
            actual: json!("act"),
            passed: true,
        };
        let j = result.to_json();
        assert_eq!(j["case_name"], "c");
        assert_eq!(j["passed"], true);
        assert!(j["avg_score"].is_number());
    }

    // -- Edge cases --

    #[test]
    fn test_eval_score_nan_clamped() {
        // NaN comparisons: clamp(0.0, 1.0) on NaN returns 0.0 in Rust
        let score = EvalScore::new("m", f64::NAN);
        // NaN.clamp(0.0, 1.0) = NaN in Rust, but is_passing should return false
        assert!(!score.is_passing(0.5));
    }

    #[test]
    fn test_contains_empty_expected() {
        let m = ContainsMetric::new(false);
        // Empty string is always contained
        let score = m.evaluate(&json!(""), &json!("anything"));
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_similarity_empty_arrays() {
        let m = JsonSimilarityMetric;
        let score = m.evaluate(&json!([]), &json!([]));
        assert!((score.score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_json_similarity_actual_shorter_array() {
        let m = JsonSimilarityMetric;
        let score = m.evaluate(&json!([1, 2, 3]), &json!([1]));
        // Only first element matches => 1/3
        assert!((score.score - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_numeric_distance_negative_numbers() {
        let m = NumericDistanceMetric::new(20.0);
        let score = m.evaluate(&json!(-10), &json!(10));
        // distance = 20, score = 1 - 20/20 = 0
        assert!((score.score - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_eval_suite_run_no_metrics() {
        let mut suite = EvalSuite::new("s");
        suite.add_case(EvalCase::builder("c").input(json!(1)).expected_output(json!(1)).build());

        let metrics: Vec<Box<dyn EvalMetric>> = vec![];
        let report = suite.run(&|input| input.clone(), &metrics);

        // No metrics means avg_score = 0.0, so not passing
        assert_eq!(report.results.len(), 1);
        assert!(!report.results[0].passed);
    }
}
