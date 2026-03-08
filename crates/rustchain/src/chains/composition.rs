//! Chain composition patterns for building complex LLM workflows.
//!
//! This module provides high-level composition primitives that mirror LangChain's
//! `SequentialChain`, `TransformChain`, and related patterns, using simple
//! `Box<dyn Fn>` handlers and `serde_json::Value` for I/O.
//!
//! # Types
//!
//! - [`ChainStep`] — A named step with input/output key declarations.
//! - [`SequentialChain`] — Runs steps in order, piping output forward.
//! - [`ParallelChain`] — Runs branches concurrently and merges results.
//! - [`TransformChain`] — Applies a single transformation function.
//! - [`MapChain`] — Applies a function to each element of a JSON array.
//! - [`ConditionalChain`] — Routes input to different handlers based on predicates.
//! - [`ChainPipeline`] — Fluent builder for composing the above primitives.
//! - [`ChainMetrics`] — Tracks per-step execution metrics.

use serde_json::{json, Value};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Handler type alias
// ---------------------------------------------------------------------------

/// A boxed handler function used throughout this module.
pub type Handler = Box<dyn Fn(Value) -> Result<Value, String> + Send + Sync>;

// ---------------------------------------------------------------------------
// ChainStep
// ---------------------------------------------------------------------------

/// A named step in a chain with declared input and output keys.
pub struct ChainStep {
    name: String,
    handler: Handler,
    input_keys: Vec<String>,
    output_keys: Vec<String>,
}

/// Builder for [`ChainStep`].
pub struct ChainStepBuilder {
    name: Option<String>,
    handler: Option<Handler>,
    input_keys: Vec<String>,
    output_keys: Vec<String>,
}

impl ChainStepBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            name: None,
            handler: None,
            input_keys: Vec::new(),
            output_keys: Vec::new(),
        }
    }

    /// Set the step name.
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the handler function.
    pub fn handler(mut self, handler: Handler) -> Self {
        self.handler = Some(handler);
        self
    }

    /// Set the expected input keys.
    pub fn input_keys(mut self, keys: Vec<String>) -> Self {
        self.input_keys = keys;
        self
    }

    /// Set the output keys this step produces.
    pub fn output_keys(mut self, keys: Vec<String>) -> Self {
        self.output_keys = keys;
        self
    }

    /// Build the [`ChainStep`]. Panics if name or handler are not set.
    pub fn build(self) -> ChainStep {
        ChainStep {
            name: self.name.expect("ChainStep requires a name"),
            handler: self.handler.expect("ChainStep requires a handler"),
            input_keys: self.input_keys,
            output_keys: self.output_keys,
        }
    }
}

impl Default for ChainStepBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ChainStep {
    /// Create a new builder.
    pub fn builder() -> ChainStepBuilder {
        ChainStepBuilder::new()
    }

    /// Get the step name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the declared input keys.
    pub fn input_keys(&self) -> &[String] {
        &self.input_keys
    }

    /// Get the declared output keys.
    pub fn output_keys(&self) -> &[String] {
        &self.output_keys
    }

    /// Execute this step with the given input.
    pub fn execute(&self, input: Value) -> Result<Value, String> {
        (self.handler)(input)
    }
}

// ---------------------------------------------------------------------------
// SequentialChain (composition-level)
// ---------------------------------------------------------------------------

/// Runs steps in sequence, passing each step's output as the next step's input.
pub struct CompositionSequentialChain {
    steps: Vec<ChainStep>,
}

impl CompositionSequentialChain {
    /// Create an empty sequential chain.
    pub fn new() -> Self {
        Self { steps: Vec::new() }
    }

    /// Add a step to the end of the chain.
    pub fn add_step(&mut self, step: ChainStep) {
        self.steps.push(step);
    }

    /// Execute all steps in order. The output of each step feeds the next.
    pub fn execute(&self, input: Value) -> Result<Value, String> {
        let mut current = input;
        for step in &self.steps {
            current = step.execute(current)?;
        }
        Ok(current)
    }

    /// Return the number of steps.
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Return the names of all steps in order.
    pub fn step_names(&self) -> Vec<&str> {
        self.steps.iter().map(|s| s.name.as_str()).collect()
    }
}

impl Default for CompositionSequentialChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ParallelChain
// ---------------------------------------------------------------------------

/// A named branch for parallel execution.
struct Branch {
    name: String,
    handler: Handler,
}

/// Runs multiple branches on the same input and merges the results into a
/// single JSON object keyed by branch name.
pub struct ParallelChain {
    branches: Vec<Branch>,
}

impl ParallelChain {
    /// Create an empty parallel chain.
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
        }
    }

    /// Add a named branch.
    pub fn add_branch(
        &mut self,
        name: impl Into<String>,
        handler: Handler,
    ) {
        self.branches.push(Branch {
            name: name.into(),
            handler,
        });
    }

    /// Execute all branches on the same input and merge results into a single
    /// JSON object. Each branch's output is stored under its name key.
    pub fn execute(&self, input: Value) -> Result<Value, String> {
        let mut merged = serde_json::Map::new();
        for branch in &self.branches {
            let result = (branch.handler)(input.clone())?;
            merged.insert(branch.name.clone(), result);
        }
        Ok(Value::Object(merged))
    }

    /// Return the number of branches.
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }
}

impl Default for ParallelChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TransformChain (composition-level)
// ---------------------------------------------------------------------------

/// Applies a single transformation function to the input.
pub struct CompositionTransformChain {
    transform: Handler,
}

impl CompositionTransformChain {
    /// Create a new transform chain with the given function.
    pub fn new(transform: Handler) -> Self {
        Self { transform }
    }

    /// Execute the transformation.
    pub fn execute(&self, input: Value) -> Result<Value, String> {
        (self.transform)(input)
    }
}

// ---------------------------------------------------------------------------
// MapChain
// ---------------------------------------------------------------------------

/// Applies a function to each element of a JSON array input, returning the
/// collected results as a new JSON array.
pub struct MapChain {
    inner: Handler,
}

impl MapChain {
    /// Create a new map chain with the given per-element handler.
    pub fn new(inner: Handler) -> Self {
        Self { inner }
    }

    /// Execute the handler on each element of the input array.
    ///
    /// Returns an error if the input is not a JSON array.
    pub fn execute(&self, input: Value) -> Result<Value, String> {
        let arr = input
            .as_array()
            .ok_or_else(|| "MapChain expects a JSON array input".to_string())?;
        let results: Result<Vec<Value>, String> =
            arr.iter().map(|item| (self.inner)(item.clone())).collect();
        Ok(Value::Array(results?))
    }
}

// ---------------------------------------------------------------------------
// ConditionalChain (composition-level)
// ---------------------------------------------------------------------------

/// A condition-handler pair for the conditional chain.
struct ConditionBranch {
    condition: Box<dyn Fn(&Value) -> bool + Send + Sync>,
    chain: Handler,
}

/// Routes input to different handlers based on predicate functions.
///
/// Conditions are evaluated in order; the first matching condition's handler
/// is executed. If no condition matches the optional `otherwise` handler is
/// used, or an error is returned.
pub struct CompositionConditionalChain {
    branches: Vec<ConditionBranch>,
    otherwise: Option<Handler>,
}

impl CompositionConditionalChain {
    /// Create a new conditional chain with no branches.
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
            otherwise: None,
        }
    }

    /// Add a condition-handler pair. Conditions are evaluated in insertion order.
    pub fn when(
        &mut self,
        condition: Box<dyn Fn(&Value) -> bool + Send + Sync>,
        chain: Handler,
    ) {
        self.branches.push(ConditionBranch { condition, chain });
    }

    /// Set the fallback handler used when no condition matches.
    pub fn otherwise(&mut self, chain: Handler) {
        self.otherwise = Some(chain);
    }

    /// Execute the chain: evaluate conditions in order and run the first match.
    pub fn execute(&self, input: Value) -> Result<Value, String> {
        for branch in &self.branches {
            if (branch.condition)(&input) {
                return (branch.chain)(input);
            }
        }
        if let Some(ref fallback) = self.otherwise {
            return (fallback)(input);
        }
        Err("No condition matched and no otherwise handler set".to_string())
    }
}

impl Default for CompositionConditionalChain {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ChainPipeline
// ---------------------------------------------------------------------------

/// A pipeline operation — one element in the fluent builder.
enum PipelineOp {
    Step(ChainStep),
    Parallel(Vec<Branch>),
    Transform(Handler),
    Map(Handler),
    Conditional {
        branches: Vec<ConditionBranch>,
        otherwise: Option<Handler>,
    },
}

/// Fluent builder for composing chain operations into a pipeline.
///
/// Each operation wraps the previous output, so the overall pipeline behaves
/// like a sequential chain of heterogeneous operations.
pub struct ChainPipeline {
    ops: Vec<PipelineOp>,
}

impl ChainPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Append a named step.
    pub fn then(mut self, step: ChainStep) -> Self {
        self.ops.push(PipelineOp::Step(step));
        self
    }

    /// Append a parallel execution stage from a list of `(name, handler)` pairs.
    pub fn parallel(mut self, branches: Vec<(String, Handler)>) -> Self {
        let b = branches
            .into_iter()
            .map(|(name, handler)| Branch { name, handler })
            .collect();
        self.ops.push(PipelineOp::Parallel(b));
        self
    }

    /// Append a transformation stage.
    pub fn transform(mut self, f: Handler) -> Self {
        self.ops.push(PipelineOp::Transform(f));
        self
    }

    /// Append a map stage (applied to each array element).
    pub fn map(mut self, f: Handler) -> Self {
        self.ops.push(PipelineOp::Map(f));
        self
    }

    /// Append a conditional routing stage.
    pub fn conditional(
        mut self,
        conditions: Vec<(Box<dyn Fn(&Value) -> bool + Send + Sync>, Handler)>,
        otherwise: Option<Handler>,
    ) -> Self {
        let branches = conditions
            .into_iter()
            .map(|(condition, chain)| ConditionBranch { condition, chain })
            .collect();
        self.ops.push(PipelineOp::Conditional {
            branches,
            otherwise,
        });
        self
    }

    /// Execute the full pipeline.
    pub fn execute(&self, input: Value) -> Result<Value, String> {
        let mut current = input;
        for op in &self.ops {
            current = match op {
                PipelineOp::Step(step) => step.execute(current)?,
                PipelineOp::Parallel(branches) => {
                    let mut merged = serde_json::Map::new();
                    for branch in branches {
                        let result = (branch.handler)(current.clone())?;
                        merged.insert(branch.name.clone(), result);
                    }
                    Value::Object(merged)
                }
                PipelineOp::Transform(f) => f(current)?,
                PipelineOp::Map(f) => {
                    let arr = current
                        .as_array()
                        .ok_or_else(|| "Pipeline map expects a JSON array".to_string())?;
                    let results: Result<Vec<Value>, String> =
                        arr.iter().map(|item| f(item.clone())).collect();
                    Value::Array(results?)
                }
                PipelineOp::Conditional {
                    branches,
                    otherwise,
                } => {
                    let mut matched = false;
                    let mut result = Value::Null;
                    for branch in branches {
                        if (branch.condition)(&current) {
                            result = (branch.chain)(current.clone())?;
                            matched = true;
                            break;
                        }
                    }
                    if !matched {
                        if let Some(ref fallback) = otherwise {
                            result = fallback(current)?;
                        } else {
                            return Err(
                                "No condition matched in pipeline conditional".to_string()
                            );
                        }
                    }
                    result
                }
            };
        }
        Ok(current)
    }
}

impl Default for ChainPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ChainMetrics
// ---------------------------------------------------------------------------

/// Metrics for a single step execution.
#[derive(Debug, Clone)]
pub struct StepMetric {
    /// Name of the step.
    pub step_name: String,
    /// Duration in milliseconds.
    pub duration_ms: u64,
    /// Whether the step succeeded.
    pub success: bool,
}

/// Tracks execution metrics across chain steps.
pub struct ChainMetrics {
    metrics: Vec<StepMetric>,
}

impl ChainMetrics {
    /// Create an empty metrics tracker.
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
        }
    }

    /// Record a step execution.
    pub fn record(&mut self, step_name: impl Into<String>, duration_ms: u64, success: bool) {
        self.metrics.push(StepMetric {
            step_name: step_name.into(),
            duration_ms,
            success,
        });
    }

    /// Total duration across all recorded steps.
    pub fn total_duration_ms(&self) -> u64 {
        self.metrics.iter().map(|m| m.duration_ms).sum()
    }

    /// Return a reference to all step metrics.
    pub fn step_metrics(&self) -> &[StepMetric] {
        &self.metrics
    }

    /// Fraction of steps that succeeded (0.0 – 1.0). Returns 0.0 if empty.
    pub fn success_rate(&self) -> f64 {
        if self.metrics.is_empty() {
            return 0.0;
        }
        let successes = self.metrics.iter().filter(|m| m.success).count();
        successes as f64 / self.metrics.len() as f64
    }

    /// Serialize metrics to a JSON value.
    pub fn to_json(&self) -> Value {
        let steps: Vec<Value> = self
            .metrics
            .iter()
            .map(|m| {
                json!({
                    "step_name": m.step_name,
                    "duration_ms": m.duration_ms,
                    "success": m.success,
                })
            })
            .collect();
        json!({
            "total_duration_ms": self.total_duration_ms(),
            "success_rate": self.success_rate(),
            "step_count": self.metrics.len(),
            "steps": steps,
        })
    }
}

impl Default for ChainMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Execute a [`CompositionSequentialChain`] while recording metrics.
pub fn execute_with_metrics(
    chain: &CompositionSequentialChain,
    input: Value,
) -> (Result<Value, String>, ChainMetrics) {
    let mut metrics = ChainMetrics::new();
    let mut current = input;
    for step in &chain.steps {
        let start = Instant::now();
        match step.execute(current.clone()) {
            Ok(output) => {
                let elapsed = start.elapsed().as_millis() as u64;
                metrics.record(step.name(), elapsed, true);
                current = output;
            }
            Err(e) => {
                let elapsed = start.elapsed().as_millis() as u64;
                metrics.record(step.name(), elapsed, false);
                return (Err(e), metrics);
            }
        }
    }
    (Ok(current), metrics)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── Helpers ──────────────────────────────────────────────────────────

    fn double_handler() -> Handler {
        Box::new(|v| {
            let n = v
                .get("value")
                .and_then(|v| v.as_i64())
                .ok_or("missing value")?;
            Ok(json!({ "value": n * 2 }))
        })
    }

    fn add_one_handler() -> Handler {
        Box::new(|v| {
            let n = v
                .get("value")
                .and_then(|v| v.as_i64())
                .ok_or("missing value")?;
            Ok(json!({ "value": n + 1 }))
        })
    }

    fn uppercase_handler() -> Handler {
        Box::new(|v| {
            let s = v
                .get("text")
                .and_then(|v| v.as_str())
                .ok_or("missing text")?;
            Ok(json!({ "text": s.to_uppercase() }))
        })
    }

    fn failing_handler() -> Handler {
        Box::new(|_| Err("intentional failure".to_string()))
    }

    fn identity_handler() -> Handler {
        Box::new(|v| Ok(v))
    }

    // ── ChainStep tests ────────────────────────────────────────────────

    #[test]
    fn test_chain_step_creation() {
        let step = ChainStep::builder()
            .name("double")
            .handler(double_handler())
            .input_keys(vec!["value".into()])
            .output_keys(vec!["value".into()])
            .build();
        assert_eq!(step.name(), "double");
        assert_eq!(step.input_keys(), &["value".to_string()]);
        assert_eq!(step.output_keys(), &["value".to_string()]);
    }

    #[test]
    fn test_chain_step_execute() {
        let step = ChainStep::builder()
            .name("double")
            .handler(double_handler())
            .build();
        let result = step.execute(json!({"value": 5})).unwrap();
        assert_eq!(result["value"], 10);
    }

    #[test]
    fn test_chain_step_execute_error() {
        let step = ChainStep::builder()
            .name("fail")
            .handler(failing_handler())
            .build();
        let result = step.execute(json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_chain_step_empty_keys() {
        let step = ChainStep::builder()
            .name("noop")
            .handler(identity_handler())
            .build();
        assert!(step.input_keys().is_empty());
        assert!(step.output_keys().is_empty());
    }

    #[test]
    #[should_panic(expected = "ChainStep requires a name")]
    fn test_chain_step_missing_name_panics() {
        ChainStep::builder().handler(identity_handler()).build();
    }

    #[test]
    #[should_panic(expected = "ChainStep requires a handler")]
    fn test_chain_step_missing_handler_panics() {
        ChainStep::builder().name("x").build();
    }

    // ── SequentialChain tests ───────────────────────────────────────────

    #[test]
    fn test_sequential_chain_empty() {
        let chain = CompositionSequentialChain::new();
        assert_eq!(chain.step_count(), 0);
        let result = chain.execute(json!({"value": 1})).unwrap();
        assert_eq!(result, json!({"value": 1}));
    }

    #[test]
    fn test_sequential_chain_single_step() {
        let mut chain = CompositionSequentialChain::new();
        chain.add_step(
            ChainStep::builder()
                .name("double")
                .handler(double_handler())
                .build(),
        );
        assert_eq!(chain.step_count(), 1);
        let result = chain.execute(json!({"value": 3})).unwrap();
        assert_eq!(result["value"], 6);
    }

    #[test]
    fn test_sequential_chain_multi_step() {
        let mut chain = CompositionSequentialChain::new();
        chain.add_step(
            ChainStep::builder()
                .name("double")
                .handler(double_handler())
                .build(),
        );
        chain.add_step(
            ChainStep::builder()
                .name("add_one")
                .handler(add_one_handler())
                .build(),
        );
        // (5 * 2) + 1 = 11
        let result = chain.execute(json!({"value": 5})).unwrap();
        assert_eq!(result["value"], 11);
    }

    #[test]
    fn test_sequential_chain_step_names() {
        let mut chain = CompositionSequentialChain::new();
        chain.add_step(ChainStep::builder().name("a").handler(identity_handler()).build());
        chain.add_step(ChainStep::builder().name("b").handler(identity_handler()).build());
        chain.add_step(ChainStep::builder().name("c").handler(identity_handler()).build());
        assert_eq!(chain.step_names(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_sequential_chain_error_propagation() {
        let mut chain = CompositionSequentialChain::new();
        chain.add_step(
            ChainStep::builder()
                .name("double")
                .handler(double_handler())
                .build(),
        );
        chain.add_step(
            ChainStep::builder()
                .name("fail")
                .handler(failing_handler())
                .build(),
        );
        chain.add_step(
            ChainStep::builder()
                .name("add_one")
                .handler(add_one_handler())
                .build(),
        );
        let result = chain.execute(json!({"value": 1}));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "intentional failure");
    }

    #[test]
    fn test_sequential_chain_three_steps() {
        let mut chain = CompositionSequentialChain::new();
        // double, double, add_one: (2*2)*2 + 1 = 9
        chain.add_step(ChainStep::builder().name("d1").handler(double_handler()).build());
        chain.add_step(ChainStep::builder().name("d2").handler(double_handler()).build());
        chain.add_step(ChainStep::builder().name("a1").handler(add_one_handler()).build());
        let result = chain.execute(json!({"value": 2})).unwrap();
        assert_eq!(result["value"], 9);
    }

    // ── ParallelChain tests ─────────────────────────────────────────────

    #[test]
    fn test_parallel_chain_empty() {
        let chain = ParallelChain::new();
        assert_eq!(chain.branch_count(), 0);
        let result = chain.execute(json!({"value": 1})).unwrap();
        assert_eq!(result, json!({}));
    }

    #[test]
    fn test_parallel_chain_single_branch() {
        let mut chain = ParallelChain::new();
        chain.add_branch("doubled", double_handler());
        assert_eq!(chain.branch_count(), 1);
        let result = chain.execute(json!({"value": 3})).unwrap();
        assert_eq!(result["doubled"]["value"], 6);
    }

    #[test]
    fn test_parallel_chain_multiple_branches() {
        let mut chain = ParallelChain::new();
        chain.add_branch("doubled", double_handler());
        chain.add_branch("plus_one", add_one_handler());
        let result = chain.execute(json!({"value": 4})).unwrap();
        assert_eq!(result["doubled"]["value"], 8);
        assert_eq!(result["plus_one"]["value"], 5);
    }

    #[test]
    fn test_parallel_chain_error_in_branch() {
        let mut chain = ParallelChain::new();
        chain.add_branch("ok", identity_handler());
        chain.add_branch("fail", failing_handler());
        let result = chain.execute(json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_parallel_chain_branch_count() {
        let mut chain = ParallelChain::new();
        chain.add_branch("a", identity_handler());
        chain.add_branch("b", identity_handler());
        chain.add_branch("c", identity_handler());
        assert_eq!(chain.branch_count(), 3);
    }

    // ── TransformChain tests ────────────────────────────────────────────

    #[test]
    fn test_transform_chain_basic() {
        let chain = CompositionTransformChain::new(Box::new(|v| {
            let n = v["value"].as_i64().unwrap_or(0);
            Ok(json!({ "result": n * n }))
        }));
        let result = chain.execute(json!({"value": 7})).unwrap();
        assert_eq!(result["result"], 49);
    }

    #[test]
    fn test_transform_chain_identity() {
        let chain = CompositionTransformChain::new(identity_handler());
        let input = json!({"a": 1, "b": 2});
        let result = chain.execute(input.clone()).unwrap();
        assert_eq!(result, input);
    }

    #[test]
    fn test_transform_chain_error() {
        let chain = CompositionTransformChain::new(failing_handler());
        let result = chain.execute(json!({}));
        assert!(result.is_err());
    }

    // ── MapChain tests ──────────────────────────────────────────────────

    #[test]
    fn test_map_chain_basic() {
        let chain = MapChain::new(Box::new(|v| {
            let n = v.as_i64().ok_or("not a number")?;
            Ok(json!(n * 2))
        }));
        let result = chain.execute(json!([1, 2, 3])).unwrap();
        assert_eq!(result, json!([2, 4, 6]));
    }

    #[test]
    fn test_map_chain_empty_array() {
        let chain = MapChain::new(identity_handler());
        let result = chain.execute(json!([])).unwrap();
        assert_eq!(result, json!([]));
    }

    #[test]
    fn test_map_chain_non_array_error() {
        let chain = MapChain::new(identity_handler());
        let result = chain.execute(json!({"not": "array"}));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("array"));
    }

    #[test]
    fn test_map_chain_error_in_element() {
        let chain = MapChain::new(Box::new(|v| {
            let n = v.as_i64().ok_or("not a number".to_string())?;
            if n > 2 {
                return Err("too large".to_string());
            }
            Ok(json!(n))
        }));
        let result = chain.execute(json!([1, 2, 3]));
        assert!(result.is_err());
    }

    #[test]
    fn test_map_chain_objects() {
        let chain = MapChain::new(Box::new(|v| {
            let name = v["name"].as_str().unwrap_or("unknown");
            Ok(json!({ "greeting": format!("Hello, {}!", name) }))
        }));
        let result = chain
            .execute(json!([{"name": "Alice"}, {"name": "Bob"}]))
            .unwrap();
        assert_eq!(result[0]["greeting"], "Hello, Alice!");
        assert_eq!(result[1]["greeting"], "Hello, Bob!");
    }

    // ── ConditionalChain tests ──────────────────────────────────────────

    #[test]
    fn test_conditional_chain_first_match() {
        let mut chain = CompositionConditionalChain::new();
        chain.when(
            Box::new(|v| v.get("type").and_then(|t| t.as_str()) == Some("a")),
            Box::new(|_| Ok(json!({"result": "matched_a"}))),
        );
        chain.when(
            Box::new(|v| v.get("type").and_then(|t| t.as_str()) == Some("b")),
            Box::new(|_| Ok(json!({"result": "matched_b"}))),
        );
        let result = chain.execute(json!({"type": "a"})).unwrap();
        assert_eq!(result["result"], "matched_a");
    }

    #[test]
    fn test_conditional_chain_second_match() {
        let mut chain = CompositionConditionalChain::new();
        chain.when(
            Box::new(|v| v.get("type").and_then(|t| t.as_str()) == Some("a")),
            Box::new(|_| Ok(json!({"result": "matched_a"}))),
        );
        chain.when(
            Box::new(|v| v.get("type").and_then(|t| t.as_str()) == Some("b")),
            Box::new(|_| Ok(json!({"result": "matched_b"}))),
        );
        let result = chain.execute(json!({"type": "b"})).unwrap();
        assert_eq!(result["result"], "matched_b");
    }

    #[test]
    fn test_conditional_chain_otherwise() {
        let mut chain = CompositionConditionalChain::new();
        chain.when(
            Box::new(|v| v.get("type").and_then(|t| t.as_str()) == Some("a")),
            Box::new(|_| Ok(json!({"result": "matched_a"}))),
        );
        chain.otherwise(Box::new(|_| Ok(json!({"result": "default"}))));
        let result = chain.execute(json!({"type": "z"})).unwrap();
        assert_eq!(result["result"], "default");
    }

    #[test]
    fn test_conditional_chain_no_match_no_otherwise() {
        let mut chain = CompositionConditionalChain::new();
        chain.when(
            Box::new(|_| false),
            Box::new(|_| Ok(json!({"unreachable": true}))),
        );
        let result = chain.execute(json!({}));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No condition matched"));
    }

    #[test]
    fn test_conditional_chain_empty() {
        let chain = CompositionConditionalChain::new();
        let result = chain.execute(json!({}));
        assert!(result.is_err());
    }

    // ── ChainPipeline tests ────────────────────────────────────────────

    #[test]
    fn test_pipeline_empty() {
        let pipeline = ChainPipeline::new();
        let result = pipeline.execute(json!({"value": 42})).unwrap();
        assert_eq!(result, json!({"value": 42}));
    }

    #[test]
    fn test_pipeline_then() {
        let pipeline = ChainPipeline::new().then(
            ChainStep::builder()
                .name("double")
                .handler(double_handler())
                .build(),
        );
        let result = pipeline.execute(json!({"value": 5})).unwrap();
        assert_eq!(result["value"], 10);
    }

    #[test]
    fn test_pipeline_transform() {
        let pipeline = ChainPipeline::new().transform(Box::new(|v| {
            let n = v["value"].as_i64().unwrap_or(0);
            Ok(json!({ "squared": n * n }))
        }));
        let result = pipeline.execute(json!({"value": 4})).unwrap();
        assert_eq!(result["squared"], 16);
    }

    #[test]
    fn test_pipeline_map() {
        let pipeline = ChainPipeline::new().map(Box::new(|v| {
            let n = v.as_i64().ok_or("not a number")?;
            Ok(json!(n + 10))
        }));
        let result = pipeline.execute(json!([1, 2, 3])).unwrap();
        assert_eq!(result, json!([11, 12, 13]));
    }

    #[test]
    fn test_pipeline_parallel() {
        let pipeline = ChainPipeline::new().parallel(vec![
            ("doubled".into(), double_handler()),
            ("plus_one".into(), add_one_handler()),
        ]);
        let result = pipeline.execute(json!({"value": 3})).unwrap();
        assert_eq!(result["doubled"]["value"], 6);
        assert_eq!(result["plus_one"]["value"], 4);
    }

    #[test]
    fn test_pipeline_conditional() {
        let pipeline = ChainPipeline::new().conditional(
            vec![(
                Box::new(|v: &Value| v.get("positive").and_then(|p| p.as_bool()) == Some(true))
                    as Box<dyn Fn(&Value) -> bool + Send + Sync>,
                Box::new(|_| Ok(json!({"sentiment": "positive"}))) as Handler,
            )],
            Some(Box::new(|_| Ok(json!({"sentiment": "negative"})))),
        );
        let pos = pipeline.execute(json!({"positive": true})).unwrap();
        assert_eq!(pos["sentiment"], "positive");
        let neg = pipeline.execute(json!({"positive": false})).unwrap();
        assert_eq!(neg["sentiment"], "negative");
    }

    #[test]
    fn test_pipeline_chained_operations() {
        // double then add_one via pipeline
        let pipeline = ChainPipeline::new()
            .then(
                ChainStep::builder()
                    .name("double")
                    .handler(double_handler())
                    .build(),
            )
            .then(
                ChainStep::builder()
                    .name("add_one")
                    .handler(add_one_handler())
                    .build(),
            );
        // (3 * 2) + 1 = 7
        let result = pipeline.execute(json!({"value": 3})).unwrap();
        assert_eq!(result["value"], 7);
    }

    #[test]
    fn test_pipeline_error_propagation() {
        let pipeline = ChainPipeline::new()
            .then(
                ChainStep::builder()
                    .name("fail")
                    .handler(failing_handler())
                    .build(),
            )
            .then(
                ChainStep::builder()
                    .name("unreachable")
                    .handler(identity_handler())
                    .build(),
            );
        let result = pipeline.execute(json!({}));
        assert!(result.is_err());
    }

    // ── ChainMetrics tests ──────────────────────────────────────────────

    #[test]
    fn test_metrics_empty() {
        let metrics = ChainMetrics::new();
        assert_eq!(metrics.total_duration_ms(), 0);
        assert_eq!(metrics.success_rate(), 0.0);
        assert!(metrics.step_metrics().is_empty());
    }

    #[test]
    fn test_metrics_record_and_query() {
        let mut metrics = ChainMetrics::new();
        metrics.record("step1", 100, true);
        metrics.record("step2", 200, true);
        assert_eq!(metrics.total_duration_ms(), 300);
        assert_eq!(metrics.success_rate(), 1.0);
        assert_eq!(metrics.step_metrics().len(), 2);
    }

    #[test]
    fn test_metrics_success_rate_partial() {
        let mut metrics = ChainMetrics::new();
        metrics.record("ok", 50, true);
        metrics.record("fail", 30, false);
        assert!((metrics.success_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_to_json() {
        let mut metrics = ChainMetrics::new();
        metrics.record("step1", 100, true);
        metrics.record("step2", 50, false);
        let j = metrics.to_json();
        assert_eq!(j["total_duration_ms"], 150);
        assert_eq!(j["step_count"], 2);
        assert_eq!(j["steps"].as_array().unwrap().len(), 2);
        assert_eq!(j["steps"][0]["step_name"], "step1");
        assert_eq!(j["steps"][1]["success"], false);
    }

    #[test]
    fn test_metrics_all_failures() {
        let mut metrics = ChainMetrics::new();
        metrics.record("f1", 10, false);
        metrics.record("f2", 20, false);
        assert_eq!(metrics.success_rate(), 0.0);
    }

    #[test]
    fn test_execute_with_metrics_success() {
        let mut chain = CompositionSequentialChain::new();
        chain.add_step(ChainStep::builder().name("double").handler(double_handler()).build());
        chain.add_step(ChainStep::builder().name("add_one").handler(add_one_handler()).build());
        let (result, metrics) = execute_with_metrics(&chain, json!({"value": 4}));
        assert_eq!(result.unwrap()["value"], 9);
        assert_eq!(metrics.step_metrics().len(), 2);
        assert_eq!(metrics.success_rate(), 1.0);
    }

    #[test]
    fn test_execute_with_metrics_failure() {
        let mut chain = CompositionSequentialChain::new();
        chain.add_step(ChainStep::builder().name("double").handler(double_handler()).build());
        chain.add_step(ChainStep::builder().name("fail").handler(failing_handler()).build());
        chain.add_step(ChainStep::builder().name("add_one").handler(add_one_handler()).build());
        let (result, metrics) = execute_with_metrics(&chain, json!({"value": 1}));
        assert!(result.is_err());
        // Only 2 steps recorded (double succeeded, fail failed, add_one skipped)
        assert_eq!(metrics.step_metrics().len(), 2);
        assert!((metrics.success_rate() - 0.5).abs() < f64::EPSILON);
    }

    // ── Edge case / integration tests ───────────────────────────────────

    #[test]
    fn test_sequential_chain_preserves_all_keys() {
        let mut chain = CompositionSequentialChain::new();
        chain.add_step(
            ChainStep::builder()
                .name("enrich")
                .handler(Box::new(|mut v| {
                    v.as_object_mut()
                        .ok_or("not object")?
                        .insert("extra".into(), json!("added"));
                    Ok(v)
                }))
                .build(),
        );
        let result = chain.execute(json!({"original": true})).unwrap();
        assert_eq!(result["original"], true);
        assert_eq!(result["extra"], "added");
    }

    #[test]
    fn test_parallel_chain_independent_results() {
        let mut chain = ParallelChain::new();
        chain.add_branch(
            "upper",
            Box::new(|v| {
                let s = v["text"].as_str().unwrap_or("");
                Ok(json!(s.to_uppercase()))
            }),
        );
        chain.add_branch(
            "lower",
            Box::new(|v| {
                let s = v["text"].as_str().unwrap_or("");
                Ok(json!(s.to_lowercase()))
            }),
        );
        let result = chain.execute(json!({"text": "Hello"})).unwrap();
        assert_eq!(result["upper"], "HELLO");
        assert_eq!(result["lower"], "hello");
    }

    #[test]
    fn test_map_chain_single_element() {
        let chain = MapChain::new(Box::new(|v| {
            let n = v.as_i64().ok_or("not a number")?;
            Ok(json!(n * 3))
        }));
        let result = chain.execute(json!([7])).unwrap();
        assert_eq!(result, json!([21]));
    }

    #[test]
    fn test_conditional_chain_first_condition_wins() {
        let mut chain = CompositionConditionalChain::new();
        // Both conditions are true, but first one should win
        chain.when(Box::new(|_| true), Box::new(|_| Ok(json!({"winner": "first"}))));
        chain.when(Box::new(|_| true), Box::new(|_| Ok(json!({"winner": "second"}))));
        let result = chain.execute(json!({})).unwrap();
        assert_eq!(result["winner"], "first");
    }

    #[test]
    fn test_pipeline_transform_then_map() {
        // Transform to produce an array, then map over it
        let pipeline = ChainPipeline::new()
            .transform(Box::new(|v| {
                let n = v["count"].as_i64().unwrap_or(0);
                Ok(json!((0..n).collect::<Vec<i64>>()))
            }))
            .map(Box::new(|v| {
                let n = v.as_i64().ok_or("not number")?;
                Ok(json!(n * n))
            }));
        let result = pipeline.execute(json!({"count": 4})).unwrap();
        assert_eq!(result, json!([0, 1, 4, 9]));
    }

    #[test]
    fn test_default_impls() {
        // Ensure Default is implemented
        let _ = CompositionSequentialChain::default();
        let _ = ParallelChain::default();
        let _ = CompositionConditionalChain::default();
        let _ = ChainPipeline::default();
        let _ = ChainMetrics::default();
        let _ = ChainStepBuilder::default();
    }
}
