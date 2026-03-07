use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::error::{Result, RustChainError};

/// A single stage in a pipeline with optional error handling and conditional execution.
#[allow(clippy::type_complexity)]
pub struct PipelineStage {
    /// Human-readable name for this stage.
    name: String,
    /// The transformation function applied to the input value.
    transform: Arc<dyn Fn(Value) -> Result<Value> + Send + Sync>,
    /// Optional error handler that produces a fallback value on failure.
    error_handler: Option<Arc<dyn Fn(RustChainError) -> Value + Send + Sync>>,
    /// Optional condition that determines whether this stage should run.
    condition: Option<Arc<dyn Fn(&Value) -> bool + Send + Sync>>,
}

impl PipelineStage {
    /// Create a new pipeline stage with the given name and transform function.
    pub fn new(
        name: impl Into<String>,
        transform: impl Fn(Value) -> Result<Value> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            transform: Arc::new(transform),
            error_handler: None,
            condition: None,
        }
    }

    /// Attach an error handler that produces a fallback value when the transform fails.
    pub fn with_error_handler(
        mut self,
        handler: impl Fn(RustChainError) -> Value + Send + Sync + 'static,
    ) -> Self {
        self.error_handler = Some(Arc::new(handler));
        self
    }

    /// Attach a condition that determines whether this stage should execute.
    ///
    /// If the condition returns `false`, the stage is skipped and the input
    /// is passed through unchanged.
    pub fn with_condition(
        mut self,
        condition: impl Fn(&Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.condition = Some(Arc::new(condition));
        self
    }

    /// Execute the stage transform on the given input.
    ///
    /// If the transform fails and an error handler is set, the handler is
    /// invoked to produce a fallback value. Otherwise the error propagates.
    pub fn execute(&self, input: Value) -> Result<Value> {
        match (self.transform)(input) {
            Ok(v) => Ok(v),
            Err(e) => {
                if let Some(ref handler) = self.error_handler {
                    Ok(handler(e))
                } else {
                    Err(e)
                }
            }
        }
    }

    /// Returns `true` if this stage should run for the given input.
    ///
    /// If no condition is set, always returns `true`.
    pub fn should_run(&self, input: &Value) -> bool {
        match &self.condition {
            Some(cond) => cond(input),
            None => true,
        }
    }
}

/// The result of executing a pipeline, including metadata about which stages
/// ran, which were skipped, timing, and any errors encountered.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// The final output value.
    pub output: Value,
    /// Names of stages that were executed.
    pub stages_executed: Vec<String>,
    /// Names of stages that were skipped due to conditions.
    pub stages_skipped: Vec<String>,
    /// Total wall-clock duration of the pipeline execution.
    pub duration: Duration,
    /// Errors that were caught by error handlers: `(stage_name, error_message)`.
    pub errors: Vec<(String, String)>,
}

impl PipelineResult {
    /// Returns `true` if no errors were encountered during execution.
    pub fn is_success(&self) -> bool {
        self.errors.is_empty()
    }

    /// Serialize the result metadata to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "output": self.output,
            "stages_executed": self.stages_executed,
            "stages_skipped": self.stages_skipped,
            "duration_ms": self.duration.as_millis() as u64,
            "errors": self.errors.iter().map(|(s, e)| json!({"stage": s, "error": e})).collect::<Vec<_>>(),
            "success": self.is_success(),
        })
    }
}

/// An ordered sequence of pipeline stages executed one after another.
///
/// Each stage receives the output of the previous stage as its input.
/// Stages may be conditional, in which case they are skipped and the
/// input passes through unchanged.
pub struct Pipeline {
    /// Human-readable name for the pipeline.
    name: String,
    /// The ordered list of stages.
    stages: Vec<PipelineStage>,
}

impl Pipeline {
    /// Create a new empty pipeline with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            stages: Vec::new(),
        }
    }

    /// Append a stage to the pipeline.
    pub fn add_stage(&mut self, stage: PipelineStage) -> &mut Self {
        self.stages.push(stage);
        self
    }

    /// Shorthand for adding a simple (unconditional, no error handler) stage.
    pub fn stage(
        &mut self,
        name: &str,
        transform: impl Fn(Value) -> Result<Value> + Send + Sync + 'static,
    ) -> &mut Self {
        self.stages.push(PipelineStage::new(name, transform));
        self
    }

    /// Add a stage that only runs when `condition` returns `true`.
    pub fn conditional_stage(
        &mut self,
        name: &str,
        transform: impl Fn(Value) -> Result<Value> + Send + Sync + 'static,
        condition: impl Fn(&Value) -> bool + Send + Sync + 'static,
    ) -> &mut Self {
        self.stages
            .push(PipelineStage::new(name, transform).with_condition(condition));
        self
    }

    /// Execute the pipeline on the given input, returning a `PipelineResult`.
    pub fn execute(&self, input: Value) -> PipelineResult {
        let start = Instant::now();
        let mut current = input;
        let mut stages_executed = Vec::new();
        let mut stages_skipped = Vec::new();
        let mut errors = Vec::new();

        for stage in &self.stages {
            if !stage.should_run(&current) {
                stages_skipped.push(stage.name.clone());
                continue;
            }

            match stage.execute(current.clone()) {
                Ok(output) => {
                    // Check if this was an error-handled result by comparing
                    // against a direct transform call.
                    if let Err(e) = (stage.transform)(current.clone()) {
                        if stage.error_handler.is_some() {
                            errors.push((stage.name.clone(), e.to_string()));
                        }
                    }
                    stages_executed.push(stage.name.clone());
                    current = output;
                }
                Err(e) => {
                    // No error handler caught this — record and stop.
                    errors.push((stage.name.clone(), e.to_string()));
                    stages_executed.push(stage.name.clone());
                    break;
                }
            }
        }

        PipelineResult {
            output: current,
            stages_executed,
            stages_skipped,
            duration: start.elapsed(),
            errors,
        }
    }

    /// Returns the number of stages in the pipeline.
    pub fn stage_count(&self) -> usize {
        self.stages.len()
    }

    /// Returns the names of all stages in order.
    pub fn stage_names(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.name.as_str()).collect()
    }

    /// Returns the pipeline name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Perform a dry run: return the names of stages that *would* execute
    /// for the given input (based on conditions), without actually running
    /// any transforms.
    pub fn dry_run(&self, input: &Value) -> Vec<String> {
        let mut would_run = Vec::new();

        for stage in &self.stages {
            if stage.should_run(input) {
                would_run.push(stage.name.clone());
            }
        }
        would_run
    }
}

/// Runs multiple pipelines in parallel (logically) on the same input and
/// collects results keyed by branch name.
pub struct ParallelPipeline {
    branches: Vec<(String, Pipeline)>,
}

impl ParallelPipeline {
    /// Create a new empty parallel pipeline.
    pub fn new() -> Self {
        Self {
            branches: Vec::new(),
        }
    }

    /// Add a named branch pipeline.
    pub fn add_branch(&mut self, name: &str, pipeline: Pipeline) {
        self.branches.push((name.to_string(), pipeline));
    }

    /// Execute all branches with the same input, returning results keyed by
    /// branch name.
    pub fn execute(&self, input: Value) -> HashMap<String, PipelineResult> {
        let mut results = HashMap::new();
        for (name, pipeline) in &self.branches {
            results.insert(name.clone(), pipeline.execute(input.clone()));
        }
        results
    }

    /// Returns the number of branches.
    pub fn branch_count(&self) -> usize {
        self.branches.len()
    }

    /// Returns the names of all branches in insertion order.
    pub fn branch_names(&self) -> Vec<&str> {
        self.branches.iter().map(|(n, _)| n.as_str()).collect()
    }
}

impl Default for ParallelPipeline {
    fn default() -> Self {
        Self::new()
    }
}

/// Fluent builder for constructing a `Pipeline`.
pub struct PipelineBuilder {
    name: String,
    stages: Vec<PipelineStage>,
}

impl PipelineBuilder {
    /// Start building a pipeline with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            stages: Vec::new(),
        }
    }

    /// Append an unconditional stage.
    pub fn then(
        mut self,
        name: &str,
        transform: impl Fn(Value) -> Result<Value> + Send + Sync + 'static,
    ) -> Self {
        self.stages.push(PipelineStage::new(name, transform));
        self
    }

    /// Append a conditional stage.
    pub fn then_if(
        mut self,
        name: &str,
        transform: impl Fn(Value) -> Result<Value> + Send + Sync + 'static,
        condition: impl Fn(&Value) -> bool + Send + Sync + 'static,
    ) -> Self {
        self.stages
            .push(PipelineStage::new(name, transform).with_condition(condition));
        self
    }

    /// Attach an error handler to the most recently added stage.
    ///
    /// # Panics
    /// Panics if no stages have been added yet.
    pub fn on_error(
        mut self,
        _name: &str,
        handler: impl Fn(RustChainError) -> Value + Send + Sync + 'static,
    ) -> Self {
        let last = self
            .stages
            .pop()
            .expect("on_error requires at least one stage");
        self.stages.push(last.with_error_handler(handler));
        self
    }

    /// Consume the builder and produce a `Pipeline`.
    pub fn build(self) -> Pipeline {
        let mut pipeline = Pipeline::new(&self.name);
        for stage in self.stages {
            pipeline.add_stage(stage);
        }
        pipeline
    }
}

/// Tracks execution statistics across multiple pipeline runs.
pub struct PipelineMetrics {
    total: usize,
    successes: usize,
    failures: usize,
    durations: Vec<Duration>,
    stage_counts: HashMap<String, usize>,
}

impl PipelineMetrics {
    /// Create a new, empty metrics tracker.
    pub fn new() -> Self {
        Self {
            total: 0,
            successes: 0,
            failures: 0,
            durations: Vec::new(),
            stage_counts: HashMap::new(),
        }
    }

    /// Record the result of a pipeline execution.
    pub fn record(&mut self, result: &PipelineResult) {
        self.total += 1;
        if result.is_success() {
            self.successes += 1;
        } else {
            self.failures += 1;
        }
        self.durations.push(result.duration);
        for stage_name in &result.stages_executed {
            *self.stage_counts.entry(stage_name.clone()).or_insert(0) += 1;
        }
    }

    /// Total number of pipeline executions recorded.
    pub fn total_executions(&self) -> usize {
        self.total
    }

    /// Number of successful executions (no errors).
    pub fn success_count(&self) -> usize {
        self.successes
    }

    /// Number of failed executions (at least one error).
    pub fn failure_count(&self) -> usize {
        self.failures
    }

    /// Average duration across all recorded executions, or `None` if no
    /// executions have been recorded.
    pub fn average_duration(&self) -> Option<Duration> {
        if self.durations.is_empty() {
            return None;
        }
        let total: Duration = self.durations.iter().sum();
        Some(total / self.durations.len() as u32)
    }

    /// Returns a map of stage name to execution count.
    pub fn stage_execution_counts(&self) -> HashMap<String, usize> {
        self.stage_counts.clone()
    }

    /// Serialize the metrics to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "total_executions": self.total,
            "successes": self.successes,
            "failures": self.failures,
            "average_duration_ms": self.average_duration().map(|d| d.as_millis() as u64),
            "stage_execution_counts": self.stage_counts,
        })
    }
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // ── PipelineStage tests ──────────────────────────────────────────

    #[test]
    fn test_stage_creation() {
        let stage = PipelineStage::new("double", |v: Value| {
            let n = v.as_i64().unwrap();
            Ok(json!(n * 2))
        });
        assert_eq!(stage.name, "double");
        assert!(stage.error_handler.is_none());
        assert!(stage.condition.is_none());
    }

    #[test]
    fn test_stage_execute_success() {
        let stage = PipelineStage::new("add_one", |v: Value| {
            let n = v.as_i64().unwrap();
            Ok(json!(n + 1))
        });
        let result = stage.execute(json!(5)).unwrap();
        assert_eq!(result, json!(6));
    }

    #[test]
    fn test_stage_execute_failure_no_handler() {
        let stage = PipelineStage::new("fail", |_v: Value| {
            Err(RustChainError::Other("boom".into()))
        });
        let result = stage.execute(json!(1));
        assert!(result.is_err());
    }

    #[test]
    fn test_stage_execute_failure_with_handler() {
        let stage = PipelineStage::new("fail", |_v: Value| {
            Err(RustChainError::Other("boom".into()))
        })
        .with_error_handler(|e| json!({"error": e.to_string()}));

        let result = stage.execute(json!(1)).unwrap();
        assert_eq!(result["error"], json!("boom"));
    }

    #[test]
    fn test_stage_should_run_no_condition() {
        let stage = PipelineStage::new("noop", |v| Ok(v));
        assert!(stage.should_run(&json!(42)));
    }

    #[test]
    fn test_stage_should_run_condition_true() {
        let stage =
            PipelineStage::new("noop", |v| Ok(v)).with_condition(|v| v.as_i64().unwrap_or(0) > 10);
        assert!(stage.should_run(&json!(20)));
    }

    #[test]
    fn test_stage_should_run_condition_false() {
        let stage =
            PipelineStage::new("noop", |v| Ok(v)).with_condition(|v| v.as_i64().unwrap_or(0) > 10);
        assert!(!stage.should_run(&json!(5)));
    }

    #[test]
    fn test_stage_error_handler_not_called_on_success() {
        let stage = PipelineStage::new("add_one", |v: Value| {
            let n = v.as_i64().unwrap();
            Ok(json!(n + 1))
        })
        .with_error_handler(|_| json!("fallback"));

        let result = stage.execute(json!(5)).unwrap();
        assert_eq!(result, json!(6)); // handler NOT invoked
    }

    // ── Pipeline tests ───────────────────────────────────────────────

    #[test]
    fn test_pipeline_sequential_execution() {
        let mut pipeline = Pipeline::new("math");
        pipeline
            .stage("add_one", |v| {
                let n = v.as_i64().unwrap();
                Ok(json!(n + 1))
            })
            .stage("double", |v| {
                let n = v.as_i64().unwrap();
                Ok(json!(n * 2))
            });

        let result = pipeline.execute(json!(5));
        assert_eq!(result.output, json!(12)); // (5+1)*2
        assert_eq!(result.stages_executed, vec!["add_one", "double"]);
        assert!(result.stages_skipped.is_empty());
        assert!(result.is_success());
    }

    #[test]
    fn test_pipeline_conditional_stage_runs() {
        let mut pipeline = Pipeline::new("cond");
        pipeline.conditional_stage(
            "double_if_positive",
            |v| {
                let n = v.as_i64().unwrap();
                Ok(json!(n * 2))
            },
            |v| v.as_i64().unwrap_or(0) > 0,
        );

        let result = pipeline.execute(json!(5));
        assert_eq!(result.output, json!(10));
        assert_eq!(result.stages_executed, vec!["double_if_positive"]);
        assert!(result.stages_skipped.is_empty());
    }

    #[test]
    fn test_pipeline_conditional_stage_skips() {
        let mut pipeline = Pipeline::new("cond");
        pipeline.conditional_stage(
            "double_if_positive",
            |v| {
                let n = v.as_i64().unwrap();
                Ok(json!(n * 2))
            },
            |v| v.as_i64().unwrap_or(0) > 0,
        );

        let result = pipeline.execute(json!(-3));
        assert_eq!(result.output, json!(-3)); // passthrough
        assert!(result.stages_executed.is_empty());
        assert_eq!(result.stages_skipped, vec!["double_if_positive"]);
    }

    #[test]
    fn test_pipeline_stage_count() {
        let mut pipeline = Pipeline::new("p");
        pipeline.stage("a", |v| Ok(v)).stage("b", |v| Ok(v));
        assert_eq!(pipeline.stage_count(), 2);
    }

    #[test]
    fn test_pipeline_stage_names() {
        let mut pipeline = Pipeline::new("p");
        pipeline.stage("alpha", |v| Ok(v)).stage("beta", |v| Ok(v));
        assert_eq!(pipeline.stage_names(), vec!["alpha", "beta"]);
    }

    #[test]
    fn test_pipeline_empty() {
        let pipeline = Pipeline::new("empty");
        let result = pipeline.execute(json!(42));
        assert_eq!(result.output, json!(42));
        assert!(result.stages_executed.is_empty());
        assert!(result.stages_skipped.is_empty());
        assert!(result.is_success());
    }

    #[test]
    fn test_pipeline_error_stops_execution() {
        let mut pipeline = Pipeline::new("err");
        pipeline
            .stage("good", |v| {
                let n = v.as_i64().unwrap();
                Ok(json!(n + 1))
            })
            .stage("bad", |_v| Err(RustChainError::Other("fail".into())))
            .stage("unreachable", |v| {
                let n = v.as_i64().unwrap();
                Ok(json!(n * 100))
            });

        let result = pipeline.execute(json!(1));
        assert_eq!(result.stages_executed, vec!["good", "bad"]);
        assert!(!result.is_success());
        assert_eq!(result.errors.len(), 1);
        assert_eq!(result.errors[0].0, "bad");
    }

    #[test]
    fn test_pipeline_error_handler_fallback() {
        let mut pipeline = Pipeline::new("fallback");
        let stage = PipelineStage::new("risky", |_v| Err(RustChainError::Other("oops".into())))
            .with_error_handler(|_| json!("recovered"));
        pipeline.add_stage(stage);
        pipeline.stage("after", |v| {
            Ok(json!(format!("got: {}", v.as_str().unwrap_or("?"))))
        });

        let result = pipeline.execute(json!(1));
        assert_eq!(result.output, json!("got: recovered"));
        assert_eq!(result.stages_executed, vec!["risky", "after"]);
        // The error was handled, so it shows up in errors list but pipeline continued
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_pipeline_dry_run() {
        let mut pipeline = Pipeline::new("dry");
        pipeline
            .stage("always", |v| Ok(v))
            .conditional_stage("only_positive", |v| Ok(v), |v| v.as_i64().unwrap_or(0) > 0)
            .conditional_stage("only_negative", |v| Ok(v), |v| v.as_i64().unwrap_or(0) < 0);

        let would_run = pipeline.dry_run(&json!(5));
        assert!(would_run.contains(&"always".to_string()));
        assert!(would_run.contains(&"only_positive".to_string()));
        assert!(!would_run.contains(&"only_negative".to_string()));
    }

    #[test]
    fn test_pipeline_dry_run_empty() {
        let pipeline = Pipeline::new("empty");
        let would_run = pipeline.dry_run(&json!(1));
        assert!(would_run.is_empty());
    }

    #[test]
    fn test_pipeline_all_stages_skipped() {
        let mut pipeline = Pipeline::new("all_skip");
        pipeline
            .conditional_stage("a", |v| Ok(v), |_| false)
            .conditional_stage("b", |v| Ok(v), |_| false);

        let result = pipeline.execute(json!("pass"));
        assert_eq!(result.output, json!("pass"));
        assert!(result.stages_executed.is_empty());
        assert_eq!(result.stages_skipped, vec!["a", "b"]);
        assert!(result.is_success());
    }

    // ── PipelineResult tests ─────────────────────────────────────────

    #[test]
    fn test_pipeline_result_to_json() {
        let result = PipelineResult {
            output: json!(42),
            stages_executed: vec!["a".into()],
            stages_skipped: vec!["b".into()],
            duration: Duration::from_millis(100),
            errors: vec![],
        };
        let j = result.to_json();
        assert_eq!(j["output"], json!(42));
        assert_eq!(j["success"], json!(true));
        assert_eq!(j["duration_ms"], json!(100));
    }

    #[test]
    fn test_pipeline_result_is_success_with_errors() {
        let result = PipelineResult {
            output: json!(null),
            stages_executed: vec![],
            stages_skipped: vec![],
            duration: Duration::ZERO,
            errors: vec![("s".into(), "e".into())],
        };
        assert!(!result.is_success());
    }

    // ── ParallelPipeline tests ───────────────────────────────────────

    #[test]
    fn test_parallel_pipeline_execution() {
        let mut branch_a = Pipeline::new("add");
        branch_a.stage("add_10", |v| {
            let n = v.as_i64().unwrap();
            Ok(json!(n + 10))
        });

        let mut branch_b = Pipeline::new("mul");
        branch_b.stage("mul_3", |v| {
            let n = v.as_i64().unwrap();
            Ok(json!(n * 3))
        });

        let mut pp = ParallelPipeline::new();
        pp.add_branch("adder", branch_a);
        pp.add_branch("multiplier", branch_b);

        let results = pp.execute(json!(5));
        assert_eq!(results["adder"].output, json!(15));
        assert_eq!(results["multiplier"].output, json!(15));
    }

    #[test]
    fn test_parallel_pipeline_branch_count() {
        let mut pp = ParallelPipeline::new();
        pp.add_branch("a", Pipeline::new("a"));
        pp.add_branch("b", Pipeline::new("b"));
        assert_eq!(pp.branch_count(), 2);
    }

    #[test]
    fn test_parallel_pipeline_branch_names() {
        let mut pp = ParallelPipeline::new();
        pp.add_branch("alpha", Pipeline::new("a"));
        pp.add_branch("beta", Pipeline::new("b"));
        assert_eq!(pp.branch_names(), vec!["alpha", "beta"]);
    }

    #[test]
    fn test_parallel_pipeline_empty() {
        let pp = ParallelPipeline::new();
        let results = pp.execute(json!(1));
        assert!(results.is_empty());
        assert_eq!(pp.branch_count(), 0);
    }

    // ── PipelineBuilder tests ────────────────────────────────────────

    #[test]
    fn test_builder_fluent_api() {
        let pipeline = PipelineBuilder::new("fluent")
            .then("add_one", |v| {
                let n = v.as_i64().unwrap();
                Ok(json!(n + 1))
            })
            .then("double", |v| {
                let n = v.as_i64().unwrap();
                Ok(json!(n * 2))
            })
            .build();

        let result = pipeline.execute(json!(4));
        assert_eq!(result.output, json!(10)); // (4+1)*2
    }

    #[test]
    fn test_builder_then_if() {
        let pipeline = PipelineBuilder::new("cond_builder")
            .then_if(
                "only_even",
                |v| {
                    let n = v.as_i64().unwrap();
                    Ok(json!(n * 10))
                },
                |v| v.as_i64().unwrap_or(0) % 2 == 0,
            )
            .build();

        let even_result = pipeline.execute(json!(4));
        assert_eq!(even_result.output, json!(40));

        let odd_result = pipeline.execute(json!(3));
        assert_eq!(odd_result.output, json!(3)); // skipped
    }

    #[test]
    fn test_builder_on_error() {
        let pipeline = PipelineBuilder::new("err_builder")
            .then("risky", |_v| Err(RustChainError::Other("failed".into())))
            .on_error("risky", |_| json!("fallback"))
            .then("after", |v| Ok(v))
            .build();

        let result = pipeline.execute(json!(1));
        assert_eq!(result.output, json!("fallback"));
    }

    #[test]
    fn test_builder_empty() {
        let pipeline = PipelineBuilder::new("empty").build();
        let result = pipeline.execute(json!(99));
        assert_eq!(result.output, json!(99));
        assert_eq!(pipeline.stage_count(), 0);
    }

    // ── PipelineMetrics tests ────────────────────────────────────────

    #[test]
    fn test_metrics_new() {
        let m = PipelineMetrics::new();
        assert_eq!(m.total_executions(), 0);
        assert_eq!(m.success_count(), 0);
        assert_eq!(m.failure_count(), 0);
        assert!(m.average_duration().is_none());
    }

    #[test]
    fn test_metrics_record_success() {
        let mut m = PipelineMetrics::new();
        let result = PipelineResult {
            output: json!(1),
            stages_executed: vec!["a".into(), "b".into()],
            stages_skipped: vec![],
            duration: Duration::from_millis(50),
            errors: vec![],
        };
        m.record(&result);
        assert_eq!(m.total_executions(), 1);
        assert_eq!(m.success_count(), 1);
        assert_eq!(m.failure_count(), 0);
    }

    #[test]
    fn test_metrics_record_failure() {
        let mut m = PipelineMetrics::new();
        let result = PipelineResult {
            output: json!(null),
            stages_executed: vec!["a".into()],
            stages_skipped: vec![],
            duration: Duration::from_millis(10),
            errors: vec![("a".into(), "err".into())],
        };
        m.record(&result);
        assert_eq!(m.failure_count(), 1);
        assert_eq!(m.success_count(), 0);
    }

    #[test]
    fn test_metrics_average_duration() {
        let mut m = PipelineMetrics::new();
        for ms in [100, 200, 300] {
            m.record(&PipelineResult {
                output: json!(null),
                stages_executed: vec![],
                stages_skipped: vec![],
                duration: Duration::from_millis(ms),
                errors: vec![],
            });
        }
        let avg = m.average_duration().unwrap();
        assert_eq!(avg.as_millis(), 200);
    }

    #[test]
    fn test_metrics_stage_execution_counts() {
        let mut m = PipelineMetrics::new();
        m.record(&PipelineResult {
            output: json!(null),
            stages_executed: vec!["a".into(), "b".into()],
            stages_skipped: vec![],
            duration: Duration::ZERO,
            errors: vec![],
        });
        m.record(&PipelineResult {
            output: json!(null),
            stages_executed: vec!["a".into()],
            stages_skipped: vec!["b".into()],
            duration: Duration::ZERO,
            errors: vec![],
        });
        let counts = m.stage_execution_counts();
        assert_eq!(counts["a"], 2);
        assert_eq!(counts["b"], 1);
    }

    #[test]
    fn test_metrics_to_json() {
        let mut m = PipelineMetrics::new();
        m.record(&PipelineResult {
            output: json!(null),
            stages_executed: vec!["x".into()],
            stages_skipped: vec![],
            duration: Duration::from_millis(42),
            errors: vec![],
        });
        let j = m.to_json();
        assert_eq!(j["total_executions"], json!(1));
        assert_eq!(j["successes"], json!(1));
        assert_eq!(j["failures"], json!(0));
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn test_all_stages_fail_with_handlers() {
        let mut pipeline = Pipeline::new("all_fail");
        pipeline.add_stage(
            PipelineStage::new("fail1", |_| Err(RustChainError::Other("e1".into())))
                .with_error_handler(|_| json!("recovered1")),
        );
        pipeline.add_stage(
            PipelineStage::new("fail2", |_| Err(RustChainError::Other("e2".into())))
                .with_error_handler(|_| json!("recovered2")),
        );

        let result = pipeline.execute(json!(0));
        assert_eq!(result.output, json!("recovered2"));
        assert_eq!(result.stages_executed.len(), 2);
        assert_eq!(result.errors.len(), 2);
    }

    #[test]
    fn test_pipeline_name() {
        let pipeline = Pipeline::new("my_pipeline");
        assert_eq!(pipeline.name(), "my_pipeline");
    }

    #[test]
    fn test_parallel_pipeline_default() {
        let pp = ParallelPipeline::default();
        assert_eq!(pp.branch_count(), 0);
    }

    #[test]
    fn test_metrics_default() {
        let m = PipelineMetrics::default();
        assert_eq!(m.total_executions(), 0);
    }
}
