//! Evaluation framework for LLM outputs.
//!
//! Provides evaluators for measuring the quality of LLM-generated text,
//! including exact match, substring containment, LLM-as-judge scoring,
//! multi-criteria evaluation, and batch evaluation with aggregate metrics.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use rustchain_core::error::{Result, RustChainError};
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::Message;

// ─── EvalResult ───

/// The result of a single evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    /// Score from 0.0 to 1.0.
    pub score: f64,
    /// Optional reasoning or explanation for the score.
    pub reasoning: Option<String>,
    /// Additional metadata about the evaluation.
    pub metadata: HashMap<String, Value>,
}

impl EvalResult {
    /// Create a new `EvalResult` with the given score, clamped to [0.0, 1.0].
    pub fn new(score: f64) -> Self {
        Self {
            score: score.clamp(0.0, 1.0),
            reasoning: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the reasoning for this result.
    pub fn with_reasoning(mut self, reasoning: impl Into<String>) -> Self {
        self.reasoning = Some(reasoning.into());
        self
    }

    /// Insert a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

// ─── Evaluator trait ───

/// Trait for evaluating LLM outputs.
#[async_trait]
pub trait Evaluator: Send + Sync {
    /// Evaluate the given output against an optional reference.
    async fn evaluate(
        &self,
        input: &str,
        output: &str,
        reference: Option<&str>,
    ) -> Result<EvalResult>;

    /// The name of this evaluator.
    fn name(&self) -> &str;
}

// ─── ExactMatchEvaluator ───

/// Evaluator that scores 1.0 if output matches the reference exactly,
/// 0.0 otherwise. Optionally case-insensitive.
pub struct ExactMatchEvaluator {
    case_insensitive: bool,
}

impl ExactMatchEvaluator {
    /// Create a new case-sensitive exact match evaluator.
    pub fn new() -> Self {
        Self {
            case_insensitive: false,
        }
    }

    /// Enable case-insensitive comparison.
    pub fn case_insensitive(mut self) -> Self {
        self.case_insensitive = true;
        self
    }
}

impl Default for ExactMatchEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Evaluator for ExactMatchEvaluator {
    async fn evaluate(
        &self,
        _input: &str,
        output: &str,
        reference: Option<&str>,
    ) -> Result<EvalResult> {
        let reference = reference.unwrap_or("");
        let matches = if self.case_insensitive {
            output.to_lowercase() == reference.to_lowercase()
        } else {
            output == reference
        };
        let score = if matches { 1.0 } else { 0.0 };
        let reasoning = if matches {
            "Output exactly matches reference."
        } else {
            "Output does not match reference."
        };
        Ok(EvalResult::new(score).with_reasoning(reasoning))
    }

    fn name(&self) -> &str {
        "exact_match"
    }
}

// ─── ContainsEvaluator ───

/// Evaluator that scores 1.0 if the output contains the reference substring,
/// 0.0 otherwise.
pub struct ContainsEvaluator;

impl ContainsEvaluator {
    pub fn new() -> Self {
        Self
    }
}

impl Default for ContainsEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Evaluator for ContainsEvaluator {
    async fn evaluate(
        &self,
        _input: &str,
        output: &str,
        reference: Option<&str>,
    ) -> Result<EvalResult> {
        let reference = reference.unwrap_or("");
        let contains = output.contains(reference);
        let score = if contains { 1.0 } else { 0.0 };
        let reasoning = if contains {
            format!("Output contains the reference substring \"{}\".", reference)
        } else {
            format!(
                "Output does not contain the reference substring \"{}\".",
                reference
            )
        };
        Ok(EvalResult::new(score).with_reasoning(reasoning))
    }

    fn name(&self) -> &str {
        "contains"
    }
}

// ─── LLMJudge ───

/// Evaluator that uses an LLM (BaseChatModel) to judge output quality.
///
/// The LLM is prompted with a configurable template and asked to return
/// a numeric score. The response is parsed to extract the score.
pub struct LLMJudge {
    model: Arc<dyn BaseChatModel>,
    prompt_template: String,
    scale: f64,
    criteria: String,
}

impl LLMJudge {
    /// Create a new `LLMJudge` builder.
    pub fn builder(model: Arc<dyn BaseChatModel>) -> LLMJudgeBuilder {
        LLMJudgeBuilder {
            model,
            prompt_template: None,
            scale: 10.0,
            criteria: "helpfulness".to_string(),
        }
    }
}

/// Builder for configuring an `LLMJudge`.
pub struct LLMJudgeBuilder {
    model: Arc<dyn BaseChatModel>,
    prompt_template: Option<String>,
    scale: f64,
    criteria: String,
}

impl LLMJudgeBuilder {
    /// Set a custom prompt template.
    ///
    /// The template can use `{input}`, `{output}`, `{reference}`, `{criteria}`,
    /// and `{scale}` as placeholders.
    pub fn prompt_template(mut self, template: impl Into<String>) -> Self {
        self.prompt_template = Some(template.into());
        self
    }

    /// Set the scoring scale (default: 10.0). Scores are normalized to 0.0-1.0.
    pub fn scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }

    /// Set the evaluation criteria (default: "helpfulness").
    pub fn criteria(mut self, criteria: impl Into<String>) -> Self {
        self.criteria = criteria.into();
        self
    }

    /// Build the `LLMJudge`.
    pub fn build(self) -> LLMJudge {
        let default_template = format!(
            "You are an expert evaluator. Rate the following output on a scale of 0 to {scale} \
             based on the criterion: {{criteria}}.\n\n\
             Input: {{input}}\n\
             Output: {{output}}\n\
             Reference: {{reference}}\n\n\
             Respond with ONLY a numeric score between 0 and {scale}.",
            scale = self.scale
        );
        LLMJudge {
            model: self.model,
            prompt_template: self.prompt_template.unwrap_or(default_template),
            scale: self.scale,
            criteria: self.criteria,
        }
    }
}

#[async_trait]
impl Evaluator for LLMJudge {
    async fn evaluate(
        &self,
        input: &str,
        output: &str,
        reference: Option<&str>,
    ) -> Result<EvalResult> {
        let prompt = self
            .prompt_template
            .replace("{input}", input)
            .replace("{output}", output)
            .replace("{reference}", reference.unwrap_or("N/A"))
            .replace("{criteria}", &self.criteria)
            .replace("{scale}", &self.scale.to_string());

        let messages = vec![Message::human(&prompt)];
        let ai_msg = self.model.invoke_messages(&messages, None).await?;
        let response_text = ai_msg.base.content.text();

        let raw_score = parse_score(&response_text).ok_or_else(|| {
            RustChainError::Other(format!(
                "Could not parse score from LLM response: {}",
                response_text
            ))
        })?;

        let normalized = (raw_score / self.scale).clamp(0.0, 1.0);

        Ok(EvalResult::new(normalized)
            .with_reasoning(format!("LLM rated {} out of {}", raw_score, self.scale))
            .with_metadata("raw_score".to_string(), serde_json::json!(raw_score))
            .with_metadata("scale".to_string(), serde_json::json!(self.scale))
            .with_metadata("llm_response".to_string(), serde_json::json!(response_text)))
    }

    fn name(&self) -> &str {
        "llm_judge"
    }
}

/// Parse the first floating-point number from a string.
fn parse_score(text: &str) -> Option<f64> {
    let re = regex::Regex::new(r"(\d+(?:\.\d+)?)").ok()?;
    re.captures(text)?.get(1)?.as_str().parse::<f64>().ok()
}

// ─── CriteriaEvaluator ───

/// Evaluates output against multiple criteria using an LLM.
///
/// Each criterion is scored independently, then averaged for the overall score.
/// The per-criterion breakdown is included in the metadata.
pub struct CriteriaEvaluator {
    model: Arc<dyn BaseChatModel>,
    criteria: Vec<String>,
    scale: f64,
}

impl CriteriaEvaluator {
    /// Create a new `CriteriaEvaluator` with the given model and criteria.
    pub fn new(model: Arc<dyn BaseChatModel>, criteria: Vec<String>) -> Self {
        Self {
            model,
            criteria,
            scale: 10.0,
        }
    }

    /// Use a predefined set of criteria: helpfulness, relevance, coherence,
    /// correctness, and conciseness.
    pub fn default_criteria(model: Arc<dyn BaseChatModel>) -> Self {
        Self::new(
            model,
            vec![
                "helpfulness".to_string(),
                "relevance".to_string(),
                "coherence".to_string(),
                "correctness".to_string(),
                "conciseness".to_string(),
            ],
        )
    }

    /// Set the scoring scale (default: 10.0).
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale;
        self
    }
}

#[async_trait]
impl Evaluator for CriteriaEvaluator {
    async fn evaluate(
        &self,
        input: &str,
        output: &str,
        reference: Option<&str>,
    ) -> Result<EvalResult> {
        let mut scores: HashMap<String, f64> = HashMap::new();
        let mut total = 0.0;

        for criterion in &self.criteria {
            let judge = LLMJudge::builder(self.model.clone())
                .criteria(criterion.clone())
                .scale(self.scale)
                .build();
            let result = judge.evaluate(input, output, reference).await?;
            scores.insert(criterion.clone(), result.score);
            total += result.score;
        }

        let avg = if self.criteria.is_empty() {
            0.0
        } else {
            total / self.criteria.len() as f64
        };

        let criteria_breakdown: Value = serde_json::to_value(&scores).unwrap_or(Value::Null);

        Ok(EvalResult::new(avg)
            .with_reasoning(format!(
                "Average score across {} criteria: {:.3}",
                self.criteria.len(),
                avg
            ))
            .with_metadata("criteria_scores".to_string(), criteria_breakdown)
            .with_metadata("criteria".to_string(), serde_json::json!(self.criteria)))
    }

    fn name(&self) -> &str {
        "criteria"
    }
}

// ─── EvalExample & EvaluationDataset ───

/// A single evaluation example with input, output, and optional reference.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalExample {
    /// The input provided to the LLM.
    pub input: String,
    /// The output produced by the LLM.
    pub output: String,
    /// The expected or reference output.
    pub reference: Option<String>,
}

/// A collection of evaluation examples.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationDataset {
    pub examples: Vec<EvalExample>,
}

impl EvaluationDataset {
    /// Create a new empty dataset.
    pub fn new() -> Self {
        Self {
            examples: Vec::new(),
        }
    }

    /// Load a dataset from a JSON string.
    ///
    /// Expects an array of objects with `input`, `output`, and optional `reference` fields,
    /// or an object with an `examples` field containing such an array.
    pub fn from_json(json: &str) -> Result<Self> {
        // Try parsing as { "examples": [...] } first
        if let Ok(dataset) = serde_json::from_str::<EvaluationDataset>(json) {
            return Ok(dataset);
        }
        // Try parsing as a bare array
        let examples: Vec<EvalExample> = serde_json::from_str(json)
            .map_err(|e| RustChainError::Other(format!("Failed to parse dataset JSON: {}", e)))?;
        Ok(Self { examples })
    }

    /// Add an example to the dataset.
    pub fn add_example(&mut self, example: EvalExample) {
        self.examples.push(example);
    }

    /// Return the number of examples.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Return true if the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }
}

impl Default for EvaluationDataset {
    fn default() -> Self {
        Self::new()
    }
}

// ─── EvaluationReport ───

/// Per-example result in an evaluation report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExampleResult {
    /// The input for this example.
    pub input: String,
    /// The output for this example.
    pub output: String,
    /// The reference for this example.
    pub reference: Option<String>,
    /// The evaluation result.
    pub result: EvalResult,
}

/// Aggregate statistics for a batch evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateMetrics {
    /// Mean score across all examples.
    pub mean: f64,
    /// Minimum score.
    pub min: f64,
    /// Maximum score.
    pub max: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Number of examples evaluated.
    pub count: usize,
}

/// Report from a batch evaluation, containing per-example and aggregate results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationReport {
    /// Name of the evaluator used.
    pub evaluator_name: String,
    /// Per-example results.
    pub results: Vec<ExampleResult>,
    /// Aggregate metrics.
    pub aggregate: AggregateMetrics,
}

// ─── BatchEvaluator ───

/// Runs an evaluator across multiple examples and produces aggregate metrics.
pub struct BatchEvaluator {
    evaluator: Arc<dyn Evaluator>,
}

impl BatchEvaluator {
    /// Create a new `BatchEvaluator` wrapping the given evaluator.
    pub fn new(evaluator: Arc<dyn Evaluator>) -> Self {
        Self { evaluator }
    }

    /// Evaluate all examples in the dataset and produce a report.
    pub async fn evaluate_dataset(&self, dataset: &EvaluationDataset) -> Result<EvaluationReport> {
        let mut results = Vec::with_capacity(dataset.examples.len());

        for example in &dataset.examples {
            let eval_result = self
                .evaluator
                .evaluate(
                    &example.input,
                    &example.output,
                    example.reference.as_deref(),
                )
                .await?;
            results.push(ExampleResult {
                input: example.input.clone(),
                output: example.output.clone(),
                reference: example.reference.clone(),
                result: eval_result,
            });
        }

        let aggregate = compute_aggregate(&results);

        Ok(EvaluationReport {
            evaluator_name: self.evaluator.name().to_string(),
            results,
            aggregate,
        })
    }
}

/// Compute aggregate metrics from a set of example results.
fn compute_aggregate(results: &[ExampleResult]) -> AggregateMetrics {
    if results.is_empty() {
        return AggregateMetrics {
            mean: 0.0,
            min: 0.0,
            max: 0.0,
            std_dev: 0.0,
            count: 0,
        };
    }

    let scores: Vec<f64> = results.iter().map(|r| r.result.score).collect();
    let count = scores.len();
    let sum: f64 = scores.iter().sum();
    let mean = sum / count as f64;
    let min = scores.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / count as f64;
    let std_dev = variance.sqrt();

    AggregateMetrics {
        mean,
        min,
        max,
        std_dev,
        count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::language_models::fake::FakeListChatModel;
    use serde_json::json;

    // ── ExactMatch correct ──

    #[tokio::test]
    async fn test_exact_match_correct() {
        let eval = ExactMatchEvaluator::new();
        let result = eval
            .evaluate("question", "Paris", Some("Paris"))
            .await
            .unwrap();
        assert!((result.score - 1.0).abs() < f64::EPSILON);
    }

    // ── ExactMatch incorrect ──

    #[tokio::test]
    async fn test_exact_match_incorrect() {
        let eval = ExactMatchEvaluator::new();
        let result = eval
            .evaluate("question", "London", Some("Paris"))
            .await
            .unwrap();
        assert!((result.score - 0.0).abs() < f64::EPSILON);
    }

    // ── ExactMatch case insensitive ──

    #[tokio::test]
    async fn test_exact_match_case_insensitive() {
        let eval = ExactMatchEvaluator::new().case_insensitive();
        let result = eval
            .evaluate("question", "paris", Some("Paris"))
            .await
            .unwrap();
        assert!((result.score - 1.0).abs() < f64::EPSILON);

        // Case-sensitive should fail
        let eval_sensitive = ExactMatchEvaluator::new();
        let result2 = eval_sensitive
            .evaluate("question", "paris", Some("Paris"))
            .await
            .unwrap();
        assert!((result2.score - 0.0).abs() < f64::EPSILON);
    }

    // ── Contains evaluator ──

    #[tokio::test]
    async fn test_contains_evaluator() {
        let eval = ContainsEvaluator::new();
        let result = eval
            .evaluate("question", "The capital of France is Paris.", Some("Paris"))
            .await
            .unwrap();
        assert!((result.score - 1.0).abs() < f64::EPSILON);

        let result2 = eval
            .evaluate("question", "The capital is London.", Some("Paris"))
            .await
            .unwrap();
        assert!((result2.score - 0.0).abs() < f64::EPSILON);
    }

    // ── LLMJudge basic scoring ──

    #[tokio::test]
    async fn test_llm_judge_basic_scoring() {
        // FakeListChatModel returns "7" which should parse as 7/10 = 0.7
        let model = Arc::new(FakeListChatModel::new(vec!["7".to_string()]));
        let judge = LLMJudge::builder(model)
            .scale(10.0)
            .criteria("helpfulness")
            .build();
        let result = judge
            .evaluate("What is 2+2?", "4", Some("4"))
            .await
            .unwrap();
        assert!((result.score - 0.7).abs() < f64::EPSILON);
        assert!(result.metadata.contains_key("raw_score"));
        assert_eq!(result.metadata["raw_score"], json!(7.0));
    }

    // ── LLMJudge custom criteria ──

    #[tokio::test]
    async fn test_llm_judge_custom_criteria() {
        let model = Arc::new(FakeListChatModel::new(vec!["9.5".to_string()]));
        let judge = LLMJudge::builder(model)
            .scale(10.0)
            .criteria("accuracy")
            .build();
        let result = judge
            .evaluate("Translate hello", "Bonjour", Some("Bonjour"))
            .await
            .unwrap();
        assert!((result.score - 0.95).abs() < f64::EPSILON);
    }

    // ── CriteriaEvaluator multi-criteria ──

    #[tokio::test]
    async fn test_criteria_evaluator_multi_criteria() {
        // The model will return "8" for each criterion call.
        // With 3 criteria, it will be called 3 times, each yielding 8/10 = 0.8.
        let model = Arc::new(FakeListChatModel::new(vec!["8".to_string()]));
        let criteria_eval = CriteriaEvaluator::new(
            model,
            vec![
                "helpfulness".to_string(),
                "relevance".to_string(),
                "coherence".to_string(),
            ],
        );
        let result = criteria_eval
            .evaluate("question", "answer", Some("reference"))
            .await
            .unwrap();
        assert!((result.score - 0.8).abs() < f64::EPSILON);
        assert!(result.metadata.contains_key("criteria_scores"));
    }

    // ── BatchEvaluator aggregate metrics ──

    #[tokio::test]
    async fn test_batch_evaluator_aggregate_metrics() {
        let eval = Arc::new(ExactMatchEvaluator::new());
        let batch = BatchEvaluator::new(eval);
        let dataset = EvaluationDataset {
            examples: vec![
                EvalExample {
                    input: "q1".to_string(),
                    output: "Paris".to_string(),
                    reference: Some("Paris".to_string()),
                },
                EvalExample {
                    input: "q2".to_string(),
                    output: "London".to_string(),
                    reference: Some("Paris".to_string()),
                },
                EvalExample {
                    input: "q3".to_string(),
                    output: "Berlin".to_string(),
                    reference: Some("Berlin".to_string()),
                },
                EvalExample {
                    input: "q4".to_string(),
                    output: "Madrid".to_string(),
                    reference: Some("Rome".to_string()),
                },
            ],
        };

        let report = batch.evaluate_dataset(&dataset).await.unwrap();
        assert_eq!(report.results.len(), 4);
        assert_eq!(report.aggregate.count, 4);
        // 2 correct out of 4 => mean = 0.5
        assert!((report.aggregate.mean - 0.5).abs() < f64::EPSILON);
        assert!((report.aggregate.min - 0.0).abs() < f64::EPSILON);
        assert!((report.aggregate.max - 1.0).abs() < f64::EPSILON);
        // std_dev of [1,0,1,0] = 0.5
        assert!((report.aggregate.std_dev - 0.5).abs() < f64::EPSILON);
    }

    // ── EvaluationDataset from JSON ──

    #[tokio::test]
    async fn test_evaluation_dataset_from_json() {
        let json = r#"[
            {"input": "q1", "output": "a1", "reference": "r1"},
            {"input": "q2", "output": "a2", "reference": null},
            {"input": "q3", "output": "a3"}
        ]"#;
        let dataset = EvaluationDataset::from_json(json).unwrap();
        assert_eq!(dataset.len(), 3);
        assert_eq!(dataset.examples[0].input, "q1");
        assert_eq!(dataset.examples[0].reference, Some("r1".to_string()));
        assert!(dataset.examples[2].reference.is_none());
    }

    // ── Score bounds (0-1) ──

    #[tokio::test]
    async fn test_score_bounds() {
        // Ensure EvalResult clamps scores to [0.0, 1.0].
        let result = EvalResult::new(1.5);
        assert!((result.score - 1.0).abs() < f64::EPSILON);

        let result2 = EvalResult::new(-0.5);
        assert!((result2.score - 0.0).abs() < f64::EPSILON);

        let result3 = EvalResult::new(0.75);
        assert!((result3.score - 0.75).abs() < f64::EPSILON);
    }

    // ── Custom prompt template ──

    #[tokio::test]
    async fn test_llm_judge_custom_prompt_template() {
        let model = Arc::new(FakeListChatModel::new(vec!["5".to_string()]));
        let judge = LLMJudge::builder(model)
            .prompt_template("Score the output: {output}. Scale: 0-{scale}.")
            .scale(5.0)
            .build();
        let result = judge.evaluate("input", "output text", None).await.unwrap();
        // 5/5 = 1.0
        assert!((result.score - 1.0).abs() < f64::EPSILON);
    }

    // ── Empty dataset ──

    #[tokio::test]
    async fn test_empty_dataset() {
        let eval = Arc::new(ExactMatchEvaluator::new());
        let batch = BatchEvaluator::new(eval);
        let dataset = EvaluationDataset::new();
        assert!(dataset.is_empty());

        let report = batch.evaluate_dataset(&dataset).await.unwrap();
        assert_eq!(report.results.len(), 0);
        assert_eq!(report.aggregate.count, 0);
        assert!((report.aggregate.mean - 0.0).abs() < f64::EPSILON);
    }

    // ── Report generation ──

    #[tokio::test]
    async fn test_report_generation() {
        let eval = Arc::new(ContainsEvaluator::new());
        let batch = BatchEvaluator::new(eval);
        let dataset = EvaluationDataset {
            examples: vec![
                EvalExample {
                    input: "q1".to_string(),
                    output: "The answer is Paris".to_string(),
                    reference: Some("Paris".to_string()),
                },
                EvalExample {
                    input: "q2".to_string(),
                    output: "I think it is Berlin".to_string(),
                    reference: Some("Berlin".to_string()),
                },
            ],
        };

        let report = batch.evaluate_dataset(&dataset).await.unwrap();
        assert_eq!(report.evaluator_name, "contains");
        assert_eq!(report.results.len(), 2);
        assert_eq!(report.aggregate.count, 2);
        assert!((report.aggregate.mean - 1.0).abs() < f64::EPSILON);

        // Verify report can be serialized to JSON
        let json = serde_json::to_value(&report).unwrap();
        assert!(json.get("evaluator_name").is_some());
        assert!(json.get("results").is_some());
        assert!(json.get("aggregate").is_some());
    }

    // ── Dataset from JSON with examples wrapper ──

    #[tokio::test]
    async fn test_dataset_from_json_with_examples_key() {
        let json = r#"{"examples": [
            {"input": "q1", "output": "a1", "reference": "r1"}
        ]}"#;
        let dataset = EvaluationDataset::from_json(json).unwrap();
        assert_eq!(dataset.len(), 1);
    }

    // ── LLMJudge with high scale ──

    #[tokio::test]
    async fn test_llm_judge_scale_normalization() {
        let model = Arc::new(FakeListChatModel::new(vec!["50".to_string()]));
        let judge = LLMJudge::builder(model).scale(100.0).build();
        let result = judge
            .evaluate("input", "output", Some("ref"))
            .await
            .unwrap();
        // 50/100 = 0.5
        assert!((result.score - 0.5).abs() < f64::EPSILON);
    }
}
