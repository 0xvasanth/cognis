//! Evaluation Pipeline Example
//!
//! Demonstrates the evaluation framework for measuring LLM output quality.
//! Shows ExactMatchEvaluator, ContainsEvaluator, LLMJudge, and BatchEvaluator
//! with aggregate metrics.
//!
//! No API keys required -- uses FakeListChatModel.
//!
//! Run with: cargo run -p rustchain-examples --example evaluation_pipeline

use std::sync::Arc;

use rustchain::evaluation::{
    BatchEvaluator, ContainsEvaluator, EvalExample, EvaluationDataset, Evaluator,
    ExactMatchEvaluator, LLMJudge,
};
use rustchain_core::language_models::FakeListChatModel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Evaluation Pipeline Example ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Create an evaluation dataset
    // -------------------------------------------------------------------------
    println!("--- Step 1: Creating evaluation dataset ---\n");

    let mut dataset = EvaluationDataset::new();

    dataset.add_example(EvalExample {
        input: "What is the capital of France?".to_string(),
        output: "Paris".to_string(),
        reference: Some("Paris".to_string()),
    });
    dataset.add_example(EvalExample {
        input: "What is the capital of Germany?".to_string(),
        output: "Berlin".to_string(),
        reference: Some("Berlin".to_string()),
    });
    dataset.add_example(EvalExample {
        input: "What is the capital of Italy?".to_string(),
        output: "Milan".to_string(),
        reference: Some("Rome".to_string()),
    });
    dataset.add_example(EvalExample {
        input: "What is the capital of Spain?".to_string(),
        output: "Madrid".to_string(),
        reference: Some("Madrid".to_string()),
    });
    dataset.add_example(EvalExample {
        input: "What is the capital of Japan?".to_string(),
        output: "Kyoto".to_string(),
        reference: Some("Tokyo".to_string()),
    });

    println!("  Created dataset with {} examples", dataset.len());
    for (i, ex) in dataset.examples.iter().enumerate() {
        let correct = ex.output == ex.reference.as_deref().unwrap_or("");
        let marker = if correct { "correct" } else { "wrong" };
        println!(
            "    [{}] Q: {} | A: {} | Ref: {} ({})",
            i + 1,
            ex.input,
            ex.output,
            ex.reference.as_deref().unwrap_or("N/A"),
            marker
        );
    }
    println!();

    // -------------------------------------------------------------------------
    // Step 2: Run ExactMatchEvaluator
    // -------------------------------------------------------------------------
    println!("--- Step 2: ExactMatchEvaluator ---\n");

    let exact_match = Arc::new(ExactMatchEvaluator::new());
    let batch_exact = BatchEvaluator::new(exact_match.clone());

    let report = batch_exact.evaluate_dataset(&dataset).await?;

    println!("  Evaluator: {}", report.evaluator_name);
    for result in &report.results {
        println!(
            "    Q: {} | Score: {:.1} | {}",
            result.input,
            result.result.score,
            result.result.reasoning.as_deref().unwrap_or("")
        );
    }
    println!("\n  Aggregate Metrics:");
    println!("    Mean:    {:.3}", report.aggregate.mean);
    println!("    Min:     {:.1}", report.aggregate.min);
    println!("    Max:     {:.1}", report.aggregate.max);
    println!("    Std Dev: {:.3}", report.aggregate.std_dev);
    println!("    Count:   {}", report.aggregate.count);
    println!();

    // -------------------------------------------------------------------------
    // Step 3: Run case-insensitive ExactMatchEvaluator
    // -------------------------------------------------------------------------
    println!("--- Step 3: Case-insensitive ExactMatchEvaluator ---\n");

    let case_insensitive = ExactMatchEvaluator::new().case_insensitive();

    // Test with a case mismatch example.
    let result = case_insensitive
        .evaluate("What is the capital of France?", "paris", Some("Paris"))
        .await?;
    println!(
        "  \"paris\" vs \"Paris\" (case-insensitive): score={:.1} | {}",
        result.score,
        result.reasoning.as_deref().unwrap_or("")
    );

    let result = case_insensitive
        .evaluate("What is the capital of France?", "london", Some("Paris"))
        .await?;
    println!(
        "  \"london\" vs \"Paris\" (case-insensitive): score={:.1} | {}",
        result.score,
        result.reasoning.as_deref().unwrap_or("")
    );
    println!();

    // -------------------------------------------------------------------------
    // Step 4: Run ContainsEvaluator
    // -------------------------------------------------------------------------
    println!("--- Step 4: ContainsEvaluator ---\n");

    let contains = Arc::new(ContainsEvaluator::new());
    let batch_contains = BatchEvaluator::new(contains);

    // Use a dataset where outputs contain the reference as a substring.
    let mut contains_dataset = EvaluationDataset::new();
    contains_dataset.add_example(EvalExample {
        input: "Capital of France?".to_string(),
        output: "The capital of France is Paris.".to_string(),
        reference: Some("Paris".to_string()),
    });
    contains_dataset.add_example(EvalExample {
        input: "Capital of Germany?".to_string(),
        output: "I believe it is Berlin, Germany.".to_string(),
        reference: Some("Berlin".to_string()),
    });
    contains_dataset.add_example(EvalExample {
        input: "Capital of Italy?".to_string(),
        output: "The capital is Milano.".to_string(),
        reference: Some("Rome".to_string()),
    });

    let contains_report = batch_contains
        .evaluate_dataset(&contains_dataset)
        .await?;

    println!("  Evaluator: {}", contains_report.evaluator_name);
    for result in &contains_report.results {
        println!(
            "    Output: \"{}\" contains \"{}\"? Score: {:.1}",
            result.output,
            result.reference.as_deref().unwrap_or(""),
            result.result.score
        );
    }
    println!(
        "\n  Aggregate: mean={:.3}, count={}",
        contains_report.aggregate.mean, contains_report.aggregate.count
    );
    println!();

    // -------------------------------------------------------------------------
    // Step 5: Run LLMJudge
    // -------------------------------------------------------------------------
    println!("--- Step 5: LLMJudge ---\n");

    // The fake model returns numeric scores that the LLMJudge parses.
    let judge_model = Arc::new(FakeListChatModel::new(vec![
        "8".to_string(),   // 8/10 = 0.8
        "9".to_string(),   // 9/10 = 0.9
        "3".to_string(),   // 3/10 = 0.3
        "7".to_string(),   // 7/10 = 0.7
        "2".to_string(),   // 2/10 = 0.2
    ]));

    let judge = Arc::new(
        LLMJudge::builder(judge_model)
            .scale(10.0)
            .criteria("accuracy and helpfulness")
            .build(),
    );

    let batch_judge = BatchEvaluator::new(judge);
    let judge_report = batch_judge.evaluate_dataset(&dataset).await?;

    println!("  Evaluator: {}", judge_report.evaluator_name);
    for result in &judge_report.results {
        let raw = result
            .result
            .metadata
            .get("raw_score")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        println!(
            "    Q: {} | Score: {:.2} (raw: {}/10) | {}",
            result.input,
            result.result.score,
            raw,
            result.result.reasoning.as_deref().unwrap_or("")
        );
    }
    println!("\n  Aggregate Metrics:");
    println!("    Mean:    {:.3}", judge_report.aggregate.mean);
    println!("    Min:     {:.3}", judge_report.aggregate.min);
    println!("    Max:     {:.3}", judge_report.aggregate.max);
    println!("    Std Dev: {:.3}", judge_report.aggregate.std_dev);
    println!("    Count:   {}", judge_report.aggregate.count);
    println!();

    // -------------------------------------------------------------------------
    // Step 6: Load dataset from JSON
    // -------------------------------------------------------------------------
    println!("--- Step 6: Loading dataset from JSON ---\n");

    let json_data = r#"[
        {"input": "Translate hello", "output": "Bonjour", "reference": "Bonjour"},
        {"input": "Translate goodbye", "output": "Au revoir", "reference": "Au revoir"},
        {"input": "Translate thanks", "output": "Merci", "reference": "Merci"},
        {"input": "Translate please", "output": "Por favor", "reference": "S'il vous plait"}
    ]"#;

    let json_dataset = EvaluationDataset::from_json(json_data)?;
    println!("  Loaded {} examples from JSON", json_dataset.len());

    let exact_report = BatchEvaluator::new(exact_match)
        .evaluate_dataset(&json_dataset)
        .await?;

    println!("  ExactMatch results:");
    for result in &exact_report.results {
        println!(
            "    \"{}\" == \"{}\"? => {:.1}",
            result.output,
            result.reference.as_deref().unwrap_or(""),
            result.result.score
        );
    }
    println!(
        "  Overall accuracy: {:.0}% ({}/{})",
        exact_report.aggregate.mean * 100.0,
        exact_report
            .results
            .iter()
            .filter(|r| r.result.score > 0.5)
            .count(),
        exact_report.aggregate.count
    );

    // -------------------------------------------------------------------------
    // Step 7: Serialize report to JSON
    // -------------------------------------------------------------------------
    println!("\n--- Step 7: Serialized report ---\n");

    let report_json = serde_json::to_string_pretty(&exact_report)?;
    // Print first 500 chars of the JSON report.
    let preview = &report_json[..report_json.len().min(500)];
    println!("{preview}...");

    println!("\nDone!");
    Ok(())
}
