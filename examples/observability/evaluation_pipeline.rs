//! Evaluation Pipeline Example
//!
//! Demonstrates the evaluation framework: ExactMatchEvaluator, ContainsEvaluator,
//! LLMJudge, and BatchEvaluator with aggregate metrics.

#[path = "../shared.rs"]
mod shared;

use std::sync::Arc;

use cognis::evaluation::{
    BatchEvaluator, ContainsEvaluator, EvalExample, EvaluationDataset, Evaluator,
    ExactMatchEvaluator, LLMJudge,
};

fn build_dataset() -> EvaluationDataset {
    let mut ds = EvaluationDataset::new();
    let examples = [
        ("What is the capital of France?", "Paris", "Paris"),
        ("What is the capital of Germany?", "Berlin", "Berlin"),
        ("What is the capital of Italy?", "Milan", "Rome"),
        ("What is the capital of Spain?", "Madrid", "Madrid"),
        ("What is the capital of Japan?", "Kyoto", "Tokyo"),
    ];
    for (input, output, reference) in examples {
        ds.add_example(EvalExample {
            input: input.to_string(),
            output: output.to_string(),
            reference: Some(reference.to_string()),
        });
    }
    ds
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = build_dataset();

    // 1. ExactMatchEvaluator
    let exact_match = Arc::new(ExactMatchEvaluator::new());
    let report = BatchEvaluator::new(exact_match.clone())
        .evaluate_dataset(&dataset)
        .await?;
    println!(
        "ExactMatch: mean={:.3}, min={:.1}, max={:.1}, count={}",
        report.aggregate.mean, report.aggregate.min, report.aggregate.max, report.aggregate.count
    );

    // 2. Case-insensitive exact match
    let ci = ExactMatchEvaluator::new().case_insensitive();
    let r1 = ci.evaluate("Q", "paris", Some("Paris")).await?;
    let r2 = ci.evaluate("Q", "london", Some("Paris")).await?;
    println!(
        "Case-insensitive: 'paris' vs 'Paris' = {:.1}, 'london' vs 'Paris' = {:.1}",
        r1.score, r2.score
    );

    // 3. ContainsEvaluator
    let contains = Arc::new(ContainsEvaluator::new());
    let mut cds = EvaluationDataset::new();
    for (output, reference) in [
        ("The capital of France is Paris.", "Paris"),
        ("I believe it is Berlin, Germany.", "Berlin"),
        ("The capital is Milano.", "Rome"),
    ] {
        cds.add_example(EvalExample {
            input: "Q".to_string(),
            output: output.to_string(),
            reference: Some(reference.to_string()),
        });
    }
    let creport = BatchEvaluator::new(contains).evaluate_dataset(&cds).await?;
    println!(
        "Contains: mean={:.3}, count={}",
        creport.aggregate.mean, creport.aggregate.count
    );

    // 4. LLMJudge
    let judge_model = shared::get_chat_model(vec![
        "8".into(),
        "9".into(),
        "3".into(),
        "7".into(),
        "2".into(),
    ]);
    let judge = Arc::new(
        LLMJudge::builder(judge_model)
            .scale(10.0)
            .criteria("accuracy and helpfulness")
            .build(),
    );
    let jreport = BatchEvaluator::new(judge)
        .evaluate_dataset(&dataset)
        .await?;
    println!(
        "LLMJudge: mean={:.3}, min={:.3}, max={:.3}, std_dev={:.3}",
        jreport.aggregate.mean,
        jreport.aggregate.min,
        jreport.aggregate.max,
        jreport.aggregate.std_dev,
    );

    // 5. Load dataset from JSON
    let json_data = r#"[
        {"input": "Translate hello", "output": "Bonjour", "reference": "Bonjour"},
        {"input": "Translate goodbye", "output": "Au revoir", "reference": "Au revoir"},
        {"input": "Translate thanks", "output": "Merci", "reference": "Merci"},
        {"input": "Translate please", "output": "Por favor", "reference": "S'il vous plait"}
    ]"#;
    let json_ds = EvaluationDataset::from_json(json_data)?;
    let ereport = BatchEvaluator::new(exact_match)
        .evaluate_dataset(&json_ds)
        .await?;
    let correct = ereport
        .results
        .iter()
        .filter(|r| r.result.score > 0.5)
        .count();
    println!(
        "JSON dataset: {:.0}% accuracy ({}/{})",
        ereport.aggregate.mean * 100.0,
        correct,
        ereport.aggregate.count,
    );

    Ok(())
}
