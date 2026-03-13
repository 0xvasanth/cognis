//! Evaluation Framework Example
//!
//! Demonstrates the DeepAgents evaluation and benchmarking framework for
//! scoring outputs against expected values using a variety of built-in metrics.
//!
//! Features shown:
//! - EvalScore creation and passing checks
//! - ExactMatchMetric evaluation
//! - ContainsMetric with case-insensitive matching
//! - NumericDistanceMetric with tolerance
//! - JsonSimilarityMetric structural comparison
//! - EvalCase building with tags and metadata
//! - EvalSuite running with multiple metrics
//! - EvalReport statistics and failure detection
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example evaluation_framework`

mod shared;
use cognisagent::evaluation::{
    ContainsMetric, EvalCase, EvalMetric, EvalScore, EvalSuite, ExactMatchMetric,
    JsonSimilarityMetric, NumericDistanceMetric,
};
use serde_json::json;

fn main() {
    println!("=== Evaluation Framework Example ===\n");

    // -----------------------------------------------------------------------
    // 1. EvalScore Creation and Passing Checks
    // -----------------------------------------------------------------------
    println!("--- 1. EvalScore Creation and Passing Checks ---");
    println!("EvalScore holds a metric result between 0.0 (worst) and 1.0 (best).\n");

    let score = EvalScore::new("my_metric", 0.85);
    println!("Score: {} = {:.2}", score.metric_name, score.score);
    println!("  Passes at 0.8 threshold? {}", score.is_passing(0.8));
    println!("  Passes at 0.9 threshold? {}", score.is_passing(0.9));

    // Scores are clamped to [0.0, 1.0]
    let clamped = EvalScore::new("clamped", 1.5);
    println!(
        "\nClamped score (input 1.5): {:.2} (clamped to 1.0)",
        clamped.score
    );

    // Scores can carry additional detail
    let detailed = EvalScore::new("detailed", 0.75)
        .with_detail("reason", json!("partial match"))
        .with_detail("matched_tokens", json!(15));
    println!("\nDetailed score JSON:");
    println!(
        "  {}",
        serde_json::to_string_pretty(&detailed.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 2. ExactMatchMetric
    // -----------------------------------------------------------------------
    println!("\n--- 2. ExactMatchMetric ---");
    println!("Checks whether the actual output is exactly equal to the expected output.\n");

    let exact = ExactMatchMetric;
    println!("Metric: '{}' - {}", exact.name(), exact.description());

    let score_match = exact.evaluate(&json!("hello"), &json!("hello"));
    println!(
        "\n  \"hello\" vs \"hello\" => {:.1} ({})",
        score_match.score,
        if score_match.score == 1.0 {
            "match"
        } else {
            "no match"
        }
    );

    let score_mismatch = exact.evaluate(&json!("hello"), &json!("world"));
    println!(
        "  \"hello\" vs \"world\" => {:.1} ({})",
        score_mismatch.score,
        if score_mismatch.score == 1.0 {
            "match"
        } else {
            "no match"
        }
    );

    let score_type = exact.evaluate(&json!("42"), &json!(42));
    println!(
        "  \"42\" (string) vs 42 (number) => {:.1} (type mismatch)",
        score_type.score
    );

    let score_obj = exact.evaluate(&json!({"a": 1, "b": 2}), &json!({"a": 1, "b": 2}));
    println!(
        "  {{\"a\":1,\"b\":2}} vs {{\"a\":1,\"b\":2}} => {:.1} (deep equality)",
        score_obj.score
    );

    // -----------------------------------------------------------------------
    // 3. ContainsMetric with Case-Insensitive Matching
    // -----------------------------------------------------------------------
    println!("\n--- 3. ContainsMetric ---");
    println!("Checks whether the actual output contains the expected value as a substring.\n");

    // Case-sensitive (default)
    let contains_cs = ContainsMetric::new(false);
    println!(
        "Metric: '{}' (case_sensitive) - {}",
        contains_cs.name(),
        contains_cs.description()
    );

    let s1 = contains_cs.evaluate(&json!("world"), &json!("hello world"));
    println!(
        "\n  expected=\"world\" in actual=\"hello world\" => {:.1}",
        s1.score
    );

    let s2 = contains_cs.evaluate(&json!("WORLD"), &json!("hello world"));
    println!(
        "  expected=\"WORLD\" in actual=\"hello world\" (case-sensitive) => {:.1}",
        s2.score
    );

    // Case-insensitive
    let contains_ci = ContainsMetric::new(true);
    println!("\nMetric: '{}' (case_insensitive)", contains_ci.name());

    let s3 = contains_ci.evaluate(&json!("WORLD"), &json!("hello world"));
    println!(
        "  expected=\"WORLD\" in actual=\"hello world\" (case-insensitive) => {:.1}",
        s3.score
    );

    let s4 = contains_ci.evaluate(&json!("missing"), &json!("hello world"));
    println!(
        "  expected=\"missing\" in actual=\"hello world\" => {:.1}",
        s4.score
    );

    // Non-string values are converted to their string representation
    let s5 = contains_cs.evaluate(&json!("42"), &json!("[1, 42, 3]"));
    println!(
        "\n  expected=\"42\" in actual=\"[1, 42, 3]\" (stringified) => {:.1}",
        s5.score
    );

    // -----------------------------------------------------------------------
    // 4. NumericDistanceMetric with Tolerance
    // -----------------------------------------------------------------------
    println!("\n--- 4. NumericDistanceMetric ---");
    println!("Scores based on distance between numeric values within a tolerance.\n");

    let numeric = NumericDistanceMetric::new(10.0);
    println!(
        "Metric: '{}' (tolerance=10.0) - {}",
        numeric.name(),
        numeric.description()
    );

    let n1 = numeric.evaluate(&json!(50.0), &json!(50.0));
    println!(
        "\n  expected=50.0, actual=50.0 => {:.2} (exact match)",
        n1.score
    );

    let n2 = numeric.evaluate(&json!(50.0), &json!(55.0));
    println!(
        "  expected=50.0, actual=55.0 => {:.2} (distance=5, within tolerance)",
        n2.score
    );

    let n3 = numeric.evaluate(&json!(50.0), &json!(60.0));
    println!(
        "  expected=50.0, actual=60.0 => {:.2} (distance=10, at tolerance boundary)",
        n3.score
    );

    let n4 = numeric.evaluate(&json!(50.0), &json!(70.0));
    println!(
        "  expected=50.0, actual=70.0 => {:.2} (distance=20, beyond tolerance)",
        n4.score
    );

    // String numbers are also supported
    let n5 = numeric.evaluate(&json!("3.14"), &json!("3.14"));
    println!(
        "  expected=\"3.14\", actual=\"3.14\" => {:.2} (string-encoded numbers)",
        n5.score
    );

    // Non-numeric values score 0.0
    let n6 = numeric.evaluate(&json!("abc"), &json!(5.0));
    println!(
        "  expected=\"abc\", actual=5.0 => {:.2} (non-numeric returns 0.0)",
        n6.score
    );

    // -----------------------------------------------------------------------
    // 5. JsonSimilarityMetric Structural Comparison
    // -----------------------------------------------------------------------
    println!("\n--- 5. JsonSimilarityMetric ---");
    println!("Compares JSON structures and scores based on matching keys/values.\n");

    let json_sim = JsonSimilarityMetric;
    println!("Metric: '{}' - {}", json_sim.name(), json_sim.description());

    // Identical objects
    let j1 = json_sim.evaluate(
        &json!({"name": "Alice", "age": 30}),
        &json!({"name": "Alice", "age": 30}),
    );
    println!("\n  Identical objects => {:.2}", j1.score);

    // Partial match (one key matches, one differs)
    let j2 = json_sim.evaluate(
        &json!({"name": "Alice", "age": 30}),
        &json!({"name": "Alice", "age": 25}),
    );
    println!("  Partial match (1 of 2 keys) => {:.2}", j2.score);

    // Missing keys
    let j3 = json_sim.evaluate(
        &json!({"name": "Alice", "age": 30, "city": "NYC"}),
        &json!({"name": "Alice"}),
    );
    println!("  Missing keys (1 of 3 present) => {:.2}", j3.score);

    // Nested object comparison
    let j4 = json_sim.evaluate(
        &json!({"user": {"name": "Alice", "age": 30}}),
        &json!({"user": {"name": "Alice", "age": 99}}),
    );
    println!("  Nested partial match => {:.2}", j4.score);

    // Array comparison
    let j5 = json_sim.evaluate(&json!([1, 2, 3, 4]), &json!([1, 2, 3, 99]));
    println!("  Array (3 of 4 elements match) => {:.2}", j5.score);

    // Scalar comparison
    let j6 = json_sim.evaluate(&json!(42), &json!(42));
    let j7 = json_sim.evaluate(&json!(42), &json!(99));
    println!("  Scalar match (42 vs 42) => {:.2}", j6.score);
    println!("  Scalar mismatch (42 vs 99) => {:.2}", j7.score);

    // -----------------------------------------------------------------------
    // 6. EvalCase Building with Tags
    // -----------------------------------------------------------------------
    println!("\n--- 6. EvalCase Building ---");
    println!("Build evaluation cases with inputs, expected outputs, tags, and metadata.\n");

    let case1 = EvalCase::builder("math_addition")
        .input(json!("What is 2+2?"))
        .expected_output(json!("4"))
        .tag("math")
        .tag("arithmetic")
        .tag("easy")
        .metadata("difficulty", json!("beginner"))
        .metadata("source", json!("textbook"))
        .build();

    println!("Case: '{}'", case1.name);
    println!("  Input: {}", case1.input);
    println!("  Expected: {}", case1.expected_output);
    println!("  Tags: {:?}", case1.tags);
    println!(
        "  Metadata keys: {:?}",
        case1.metadata.keys().collect::<Vec<_>>()
    );

    let case2 = EvalCase::builder("json_extraction")
        .input(json!({"text": "Extract the name and age from: Alice is 30 years old"}))
        .expected_output(json!({"name": "Alice", "age": 30}))
        .tag("extraction")
        .tag("structured")
        .build();

    println!("\nCase: '{}'", case2.name);
    println!("  Tags: {:?}", case2.tags);
    println!("\nCase JSON:");
    println!(
        "  {}",
        serde_json::to_string_pretty(&case1.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 7. EvalSuite Running with Multiple Metrics
    // -----------------------------------------------------------------------
    println!("\n--- 7. EvalSuite with Multiple Metrics ---");
    println!("Run a suite of test cases against multiple metrics simultaneously.\n");

    let mut suite = EvalSuite::new("qa_evaluation");

    // Add cases that will pass
    suite.add_case(
        EvalCase::builder("exact_answer")
            .input(json!("hello world"))
            .expected_output(json!("hello world"))
            .tag("text")
            .build(),
    );

    suite.add_case(
        EvalCase::builder("contains_answer")
            .input(json!("The answer is 42, as we all know."))
            .expected_output(json!("42"))
            .tag("numeric")
            .build(),
    );

    // Add a case that will fail all metrics
    suite.add_case(
        EvalCase::builder("wrong_answer")
            .input(json!("completely wrong"))
            .expected_output(json!("correct answer"))
            .tag("text")
            .build(),
    );

    // Add a partial match case
    suite.add_case(
        EvalCase::builder("partial_match")
            .input(json!("hello everyone"))
            .expected_output(json!("hello"))
            .tag("text")
            .build(),
    );

    println!("Suite '{}' has {} cases", suite.name, suite.case_count());

    // Filter by tag
    let text_cases = suite.by_tag("text");
    println!("  Cases tagged 'text': {}", text_cases.len());
    let numeric_cases = suite.by_tag("numeric");
    println!("  Cases tagged 'numeric': {}", numeric_cases.len());

    // Define metrics
    let metrics: Vec<Box<dyn EvalMetric>> = vec![
        Box::new(ExactMatchMetric),
        Box::new(ContainsMetric::new(true)),
    ];

    // Run the suite: the evaluator simply returns the input as-is
    let report = suite.run(&|input| input.clone(), &metrics);

    println!("\nResults per case:");
    for result in &report.results {
        println!(
            "  Case '{}': avg_score={:.2}, passed={}",
            result.case_name,
            result.avg_score(),
            result.passed
        );
        for score in &result.scores {
            println!("    - {}: {:.2}", score.metric_name, score.score);
        }
    }

    // -----------------------------------------------------------------------
    // 8. EvalReport Statistics and Failure Detection
    // -----------------------------------------------------------------------
    println!("\n--- 8. EvalReport Statistics ---");
    println!("Analyze the evaluation report for overall statistics and failures.\n");

    println!("Suite: '{}'", report.suite_name);
    println!("Overall score: {:.2}", report.overall_score);
    println!("Duration: {}ms", report.duration_ms);
    println!("Total cases: {}", report.results.len());

    // Passing rate at various thresholds
    let rate_50 = report.passing_rate(0.5);
    let rate_80 = report.passing_rate(0.8);
    let rate_100 = report.passing_rate(1.0);
    println!(
        "\nPassing rates:\n  at 0.5 threshold: {:.1}%\n  at 0.8 threshold: {:.1}%\n  at 1.0 threshold: {:.1}%",
        rate_50 * 100.0,
        rate_80 * 100.0,
        rate_100 * 100.0,
    );

    // Detect failures
    let failures = report.failures(0.5);
    println!("\nFailures (below 0.5 threshold): {}", failures.len());
    for f in &failures {
        println!(
            "  - '{}': avg_score={:.2}, expected={}, actual={}",
            f.case_name,
            f.avg_score(),
            f.expected,
            f.actual,
        );
    }

    // Full report JSON
    let report_json = report.to_json();
    println!("\nReport JSON (summary):");
    println!("  suite_name: {}", report_json["suite_name"]);
    println!("  overall_score: {}", report_json["overall_score"]);
    println!("  case_count: {}", report_json["case_count"]);
    println!("  duration_ms: {}", report_json["duration_ms"]);

    println!("\n=== Evaluation Framework Example Complete ===");
}
