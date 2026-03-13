//! Chain Composition Example
//!
//! Demonstrates the chain composition primitives for building complex LLM
//! workflows without needing a live model. All operations use pure functions
//! over `serde_json::Value`, making them easy to test and compose.
//!
//! Features shown:
//! - ChainStep creation with input/output key declarations
//! - SequentialChain running steps in order
//! - ParallelChain running branches concurrently and merging results
//! - TransformChain applying a single transformation
//! - MapChain processing each element of a JSON array
//! - ConditionalChain routing input based on predicates
//! - ChainPipeline fluent composition of heterogeneous operations
//! - ChainMetrics tracking per-step execution statistics
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example chain_composition`

mod shared;
use cognis::chains::composition::Handler;
use cognis::chains::{
    execute_with_metrics, ChainMetrics, ChainPipeline, ChainStep, CompositionConditionalChain,
    CompositionSequentialChain, CompositionTransformChain, MapChain, ParallelChain,
};
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Helper handlers
// ---------------------------------------------------------------------------

/// Doubles the numeric "value" field.
fn double_handler() -> Handler {
    Box::new(|v: Value| {
        let n = v
            .get("value")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| "missing 'value' field".to_string())?;
        Ok(json!({ "value": n * 2 }))
    })
}

/// Adds 10 to the numeric "value" field.
fn add_ten_handler() -> Handler {
    Box::new(|v: Value| {
        let n = v
            .get("value")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| "missing 'value' field".to_string())?;
        Ok(json!({ "value": n + 10 }))
    })
}

/// Uppercases the "text" field.
fn uppercase_handler() -> Handler {
    Box::new(|v: Value| {
        let s = v
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "missing 'text' field".to_string())?;
        Ok(json!({ "text": s.to_uppercase() }))
    })
}

/// Reverses the "text" field.
fn reverse_handler() -> Handler {
    Box::new(|v: Value| {
        let s = v
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "missing 'text' field".to_string())?;
        Ok(json!({ "text": s.chars().rev().collect::<String>() }))
    })
}

/// Counts the length of the "text" field.
fn length_handler() -> Handler {
    Box::new(|v: Value| {
        let s = v
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "missing 'text' field".to_string())?;
        Ok(json!({ "length": s.len() }))
    })
}

fn main() {
    println!("=== Chain Composition Example ===\n");

    // -----------------------------------------------------------------------
    // 1. ChainStep — Named steps with input/output key declarations
    // -----------------------------------------------------------------------
    println!("--- 1. ChainStep Creation ---");
    println!("A ChainStep wraps a handler with a name and declared I/O keys.\n");

    let step = ChainStep::builder()
        .name("double")
        .handler(double_handler())
        .input_keys(vec!["value".into()])
        .output_keys(vec!["value".into()])
        .build();

    println!("  Step name:    {}", step.name());
    println!("  Input keys:   {:?}", step.input_keys());
    println!("  Output keys:  {:?}", step.output_keys());

    let result = step.execute(json!({"value": 7})).unwrap();
    println!("  execute(7) -> {}", result);

    // -----------------------------------------------------------------------
    // 2. SequentialChain — Steps executed in order
    // -----------------------------------------------------------------------
    println!("\n--- 2. SequentialChain ---");
    println!("Runs steps in sequence, piping each output to the next input.\n");

    let mut seq = CompositionSequentialChain::new();
    seq.add_step(
        ChainStep::builder()
            .name("double")
            .handler(double_handler())
            .input_keys(vec!["value".into()])
            .output_keys(vec!["value".into()])
            .build(),
    );
    seq.add_step(
        ChainStep::builder()
            .name("add_ten")
            .handler(add_ten_handler())
            .input_keys(vec!["value".into()])
            .output_keys(vec!["value".into()])
            .build(),
    );
    seq.add_step(
        ChainStep::builder()
            .name("double_again")
            .handler(double_handler())
            .input_keys(vec!["value".into()])
            .output_keys(vec!["value".into()])
            .build(),
    );

    println!("  Steps: {:?}", seq.step_names());
    println!("  Count: {}", seq.step_count());

    let input = json!({"value": 5});
    let result = seq.execute(input.clone()).unwrap();
    println!("  Pipeline: double -> add_ten -> double_again");
    println!("  Input:  {}", input);
    println!("  Output: {} (5*2=10, 10+10=20, 20*2=40)", result);

    // -----------------------------------------------------------------------
    // 3. ParallelChain — Branches run concurrently and merged
    // -----------------------------------------------------------------------
    println!("\n--- 3. ParallelChain ---");
    println!("Runs multiple named branches on the same input and merges results.\n");

    let mut parallel = ParallelChain::new();
    parallel.add_branch("uppercase", uppercase_handler());
    parallel.add_branch("reversed", reverse_handler());
    parallel.add_branch("length", length_handler());

    println!("  Branches: {}", parallel.branch_count());

    let input = json!({"text": "hello world"});
    let result = parallel.execute(input.clone()).unwrap();
    println!("  Input:  {}", input);
    println!("  Merged output:");
    println!("    uppercase -> {}", result["uppercase"]);
    println!("    reversed  -> {}", result["reversed"]);
    println!("    length    -> {}", result["length"]);

    // -----------------------------------------------------------------------
    // 4. TransformChain — Single transformation
    // -----------------------------------------------------------------------
    println!("\n--- 4. TransformChain ---");
    println!("Applies a single transformation function to the input.\n");

    let transform = CompositionTransformChain::new(Box::new(|v: Value| {
        // Extract words and build a summary object
        let text = v.get("text").and_then(|t| t.as_str()).unwrap_or_default();
        let words: Vec<&str> = text.split_whitespace().collect();
        Ok(json!({
            "original": text,
            "word_count": words.len(),
            "first_word": words.first().copied().unwrap_or(""),
            "last_word": words.last().copied().unwrap_or(""),
        }))
    }));

    let input = json!({"text": "the quick brown fox jumps"});
    let result = transform.execute(input.clone()).unwrap();
    println!("  Input:  {}", input);
    println!(
        "  Output: {}",
        serde_json::to_string_pretty(&result).unwrap()
    );

    // -----------------------------------------------------------------------
    // 5. MapChain — Process each element of a JSON array
    // -----------------------------------------------------------------------
    println!("\n--- 5. MapChain ---");
    println!("Applies a handler to each element of a JSON array input.\n");

    let map_chain = MapChain::new(Box::new(|v: Value| {
        let n = v.as_i64().ok_or_else(|| "expected integer".to_string())?;
        Ok(json!(n * n))
    }));

    let input = json!([1, 2, 3, 4, 5]);
    let result = map_chain.execute(input.clone()).unwrap();
    println!("  Input:  {}", input);
    println!("  Output: {} (each element squared)", result);

    // Map over objects
    let map_text = MapChain::new(Box::new(|v: Value| {
        let s = v
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| "missing name".to_string())?;
        Ok(json!({ "name": s, "greeting": format!("Hello, {}!", s) }))
    }));

    let input = json!([
        {"name": "Alice"},
        {"name": "Bob"},
        {"name": "Charlie"}
    ]);
    let result = map_text.execute(input.clone()).unwrap();
    println!("  Input:  {}", input);
    println!(
        "  Output: {}",
        serde_json::to_string_pretty(&result).unwrap()
    );

    // -----------------------------------------------------------------------
    // 6. ConditionalChain — Route based on predicates
    // -----------------------------------------------------------------------
    println!("\n--- 6. ConditionalChain ---");
    println!("Routes input to different handlers based on predicate functions.\n");

    let mut cond = CompositionConditionalChain::new();

    // Route 1: if value > 100, label it "large"
    cond.when(
        Box::new(|v: &Value| {
            v.get("value")
                .and_then(|n| n.as_i64())
                .map_or(false, |n| n > 100)
        }),
        Box::new(|v: Value| {
            let n = v["value"].as_i64().unwrap();
            Ok(json!({ "value": n, "category": "large" }))
        }),
    );

    // Route 2: if value > 10, label it "medium"
    cond.when(
        Box::new(|v: &Value| {
            v.get("value")
                .and_then(|n| n.as_i64())
                .map_or(false, |n| n > 10)
        }),
        Box::new(|v: Value| {
            let n = v["value"].as_i64().unwrap();
            Ok(json!({ "value": n, "category": "medium" }))
        }),
    );

    // Fallback: label it "small"
    cond.otherwise(Box::new(|v: Value| {
        let n = v["value"].as_i64().unwrap();
        Ok(json!({ "value": n, "category": "small" }))
    }));

    for &val in &[5, 50, 500] {
        let input = json!({"value": val});
        let result = cond.execute(input).unwrap();
        println!("  value={:>3} -> category={}", val, result["category"]);
    }

    // -----------------------------------------------------------------------
    // 7. ChainPipeline — Fluent composition of mixed operations
    // -----------------------------------------------------------------------
    println!("\n--- 7. ChainPipeline ---");
    println!("Fluent builder composing steps, transforms, and parallel stages.\n");

    let pipeline = ChainPipeline::new()
        // Step 1: normalize the text to lowercase
        .then(
            ChainStep::builder()
                .name("normalize")
                .handler(Box::new(|v: Value| {
                    let text = v
                        .get("text")
                        .and_then(|t| t.as_str())
                        .ok_or_else(|| "missing text".to_string())?;
                    Ok(json!({ "text": text.to_lowercase() }))
                }))
                .input_keys(vec!["text".into()])
                .output_keys(vec!["text".into()])
                .build(),
        )
        // Step 2: trim whitespace
        .then(
            ChainStep::builder()
                .name("trim")
                .handler(Box::new(|v: Value| {
                    let text = v["text"].as_str().unwrap_or_default().trim().to_string();
                    Ok(json!({ "text": text }))
                }))
                .build(),
        )
        // Step 3: transform to extract analytics
        .transform(Box::new(|v: Value| {
            let text = v["text"].as_str().unwrap_or_default();
            let words: Vec<&str> = text.split_whitespace().collect();
            Ok(json!({
                "text": text,
                "word_count": words.len(),
                "char_count": text.len(),
            }))
        }));

    let input = json!({"text": "  Hello WORLD from Cognis  "});
    let result = pipeline.execute(input.clone()).unwrap();
    println!("  Input:  {}", input);
    println!(
        "  Output: {}",
        serde_json::to_string_pretty(&result).unwrap()
    );

    // Pipeline with parallel stage
    println!("\n  Pipeline with parallel branches:");
    let pipeline2 = ChainPipeline::new().parallel(vec![
        (
            "sum".to_string(),
            Box::new(|v: Value| {
                let arr = v.get("numbers").and_then(|a| a.as_array()).unwrap();
                let total: i64 = arr.iter().filter_map(|n| n.as_i64()).sum();
                Ok(json!(total))
            }) as Handler,
        ),
        (
            "count".to_string(),
            Box::new(|v: Value| {
                let arr = v.get("numbers").and_then(|a| a.as_array()).unwrap();
                Ok(json!(arr.len()))
            }) as Handler,
        ),
        (
            "max".to_string(),
            Box::new(|v: Value| {
                let arr = v.get("numbers").and_then(|a| a.as_array()).unwrap();
                let max = arr.iter().filter_map(|n| n.as_i64()).max().unwrap_or(0);
                Ok(json!(max))
            }) as Handler,
        ),
    ]);

    let input = json!({"numbers": [3, 7, 1, 9, 4]});
    let result = pipeline2.execute(input.clone()).unwrap();
    println!("  Input:  {}", input);
    println!(
        "  Output: {}",
        serde_json::to_string_pretty(&result).unwrap()
    );

    // -----------------------------------------------------------------------
    // 8. ChainMetrics — Track execution statistics
    // -----------------------------------------------------------------------
    println!("\n--- 8. ChainMetrics ---");
    println!("Track per-step execution timing and success rates.\n");

    // Build a sequential chain to run with metrics
    let mut chain = CompositionSequentialChain::new();
    chain.add_step(
        ChainStep::builder()
            .name("parse_input")
            .handler(Box::new(|v: Value| {
                let n = v
                    .get("raw")
                    .and_then(|r| r.as_str())
                    .ok_or_else(|| "missing raw".to_string())?;
                let parsed: i64 = n
                    .parse()
                    .map_err(|e: std::num::ParseIntError| e.to_string())?;
                Ok(json!({ "value": parsed }))
            }))
            .build(),
    );
    chain.add_step(
        ChainStep::builder()
            .name("multiply_by_three")
            .handler(Box::new(|v: Value| {
                let n = v["value"].as_i64().unwrap();
                Ok(json!({ "value": n * 3 }))
            }))
            .build(),
    );
    chain.add_step(
        ChainStep::builder()
            .name("format_output")
            .handler(Box::new(|v: Value| {
                let n = v["value"].as_i64().unwrap();
                Ok(json!({ "result": format!("The answer is {}", n) }))
            }))
            .build(),
    );

    let input = json!({"raw": "42"});
    let (result, metrics) = execute_with_metrics(&chain, input.clone());
    let result = result.unwrap();
    println!("  Input:  {}", input);
    println!("  Output: {}", result);

    println!("\n  Metrics summary:");
    println!("    Total steps:    {}", metrics.step_metrics().len());
    println!("    Total duration: {} ms", metrics.total_duration_ms());
    println!("    Success rate:   {:.0}%", metrics.success_rate() * 100.0);

    println!("\n  Per-step breakdown:");
    for m in metrics.step_metrics() {
        println!(
            "    [{}] {} ms, success={}",
            m.step_name, m.duration_ms, m.success
        );
    }

    // Manual metrics tracking
    println!("\n  Manual metrics tracking:");
    let mut manual_metrics = ChainMetrics::new();
    manual_metrics.record("step_a", 15, true);
    manual_metrics.record("step_b", 8, true);
    manual_metrics.record("step_c", 3, false);

    let metrics_json = manual_metrics.to_json();
    println!("  Metrics JSON:");
    println!("  {}", serde_json::to_string_pretty(&metrics_json).unwrap());

    println!("\n=== Chain Composition Example Complete ===");
}
