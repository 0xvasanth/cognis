//! Conditional Chains Example
//!
//! Demonstrates ConditionalChain, BranchChain, and SwitchChain for routing
//! execution based on input values. All chains use RunnableLambda functions
//! as branches -- no LLM or API keys required.

use std::sync::Arc;

use serde_json::{json, Value};

use cognis::chains::{
    BranchChain, ClosureCondition, ConditionalChain, KeyContainsCondition, KeyEqualsCondition,
    KeyExistsCondition, SwitchChain,
};
use cognis_core::runnables::{Runnable, RunnableLambda};

/// Helper: create a lambda that uppercases the "text" field.
fn uppercase_lambda() -> Arc<dyn Runnable> {
    Arc::new(RunnableLambda::new("uppercase", |v: Value| async move {
        let s = v.get("text").and_then(|t| t.as_str()).unwrap_or_default();
        Ok(json!({ "text": s.to_uppercase(), "transform": "uppercase" }))
    }))
}

/// Helper: create a lambda that lowercases the "text" field.
fn lowercase_lambda() -> Arc<dyn Runnable> {
    Arc::new(RunnableLambda::new("lowercase", |v: Value| async move {
        let s = v.get("text").and_then(|t| t.as_str()).unwrap_or_default();
        Ok(json!({ "text": s.to_lowercase(), "transform": "lowercase" }))
    }))
}

/// Helper: create a lambda that adds a tag to the input.
fn tag_lambda(tag: &'static str) -> Arc<dyn Runnable> {
    Arc::new(RunnableLambda::new("add_tag", move |v: Value| async move {
        let mut obj = v.as_object().cloned().unwrap_or_default();
        obj.insert("tag".to_string(), json!(tag));
        Ok(Value::Object(obj))
    }))
}

/// Helper: create a lambda that doubles a numeric "value" field.
fn double_lambda() -> Arc<dyn Runnable> {
    Arc::new(RunnableLambda::new("double", |v: Value| async move {
        let n = v.get("value").and_then(|x| x.as_i64()).unwrap_or(0);
        Ok(json!({ "value": n * 2, "operation": "doubled" }))
    }))
}

/// Helper: create a lambda that negates a numeric "value" field.
fn negate_lambda() -> Arc<dyn Runnable> {
    Arc::new(RunnableLambda::new("negate", |v: Value| async move {
        let n = v.get("value").and_then(|x| x.as_i64()).unwrap_or(0);
        Ok(json!({ "value": -n, "operation": "negated" }))
    }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Conditional Chains Example ===\n");

    // =========================================================================
    // 1. ConditionalChain: if/else routing
    // =========================================================================
    println!("--- 1. ConditionalChain ---\n");

    // Route based on whether "mode" equals "upper"
    let conditional = ConditionalChain::builder(KeyEqualsCondition::new("mode", json!("upper")))
        .then(uppercase_lambda())
        .otherwise(lowercase_lambda())
        .build();

    let input_upper = json!({ "mode": "upper", "text": "Hello World" });
    let result = conditional.invoke(input_upper, None).await?;
    println!(
        "  Input mode=upper: {} -> {}",
        "Hello World", result["text"]
    );

    let input_lower = json!({ "mode": "lower", "text": "Hello World" });
    let result = conditional.invoke(input_lower, None).await?;
    println!(
        "  Input mode=lower: {} -> {}",
        "Hello World", result["text"]
    );

    // ConditionalChain with a closure condition (numeric threshold)
    println!("\n  Closure condition (value > 10):");
    let numeric_conditional = ConditionalChain::builder(ClosureCondition::new(|v: &Value| {
        Ok(v.get("value").and_then(|x| x.as_i64()).unwrap_or(0) > 10)
    }))
    .then(double_lambda())
    .otherwise(negate_lambda())
    .build();

    let result = numeric_conditional
        .invoke(json!({ "value": 20 }), None)
        .await?;
    println!("    value=20 (>10): {}", result);

    let result = numeric_conditional
        .invoke(json!({ "value": 5 }), None)
        .await?;
    println!("    value=5  (<=10): {}", result);

    // ConditionalChain with passthrough (no else branch)
    println!("\n  Passthrough (no else branch):");
    let passthrough_chain = ConditionalChain::builder(KeyExistsCondition::new("premium"))
        .then(tag_lambda("premium_user"))
        .build();

    let result = passthrough_chain
        .invoke(json!({ "premium": true, "name": "Alice" }), None)
        .await?;
    println!("    With premium key: {}", result);

    let input_no_premium = json!({ "name": "Bob" });
    let result = passthrough_chain
        .invoke(input_no_premium.clone(), None)
        .await?;
    println!("    Without premium key (passthrough): {}", result);

    // =========================================================================
    // 2. BranchChain: multi-condition routing (first match wins)
    // =========================================================================
    println!("\n--- 2. BranchChain ---\n");

    let branch = BranchChain::builder()
        .branch(
            KeyContainsCondition::new("text", "error"),
            tag_lambda("error_handler"),
        )
        .branch(
            KeyEqualsCondition::new("priority", json!("high")),
            tag_lambda("high_priority"),
        )
        .branch(KeyExistsCondition::new("debug"), tag_lambda("debug_mode"))
        .default(tag_lambda("normal"))
        .build();

    let test_cases = vec![
        json!({ "text": "an error occurred", "priority": "low" }),
        json!({ "text": "all good", "priority": "high" }),
        json!({ "text": "testing", "debug": true }),
        json!({ "text": "regular message" }),
    ];

    for input in test_cases {
        let result = branch.invoke(input.clone(), None).await?;
        let tag = result["tag"].as_str().unwrap_or("(none)");
        println!("  Input: {} -> tag: {}", input, tag);
    }

    // =========================================================================
    // 3. SwitchChain: route by key value (like a switch/case statement)
    // =========================================================================
    println!("\n--- 3. SwitchChain ---\n");

    let switch = SwitchChain::builder("language")
        .case("rust", tag_lambda("rust_handler"))
        .case("python", tag_lambda("python_handler"))
        .case("javascript", tag_lambda("js_handler"))
        .default(tag_lambda("unknown_language"))
        .build();

    let languages = vec!["rust", "python", "javascript", "go"];
    for lang in languages {
        let input = json!({ "language": lang, "code": "println!()" });
        let result = switch.invoke(input, None).await?;
        let tag = result["tag"].as_str().unwrap_or("(none)");
        println!("  language={} -> tag: {}", lang, tag);
    }

    // SwitchChain with missing key (falls through to default)
    let no_lang = json!({ "code": "some code" });
    let result = switch.invoke(no_lang, None).await?;
    println!(
        "  (no language key) -> tag: {}",
        result["tag"].as_str().unwrap_or("(none)")
    );

    // =========================================================================
    // 4. Combining conditions
    // =========================================================================
    println!("\n--- 4. Combined Example: Request Router ---\n");

    // Simulate a request router that handles different request types
    let router = BranchChain::builder()
        .branch(
            KeyEqualsCondition::new("method", json!("GET")),
            Arc::new(RunnableLambda::new("get_handler", |v: Value| async move {
                let path = v.get("path").and_then(|p| p.as_str()).unwrap_or("/");
                Ok(json!({ "status": 200, "body": format!("GET {}", path) }))
            })),
        )
        .branch(
            KeyEqualsCondition::new("method", json!("POST")),
            Arc::new(RunnableLambda::new("post_handler", |v: Value| async move {
                let path = v.get("path").and_then(|p| p.as_str()).unwrap_or("/");
                Ok(json!({ "status": 201, "body": format!("Created at {}", path) }))
            })),
        )
        .default(Arc::new(RunnableLambda::new(
            "method_not_allowed",
            |v: Value| async move {
                let method = v
                    .get("method")
                    .and_then(|m| m.as_str())
                    .unwrap_or("UNKNOWN");
                Ok(json!({ "status": 405, "body": format!("{} not allowed", method) }))
            },
        )))
        .build();

    let requests = vec![
        json!({ "method": "GET", "path": "/users" }),
        json!({ "method": "POST", "path": "/users" }),
        json!({ "method": "DELETE", "path": "/users/1" }),
    ];

    for req in requests {
        let result = router.invoke(req.clone(), None).await?;
        println!(
            "  {} {} -> status={}, body={}",
            req["method"].as_str().unwrap_or("?"),
            req["path"].as_str().unwrap_or("?"),
            result["status"],
            result["body"]
        );
    }

    println!("\n=== Conditional Chains Example Complete ===");
    Ok(())
}
