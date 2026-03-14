//! Conditional Chains Example
//!
//! Demonstrates ConditionalChain, BranchChain, and SwitchChain for routing.

#[path = "../shared.rs"]
mod shared;
use std::sync::Arc;

use serde_json::{json, Value};

use cognis::chains::{
    BranchChain, ConditionalChain, KeyContainsCondition, KeyEqualsCondition, KeyExistsCondition,
    SwitchChain,
};
use cognis_core::messages::Message;
use cognis_core::runnables::{Runnable, RunnableLambda};

fn text_transform(name: &'static str, upper: bool) -> Arc<dyn Runnable> {
    Arc::new(RunnableLambda::new(name, move |v: Value| async move {
        let s = v.get("text").and_then(|t| t.as_str()).unwrap_or_default();
        let t = if upper {
            s.to_uppercase()
        } else {
            s.to_lowercase()
        };
        Ok(json!({ "text": t }))
    }))
}

fn tag_lambda(tag: &'static str) -> Arc<dyn Runnable> {
    Arc::new(RunnableLambda::new("add_tag", move |v: Value| async move {
        let mut obj = v.as_object().cloned().unwrap_or_default();
        obj.insert("tag".to_string(), json!(tag));
        Ok(Value::Object(obj))
    }))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. ConditionalChain: if/else routing by key value
    let cond = ConditionalChain::builder(KeyEqualsCondition::new("mode", json!("upper")))
        .then(text_transform("upper", true))
        .otherwise(text_transform("lower", false))
        .build();
    let r1 = cond
        .invoke(json!({ "mode": "upper", "text": "Hello" }), None)
        .await?;
    let r2 = cond
        .invoke(json!({ "mode": "lower", "text": "Hello" }), None)
        .await?;
    println!(
        "ConditionalChain: upper={}, lower={}",
        r1["text"], r2["text"]
    );

    // 2. BranchChain: multi-condition routing (first match wins)
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

    for input in [
        json!({ "text": "an error occurred", "priority": "low" }),
        json!({ "text": "all good", "priority": "high" }),
        json!({ "text": "testing", "debug": true }),
        json!({ "text": "regular message" }),
    ] {
        let result = branch.invoke(input.clone(), None).await?;
        println!("BranchChain: {} -> tag={}", input, result["tag"]);
    }

    // 3. SwitchChain: route by key value
    let switch = SwitchChain::builder("language")
        .case("rust", tag_lambda("rust_handler"))
        .case("python", tag_lambda("python_handler"))
        .case("javascript", tag_lambda("js_handler"))
        .default(tag_lambda("unknown_language"))
        .build();

    for lang in ["rust", "python", "javascript", "go"] {
        let result = switch.invoke(json!({ "language": lang }), None).await?;
        println!("SwitchChain: language={} -> tag={}", lang, result["tag"]);
    }

    // 4. LLM-driven classification routed through SwitchChain
    let model = shared::get_chat_model(vec!["technical".into()]);
    let messages = vec![
        Message::system("Classify as: technical, billing, general. Reply with just the category."),
        Message::human("How do I configure async timeouts in Rust?"),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        let category = gen.message.content().text().trim().to_lowercase();
        let routed = switch.invoke(json!({ "language": category }), None).await?;
        println!(
            "LLM classified '{}', routed to tag={}",
            category, routed["tag"]
        );
    }

    Ok(())
}
