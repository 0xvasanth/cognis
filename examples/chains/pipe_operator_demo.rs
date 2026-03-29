//! Pipe Operator Demo
//!
//! Demonstrates LCEL-style chain composition using the `|` operator
//! and schema introspection on the Runnable trait.
//!
//! Run with: `cargo run -p cognis-examples --example pipe_operator_demo`

#[path = "../shared.rs"]
mod shared;

use std::sync::Arc;

use serde_json::{json, Value};

use cognis_core::runnables::lambda::RunnableLambda;
use cognis_core::runnables::pipe::RunnableRef;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Pipe Operator Demo ===\n");

    // ── Build steps ──────────────────────────────────────────────────

    // Step 1: Format a topic into a prompt
    let prompt = RunnableRef::new(Arc::new(RunnableLambda::new(
        "format_prompt",
        |input: Value| async move {
            let topic = input.as_str().unwrap_or("Rust");
            Ok(json!({
                "messages": [
                    {"type": "system", "content": "Explain the topic in one sentence."},
                    {"type": "human", "content": topic}
                ]
            }))
        },
    )));

    // Step 2: Call the LLM
    let model = shared::get_chat_model(vec![
        "Rust is a systems programming language focused on safety and performance.".into(),
        "Python is a high-level interpreted language popular for data science.".into(),
    ]);
    let model_ref = model.clone();
    let call_model = RunnableRef::new(Arc::new(RunnableLambda::new(
        "call_model",
        move |input: Value| {
            let m = model_ref.clone();
            Box::pin(async move {
                let messages: Vec<cognis_core::messages::Message> = input
                    .get("messages")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                let response = m
                    .invoke_messages(&messages, None)
                    .await
                    .map_err(|e| cognis_core::error::CognisError::Other(e.to_string()))?;
                Ok(json!({ "response": response.base.content.text().trim() }))
            })
        },
    )));

    // Step 3: Extract the text
    let extract = RunnableRef::new(Arc::new(RunnableLambda::new(
        "extract_text",
        |input: Value| async move {
            let text = input
                .get("response")
                .and_then(|v| v.as_str())
                .unwrap_or("(no response)");
            Ok(Value::String(text.to_string()))
        },
    )));

    // ── Compose with | operator ─────────────────────────────────────

    println!("--- Chain Composition ---\n");
    let chain = prompt | call_model | extract;
    println!("Chain: {}\n", chain.runnable().name());

    // ── Schema introspection ────────────────────────────────────────

    println!("--- Schema Introspection ---\n");
    println!("Input schema:  {}", chain.runnable().input_schema());
    println!("Output schema: {}\n", chain.runnable().output_schema());

    // ── Single invocation ───────────────────────────────────────────

    println!("--- Single Invocation ---\n");
    let result = chain.runnable().invoke(json!("Rust"), None).await?;
    println!("Topic: Rust");
    println!("Result: {}\n", result);

    // ── Batch invocation ────────────────────────────────────────────

    println!("--- Batch Invocation ---\n");
    let results = chain
        .runnable()
        .batch(vec![json!("Python"), json!("TypeScript")], None)
        .await?;
    for (i, r) in results.iter().enumerate() {
        println!("  [{}] {}", i + 1, r);
    }

    println!("\n=== Done ===");
    Ok(())
}
