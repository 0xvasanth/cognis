//! Chain Composition Example
//!
//! Demonstrates key composition patterns: SequentialChain, ParallelChain,
//! TransformChain, and an LLM chain using the `chain!` macro.
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example chain_composition`

#[path = "../shared.rs"]
mod shared;
use cognis::chains::composition::Handler;
use cognis::chains::{
    ChainStep, CompositionSequentialChain, CompositionTransformChain, ParallelChain,
};
use cognis_core::chain;
use cognis_core::language_models::ChatModelRunnable;
use cognis_core::output_parsers::StrOutputParser;
use cognis_core::prompts::ChatPromptTemplate;
use cognis_core::runnables::Runnable;
use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// Helper handlers
// ---------------------------------------------------------------------------

fn double_handler() -> Handler {
    Box::new(|v: Value| {
        let n = v
            .get("value")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| "missing 'value' field".to_string())?;
        Ok(json!({ "value": n * 2 }))
    })
}

fn add_ten_handler() -> Handler {
    Box::new(|v: Value| {
        let n = v
            .get("value")
            .and_then(|v| v.as_i64())
            .ok_or_else(|| "missing 'value' field".to_string())?;
        Ok(json!({ "value": n + 10 }))
    })
}

fn uppercase_handler() -> Handler {
    Box::new(|v: Value| {
        let s = v
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "missing 'text' field".to_string())?;
        Ok(json!({ "text": s.to_uppercase() }))
    })
}

fn reverse_handler() -> Handler {
    Box::new(|v: Value| {
        let s = v
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "missing 'text' field".to_string())?;
        Ok(json!({ "text": s.chars().rev().collect::<String>() }))
    })
}

fn length_handler() -> Handler {
    Box::new(|v: Value| {
        let s = v
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| "missing 'text' field".to_string())?;
        Ok(json!({ "length": s.len() }))
    })
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Chain Composition Example ===\n");

    // 1. SequentialChain — steps executed in order
    println!("--- 1. SequentialChain ---");
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

    let input = json!({"value": 5});
    let result = seq.execute(input.clone()).unwrap();
    println!("  Pipeline: double -> add_ten -> double");
    println!(
        "  Input:  {}  Output: {} (5*2=10, +10=20, *2=40)\n",
        input, result
    );

    // 2. ParallelChain — branches run concurrently, results merged
    println!("--- 2. ParallelChain ---");
    let mut parallel = ParallelChain::new();
    parallel.add_branch("uppercase", uppercase_handler());
    parallel.add_branch("reversed", reverse_handler());
    parallel.add_branch("length", length_handler());

    let input = json!({"text": "hello world"});
    let result = parallel.execute(input.clone()).unwrap();
    println!("  Input:  {}", input);
    println!(
        "  Output: {}\n",
        serde_json::to_string_pretty(&result).unwrap()
    );

    // 3. TransformChain — single transformation
    println!("--- 3. TransformChain ---");
    let transform = CompositionTransformChain::new(Box::new(|v: Value| {
        let text = v.get("text").and_then(|t| t.as_str()).unwrap_or_default();
        let words: Vec<&str> = text.split_whitespace().collect();
        Ok(json!({
            "word_count": words.len(),
            "first_word": words.first().copied().unwrap_or(""),
            "last_word": words.last().copied().unwrap_or(""),
        }))
    }));

    let input = json!({"text": "the quick brown fox jumps"});
    let result = transform.execute(input.clone()).unwrap();
    println!("  Input:  {}", input);
    println!(
        "  Output: {}\n",
        serde_json::to_string_pretty(&result).unwrap()
    );

    // 4. LLM Chain — prompt -> model -> parser using the chain! macro
    println!("--- 4. LLM Chain (prompt -> model -> parser) ---");
    let prompt = ChatPromptTemplate::from_messages(vec![
        (
            "system",
            "You are a helpful assistant that explains topics concisely.",
        ),
        ("human", "Explain {topic} in one sentence."),
    ])?;

    let model = shared::get_chat_model(vec![
        "A chain composes multiple processing steps into a single callable unit, \
         passing each step's output as the next step's input."
            .into(),
    ]);

    let llm_chain = chain!(prompt, ChatModelRunnable::new(model), StrOutputParser)?;

    let input = json!({ "topic": "chain composition in LLM frameworks" });
    let result = llm_chain.invoke(input.clone(), None).await?;
    let text = result
        .as_str()
        .map(String::from)
        .unwrap_or_else(|| result.to_string());
    println!("  Input:    {}", input);
    println!("  Response: {}", text);

    println!("\n=== Done ===");
    Ok(())
}
