//! Streaming Chat with Callbacks Example
//!
//! Demonstrates streaming responses with callback handlers that track
//! metrics and log events using LoggingCallbackHandler and MetricsCallbackHandler.

#[path = "../shared.rs"]
mod shared;

use std::sync::Arc;

use futures::StreamExt;
use serde_json::json;
use uuid::Uuid;

use cognis_core::callbacks::base::CallbackHandler;
use cognis_core::callbacks::handlers::{LogLevel, LoggingCallbackHandler, MetricsCallbackHandler};
use cognis_core::messages::{HumanMessage, Message};
use cognis_core::outputs::LLMResult;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let logging_handler = Arc::new(LoggingCallbackHandler::new(LogLevel::Info));
    let metrics_handler = Arc::new(MetricsCallbackHandler::new());

    // --- Token-level streaming with callbacks ---
    let response_text = "Rust is a systems programming language that guarantees memory safety \
        without garbage collection through its innovative ownership and borrowing system.";
    let model = shared::get_streaming_model(vec![response_text.into()]);
    let messages = vec![Message::Human(HumanMessage::new("What is Rust?"))];

    let run_id = Uuid::new_v4();
    logging_handler
        .on_llm_start(&json!({}), &["What is Rust?".into()], run_id, None)
        .await?;
    metrics_handler
        .on_llm_start(&json!({}), &["What is Rust?".into()], run_id, None)
        .await?;

    print!("Streaming: ");
    let mut stream = model._stream(&messages, None).await?;
    let mut full_response = String::new();
    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        let token = chunk.message.base.content.text();
        print!("{token}");
        full_response.push_str(&token);
        logging_handler
            .on_llm_new_token(&token, run_id, None)
            .await?;
    }
    println!("\n({} chars)\n", full_response.len());

    let llm_result = LLMResult {
        generations: vec![],
        llm_output: None,
        run: None,
    };
    logging_handler
        .on_llm_end(&llm_result, run_id, None)
        .await?;
    metrics_handler
        .on_llm_end(&llm_result, run_id, None)
        .await?;

    // --- Collecting streamed output ---
    let collect_model =
        shared::get_streaming_model(vec!["Ownership, borrowing, and lifetimes.".into()]);
    let messages = vec![Message::Human(HumanMessage::new("Key Rust concepts?"))];
    let stream = collect_model._stream(&messages, None).await?;
    let chunks: Vec<_> = stream.collect().await;
    let collected: String = chunks
        .into_iter()
        .filter_map(|r| r.ok())
        .map(|chunk| chunk.message.base.content.text())
        .collect();
    println!("Collected: \"{collected}\"");

    // --- Metrics summary ---
    let metrics = metrics_handler.get_metrics();
    println!(
        "\nMetrics: {} LLM calls, {} errors, ~{} tokens",
        metrics.total_llm_calls, metrics.total_errors, metrics.total_tokens
    );

    // --- Captured logs ---
    let logs = logging_handler.get_logs();
    println!("Log entries: {}", logs.len());
    for log in logs.iter().take(3) {
        println!("  {log}");
    }
    if logs.len() > 3 {
        println!("  ... ({} more)", logs.len() - 3);
    }

    Ok(())
}
