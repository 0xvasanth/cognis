//! Streaming Chat with Callbacks Example
//!
//! Demonstrates streaming responses from a chat model with callback handlers
//! that track metrics and log events. Shows token-level streaming using
//! GenericFakeChatModel and LoggingCallbackHandler + MetricsCallbackHandler.
//!
//! No API keys required -- uses GenericFakeChatModel and FakeListChatModel.
//!
//! Run with: cargo run -p cognis-examples --example streaming_chat

mod shared;

use std::sync::Arc;

use futures::StreamExt;
use serde_json::json;
use uuid::Uuid;

use cognis_core::callbacks::base::CallbackHandler;
use cognis_core::callbacks::handlers::{
    LogLevel, LoggingCallbackHandler, MetricsCallbackHandler,
};
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{HumanMessage, Message};
use cognis_core::outputs::LLMResult;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Streaming Chat with Callbacks Example ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Set up callback handlers
    // -------------------------------------------------------------------------
    println!("--- Step 1: Setting up callback handlers ---\n");

    let logging_handler = Arc::new(LoggingCallbackHandler::new(LogLevel::Info));
    let metrics_handler = Arc::new(MetricsCallbackHandler::new());

    println!("  Created LoggingCallbackHandler (level=INFO)");
    println!("  Created MetricsCallbackHandler");
    println!();

    // -------------------------------------------------------------------------
    // Step 2: Token-level streaming
    // -------------------------------------------------------------------------
    println!("--- Step 2: Token-level streaming ---\n");

    let response_text = "Rust is a systems programming language that guarantees memory safety \
        without garbage collection through its innovative ownership and borrowing system.";

    let model = shared::get_streaming_model(vec![response_text.into()]);

    let messages = vec![Message::Human(HumanMessage::new(
        "What is Rust programming language?",
    ))];

    // Fire callback to track the LLM call.
    let run_id = Uuid::new_v4();
    logging_handler
        .on_llm_start(&json!({}), &["What is Rust?".to_string()], run_id, None)
        .await?;
    metrics_handler
        .on_llm_start(&json!({}), &["What is Rust?".to_string()], run_id, None)
        .await?;

    println!("  Streaming response token by token:");
    print!("  ");

    let mut stream = model._stream(&messages, None).await?;
    let mut token_count = 0;
    let mut full_response = String::new();

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        let token = chunk.message.base.content.text();
        print!("{token}");
        full_response.push_str(&token);
        token_count += 1;

        // Fire new-token callback for each token.
        logging_handler
            .on_llm_new_token(&token, run_id, None)
            .await?;
    }
    println!("\n");

    // Signal LLM end.
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

    println!(
        "  Received {token_count} tokens, {} chars total",
        full_response.len()
    );
    println!();

    // -------------------------------------------------------------------------
    // Step 3: Character-level streaming with FakeListChatModel
    // -------------------------------------------------------------------------
    println!("--- Step 3: Character-level streaming (FakeListChatModel) ---\n");

    let char_model = shared::get_streaming_model(vec![
        "The borrow checker prevents data races at compile time!".into(),
    ]);

    let messages = vec![Message::Human(HumanMessage::new(
        "Tell me about the borrow checker",
    ))];

    // Fire callbacks.
    let run_id2 = Uuid::new_v4();
    metrics_handler
        .on_llm_start(&json!({}), &["borrow checker".to_string()], run_id2, None)
        .await?;

    println!("  Streaming character by character:");
    print!("  ");

    let mut stream = char_model._stream(&messages, None).await?;
    let mut char_count = 0;

    while let Some(chunk_result) = stream.next().await {
        let chunk = chunk_result?;
        let ch = chunk.message.base.content.text();
        print!("{ch}");
        char_count += 1;
    }
    println!("\n");

    metrics_handler
        .on_llm_end(&llm_result, run_id2, None)
        .await?;

    println!("  Received {char_count} chunks (one per character)");
    println!();

    // -------------------------------------------------------------------------
    // Step 4: Collecting streamed output
    // -------------------------------------------------------------------------
    println!("--- Step 4: Collecting streamed output ---\n");

    let collect_model = shared::get_streaming_model(vec!["Ownership, borrowing, and lifetimes.".into()]);
    let messages = vec![Message::Human(HumanMessage::new("Key Rust concepts?"))];

    let stream = collect_model._stream(&messages, None).await?;
    let chunks: Vec<_> = stream.collect().await;

    let collected: String = chunks
        .into_iter()
        .filter_map(|r| r.ok())
        .map(|chunk| chunk.message.base.content.text())
        .collect();

    println!("  Collected text: \"{collected}\"");
    println!("  Length: {} chars", collected.len());
    println!();

    // -------------------------------------------------------------------------
    // Step 5: Review metrics
    // -------------------------------------------------------------------------
    println!("--- Step 5: Metrics Summary ---\n");

    let metrics = metrics_handler.get_metrics();
    println!("  Total LLM calls:   {}", metrics.total_llm_calls);
    println!("  Total tool calls:  {}", metrics.total_tool_calls);
    println!("  Total chain calls: {}", metrics.total_chain_calls);
    println!("  Total errors:      {}", metrics.total_errors);
    println!("  Estimated tokens:  {}", metrics.total_tokens);
    println!();

    // -------------------------------------------------------------------------
    // Step 6: Review captured logs
    // -------------------------------------------------------------------------
    println!("--- Step 6: Captured Logs ---\n");

    let logs = logging_handler.get_logs();
    println!("  Total log entries: {}", logs.len());
    // Show first 5 and last 3 log entries.
    let show_count = logs.len().min(5);
    for log in &logs[..show_count] {
        println!("  {log}");
    }
    if logs.len() > 8 {
        println!("  ... ({} more entries) ...", logs.len() - 8);
    }
    if logs.len() > 5 {
        let tail_start = logs.len().saturating_sub(3);
        for log in &logs[tail_start..] {
            println!("  {log}");
        }
    }

    // -------------------------------------------------------------------------
    // Step 7: Reset metrics and verify
    // -------------------------------------------------------------------------
    println!("\n--- Step 7: Reset and verify ---\n");

    metrics_handler.reset();
    let reset_metrics = metrics_handler.get_metrics();
    println!("  After reset:");
    println!("    LLM calls: {}", reset_metrics.total_llm_calls);
    println!("    Errors:    {}", reset_metrics.total_errors);

    logging_handler.clear_logs();
    println!("    Log entries: {}", logging_handler.get_logs().len());

    println!("\nDone!");
    Ok(())
}
