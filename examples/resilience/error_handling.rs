//! Error Handling Example
//!
//! Demonstrates classifying LLM errors, using a circuit breaker to prevent
//! cascading failures, and recovering with fallback responses.
//!
//! Run with: `cargo run -p cognis-examples --example error_handling`

#[path = "../shared.rs"]
mod shared;

use std::time::Duration;

use cognis_core::messages::{HumanMessage, Message, SystemMessage};
use cognis_core::runnables::{
    CircuitBreaker, ClassifiedError, ErrorChain, ErrorClassifier, ErrorHandler, ErrorKind,
    MapErrorHandler, PatternErrorClassifier, RecoveryHandler,
};
use serde_json::json;

fn main() {
    println!("=== Error Handling with LLM Calls ===\n");

    // -- Set up error classification with custom LLM-specific patterns ----------
    let mut classifier = PatternErrorClassifier::new();
    classifier.add_pattern("quota exceeded", ErrorKind::RateLimit);
    classifier.add_pattern("model not available", ErrorKind::Transient);
    classifier.add_pattern("invalid api key", ErrorKind::Permanent);

    // -- Set up a circuit breaker (trips after 3 failures, resets after 2s) -----
    let mut circuit = CircuitBreaker::new(3, Duration::from_secs(2));

    // -- Set up a fallback recovery handler --------------------------------------
    let fallback = RecoveryHandler::new(json!({
        "response": "Service temporarily unavailable. Please try again later.",
        "source": "fallback"
    }));

    // -- Call the LLM with error handling ----------------------------------------
    let model = shared::get_chat_model(vec![
        "Error classification matters because LLM APIs fail in different ways: \
         transient network issues should be retried, rate limits need backoff, \
         and permanent errors like invalid keys require immediate attention."
            .into(),
    ]);

    let messages = vec![
        Message::System(SystemMessage::new("You are a software engineering expert.")),
        Message::Human(HumanMessage::new(
            "In 2-3 sentences, explain why error classification matters in LLM apps.",
        )),
    ];

    let rt = tokio::runtime::Runtime::new().unwrap();

    // Simulate up to 3 attempts with circuit breaker protection
    let mut errors = ErrorChain::new();

    for attempt in 1..=3 {
        if !circuit.allow_request() {
            println!("Circuit breaker OPEN — skipping attempt {attempt}");
            continue;
        }

        println!("Attempt {attempt}: calling LLM...");
        let result = rt.block_on(async { model.invoke_messages(&messages, None).await });

        match result {
            Ok(response) => {
                circuit.record_success();
                println!("Success: {}\n", response.base.content.text());
                break;
            }
            Err(e) => {
                let kind = classifier.classify(&e.to_string());
                let retryable = kind.is_retryable();
                println!("Failed: {e} (classified: {kind}, retryable: {retryable})");

                let classified =
                    ClassifiedError::new(kind, &e.to_string()).with_source("llm_provider");
                errors.add(classified.clone());
                circuit.record_failure();

                // On non-retryable errors, use fallback immediately
                if !retryable {
                    let action = fallback.handle(&classified, &json!({}));
                    println!("Non-retryable error — fallback: {action:?}");
                    break;
                }
            }
        }
    }

    // -- Summary ----------------------------------------------------------------
    if !errors.is_empty() {
        println!(
            "\nError summary: {} error(s), {} retryable, permanent: {}",
            errors.len(),
            errors.retry_count(),
            errors.has_permanent()
        );
    }

    // Demonstrate error message transformation for logging
    let mapper = MapErrorHandler::new(|msg| format!("[LLM] {}", msg.to_uppercase()));
    let sample = ClassifiedError::new(ErrorKind::Timeout, "request timed out");
    let action = mapper.handle(&sample, &json!({}));
    println!("Mapped error example: {action:?}");

    println!("\n=== Done ===");
}
