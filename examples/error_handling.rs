//! Error Handling Example
//!
//! Demonstrates the runnable error handling module from `cognis-core`,
//! which provides structured error classification, recovery strategies,
//! circuit breaker patterns, and error chain tracking.
//!
//! Features shown:
//! - ErrorKind variants and retryable classification
//! - PatternErrorClassifier matching error messages to ErrorKind
//! - Custom pattern addition to the classifier
//! - RecoveryHandler providing default values on error
//! - MapErrorHandler transforming error messages
//! - CircuitBreaker state transitions (Closed -> Open -> HalfOpen)
//! - ErrorChain collecting and inspecting errors
//! - ErrorAction variants
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example error_handling`

mod shared;
use std::thread;
use std::time::Duration;

use cognis_core::runnables::{
    CircuitBreaker, CircuitState, ClassifiedError, ErrorAction, ErrorChain, ErrorClassifier,
    ErrorHandler, ErrorKind, MapErrorHandler, PatternErrorClassifier, RecoveryHandler,
};
use serde_json::json;

fn main() {
    println!("=== Error Handling Example ===\n");

    // -----------------------------------------------------------------------
    // 1. ErrorKind Variants and Retryable Classification
    // -----------------------------------------------------------------------
    println!("--- 1. ErrorKind Variants ---");
    println!("Each ErrorKind classifies an error and determines if it is retryable.\n");

    let kinds = [
        ErrorKind::Transient,
        ErrorKind::Permanent,
        ErrorKind::Timeout,
        ErrorKind::RateLimit,
        ErrorKind::Validation,
        ErrorKind::Unknown,
    ];

    for kind in &kinds {
        println!(
            "  {:12} -> retryable: {}",
            kind.to_string(),
            kind.is_retryable()
        );
    }

    println!();
    println!("Retryable kinds: Transient, Timeout, RateLimit");
    println!("Non-retryable:   Permanent, Validation, Unknown");

    // -----------------------------------------------------------------------
    // 2. PatternErrorClassifier — Built-in Patterns
    // -----------------------------------------------------------------------
    println!("\n--- 2. PatternErrorClassifier ---");
    println!("Classifies error messages by matching substrings against known patterns.\n");

    let classifier = PatternErrorClassifier::new();

    let test_messages = [
        "request timed out after 30s",
        "HTTP 429 Too Many Requests",
        "connection refused by remote host",
        "401 unauthorized access",
        "validation failed for field 'email'",
        "something completely unexpected happened",
    ];

    for msg in &test_messages {
        let kind = classifier.classify(msg);
        println!("  \"{msg}\"");
        println!("    -> {kind}\n");
    }

    // -----------------------------------------------------------------------
    // 3. Custom Pattern Addition
    // -----------------------------------------------------------------------
    println!("--- 3. Custom Patterns ---");
    println!("Add domain-specific patterns to the classifier.\n");

    let mut custom_classifier = PatternErrorClassifier::new();
    custom_classifier.add_pattern("quota exceeded", ErrorKind::RateLimit);
    custom_classifier.add_pattern("model not available", ErrorKind::Transient);
    custom_classifier.add_pattern("invalid api key", ErrorKind::Permanent);

    let custom_messages = [
        "quota exceeded for project X",
        "model not available, try again later",
        "invalid api key provided",
    ];

    for msg in &custom_messages {
        let kind = custom_classifier.classify(msg);
        println!("  \"{msg}\"");
        println!("    -> {kind}\n");
    }

    // -----------------------------------------------------------------------
    // 4. RecoveryHandler — Default Values on Error
    // -----------------------------------------------------------------------
    println!("--- 4. RecoveryHandler ---");
    println!("Provides a default recovery value when an error occurs.\n");

    let handler = RecoveryHandler::new(json!({
        "status": "fallback",
        "response": "Service temporarily unavailable. Please try again later.",
        "cached": true
    }));

    let error = ClassifiedError::new(ErrorKind::Transient, "connection reset by peer");
    let input = json!({"query": "What is Rust?"});

    let action = handler.handle(&error, &input);
    println!("  Error:  {} ({})", error.message, error.kind);
    println!("  Input:  {input}");
    println!("  Action: {action:?}");

    // -----------------------------------------------------------------------
    // 5. MapErrorHandler — Transforming Error Messages
    // -----------------------------------------------------------------------
    println!("\n--- 5. MapErrorHandler ---");
    println!("Transforms error messages before propagating them.\n");

    let mapper = MapErrorHandler::new(|msg| format!("[LLM Pipeline Error] {}", msg.to_uppercase()));

    // Direct mapping
    let original = "model returned empty response";
    let mapped = mapper.map(original);
    println!("  Original: \"{original}\"");
    println!("  Mapped:   \"{mapped}\"");

    // Via the ErrorHandler trait
    let error = ClassifiedError::new(ErrorKind::Permanent, "invalid prompt format");
    let action = mapper.handle(&error, &json!(null));
    println!("\n  ErrorHandler action: {action:?}");

    // -----------------------------------------------------------------------
    // 6. CircuitBreaker — State Transitions
    // -----------------------------------------------------------------------
    println!("\n--- 6. CircuitBreaker ---");
    println!("Tracks failures and prevents cascading errors using state transitions.");
    println!("States: Closed (normal) -> Open (blocked) -> HalfOpen (probe)\n");

    let mut cb = CircuitBreaker::new(3, Duration::from_millis(100));

    // Start in Closed state
    println!("  Initial state:    {}", cb.state());
    println!("  Allow request:    {}", cb.allow_request());
    println!("  Failure count:    {}", cb.failure_count());

    // Record some failures, but stay below threshold
    cb.record_failure();
    cb.record_failure();
    println!("\n  After 2 failures: {}", cb.state());
    println!("  Allow request:    {}", cb.allow_request());

    // Third failure trips the breaker
    cb.record_failure();
    println!("\n  After 3 failures: {}", cb.state());
    println!("  Allow request:    {}", cb.allow_request());
    println!("  Failure count:    {}", cb.failure_count());

    // Wait for the reset timeout to elapse -> HalfOpen
    println!("\n  Waiting 150ms for reset timeout...");
    thread::sleep(Duration::from_millis(150));
    println!("  State after wait: {}", cb.state());
    println!(
        "  Allow request:    {} (one probe allowed)",
        cb.allow_request()
    );

    // A successful probe closes the circuit
    cb.record_success();
    println!("\n  After success:    {}", cb.state());
    println!("  Allow request:    {}", cb.allow_request());
    println!("  Failure count:    {} (reset)", cb.failure_count());

    // Full cycle summary
    println!("\n  Full cycle: Closed -> Open -> HalfOpen -> Closed");

    // Demonstrate reset()
    cb.record_failure();
    cb.record_failure();
    cb.record_failure();
    assert_eq!(cb.state(), CircuitState::Open);
    cb.reset();
    println!("  After reset():    {} (manual reset)", cb.state());

    // -----------------------------------------------------------------------
    // 7. ErrorChain — Collecting and Inspecting Errors
    // -----------------------------------------------------------------------
    println!("\n--- 7. ErrorChain ---");
    println!("Collects classified errors through a processing pipeline.\n");

    let mut chain = ErrorChain::new();
    println!(
        "  Empty chain: len={}, is_empty={}",
        chain.len(),
        chain.is_empty()
    );

    // Add errors from a hypothetical multi-step pipeline
    chain.add(
        ClassifiedError::new(ErrorKind::Transient, "connection reset on attempt 1")
            .with_source("http_client"),
    );
    chain.add(
        ClassifiedError::new(ErrorKind::Timeout, "request timed out on attempt 2")
            .with_source("llm_provider"),
    );
    chain.add(
        ClassifiedError::new(ErrorKind::RateLimit, "rate limit hit on attempt 3")
            .with_source("api_gateway"),
    );
    chain.add(
        ClassifiedError::new(ErrorKind::Permanent, "invalid model ID")
            .with_source("model_registry"),
    );

    println!("  After adding 4 errors:");
    println!("    Total errors:    {}", chain.len());
    println!("    Retryable count: {}", chain.retry_count());
    println!("    Has permanent:   {}", chain.has_permanent());

    if let Some(latest) = chain.latest() {
        println!(
            "    Latest error:    \"{}\" ({}) from {:?}",
            latest.message, latest.kind, latest.source
        );
    }

    // Inspect all errors
    println!("\n  All errors in chain:");
    for (i, err) in chain.errors().iter().enumerate() {
        println!(
            "    [{}] {} - \"{}\" (source: {})",
            i,
            err.kind,
            err.message,
            err.source.as_deref().unwrap_or("none")
        );
    }

    // JSON serialization
    let chain_json = chain.to_json();
    println!("\n  Chain as JSON:");
    println!("    {}", serde_json::to_string_pretty(&chain_json).unwrap());

    // -----------------------------------------------------------------------
    // 8. ErrorAction Variants
    // -----------------------------------------------------------------------
    println!("\n--- 8. ErrorAction Variants ---");
    println!("ErrorAction tells the runtime how to handle a classified error.\n");

    let actions: Vec<(&str, ErrorAction)> = vec![
        ("Propagate", ErrorAction::Propagate),
        ("Recover", ErrorAction::Recover(json!({"default": "value"}))),
        ("Retry", ErrorAction::Retry),
        (
            "Fallback",
            ErrorAction::Fallback(json!({"cached_answer": "42"})),
        ),
        ("CircuitBreak", ErrorAction::CircuitBreak),
    ];

    for (label, action) in &actions {
        println!("  {label:14} -> {action:?}");
    }

    println!("\n=== Error Handling Example Complete ===");
}
