//! Error Recovery Example
//!
//! Demonstrates error recovery strategies: classify errors, retry with
//! exponential backoff, and fall back to an alternative when retries fail.
//!
//! Run with: `cargo run -p cognis-examples --example error_recovery`

#[path = "../shared.rs"]
mod shared;

use std::time::Duration;

use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{HumanMessage, Message, SystemMessage};
use cognisagent::recovery::{
    BackoffStrategy, ErrorCategory, ErrorClassifier, RecoveryManager, RecoveryPolicy,
    RecoveryStrategy,
};

fn main() {
    println!("=== Error Recovery Example ===\n");

    // -- Step 1: Classify errors into actionable categories --------------------
    let mut classifier = ErrorClassifier::new();
    classifier.add_pattern("quota_exceeded", ErrorCategory::RateLimit);

    let sample_errors = [
        "rate limit exceeded, please retry after 30s",
        "unauthorized: invalid api key",
        "quota_exceeded for project X",
        "request timeout after 30s",
    ];

    println!("Error classification:");
    for err in &sample_errors {
        let cat = classifier.classify(err);
        println!("  [{cat}] \"{err}\" (retryable: {})", cat.is_retryable());
    }

    // -- Step 2: Configure recovery policies -----------------------------------
    let mut policy = RecoveryPolicy::with_default_retry(3);
    policy.set_strategy(
        ErrorCategory::RateLimit,
        RecoveryStrategy::Retry {
            max_attempts: 3,
            backoff: BackoffStrategy::Exponential {
                initial: Duration::from_secs(1),
                multiplier: 2.0,
                max: Duration::from_secs(30),
            },
        },
    );
    policy.set_strategy(
        ErrorCategory::AuthFailure,
        RecoveryStrategy::Fallback("backup-api-key".into()),
    );

    println!("\nRecovery policies:");
    for cat in [
        ErrorCategory::RateLimit,
        ErrorCategory::AuthFailure,
        ErrorCategory::Timeout,
    ] {
        println!("  {cat} -> {:?}", policy.get_strategy(&cat));
    }

    // -- Step 3: Simulate error handling with the RecoveryManager ---------------
    let mut manager = RecoveryManager::new(policy);

    println!("\nSimulating repeated rate-limit errors:");
    for i in 1..=4 {
        let action = manager.handle_error("rate limit exceeded");
        println!(
            "  attempt {i}: retry={}, delay={:?}",
            action.should_retry(),
            action.delay,
        );
    }

    println!("\nHandling auth failure (non-retryable):");
    let auth = manager.handle_error("unauthorized: invalid credentials");
    println!("  retry={}, action={:?}", auth.should_retry(), auth.action);

    // -- Step 4: Recovery log summary ------------------------------------------
    let log = manager.log();
    println!(
        "\nRecovery log: {} attempts, {:.0}% success rate",
        log.len(),
        log.success_rate() * 100.0,
    );

    // -- Step 5: Real LLM call with error recovery -----------------------------
    println!("\n--- LLM call with recovery fallback ---\n");

    let model = shared::get_chat_model(vec![
        "Best practices: 1) Classify errors as transient or permanent, \
         2) Use exponential backoff for retries, \
         3) Fall back to cached responses when retries are exhausted."
            .into(),
    ]);

    let messages =
        vec![
        Message::System(SystemMessage::new("You are a reliability engineering expert.")),
        Message::Human(HumanMessage::new(
            "What are best practices for error recovery in LLM API calls? Answer in 2-3 sentences.",
        )),
    ];

    let rt = tokio::runtime::Runtime::new().unwrap();
    match rt.block_on(model.invoke_messages(&messages, None)) {
        Ok(resp) => println!("  Response: {}", resp.base.content.text()),
        Err(e) => {
            println!("  Call failed: {e}");
            let mut recovery = RecoveryManager::new(RecoveryPolicy::with_default_retry(3));
            let action = recovery.handle_error(&e.to_string());
            println!(
                "  Recovery: category={}, retry={}, delay={:?}",
                action.category,
                action.should_retry(),
                action.delay,
            );
        }
    }

    println!("\n=== Done ===");
}
