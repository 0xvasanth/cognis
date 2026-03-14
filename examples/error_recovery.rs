//! Error Recovery Example
//!
//! Demonstrates the error recovery system from cognisagent:
//!
//! - `ErrorClassifier` — classify error messages into categories
//! - `RecoveryPolicy` — configure recovery strategies per error category
//! - `RecoveryManager` — handle errors end-to-end with backoff and logging
//! - `BackoffStrategy` — fixed, linear, and exponential backoff
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example error_recovery`

mod shared;
use std::time::Duration;

use cognisagent::recovery::{
    BackoffStrategy, ErrorCategory, ErrorClassifier, RecoveryManager, RecoveryPolicy,
    RecoveryStrategy,
};

fn main() {
    println!("=== Error Recovery Example ===\n");

    // -----------------------------------------------------------------------
    // 1. ErrorClassifier: classify various error messages
    // -----------------------------------------------------------------------
    println!("--- 1. ErrorClassifier ---");

    let classifier = ErrorClassifier::new();

    let test_errors = vec![
        "rate limit exceeded, please retry after 30s",
        "HTTP 429 Too Many Requests",
        "unauthorized: invalid api key provided",
        "model is overloaded, try again later",
        "503 service unavailable",
        "connection refused to api.example.com",
        "request timeout after 30s",
        "tool execution failed: calculator returned NaN",
        "parse error: invalid json at line 5",
        "resource exhausted: token limit reached",
        "something completely unexpected happened",
    ];

    for error in &test_errors {
        let category = classifier.classify(error);
        println!(
            "  \"{}\" -> {} (retryable: {})",
            error,
            category,
            category.is_retryable()
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 2. Custom patterns
    // -----------------------------------------------------------------------
    println!("--- 2. Custom classifier patterns ---");

    let mut custom_classifier = ErrorClassifier::new();
    custom_classifier.add_pattern("quota_exceeded", ErrorCategory::RateLimit);
    custom_classifier.add_pattern("gpu_busy", ErrorCategory::ModelOverloaded);

    let custom_errors = vec![
        "quota_exceeded for project X",
        "gpu_busy: all inference slots occupied",
        "normal timeout error",
    ];

    for error in &custom_errors {
        let category = custom_classifier.classify(error);
        println!("  \"{}\" -> {}", error, category);
    }
    println!();

    // -----------------------------------------------------------------------
    // 3. BackoffStrategy demonstrations
    // -----------------------------------------------------------------------
    println!("--- 3. BackoffStrategy ---");

    println!("  Fixed backoff (3s):");
    let fixed = BackoffStrategy::Fixed(Duration::from_secs(3));
    for attempt in 0..4 {
        println!(
            "    attempt {}: {:?}",
            attempt,
            fixed.delay_for_attempt(attempt)
        );
    }

    println!("  Linear backoff (initial=1s, increment=2s):");
    let linear = BackoffStrategy::Linear {
        initial: Duration::from_secs(1),
        increment: Duration::from_secs(2),
    };
    for attempt in 0..4 {
        println!(
            "    attempt {}: {:?}",
            attempt,
            linear.delay_for_attempt(attempt)
        );
    }

    println!("  Exponential backoff (initial=1s, multiplier=2, max=30s):");
    let exponential = BackoffStrategy::Exponential {
        initial: Duration::from_secs(1),
        multiplier: 2.0,
        max: Duration::from_secs(30),
    };
    for attempt in 0..6 {
        println!(
            "    attempt {}: {:?}",
            attempt,
            exponential.delay_for_attempt(attempt)
        );
    }
    println!();

    // -----------------------------------------------------------------------
    // 4. RecoveryPolicy: default strategies
    // -----------------------------------------------------------------------
    println!("--- 4. RecoveryPolicy (defaults) ---");

    let policy = RecoveryPolicy::new();

    let categories = vec![
        ErrorCategory::RateLimit,
        ErrorCategory::AuthFailure,
        ErrorCategory::ModelOverloaded,
        ErrorCategory::ToolError,
        ErrorCategory::Timeout,
        ErrorCategory::NetworkError,
        ErrorCategory::ParseError,
        ErrorCategory::ResourceExhausted,
    ];

    for cat in &categories {
        let strategy = policy.get_strategy(cat);
        println!("  {} -> {:?}", cat, strategy);
    }
    println!();

    // -----------------------------------------------------------------------
    // 5. Custom RecoveryPolicy
    // -----------------------------------------------------------------------
    println!("--- 5. Custom RecoveryPolicy ---");

    let mut custom_policy = RecoveryPolicy::new();
    custom_policy.set_strategy(
        ErrorCategory::ToolError,
        RecoveryStrategy::Retry {
            max_attempts: 2,
            backoff: BackoffStrategy::Fixed(Duration::from_millis(500)),
        },
    );
    custom_policy.set_strategy(
        ErrorCategory::AuthFailure,
        RecoveryStrategy::Fallback("backup-api-key".to_string()),
    );

    println!(
        "  ToolError -> {:?}",
        custom_policy.get_strategy(&ErrorCategory::ToolError)
    );
    println!(
        "  AuthFailure -> {:?}",
        custom_policy.get_strategy(&ErrorCategory::AuthFailure)
    );
    println!();

    // -----------------------------------------------------------------------
    // 6. RecoveryManager: handle errors end-to-end
    // -----------------------------------------------------------------------
    println!("--- 6. RecoveryManager: handling errors ---");

    let policy = RecoveryPolicy::with_default_retry(3);
    let mut manager = RecoveryManager::new(policy);

    // Simulate repeated rate limit errors
    println!("  Simulating rate limit errors:");
    for i in 1..=4 {
        let action = manager.handle_error("rate limit exceeded");
        println!(
            "    Error #{}: category={}, attempt={}, should_retry={}, delay={:?}",
            i,
            action.category,
            action.attempt,
            action.should_retry(),
            action.delay,
        );
    }
    println!();

    // Handle a non-retryable error
    println!("  Handling auth failure:");
    let auth_action = manager.handle_error("unauthorized: invalid credentials");
    println!(
        "    category={}, should_retry={}, action={:?}",
        auth_action.category,
        auth_action.should_retry(),
        auth_action.action,
    );
    println!();

    // Handle a timeout
    println!("  Handling timeout:");
    let timeout_action = manager.handle_error("request timed out");
    println!(
        "    category={}, attempt={}, should_retry={}, delay={:?}",
        timeout_action.category,
        timeout_action.attempt,
        timeout_action.should_retry(),
        timeout_action.delay,
    );
    println!();

    // -----------------------------------------------------------------------
    // 7. Recovery log analysis
    // -----------------------------------------------------------------------
    println!("--- 7. Recovery log ---");

    let log = manager.log();
    println!("  Total attempts: {}", log.len());
    println!("  Success rate: {:.1}%", log.success_rate() * 100.0);
    println!("  Total recovery time: {:?}", log.total_recovery_time());

    let rate_limit_attempts = log.attempts_by_category(&ErrorCategory::RateLimit);
    println!(
        "  Rate limit attempts: {} (of {} total)",
        rate_limit_attempts.len(),
        log.len()
    );

    println!(
        "  Log JSON:\n  {}",
        serde_json::to_string_pretty(&log.to_json()).unwrap()
    );
    println!();

    // -----------------------------------------------------------------------
    // 8. Reset and fresh start
    // -----------------------------------------------------------------------
    println!("--- 8. Reset manager ---");

    manager.reset();
    println!("  After reset: log entries={}", manager.log().len());

    let fresh_action = manager.handle_error("rate limit exceeded");
    println!(
        "  Fresh attempt after reset: attempt={}, should_retry={}",
        fresh_action.attempt,
        fresh_action.should_retry()
    );
    println!();

    // -----------------------------------------------------------------------
    // 9. Suggested delays by category
    // -----------------------------------------------------------------------
    println!("--- 9. Suggested delays by category ---");

    let all_categories = vec![
        ErrorCategory::RateLimit,
        ErrorCategory::ModelOverloaded,
        ErrorCategory::NetworkError,
        ErrorCategory::Timeout,
        ErrorCategory::AuthFailure,
        ErrorCategory::ToolError,
        ErrorCategory::ParseError,
        ErrorCategory::ResourceExhausted,
    ];

    for cat in &all_categories {
        println!("  {}: {:?}", cat, cat.suggested_delay());
    }
    println!();

    // -----------------------------------------------------------------------
    // 10. Real LLM Demo — Error recovery with a real model
    // -----------------------------------------------------------------------
    println!("--- 10. Real LLM Demo (Error Recovery with Real Model) ---\n");

    let model = shared::get_chat_model(vec![
        "When an LLM call fails, best practices include: 1) Classify the error (transient vs permanent), 2) Apply exponential backoff for retries, 3) Fall back to cached responses when retries are exhausted.".into(),
    ]);

    let messages = vec![
        cognis_core::messages::Message::System(cognis_core::messages::SystemMessage::new(
            "You are a reliability engineering expert.",
        )),
        cognis_core::messages::Message::Human(cognis_core::messages::HumanMessage::new(
            "What are the best practices for error recovery when making LLM API calls? Answer in 2-3 sentences.",
        )),
    ];

    let rt = tokio::runtime::Runtime::new().unwrap();
    let result = rt.block_on(async {
        model.invoke_messages(&messages, None).await
    });

    match result {
        Ok(response) => {
            println!("  LLM response: {}", response.base.content.text());
        }
        Err(e) => {
            // Demonstrate recovery: classify the error and determine action
            println!("  LLM call failed: {}", e);
            let policy = RecoveryPolicy::with_default_retry(3);
            let mut recovery_manager = RecoveryManager::new(policy);
            let action = recovery_manager.handle_error(&e.to_string());
            println!(
                "  Recovery action: category={}, should_retry={}, delay={:?}",
                action.category,
                action.should_retry(),
                action.delay,
            );
        }
    }
    println!();

    println!("=== Error Recovery Example Complete ===");
}
