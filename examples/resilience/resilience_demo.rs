//! Resilience Patterns Demo
//!
//! Shows how to protect LLM calls with retry policies, a circuit breaker,
//! and a fallback chain — the most common production resilience patterns.
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example resilience_demo`

#[path = "../shared.rs"]
mod shared;

use cognis::resilience::{
    CircuitBreaker, FallbackChain, ResiliencePolicy, RetryConfig, RetryStrategy,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Resilience Patterns Demo ===\n");

    // -- 1. Configure retry with exponential backoff -------------------------
    println!("--- Retry Policy (exponential backoff) ---");
    let retry = RetryConfig::new(
        3,
        RetryStrategy::Exponential {
            base_ms: 200,
            max_ms: 5000,
            multiplier: 2.0,
        },
    )
    .with_retryable_error("timeout")
    .with_retryable_error("rate_limit");

    println!("Max attempts: {}", retry.max_attempts);
    println!(
        "Will retry 'timeout'?  {}",
        retry.should_retry(0, "connection timeout")
    );
    println!(
        "Will retry 'auth err'? {}",
        retry.should_retry(0, "authentication failure")
    );

    // -- 2. Circuit breaker --------------------------------------------------
    println!("\n--- Circuit Breaker ---");
    let breaker = CircuitBreaker::new(3, 30_000); // opens after 3 failures, 30s recovery
    println!(
        "State: {} (available: {})",
        breaker.state(),
        breaker.is_available()
    );

    // Simulate consecutive failures tripping the breaker
    for _ in 0..3 {
        breaker.record_failure();
    }
    println!("After 3 failures: state={}", breaker.state());

    breaker.reset();
    println!("After reset: state={}", breaker.state());

    // -- 3. Fallback chain ---------------------------------------------------
    println!("\n--- Fallback Chain ---");
    let mut chain = FallbackChain::new();
    chain.add_fallback("anthropic".into(), 1);
    chain.add_fallback("openai".into(), 2);
    chain.add_fallback("ollama".into(), 3);

    println!("Primary provider: {:?}", chain.next_fallback());

    chain.mark_failed("anthropic");
    println!("After anthropic fails: {:?}", chain.next_fallback());

    chain.mark_recovered("anthropic");
    println!("After recovery:        {:?}", chain.next_fallback());

    // -- 4. Combined resilience policy ---------------------------------------
    println!("\n--- Combined Resilience Policy ---");
    let policy = ResiliencePolicy::new("llm-api")
        .with_retry(retry)
        .with_circuit_breaker(CircuitBreaker::new(5, 30_000));

    println!(
        "Policy '{}': can_execute={}",
        policy.name,
        policy.can_execute()
    );

    policy.record_success();
    println!("After success: can_execute={}", policy.can_execute());

    for i in 1..=5 {
        policy.record_failure(&format!("error_{}", i));
    }
    println!("After 5 failures: can_execute={}", policy.can_execute());

    // -- 5. LLM call with resilience context ---------------------------------
    println!("\n--- LLM Call ---");
    let model = shared::get_chat_model(vec![
        "Key resilience patterns: 1) Retry with backoff for transient errors, \
         2) Circuit breaker to stop cascading failures, \
         3) Fallback chain to switch providers."
            .into(),
    ]);

    let messages = vec![cognis_core::messages::Message::human(
        "What resilience patterns matter most for production LLM APIs?",
    )];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("Response: {}", gen.message.content().text());
    }

    println!("\n=== Done ===");
    Ok(())
}
