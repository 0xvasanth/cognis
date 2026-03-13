//! Resilience Patterns Demo
//!
//! Demonstrates the `cognis::resilience` module:
//! - `RetryStrategy` variants (Fixed, Exponential, Linear) with delay calculations
//! - `RetryConfig` with retryable error filtering
//! - `CircuitBreaker` with state transitions (Closed -> Open -> HalfOpen)
//! - `Bulkhead` with permit acquisition and RAII release
//! - `FallbackChain` for provider failover
//! - `ResiliencePolicy` combining all patterns
//! - `ResilienceMetrics` for tracking events
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example resilience_demo`

mod shared;
use cognis::resilience::{
    Bulkhead, CircuitBreaker, FallbackChain, ResilienceMetrics, ResiliencePolicy, RetryConfig,
    RetryStrategy,
};

fn main() {
    println!("=== Resilience Patterns Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. RetryStrategy variants
    // -----------------------------------------------------------------------
    println!("--- 1. RetryStrategy ---");

    // Fixed delay
    let fixed = RetryStrategy::Fixed { delay_ms: 500 };
    println!("Strategy: {}", fixed);
    for attempt in 0..4 {
        println!(
            "  attempt {}: delay = {}ms",
            attempt,
            fixed.delay_for_attempt(attempt)
        );
    }

    // Exponential backoff
    let exponential = RetryStrategy::Exponential {
        base_ms: 100,
        max_ms: 5000,
        multiplier: 2.0,
    };
    println!("\nStrategy: {}", exponential);
    for attempt in 0..8 {
        println!(
            "  attempt {}: delay = {}ms",
            attempt,
            exponential.delay_for_attempt(attempt)
        );
    }

    // Linear increase
    let linear = RetryStrategy::Linear {
        initial_ms: 200,
        increment_ms: 100,
        max_ms: 1000,
    };
    println!("\nStrategy: {}", linear);
    for attempt in 0..12 {
        println!(
            "  attempt {}: delay = {}ms",
            attempt,
            linear.delay_for_attempt(attempt)
        );
    }

    // None strategy
    let none = RetryStrategy::None;
    println!(
        "\nStrategy: {} -> delay = {}ms",
        none,
        none.delay_for_attempt(0)
    );

    // Serialization
    println!("\nFixed as JSON: {}", fixed.to_json());

    // -----------------------------------------------------------------------
    // 2. RetryConfig with error filtering
    // -----------------------------------------------------------------------
    println!("\n--- 2. RetryConfig ---");

    let config = RetryConfig::new(
        3,
        RetryStrategy::Exponential {
            base_ms: 100,
            max_ms: 2000,
            multiplier: 2.0,
        },
    )
    .with_retryable_error("timeout")
    .with_retryable_error("rate_limit")
    .with_retryable_error("503");

    println!("Max attempts: {}", config.max_attempts);
    println!("Retryable errors: {:?}", config.retryable_errors);

    // Test should_retry with various errors
    let test_cases = [
        (0, "connection timeout", true),
        (0, "rate_limit exceeded", true),
        (0, "authentication failure", false),
        (1, "503 Service Unavailable", true),
        (2, "timeout again", true),
        (3, "timeout but max reached", false),
    ];

    println!("\nRetry decisions:");
    for (attempt, error, expected) in &test_cases {
        let result = config.should_retry(*attempt, error);
        println!(
            "  attempt={}, error='{}' -> should_retry={} (expected={})",
            attempt, error, result, expected
        );
        assert_eq!(result, *expected);
    }

    println!(
        "\nConfig JSON: {}",
        serde_json::to_string_pretty(&config.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 3. CircuitBreaker
    // -----------------------------------------------------------------------
    println!("\n--- 3. CircuitBreaker ---");

    let cb = CircuitBreaker::new(3, 0); // threshold=3, recovery_timeout=0ms for demo
    println!(
        "Initial state: {} (available={})",
        cb.state(),
        cb.is_available()
    );

    // Record successes
    cb.record_success();
    cb.record_success();
    println!(
        "After 2 successes: state={}, failures={}",
        cb.state(),
        cb.failure_count()
    );

    // Record failures up to threshold
    cb.record_failure();
    println!(
        "After 1 failure: state={}, failures={}",
        cb.state(),
        cb.failure_count()
    );
    cb.record_failure();
    println!(
        "After 2 failures: state={}, failures={}",
        cb.state(),
        cb.failure_count()
    );
    cb.record_failure();
    println!(
        "After 3 failures (threshold): state={}, available={}",
        cb.state(),
        cb.is_available()
    );

    // With 0ms timeout, is_available() transitions to HalfOpen
    // (we already called is_available above which triggered the transition)
    println!("State after availability check: {}", cb.state());

    // A success in HalfOpen closes the circuit
    cb.record_success();
    println!(
        "After success in HalfOpen: state={}, failures={}",
        cb.state(),
        cb.failure_count()
    );

    // Demonstrate reset
    cb.record_failure();
    cb.record_failure();
    cb.record_failure();
    println!("\nAfter 3 more failures: state={}", cb.state());
    cb.reset();
    println!(
        "After reset(): state={}, available={}",
        cb.state(),
        cb.is_available()
    );

    println!(
        "Breaker JSON: {}",
        serde_json::to_string_pretty(&cb.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 4. Bulkhead
    // -----------------------------------------------------------------------
    println!("\n--- 4. Bulkhead ---");

    let bh = Bulkhead::new(3);
    println!(
        "Bulkhead: max={}, available={}, active={}",
        bh.max_concurrent(),
        bh.available(),
        bh.active()
    );

    // Acquire permits
    let p1 = bh.try_acquire();
    println!(
        "Acquired permit 1: success={}, available={}, active={}",
        p1.is_some(),
        bh.available(),
        bh.active()
    );

    let p2 = bh.try_acquire();
    println!(
        "Acquired permit 2: success={}, available={}, active={}",
        p2.is_some(),
        bh.available(),
        bh.active()
    );

    let p3 = bh.try_acquire();
    println!(
        "Acquired permit 3: success={}, available={}, active={}",
        p3.is_some(),
        bh.available(),
        bh.active()
    );

    // Fourth should fail
    let p4 = bh.try_acquire();
    println!(
        "Acquired permit 4: success={} (bulkhead full)",
        p4.is_some()
    );

    // Drop a permit to free a slot
    drop(p1);
    println!(
        "After dropping permit 1: available={}, active={}",
        bh.available(),
        bh.active()
    );

    let p5 = bh.try_acquire();
    println!(
        "Acquired permit 5 (after release): success={}",
        p5.is_some()
    );

    // Drop all remaining permits
    drop(p2);
    drop(p3);
    drop(p5);
    println!(
        "After dropping all permits: available={}, active={}",
        bh.available(),
        bh.active()
    );

    println!(
        "Bulkhead JSON: {}",
        serde_json::to_string_pretty(&bh.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 5. FallbackChain
    // -----------------------------------------------------------------------
    println!("\n--- 5. FallbackChain ---");

    let mut chain = FallbackChain::new();
    chain.add_fallback("anthropic".into(), 1);
    chain.add_fallback("openai".into(), 2);
    chain.add_fallback("google".into(), 3);
    chain.add_fallback("ollama".into(), 4);

    println!("Providers: available={}", chain.available_count());
    println!("Next provider: {:?}", chain.next_fallback());

    // Simulate primary provider failure
    chain.mark_failed("anthropic");
    println!("\nAfter anthropic fails:");
    println!(
        "  available={}, next={:?}",
        chain.available_count(),
        chain.next_fallback()
    );

    // Second provider also fails
    chain.mark_failed("openai");
    println!("After openai also fails:");
    println!(
        "  available={}, next={:?}",
        chain.available_count(),
        chain.next_fallback()
    );

    // Primary recovers
    chain.mark_recovered("anthropic");
    println!("After anthropic recovers:");
    println!(
        "  available={}, next={:?}",
        chain.available_count(),
        chain.next_fallback()
    );

    // Mark all remaining as failed
    chain.mark_failed("anthropic");
    chain.mark_failed("google");
    chain.mark_failed("ollama");
    println!("After all fail:");
    println!(
        "  available={}, next={:?}",
        chain.available_count(),
        chain.next_fallback()
    );

    println!(
        "Chain JSON: {}",
        serde_json::to_string_pretty(&chain.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 6. ResiliencePolicy — combining patterns
    // -----------------------------------------------------------------------
    println!("\n--- 6. ResiliencePolicy ---");

    let policy = ResiliencePolicy::new("llm-api-policy")
        .with_retry(
            RetryConfig::new(
                3,
                RetryStrategy::Exponential {
                    base_ms: 200,
                    max_ms: 5000,
                    multiplier: 2.0,
                },
            )
            .with_retryable_error("timeout")
            .with_retryable_error("rate_limit"),
        )
        .with_circuit_breaker(CircuitBreaker::new(5, 30000))
        .with_bulkhead(Bulkhead::new(10));

    println!(
        "Policy '{}': can_execute={}",
        policy.name,
        policy.can_execute()
    );

    // Simulate successful calls
    policy.record_success();
    policy.record_success();
    println!("After 2 successes: can_execute={}", policy.can_execute());

    // Simulate failures
    for i in 1..=5 {
        policy.record_failure(&format!("error_{}", i));
        println!("After failure {}: can_execute={}", i, policy.can_execute());
    }

    // Check retry config
    if let Some(ref retry) = policy.retry_config {
        println!("\nRetry config: max_attempts={}", retry.max_attempts);
        println!(
            "  should_retry(0, 'timeout')={}",
            retry.should_retry(0, "timeout")
        );
        println!(
            "  should_retry(0, 'auth_error')={}",
            retry.should_retry(0, "auth_error")
        );
    }

    println!(
        "\nPolicy JSON: {}",
        serde_json::to_string_pretty(&policy.to_json()).unwrap()
    );

    // -----------------------------------------------------------------------
    // 7. ResilienceMetrics
    // -----------------------------------------------------------------------
    println!("\n--- 7. ResilienceMetrics ---");

    let metrics = ResilienceMetrics::new();

    // Simulate a series of operations
    // Operation 1: succeeds on first try
    metrics.record_attempt();
    metrics.record_success();

    // Operation 2: succeeds after 2 retries
    metrics.record_attempt();
    metrics.record_retry(1);
    metrics.record_retry(2);
    metrics.record_success();

    // Operation 3: circuit breaker opens
    metrics.record_attempt();
    metrics.record_circuit_open();

    // Operation 4: bulkhead rejects
    metrics.record_bulkhead_reject();

    // Operation 5: succeeds on first try
    metrics.record_attempt();
    metrics.record_success();

    println!("Total attempts: {}", metrics.total_attempts());
    println!("Total retries: {}", metrics.total_retries());
    println!("Circuit opens: {}", metrics.circuit_opens());
    println!("Bulkhead rejects: {}", metrics.bulkhead_rejects());
    println!("Success rate: {:.2}%", metrics.success_rate() * 100.0);

    println!(
        "\nMetrics JSON: {}",
        serde_json::to_string_pretty(&metrics.to_json()).unwrap()
    );

    println!("\n=== Resilience Patterns Demo Complete ===");
}
