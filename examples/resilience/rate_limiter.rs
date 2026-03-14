//! Rate Limiter Example
//!
//! Demonstrates the advanced rate limiting system from `cognisagent::rate_limiting`:
//! - `TokenBucket` — classic token bucket with capacity and refill rate
//! - `SlidingWindowLimiter` — per-second, per-minute, and per-hour windows
//! - `CostBasedLimiter` — API budget tracking with per-model costs
//! - `CompositeLimiter` — combining multiple limiters
//! - `QuotaManager` — per-model and per-provider quotas
//! - `UsageTracker` and `UsageReport` — consumption monitoring
//! - Rate-limited LLM calls using the chat model
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example rate_limiter`

#[path = "../shared.rs"]
mod shared;

use std::time::Duration;

use cognisagent::rate_limiting::{
    CompositeLimiter, CostBasedLimiter, QuotaManager, RateLimitPolicy, SlidingWindowLimiter,
    TimeWindow, TokenBucket, UsageTracker,
};

use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Rate Limiter Example ===\n");

    // -----------------------------------------------------------------------
    // 1. TokenBucket — classic token bucket rate limiter
    // -----------------------------------------------------------------------
    println!("--- 1. TokenBucket ---");
    println!("Tokens refill continuously at a fixed rate up to a capacity.\n");

    let bucket = TokenBucket::new(5.0, 2.0); // 5 capacity, 2 tokens/sec refill
    println!("  Capacity: {}", bucket.capacity());
    println!("  Refill rate: {} tokens/sec", bucket.refill_rate());
    println!("  Available: {:.1}", bucket.available());

    // Acquire tokens
    for i in 1..=7 {
        let result = bucket.try_acquire(1.0);
        println!(
            "  Request {}: allowed={}, remaining={:.1}{}",
            i,
            result.allowed,
            result.remaining_tokens,
            if !result.allowed {
                format!(", wait={:?}", result.wait_time)
            } else {
                String::new()
            }
        );
    }

    // Show wait time calculation
    let wait = bucket.wait_time_for(3.0);
    println!("  Wait time for 3 tokens: {:?}", wait);

    // Reset and verify
    bucket.reset();
    println!("  After reset, available: {:.1}\n", bucket.available());

    // -----------------------------------------------------------------------
    // 2. SlidingWindowLimiter — multi-window request tracking
    // -----------------------------------------------------------------------
    println!("--- 2. SlidingWindowLimiter ---");
    println!("Track requests across per-second, per-minute, and per-hour windows.\n");

    let window_limiter = SlidingWindowLimiter::new(RateLimitPolicy::Reject);
    window_limiter.add_window(TimeWindow::PerSecond, 3); // max 3 per second
    window_limiter.add_window(TimeWindow::PerMinute, 10); // max 10 per minute

    println!("  Windows configured: PerSecond(3), PerMinute(10)");
    for i in 1..=5 {
        let result = window_limiter.check_and_record();
        println!(
            "  Request {}: allowed={}, remaining={:.0}{}",
            i,
            result.allowed,
            result.remaining_tokens,
            if let Some(ref reason) = result.reason {
                format!(", reason={}", reason)
            } else {
                String::new()
            }
        );
    }

    println!(
        "  Current count (PerSecond): {}",
        window_limiter.count_for(TimeWindow::PerSecond)
    );
    println!(
        "  Current count (PerMinute): {}",
        window_limiter.count_for(TimeWindow::PerMinute)
    );

    // Reset and verify
    window_limiter.reset();
    println!(
        "  After reset, PerSecond count: {}\n",
        window_limiter.count_for(TimeWindow::PerSecond)
    );

    // -----------------------------------------------------------------------
    // 3. CostBasedLimiter — API budget management
    // -----------------------------------------------------------------------
    println!("--- 3. CostBasedLimiter ---");
    println!("Track API costs against a budget cap with per-model pricing.\n");

    let cost_limiter = CostBasedLimiter::new(
        1.00, // $1.00 budget
        0.01, // $0.01 default cost per call
        RateLimitPolicy::Reject,
    );
    cost_limiter.set_model_cost("gpt-4", 0.03);
    cost_limiter.set_model_cost("gpt-3.5", 0.002);
    cost_limiter.set_model_cost("claude-sonnet", 0.015);

    println!("  Budget: ${:.2}", cost_limiter.budget());
    println!(
        "  GPT-4 cost: ${:.3}/call",
        cost_limiter.cost_for_model("gpt-4")
    );
    println!(
        "  GPT-3.5 cost: ${:.4}/call",
        cost_limiter.cost_for_model("gpt-3.5")
    );
    println!(
        "  Unknown model cost: ${:.3}/call (default)",
        cost_limiter.cost_for_model("unknown")
    );

    // Simulate API calls
    let models = ["gpt-4", "gpt-3.5", "claude-sonnet", "gpt-4", "gpt-3.5"];
    for model in &models {
        let cost = cost_limiter.cost_for_model(model);
        let result = cost_limiter.check_and_record(cost);
        println!(
            "  Call to {}: allowed={}, budget remaining=${:.4}",
            model, result.allowed, result.cost_remaining
        );
    }
    println!("  Total spent: ${:.4}", cost_limiter.total_spent());
    println!(
        "  Remaining budget: ${:.4}\n",
        cost_limiter.remaining_budget()
    );

    // -----------------------------------------------------------------------
    // 4. CompositeLimiter — combining multiple limiters
    // -----------------------------------------------------------------------
    println!("--- 4. CompositeLimiter ---");
    println!("Combine multiple limiters — requests pass only if ALL approve.\n");

    let mut composite = CompositeLimiter::new();

    // Add a token bucket (3 capacity, 1/sec refill)
    composite.add_limiter(Box::new(TokenBucket::new(3.0, 1.0)));

    // Add a sliding window (5 per minute)
    let sw = SlidingWindowLimiter::new(RateLimitPolicy::Reject);
    sw.add_window(TimeWindow::PerMinute, 5);
    composite.add_limiter(Box::new(sw));

    println!("  Sub-limiters: {}", composite.len());

    for i in 1..=5 {
        let result = composite.check_and_record();
        println!(
            "  Request {}: allowed={}, tokens={:.1}, cost_rem={:.1}{}",
            i,
            result.allowed,
            result.remaining_tokens,
            result.cost_remaining,
            if let Some(ref reason) = result.reason {
                format!(", reason={}", reason)
            } else {
                String::new()
            }
        );
    }

    composite.reset_all();
    println!(
        "  After reset, check: allowed={}\n",
        composite.check_all().allowed
    );

    // -----------------------------------------------------------------------
    // 5. QuotaManager — per-model and per-provider quotas
    // -----------------------------------------------------------------------
    println!("--- 5. QuotaManager ---");
    println!("Enforce per-model or per-provider quotas with windowed resets.\n");

    let quotas = QuotaManager::new(Duration::from_secs(60));
    quotas.set_quota("gpt-4", 5, 0.50); // 5 requests, $0.50 max per minute
    quotas.set_quota("gpt-3.5", 20, 0.10); // 20 requests, $0.10 max per minute
    quotas.set_quota("claude-sonnet", 10, 0.30);

    println!("  Configured quotas: {:?}", quotas.keys());

    // Use GPT-4 quota
    for i in 1..=6 {
        let result = quotas.check_and_record("gpt-4", 0.03);
        println!(
            "  gpt-4 call {}: allowed={}, remaining_tokens={:.0}, cost_remaining={:.2}{}",
            i,
            result.allowed,
            result.remaining_tokens,
            result.cost_remaining,
            if let Some(ref reason) = result.reason {
                format!(", reason={}", reason)
            } else {
                String::new()
            }
        );
    }

    if let Some((reqs, cost)) = quotas.usage("gpt-4") {
        println!("  gpt-4 usage: {} requests, ${:.4} cost", reqs, cost);
    }

    // Reset GPT-4 quota
    quotas.reset("gpt-4");
    if let Some((reqs, cost)) = quotas.usage("gpt-4") {
        println!("  After reset: {} requests, ${:.4} cost\n", reqs, cost);
    }

    // -----------------------------------------------------------------------
    // 6. UsageTracker and UsageReport — consumption monitoring
    // -----------------------------------------------------------------------
    println!("--- 6. UsageTracker & UsageReport ---");
    println!("Record and report on token/cost consumption over time.\n");

    let tracker = UsageTracker::new();

    // Simulate a series of LLM calls with varying token counts and costs
    let calls = vec![
        (150u64, 0.003, "gpt-3.5"),
        (500, 0.015, "gpt-4"),
        (200, 0.004, "gpt-3.5"),
        (800, 0.024, "gpt-4"),
        (300, 0.009, "claude-sonnet"),
        (100, 0.002, "gpt-3.5"),
        (600, 0.018, "gpt-4"),
        (250, 0.0075, "claude-sonnet"),
    ];

    for (tokens, cost, model) in &calls {
        tracker.record(*tokens, *cost, Some(model));
    }

    println!("  Total requests: {}", tracker.total_requests());
    println!("  Total tokens: {}", tracker.total_tokens());
    println!("  Total cost: ${:.4}", tracker.total_cost());

    let report = tracker.report();
    println!("\n  Usage Report:");
    println!("    Period: {:?}", report.period);
    println!("    Average RPS: {:.2}", report.average_rps);
    println!("    Peak RPS: {:.2}", report.peak_rps);
    println!(
        "    Avg tokens/request: {:.1}",
        report.average_tokens_per_request
    );
    println!(
        "    Avg cost/request: ${:.4}",
        report.average_cost_per_request
    );

    println!("\n    Cost breakdown by model:");
    let mut cost_entries: Vec<_> = report.cost_breakdown.iter().collect();
    cost_entries.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (model, cost) in &cost_entries {
        let reqs = report.request_breakdown.get(model.as_str()).unwrap_or(&0);
        println!("      {}: ${:.4} ({} requests)", model, cost, reqs);
    }

    // -----------------------------------------------------------------------
    // 7. Rate-limited LLM calls
    // -----------------------------------------------------------------------
    println!("\n--- 7. Rate-Limited LLM Calls ---");
    println!("Demonstrate rate limiting applied to actual chat model invocations.\n");

    let model = shared::get_chat_model(vec![
        "I am the first response within the rate limit.".to_string(),
        "I am the second response within the rate limit.".to_string(),
        "I am the third response within the rate limit.".to_string(),
        "Rate limit analysis: The token bucket pattern is effective for burst control. \
         Combined with sliding windows for sustained rate enforcement and cost-based \
         limits for budget management, you get a robust multi-layered rate limiting \
         strategy suitable for production LLM applications."
            .to_string(),
    ]);

    // Set up rate limiting for model calls
    let model_bucket = TokenBucket::new(3.0, 1.0); // 3 calls burst, 1/sec refill
    let usage = UsageTracker::new();

    let prompts = vec![
        "What is Rust?",
        "What is async/await?",
        "What are traits?",
        "What is ownership?", // this one may be rate-limited
    ];

    for prompt in &prompts {
        let check = model_bucket.try_acquire(1.0);
        if check.allowed {
            let messages = vec![Message::human(*prompt)];
            let response = model.invoke_messages(&messages, None).await?;
            usage.record(50, 0.01, Some("llama3.2"));
            println!("  Q: {}", prompt);
            println!(
                "  A: {} (tokens remaining: {:.0})\n",
                response.base.content.text(),
                check.remaining_tokens
            );
        } else {
            println!(
                "  Q: {} => RATE LIMITED (wait {:?})\n",
                prompt, check.wait_time
            );
        }
    }

    // Final usage report
    let final_report = usage.report();
    println!("  Final usage summary:");
    println!("    Requests made: {}", final_report.total_requests);
    println!("    Tokens consumed: {}", final_report.total_tokens);
    println!("    Total cost: ${:.4}", final_report.total_cost);

    // Ask the model to analyze the rate limiting strategy
    println!("\n--- 8. LLM Analysis of Rate Limiting ---");

    let analysis_prompt = format!(
        "We have a rate limiting setup with:\n\
         - Token bucket: 3 capacity, 1/sec refill\n\
         - Sliding window: 3/sec, 10/min\n\
         - Cost budget: $1.00 with per-model costs\n\
         - {} requests processed, {} rate-limited\n\n\
         Briefly analyze this rate limiting strategy.",
        final_report.total_requests,
        prompts.len() as u64 - final_report.total_requests
    );

    let messages = vec![Message::human(analysis_prompt)];
    let response = model.invoke_messages(&messages, None).await?;
    println!("  {}\n", response.base.content.text());

    println!("=== Rate Limiter Example Complete ===");
    Ok(())
}
