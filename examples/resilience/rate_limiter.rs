//! Rate Limiter Example
//!
//! Shows how to rate-limit LLM API calls using a token bucket
//! combined with a sliding window limiter and usage tracking.
//!
//! Run with: `cargo run -p cognis-examples --example rate_limiter`

#[path = "../shared.rs"]
mod shared;

use cognis_core::messages::Message;
use cognisagent::rate_limiting::{
    CompositeLimiter, RateLimitPolicy, SlidingWindowLimiter, TimeWindow, TokenBucket, UsageTracker,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Rate-Limited LLM Calls ===\n");

    // -- Build a composite rate limiter ------------------------------------------
    // Token bucket: allow bursts of 3 calls, refilling at 1/sec
    // Sliding window: max 10 requests per minute
    let mut limiter = CompositeLimiter::new();
    limiter.add_limiter(Box::new(TokenBucket::new(3.0, 1.0)));

    let window = SlidingWindowLimiter::new(RateLimitPolicy::Reject);
    window.add_window(TimeWindow::PerMinute, 10);
    limiter.add_limiter(Box::new(window));

    // -- Set up usage tracking and the chat model --------------------------------
    let tracker = UsageTracker::new();
    let model = shared::get_chat_model(vec![
        "Rust is a systems programming language focused on safety and performance.".into(),
        "Async/await lets you write non-blocking code that reads like synchronous code.".into(),
        "Traits define shared behavior, similar to interfaces in other languages.".into(),
        "Ownership ensures memory safety without a garbage collector.".into(),
    ]);

    // -- Send prompts, respecting the rate limit ---------------------------------
    let prompts = [
        "What is Rust?",
        "What is async/await?",
        "What are traits?",
        "What is ownership?",
    ];

    for prompt in &prompts {
        let check = limiter.check_and_record();
        if check.allowed {
            let messages = vec![Message::human(*prompt)];
            let response = model.invoke_messages(&messages, None).await?;
            tracker.record(50, 0.01, Some("llama3.2"));
            println!("Q: {}", prompt);
            println!("A: {}\n", response.base.content.text());
        } else {
            let reason = check.reason.as_deref().unwrap_or("rate limited");
            println!("Q: {} => THROTTLED ({})\n", prompt, reason);
        }
    }

    // -- Print usage summary -----------------------------------------------------
    let report = tracker.report();
    println!(
        "Usage: {} requests, {} tokens, ${:.4} total cost",
        report.total_requests, report.total_tokens, report.total_cost
    );

    println!("\n=== Done ===");
    Ok(())
}
