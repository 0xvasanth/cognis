//! Caching Demo
//!
//! Shows how to cache LLM responses using `InMemoryCache` to avoid
//! redundant API calls. Demonstrates cache misses, hits, and stats.
//!
//! Run with: `cargo run -p cognis-examples --example caching_demo`

#[path = "../shared.rs"]
mod shared;

use cognis::caching::{CacheEntry, CacheKey, CacheStats, CacheStore, InMemoryCache};
use cognis_core::messages::Message;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LLM Response Caching Demo ===\n");

    let model = shared::get_chat_model(vec![
        "Rust uses ownership and borrowing to guarantee memory safety at compile time.".into(),
        "Rust uses ownership and borrowing to guarantee memory safety at compile time.".into(),
    ]);

    let mut cache = InMemoryCache::new(100);
    let mut stats = CacheStats::new();

    let prompt = "Explain Rust ownership in one sentence.";
    let cache_key = CacheKey::from_parts(
        "demo_model",
        &[json!({"role": "user", "content": prompt})],
        Some(0.3),
        None,
    );

    // --- First call: expect a cache miss, so we call the LLM ---
    println!("Query: \"{}\"\n", prompt);

    let response_text = match cache.get(&cache_key) {
        Some(entry) => {
            entry.record_hit();
            stats.record_hit();
            println!("[CACHE HIT]");
            entry.response.as_str().unwrap_or_default().to_string()
        }
        None => {
            stats.record_miss();
            println!("[CACHE MISS] Calling LLM...");
            let messages = vec![Message::human(prompt)];
            let result = model._generate(&messages, None).await?;
            let text = result.generations[0].message.content().text();

            let entry = CacheEntry::new(json!(text), "demo_model").with_ttl_secs(600);
            cache.put(cache_key.clone(), entry);
            stats.record_insertion();

            text
        }
    };
    println!("Response: {}\n", response_text);

    // --- Second call: same query, should be a cache hit ---
    let response_text = match cache.get(&cache_key) {
        Some(entry) => {
            entry.record_hit();
            stats.record_hit();
            println!("[CACHE HIT] No LLM call needed.");
            entry.response.as_str().unwrap_or_default().to_string()
        }
        None => {
            stats.record_miss();
            println!("[CACHE MISS] Calling LLM...");
            let messages = vec![Message::human(prompt)];
            let result = model._generate(&messages, None).await?;
            let text = result.generations[0].message.content().text();

            let entry = CacheEntry::new(json!(text), "demo_model").with_ttl_secs(600);
            cache.put(cache_key.clone(), entry);
            stats.record_insertion();

            text
        }
    };
    println!("Response: {}\n", response_text);

    // --- A different query: expect another miss ---
    let other_prompt = "What is borrowing in Rust?";
    let other_key = CacheKey::from_parts(
        "demo_model",
        &[json!({"role": "user", "content": other_prompt})],
        Some(0.3),
        None,
    );

    match cache.get(&other_key) {
        Some(_) => println!("[CACHE HIT] for \"{}\"", other_prompt),
        None => {
            stats.record_miss();
            println!("[CACHE MISS] for \"{}\" (different query)", other_prompt);
        }
    }

    // --- Summary ---
    println!("\n--- Cache Statistics ---");
    println!("  Hits:    {}", stats.total_hits());
    println!("  Misses:  {}", stats.total_misses());
    println!("  Hit rate: {:.0}%", stats.hit_rate() * 100.0);
    println!("  Entries: {}", cache.len());

    println!("\n=== Done ===");
    Ok(())
}
