//! Caching System Demo
//!
//! Demonstrates the caching system from `cognis::caching`:
//! - CacheKey creation from model/messages/temperature
//! - InMemoryCache for storing and retrieving cached entries
//! - TTL expiration behavior
//! - CachePolicy for controlling when caching is applied
//! - CacheStats for tracking hit/miss rates
//! - SemanticCache with embedding similarity matching
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example caching_demo`

mod shared;
use cognis::caching::{
    CacheEntry, CacheKey, CachePolicy, CacheStats, CacheStore, CacheWarmer, InMemoryCache,
    SemanticCache,
};
use cognis_core::messages::Message;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Caching System Demo ===\n");

    // -----------------------------------------------------------------------
    // 1. Creating CacheKeys
    // -----------------------------------------------------------------------
    println!("--- 1. Creating CacheKeys ---\n");

    let messages = vec![json!({"role": "user", "content": "What is Rust?"})];

    let key1 = CacheKey::from_parts("gpt-4", &messages, Some(0.7), None);
    println!("Key 1 (gpt-4, temp=0.7): {}", key1);
    println!("Key 1 hash: {}", key1.hash_hex());

    let key2 = CacheKey::from_parts("gpt-4", &messages, Some(0.7), None);
    println!("\nKey 2 (same params): {}", key2);
    println!("Key 2 hash: {}", key2.hash_hex());
    println!("Keys are equal: {}", key1 == key2);

    let key3 = CacheKey::from_parts("claude-3", &messages, Some(0.5), None);
    println!("\nKey 3 (different model/temp): {}", key3);
    println!("Key 3 hash: {}", key3.hash_hex());
    println!("Key 1 == Key 3: {}", key1 == key3);

    let tools = vec![json!({"name": "calculator", "description": "Math tool"})];
    let key_with_tools = CacheKey::from_parts("gpt-4", &messages, None, Some(&tools));
    println!("\nKey with tools: {}", key_with_tools);

    // -----------------------------------------------------------------------
    // 2. InMemoryCache: Store and Retrieve
    // -----------------------------------------------------------------------
    println!("\n--- 2. InMemoryCache: Store and Retrieve ---\n");

    let mut cache = InMemoryCache::new(5);

    let response1 = json!({
        "text": "Rust is a systems programming language.",
        "usage": {"total_tokens": 150}
    });
    cache.put(key1.clone(), CacheEntry::new(response1.clone(), "gpt-4"));
    println!("Stored response for key1");
    println!("Cache size: {}", cache.len());
    println!("Cache contains key1: {}", cache.contains(&key1));
    println!("Cache contains key3: {}", cache.contains(&key3));

    if let Some(entry) = cache.get(&key1) {
        entry.record_hit();
        println!("\nCache HIT for key1:");
        println!("  Model: {}", entry.model);
        println!("  Response: {}", entry.response);
        println!("  Hit count: {}", entry.hit_count);
        println!("  Age (seconds): {}", entry.age_secs());
    }

    // Second hit on same key
    if let Some(entry) = cache.get(&key1) {
        entry.record_hit();
        println!("  Hit count after second access: {}", entry.hit_count);
    }

    // -----------------------------------------------------------------------
    // 3. TTL Expiration Behavior
    // -----------------------------------------------------------------------
    println!("\n--- 3. TTL Expiration Behavior ---\n");

    let long_lived = CacheEntry::new(json!("long-lived response"), "gpt-4").with_ttl_secs(3600);
    println!("Long-lived entry (TTL=3600s):");
    println!("  Expires at: {:?}", long_lived.expires_at);
    println!("  Is expired: {}", long_lived.is_expired());

    let mut already_expired = CacheEntry::new(json!("expired response"), "gpt-4");
    // Simulate an entry that expired in the past
    already_expired.expires_at = Some("0".to_string());
    // Force the internal epoch to 0 so is_expired() returns true
    println!("\nAlready-expired entry:");
    println!(
        "  Is expired (no expiry set internally): {}",
        already_expired.is_expired()
    );

    // Use with_ttl_secs(0) — expires immediately at creation epoch
    let zero_ttl = CacheEntry::new(json!("zero-ttl response"), "gpt-4").with_ttl_secs(0);
    println!("\nZero-TTL entry:");
    println!("  Is expired: {}", zero_ttl.is_expired());

    // Default TTL on cache
    let mut ttl_cache = InMemoryCache::new(10).with_default_ttl(60);
    let ttl_key = CacheKey::from_parts("m", &[json!("hello")], None, None);
    ttl_cache.put(ttl_key.clone(), CacheEntry::new(json!("auto-ttl"), "m"));
    if let Some(entry) = ttl_cache.get(&ttl_key) {
        println!("\nCache with default TTL=60s:");
        println!("  Entry expires_at: {:?}", entry.expires_at);
    }

    // -----------------------------------------------------------------------
    // 4. LRU Eviction
    // -----------------------------------------------------------------------
    println!("\n--- 4. LRU Eviction ---\n");

    let mut small_cache = InMemoryCache::new(3);
    for i in 0..5 {
        let key = CacheKey::from_parts("m", &[json!(i)], None, None);
        small_cache.put(key, CacheEntry::new(json!(format!("response {}", i)), "m"));
        println!("Inserted entry {}, cache size: {}", i, small_cache.len());
    }
    println!("Final cache size (max 3): {}", small_cache.len());

    // -----------------------------------------------------------------------
    // 5. CachePolicy
    // -----------------------------------------------------------------------
    println!("\n--- 5. CachePolicy ---\n");

    let policy = CachePolicy::new()
        .cache_if_tokens_above(50)
        .skip_if_streaming()
        .skip_if_tools_used()
        .max_ttl_secs(300);

    println!("Policy settings: {}", policy.to_json());

    let req_normal = json!({"model": "gpt-4"});
    let resp_high_tokens = json!({"usage": {"total_tokens": 200}});
    let resp_low_tokens = json!({"usage": {"total_tokens": 10}});
    let req_streaming = json!({"model": "gpt-4", "stream": true});
    let req_with_tools = json!({"model": "gpt-4", "tools": [{"name": "calc"}]});

    println!(
        "Normal request, high tokens: should_cache = {}",
        policy.should_cache(&req_normal, &resp_high_tokens)
    );
    println!(
        "Normal request, low tokens:  should_cache = {}",
        policy.should_cache(&req_normal, &resp_low_tokens)
    );
    println!(
        "Streaming request:           should_cache = {}",
        policy.should_cache(&req_streaming, &resp_high_tokens)
    );
    println!(
        "Request with tools:          should_cache = {}",
        policy.should_cache(&req_with_tools, &resp_high_tokens)
    );

    // Permissive policy
    let permissive = CachePolicy::default();
    println!(
        "\nPermissive policy, any request: should_cache = {}",
        permissive.should_cache(&json!({}), &json!({}))
    );

    // -----------------------------------------------------------------------
    // 6. CacheStats
    // -----------------------------------------------------------------------
    println!("\n--- 6. CacheStats ---\n");

    let mut stats = CacheStats::new();

    // Simulate a series of cache lookups
    stats.record_miss();
    stats.record_insertion();
    println!("After first miss + insertion:");

    stats.record_hit();
    stats.record_saved_tokens(150);
    println!("After first hit (saved 150 tokens):");

    stats.record_hit();
    stats.record_saved_tokens(200);
    println!("After second hit (saved 200 tokens):");

    stats.record_miss();
    stats.record_insertion();
    println!("After second miss + insertion:");

    stats.record_hit();
    stats.record_saved_tokens(100);

    stats.record_eviction();

    println!("\nFinal stats:");
    println!("  Total hits:     {}", stats.total_hits());
    println!("  Total misses:   {}", stats.total_misses());
    println!("  Total lookups:  {}", stats.total_lookups());
    println!("  Hit rate:       {:.1}%", stats.hit_rate() * 100.0);
    println!("  Evictions:      {}", stats.total_evictions());
    println!("  Saved tokens:   {}", stats.estimated_savings_tokens());
    println!("  Stats JSON:     {}", stats.to_json());

    // -----------------------------------------------------------------------
    // 7. SemanticCache with Embedding Similarity
    // -----------------------------------------------------------------------
    println!("\n--- 7. SemanticCache with Embedding Similarity ---\n");

    let mut semantic_cache = SemanticCache::new(0.90);

    // Simulate embeddings for cached queries
    semantic_cache.put(
        "What is Rust?".to_string(),
        vec![0.9, 0.1, 0.0, 0.05],
        CacheEntry::new(json!("Rust is a systems programming language."), "gpt-4"),
    );
    semantic_cache.put(
        "Explain Python".to_string(),
        vec![0.1, 0.9, 0.0, 0.05],
        CacheEntry::new(
            json!("Python is a high-level interpreted language."),
            "gpt-4",
        ),
    );
    semantic_cache.put(
        "What is machine learning?".to_string(),
        vec![0.0, 0.1, 0.9, 0.05],
        CacheEntry::new(json!("ML is a subset of artificial intelligence."), "gpt-4"),
    );

    println!("Semantic cache size: {}", semantic_cache.len());

    // Query with a very similar embedding to "What is Rust?"
    let query_embedding = vec![0.88, 0.12, 0.01, 0.04];
    println!("\nQuery: 'Tell me about Rust' (similar to 'What is Rust?')");
    if let Some(entry) = semantic_cache.get_similar(&query_embedding) {
        println!("  SEMANTIC HIT: {}", entry.response);
    } else {
        println!("  No semantic match found");
    }

    // Query with a very similar embedding to "Explain Python"
    let python_query = vec![0.12, 0.88, 0.02, 0.04];
    println!("\nQuery: 'How does Python work?' (similar to 'Explain Python')");
    if let Some(entry) = semantic_cache.get_similar(&python_query) {
        println!("  SEMANTIC HIT: {}", entry.response);
    } else {
        println!("  No semantic match found");
    }

    // Query with an orthogonal embedding (no match expected)
    let unrelated_query = vec![0.0, 0.0, 0.0, 1.0];
    println!("\nQuery: 'Something completely different' (orthogonal)");
    if let Some(entry) = semantic_cache.get_similar(&unrelated_query) {
        println!("  SEMANTIC HIT: {}", entry.response);
    } else {
        println!("  No semantic match found (as expected)");
    }

    // -----------------------------------------------------------------------
    // 8. CacheWarmer
    // -----------------------------------------------------------------------
    println!("\n--- 8. CacheWarmer ---\n");

    let mut warmer = CacheWarmer::new();
    warmer.add_query(
        CacheKey::from_parts(
            "gpt-4",
            &[json!({"role": "user", "content": "Hello"})],
            None,
            None,
        ),
        CacheEntry::new(json!("Hi there!"), "gpt-4"),
    );
    warmer.add_query(
        CacheKey::from_parts(
            "gpt-4",
            &[json!({"role": "user", "content": "Help"})],
            None,
            None,
        ),
        CacheEntry::new(json!("How can I help?"), "gpt-4"),
    );
    println!("Warmer queued queries: {}", warmer.queries_count());

    let mut warm_cache = InMemoryCache::new(100);
    let warmed = warmer.warm(&mut warm_cache);
    println!("Warmed {} entries into cache", warmed);
    println!("Cache size after warming: {}", warm_cache.len());
    println!("Warmer remaining queries: {}", warmer.queries_count());

    // -----------------------------------------------------------------------
    // 9. Full Round-Trip Integration
    // -----------------------------------------------------------------------
    println!("\n--- 9. Full Round-Trip Integration ---\n");

    let mut store = InMemoryCache::new(100);
    let mut stats = CacheStats::new();
    let policy = CachePolicy::new().cache_if_tokens_above(10);

    let query_key = CacheKey::from_parts(
        "claude-3",
        &[json!({"role": "user", "content": "Explain ownership in Rust"})],
        Some(0.7),
        None,
    );
    let request = json!({"model": "claude-3"});
    let response = json!({
        "text": "Ownership is Rust's central feature for memory safety...",
        "usage": {"total_tokens": 250}
    });

    // First lookup: miss
    if store.get(&query_key).is_none() {
        stats.record_miss();
        println!("First lookup: MISS");

        if policy.should_cache(&request, &response) {
            let entry = CacheEntry::new(response.clone(), "claude-3").with_ttl_secs(600);
            store.put(query_key.clone(), entry);
            stats.record_insertion();
            println!("  Cached response (TTL=600s)");
        }
    }

    // Second lookup: hit
    if let Some(entry) = store.get(&query_key) {
        entry.record_hit();
        stats.record_hit();
        stats.record_saved_tokens(250);
        println!("Second lookup: HIT");
        println!("  Saved 250 tokens");
        println!("  Entry JSON: {}", entry.to_json());
    }

    // Third lookup: hit
    if let Some(entry) = store.get(&query_key) {
        entry.record_hit();
        stats.record_hit();
        stats.record_saved_tokens(250);
        println!("Third lookup: HIT");
    }

    println!("\nFinal statistics:");
    println!("  {}", stats.to_json());

    // --- Real LLM Demo ---
    println!("\n--- Real LLM Demo ---\n");
    println!("Demonstrating caching with a real LLM call.\n");

    let model = shared::get_chat_model(vec![
        "Rust's ownership system ensures memory safety without garbage collection by enforcing strict rules at compile time.".into(),
    ]);

    let cache_key = CacheKey::from_parts(
        "shared_model",
        &[json!({"role": "user", "content": "Explain Rust ownership"})],
        Some(0.3),
        None,
    );

    // First call: cache miss, call the LLM
    let mut demo_cache = InMemoryCache::new(100);
    let mut demo_stats = CacheStats::new();

    if demo_cache.get(&cache_key).is_none() {
        demo_stats.record_miss();
        println!("Cache MISS - calling LLM...");
        let messages = vec![Message::human("Explain Rust ownership in one sentence.")];
        let result = model._generate(&messages, None).await?;
        if let Some(gen) = result.generations.first() {
            let response_text = gen.message.content().text();
            println!("LLM Response: {}", response_text);
            let entry = CacheEntry::new(json!(response_text), "shared_model").with_ttl_secs(600);
            demo_cache.put(cache_key.clone(), entry);
            demo_stats.record_insertion();
            println!("Response cached.");
        }
    }

    // Second call: cache hit
    if let Some(entry) = demo_cache.get(&cache_key) {
        entry.record_hit();
        demo_stats.record_hit();
        println!("\nCache HIT: {}", entry.response);
        println!("Saved an LLM call!");
    }

    println!("Cache stats: {}", demo_stats.to_json());

    println!("\n=== Caching Demo Complete ===");
    Ok(())
}
