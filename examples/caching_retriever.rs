//! Caching Retriever Example
//!
//! Demonstrates the CachingRetriever that wraps another retriever and caches
//! results to avoid redundant lookups. Shows cache hits, misses, stats,
//! invalidation, TTL configuration, and LRU eviction.
//!
//! No API keys required -- uses a mock retriever.
//!
//! Run with: `cargo run -p cognis-examples --example caching_retriever`

mod shared;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use cognis::retrievers::caching::{CacheConfig, CachingRetriever};
use cognis_core::documents::Document;
use cognis_core::error::Result;
use cognis_core::retrievers::BaseRetriever;

/// A mock retriever that returns fixed documents and counts how many times
/// the underlying retrieval was actually called.
struct MockRetriever {
    documents: Vec<Document>,
    call_count: AtomicUsize,
}

impl MockRetriever {
    fn new(docs: Vec<Document>) -> Self {
        Self {
            documents: docs,
            call_count: AtomicUsize::new(0),
        }
    }

    fn calls(&self) -> usize {
        self.call_count.load(Ordering::Relaxed)
    }
}

#[async_trait]
impl BaseRetriever for MockRetriever {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        println!("    [MockRetriever] Actually fetching for query: \"{query}\"");
        Ok(self.documents.clone())
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Caching Retriever Example ===\n");

    // -----------------------------------------------------------------------
    // 1. Basic cache hits and misses
    // -----------------------------------------------------------------------
    println!("--- 1. Basic Cache Hits and Misses ---\n");

    let docs = vec![
        Document::new("Rust was first released in 2010."),
        Document::new("Rust is maintained by the Rust Foundation."),
    ];
    let inner = Arc::new(MockRetriever::new(docs));
    let config = CacheConfig::default()
        .with_max_entries(100)
        .with_ttl(Duration::from_secs(60));
    let caching = CachingRetriever::new(inner.clone(), config);

    // First call: cache miss
    println!("  Query 1: \"What is Rust?\" (should be a cache miss)");
    let result = caching.get_relevant_documents("What is Rust?").await?;
    println!("    Returned {} documents", result.len());
    let stats = caching.cache_stats().await;
    println!(
        "    Stats: hits={}, misses={}, size={}\n",
        stats.hits, stats.misses, stats.size
    );

    // Second call with same query: cache hit
    println!("  Query 2: \"What is Rust?\" (should be a cache hit)");
    let _result = caching.get_relevant_documents("What is Rust?").await?;
    let stats = caching.cache_stats().await;
    println!(
        "    Stats: hits={}, misses={}, size={}",
        stats.hits, stats.misses, stats.size
    );
    println!("    Inner retriever was called {} time(s)\n", inner.calls());

    // Different query: cache miss
    println!("  Query 3: \"Tell me about Python\" (different query, cache miss)");
    let _result = caching
        .get_relevant_documents("Tell me about Python")
        .await?;
    let stats = caching.cache_stats().await;
    println!(
        "    Stats: hits={}, misses={}, size={}",
        stats.hits, stats.misses, stats.size
    );
    println!("    Inner retriever was called {} time(s)\n", inner.calls());

    // -----------------------------------------------------------------------
    // 2. Query normalization
    // -----------------------------------------------------------------------
    println!("--- 2. Query Normalization ---\n");

    let inner2 = Arc::new(MockRetriever::new(vec![Document::new(
        "Normalized result.",
    )]));
    let config2 = CacheConfig::default().with_normalize_queries(true);
    let caching2 = CachingRetriever::new(inner2.clone(), config2);

    println!("  Query: \"  Hello World  \" (with extra whitespace and caps)");
    let _r = caching2.get_relevant_documents("  Hello World  ").await?;
    let stats = caching2.cache_stats().await;
    println!("    Stats: hits={}, misses={}\n", stats.hits, stats.misses);

    println!("  Query: \"hello world\" (normalized should match)");
    let _r = caching2.get_relevant_documents("hello world").await?;
    let stats = caching2.cache_stats().await;
    println!("    Stats: hits={}, misses={}", stats.hits, stats.misses);
    println!(
        "    Inner retriever called {} time(s) (only once due to normalization)\n",
        inner2.calls()
    );

    // -----------------------------------------------------------------------
    // 3. Cache invalidation
    // -----------------------------------------------------------------------
    println!("--- 3. Cache Invalidation ---\n");

    let inner3 = Arc::new(MockRetriever::new(vec![Document::new(
        "Invalidation test.",
    )]));
    let caching3 = CachingRetriever::with_defaults(inner3.clone());

    println!("  Populating cache with query \"test\"...");
    let _r = caching3.get_relevant_documents("test").await?;
    let stats = caching3.cache_stats().await;
    println!("    Cache size: {}, misses: {}\n", stats.size, stats.misses);

    println!("  Invalidating entry for \"test\"...");
    caching3.invalidate("test").await;
    let stats = caching3.cache_stats().await;
    println!(
        "    Cache size: {}, evictions: {}\n",
        stats.size, stats.evictions
    );

    println!("  Querying \"test\" again (should be a miss after invalidation)...");
    let _r = caching3.get_relevant_documents("test").await?;
    let stats = caching3.cache_stats().await;
    println!(
        "    Stats: hits={}, misses={}, evictions={}",
        stats.hits, stats.misses, stats.evictions
    );
    println!("    Inner retriever called {} time(s)\n", inner3.calls());

    // Clear entire cache
    println!("  Clearing entire cache...");
    caching3.clear_cache().await;
    let stats = caching3.cache_stats().await;
    println!(
        "    Cache size: {}, evictions: {}\n",
        stats.size, stats.evictions
    );

    // -----------------------------------------------------------------------
    // 4. TTL expiration
    // -----------------------------------------------------------------------
    println!("--- 4. TTL Expiration ---\n");

    let inner4 = Arc::new(MockRetriever::new(vec![Document::new("TTL test.")]));
    let config4 = CacheConfig::default().with_ttl(Duration::from_millis(200));
    let caching4 = CachingRetriever::new(inner4.clone(), config4);

    println!("  TTL set to 200ms");
    println!("  Query: \"ttl-query\" (miss)");
    let _r = caching4.get_relevant_documents("ttl-query").await?;
    println!("    Inner calls: {}", inner4.calls());

    println!("  Query again immediately (hit)");
    let _r = caching4.get_relevant_documents("ttl-query").await?;
    println!("    Inner calls: {}", inner4.calls());

    println!("  Sleeping 300ms to let TTL expire...");
    tokio::time::sleep(Duration::from_millis(300)).await;

    println!("  Query again after TTL expiry (miss)");
    let _r = caching4.get_relevant_documents("ttl-query").await?;
    println!(
        "    Inner calls: {} (called again because cache expired)\n",
        inner4.calls()
    );

    // -----------------------------------------------------------------------
    // 5. LRU eviction
    // -----------------------------------------------------------------------
    println!("--- 5. LRU Eviction ---\n");

    let inner5 = Arc::new(MockRetriever::new(vec![Document::new("LRU test.")]));
    let config5 = CacheConfig::default().with_max_entries(2);
    let caching5 = CachingRetriever::new(inner5.clone(), config5);

    println!("  Max cache entries: 2");

    println!("  Adding: \"query-a\"");
    let _r = caching5.get_relevant_documents("query-a").await?;

    println!("  Adding: \"query-b\"");
    let _r = caching5.get_relevant_documents("query-b").await?;

    let stats = caching5.cache_stats().await;
    println!(
        "    Cache size: {}, evictions: {}",
        stats.size, stats.evictions
    );

    println!("  Adding: \"query-c\" (should trigger eviction of oldest)");
    let _r = caching5.get_relevant_documents("query-c").await?;

    let stats = caching5.cache_stats().await;
    println!(
        "    Cache size: {}, evictions: {}",
        stats.size, stats.evictions
    );
    println!("    Total inner calls: {}\n", inner5.calls());

    println!("=== Done ===");
    Ok(())
}
