//! Caching Retriever Example
//!
//! Demonstrates CachingRetriever: cache hits/misses, query normalization,
//! invalidation, TTL expiration, and LRU eviction.

#[path = "../shared.rs"]
mod shared;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;

use cognis::retrievers::caching::{CacheConfig, CachingRetriever};
use cognis_core::documents::Document;
use cognis_core::error::Result;
use cognis_core::messages::Message;
use cognis_core::retrievers::BaseRetriever;

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
    async fn get_relevant_documents(&self, _query: &str) -> Result<Vec<Document>> {
        self.call_count.fetch_add(1, Ordering::Relaxed);
        Ok(self.documents.clone())
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let docs = vec![
        Document::new("Rust was first released in 2010."),
        Document::new("Rust is maintained by the Rust Foundation."),
    ];

    // 1. Basic cache hits and misses
    let inner = Arc::new(MockRetriever::new(docs.clone()));
    let config = CacheConfig::default()
        .with_max_entries(100)
        .with_ttl(Duration::from_secs(60));
    let caching = CachingRetriever::new(inner.clone(), config);

    let _r = caching.get_relevant_documents("What is Rust?").await?;
    let _r = caching.get_relevant_documents("What is Rust?").await?;
    let stats = caching.cache_stats().await;
    println!(
        "Basic: hits={}, misses={}, inner_calls={}",
        stats.hits,
        stats.misses,
        inner.calls()
    );

    // 2. Query normalization
    let inner2 = Arc::new(MockRetriever::new(vec![Document::new("Normalized.")]));
    let caching2 = CachingRetriever::new(
        inner2.clone(),
        CacheConfig::default().with_normalize_queries(true),
    );
    let _r = caching2.get_relevant_documents("  Hello World  ").await?;
    let _r = caching2.get_relevant_documents("hello world").await?;
    let stats = caching2.cache_stats().await;
    println!(
        "Normalization: hits={}, misses={}, inner_calls={}",
        stats.hits,
        stats.misses,
        inner2.calls()
    );

    // 3. Cache invalidation
    let inner3 = Arc::new(MockRetriever::new(vec![Document::new("Test.")]));
    let caching3 = CachingRetriever::with_defaults(inner3.clone());
    let _r = caching3.get_relevant_documents("test").await?;
    caching3.invalidate("test").await;
    let _r = caching3.get_relevant_documents("test").await?;
    println!("Invalidation: inner_calls={} (expected 2)", inner3.calls());

    // 4. TTL expiration
    let inner4 = Arc::new(MockRetriever::new(vec![Document::new("TTL test.")]));
    let caching4 = CachingRetriever::new(
        inner4.clone(),
        CacheConfig::default().with_ttl(Duration::from_millis(200)),
    );
    let _r = caching4.get_relevant_documents("ttl-query").await?;
    let _r = caching4.get_relevant_documents("ttl-query").await?;
    tokio::time::sleep(Duration::from_millis(300)).await;
    let _r = caching4.get_relevant_documents("ttl-query").await?;
    println!(
        "TTL: inner_calls={} (expected 2, cache expired after 200ms)",
        inner4.calls()
    );

    // 5. LRU eviction
    let inner5 = Arc::new(MockRetriever::new(vec![Document::new("LRU test.")]));
    let caching5 =
        CachingRetriever::new(inner5.clone(), CacheConfig::default().with_max_entries(2));
    for q in ["query-a", "query-b", "query-c"] {
        let _r = caching5.get_relevant_documents(q).await?;
    }
    let stats = caching5.cache_stats().await;
    println!("LRU: size={}, evictions={}", stats.size, stats.evictions);

    // 6. LLM demo with cached retrieval
    let inner_llm = Arc::new(MockRetriever::new(docs));
    let caching_llm = CachingRetriever::with_defaults(inner_llm);
    let retrieved = caching_llm.get_relevant_documents("What is Rust?").await?;
    let context: String = retrieved
        .iter()
        .map(|d| d.page_content.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    let model = shared::get_chat_model(vec![
        "Rust was first released in 2010 and is maintained by the Rust Foundation.".into(),
    ]);
    let messages = vec![
        Message::system("Answer based on context only."),
        Message::human(&format!(
            "Context: {}\n\nQuestion: What do we know about Rust?",
            context
        )),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("LLM: {}", gen.message.content().text());
    }

    Ok(())
}
