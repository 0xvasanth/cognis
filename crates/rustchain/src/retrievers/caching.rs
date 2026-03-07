//! Caching retriever that wraps another retriever and caches results to avoid
//! redundant lookups.
//!
//! Supports TTL-based expiration, LRU eviction when the cache reaches a
//! configurable maximum size, and optional query normalization.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::RwLock;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use rustchain_core::retrievers::BaseRetriever;

/// A single cache entry storing retrieved documents and access metadata.
#[derive(Debug)]
pub struct CacheEntry {
    /// The cached documents.
    pub documents: Vec<Document>,
    /// When this entry was created.
    pub created_at: Instant,
    /// When this entry was last accessed.
    pub last_accessed: Instant,
    /// Number of cache hits for this entry.
    pub hit_count: AtomicUsize,
}

impl CacheEntry {
    fn new(documents: Vec<Document>) -> Self {
        let now = Instant::now();
        Self {
            documents,
            created_at: now,
            last_accessed: now,
            hit_count: AtomicUsize::new(0),
        }
    }

    fn is_expired(&self, ttl: Duration) -> bool {
        self.created_at.elapsed() > ttl
    }
}

/// Configuration for the caching retriever.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries in the cache (default: 1000).
    pub max_entries: usize,
    /// Time-to-live for cache entries (default: 5 minutes).
    pub ttl: Duration,
    /// Whether to normalize queries before cache lookup (lowercase + trim).
    pub normalize_queries: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl: Duration::from_secs(300),
            normalize_queries: true,
        }
    }
}

impl CacheConfig {
    /// Create a new config with the given max entries.
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Create a new config with the given TTL.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = ttl;
        self
    }

    /// Create a new config with query normalization enabled or disabled.
    pub fn with_normalize_queries(mut self, normalize: bool) -> Self {
        self.normalize_queries = normalize;
        self
    }
}

/// Statistics about cache usage.
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// Total number of cache hits.
    pub hits: usize,
    /// Total number of cache misses.
    pub misses: usize,
    /// Total number of evictions (TTL or LRU).
    pub evictions: usize,
    /// Current number of entries in the cache.
    pub size: usize,
}

/// A retriever that caches results from an inner retriever to avoid redundant
/// lookups.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::retrievers::caching::{CachingRetriever, CacheConfig};
/// use std::sync::Arc;
/// use std::time::Duration;
///
/// let config = CacheConfig::default().with_ttl(Duration::from_secs(60));
/// let caching = CachingRetriever::new(inner_retriever, config);
/// let docs = caching.get_relevant_documents("query").await?;
/// ```
pub struct CachingRetriever {
    /// The wrapped retriever.
    inner: Arc<dyn BaseRetriever>,
    /// Thread-safe cache storage.
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Cache configuration.
    config: CacheConfig,
    /// Running statistics.
    stats: Arc<RwLock<CacheStats>>,
}

impl CachingRetriever {
    /// Create a new caching retriever wrapping the given inner retriever.
    pub fn new(inner: Arc<dyn BaseRetriever>, config: CacheConfig) -> Self {
        Self {
            inner,
            cache: Arc::new(RwLock::new(HashMap::new())),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// Create a new caching retriever with default configuration.
    pub fn with_defaults(inner: Arc<dyn BaseRetriever>) -> Self {
        Self::new(inner, CacheConfig::default())
    }

    /// Normalize a query string (lowercase, trim whitespace).
    fn normalize_query(&self, query: &str) -> String {
        if self.config.normalize_queries {
            query.trim().to_lowercase()
        } else {
            query.to_string()
        }
    }

    /// Invalidate (remove) a specific cache entry.
    pub async fn invalidate(&self, query: &str) {
        let key = self.normalize_query(query);
        let mut cache = self.cache.write().await;
        if cache.remove(&key).is_some() {
            let mut stats = self.stats.write().await;
            stats.evictions += 1;
        }
    }

    /// Clear all entries from the cache.
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;
        stats.evictions += cache.len();
        cache.clear();
    }

    /// Return current cache statistics.
    pub async fn cache_stats(&self) -> CacheStats {
        let stats = self.stats.read().await;
        let cache = self.cache.read().await;
        CacheStats {
            hits: stats.hits,
            misses: stats.misses,
            evictions: stats.evictions,
            size: cache.len(),
        }
    }

    /// Evict all expired entries from the cache.
    async fn evict_expired(&self) {
        let mut cache = self.cache.write().await;
        let mut stats = self.stats.write().await;
        let before = cache.len();
        cache.retain(|_, entry| !entry.is_expired(self.config.ttl));
        let removed = before - cache.len();
        stats.evictions += removed;
    }

    /// Evict the least recently used entry when the cache exceeds max_entries.
    async fn evict_lru(&self) {
        let mut cache = self.cache.write().await;
        if cache.len() <= self.config.max_entries {
            return;
        }

        let mut stats = self.stats.write().await;
        while cache.len() > self.config.max_entries {
            // Find the entry with the oldest last_accessed time.
            if let Some(lru_key) = cache
                .iter()
                .min_by_key(|(_, entry)| entry.last_accessed)
                .map(|(key, _)| key.clone())
            {
                cache.remove(&lru_key);
                stats.evictions += 1;
            } else {
                break;
            }
        }
    }
}

#[async_trait]
impl BaseRetriever for CachingRetriever {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
        let key = self.normalize_query(query);

        // Check cache (read lock).
        {
            let cache = self.cache.read().await;
            if let Some(entry) = cache.get(&key) {
                if !entry.is_expired(self.config.ttl) {
                    entry.hit_count.fetch_add(1, Ordering::Relaxed);
                    let docs = entry.documents.clone();
                    drop(cache);
                    let mut stats = self.stats.write().await;
                    stats.hits += 1;
                    return Ok(docs);
                }
            }
        }

        // Cache miss — query the inner retriever.
        let docs = self.inner.get_relevant_documents(query).await?;

        // Evict expired entries before inserting.
        self.evict_expired().await;

        // Insert into cache.
        {
            let mut cache = self.cache.write().await;
            cache.insert(key, CacheEntry::new(docs.clone()));
        }

        // Evict LRU if over capacity.
        self.evict_lru().await;

        // Record miss.
        {
            let mut stats = self.stats.write().await;
            stats.misses += 1;
        }

        Ok(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize as StdAtomicUsize;

    /// A mock retriever that counts invocations.
    struct CountingRetriever {
        docs: Vec<Document>,
        call_count: StdAtomicUsize,
    }

    impl CountingRetriever {
        fn new(docs: Vec<Document>) -> Self {
            Self {
                docs,
                call_count: StdAtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.call_count.load(Ordering::Relaxed)
        }
    }

    #[async_trait]
    impl BaseRetriever for CountingRetriever {
        async fn get_relevant_documents(&self, _query: &str) -> Result<Vec<Document>> {
            self.call_count.fetch_add(1, Ordering::Relaxed);
            Ok(self.docs.clone())
        }
    }

    fn make_docs(contents: &[&str]) -> Vec<Document> {
        contents.iter().map(|c| Document::new(*c)).collect()
    }

    #[tokio::test]
    async fn test_cache_hit_avoids_inner_call() {
        let inner = Arc::new(CountingRetriever::new(make_docs(&["doc1"])));
        let caching = CachingRetriever::with_defaults(inner.clone());

        let docs1 = caching.get_relevant_documents("hello").await.unwrap();
        let docs2 = caching.get_relevant_documents("hello").await.unwrap();

        assert_eq!(docs1, docs2);
        assert_eq!(inner.calls(), 1); // Only called once.
    }

    #[tokio::test]
    async fn test_different_queries_cause_separate_calls() {
        let inner = Arc::new(CountingRetriever::new(make_docs(&["doc1"])));
        let caching = CachingRetriever::with_defaults(inner.clone());

        caching.get_relevant_documents("query_a").await.unwrap();
        caching.get_relevant_documents("query_b").await.unwrap();

        assert_eq!(inner.calls(), 2);
    }

    #[tokio::test]
    async fn test_query_normalization_lowercase() {
        let inner = Arc::new(CountingRetriever::new(make_docs(&["doc1"])));
        let config = CacheConfig::default().with_normalize_queries(true);
        let caching = CachingRetriever::new(inner.clone(), config);

        caching.get_relevant_documents("Hello World").await.unwrap();
        caching.get_relevant_documents("hello world").await.unwrap();

        assert_eq!(inner.calls(), 1);
    }

    #[tokio::test]
    async fn test_query_normalization_trim() {
        let inner = Arc::new(CountingRetriever::new(make_docs(&["doc1"])));
        let config = CacheConfig::default().with_normalize_queries(true);
        let caching = CachingRetriever::new(inner.clone(), config);

        caching.get_relevant_documents("  hello  ").await.unwrap();
        caching.get_relevant_documents("hello").await.unwrap();

        assert_eq!(inner.calls(), 1);
    }

    #[tokio::test]
    async fn test_normalization_disabled() {
        let inner = Arc::new(CountingRetriever::new(make_docs(&["doc1"])));
        let config = CacheConfig::default().with_normalize_queries(false);
        let caching = CachingRetriever::new(inner.clone(), config);

        caching.get_relevant_documents("Hello").await.unwrap();
        caching.get_relevant_documents("hello").await.unwrap();

        assert_eq!(inner.calls(), 2); // Not normalized, so different keys.
    }

    #[tokio::test]
    async fn test_ttl_expiration() {
        let inner = Arc::new(CountingRetriever::new(make_docs(&["doc1"])));
        let config = CacheConfig::default().with_ttl(Duration::from_millis(50));
        let caching = CachingRetriever::new(inner.clone(), config);

        caching.get_relevant_documents("query").await.unwrap();
        assert_eq!(inner.calls(), 1);

        // Wait for TTL to expire.
        tokio::time::sleep(Duration::from_millis(100)).await;

        caching.get_relevant_documents("query").await.unwrap();
        assert_eq!(inner.calls(), 2); // Cache expired, called again.
    }

    #[tokio::test]
    async fn test_invalidate_removes_entry() {
        let inner = Arc::new(CountingRetriever::new(make_docs(&["doc1"])));
        let caching = CachingRetriever::with_defaults(inner.clone());

        caching.get_relevant_documents("query").await.unwrap();
        assert_eq!(inner.calls(), 1);

        caching.invalidate("query").await;

        caching.get_relevant_documents("query").await.unwrap();
        assert_eq!(inner.calls(), 2);
    }

    #[tokio::test]
    async fn test_clear_cache() {
        let inner = Arc::new(CountingRetriever::new(make_docs(&["doc1"])));
        let caching = CachingRetriever::with_defaults(inner.clone());

        caching.get_relevant_documents("a").await.unwrap();
        caching.get_relevant_documents("b").await.unwrap();
        assert_eq!(inner.calls(), 2);

        caching.clear_cache().await;

        caching.get_relevant_documents("a").await.unwrap();
        caching.get_relevant_documents("b").await.unwrap();
        assert_eq!(inner.calls(), 4);
    }

    #[tokio::test]
    async fn test_cache_stats_hits_and_misses() {
        let inner = Arc::new(CountingRetriever::new(make_docs(&["doc1"])));
        let caching = CachingRetriever::with_defaults(inner.clone());

        caching.get_relevant_documents("a").await.unwrap(); // miss
        caching.get_relevant_documents("a").await.unwrap(); // hit
        caching.get_relevant_documents("b").await.unwrap(); // miss
        caching.get_relevant_documents("a").await.unwrap(); // hit

        let stats = caching.cache_stats().await;
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.size, 2);
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let inner = Arc::new(CountingRetriever::new(make_docs(&["doc1"])));
        let config = CacheConfig::default().with_max_entries(2);
        let caching = CachingRetriever::new(inner.clone(), config);

        caching.get_relevant_documents("a").await.unwrap();
        caching.get_relevant_documents("b").await.unwrap();
        caching.get_relevant_documents("c").await.unwrap(); // Should evict "a"

        let stats = caching.cache_stats().await;
        assert_eq!(stats.size, 2);
        assert!(stats.evictions >= 1);
    }

    #[tokio::test]
    async fn test_cache_returns_correct_documents() {
        let inner = Arc::new(CountingRetriever::new(make_docs(&["alpha", "beta"])));
        let caching = CachingRetriever::with_defaults(inner.clone());

        let docs = caching.get_relevant_documents("test").await.unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].page_content, "alpha");
        assert_eq!(docs[1].page_content, "beta");

        // Second call should return same results from cache.
        let docs2 = caching.get_relevant_documents("test").await.unwrap();
        assert_eq!(docs, docs2);
    }

    #[tokio::test]
    async fn test_empty_results_are_cached() {
        let inner = Arc::new(CountingRetriever::new(vec![]));
        let caching = CachingRetriever::with_defaults(inner.clone());

        let docs1 = caching.get_relevant_documents("empty").await.unwrap();
        let docs2 = caching.get_relevant_documents("empty").await.unwrap();

        assert!(docs1.is_empty());
        assert!(docs2.is_empty());
        assert_eq!(inner.calls(), 1); // Still cached even though empty.
    }

    #[tokio::test]
    async fn test_invalidate_nonexistent_key_is_noop() {
        let inner = Arc::new(CountingRetriever::new(make_docs(&["doc1"])));
        let caching = CachingRetriever::with_defaults(inner.clone());

        // Should not panic or error.
        caching.invalidate("nonexistent").await;
        let stats = caching.cache_stats().await;
        assert_eq!(stats.evictions, 0);
    }

    #[tokio::test]
    async fn test_evict_expired_cleans_up() {
        let inner = Arc::new(CountingRetriever::new(make_docs(&["doc1"])));
        let config = CacheConfig::default().with_ttl(Duration::from_millis(50));
        let caching = CachingRetriever::new(inner.clone(), config);

        caching.get_relevant_documents("a").await.unwrap();
        caching.get_relevant_documents("b").await.unwrap();

        tokio::time::sleep(Duration::from_millis(100)).await;

        // Trigger eviction by inserting a new entry.
        caching.get_relevant_documents("c").await.unwrap();

        let stats = caching.cache_stats().await;
        // "a" and "b" should be evicted, only "c" remains.
        assert_eq!(stats.size, 1);
        assert!(stats.evictions >= 2);
    }
}
