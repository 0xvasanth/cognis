//! Embedding cache layer for avoiding redundant API calls.
//!
//! Provides [`CachedEmbeddings`], a wrapper around any [`Embeddings`] implementation
//! that caches results to avoid repeated calls for the same text inputs.

use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, VecDeque};
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock};

use async_trait::async_trait;

use cognis_core::embeddings::Embeddings;
use cognis_core::error::Result;

/// Statistics about cache performance.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: usize,
    /// Number of cache misses.
    pub misses: usize,
    /// Hit rate as a fraction (0.0 to 1.0). Returns 0.0 if no lookups have occurred.
    pub hit_rate: f64,
}

/// Trait for embedding caches.
///
/// Implementations store and retrieve embedding vectors keyed by a string
/// derived from the input text (typically a hash).
pub trait EmbeddingCache: Send + Sync {
    /// Retrieve a cached embedding by key.
    fn get(&self, key: &str) -> Option<Vec<f32>>;

    /// Store an embedding under the given key.
    fn put(&self, key: &str, embedding: Vec<f32>);

    /// Retrieve multiple cached embeddings at once.
    ///
    /// Returns a vector of the same length as `keys`, with `None` for cache misses.
    fn get_many(&self, keys: &[String]) -> Vec<Option<Vec<f32>>>;

    /// Store multiple embeddings at once.
    fn put_many(&self, entries: &[(String, Vec<f32>)]);

    /// Remove all entries from the cache.
    fn clear(&self);

    /// Return the number of entries currently in the cache.
    fn len(&self) -> usize;

    /// Return true if the cache is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Thread-safe in-memory embedding cache backed by a `HashMap`.
///
/// Supports optional LRU eviction when a `max_size` is set.
pub struct InMemoryEmbeddingCache {
    store: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    /// Insertion order for LRU eviction. Front = oldest.
    order: Arc<RwLock<VecDeque<String>>>,
    /// Maximum number of entries. `None` means unlimited.
    max_size: Option<usize>,
}

impl InMemoryEmbeddingCache {
    /// Create a new unbounded in-memory cache.
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
            order: Arc::new(RwLock::new(VecDeque::new())),
            max_size: None,
        }
    }

    /// Create a new in-memory cache with a maximum number of entries.
    ///
    /// When the cache exceeds this size, the least recently inserted entries
    /// are evicted.
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
            order: Arc::new(RwLock::new(VecDeque::new())),
            max_size: Some(max_size),
        }
    }

    /// Evict entries until the cache is at or below `max_size`.
    fn evict_if_needed(&self) {
        if let Some(max) = self.max_size {
            let mut store = self.store.write().unwrap();
            let mut order = self.order.write().unwrap();
            while store.len() > max {
                if let Some(oldest_key) = order.pop_front() {
                    store.remove(&oldest_key);
                } else {
                    break;
                }
            }
        }
    }
}

impl Default for InMemoryEmbeddingCache {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingCache for InMemoryEmbeddingCache {
    fn get(&self, key: &str) -> Option<Vec<f32>> {
        let store = self.store.read().unwrap();
        store.get(key).cloned()
    }

    fn put(&self, key: &str, embedding: Vec<f32>) {
        {
            let mut store = self.store.write().unwrap();
            let is_new = !store.contains_key(key);
            store.insert(key.to_string(), embedding);
            if is_new {
                let mut order = self.order.write().unwrap();
                order.push_back(key.to_string());
            }
        }
        self.evict_if_needed();
    }

    fn get_many(&self, keys: &[String]) -> Vec<Option<Vec<f32>>> {
        let store = self.store.read().unwrap();
        keys.iter().map(|k| store.get(k).cloned()).collect()
    }

    fn put_many(&self, entries: &[(String, Vec<f32>)]) {
        {
            let mut store = self.store.write().unwrap();
            let mut order = self.order.write().unwrap();
            for (key, embedding) in entries {
                let is_new = !store.contains_key(key);
                store.insert(key.clone(), embedding.clone());
                if is_new {
                    order.push_back(key.clone());
                }
            }
        }
        self.evict_if_needed();
    }

    fn clear(&self) {
        let mut store = self.store.write().unwrap();
        let mut order = self.order.write().unwrap();
        store.clear();
        order.clear();
    }

    fn len(&self) -> usize {
        let store = self.store.read().unwrap();
        store.len()
    }
}

/// Compute a deterministic cache key from text content using hashing.
pub fn cache_key(text: &str) -> String {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Wrapper around an [`Embeddings`] implementation that caches results.
///
/// On each call to `embed_documents` or `embed_query`, the cache is consulted
/// first. Only texts not found in the cache are forwarded to the inner embeddings
/// provider. Results are cached for future use.
///
/// # Example
///
/// ```no_run
/// use cognis::embeddings::cached::{CachedEmbeddings, InMemoryEmbeddingCache};
///
/// # fn example(inner: Box<dyn cognis_core::embeddings::Embeddings>) {
/// let cached = CachedEmbeddings::new(
///     inner,
///     Box::new(InMemoryEmbeddingCache::new()),
/// );
/// # }
/// ```
pub struct CachedEmbeddings {
    inner: Box<dyn Embeddings>,
    cache: Box<dyn EmbeddingCache>,
    hits: AtomicUsize,
    misses: AtomicUsize,
}

impl CachedEmbeddings {
    /// Create a new cached embeddings wrapper.
    pub fn new(inner: Box<dyn Embeddings>, cache: Box<dyn EmbeddingCache>) -> Self {
        Self {
            inner,
            cache,
            hits: AtomicUsize::new(0),
            misses: AtomicUsize::new(0),
        }
    }

    /// Return current cache performance statistics.
    pub fn cache_stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::Relaxed);
        let misses = self.misses.load(Ordering::Relaxed);
        let total = hits + misses;
        let hit_rate = if total == 0 {
            0.0
        } else {
            hits as f64 / total as f64
        };
        CacheStats {
            hits,
            misses,
            hit_rate,
        }
    }

    /// Reset cache statistics counters to zero.
    pub fn reset_stats(&self) {
        self.hits.store(0, Ordering::Relaxed);
        self.misses.store(0, Ordering::Relaxed);
    }

    /// Clear the cache and reset statistics.
    pub fn clear(&self) {
        self.cache.clear();
        self.reset_stats();
    }

    /// Return a reference to the underlying cache.
    pub fn cache(&self) -> &dyn EmbeddingCache {
        self.cache.as_ref()
    }
}

#[async_trait]
impl Embeddings for CachedEmbeddings {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Build cache keys for all texts
        let keys: Vec<String> = texts.iter().map(|t| cache_key(t)).collect();
        let cached = self.cache.get_many(&keys);

        // Identify misses
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_texts: Vec<String> = Vec::new();
        for (i, entry) in cached.iter().enumerate() {
            if entry.is_none() {
                miss_indices.push(i);
                miss_texts.push(texts[i].clone());
            }
        }

        let hit_count = texts.len() - miss_indices.len();
        self.hits.fetch_add(hit_count, Ordering::Relaxed);
        self.misses.fetch_add(miss_indices.len(), Ordering::Relaxed);

        // Fetch embeddings for misses from the inner provider
        let miss_embeddings = if miss_texts.is_empty() {
            Vec::new()
        } else {
            self.inner.embed_documents(miss_texts).await?
        };

        // Cache the new embeddings
        let new_entries: Vec<(String, Vec<f32>)> = miss_indices
            .iter()
            .zip(miss_embeddings.iter())
            .map(|(&idx, emb)| (keys[idx].clone(), emb.clone()))
            .collect();
        if !new_entries.is_empty() {
            self.cache.put_many(&new_entries);
        }

        // Assemble the full result in original order
        let mut results: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        let mut miss_iter = miss_embeddings.into_iter();
        for entry in cached {
            match entry {
                Some(emb) => results.push(emb),
                None => results.push(miss_iter.next().unwrap()),
            }
        }

        Ok(results)
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let key = cache_key(text);
        if let Some(cached) = self.cache.get(&key) {
            self.hits.fetch_add(1, Ordering::Relaxed);
            return Ok(cached);
        }

        self.misses.fetch_add(1, Ordering::Relaxed);
        let embedding = self.inner.embed_query(text).await?;
        self.cache.put(&key, embedding.clone());
        Ok(embedding)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::embeddings_fake::DeterministicFakeEmbedding;
    use std::sync::Arc;

    /// Helper: create a CachedEmbeddings with DeterministicFakeEmbedding and unbounded cache.
    fn make_cached(size: usize) -> CachedEmbeddings {
        CachedEmbeddings::new(
            Box::new(DeterministicFakeEmbedding::new(size)),
            Box::new(InMemoryEmbeddingCache::new()),
        )
    }

    /// Helper: create a CachedEmbeddings with a bounded cache.
    fn make_cached_bounded(size: usize, max_cache: usize) -> CachedEmbeddings {
        CachedEmbeddings::new(
            Box::new(DeterministicFakeEmbedding::new(size)),
            Box::new(InMemoryEmbeddingCache::with_max_size(max_cache)),
        )
    }

    #[tokio::test]
    async fn test_cache_miss_calls_inner() {
        let cached = make_cached(8);

        let result = cached
            .embed_documents(vec!["hello".to_string()])
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 8);

        let stats = cached.cache_stats();
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 0);
    }

    #[tokio::test]
    async fn test_cache_hit_returns_cached_value() {
        let cached = make_cached(8);

        // First call: miss
        let first = cached
            .embed_documents(vec!["hello".to_string()])
            .await
            .unwrap();

        // Second call: hit
        let second = cached
            .embed_documents(vec!["hello".to_string()])
            .await
            .unwrap();

        assert_eq!(first, second);

        let stats = cached.cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn test_mixed_hits_and_misses_in_batch() {
        let cached = make_cached(8);

        // Populate cache with "hello"
        cached
            .embed_documents(vec!["hello".to_string()])
            .await
            .unwrap();

        // Now request "hello" (hit) and "world" (miss)
        let results = cached
            .embed_documents(vec!["hello".to_string(), "world".to_string()])
            .await
            .unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].len(), 8);
        assert_eq!(results[1].len(), 8);

        let stats = cached.cache_stats();
        // First call: 1 miss. Second call: 1 hit + 1 miss.
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
    }

    #[tokio::test]
    async fn test_cache_stats_tracking() {
        let cached = make_cached(4);

        // 3 misses
        cached
            .embed_documents(vec!["a".to_string(), "b".to_string(), "c".to_string()])
            .await
            .unwrap();

        // 3 hits
        cached
            .embed_documents(vec!["a".to_string(), "b".to_string(), "c".to_string()])
            .await
            .unwrap();

        let stats = cached.cache_stats();
        assert_eq!(stats.hits, 3);
        assert_eq!(stats.misses, 3);
        assert!((stats.hit_rate - 0.5).abs() < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_lru_eviction_when_max_size_exceeded() {
        let cached = make_cached_bounded(4, 2);

        // Insert 3 items into a cache that holds 2
        cached.embed_documents(vec!["a".to_string()]).await.unwrap();
        cached.embed_documents(vec!["b".to_string()]).await.unwrap();
        cached.embed_documents(vec!["c".to_string()]).await.unwrap();

        // Cache should have only 2 entries
        assert_eq!(cached.cache().len(), 2);

        // "a" should have been evicted (oldest)
        let key_a = cache_key("a");
        assert!(cached.cache().get(&key_a).is_none());

        // "b" and "c" should still be present
        let key_b = cache_key("b");
        let key_c = cache_key("c");
        assert!(cached.cache().get(&key_b).is_some());
        assert!(cached.cache().get(&key_c).is_some());
    }

    #[tokio::test]
    async fn test_embed_query_caching() {
        let cached = make_cached(8);

        let first = cached.embed_query("test query").await.unwrap();
        let second = cached.embed_query("test query").await.unwrap();

        assert_eq!(first, second);

        let stats = cached.cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 1);
    }

    #[tokio::test]
    async fn test_clear_cache_resets_stats() {
        let cached = make_cached(4);

        cached.embed_query("foo").await.unwrap();
        cached.embed_query("foo").await.unwrap();

        assert_eq!(cached.cache().len(), 1);
        assert_eq!(cached.cache_stats().hits, 1);

        cached.clear();

        assert_eq!(cached.cache().len(), 0);
        let stats = cached.cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[tokio::test]
    async fn test_thread_safety_concurrent_access() {
        let cached = Arc::new(make_cached(8));

        let mut handles = Vec::new();
        for i in 0..10 {
            let cached_clone = Arc::clone(&cached);
            handles.push(tokio::spawn(async move {
                let text = format!("text_{}", i);
                cached_clone.embed_query(&text).await.unwrap()
            }));
        }

        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await.unwrap());
        }

        assert_eq!(results.len(), 10);
        // All should be cached now
        assert_eq!(cached.cache().len(), 10);

        let stats = cached.cache_stats();
        assert_eq!(stats.misses, 10);
        assert_eq!(stats.hits, 0);
    }

    #[tokio::test]
    async fn test_empty_input_handling() {
        let cached = make_cached(8);

        let result = cached.embed_documents(vec![]).await.unwrap();
        assert!(result.is_empty());

        let stats = cached.cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[tokio::test]
    async fn test_cache_key_consistency() {
        // Same text should always produce the same cache key
        let key1 = cache_key("consistent text");
        let key2 = cache_key("consistent text");
        assert_eq!(key1, key2);

        // Different text should produce different keys
        let key3 = cache_key("different text");
        assert_ne!(key1, key3);
    }

    #[tokio::test]
    async fn test_embed_query_and_documents_share_cache() {
        let cached = make_cached(8);

        // embed_query populates cache
        let query_result = cached.embed_query("shared text").await.unwrap();

        // embed_documents should hit cache for same text
        let doc_results = cached
            .embed_documents(vec!["shared text".to_string()])
            .await
            .unwrap();

        assert_eq!(query_result, doc_results[0]);

        let stats = cached.cache_stats();
        // 1 miss from embed_query, 1 hit from embed_documents
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.hits, 1);
    }

    #[test]
    fn test_in_memory_cache_is_empty() {
        let cache = InMemoryEmbeddingCache::new();
        assert!(cache.is_empty());
        cache.put("key", vec![1.0, 2.0]);
        assert!(!cache.is_empty());
    }

    #[test]
    fn test_cache_stats_zero_lookups() {
        let cached = make_cached(4);
        let stats = cached.cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.hit_rate, 0.0);
    }
}
