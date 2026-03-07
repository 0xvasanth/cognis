//! Embedding cache layer with eviction policies and statistics.
//!
//! Provides a configurable embedding cache that avoids redundant API calls
//! for the same text. Supports LRU, LFU, FIFO, and TTL eviction policies,
//! cache hit/miss tracking, and a [`CachedEmbeddingProvider`] wrapper that
//! integrates caching with embedding computation.

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use serde_json::Value;

// ---------------------------------------------------------------------------
// EmbeddingCacheKey
// ---------------------------------------------------------------------------

/// A hashable key combining text content and model name.
#[derive(Debug, Clone)]
pub struct EmbeddingCacheKey {
    text: String,
    model: String,
}

impl EmbeddingCacheKey {
    /// Create a new cache key from text and model name.
    pub fn new(text: &str, model: &str) -> Self {
        Self {
            text: text.to_string(),
            model: model.to_string(),
        }
    }

    /// Return the text portion of this key.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Return the model name portion of this key.
    pub fn model(&self) -> &str {
        &self.model
    }
}

impl PartialEq for EmbeddingCacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.text == other.text && self.model == other.model
    }
}

impl Eq for EmbeddingCacheKey {}

impl Hash for EmbeddingCacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.text.hash(state);
        self.model.hash(state);
    }
}

// ---------------------------------------------------------------------------
// CachedEmbedding
// ---------------------------------------------------------------------------

/// A cached embedding vector with metadata for eviction decisions.
#[derive(Debug, Clone)]
pub struct CachedEmbedding {
    /// The embedding vector.
    pub vector: Vec<f32>,
    /// When this entry was created.
    pub created_at: Instant,
    /// How many times this entry has been accessed.
    pub access_count: u64,
    /// The model that produced this embedding.
    pub model: String,
}

impl CachedEmbedding {
    /// Create a new cached embedding.
    pub fn new(vector: Vec<f32>, model: &str) -> Self {
        Self {
            vector,
            created_at: Instant::now(),
            access_count: 0,
            model: model.to_string(),
        }
    }

    /// Return the time elapsed since this entry was created.
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Return `true` if this entry is older than `ttl`.
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.age() > ttl
    }

    /// Increment the access counter.
    pub fn touch(&mut self) {
        self.access_count += 1;
    }
}

// ---------------------------------------------------------------------------
// EvictionPolicy
// ---------------------------------------------------------------------------

/// Policy used to decide which entries to evict when the cache is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used — evict entries with the oldest last-access time.
    LRU,
    /// Least Frequently Used — evict entries with the lowest access count.
    LFU,
    /// First In, First Out — evict the oldest entries by creation time.
    FIFO,
    /// Time To Live — evict entries that have exceeded their TTL.
    TTL,
}

// ---------------------------------------------------------------------------
// EmbeddingCacheConfig
// ---------------------------------------------------------------------------

/// Configuration for [`EmbeddingCache`].
#[derive(Debug, Clone)]
pub struct EmbeddingCacheConfig {
    /// Maximum number of entries the cache will hold.
    pub max_entries: usize,
    /// Optional time-to-live for cache entries.
    pub ttl: Option<Duration>,
    /// Eviction policy when the cache exceeds `max_entries`.
    pub eviction: EvictionPolicy,
}

impl Default for EmbeddingCacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            ttl: None,
            eviction: EvictionPolicy::LRU,
        }
    }
}

impl EmbeddingCacheConfig {
    /// Set the maximum number of entries.
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Set the time-to-live for cache entries.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Set the eviction policy.
    pub fn with_eviction(mut self, eviction: EvictionPolicy) -> Self {
        self.eviction = eviction;
        self
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Aggregate statistics about cache usage.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total number of cache hits.
    pub hits: u64,
    /// Total number of cache misses.
    pub misses: u64,
    /// Total number of evictions performed.
    pub evictions: u64,
    /// Current number of entries in the cache.
    pub size: usize,
    /// Maximum number of entries allowed.
    pub max_size: usize,
}

impl CacheStats {
    /// Compute the hit rate as a value between 0.0 and 1.0.
    ///
    /// Returns 0.0 if no lookups have been recorded.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Serialize these statistics to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "size": self.size,
            "max_size": self.max_size,
            "hit_rate": self.hit_rate(),
        })
    }
}

// ---------------------------------------------------------------------------
// EmbeddingCache
// ---------------------------------------------------------------------------

/// Internal metadata for tracking access order / frequency.
#[derive(Debug, Clone)]
struct EntryMeta {
    /// Monotonically increasing counter assigned on last access (for LRU).
    last_access_order: u64,
    /// Monotonically increasing counter assigned on insertion (for FIFO).
    insertion_order: u64,
}

/// A configurable in-memory embedding cache with eviction policies and stats.
pub struct EmbeddingCache {
    config: EmbeddingCacheConfig,
    entries: HashMap<EmbeddingCacheKey, CachedEmbedding>,
    meta: HashMap<EmbeddingCacheKey, EntryMeta>,
    /// Global counter incremented on every access (used for LRU ordering).
    access_counter: u64,
    /// Global counter incremented on every insertion (used for FIFO ordering).
    insertion_counter: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl EmbeddingCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: EmbeddingCacheConfig) -> Self {
        Self {
            config,
            entries: HashMap::new(),
            meta: HashMap::new(),
            access_counter: 0,
            insertion_counter: 0,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Look up an embedding by key.
    ///
    /// On a hit the entry is touched and access metadata is updated.
    /// Returns `None` on a miss or if the entry has expired.
    pub fn get(&mut self, key: &EmbeddingCacheKey) -> Option<&CachedEmbedding> {
        // Check TTL expiration first.
        if let Some(ttl) = self.config.ttl {
            if let Some(entry) = self.entries.get(key) {
                if entry.is_expired(ttl) {
                    self.entries.remove(key);
                    self.meta.remove(key);
                    self.misses += 1;
                    return None;
                }
            }
        }

        if self.entries.contains_key(key) {
            self.hits += 1;
            self.access_counter += 1;
            let order = self.access_counter;

            // Update access metadata.
            if let Some(m) = self.meta.get_mut(key) {
                m.last_access_order = order;
            }
            if let Some(entry) = self.entries.get_mut(key) {
                entry.touch();
            }

            // Re-borrow immutably to return reference.
            self.entries.get(key)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert an embedding into the cache, evicting entries if necessary.
    pub fn put(&mut self, key: EmbeddingCacheKey, embedding: CachedEmbedding) {
        // If the key already exists, just replace.
        if self.entries.contains_key(&key) {
            self.entries.insert(key.clone(), embedding);
            return;
        }

        // Evict if at capacity.
        if self.entries.len() >= self.config.max_entries {
            self.evict();
        }

        self.insertion_counter += 1;
        self.access_counter += 1;
        let ins = self.insertion_counter;
        let acc = self.access_counter;

        self.meta.insert(
            key.clone(),
            EntryMeta {
                last_access_order: acc,
                insertion_order: ins,
            },
        );
        self.entries.insert(key, embedding);
    }

    /// Return `true` if the cache contains an entry for `key`.
    pub fn contains(&self, key: &EmbeddingCacheKey) -> bool {
        self.entries.contains_key(key)
    }

    /// Remove the entry for `key`, returning `true` if it was present.
    pub fn remove(&mut self, key: &EmbeddingCacheKey) -> bool {
        self.meta.remove(key);
        self.entries.remove(key).is_some()
    }

    /// Remove all entries from the cache (stats are preserved).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.meta.clear();
    }

    /// Return the number of entries currently stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Evict one entry according to the configured [`EvictionPolicy`].
    pub fn evict(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        let victim = match self.config.eviction {
            EvictionPolicy::LRU => {
                // Evict the entry with the smallest last_access_order.
                self.meta
                    .iter()
                    .min_by_key(|(_, m)| m.last_access_order)
                    .map(|(k, _)| k.clone())
            }
            EvictionPolicy::LFU => {
                // Evict the entry with the lowest access_count.
                self.entries
                    .iter()
                    .min_by_key(|(_, e)| e.access_count)
                    .map(|(k, _)| k.clone())
            }
            EvictionPolicy::FIFO => {
                // Evict the entry with the smallest insertion_order.
                self.meta
                    .iter()
                    .min_by_key(|(_, m)| m.insertion_order)
                    .map(|(k, _)| k.clone())
            }
            EvictionPolicy::TTL => {
                // Evict the oldest entry by creation time.
                self.entries
                    .iter()
                    .min_by_key(|(_, e)| e.created_at)
                    .map(|(k, _)| k.clone())
            }
        };

        if let Some(key) = victim {
            self.entries.remove(&key);
            self.meta.remove(&key);
            self.evictions += 1;
        }
    }

    /// Remove all entries whose TTL has expired, returning the count removed.
    pub fn cleanup_expired(&mut self) -> usize {
        let ttl = match self.config.ttl {
            Some(ttl) => ttl,
            None => return 0,
        };

        let expired_keys: Vec<EmbeddingCacheKey> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired(ttl))
            .map(|(k, _)| k.clone())
            .collect();

        let count = expired_keys.len();
        for key in expired_keys {
            self.entries.remove(&key);
            self.meta.remove(&key);
            self.evictions += 1;
        }
        count
    }

    /// Compute the cache hit rate as a value between 0.0 and 1.0.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Return a snapshot of cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            hits: self.hits,
            misses: self.misses,
            evictions: self.evictions,
            size: self.entries.len(),
            max_size: self.config.max_entries,
        }
    }
}

// ---------------------------------------------------------------------------
// CachedEmbeddingProvider
// ---------------------------------------------------------------------------

/// A simple embedding provider that wraps an [`EmbeddingCache`].
///
/// On a cache miss, a deterministic hash-based fake embedding is produced.
/// This is primarily useful for testing and demonstration purposes.
pub struct CachedEmbeddingProvider {
    model_name: String,
    cache: EmbeddingCache,
    /// Dimensionality of fake embeddings produced on cache miss.
    dimensions: usize,
}

impl CachedEmbeddingProvider {
    /// Create a new provider with the given model name and cache.
    ///
    /// Fake embeddings will have 8 dimensions by default.
    pub fn new(model_name: &str, cache: EmbeddingCache) -> Self {
        Self {
            model_name: model_name.to_string(),
            cache,
            dimensions: 8,
        }
    }

    /// Embed a single text, using the cache when possible.
    pub fn embed(&mut self, text: &str) -> Vec<f32> {
        let key = EmbeddingCacheKey::new(text, &self.model_name);

        if let Some(cached) = self.cache.get(&key) {
            return cached.vector.clone();
        }

        let vector = self.fake_embedding(text);
        let entry = CachedEmbedding::new(vector.clone(), &self.model_name);
        self.cache.put(key, entry);
        vector
    }

    /// Embed a batch of texts, looking up each in the cache first.
    pub fn embed_batch(&mut self, texts: &[&str]) -> Vec<Vec<f32>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Return a snapshot of cache statistics.
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.stats()
    }

    /// Remove the cache entry for a specific text.
    pub fn invalidate(&mut self, text: &str) {
        let key = EmbeddingCacheKey::new(text, &self.model_name);
        self.cache.remove(&key);
    }

    /// Produce a deterministic fake embedding from text content.
    fn fake_embedding(&self, text: &str) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.dimensions);
        for i in 0..self.dimensions {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            text.hash(&mut hasher);
            i.hash(&mut hasher);
            let h = hasher.finish();
            // Map hash to a value in [-1, 1].
            vector.push(((h % 20000) as f32 / 10000.0) - 1.0);
        }
        vector
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- EmbeddingCacheKey tests ----

    #[test]
    fn test_cache_key_new_and_getters() {
        let key = EmbeddingCacheKey::new("hello world", "model-v1");
        assert_eq!(key.text(), "hello world");
        assert_eq!(key.model(), "model-v1");
    }

    #[test]
    fn test_cache_key_equality_same() {
        let a = EmbeddingCacheKey::new("text", "model");
        let b = EmbeddingCacheKey::new("text", "model");
        assert_eq!(a, b);
    }

    #[test]
    fn test_cache_key_inequality_different_text() {
        let a = EmbeddingCacheKey::new("alpha", "model");
        let b = EmbeddingCacheKey::new("beta", "model");
        assert_ne!(a, b);
    }

    #[test]
    fn test_cache_key_inequality_different_model() {
        let a = EmbeddingCacheKey::new("text", "model-a");
        let b = EmbeddingCacheKey::new("text", "model-b");
        assert_ne!(a, b);
    }

    #[test]
    fn test_cache_key_hashing_consistent() {
        use std::collections::hash_map::DefaultHasher;
        let key = EmbeddingCacheKey::new("foo", "bar");
        let mut h1 = DefaultHasher::new();
        let mut h2 = DefaultHasher::new();
        key.hash(&mut h1);
        key.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish());
    }

    #[test]
    fn test_cache_key_hashing_different_keys() {
        use std::collections::hash_map::DefaultHasher;
        let a = EmbeddingCacheKey::new("foo", "bar");
        let b = EmbeddingCacheKey::new("baz", "bar");
        let mut ha = DefaultHasher::new();
        let mut hb = DefaultHasher::new();
        a.hash(&mut ha);
        b.hash(&mut hb);
        assert_ne!(ha.finish(), hb.finish());
    }

    #[test]
    fn test_cache_key_usable_as_hashmap_key() {
        let mut map = HashMap::new();
        let key = EmbeddingCacheKey::new("text", "model");
        map.insert(key.clone(), 42);
        assert_eq!(map.get(&EmbeddingCacheKey::new("text", "model")), Some(&42));
    }

    // ---- CachedEmbedding tests ----

    #[test]
    fn test_cached_embedding_new() {
        let emb = CachedEmbedding::new(vec![1.0, 2.0, 3.0], "test-model");
        assert_eq!(emb.vector, vec![1.0, 2.0, 3.0]);
        assert_eq!(emb.model, "test-model");
        assert_eq!(emb.access_count, 0);
    }

    #[test]
    fn test_cached_embedding_age_is_small() {
        let emb = CachedEmbedding::new(vec![1.0], "m");
        // Freshly created — age should be very small.
        assert!(emb.age() < Duration::from_secs(1));
    }

    #[test]
    fn test_cached_embedding_not_expired_without_wait() {
        let emb = CachedEmbedding::new(vec![1.0], "m");
        assert!(!emb.is_expired(Duration::from_secs(60)));
    }

    #[test]
    fn test_cached_embedding_expired_with_zero_ttl() {
        let emb = CachedEmbedding::new(vec![1.0], "m");
        // A TTL of zero means everything is expired.
        assert!(emb.is_expired(Duration::from_nanos(0)));
    }

    #[test]
    fn test_cached_embedding_touch_increments() {
        let mut emb = CachedEmbedding::new(vec![1.0], "m");
        assert_eq!(emb.access_count, 0);
        emb.touch();
        assert_eq!(emb.access_count, 1);
        emb.touch();
        emb.touch();
        assert_eq!(emb.access_count, 3);
    }

    // ---- EmbeddingCacheConfig tests ----

    #[test]
    fn test_config_defaults() {
        let cfg = EmbeddingCacheConfig::default();
        assert_eq!(cfg.max_entries, 10_000);
        assert!(cfg.ttl.is_none());
        assert_eq!(cfg.eviction, EvictionPolicy::LRU);
    }

    #[test]
    fn test_config_builder() {
        let cfg = EmbeddingCacheConfig::default()
            .with_max_entries(500)
            .with_ttl(Duration::from_secs(120))
            .with_eviction(EvictionPolicy::LFU);
        assert_eq!(cfg.max_entries, 500);
        assert_eq!(cfg.ttl, Some(Duration::from_secs(120)));
        assert_eq!(cfg.eviction, EvictionPolicy::LFU);
    }

    // ---- EmbeddingCache CRUD tests ----

    fn default_cache(max: usize) -> EmbeddingCache {
        EmbeddingCache::new(EmbeddingCacheConfig::default().with_max_entries(max))
    }

    #[test]
    fn test_cache_put_and_get() {
        let mut cache = default_cache(10);
        let key = EmbeddingCacheKey::new("hi", "m");
        let emb = CachedEmbedding::new(vec![1.0, 2.0], "m");
        cache.put(key.clone(), emb);

        let result = cache.get(&key).unwrap();
        assert_eq!(result.vector, vec![1.0, 2.0]);
    }

    #[test]
    fn test_cache_get_miss() {
        let mut cache = default_cache(10);
        let key = EmbeddingCacheKey::new("missing", "m");
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cache_contains() {
        let mut cache = default_cache(10);
        let key = EmbeddingCacheKey::new("x", "m");
        assert!(!cache.contains(&key));
        cache.put(key.clone(), CachedEmbedding::new(vec![1.0], "m"));
        assert!(cache.contains(&key));
    }

    #[test]
    fn test_cache_remove() {
        let mut cache = default_cache(10);
        let key = EmbeddingCacheKey::new("x", "m");
        cache.put(key.clone(), CachedEmbedding::new(vec![1.0], "m"));
        assert!(cache.remove(&key));
        assert!(!cache.contains(&key));
        assert!(!cache.remove(&key));
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = default_cache(10);
        cache.put(
            EmbeddingCacheKey::new("a", "m"),
            CachedEmbedding::new(vec![1.0], "m"),
        );
        cache.put(
            EmbeddingCacheKey::new("b", "m"),
            CachedEmbedding::new(vec![2.0], "m"),
        );
        assert_eq!(cache.len(), 2);
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn test_cache_len_and_is_empty() {
        let mut cache = default_cache(10);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        cache.put(
            EmbeddingCacheKey::new("a", "m"),
            CachedEmbedding::new(vec![1.0], "m"),
        );
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    // ---- Eviction policy tests ----

    #[test]
    fn test_eviction_lru() {
        let mut cache = EmbeddingCache::new(
            EmbeddingCacheConfig::default()
                .with_max_entries(2)
                .with_eviction(EvictionPolicy::LRU),
        );
        let k1 = EmbeddingCacheKey::new("a", "m");
        let k2 = EmbeddingCacheKey::new("b", "m");
        let k3 = EmbeddingCacheKey::new("c", "m");

        cache.put(k1.clone(), CachedEmbedding::new(vec![1.0], "m"));
        cache.put(k2.clone(), CachedEmbedding::new(vec![2.0], "m"));

        // Access k1 so it becomes more recently used.
        cache.get(&k1);

        // Inserting k3 should evict k2 (least recently used).
        cache.put(k3.clone(), CachedEmbedding::new(vec![3.0], "m"));

        assert!(cache.contains(&k1));
        assert!(!cache.contains(&k2));
        assert!(cache.contains(&k3));
    }

    #[test]
    fn test_eviction_lfu() {
        let mut cache = EmbeddingCache::new(
            EmbeddingCacheConfig::default()
                .with_max_entries(2)
                .with_eviction(EvictionPolicy::LFU),
        );
        let k1 = EmbeddingCacheKey::new("a", "m");
        let k2 = EmbeddingCacheKey::new("b", "m");
        let k3 = EmbeddingCacheKey::new("c", "m");

        cache.put(k1.clone(), CachedEmbedding::new(vec![1.0], "m"));
        cache.put(k2.clone(), CachedEmbedding::new(vec![2.0], "m"));

        // Access k1 multiple times so it has a higher access count.
        cache.get(&k1);
        cache.get(&k1);

        // k2 has fewer accesses, so it should be evicted.
        cache.put(k3.clone(), CachedEmbedding::new(vec![3.0], "m"));

        assert!(cache.contains(&k1));
        assert!(!cache.contains(&k2));
        assert!(cache.contains(&k3));
    }

    #[test]
    fn test_eviction_fifo() {
        let mut cache = EmbeddingCache::new(
            EmbeddingCacheConfig::default()
                .with_max_entries(2)
                .with_eviction(EvictionPolicy::FIFO),
        );
        let k1 = EmbeddingCacheKey::new("a", "m");
        let k2 = EmbeddingCacheKey::new("b", "m");
        let k3 = EmbeddingCacheKey::new("c", "m");

        cache.put(k1.clone(), CachedEmbedding::new(vec![1.0], "m"));
        cache.put(k2.clone(), CachedEmbedding::new(vec![2.0], "m"));

        // Even though k1 is accessed, FIFO ignores access — oldest inserted is evicted.
        cache.get(&k1);
        cache.put(k3.clone(), CachedEmbedding::new(vec![3.0], "m"));

        assert!(!cache.contains(&k1));
        assert!(cache.contains(&k2));
        assert!(cache.contains(&k3));
    }

    // ---- TTL tests ----

    #[test]
    fn test_ttl_expiration_on_get() {
        let mut cache = EmbeddingCache::new(
            EmbeddingCacheConfig::default()
                .with_max_entries(10)
                .with_ttl(Duration::from_nanos(1)),
        );
        let key = EmbeddingCacheKey::new("x", "m");
        cache.put(key.clone(), CachedEmbedding::new(vec![1.0], "m"));

        // With a 1ns TTL the entry should be expired almost immediately.
        std::thread::sleep(Duration::from_millis(1));
        assert!(cache.get(&key).is_none());
    }

    #[test]
    fn test_cleanup_expired_removes_entries() {
        let mut cache = EmbeddingCache::new(
            EmbeddingCacheConfig::default()
                .with_max_entries(10)
                .with_ttl(Duration::from_nanos(1)),
        );
        cache.put(
            EmbeddingCacheKey::new("a", "m"),
            CachedEmbedding::new(vec![1.0], "m"),
        );
        cache.put(
            EmbeddingCacheKey::new("b", "m"),
            CachedEmbedding::new(vec![2.0], "m"),
        );

        std::thread::sleep(Duration::from_millis(1));
        let removed = cache.cleanup_expired();
        assert_eq!(removed, 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cleanup_expired_no_ttl() {
        let mut cache = default_cache(10);
        cache.put(
            EmbeddingCacheKey::new("a", "m"),
            CachedEmbedding::new(vec![1.0], "m"),
        );
        assert_eq!(cache.cleanup_expired(), 0);
    }

    // ---- Hit / miss tracking ----

    #[test]
    fn test_hit_miss_tracking() {
        let mut cache = default_cache(10);
        let key = EmbeddingCacheKey::new("a", "m");

        // Miss.
        cache.get(&key);
        assert_eq!(cache.stats().misses, 1);
        assert_eq!(cache.stats().hits, 0);

        // Put, then hit.
        cache.put(key.clone(), CachedEmbedding::new(vec![1.0], "m"));
        cache.get(&key);
        assert_eq!(cache.stats().hits, 1);
        assert_eq!(cache.stats().misses, 1);
    }

    #[test]
    fn test_hit_rate_calculation() {
        let mut cache = default_cache(10);
        let key = EmbeddingCacheKey::new("a", "m");
        cache.put(key.clone(), CachedEmbedding::new(vec![1.0], "m"));

        // 1 hit, 1 miss → 0.5
        cache.get(&EmbeddingCacheKey::new("missing", "m"));
        cache.get(&key);
        assert!((cache.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_hit_rate_empty() {
        let cache = default_cache(10);
        assert_eq!(cache.hit_rate(), 0.0);
    }

    // ---- CacheStats tests ----

    #[test]
    fn test_cache_stats_snapshot() {
        let mut cache = default_cache(100);
        let key = EmbeddingCacheKey::new("a", "m");
        cache.put(key.clone(), CachedEmbedding::new(vec![1.0], "m"));
        cache.get(&key);

        let stats = cache.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.size, 1);
        assert_eq!(stats.max_size, 100);
    }

    #[test]
    fn test_cache_stats_hit_rate_method() {
        let stats = CacheStats {
            hits: 3,
            misses: 1,
            evictions: 0,
            size: 4,
            max_size: 10,
        };
        assert!((stats.hit_rate() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_stats_to_json() {
        let stats = CacheStats {
            hits: 10,
            misses: 5,
            evictions: 2,
            size: 8,
            max_size: 100,
        };
        let json = stats.to_json();
        assert_eq!(json["hits"], 10);
        assert_eq!(json["misses"], 5);
        assert_eq!(json["evictions"], 2);
        assert_eq!(json["size"], 8);
        assert_eq!(json["max_size"], 100);
    }

    #[test]
    fn test_eviction_counter_increments() {
        let mut cache = EmbeddingCache::new(
            EmbeddingCacheConfig::default()
                .with_max_entries(1)
                .with_eviction(EvictionPolicy::FIFO),
        );
        cache.put(
            EmbeddingCacheKey::new("a", "m"),
            CachedEmbedding::new(vec![1.0], "m"),
        );
        cache.put(
            EmbeddingCacheKey::new("b", "m"),
            CachedEmbedding::new(vec![2.0], "m"),
        );
        assert_eq!(cache.stats().evictions, 1);
    }

    // ---- CachedEmbeddingProvider tests ----

    fn make_provider() -> CachedEmbeddingProvider {
        CachedEmbeddingProvider::new(
            "test-model",
            EmbeddingCache::new(EmbeddingCacheConfig::default().with_max_entries(100)),
        )
    }

    #[test]
    fn test_provider_embed_cache_miss() {
        let mut prov = make_provider();
        let v = prov.embed("hello");
        assert_eq!(v.len(), 8);
        assert_eq!(prov.cache_stats().misses, 1);
        assert_eq!(prov.cache_stats().hits, 0);
    }

    #[test]
    fn test_provider_embed_cache_hit() {
        let mut prov = make_provider();
        let v1 = prov.embed("hello");
        let v2 = prov.embed("hello");
        assert_eq!(v1, v2);
        assert_eq!(prov.cache_stats().hits, 1);
        assert_eq!(prov.cache_stats().misses, 1);
    }

    #[test]
    fn test_provider_embed_batch() {
        let mut prov = make_provider();
        let results = prov.embed_batch(&["a", "b", "c"]);
        assert_eq!(results.len(), 3);
        assert_eq!(prov.cache_stats().misses, 3);
    }

    #[test]
    fn test_provider_batch_partial_cache_hit() {
        let mut prov = make_provider();

        // Warm the cache with "a".
        prov.embed("a");

        // Batch with "a" (hit) and "b" (miss).
        let results = prov.embed_batch(&["a", "b"]);
        assert_eq!(results.len(), 2);

        let stats = prov.cache_stats();
        // 1 miss for initial "a", 1 hit for "a" in batch, 1 miss for "b".
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
    }

    #[test]
    fn test_provider_invalidate() {
        let mut prov = make_provider();
        prov.embed("hello");
        assert_eq!(prov.cache_stats().size, 1);

        prov.invalidate("hello");
        assert_eq!(prov.cache_stats().size, 0);

        // Should be a miss again.
        prov.embed("hello");
        assert_eq!(prov.cache_stats().misses, 2);
    }

    #[test]
    fn test_provider_deterministic_embedding() {
        let mut prov = make_provider();
        // Two separate providers should produce the same fake embedding for the same text.
        let mut prov2 = make_provider();
        assert_eq!(prov.embed("same"), prov2.embed("same"));
    }

    // ---- Edge cases ----

    #[test]
    fn test_cache_capacity_one() {
        let mut cache = default_cache(1);
        cache.put(
            EmbeddingCacheKey::new("a", "m"),
            CachedEmbedding::new(vec![1.0], "m"),
        );
        cache.put(
            EmbeddingCacheKey::new("b", "m"),
            CachedEmbedding::new(vec![2.0], "m"),
        );
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(&EmbeddingCacheKey::new("b", "m")));
        assert!(!cache.contains(&EmbeddingCacheKey::new("a", "m")));
    }

    #[test]
    fn test_empty_cache_operations() {
        let mut cache = default_cache(10);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        assert!(!cache.remove(&EmbeddingCacheKey::new("x", "m")));
        assert!(cache.get(&EmbeddingCacheKey::new("x", "m")).is_none());
        cache.evict(); // should not panic
        cache.clear(); // should not panic
    }

    #[test]
    fn test_all_entries_expired_cleanup() {
        let mut cache = EmbeddingCache::new(
            EmbeddingCacheConfig::default()
                .with_max_entries(100)
                .with_ttl(Duration::from_nanos(1)),
        );
        for i in 0..5 {
            cache.put(
                EmbeddingCacheKey::new(&format!("k{i}"), "m"),
                CachedEmbedding::new(vec![i as f32], "m"),
            );
        }
        std::thread::sleep(Duration::from_millis(1));
        let removed = cache.cleanup_expired();
        assert_eq!(removed, 5);
        assert!(cache.is_empty());
    }

    #[test]
    fn test_replace_existing_key() {
        let mut cache = default_cache(10);
        let key = EmbeddingCacheKey::new("a", "m");
        cache.put(key.clone(), CachedEmbedding::new(vec![1.0], "m"));
        cache.put(key.clone(), CachedEmbedding::new(vec![9.0], "m"));
        assert_eq!(cache.len(), 1);
        let entry = cache.get(&key).unwrap();
        assert_eq!(entry.vector, vec![9.0]);
    }
}
