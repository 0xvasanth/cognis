//! Caching wrapper for runnables.
//!
//! Provides a [`Runnable`] wrapper that memoizes `invoke` results keyed by
//! the serialized input (or a custom key function). The cache supports
//! TTL-based expiry and LRU eviction when the maximum entry count is exceeded.
//!
//! - [`CacheConfig`] — configuration for max entries, TTL, and custom key generation
//! - [`CacheStats`] — snapshot of cache hit/miss/eviction counters
//! - [`RunnableCache`] — the caching wrapper that implements [`Runnable`]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::RwLock;

use crate::error::Result;

use super::base::Runnable;
use super::config::RunnableConfig;
use super::RunnableStream;

/// A thread-safe function that generates a cache key from a JSON value.
type CacheKeyFn = Arc<dyn Fn(&Value) -> String + Send + Sync>;

/// Configuration for the caching wrapper.
///
/// Controls cache size, TTL, and key generation.
///
/// # Example
/// ```ignore
/// use std::time::Duration;
/// use rustchain_core::runnables::cache::CacheConfig;
///
/// let config = CacheConfig::new()
///     .with_max_entries(500)
///     .with_ttl(Duration::from_secs(60));
/// ```
#[derive(Clone)]
pub struct CacheConfig {
    /// Maximum number of entries the cache can hold before LRU eviction.
    pub max_entries: usize,
    /// Optional time-to-live for cache entries. Expired entries are evicted on access.
    pub ttl: Option<Duration>,
    /// Optional custom key generation function. Defaults to JSON serialization of input.
    pub cache_key_fn: Option<CacheKeyFn>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            ttl: None,
            cache_key_fn: None,
        }
    }
}

impl CacheConfig {
    /// Create a new cache configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of cache entries.
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Set the time-to-live for cache entries.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Set a custom cache key generation function.
    pub fn with_cache_key_fn(
        mut self,
        f: impl Fn(&Value) -> String + Send + Sync + 'static,
    ) -> Self {
        self.cache_key_fn = Some(Arc::new(f));
        self
    }
}

impl std::fmt::Debug for CacheConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheConfig")
            .field("max_entries", &self.max_entries)
            .field("ttl", &self.ttl)
            .field("cache_key_fn", &self.cache_key_fn.is_some())
            .finish()
    }
}

/// An entry in the cache storing a computed value and metadata.
pub(crate) struct CacheEntry {
    /// The cached output value.
    value: Value,
    /// When this entry was created.
    created_at: Instant,
    /// When this entry was last accessed (for LRU eviction).
    last_accessed: Instant,
    /// Number of cache hits for this entry.
    hit_count: AtomicUsize,
}

impl CacheEntry {
    fn new(value: Value) -> Self {
        let now = Instant::now();
        Self {
            value,
            created_at: now,
            last_accessed: now,
            hit_count: AtomicUsize::new(0),
        }
    }

    /// Check if this entry has expired given a TTL.
    fn is_expired(&self, ttl: Option<Duration>) -> bool {
        match ttl {
            Some(ttl) => self.created_at.elapsed() > ttl,
            None => false,
        }
    }
}

/// Statistics about cache usage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStats {
    /// Number of cache hits.
    pub hits: usize,
    /// Number of cache misses.
    pub misses: usize,
    /// Number of entries evicted (TTL expiry or LRU).
    pub evictions: usize,
    /// Current number of entries in the cache.
    pub size: usize,
}

/// Internal shared state for the cache.
struct CacheState {
    entries: HashMap<String, CacheEntry>,
    hits: usize,
    misses: usize,
    evictions: usize,
}

impl CacheState {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }
}

/// A runnable wrapper that memoizes `invoke` results based on input.
///
/// Cache entries are keyed by the JSON serialization of the input (or a custom
/// key function). Supports TTL-based expiry and LRU eviction when the cache
/// exceeds `max_entries`.
///
/// Streaming is not cached and delegates directly to the inner runnable.
///
/// # Example
/// ```ignore
/// use std::time::Duration;
/// use rustchain_core::runnables::{RunnableLambda, RunnableExt};
/// use rustchain_core::runnables::cache::CacheConfig;
///
/// let config = CacheConfig::new()
///     .with_max_entries(100)
///     .with_ttl(Duration::from_secs(300));
///
/// let cached = my_runnable.with_cache(config);
/// ```
pub struct RunnableCache {
    /// The wrapped runnable.
    inner: Arc<dyn Runnable>,
    /// Thread-safe cache state.
    state: Arc<RwLock<CacheState>>,
    /// Cache configuration.
    config: CacheConfig,
}

impl RunnableCache {
    /// Create a new cached runnable wrapper.
    pub fn new(inner: Arc<dyn Runnable>, config: CacheConfig) -> Self {
        Self {
            inner,
            state: Arc::new(RwLock::new(CacheState::new())),
            config,
        }
    }

    /// Compute the cache key for a given input value.
    fn cache_key(&self, input: &Value) -> String {
        match &self.config.cache_key_fn {
            Some(f) => f(input),
            None => serde_json::to_string(input).unwrap_or_default(),
        }
    }

    /// Set a custom cache key function, returning self for chaining.
    pub fn with_cache_key(mut self, f: impl Fn(&Value) -> String + Send + Sync + 'static) -> Self {
        self.config.cache_key_fn = Some(Arc::new(f));
        self
    }

    /// Invalidate (remove) a specific cache entry by key.
    pub async fn invalidate(&self, key: &str) {
        let mut state = self.state.write().await;
        if state.entries.remove(key).is_some() {
            state.evictions += 1;
        }
    }

    /// Clear all entries from the cache.
    pub async fn invalidate_all(&self) {
        let mut state = self.state.write().await;
        let count = state.entries.len();
        state.entries.clear();
        state.evictions += count;
    }

    /// Get current cache statistics.
    pub async fn stats(&self) -> CacheStats {
        let state = self.state.read().await;
        CacheStats {
            hits: state.hits,
            misses: state.misses,
            evictions: state.evictions,
            size: state.entries.len(),
        }
    }

    /// Evict expired entries from the cache.
    fn evict_expired(state: &mut CacheState, ttl: Option<Duration>) -> usize {
        if ttl.is_none() {
            return 0;
        }
        let before = state.entries.len();
        state.entries.retain(|_, entry| !entry.is_expired(ttl));
        let evicted = before - state.entries.len();
        state.evictions += evicted;
        evicted
    }

    /// Evict the least recently used entry to make room.
    fn evict_lru(state: &mut CacheState) {
        if state.entries.is_empty() {
            return;
        }
        let lru_key = state
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(key, _)| key.clone());

        if let Some(key) = lru_key {
            state.entries.remove(&key);
            state.evictions += 1;
        }
    }
}

#[async_trait]
impl Runnable for RunnableCache {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let key = self.cache_key(&input);

        // Try to read from cache first.
        {
            let state = self.state.read().await;
            if let Some(entry) = state.entries.get(&key) {
                if !entry.is_expired(self.config.ttl) {
                    entry.hit_count.fetch_add(1, Ordering::Relaxed);
                    // We need a write lock to update last_accessed and hits,
                    // so drop the read lock and re-acquire as write below.
                    let value = entry.value.clone();
                    drop(state);
                    let mut state = self.state.write().await;
                    state.hits += 1;
                    if let Some(entry) = state.entries.get_mut(&key) {
                        entry.last_accessed = Instant::now();
                    }
                    return Ok(value);
                }
            }
        }

        // Cache miss: invoke the inner runnable.
        let result = self.inner.invoke(input, config).await?;

        // Store the result in the cache.
        {
            let mut state = self.state.write().await;
            state.misses += 1;

            // Evict expired entries first.
            Self::evict_expired(&mut state, self.config.ttl);

            // If still at capacity, evict the LRU entry.
            while state.entries.len() >= self.config.max_entries {
                Self::evict_lru(&mut state);
            }

            state.entries.insert(key, CacheEntry::new(result.clone()));
        }

        Ok(result)
    }

    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        // Streams are not cached; delegate directly.
        self.inner.stream(input, config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::ext::RunnableExt;
    use crate::runnables::lambda::RunnableLambda;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Helper: creates a RunnableLambda that counts invocations via an AtomicUsize counter.
    fn counting_doubler(counter: Arc<AtomicUsize>) -> RunnableLambda {
        RunnableLambda::new("counting_doubler", move |v: Value| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                let n = v.as_i64().unwrap();
                Ok(json!(n * 2))
            }
        })
    }

    fn counting_identity(counter: Arc<AtomicUsize>) -> RunnableLambda {
        RunnableLambda::new("counting_identity", move |v: Value| {
            let counter = counter.clone();
            async move {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(v)
            }
        })
    }

    // ==================== CacheConfig tests ====================

    #[test]
    fn test_cache_config_defaults() {
        let config = CacheConfig::new();
        assert_eq!(config.max_entries, 1000);
        assert!(config.ttl.is_none());
        assert!(config.cache_key_fn.is_none());
    }

    #[test]
    fn test_cache_config_builder() {
        let config = CacheConfig::new()
            .with_max_entries(500)
            .with_ttl(Duration::from_secs(60));
        assert_eq!(config.max_entries, 500);
        assert_eq!(config.ttl, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_cache_config_custom_key_fn() {
        let config = CacheConfig::new().with_cache_key_fn(|v: &Value| format!("custom:{}", v));
        assert!(config.cache_key_fn.is_some());
        let key = (config.cache_key_fn.unwrap())(&json!(42));
        assert_eq!(key, "custom:42");
    }

    #[test]
    fn test_cache_config_debug() {
        let config = CacheConfig::new();
        let debug = format!("{:?}", config);
        assert!(debug.contains("CacheConfig"));
        assert!(debug.contains("max_entries"));
    }

    // ==================== Basic caching tests ====================

    #[tokio::test]
    async fn test_cache_hit_avoids_recomputation() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_doubler(counter.clone());
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());

        let r1 = cached.invoke(json!(5), None).await.unwrap();
        assert_eq!(r1, json!(10));
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        let r2 = cached.invoke(json!(5), None).await.unwrap();
        assert_eq!(r2, json!(10));
        assert_eq!(counter.load(Ordering::SeqCst), 1); // Not called again
    }

    #[tokio::test]
    async fn test_cache_miss_for_different_inputs() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_doubler(counter.clone());
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());

        cached.invoke(json!(5), None).await.unwrap();
        cached.invoke(json!(10), None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_cache_returns_correct_values() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_doubler(counter.clone());
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());

        assert_eq!(cached.invoke(json!(3), None).await.unwrap(), json!(6));
        assert_eq!(cached.invoke(json!(7), None).await.unwrap(), json!(14));
        // Repeat, should be from cache
        assert_eq!(cached.invoke(json!(3), None).await.unwrap(), json!(6));
        assert_eq!(cached.invoke(json!(7), None).await.unwrap(), json!(14));
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_cache_name_delegates() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_doubler(counter);
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());
        assert_eq!(cached.name(), "counting_doubler");
    }

    // ==================== Stats tests ====================

    #[tokio::test]
    async fn test_cache_stats_initial() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter);
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());

        let stats = cached.stats().await;
        assert_eq!(
            stats,
            CacheStats {
                hits: 0,
                misses: 0,
                evictions: 0,
                size: 0
            }
        );
    }

    #[tokio::test]
    async fn test_cache_stats_hits_and_misses() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter);
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());

        cached.invoke(json!(1), None).await.unwrap(); // miss
        cached.invoke(json!(2), None).await.unwrap(); // miss
        cached.invoke(json!(1), None).await.unwrap(); // hit
        cached.invoke(json!(1), None).await.unwrap(); // hit
        cached.invoke(json!(3), None).await.unwrap(); // miss

        let stats = cached.stats().await;
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 3);
        assert_eq!(stats.size, 3);
    }

    // ==================== TTL tests ====================

    #[tokio::test]
    async fn test_cache_ttl_expiry() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_doubler(counter.clone());
        let config = CacheConfig::new().with_ttl(Duration::from_millis(50));
        let cached = RunnableCache::new(Arc::new(runnable), config);

        cached.invoke(json!(5), None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Entry should still be valid.
        cached.invoke(json!(5), None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Wait for TTL to expire.
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Should re-invoke because entry expired.
        cached.invoke(json!(5), None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_cache_no_ttl_entries_persist() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let config = CacheConfig::new(); // no TTL
        let cached = RunnableCache::new(Arc::new(runnable), config);

        cached.invoke(json!("hello"), None).await.unwrap();
        tokio::time::sleep(Duration::from_millis(50)).await;
        cached.invoke(json!("hello"), None).await.unwrap();

        assert_eq!(counter.load(Ordering::SeqCst), 1); // Still cached
    }

    // ==================== LRU eviction tests ====================

    #[tokio::test]
    async fn test_cache_lru_eviction() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let config = CacheConfig::new().with_max_entries(2);
        let cached = RunnableCache::new(Arc::new(runnable), config);

        cached.invoke(json!(1), None).await.unwrap(); // miss, cache: {1}
        cached.invoke(json!(2), None).await.unwrap(); // miss, cache: {1, 2}
        cached.invoke(json!(3), None).await.unwrap(); // miss, evicts 1, cache: {2, 3}

        assert_eq!(counter.load(Ordering::SeqCst), 3);

        let stats = cached.stats().await;
        assert_eq!(stats.size, 2);
        assert_eq!(stats.evictions, 1);
    }

    #[tokio::test]
    async fn test_cache_lru_evicts_least_recently_used() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let config = CacheConfig::new().with_max_entries(2);
        let cached = RunnableCache::new(Arc::new(runnable), config);

        cached.invoke(json!(1), None).await.unwrap(); // miss
        cached.invoke(json!(2), None).await.unwrap(); // miss
                                                      // Access 1 again to make it recently used.
        cached.invoke(json!(1), None).await.unwrap(); // hit

        // Now insert 3 -> should evict 2 (least recently used), not 1.
        cached.invoke(json!(3), None).await.unwrap(); // miss, evicts 2

        // 1 should still be cached.
        cached.invoke(json!(1), None).await.unwrap(); // hit
        assert_eq!(counter.load(Ordering::SeqCst), 3); // only 1, 2, 3 were actual calls

        // 2 should have been evicted and needs recomputation.
        cached.invoke(json!(2), None).await.unwrap(); // miss
        assert_eq!(counter.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn test_cache_max_entries_one() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let config = CacheConfig::new().with_max_entries(1);
        let cached = RunnableCache::new(Arc::new(runnable), config);

        cached.invoke(json!(1), None).await.unwrap();
        cached.invoke(json!(1), None).await.unwrap(); // hit
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        cached.invoke(json!(2), None).await.unwrap(); // miss, evicts 1
        assert_eq!(counter.load(Ordering::SeqCst), 2);

        cached.invoke(json!(1), None).await.unwrap(); // miss, evicts 2
        assert_eq!(counter.load(Ordering::SeqCst), 3);

        let stats = cached.stats().await;
        assert_eq!(stats.size, 1);
        assert_eq!(stats.evictions, 2);
    }

    // ==================== Invalidation tests ====================

    #[tokio::test]
    async fn test_invalidate_specific_key() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());

        cached.invoke(json!(42), None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        let key = serde_json::to_string(&json!(42)).unwrap();
        cached.invalidate(&key).await;

        cached.invoke(json!(42), None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 2); // Re-invoked
    }

    #[tokio::test]
    async fn test_invalidate_nonexistent_key() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter);
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());

        // Should not panic or change eviction count.
        cached.invalidate("nonexistent").await;
        let stats = cached.stats().await;
        assert_eq!(stats.evictions, 0);
    }

    #[tokio::test]
    async fn test_invalidate_all() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());

        cached.invoke(json!(1), None).await.unwrap();
        cached.invoke(json!(2), None).await.unwrap();
        cached.invoke(json!(3), None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 3);

        cached.invalidate_all().await;
        let stats = cached.stats().await;
        assert_eq!(stats.size, 0);
        assert_eq!(stats.evictions, 3);

        // All should be misses now.
        cached.invoke(json!(1), None).await.unwrap();
        cached.invoke(json!(2), None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }

    // ==================== Custom key function tests ====================

    #[tokio::test]
    async fn test_custom_cache_key_fn() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let config = CacheConfig::new().with_cache_key_fn(|v: &Value| {
            // Only use the "id" field as the key, ignoring other fields.
            v.get("id")
                .and_then(|id| id.as_str())
                .unwrap_or("unknown")
                .to_string()
        });
        let cached = RunnableCache::new(Arc::new(runnable), config);

        let input1 = json!({"id": "abc", "data": 1});
        let input2 = json!({"id": "abc", "data": 2}); // same id, different data

        cached.invoke(input1.clone(), None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Should be a cache hit because the key ("abc") is the same.
        let result = cached.invoke(input2, None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        // Returns the first cached value.
        assert_eq!(result, input1);
    }

    #[tokio::test]
    async fn test_with_cache_key_method() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new())
            .with_cache_key(|v: &Value| format!("prefix:{}", v));

        cached.invoke(json!(1), None).await.unwrap();
        cached.invoke(json!(1), None).await.unwrap(); // hit
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    // ==================== Stream delegation tests ====================

    #[tokio::test]
    async fn test_stream_delegates_without_caching() {
        use futures::StreamExt;

        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());

        let mut stream = cached.stream(json!(42), None).await.unwrap();
        let item = stream.next().await.unwrap().unwrap();
        assert_eq!(item, json!(42));

        // Streaming does not populate the cache.
        let stats = cached.stats().await;
        assert_eq!(stats.size, 0);
    }

    // ==================== Thread safety tests ====================

    #[tokio::test]
    async fn test_cache_thread_safety() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let cached = Arc::new(RunnableCache::new(Arc::new(runnable), CacheConfig::new()));

        let mut handles = vec![];
        for i in 0..10 {
            let cached = cached.clone();
            handles.push(tokio::spawn(async move {
                // All tasks invoke the same key.
                cached.invoke(json!(i % 3), None).await.unwrap()
            }));
        }

        for handle in handles {
            handle.await.unwrap();
        }

        // At most 3 unique keys were invoked.
        assert!(counter.load(Ordering::SeqCst) <= 10);
        let stats = cached.stats().await;
        assert!(stats.size <= 3);
    }

    // ==================== Extension trait test ====================

    #[tokio::test]
    async fn test_with_cache_extension() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_doubler(counter.clone());
        let cached = runnable.with_cache(CacheConfig::new().with_max_entries(10));

        let r1 = cached.invoke(json!(5), None).await.unwrap();
        assert_eq!(r1, json!(10));

        let r2 = cached.invoke(json!(5), None).await.unwrap();
        assert_eq!(r2, json!(10));
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    // ==================== Complex input types ====================

    #[tokio::test]
    async fn test_cache_with_object_inputs() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());

        let input = json!({"key": "value", "nested": {"a": 1}});
        cached.invoke(input.clone(), None).await.unwrap();
        cached.invoke(input.clone(), None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_cache_with_array_inputs() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());

        cached.invoke(json!([1, 2, 3]), None).await.unwrap();
        cached.invoke(json!([1, 2, 3]), None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        cached.invoke(json!([1, 2, 4]), None).await.unwrap(); // different array
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_cache_with_null_input() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());

        cached.invoke(json!(null), None).await.unwrap();
        cached.invoke(json!(null), None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    // ==================== TTL + LRU combined ====================

    #[tokio::test]
    async fn test_ttl_eviction_frees_space_before_lru() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let config = CacheConfig::new()
            .with_max_entries(2)
            .with_ttl(Duration::from_millis(50));
        let cached = RunnableCache::new(Arc::new(runnable), config);

        cached.invoke(json!(1), None).await.unwrap(); // miss
        cached.invoke(json!(2), None).await.unwrap(); // miss

        // Wait for TTL to expire.
        tokio::time::sleep(Duration::from_millis(60)).await;

        // Insert 3: expired entries should be evicted first via TTL, not LRU.
        cached.invoke(json!(3), None).await.unwrap(); // miss, TTL evicts 1 & 2
        assert_eq!(counter.load(Ordering::SeqCst), 3);

        let stats = cached.stats().await;
        assert_eq!(stats.size, 1); // Only entry 3 remains.
    }

    // ==================== Error propagation ====================

    #[tokio::test]
    async fn test_cache_does_not_cache_errors() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();
        let runnable = RunnableLambda::new("failing", move |v: Value| {
            let c = c.clone();
            async move {
                c.fetch_add(1, Ordering::SeqCst);
                let n = v.as_i64().unwrap();
                if n < 0 {
                    Err(crate::error::RustChainError::Other(
                        "negative input".to_string(),
                    ))
                } else {
                    Ok(json!(n * 2))
                }
            }
        });
        let cached = RunnableCache::new(Arc::new(runnable), CacheConfig::new());

        // Error should not be cached.
        assert!(cached.invoke(json!(-1), None).await.is_err());
        assert!(cached.invoke(json!(-1), None).await.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 2); // Called twice (not cached)

        // Successful result should be cached.
        assert_eq!(cached.invoke(json!(5), None).await.unwrap(), json!(10));
        assert_eq!(cached.invoke(json!(5), None).await.unwrap(), json!(10));
        assert_eq!(counter.load(Ordering::SeqCst), 3); // Only one more call
    }
}
