//! Caching wrapper for runnables.
//!
//! Provides caching infrastructure to avoid redundant LLM calls. Supports
//! multiple eviction policies (LRU, LFU, FIFO, TTL) and pattern-based
//! invalidation.
//!
//! - [`CacheKey`] — hashed key for cache lookups
//! - [`CacheEntry`] — cached value with metadata
//! - [`CacheConfig`] — configuration with eviction policy, max entries, TTL
//! - [`EvictionPolicy`] — LRU, LFU, FIFO, or TTL-based eviction
//! - [`RunnableCache`] — the cache store with get/put/invalidate operations
//! - [`CacheStats`] — hit/miss/eviction counters
//! - [`CachedRunnable`] — wraps a [`Runnable`] with transparent caching
//! - [`CacheInvalidator`] — pattern-based cache invalidation utility

use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::error::Result;

use super::base::Runnable;
use super::config::RunnableConfig;
use super::RunnableStream;

// ---------------------------------------------------------------------------
// CacheKey
// ---------------------------------------------------------------------------

/// A cache key wrapping a hashed string representation of a value.
#[derive(Debug, Clone, Eq)]
pub struct CacheKey {
    hash: String,
}

impl CacheKey {
    /// Create a `CacheKey` by serializing the given JSON value to a string
    /// and producing a simple string-based hash.
    pub fn from_value(value: &Value) -> CacheKey {
        let serialized = serde_json::to_string(value).unwrap_or_default();
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        serialized.hash(&mut hasher);
        let hash_val = hasher.finish();
        CacheKey {
            hash: format!("{:016x}", hash_val),
        }
    }

    /// Return the key as a string slice.
    pub fn as_str(&self) -> &str {
        &self.hash
    }
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        self.hash == other.hash
    }
}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash.hash(state);
    }
}

// ---------------------------------------------------------------------------
// CacheEntry
// ---------------------------------------------------------------------------

/// A single entry in the cache, storing value and metadata.
pub struct CacheEntry {
    /// The cache key for this entry.
    pub key: CacheKey,
    /// The cached JSON value.
    pub value: Value,
    /// When this entry was created.
    pub created_at: Instant,
    /// Number of times this entry has been accessed.
    pub access_count: u64,
    /// Optional per-entry TTL.
    pub ttl: Option<Duration>,
    /// When this entry was last accessed (for LRU).
    last_accessed: Instant,
    /// Insertion order index (for FIFO).
    insertion_order: u64,
}

impl CacheEntry {
    /// Create a new cache entry.
    pub fn new(key: CacheKey, value: Value, ttl: Option<Duration>) -> Self {
        let now = Instant::now();
        Self {
            key,
            value,
            created_at: now,
            access_count: 0,
            ttl,
            last_accessed: now,
            insertion_order: 0,
        }
    }

    /// Check whether this entry has expired based on its TTL.
    pub fn is_expired(&self) -> bool {
        match self.ttl {
            Some(ttl) => self.created_at.elapsed() > ttl,
            None => false,
        }
    }

    /// Record an access: increment `access_count` and update `last_accessed`.
    pub fn touch(&mut self) {
        self.access_count += 1;
        self.last_accessed = Instant::now();
    }
}

// ---------------------------------------------------------------------------
// EvictionPolicy
// ---------------------------------------------------------------------------

/// Eviction strategy used when the cache is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvictionPolicy {
    /// Least Recently Used — evicts the entry that was accessed least recently.
    LRU,
    /// Least Frequently Used — evicts the entry with the fewest accesses.
    LFU,
    /// First In, First Out — evicts the oldest inserted entry.
    FIFO,
    /// TTL-only — only evicts expired entries (no forced eviction).
    TTL,
}

impl EvictionPolicy {
    /// Return a human-readable name for this policy.
    pub fn name(&self) -> &str {
        match self {
            EvictionPolicy::LRU => "LRU",
            EvictionPolicy::LFU => "LFU",
            EvictionPolicy::FIFO => "FIFO",
            EvictionPolicy::TTL => "TTL",
        }
    }
}

// ---------------------------------------------------------------------------
// CacheConfig
// ---------------------------------------------------------------------------

/// Configuration for [`RunnableCache`].
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum number of entries the cache can hold.
    pub max_entries: usize,
    /// Default TTL applied to entries that do not specify their own.
    pub default_ttl: Option<Duration>,
    /// The eviction policy to use when the cache is full.
    pub eviction_policy: EvictionPolicy,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 1000,
            default_ttl: None,
            eviction_policy: EvictionPolicy::LRU,
        }
    }
}

impl CacheConfig {
    /// Create a new configuration with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of entries.
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Set the default TTL.
    pub fn with_default_ttl(mut self, ttl: Duration) -> Self {
        self.default_ttl = Some(ttl);
        self
    }

    /// Set the eviction policy.
    pub fn with_eviction_policy(mut self, policy: EvictionPolicy) -> Self {
        self.eviction_policy = policy;
        self
    }
}

// ---------------------------------------------------------------------------
// CacheStats
// ---------------------------------------------------------------------------

/// Snapshot of cache statistics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheStats {
    /// Total cache hits.
    pub hits: u64,
    /// Total cache misses.
    pub misses: u64,
    /// Total evictions (TTL + policy-based).
    pub evictions: u64,
    /// Current number of entries in the cache.
    pub current_size: usize,
}

impl CacheStats {
    /// Compute the hit rate as a ratio (0.0 to 1.0).
    /// Returns 0.0 if there have been no lookups.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Serialize stats to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "current_size": self.current_size,
            "hit_rate": self.hit_rate(),
        })
    }
}

// ---------------------------------------------------------------------------
// RunnableCache (the cache store)
// ---------------------------------------------------------------------------

/// A cache store supporting configurable eviction policies.
pub struct RunnableCache {
    entries: HashMap<CacheKey, CacheEntry>,
    /// Insertion order tracking for FIFO.
    insertion_counter: u64,
    config: CacheConfig,
    hits: u64,
    misses: u64,
    evictions: u64,
}

impl RunnableCache {
    /// Create a new cache with the given configuration.
    pub fn new(config: CacheConfig) -> Self {
        Self {
            entries: HashMap::new(),
            insertion_counter: 0,
            config,
            hits: 0,
            misses: 0,
            evictions: 0,
        }
    }

    /// Look up a value by key. Returns `None` if the key is missing or expired.
    /// On hit, the entry is touched (access count incremented, last_accessed updated).
    pub fn get(&mut self, key: &CacheKey) -> Option<&Value> {
        // Check expiration first.
        if let Some(entry) = self.entries.get(key) {
            if entry.is_expired() {
                self.entries.remove(key);
                self.evictions += 1;
                self.misses += 1;
                return None;
            }
        }

        if let Some(entry) = self.entries.get_mut(key) {
            entry.touch();
            self.hits += 1;
            Some(&entry.value)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Insert or update a cache entry. Evicts entries if the cache is full.
    pub fn put(&mut self, key: CacheKey, value: Value) {
        // If key already exists, just update the value.
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.value = value;
            entry.touch();
            return;
        }

        // Evict expired entries first.
        self.evict_expired();

        // Evict based on policy while at capacity.
        while self.entries.len() >= self.config.max_entries && self.config.max_entries > 0 {
            self.evict_one();
        }

        // If max_entries is 0, we cannot store anything.
        if self.config.max_entries == 0 {
            return;
        }

        let mut entry = CacheEntry::new(key.clone(), value, self.config.default_ttl);
        entry.insertion_order = self.insertion_counter;
        self.insertion_counter += 1;
        self.entries.insert(key, entry);
    }

    /// Remove a specific entry by key.
    pub fn invalidate(&mut self, key: &CacheKey) {
        if self.entries.remove(key).is_some() {
            self.evictions += 1;
        }
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        let count = self.entries.len() as u64;
        self.entries.clear();
        self.evictions += count;
    }

    /// Return the number of entries currently in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Compute the cache hit rate (0.0 to 1.0).
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
            current_size: self.entries.len(),
        }
    }

    /// Evict all expired entries.
    fn evict_expired(&mut self) {
        let expired_keys: Vec<CacheKey> = self
            .entries
            .iter()
            .filter(|(_, e)| e.is_expired())
            .map(|(k, _)| k.clone())
            .collect();
        for key in &expired_keys {
            self.entries.remove(key);
        }
        self.evictions += expired_keys.len() as u64;
    }

    /// Evict one entry based on the configured policy.
    fn evict_one(&mut self) {
        if self.entries.is_empty() {
            return;
        }

        let victim_key = match self.config.eviction_policy {
            EvictionPolicy::LRU => self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.last_accessed)
                .map(|(k, _)| k.clone()),
            EvictionPolicy::LFU => self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.access_count)
                .map(|(k, _)| k.clone()),
            EvictionPolicy::FIFO => self
                .entries
                .iter()
                .min_by_key(|(_, e)| e.insertion_order)
                .map(|(k, _)| k.clone()),
            EvictionPolicy::TTL => {
                // TTL-only: only evict expired. If none expired, evict oldest.
                let expired = self
                    .entries
                    .iter()
                    .filter(|(_, e)| e.is_expired())
                    .min_by_key(|(_, e)| e.created_at)
                    .map(|(k, _)| k.clone());
                expired.or_else(|| {
                    self.entries
                        .iter()
                        .min_by_key(|(_, e)| e.created_at)
                        .map(|(k, _)| k.clone())
                })
            }
        };

        if let Some(key) = victim_key {
            self.entries.remove(&key);
            self.evictions += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// CachedRunnable
// ---------------------------------------------------------------------------

/// A [`Runnable`] wrapper that transparently caches `invoke` results.
///
/// On `invoke`, the input is hashed to a [`CacheKey`]. If a matching,
/// non-expired entry exists in the cache, it is returned directly without
/// calling the inner runnable. On a miss, the inner runnable is invoked and
/// the result is cached.
pub struct CachedRunnable {
    inner: Arc<dyn Runnable>,
    cache: Arc<Mutex<RunnableCache>>,
}

impl CachedRunnable {
    /// Create a new `CachedRunnable` wrapping the given runnable and cache.
    pub fn new(inner: Arc<dyn Runnable>, cache: Arc<Mutex<RunnableCache>>) -> Self {
        Self { inner, cache }
    }
}

#[async_trait]
impl Runnable for CachedRunnable {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let key = CacheKey::from_value(&input);

        // Check cache.
        {
            let mut cache = self.cache.lock().await;
            if let Some(value) = cache.get(&key) {
                return Ok(value.clone());
            }
        }

        // Cache miss: invoke inner runnable.
        let result = self.inner.invoke(input, config).await?;

        // Store result in cache.
        {
            let mut cache = self.cache.lock().await;
            cache.put(key, result.clone());
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

// ---------------------------------------------------------------------------
// CacheInvalidator
// ---------------------------------------------------------------------------

/// Utility for pattern-based cache invalidation.
///
/// Stores a set of patterns (simple substring matches on key strings) and
/// can check individual keys or bulk-invalidate matching entries from a cache.
pub struct CacheInvalidator {
    patterns: Vec<String>,
}

impl CacheInvalidator {
    /// Create a new invalidator with no patterns.
    pub fn new() -> Self {
        Self {
            patterns: Vec::new(),
        }
    }

    /// Add a pattern (substring match) to the invalidator.
    pub fn add_pattern(&mut self, pattern: &str) {
        self.patterns.push(pattern.to_string());
    }

    /// Check whether a key matches any of the registered patterns.
    pub fn should_invalidate(&self, key: &CacheKey) -> bool {
        let key_str = key.as_str();
        self.patterns.iter().any(|p| key_str.contains(p))
    }

    /// Remove all entries from the cache whose keys match any registered pattern.
    pub fn invalidate_matching(&self, cache: &mut RunnableCache) {
        let matching_keys: Vec<CacheKey> = cache
            .entries
            .keys()
            .filter(|k| self.should_invalidate(k))
            .cloned()
            .collect();
        for key in matching_keys {
            cache.invalidate(&key);
        }
    }
}

impl Default for CacheInvalidator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::lambda::RunnableLambda;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Helper: creates a RunnableLambda that counts invocations.
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

    // ==================== CacheKey tests ====================

    #[test]
    fn test_cache_key_from_value_deterministic() {
        let k1 = CacheKey::from_value(&json!(42));
        let k2 = CacheKey::from_value(&json!(42));
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_different_values_differ() {
        let k1 = CacheKey::from_value(&json!(1));
        let k2 = CacheKey::from_value(&json!(2));
        assert_ne!(k1, k2);
    }

    #[test]
    fn test_cache_key_as_str() {
        let k = CacheKey::from_value(&json!("hello"));
        assert!(!k.as_str().is_empty());
        assert_eq!(k.as_str().len(), 16); // 16-char hex
    }

    #[test]
    fn test_cache_key_hash_impl() {
        use std::collections::HashSet;
        let k1 = CacheKey::from_value(&json!(1));
        let k2 = CacheKey::from_value(&json!(1));
        let k3 = CacheKey::from_value(&json!(2));
        let mut set = HashSet::new();
        set.insert(k1);
        set.insert(k2);
        set.insert(k3);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_cache_key_from_complex_value() {
        let k1 = CacheKey::from_value(&json!({"a": 1, "b": [2, 3]}));
        let k2 = CacheKey::from_value(&json!({"a": 1, "b": [2, 3]}));
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_cache_key_from_null() {
        let k = CacheKey::from_value(&json!(null));
        assert!(!k.as_str().is_empty());
    }

    // ==================== CacheEntry tests ====================

    #[test]
    fn test_cache_entry_creation() {
        let key = CacheKey::from_value(&json!(1));
        let entry = CacheEntry::new(key.clone(), json!(42), None);
        assert_eq!(entry.value, json!(42));
        assert_eq!(entry.access_count, 0);
        assert!(entry.ttl.is_none());
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_cache_entry_with_ttl_not_expired() {
        let key = CacheKey::from_value(&json!(1));
        let entry = CacheEntry::new(key, json!(42), Some(Duration::from_secs(60)));
        assert!(!entry.is_expired());
    }

    #[test]
    fn test_cache_entry_with_zero_ttl_expired() {
        let key = CacheKey::from_value(&json!(1));
        let entry = CacheEntry::new(key, json!(42), Some(Duration::from_nanos(0)));
        // The entry is created and immediately checked — with zero TTL it should be expired.
        std::thread::sleep(Duration::from_millis(1));
        assert!(entry.is_expired());
    }

    #[test]
    fn test_cache_entry_touch() {
        let key = CacheKey::from_value(&json!(1));
        let mut entry = CacheEntry::new(key, json!(42), None);
        assert_eq!(entry.access_count, 0);
        entry.touch();
        assert_eq!(entry.access_count, 1);
        entry.touch();
        entry.touch();
        assert_eq!(entry.access_count, 3);
    }

    #[test]
    fn test_cache_entry_no_ttl_never_expires() {
        let key = CacheKey::from_value(&json!(1));
        let entry = CacheEntry::new(key, json!(42), None);
        assert!(!entry.is_expired());
    }

    // ==================== CacheConfig tests ====================

    #[test]
    fn test_cache_config_defaults() {
        let config = CacheConfig::new();
        assert_eq!(config.max_entries, 1000);
        assert!(config.default_ttl.is_none());
        assert_eq!(config.eviction_policy, EvictionPolicy::LRU);
    }

    #[test]
    fn test_cache_config_builder() {
        let config = CacheConfig::new()
            .with_max_entries(500)
            .with_default_ttl(Duration::from_secs(60))
            .with_eviction_policy(EvictionPolicy::LFU);
        assert_eq!(config.max_entries, 500);
        assert_eq!(config.default_ttl, Some(Duration::from_secs(60)));
        assert_eq!(config.eviction_policy, EvictionPolicy::LFU);
    }

    #[test]
    fn test_cache_config_debug() {
        let config = CacheConfig::new();
        let debug = format!("{:?}", config);
        assert!(debug.contains("CacheConfig"));
        assert!(debug.contains("max_entries"));
    }

    // ==================== EvictionPolicy tests ====================

    #[test]
    fn test_eviction_policy_name() {
        assert_eq!(EvictionPolicy::LRU.name(), "LRU");
        assert_eq!(EvictionPolicy::LFU.name(), "LFU");
        assert_eq!(EvictionPolicy::FIFO.name(), "FIFO");
        assert_eq!(EvictionPolicy::TTL.name(), "TTL");
    }

    #[test]
    fn test_eviction_policy_equality() {
        assert_eq!(EvictionPolicy::LRU, EvictionPolicy::LRU);
        assert_ne!(EvictionPolicy::LRU, EvictionPolicy::LFU);
    }

    // ==================== RunnableCache put/get/invalidate/clear ====================

    #[test]
    fn test_runnable_cache_put_get() {
        let mut cache = RunnableCache::new(CacheConfig::new());
        let key = CacheKey::from_value(&json!(1));
        cache.put(key.clone(), json!(42));
        assert_eq!(cache.get(&key), Some(&json!(42)));
    }

    #[test]
    fn test_runnable_cache_get_missing() {
        let mut cache = RunnableCache::new(CacheConfig::new());
        let key = CacheKey::from_value(&json!(999));
        assert_eq!(cache.get(&key), None);
    }

    #[test]
    fn test_runnable_cache_invalidate() {
        let mut cache = RunnableCache::new(CacheConfig::new());
        let key = CacheKey::from_value(&json!(1));
        cache.put(key.clone(), json!(42));
        assert_eq!(cache.len(), 1);
        cache.invalidate(&key);
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.get(&key), None);
    }

    #[test]
    fn test_runnable_cache_invalidate_nonexistent() {
        let mut cache = RunnableCache::new(CacheConfig::new());
        let key = CacheKey::from_value(&json!(999));
        cache.invalidate(&key); // should not panic
        assert_eq!(cache.stats().evictions, 0);
    }

    #[test]
    fn test_runnable_cache_clear() {
        let mut cache = RunnableCache::new(CacheConfig::new());
        cache.put(CacheKey::from_value(&json!(1)), json!(10));
        cache.put(CacheKey::from_value(&json!(2)), json!(20));
        cache.put(CacheKey::from_value(&json!(3)), json!(30));
        assert_eq!(cache.len(), 3);
        cache.clear();
        assert_eq!(cache.len(), 0);
        assert!(cache.is_empty());
        assert_eq!(cache.stats().evictions, 3);
    }

    #[test]
    fn test_runnable_cache_len_is_empty() {
        let mut cache = RunnableCache::new(CacheConfig::new());
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        cache.put(CacheKey::from_value(&json!(1)), json!(10));
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_runnable_cache_update_existing_key() {
        let mut cache = RunnableCache::new(CacheConfig::new());
        let key = CacheKey::from_value(&json!(1));
        cache.put(key.clone(), json!(10));
        cache.put(key.clone(), json!(20));
        assert_eq!(cache.get(&key), Some(&json!(20)));
        assert_eq!(cache.len(), 1);
    }

    // ==================== LRU eviction ====================

    #[test]
    fn test_lru_eviction_evicts_least_recently_accessed() {
        let config = CacheConfig::new()
            .with_max_entries(2)
            .with_eviction_policy(EvictionPolicy::LRU);
        let mut cache = RunnableCache::new(config);

        let k1 = CacheKey::from_value(&json!(1));
        let k2 = CacheKey::from_value(&json!(2));
        let k3 = CacheKey::from_value(&json!(3));

        cache.put(k1.clone(), json!(10));
        cache.put(k2.clone(), json!(20));

        // Access k1 to make it recently used.
        cache.get(&k1);

        // Insert k3 — should evict k2 (least recently used).
        cache.put(k3.clone(), json!(30));

        assert_eq!(cache.len(), 2);
        assert!(cache.entries.contains_key(&k1));
        assert!(!cache.entries.contains_key(&k2));
        assert!(cache.entries.contains_key(&k3));
    }

    #[test]
    fn test_lru_eviction_multiple() {
        let config = CacheConfig::new()
            .with_max_entries(2)
            .with_eviction_policy(EvictionPolicy::LRU);
        let mut cache = RunnableCache::new(config);

        for i in 0..5 {
            cache.put(CacheKey::from_value(&json!(i)), json!(i * 10));
        }
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.stats().evictions, 3);
    }

    // ==================== LFU eviction ====================

    #[test]
    fn test_lfu_eviction_evicts_least_frequently_accessed() {
        let config = CacheConfig::new()
            .with_max_entries(2)
            .with_eviction_policy(EvictionPolicy::LFU);
        let mut cache = RunnableCache::new(config);

        let k1 = CacheKey::from_value(&json!(1));
        let k2 = CacheKey::from_value(&json!(2));
        let k3 = CacheKey::from_value(&json!(3));

        cache.put(k1.clone(), json!(10));
        cache.put(k2.clone(), json!(20));

        // Access k1 multiple times to increase its frequency.
        cache.get(&k1);
        cache.get(&k1);
        cache.get(&k1);
        // Access k2 once.
        cache.get(&k2);

        // Insert k3 — should evict k2 (least frequently used).
        cache.put(k3.clone(), json!(30));

        assert_eq!(cache.len(), 2);
        assert!(cache.entries.contains_key(&k1));
        assert!(!cache.entries.contains_key(&k2));
        assert!(cache.entries.contains_key(&k3));
    }

    // ==================== FIFO eviction ====================

    #[test]
    fn test_fifo_eviction_evicts_oldest() {
        let config = CacheConfig::new()
            .with_max_entries(2)
            .with_eviction_policy(EvictionPolicy::FIFO);
        let mut cache = RunnableCache::new(config);

        let k1 = CacheKey::from_value(&json!(1));
        let k2 = CacheKey::from_value(&json!(2));
        let k3 = CacheKey::from_value(&json!(3));

        cache.put(k1.clone(), json!(10));
        cache.put(k2.clone(), json!(20));

        // Access k1 (should not matter for FIFO).
        cache.get(&k1);

        // Insert k3 — should evict k1 (first inserted).
        cache.put(k3.clone(), json!(30));

        assert_eq!(cache.len(), 2);
        assert!(!cache.entries.contains_key(&k1));
        assert!(cache.entries.contains_key(&k2));
        assert!(cache.entries.contains_key(&k3));
    }

    #[test]
    fn test_fifo_order_preserved_on_access() {
        let config = CacheConfig::new()
            .with_max_entries(3)
            .with_eviction_policy(EvictionPolicy::FIFO);
        let mut cache = RunnableCache::new(config);

        let k1 = CacheKey::from_value(&json!(1));
        let k2 = CacheKey::from_value(&json!(2));
        let k3 = CacheKey::from_value(&json!(3));
        let k4 = CacheKey::from_value(&json!(4));

        cache.put(k1.clone(), json!(10));
        cache.put(k2.clone(), json!(20));
        cache.put(k3.clone(), json!(30));

        // Access k1 heavily — FIFO should still evict it first.
        for _ in 0..10 {
            cache.get(&k1);
        }

        cache.put(k4.clone(), json!(40));
        assert!(
            !cache.entries.contains_key(&k1),
            "FIFO should evict k1 regardless of access"
        );
    }

    // ==================== TTL expiration ====================

    #[test]
    fn test_ttl_expiration_on_get() {
        let config = CacheConfig::new()
            .with_default_ttl(Duration::from_millis(1))
            .with_eviction_policy(EvictionPolicy::TTL);
        let mut cache = RunnableCache::new(config);

        let key = CacheKey::from_value(&json!(1));
        cache.put(key.clone(), json!(42));

        std::thread::sleep(Duration::from_millis(5));
        assert_eq!(cache.get(&key), None);
    }

    #[test]
    fn test_ttl_not_expired_returns_value() {
        let config = CacheConfig::new()
            .with_default_ttl(Duration::from_secs(60))
            .with_eviction_policy(EvictionPolicy::TTL);
        let mut cache = RunnableCache::new(config);

        let key = CacheKey::from_value(&json!(1));
        cache.put(key.clone(), json!(42));
        assert_eq!(cache.get(&key), Some(&json!(42)));
    }

    // ==================== CacheStats tracking ====================

    #[test]
    fn test_cache_stats_initial() {
        let cache = RunnableCache::new(CacheConfig::new());
        let stats = cache.stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
        assert_eq!(stats.evictions, 0);
        assert_eq!(stats.current_size, 0);
    }

    #[test]
    fn test_cache_stats_hits_and_misses() {
        let mut cache = RunnableCache::new(CacheConfig::new());
        let k1 = CacheKey::from_value(&json!(1));
        let k2 = CacheKey::from_value(&json!(2));

        cache.put(k1.clone(), json!(10));
        cache.get(&k1); // hit
        cache.get(&k1); // hit
        cache.get(&k2); // miss

        let stats = cache.stats();
        assert_eq!(stats.hits, 2);
        assert_eq!(stats.misses, 1);
        assert_eq!(stats.current_size, 1);
    }

    #[test]
    fn test_cache_stats_hit_rate() {
        let stats = CacheStats {
            hits: 3,
            misses: 1,
            evictions: 0,
            current_size: 2,
        };
        assert!((stats.hit_rate() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_stats_hit_rate_no_lookups() {
        let stats = CacheStats {
            hits: 0,
            misses: 0,
            evictions: 0,
            current_size: 0,
        };
        assert!((stats.hit_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cache_stats_to_json() {
        let stats = CacheStats {
            hits: 10,
            misses: 5,
            evictions: 2,
            current_size: 8,
        };
        let j = stats.to_json();
        assert_eq!(j["hits"], 10);
        assert_eq!(j["misses"], 5);
        assert_eq!(j["evictions"], 2);
        assert_eq!(j["current_size"], 8);
    }

    #[test]
    fn test_runnable_cache_hit_rate() {
        let mut cache = RunnableCache::new(CacheConfig::new());
        assert!((cache.hit_rate() - 0.0).abs() < f64::EPSILON);

        let key = CacheKey::from_value(&json!(1));
        cache.put(key.clone(), json!(10));
        cache.get(&key); // hit
        cache.get(&CacheKey::from_value(&json!(2))); // miss

        assert!((cache.hit_rate() - 0.5).abs() < f64::EPSILON);
    }

    // ==================== Edge cases ====================

    #[test]
    fn test_zero_max_entries() {
        let config = CacheConfig::new().with_max_entries(0);
        let mut cache = RunnableCache::new(config);

        let key = CacheKey::from_value(&json!(1));
        cache.put(key.clone(), json!(42));
        // Nothing should be stored.
        assert_eq!(cache.len(), 0);
        assert_eq!(cache.get(&key), None);
    }

    #[test]
    fn test_immediate_ttl_expiry() {
        let config = CacheConfig::new().with_default_ttl(Duration::from_nanos(1));
        let mut cache = RunnableCache::new(config);

        let key = CacheKey::from_value(&json!(1));
        cache.put(key.clone(), json!(42));
        std::thread::sleep(Duration::from_millis(1));

        // Should be expired on get.
        assert_eq!(cache.get(&key), None);
    }

    #[test]
    fn test_max_entries_one() {
        let config = CacheConfig::new().with_max_entries(1);
        let mut cache = RunnableCache::new(config);

        let k1 = CacheKey::from_value(&json!(1));
        let k2 = CacheKey::from_value(&json!(2));

        cache.put(k1.clone(), json!(10));
        assert_eq!(cache.len(), 1);

        cache.put(k2.clone(), json!(20));
        assert_eq!(cache.len(), 1);
        assert!(cache.entries.contains_key(&k2));
        assert!(!cache.entries.contains_key(&k1));
    }

    // ==================== CachedRunnable tests ====================

    #[tokio::test]
    async fn test_cached_runnable_cache_hit() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_doubler(counter.clone());
        let cache = Arc::new(Mutex::new(RunnableCache::new(CacheConfig::new())));
        let cached = CachedRunnable::new(Arc::new(runnable), cache.clone());

        let r1 = cached.invoke(json!(5), None).await.unwrap();
        assert_eq!(r1, json!(10));
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Second call with same input should be a cache hit.
        let r2 = cached.invoke(json!(5), None).await.unwrap();
        assert_eq!(r2, json!(10));
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_cached_runnable_cache_miss() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_doubler(counter.clone());
        let cache = Arc::new(Mutex::new(RunnableCache::new(CacheConfig::new())));
        let cached = CachedRunnable::new(Arc::new(runnable), cache);

        cached.invoke(json!(5), None).await.unwrap();
        cached.invoke(json!(10), None).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 2); // Two different inputs
    }

    #[tokio::test]
    async fn test_cached_runnable_name_delegates() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_doubler(counter);
        let cache = Arc::new(Mutex::new(RunnableCache::new(CacheConfig::new())));
        let cached = CachedRunnable::new(Arc::new(runnable), cache);
        assert_eq!(cached.name(), "counting_doubler");
    }

    #[tokio::test]
    async fn test_cached_runnable_does_not_cache_errors() {
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
        let cache = Arc::new(Mutex::new(RunnableCache::new(CacheConfig::new())));
        let cached = CachedRunnable::new(Arc::new(runnable), cache);

        assert!(cached.invoke(json!(-1), None).await.is_err());
        assert!(cached.invoke(json!(-1), None).await.is_err());
        assert_eq!(counter.load(Ordering::SeqCst), 2); // Called twice (not cached)
    }

    #[tokio::test]
    async fn test_cached_runnable_stats() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let cache = Arc::new(Mutex::new(RunnableCache::new(CacheConfig::new())));
        let cached = CachedRunnable::new(Arc::new(runnable), cache.clone());

        cached.invoke(json!(1), None).await.unwrap(); // miss
        cached.invoke(json!(1), None).await.unwrap(); // hit
        cached.invoke(json!(2), None).await.unwrap(); // miss

        let stats = cache.lock().await.stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert_eq!(stats.current_size, 2);
    }

    #[tokio::test]
    async fn test_cached_runnable_with_eviction() {
        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let config = CacheConfig::new().with_max_entries(2);
        let cache = Arc::new(Mutex::new(RunnableCache::new(config)));
        let cached = CachedRunnable::new(Arc::new(runnable), cache.clone());

        cached.invoke(json!(1), None).await.unwrap();
        cached.invoke(json!(2), None).await.unwrap();
        cached.invoke(json!(3), None).await.unwrap(); // evicts 1

        assert_eq!(counter.load(Ordering::SeqCst), 3);
        let stats = cache.lock().await.stats();
        assert_eq!(stats.current_size, 2);
        assert_eq!(stats.evictions, 1);
    }

    #[tokio::test]
    async fn test_cached_runnable_stream_not_cached() {
        use futures::StreamExt;

        let counter = Arc::new(AtomicUsize::new(0));
        let runnable = counting_identity(counter.clone());
        let cache = Arc::new(Mutex::new(RunnableCache::new(CacheConfig::new())));
        let cached = CachedRunnable::new(Arc::new(runnable), cache.clone());

        let mut stream = cached.stream(json!(42), None).await.unwrap();
        let item = stream.next().await.unwrap().unwrap();
        assert_eq!(item, json!(42));

        // Stream should not populate cache.
        let stats = cache.lock().await.stats();
        assert_eq!(stats.current_size, 0);
    }

    // ==================== CacheInvalidator tests ====================

    #[test]
    fn test_cache_invalidator_no_patterns() {
        let invalidator = CacheInvalidator::new();
        let key = CacheKey::from_value(&json!(1));
        assert!(!invalidator.should_invalidate(&key));
    }

    #[test]
    fn test_cache_invalidator_matching_pattern() {
        let mut invalidator = CacheInvalidator::new();
        let key = CacheKey::from_value(&json!(1));
        // Add a pattern that matches a substring of the key hash.
        let prefix = key.as_str()[..4].to_string();
        invalidator.add_pattern(&prefix);
        assert!(invalidator.should_invalidate(&key));
    }

    #[test]
    fn test_cache_invalidator_non_matching_pattern() {
        let mut invalidator = CacheInvalidator::new();
        invalidator.add_pattern("zzzzzzzzz_nonexistent");
        let key = CacheKey::from_value(&json!(1));
        assert!(!invalidator.should_invalidate(&key));
    }

    #[test]
    fn test_cache_invalidator_invalidate_matching() {
        let mut cache = RunnableCache::new(CacheConfig::new());
        let k1 = CacheKey::from_value(&json!(1));
        let k2 = CacheKey::from_value(&json!(2));
        cache.put(k1.clone(), json!(10));
        cache.put(k2.clone(), json!(20));
        assert_eq!(cache.len(), 2);

        let mut invalidator = CacheInvalidator::new();
        // Match the first key's prefix.
        let prefix = k1.as_str()[..4].to_string();
        invalidator.add_pattern(&prefix);

        invalidator.invalidate_matching(&mut cache);

        // k1 should be gone (or if k2's hash also starts with the same prefix, both).
        assert!(!cache.entries.contains_key(&k1));
    }

    #[test]
    fn test_cache_invalidator_default() {
        let invalidator = CacheInvalidator::default();
        assert!(!invalidator.should_invalidate(&CacheKey::from_value(&json!(1))));
    }
}
