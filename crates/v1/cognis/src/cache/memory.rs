//! In-memory LRU cache with optional TTL.
//!
//! [`InMemoryCache`] stores [`ChatResult`] entries in a `HashMap` with LRU
//! eviction when a maximum capacity is set. Each entry can optionally expire
//! after a configurable time-to-live (TTL).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::RwLock;

use cognis_core::outputs::ChatResult;

use super::LlmCache;

/// A single cache entry with its value and metadata.
#[derive(Clone)]
struct CacheEntry {
    value: ChatResult,
    /// When this entry was last accessed (for LRU ordering).
    last_accessed: Instant,
    /// When this entry was inserted (for TTL expiry).
    created_at: Instant,
}

/// Thread-safe, bounded in-memory LLM response cache with LRU eviction and
/// optional TTL.
///
/// # Example
///
/// ```rust
/// use cognis::cache::InMemoryCache;
/// use std::time::Duration;
///
/// // Cache up to 100 entries, each valid for 5 minutes.
/// let cache = InMemoryCache::builder()
///     .max_size(100)
///     .ttl(Duration::from_secs(300))
///     .build();
/// ```
pub struct InMemoryCache {
    store: Arc<RwLock<HashMap<String, CacheEntry>>>,
    max_size: Option<usize>,
    ttl: Option<Duration>,
}

/// Builder for [`InMemoryCache`].
pub struct InMemoryCacheBuilder {
    max_size: Option<usize>,
    ttl: Option<Duration>,
}

impl InMemoryCacheBuilder {
    /// Set the maximum number of entries before LRU eviction kicks in.
    pub fn max_size(mut self, size: usize) -> Self {
        self.max_size = Some(size);
        self
    }

    /// Set the time-to-live for each cache entry.
    pub fn ttl(mut self, duration: Duration) -> Self {
        self.ttl = Some(duration);
        self
    }

    /// Build the cache.
    pub fn build(self) -> InMemoryCache {
        InMemoryCache {
            store: Arc::new(RwLock::new(HashMap::new())),
            max_size: self.max_size,
            ttl: self.ttl,
        }
    }
}

impl InMemoryCache {
    /// Create a new unbounded in-memory cache with no TTL.
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
            max_size: None,
            ttl: None,
        }
    }

    /// Return a builder for configuring the cache.
    pub fn builder() -> InMemoryCacheBuilder {
        InMemoryCacheBuilder {
            max_size: None,
            ttl: None,
        }
    }

    /// Return the current number of entries (including possibly expired ones).
    pub async fn len(&self) -> usize {
        self.store.read().await.len()
    }

    /// Return `true` if the cache contains no entries.
    pub async fn is_empty(&self) -> bool {
        self.store.read().await.is_empty()
    }

    /// Check whether an entry has expired according to the configured TTL.
    fn is_expired(&self, entry: &CacheEntry) -> bool {
        if let Some(ttl) = self.ttl {
            entry.created_at.elapsed() > ttl
        } else {
            false
        }
    }

    /// Evict the least-recently-used entry from the store.
    ///
    /// Must be called while holding a write lock (the store is passed by ref).
    fn evict_lru(store: &mut HashMap<String, CacheEntry>) {
        if let Some(lru_key) = store
            .iter()
            .min_by_key(|(_, entry)| entry.last_accessed)
            .map(|(k, _)| k.clone())
        {
            store.remove(&lru_key);
        }
    }
}

impl Default for InMemoryCache {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmCache for InMemoryCache {
    async fn get(&self, key: &str) -> Option<ChatResult> {
        // First try with a read lock.
        {
            let store = self.store.read().await;
            let entry = store.get(key)?;
            if self.is_expired(entry) {
                // Need write lock to remove expired entry — drop read lock first.
                drop(store);
                let mut store = self.store.write().await;
                store.remove(key);
                return None;
            }
            // Clone the value; we'll update last_accessed below.
            let value = entry.value.clone();
            drop(store);

            // Update last_accessed under write lock.
            let mut store = self.store.write().await;
            if let Some(entry) = store.get_mut(key) {
                entry.last_accessed = Instant::now();
            }
            Some(value)
        }
    }

    async fn put(&self, key: &str, result: &ChatResult) {
        let mut store = self.store.write().await;

        let now = Instant::now();
        let entry = CacheEntry {
            value: result.clone(),
            last_accessed: now,
            created_at: now,
        };

        // If the key already exists, just update it.
        if store.contains_key(key) {
            store.insert(key.to_string(), entry);
            return;
        }

        // Evict if at capacity.
        if let Some(max) = self.max_size {
            while store.len() >= max {
                Self::evict_lru(&mut store);
            }
        }

        store.insert(key.to_string(), entry);
    }

    async fn clear(&self) {
        let mut store = self.store.write().await;
        store.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::messages::AIMessage;
    use cognis_core::outputs::{ChatGeneration, ChatResult};

    fn make_result(text: &str) -> ChatResult {
        ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new(text))],
            llm_output: None,
        }
    }

    #[tokio::test]
    async fn test_put_and_get() {
        let cache = InMemoryCache::new();
        let result = make_result("hello");

        cache.put("k1", &result).await;
        let got = cache.get("k1").await;

        assert!(got.is_some());
        assert_eq!(got.unwrap(), result);
    }

    #[tokio::test]
    async fn test_get_returns_none_on_miss() {
        let cache = InMemoryCache::new();
        assert!(cache.get("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn test_clear_empties_cache() {
        let cache = InMemoryCache::new();
        cache.put("a", &make_result("a")).await;
        cache.put("b", &make_result("b")).await;
        assert_eq!(cache.len().await, 2);

        cache.clear().await;
        assert!(cache.is_empty().await);
        assert!(cache.get("a").await.is_none());
    }

    #[tokio::test]
    async fn test_lru_eviction() {
        let cache = InMemoryCache::builder().max_size(2).build();

        cache.put("a", &make_result("first")).await;
        cache.put("b", &make_result("second")).await;

        // Access "a" to make it more recently used.
        let _ = cache.get("a").await;

        // Insert "c" — should evict "b" (least recently used).
        cache.put("c", &make_result("third")).await;

        assert!(
            cache.get("a").await.is_some(),
            "a should survive (recently accessed)"
        );
        assert!(cache.get("b").await.is_none(), "b should be evicted (LRU)");
        assert!(cache.get("c").await.is_some(), "c should be present");
    }

    #[tokio::test]
    async fn test_ttl_expiry() {
        let cache = InMemoryCache::builder()
            .ttl(Duration::from_millis(50))
            .build();

        cache.put("k", &make_result("ephemeral")).await;
        assert!(cache.get("k").await.is_some());

        // Wait for TTL to expire.
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(cache.get("k").await.is_none(), "entry should have expired");
    }

    #[tokio::test]
    async fn test_overwrite_existing_key() {
        let cache = InMemoryCache::builder().max_size(2).build();

        cache.put("k", &make_result("v1")).await;
        cache.put("k", &make_result("v2")).await;

        let got = cache.get("k").await.unwrap();
        assert_eq!(got.generations[0].text, "v2");
        assert_eq!(cache.len().await, 1, "overwrite should not increase size");
    }

    #[tokio::test]
    async fn test_unbounded_cache() {
        let cache = InMemoryCache::new();
        for i in 0..100 {
            cache
                .put(&format!("k{i}"), &make_result(&format!("v{i}")))
                .await;
        }
        assert_eq!(cache.len().await, 100);
        assert!(cache.get("k0").await.is_some());
        assert!(cache.get("k99").await.is_some());
    }
}
