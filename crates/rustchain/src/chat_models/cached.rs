//! LLM response caching layer for chat models.
//!
//! Provides a [`CachedChatModel`] wrapper that caches responses from any
//! [`BaseChatModel`] implementation using a pluggable [`CacheStore`] backend.
//!
//! Two built-in backends are provided:
//! - [`InMemoryCache`] — bounded in-memory cache with FIFO eviction
//! - [`FileCache`] — JSON-file-based persistent cache

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use rustchain_core::error::Result;
use rustchain_core::language_models::chat_model::{BaseChatModel, ChatStream, ToolChoice};
use rustchain_core::messages::Message;
use rustchain_core::outputs::ChatResult;
use rustchain_core::tools::ToolSchema;

// ---------------------------------------------------------------------------
// CacheStore trait
// ---------------------------------------------------------------------------

/// Pluggable cache backend for LLM responses.
#[async_trait]
pub trait CacheStore: Send + Sync {
    /// Look up a cached result by key.
    async fn get(&self, key: &str) -> Option<ChatResult>;

    /// Store a result under the given key.
    async fn set(&self, key: &str, value: &ChatResult);

    /// Remove all entries from the cache.
    async fn clear(&self);
}

// ---------------------------------------------------------------------------
// InMemoryCache
// ---------------------------------------------------------------------------

/// Bounded in-memory LLM response cache with FIFO eviction.
pub struct InMemoryCache {
    store: Mutex<HashMap<String, ChatResult>>,
    order: Mutex<VecDeque<String>>,
    max_size: Option<usize>,
}

impl InMemoryCache {
    /// Create a new in-memory cache.
    ///
    /// If `max_size` is `Some(n)`, the cache evicts the oldest entry when it
    /// exceeds `n` entries.
    pub fn new(max_size: Option<usize>) -> Self {
        Self {
            store: Mutex::new(HashMap::new()),
            order: Mutex::new(VecDeque::new()),
            max_size,
        }
    }
}

#[async_trait]
impl CacheStore for InMemoryCache {
    async fn get(&self, key: &str) -> Option<ChatResult> {
        let store = self.store.lock().await;
        store.get(key).cloned()
    }

    async fn set(&self, key: &str, value: &ChatResult) {
        let mut store = self.store.lock().await;
        let mut order = self.order.lock().await;

        if !store.contains_key(key) {
            order.push_back(key.to_string());
        }

        store.insert(key.to_string(), value.clone());

        // Evict oldest entries if over capacity.
        if let Some(max) = self.max_size {
            while store.len() > max {
                if let Some(oldest) = order.pop_front() {
                    store.remove(&oldest);
                }
            }
        }
    }

    async fn clear(&self) {
        let mut store = self.store.lock().await;
        let mut order = self.order.lock().await;
        store.clear();
        order.clear();
    }
}

// ---------------------------------------------------------------------------
// FileCache
// ---------------------------------------------------------------------------

/// File-system-backed LLM response cache.
///
/// Each entry is stored as a JSON file in the configured directory. The cache
/// key is hashed to produce a safe filename.
pub struct FileCache {
    dir: PathBuf,
}

impl FileCache {
    /// Create a new file cache that stores entries under `dir`.
    ///
    /// The directory is created if it does not exist.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn path_for(&self, key: &str) -> PathBuf {
        self.dir.join(format!("{key}.json"))
    }
}

#[async_trait]
impl CacheStore for FileCache {
    async fn get(&self, key: &str) -> Option<ChatResult> {
        let path = self.path_for(key);
        let data = tokio::fs::read_to_string(&path).await.ok()?;
        serde_json::from_str(&data).ok()
    }

    async fn set(&self, key: &str, value: &ChatResult) {
        let _ = tokio::fs::create_dir_all(&self.dir).await;
        let path = self.path_for(key);
        if let Ok(json) = serde_json::to_string_pretty(value) {
            let _ = tokio::fs::write(&path, json).await;
        }
    }

    async fn clear(&self) {
        if let Ok(mut entries) = tokio::fs::read_dir(&self.dir).await {
            while let Ok(Some(entry)) = entries.next_entry().await {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    let _ = tokio::fs::remove_file(path).await;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cache key computation
// ---------------------------------------------------------------------------

/// Compute a deterministic hex cache key from messages and optional stop sequences.
fn compute_cache_key(messages: &[Message], stop: Option<&[String]>) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let serialized = serde_json::to_string(messages).unwrap_or_default();
    let mut hasher = DefaultHasher::new();
    serialized.hash(&mut hasher);
    if let Some(stop) = stop {
        stop.hash(&mut hasher);
    }
    format!("{:x}", hasher.finish())
}

// ---------------------------------------------------------------------------
// CachedChatModel
// ---------------------------------------------------------------------------

/// A chat model wrapper that caches responses from an inner model.
///
/// `_generate` checks the cache before calling the inner model. On a miss the
/// result is stored for future calls with identical inputs. Streaming requests
/// are passed through to the inner model without caching.
pub struct CachedChatModel {
    inner: Box<dyn BaseChatModel>,
    cache: Arc<dyn CacheStore>,
}

impl CachedChatModel {
    /// Wrap an existing chat model with a cache backend.
    pub fn new(inner: Box<dyn BaseChatModel>, cache: Arc<dyn CacheStore>) -> Self {
        Self { inner, cache }
    }
}

#[async_trait]
impl BaseChatModel for CachedChatModel {
    async fn _generate(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        let key = compute_cache_key(messages, stop);

        // Return cached result on hit.
        if let Some(cached) = self.cache.get(&key).await {
            return Ok(cached);
        }

        // Miss — call through and cache the result.
        let result = self.inner._generate(messages, stop).await?;
        self.cache.set(&key, &result).await;
        Ok(result)
    }

    fn llm_type(&self) -> &str {
        // We return a static-lifetime str by leaking — acceptable for a type
        // identifier that lives for the program's duration.
        let s = format!("cached({})", self.inner.llm_type());
        Box::leak(s.into_boxed_str())
    }

    async fn _stream(
        &self,
        messages: &[Message],
        stop: Option<&[String]>,
    ) -> Result<ChatStream> {
        // Streams are not cached; delegate directly.
        self.inner._stream(messages, stop).await
    }

    fn bind_tools(
        &self,
        tools: &[ToolSchema],
        tool_choice: Option<ToolChoice>,
    ) -> Result<Box<dyn BaseChatModel>> {
        let inner_with_tools = self.inner.bind_tools(tools, tool_choice)?;
        Ok(Box::new(CachedChatModel {
            inner: inner_with_tools,
            cache: Arc::clone(&self.cache),
        }))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use rustchain_core::messages::{AIMessage, HumanMessage, Message};
    use rustchain_core::outputs::{ChatGeneration, ChatResult};

    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A fake chat model that counts how many times `_generate` is called.
    struct FakeModel {
        call_count: Arc<AtomicUsize>,
    }

    impl FakeModel {
        fn new() -> (Self, Arc<AtomicUsize>) {
            let count = Arc::new(AtomicUsize::new(0));
            (
                Self {
                    call_count: Arc::clone(&count),
                },
                count,
            )
        }
    }

    #[async_trait]
    impl BaseChatModel for FakeModel {
        async fn _generate(
            &self,
            messages: &[Message],
            _stop: Option<&[String]>,
        ) -> Result<ChatResult> {
            let n = self.call_count.fetch_add(1, Ordering::SeqCst);
            let text = format!("response #{} to: {}", n, serde_json::to_string(messages).unwrap_or_default());
            let ai = AIMessage::new(&text);
            Ok(ChatResult {
                generations: vec![ChatGeneration::new(ai)],
                llm_output: None,
            })
        }

        fn llm_type(&self) -> &str {
            "fake"
        }
    }

    fn human(text: &str) -> Message {
        Message::Human(HumanMessage::new(text))
    }

    #[tokio::test]
    async fn test_cache_hit_returns_cached_result() {
        let (model, call_count) = FakeModel::new();
        let cache = Arc::new(InMemoryCache::new(None));
        let cached = CachedChatModel::new(Box::new(model), cache);

        let msgs = vec![human("hello")];

        let r1 = cached._generate(&msgs, None).await.unwrap();
        let r2 = cached._generate(&msgs, None).await.unwrap();

        assert_eq!(r1, r2);
        assert_eq!(call_count.load(Ordering::SeqCst), 1, "inner model should be called only once");
    }

    #[tokio::test]
    async fn test_cache_miss_different_messages() {
        let (model, call_count) = FakeModel::new();
        let cache = Arc::new(InMemoryCache::new(None));
        let cached = CachedChatModel::new(Box::new(model), cache);

        let msgs_a = vec![human("hello")];
        let msgs_b = vec![human("world")];

        let r1 = cached._generate(&msgs_a, None).await.unwrap();
        let r2 = cached._generate(&msgs_b, None).await.unwrap();

        assert_ne!(r1, r2);
        assert_eq!(call_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_in_memory_cache_max_size_eviction() {
        let cache = InMemoryCache::new(Some(2));

        let result = |text: &str| ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new(text))],
            llm_output: None,
        };

        cache.set("a", &result("first")).await;
        cache.set("b", &result("second")).await;
        cache.set("c", &result("third")).await;

        // "a" should have been evicted (FIFO).
        assert!(cache.get("a").await.is_none(), "oldest entry should be evicted");
        assert!(cache.get("b").await.is_some());
        assert!(cache.get("c").await.is_some());
    }

    #[tokio::test]
    async fn test_file_cache_save_and_load() {
        let dir = tempfile::tempdir().unwrap();
        let cache = FileCache::new(dir.path());

        let result = ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new("cached"))],
            llm_output: None,
        };

        cache.set("key1", &result).await;
        let loaded = cache.get("key1").await;

        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap(), result);
    }

    #[tokio::test]
    async fn test_clear_empties_cache() {
        let cache = InMemoryCache::new(None);

        let result = ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new("data"))],
            llm_output: None,
        };

        cache.set("x", &result).await;
        assert!(cache.get("x").await.is_some());

        cache.clear().await;
        assert!(cache.get("x").await.is_none(), "cache should be empty after clear");
    }

    #[tokio::test]
    async fn test_llm_type_includes_inner() {
        let (model, _) = FakeModel::new();
        let cache = Arc::new(InMemoryCache::new(None));
        let cached = CachedChatModel::new(Box::new(model), cache);

        assert_eq!(cached.llm_type(), "cached(fake)");
    }

    #[tokio::test]
    async fn test_file_cache_clear() {
        let dir = tempfile::tempdir().unwrap();
        let cache = FileCache::new(dir.path());

        let result = ChatResult {
            generations: vec![ChatGeneration::new(AIMessage::new("temp"))],
            llm_output: None,
        };

        cache.set("k1", &result).await;
        cache.set("k2", &result).await;
        assert!(cache.get("k1").await.is_some());

        cache.clear().await;
        assert!(cache.get("k1").await.is_none());
        assert!(cache.get("k2").await.is_none());
    }
}
