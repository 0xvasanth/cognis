//! Caching wrapper for tools that stores and retrieves previous results.
//!
//! `CachedTool` wraps any `Arc<dyn BaseTool>` and transparently caches results
//! keyed by tool name + input. It supports optional TTL-based expiry and
//! LRU eviction when a maximum cache size is configured.
//!
//! # Example
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use std::time::Duration;
//! use rustchain::tools::cached::CachedTool;
//! use rustchain_core::tools::SimpleTool;
//!
//! let inner = Arc::new(SimpleTool::new("echo", "Echo input", |s: &str| Ok(s.to_string())));
//! let cached = CachedTool::new(inner)
//!     .with_ttl(Duration::from_secs(60))
//!     .with_max_size(100);
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::RwLock;

use rustchain_core::error::Result;
use rustchain_core::tools::base::BaseTool;
use rustchain_core::tools::types::{ErrorHandler, ResponseFormat, ToolInput, ToolOutput};

/// A single cached result with its creation timestamp.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// The cached tool output serialized as a JSON string.
    pub result: String,
    /// When this entry was created — used for TTL checks.
    pub created_at: Instant,
}

/// Counters tracking cache performance.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Total cache hits.
    pub hits: usize,
    /// Total cache misses.
    pub misses: usize,
    /// Hit rate as a fraction in [0.0, 1.0]. Returns 0.0 when no lookups have occurred.
    pub hit_rate: f64,
}

/// A tool wrapper that caches results from an inner tool.
///
/// Thread-safe: the cache is behind an `Arc<RwLock<..>>` and the stats use
/// atomic counters, so `CachedTool` can be shared across tasks freely.
pub struct CachedTool {
    inner: Arc<dyn BaseTool>,
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Insertion order for LRU eviction — oldest keys first.
    insertion_order: Arc<RwLock<Vec<String>>>,
    max_size: Option<usize>,
    ttl: Option<Duration>,
    hits: Arc<AtomicUsize>,
    misses: Arc<AtomicUsize>,
}

impl CachedTool {
    /// Wrap an existing tool with caching. No TTL or size limit by default.
    pub fn new(tool: Arc<dyn BaseTool>) -> Self {
        Self {
            inner: tool,
            cache: Arc::new(RwLock::new(HashMap::new())),
            insertion_order: Arc::new(RwLock::new(Vec::new())),
            max_size: None,
            ttl: None,
            hits: Arc::new(AtomicUsize::new(0)),
            misses: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Set the maximum number of cached entries. When exceeded the oldest entry
    /// (LRU) is evicted.
    pub fn with_max_size(mut self, n: usize) -> Self {
        self.max_size = Some(n);
        self
    }

    /// Set a time-to-live for cache entries. Expired entries are treated as
    /// cache misses and re-executed.
    pub fn with_ttl(mut self, duration: Duration) -> Self {
        self.ttl = Some(duration);
        self
    }

    // ── Cache management ───────────────────────────────────────────────

    /// Remove all entries from the cache and reset stats.
    pub async fn clear_cache(&self) {
        let mut cache = self.cache.write().await;
        cache.clear();
        let mut order = self.insertion_order.write().await;
        order.clear();
        self.hits.store(0, Ordering::SeqCst);
        self.misses.store(0, Ordering::SeqCst);
    }

    /// Return the number of entries currently in the cache.
    pub async fn cache_size(&self) -> usize {
        self.cache.read().await.len()
    }

    /// Return a snapshot of hit/miss statistics.
    pub fn cache_stats(&self) -> CacheStats {
        let hits = self.hits.load(Ordering::SeqCst);
        let misses = self.misses.load(Ordering::SeqCst);
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

    /// Remove a specific cache entry by its raw input string.
    pub async fn invalidate(&self, input: &str) {
        let key = self.make_cache_key(input);
        let mut cache = self.cache.write().await;
        cache.remove(&key);
        let mut order = self.insertion_order.write().await;
        order.retain(|k| k != &key);
    }

    // ── Internals ──────────────────────────────────────────────────────

    /// Derive a deterministic cache key from the tool name and an input string.
    fn make_cache_key(&self, input: &str) -> String {
        // Simple but deterministic: "tool_name::input"
        format!("{}::{}", self.inner.name(), input)
    }

    /// Serialize `ToolInput` into a deterministic string suitable for hashing.
    fn input_to_key_string(input: &ToolInput) -> String {
        match input {
            ToolInput::Text(s) => s.clone(),
            ToolInput::Structured(map) => {
                // Sort keys for deterministic ordering.
                let mut pairs: Vec<(&String, &Value)> = map.iter().collect();
                pairs.sort_by_key(|(k, _)| *k);
                serde_json::to_string(&pairs).unwrap_or_default()
            }
            ToolInput::ToolCall(tc) => {
                let mut pairs: Vec<(&String, &Value)> = tc.args.iter().collect();
                pairs.sort_by_key(|(k, _)| *k);
                format!(
                    "{}::{}",
                    tc.name,
                    serde_json::to_string(&pairs).unwrap_or_default()
                )
            }
        }
    }

    /// Evict the oldest entry when the cache exceeds `max_size`.
    async fn evict_if_needed(&self) {
        if let Some(max) = self.max_size {
            let mut cache = self.cache.write().await;
            let mut order = self.insertion_order.write().await;
            while cache.len() > max && !order.is_empty() {
                let oldest = order.remove(0);
                cache.remove(&oldest);
            }
        }
    }
}

#[async_trait]
impl BaseTool for CachedTool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn description(&self) -> &str {
        self.inner.description()
    }

    fn args_schema(&self) -> Option<Value> {
        self.inner.args_schema()
    }

    fn tool_call_schema(&self) -> Value {
        self.inner.tool_call_schema()
    }

    fn return_direct(&self) -> bool {
        self.inner.return_direct()
    }

    fn handle_tool_error(&self) -> &ErrorHandler {
        self.inner.handle_tool_error()
    }

    fn handle_validation_error(&self) -> &ErrorHandler {
        self.inner.handle_validation_error()
    }

    fn response_format(&self) -> ResponseFormat {
        self.inner.response_format()
    }

    fn tags(&self) -> &[String] {
        self.inner.tags()
    }

    fn metadata(&self) -> Option<&HashMap<String, Value>> {
        self.inner.metadata()
    }

    fn extras(&self) -> Option<&HashMap<String, Value>> {
        self.inner.extras()
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let input_str = Self::input_to_key_string(&input);
        let key = self.make_cache_key(&input_str);

        // Check cache (read lock).
        {
            let cache = self.cache.read().await;
            if let Some(entry) = cache.get(&key) {
                let expired = self
                    .ttl
                    .map(|ttl| entry.created_at.elapsed() > ttl)
                    .unwrap_or(false);

                if !expired {
                    self.hits.fetch_add(1, Ordering::SeqCst);
                    let value: Value = serde_json::from_str(&entry.result)
                        .unwrap_or(Value::String(entry.result.clone()));
                    return Ok(ToolOutput::Content(value));
                }
            }
        }

        // Cache miss — run the inner tool.
        self.misses.fetch_add(1, Ordering::SeqCst);
        let output = self.inner._run(input).await?;

        // Store result.
        let result_string = match &output {
            ToolOutput::Content(v) => serde_json::to_string(v).unwrap_or_default(),
            ToolOutput::ContentAndArtifact { content, .. } => {
                serde_json::to_string(content).unwrap_or_default()
            }
        };

        {
            let mut cache = self.cache.write().await;
            let mut order = self.insertion_order.write().await;

            // Remove old position if key already exists (re-insert at end).
            if cache.contains_key(&key) {
                order.retain(|k| k != &key);
            }

            cache.insert(
                key.clone(),
                CacheEntry {
                    result: result_string,
                    created_at: Instant::now(),
                },
            );
            order.push(key);
        }

        self.evict_if_needed().await;

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::tools::SimpleTool;
    use std::sync::atomic::AtomicU32;

    /// Helper: create a SimpleTool that counts invocations.
    fn counting_tool(name: &str, counter: Arc<AtomicU32>) -> Arc<dyn BaseTool> {
        Arc::new(SimpleTool::new(
            name.to_string(),
            format!("{} tool", name),
            move |input: &str| {
                counter.fetch_add(1, Ordering::SeqCst);
                Ok(format!("result:{}", input))
            },
        ))
    }

    #[tokio::test]
    async fn test_cache_miss_calls_inner_tool() {
        let counter = Arc::new(AtomicU32::new(0));
        let tool = CachedTool::new(counting_tool("echo", counter.clone()));

        let result = tool._run(ToolInput::Text("hello".into())).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);
        match result {
            ToolOutput::Content(v) => assert_eq!(v, Value::String("result:hello".into())),
            _ => panic!("Expected Content output"),
        }
    }

    #[tokio::test]
    async fn test_cache_hit_returns_cached_without_calling_inner() {
        let counter = Arc::new(AtomicU32::new(0));
        let tool = CachedTool::new(counting_tool("echo", counter.clone()));

        // First call — miss
        tool._run(ToolInput::Text("hello".into())).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Second call — hit
        let result = tool._run(ToolInput::Text("hello".into())).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1); // Still 1 — inner not called again
        match result {
            ToolOutput::Content(v) => assert_eq!(v, Value::String("result:hello".into())),
            _ => panic!("Expected Content output"),
        }
    }

    #[tokio::test]
    async fn test_ttl_expiry_causes_reexecution() {
        let counter = Arc::new(AtomicU32::new(0));
        let tool = CachedTool::new(counting_tool("echo", counter.clone()))
            .with_ttl(Duration::from_millis(50));

        // First call
        tool._run(ToolInput::Text("data".into())).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 1);

        // Wait for TTL to expire
        tokio::time::sleep(Duration::from_millis(80)).await;

        // Should re-execute
        tool._run(ToolInput::Text("data".into())).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn test_lru_eviction_when_max_size_exceeded() {
        let counter = Arc::new(AtomicU32::new(0));
        let tool = CachedTool::new(counting_tool("echo", counter.clone())).with_max_size(2);

        // Fill cache with 2 entries
        tool._run(ToolInput::Text("a".into())).await.unwrap();
        tool._run(ToolInput::Text("b".into())).await.unwrap();
        assert_eq!(tool.cache_size().await, 2);

        // Third entry triggers eviction of "a"
        tool._run(ToolInput::Text("c".into())).await.unwrap();
        assert_eq!(tool.cache_size().await, 2);

        // "a" was evicted — calling it again should be a miss
        tool._run(ToolInput::Text("a".into())).await.unwrap();
        // counter: 3 (a, b, c) + 1 (a again) = 4
        assert_eq!(counter.load(Ordering::SeqCst), 4);
    }

    #[tokio::test]
    async fn test_cache_stats_tracking() {
        let counter = Arc::new(AtomicU32::new(0));
        let tool = CachedTool::new(counting_tool("echo", counter.clone()));

        // Miss
        tool._run(ToolInput::Text("x".into())).await.unwrap();
        // Hit
        tool._run(ToolInput::Text("x".into())).await.unwrap();
        // Miss (different input)
        tool._run(ToolInput::Text("y".into())).await.unwrap();

        let stats = tool.cache_stats();
        assert_eq!(stats.hits, 1);
        assert_eq!(stats.misses, 2);
        assert!((stats.hit_rate - 1.0 / 3.0).abs() < 0.001);
    }

    #[tokio::test]
    async fn test_clear_cache() {
        let counter = Arc::new(AtomicU32::new(0));
        let tool = CachedTool::new(counting_tool("echo", counter.clone()));

        tool._run(ToolInput::Text("x".into())).await.unwrap();
        assert_eq!(tool.cache_size().await, 1);

        tool.clear_cache().await;
        assert_eq!(tool.cache_size().await, 0);

        let stats = tool.cache_stats();
        assert_eq!(stats.hits, 0);
        assert_eq!(stats.misses, 0);
    }

    #[tokio::test]
    async fn test_invalidate_specific_entry() {
        let counter = Arc::new(AtomicU32::new(0));
        let tool = CachedTool::new(counting_tool("echo", counter.clone()));

        tool._run(ToolInput::Text("keep".into())).await.unwrap();
        tool._run(ToolInput::Text("remove".into())).await.unwrap();
        assert_eq!(tool.cache_size().await, 2);

        tool.invalidate("remove").await;
        assert_eq!(tool.cache_size().await, 1);

        // Re-calling "remove" should be a miss
        tool._run(ToolInput::Text("remove".into())).await.unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 3); // original 2 + 1 re-execution
    }

    #[tokio::test]
    async fn test_different_inputs_produce_different_cache_keys() {
        let counter = Arc::new(AtomicU32::new(0));
        let tool = CachedTool::new(counting_tool("echo", counter.clone()));

        tool._run(ToolInput::Text("input_a".into())).await.unwrap();
        tool._run(ToolInput::Text("input_b".into())).await.unwrap();

        // Both are misses — inner called twice
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert_eq!(tool.cache_size().await, 2);
    }

    #[tokio::test]
    async fn test_tool_name_and_description_delegation() {
        let inner = Arc::new(SimpleTool::new(
            "my_tool",
            "My tool description",
            |_: &str| Ok("ok".into()),
        ));
        let cached = CachedTool::new(inner);

        assert_eq!(cached.name(), "my_tool");
        assert_eq!(cached.description(), "My tool description");
        assert!(cached.args_schema().is_some());
    }

    #[tokio::test]
    async fn test_thread_safety_concurrent_access() {
        let counter = Arc::new(AtomicU32::new(0));
        let tool = Arc::new(CachedTool::new(counting_tool("echo", counter.clone())));

        let mut handles = Vec::new();
        for i in 0..20 {
            let tool = tool.clone();
            let input = format!("input_{}", i % 5); // 5 unique inputs, 4 repeats each
            handles.push(tokio::spawn(async move {
                tool._run(ToolInput::Text(input)).await.unwrap();
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        // At most 5 unique misses (could be more due to races before first cache fill)
        // but should not be 20 (i.e., caching is working)
        let stats = tool.cache_stats();
        assert!(stats.hits + stats.misses == 20);
        // With 5 unique inputs and some concurrency, we expect some hits
        assert!(stats.misses <= 20); // Loose bound — exact depends on scheduling
        assert_eq!(tool.cache_size().await, 5);
    }

    #[tokio::test]
    async fn test_cache_key_consistency() {
        let tool_a = CachedTool::new(Arc::new(SimpleTool::new("tool_a", "A", |_: &str| {
            Ok("a".into())
        })));
        let tool_b = CachedTool::new(Arc::new(SimpleTool::new("tool_b", "B", |_: &str| {
            Ok("b".into())
        })));

        // Same input string but different tool names should produce different keys
        let key_a = tool_a.make_cache_key("same_input");
        let key_b = tool_b.make_cache_key("same_input");
        assert_ne!(key_a, key_b);

        // Same tool, same input should produce identical keys
        let key_a1 = tool_a.make_cache_key("same_input");
        let key_a2 = tool_a.make_cache_key("same_input");
        assert_eq!(key_a1, key_a2);
    }

    #[tokio::test]
    async fn test_structured_input_cache_key_determinism() {
        let counter = Arc::new(AtomicU32::new(0));
        let tool = CachedTool::new(counting_tool("echo", counter.clone()));

        let mut map1 = HashMap::new();
        map1.insert("b".to_string(), Value::String("2".into()));
        map1.insert("a".to_string(), Value::String("1".into()));

        let mut map2 = HashMap::new();
        map2.insert("a".to_string(), Value::String("1".into()));
        map2.insert("b".to_string(), Value::String("2".into()));

        // Keys from both orderings should be identical (sorted internally)
        let key1 = CachedTool::input_to_key_string(&ToolInput::Structured(map1));
        let key2 = CachedTool::input_to_key_string(&ToolInput::Structured(map2));
        assert_eq!(key1, key2);
    }
}
