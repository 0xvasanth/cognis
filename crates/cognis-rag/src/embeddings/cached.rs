//! Cached embeddings — wraps any [`Embeddings`] with an in-process FIFO
//! cache keyed by the input text.
//!
//! The cache is a fixed-capacity FIFO (insertion-order eviction). Not
//! a true LRU — that would need an additional crate. For typical
//! embedding workloads (warm-cache lookups across re-indexing runs)
//! FIFO is enough.
//!
//! Customization:
//! - [`CachedEmbeddings::with_capacity`] — cache size.
//! - [`CachedEmbeddings::with_normalizer`] — pre-cache key normalizer
//!   (e.g. trim whitespace + lowercase, so equivalent inputs share a
//!   cache slot).

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Mutex;

use cognis_core::Result;

use super::Embeddings;

type Normalizer = Arc<dyn Fn(&str) -> String + Send + Sync>;

/// Fixed-capacity FIFO cache.
struct FifoCache {
    capacity: usize,
    map: HashMap<String, Vec<f32>>,
    order: VecDeque<String>,
}

impl FifoCache {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            map: HashMap::with_capacity(capacity.max(1)),
            order: VecDeque::with_capacity(capacity.max(1)),
        }
    }
    fn get(&self, k: &str) -> Option<&Vec<f32>> {
        self.map.get(k)
    }
    fn put(&mut self, k: String, v: Vec<f32>) {
        use std::collections::hash_map::Entry;
        match self.map.entry(k.clone()) {
            Entry::Occupied(mut o) => {
                o.insert(v);
            }
            Entry::Vacant(v_entry) => {
                if self.order.len() >= self.capacity {
                    if let Some(old) = self.order.pop_front() {
                        // We can't reach back into `self.map` mid-borrow,
                        // so drop the vacant entry and re-enter cleanly.
                        let _ = v_entry;
                        self.map.remove(&old);
                        self.order.push_back(k.clone());
                        self.map.insert(k, v);
                        return;
                    }
                }
                v_entry.insert(v);
                self.order.push_back(k);
            }
        }
    }
    fn len(&self) -> usize {
        self.map.len()
    }
}

/// In-process cache for embeddings.
pub struct CachedEmbeddings {
    inner: Arc<dyn Embeddings>,
    cache: Mutex<FifoCache>,
    normalizer: Option<Normalizer>,
}

impl CachedEmbeddings {
    /// Wrap with default capacity (1024).
    pub fn new(inner: Arc<dyn Embeddings>) -> Self {
        Self::with_capacity(inner, 1024)
    }

    /// Wrap with a specific capacity (>= 1).
    pub fn with_capacity(inner: Arc<dyn Embeddings>, capacity: usize) -> Self {
        Self {
            inner,
            cache: Mutex::new(FifoCache::new(capacity)),
            normalizer: None,
        }
    }

    /// Plug in a key normalizer (e.g. case folding).
    pub fn with_normalizer<F>(mut self, f: F) -> Self
    where
        F: Fn(&str) -> String + Send + Sync + 'static,
    {
        self.normalizer = Some(Arc::new(f));
        self
    }

    /// Number of entries currently cached.
    pub async fn len(&self) -> usize {
        self.cache.lock().await.len()
    }

    /// True if the cache is empty.
    pub async fn is_empty(&self) -> bool {
        self.cache.lock().await.len() == 0
    }

    fn key(&self, s: &str) -> String {
        match &self.normalizer {
            Some(n) => n(s),
            None => s.to_string(),
        }
    }
}

#[async_trait]
impl Embeddings for CachedEmbeddings {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let mut out: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut to_compute: Vec<(usize, String)> = Vec::new();
        {
            let cache = self.cache.lock().await;
            for (i, t) in texts.iter().enumerate() {
                let k = self.key(t);
                if let Some(v) = cache.get(&k) {
                    out[i] = Some(v.clone());
                } else {
                    to_compute.push((i, t.clone()));
                }
            }
        }
        if !to_compute.is_empty() {
            let pending_texts: Vec<String> = to_compute.iter().map(|(_, t)| t.clone()).collect();
            let computed = self.inner.embed_documents(pending_texts).await?;
            let mut cache = self.cache.lock().await;
            for ((i, original), v) in to_compute.into_iter().zip(computed) {
                cache.put(self.key(&original), v.clone());
                out[i] = Some(v);
            }
        }
        Ok(out.into_iter().map(|o| o.unwrap_or_default()).collect())
    }

    async fn embed_query(&self, text: String) -> Result<Vec<f32>> {
        let k = self.key(&text);
        {
            let cache = self.cache.lock().await;
            if let Some(v) = cache.get(&k) {
                return Ok(v.clone());
            }
        }
        let v = self.inner.embed_query(text).await?;
        self.cache.lock().await.put(k, v.clone());
        Ok(v)
    }

    fn dimensions(&self) -> Option<usize> {
        self.inner.dimensions()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::FakeEmbeddings;

    /// Wrap an embeddings impl with a call counter.
    struct Counting {
        inner: Arc<dyn Embeddings>,
        calls: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl Embeddings for Counting {
        async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
            self.calls
                .fetch_add(texts.len(), std::sync::atomic::Ordering::SeqCst);
            self.inner.embed_documents(texts).await
        }
        fn model(&self) -> &str {
            "counting"
        }
    }

    fn counted(dim: usize) -> Arc<Counting> {
        Arc::new(Counting {
            inner: Arc::new(FakeEmbeddings::new(dim)),
            calls: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    #[tokio::test]
    async fn second_call_hits_cache() {
        let counter = counted(8);
        let cached = CachedEmbeddings::new(counter.clone() as Arc<dyn Embeddings>);
        let _ = cached.embed_query("hello".into()).await.unwrap();
        let _ = cached.embed_query("hello".into()).await.unwrap();
        let _ = cached.embed_query("hello".into()).await.unwrap();
        assert_eq!(counter.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn batch_partial_hits_only_recomputes_misses() {
        let counter = counted(8);
        let cached = CachedEmbeddings::new(counter.clone() as Arc<dyn Embeddings>);
        let _ = cached
            .embed_documents(vec!["a".into(), "b".into()])
            .await
            .unwrap();
        let _ = cached
            .embed_documents(vec!["a".into(), "c".into()])
            .await
            .unwrap();
        // First call: 2; second call: only "c" → 1. Total 3.
        assert_eq!(counter.calls.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn normalizer_collapses_equivalent_inputs() {
        let counter = counted(8);
        let cached = CachedEmbeddings::new(counter.clone() as Arc<dyn Embeddings>)
            .with_normalizer(|s| s.trim().to_lowercase());
        let _ = cached.embed_query("Hello".into()).await.unwrap();
        let _ = cached.embed_query("  HELLO ".into()).await.unwrap();
        assert_eq!(counter.calls.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn capacity_eviction_drops_oldest() {
        let counter = counted(4);
        let cached = CachedEmbeddings::with_capacity(counter.clone() as Arc<dyn Embeddings>, 2);
        let _ = cached.embed_query("a".into()).await.unwrap();
        let _ = cached.embed_query("b".into()).await.unwrap();
        // Forces eviction of "a".
        let _ = cached.embed_query("c".into()).await.unwrap();
        // "a" should now be a miss.
        let _ = cached.embed_query("a".into()).await.unwrap();
        // a, b, c, then a again = 4 underlying calls.
        assert_eq!(counter.calls.load(std::sync::atomic::Ordering::SeqCst), 4);
        assert_eq!(cached.len().await, 2);
    }
}
