//! In-memory key-based cache wrapper.

use std::collections::HashMap;
use std::hash::Hash;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use crate::runnable::{Runnable, RunnableConfig};
use crate::Result;

/// Pluggable cache backend.
///
/// Implementations decide eviction, TTL, and persistence. The default
/// [`MemoryCache`] is unbounded and lives only as long as the process.
#[async_trait]
pub trait CacheBackend<K, V>: Send + Sync
where
    K: Send + Sync + 'static,
    V: Send + Sync + 'static,
{
    /// Look up a value. `None` if not present.
    async fn get(&self, key: &K) -> Option<V>;
    /// Store a value.
    async fn set(&self, key: K, value: V);
}

/// Unbounded in-memory `HashMap`-backed cache.
pub struct MemoryCache<K, V> {
    inner: RwLock<HashMap<K, V>>,
}

impl<K, V> Default for MemoryCache<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> MemoryCache<K, V> {
    /// Construct an empty `MemoryCache`.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
        }
    }
}

#[async_trait]
impl<K, V> CacheBackend<K, V> for MemoryCache<K, V>
where
    K: Hash + Eq + Send + Sync + Clone + 'static,
    V: Clone + Send + Sync + 'static,
{
    async fn get(&self, key: &K) -> Option<V> {
        self.inner.read().await.get(key).cloned()
    }
    async fn set(&self, key: K, value: V) {
        self.inner.write().await.insert(key, value);
    }
}

type KeyFn<I, K> = dyn Fn(&I) -> K + Send + Sync;

/// Caches results of the inner runnable, keyed by a user-supplied
/// `key_fn(&I)`. On miss, runs the inner and stores the output.
pub struct Cache<R, I, O, K, B> {
    inner: R,
    backend: Arc<B>,
    key_fn: Arc<KeyFn<I, K>>,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<R, I, O, K, B> Cache<R, I, O, K, B>
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Send + Sync + Clone + 'static,
    K: Send + Sync + 'static,
    B: CacheBackend<K, O>,
{
    /// Build a cache wrapper.
    pub fn new<F>(inner: R, backend: Arc<B>, key_fn: F) -> Self
    where
        F: Fn(&I) -> K + Send + Sync + 'static,
    {
        Self {
            inner,
            backend,
            key_fn: Arc::new(key_fn),
            _phantom: PhantomData,
        }
    }
}

#[async_trait]
impl<R, I, O, K, B> Runnable<I, O> for Cache<R, I, O, K, B>
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Clone + Send + Sync + 'static,
    K: Send + Sync + 'static,
    B: CacheBackend<K, O> + 'static,
{
    async fn invoke(&self, input: I, config: RunnableConfig) -> Result<O> {
        let key = (self.key_fn)(&input);
        if let Some(hit) = self.backend.get(&key).await {
            return Ok(hit);
        }
        let out = self.inner.invoke(input, config).await?;
        self.backend.set(key, out.clone()).await;
        Ok(out)
    }
    fn name(&self) -> &str {
        "Cache"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    struct Counter {
        calls: Arc<AtomicU32>,
    }

    #[async_trait]
    impl Runnable<u32, u32> for Counter {
        async fn invoke(&self, input: u32, _: RunnableConfig) -> Result<u32> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(input * 10)
        }
    }

    #[tokio::test]
    async fn caches_on_repeated_input() {
        let calls = Arc::new(AtomicU32::new(0));
        let backend = Arc::new(MemoryCache::<u32, u32>::new());
        let cached = Cache::new(
            Counter {
                calls: calls.clone(),
            },
            backend,
            |i: &u32| *i,
        );
        let cfg = RunnableConfig::default();
        assert_eq!(cached.invoke(3, cfg.clone()).await.unwrap(), 30);
        assert_eq!(cached.invoke(3, cfg.clone()).await.unwrap(), 30);
        assert_eq!(cached.invoke(4, cfg.clone()).await.unwrap(), 40);
        assert_eq!(calls.load(Ordering::SeqCst), 2); // 3 hit cache once, 4 ran
    }
}
