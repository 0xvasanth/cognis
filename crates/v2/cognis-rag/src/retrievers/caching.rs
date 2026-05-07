//! Cache retriever results by query string.

use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::wrappers::{CacheBackend, MemoryCache};
use cognis2_core::{Result, Runnable, RunnableConfig};

use crate::document::Document;

/// Wraps any `Runnable<String, Vec<Document>>` so identical query
/// strings return cached document lists.
pub struct CachingRetriever {
    inner: Arc<dyn Runnable<String, Vec<Document>>>,
    backend: Arc<dyn CacheBackend<String, Vec<Document>>>,
}

impl CachingRetriever {
    /// Build with a memory cache.
    pub fn new(inner: Arc<dyn Runnable<String, Vec<Document>>>) -> Self {
        Self {
            inner,
            backend: Arc::new(MemoryCache::<String, Vec<Document>>::new()),
        }
    }

    /// Override the cache backend.
    pub fn with_backend(mut self, b: Arc<dyn CacheBackend<String, Vec<Document>>>) -> Self {
        self.backend = b;
        self
    }
}

#[async_trait]
impl Runnable<String, Vec<Document>> for CachingRetriever {
    async fn invoke(&self, query: String, config: RunnableConfig) -> Result<Vec<Document>> {
        if let Some(hit) = self.backend.get(&query).await {
            return Ok(hit);
        }
        let out = self.inner.invoke(query.clone(), config).await?;
        self.backend.set(query, out.clone()).await;
        Ok(out)
    }
    fn name(&self) -> &str {
        "CachingRetriever"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counter {
        calls: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl Runnable<String, Vec<Document>> for Counter {
        async fn invoke(&self, q: String, _: RunnableConfig) -> Result<Vec<Document>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![Document::new(q)])
        }
    }

    #[tokio::test]
    async fn second_identical_query_hits_cache() {
        let calls = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn Runnable<String, Vec<Document>>> = Arc::new(Counter {
            calls: calls.clone(),
        });
        let r = CachingRetriever::new(inner);
        let _ = r
            .invoke("a".into(), RunnableConfig::default())
            .await
            .unwrap();
        let _ = r
            .invoke("a".into(), RunnableConfig::default())
            .await
            .unwrap();
        let _ = r
            .invoke("b".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
