//! Wrap any [`Tool`] with input-keyed result caching.

use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::wrappers::{CacheBackend, MemoryCache};
use cognis2_core::Result;
use cognis2_llm::tools::{Tool, ToolInput, ToolOutput};

type KeyFn = dyn Fn(&ToolInput) -> String + Send + Sync;

/// Wraps a `Tool` so identical inputs (per `key_fn`) return cached outputs.
///
/// Defaults to a process-local [`MemoryCache`]; pair with a sqlite cache
/// or other [`CacheBackend`] for persistence.
pub struct CachedTool {
    inner: Arc<dyn Tool>,
    backend: Arc<dyn CacheBackend<String, ToolOutput>>,
    key_fn: Arc<KeyFn>,
}

impl CachedTool {
    /// Wrap with a memory backend and the default JSON-serialized input as key.
    pub fn new(inner: Arc<dyn Tool>) -> Self {
        Self {
            inner,
            backend: Arc::new(MemoryCache::<String, ToolOutput>::new()),
            key_fn: Arc::new(default_key),
        }
    }

    /// Override the cache backend.
    pub fn with_backend(mut self, b: Arc<dyn CacheBackend<String, ToolOutput>>) -> Self {
        self.backend = b;
        self
    }

    /// Override the key function (defaults to `serde_json::to_string` of input).
    pub fn with_key_fn<F>(mut self, f: F) -> Self
    where
        F: Fn(&ToolInput) -> String + Send + Sync + 'static,
    {
        self.key_fn = Arc::new(f);
        self
    }
}

fn default_key(input: &ToolInput) -> String {
    serde_json::to_string(&input.clone().into_json()).unwrap_or_default()
}

#[async_trait]
impl Tool for CachedTool {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn description(&self) -> &str {
        self.inner.description()
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        self.inner.args_schema()
    }
    fn return_direct(&self) -> bool {
        self.inner.return_direct()
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let key = (self.key_fn)(&input);
        if let Some(hit) = self.backend.get(&key).await {
            return Ok(hit);
        }
        let out = self.inner._run(input).await?;
        self.backend.set(key, out.clone()).await;
        Ok(out)
    }
}

// `ToolOutput` already derives `Clone` (it's `Serialize + Deserialize` and
// holds owned data).

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counter {
        calls: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl Tool for Counter {
        fn name(&self) -> &str {
            "counter"
        }
        fn description(&self) -> &str {
            "counts"
        }
        fn args_schema(&self) -> Option<serde_json::Value> {
            None
        }
        async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ToolOutput::Content(input.into_json()))
        }
    }

    #[tokio::test]
    async fn caches_repeated_input() {
        let calls = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn Tool> = Arc::new(Counter {
            calls: calls.clone(),
        });
        let cached = CachedTool::new(inner);
        cached._run(ToolInput::Text("a".into())).await.unwrap();
        cached._run(ToolInput::Text("a".into())).await.unwrap();
        cached._run(ToolInput::Text("b".into())).await.unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }
}
