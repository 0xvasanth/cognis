//! Fluent extension methods for any `Runnable`.
//!
//! Imported via the prelude or directly:
//!
//! ```ignore
//! use cognis2_core::RunnableExt;
//!
//! let chain = prompt.pipe(model).pipe(parser);
//! let resilient = model.with_max_retries(3).with_timeout(Duration::from_secs(30));
//! ```

use std::sync::Arc;
use std::time::Duration;

use crate::compose::{Each, Pipe};
use crate::runnable::Runnable;
use crate::wrappers::{Cache, Fallback, MemoryCache, Retry, RetryPolicy, Timeout};

/// Adds composition + wrapper methods to any `Runnable`.
pub trait RunnableExt<I, O>: Runnable<I, O> + Sized
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Pipe this runnable into another, building a `Pipe<Self, Next>`.
    fn pipe<R2, O2>(self, next: R2) -> Pipe<Self, R2, I, O, O2>
    where
        R2: Runnable<O, O2>,
        O2: Send + 'static,
    {
        Pipe::new(self, next)
    }

    /// Wrap with a retry policy.
    fn with_retry(self, policy: RetryPolicy) -> Retry<Self, I, O>
    where
        I: Clone,
    {
        Retry::new(self, policy)
    }

    /// Shortcut: retry with default policy and N attempts.
    fn with_max_retries(self, attempts: u32) -> Retry<Self, I, O>
    where
        I: Clone,
    {
        Retry::new(self, RetryPolicy::new(attempts))
    }

    /// Wrap with a per-call timeout.
    fn with_timeout(self, duration: Duration) -> Timeout<Self, I, O> {
        Timeout::new(self, duration)
    }

    /// Wrap with a fallback runnable.
    fn with_fallback<F>(self, fallback: F) -> Fallback<Self, F, I, O>
    where
        F: Runnable<I, O>,
        I: Clone,
    {
        Fallback::new(self, fallback)
    }

    /// Wrap with an in-memory cache keyed by `key_fn(&I)`.
    fn with_memory_cache<K, F>(self, key_fn: F) -> Cache<Self, I, O, K, MemoryCache<K, O>>
    where
        K: std::hash::Hash + Eq + Clone + Send + Sync + 'static,
        O: Clone + Send + Sync + 'static,
        F: Fn(&I) -> K + Send + Sync + 'static,
    {
        Cache::new(self, Arc::new(MemoryCache::new()), key_fn)
    }

    /// Apply this runnable to each element of a `Vec<I>` (preserves order).
    fn each(self) -> Each<Self, I, O> {
        Each::new(self)
    }
}

impl<R, I, O> RunnableExt<I, O> for R
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Send + 'static,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnable::RunnableConfig;
    use crate::Result;
    use async_trait::async_trait;

    struct Inc;

    #[async_trait]
    impl Runnable<u32, u32> for Inc {
        async fn invoke(&self, input: u32, _: RunnableConfig) -> Result<u32> {
            Ok(input + 1)
        }
    }

    struct Double;

    #[async_trait]
    impl Runnable<u32, u32> for Double {
        async fn invoke(&self, input: u32, _: RunnableConfig) -> Result<u32> {
            Ok(input * 2)
        }
    }

    #[tokio::test]
    async fn pipe_method_works() {
        let chain = Inc.pipe(Double).pipe(Inc);
        let out = chain.invoke(3, RunnableConfig::default()).await.unwrap();
        assert_eq!(out, ((3 + 1) * 2) + 1);
    }

    #[tokio::test]
    async fn each_works() {
        let mapper = Inc.each();
        let out = mapper
            .invoke(vec![1, 2, 3], RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out, vec![2, 3, 4]);
    }
}
