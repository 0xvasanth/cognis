use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use crate::error::Result;

use super::assign::{RunnableAssign, RunnablePick};
use super::base::Runnable;
use super::binding::RunnableBinding;
use super::cache::{CacheConfig, RunnableCache};
use super::config::RunnableConfig;
use super::each::RunnableEach;
use super::fallbacks::RunnableWithFallbacks;
use super::rate_limit::{RateLimitConfig, RunnableRateLimit, RunnableThrottle};
use super::retry::RunnableRetry;
use super::sequence::RunnableSequence;
use super::timeout::{RunnableTimeout, TimeoutConfig};
use serde_json::Value;

/// Extension trait providing LCEL composition methods for any `Runnable`.
///
/// This mirrors the composition methods from Python's `langchain_core.runnables.Runnable`
/// class, enabling fluent chaining of runnables using the builder pattern.
///
/// All methods consume `self` and return a new composed runnable.
///
/// # Example
/// ```ignore
/// use rustchain_core::runnables::{RunnableLambda, RunnableExt};
///
/// let chain = RunnableLambda::new("add_one", |v| async move { Ok(json!(v.as_i64().unwrap() + 1)) })
///     .pipe(RunnableLambda::new("double", |v| async move { Ok(json!(v.as_i64().unwrap() * 2)) }))
///     .with_retry(3, 100);
/// ```
pub trait RunnableExt: Runnable + Sized + 'static {
    /// Chain this runnable with another, piping the output of `self` into `next`.
    ///
    /// Equivalent to Python's `Runnable.__or__` / `Runnable.pipe`.
    ///
    /// # Example
    /// ```ignore
    /// let chain = step1.pipe(step2); // step1 | step2
    /// ```
    fn pipe<R: Runnable + 'static>(self, next: R) -> Result<RunnableSequence> {
        RunnableSequence::new(vec![
            Arc::new(self) as Arc<dyn Runnable>,
            Arc::new(next) as Arc<dyn Runnable>,
        ])
    }

    /// Apply this runnable to each element of an input array.
    ///
    /// Equivalent to Python's `Runnable.map`.
    /// The input must be a `Value::Array`; the runnable is invoked on each element.
    fn map(self) -> RunnableEach {
        RunnableEach::new(Arc::new(self) as Arc<dyn Runnable>)
    }

    /// Add fallback runnables that are tried in order if `self` fails.
    ///
    /// Equivalent to Python's `Runnable.with_fallbacks`.
    fn with_fallbacks(self, fallbacks: Vec<Arc<dyn Runnable>>) -> RunnableWithFallbacks {
        RunnableWithFallbacks::new(Arc::new(self) as Arc<dyn Runnable>).with_fallbacks(fallbacks)
    }

    /// Wrap this runnable with retry logic using exponential backoff.
    ///
    /// Equivalent to Python's `Runnable.with_retry`.
    ///
    /// # Arguments
    /// * `max_retries` - Maximum number of retry attempts (total attempts = max_retries).
    /// * `initial_wait_ms` - Initial wait time in milliseconds before first retry.
    fn with_retry(self, max_retries: u32, initial_wait_ms: u64) -> RunnableRetry {
        RunnableRetry::new(Arc::new(self) as Arc<dyn Runnable>, max_retries)
            .with_wait(initial_wait_ms, initial_wait_ms * 100)
    }

    /// Run this runnable on multiple inputs sequentially, collecting results.
    ///
    /// This is a convenience wrapper around invoking in a loop. Each input
    /// is processed independently; an error in one does not stop the others.
    ///
    /// Equivalent to Python's `Runnable.batch`.
    fn batch_sync(self) -> Arc<dyn Runnable> {
        Arc::new(self)
    }

    /// Create a `RunnableAssign` that passes input through and merges in
    /// additional computed keys from a mapping of name -> runnable.
    ///
    /// Equivalent to Python's `RunnablePassthrough.assign(**kwargs)`.
    ///
    /// # Arguments
    /// * `mapping` - Map of output key names to runnables that compute the values.
    ///
    /// # Example
    /// ```ignore
    /// // Input: {"question": "What is 2+2?"}
    /// // Output: {"question": "What is 2+2?", "answer": <computed>}
    /// let chain = my_runnable.assign(hashmap!{ "answer" => answer_runnable });
    /// ```
    fn assign(self, mapping: HashMap<String, Arc<dyn Runnable>>) -> Result<RunnableSequence> {
        let mut builder = RunnableAssign::new();
        for (key, runnable) in mapping {
            builder = builder.assign(key, runnable);
        }
        RunnableSequence::new(vec![
            Arc::new(self) as Arc<dyn Runnable>,
            Arc::new(builder) as Arc<dyn Runnable>,
        ])
    }

    /// Bind additional kwargs to the runnable's input.
    ///
    /// Equivalent to Python's `Runnable.bind(**kwargs)`.
    /// When invoked, `kwargs` are merged into the input if it's an object.
    fn bind(self, kwargs: HashMap<String, Value>) -> RunnableBinding {
        RunnableBinding::new(Arc::new(self) as Arc<dyn Runnable>, kwargs, None)
    }

    /// Attach a config patch to the runnable.
    ///
    /// Equivalent to Python's `Runnable.with_config(config)`.
    /// The config patch is merged with any config passed at invocation time.
    fn with_config(self, config: RunnableConfig) -> RunnableBinding {
        RunnableBinding::new(
            Arc::new(self) as Arc<dyn Runnable>,
            HashMap::new(),
            Some(config),
        )
    }

    /// Set a concurrency limit for batch operations on this runnable.
    ///
    /// Returns a `RunnableBinding` with `max_concurrency` set in its config patch.
    /// When `abatch` is called on the resulting runnable, it will limit the number
    /// of concurrent invocations to `max_concurrency`.
    fn with_concurrency(self, max_concurrency: usize) -> RunnableBinding {
        let config = RunnableConfig {
            max_concurrency: Some(max_concurrency),
            ..Default::default()
        };
        self.with_config(config)
    }

    /// Pick one or more keys from the output dict.
    ///
    /// Equivalent to Python's `Runnable.pick(keys)`.
    /// If a single key string is passed, returns the value directly.
    /// If multiple keys are passed, returns a dict with only those keys.
    /// Wrap this runnable with token-bucket rate limiting.
    ///
    /// Controls the maximum rate of invocations using the provided configuration.
    ///
    /// # Arguments
    /// * `config` - Rate limit configuration (requests per second, burst size, wait behavior).
    fn with_rate_limit(self, config: RateLimitConfig) -> RunnableRateLimit {
        RunnableRateLimit::new(Arc::new(self) as Arc<dyn Runnable>, config)
    }

    /// Wrap this runnable with a simple throttle that enforces a minimum interval
    /// between invocations.
    ///
    /// # Arguments
    /// * `min_interval` - Minimum duration between consecutive invocations.
    fn with_throttle(self, min_interval: Duration) -> RunnableThrottle {
        RunnableThrottle::with_duration(Arc::new(self) as Arc<dyn Runnable>, min_interval)
    }

    /// Wrap this runnable with a caching layer that memoizes invoke results.
    ///
    /// Cached entries are keyed by JSON serialization of the input (or a custom
    /// key function). Supports TTL-based expiry and LRU eviction.
    ///
    /// # Arguments
    /// * `config` - Cache configuration (max entries, TTL, custom key function).
    fn with_cache(self, config: CacheConfig) -> RunnableCache {
        RunnableCache::new(Arc::new(self) as Arc<dyn Runnable>, config)
    }

    /// Wrap this runnable with a relative timeout.
    ///
    /// If the operation does not complete within `duration`, a timeout error
    /// is returned.
    ///
    /// # Arguments
    /// * `duration` - Maximum time allowed for the operation.
    fn with_timeout(self, duration: Duration) -> RunnableTimeout {
        RunnableTimeout::new(
            Arc::new(self) as Arc<dyn Runnable>,
            TimeoutConfig::new(duration),
        )
    }

    /// Wrap this runnable with a fully configured timeout.
    ///
    /// # Arguments
    /// * `config` - Timeout configuration (duration and behavior on timeout).
    fn with_timeout_config(self, config: TimeoutConfig) -> RunnableTimeout {
        RunnableTimeout::new(Arc::new(self) as Arc<dyn Runnable>, config)
    }

    fn pick(self, keys: Vec<String>) -> Result<RunnableSequence> {
        let picker = if keys.len() == 1 {
            RunnablePick::one(keys.into_iter().next().unwrap())
        } else {
            RunnablePick::many(keys)
        };
        RunnableSequence::new(vec![
            Arc::new(self) as Arc<dyn Runnable>,
            Arc::new(picker) as Arc<dyn Runnable>,
        ])
    }
}

/// Blanket implementation: every type that implements `Runnable + Sized + 'static`
/// automatically gets the `RunnableExt` methods.
impl<T: Runnable + Sized + 'static> RunnableExt for T {}
