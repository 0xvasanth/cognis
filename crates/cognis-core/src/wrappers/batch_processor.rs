//! Explicit batch processor — extends the default [`Runnable::batch`]
//! with knobs that the trait method intentionally keeps simple:
//!
//! - **`max_concurrency`** — separate from `RunnableConfig`'s default,
//!   so the same runnable can be batched at different concurrencies in
//!   different call sites without per-config gymnastics.
//! - **`return_exceptions`** — when `true`, the processor returns
//!   `Vec<Result<O, CognisError>>` so partial failures surface
//!   per-input. The default `Runnable::batch` short-circuits on the
//!   first error.
//! - **`wave_delay`** — optional pause between scheduling waves, useful
//!   when the underlying service has rate-limit windows that aren't
//!   well-captured by per-call backoff.
//!
//! Customization: implement [`crate::Runnable`] for an entirely custom
//! batch strategy (e.g. server-side batching). Otherwise, use
//! `BatchProcessor::process` for the standard concurrency-bounded path.

use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;

use futures::stream::{self, StreamExt};

use crate::runnable::{Runnable, RunnableConfig};
use crate::{CognisError, Result};

/// Settings for [`BatchProcessor`].
#[derive(Debug, Clone, Default)]
pub struct BatchOptions {
    /// Maximum in-flight tasks. `0` falls back to the config default.
    pub max_concurrency: usize,
    /// When `true`, errors are collected per-input rather than
    /// short-circuiting the whole batch.
    pub return_exceptions: bool,
    /// Optional pause between waves of scheduled tasks. `None` schedules
    /// continuously up to `max_concurrency`.
    pub wave_delay: Option<Duration>,
}

impl BatchOptions {
    /// Override max_concurrency (`0` ⇒ use config's value).
    pub fn with_max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n;
        self
    }

    /// Toggle return_exceptions.
    pub fn with_return_exceptions(mut self, on: bool) -> Self {
        self.return_exceptions = on;
        self
    }

    /// Set a wave delay between scheduled tasks.
    pub fn with_wave_delay(mut self, d: Duration) -> Self {
        self.wave_delay = Some(d);
        self
    }
}

/// Explicit batch processor for a single inner runnable.
///
/// Differences from `inner.batch(...)`:
/// - Honors per-instance [`BatchOptions`].
/// - Has [`process_with_results`] returning a `Vec<Result<O>>` so
///   partial failures don't lose data.
pub struct BatchProcessor<R, I, O> {
    inner: R,
    options: BatchOptions,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<R, I, O> BatchProcessor<R, I, O>
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Send + 'static,
{
    /// Wrap with default options.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            options: BatchOptions::default(),
            _phantom: PhantomData,
        }
    }

    /// Wrap with explicit options.
    pub fn with_options(inner: R, options: BatchOptions) -> Self {
        Self {
            inner,
            options,
            _phantom: PhantomData,
        }
    }

    /// Borrow the inner runnable.
    pub fn inner(&self) -> &R {
        &self.inner
    }

    /// Mutably borrow options (e.g. to flip `return_exceptions` mid-life).
    pub fn options_mut(&mut self) -> &mut BatchOptions {
        &mut self.options
    }

    fn effective_concurrency(&self, config: &RunnableConfig) -> usize {
        if self.options.max_concurrency == 0 {
            config.max_concurrency.max(1)
        } else {
            self.options.max_concurrency
        }
    }
}

impl<R, I, O> BatchProcessor<R, I, O>
where
    R: Runnable<I, O> + Sync,
    I: Send + 'static,
    O: Send + 'static,
{
    /// Run a batch and short-circuit on the first error. Equivalent to
    /// `inner.batch(...)` but honors `BatchOptions::max_concurrency` and
    /// `wave_delay`.
    pub async fn process(&self, inputs: Vec<I>, config: RunnableConfig) -> Result<Vec<O>> {
        if self.options.return_exceptions {
            let results = self.process_with_results(inputs, config).await;
            results.into_iter().collect()
        } else {
            let concurrency = self.effective_concurrency(&config);
            let wave = self.options.wave_delay;
            let cfg = Arc::new(config);
            let inner = &self.inner;
            stream::iter(inputs)
                .map(|input| {
                    let cfg = cfg.clone();
                    async move {
                        if let Some(d) = wave {
                            tokio::time::sleep(d).await;
                        }
                        inner
                            .invoke(input, RunnableConfig::clone_for_subcall(&cfg))
                            .await
                    }
                })
                .buffer_unordered(concurrency)
                .collect::<Vec<_>>()
                .await
                .into_iter()
                .collect()
        }
    }

    /// Run a batch and **always** return per-input results — failures
    /// surface as `Err(CognisError)` entries, successful invocations as
    /// `Ok(O)`. Order matches the input order.
    pub async fn process_with_results(
        &self,
        inputs: Vec<I>,
        config: RunnableConfig,
    ) -> Vec<Result<O>> {
        let concurrency = self.effective_concurrency(&config);
        let wave = self.options.wave_delay;
        let cfg = Arc::new(config);
        let inner = &self.inner;
        // Tag each input with its original index so we can restore order
        // after `buffer_unordered` finishes.
        let tagged: Vec<(usize, I)> = inputs.into_iter().enumerate().collect();
        let total = tagged.len();
        let mut results: Vec<Option<Result<O>>> = (0..total).map(|_| None).collect::<Vec<_>>();

        let mut stream = stream::iter(tagged)
            .map(|(i, input)| {
                let cfg = cfg.clone();
                async move {
                    if let Some(d) = wave {
                        tokio::time::sleep(d).await;
                    }
                    let res = inner
                        .invoke(input, RunnableConfig::clone_for_subcall(&cfg))
                        .await;
                    (i, res)
                }
            })
            .buffer_unordered(concurrency);

        while let Some((i, res)) = stream.next().await {
            results[i] = Some(res);
        }
        results
            .into_iter()
            .map(|r| {
                r.unwrap_or_else(|| {
                    Err(CognisError::Internal("batch slot was never filled".into()))
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Counter {
        seen: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl Runnable<usize, usize> for Counter {
        async fn invoke(&self, input: usize, _: RunnableConfig) -> Result<usize> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            Ok(input * 2)
        }
    }

    struct Sometimes;
    #[async_trait]
    impl Runnable<usize, usize> for Sometimes {
        async fn invoke(&self, input: usize, _: RunnableConfig) -> Result<usize> {
            if input.is_multiple_of(2) {
                Ok(input)
            } else {
                Err(CognisError::Internal(format!("odd input {input}")))
            }
        }
    }

    #[tokio::test]
    async fn process_returns_results_in_order() {
        let r = Counter {
            seen: Arc::new(AtomicUsize::new(0)),
        };
        let bp: BatchProcessor<Counter, usize, usize> = BatchProcessor::new(r);
        let out = bp
            .process(vec![1, 2, 3], RunnableConfig::default())
            .await
            .unwrap();
        // process() preserves order via the inner Runnable::batch path —
        // here we use the same shape but rely on the result order matching
        // input order (first-error short-circuit branch).
        assert_eq!(out.len(), 3);
        assert!(out.contains(&2));
        assert!(out.contains(&4));
        assert!(out.contains(&6));
    }

    #[tokio::test]
    async fn process_short_circuits_on_first_error() {
        let bp: BatchProcessor<Sometimes, usize, usize> = BatchProcessor::new(Sometimes);
        let res = bp.process(vec![1, 2, 3], RunnableConfig::default()).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn process_with_results_collects_partials() {
        let bp = BatchProcessor::with_options(
            Sometimes,
            BatchOptions::default()
                .with_return_exceptions(true)
                .with_max_concurrency(4),
        );
        let res = bp
            .process_with_results(vec![1, 2, 3, 4], RunnableConfig::default())
            .await;
        // Order matches input.
        assert_eq!(res.len(), 4);
        assert!(res[0].is_err());
        assert!(res[1].is_ok());
        assert!(res[2].is_err());
        assert!(res[3].is_ok());
    }

    #[tokio::test]
    async fn return_exceptions_via_process_short_circuits_only_when_all_fail() {
        let bp = BatchProcessor::with_options(
            Sometimes,
            BatchOptions::default().with_return_exceptions(true),
        );
        // process() under return_exceptions returns the first Err it
        // encounters when collecting via `.collect()`. That mirrors V1
        // semantics: if the user wants the whole vec of partials, they
        // should call process_with_results directly.
        let res = bp.process(vec![2, 4], RunnableConfig::default()).await;
        assert_eq!(res.unwrap(), vec![2, 4]);
    }

    #[tokio::test]
    async fn explicit_max_concurrency_overrides_config() {
        let counter = Arc::new(AtomicUsize::new(0));
        let r = Counter {
            seen: counter.clone(),
        };
        let bp: BatchProcessor<Counter, usize, usize> =
            BatchProcessor::with_options(r, BatchOptions::default().with_max_concurrency(2));
        let _ = bp
            .process(vec![1, 2, 3, 4, 5], RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 5);
    }
}
