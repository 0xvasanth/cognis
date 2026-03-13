//! Batch and parallel embeddings processing utilities.
//!
//! Provides tools for processing large sets of texts through embedding models
//! with configurable concurrency, retry logic, rate limiting, and multi-provider
//! distribution.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::sync::{Mutex, Semaphore};

use cognis_core::embeddings::Embeddings;
use cognis_core::error::{Result, CognisError};

/// Configuration for batch embedding processing.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of texts per batch sent to the embedding provider.
    pub batch_size: usize,
    /// Maximum number of batches processed concurrently.
    pub max_concurrency: usize,
    /// Whether to retry failed batches.
    pub retry_failed: bool,
    /// Maximum number of retry attempts per batch.
    pub max_retries: usize,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 100,
            max_concurrency: 4,
            retry_failed: true,
            max_retries: 3,
        }
    }
}

impl BatchConfig {
    /// Create a new `BatchConfig` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the maximum concurrency level.
    pub fn max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    /// Set whether to retry failed batches.
    pub fn retry_failed(mut self, retry_failed: bool) -> Self {
        self.retry_failed = retry_failed;
        self
    }

    /// Set the maximum number of retries per batch.
    pub fn max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }
}

/// Result of a batch embedding operation.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// The computed embeddings, indexed to match input ordering.
    /// Failed indices will have empty vectors.
    pub embeddings: Vec<Vec<f32>>,
    /// Indices of texts that failed to embed after all retries.
    pub failed_indices: Vec<usize>,
    /// Total number of texts successfully processed.
    pub total_processed: usize,
    /// Wall-clock time for the entire batch operation.
    pub total_time: Duration,
    /// Number of batches that were processed.
    pub batches_processed: usize,
}

/// Processes embeddings in configurable batches with concurrency control.
pub struct BatchEmbedder {
    embeddings: Arc<dyn Embeddings>,
    config: BatchConfig,
}

impl BatchEmbedder {
    /// Create a new `BatchEmbedder` wrapping an embedding provider.
    pub fn new(embeddings: Arc<dyn Embeddings>, config: BatchConfig) -> Self {
        Self { embeddings, config }
    }

    /// Embed a collection of texts in batches with concurrency limiting.
    ///
    /// Splits `texts` into chunks of `config.batch_size`, then processes up to
    /// `config.max_concurrency` chunks in parallel using a semaphore. Failed
    /// batches are retried up to `config.max_retries` times when `config.retry_failed`
    /// is enabled.
    pub async fn embed_batch(&self, texts: Vec<String>) -> Result<BatchResult> {
        self.embed_with_progress_internal(texts, None).await
    }

    /// Embed texts with a progress callback invoked after each batch completes.
    ///
    /// The callback receives `(completed_batches, total_batches)`.
    pub async fn embed_with_progress(
        &self,
        texts: Vec<String>,
        callback: impl Fn(usize, usize) + Send + Sync + 'static,
    ) -> Result<BatchResult> {
        self.embed_with_progress_internal(texts, Some(Box::new(callback)))
            .await
    }

    async fn embed_with_progress_internal(
        &self,
        texts: Vec<String>,
        callback: Option<Box<dyn Fn(usize, usize) + Send + Sync>>,
    ) -> Result<BatchResult> {
        let start = Instant::now();

        if texts.is_empty() {
            return Ok(BatchResult {
                embeddings: vec![],
                failed_indices: vec![],
                total_processed: 0,
                total_time: start.elapsed(),
                batches_processed: 0,
            });
        }

        let total_texts = texts.len();
        let batch_size = self.config.batch_size.max(1);
        let batches: Vec<(usize, Vec<String>)> = texts
            .chunks(batch_size)
            .enumerate()
            .map(|(i, chunk)| (i, chunk.to_vec()))
            .collect();
        let total_batches = batches.len();

        let semaphore = Arc::new(Semaphore::new(self.config.max_concurrency.max(1)));
        #[allow(clippy::type_complexity)]
        let results: Arc<Mutex<Vec<(usize, std::result::Result<Vec<Vec<f32>>, String>)>>> =
            Arc::new(Mutex::new(Vec::with_capacity(total_batches)));
        let completed = Arc::new(AtomicUsize::new(0));
        let callback = Arc::new(callback);

        let mut handles = Vec::with_capacity(total_batches);

        for (batch_idx, batch_texts) in batches {
            let sem = semaphore.clone();
            let embedder = self.embeddings.clone();
            let results = results.clone();
            let completed = completed.clone();
            let callback = callback.clone();
            let retry_failed = self.config.retry_failed;
            let max_retries = self.config.max_retries;

            let handle = tokio::spawn(async move {
                let _permit = sem.acquire().await.expect("semaphore closed");

                let mut last_err = String::new();
                let attempts = if retry_failed { max_retries + 1 } else { 1 };

                for attempt in 0..attempts {
                    match embedder.embed_documents(batch_texts.clone()).await {
                        Ok(embs) => {
                            let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
                            if let Some(ref cb) = *callback {
                                cb(done, total_batches);
                            }
                            results.lock().await.push((batch_idx, Ok(embs)));
                            return;
                        }
                        Err(e) => {
                            last_err = e.to_string();
                            if attempt < attempts - 1 {
                                // Brief backoff before retry
                                tokio::time::sleep(Duration::from_millis(
                                    50 * (attempt as u64 + 1),
                                ))
                                .await;
                            }
                        }
                    }
                }

                let done = completed.fetch_add(1, Ordering::SeqCst) + 1;
                if let Some(ref cb) = *callback {
                    cb(done, total_batches);
                }
                results.lock().await.push((batch_idx, Err(last_err)));
            });

            handles.push(handle);
        }

        for handle in handles {
            handle
                .await
                .map_err(|e| CognisError::Other(format!("Task join error: {e}")))?;
        }

        // Assemble results in order
        let mut all_results = results.lock().await;
        all_results.sort_by_key(|(idx, _)| *idx);

        let mut embeddings = vec![vec![]; total_texts];
        let mut failed_indices = Vec::new();
        let mut total_processed = 0;

        for (batch_idx, result) in all_results.iter() {
            let offset = batch_idx * batch_size;
            match result {
                Ok(embs) => {
                    for (j, emb) in embs.iter().enumerate() {
                        let global_idx = offset + j;
                        if global_idx < total_texts {
                            embeddings[global_idx] = emb.clone();
                            total_processed += 1;
                        }
                    }
                }
                Err(_) => {
                    let batch_end = (offset + batch_size).min(total_texts);
                    for idx in offset..batch_end {
                        failed_indices.push(idx);
                    }
                }
            }
        }

        failed_indices.sort();

        Ok(BatchResult {
            embeddings,
            failed_indices,
            total_processed,
            total_time: start.elapsed(),
            batches_processed: total_batches,
        })
    }
}

/// Distributes embedding requests across multiple providers for throughput.
pub struct ParallelEmbedder {
    providers: Vec<Arc<dyn Embeddings>>,
}

impl ParallelEmbedder {
    /// Create a new `ParallelEmbedder` with multiple embedding providers.
    ///
    /// # Panics
    /// Panics if `providers` is empty.
    pub fn new(providers: Vec<Arc<dyn Embeddings>>) -> Self {
        assert!(!providers.is_empty(), "providers must not be empty");
        Self { providers }
    }

    /// Distribute texts across providers in round-robin fashion and collect results.
    pub async fn embed_round_robin(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        let n = self.providers.len();
        // Group texts by provider index, preserving original indices
        let mut groups: Vec<Vec<(usize, String)>> = vec![vec![]; n];
        for (i, text) in texts.iter().enumerate() {
            groups[i % n].push((i, text.clone()));
        }

        let mut handles = Vec::with_capacity(n);
        for (provider_idx, group) in groups.into_iter().enumerate() {
            if group.is_empty() {
                continue;
            }
            let provider = self.providers[provider_idx].clone();
            let handle = tokio::spawn(async move {
                let indices: Vec<usize> = group.iter().map(|(i, _)| *i).collect();
                let batch: Vec<String> = group.into_iter().map(|(_, t)| t).collect();
                let embs = provider.embed_documents(batch).await?;
                Ok::<Vec<(usize, Vec<f32>)>, CognisError>(
                    indices.into_iter().zip(embs).collect(),
                )
            });
            handles.push(handle);
        }

        let mut results = vec![vec![]; texts.len()];
        for handle in handles {
            let pairs = handle
                .await
                .map_err(|e| CognisError::Other(format!("Task join error: {e}")))??;
            for (idx, emb) in pairs {
                results[idx] = emb;
            }
        }

        Ok(results)
    }

    /// Race all providers to embed a single text, returning the first successful result.
    pub async fn embed_fastest(&self, text: &str) -> Result<Vec<f32>> {
        let (tx, mut rx) = tokio::sync::mpsc::channel::<Result<Vec<f32>>>(self.providers.len());

        for provider in &self.providers {
            let provider = provider.clone();
            let text = text.to_string();
            let tx = tx.clone();
            tokio::spawn(async move {
                let result = provider.embed_query(&text).await;
                let _ = tx.send(result).await;
            });
        }

        drop(tx); // Drop our sender so the channel closes when all tasks finish

        // Return the first successful result
        let mut last_err = None;
        while let Some(result) = rx.recv().await {
            match result {
                Ok(emb) => return Ok(emb),
                Err(e) => last_err = Some(e),
            }
        }

        Err(last_err.unwrap_or_else(|| CognisError::Other("All providers failed".to_string())))
    }
}

/// Token-bucket rate limiter for embedding requests.
///
/// Wraps an inner embedding provider and throttles requests to stay within
/// a configured token rate.
pub struct EmbeddingRateLimiter {
    inner: Arc<dyn Embeddings>,
    /// Tokens replenished per second.
    pub tokens_per_second: f64,
    /// Maximum burst capacity.
    pub max_burst: usize,
    state: Mutex<RateLimiterState>,
}

struct RateLimiterState {
    available_tokens: f64,
    last_refill: Instant,
}

impl EmbeddingRateLimiter {
    /// Create a new rate limiter wrapping an embedding provider.
    ///
    /// - `tokens_per_second` — steady-state rate of token replenishment.
    /// - `max_burst` — maximum tokens that can accumulate.
    pub fn new(inner: Arc<dyn Embeddings>, tokens_per_second: f64, max_burst: usize) -> Self {
        Self {
            inner,
            tokens_per_second,
            max_burst,
            state: Mutex::new(RateLimiterState {
                available_tokens: max_burst as f64,
                last_refill: Instant::now(),
            }),
        }
    }

    /// Wait until `count` tokens are available, then consume them.
    pub async fn acquire(&self, count: usize) {
        loop {
            {
                let mut state = self.state.lock().await;
                let now = Instant::now();
                let elapsed = now.duration_since(state.last_refill).as_secs_f64();
                state.available_tokens = (state.available_tokens
                    + elapsed * self.tokens_per_second)
                    .min(self.max_burst as f64);
                state.last_refill = now;

                if state.available_tokens >= count as f64 {
                    state.available_tokens -= count as f64;
                    return;
                }
            }
            // Wait a bit before checking again
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

#[async_trait]
impl Embeddings for EmbeddingRateLimiter {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        self.acquire(texts.len()).await;
        self.inner.embed_documents(texts).await
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.acquire(1).await;
        self.inner.embed_query(text).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::embeddings_fake::DeterministicFakeEmbedding;
    fn fake_embedder() -> Arc<dyn Embeddings> {
        Arc::new(DeterministicFakeEmbedding::new(8))
    }

    // --- BatchConfig tests ---

    #[test]
    fn test_batch_config_defaults() {
        let config = BatchConfig::default();
        assert_eq!(config.batch_size, 100);
        assert_eq!(config.max_concurrency, 4);
        assert!(config.retry_failed);
        assert_eq!(config.max_retries, 3);
    }

    #[test]
    fn test_batch_config_builder() {
        let config = BatchConfig::new()
            .batch_size(50)
            .max_concurrency(8)
            .retry_failed(false)
            .max_retries(5);
        assert_eq!(config.batch_size, 50);
        assert_eq!(config.max_concurrency, 8);
        assert!(!config.retry_failed);
        assert_eq!(config.max_retries, 5);
    }

    // --- BatchEmbedder tests ---

    #[tokio::test]
    async fn test_batch_embed_empty() {
        let embedder = BatchEmbedder::new(fake_embedder(), BatchConfig::new());
        let result = embedder.embed_batch(vec![]).await.unwrap();
        assert_eq!(result.embeddings.len(), 0);
        assert_eq!(result.total_processed, 0);
        assert_eq!(result.batches_processed, 0);
        assert!(result.failed_indices.is_empty());
    }

    #[tokio::test]
    async fn test_batch_embed_single_text() {
        let embedder = BatchEmbedder::new(fake_embedder(), BatchConfig::new());
        let result = embedder
            .embed_batch(vec!["hello".to_string()])
            .await
            .unwrap();
        assert_eq!(result.embeddings.len(), 1);
        assert_eq!(result.embeddings[0].len(), 8);
        assert_eq!(result.total_processed, 1);
        assert_eq!(result.batches_processed, 1);
    }

    #[tokio::test]
    async fn test_batch_embed_multiple_batches() {
        let config = BatchConfig::new().batch_size(3).max_concurrency(2);
        let embedder = BatchEmbedder::new(fake_embedder(), config);
        let texts: Vec<String> = (0..10).map(|i| format!("text_{i}")).collect();
        let result = embedder.embed_batch(texts).await.unwrap();

        assert_eq!(result.embeddings.len(), 10);
        assert_eq!(result.total_processed, 10);
        assert_eq!(result.batches_processed, 4); // ceil(10/3) = 4
        assert!(result.failed_indices.is_empty());

        // Each embedding should have length 8
        for emb in &result.embeddings {
            assert_eq!(emb.len(), 8);
        }
    }

    #[tokio::test]
    async fn test_batch_embed_deterministic_results() {
        let config = BatchConfig::new().batch_size(2);
        let embedder = BatchEmbedder::new(fake_embedder(), config);
        let texts = vec!["alpha".to_string(), "beta".to_string()];

        let result1 = embedder.embed_batch(texts.clone()).await.unwrap();
        let result2 = embedder.embed_batch(texts).await.unwrap();

        assert_eq!(result1.embeddings, result2.embeddings);
    }

    #[tokio::test]
    async fn test_batch_embed_preserves_order() {
        let config = BatchConfig::new().batch_size(2).max_concurrency(4);
        let embedder = BatchEmbedder::new(fake_embedder(), config);
        let texts: Vec<String> = (0..8).map(|i| format!("doc_{i}")).collect();

        // Embed all at once for reference
        let reference = fake_embedder()
            .embed_documents(texts.clone())
            .await
            .unwrap();

        let result = embedder.embed_batch(texts).await.unwrap();
        assert_eq!(result.embeddings, reference);
    }

    #[tokio::test]
    async fn test_batch_embed_with_progress() {
        let config = BatchConfig::new().batch_size(3).max_concurrency(1);
        let embedder = BatchEmbedder::new(fake_embedder(), config);
        let texts: Vec<String> = (0..9).map(|i| format!("item_{i}")).collect();

        let progress_count = Arc::new(AtomicUsize::new(0));
        let pc = progress_count.clone();

        let result = embedder
            .embed_with_progress(texts, move |completed, total| {
                assert_eq!(total, 3);
                assert!(completed <= total);
                pc.fetch_add(1, Ordering::SeqCst);
            })
            .await
            .unwrap();

        assert_eq!(result.total_processed, 9);
        assert_eq!(progress_count.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn test_batch_embed_concurrency_limit() {
        // Verify that concurrency is actually limited
        let active = Arc::new(AtomicUsize::new(0));
        let max_active = Arc::new(AtomicUsize::new(0));

        struct TrackingEmbedder {
            inner: DeterministicFakeEmbedding,
            active: Arc<AtomicUsize>,
            max_active: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl Embeddings for TrackingEmbedder {
            async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
                let current = self.active.fetch_add(1, Ordering::SeqCst) + 1;
                self.max_active.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(50)).await;
                let result = self.inner.embed_documents(texts).await;
                self.active.fetch_sub(1, Ordering::SeqCst);
                result
            }

            async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
                self.inner.embed_query(text).await
            }
        }

        let embedder_impl = TrackingEmbedder {
            inner: DeterministicFakeEmbedding::new(4),
            active: active.clone(),
            max_active: max_active.clone(),
        };

        let config = BatchConfig::new().batch_size(1).max_concurrency(2);
        let embedder = BatchEmbedder::new(Arc::new(embedder_impl), config);
        let texts: Vec<String> = (0..6).map(|i| format!("t{i}")).collect();

        let result = embedder.embed_batch(texts).await.unwrap();
        assert_eq!(result.total_processed, 6);
        assert!(max_active.load(Ordering::SeqCst) <= 2);
    }

    #[tokio::test]
    async fn test_batch_result_timing() {
        let config = BatchConfig::new().batch_size(5);
        let embedder = BatchEmbedder::new(fake_embedder(), config);
        let texts: Vec<String> = (0..5).map(|i| format!("text_{i}")).collect();
        let result = embedder.embed_batch(texts).await.unwrap();
        // Should complete very quickly with fake embedder
        assert!(result.total_time < Duration::from_secs(5));
    }

    // --- ParallelEmbedder tests ---

    #[tokio::test]
    async fn test_parallel_round_robin_empty() {
        let pe = ParallelEmbedder::new(vec![fake_embedder()]);
        let result = pe.embed_round_robin(vec![]).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_parallel_round_robin_single_provider() {
        let pe = ParallelEmbedder::new(vec![fake_embedder()]);
        let texts: Vec<String> = (0..5).map(|i| format!("doc_{i}")).collect();
        let result = pe.embed_round_robin(texts.clone()).await.unwrap();

        let reference = fake_embedder().embed_documents(texts).await.unwrap();
        assert_eq!(result, reference);
    }

    #[tokio::test]
    async fn test_parallel_round_robin_multiple_providers() {
        let pe = ParallelEmbedder::new(vec![fake_embedder(), fake_embedder()]);
        let texts: Vec<String> = (0..6).map(|i| format!("x_{i}")).collect();
        let result = pe.embed_round_robin(texts.clone()).await.unwrap();

        assert_eq!(result.len(), 6);
        // Since all providers are the same fake embedder, results should match
        let reference = fake_embedder().embed_documents(texts).await.unwrap();
        assert_eq!(result, reference);
    }

    #[tokio::test]
    async fn test_parallel_embed_fastest() {
        let pe = ParallelEmbedder::new(vec![fake_embedder(), fake_embedder()]);
        let result = pe.embed_fastest("hello").await.unwrap();
        assert_eq!(result.len(), 8);

        let reference = fake_embedder().embed_query("hello").await.unwrap();
        assert_eq!(result, reference);
    }

    #[test]
    #[should_panic(expected = "providers must not be empty")]
    fn test_parallel_embedder_empty_providers() {
        ParallelEmbedder::new(vec![]);
    }

    // --- EmbeddingRateLimiter tests ---

    #[tokio::test]
    async fn test_rate_limiter_basic() {
        let limiter = EmbeddingRateLimiter::new(fake_embedder(), 1000.0, 100);
        let result = limiter.embed_query("test").await.unwrap();
        assert_eq!(result.len(), 8);
    }

    #[tokio::test]
    async fn test_rate_limiter_documents() {
        let limiter = EmbeddingRateLimiter::new(fake_embedder(), 1000.0, 100);
        let texts = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let result = limiter.embed_documents(texts).await.unwrap();
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn test_rate_limiter_acquire_consumes_tokens() {
        let limiter = EmbeddingRateLimiter::new(fake_embedder(), 10.0, 5);
        // Burst of 5 should succeed immediately
        limiter.acquire(5).await;
        // Next acquire should wait for refill
        let start = Instant::now();
        limiter.acquire(1).await;
        // Should have waited at least ~100ms (1 token / 10 tokens_per_sec)
        assert!(start.elapsed() >= Duration::from_millis(50));
    }

    #[tokio::test]
    async fn test_rate_limiter_deterministic_output() {
        let limiter = EmbeddingRateLimiter::new(fake_embedder(), 1000.0, 100);
        let r1 = limiter.embed_query("same_text").await.unwrap();
        let r2 = limiter.embed_query("same_text").await.unwrap();
        assert_eq!(r1, r2);
    }

    // --- Failing embedder for retry tests ---

    #[tokio::test]
    async fn test_batch_embed_with_failing_provider() {
        struct FailingEmbedder;

        #[async_trait]
        impl Embeddings for FailingEmbedder {
            async fn embed_documents(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
                Err(CognisError::Other("simulated failure".to_string()))
            }
            async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
                Err(CognisError::Other("simulated failure".to_string()))
            }
        }

        let config = BatchConfig::new()
            .batch_size(2)
            .max_retries(1)
            .retry_failed(true);
        let embedder = BatchEmbedder::new(Arc::new(FailingEmbedder), config);
        let texts = vec!["a".to_string(), "b".to_string(), "c".to_string()];

        let result = embedder.embed_batch(texts).await.unwrap();
        assert_eq!(result.total_processed, 0);
        assert_eq!(result.failed_indices, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn test_batch_embed_retry_disabled() {
        let call_count = Arc::new(AtomicUsize::new(0));

        struct CountingFailer {
            count: Arc<AtomicUsize>,
        }

        #[async_trait]
        impl Embeddings for CountingFailer {
            async fn embed_documents(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
                self.count.fetch_add(1, Ordering::SeqCst);
                Err(CognisError::Other("fail".to_string()))
            }
            async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
                Err(CognisError::Other("fail".to_string()))
            }
        }

        let config = BatchConfig::new().batch_size(2).retry_failed(false);
        let embedder = BatchEmbedder::new(
            Arc::new(CountingFailer {
                count: call_count.clone(),
            }),
            config,
        );

        let _result = embedder
            .embed_batch(vec!["x".to_string(), "y".to_string()])
            .await
            .unwrap();

        // With retry disabled, should only attempt once
        assert_eq!(call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_parallel_embed_fastest_all_fail() {
        struct FailingEmbedder;

        #[async_trait]
        impl Embeddings for FailingEmbedder {
            async fn embed_documents(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
                Err(CognisError::Other("fail".to_string()))
            }
            async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
                Err(CognisError::Other("fail".to_string()))
            }
        }

        let pe = ParallelEmbedder::new(vec![
            Arc::new(FailingEmbedder) as Arc<dyn Embeddings>,
            Arc::new(FailingEmbedder),
        ]);
        let result = pe.embed_fastest("test").await;
        assert!(result.is_err());
    }
}
