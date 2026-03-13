use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;
use serde_json::{json, Value};

use crate::error::{Result, CognisError};

use super::base::Runnable;
use super::config::RunnableConfig;

// ---------------------------------------------------------------------------
// BatchConfig
// ---------------------------------------------------------------------------

/// Configuration for batch processing of runnables.
///
/// Uses a builder pattern for ergonomic construction.
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Maximum number of items processed concurrently within a chunk.
    pub max_concurrency: usize,
    /// Number of items per processing chunk.
    pub chunk_size: usize,
    /// If `true`, abort the entire batch on the first failure.
    pub fail_fast: bool,
    /// Optional per-item timeout in milliseconds.
    pub timeout_per_item_ms: Option<u64>,
    /// If `true`, failed items will be retried once.
    pub retry_failed: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 10,
            chunk_size: 100,
            fail_fast: false,
            timeout_per_item_ms: None,
            retry_failed: false,
        }
    }
}

impl BatchConfig {
    /// Create a new `BatchConfig` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum concurrency.
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = max_concurrency;
        self
    }

    /// Set the chunk size.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }

    /// Set the fail-fast flag.
    pub fn with_fail_fast(mut self, fail_fast: bool) -> Self {
        self.fail_fast = fail_fast;
        self
    }

    /// Set the per-item timeout in milliseconds.
    pub fn with_timeout_per_item_ms(mut self, timeout_ms: u64) -> Self {
        self.timeout_per_item_ms = Some(timeout_ms);
        self
    }

    /// Set whether to retry failed items.
    pub fn with_retry_failed(mut self, retry_failed: bool) -> Self {
        self.retry_failed = retry_failed;
        self
    }
}

// ---------------------------------------------------------------------------
// BatchItemResult
// ---------------------------------------------------------------------------

/// The outcome of processing a single item in a batch.
#[derive(Debug, Clone)]
pub enum BatchItemResult {
    /// The item was processed successfully.
    Success {
        /// Original index in the input vector.
        index: usize,
        /// The resulting value.
        value: Value,
    },
    /// The item failed to process.
    Failure {
        /// Original index in the input vector.
        index: usize,
        /// A description of the error.
        error: String,
    },
}

impl BatchItemResult {
    /// Returns `true` if this is a successful result.
    pub fn is_success(&self) -> bool {
        matches!(self, BatchItemResult::Success { .. })
    }

    /// Returns `true` if this is a failure result.
    pub fn is_failure(&self) -> bool {
        matches!(self, BatchItemResult::Failure { .. })
    }

    /// Returns the index of this item in the original input.
    pub fn index(&self) -> usize {
        match self {
            BatchItemResult::Success { index, .. } => *index,
            BatchItemResult::Failure { index, .. } => *index,
        }
    }

    /// Consume and return the value if successful, or `None` on failure.
    pub fn into_value(self) -> Option<Value> {
        match self {
            BatchItemResult::Success { value, .. } => Some(value),
            BatchItemResult::Failure { .. } => None,
        }
    }
}

// ---------------------------------------------------------------------------
// BatchResult
// ---------------------------------------------------------------------------

/// Aggregated result of a batch processing run.
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Individual item results, ordered by original index.
    pub results: Vec<BatchItemResult>,
    /// Total number of items submitted.
    pub total: usize,
    /// Number of successfully processed items.
    pub succeeded: usize,
    /// Number of failed items.
    pub failed: usize,
    /// Wall-clock duration of the batch in milliseconds.
    pub duration_ms: u64,
}

impl BatchResult {
    /// Returns the fraction of items that succeeded (0.0 to 1.0).
    ///
    /// Returns 0.0 if total is zero.
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.succeeded as f64 / self.total as f64
    }

    /// Returns references to all failure results.
    pub fn failures(&self) -> Vec<&BatchItemResult> {
        self.results.iter().filter(|r| r.is_failure()).collect()
    }

    /// Returns references to all success results.
    pub fn successes(&self) -> Vec<&BatchItemResult> {
        self.results.iter().filter(|r| r.is_success()).collect()
    }

    /// Serialize this result to a JSON value.
    pub fn to_json(&self) -> Value {
        let items: Vec<Value> = self
            .results
            .iter()
            .map(|r| match r {
                BatchItemResult::Success { index, value } => {
                    json!({ "index": index, "status": "success", "value": value })
                }
                BatchItemResult::Failure { index, error } => {
                    json!({ "index": index, "status": "failure", "error": error })
                }
            })
            .collect();

        json!({
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "duration_ms": self.duration_ms,
            "success_rate": self.success_rate(),
            "results": items,
        })
    }
}

// ---------------------------------------------------------------------------
// BatchProgress
// ---------------------------------------------------------------------------

/// Thread-safe progress tracker for batch execution.
pub struct BatchProgress {
    total: usize,
    succeeded: AtomicUsize,
    failed: AtomicUsize,
}

impl BatchProgress {
    /// Create a new progress tracker for `total` items.
    pub fn new(total: usize) -> Self {
        Self {
            total,
            succeeded: AtomicUsize::new(0),
            failed: AtomicUsize::new(0),
        }
    }

    /// Record a successful item.
    pub fn record_success(&self) {
        self.succeeded.fetch_add(1, Ordering::SeqCst);
    }

    /// Record a failed item.
    pub fn record_failure(&self) {
        self.failed.fetch_add(1, Ordering::SeqCst);
    }

    /// Number of completed items (success + failure).
    pub fn completed(&self) -> usize {
        self.succeeded.load(Ordering::SeqCst) + self.failed.load(Ordering::SeqCst)
    }

    /// Number of remaining items.
    pub fn remaining(&self) -> usize {
        self.total.saturating_sub(self.completed())
    }

    /// Progress as a percentage (0.0 to 100.0).
    pub fn progress_percent(&self) -> f64 {
        if self.total == 0 {
            return 100.0;
        }
        (self.completed() as f64 / self.total as f64) * 100.0
    }

    /// Serialize current progress to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "total": self.total,
            "succeeded": self.succeeded.load(Ordering::SeqCst),
            "failed": self.failed.load(Ordering::SeqCst),
            "completed": self.completed(),
            "remaining": self.remaining(),
            "progress_percent": self.progress_percent(),
        })
    }
}

// ---------------------------------------------------------------------------
// ChunkIterator
// ---------------------------------------------------------------------------

/// Splits a vector of values into fixed-size chunks.
pub struct ChunkIterator {
    items: Vec<Value>,
    chunk_size: usize,
    offset: usize,
}

impl ChunkIterator {
    /// Create a new chunk iterator.
    ///
    /// `chunk_size` is clamped to at least 1.
    pub fn new(items: Vec<Value>, chunk_size: usize) -> Self {
        Self {
            items,
            chunk_size: chunk_size.max(1),
            offset: 0,
        }
    }
}

impl Iterator for ChunkIterator {
    type Item = Vec<Value>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.offset >= self.items.len() {
            return None;
        }
        let end = (self.offset + self.chunk_size).min(self.items.len());
        let chunk = self.items[self.offset..end].to_vec();
        self.offset = end;
        Some(chunk)
    }
}

// ---------------------------------------------------------------------------
// batch_invoke (enhanced version with BatchConfig)
// ---------------------------------------------------------------------------

/// Process a batch of inputs through a runnable with full batch configuration.
///
/// Items are split into chunks of `config.chunk_size`, and within each chunk
/// up to `config.max_concurrency` items are processed concurrently.
///
/// If `config.fail_fast` is `true`, processing stops at the first failure.
///
/// Returns a [`BatchResult`] with per-item outcomes and aggregate statistics.
pub async fn batch_process(
    runnable: Arc<dyn Runnable>,
    inputs: Vec<Value>,
    config: BatchConfig,
    runnable_config: Option<&RunnableConfig>,
) -> BatchResult {
    let total = inputs.len();
    let start = Instant::now();

    if total == 0 {
        return BatchResult {
            results: Vec::new(),
            total: 0,
            succeeded: 0,
            failed: 0,
            duration_ms: 0,
        };
    }

    let progress = Arc::new(BatchProgress::new(total));
    let mut all_results: Vec<BatchItemResult> = Vec::with_capacity(total);

    let chunks = ChunkIterator::new(inputs, config.chunk_size);
    let mut global_offset: usize = 0;
    let mut early_stop = false;

    for chunk in chunks {
        if early_stop {
            // Mark remaining items as failures
            for (local_idx, _) in chunk.iter().enumerate() {
                let idx = global_offset + local_idx;
                all_results.push(BatchItemResult::Failure {
                    index: idx,
                    error: "batch aborted due to fail_fast".to_string(),
                });
                progress.record_failure();
            }
            global_offset += chunk.len();
            continue;
        }

        let chunk_len = chunk.len();

        // Process chunk items concurrently using semaphore-bounded futures.
        use futures::stream::{self, StreamExt};
        let semaphore = Arc::new(tokio::sync::Semaphore::new(config.max_concurrency));

        let chunk_results: Vec<BatchItemResult> =
            stream::iter(chunk.into_iter().enumerate().map(|(local_idx, input)| {
                let idx = global_offset + local_idx;
                let runnable = Arc::clone(&runnable);
                let sem = Arc::clone(&semaphore);
                let cfg = runnable_config.cloned();
                let timeout_ms = config.timeout_per_item_ms;
                let retry = config.retry_failed;

                async move {
                    let _permit = sem.acquire().await.unwrap();
                    let result =
                        invoke_with_options(&runnable, input.clone(), cfg.as_ref(), timeout_ms)
                            .await;

                    match result {
                        Ok(value) => BatchItemResult::Success { index: idx, value },
                        Err(e) => {
                            // Retry once if configured
                            if retry {
                                match invoke_with_options(
                                    &runnable,
                                    input,
                                    cfg.as_ref(),
                                    timeout_ms,
                                )
                                .await
                                {
                                    Ok(value) => BatchItemResult::Success { index: idx, value },
                                    Err(e2) => BatchItemResult::Failure {
                                        index: idx,
                                        error: e2.to_string(),
                                    },
                                }
                            } else {
                                BatchItemResult::Failure {
                                    index: idx,
                                    error: e.to_string(),
                                }
                            }
                        }
                    }
                }
            }))
            .buffer_unordered(config.max_concurrency)
            .collect()
            .await;

        // Sort by index to maintain order within the chunk
        let mut sorted = chunk_results;
        sorted.sort_by_key(|r| r.index());

        for item in &sorted {
            if item.is_success() {
                progress.record_success();
            } else {
                progress.record_failure();
                if config.fail_fast {
                    early_stop = true;
                }
            }
        }

        all_results.extend(sorted);
        global_offset += chunk_len;
    }

    // Sort all results by index for a consistent ordering
    all_results.sort_by_key(|r| r.index());

    let succeeded = all_results.iter().filter(|r| r.is_success()).count();
    let failed = all_results.iter().filter(|r| r.is_failure()).count();
    let duration_ms = start.elapsed().as_millis() as u64;

    BatchResult {
        results: all_results,
        total,
        succeeded,
        failed,
        duration_ms,
    }
}

/// Invoke a runnable with optional timeout.
async fn invoke_with_options(
    runnable: &Arc<dyn Runnable>,
    input: Value,
    config: Option<&RunnableConfig>,
    timeout_ms: Option<u64>,
) -> Result<Value> {
    match timeout_ms {
        Some(ms) => {
            let duration = std::time::Duration::from_millis(ms);
            match tokio::time::timeout(duration, runnable.invoke(input, config)).await {
                Ok(result) => result,
                Err(_) => Err(CognisError::Other("batch item timed out".to_string())),
            }
        }
        None => runnable.invoke(input, config).await,
    }
}

// ---------------------------------------------------------------------------
// RunnableBatchProcessor
// ---------------------------------------------------------------------------

/// A runnable wrapper that applies batch processing with full [`BatchConfig`].
///
/// Input: a JSON array of items.
/// Output: a JSON object with batch result details, or a JSON array of values.
pub struct RunnableBatchProcessor {
    inner: Arc<dyn Runnable>,
    config: BatchConfig,
    name: String,
}

impl RunnableBatchProcessor {
    /// Create a new batch processor wrapping the given runnable.
    pub fn new(inner: Arc<dyn Runnable>, config: BatchConfig) -> Self {
        let name = format!("RunnableBatchProcessor<{}>", inner.name());
        Self {
            inner,
            config,
            name,
        }
    }

    /// Run the batch and return a structured [`BatchResult`].
    pub async fn invoke_batch(&self, inputs: Vec<Value>) -> BatchResult {
        batch_process(Arc::clone(&self.inner), inputs, self.config.clone(), None).await
    }
}

#[async_trait]
impl Runnable for RunnableBatchProcessor {
    fn name(&self) -> &str {
        &self.name
    }

    /// Invoke expects a JSON array. Returns a JSON array of results where
    /// successful items are their values and failures are `{"error": "..."}`.
    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let items = input
            .as_array()
            .ok_or_else(|| CognisError::TypeMismatch {
                expected: "Array".into(),
                got: input_type_name(&input).to_string(),
            })?
            .clone();

        let batch_result =
            batch_process(Arc::clone(&self.inner), items, self.config.clone(), _config).await;

        let output: Vec<Value> = batch_result
            .results
            .into_iter()
            .map(|r| match r {
                BatchItemResult::Success { value, .. } => value,
                BatchItemResult::Failure { error, .. } => json!({ "error": error }),
            })
            .collect();

        Ok(Value::Array(output))
    }
}

/// Helper to get a human-readable type name for a JSON value.
fn input_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "Null",
        Value::Bool(_) => "Bool",
        Value::Number(_) => "Number",
        Value::String(_) => "String",
        Value::Array(_) => "Array",
        Value::Object(_) => "Object",
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::RunnableLambda;
    use serde_json::json;
    use std::time::Duration;

    /// Helper: a simple doubler runnable.
    fn doubler() -> RunnableLambda {
        RunnableLambda::new("doubler", |v: Value| async move {
            let n = v.as_i64().ok_or_else(|| CognisError::TypeMismatch {
                expected: "integer".into(),
                got: format!("{v}"),
            })?;
            Ok(json!(n * 2))
        })
    }

    /// Helper: a slow doubler for timing tests.
    fn slow_doubler(delay_ms: u64) -> RunnableLambda {
        RunnableLambda::new("slow_doubler", move |v: Value| async move {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            let n = v.as_i64().unwrap();
            Ok(json!(n * 2))
        })
    }

    /// Helper: a runnable that fails on even numbers.
    fn fail_even() -> RunnableLambda {
        RunnableLambda::new("fail_even", |v: Value| async move {
            let n = v.as_i64().unwrap();
            if n % 2 == 0 {
                Err(CognisError::Other(format!("even number: {n}")))
            } else {
                Ok(json!(n * 10))
            }
        })
    }

    // -----------------------------------------------------------------------
    // BatchConfig tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_config_defaults() {
        let cfg = BatchConfig::default();
        assert_eq!(cfg.max_concurrency, 10);
        assert_eq!(cfg.chunk_size, 100);
        assert!(!cfg.fail_fast);
        assert!(cfg.timeout_per_item_ms.is_none());
        assert!(!cfg.retry_failed);
    }

    #[test]
    fn test_batch_config_new_equals_default() {
        let a = BatchConfig::new();
        let b = BatchConfig::default();
        assert_eq!(a.max_concurrency, b.max_concurrency);
        assert_eq!(a.chunk_size, b.chunk_size);
        assert_eq!(a.fail_fast, b.fail_fast);
        assert_eq!(a.timeout_per_item_ms, b.timeout_per_item_ms);
        assert_eq!(a.retry_failed, b.retry_failed);
    }

    #[test]
    fn test_batch_config_builder() {
        let cfg = BatchConfig::new()
            .with_max_concurrency(5)
            .with_chunk_size(50)
            .with_fail_fast(true)
            .with_timeout_per_item_ms(2000)
            .with_retry_failed(true);

        assert_eq!(cfg.max_concurrency, 5);
        assert_eq!(cfg.chunk_size, 50);
        assert!(cfg.fail_fast);
        assert_eq!(cfg.timeout_per_item_ms, Some(2000));
        assert!(cfg.retry_failed);
    }

    #[test]
    fn test_batch_config_builder_chaining() {
        let cfg = BatchConfig::new()
            .with_max_concurrency(3)
            .with_chunk_size(10);

        assert_eq!(cfg.max_concurrency, 3);
        assert_eq!(cfg.chunk_size, 10);
        // Unchanged defaults
        assert!(!cfg.fail_fast);
        assert!(!cfg.retry_failed);
    }

    // -----------------------------------------------------------------------
    // BatchItemResult tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_item_result_success() {
        let item = BatchItemResult::Success {
            index: 0,
            value: json!(42),
        };
        assert!(item.is_success());
        assert!(!item.is_failure());
        assert_eq!(item.index(), 0);
    }

    #[test]
    fn test_batch_item_result_failure() {
        let item = BatchItemResult::Failure {
            index: 3,
            error: "something went wrong".into(),
        };
        assert!(!item.is_success());
        assert!(item.is_failure());
        assert_eq!(item.index(), 3);
    }

    #[test]
    fn test_batch_item_result_into_value_success() {
        let item = BatchItemResult::Success {
            index: 0,
            value: json!("hello"),
        };
        assert_eq!(item.into_value(), Some(json!("hello")));
    }

    #[test]
    fn test_batch_item_result_into_value_failure() {
        let item = BatchItemResult::Failure {
            index: 0,
            error: "err".into(),
        };
        assert_eq!(item.into_value(), None);
    }

    // -----------------------------------------------------------------------
    // BatchResult tests
    // -----------------------------------------------------------------------

    fn make_batch_result(successes: usize, failures: usize) -> BatchResult {
        let mut results = Vec::new();
        for i in 0..successes {
            results.push(BatchItemResult::Success {
                index: i,
                value: json!(i),
            });
        }
        for i in 0..failures {
            results.push(BatchItemResult::Failure {
                index: successes + i,
                error: format!("error {i}"),
            });
        }
        BatchResult {
            total: successes + failures,
            succeeded: successes,
            failed: failures,
            duration_ms: 100,
            results,
        }
    }

    #[test]
    fn test_batch_result_success_rate_all_success() {
        let r = make_batch_result(10, 0);
        assert!((r.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_result_success_rate_all_failure() {
        let r = make_batch_result(0, 5);
        assert!((r.success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_result_success_rate_mixed() {
        let r = make_batch_result(3, 1);
        assert!((r.success_rate() - 0.75).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_result_success_rate_empty() {
        let r = BatchResult {
            results: vec![],
            total: 0,
            succeeded: 0,
            failed: 0,
            duration_ms: 0,
        };
        assert!((r.success_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_result_failures_filter() {
        let r = make_batch_result(2, 3);
        let failures = r.failures();
        assert_eq!(failures.len(), 3);
        for f in &failures {
            assert!(f.is_failure());
        }
    }

    #[test]
    fn test_batch_result_successes_filter() {
        let r = make_batch_result(4, 2);
        let successes = r.successes();
        assert_eq!(successes.len(), 4);
        for s in &successes {
            assert!(s.is_success());
        }
    }

    #[test]
    fn test_batch_result_to_json() {
        let r = make_batch_result(1, 1);
        let j = r.to_json();
        assert_eq!(j["total"], 2);
        assert_eq!(j["succeeded"], 1);
        assert_eq!(j["failed"], 1);
        assert!(j["duration_ms"].is_number());
        assert!(j["results"].is_array());
        assert_eq!(j["results"].as_array().unwrap().len(), 2);
    }

    // -----------------------------------------------------------------------
    // batch_process tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_batch_process_all_success() {
        let runnable = Arc::new(doubler()) as Arc<dyn Runnable>;
        let inputs: Vec<Value> = (1..=5).map(|i| json!(i)).collect();
        let config = BatchConfig::new().with_chunk_size(3);

        let result = batch_process(runnable, inputs, config, None).await;

        assert_eq!(result.total, 5);
        assert_eq!(result.succeeded, 5);
        assert_eq!(result.failed, 0);
        assert_eq!(result.results.len(), 5);
        for (i, item) in result.results.iter().enumerate() {
            let expected = (i as i64 + 1) * 2;
            assert_eq!(item.clone().into_value(), Some(json!(expected)));
        }
    }

    #[tokio::test]
    async fn test_batch_process_mixed_results() {
        let runnable = Arc::new(fail_even()) as Arc<dyn Runnable>;
        let inputs = vec![json!(1), json!(2), json!(3), json!(4), json!(5)];
        let config = BatchConfig::new();

        let result = batch_process(runnable, inputs, config, None).await;

        assert_eq!(result.total, 5);
        assert_eq!(result.succeeded, 3);
        assert_eq!(result.failed, 2);
        assert!(result.results[0].is_success()); // 1 (odd)
        assert!(result.results[1].is_failure()); // 2 (even)
        assert!(result.results[2].is_success()); // 3 (odd)
        assert!(result.results[3].is_failure()); // 4 (even)
        assert!(result.results[4].is_success()); // 5 (odd)
    }

    #[tokio::test]
    async fn test_batch_process_fail_fast() {
        let runnable = Arc::new(fail_even()) as Arc<dyn Runnable>;
        // Items: 1, 2, 3, 4 — process in chunk_size=1 so failures detected immediately
        let inputs = vec![json!(1), json!(2), json!(3), json!(4)];
        let config = BatchConfig::new().with_fail_fast(true).with_chunk_size(1);

        let result = batch_process(runnable, inputs, config, None).await;

        assert_eq!(result.total, 4);
        // First chunk (1) succeeds, second chunk (2) fails, rest should be aborted
        assert!(result.results[0].is_success());
        assert!(result.results[1].is_failure());
        // Items 3 and 4 should also be failures (aborted)
        assert!(result.results[2].is_failure());
        assert!(result.results[3].is_failure());
    }

    #[tokio::test]
    async fn test_batch_process_fail_fast_all_succeed() {
        let runnable = Arc::new(doubler()) as Arc<dyn Runnable>;
        let inputs = vec![json!(1), json!(3), json!(5)];
        let config = BatchConfig::new().with_fail_fast(true);

        let result = batch_process(runnable, inputs, config, None).await;

        assert_eq!(result.total, 3);
        assert_eq!(result.succeeded, 3);
        assert_eq!(result.failed, 0);
    }

    #[tokio::test]
    async fn test_batch_process_empty_input() {
        let runnable = Arc::new(doubler()) as Arc<dyn Runnable>;
        let config = BatchConfig::new();

        let result = batch_process(runnable, vec![], config, None).await;

        assert_eq!(result.total, 0);
        assert_eq!(result.succeeded, 0);
        assert_eq!(result.failed, 0);
        assert!(result.results.is_empty());
        assert_eq!(result.duration_ms, 0);
    }

    #[tokio::test]
    async fn test_batch_process_chunk_size_1() {
        let runnable = Arc::new(doubler()) as Arc<dyn Runnable>;
        let inputs = vec![json!(10), json!(20), json!(30)];
        let config = BatchConfig::new().with_chunk_size(1);

        let result = batch_process(runnable, inputs, config, None).await;

        assert_eq!(result.total, 3);
        assert_eq!(result.succeeded, 3);
        assert_eq!(result.results[0].clone().into_value(), Some(json!(20)));
        assert_eq!(result.results[1].clone().into_value(), Some(json!(40)));
        assert_eq!(result.results[2].clone().into_value(), Some(json!(60)));
    }

    #[tokio::test]
    async fn test_batch_process_chunk_size_larger_than_total() {
        let runnable = Arc::new(doubler()) as Arc<dyn Runnable>;
        let inputs = vec![json!(1), json!(2)];
        let config = BatchConfig::new().with_chunk_size(1000);

        let result = batch_process(runnable, inputs, config, None).await;

        assert_eq!(result.total, 2);
        assert_eq!(result.succeeded, 2);
    }

    #[tokio::test]
    async fn test_batch_process_concurrency_control() {
        let counter = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let c1 = counter.clone();
        let p1 = peak.clone();

        let runnable = Arc::new(RunnableLambda::new("track", move |v: Value| {
            let counter = c1.clone();
            let peak = p1.clone();
            async move {
                let cur = counter.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(cur, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(30)).await;
                counter.fetch_sub(1, Ordering::SeqCst);
                Ok(v)
            }
        })) as Arc<dyn Runnable>;

        let inputs: Vec<Value> = (0..10).map(|i| json!(i)).collect();
        let config = BatchConfig::new()
            .with_max_concurrency(2)
            .with_chunk_size(10);

        let _result = batch_process(runnable, inputs, config, None).await;

        let observed_peak = peak.load(Ordering::SeqCst);
        assert!(
            observed_peak <= 2,
            "peak concurrency should be <= 2, was {observed_peak}"
        );
    }

    #[tokio::test]
    async fn test_batch_process_with_timeout() {
        let runnable = Arc::new(RunnableLambda::new("slow", |v: Value| async move {
            let n = v.as_i64().unwrap();
            // Even items sleep too long
            if n % 2 == 0 {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            Ok(json!(n))
        })) as Arc<dyn Runnable>;

        let inputs = vec![json!(1), json!(2), json!(3)];
        let config = BatchConfig::new().with_timeout_per_item_ms(100);

        let result = batch_process(runnable, inputs, config, None).await;

        assert_eq!(result.total, 3);
        assert!(result.results[0].is_success()); // 1 - fast
        assert!(result.results[1].is_failure()); // 2 - times out
        assert!(result.results[2].is_success()); // 3 - fast
    }

    #[tokio::test]
    async fn test_batch_process_retry_failed() {
        // A runnable that fails the first time for item 0, succeeds second time
        let attempt = Arc::new(AtomicUsize::new(0));
        let attempt_clone = attempt.clone();

        let runnable = Arc::new(RunnableLambda::new("retry_test", move |v: Value| {
            let attempt = attempt_clone.clone();
            async move {
                let n = v.as_i64().unwrap();
                if n == 0 {
                    let count = attempt.fetch_add(1, Ordering::SeqCst);
                    if count == 0 {
                        return Err(CognisError::Other("transient error".into()));
                    }
                }
                Ok(json!(n * 10))
            }
        })) as Arc<dyn Runnable>;

        let inputs = vec![json!(0), json!(1)];
        let config = BatchConfig::new().with_retry_failed(true);

        let result = batch_process(runnable, inputs, config, None).await;

        assert_eq!(result.total, 2);
        assert_eq!(result.succeeded, 2);
        assert!(result.results[0].is_success());
        assert!(result.results[1].is_success());
    }

    #[tokio::test]
    async fn test_batch_process_preserves_order() {
        let runnable = Arc::new(RunnableLambda::new(
            "delay_by_value",
            |v: Value| async move {
                let n = v.as_i64().unwrap();
                // Higher index items finish faster
                let delay = (5 - n) as u64 * 10;
                tokio::time::sleep(Duration::from_millis(delay)).await;
                Ok(json!(n * 100))
            },
        )) as Arc<dyn Runnable>;

        let inputs: Vec<Value> = (0..5).map(|i| json!(i)).collect();
        let config = BatchConfig::new().with_chunk_size(5);

        let result = batch_process(runnable, inputs, config, None).await;

        for (i, item) in result.results.iter().enumerate() {
            assert_eq!(item.index(), i);
            assert_eq!(item.clone().into_value(), Some(json!(i as i64 * 100)));
        }
    }

    // -----------------------------------------------------------------------
    // RunnableBatchProcessor tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_runnable_batch_processor_invoke_array() {
        let inner = Arc::new(doubler()) as Arc<dyn Runnable>;
        let config = BatchConfig::new();
        let processor = RunnableBatchProcessor::new(inner, config);

        let input = json!([1, 2, 3]);
        let output = processor.invoke(input, None).await.unwrap();
        let arr = output.as_array().unwrap();

        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0], json!(2));
        assert_eq!(arr[1], json!(4));
        assert_eq!(arr[2], json!(6));
    }

    #[tokio::test]
    async fn test_runnable_batch_processor_invoke_non_array() {
        let inner = Arc::new(doubler()) as Arc<dyn Runnable>;
        let config = BatchConfig::new();
        let processor = RunnableBatchProcessor::new(inner, config);

        let result = processor.invoke(json!("not array"), None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_runnable_batch_processor_invoke_batch_method() {
        let inner = Arc::new(doubler()) as Arc<dyn Runnable>;
        let config = BatchConfig::new().with_chunk_size(2);
        let processor = RunnableBatchProcessor::new(inner, config);

        let inputs = vec![json!(10), json!(20), json!(30)];
        let result = processor.invoke_batch(inputs).await;

        assert_eq!(result.total, 3);
        assert_eq!(result.succeeded, 3);
        assert_eq!(result.results[0].clone().into_value(), Some(json!(20)));
        assert_eq!(result.results[1].clone().into_value(), Some(json!(40)));
        assert_eq!(result.results[2].clone().into_value(), Some(json!(60)));
    }

    #[tokio::test]
    async fn test_runnable_batch_processor_name() {
        let inner = Arc::new(doubler()) as Arc<dyn Runnable>;
        let config = BatchConfig::new();
        let processor = RunnableBatchProcessor::new(inner, config);
        assert_eq!(processor.name(), "RunnableBatchProcessor<doubler>");
    }

    #[tokio::test]
    async fn test_runnable_batch_processor_with_failures() {
        let inner = Arc::new(fail_even()) as Arc<dyn Runnable>;
        let config = BatchConfig::new();
        let processor = RunnableBatchProcessor::new(inner, config);

        let input = json!([1, 2, 3]);
        let output = processor.invoke(input, None).await.unwrap();
        let arr = output.as_array().unwrap();

        assert_eq!(arr[0], json!(10));
        assert!(arr[1].get("error").is_some());
        assert_eq!(arr[2], json!(30));
    }

    // -----------------------------------------------------------------------
    // BatchProgress tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_progress_initial() {
        let p = BatchProgress::new(10);
        assert_eq!(p.completed(), 0);
        assert_eq!(p.remaining(), 10);
        assert!((p.progress_percent() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_progress_record_success() {
        let p = BatchProgress::new(4);
        p.record_success();
        p.record_success();
        assert_eq!(p.completed(), 2);
        assert_eq!(p.remaining(), 2);
        assert!((p.progress_percent() - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_progress_record_failure() {
        let p = BatchProgress::new(4);
        p.record_failure();
        assert_eq!(p.completed(), 1);
        assert_eq!(p.remaining(), 3);
    }

    #[test]
    fn test_batch_progress_mixed() {
        let p = BatchProgress::new(5);
        p.record_success();
        p.record_success();
        p.record_failure();
        assert_eq!(p.completed(), 3);
        assert_eq!(p.remaining(), 2);
        assert!((p.progress_percent() - 60.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_progress_complete() {
        let p = BatchProgress::new(2);
        p.record_success();
        p.record_failure();
        assert_eq!(p.completed(), 2);
        assert_eq!(p.remaining(), 0);
        assert!((p.progress_percent() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_progress_zero_total() {
        let p = BatchProgress::new(0);
        assert_eq!(p.completed(), 0);
        assert_eq!(p.remaining(), 0);
        assert!((p.progress_percent() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_progress_to_json() {
        let p = BatchProgress::new(10);
        p.record_success();
        p.record_success();
        p.record_failure();
        let j = p.to_json();
        assert_eq!(j["total"], 10);
        assert_eq!(j["succeeded"], 2);
        assert_eq!(j["failed"], 1);
        assert_eq!(j["completed"], 3);
        assert_eq!(j["remaining"], 7);
        assert!((j["progress_percent"].as_f64().unwrap() - 30.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // ChunkIterator tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chunk_iterator_basic() {
        let items: Vec<Value> = (0..5).map(|i| json!(i)).collect();
        let chunks: Vec<Vec<Value>> = ChunkIterator::new(items, 2).collect();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], vec![json!(0), json!(1)]);
        assert_eq!(chunks[1], vec![json!(2), json!(3)]);
        assert_eq!(chunks[2], vec![json!(4)]);
    }

    #[test]
    fn test_chunk_iterator_exact_division() {
        let items: Vec<Value> = (0..6).map(|i| json!(i)).collect();
        let chunks: Vec<Vec<Value>> = ChunkIterator::new(items, 3).collect();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].len(), 3);
        assert_eq!(chunks[1].len(), 3);
    }

    #[test]
    fn test_chunk_iterator_chunk_size_1() {
        let items = vec![json!("a"), json!("b"), json!("c")];
        let chunks: Vec<Vec<Value>> = ChunkIterator::new(items, 1).collect();

        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert_eq!(chunk.len(), 1);
        }
    }

    #[test]
    fn test_chunk_iterator_chunk_size_larger_than_items() {
        let items = vec![json!(1), json!(2)];
        let chunks: Vec<Vec<Value>> = ChunkIterator::new(items, 100).collect();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].len(), 2);
    }

    #[test]
    fn test_chunk_iterator_empty() {
        let chunks: Vec<Vec<Value>> = ChunkIterator::new(vec![], 5).collect();
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_chunk_iterator_chunk_size_zero_clamped() {
        // chunk_size of 0 should be clamped to 1
        let items = vec![json!(1), json!(2), json!(3)];
        let chunks: Vec<Vec<Value>> = ChunkIterator::new(items, 0).collect();

        assert_eq!(chunks.len(), 3);
        for chunk in &chunks {
            assert_eq!(chunk.len(), 1);
        }
    }

    #[test]
    fn test_chunk_iterator_single_item() {
        let items = vec![json!(42)];
        let chunks: Vec<Vec<Value>> = ChunkIterator::new(items, 10).collect();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], vec![json!(42)]);
    }
}
