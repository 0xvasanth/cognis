use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Semaphore;

use crate::error::{CognisError, Result};

use super::base::Runnable;
use super::config::RunnableConfig;

/// Invoke a runnable on multiple inputs concurrently, with optional concurrency limits.
///
/// Uses `tokio::sync::Semaphore` to bound parallelism. Results are returned in the
/// same order as inputs, regardless of completion order. Individual failures do not
/// prevent other inputs from being processed.
///
/// # Arguments
/// * `runnable` - The runnable to invoke on each input.
/// * `inputs` - A vector of JSON values to process.
/// * `max_concurrency` - Optional upper bound on concurrent invocations. If `None`,
///   all inputs are processed concurrently.
/// * `config` - Optional config forwarded to each invocation.
///
/// # Returns
/// A `Vec<Result<Value>>` with one result per input, preserving order.
pub async fn batch_invoke(
    runnable: &dyn Runnable,
    inputs: Vec<Value>,
    max_concurrency: Option<usize>,
    config: Option<&RunnableConfig>,
) -> Vec<Result<Value>> {
    if inputs.is_empty() {
        return Vec::new();
    }

    let concurrency = max_concurrency.unwrap_or(inputs.len());
    let semaphore = Arc::new(Semaphore::new(concurrency));

    let mut handles = Vec::with_capacity(inputs.len());

    // We need to collect futures that can be joined. Since `runnable` is not 'static,
    // we use `tokio::task::spawn` indirectly by driving futures on the current task
    // with index tracking.
    let config_owned = config.cloned();

    for (idx, input) in inputs.into_iter().enumerate() {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let cfg = config_owned.clone();

        // We can't spawn because `runnable` isn't 'static, so we collect futures
        // and drive them with buffer_unordered. But for semaphore-based control,
        // we use a different approach: spawn tasks with Arc<dyn Runnable>.
        // However, `runnable` is a reference. We'll use the futures approach instead.
        drop(permit); // We'll use a different pattern below
        handles.push((idx, input, cfg));
    }

    // Use futures stream with semaphore for proper concurrency control
    use futures::stream::{self, StreamExt};

    let results: Vec<(usize, Result<Value>)> = stream::iter(handles)
        .map(|(idx, input, cfg)| {
            let sem = semaphore.clone();
            async move {
                let _permit = sem.acquire().await.unwrap();
                let result = runnable.invoke(input, cfg.as_ref()).await;
                (idx, result)
            }
        })
        .buffer_unordered(concurrency)
        .collect()
        .await;

    // Reconstruct ordered results
    let len = results.len();
    let mut ordered = Vec::with_capacity(len);
    ordered.resize_with(len, || Err(CognisError::Other("missing result".into())));
    for (idx, result) in results {
        ordered[idx] = result;
    }

    ordered
}

/// A runnable wrapper that pre-configures batch processing settings.
///
/// `RunnableBatch` wraps an inner `Runnable` and applies batch semantics:
/// - Input: a JSON array of items to process
/// - Output: a JSON array of results (one per input item)
///
/// If any individual invocation fails, the corresponding entry in the output
/// array will be `null` and the error is collected separately.
///
/// Batch concurrency is controlled by the `max_concurrency` field.
pub struct RunnableBatch {
    inner: Arc<dyn Runnable>,
    max_concurrency: Option<usize>,
    name: String,
    /// If true, return errors as `{"error": "..."}` objects in the output array
    /// instead of propagating the first error.
    return_exceptions: bool,
}

impl RunnableBatch {
    /// Create a new `RunnableBatch` wrapping the given runnable.
    pub fn new(inner: Arc<dyn Runnable>) -> Self {
        let name = format!("RunnableBatch<{}>", inner.name());
        Self {
            inner,
            max_concurrency: None,
            name,
            return_exceptions: true,
        }
    }

    /// Set the maximum number of concurrent invocations.
    pub fn with_max_concurrency(mut self, max_concurrency: usize) -> Self {
        self.max_concurrency = Some(max_concurrency);
        self
    }

    /// Set whether to return exceptions inline or propagate the first error.
    ///
    /// When `true` (default), failed items produce `{"error": "..."}` in the output array.
    /// When `false`, the first error causes the entire batch to fail.
    pub fn with_return_exceptions(mut self, return_exceptions: bool) -> Self {
        self.return_exceptions = return_exceptions;
        self
    }
}

#[async_trait]
impl Runnable for RunnableBatch {
    fn name(&self) -> &str {
        &self.name
    }

    /// Invoke the batch runnable.
    ///
    /// Input must be a JSON array. Each element is processed through the inner
    /// runnable. Output is a JSON array of results.
    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let items = input
            .as_array()
            .ok_or_else(|| CognisError::TypeMismatch {
                expected: "Array".into(),
                got: input_type_name(&input).to_string(),
            })?
            .clone();

        // Determine effective max_concurrency: explicit setting takes precedence,
        // then config, then unlimited.
        let effective_concurrency = self
            .max_concurrency
            .or_else(|| config.and_then(|c| c.max_concurrency));

        let results = batch_invoke(self.inner.as_ref(), items, effective_concurrency, config).await;

        if self.return_exceptions {
            // Convert results to JSON array, representing errors as objects
            let output: Vec<Value> = results
                .into_iter()
                .map(|r| match r {
                    Ok(v) => v,
                    Err(e) => serde_json::json!({"error": e.to_string()}),
                })
                .collect();
            Ok(Value::Array(output))
        } else {
            // Propagate the first error, or collect all successes
            let mut output = Vec::with_capacity(results.len());
            for r in results {
                output.push(r?);
            }
            Ok(Value::Array(output))
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::RunnableLambda;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    /// Helper: creates a RunnableLambda that doubles an integer input.
    fn doubler() -> RunnableLambda {
        RunnableLambda::new("doubler", |v: Value| async move {
            let n = v.as_i64().ok_or_else(|| CognisError::TypeMismatch {
                expected: "integer".into(),
                got: format!("{v}"),
            })?;
            Ok(json!(n * 2))
        })
    }

    /// Helper: creates a slow doubler for timing tests.
    fn slow_doubler(delay_ms: u64) -> RunnableLambda {
        RunnableLambda::new("slow_doubler", move |v: Value| async move {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            let n = v.as_i64().unwrap();
            Ok(json!(n * 2))
        })
    }

    // ---- batch_invoke tests ----

    #[tokio::test]
    async fn test_batch_invoke_basic() {
        let runnable = doubler();
        let inputs = vec![json!(1), json!(2), json!(3), json!(4)];
        let results = batch_invoke(&runnable, inputs, None, None).await;

        assert_eq!(results.len(), 4);
        assert_eq!(results[0].as_ref().unwrap(), &json!(2));
        assert_eq!(results[1].as_ref().unwrap(), &json!(4));
        assert_eq!(results[2].as_ref().unwrap(), &json!(6));
        assert_eq!(results[3].as_ref().unwrap(), &json!(8));
    }

    #[tokio::test]
    async fn test_batch_invoke_preserves_order() {
        // Use varying delays to force out-of-order completion
        let runnable = RunnableLambda::new("reverse_delay", |v: Value| async move {
            let n = v.as_i64().unwrap();
            let delay = (10 - n) as u64 * 15;
            tokio::time::sleep(Duration::from_millis(delay)).await;
            Ok(json!(n * 100))
        });

        let inputs: Vec<Value> = (0..10).map(|i| json!(i)).collect();
        let results = batch_invoke(&runnable, inputs, None, None).await;

        assert_eq!(results.len(), 10);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(
                r.as_ref().unwrap(),
                &json!(i as i64 * 100),
                "index {i} should be {}",
                i * 100
            );
        }
    }

    #[tokio::test]
    async fn test_batch_invoke_error_handling() {
        // Fail on even numbers, succeed on odd
        let runnable = RunnableLambda::new("fail_even", |v: Value| async move {
            let n = v.as_i64().unwrap();
            if n % 2 == 0 {
                Err(CognisError::Other(format!("even number: {n}")))
            } else {
                Ok(json!(n * 10))
            }
        });

        let inputs = vec![json!(1), json!(2), json!(3), json!(4), json!(5)];
        let results = batch_invoke(&runnable, inputs, None, None).await;

        assert_eq!(results.len(), 5);
        // Odd numbers succeed
        assert_eq!(results[0].as_ref().unwrap(), &json!(10));
        assert_eq!(results[2].as_ref().unwrap(), &json!(30));
        assert_eq!(results[4].as_ref().unwrap(), &json!(50));
        // Even numbers fail
        assert!(results[1].is_err());
        assert!(results[3].is_err());
    }

    #[tokio::test]
    async fn test_batch_invoke_max_concurrency_1_sequential() {
        let delay_ms: u64 = 40;
        let runnable = slow_doubler(delay_ms);
        let inputs: Vec<Value> = (0..4).map(|i| json!(i)).collect();

        let start = Instant::now();
        let results = batch_invoke(&runnable, inputs, Some(1), None).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 4);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.as_ref().unwrap(), &json!(i as i64 * 2));
        }

        // With max_concurrency=1, should take ~4 * delay_ms
        assert!(
            elapsed >= Duration::from_millis(delay_ms * 4 - 20),
            "max_concurrency=1 should be sequential, elapsed: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn test_batch_invoke_concurrency_limiting() {
        // Track peak concurrency using an atomic counter
        let counter = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let counter_clone = counter.clone();
        let peak_clone = peak.clone();

        let runnable = RunnableLambda::new("track_concurrency", move |v: Value| {
            let counter = counter_clone.clone();
            let peak = peak_clone.clone();
            async move {
                let current = counter.fetch_add(1, Ordering::SeqCst) + 1;
                // Update peak
                peak.fetch_max(current, Ordering::SeqCst);
                tokio::time::sleep(Duration::from_millis(50)).await;
                counter.fetch_sub(1, Ordering::SeqCst);
                Ok(v)
            }
        });

        let inputs: Vec<Value> = (0..10).map(|i| json!(i)).collect();
        let _results = batch_invoke(&runnable, inputs, Some(3), None).await;

        let observed_peak = peak.load(Ordering::SeqCst);
        assert!(
            observed_peak <= 3,
            "peak concurrency should be <= 3, was {observed_peak}"
        );
    }

    #[tokio::test]
    async fn test_batch_invoke_empty_inputs() {
        let runnable = doubler();
        let results = batch_invoke(&runnable, vec![], None, None).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_batch_invoke_with_config() {
        let runnable = RunnableLambda::with_config(
            "config_reader",
            |v, config: Option<RunnableConfig>| async move {
                let tag = config
                    .and_then(|c| c.tags.first().cloned())
                    .unwrap_or_else(|| "none".to_string());
                Ok(json!({"value": v, "tag": tag}))
            },
        );

        let mut config = RunnableConfig::default();
        config.tags = vec!["test_tag".to_string()];

        let inputs = vec![json!(1), json!(2)];
        let results = batch_invoke(&runnable, inputs, None, Some(&config)).await;

        assert_eq!(results.len(), 2);
        for r in &results {
            let v = r.as_ref().unwrap();
            assert_eq!(v["tag"], "test_tag");
        }
    }

    // ---- RunnableBatch tests ----

    #[tokio::test]
    async fn test_runnable_batch_basic() {
        let inner = Arc::new(doubler()) as Arc<dyn Runnable>;
        let batch = RunnableBatch::new(inner);

        let input = json!([1, 2, 3, 4, 5]);
        let output = batch.invoke(input, None).await.unwrap();

        let arr = output.as_array().unwrap();
        assert_eq!(arr.len(), 5);
        assert_eq!(arr[0], json!(2));
        assert_eq!(arr[1], json!(4));
        assert_eq!(arr[2], json!(6));
        assert_eq!(arr[3], json!(8));
        assert_eq!(arr[4], json!(10));
    }

    #[tokio::test]
    async fn test_runnable_batch_with_max_concurrency() {
        let delay_ms: u64 = 40;
        let inner = Arc::new(slow_doubler(delay_ms)) as Arc<dyn Runnable>;
        let batch = RunnableBatch::new(inner).with_max_concurrency(2);

        let input = json!([1, 2, 3, 4]);
        let start = Instant::now();
        let output = batch.invoke(input, None).await.unwrap();
        let elapsed = start.elapsed();

        let arr = output.as_array().unwrap();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr[0], json!(2));
        assert_eq!(arr[3], json!(8));

        // With concurrency=2 and 4 inputs: ~2 rounds of 2 = ~80ms
        // Should be faster than sequential (~160ms)
        assert!(
            elapsed < Duration::from_millis(delay_ms * 4 - 20),
            "batch with concurrency=2 should parallelize, elapsed: {elapsed:?}"
        );
    }

    #[tokio::test]
    async fn test_runnable_batch_return_exceptions() {
        let inner = Arc::new(RunnableLambda::new("fail_even", |v: Value| async move {
            let n = v.as_i64().unwrap();
            if n % 2 == 0 {
                Err(CognisError::Other(format!("even: {n}")))
            } else {
                Ok(json!(n * 10))
            }
        })) as Arc<dyn Runnable>;

        let batch = RunnableBatch::new(inner).with_return_exceptions(true);
        let input = json!([1, 2, 3, 4]);
        let output = batch.invoke(input, None).await.unwrap();

        let arr = output.as_array().unwrap();
        assert_eq!(arr.len(), 4);
        assert_eq!(arr[0], json!(10));
        assert!(arr[1].get("error").is_some());
        assert_eq!(arr[2], json!(30));
        assert!(arr[3].get("error").is_some());
    }

    #[tokio::test]
    async fn test_runnable_batch_propagate_errors() {
        let inner = Arc::new(RunnableLambda::new("fail_all", |_v: Value| async move {
            Err(CognisError::Other("always fails".into()))
        })) as Arc<dyn Runnable>;

        let batch = RunnableBatch::new(inner).with_return_exceptions(false);
        let input = json!([1, 2]);
        let result = batch.invoke(input, None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_runnable_batch_invalid_input_type() {
        let inner = Arc::new(doubler()) as Arc<dyn Runnable>;
        let batch = RunnableBatch::new(inner);

        let result = batch.invoke(json!("not an array"), None).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("expected Array"));
    }

    #[tokio::test]
    async fn test_runnable_batch_empty_array() {
        let inner = Arc::new(doubler()) as Arc<dyn Runnable>;
        let batch = RunnableBatch::new(inner);

        let output = batch.invoke(json!([]), None).await.unwrap();
        assert_eq!(output, json!([]));
    }

    #[tokio::test]
    async fn test_runnable_batch_name() {
        let inner = Arc::new(doubler()) as Arc<dyn Runnable>;
        let batch = RunnableBatch::new(inner);
        assert_eq!(batch.name(), "RunnableBatch<doubler>");
    }
}
