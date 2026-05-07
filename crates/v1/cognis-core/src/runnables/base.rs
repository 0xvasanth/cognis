use async_trait::async_trait;
use futures::stream::{self, StreamExt};
use serde_json::Value;

use crate::error::Result;

use super::config::RunnableConfig;
use super::RunnableStream;

/// Core trait for composable execution units in LCEL.
///
/// All runnables accept `serde_json::Value` as input and output,
/// enabling heterogeneous composition (sequences, parallels, branches).
#[async_trait]
pub trait Runnable: Send + Sync {
    /// Returns the name of this runnable.
    fn name(&self) -> &str;

    /// Invoke the runnable with a single input.
    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value>;

    /// Invoke the runnable with multiple inputs, returning results in order.
    async fn batch(
        &self,
        inputs: Vec<Value>,
        config: Option<&RunnableConfig>,
    ) -> Result<Vec<Value>> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.invoke(input, config).await?);
        }
        Ok(results)
    }

    /// Process multiple inputs concurrently with optional concurrency limit.
    ///
    /// Unlike `batch` which processes sequentially by default, `abatch` uses
    /// concurrent futures to process inputs in parallel, respecting
    /// `max_concurrency` from the config.
    ///
    /// Results are returned in the same order as the inputs, regardless of
    /// the order in which the futures complete.
    async fn abatch(
        &self,
        inputs: Vec<Value>,
        config: Option<&RunnableConfig>,
    ) -> Vec<Result<Value>> {
        let concurrency = config
            .and_then(|c| c.max_concurrency)
            .unwrap_or(inputs.len().max(1));

        let results: Vec<Result<Value>> = stream::iter(inputs.into_iter().enumerate())
            .map(|(idx, input)| async move {
                let result = self.invoke(input, config).await;
                (idx, result)
            })
            .buffer_unordered(concurrency)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .fold(Vec::new(), |mut acc, (idx, result)| {
                // Ensure we have enough capacity
                if acc.len() <= idx {
                    acc.resize_with(idx + 1, || Ok(Value::Null));
                }
                acc[idx] = result;
                acc
            });

        results
    }

    /// Stream the runnable output. Default yields a single result from invoke.
    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        let result = self.invoke(input, config).await;
        Ok(Box::pin(stream::once(async { result })))
    }

    /// Returns the JSON Schema describing valid inputs for this runnable.
    ///
    /// Default returns a permissive schema that accepts any JSON value.
    /// Implementations should override to provide specific schemas for
    /// validation, documentation, and API serving.
    fn input_schema(&self) -> Value {
        serde_json::json!({
            "description": format!("Input for {}", self.name())
        })
    }

    /// Returns the JSON Schema describing the output of this runnable.
    ///
    /// Default returns a permissive schema that accepts any JSON value.
    fn output_schema(&self) -> Value {
        serde_json::json!({
            "description": format!("Output of {}", self.name())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::{RunnableExt, RunnableLambda};
    use serde_json::json;
    use std::time::{Duration, Instant};

    /// Helper: creates a RunnableLambda that sleeps for the given duration
    /// then returns the input doubled (assumes integer input).
    fn slow_double(delay_ms: u64) -> RunnableLambda {
        RunnableLambda::new("slow_double", move |v: Value| async move {
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;
            let n = v.as_i64().unwrap();
            Ok(json!(n * 2))
        })
    }

    #[tokio::test]
    async fn test_abatch_concurrent_faster_than_sequential() {
        let delay_ms = 50;
        let runnable = slow_double(delay_ms);
        let inputs: Vec<Value> = (0..4).map(|i| json!(i)).collect();

        // Sequential batch
        let start = Instant::now();
        let seq_results = runnable.batch(inputs.clone(), None).await.unwrap();
        let seq_elapsed = start.elapsed();

        // Concurrent abatch (unlimited concurrency)
        let start = Instant::now();
        let conc_results = runnable.abatch(inputs.clone(), None).await;
        let conc_elapsed = start.elapsed();

        // Verify correctness
        assert_eq!(seq_results.len(), 4);
        assert_eq!(conc_results.len(), 4);
        for (i, r) in conc_results.iter().enumerate() {
            assert_eq!(r.as_ref().unwrap(), &json!(i as i64 * 2));
        }

        // Concurrent should be significantly faster than sequential
        // Sequential: ~4*50ms = 200ms, Concurrent: ~50ms
        assert!(
            conc_elapsed < seq_elapsed,
            "abatch ({:?}) should be faster than batch ({:?})",
            conc_elapsed,
            seq_elapsed
        );
    }

    #[tokio::test]
    async fn test_abatch_concurrency_limit_1_is_sequential() {
        let delay_ms = 50;
        let runnable = slow_double(delay_ms);
        let inputs: Vec<Value> = (0..4).map(|i| json!(i)).collect();

        let mut config = RunnableConfig::default();
        config.max_concurrency = Some(1);

        let start = Instant::now();
        let results = runnable.abatch(inputs, Some(&config)).await;
        let elapsed = start.elapsed();

        // With concurrency=1, should take ~4*50ms = 200ms minimum
        assert_eq!(results.len(), 4);
        assert!(
            elapsed >= Duration::from_millis(delay_ms * 4 - 20),
            "concurrency=1 should be sequential-like, elapsed: {:?}",
            elapsed
        );
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.as_ref().unwrap(), &json!(i as i64 * 2));
        }
    }

    #[tokio::test]
    async fn test_abatch_concurrency_limit_2_on_4_inputs() {
        let delay_ms = 50;
        let runnable = slow_double(delay_ms);
        let inputs: Vec<Value> = (0..4).map(|i| json!(i)).collect();

        let mut config = RunnableConfig::default();
        config.max_concurrency = Some(2);

        let start = Instant::now();
        let results = runnable.abatch(inputs, Some(&config)).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 4);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.as_ref().unwrap(), &json!(i as i64 * 2));
        }

        // With concurrency=2 and 4 inputs at 50ms each: ~2 batches of 2 = ~100ms
        // Should be faster than sequential (~200ms) but slower than fully concurrent (~50ms)
        assert!(
            elapsed < Duration::from_millis(delay_ms * 4 - 20),
            "concurrency=2 should be faster than sequential, elapsed: {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_abatch_preserves_input_order() {
        // Use varying delays to ensure out-of-order completion
        let runnable = RunnableLambda::new("delay_by_value", |v: Value| async move {
            let n = v.as_i64().unwrap();
            // Higher index items finish faster to force reordering
            let delay = (4 - n) as u64 * 20;
            tokio::time::sleep(Duration::from_millis(delay)).await;
            Ok(json!(n * 10))
        });

        let inputs: Vec<Value> = (0..5).map(|i| json!(i)).collect();
        let results = runnable.abatch(inputs, None).await;

        assert_eq!(results.len(), 5);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(
                r.as_ref().unwrap(),
                &json!(i as i64 * 10),
                "Result at index {} should be {} but got {:?}",
                i,
                i * 10,
                r
            );
        }
    }

    #[tokio::test]
    async fn test_with_concurrency_extension() {
        let delay_ms = 50;
        let runnable = slow_double(delay_ms);
        let bound = runnable.with_concurrency(2);

        // Verify the binding works by invoking
        let result = bound.invoke(json!(5), None).await.unwrap();
        assert_eq!(result, json!(10));

        // Verify abatch respects the concurrency from the bound config
        let inputs: Vec<Value> = (0..4).map(|i| json!(i)).collect();
        let start = Instant::now();
        let results = bound.abatch(inputs, None).await;
        let elapsed = start.elapsed();

        assert_eq!(results.len(), 4);
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.as_ref().unwrap(), &json!(i as i64 * 2));
        }

        // Should be faster than fully sequential
        assert!(
            elapsed < Duration::from_millis(delay_ms * 4 - 20),
            "with_concurrency(2) should limit but still parallelize, elapsed: {:?}",
            elapsed
        );
    }

    #[tokio::test]
    async fn test_abatch_empty_inputs() {
        let runnable = slow_double(10);
        let results = runnable.abatch(vec![], None).await;
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_abatch_single_input() {
        let runnable = slow_double(10);
        let results = runnable.abatch(vec![json!(7)], None).await;
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].as_ref().unwrap(), &json!(14));
    }

    #[test]
    fn test_default_input_schema() {
        let r = RunnableLambda::new("test_fn", |v: Value| async move { Ok(v) });
        let schema = r.input_schema();
        // Default schema is permissive (no "type" restriction) with a description.
        assert!(schema.get("type").is_none());
        assert!(schema["description"].as_str().unwrap().contains("test_fn"));
    }

    #[test]
    fn test_default_output_schema() {
        let r = RunnableLambda::new("test_fn", |v: Value| async move { Ok(v) });
        let schema = r.output_schema();
        assert!(schema.get("type").is_none());
        assert!(schema["description"].as_str().unwrap().contains("test_fn"));
    }
}
