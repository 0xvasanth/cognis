//! Type-safe runnable trait for compile-time checked pipelines.
//!
//! [`TypedRunnable<I, O>`] provides the same semantics as [`Runnable`] but
//! with concrete input/output types. Use it when types flowing through a
//! pipeline are known at compile time.
//!
//! For heterogeneous composition (mixing different I/O types), use
//! [`DynRunnable`] to erase types back to `Value`.

use std::sync::Arc;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value;

use crate::error::{CognisError, Result};

use super::base::Runnable;
use super::config::RunnableConfig;

/// A runnable with concrete input and output types.
#[async_trait]
pub trait TypedRunnable<I, O>: Send + Sync
where
    I: Serialize + Send + 'static,
    O: DeserializeOwned + Send + 'static,
{
    /// Returns the name of this runnable.
    fn name(&self) -> &str;

    /// Invoke with a typed input, returning a typed output.
    async fn invoke(&self, input: I, config: Option<&RunnableConfig>) -> Result<O>;

    /// Process multiple inputs sequentially.
    async fn batch(&self, inputs: Vec<I>, config: Option<&RunnableConfig>) -> Result<Vec<O>> {
        let mut results = Vec::with_capacity(inputs.len());
        for input in inputs {
            results.push(self.invoke(input, config).await?);
        }
        Ok(results)
    }
}

/// Wraps a [`TypedRunnable<I, O>`] as a [`Runnable`] (Value-based).
pub struct DynRunnable<I, O>
where
    I: DeserializeOwned + Serialize + Send + 'static,
    O: Serialize + DeserializeOwned + Send + 'static,
{
    inner: Arc<dyn TypedRunnable<I, O>>,
}

impl<I, O> DynRunnable<I, O>
where
    I: DeserializeOwned + Serialize + Send + 'static,
    O: Serialize + DeserializeOwned + Send + 'static,
{
    /// Wrap a typed runnable for dynamic composition.
    pub fn new(inner: Arc<dyn TypedRunnable<I, O>>) -> Self {
        Self { inner }
    }
}

#[async_trait]
impl<I, O> Runnable for DynRunnable<I, O>
where
    I: DeserializeOwned + Serialize + Send + Sync + 'static,
    O: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let typed_input: I = serde_json::from_value(input)
            .map_err(|e| CognisError::Other(format!("input deserialization: {}", e)))?;
        let typed_output = self.inner.invoke(typed_input, config).await?;
        serde_json::to_value(typed_output)
            .map_err(|e| CognisError::Other(format!("output serialization: {}", e)))
    }
}

/// Wraps a [`Runnable`] as a [`TypedRunnable<I, O>`].
pub struct FromDynRunnable<I, O> {
    inner: Arc<dyn Runnable>,
    _phantom: std::marker::PhantomData<(I, O)>,
}

impl<I, O> FromDynRunnable<I, O> {
    /// Wrap a dynamic runnable for typed consumption.
    pub fn new(inner: Arc<dyn Runnable>) -> Self {
        Self {
            inner,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[async_trait]
impl<I, O> TypedRunnable<I, O> for FromDynRunnable<I, O>
where
    I: Serialize + Send + Sync + 'static,
    O: DeserializeOwned + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(&self, input: I, config: Option<&RunnableConfig>) -> Result<O> {
        let value_input = serde_json::to_value(input)
            .map_err(|e| CognisError::Other(format!("input serialization: {}", e)))?;
        let value_output = self.inner.invoke(value_input, config).await?;
        serde_json::from_value(value_output)
            .map_err(|e| CognisError::Other(format!("output deserialization: {}", e)))
    }
}

/// Composes two typed runnables in sequence.
pub struct TypedSequence<A, B, Mid> {
    first: Arc<dyn TypedRunnable<A, Mid>>,
    second: Arc<dyn TypedRunnable<Mid, B>>,
}

impl<A, B, Mid> TypedSequence<A, B, Mid>
where
    A: Serialize + Send + 'static,
    B: DeserializeOwned + Send + 'static,
    Mid: Serialize + DeserializeOwned + Send + 'static,
{
    /// Compose two typed runnables.
    pub fn new(
        first: Arc<dyn TypedRunnable<A, Mid>>,
        second: Arc<dyn TypedRunnable<Mid, B>>,
    ) -> Self {
        Self { first, second }
    }
}

#[async_trait]
impl<A, B, Mid> TypedRunnable<A, B> for TypedSequence<A, B, Mid>
where
    A: Serialize + Send + Sync + 'static,
    B: DeserializeOwned + Send + Sync + 'static,
    Mid: Serialize + DeserializeOwned + Send + Sync + 'static,
{
    fn name(&self) -> &str {
        "TypedSequence"
    }

    async fn invoke(&self, input: A, config: Option<&RunnableConfig>) -> Result<B> {
        let mid = self.first.invoke(input, config).await?;
        self.second.invoke(mid, config).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct AddOne;

    #[async_trait]
    impl TypedRunnable<i64, i64> for AddOne {
        fn name(&self) -> &str {
            "add_one"
        }
        async fn invoke(&self, input: i64, _config: Option<&RunnableConfig>) -> Result<i64> {
            Ok(input + 1)
        }
    }

    struct Double;

    #[async_trait]
    impl TypedRunnable<i64, i64> for Double {
        fn name(&self) -> &str {
            "double"
        }
        async fn invoke(&self, input: i64, _config: Option<&RunnableConfig>) -> Result<i64> {
            Ok(input * 2)
        }
    }

    struct IntToString;

    #[async_trait]
    impl TypedRunnable<i64, String> for IntToString {
        fn name(&self) -> &str {
            "to_string"
        }
        async fn invoke(&self, input: i64, _config: Option<&RunnableConfig>) -> Result<String> {
            Ok(format!("result: {}", input))
        }
    }

    #[tokio::test]
    async fn test_typed_invoke() {
        let result = AddOne.invoke(5, None).await.unwrap();
        assert_eq!(result, 6);
    }

    #[tokio::test]
    async fn test_typed_batch() {
        let results = AddOne.batch(vec![1, 2, 3], None).await.unwrap();
        assert_eq!(results, vec![2, 3, 4]);
    }

    #[tokio::test]
    async fn test_typed_sequence() {
        let seq = TypedSequence::new(
            Arc::new(AddOne) as Arc<dyn TypedRunnable<i64, i64>>,
            Arc::new(Double) as Arc<dyn TypedRunnable<i64, i64>>,
        );
        let result = seq.invoke(5, None).await.unwrap();
        assert_eq!(result, 12); // (5 + 1) * 2
    }

    #[tokio::test]
    async fn test_typed_sequence_mixed_types() {
        let seq = TypedSequence::new(
            Arc::new(Double) as Arc<dyn TypedRunnable<i64, i64>>,
            Arc::new(IntToString) as Arc<dyn TypedRunnable<i64, String>>,
        );
        let result = seq.invoke(7, None).await.unwrap();
        assert_eq!(result, "result: 14");
    }

    #[tokio::test]
    async fn test_dyn_runnable_bridge() {
        let typed: Arc<dyn TypedRunnable<i64, i64>> = Arc::new(AddOne);
        let dynamic: Arc<dyn Runnable> = Arc::new(DynRunnable::new(typed));
        let result = dynamic.invoke(json!(10), None).await.unwrap();
        assert_eq!(result, json!(11));
    }

    #[tokio::test]
    async fn test_from_dyn_runnable_bridge() {
        let dynamic = Arc::new(crate::runnables::lambda::RunnableLambda::new(
            "add_ten",
            |v: Value| async move {
                let n = v.as_i64().unwrap_or(0);
                Ok(json!(n + 10))
            },
        )) as Arc<dyn Runnable>;
        let typed: FromDynRunnable<i64, i64> = FromDynRunnable::new(dynamic);
        let result = typed.invoke(5, None).await.unwrap();
        assert_eq!(result, 15);
    }

    #[tokio::test]
    async fn test_dyn_runnable_bad_input_error() {
        let typed: Arc<dyn TypedRunnable<i64, i64>> = Arc::new(AddOne);
        let dynamic: Arc<dyn Runnable> = Arc::new(DynRunnable::new(typed));
        let result = dynamic.invoke(json!("not a number"), None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("deserialization"));
    }
}
