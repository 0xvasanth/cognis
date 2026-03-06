use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;

use super::base::Runnable;
use super::config::RunnableConfig;
use super::RunnableStream;

/// Tries the primary runnable, falling back to alternatives on error.
///
/// On invocation, the primary runnable is tried first. If it fails, each
/// fallback is tried in order. If all fallbacks also fail, the primary
/// error is returned (preserving the original failure context).
///
/// # Example
/// ```ignore
/// use rustchain_core::runnables::{RunnableLambda, RunnableExt};
///
/// let chain = primary_runnable.with_fallbacks(vec![
///     Arc::new(fallback1) as Arc<dyn Runnable>,
///     Arc::new(fallback2) as Arc<dyn Runnable>,
/// ]);
/// ```
pub struct RunnableWithFallbacks {
    /// The primary runnable to try first.
    primary: Arc<dyn Runnable>,
    /// Fallbacks tried in order if the primary fails.
    fallbacks: Vec<Arc<dyn Runnable>>,
}

impl RunnableWithFallbacks {
    /// Create a new `RunnableWithFallbacks` with a primary runnable and a list of fallbacks.
    pub fn new(primary: Arc<dyn Runnable>, fallbacks: Vec<Arc<dyn Runnable>>) -> Self {
        Self { primary, fallbacks }
    }
}

#[async_trait]
impl Runnable for RunnableWithFallbacks {
    fn name(&self) -> &str {
        "RunnableWithFallbacks"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        match self.primary.invoke(input.clone(), config).await {
            Ok(result) => Ok(result),
            Err(primary_err) => {
                for fallback in &self.fallbacks {
                    match fallback.invoke(input.clone(), config).await {
                        Ok(result) => return Ok(result),
                        Err(_) => continue,
                    }
                }
                // Return the original primary error if all fallbacks fail.
                Err(primary_err)
            }
        }
    }

    /// Batch invocation: for each input, try primary then fallbacks.
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

    /// Stream invocation: try primary stream, fall back on failure.
    async fn stream(
        &self,
        input: Value,
        config: Option<&RunnableConfig>,
    ) -> Result<RunnableStream> {
        match self.primary.stream(input.clone(), config).await {
            Ok(stream) => Ok(stream),
            Err(primary_err) => {
                for fallback in &self.fallbacks {
                    match fallback.stream(input.clone(), config).await {
                        Ok(stream) => return Ok(stream),
                        Err(_) => continue,
                    }
                }
                Err(primary_err)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RustChainError;

    /// A test runnable that always succeeds, returning its input.
    struct Succeeds {
        label: &'static str,
    }

    impl Succeeds {
        fn new(label: &'static str) -> Self {
            Self { label }
        }
    }

    #[async_trait]
    impl Runnable for Succeeds {
        fn name(&self) -> &str {
            self.label
        }

        async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
            // Return the input tagged with our label so tests can verify which runnable ran.
            Ok(serde_json::json!({ "from": self.label, "value": input }))
        }
    }

    /// A test runnable that always fails with a given message.
    struct Fails {
        message: String,
    }

    impl Fails {
        fn new(msg: &str) -> Self {
            Self {
                message: msg.to_string(),
            }
        }
    }

    #[async_trait]
    impl Runnable for Fails {
        fn name(&self) -> &str {
            "Fails"
        }

        async fn invoke(&self, _input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
            Err(RustChainError::Other(self.message.clone()))
        }
    }

    #[tokio::test]
    async fn test_fallback_primary_succeeds() {
        let primary = Arc::new(Succeeds::new("primary")) as Arc<dyn Runnable>;
        let fallback = Arc::new(Succeeds::new("fallback1")) as Arc<dyn Runnable>;
        let chain = RunnableWithFallbacks::new(primary, vec![fallback]);

        let result = chain.invoke(serde_json::json!("data"), None).await;
        assert!(result.is_ok());
        let val = result.unwrap();
        assert_eq!(val["from"], "primary");
    }

    #[tokio::test]
    async fn test_fallback_to_first() {
        let primary = Arc::new(Fails::new("primary failed")) as Arc<dyn Runnable>;
        let fallback1 = Arc::new(Succeeds::new("fallback1")) as Arc<dyn Runnable>;
        let fallback2 = Arc::new(Succeeds::new("fallback2")) as Arc<dyn Runnable>;
        let chain = RunnableWithFallbacks::new(primary, vec![fallback1, fallback2]);

        let result = chain.invoke(serde_json::json!("data"), None).await;
        assert!(result.is_ok());
        let val = result.unwrap();
        assert_eq!(val["from"], "fallback1");
    }

    #[tokio::test]
    async fn test_fallback_to_second() {
        let primary = Arc::new(Fails::new("primary failed")) as Arc<dyn Runnable>;
        let fallback1 = Arc::new(Fails::new("fallback1 failed")) as Arc<dyn Runnable>;
        let fallback2 = Arc::new(Succeeds::new("fallback2")) as Arc<dyn Runnable>;
        let chain = RunnableWithFallbacks::new(primary, vec![fallback1, fallback2]);

        let result = chain.invoke(serde_json::json!("data"), None).await;
        assert!(result.is_ok());
        let val = result.unwrap();
        assert_eq!(val["from"], "fallback2");
    }

    #[tokio::test]
    async fn test_fallback_all_fail() {
        let primary = Arc::new(Fails::new("primary failed")) as Arc<dyn Runnable>;
        let fallback1 = Arc::new(Fails::new("fallback1 failed")) as Arc<dyn Runnable>;
        let fallback2 = Arc::new(Fails::new("fallback2 failed")) as Arc<dyn Runnable>;
        let chain = RunnableWithFallbacks::new(primary, vec![fallback1, fallback2]);

        let result = chain.invoke(serde_json::json!("data"), None).await;
        assert!(result.is_err());
        // Should return the PRIMARY error, not a fallback error.
        let err = format!("{}", result.unwrap_err());
        assert!(
            err.contains("primary failed"),
            "Expected primary error, got: {}",
            err
        );
    }
}
