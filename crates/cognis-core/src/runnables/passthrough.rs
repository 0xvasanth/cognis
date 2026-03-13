use std::future::Future;
use std::pin::Pin;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::Result;

use super::base::Runnable;
use super::config::RunnableConfig;

type SideEffectFn =
    Box<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync>;

/// Returns the input unchanged. Optionally runs a side-effect function.
pub struct RunnablePassthrough {
    side_effect: Option<SideEffectFn>,
}

impl RunnablePassthrough {
    pub fn new() -> Self {
        Self { side_effect: None }
    }

    /// Create with an async side-effect that receives a clone of the input.
    pub fn with_side_effect<F, Fut>(f: F) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static,
    {
        Self {
            side_effect: Some(Box::new(move |v| Box::pin(f(v)))),
        }
    }
}

impl Default for RunnablePassthrough {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Runnable for RunnablePassthrough {
    fn name(&self) -> &str {
        "RunnablePassthrough"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        if let Some(ref side_effect) = self.side_effect {
            side_effect(input.clone()).await?;
        }
        Ok(input)
    }
}

// RunnableAssign and RunnablePick have been moved to the `assign` module.
// Re-export them here for backward compatibility.
pub use super::assign::{RunnableAssign, RunnablePick};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_passthrough_returns_input() {
        let p = RunnablePassthrough::new();
        let input = json!({"key": "value"});
        let result = p.invoke(input.clone(), None).await.unwrap();
        assert_eq!(result, input);
    }

    #[tokio::test]
    async fn test_passthrough_with_side_effect() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let called = Arc::new(AtomicBool::new(false));
        let called_clone = called.clone();
        let p = RunnablePassthrough::with_side_effect(move |_| {
            let c = called_clone.clone();
            async move {
                c.store(true, Ordering::SeqCst);
                Ok(())
            }
        });
        let result = p.invoke(json!("hello"), None).await.unwrap();
        assert_eq!(result, json!("hello"));
        assert!(called.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_pick_single_key() {
        let pick = RunnablePick::one("name");
        let input = json!({"name": "John", "age": 30, "city": "NYC"});
        let result = pick.invoke(input, None).await.unwrap();
        assert_eq!(result, json!("John"));
    }

    #[tokio::test]
    async fn test_pick_single_key_missing() {
        let pick = RunnablePick::one("missing");
        let input = json!({"name": "John"});
        let result = pick.invoke(input, None).await.unwrap();
        assert_eq!(result, Value::Null);
    }

    #[tokio::test]
    async fn test_pick_multiple_keys() {
        let pick = RunnablePick::many(vec!["name".into(), "age".into()]);
        let input = json!({"name": "John", "age": 30, "city": "NYC"});
        let result = pick.invoke(input, None).await.unwrap();
        assert_eq!(result, json!({"name": "John", "age": 30}));
    }

    #[tokio::test]
    async fn test_pick_multiple_keys_partial() {
        let pick = RunnablePick::many(vec!["name".into(), "missing".into()]);
        let input = json!({"name": "John", "age": 30});
        let result = pick.invoke(input, None).await.unwrap();
        assert_eq!(result, json!({"name": "John"}));
    }

    #[tokio::test]
    async fn test_pick_non_object_errors() {
        let pick = RunnablePick::one("key");
        let result = pick.invoke(json!("not an object"), None).await;
        assert!(result.is_err());
    }
}
