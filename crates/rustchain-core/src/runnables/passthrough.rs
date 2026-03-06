use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, RustChainError};

use super::base::Runnable;
use super::config::RunnableConfig;
use super::parallel::RunnableParallel;

type SideEffectFn = Box<
    dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<()>> + Send>> + Send + Sync,
>;

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

/// Runs a `RunnableParallel` and merges its output keys into the input object.
///
/// Input must be a `Value::Object`.
pub struct RunnableAssign {
    parallel: Arc<RunnableParallel>,
}

impl RunnableAssign {
    pub fn new(parallel: RunnableParallel) -> Self {
        Self {
            parallel: Arc::new(parallel),
        }
    }
}

#[async_trait]
impl Runnable for RunnableAssign {
    fn name(&self) -> &str {
        "RunnableAssign"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let obj = input.as_object().ok_or_else(|| RustChainError::TypeMismatch {
            expected: "Object".into(),
            got: value_type_name(&input).to_string(),
        })?;

        let parallel_output = self.parallel.invoke(input.clone(), config).await?;
        let parallel_obj = parallel_output
            .as_object()
            .ok_or_else(|| RustChainError::Other("RunnableParallel did not return Object".into()))?;

        let mut merged = obj.clone();
        for (k, v) in parallel_obj {
            merged.insert(k.clone(), v.clone());
        }

        Ok(Value::Object(merged))
    }
}

/// Selects keys from a dict input.
///
/// When constructed with a single key, returns the value directly.
/// When constructed with multiple keys, returns a dict with only those keys.
pub struct RunnablePick {
    keys: Vec<String>,
}

impl RunnablePick {
    /// Create a `RunnablePick` that selects a single key (returns value directly).
    pub fn one(key: impl Into<String>) -> Self {
        Self {
            keys: vec![key.into()],
        }
    }

    /// Create a `RunnablePick` that selects multiple keys (returns a dict).
    pub fn many(keys: Vec<String>) -> Self {
        Self { keys }
    }
}

#[async_trait]
impl Runnable for RunnablePick {
    fn name(&self) -> &str {
        "RunnablePick"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        let obj = input.as_object().ok_or_else(|| RustChainError::TypeMismatch {
            expected: "Object".into(),
            got: value_type_name(&input).to_string(),
        })?;

        if self.keys.len() == 1 {
            Ok(obj
                .get(&self.keys[0])
                .cloned()
                .unwrap_or(Value::Null))
        } else {
            let mut result = serde_json::Map::new();
            for key in &self.keys {
                if let Some(val) = obj.get(key) {
                    result.insert(key.clone(), val.clone());
                }
            }
            Ok(Value::Object(result))
        }
    }
}

fn value_type_name(v: &Value) -> &'static str {
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
