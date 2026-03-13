use std::collections::HashMap;
use std::future::Future;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{Map, Value};
use tokio::task::JoinSet;

use crate::error::{Result, CognisError};

use super::base::Runnable;
use super::config::{ensure_config, RunnableConfig};
use super::lambda::RunnableLambda;

/// Computes new fields in parallel and merges them into the input object.
///
/// Each assignment maps a field name to a `Runnable` that receives the full
/// input and produces the value for that field. The original input fields are
/// preserved; assignment fields are added or overwritten.
///
/// Input **must** be a JSON object; otherwise an error is returned.
///
/// # Example
/// ```ignore
/// let assign = RunnableAssign::new()
///     .assign("upper", Arc::new(RunnableLambda::new("upper", |v| async move {
///         let s = v["name"].as_str().unwrap().to_uppercase();
///         Ok(Value::String(s))
///     })))
///     .assign_value("version", json!(2));
/// ```
pub struct RunnableAssign {
    assignments: HashMap<String, Arc<dyn Runnable>>,
}

impl RunnableAssign {
    /// Create an empty `RunnableAssign`.
    pub fn new() -> Self {
        Self {
            assignments: HashMap::new(),
        }
    }

    /// Add an assignment that computes a field value using a `Runnable`.
    pub fn assign(mut self, field_name: impl Into<String>, runnable: Arc<dyn Runnable>) -> Self {
        self.assignments.insert(field_name.into(), runnable);
        self
    }

    /// Add an assignment that sets a field to a static JSON value.
    pub fn assign_value(self, field_name: impl Into<String>, value: Value) -> Self {
        let v = value.clone();
        let lambda = RunnableLambda::new("static_value", move |_input: Value| {
            let v = v.clone();
            async move { Ok(v) }
        });
        self.assign(field_name, Arc::new(lambda))
    }

    /// Add an assignment that computes a field value from a closure.
    ///
    /// The closure receives the full input `Value` and must return a `Result<Value>`.
    pub fn assign_fn<F, Fut>(self, field_name: impl Into<String>, f: F) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        let lambda = RunnableLambda::new("assign_fn", f);
        self.assign(field_name, Arc::new(lambda))
    }
}

impl Default for RunnableAssign {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Runnable for RunnableAssign {
    fn name(&self) -> &str {
        // We return a static string here; for dynamic naming see `display_name`.
        "RunnableAssign"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        let obj = input
            .as_object()
            .ok_or_else(|| CognisError::TypeMismatch {
                expected: "Object".into(),
                got: value_type_name(&input).to_string(),
            })?;

        let cfg = ensure_config(config);
        let mut join_set = JoinSet::new();

        for (key, runnable) in &self.assignments {
            let key = key.clone();
            let runnable = Arc::clone(runnable);
            let input = input.clone();
            let cfg = cfg.clone();

            join_set.spawn(async move {
                let result = runnable.invoke(input, Some(&cfg)).await;
                (key, result)
            });
        }

        let mut merged = obj.clone();
        while let Some(result) = join_set.join_next().await {
            let (key, value) = result.map_err(|e| CognisError::Other(e.to_string()))?;
            merged.insert(key, value?);
        }

        Ok(Value::Object(merged))
    }
}

/// Keeps only the specified fields from a JSON object input, dropping all others.
///
/// When constructed with a single key via [`RunnablePick::one`], returns the
/// value directly (not wrapped in an object). When constructed with multiple
/// keys via [`RunnablePick::many`], returns an object with only those keys.
///
/// Useful in pipelines: `input | assign(extra) | pick(needed)`.
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

    /// Create a `RunnablePick` that selects multiple keys (returns an object).
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
        let obj = input
            .as_object()
            .ok_or_else(|| CognisError::TypeMismatch {
                expected: "Object".into(),
                got: value_type_name(&input).to_string(),
            })?;

        if self.keys.len() == 1 {
            Ok(obj.get(&self.keys[0]).cloned().unwrap_or(Value::Null))
        } else {
            let mut result = Map::new();
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

    // ── RunnableAssign tests ──────────────────────────────────────────

    #[tokio::test]
    async fn test_assign_adds_new_field() {
        let assign = RunnableAssign::new().assign_fn("greeting", |v: Value| async move {
            let name = v["name"].as_str().unwrap().to_string();
            Ok(Value::String(format!("Hello, {}!", name)))
        });

        let input = json!({"name": "Alice"});
        let result = assign.invoke(input, None).await.unwrap();
        assert_eq!(result["name"], json!("Alice"));
        assert_eq!(result["greeting"], json!("Hello, Alice!"));
    }

    #[tokio::test]
    async fn test_assign_multiple_assignments_parallel() {
        let assign = RunnableAssign::new()
            .assign_fn("upper", |v: Value| async move {
                let s = v["name"].as_str().unwrap().to_uppercase();
                Ok(Value::String(s))
            })
            .assign_fn("lower", |v: Value| async move {
                let s = v["name"].as_str().unwrap().to_lowercase();
                Ok(Value::String(s))
            });

        let input = json!({"name": "Alice"});
        let result = assign.invoke(input, None).await.unwrap();
        assert_eq!(result["name"], json!("Alice"));
        assert_eq!(result["upper"], json!("ALICE"));
        assert_eq!(result["lower"], json!("alice"));
    }

    #[tokio::test]
    async fn test_assign_preserves_original_fields() {
        let assign = RunnableAssign::new().assign_value("extra", json!(42));

        let input = json!({"a": 1, "b": 2, "c": 3});
        let result = assign.invoke(input, None).await.unwrap();
        assert_eq!(result["a"], json!(1));
        assert_eq!(result["b"], json!(2));
        assert_eq!(result["c"], json!(3));
        assert_eq!(result["extra"], json!(42));
    }

    #[tokio::test]
    async fn test_assign_overwrites_existing_field() {
        let assign = RunnableAssign::new().assign_value("name", json!("Bob"));

        let input = json!({"name": "Alice", "age": 30});
        let result = assign.invoke(input, None).await.unwrap();
        assert_eq!(result["name"], json!("Bob"));
        assert_eq!(result["age"], json!(30));
    }

    #[tokio::test]
    async fn test_assign_non_object_input_errors() {
        let assign = RunnableAssign::new().assign_value("key", json!("val"));

        let result = assign.invoke(json!("not an object"), None).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("expected Object"));
    }

    #[tokio::test]
    async fn test_assign_value_static() {
        let assign = RunnableAssign::new()
            .assign_value("pi", json!(3.14))
            .assign_value("tags", json!(["a", "b"]));

        let input = json!({"id": 1});
        let result = assign.invoke(input, None).await.unwrap();
        assert_eq!(result["id"], json!(1));
        assert_eq!(result["pi"], json!(3.14));
        assert_eq!(result["tags"], json!(["a", "b"]));
    }

    #[tokio::test]
    async fn test_assign_fn_computed() {
        let assign = RunnableAssign::new().assign_fn("doubled", |v: Value| async move {
            let n = v["value"].as_i64().unwrap();
            Ok(json!(n * 2))
        });

        let input = json!({"value": 21});
        let result = assign.invoke(input, None).await.unwrap();
        assert_eq!(result["value"], json!(21));
        assert_eq!(result["doubled"], json!(42));
    }

    #[tokio::test]
    async fn test_assign_config_passed_through() {
        let lambda = RunnableLambda::with_config("config_check", |_input, config| async move {
            let has_config = config.is_some();
            Ok(json!(has_config))
        });

        let assign = RunnableAssign::new().assign("has_config", Arc::new(lambda));

        let cfg = RunnableConfig::default();
        let input = json!({"x": 1});
        let result = assign.invoke(input, Some(&cfg)).await.unwrap();
        assert_eq!(result["has_config"], json!(true));
    }

    #[tokio::test]
    async fn test_assign_nested() {
        // Inner assign adds "b", outer assign adds "c" based on input (which includes "a")
        let inner = RunnableAssign::new().assign_value("b", json!(2));
        let outer = RunnableAssign::new()
            .assign("inner_result", Arc::new(inner))
            .assign_value("c", json!(3));

        // The outer assign passes the original input to each assignment.
        // The inner assignment (RunnableAssign) will receive {"a": 1} and return {"a":1, "b":2}
        let input = json!({"a": 1});
        let result = outer.invoke(input, None).await.unwrap();
        assert_eq!(result["a"], json!(1));
        assert_eq!(result["c"], json!(3));
        // inner_result is the output of inner assign: {"a":1, "b":2}
        assert_eq!(result["inner_result"]["a"], json!(1));
        assert_eq!(result["inner_result"]["b"], json!(2));
    }

    // ── RunnablePick tests ────────────────────────────────────────────

    #[tokio::test]
    async fn test_pick_keeps_specified_fields() {
        let pick = RunnablePick::many(vec!["name".into(), "age".into()]);
        let input = json!({"name": "Alice", "age": 30, "city": "NYC", "country": "US"});
        let result = pick.invoke(input, None).await.unwrap();
        assert_eq!(result, json!({"name": "Alice", "age": 30}));
    }

    #[tokio::test]
    async fn test_pick_nonexistent_fields_returns_empty() {
        let pick = RunnablePick::many(vec!["missing1".into(), "missing2".into()]);
        let input = json!({"name": "Alice"});
        let result = pick.invoke(input, None).await.unwrap();
        assert_eq!(result, json!({}));
    }

    #[tokio::test]
    async fn test_pick_non_object_errors() {
        let pick = RunnablePick::one("key");
        let result = pick.invoke(json!(42), None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_assign_then_pick_pipeline() {
        // Simulate a pipeline: assign adds fields, then pick selects subset
        let assign = RunnableAssign::new()
            .assign_fn("full_name", |v: Value| async move {
                let first = v["first"].as_str().unwrap();
                let last = v["last"].as_str().unwrap();
                Ok(Value::String(format!("{} {}", first, last)))
            })
            .assign_value("processed", json!(true));

        let input = json!({"first": "John", "last": "Doe", "age": 42});
        let assigned = assign.invoke(input, None).await.unwrap();

        let pick = RunnablePick::many(vec!["full_name".into(), "processed".into()]);
        let result = pick.invoke(assigned, None).await.unwrap();

        assert_eq!(result, json!({"full_name": "John Doe", "processed": true}));
        // Original fields should NOT be present
        assert!(result.get("first").is_none());
        assert!(result.get("last").is_none());
        assert!(result.get("age").is_none());
    }
}
