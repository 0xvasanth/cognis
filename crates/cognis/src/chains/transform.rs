use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use regex::Regex;
use serde_json::Value;

use cognis_core::error::{Result, CognisError};
use cognis_core::runnables::base::Runnable;
use cognis_core::runnables::config::RunnableConfig;

/// A synchronous transform function that maps a JSON value to another.
pub type TransformFn = Box<dyn Fn(Value) -> Result<Value> + Send + Sync>;

/// An asynchronous transform function that maps a JSON value to another.
pub type AsyncTransformFn =
    Box<dyn Fn(Value) -> Pin<Box<dyn Future<Output = Result<Value>> + Send>> + Send + Sync>;

/// A chain that applies an arbitrary synchronous transformation to its input.
///
/// Useful for data manipulation steps within a chain pipeline (e.g., extracting
/// fields, reformatting strings, filtering keys).
pub struct TransformChain {
    name: String,
    transform: TransformFn,
    input_keys: Vec<String>,
    output_keys: Vec<String>,
}

/// Builder for [`TransformChain`].
pub struct TransformChainBuilder {
    name: String,
    transform: Option<TransformFn>,
    input_keys: Vec<String>,
    output_keys: Vec<String>,
}

impl TransformChainBuilder {
    /// Create a new builder with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            transform: None,
            input_keys: Vec::new(),
            output_keys: Vec::new(),
        }
    }

    /// Set the transform function.
    pub fn transform(mut self, f: impl Fn(Value) -> Result<Value> + Send + Sync + 'static) -> Self {
        self.transform = Some(Box::new(f));
        self
    }

    /// Set the expected input keys.
    pub fn input_keys(mut self, keys: Vec<String>) -> Self {
        self.input_keys = keys;
        self
    }

    /// Set the output keys.
    pub fn output_keys(mut self, keys: Vec<String>) -> Self {
        self.output_keys = keys;
        self
    }

    /// Build the [`TransformChain`].
    ///
    /// # Panics
    /// Panics if no transform function was provided.
    pub fn build(self) -> TransformChain {
        TransformChain {
            name: self.name,
            transform: self.transform.expect("transform function is required"),
            input_keys: self.input_keys,
            output_keys: self.output_keys,
        }
    }
}

impl TransformChain {
    /// Create a new builder.
    pub fn builder(name: impl Into<String>) -> TransformChainBuilder {
        TransformChainBuilder::new(name)
    }

    /// Create a `TransformChain` directly from a name and function.
    pub fn new(
        name: impl Into<String>,
        f: impl Fn(Value) -> Result<Value> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            transform: Box::new(f),
            input_keys: Vec::new(),
            output_keys: Vec::new(),
        }
    }

    /// Returns the configured input keys.
    pub fn input_keys(&self) -> &[String] {
        &self.input_keys
    }

    /// Returns the configured output keys.
    pub fn output_keys(&self) -> &[String] {
        &self.output_keys
    }
}

#[async_trait]
impl Runnable for TransformChain {
    fn name(&self) -> &str {
        &self.name
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        (self.transform)(input)
    }
}

/// A chain that applies an arbitrary asynchronous transformation to its input.
pub struct AsyncTransformChain {
    name: String,
    transform: AsyncTransformFn,
    input_keys: Vec<String>,
    output_keys: Vec<String>,
}

impl AsyncTransformChain {
    /// Create a new `AsyncTransformChain` from a name and async function.
    pub fn new<F, Fut>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        Self {
            name: name.into(),
            transform: Box::new(move |v| Box::pin(f(v))),
            input_keys: Vec::new(),
            output_keys: Vec::new(),
        }
    }

    /// Set the expected input keys.
    pub fn with_input_keys(mut self, keys: Vec<String>) -> Self {
        self.input_keys = keys;
        self
    }

    /// Set the output keys.
    pub fn with_output_keys(mut self, keys: Vec<String>) -> Self {
        self.output_keys = keys;
        self
    }

    /// Returns the configured input keys.
    pub fn input_keys(&self) -> &[String] {
        &self.input_keys
    }

    /// Returns the configured output keys.
    pub fn output_keys(&self) -> &[String] {
        &self.output_keys
    }
}

#[async_trait]
impl Runnable for AsyncTransformChain {
    fn name(&self) -> &str {
        &self.name
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        (self.transform)(input).await
    }
}

// ---------------------------------------------------------------------------
// Built-in transform factories
// ---------------------------------------------------------------------------

/// Apply a function to every string value in the JSON tree (recursively).
fn map_strings(value: Value, f: &dyn Fn(&str) -> String) -> Value {
    match value {
        Value::String(s) => Value::String(f(&s)),
        Value::Array(arr) => Value::Array(arr.into_iter().map(|v| map_strings(v, f)).collect()),
        Value::Object(map) => {
            let mapped = map
                .into_iter()
                .map(|(k, v)| (k, map_strings(v, f)))
                .collect();
            Value::Object(mapped)
        }
        other => other,
    }
}

/// Creates a transform that uppercases all string values in the input.
pub fn uppercase_transform() -> TransformChain {
    TransformChain::new("uppercase", |v| Ok(map_strings(v, &|s| s.to_uppercase())))
}

/// Creates a transform that lowercases all string values in the input.
pub fn lowercase_transform() -> TransformChain {
    TransformChain::new("lowercase", |v| Ok(map_strings(v, &|s| s.to_lowercase())))
}

/// Creates a transform that trims whitespace from all string values in the input.
pub fn trim_transform() -> TransformChain {
    TransformChain::new("trim", |v| Ok(map_strings(v, &|s| s.trim().to_string())))
}

/// Creates a transform that extracts a specific key from a JSON object.
///
/// If the input is a JSON object, this returns the value at `key`.
/// Returns an error if the key is not found.
pub fn json_extract_transform(key: &str) -> TransformChain {
    let key = key.to_string();
    TransformChain::new("json_extract", move |v| {
        match &v {
            Value::Object(map) => match map.get(&key) {
                Some(val) => Ok(val.clone()),
                None => Err(CognisError::Other(format!(
                    "key '{}' not found in object",
                    key
                ))),
            },
            // If the input is a JSON string, try to parse it first
            Value::String(s) => {
                let parsed: Value = serde_json::from_str(s).map_err(|e| {
                    CognisError::Other(format!("failed to parse JSON string: {}", e))
                })?;
                match parsed.get(&key) {
                    Some(val) => Ok(val.clone()),
                    None => Err(CognisError::Other(format!(
                        "key '{}' not found in parsed object",
                        key
                    ))),
                }
            }
            _ => Err(CognisError::Other(
                "json_extract requires an object or JSON string input".to_string(),
            )),
        }
    })
}

/// Creates a transform that applies a regex replacement to all string values.
///
/// # Panics
/// Panics if the provided pattern is not a valid regex.
pub fn regex_replace_transform(pattern: &str, replacement: &str) -> TransformChain {
    let re = Regex::new(pattern).expect("invalid regex pattern");
    let replacement = replacement.to_string();
    TransformChain::new("regex_replace", move |v| {
        Ok(map_strings(v, &|s| {
            re.replace_all(s, &*replacement).into_owned()
        }))
    })
}

/// Creates a transform that applies a custom function to the value at a specific key.
///
/// The function receives the string value at `key` and must return a new string.
/// Other keys in the object are passed through unchanged.
pub fn map_transform(
    key: &str,
    f: impl Fn(&str) -> String + Send + Sync + 'static,
) -> TransformChain {
    let key = key.to_string();
    TransformChain::new("map_key", move |v| match v {
        Value::Object(mut map) => {
            if let Some(val) = map.get(&key) {
                let s = val.as_str().ok_or_else(|| {
                    CognisError::Other(format!("value at key '{}' is not a string", key))
                })?;
                map.insert(key.clone(), Value::String(f(s)));
            }
            Ok(Value::Object(map))
        }
        _ => Err(CognisError::Other(
            "map_transform requires an object input".to_string(),
        )),
    })
}

// ---------------------------------------------------------------------------
// TransformPipeline
// ---------------------------------------------------------------------------

/// Chains multiple [`TransformChain`]s together, executing them in sequence.
///
/// Each transform receives the output of the previous one.
pub struct TransformPipeline {
    transforms: Vec<Arc<dyn Runnable>>,
}

impl TransformPipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Add a transform to the pipeline. Returns `self` for chaining.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, transform: TransformChain) -> Self {
        self.transforms.push(Arc::new(transform));
        self
    }

    /// Add any `Runnable` to the pipeline.
    pub fn add_runnable(mut self, runnable: Arc<dyn Runnable>) -> Self {
        self.transforms.push(runnable);
        self
    }

    /// Execute all transforms in sequence, returning the final result.
    pub async fn execute(&self, input: Value) -> Result<Value> {
        let mut current = input;
        for t in &self.transforms {
            current = t.invoke(current, None).await?;
        }
        Ok(current)
    }

    /// Returns the number of transforms in the pipeline.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Returns true if the pipeline has no transforms.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl Default for TransformPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Runnable for TransformPipeline {
    fn name(&self) -> &str {
        "TransformPipeline"
    }

    async fn invoke(&self, input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
        self.execute(input).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // TransformChain basic tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_transform_chain_identity() {
        let chain = TransformChain::new("identity", |v| Ok(v));
        let input = json!({"hello": "world"});
        let result = chain.invoke(input.clone(), None).await.unwrap();
        assert_eq!(result, input);
    }

    #[tokio::test]
    async fn test_transform_chain_name() {
        let chain = TransformChain::new("my_transform", |v| Ok(v));
        assert_eq!(chain.name(), "my_transform");
    }

    #[tokio::test]
    async fn test_transform_chain_builder() {
        let chain = TransformChain::builder("builder_test")
            .transform(|v| Ok(v))
            .input_keys(vec!["in".to_string()])
            .output_keys(vec!["out".to_string()])
            .build();
        assert_eq!(chain.name(), "builder_test");
        assert_eq!(chain.input_keys(), &["in".to_string()]);
        assert_eq!(chain.output_keys(), &["out".to_string()]);
    }

    #[tokio::test]
    async fn test_transform_chain_error_propagation() {
        let chain = TransformChain::new("fail", |_| {
            Err(CognisError::Other("intentional error".to_string()))
        });
        let result = chain.invoke(json!(null), None).await;
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(err_msg.contains("intentional error"));
    }

    // -----------------------------------------------------------------------
    // Built-in transforms
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_uppercase_transform() {
        let chain = uppercase_transform();
        let result = chain
            .invoke(json!({"text": "hello world"}), None)
            .await
            .unwrap();
        assert_eq!(result, json!({"text": "HELLO WORLD"}));
    }

    #[tokio::test]
    async fn test_uppercase_transform_nested() {
        let chain = uppercase_transform();
        let result = chain
            .invoke(json!({"a": "foo", "b": ["bar", "baz"]}), None)
            .await
            .unwrap();
        assert_eq!(result, json!({"a": "FOO", "b": ["BAR", "BAZ"]}));
    }

    #[tokio::test]
    async fn test_lowercase_transform() {
        let chain = lowercase_transform();
        let result = chain.invoke(json!("HELLO WORLD"), None).await.unwrap();
        assert_eq!(result, json!("hello world"));
    }

    #[tokio::test]
    async fn test_trim_transform() {
        let chain = trim_transform();
        let result = chain
            .invoke(json!({"text": "  hello  "}), None)
            .await
            .unwrap();
        assert_eq!(result, json!({"text": "hello"}));
    }

    #[tokio::test]
    async fn test_trim_transform_preserves_non_strings() {
        let chain = trim_transform();
        let result = chain
            .invoke(json!({"num": 42, "text": " hi "}), None)
            .await
            .unwrap();
        assert_eq!(result["num"], 42);
        assert_eq!(result["text"], "hi");
    }

    #[tokio::test]
    async fn test_json_extract_transform() {
        let chain = json_extract_transform("name");
        let result = chain
            .invoke(json!({"name": "Alice", "age": 30}), None)
            .await
            .unwrap();
        assert_eq!(result, json!("Alice"));
    }

    #[tokio::test]
    async fn test_json_extract_transform_missing_key() {
        let chain = json_extract_transform("missing");
        let result = chain.invoke(json!({"name": "Alice"}), None).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_json_extract_transform_from_string() {
        let chain = json_extract_transform("key");
        let result = chain
            .invoke(json!("{\"key\": \"value\"}"), None)
            .await
            .unwrap();
        assert_eq!(result, json!("value"));
    }

    #[tokio::test]
    async fn test_regex_replace_transform() {
        let chain = regex_replace_transform(r"\d+", "NUM");
        let result = chain
            .invoke(json!({"text": "order 123 item 456"}), None)
            .await
            .unwrap();
        assert_eq!(result, json!({"text": "order NUM item NUM"}));
    }

    #[tokio::test]
    async fn test_map_transform() {
        let chain = map_transform("name", |s| format!("Dr. {}", s));
        let result = chain
            .invoke(json!({"name": "Smith", "age": 40}), None)
            .await
            .unwrap();
        assert_eq!(result["name"], "Dr. Smith");
        assert_eq!(result["age"], 40);
    }

    #[tokio::test]
    async fn test_map_transform_non_object_error() {
        let chain = map_transform("key", |s| s.to_uppercase());
        let result = chain.invoke(json!("not an object"), None).await;
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // AsyncTransformChain
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_async_transform_chain() {
        let chain = AsyncTransformChain::new("async_upper", |v| async move {
            Ok(map_strings(v, &|s| s.to_uppercase()))
        });
        let result = chain.invoke(json!({"msg": "hello"}), None).await.unwrap();
        assert_eq!(result, json!({"msg": "HELLO"}));
    }

    #[tokio::test]
    async fn test_async_transform_chain_with_keys() {
        let chain = AsyncTransformChain::new("async_id", |v| async move { Ok(v) })
            .with_input_keys(vec!["in".to_string()])
            .with_output_keys(vec!["out".to_string()]);
        assert_eq!(chain.name(), "async_id");
        assert_eq!(chain.input_keys(), &["in".to_string()]);
        assert_eq!(chain.output_keys(), &["out".to_string()]);
    }

    // -----------------------------------------------------------------------
    // TransformPipeline
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_pipeline_empty() {
        let pipeline = TransformPipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
        let result = pipeline.execute(json!({"x": 1})).await.unwrap();
        assert_eq!(result, json!({"x": 1}));
    }

    #[tokio::test]
    async fn test_pipeline_single_transform() {
        let pipeline = TransformPipeline::new().add(uppercase_transform());
        assert_eq!(pipeline.len(), 1);
        let result = pipeline.execute(json!({"text": "hello"})).await.unwrap();
        assert_eq!(result, json!({"text": "HELLO"}));
    }

    #[tokio::test]
    async fn test_pipeline_multiple_transforms() {
        let pipeline = TransformPipeline::new()
            .add(trim_transform())
            .add(uppercase_transform());
        let result = pipeline
            .execute(json!({"text": "  hello  "}))
            .await
            .unwrap();
        assert_eq!(result, json!({"text": "HELLO"}));
    }

    #[tokio::test]
    async fn test_pipeline_as_runnable() {
        let pipeline = TransformPipeline::new().add(lowercase_transform());
        let runnable: &dyn Runnable = &pipeline;
        assert_eq!(runnable.name(), "TransformPipeline");
        let result = runnable.invoke(json!("HELLO"), None).await.unwrap();
        assert_eq!(result, json!("hello"));
    }

    #[tokio::test]
    async fn test_pipeline_error_stops_execution() {
        let pipeline = TransformPipeline::new()
            .add(TransformChain::new("fail", |_| {
                Err(CognisError::Other("stop".to_string()))
            }))
            .add(uppercase_transform());
        let result = pipeline.execute(json!("anything")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_pipeline_complex_chain() {
        // trim -> regex replace digits -> uppercase
        let pipeline = TransformPipeline::new()
            .add(trim_transform())
            .add(regex_replace_transform(r"\d+", "N"))
            .add(uppercase_transform());
        let result = pipeline
            .execute(json!({"msg": "  order 42  "}))
            .await
            .unwrap();
        assert_eq!(result, json!({"msg": "ORDER N"}));
    }
}
