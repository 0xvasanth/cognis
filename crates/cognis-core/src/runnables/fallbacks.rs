use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::error::{Result, CognisError};

use super::base::Runnable;
use super::config::RunnableConfig;
use super::RunnableStream;

/// Tries the primary runnable, falling back to alternatives on error.
///
/// On invocation, the primary runnable is tried first. If it fails, each
/// fallback is tried in order. If all fallbacks also fail, the last error
/// is returned.
///
/// An optional `exceptions_to_handle` filter controls which error types
/// trigger fallback behaviour. If the filter is set and the error does not
/// match any of the listed exception names, it is propagated immediately
/// without trying fallbacks.
///
/// # Example
/// ```ignore
/// use cognis_core::runnables::{RunnableLambda, RunnableExt};
///
/// let chain = RunnableWithFallbacks::new(Arc::new(primary))
///     .with_fallback(Arc::new(fallback1))
///     .with_fallback(Arc::new(fallback2))
///     .with_exceptions_to_handle(vec!["ToolException".to_string()]);
/// ```
pub struct RunnableWithFallbacks {
    /// The primary runnable to try first.
    primary: Arc<dyn Runnable>,
    /// Fallbacks tried in order if the primary fails.
    fallbacks: Vec<Arc<dyn Runnable>>,
    /// Optional filter: if set, only errors whose variant name appears in this
    /// list will trigger fallback. Errors not in the list are returned immediately.
    exceptions_to_handle: Option<Vec<String>>,
}

/// Returns the variant name of a `CognisError` for matching against
/// the `exceptions_to_handle` filter.
fn error_variant_name(err: &CognisError) -> &'static str {
    match err {
        CognisError::OutputParserError { .. } => "OutputParserError",
        CognisError::ContextOverflow(_) => "ContextOverflow",
        CognisError::TracerError(_) => "TracerError",
        CognisError::InvalidKey(_) => "InvalidKey",
        CognisError::SerializationError(_) => "SerializationError",
        CognisError::NotImplemented(_) => "NotImplemented",
        CognisError::ToolException(_) => "ToolException",
        CognisError::ToolValidationError(_) => "ToolValidationError",
        CognisError::SchemaAnnotationError(_) => "SchemaAnnotationError",
        CognisError::RecursionLimitExceeded(_) => "RecursionLimitExceeded",
        CognisError::TypeMismatch { .. } => "TypeMismatch",
        CognisError::HttpError { .. } => "HttpError",
        CognisError::IoError(_) => "IoError",
        CognisError::Other(_) => "Other",
    }
}

/// Returns `true` if the error should trigger fallback behaviour based on the
/// configured exception filter.
fn should_fallback(err: &CognisError, filter: &Option<Vec<String>>) -> bool {
    match filter {
        None => true,
        Some(allowed) => {
            let name = error_variant_name(err);
            allowed.iter().any(|a| a == name)
        }
    }
}

impl RunnableWithFallbacks {
    /// Create a new `RunnableWithFallbacks` with a primary runnable.
    ///
    /// Use the builder methods to add fallbacks and configure exception filters.
    pub fn new(primary: Arc<dyn Runnable>) -> Self {
        Self {
            primary,
            fallbacks: Vec::new(),
            exceptions_to_handle: None,
        }
    }

    /// Add a single fallback runnable to the end of the fallback list.
    pub fn with_fallback(mut self, fallback: Arc<dyn Runnable>) -> Self {
        self.fallbacks.push(fallback);
        self
    }

    /// Add multiple fallback runnables at once.
    pub fn with_fallbacks(mut self, fallbacks: Vec<Arc<dyn Runnable>>) -> Self {
        self.fallbacks.extend(fallbacks);
        self
    }

    /// Set the exception filter. Only errors whose variant name appears in the
    /// list will trigger fallback execution. Errors not in the list are returned
    /// immediately.
    pub fn with_exceptions_to_handle(mut self, exceptions: Vec<String>) -> Self {
        self.exceptions_to_handle = Some(exceptions);
        self
    }
}

#[async_trait]
impl Runnable for RunnableWithFallbacks {
    fn name(&self) -> &str {
        // We cannot return a dynamically formatted &str from name() since it
        // requires a 'static or self-borrowed lifetime. We use a fixed label.
        // Use `display_name()` (below) for a fully formatted name if needed.
        "RunnableWithFallbacks"
    }

    async fn invoke(&self, input: Value, config: Option<&RunnableConfig>) -> Result<Value> {
        match self.primary.invoke(input.clone(), config).await {
            Ok(result) => Ok(result),
            Err(primary_err) => {
                if !should_fallback(&primary_err, &self.exceptions_to_handle) {
                    return Err(primary_err);
                }

                let mut last_err = primary_err;
                for fallback in &self.fallbacks {
                    match fallback.invoke(input.clone(), config).await {
                        Ok(result) => return Ok(result),
                        Err(e) => {
                            last_err = e;
                            continue;
                        }
                    }
                }
                Err(last_err)
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
                if !should_fallback(&primary_err, &self.exceptions_to_handle) {
                    return Err(primary_err);
                }

                let mut last_err = primary_err;
                for fallback in &self.fallbacks {
                    match fallback.stream(input.clone(), config).await {
                        Ok(stream) => return Ok(stream),
                        Err(e) => {
                            last_err = e;
                            continue;
                        }
                    }
                }
                Err(last_err)
            }
        }
    }
}

impl RunnableWithFallbacks {
    /// Returns a formatted display name: `"primary_name with fallbacks [f1, f2, ...]"`.
    pub fn display_name(&self) -> String {
        let fb_names: Vec<&str> = self.fallbacks.iter().map(|f| f.name()).collect();
        format!(
            "{} with fallbacks [{}]",
            self.primary.name(),
            fb_names.join(", ")
        )
    }
}

/// Convenience function to create a `RunnableWithFallbacks` from a primary
/// runnable and a list of fallbacks, returned as a trait object.
pub fn with_fallbacks(
    primary: Arc<dyn Runnable>,
    fallbacks: Vec<Arc<dyn Runnable>>,
) -> Arc<dyn Runnable> {
    Arc::new(RunnableWithFallbacks::new(primary).with_fallbacks(fallbacks))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::CognisError;
    use serde_json::json;

    // ── Test helpers ──────────────────────────────────────────────────

    /// A test runnable that always succeeds, returning its input tagged with a label.
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
            Ok(json!({ "from": self.label, "value": input }))
        }
    }

    /// A test runnable that always fails with a configurable error variant.
    struct Fails {
        label: &'static str,
        error: CognisError,
    }

    impl Fails {
        fn other(label: &'static str, msg: &str) -> Self {
            Self {
                label,
                error: CognisError::Other(msg.to_string()),
            }
        }

        fn tool_exception(label: &'static str, msg: &str) -> Self {
            Self {
                label,
                error: CognisError::ToolException(msg.to_string()),
            }
        }

        fn http_error(label: &'static str, status: u16, body: &str) -> Self {
            Self {
                label,
                error: CognisError::HttpError {
                    status,
                    body: body.to_string(),
                },
            }
        }
    }

    #[async_trait]
    impl Runnable for Fails {
        fn name(&self) -> &str {
            self.label
        }

        async fn invoke(&self, _input: Value, _config: Option<&RunnableConfig>) -> Result<Value> {
            Err(match &self.error {
                CognisError::Other(m) => CognisError::Other(m.clone()),
                CognisError::ToolException(m) => CognisError::ToolException(m.clone()),
                CognisError::HttpError { status, body } => CognisError::HttpError {
                    status: *status,
                    body: body.clone(),
                },
                _ => CognisError::Other("unexpected".into()),
            })
        }
    }

    // ── Tests ─────────────────────────────────────────────────────────

    #[tokio::test]
    async fn test_primary_succeeds_fallbacks_not_called() {
        let chain = RunnableWithFallbacks::new(Arc::new(Succeeds::new("primary")))
            .with_fallback(Arc::new(Succeeds::new("fallback1")));

        let result = chain.invoke(json!("data"), None).await.unwrap();
        assert_eq!(result["from"], "primary");
    }

    #[tokio::test]
    async fn test_primary_fails_first_fallback_succeeds() {
        let chain = RunnableWithFallbacks::new(Arc::new(Fails::other("primary", "boom")))
            .with_fallback(Arc::new(Succeeds::new("fb1")))
            .with_fallback(Arc::new(Succeeds::new("fb2")));

        let result = chain.invoke(json!("data"), None).await.unwrap();
        assert_eq!(result["from"], "fb1");
    }

    #[tokio::test]
    async fn test_primary_and_first_fallback_fail_second_succeeds() {
        let chain = RunnableWithFallbacks::new(Arc::new(Fails::other("primary", "err1")))
            .with_fallback(Arc::new(Fails::other("fb1", "err2")))
            .with_fallback(Arc::new(Succeeds::new("fb2")));

        let result = chain.invoke(json!("data"), None).await.unwrap();
        assert_eq!(result["from"], "fb2");
    }

    #[tokio::test]
    async fn test_all_fail_returns_last_error() {
        let chain = RunnableWithFallbacks::new(Arc::new(Fails::other("primary", "primary_err")))
            .with_fallback(Arc::new(Fails::other("fb1", "fb1_err")))
            .with_fallback(Arc::new(Fails::other("fb2", "fb2_err")));

        let err = chain.invoke(json!("data"), None).await.unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("fb2_err"),
            "Expected last fallback error, got: {}",
            msg
        );
    }

    #[tokio::test]
    async fn test_exception_filter_matching_error_triggers_fallback() {
        // Primary fails with ToolException which IS in the filter → fallback runs
        let chain =
            RunnableWithFallbacks::new(Arc::new(Fails::tool_exception("primary", "tool broke")))
                .with_fallback(Arc::new(Succeeds::new("fb1")))
                .with_exceptions_to_handle(vec!["ToolException".to_string()]);

        let result = chain.invoke(json!("data"), None).await.unwrap();
        assert_eq!(result["from"], "fb1");
    }

    #[tokio::test]
    async fn test_exception_filter_non_matching_error_propagates_immediately() {
        // Primary fails with Other which is NOT in the filter → error propagates
        let chain = RunnableWithFallbacks::new(Arc::new(Fails::other("primary", "other err")))
            .with_fallback(Arc::new(Succeeds::new("fb1")))
            .with_exceptions_to_handle(vec!["ToolException".to_string()]);

        let err = chain.invoke(json!("data"), None).await.unwrap_err();
        let msg = format!("{}", err);
        assert!(
            msg.contains("other err"),
            "Non-matching error should propagate immediately, got: {}",
            msg
        );
    }

    #[tokio::test]
    async fn test_empty_fallbacks_returns_primary_error() {
        let chain = RunnableWithFallbacks::new(Arc::new(Fails::other("primary", "solo error")));

        let err = chain.invoke(json!("data"), None).await.unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("solo error"));
    }

    #[tokio::test]
    async fn test_fallback_with_different_output_value() {
        // Primary fails, fallback returns a completely different shape
        let chain = RunnableWithFallbacks::new(Arc::new(Fails::other("primary", "fail")))
            .with_fallback(Arc::new(Succeeds::new("alternative")));

        let result = chain.invoke(json!(42), None).await.unwrap();
        assert_eq!(result["from"], "alternative");
        assert_eq!(result["value"], json!(42));
    }

    #[tokio::test]
    async fn test_config_passing_through_fallbacks() {
        /// A runnable that reads a tag from config metadata and includes it in output.
        struct ConfigReader;

        #[async_trait]
        impl Runnable for ConfigReader {
            fn name(&self) -> &str {
                "ConfigReader"
            }

            async fn invoke(
                &self,
                _input: Value,
                config: Option<&RunnableConfig>,
            ) -> Result<Value> {
                let tag = config
                    .and_then(|c| c.metadata.get("tag"))
                    .cloned()
                    .unwrap_or_else(|| json!("none"));
                Ok(json!({ "tag": tag }))
            }
        }

        let chain = RunnableWithFallbacks::new(Arc::new(Fails::other("primary", "fail")))
            .with_fallback(Arc::new(ConfigReader));

        let mut config = RunnableConfig::default();
        config.metadata.insert("tag".to_string(), json!("hello"));

        let result = chain.invoke(json!(null), Some(&config)).await.unwrap();
        assert_eq!(result["tag"], "hello");
    }

    #[tokio::test]
    async fn test_display_name_formatting() {
        let chain = RunnableWithFallbacks::new(Arc::new(Succeeds::new("main_model")))
            .with_fallback(Arc::new(Succeeds::new("gpt4")))
            .with_fallback(Arc::new(Succeeds::new("claude")));

        assert_eq!(
            chain.display_name(),
            "main_model with fallbacks [gpt4, claude]"
        );
    }

    #[tokio::test]
    async fn test_builder_pattern_chaining() {
        let chain = RunnableWithFallbacks::new(Arc::new(Fails::other("p", "err")))
            .with_fallback(Arc::new(Fails::other("f1", "err1")))
            .with_fallbacks(vec![
                Arc::new(Fails::other("f2", "err2")) as Arc<dyn Runnable>,
                Arc::new(Succeeds::new("f3")),
            ])
            .with_exceptions_to_handle(vec!["Other".to_string()]);

        let result = chain.invoke(json!("x"), None).await.unwrap();
        assert_eq!(result["from"], "f3");
    }

    #[tokio::test]
    async fn test_batch_invoke_with_fallbacks() {
        let chain = RunnableWithFallbacks::new(Arc::new(Fails::other("primary", "fail")))
            .with_fallback(Arc::new(Succeeds::new("fb")));

        let results = chain
            .batch(vec![json!(1), json!(2), json!(3)], None)
            .await
            .unwrap();

        assert_eq!(results.len(), 3);
        for (i, result) in results.iter().enumerate() {
            assert_eq!(result["from"], "fb");
            assert_eq!(result["value"], json!(i + 1));
        }
    }

    #[tokio::test]
    async fn test_with_fallbacks_helper_function() {
        let runnable = with_fallbacks(
            Arc::new(Fails::other("primary", "err")),
            vec![Arc::new(Succeeds::new("helper_fb")) as Arc<dyn Runnable>],
        );

        let result = runnable.invoke(json!("test"), None).await.unwrap();
        assert_eq!(result["from"], "helper_fb");
    }

    #[tokio::test]
    async fn test_exception_filter_with_http_error() {
        // Filter allows HttpError → fallback should be tried
        let chain =
            RunnableWithFallbacks::new(Arc::new(Fails::http_error("primary", 500, "server error")))
                .with_fallback(Arc::new(Succeeds::new("fb")))
                .with_exceptions_to_handle(vec!["HttpError".to_string()]);

        let result = chain.invoke(json!("data"), None).await.unwrap();
        assert_eq!(result["from"], "fb");
    }
}
