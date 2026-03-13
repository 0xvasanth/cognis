use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use super::base::BaseTool;
use super::types::{ErrorHandler, ResponseFormat, ToolInput, ToolOutput};
use crate::error::{Result, CognisError};

/// A sync function that takes a single string and returns a string.
type SimpleSyncFn = Arc<dyn Fn(&str) -> Result<String> + Send + Sync>;

/// An async function that takes a single string and returns a string.
type SimpleAsyncFn = Arc<
    dyn Fn(String) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send>>
        + Send
        + Sync,
>;

/// A simple tool that wraps a function accepting a single string input.
///
/// This is the Rust equivalent of Python's `langchain_core.tools.simple.Tool`.
/// It enforces single-input semantics: the tool takes exactly one string argument
/// and returns a string result.
///
/// For tools that accept structured (named) arguments, use [`StructuredTool`] instead.
///
/// # Example
///
/// ```rust
/// use cognis_core::tools::SimpleTool;
///
/// let tool = SimpleTool::new(
///     "search",
///     "Search the web for a query",
///     |query: &str| Ok(format!("Results for: {}", query)),
/// );
/// ```
pub struct SimpleTool {
    name: String,
    description: String,
    return_direct: bool,
    response_format: ResponseFormat,
    error_handler: ErrorHandler,
    validation_error_handler: ErrorHandler,
    tags: Vec<String>,
    metadata: HashMap<String, Value>,
    extras: Option<HashMap<String, Value>>,
    func: Option<SimpleSyncFn>,
    async_func: Option<SimpleAsyncFn>,
}

impl SimpleTool {
    /// Create a new `SimpleTool` from a synchronous function.
    ///
    /// - `name` — tool name exposed to the agent.
    /// - `description` — human-readable description of what the tool does.
    /// - `func` — synchronous function taking a `&str` and returning a `Result<String>`.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        func: impl Fn(&str) -> Result<String> + Send + Sync + 'static,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            return_direct: false,
            response_format: ResponseFormat::Content,
            error_handler: ErrorHandler::Propagate,
            validation_error_handler: ErrorHandler::Propagate,
            tags: Vec::new(),
            metadata: HashMap::new(),
            extras: None,
            func: Some(Arc::new(func)),
            async_func: None,
        }
    }

    /// Create a new `SimpleTool` from an async function.
    ///
    /// - `name` — tool name exposed to the agent.
    /// - `description` — human-readable description of what the tool does.
    /// - `func` — async function taking a `String` and returning a `Result<String>`.
    pub fn new_async<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        func: F,
    ) -> Self
    where
        F: Fn(String) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<String>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            return_direct: false,
            response_format: ResponseFormat::Content,
            error_handler: ErrorHandler::Propagate,
            validation_error_handler: ErrorHandler::Propagate,
            tags: Vec::new(),
            metadata: HashMap::new(),
            extras: None,
            func: None,
            async_func: Some(Arc::new(move |input| Box::pin(func(input)))),
        }
    }

    /// Create a `SimpleTool` from a function, equivalent to Python's `Tool.from_function()`.
    ///
    /// Accepts both a sync function and an optional async function.
    pub fn from_function(
        name: impl Into<String>,
        description: impl Into<String>,
        func: impl Fn(&str) -> Result<String> + Send + Sync + 'static,
    ) -> Self {
        Self::new(name, description, func)
    }

    /// Set whether the tool output should be returned directly to the user.
    pub fn with_return_direct(mut self, return_direct: bool) -> Self {
        self.return_direct = return_direct;
        self
    }

    /// Set the response format for this tool.
    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = format;
        self
    }

    /// Set the error handler for tool execution errors.
    pub fn with_error_handler(mut self, handler: ErrorHandler) -> Self {
        self.error_handler = handler;
        self
    }

    /// Set the error handler for tool validation errors.
    pub fn with_validation_error_handler(mut self, handler: ErrorHandler) -> Self {
        self.validation_error_handler = handler;
        self
    }

    /// Set the tags for this tool.
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Set the metadata for this tool.
    pub fn with_metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set the extras for this tool.
    pub fn with_extras(mut self, extras: HashMap<String, Value>) -> Self {
        self.extras = Some(extras);
        self
    }

    /// Extract a single string argument from the tool input.
    ///
    /// Mirrors the single-input enforcement from Python's `Tool._to_args_and_kwargs`.
    fn extract_string_input(&self, input: ToolInput) -> Result<String> {
        match input {
            ToolInput::Text(s) => Ok(s),
            ToolInput::Structured(map) => {
                // For structured input, expect exactly one value.
                let values: Vec<Value> = map.into_values().collect();
                if values.len() != 1 {
                    return Err(CognisError::ToolException(format!(
                        "Too many arguments to single-input tool '{}'. \
                         Consider using StructuredTool instead. Args: {:?}",
                        self.name, values
                    )));
                }
                match &values[0] {
                    Value::String(s) => Ok(s.clone()),
                    other => Ok(other.to_string()),
                }
            }
            ToolInput::ToolCall(tc) => {
                let values: Vec<Value> = tc.args.into_values().collect();
                if values.len() != 1 {
                    return Err(CognisError::ToolException(format!(
                        "Too many arguments to single-input tool '{}'. \
                         Consider using StructuredTool instead. Args: {:?}",
                        self.name, values
                    )));
                }
                match &values[0] {
                    Value::String(s) => Ok(s.clone()),
                    other => Ok(other.to_string()),
                }
            }
        }
    }
}

#[async_trait]
impl BaseTool for SimpleTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn args_schema(&self) -> Option<Value> {
        Some(serde_json::json!({
            "type": "object",
            "properties": {
                "tool_input": { "type": "string" }
            },
            "required": ["tool_input"]
        }))
    }

    fn return_direct(&self) -> bool {
        self.return_direct
    }

    fn handle_tool_error(&self) -> &ErrorHandler {
        &self.error_handler
    }

    fn handle_validation_error(&self) -> &ErrorHandler {
        &self.validation_error_handler
    }

    fn response_format(&self) -> ResponseFormat {
        self.response_format
    }

    fn tags(&self) -> &[String] {
        &self.tags
    }

    fn metadata(&self) -> Option<&HashMap<String, Value>> {
        if self.metadata.is_empty() {
            None
        } else {
            Some(&self.metadata)
        }
    }

    fn extras(&self) -> Option<&HashMap<String, Value>> {
        self.extras.as_ref()
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let text = self.extract_string_input(input)?;

        if let Some(ref async_func) = self.async_func {
            let result = (async_func)(text).await?;
            return Ok(ToolOutput::Content(Value::String(result)));
        }

        if let Some(ref func) = self.func {
            let result = (func)(&text)?;
            return Ok(ToolOutput::Content(Value::String(result)));
        }

        Err(CognisError::NotImplemented(
            "Tool does not support invocation: no function provided".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn test_simple_tool_with_text_input() {
        let tool = SimpleTool::new("echo", "Echo the input", |input: &str| {
            Ok(format!("echo: {}", input))
        });

        let result = tool
            ._run(ToolInput::Text("hello".to_string()))
            .await
            .unwrap();
        match result {
            ToolOutput::Content(v) => assert_eq!(v, Value::String("echo: hello".to_string())),
            _ => panic!("Expected Content output"),
        }
    }

    #[tokio::test]
    async fn test_simple_tool_with_structured_single_arg() {
        let tool = SimpleTool::new("echo", "Echo the input", |input: &str| {
            Ok(format!("echo: {}", input))
        });

        let mut args = HashMap::new();
        args.insert("tool_input".to_string(), json!("world"));
        let result = tool._run(ToolInput::Structured(args)).await.unwrap();
        match result {
            ToolOutput::Content(v) => assert_eq!(v, Value::String("echo: world".to_string())),
            _ => panic!("Expected Content output"),
        }
    }

    #[tokio::test]
    async fn test_simple_tool_rejects_multiple_args() {
        let tool = SimpleTool::new("echo", "Echo the input", |input: &str| {
            Ok(format!("echo: {}", input))
        });

        let mut args = HashMap::new();
        args.insert("a".to_string(), json!("x"));
        args.insert("b".to_string(), json!("y"));
        let result = tool._run(ToolInput::Structured(args)).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_simple_tool_async() {
        let tool = SimpleTool::new_async("async_echo", "Async echo", |input: String| async move {
            Ok(format!("async: {}", input))
        });

        let result = tool
            ._run(ToolInput::Text("test".to_string()))
            .await
            .unwrap();
        match result {
            ToolOutput::Content(v) => assert_eq!(v, Value::String("async: test".to_string())),
            _ => panic!("Expected Content output"),
        }
    }

    #[tokio::test]
    async fn test_simple_tool_from_function() {
        let tool = SimpleTool::from_function("greet", "Greet someone", |name: &str| {
            Ok(format!("Hello, {}!", name))
        });

        assert_eq!(tool.name(), "greet");
        assert_eq!(tool.description(), "Greet someone");

        let result = tool
            ._run(ToolInput::Text("Alice".to_string()))
            .await
            .unwrap();
        match result {
            ToolOutput::Content(v) => assert_eq!(v, Value::String("Hello, Alice!".to_string())),
            _ => panic!("Expected Content output"),
        }
    }

    #[test]
    fn test_simple_tool_args_schema() {
        let tool = SimpleTool::new("test", "A test tool", |_: &str| Ok("ok".to_string()));
        let schema = tool.args_schema().unwrap();
        assert_eq!(schema["type"], "object");
        assert!(schema["properties"]["tool_input"].is_object());
    }

    #[test]
    fn test_simple_tool_builder_methods() {
        let tool = SimpleTool::new("test", "A test tool", |_: &str| Ok("ok".to_string()))
            .with_return_direct(true)
            .with_response_format(ResponseFormat::ContentAndArtifact)
            .with_tags(vec!["tag1".to_string()]);

        assert!(tool.return_direct());
        assert_eq!(tool.response_format(), ResponseFormat::ContentAndArtifact);
        assert_eq!(tool.tags(), &["tag1".to_string()]);
    }

    #[tokio::test]
    async fn test_simple_tool_no_func_errors() {
        let tool = SimpleTool {
            name: "broken".to_string(),
            description: "No function".to_string(),
            return_direct: false,
            response_format: ResponseFormat::Content,
            error_handler: ErrorHandler::Propagate,
            validation_error_handler: ErrorHandler::Propagate,
            tags: Vec::new(),
            metadata: HashMap::new(),
            extras: None,
            func: None,
            async_func: None,
        };

        let result = tool._run(ToolInput::Text("test".to_string())).await;
        assert!(result.is_err());
    }
}
