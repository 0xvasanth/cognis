use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use serde_json::Value;
use crate::error::Result;
use super::base::{BaseTool, ToolSchema};
use super::types::{ErrorHandler, ResponseFormat, ToolInput, ToolOutput};

type SyncToolFn = Arc<dyn Fn(ToolInput) -> Result<ToolOutput> + Send + Sync>;

/// A tool backed by a simple function closure.
pub struct FunctionTool {
    schema: ToolSchema,
    return_direct: bool,
    response_format: ResponseFormat,
    error_handler: ErrorHandler,
    validation_error_handler: ErrorHandler,
    tags: Vec<String>,
    metadata: HashMap<String, Value>,
    func: SyncToolFn,
}

impl FunctionTool {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        parameters: Option<Value>,
        func: impl Fn(ToolInput) -> Result<ToolOutput> + Send + Sync + 'static,
    ) -> Self {
        Self {
            schema: ToolSchema {
                name: name.into(),
                description: description.into(),
                parameters,
                extras: None,
            },
            return_direct: false,
            response_format: ResponseFormat::Content,
            error_handler: ErrorHandler::Propagate,
            validation_error_handler: ErrorHandler::Propagate,
            tags: Vec::new(),
            metadata: HashMap::new(),
            func: Arc::new(func),
        }
    }

    pub fn with_return_direct(mut self, return_direct: bool) -> Self {
        self.return_direct = return_direct;
        self
    }

    pub fn with_response_format(mut self, format: ResponseFormat) -> Self {
        self.response_format = format;
        self
    }

    pub fn with_error_handler(mut self, handler: ErrorHandler) -> Self {
        self.error_handler = handler;
        self
    }

    pub fn with_validation_error_handler(mut self, handler: ErrorHandler) -> Self {
        self.validation_error_handler = handler;
        self
    }

    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_extras(mut self, extras: HashMap<String, Value>) -> Self {
        self.schema.extras = Some(extras);
        self
    }
}

#[async_trait]
impl BaseTool for FunctionTool {
    fn name(&self) -> &str {
        &self.schema.name
    }

    fn description(&self) -> &str {
        &self.schema.description
    }

    fn args_schema(&self) -> Option<Value> {
        self.schema.parameters.clone()
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
        self.schema.extras.as_ref()
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        (self.func)(input)
    }
}

/// Create a tool from a synchronous function that takes and returns JSON values.
///
/// This is the Rust equivalent of Python's `@tool` decorator. It provides a
/// simpler API than constructing a [`FunctionTool`] directly when your function
/// works with raw JSON values.
///
/// # Arguments
/// * `name` - The name of the tool.
/// * `description` - A human-readable description of what the tool does.
/// * `schema` - JSON Schema for the tool's input parameters.
/// * `func` - A function that accepts a `serde_json::Value` input and returns a `Result<Value>`.
///
/// # Example
///
/// ```ignore
/// use rustchain_core::tools::tool_from_function;
/// use serde_json::json;
///
/// let tool = tool_from_function(
///     "add",
///     "Add two numbers",
///     json!({"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}}),
///     |input| {
///         let a = input["a"].as_f64().unwrap_or(0.0);
///         let b = input["b"].as_f64().unwrap_or(0.0);
///         Ok(serde_json::json!(a + b))
///     },
/// );
/// ```
pub fn tool_from_function<F>(
    name: impl Into<String>,
    description: impl Into<String>,
    schema: Value,
    func: F,
) -> FunctionTool
where
    F: Fn(Value) -> Result<Value> + Send + Sync + 'static,
{
    FunctionTool::new(
        name,
        description,
        Some(schema),
        move |input: ToolInput| {
            let json_input = match input {
                ToolInput::Text(s) => {
                    serde_json::from_str::<Value>(&s).unwrap_or(Value::String(s))
                }
                ToolInput::Structured(map) => Value::Object(
                    map.into_iter().collect::<serde_json::Map<String, Value>>(),
                ),
                ToolInput::ToolCall(tc) => Value::Object(
                    tc.args.into_iter().collect::<serde_json::Map<String, Value>>(),
                ),
            };
            let result = func(json_input)?;
            Ok(ToolOutput::Content(result))
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_upper_tool() -> FunctionTool {
        FunctionTool::new(
            "upper",
            "Convert text to uppercase",
            None,
            |input: ToolInput| match input {
                ToolInput::Text(s) => Ok(ToolOutput::Content(Value::String(s.to_uppercase()))),
                _ => Ok(ToolOutput::Content(Value::String("unsupported".to_string()))),
            },
        )
    }

    #[tokio::test]
    async fn test_function_tool_text_input() {
        let tool = make_upper_tool();
        let result = tool
            ._run(ToolInput::Text("hello".to_string()))
            .await
            .unwrap();
        match result {
            ToolOutput::Content(v) => assert_eq!(v, Value::String("HELLO".to_string())),
            _ => panic!("Expected Content output"),
        }
    }

    #[tokio::test]
    async fn test_function_tool_run_str() {
        let tool = make_upper_tool();
        let result = tool.run_str("world").await.unwrap();
        assert_eq!(result, Value::String("WORLD".to_string()));
    }

    #[test]
    fn test_function_tool_name_and_description() {
        let tool = make_upper_tool();
        assert_eq!(tool.name(), "upper");
        assert_eq!(tool.description(), "Convert text to uppercase");
        assert!(tool.args_schema().is_none());
    }

    #[test]
    fn test_function_tool_builder_methods() {
        let tool = make_upper_tool()
            .with_return_direct(true)
            .with_response_format(ResponseFormat::ContentAndArtifact)
            .with_tags(vec!["test".to_string()])
            .with_error_handler(ErrorHandler::DefaultMessage);

        assert!(tool.return_direct());
        assert_eq!(tool.response_format(), ResponseFormat::ContentAndArtifact);
        assert_eq!(tool.tags(), &["test".to_string()]);
        assert!(matches!(tool.handle_tool_error(), ErrorHandler::DefaultMessage));
    }

    #[test]
    fn test_function_tool_with_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "text": { "type": "string" }
            }
        });
        let tool = FunctionTool::new(
            "echo",
            "Echo tool",
            Some(schema.clone()),
            |input: ToolInput| match input {
                ToolInput::Text(s) => Ok(ToolOutput::Content(Value::String(s))),
                _ => Ok(ToolOutput::Content(Value::Null)),
            },
        );
        assert_eq!(tool.args_schema(), Some(schema));
    }

    #[test]
    fn test_function_tool_with_metadata() {
        let mut metadata = HashMap::new();
        metadata.insert("version".to_string(), json!("1.0"));
        let tool = make_upper_tool().with_metadata(metadata);
        assert!(tool.metadata().is_some());
        assert_eq!(tool.metadata().unwrap()["version"], json!("1.0"));
    }

    #[test]
    fn test_function_tool_with_extras() {
        let mut extras = HashMap::new();
        extras.insert("cache_control".to_string(), json!({"type": "ephemeral"}));
        let tool = make_upper_tool().with_extras(extras);
        assert!(tool.extras().is_some());
    }

    #[tokio::test]
    async fn test_function_tool_error_handling() {
        let tool = FunctionTool::new(
            "failing",
            "A tool that fails",
            None,
            |_input: ToolInput| {
                Err(crate::error::RustChainError::ToolException(
                    "something went wrong".to_string(),
                ))
            },
        )
        .with_error_handler(ErrorHandler::StaticMessage("handled".to_string()));

        // Using `run` (not `_run`) to trigger error handling.
        let result = tool
            .run(ToolInput::Text("test".to_string()), None)
            .await
            .unwrap();
        assert_eq!(result, Value::String("handled".to_string()));
    }
}
