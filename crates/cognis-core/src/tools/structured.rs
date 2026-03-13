use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

use async_trait::async_trait;
use serde_json::Value;

use super::base::BaseTool;
use super::types::{ErrorHandler, ResponseFormat, ToolInput, ToolOutput};
use crate::error::{Result, CognisError};

/// An async function that takes structured arguments and returns a JSON value.
type StructuredFn = Box<
    dyn Fn(HashMap<String, Value>) -> Pin<Box<dyn Future<Output = Result<Value>> + Send>>
        + Send
        + Sync,
>;

/// A tool that accepts structured (named) arguments validated against a JSON Schema.
///
/// Mirrors Python's `langchain_core.tools.StructuredTool`. Unlike `FunctionTool`,
/// which takes raw `ToolInput`, `StructuredTool` expects a `HashMap<String, Value>`
/// and validates required fields from the schema before invoking the function.
///
/// # Example
///
/// ```ignore
/// use cognis_core::tools::StructuredTool;
/// use serde_json::json;
///
/// let tool = StructuredTool::new(
///     "add",
///     "Add two numbers",
///     json!({
///         "type": "object",
///         "properties": {
///             "a": { "type": "number" },
///             "b": { "type": "number" }
///         },
///         "required": ["a", "b"]
///     }),
///     |args| Box::pin(async move {
///         let a = args["a"].as_f64().unwrap_or(0.0);
///         let b = args["b"].as_f64().unwrap_or(0.0);
///         Ok(serde_json::json!(a + b))
///     }),
/// );
/// ```
pub struct StructuredTool {
    name: String,
    description: String,
    args_schema: Value,
    return_direct: bool,
    func: StructuredFn,
}

impl StructuredTool {
    /// Create a new `StructuredTool`.
    ///
    /// - `name` — tool name exposed to the agent.
    /// - `description` — human-readable description of what the tool does.
    /// - `schema` — JSON Schema (object with `properties` and optional `required`).
    /// - `func` — async function receiving validated arguments.
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: Value,
        func: F,
    ) -> Self
    where
        F: Fn(HashMap<String, Value>) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<Value>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            args_schema: schema,
            return_direct: false,
            func: Box::new(move |args| Box::pin(func(args))),
        }
    }

    /// Set whether the tool output should be returned directly to the user.
    pub fn with_return_direct(mut self, return_direct: bool) -> Self {
        self.return_direct = return_direct;
        self
    }

    /// Validate that all fields listed in the schema's `"required"` array are
    /// present in `args`. Returns an error describing the first missing field.
    fn validate_required(&self, args: &HashMap<String, Value>) -> Result<()> {
        if let Some(required) = self.args_schema.get("required").and_then(|v| v.as_array()) {
            for field in required {
                if let Some(field_name) = field.as_str() {
                    if !args.contains_key(field_name) {
                        return Err(CognisError::ToolValidationError(format!(
                            "Missing required argument: '{}'",
                            field_name
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    /// Extract a `HashMap<String, Value>` from the given `ToolInput`.
    fn extract_args(&self, input: ToolInput) -> Result<HashMap<String, Value>> {
        match input {
            ToolInput::Structured(map) => Ok(map),
            ToolInput::ToolCall(tc) => Ok(tc.args),
            ToolInput::Text(text) => {
                // Attempt to parse the text as a JSON object.
                let parsed: Value = serde_json::from_str(&text).map_err(|_| {
                    CognisError::ToolValidationError(format!(
                        "Expected JSON object input for structured tool '{}', got plain text",
                        self.name
                    ))
                })?;
                match parsed {
                    Value::Object(map) => Ok(map.into_iter().collect()),
                    _ => Err(CognisError::ToolValidationError(format!(
                        "Expected JSON object input for structured tool '{}', got {}",
                        self.name, parsed
                    ))),
                }
            }
        }
    }
}

#[async_trait]
impl BaseTool for StructuredTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn args_schema(&self) -> Option<Value> {
        Some(self.args_schema.clone())
    }

    fn return_direct(&self) -> bool {
        self.return_direct
    }

    fn handle_tool_error(&self) -> &ErrorHandler {
        &ErrorHandler::Propagate
    }

    fn handle_validation_error(&self) -> &ErrorHandler {
        &ErrorHandler::Propagate
    }

    fn response_format(&self) -> ResponseFormat {
        ResponseFormat::Content
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let args = self.extract_args(input)?;
        self.validate_required(&args)?;
        let value = (self.func)(args).await?;
        Ok(ToolOutput::Content(value))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn add_schema() -> Value {
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "number" },
                "b": { "type": "number" }
            },
            "required": ["a", "b"]
        })
    }

    fn make_add_tool() -> StructuredTool {
        StructuredTool::new("add", "Add two numbers", add_schema(), |args| async move {
            let a = args["a"].as_f64().unwrap_or(0.0);
            let b = args["b"].as_f64().unwrap_or(0.0);
            Ok(json!(a + b))
        })
    }

    #[tokio::test]
    async fn test_structured_tool_run() {
        let tool = make_add_tool();
        let mut args = HashMap::new();
        args.insert("a".to_string(), json!(2));
        args.insert("b".to_string(), json!(3));

        let result = tool._run(ToolInput::Structured(args)).await.unwrap();
        match result {
            ToolOutput::Content(v) => assert_eq!(v, json!(5.0)),
            _ => panic!("Expected Content output"),
        }
    }

    #[tokio::test]
    async fn test_missing_required_field() {
        let tool = make_add_tool();
        let mut args = HashMap::new();
        args.insert("a".to_string(), json!(2));

        let result = tool._run(ToolInput::Structured(args)).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(matches!(err, CognisError::ToolValidationError(_)));
    }

    #[tokio::test]
    async fn test_text_json_input() {
        let tool = make_add_tool();
        let input = ToolInput::Text(r#"{"a": 10, "b": 20}"#.to_string());
        let result = tool._run(input).await.unwrap();
        match result {
            ToolOutput::Content(v) => assert_eq!(v, json!(30.0)),
            _ => panic!("Expected Content output"),
        }
    }

    #[tokio::test]
    async fn test_return_direct() {
        let tool = make_add_tool().with_return_direct(true);
        assert!(tool.return_direct());
    }

    #[test]
    fn test_name_and_description() {
        let tool = make_add_tool();
        assert_eq!(tool.name(), "add");
        assert_eq!(tool.description(), "Add two numbers");
        assert!(tool.args_schema().is_some());
    }
}
