use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use crate::error::{Result, RustChainError};
use super::types::{ErrorHandler, ResponseFormat, ToolInput, ToolOutput};

/// Schema description for a tool's input.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<HashMap<String, Value>>,
}

/// A collection of related tools.
pub trait BaseToolkit: Send + Sync {
    fn get_tools(&self) -> Vec<Box<dyn BaseTool>>;
}

/// Interface for tools that can be called by agents.
#[async_trait]
pub trait BaseTool: Send + Sync {
    /// The name of the tool.
    fn name(&self) -> &str;

    /// A description of what the tool does.
    fn description(&self) -> &str;

    /// The JSON schema for the tool's arguments.
    fn args_schema(&self) -> Option<Value> {
        None
    }

    /// The full tool call schema (defaults to args_schema or empty object).
    fn tool_call_schema(&self) -> Value {
        self.args_schema()
            .unwrap_or(Value::Object(Default::default()))
    }

    /// Whether to return the tool output directly to the user.
    fn return_direct(&self) -> bool {
        false
    }

    /// How to handle tool execution errors.
    fn handle_tool_error(&self) -> &ErrorHandler {
        &ErrorHandler::Propagate
    }

    /// How to handle tool validation errors.
    fn handle_validation_error(&self) -> &ErrorHandler {
        &ErrorHandler::Propagate
    }

    /// The response format for the tool output.
    fn response_format(&self) -> ResponseFormat {
        ResponseFormat::Content
    }

    /// Tags associated with the tool.
    fn tags(&self) -> &[String] {
        &[]
    }

    /// Metadata associated with the tool.
    fn metadata(&self) -> Option<&HashMap<String, Value>> {
        None
    }

    /// Extra configuration for the tool.
    fn extras(&self) -> Option<&HashMap<String, Value>> {
        None
    }

    /// Core implementation of the tool logic.
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput>;

    /// Run the tool with error handling.
    async fn run(&self, input: ToolInput, _tool_call_id: Option<&str>) -> Result<Value> {
        match self._run(input).await {
            Ok(output) => {
                let content = match output {
                    ToolOutput::Content(v) => v,
                    ToolOutput::ContentAndArtifact { content, .. } => content,
                };
                Ok(content)
            }
            Err(RustChainError::ToolException(msg)) => match self.handle_tool_error() {
                ErrorHandler::Propagate => Err(RustChainError::ToolException(msg)),
                ErrorHandler::DefaultMessage => Ok(Value::String(msg)),
                ErrorHandler::StaticMessage(s) => Ok(Value::String(s.clone())),
                ErrorHandler::Dynamic(f) => Ok(Value::String(f(&msg))),
            },
            Err(RustChainError::ToolValidationError(msg)) => {
                match self.handle_validation_error() {
                    ErrorHandler::Propagate => Err(RustChainError::ToolValidationError(msg)),
                    ErrorHandler::DefaultMessage => Ok(Value::String(msg)),
                    ErrorHandler::StaticMessage(s) => Ok(Value::String(s.clone())),
                    ErrorHandler::Dynamic(f) => Ok(Value::String(f(&msg))),
                }
            }
            Err(e) => Err(e),
        }
    }

    /// Convenience method to run the tool with a string input.
    async fn run_str(&self, input: &str) -> Result<Value> {
        self.run(ToolInput::Text(input.to_string()), None).await
    }

    /// Run the tool with structured (JSON) input.
    async fn run_json(&self, input: &Value) -> Result<Value> {
        let map: HashMap<String, Value> = match input {
            Value::Object(m) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            Value::String(s) => return self.run(ToolInput::Text(s.clone()), None).await,
            _ => return self.run(ToolInput::Text(input.to_string()), None).await,
        };
        self.run(ToolInput::Structured(map), None).await
    }
}
