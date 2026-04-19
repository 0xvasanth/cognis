use super::types::{ErrorHandler, ResponseFormat, ToolInput, ToolOutput};
use crate::error::{CognisError, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

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

/// Apply an error-handler policy to a tool error, converting it (where
/// configured) into a `Value::String` observation the agent can feed back
/// to the model, or re-raising the error for caller-side handling.
///
/// Used by `BaseTool::run` and the agent executor (which calls `_run`
/// directly and must replicate policy outside the trait).
pub fn apply_error_handler(handler: &ErrorHandler, error: CognisError) -> Result<Value> {
    match error {
        CognisError::ToolException(msg) => match handler {
            ErrorHandler::Propagate => Err(CognisError::ToolException(msg)),
            ErrorHandler::DefaultMessage => Ok(Value::String(msg)),
            ErrorHandler::StaticMessage(s) => Ok(Value::String(s.clone())),
            ErrorHandler::Dynamic(f) => Ok(Value::String(f(&msg))),
        },
        CognisError::ToolValidationError(msg) => match handler {
            ErrorHandler::Propagate => Err(CognisError::ToolValidationError(msg)),
            ErrorHandler::DefaultMessage => Ok(Value::String(msg)),
            ErrorHandler::StaticMessage(s) => Ok(Value::String(s.clone())),
            ErrorHandler::Dynamic(f) => Ok(Value::String(f(&msg))),
        },
        other => Err(other),
    }
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
    ///
    /// Returns only the content `Value`. If the tool produces
    /// `ToolOutput::ContentAndArtifact`, the artifact is discarded.
    /// Callers that need the artifact (such as UI callback consumers)
    /// should call [`_run`](Self::_run) directly and match on `ToolOutput`.
    async fn run(&self, input: ToolInput, _tool_call_id: Option<&str>) -> Result<Value> {
        match self._run(input).await {
            Ok(output) => Ok(match output {
                ToolOutput::Content(v) => v,
                ToolOutput::ContentAndArtifact { content, .. } => content,
            }),
            Err(e @ CognisError::ToolException(_)) => {
                apply_error_handler(self.handle_tool_error(), e)
            }
            Err(e @ CognisError::ToolValidationError(_)) => {
                apply_error_handler(self.handle_validation_error(), e)
            }
            Err(e) => Err(e),
        }
    }

    /// Convenience method to run the tool with a string input.
    ///
    /// Returns only the content `Value`; see [`run`](Self::run) for the
    /// artifact-discard caveat.
    async fn run_str(&self, input: &str) -> Result<Value> {
        self.run(ToolInput::Text(input.to_string()), None).await
    }

    /// Run the tool with structured (JSON) input.
    ///
    /// Returns only the content `Value`; see [`run`](Self::run) for the
    /// artifact-discard caveat.
    async fn run_json(&self, input: &Value) -> Result<Value> {
        let map: HashMap<String, Value> = match input {
            Value::Object(m) => m.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            Value::String(s) => return self.run(ToolInput::Text(s.clone()), None).await,
            _ => return self.run(ToolInput::Text(input.to_string()), None).await,
        };
        self.run(ToolInput::Structured(map), None).await
    }
}

#[cfg(test)]
mod error_handler_tests {
    use super::*;

    #[test]
    fn propagate_returns_err() {
        let handler = ErrorHandler::Propagate;
        let result = apply_error_handler(&handler, CognisError::ToolException("boom".into()));
        match result {
            Err(CognisError::ToolException(msg)) => assert_eq!(msg, "boom"),
            other => panic!("expected ToolException, got {other:?}"),
        }
    }

    #[test]
    fn default_message_returns_error_text() {
        let handler = ErrorHandler::DefaultMessage;
        let result = apply_error_handler(&handler, CognisError::ToolException("boom".into()));
        assert_eq!(result.unwrap(), Value::String("boom".to_string()));
    }

    #[test]
    fn static_message_returns_configured_text() {
        let handler = ErrorHandler::StaticMessage("safe fallback".into());
        let result = apply_error_handler(&handler, CognisError::ToolException("boom".into()));
        assert_eq!(result.unwrap(), Value::String("safe fallback".to_string()));
    }

    #[test]
    fn dynamic_uses_callback() {
        let handler =
            ErrorHandler::Dynamic(std::sync::Arc::new(|msg: &str| format!("wrapped: {msg}")));
        let result = apply_error_handler(&handler, CognisError::ToolException("boom".into()));
        assert_eq!(result.unwrap(), Value::String("wrapped: boom".to_string()));
    }

    #[test]
    fn validation_error_respects_static_message() {
        let handler = ErrorHandler::StaticMessage("bad input".into());
        let result = apply_error_handler(
            &handler,
            CognisError::ToolValidationError("schema mismatch".into()),
        );
        assert_eq!(result.unwrap(), Value::String("bad input".to_string()));
    }

    #[test]
    fn validation_error_propagate_preserves_variant_and_message() {
        let handler = ErrorHandler::Propagate;
        let result = apply_error_handler(
            &handler,
            CognisError::ToolValidationError("schema mismatch".into()),
        );
        match result {
            Err(CognisError::ToolValidationError(msg)) => assert_eq!(msg, "schema mismatch"),
            other => panic!("expected ToolValidationError, got {other:?}"),
        }
    }

    #[test]
    fn non_tool_error_passes_through_unchanged() {
        // `CognisError::Other` is not a tool-level error; the `other => Err(other)`
        // fallthrough in `apply_error_handler` must propagate it unchanged
        // regardless of the handler policy.
        let handler = ErrorHandler::DefaultMessage;
        let error = CognisError::Other("unexpected failure".into());
        let result = apply_error_handler(&handler, error);
        match result {
            Err(CognisError::Other(msg)) => assert_eq!(msg, "unexpected failure"),
            other => panic!("expected Other, got {other:?}"),
        }
    }
}
