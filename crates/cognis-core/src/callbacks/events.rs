//! Struct payloads for tool-lifecycle callback events.

use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// Emitted when a tool invocation begins.
#[derive(Debug, Clone)]
pub struct ToolStartEvent {
    /// Name of the tool being invoked.
    pub tool: String,
    /// Serialized tool metadata (name + arguments schema).
    pub serialized: Value,
    /// JSON-stringified tool arguments, suitable for logs.
    pub input_str: String,
    /// Structured tool arguments as a `Value`.
    pub inputs: Value,
    /// Identifier from the AI message's `tool_call`, if present.
    pub tool_call_id: Option<String>,
    /// Unique identifier for this tool invocation.
    pub run_id: Uuid,
    /// Identifier of the enclosing run (e.g., chain or agent).
    pub parent_run_id: Option<Uuid>,
    /// Tags propagated from the run configuration.
    pub tags: Vec<String>,
    /// Run-scoped metadata.
    pub metadata: HashMap<String, Value>,
}

/// Emitted when a tool invocation completes successfully.
///
/// `output_str` is the stringified form given to the LLM scratchpad.
/// `output_value` preserves the original JSON shape so UIs can render
/// typed payloads without re-parsing. `artifact` is populated only when
/// the tool returned `ToolOutput::ContentAndArtifact`.
#[derive(Debug, Clone)]
pub struct ToolEndEvent {
    /// Name of the tool that completed.
    pub tool: String,
    /// Stringified form of the output; this is what the LLM scratchpad sees.
    pub output_str: String,
    /// Original JSON shape of the output, preserved for UI consumers that render typed payloads.
    pub output_value: Value,
    /// Structured side-channel payload from tools that return `ToolOutput::ContentAndArtifact`;
    /// `None` for plain `ToolOutput::Content`.
    pub artifact: Option<Value>,
    /// Identifier from the AI message's `tool_call`, if present.
    pub tool_call_id: Option<String>,
    /// Unique identifier for this tool invocation.
    pub run_id: Uuid,
    /// Identifier of the enclosing run.
    pub parent_run_id: Option<Uuid>,
}

/// Emitted when a tool invocation fails.
#[derive(Debug, Clone)]
pub struct ToolErrorEvent {
    /// Name of the tool that failed.
    pub tool: String,
    /// Display-formatted error message.
    pub error: String,
    /// Structured discriminator for programmatic handling.
    pub error_kind: ToolErrorKind,
    /// Identifier from the AI message's `tool_call`, if present.
    pub tool_call_id: Option<String>,
    /// Unique identifier for this tool invocation.
    pub run_id: Uuid,
    /// Identifier of the enclosing run.
    pub parent_run_id: Option<Uuid>,
}

/// Structured discriminator for tool-invocation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolErrorKind {
    /// Tool name did not resolve in the executor's registry.
    NotFound,
    /// Tool `_run` returned an error (e.g., `CognisError::ToolException` or
    /// `CognisError::ToolValidationError`).
    Execution,
    /// Any other failure (serde errors, panics surfaced as errors, etc.).
    Other,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_end_event_is_clone_debug() {
        let event = ToolEndEvent {
            tool: "search".into(),
            output_str: "\"hit\"".into(),
            output_value: Value::String("hit".into()),
            artifact: None,
            tool_call_id: Some("call_1".into()),
            run_id: Uuid::new_v4(),
            parent_run_id: None,
        };
        let cloned = event.clone();
        assert_eq!(cloned.tool, "search");
        // Debug impl should not panic.
        let _ = format!("{event:?}");
    }

    #[test]
    fn tool_error_kind_equality() {
        assert_eq!(ToolErrorKind::NotFound, ToolErrorKind::NotFound);
        assert_ne!(ToolErrorKind::NotFound, ToolErrorKind::Execution);
    }
}
