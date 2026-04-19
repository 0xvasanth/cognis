//! Struct payloads for tool-lifecycle callback events.

use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

/// Emitted when a tool invocation begins.
#[derive(Debug, Clone)]
pub struct ToolStartEvent {
    pub tool: String,
    pub serialized: Value,
    pub input_str: String,
    pub inputs: Value,
    pub tool_call_id: Option<String>,
    pub run_id: Uuid,
    pub parent_run_id: Option<Uuid>,
    pub tags: Vec<String>,
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
    pub tool: String,
    pub output_str: String,
    pub output_value: Value,
    pub artifact: Option<Value>,
    pub tool_call_id: Option<String>,
    pub run_id: Uuid,
    pub parent_run_id: Option<Uuid>,
}

/// Emitted when a tool invocation fails.
#[derive(Debug, Clone)]
pub struct ToolErrorEvent {
    pub tool: String,
    pub error: String,
    pub error_kind: ToolErrorKind,
    pub tool_call_id: Option<String>,
    pub run_id: Uuid,
    pub parent_run_id: Option<Uuid>,
}

/// Structured discriminator for tool-invocation errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolErrorKind {
    /// Tool name did not resolve in the executor's registry.
    NotFound,
    /// Tool `_run` returned `Err(ToolException | ToolValidationError)`.
    Execution,
    /// Any other failure (serde, panics surfaced as errors, etc.).
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
