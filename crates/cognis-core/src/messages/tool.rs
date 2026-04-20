use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::base::{BaseMessageFields, MessageContent};

/// Status of a tool invocation.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolStatus {
    #[default]
    Success,
    Error,
}

/// A message containing the result of a tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolMessage {
    #[serde(flatten)]
    pub base: BaseMessageFields,
    pub tool_call_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<Value>,
    #[serde(default)]
    pub status: ToolStatus,
}

impl ToolMessage {
    pub fn new(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::Text(content.into())),
            tool_call_id: tool_call_id.into(),
            artifact: None,
            status: ToolStatus::Success,
        }
    }

    pub fn with_error(mut self) -> Self {
        self.status = ToolStatus::Error;
        self
    }

    /// Constructor that also sets the structured artifact payload.
    ///
    /// Use when the tool returned `ToolOutput::ContentAndArtifact`: `content`
    /// goes to the LLM scratchpad, `artifact` is the structured payload for
    /// UIs and callback consumers.
    pub fn with_artifact(
        content: impl Into<String>,
        tool_call_id: impl Into<String>,
        artifact: Value,
    ) -> Self {
        let mut msg = Self::new(content, tool_call_id);
        msg.artifact = Some(artifact);
        msg
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn with_artifact_sets_the_field() {
        let artifact = json!({"chart": [1, 2, 3]});
        let msg = ToolMessage::with_artifact("summary text", "call_abc", artifact.clone());
        assert_eq!(msg.base.content.text(), "summary text");
        assert_eq!(msg.tool_call_id, "call_abc");
        assert_eq!(msg.artifact, Some(artifact));
    }

    #[test]
    fn new_leaves_artifact_none() {
        let msg = ToolMessage::new("just text", "call_abc");
        assert_eq!(msg.artifact, None);
    }
}
