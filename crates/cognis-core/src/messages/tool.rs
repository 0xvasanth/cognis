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

    pub fn with_artifact(mut self, artifact: Value) -> Self {
        self.artifact = Some(artifact);
        self
    }
}
