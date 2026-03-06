use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::base::{BaseMessageFields, MessageContent};
use super::tool::ToolStatus;
use super::tool_types::ToolCallChunk;
use crate::utils::merge_dicts;

/// Trait for chunk types that support concatenation during streaming.
pub trait MessageChunkTrait: Sized {
    /// Concatenate two chunks of the same type.
    fn add(self, other: Self) -> Self;
}

/// Merge the base fields (content, additional_kwargs, response_metadata) from `other` into `base`.
fn merge_base_fields(base: &mut BaseMessageFields, other: &BaseMessageFields) {
    let combined = format!("{}{}", base.content.text(), other.content.text());
    base.content = MessageContent::Text(combined);

    // Merge additional_kwargs
    if !other.additional_kwargs.is_empty() {
        let left = serde_json::to_value(&base.additional_kwargs).unwrap_or_default();
        let right = serde_json::to_value(&other.additional_kwargs).unwrap_or_default();
        let merged = merge_dicts(&left, &[&right]).unwrap_or(left);
        if let Ok(map) = serde_json::from_value(merged) {
            base.additional_kwargs = map;
        }
    }

    // Merge response_metadata
    if !other.response_metadata.is_empty() {
        let left = serde_json::to_value(&base.response_metadata).unwrap_or_default();
        let right = serde_json::to_value(&other.response_metadata).unwrap_or_default();
        let merged = merge_dicts(&left, &[&right]).unwrap_or(left);
        if let Ok(map) = serde_json::from_value(merged) {
            base.response_metadata = map;
        }
    }
}

/// A streaming chunk of a human message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HumanMessageChunk {
    #[serde(flatten)]
    pub base: BaseMessageFields,
}

impl HumanMessageChunk {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::Text(content.into())),
        }
    }
}

impl MessageChunkTrait for HumanMessageChunk {
    fn add(mut self, other: Self) -> Self {
        merge_base_fields(&mut self.base, &other.base);
        self
    }
}

/// A streaming chunk of a system message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemMessageChunk {
    #[serde(flatten)]
    pub base: BaseMessageFields,
}

impl SystemMessageChunk {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::Text(content.into())),
        }
    }
}

impl MessageChunkTrait for SystemMessageChunk {
    fn add(mut self, other: Self) -> Self {
        merge_base_fields(&mut self.base, &other.base);
        self
    }
}

/// A streaming chunk of a chat message with an explicit role.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessageChunk {
    pub role: String,
    #[serde(flatten)]
    pub base: BaseMessageFields,
}

impl ChatMessageChunk {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            base: BaseMessageFields::new(MessageContent::Text(content.into())),
        }
    }
}

impl MessageChunkTrait for ChatMessageChunk {
    fn add(mut self, other: Self) -> Self {
        assert_eq!(
            self.role, other.role,
            "Cannot concatenate ChatMessageChunks with different roles: '{}' and '{}'",
            self.role, other.role
        );
        merge_base_fields(&mut self.base, &other.base);
        self
    }
}

/// A streaming chunk of a legacy function message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionMessageChunk {
    #[serde(flatten)]
    pub base: BaseMessageFields,
}

impl FunctionMessageChunk {
    pub fn new(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::Text(content.into())).with_name(name),
        }
    }

    pub fn name(&self) -> Option<&str> {
        self.base.name.as_deref()
    }
}

impl MessageChunkTrait for FunctionMessageChunk {
    fn add(mut self, other: Self) -> Self {
        assert_eq!(
            self.base.name, other.base.name,
            "Cannot concatenate FunctionMessageChunks with different names: '{:?}' and '{:?}'",
            self.base.name, other.base.name
        );
        merge_base_fields(&mut self.base, &other.base);
        self
    }
}

/// A streaming chunk of a tool message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolMessageChunk {
    #[serde(flatten)]
    pub base: BaseMessageFields,
    pub tool_call_id: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_call_chunks: Vec<ToolCallChunk>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<Value>,
    #[serde(default)]
    pub status: ToolStatus,
}

impl ToolMessageChunk {
    pub fn new(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::Text(content.into())),
            tool_call_id: tool_call_id.into(),
            tool_call_chunks: Vec::new(),
            artifact: None,
            status: ToolStatus::Success,
        }
    }
}

/// Merge two artifact values. Strings concatenate, objects merge, arrays merge.
fn merge_artifact(left: Option<Value>, right: Option<Value>) -> Option<Value> {
    match (left, right) {
        (None, r) => r,
        (l, None) => l,
        (Some(Value::String(l)), Some(Value::String(r))) => {
            Some(Value::String(format!("{}{}", l, r)))
        }
        (Some(l @ Value::Object(_)), Some(r @ Value::Object(_))) => {
            Some(merge_dicts(&l, &[&r]).unwrap_or(l))
        }
        (Some(Value::Array(mut l)), Some(Value::Array(r))) => {
            l.extend(r);
            Some(Value::Array(l))
        }
        (Some(l), Some(_)) => Some(l), // fallback: keep left
    }
}

/// Merge two statuses: error is sticky.
fn merge_status(left: ToolStatus, right: ToolStatus) -> ToolStatus {
    if left == ToolStatus::Error || right == ToolStatus::Error {
        ToolStatus::Error
    } else {
        ToolStatus::Success
    }
}

impl MessageChunkTrait for ToolMessageChunk {
    fn add(mut self, other: Self) -> Self {
        assert_eq!(
            self.tool_call_id, other.tool_call_id,
            "Cannot concatenate ToolMessageChunks with different tool_call_ids: '{}' and '{}'",
            self.tool_call_id, other.tool_call_id
        );
        merge_base_fields(&mut self.base, &other.base);
        self.tool_call_chunks.extend(other.tool_call_chunks);
        self.artifact = merge_artifact(self.artifact, other.artifact);
        self.status = merge_status(self.status, other.status);
        self
    }
}

/// A signal to remove a message by ID from a message list.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoveMessage {
    pub id: String,
}

impl RemoveMessage {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}
