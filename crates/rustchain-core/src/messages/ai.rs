use serde::{Deserialize, Serialize};

use super::base::{BaseMessageFields, MessageContent};
use super::tool_types::{InvalidToolCall, ToolCall, ToolCallChunk};
use crate::utils::merge_dicts;

/// Token usage details for input tokens.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct InputTokenDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_creation: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_read: Option<u64>,
}

/// Token usage details for output tokens.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct OutputTokenDetails {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub audio: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<u64>,
}

/// Token usage metadata for a message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct UsageMetadata {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub total_tokens: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_token_details: Option<InputTokenDetails>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_token_details: Option<OutputTokenDetails>,
}

impl UsageMetadata {
    pub fn new(input_tokens: u64, output_tokens: u64, total_tokens: u64) -> Self {
        Self {
            input_tokens,
            output_tokens,
            total_tokens,
            input_token_details: None,
            output_token_details: None,
        }
    }

    /// Add two UsageMetadata values together.
    pub fn add(&self, other: &UsageMetadata) -> UsageMetadata {
        UsageMetadata {
            input_tokens: self.input_tokens + other.input_tokens,
            output_tokens: self.output_tokens + other.output_tokens,
            total_tokens: self.total_tokens + other.total_tokens,
            input_token_details: match (&self.input_token_details, &other.input_token_details) {
                (Some(l), Some(r)) => Some(InputTokenDetails {
                    audio: add_optional(l.audio, r.audio),
                    cache_creation: add_optional(l.cache_creation, r.cache_creation),
                    cache_read: add_optional(l.cache_read, r.cache_read),
                }),
                (Some(d), None) | (None, Some(d)) => Some(d.clone()),
                (None, None) => None,
            },
            output_token_details: match (&self.output_token_details, &other.output_token_details) {
                (Some(l), Some(r)) => Some(OutputTokenDetails {
                    audio: add_optional(l.audio, r.audio),
                    reasoning: add_optional(l.reasoning, r.reasoning),
                }),
                (Some(d), None) | (None, Some(d)) => Some(d.clone()),
                (None, None) => None,
            },
        }
    }
}

fn add_optional(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x + y),
        (Some(x), None) | (None, Some(x)) => Some(x),
        (None, None) => None,
    }
}

fn sub_optional(a: Option<u64>, b: Option<u64>) -> Option<u64> {
    match (a, b) {
        (Some(x), Some(y)) => Some(x.saturating_sub(y)),
        (Some(x), None) => Some(x),
        (None, Some(_)) => Some(0),
        (None, None) => None,
    }
}

impl UsageMetadata {
    /// Subtract another UsageMetadata from this one (saturating).
    pub fn subtract(&self, other: &UsageMetadata) -> UsageMetadata {
        UsageMetadata {
            input_tokens: self.input_tokens.saturating_sub(other.input_tokens),
            output_tokens: self.output_tokens.saturating_sub(other.output_tokens),
            total_tokens: self.total_tokens.saturating_sub(other.total_tokens),
            input_token_details: match (&self.input_token_details, &other.input_token_details) {
                (Some(l), Some(r)) => Some(InputTokenDetails {
                    audio: sub_optional(l.audio, r.audio),
                    cache_creation: sub_optional(l.cache_creation, r.cache_creation),
                    cache_read: sub_optional(l.cache_read, r.cache_read),
                }),
                (Some(d), None) => Some(d.clone()),
                (None, _) => None,
            },
            output_token_details: match (&self.output_token_details, &other.output_token_details) {
                (Some(l), Some(r)) => Some(OutputTokenDetails {
                    audio: sub_optional(l.audio, r.audio),
                    reasoning: sub_optional(l.reasoning, r.reasoning),
                }),
                (Some(d), None) => Some(d.clone()),
                (None, _) => None,
            },
        }
    }
}

/// Free function to add two UsageMetadata values.
pub fn add_usage(a: &UsageMetadata, b: &UsageMetadata) -> UsageMetadata {
    a.add(b)
}

/// A message from an AI model.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AIMessage {
    #[serde(flatten)]
    pub base: BaseMessageFields,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub invalid_tool_calls: Vec<InvalidToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>,
}

impl AIMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::Text(content.into())),
            tool_calls: Vec::new(),
            invalid_tool_calls: Vec::new(),
            usage_metadata: None,
        }
    }

    pub fn with_tool_calls(mut self, tool_calls: Vec<ToolCall>) -> Self {
        self.tool_calls = tool_calls;
        self
    }

    pub fn with_usage(mut self, usage: UsageMetadata) -> Self {
        self.usage_metadata = Some(usage);
        self
    }
}

/// An AI message chunk yielded during streaming.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AIMessageChunk {
    #[serde(flatten)]
    pub base: BaseMessageFields,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub invalid_tool_calls: Vec<InvalidToolCall>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_call_chunks: Vec<ToolCallChunk>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage_metadata: Option<UsageMetadata>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk_position: Option<String>,
}

impl AIMessageChunk {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::Text(content.into())),
            tool_calls: Vec::new(),
            invalid_tool_calls: Vec::new(),
            tool_call_chunks: Vec::new(),
            usage_metadata: None,
            chunk_position: None,
        }
    }

    /// Concatenate two AI message chunks, merging content, additional_kwargs,
    /// response_metadata, tool_call_chunks, and usage_metadata.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, other: Self) -> Self {
        let combined = format!("{}{}", self.base.content.text(), other.base.content.text());
        self.base.content = MessageContent::Text(combined);

        // Merge additional_kwargs
        if !other.base.additional_kwargs.is_empty() {
            let left = serde_json::to_value(&self.base.additional_kwargs).unwrap_or_default();
            let right = serde_json::to_value(&other.base.additional_kwargs).unwrap_or_default();
            let merged = merge_dicts(&left, &[&right]).unwrap_or(left);
            if let Ok(map) = serde_json::from_value(merged) {
                self.base.additional_kwargs = map;
            }
        }

        // Merge response_metadata
        if !other.base.response_metadata.is_empty() {
            let left = serde_json::to_value(&self.base.response_metadata).unwrap_or_default();
            let right = serde_json::to_value(&other.base.response_metadata).unwrap_or_default();
            let merged = merge_dicts(&left, &[&right]).unwrap_or(left);
            if let Ok(map) = serde_json::from_value(merged) {
                self.base.response_metadata = map;
            }
        }

        self.tool_calls.extend(other.tool_calls);
        self.invalid_tool_calls.extend(other.invalid_tool_calls);
        self.tool_call_chunks.extend(other.tool_call_chunks);
        self.usage_metadata = match (self.usage_metadata, other.usage_metadata) {
            (Some(a), Some(b)) => Some(a.add(&b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        };
        // chunk_position: "last" is sticky (OR logic)
        if other.chunk_position.as_deref() == Some("last") || self.chunk_position.is_none() {
            self.chunk_position = other.chunk_position;
        }
        self
    }
}
