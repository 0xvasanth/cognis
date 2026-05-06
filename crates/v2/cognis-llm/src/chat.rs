//! Chat-completion request/response types shared across providers.

use serde::{Deserialize, Serialize};

/// Per-call options that override `ClientConfig` defaults.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatOptions {
    /// Model identifier override (e.g. "gpt-4o", "llama3.2").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// Sampling temperature [0.0, 2.0].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    /// Top-p nucleus sampling [0.0, 1.0].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    /// Max tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    /// Stop sequences.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stop: Vec<String>,
    /// Frequency penalty [-2.0, 2.0].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty [-2.0, 2.0].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
}

/// Response payload from `chat_completion`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatResponse {
    /// The assistant's reply.
    pub message: crate::Message,
    /// Token usage if the provider reports it.
    pub usage: Option<Usage>,
    /// Why generation stopped ("stop", "length", "tool_calls", …).
    pub finish_reason: String,
    /// Model that produced this response.
    pub model: String,
}

/// Token usage report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Usage {
    /// Tokens in the prompt.
    pub prompt_tokens: u32,
    /// Tokens in the completion.
    pub completion_tokens: u32,
    /// Total tokens.
    pub total_tokens: u32,
}

/// One chunk of a streaming response.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamChunk {
    /// Text delta (may be empty for tool-call chunks).
    pub content: String,
    /// True if this chunk is a partial token, false if it's a complete unit.
    pub is_delta: bool,
    /// True if this is the terminal chunk.
    pub is_done: bool,
    /// Reason the stream terminated, only set when is_done=true.
    pub finish_reason: Option<String>,
    /// Final usage stats, only set when is_done=true.
    pub usage: Option<Usage>,
    /// Tool-call deltas accumulated this chunk.
    #[serde(default)]
    pub tool_calls_delta: Vec<ToolCallDelta>,
}

/// Partial tool-call info streamed across chunks. Streams accumulate these
/// into complete tool calls at the framework boundary.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolCallDelta {
    /// Index in the assistant message's tool_calls array.
    pub index: u32,
    /// ID (only on the first chunk of this call).
    pub id: Option<String>,
    /// Function name (only on the first chunk).
    pub name: Option<String>,
    /// Argument fragment (concatenated across chunks).
    pub arguments_delta: Option<String>,
}

/// Provider connectivity probe result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Reachable and responding.
    Healthy {
        /// Round-trip latency in milliseconds.
        latency_ms: u64,
    },
    /// Reachable but slow / partial.
    Degraded {
        /// Why it's degraded.
        reason: String,
    },
    /// Not reachable.
    Unhealthy {
        /// Why it's unhealthy.
        reason: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Message;

    #[test]
    fn chat_options_default_minimal() {
        let opts = ChatOptions::default();
        let s = serde_json::to_string(&opts).unwrap();
        // Empty options serialize to an empty object — model/temp/etc. are skipped when None.
        assert_eq!(s, "{}");
    }

    #[test]
    fn chat_response_roundtrip() {
        let r = ChatResponse {
            message: Message::ai("hello"),
            usage: Some(Usage { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 }),
            finish_reason: "stop".into(),
            model: "gpt-4o".into(),
        };
        let s = serde_json::to_string(&r).unwrap();
        let back: ChatResponse = serde_json::from_str(&s).unwrap();
        assert_eq!(back.finish_reason, "stop");
        assert_eq!(back.usage.unwrap().total_tokens, 15);
    }
}
