//! Message types for LLM conversations.
//!
//! `Message` is a tagged enum. Each variant carries a small content struct;
//! the variants stay flat to keep pattern matching ergonomic.

use serde::{Deserialize, Serialize};

/// A single message in an LLM conversation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// User input.
    Human(HumanMessage),
    /// Assistant response.
    Ai(AiMessage),
    /// System prompt or instruction.
    System(SystemMessage),
    /// Tool execution result.
    Tool(ToolMessage),
}

/// Convenience constructors.
impl Message {
    /// Build a `Human` message from arbitrary content.
    pub fn human(content: impl Into<String>) -> Self {
        Self::Human(HumanMessage {
            content: content.into(),
        })
    }

    /// Build an `Ai` message with text only (no tool calls).
    pub fn ai(content: impl Into<String>) -> Self {
        Self::Ai(AiMessage {
            content: content.into(),
            tool_calls: Vec::new(),
        })
    }

    /// Build a `System` message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::System(SystemMessage {
            content: content.into(),
        })
    }

    /// Build a `Tool` message.
    pub fn tool(call_id: impl Into<String>, content: impl Into<String>) -> Self {
        Self::Tool(ToolMessage {
            tool_call_id: call_id.into(),
            content: content.into(),
        })
    }

    /// Get the message's primary text content (empty string for messages
    /// that are tool-call-only with no text).
    pub fn content(&self) -> &str {
        match self {
            Self::Human(m) => &m.content,
            Self::Ai(m) => &m.content,
            Self::System(m) => &m.content,
            Self::Tool(m) => &m.content,
        }
    }

    /// Returns the tool calls if this is an `Ai` message; empty otherwise.
    pub fn tool_calls(&self) -> &[ToolCall] {
        match self {
            Self::Ai(m) => &m.tool_calls,
            _ => &[],
        }
    }

    /// True if this is an `Ai` message with at least one tool call.
    pub fn has_tool_calls(&self) -> bool {
        matches!(self, Self::Ai(m) if !m.tool_calls.is_empty())
    }
}

/// A human/user message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HumanMessage {
    /// The message text.
    pub content: String,
}

/// An AI/assistant message, optionally carrying tool call requests.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct AiMessage {
    /// The message text.
    pub content: String,
    /// Tool calls requested by the model (omitted from JSON when empty).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
}

/// A system prompt or instruction message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemMessage {
    /// The system prompt text.
    pub content: String,
}

/// A tool execution result message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolMessage {
    /// The ID of the tool call this result corresponds to.
    pub tool_call_id: String,
    /// The result content.
    pub content: String,
}

/// One tool invocation requested by the LLM in an `AiMessage`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    /// Provider-assigned ID (used to match tool results back to calls).
    pub id: String,
    /// Tool name as registered with the LLM.
    pub name: String,
    /// Arguments — opaque JSON, deserialized by the tool.
    pub arguments: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convenience_constructors() {
        assert_eq!(Message::human("hi").content(), "hi");
        assert_eq!(Message::ai("hello").content(), "hello");
        assert_eq!(Message::system("be terse").content(), "be terse");
        let t = Message::tool("call_1", "result");
        assert_eq!(t.content(), "result");
        if let Message::Tool(tm) = t {
            assert_eq!(tm.tool_call_id, "call_1");
        }
    }

    #[test]
    fn tool_calls_accessor() {
        let m = Message::ai("none here");
        assert!(m.tool_calls().is_empty());
        assert!(!m.has_tool_calls());

        let m = Message::Ai(AiMessage {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: "c".into(),
                name: "search".into(),
                arguments: serde_json::json!({"q": "rust"}),
            }],
        });
        assert_eq!(m.tool_calls().len(), 1);
        assert!(m.has_tool_calls());
    }

    #[test]
    fn roundtrip_serde() {
        let m = Message::human("hi");
        let s = serde_json::to_string(&m).unwrap();
        let back: Message = serde_json::from_str(&s).unwrap();
        assert_eq!(m, back);
        assert!(s.contains("\"role\":\"human\""));
    }
}
