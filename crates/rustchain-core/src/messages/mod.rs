pub mod content;
pub mod base;
pub mod tool_types;
pub mod human;
pub mod ai;
pub mod system;
pub mod tool;
pub mod function;
pub mod chat;
pub mod chunks;
pub mod openai;
pub mod utils;
pub mod block_translators;

use serde::{Deserialize, Serialize};

pub use self::ai::{AIMessage, AIMessageChunk, InputTokenDetails, OutputTokenDetails, UsageMetadata, add_usage};
pub use self::base::{BaseMessageFields, MessageContent, MessageType, merge_content};
pub use self::chat::ChatMessage;
pub use self::chunks::{
    ChatMessageChunk, FunctionMessageChunk, HumanMessageChunk, MessageChunkTrait, RemoveMessage,
    SystemMessageChunk, ToolMessageChunk,
};
pub use self::content::{
    Annotation, Citation, ContentBlock, ImageUrlInfo,
    KNOWN_BLOCK_TYPES, is_data_content_block_type,
};
pub use self::function::FunctionMessage;
pub use self::human::HumanMessage;
pub use self::system::SystemMessage;
pub use self::tool::{ToolMessage, ToolStatus};
pub use self::openai::{convert_to_openai_messages, count_tokens_approximately};
pub use self::tool_types::{
    default_tool_chunk_parser, default_tool_parser, invalid_tool_call, tool_call, tool_call_chunk,
    InvalidToolCall, ToolCall, ToolCallChunk,
};
pub use self::block_translators::{
    BlockTranslator, OpenAIBlockTranslator, AnthropicBlockTranslator,
    GoogleGenAIBlockTranslator, get_translator,
};
pub use self::utils::{
    convert_to_messages, convert_to_messages_flex, filter_messages, filter_messages_full,
    get_buffer_string, get_buffer_string_full, merge_message_runs, merge_message_runs_with_separator,
    message_chunk_to_message, messages_from_dict, messages_to_dict, trim_messages, trim_messages_full,
    MessageLike, TrimStrategy,
};

/// Unified message enum for dispatching across message types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Message {
    Human(HumanMessage),
    Ai(AIMessage),
    System(SystemMessage),
    Tool(ToolMessage),
    Function(FunctionMessage),
    Chat(ChatMessage),
    HumanChunk(HumanMessageChunk),
    AiChunk(AIMessageChunk),
    SystemChunk(SystemMessageChunk),
    ToolChunk(ToolMessageChunk),
    FunctionChunk(FunctionMessageChunk),
    ChatChunk(ChatMessageChunk),
    Remove(RemoveMessage),
}

impl Message {
    pub fn message_type(&self) -> MessageType {
        match self {
            Self::Human(_) | Self::HumanChunk(_) => MessageType::Human,
            Self::Ai(_) | Self::AiChunk(_) => MessageType::Ai,
            Self::System(_) | Self::SystemChunk(_) => MessageType::System,
            Self::Tool(_) | Self::ToolChunk(_) => MessageType::Tool,
            Self::Function(_) | Self::FunctionChunk(_) => MessageType::Function,
            Self::Chat(_) | Self::ChatChunk(_) => MessageType::Chat,
            Self::Remove(_) => MessageType::Remove,
        }
    }

    pub fn content(&self) -> &MessageContent {
        match self {
            Self::Human(m) => &m.base.content,
            Self::Ai(m) => &m.base.content,
            Self::System(m) => &m.base.content,
            Self::Tool(m) => &m.base.content,
            Self::Function(m) => &m.base.content,
            Self::Chat(m) => &m.base.content,
            Self::HumanChunk(m) => &m.base.content,
            Self::AiChunk(m) => &m.base.content,
            Self::SystemChunk(m) => &m.base.content,
            Self::ToolChunk(m) => &m.base.content,
            Self::FunctionChunk(m) => &m.base.content,
            Self::ChatChunk(m) => &m.base.content,
            Self::Remove(_) => {
                // RemoveMessage has no content; return a static empty reference
                static EMPTY: MessageContent = MessageContent::Text(String::new());
                &EMPTY
            }
        }
    }

    /// Create a human message.
    pub fn human(content: impl Into<String>) -> Self {
        Self::Human(HumanMessage::new(content))
    }

    /// Create an AI message.
    pub fn ai(content: impl Into<String>) -> Self {
        Self::Ai(AIMessage::new(content))
    }

    /// Create an AI message with tool calls.
    pub fn ai_with_tool_calls(content: impl Into<String>, tool_calls: Vec<serde_json::Value>) -> Self {
        let mut msg = AIMessage::new(content);
        msg.tool_calls = tool_calls
            .into_iter()
            .filter_map(|tc| serde_json::from_value(tc).ok())
            .collect();
        Self::Ai(msg)
    }

    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self::System(SystemMessage::new(content))
    }

    /// Create a tool message.
    pub fn tool(content: impl Into<String>, tool_call_id: impl Into<String>) -> Self {
        Self::Tool(ToolMessage::new(content, tool_call_id))
    }

    /// Get a pretty representation of the message.
    ///
    /// Produces output like:
    /// ```text
    /// ================================ Human Message =================================
    ///
    /// What is the capital of France?
    /// ```
    pub fn pretty_repr(&self) -> String {
        let type_name = match self.message_type() {
            MessageType::Human => "Human",
            MessageType::Ai => "Ai",
            MessageType::System => "System",
            MessageType::Tool => "Tool",
            MessageType::Function => "Function",
            MessageType::Chat => "Chat",
            MessageType::Remove => "Remove",
        };
        let title = get_msg_title_repr(&format!("{} Message", type_name));
        let name_line = self
            .base()
            .and_then(|b| b.name.as_ref())
            .map(|n| format!("\nName: {}", n))
            .unwrap_or_default();
        format!("{}{}\n\n{}", title, name_line, self.content().text())
    }

    /// Print a pretty representation of the message to stdout.
    pub fn pretty_print(&self) {
        println!("{}", self.pretty_repr());
    }

    pub fn base(&self) -> Option<&BaseMessageFields> {
        match self {
            Self::Human(m) => Some(&m.base),
            Self::Ai(m) => Some(&m.base),
            Self::System(m) => Some(&m.base),
            Self::Tool(m) => Some(&m.base),
            Self::Function(m) => Some(&m.base),
            Self::Chat(m) => Some(&m.base),
            Self::HumanChunk(m) => Some(&m.base),
            Self::AiChunk(m) => Some(&m.base),
            Self::SystemChunk(m) => Some(&m.base),
            Self::ToolChunk(m) => Some(&m.base),
            Self::FunctionChunk(m) => Some(&m.base),
            Self::ChatChunk(m) => Some(&m.base),
            Self::Remove(_) => None,
        }
    }
}

impl std::fmt::Display for Message {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let type_name = self.message_type().as_str();
        let content = self.content().text();
        let truncated = if content.len() > 100 {
            format!("{}...", &content[..100])
        } else {
            content
        };
        write!(f, "{}({})", type_name, truncated)
    }
}

/// Get a title representation for a message (e.g., "== Human Message ==").
///
/// Centers the title in a line of 80 `=` characters.
pub fn get_msg_title_repr(title: &str) -> String {
    let padded = format!(" {} ", title);
    let sep_len = (80usize.saturating_sub(padded.len())) / 2;
    let sep: String = "=".repeat(sep_len);
    let second_sep = if padded.len() % 2 == 1 {
        format!("{}=", sep)
    } else {
        sep.clone()
    };
    format!("{}{}{}", sep, padded, second_sep)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_msg_title_repr_length() {
        let title = get_msg_title_repr("Human Message");
        assert_eq!(title.len(), 80);
        assert!(title.contains(" Human Message "));
        assert!(title.starts_with('='));
        assert!(title.ends_with('='));
    }

    #[test]
    fn test_get_msg_title_repr_various() {
        let ai = get_msg_title_repr("Ai Message");
        assert_eq!(ai.len(), 80);
        assert!(ai.contains(" Ai Message "));

        let system = get_msg_title_repr("System Message");
        assert_eq!(system.len(), 80);
    }

    #[test]
    fn test_pretty_repr_human() {
        let msg = Message::human("Hello world");
        let repr = msg.pretty_repr();
        assert!(repr.contains("Human Message"));
        assert!(repr.contains("Hello world"));
    }

    #[test]
    fn test_pretty_repr_ai() {
        let msg = Message::ai("The answer is 42");
        let repr = msg.pretty_repr();
        assert!(repr.contains("Ai Message"));
        assert!(repr.contains("The answer is 42"));
    }

    #[test]
    fn test_pretty_repr_with_name() {
        let mut human = HumanMessage::new("Question");
        human.base.name = Some("Alice".to_string());
        let msg = Message::Human(human);
        let repr = msg.pretty_repr();
        assert!(repr.contains("Name: Alice"));
        assert!(repr.contains("Question"));
    }
}
