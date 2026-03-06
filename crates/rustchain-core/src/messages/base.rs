use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

use super::content::ContentBlock;

/// The type of a message.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    Human,
    Ai,
    System,
    Tool,
    Function,
    Chat,
    Remove,
}

impl MessageType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Human => "human",
            Self::Ai => "ai",
            Self::System => "system",
            Self::Tool => "tool",
            Self::Function => "function",
            Self::Chat => "chat",
            Self::Remove => "remove",
        }
    }
}

/// Message content can be plain text or a list of content blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

impl Default for MessageContent {
    fn default() -> Self {
        Self::Text(String::new())
    }
}

impl MessageContent {
    /// Extract all text content as a single string.
    pub fn text(&self) -> String {
        match self {
            Self::Text(s) => s.clone(),
            Self::Blocks(blocks) => blocks
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Merge two message contents.
    ///
    /// Follows the same semantics as Python's `merge_content`:
    /// - Text + Text → concatenated Text
    /// - Text + Blocks → Blocks with text prepended
    /// - Blocks + Blocks → Blocks merged (last text extended if applicable)
    /// - Blocks + Text → last text block extended, or new text block appended
    pub fn merge(&self, other: &MessageContent) -> MessageContent {
        match (self, other) {
            (Self::Text(a), Self::Text(b)) => Self::Text(format!("{}{}", a, b)),
            (Self::Text(a), Self::Blocks(b)) => {
                let mut blocks = vec![ContentBlock::text_only(a.clone())];
                blocks.extend(b.iter().cloned());
                Self::Blocks(blocks)
            }
            (Self::Blocks(a), Self::Blocks(b)) => {
                let mut merged = a.clone();
                for block in b {
                    // Try to merge adjacent text blocks
                    if let ContentBlock::Text { text, .. } = block {
                        if let Some(ContentBlock::Text { text: last_text, .. }) = merged.last_mut() {
                            last_text.push_str(text);
                            continue;
                        }
                    }
                    merged.push(block.clone());
                }
                Self::Blocks(merged)
            }
            (Self::Blocks(a), Self::Text(b)) => {
                if b.is_empty() {
                    return Self::Blocks(a.clone());
                }
                let mut merged = a.clone();
                if let Some(ContentBlock::Text { text: last_text, .. }) = merged.last_mut() {
                    last_text.push_str(b);
                } else {
                    merged.push(ContentBlock::text_only(b.clone()));
                }
                Self::Blocks(merged)
            }
        }
    }
}

/// Common fields shared by all message types.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BaseMessageFields {
    pub content: MessageContent,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub additional_kwargs: HashMap<String, Value>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub response_metadata: HashMap<String, Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
}

impl BaseMessageFields {
    pub fn new(content: MessageContent) -> Self {
        Self {
            content,
            additional_kwargs: HashMap::new(),
            response_metadata: HashMap::new(),
            name: None,
            id: None,
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }
}

/// Merge multiple message contents into one.
///
/// Convenience wrapper around `MessageContent::merge`.
pub fn merge_content(first: &MessageContent, rest: &[MessageContent]) -> MessageContent {
    let mut result = first.clone();
    for content in rest {
        result = result.merge(content);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_merge_text_text() {
        let a = MessageContent::Text("hello ".into());
        let b = MessageContent::Text("world".into());
        assert_eq!(a.merge(&b), MessageContent::Text("hello world".into()));
    }

    #[test]
    fn test_merge_text_blocks() {
        let a = MessageContent::Text("prefix ".into());
        let b = MessageContent::Blocks(vec![ContentBlock::text_only("suffix")]);
        if let MessageContent::Blocks(blocks) = a.merge(&b) {
            assert_eq!(blocks.len(), 2);
        } else {
            panic!("Expected Blocks");
        }
    }

    #[test]
    fn test_merge_blocks_text() {
        let a = MessageContent::Blocks(vec![ContentBlock::text_only("hello")]);
        let b = MessageContent::Text(" world".into());
        if let MessageContent::Blocks(blocks) = a.merge(&b) {
            assert_eq!(blocks.len(), 1);
            if let ContentBlock::Text { text, .. } = &blocks[0] {
                assert_eq!(text, "hello world");
            }
        } else {
            panic!("Expected Blocks");
        }
    }

    #[test]
    fn test_merge_blocks_blocks_adjacent_text() {
        let a = MessageContent::Blocks(vec![ContentBlock::text_only("hello")]);
        let b = MessageContent::Blocks(vec![ContentBlock::text_only(" world")]);
        if let MessageContent::Blocks(blocks) = a.merge(&b) {
            assert_eq!(blocks.len(), 1);
            if let ContentBlock::Text { text, .. } = &blocks[0] {
                assert_eq!(text, "hello world");
            }
        } else {
            panic!("Expected Blocks");
        }
    }

    #[test]
    fn test_merge_content_multiple() {
        let a = MessageContent::Text("a".into());
        let b = MessageContent::Text("b".into());
        let c = MessageContent::Text("c".into());
        let result = merge_content(&a, &[b, c]);
        assert_eq!(result, MessageContent::Text("abc".into()));
    }
}
