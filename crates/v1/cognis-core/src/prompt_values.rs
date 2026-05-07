use serde::{Deserialize, Serialize};

use crate::messages::{HumanMessage, Message};

/// Base trait for inputs to any language model.
pub trait PromptValue: Send + Sync {
    /// Return prompt value as a string.
    fn to_string(&self) -> String;

    /// Return prompt as a list of messages.
    fn to_messages(&self) -> Vec<Message>;
}

/// A simple string prompt value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StringPromptValue {
    pub text: String,
}

impl StringPromptValue {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl PromptValue for StringPromptValue {
    fn to_string(&self) -> String {
        self.text.clone()
    }

    fn to_messages(&self) -> Vec<Message> {
        vec![Message::Human(HumanMessage::new(&self.text))]
    }
}

/// A chat prompt value built from messages.
#[derive(Debug, Clone)]
pub struct ChatPromptValue {
    pub messages: Vec<Message>,
}

impl ChatPromptValue {
    pub fn new(messages: Vec<Message>) -> Self {
        Self { messages }
    }
}

impl PromptValue for ChatPromptValue {
    fn to_string(&self) -> String {
        self.messages
            .iter()
            .map(|m| {
                let typ = m.message_type().as_str();
                let content = m.content().text();
                format!("{}: {}", typ, content)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn to_messages(&self) -> Vec<Message> {
        self.messages.clone()
    }
}

/// An image prompt value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImagePromptValue {
    pub url: String,
    pub detail: Option<String>,
}

impl ImagePromptValue {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            detail: None,
        }
    }
}

impl PromptValue for ImagePromptValue {
    fn to_string(&self) -> String {
        self.url.clone()
    }

    fn to_messages(&self) -> Vec<Message> {
        vec![Message::Human(HumanMessage::new(&self.url))]
    }
}
