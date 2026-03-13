use serde::{Deserialize, Serialize};

use super::base::{BaseMessageFields, MessageContent};

/// A chat message with an explicit role string.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ChatMessage {
    pub role: String,
    #[serde(flatten)]
    pub base: BaseMessageFields,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            base: BaseMessageFields::new(MessageContent::Text(content.into())),
        }
    }
}
