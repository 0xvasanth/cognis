use serde::{Deserialize, Serialize};

use super::base::{BaseMessageFields, MessageContent};

/// A system message that sets the behavior of the AI.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SystemMessage {
    #[serde(flatten)]
    pub base: BaseMessageFields,
}

impl SystemMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::Text(content.into())),
        }
    }
}
