use serde::{Deserialize, Serialize};

use super::base::{BaseMessageFields, MessageContent};

/// A message from a human user.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HumanMessage {
    #[serde(flatten)]
    pub base: BaseMessageFields,
}

impl HumanMessage {
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::Text(content.into())),
        }
    }

    pub fn with_blocks(blocks: Vec<super::content::ContentBlock>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::Blocks(blocks)),
        }
    }
}
