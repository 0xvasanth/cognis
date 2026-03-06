use serde::{Deserialize, Serialize};

use super::base::{BaseMessageFields, MessageContent};

/// A legacy function message (deprecated in favor of ToolMessage).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionMessage {
    #[serde(flatten)]
    pub base: BaseMessageFields,
}

impl FunctionMessage {
    pub fn new(name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::Text(content.into())).with_name(name),
        }
    }
}
