use serde::{Deserialize, Serialize};

use super::base::{BaseMessageFields, MessageContent};
use super::multimodal::ContentPart;

/// A message from a human user.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HumanMessage {
    #[serde(flatten)]
    pub base: BaseMessageFields,
}

impl HumanMessage {
    /// Create a text-only human message.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::Text(content.into())),
        }
    }

    /// Create a human message from content blocks.
    pub fn with_blocks(blocks: Vec<super::content::ContentBlock>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::Blocks(blocks)),
        }
    }

    /// Create a multi-modal human message from content parts.
    ///
    /// # Example
    ///
    /// ```
    /// use cognis_core::messages::{HumanMessage, ContentPart};
    ///
    /// let msg = HumanMessage::from_parts(vec![
    ///     ContentPart::text("What is in this image?"),
    ///     ContentPart::image_url("https://example.com/photo.jpg"),
    /// ]);
    /// assert!(msg.base.content.is_multimodal());
    /// ```
    pub fn from_parts(parts: Vec<ContentPart>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::from_parts(parts)),
        }
    }

    /// Create a human message with text and an image URL.
    pub fn with_image_url(text: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::with_image_url(text, url)),
        }
    }

    /// Create a human message with text and a base64-encoded image.
    pub fn with_image_base64(
        text: impl Into<String>,
        base64: impl Into<String>,
        mime_type: impl Into<String>,
    ) -> Self {
        Self {
            base: BaseMessageFields::new(MessageContent::with_image_base64(
                text, base64, mime_type,
            )),
        }
    }
}
