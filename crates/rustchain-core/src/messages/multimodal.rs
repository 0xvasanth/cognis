//! Multi-modal message content support for text + images.
//!
//! Provides [`ContentPart`], [`ImageDetail`], and convenience constructors
//! for building multi-modal messages compatible with OpenAI's content format.
//!
//! These types offer a simplified API on top of the existing [`ContentBlock`]
//! system, making it easy to construct messages with mixed text and image
//! content.

use serde::{Deserialize, Serialize};

use super::base::MessageContent;
use super::content::{ContentBlock, ImageUrlInfo};

/// Image detail level for vision models.
///
/// Controls the resolution at which the model processes the image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ImageDetail {
    /// Let the model decide the appropriate detail level.
    Auto,
    /// Use a low-resolution representation (faster, fewer tokens).
    Low,
    /// Use a high-resolution representation (more detailed, more tokens).
    High,
}

impl ImageDetail {
    /// Return the detail level as a string slice.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Low => "low",
            Self::High => "high",
        }
    }
}

impl std::fmt::Display for ImageDetail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A single part of a multi-modal message.
///
/// This is a simplified representation that maps to OpenAI's content part
/// format. Each variant can be converted to a [`ContentBlock`] for use in
/// the existing message system.
///
/// # Serialization
///
/// Serializes to OpenAI's multi-modal format:
/// ```json
/// {"type": "text", "text": "Describe this image"}
/// {"type": "image_url", "image_url": {"url": "https://...", "detail": "auto"}}
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text content.
    Text {
        /// The text content.
        text: String,
    },
    /// An image referenced by URL (including data URIs for base64).
    ImageUrl {
        /// The image URL information.
        image_url: ImageUrlContent,
    },
}

/// Image URL content with optional detail level.
///
/// Used within [`ContentPart::ImageUrl`] to hold the URL and detail setting.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ImageUrlContent {
    /// The URL of the image. Can be an HTTP(S) URL or a `data:` URI.
    pub url: String,
    /// Optional detail level for vision processing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<ImageDetail>,
}

impl ContentPart {
    /// Create a text content part.
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    /// Create an image URL content part.
    pub fn image_url(url: impl Into<String>) -> Self {
        Self::ImageUrl {
            image_url: ImageUrlContent {
                url: url.into(),
                detail: None,
            },
        }
    }

    /// Create an image URL content part with a specific detail level.
    pub fn image_url_with_detail(url: impl Into<String>, detail: ImageDetail) -> Self {
        Self::ImageUrl {
            image_url: ImageUrlContent {
                url: url.into(),
                detail: Some(detail),
            },
        }
    }

    /// Create an image content part from base64-encoded data and MIME type.
    ///
    /// This constructs a `data:` URI from the base64 data and MIME type,
    /// then wraps it as an [`ContentPart::ImageUrl`].
    pub fn image_base64(base64: impl Into<String>, mime_type: impl Into<String>) -> Self {
        let b64 = base64.into();
        let mime = mime_type.into();
        let data_uri = format!("data:{};base64,{}", mime, b64);
        Self::ImageUrl {
            image_url: ImageUrlContent {
                url: data_uri,
                detail: None,
            },
        }
    }

    /// Create an image content part from base64-encoded data with a detail level.
    pub fn image_base64_with_detail(
        base64: impl Into<String>,
        mime_type: impl Into<String>,
        detail: ImageDetail,
    ) -> Self {
        let b64 = base64.into();
        let mime = mime_type.into();
        let data_uri = format!("data:{};base64,{}", mime, b64);
        Self::ImageUrl {
            image_url: ImageUrlContent {
                url: data_uri,
                detail: Some(detail),
            },
        }
    }

    /// Convert this content part to a [`ContentBlock`].
    pub fn to_content_block(&self) -> ContentBlock {
        match self {
            Self::Text { text } => ContentBlock::text_only(text.clone()),
            Self::ImageUrl { image_url } => ContentBlock::ImageUrl {
                image_url: ImageUrlInfo {
                    url: image_url.url.clone(),
                    detail: image_url.detail.map(|d| d.as_str().to_string()),
                },
                extras: None,
            },
        }
    }
}

// ── MessageContent extensions ───────────────────────────────────────────

impl MessageContent {
    /// Create a text-only `MessageContent`.
    pub fn new_text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    /// Create a multi-modal `MessageContent` with text and an image URL.
    pub fn with_image_url(text: impl Into<String>, url: impl Into<String>) -> Self {
        Self::Blocks(vec![
            ContentBlock::text_only(text.into()),
            ContentBlock::ImageUrl {
                image_url: ImageUrlInfo {
                    url: url.into(),
                    detail: None,
                },
                extras: None,
            },
        ])
    }

    /// Create a multi-modal `MessageContent` with text and a base64-encoded image.
    pub fn with_image_base64(
        text: impl Into<String>,
        base64: impl Into<String>,
        mime_type: impl Into<String>,
    ) -> Self {
        let b64 = base64.into();
        let mime = mime_type.into();
        let data_uri = format!("data:{};base64,{}", mime, b64);
        Self::Blocks(vec![
            ContentBlock::text_only(text.into()),
            ContentBlock::ImageUrl {
                image_url: ImageUrlInfo {
                    url: data_uri,
                    detail: None,
                },
                extras: None,
            },
        ])
    }

    /// Create a `MessageContent` from a list of [`ContentPart`]s.
    pub fn from_parts(parts: Vec<ContentPart>) -> Self {
        Self::Blocks(parts.iter().map(|p| p.to_content_block()).collect())
    }

    /// Check whether this content contains non-text parts (images, etc.).
    pub fn is_multimodal(&self) -> bool {
        match self {
            Self::Text(_) => false,
            Self::Blocks(blocks) => blocks
                .iter()
                .any(|b| !matches!(b, ContentBlock::Text { .. })),
        }
    }

    /// Return the text content as a string reference.
    ///
    /// For `Text` variants, returns the inner string directly.
    /// For `Blocks` variants, concatenates all text blocks.
    ///
    /// Note: For `Blocks`, this allocates a new `String` each call.
    /// If you only need owned text, use [`text()`](Self::text) instead.
    pub fn content_as_text(&self) -> String {
        self.text()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Text-only MessageContent ────────────────────────────────────────

    #[test]
    fn test_text_only_message_content() {
        let content = MessageContent::new_text("Hello, world!");
        assert_eq!(content, MessageContent::Text("Hello, world!".to_string()));
        assert_eq!(content.text(), "Hello, world!");
    }

    // ── Multi-part with text + image URL ────────────────────────────────

    #[test]
    fn test_multipart_text_and_image_url() {
        let content =
            MessageContent::with_image_url("Describe this image", "https://example.com/cat.png");
        match &content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                match &blocks[0] {
                    ContentBlock::Text { text, .. } => assert_eq!(text, "Describe this image"),
                    other => panic!("Expected Text block, got {:?}", other),
                }
                match &blocks[1] {
                    ContentBlock::ImageUrl { image_url, .. } => {
                        assert_eq!(image_url.url, "https://example.com/cat.png");
                        assert!(image_url.detail.is_none());
                    }
                    other => panic!("Expected ImageUrl block, got {:?}", other),
                }
            }
            other => panic!("Expected Blocks, got {:?}", other),
        }
    }

    // ── Multi-part with text + base64 image ─────────────────────────────

    #[test]
    fn test_multipart_text_and_base64_image() {
        let content =
            MessageContent::with_image_base64("What is in this photo?", "aGVsbG8=", "image/png");
        match &content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                match &blocks[1] {
                    ContentBlock::ImageUrl { image_url, .. } => {
                        assert_eq!(image_url.url, "data:image/png;base64,aGVsbG8=");
                    }
                    other => panic!("Expected ImageUrl block, got {:?}", other),
                }
            }
            other => panic!("Expected Blocks, got {:?}", other),
        }
    }

    // ── Serialization round-trip ────────────────────────────────────────

    #[test]
    fn test_content_part_serialization_roundtrip() {
        let parts = vec![
            ContentPart::text("Hello"),
            ContentPart::image_url("https://example.com/img.png"),
        ];

        let json = serde_json::to_string(&parts).unwrap();
        let deserialized: Vec<ContentPart> = serde_json::from_str(&json).unwrap();
        assert_eq!(parts, deserialized);
    }

    #[test]
    fn test_content_part_text_serialization_format() {
        let part = ContentPart::text("Hello");
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "text");
        assert_eq!(json["text"], "Hello");
    }

    #[test]
    fn test_content_part_image_url_serialization_format() {
        let part =
            ContentPart::image_url_with_detail("https://example.com/img.png", ImageDetail::Auto);
        let json = serde_json::to_value(&part).unwrap();
        assert_eq!(json["type"], "image_url");
        assert_eq!(json["image_url"]["url"], "https://example.com/img.png");
        assert_eq!(json["image_url"]["detail"], "auto");
    }

    // ── content_as_text for text-only ───────────────────────────────────

    #[test]
    fn test_content_as_text_text_only() {
        let content = MessageContent::new_text("Simple text");
        assert_eq!(content.content_as_text(), "Simple text");
    }

    // ── content_as_text for multi-part ──────────────────────────────────

    #[test]
    fn test_content_as_text_multipart() {
        let content =
            MessageContent::with_image_url("Describe this", "https://example.com/img.png");
        // Should return only the text parts concatenated
        assert_eq!(content.content_as_text(), "Describe this");
    }

    // ── is_multimodal checks ────────────────────────────────────────────

    #[test]
    fn test_is_multimodal_text_only() {
        let content = MessageContent::new_text("Just text");
        assert!(!content.is_multimodal());
    }

    #[test]
    fn test_is_multimodal_with_image() {
        let content = MessageContent::with_image_url("Check this", "https://example.com/img.png");
        assert!(content.is_multimodal());
    }

    #[test]
    fn test_is_multimodal_blocks_text_only() {
        let content = MessageContent::Blocks(vec![
            ContentBlock::text_only("Just"),
            ContentBlock::text_only(" text blocks"),
        ]);
        assert!(!content.is_multimodal());
    }

    // ── HumanMessage with multi-modal content ───────────────────────────

    #[test]
    fn test_human_message_with_multimodal_content() {
        use crate::messages::HumanMessage;

        let msg = HumanMessage::with_blocks(vec![
            ContentBlock::text_only("What is this?"),
            ContentBlock::ImageUrl {
                image_url: ImageUrlInfo {
                    url: "https://example.com/photo.jpg".to_string(),
                    detail: Some("high".to_string()),
                },
                extras: None,
            },
        ]);

        assert!(msg.base.content.is_multimodal());
        assert_eq!(msg.base.content.content_as_text(), "What is this?");
    }

    // ── ImageDetail serialization ───────────────────────────────────────

    #[test]
    fn test_image_detail_serialization() {
        assert_eq!(
            serde_json::to_string(&ImageDetail::Auto).unwrap(),
            "\"auto\""
        );
        assert_eq!(serde_json::to_string(&ImageDetail::Low).unwrap(), "\"low\"");
        assert_eq!(
            serde_json::to_string(&ImageDetail::High).unwrap(),
            "\"high\""
        );

        let auto: ImageDetail = serde_json::from_str("\"auto\"").unwrap();
        assert_eq!(auto, ImageDetail::Auto);
        let low: ImageDetail = serde_json::from_str("\"low\"").unwrap();
        assert_eq!(low, ImageDetail::Low);
        let high: ImageDetail = serde_json::from_str("\"high\"").unwrap();
        assert_eq!(high, ImageDetail::High);
    }

    // ── Multiple images in one message ──────────────────────────────────

    #[test]
    fn test_multiple_images_in_one_message() {
        let content = MessageContent::from_parts(vec![
            ContentPart::text("Compare these images"),
            ContentPart::image_url("https://example.com/img1.png"),
            ContentPart::image_url("https://example.com/img2.png"),
        ]);

        match &content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 3);
                // First is text
                assert!(matches!(&blocks[0], ContentBlock::Text { .. }));
                // Second and third are images
                assert!(matches!(&blocks[1], ContentBlock::ImageUrl { .. }));
                assert!(matches!(&blocks[2], ContentBlock::ImageUrl { .. }));
            }
            other => panic!("Expected Blocks, got {:?}", other),
        }
        assert!(content.is_multimodal());
        assert_eq!(content.content_as_text(), "Compare these images");
    }

    // ── Backwards compatibility ─────────────────────────────────────────

    #[test]
    fn test_backwards_compatibility_human_message_new() {
        use crate::messages::HumanMessage;

        // Existing API must still work unchanged
        let msg = HumanMessage::new("Hello, world!");
        assert_eq!(msg.base.content.text(), "Hello, world!");
        assert!(!msg.base.content.is_multimodal());
    }

    // ── ContentPart::image_base64 ───────────────────────────────────────

    #[test]
    fn test_content_part_image_base64() {
        let part = ContentPart::image_base64("aGVsbG8=", "image/jpeg");
        match &part {
            ContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "data:image/jpeg;base64,aGVsbG8=");
                assert!(image_url.detail.is_none());
            }
            other => panic!("Expected ImageUrl, got {:?}", other),
        }
    }

    #[test]
    fn test_content_part_image_base64_with_detail() {
        let part =
            ContentPart::image_base64_with_detail("aGVsbG8=", "image/png", ImageDetail::High);
        match &part {
            ContentPart::ImageUrl { image_url } => {
                assert_eq!(image_url.url, "data:image/png;base64,aGVsbG8=");
                assert_eq!(image_url.detail, Some(ImageDetail::High));
            }
            other => panic!("Expected ImageUrl, got {:?}", other),
        }
    }

    // ── from_parts ──────────────────────────────────────────────────────

    #[test]
    fn test_from_parts_conversion() {
        let content = MessageContent::from_parts(vec![
            ContentPart::text("Hello"),
            ContentPart::image_url_with_detail("https://img.com/a.png", ImageDetail::Low),
        ]);

        match &content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                match &blocks[1] {
                    ContentBlock::ImageUrl { image_url, .. } => {
                        assert_eq!(image_url.url, "https://img.com/a.png");
                        assert_eq!(image_url.detail.as_deref(), Some("low"));
                    }
                    other => panic!("Expected ImageUrl, got {:?}", other),
                }
            }
            other => panic!("Expected Blocks, got {:?}", other),
        }
    }
}
