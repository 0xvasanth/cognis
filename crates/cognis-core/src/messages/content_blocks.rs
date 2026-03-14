//! Standardized multimodal content blocks for Large Language Model I/O.
//!
//! This module provides structured data types for representing multimodal content
//! (text, images, audio, video, PDFs, citations, and annotations) in a unified,
//! provider-agnostic format. Each block carries an auto-generated UUID for
//! identification and tracking.
//!
//! # Factory functions
//!
//! Convenience constructors are provided for every block type:
//!
//! ```
//! use cognis_core::messages::content_blocks::*;
//!
//! let t = text_block("Hello, world!");
//! let img = image_block_url("https://example.com/cat.png", "image/png");
//! ```

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Source enum – shared between image, audio, video, and PDF blocks
// ---------------------------------------------------------------------------

/// Describes where the binary data for a media block comes from.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "source_type", rename_all = "snake_case")]
pub enum MediaSource {
    /// A URL pointing to the resource.
    Url {
        /// The URL of the resource.
        url: String,
    },
    /// Base64-encoded inline data.
    Base64 {
        /// The base64-encoded data.
        data: String,
    },
}

// ---------------------------------------------------------------------------
// Struct-based content blocks
// ---------------------------------------------------------------------------

/// A block of plain text content.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TextContentBlock {
    /// Unique identifier for this block.
    pub id: String,
    /// The text content.
    pub text: String,
}

/// A block representing image data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageContentBlock {
    /// Unique identifier for this block.
    pub id: String,
    /// Source of the image data.
    pub source: MediaSource,
    /// MIME type of the image (e.g. `"image/png"`).
    pub media_type: String,
    /// Alternative text description for accessibility.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alt_text: Option<String>,
}

/// A block representing audio data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AudioContentBlock {
    /// Unique identifier for this block.
    pub id: String,
    /// Source of the audio data.
    pub source: MediaSource,
    /// MIME type of the audio (e.g. `"audio/mp3"`).
    pub media_type: String,
    /// Optional transcript of the audio content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub transcript: Option<String>,
}

/// A block representing video data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VideoContentBlock {
    /// Unique identifier for this block.
    pub id: String,
    /// Source of the video data.
    pub source: MediaSource,
    /// MIME type of the video (e.g. `"video/mp4"`).
    pub media_type: String,
}

/// A block representing a PDF document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PdfContentBlock {
    /// Unique identifier for this block.
    pub id: String,
    /// Source of the PDF data.
    pub source: MediaSource,
    /// Optional page range to consider (e.g. `"1-5"`, `"3,7,10-12"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub page_range: Option<String>,
}

/// A citation referencing a source document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CitationBlock {
    /// Identifier of the source being cited.
    pub source_id: String,
    /// The quoted text from the source.
    pub quote: String,
    /// Start character index of the citation in the response text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_index: Option<usize>,
    /// End character index of the citation in the response text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_index: Option<usize>,
}

/// An annotation attached to content, carrying arbitrary typed data.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AnnotationBlock {
    /// Unique identifier for this annotation.
    pub id: String,
    /// The kind of annotation (e.g. `"highlight"`, `"comment"`, `"footnote"`).
    pub annotation_type: String,
    /// Arbitrary JSON payload associated with the annotation.
    pub data: Value,
}

// ---------------------------------------------------------------------------
// Unified ContentBlock enum
// ---------------------------------------------------------------------------

/// A single content block in a multimodal message.
///
/// Each variant wraps a strongly-typed struct, making it straightforward to
/// pattern-match and extract data. The `Custom` variant allows provider-specific
/// extensions without breaking the standard structure.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "block_type", rename_all = "snake_case")]
pub enum MultimodalContentBlock {
    /// Plain text content.
    Text(TextContentBlock),
    /// Image data (URL or base64).
    Image(ImageContentBlock),
    /// Audio data (URL or base64).
    Audio(AudioContentBlock),
    /// Video data (URL or base64).
    Video(VideoContentBlock),
    /// PDF document data (URL or base64).
    Pdf(PdfContentBlock),
    /// A citation referencing source material.
    Citation(CitationBlock),
    /// An annotation on content.
    Annotation(AnnotationBlock),
    /// Provider-specific or custom content not covered by the standard types.
    Custom {
        /// An identifier for the custom block type.
        custom_type: String,
        /// Arbitrary JSON payload.
        data: Value,
    },
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

/// Create a new unique block ID.
fn new_block_id() -> String {
    Uuid::new_v4().to_string()
}

/// Create a [`MultimodalContentBlock::Text`] block.
pub fn text_block(text: impl Into<String>) -> MultimodalContentBlock {
    MultimodalContentBlock::Text(TextContentBlock {
        id: new_block_id(),
        text: text.into(),
    })
}

/// Create a [`MultimodalContentBlock::Image`] block from a URL.
pub fn image_block_url(
    url: impl Into<String>,
    media_type: impl Into<String>,
) -> MultimodalContentBlock {
    MultimodalContentBlock::Image(ImageContentBlock {
        id: new_block_id(),
        source: MediaSource::Url {
            url: url.into(),
        },
        media_type: media_type.into(),
        alt_text: None,
    })
}

/// Create a [`MultimodalContentBlock::Image`] block from base64 data.
pub fn image_block_base64(
    data: impl Into<String>,
    media_type: impl Into<String>,
) -> MultimodalContentBlock {
    MultimodalContentBlock::Image(ImageContentBlock {
        id: new_block_id(),
        source: MediaSource::Base64 {
            data: data.into(),
        },
        media_type: media_type.into(),
        alt_text: None,
    })
}

/// Create a [`MultimodalContentBlock::Image`] block with alt text from a URL.
pub fn image_block_url_with_alt(
    url: impl Into<String>,
    media_type: impl Into<String>,
    alt_text: impl Into<String>,
) -> MultimodalContentBlock {
    MultimodalContentBlock::Image(ImageContentBlock {
        id: new_block_id(),
        source: MediaSource::Url {
            url: url.into(),
        },
        media_type: media_type.into(),
        alt_text: Some(alt_text.into()),
    })
}

/// Create a [`MultimodalContentBlock::Audio`] block from a URL.
pub fn audio_block_url(
    url: impl Into<String>,
    media_type: impl Into<String>,
) -> MultimodalContentBlock {
    MultimodalContentBlock::Audio(AudioContentBlock {
        id: new_block_id(),
        source: MediaSource::Url {
            url: url.into(),
        },
        media_type: media_type.into(),
        transcript: None,
    })
}

/// Create a [`MultimodalContentBlock::Audio`] block from base64 data.
pub fn audio_block_base64(
    data: impl Into<String>,
    media_type: impl Into<String>,
) -> MultimodalContentBlock {
    MultimodalContentBlock::Audio(AudioContentBlock {
        id: new_block_id(),
        source: MediaSource::Base64 {
            data: data.into(),
        },
        media_type: media_type.into(),
        transcript: None,
    })
}

/// Create a [`MultimodalContentBlock::Audio`] block from a URL with a transcript.
pub fn audio_block_url_with_transcript(
    url: impl Into<String>,
    media_type: impl Into<String>,
    transcript: impl Into<String>,
) -> MultimodalContentBlock {
    MultimodalContentBlock::Audio(AudioContentBlock {
        id: new_block_id(),
        source: MediaSource::Url {
            url: url.into(),
        },
        media_type: media_type.into(),
        transcript: Some(transcript.into()),
    })
}

/// Create a [`MultimodalContentBlock::Video`] block from a URL.
pub fn video_block_url(
    url: impl Into<String>,
    media_type: impl Into<String>,
) -> MultimodalContentBlock {
    MultimodalContentBlock::Video(VideoContentBlock {
        id: new_block_id(),
        source: MediaSource::Url {
            url: url.into(),
        },
        media_type: media_type.into(),
    })
}

/// Create a [`MultimodalContentBlock::Video`] block from base64 data.
pub fn video_block_base64(
    data: impl Into<String>,
    media_type: impl Into<String>,
) -> MultimodalContentBlock {
    MultimodalContentBlock::Video(VideoContentBlock {
        id: new_block_id(),
        source: MediaSource::Base64 {
            data: data.into(),
        },
        media_type: media_type.into(),
    })
}

/// Create a [`MultimodalContentBlock::Pdf`] block from a URL.
pub fn pdf_block_url(url: impl Into<String>) -> MultimodalContentBlock {
    MultimodalContentBlock::Pdf(PdfContentBlock {
        id: new_block_id(),
        source: MediaSource::Url {
            url: url.into(),
        },
        page_range: None,
    })
}

/// Create a [`MultimodalContentBlock::Pdf`] block from base64 data.
pub fn pdf_block_base64(data: impl Into<String>) -> MultimodalContentBlock {
    MultimodalContentBlock::Pdf(PdfContentBlock {
        id: new_block_id(),
        source: MediaSource::Base64 {
            data: data.into(),
        },
        page_range: None,
    })
}

/// Create a [`MultimodalContentBlock::Pdf`] block from a URL with a page range.
pub fn pdf_block_url_with_pages(
    url: impl Into<String>,
    page_range: impl Into<String>,
) -> MultimodalContentBlock {
    MultimodalContentBlock::Pdf(PdfContentBlock {
        id: new_block_id(),
        source: MediaSource::Url {
            url: url.into(),
        },
        page_range: Some(page_range.into()),
    })
}

/// Create a [`MultimodalContentBlock::Citation`] block.
pub fn citation_block(
    source_id: impl Into<String>,
    quote: impl Into<String>,
) -> MultimodalContentBlock {
    MultimodalContentBlock::Citation(CitationBlock {
        source_id: source_id.into(),
        quote: quote.into(),
        start_index: None,
        end_index: None,
    })
}

/// Create a [`MultimodalContentBlock::Citation`] block with index range.
pub fn citation_block_with_indices(
    source_id: impl Into<String>,
    quote: impl Into<String>,
    start_index: usize,
    end_index: usize,
) -> MultimodalContentBlock {
    MultimodalContentBlock::Citation(CitationBlock {
        source_id: source_id.into(),
        quote: quote.into(),
        start_index: Some(start_index),
        end_index: Some(end_index),
    })
}

/// Create a [`MultimodalContentBlock::Annotation`] block.
pub fn annotation_block(
    annotation_type: impl Into<String>,
    data: Value,
) -> MultimodalContentBlock {
    MultimodalContentBlock::Annotation(AnnotationBlock {
        id: new_block_id(),
        annotation_type: annotation_type.into(),
        data,
    })
}

/// Create a [`MultimodalContentBlock::Custom`] block.
pub fn custom_block(
    custom_type: impl Into<String>,
    data: Value,
) -> MultimodalContentBlock {
    MultimodalContentBlock::Custom {
        custom_type: custom_type.into(),
        data,
    }
}

// ---------------------------------------------------------------------------
// ContentBlockList wrapper
// ---------------------------------------------------------------------------

/// An ordered list of [`MultimodalContentBlock`]s with convenience query methods.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ContentBlockList {
    blocks: Vec<MultimodalContentBlock>,
}

impl ContentBlockList {
    /// Create an empty list.
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Create a list from an existing vector of blocks.
    pub fn from_blocks(blocks: Vec<MultimodalContentBlock>) -> Self {
        Self { blocks }
    }

    /// Push a block onto the end of the list.
    pub fn push(&mut self, block: MultimodalContentBlock) {
        self.blocks.push(block);
    }

    /// Return the number of blocks.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Return `true` if there are no blocks.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Return a slice of all blocks.
    pub fn blocks(&self) -> &[MultimodalContentBlock] {
        &self.blocks
    }

    /// Consume the list and return the inner vector.
    pub fn into_blocks(self) -> Vec<MultimodalContentBlock> {
        self.blocks
    }

    /// Return an iterator over the blocks.
    pub fn iter(&self) -> std::slice::Iter<'_, MultimodalContentBlock> {
        self.blocks.iter()
    }

    // -- Query helpers --

    /// Return `true` if the list contains only text blocks.
    pub fn text_only(&self) -> bool {
        !self.blocks.is_empty()
            && self
                .blocks
                .iter()
                .all(|b| matches!(b, MultimodalContentBlock::Text(_)))
    }

    /// Return `true` if the list contains at least one image block.
    pub fn has_images(&self) -> bool {
        self.blocks
            .iter()
            .any(|b| matches!(b, MultimodalContentBlock::Image(_)))
    }

    /// Return `true` if the list contains at least one audio block.
    pub fn has_audio(&self) -> bool {
        self.blocks
            .iter()
            .any(|b| matches!(b, MultimodalContentBlock::Audio(_)))
    }

    /// Return `true` if the list contains at least one video block.
    pub fn has_video(&self) -> bool {
        self.blocks
            .iter()
            .any(|b| matches!(b, MultimodalContentBlock::Video(_)))
    }

    /// Return `true` if the list contains at least one PDF block.
    pub fn has_pdfs(&self) -> bool {
        self.blocks
            .iter()
            .any(|b| matches!(b, MultimodalContentBlock::Pdf(_)))
    }

    /// Return `true` if the list contains at least one citation block.
    pub fn has_citations(&self) -> bool {
        self.blocks
            .iter()
            .any(|b| matches!(b, MultimodalContentBlock::Citation(_)))
    }

    /// Return `true` if the list contains at least one annotation block.
    pub fn has_annotations(&self) -> bool {
        self.blocks
            .iter()
            .any(|b| matches!(b, MultimodalContentBlock::Annotation(_)))
    }

    /// Return `true` if the list contains any media blocks (image, audio, video, or PDF).
    pub fn has_media(&self) -> bool {
        self.blocks.iter().any(|b| {
            matches!(
                b,
                MultimodalContentBlock::Image(_)
                    | MultimodalContentBlock::Audio(_)
                    | MultimodalContentBlock::Video(_)
                    | MultimodalContentBlock::Pdf(_)
            )
        })
    }

    /// Extract all text blocks, returning their content concatenated with newlines.
    pub fn extract_text(&self) -> String {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                MultimodalContentBlock::Text(t) => Some(t.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Return only the text blocks.
    pub fn text_blocks(&self) -> Vec<&TextContentBlock> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                MultimodalContentBlock::Text(t) => Some(t),
                _ => None,
            })
            .collect()
    }

    /// Return only the image blocks.
    pub fn image_blocks(&self) -> Vec<&ImageContentBlock> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                MultimodalContentBlock::Image(i) => Some(i),
                _ => None,
            })
            .collect()
    }

    /// Return only the audio blocks.
    pub fn audio_blocks(&self) -> Vec<&AudioContentBlock> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                MultimodalContentBlock::Audio(a) => Some(a),
                _ => None,
            })
            .collect()
    }

    /// Return only the video blocks.
    pub fn video_blocks(&self) -> Vec<&VideoContentBlock> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                MultimodalContentBlock::Video(v) => Some(v),
                _ => None,
            })
            .collect()
    }

    /// Return only the PDF blocks.
    pub fn pdf_blocks(&self) -> Vec<&PdfContentBlock> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                MultimodalContentBlock::Pdf(p) => Some(p),
                _ => None,
            })
            .collect()
    }

    /// Return only the citation blocks.
    pub fn citation_blocks(&self) -> Vec<&CitationBlock> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                MultimodalContentBlock::Citation(c) => Some(c),
                _ => None,
            })
            .collect()
    }

    /// Return only the annotation blocks.
    pub fn annotation_blocks(&self) -> Vec<&AnnotationBlock> {
        self.blocks
            .iter()
            .filter_map(|b| match b {
                MultimodalContentBlock::Annotation(a) => Some(a),
                _ => None,
            })
            .collect()
    }
}

impl IntoIterator for ContentBlockList {
    type Item = MultimodalContentBlock;
    type IntoIter = std::vec::IntoIter<MultimodalContentBlock>;

    fn into_iter(self) -> Self::IntoIter {
        self.blocks.into_iter()
    }
}

impl<'a> IntoIterator for &'a ContentBlockList {
    type Item = &'a MultimodalContentBlock;
    type IntoIter = std::slice::Iter<'a, MultimodalContentBlock>;

    fn into_iter(self) -> Self::IntoIter {
        self.blocks.iter()
    }
}

impl From<Vec<MultimodalContentBlock>> for ContentBlockList {
    fn from(blocks: Vec<MultimodalContentBlock>) -> Self {
        Self { blocks }
    }
}

impl From<ContentBlockList> for Vec<MultimodalContentBlock> {
    fn from(list: ContentBlockList) -> Self {
        list.blocks
    }
}

// ---------------------------------------------------------------------------
// Helper trait implementations for MultimodalContentBlock
// ---------------------------------------------------------------------------

impl MultimodalContentBlock {
    /// Return `true` if this is a text block.
    pub fn is_text(&self) -> bool {
        matches!(self, Self::Text(_))
    }

    /// Return `true` if this is an image block.
    pub fn is_image(&self) -> bool {
        matches!(self, Self::Image(_))
    }

    /// Return `true` if this is an audio block.
    pub fn is_audio(&self) -> bool {
        matches!(self, Self::Audio(_))
    }

    /// Return `true` if this is a video block.
    pub fn is_video(&self) -> bool {
        matches!(self, Self::Video(_))
    }

    /// Return `true` if this is a PDF block.
    pub fn is_pdf(&self) -> bool {
        matches!(self, Self::Pdf(_))
    }

    /// Return `true` if this is a citation block.
    pub fn is_citation(&self) -> bool {
        matches!(self, Self::Citation(_))
    }

    /// Return `true` if this is an annotation block.
    pub fn is_annotation(&self) -> bool {
        matches!(self, Self::Annotation(_))
    }

    /// Return `true` if this is a custom block.
    pub fn is_custom(&self) -> bool {
        matches!(self, Self::Custom { .. })
    }

    /// Return the block type as a string.
    pub fn block_type_str(&self) -> &str {
        match self {
            Self::Text(_) => "text",
            Self::Image(_) => "image",
            Self::Audio(_) => "audio",
            Self::Video(_) => "video",
            Self::Pdf(_) => "pdf",
            Self::Citation(_) => "citation",
            Self::Annotation(_) => "annotation",
            Self::Custom { .. } => "custom",
        }
    }

    /// Attempt to extract plain text from the block. Returns `Some` only for text blocks.
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Self::Text(t) => Some(&t.text),
            _ => None,
        }
    }
}

impl std::fmt::Display for MultimodalContentBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Text(t) => write!(f, "Text({})", truncate_str(&t.text, 60)),
            Self::Image(i) => write!(f, "Image(media_type={})", i.media_type),
            Self::Audio(a) => write!(f, "Audio(media_type={})", a.media_type),
            Self::Video(v) => write!(f, "Video(media_type={})", v.media_type),
            Self::Pdf(p) => {
                let pages = p.page_range.as_deref().unwrap_or("all");
                write!(f, "Pdf(pages={})", pages)
            }
            Self::Citation(c) => {
                write!(f, "Citation(source={})", truncate_str(&c.source_id, 30))
            }
            Self::Annotation(a) => write!(f, "Annotation(type={})", a.annotation_type),
            Self::Custom { custom_type, .. } => write!(f, "Custom(type={})", custom_type),
        }
    }
}

/// Truncate a string for display purposes.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() > max_len {
        format!("{}...", &s[..max_len])
    } else {
        s.to_string()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- TextContentBlock tests --

    #[test]
    fn test_text_block_creation() {
        let block = text_block("Hello, world!");
        assert!(block.is_text());
        assert_eq!(block.as_text(), Some("Hello, world!"));
    }

    #[test]
    fn test_text_block_has_uuid() {
        let block = text_block("test");
        if let MultimodalContentBlock::Text(t) = &block {
            assert!(!t.id.is_empty());
            // Should be a valid UUID
            assert!(Uuid::parse_str(&t.id).is_ok());
        } else {
            panic!("expected text block");
        }
    }

    #[test]
    fn test_text_block_unique_ids() {
        let b1 = text_block("a");
        let b2 = text_block("b");
        if let (MultimodalContentBlock::Text(t1), MultimodalContentBlock::Text(t2)) = (&b1, &b2) {
            assert_ne!(t1.id, t2.id);
        }
    }

    #[test]
    fn test_text_block_display() {
        let block = text_block("short text");
        let display = format!("{}", block);
        assert!(display.contains("Text("));
        assert!(display.contains("short text"));
    }

    #[test]
    fn test_text_block_long_display_truncates() {
        let long = "a".repeat(100);
        let block = text_block(&long);
        let display = format!("{}", block);
        assert!(display.contains("..."));
    }

    // -- ImageContentBlock tests --

    #[test]
    fn test_image_block_url() {
        let block = image_block_url("https://example.com/img.png", "image/png");
        assert!(block.is_image());
        if let MultimodalContentBlock::Image(i) = &block {
            assert_eq!(i.media_type, "image/png");
            assert!(matches!(&i.source, MediaSource::Url { url } if url == "https://example.com/img.png"));
            assert!(i.alt_text.is_none());
        }
    }

    #[test]
    fn test_image_block_base64() {
        let block = image_block_base64("abc123==", "image/jpeg");
        if let MultimodalContentBlock::Image(i) = &block {
            assert!(matches!(&i.source, MediaSource::Base64 { data } if data == "abc123=="));
            assert_eq!(i.media_type, "image/jpeg");
        }
    }

    #[test]
    fn test_image_block_with_alt() {
        let block = image_block_url_with_alt("https://img.com/x.jpg", "image/jpeg", "A cat");
        if let MultimodalContentBlock::Image(i) = &block {
            assert_eq!(i.alt_text, Some("A cat".to_string()));
        }
    }

    #[test]
    fn test_image_block_has_uuid() {
        let block = image_block_url("https://img.com/x.png", "image/png");
        if let MultimodalContentBlock::Image(i) = &block {
            assert!(Uuid::parse_str(&i.id).is_ok());
        }
    }

    #[test]
    fn test_image_block_display() {
        let block = image_block_url("https://img.com/x.png", "image/png");
        let display = format!("{}", block);
        assert!(display.contains("image/png"));
    }

    // -- AudioContentBlock tests --

    #[test]
    fn test_audio_block_url() {
        let block = audio_block_url("https://example.com/audio.mp3", "audio/mp3");
        assert!(block.is_audio());
        if let MultimodalContentBlock::Audio(a) = &block {
            assert_eq!(a.media_type, "audio/mp3");
            assert!(a.transcript.is_none());
        }
    }

    #[test]
    fn test_audio_block_base64() {
        let block = audio_block_base64("base64data==", "audio/wav");
        if let MultimodalContentBlock::Audio(a) = &block {
            assert!(matches!(&a.source, MediaSource::Base64 { data } if data == "base64data=="));
        }
    }

    #[test]
    fn test_audio_block_with_transcript() {
        let block = audio_block_url_with_transcript(
            "https://example.com/speech.mp3",
            "audio/mp3",
            "Hello there",
        );
        if let MultimodalContentBlock::Audio(a) = &block {
            assert_eq!(a.transcript, Some("Hello there".to_string()));
        }
    }

    #[test]
    fn test_audio_block_has_uuid() {
        let block = audio_block_url("https://example.com/a.mp3", "audio/mp3");
        if let MultimodalContentBlock::Audio(a) = &block {
            assert!(Uuid::parse_str(&a.id).is_ok());
        }
    }

    // -- VideoContentBlock tests --

    #[test]
    fn test_video_block_url() {
        let block = video_block_url("https://example.com/video.mp4", "video/mp4");
        assert!(block.is_video());
        if let MultimodalContentBlock::Video(v) = &block {
            assert_eq!(v.media_type, "video/mp4");
        }
    }

    #[test]
    fn test_video_block_base64() {
        let block = video_block_base64("videodata==", "video/webm");
        if let MultimodalContentBlock::Video(v) = &block {
            assert!(matches!(&v.source, MediaSource::Base64 { data } if data == "videodata=="));
            assert_eq!(v.media_type, "video/webm");
        }
    }

    #[test]
    fn test_video_block_has_uuid() {
        let block = video_block_url("https://example.com/v.mp4", "video/mp4");
        if let MultimodalContentBlock::Video(v) = &block {
            assert!(Uuid::parse_str(&v.id).is_ok());
        }
    }

    #[test]
    fn test_video_block_display() {
        let block = video_block_url("https://example.com/v.mp4", "video/mp4");
        let display = format!("{}", block);
        assert!(display.contains("video/mp4"));
    }

    // -- PdfContentBlock tests --

    #[test]
    fn test_pdf_block_url() {
        let block = pdf_block_url("https://example.com/doc.pdf");
        assert!(block.is_pdf());
        if let MultimodalContentBlock::Pdf(p) = &block {
            assert!(p.page_range.is_none());
        }
    }

    #[test]
    fn test_pdf_block_base64() {
        let block = pdf_block_base64("pdfdata==");
        if let MultimodalContentBlock::Pdf(p) = &block {
            assert!(matches!(&p.source, MediaSource::Base64 { data } if data == "pdfdata=="));
        }
    }

    #[test]
    fn test_pdf_block_with_pages() {
        let block = pdf_block_url_with_pages("https://example.com/doc.pdf", "1-5");
        if let MultimodalContentBlock::Pdf(p) = &block {
            assert_eq!(p.page_range, Some("1-5".to_string()));
        }
    }

    #[test]
    fn test_pdf_block_has_uuid() {
        let block = pdf_block_url("https://example.com/doc.pdf");
        if let MultimodalContentBlock::Pdf(p) = &block {
            assert!(Uuid::parse_str(&p.id).is_ok());
        }
    }

    #[test]
    fn test_pdf_block_display() {
        let block = pdf_block_url_with_pages("https://example.com/doc.pdf", "3-7");
        let display = format!("{}", block);
        assert!(display.contains("3-7"));
    }

    #[test]
    fn test_pdf_block_display_all_pages() {
        let block = pdf_block_url("https://example.com/doc.pdf");
        let display = format!("{}", block);
        assert!(display.contains("all"));
    }

    // -- CitationBlock tests --

    #[test]
    fn test_citation_block() {
        let block = citation_block("src-1", "The earth is round.");
        assert!(block.is_citation());
        if let MultimodalContentBlock::Citation(c) = &block {
            assert_eq!(c.source_id, "src-1");
            assert_eq!(c.quote, "The earth is round.");
            assert!(c.start_index.is_none());
            assert!(c.end_index.is_none());
        }
    }

    #[test]
    fn test_citation_block_with_indices() {
        let block = citation_block_with_indices("src-2", "quote text", 10, 20);
        if let MultimodalContentBlock::Citation(c) = &block {
            assert_eq!(c.start_index, Some(10));
            assert_eq!(c.end_index, Some(20));
        }
    }

    #[test]
    fn test_citation_block_display() {
        let block = citation_block("my-source", "some quote");
        let display = format!("{}", block);
        assert!(display.contains("my-source"));
    }

    // -- AnnotationBlock tests --

    #[test]
    fn test_annotation_block() {
        let block = annotation_block("highlight", json!({"color": "yellow"}));
        assert!(block.is_annotation());
        if let MultimodalContentBlock::Annotation(a) = &block {
            assert_eq!(a.annotation_type, "highlight");
            assert_eq!(a.data, json!({"color": "yellow"}));
        }
    }

    #[test]
    fn test_annotation_block_has_uuid() {
        let block = annotation_block("comment", json!("test"));
        if let MultimodalContentBlock::Annotation(a) = &block {
            assert!(Uuid::parse_str(&a.id).is_ok());
        }
    }

    #[test]
    fn test_annotation_block_display() {
        let block = annotation_block("footnote", json!(null));
        let display = format!("{}", block);
        assert!(display.contains("footnote"));
    }

    // -- Custom block tests --

    #[test]
    fn test_custom_block() {
        let block = custom_block("provider_x_tool_result", json!({"output": 42}));
        assert!(block.is_custom());
        if let MultimodalContentBlock::Custom { custom_type, data } = &block {
            assert_eq!(custom_type, "provider_x_tool_result");
            assert_eq!(data, &json!({"output": 42}));
        }
    }

    #[test]
    fn test_custom_block_display() {
        let block = custom_block("my_type", json!(null));
        let display = format!("{}", block);
        assert!(display.contains("my_type"));
    }

    // -- Type predicate tests --

    #[test]
    fn test_is_text_false_for_image() {
        let block = image_block_url("https://img.com/a.png", "image/png");
        assert!(!block.is_text());
    }

    #[test]
    fn test_block_type_str() {
        assert_eq!(text_block("x").block_type_str(), "text");
        assert_eq!(
            image_block_url("u", "image/png").block_type_str(),
            "image"
        );
        assert_eq!(audio_block_url("u", "audio/mp3").block_type_str(), "audio");
        assert_eq!(video_block_url("u", "video/mp4").block_type_str(), "video");
        assert_eq!(pdf_block_url("u").block_type_str(), "pdf");
        assert_eq!(citation_block("s", "q").block_type_str(), "citation");
        assert_eq!(
            annotation_block("t", json!(null)).block_type_str(),
            "annotation"
        );
        assert_eq!(
            custom_block("c", json!(null)).block_type_str(),
            "custom"
        );
    }

    #[test]
    fn test_as_text_returns_none_for_non_text() {
        let block = image_block_url("u", "image/png");
        assert!(block.as_text().is_none());
    }

    // -- Serialization / Deserialization tests --

    #[test]
    fn test_text_block_serialize_deserialize() {
        let block = text_block("hello");
        let json = serde_json::to_string(&block).unwrap();
        let deserialized: MultimodalContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn test_image_block_serialize_deserialize() {
        let block = image_block_url_with_alt("https://img.com/x.jpg", "image/jpeg", "A dog");
        let json = serde_json::to_string(&block).unwrap();
        let deserialized: MultimodalContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn test_audio_block_serialize_deserialize() {
        let block = audio_block_url_with_transcript("https://a.com/s.mp3", "audio/mp3", "Hi");
        let json = serde_json::to_string(&block).unwrap();
        let deserialized: MultimodalContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn test_video_block_serialize_deserialize() {
        let block = video_block_base64("data==", "video/mp4");
        let json = serde_json::to_string(&block).unwrap();
        let deserialized: MultimodalContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn test_pdf_block_serialize_deserialize() {
        let block = pdf_block_url_with_pages("https://x.com/d.pdf", "1-3");
        let json = serde_json::to_string(&block).unwrap();
        let deserialized: MultimodalContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn test_citation_block_serialize_deserialize() {
        let block = citation_block_with_indices("src", "quote", 5, 15);
        let json = serde_json::to_string(&block).unwrap();
        let deserialized: MultimodalContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn test_annotation_block_serialize_deserialize() {
        let block = annotation_block("comment", json!({"text": "note"}));
        let json = serde_json::to_string(&block).unwrap();
        let deserialized: MultimodalContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn test_custom_block_serialize_deserialize() {
        let block = custom_block("special", json!({"key": "value"}));
        let json = serde_json::to_string(&block).unwrap();
        let deserialized: MultimodalContentBlock = serde_json::from_str(&json).unwrap();
        assert_eq!(block, deserialized);
    }

    #[test]
    fn test_media_source_url_serde() {
        let src = MediaSource::Url { url: "https://example.com".into() };
        let json = serde_json::to_string(&src).unwrap();
        assert!(json.contains("\"source_type\":\"url\""));
        let deserialized: MediaSource = serde_json::from_str(&json).unwrap();
        assert_eq!(src, deserialized);
    }

    #[test]
    fn test_media_source_base64_serde() {
        let src = MediaSource::Base64 { data: "abc==".into() };
        let json = serde_json::to_string(&src).unwrap();
        assert!(json.contains("\"source_type\":\"base64\""));
        let deserialized: MediaSource = serde_json::from_str(&json).unwrap();
        assert_eq!(src, deserialized);
    }

    // -- ContentBlockList tests --

    #[test]
    fn test_content_block_list_new_is_empty() {
        let list = ContentBlockList::new();
        assert!(list.is_empty());
        assert_eq!(list.len(), 0);
    }

    #[test]
    fn test_content_block_list_push() {
        let mut list = ContentBlockList::new();
        list.push(text_block("hello"));
        assert_eq!(list.len(), 1);
        assert!(!list.is_empty());
    }

    #[test]
    fn test_content_block_list_from_blocks() {
        let blocks = vec![text_block("a"), text_block("b")];
        let list = ContentBlockList::from_blocks(blocks);
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_content_block_list_text_only_true() {
        let list = ContentBlockList::from_blocks(vec![
            text_block("a"),
            text_block("b"),
        ]);
        assert!(list.text_only());
    }

    #[test]
    fn test_content_block_list_text_only_false_with_image() {
        let list = ContentBlockList::from_blocks(vec![
            text_block("a"),
            image_block_url("https://img.com/x.png", "image/png"),
        ]);
        assert!(!list.text_only());
    }

    #[test]
    fn test_content_block_list_text_only_false_when_empty() {
        let list = ContentBlockList::new();
        assert!(!list.text_only());
    }

    #[test]
    fn test_content_block_list_has_images() {
        let mut list = ContentBlockList::new();
        assert!(!list.has_images());
        list.push(image_block_url("https://img.com/x.png", "image/png"));
        assert!(list.has_images());
    }

    #[test]
    fn test_content_block_list_has_audio() {
        let list = ContentBlockList::from_blocks(vec![
            audio_block_url("https://a.com/s.mp3", "audio/mp3"),
        ]);
        assert!(list.has_audio());
    }

    #[test]
    fn test_content_block_list_has_video() {
        let list = ContentBlockList::from_blocks(vec![
            video_block_url("https://v.com/v.mp4", "video/mp4"),
        ]);
        assert!(list.has_video());
    }

    #[test]
    fn test_content_block_list_has_pdfs() {
        let list = ContentBlockList::from_blocks(vec![
            pdf_block_url("https://d.com/d.pdf"),
        ]);
        assert!(list.has_pdfs());
    }

    #[test]
    fn test_content_block_list_has_citations() {
        let list = ContentBlockList::from_blocks(vec![
            citation_block("src", "quote"),
        ]);
        assert!(list.has_citations());
    }

    #[test]
    fn test_content_block_list_has_annotations() {
        let list = ContentBlockList::from_blocks(vec![
            annotation_block("highlight", json!({})),
        ]);
        assert!(list.has_annotations());
    }

    #[test]
    fn test_content_block_list_has_media() {
        let list = ContentBlockList::from_blocks(vec![text_block("x")]);
        assert!(!list.has_media());

        let list2 = ContentBlockList::from_blocks(vec![
            text_block("x"),
            image_block_url("u", "image/png"),
        ]);
        assert!(list2.has_media());
    }

    #[test]
    fn test_content_block_list_extract_text() {
        let list = ContentBlockList::from_blocks(vec![
            text_block("Hello"),
            image_block_url("u", "image/png"),
            text_block("World"),
        ]);
        assert_eq!(list.extract_text(), "Hello\nWorld");
    }

    #[test]
    fn test_content_block_list_extract_text_empty() {
        let list = ContentBlockList::from_blocks(vec![
            image_block_url("u", "image/png"),
        ]);
        assert_eq!(list.extract_text(), "");
    }

    #[test]
    fn test_content_block_list_text_blocks() {
        let list = ContentBlockList::from_blocks(vec![
            text_block("a"),
            image_block_url("u", "image/png"),
            text_block("b"),
        ]);
        let texts = list.text_blocks();
        assert_eq!(texts.len(), 2);
        assert_eq!(texts[0].text, "a");
        assert_eq!(texts[1].text, "b");
    }

    #[test]
    fn test_content_block_list_image_blocks() {
        let list = ContentBlockList::from_blocks(vec![
            text_block("x"),
            image_block_url("u1", "image/png"),
            image_block_base64("d", "image/jpeg"),
        ]);
        assert_eq!(list.image_blocks().len(), 2);
    }

    #[test]
    fn test_content_block_list_audio_blocks() {
        let list = ContentBlockList::from_blocks(vec![
            audio_block_url("u", "audio/mp3"),
            text_block("x"),
        ]);
        assert_eq!(list.audio_blocks().len(), 1);
    }

    #[test]
    fn test_content_block_list_video_blocks() {
        let list = ContentBlockList::from_blocks(vec![
            video_block_url("u", "video/mp4"),
        ]);
        assert_eq!(list.video_blocks().len(), 1);
    }

    #[test]
    fn test_content_block_list_pdf_blocks() {
        let list = ContentBlockList::from_blocks(vec![
            pdf_block_url("u"),
            pdf_block_base64("d"),
        ]);
        assert_eq!(list.pdf_blocks().len(), 2);
    }

    #[test]
    fn test_content_block_list_citation_blocks() {
        let list = ContentBlockList::from_blocks(vec![
            citation_block("s1", "q1"),
            citation_block("s2", "q2"),
            text_block("x"),
        ]);
        assert_eq!(list.citation_blocks().len(), 2);
    }

    #[test]
    fn test_content_block_list_annotation_blocks() {
        let list = ContentBlockList::from_blocks(vec![
            annotation_block("highlight", json!({})),
            text_block("x"),
        ]);
        assert_eq!(list.annotation_blocks().len(), 1);
    }

    #[test]
    fn test_content_block_list_into_blocks() {
        let list = ContentBlockList::from_blocks(vec![text_block("a"), text_block("b")]);
        let blocks = list.into_blocks();
        assert_eq!(blocks.len(), 2);
    }

    #[test]
    fn test_content_block_list_iter() {
        let list = ContentBlockList::from_blocks(vec![text_block("a"), text_block("b")]);
        let count = list.iter().count();
        assert_eq!(count, 2);
    }

    #[test]
    fn test_content_block_list_into_iter() {
        let list = ContentBlockList::from_blocks(vec![text_block("a")]);
        let collected: Vec<_> = list.into_iter().collect();
        assert_eq!(collected.len(), 1);
    }

    #[test]
    fn test_content_block_list_ref_into_iter() {
        let list = ContentBlockList::from_blocks(vec![text_block("a"), text_block("b")]);
        let count = (&list).into_iter().count();
        assert_eq!(count, 2);
        // list is still usable
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_content_block_list_from_vec() {
        let blocks = vec![text_block("a")];
        let list: ContentBlockList = blocks.into();
        assert_eq!(list.len(), 1);
    }

    #[test]
    fn test_content_block_list_into_vec() {
        let list = ContentBlockList::from_blocks(vec![text_block("a")]);
        let vec: Vec<MultimodalContentBlock> = list.into();
        assert_eq!(vec.len(), 1);
    }

    #[test]
    fn test_content_block_list_default() {
        let list = ContentBlockList::default();
        assert!(list.is_empty());
    }

    #[test]
    fn test_content_block_list_serialize_deserialize() {
        let list = ContentBlockList::from_blocks(vec![
            text_block("hello"),
            image_block_url("https://img.com/x.png", "image/png"),
        ]);
        let json = serde_json::to_string(&list).unwrap();
        let deserialized: ContentBlockList = serde_json::from_str(&json).unwrap();
        assert_eq!(list, deserialized);
    }

    #[test]
    fn test_mixed_multimodal_list() {
        let list = ContentBlockList::from_blocks(vec![
            text_block("Describe this image:"),
            image_block_url("https://img.com/photo.jpg", "image/jpeg"),
            audio_block_url("https://a.com/narration.mp3", "audio/mp3"),
            video_block_url("https://v.com/clip.mp4", "video/mp4"),
            pdf_block_url_with_pages("https://d.com/paper.pdf", "1-10"),
            citation_block_with_indices("paper-1", "relevant excerpt", 0, 50),
            annotation_block("footnote", json!({"note": "see appendix"})),
            custom_block("provider_metadata", json!({"model": "gpt-4"})),
        ]);
        assert_eq!(list.len(), 8);
        assert!(list.has_images());
        assert!(list.has_audio());
        assert!(list.has_video());
        assert!(list.has_pdfs());
        assert!(list.has_citations());
        assert!(list.has_annotations());
        assert!(list.has_media());
        assert!(!list.text_only());
        assert_eq!(list.extract_text(), "Describe this image:");
    }

    #[test]
    fn test_clone_equality() {
        let block = text_block("cloneable");
        let cloned = block.clone();
        assert_eq!(block, cloned);
    }

    #[test]
    fn test_content_block_list_clone() {
        let list = ContentBlockList::from_blocks(vec![text_block("a")]);
        let cloned = list.clone();
        assert_eq!(list, cloned);
    }
}
