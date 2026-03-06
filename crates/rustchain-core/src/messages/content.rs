use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A citation annotation referencing a source document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Citation {
    /// URL of the document source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    /// Source document title.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Start index of the response text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_index: Option<usize>,
    /// End index of the response text.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_index: Option<usize>,
    /// Excerpt of source text being cited.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cited_text: Option<String>,
    /// Provider-specific metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extras: Option<Value>,
}

/// An annotation on a content block.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Annotation {
    /// A citation annotation.
    Citation(Citation),
    /// A provider-specific annotation.
    NonStandardAnnotation {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        value: Value,
    },
}

/// Standard content block types for LLM I/O.
///
/// Each variant corresponds to a typed content block from the Python
/// `langchain_core.messages.content` module. All multimodal data blocks
/// (Image, Video, Audio, File, PlainText) support URL, base64, and file_id
/// sources.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    /// Plain text content from a user or model.
    Text {
        text: String,
        /// Optional unique identifier for this block.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Citations and other annotations.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        annotations: Option<Vec<Annotation>>,
        /// Index of block in aggregate response (streaming).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        /// Provider-specific metadata.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// Image data (URL, base64, or file reference).
    Image {
        /// Optional unique identifier for this block.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// URL of the image.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        /// Base64-encoded image data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base64: Option<String>,
        /// Reference to an image in an external file storage system.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        /// MIME type of the image. Required for base64 data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        /// Index of block in aggregate response (streaming).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        // Legacy fields for backward compatibility with old-style blocks.
        /// Legacy: inline image data (base64 or URL).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        image: Option<String>,
        /// Legacy: source type ("url", "base64", "id").
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_type: Option<String>,
        /// Legacy: media type alias.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
        /// Provider-specific metadata.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// OpenAI-style image_url block with optional detail level.
    ImageUrl {
        /// The image URL info (url, detail).
        image_url: ImageUrlInfo,
        /// Provider-specific metadata.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// Audio data (URL, base64, or file reference).
    Audio {
        /// Optional unique identifier for this block.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// URL of the audio.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        /// Base64-encoded audio data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base64: Option<String>,
        /// Reference to an audio file in an external file storage system.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        /// MIME type of the audio. Required for base64 data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        /// Index of block in aggregate response (streaming).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        // Legacy fields for backward compatibility.
        /// Legacy: inline audio data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        audio: Option<String>,
        /// Legacy: source type.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_type: Option<String>,
        /// Legacy: media type alias.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
        /// Provider-specific metadata.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// Video data (URL, base64, or file reference).
    Video {
        /// Optional unique identifier for this block.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// URL of the video.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        /// Base64-encoded video data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base64: Option<String>,
        /// Reference to a video file in an external file storage system.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        /// MIME type of the video. Required for base64 data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        /// Index of block in aggregate response (streaming).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        // Legacy fields for backward compatibility.
        /// Legacy: inline video data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        video: Option<String>,
        /// Legacy: source type.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_type: Option<String>,
        /// Legacy: media type alias.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
        /// Provider-specific metadata.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// File data (PDFs, documents, etc. — not images/audio/video).
    File {
        /// Optional unique identifier for this block.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// URL of the file.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        /// Base64-encoded file data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base64: Option<String>,
        /// Reference to the file in an external file storage system.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        /// MIME type of the file. Required for base64 data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        /// Index of block in aggregate response (streaming).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        // Legacy fields.
        /// Legacy: inline file data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file: Option<String>,
        /// Legacy: source type.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        source_type: Option<String>,
        /// Legacy: media type alias.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        media_type: Option<String>,
        /// Provider-specific metadata.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// Plain text document data (e.g., `.txt` or `.md`).
    #[serde(rename = "text-plain")]
    PlainText {
        /// Optional unique identifier for this block.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// URL of the plaintext.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        url: Option<String>,
        /// Base64-encoded data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        base64: Option<String>,
        /// Reference to the file in an external file storage system.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        file_id: Option<String>,
        /// MIME type. Should be "text/plain".
        #[serde(default, skip_serializing_if = "Option::is_none")]
        mime_type: Option<String>,
        /// Plaintext content.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        text: Option<String>,
        /// Title of the text data.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        /// Context for the text (description or summary).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        context: Option<String>,
        /// Index of block in aggregate response (streaming).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        /// Provider-specific metadata.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// Chain-of-thought reasoning output from a model.
    Reasoning {
        /// The reasoning text (thought summary or raw reasoning).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        reasoning: Option<String>,
        /// Optional unique identifier for this block.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        /// Index of block in aggregate response (streaming).
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        /// Provider-specific metadata.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// Redacted thinking/reasoning block (e.g., Anthropic's redacted thinking).
    RedactedThinking {
        /// Opaque data representing the redacted thinking.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        data: Option<String>,
        /// Provider-specific metadata.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// Cache control marker (e.g., Anthropic's cache control).
    CacheControl {
        /// Cache control type (e.g., "ephemeral").
        #[serde(default, skip_serializing_if = "Option::is_none")]
        cache_type: Option<String>,
        /// Provider-specific metadata.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// An AI's request to call a tool.
    ToolCall {
        name: String,
        args: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// A chunk of a tool call (yielded during streaming).
    ToolCallChunk {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// A tool call that failed to parse.
    InvalidToolCall {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        error: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// A tool call that is executed server-side.
    ServerToolCall {
        name: String,
        args: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// A chunk of a server-side tool call (streaming).
    ServerToolCallChunk {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        name: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        args: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// Result of a server-side tool call.
    ServerToolResult {
        /// ID of the corresponding server tool call.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        tool_call_id: Option<String>,
        /// Execution status ("success" or "error").
        #[serde(default, skip_serializing_if = "Option::is_none")]
        status: Option<String>,
        /// Output of the executed tool.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        output: Option<Value>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// Tool result content (tool_use_id based, e.g., for Anthropic).
    ToolResult {
        /// The tool_use_id this result corresponds to.
        tool_use_id: String,
        /// The result content.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        content: Option<Value>,
        /// Whether this result represents an error.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
        /// Provider-specific metadata.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// Arbitrary data payload.
    Data {
        data: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        extras: Option<Value>,
    },

    /// Provider-specific content that does not fit standard types.
    #[serde(rename = "non_standard")]
    NonStandard {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        id: Option<String>,
        value: Value,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        index: Option<usize>,
    },
}

/// Image URL info for OpenAI-style image_url blocks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ImageUrlInfo {
    /// The URL of the image.
    pub url: String,
    /// Detail level ("auto", "low", "high").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

impl ContentBlock {
    /// Create a text-only content block with no metadata.
    pub fn text_only(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            id: None,
            annotations: None,
            index: None,
            extras: None,
        }
    }
}

/// Known content block type strings.
pub const KNOWN_BLOCK_TYPES: &[&str] = &[
    "text",
    "reasoning",
    "redacted_thinking",
    "cache_control",
    "tool_call",
    "invalid_tool_call",
    "tool_call_chunk",
    "tool_result",
    "image",
    "image_url",
    "audio",
    "file",
    "text-plain",
    "video",
    "server_tool_call",
    "server_tool_call_chunk",
    "server_tool_result",
    "data",
    "non_standard",
];

/// Check if a content block type string is a known data content block type.
pub fn is_data_content_block_type(block_type: &str) -> bool {
    matches!(block_type, "image" | "video" | "audio" | "text-plain" | "file")
}
