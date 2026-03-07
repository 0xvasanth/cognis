//! Text splitting strategies for document chunking.
//!
//! This module provides multiple text splitting strategies mirroring LangChain's
//! `text_splitters` module. Each splitter implements the [`TextSplitter`] trait
//! and supports configurable chunk size, overlap, and various splitting heuristics.
//!
//! ## Splitters
//!
//! - [`CharacterTextSplitter`] -- splits on a single separator (default `"\n\n"`).
//! - [`RecursiveCharacterTextSplitter`] -- tries separators in order, recursing on oversized chunks.
//! - [`TokenTextSplitter`] -- splits by estimated token count (whitespace-delimited words).
//! - [`MarkdownTextSplitter`] -- splits markdown by header levels, falling back to recursive splitting.
//! - [`CodeTextSplitter`] -- splits code using language-specific separators.
//!
//! ## Shared Configuration
//!
//! [`SplitConfig`] holds common options (chunk size, overlap, length function,
//! keep-separator flag, strip-whitespace flag) and is used internally by all splitters.

use std::sync::Arc;

use rustchain_core::documents::Document;

// ---------------------------------------------------------------------------
// Length function
// ---------------------------------------------------------------------------

/// How to measure the length of a text chunk.
#[derive(Clone)]
pub enum LengthFn {
    /// Count Unicode characters (default).
    Chars,
    /// Count whitespace-delimited words.
    Words,
    /// A user-supplied length function.
    Custom(Arc<dyn Fn(&str) -> usize + Send + Sync>),
}

impl Default for LengthFn {
    fn default() -> Self {
        Self::Chars
    }
}

impl LengthFn {
    fn measure(&self, text: &str) -> usize {
        match self {
            LengthFn::Chars => text.chars().count(),
            LengthFn::Words => text.split_whitespace().count(),
            LengthFn::Custom(f) => f(text),
        }
    }
}

impl std::fmt::Debug for LengthFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LengthFn::Chars => write!(f, "LengthFn::Chars"),
            LengthFn::Words => write!(f, "LengthFn::Words"),
            LengthFn::Custom(_) => write!(f, "LengthFn::Custom(...)"),
        }
    }
}

// ---------------------------------------------------------------------------
// SplitConfig
// ---------------------------------------------------------------------------

/// Shared configuration for text splitters.
#[derive(Clone)]
pub struct SplitConfig {
    /// Maximum length of each chunk (measured by `length_fn`).
    pub chunk_size: usize,
    /// Number of overlapping units between adjacent chunks.
    pub chunk_overlap: usize,
    /// How to measure chunk length.
    pub length_fn: LengthFn,
    /// Whether to keep the separator in the output chunks.
    pub keep_separator: bool,
    /// Strip leading/trailing whitespace from chunks.
    pub strip_whitespace: bool,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
            length_fn: LengthFn::default(),
            keep_separator: false,
            strip_whitespace: true,
        }
    }
}

impl SplitConfig {
    /// Create a new `SplitConfig` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set the overlap between adjacent chunks.
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    /// Set the length function.
    pub fn with_length_fn(mut self, f: LengthFn) -> Self {
        self.length_fn = f;
        self
    }

    /// Set whether to keep separators in the output.
    pub fn with_keep_separator(mut self, keep: bool) -> Self {
        self.keep_separator = keep;
        self
    }

    /// Set whether to strip whitespace from chunks.
    pub fn with_strip_whitespace(mut self, strip: bool) -> Self {
        self.strip_whitespace = strip;
        self
    }

    /// Measure text length using the configured length function.
    fn len(&self, text: &str) -> usize {
        self.length_fn.measure(text)
    }
}

// ---------------------------------------------------------------------------
// TextSplitter trait
// ---------------------------------------------------------------------------

/// Trait for splitting text into chunks.
///
/// Implementations must provide [`split_text`](TextSplitter::split_text).
/// A default [`split_documents`](TextSplitter::split_documents) implementation
/// is provided that splits each document and preserves metadata.
pub trait TextSplitter: Send + Sync {
    /// Split a single text string into chunks.
    fn split_text(&self, text: &str) -> Vec<String>;

    /// Split documents into smaller documents, preserving metadata.
    fn split_documents(&self, docs: Vec<Document>) -> Vec<Document> {
        let mut result = Vec::new();
        for doc in &docs {
            let chunks = self.split_text(&doc.page_content);
            for chunk in chunks {
                result.push(Document::new(chunk).with_metadata(doc.metadata.clone()));
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Helper: merge splits with overlap
// ---------------------------------------------------------------------------

/// Merge small pieces into chunks up to `chunk_size`, with `chunk_overlap`.
fn merge_splits(
    splits: &[&str],
    separator: &str,
    config: &SplitConfig,
) -> Vec<String> {
    let sep_len = config.len(separator);
    let mut docs: Vec<String> = Vec::new();
    let mut current_doc: Vec<&str> = Vec::new();
    let mut total: usize = 0;

    for piece in splits {
        let len = config.len(piece);
        let added = if current_doc.is_empty() {
            len
        } else {
            len + sep_len
        };

        if total + added > config.chunk_size && !current_doc.is_empty() {
            let doc = current_doc.join(separator);
            let doc = if config.strip_whitespace {
                doc.trim().to_string()
            } else {
                doc
            };
            if !doc.is_empty() {
                docs.push(doc);
            }
            // Keep overlap.
            if config.chunk_overlap == 0 {
                current_doc.clear();
                total = 0;
            } else {
                while total > config.chunk_overlap && current_doc.len() > 1 {
                    let removed = config.len(current_doc[0]) + sep_len;
                    total = total.saturating_sub(removed);
                    current_doc.remove(0);
                }
            }
        }

        current_doc.push(piece);
        total = if current_doc.len() == 1 {
            len
        } else {
            total + len + sep_len
        };
    }

    if !current_doc.is_empty() {
        let doc = current_doc.join(separator);
        let doc = if config.strip_whitespace {
            doc.trim().to_string()
        } else {
            doc
        };
        if !doc.is_empty() {
            docs.push(doc);
        }
    }
    docs
}

// ---------------------------------------------------------------------------
// CharacterTextSplitter
// ---------------------------------------------------------------------------

/// Splits text on a single separator string.
///
/// The default separator is `"\n\n"`. After splitting, pieces are merged back
/// into chunks that respect `chunk_size` and `chunk_overlap`.
pub struct CharacterTextSplitter {
    /// The separator to split on.
    pub separator: String,
    /// Shared split configuration.
    pub config: SplitConfig,
}

impl Default for CharacterTextSplitter {
    fn default() -> Self {
        Self {
            separator: "\n\n".to_string(),
            config: SplitConfig::default(),
        }
    }
}

impl CharacterTextSplitter {
    /// Create a new `CharacterTextSplitter` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the separator string.
    pub fn with_separator(mut self, sep: impl Into<String>) -> Self {
        self.separator = sep.into();
        self
    }

    /// Set the maximum chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    /// Set the overlap between adjacent chunks.
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.config.chunk_overlap = overlap;
        self
    }

    /// Set the split configuration.
    pub fn with_config(mut self, config: SplitConfig) -> Self {
        self.config = config;
        self
    }
}

impl TextSplitter for CharacterTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        let splits: Vec<&str> = if self.separator.is_empty() {
            // Character-level fallback: each character is a split.
            text.char_indices()
                .map(|(i, c)| &text[i..i + c.len_utf8()])
                .collect()
        } else {
            text.split(&self.separator).collect()
        };

        let good_splits: Vec<&str> = splits
            .iter()
            .copied()
            .map(|s| if self.config.strip_whitespace { s.trim() } else { s })
            .filter(|s| !s.is_empty())
            .collect();

        merge_splits(&good_splits, &self.separator, &self.config)
    }
}

// ---------------------------------------------------------------------------
// RecursiveCharacterTextSplitter
// ---------------------------------------------------------------------------

/// Recursively splits text trying each separator in order, falling back to
/// smaller separators when chunks exceed `chunk_size`.
///
/// Default separators: `["\n\n", "\n", " ", ""]`.
pub struct RecursiveCharacterTextSplitter {
    /// Hierarchy of separators, tried in order.
    pub separators: Vec<String>,
    /// Shared split configuration.
    pub config: SplitConfig,
}

impl Default for RecursiveCharacterTextSplitter {
    fn default() -> Self {
        Self {
            separators: vec![
                "\n\n".into(),
                "\n".into(),
                " ".into(),
                "".into(),
            ],
            config: SplitConfig::default(),
        }
    }
}

impl RecursiveCharacterTextSplitter {
    /// Create a new splitter with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the hierarchy of separators.
    pub fn with_separators(mut self, seps: Vec<impl Into<String>>) -> Self {
        self.separators = seps.into_iter().map(|s| s.into()).collect();
        self
    }

    /// Set the maximum chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.config.chunk_size = size;
        self
    }

    /// Set the overlap between adjacent chunks.
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.config.chunk_overlap = overlap;
        self
    }

    /// Set the split configuration.
    pub fn with_config(mut self, config: SplitConfig) -> Self {
        self.config = config;
        self
    }

    /// Core recursive algorithm.
    fn split_text_recursive(&self, text: &str, separators: &[String]) -> Vec<String> {
        let mut final_chunks: Vec<String> = Vec::new();

        // Find the appropriate separator.
        let mut separator = separators.last().map(|s| s.as_str()).unwrap_or("");
        let mut remaining_separators: &[String] = &[];

        for (i, sep) in separators.iter().enumerate() {
            if sep.is_empty() || text.contains(sep.as_str()) {
                separator = sep.as_str();
                remaining_separators = &separators[i + 1..];
                break;
            }
        }

        let splits: Vec<&str> = if separator.is_empty() {
            text.char_indices()
                .map(|(i, c)| &text[i..i + c.len_utf8()])
                .collect()
        } else if self.config.keep_separator {
            // Split and keep separator at the start of each piece (except first).
            let mut result = Vec::new();
            let mut last = 0;
            for (idx, _) in text.match_indices(separator) {
                if last < idx {
                    result.push(&text[last..idx]);
                }
                last = idx;
            }
            if last < text.len() {
                result.push(&text[last..]);
            }
            if result.is_empty() {
                result.push(text);
            }
            result
        } else {
            text.split(separator).collect()
        };

        let merge_sep = if self.config.keep_separator { "" } else { separator };

        let mut good_splits: Vec<&str> = Vec::new();

        for s in &splits {
            let piece = if self.config.strip_whitespace { s.trim() } else { *s };
            if piece.is_empty() {
                continue;
            }
            if self.config.len(piece) < self.config.chunk_size {
                good_splits.push(piece);
            } else {
                if !good_splits.is_empty() {
                    let merged = merge_splits(&good_splits, merge_sep, &self.config);
                    final_chunks.extend(merged);
                    good_splits.clear();
                }
                if remaining_separators.is_empty() {
                    final_chunks.push(piece.to_string());
                } else {
                    let sub = self.split_text_recursive(piece, remaining_separators);
                    final_chunks.extend(sub);
                }
            }
        }

        if !good_splits.is_empty() {
            let merged = merge_splits(&good_splits, merge_sep, &self.config);
            final_chunks.extend(merged);
        }

        final_chunks
    }
}

impl TextSplitter for RecursiveCharacterTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        self.split_text_recursive(text, &self.separators)
    }
}

// ---------------------------------------------------------------------------
// TokenTextSplitter
// ---------------------------------------------------------------------------

/// Splits text by estimated token count using simple word-based estimation
/// (split on whitespace).
///
/// `chunk_size` and `chunk_overlap` are measured in tokens (words).
pub struct TokenTextSplitter {
    /// Maximum number of tokens per chunk.
    pub chunk_size: usize,
    /// Number of overlapping tokens between adjacent chunks.
    pub chunk_overlap: usize,
    /// Strip whitespace from chunks.
    pub strip_whitespace: bool,
}

impl Default for TokenTextSplitter {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
            strip_whitespace: true,
        }
    }
}

impl TokenTextSplitter {
    /// Create a new `TokenTextSplitter` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of tokens per chunk.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set the overlap in tokens between adjacent chunks.
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    /// Set whether to strip whitespace from chunks.
    pub fn with_strip_whitespace(mut self, strip: bool) -> Self {
        self.strip_whitespace = strip;
        self
    }
}

impl TextSplitter for TokenTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        let words: Vec<&str> = text.split_whitespace().collect();
        if words.is_empty() {
            return Vec::new();
        }

        let chunk_size = if self.chunk_size == 0 { 1 } else { self.chunk_size };
        // Clamp overlap to be less than chunk_size.
        let overlap = self.chunk_overlap.min(chunk_size.saturating_sub(1));

        let mut chunks = Vec::new();
        let mut start = 0;

        while start < words.len() {
            let end = (start + chunk_size).min(words.len());
            let chunk = words[start..end].join(" ");
            let chunk = if self.strip_whitespace {
                chunk.trim().to_string()
            } else {
                chunk
            };
            if !chunk.is_empty() {
                chunks.push(chunk);
            }
            if end >= words.len() {
                break;
            }
            let step = chunk_size.saturating_sub(overlap);
            let new_start = start + step.max(1);
            if new_start <= start {
                break;
            }
            start = new_start;
        }

        chunks
    }
}

// ---------------------------------------------------------------------------
// MarkdownTextSplitter
// ---------------------------------------------------------------------------

/// Splits markdown text by header levels, then falls back to
/// [`RecursiveCharacterTextSplitter`] for sections that exceed `chunk_size`.
pub struct MarkdownTextSplitter {
    /// Maximum chunk size.
    pub chunk_size: usize,
    /// Overlap between chunks.
    pub chunk_overlap: usize,
    /// Strip whitespace from chunks.
    pub strip_whitespace: bool,
}

impl Default for MarkdownTextSplitter {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
            strip_whitespace: true,
        }
    }
}

impl MarkdownTextSplitter {
    /// Create a new `MarkdownTextSplitter` with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set the overlap between chunks.
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    /// Set whether to strip whitespace from chunks.
    pub fn with_strip_whitespace(mut self, strip: bool) -> Self {
        self.strip_whitespace = strip;
        self
    }
}

impl TextSplitter for MarkdownTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        // Use markdown-aware separators: headers first, then paragraph/line/word/char.
        let separators: Vec<String> = vec![
            "\n# ".into(),
            "\n## ".into(),
            "\n### ".into(),
            "\n#### ".into(),
            "\n##### ".into(),
            "\n###### ".into(),
            "\n\n".into(),
            "\n".into(),
            " ".into(),
            "".into(),
        ];

        let splitter = RecursiveCharacterTextSplitter {
            separators,
            config: SplitConfig {
                chunk_size: self.chunk_size,
                chunk_overlap: self.chunk_overlap,
                length_fn: LengthFn::Chars,
                keep_separator: false,
                strip_whitespace: self.strip_whitespace,
            },
        };

        splitter.split_text(text)
    }
}

// ---------------------------------------------------------------------------
// CodeTextSplitter
// ---------------------------------------------------------------------------

/// Supported programming languages for [`CodeTextSplitter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodeLanguage {
    /// Rust: `fn`, `impl`, `struct`, `enum`, `mod`, `trait`.
    Rust,
    /// Python: `def`, `class`.
    Python,
    /// JavaScript: `function`, `class`, `const`.
    JavaScript,
}

impl CodeLanguage {
    /// Get the ordered list of separators for this language.
    fn separators(&self) -> Vec<String> {
        match self {
            CodeLanguage::Rust => vec![
                "\nfn ".into(),
                "\npub fn ".into(),
                "\nimpl ".into(),
                "\nstruct ".into(),
                "\npub struct ".into(),
                "\nenum ".into(),
                "\npub enum ".into(),
                "\nmod ".into(),
                "\npub mod ".into(),
                "\ntrait ".into(),
                "\npub trait ".into(),
                "\n\n".into(),
                "\n".into(),
                " ".into(),
                "".into(),
            ],
            CodeLanguage::Python => vec![
                "\nclass ".into(),
                "\ndef ".into(),
                "\n\tdef ".into(),
                "\n    def ".into(),
                "\n\n".into(),
                "\n".into(),
                " ".into(),
                "".into(),
            ],
            CodeLanguage::JavaScript => vec![
                "\nfunction ".into(),
                "\nclass ".into(),
                "\nconst ".into(),
                "\nlet ".into(),
                "\nvar ".into(),
                "\n\n".into(),
                "\n".into(),
                " ".into(),
                "".into(),
            ],
        }
    }
}

/// Splits code using language-specific separators.
///
/// Falls back to [`RecursiveCharacterTextSplitter`] with language-aware
/// separator ordering.
pub struct CodeTextSplitter {
    /// The programming language to split for.
    pub language: CodeLanguage,
    /// Maximum chunk size.
    pub chunk_size: usize,
    /// Overlap between chunks.
    pub chunk_overlap: usize,
    /// Strip whitespace from chunks.
    pub strip_whitespace: bool,
}

impl Default for CodeTextSplitter {
    fn default() -> Self {
        Self {
            language: CodeLanguage::Rust,
            chunk_size: 1000,
            chunk_overlap: 200,
            strip_whitespace: true,
        }
    }
}

impl CodeTextSplitter {
    /// Create a new `CodeTextSplitter` with default settings (Rust).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the programming language.
    pub fn with_language(mut self, lang: CodeLanguage) -> Self {
        self.language = lang;
        self
    }

    /// Set the maximum chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Set the overlap between chunks.
    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    /// Set whether to strip whitespace from chunks.
    pub fn with_strip_whitespace(mut self, strip: bool) -> Self {
        self.strip_whitespace = strip;
        self
    }
}

impl TextSplitter for CodeTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        let splitter = RecursiveCharacterTextSplitter {
            separators: self.language.separators(),
            config: SplitConfig {
                chunk_size: self.chunk_size,
                chunk_overlap: self.chunk_overlap,
                length_fn: LengthFn::Chars,
                keep_separator: false,
                strip_whitespace: self.strip_whitespace,
            },
        };

        splitter.split_text(text)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use serde_json::json;

    fn doc(content: &str) -> Document {
        Document::new(content)
    }

    fn doc_with_meta(content: &str, key: &str, value: &str) -> Document {
        let mut m = HashMap::new();
        m.insert(key.to_string(), json!(value));
        Document::new(content).with_metadata(m)
    }

    // ===================================================================
    // SplitConfig tests
    // ===================================================================

    #[test]
    fn test_split_config_defaults() {
        let config = SplitConfig::new();
        assert_eq!(config.chunk_size, 1000);
        assert_eq!(config.chunk_overlap, 200);
        assert!(config.strip_whitespace);
        assert!(!config.keep_separator);
    }

    #[test]
    fn test_split_config_builder() {
        let config = SplitConfig::new()
            .with_chunk_size(500)
            .with_chunk_overlap(50)
            .with_keep_separator(true)
            .with_strip_whitespace(false);
        assert_eq!(config.chunk_size, 500);
        assert_eq!(config.chunk_overlap, 50);
        assert!(config.keep_separator);
        assert!(!config.strip_whitespace);
    }

    #[test]
    fn test_split_config_length_fn_chars() {
        let config = SplitConfig::new();
        assert_eq!(config.len("hello"), 5);
    }

    #[test]
    fn test_split_config_length_fn_words() {
        let config = SplitConfig::new().with_length_fn(LengthFn::Words);
        assert_eq!(config.len("one two three"), 3);
    }

    #[test]
    fn test_split_config_length_fn_custom() {
        let config = SplitConfig::new()
            .with_length_fn(LengthFn::Custom(Arc::new(|s: &str| s.len())));
        assert_eq!(config.len("abc"), 3);
    }

    // ===================================================================
    // CharacterTextSplitter tests
    // ===================================================================

    #[test]
    fn test_character_splitter_default_separator() {
        let splitter = CharacterTextSplitter::new()
            .with_chunk_size(20)
            .with_chunk_overlap(0);
        let text = "Hello world.\n\nSecond paragraph.\n\nThird paragraph.";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_character_splitter_custom_separator() {
        let splitter = CharacterTextSplitter::new()
            .with_separator("|")
            .with_chunk_size(5)
            .with_chunk_overlap(0);
        let text = "abc|defgh|ijklm";
        let chunks = splitter.split_text(text);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "abc");
        assert_eq!(chunks[1], "defgh");
        assert_eq!(chunks[2], "ijklm");
    }

    #[test]
    fn test_character_splitter_empty_text() {
        let splitter = CharacterTextSplitter::new();
        let chunks = splitter.split_text("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_character_splitter_text_smaller_than_chunk() {
        let splitter = CharacterTextSplitter::new()
            .with_chunk_size(1000)
            .with_chunk_overlap(0);
        let chunks = splitter.split_text("short text");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "short text");
    }

    #[test]
    fn test_character_splitter_single_char() {
        let splitter = CharacterTextSplitter::new()
            .with_chunk_size(10)
            .with_chunk_overlap(0);
        let chunks = splitter.split_text("x");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "x");
    }

    #[test]
    fn test_character_splitter_no_separator_found() {
        let splitter = CharacterTextSplitter::new()
            .with_separator("|||")
            .with_chunk_size(100)
            .with_chunk_overlap(0);
        let chunks = splitter.split_text("no separators here at all");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "no separators here at all");
    }

    #[test]
    fn test_character_splitter_overlap() {
        let splitter = CharacterTextSplitter::new()
            .with_separator(" ")
            .with_chunk_size(10)
            .with_chunk_overlap(5);
        let text = "one two three four five six";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() > 1);
        // With overlap, some words should appear in consecutive chunks.
        let all_text: String = chunks.join(" ");
        assert!(all_text.contains("one"));
        assert!(all_text.contains("six"));
    }

    #[test]
    fn test_character_splitter_with_config() {
        let config = SplitConfig::new()
            .with_chunk_size(5)
            .with_chunk_overlap(0)
            .with_strip_whitespace(true);
        let splitter = CharacterTextSplitter::new()
            .with_separator(",")
            .with_config(config);
        let chunks = splitter.split_text("a,bb,ccc,dddd");
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert_eq!(chunk, chunk.trim());
        }
    }

    // ===================================================================
    // RecursiveCharacterTextSplitter tests
    // ===================================================================

    #[test]
    fn test_recursive_splitter_default() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(20)
            .with_chunk_overlap(0);
        let text = "Hello world.\n\nSecond paragraph here.\n\nThird.";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            assert!(
                chunk.len() <= 25,
                "Chunk too large: {:?} (len {})",
                chunk,
                chunk.len()
            );
        }
    }

    #[test]
    fn test_recursive_splitter_custom_separators() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_separators(vec!["||", " ", ""])
            .with_chunk_size(12)
            .with_chunk_overlap(0);
        let text = "chunk one||chunk two||chunk three";
        let chunks = splitter.split_text(text);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "chunk one");
        assert_eq!(chunks[1], "chunk two");
        assert_eq!(chunks[2], "chunk three");
    }

    #[test]
    fn test_recursive_splitter_empty_text() {
        let splitter = RecursiveCharacterTextSplitter::new();
        let chunks = splitter.split_text("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_recursive_splitter_text_smaller_than_chunk() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(1000)
            .with_chunk_overlap(0);
        let chunks = splitter.split_text("Small text.");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Small text.");
    }

    #[test]
    fn test_recursive_splitter_overlap() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(15)
            .with_chunk_overlap(5);
        let text = "alpha beta gamma delta epsilon zeta eta theta";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() > 1);
        // Verify overlap: some words appear in consecutive chunks.
        let mut found_overlap = false;
        for i in 1..chunks.len() {
            let prev_words: Vec<&str> = chunks[i - 1].split_whitespace().collect();
            let curr_words: Vec<&str> = chunks[i].split_whitespace().collect();
            for w in &prev_words {
                if curr_words.contains(w) {
                    found_overlap = true;
                    break;
                }
            }
            if found_overlap {
                break;
            }
        }
        assert!(found_overlap, "Expected overlap between chunks");
    }

    #[test]
    fn test_recursive_splitter_falls_back_to_smaller_separators() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(10)
            .with_chunk_overlap(0);
        // No "\n\n" or "\n", so should fall to " " then "".
        let text = "abcdefghij klmnopqrst";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_recursive_splitter_single_char_text() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(10)
            .with_chunk_overlap(0);
        let chunks = splitter.split_text("x");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "x");
    }

    // ===================================================================
    // TokenTextSplitter tests
    // ===================================================================

    #[test]
    fn test_token_splitter_basic() {
        let splitter = TokenTextSplitter::new()
            .with_chunk_size(3)
            .with_chunk_overlap(0);
        let text = "one two three four five six seven";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() >= 2);
        for chunk in &chunks {
            let word_count = chunk.split_whitespace().count();
            assert!(word_count <= 3, "Chunk has {} words: {:?}", word_count, chunk);
        }
    }

    #[test]
    fn test_token_splitter_overlap() {
        let splitter = TokenTextSplitter::new()
            .with_chunk_size(4)
            .with_chunk_overlap(2);
        let text = "one two three four five six seven eight";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() > 1);
        // With overlap of 2, consecutive chunks should share 2 words.
        if chunks.len() >= 2 {
            let first_words: Vec<&str> = chunks[0].split_whitespace().collect();
            let second_words: Vec<&str> = chunks[1].split_whitespace().collect();
            let shared: usize = first_words
                .iter()
                .filter(|w| second_words.contains(w))
                .count();
            assert!(shared >= 1, "Expected overlap, found {} shared words", shared);
        }
    }

    #[test]
    fn test_token_splitter_empty_text() {
        let splitter = TokenTextSplitter::new();
        let chunks = splitter.split_text("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_token_splitter_whitespace_only() {
        let splitter = TokenTextSplitter::new();
        let chunks = splitter.split_text("   \t\n  ");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_token_splitter_text_smaller_than_chunk() {
        let splitter = TokenTextSplitter::new()
            .with_chunk_size(100)
            .with_chunk_overlap(0);
        let chunks = splitter.split_text("just two");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "just two");
    }

    #[test]
    fn test_token_splitter_overlap_ge_chunk_size() {
        // overlap >= chunk_size should be clamped to chunk_size - 1.
        let splitter = TokenTextSplitter::new()
            .with_chunk_size(3)
            .with_chunk_overlap(10);
        let text = "one two three four five";
        let chunks = splitter.split_text(text);
        assert!(!chunks.is_empty());
        // Should not infinite loop.
    }

    #[test]
    fn test_token_splitter_single_word() {
        let splitter = TokenTextSplitter::new()
            .with_chunk_size(5)
            .with_chunk_overlap(0);
        let chunks = splitter.split_text("hello");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello");
    }

    // ===================================================================
    // MarkdownTextSplitter tests
    // ===================================================================

    #[test]
    fn test_markdown_splitter_header_detection() {
        let splitter = MarkdownTextSplitter::new()
            .with_chunk_size(50)
            .with_chunk_overlap(0);
        let text = "# Title\n\nIntro text here.\n\n## Section One\n\nContent one.\n\n## Section Two\n\nContent two.";
        let chunks = splitter.split_text(text);
        assert!(
            chunks.len() >= 2,
            "Expected header-based splits, got {:?}",
            chunks
        );
    }

    #[test]
    fn test_markdown_splitter_nested_headers() {
        let splitter = MarkdownTextSplitter::new()
            .with_chunk_size(40)
            .with_chunk_overlap(0);
        let text = "# Top\n\nText.\n\n## Mid\n\nMore.\n\n### Low\n\nDeep text.";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_markdown_splitter_empty_text() {
        let splitter = MarkdownTextSplitter::new();
        let chunks = splitter.split_text("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_markdown_splitter_no_headers() {
        let splitter = MarkdownTextSplitter::new()
            .with_chunk_size(20)
            .with_chunk_overlap(0);
        let text = "Just plain text without any markdown headers at all.";
        let chunks = splitter.split_text(text);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_markdown_splitter_small_text() {
        let splitter = MarkdownTextSplitter::new()
            .with_chunk_size(1000)
            .with_chunk_overlap(0);
        let text = "# Title\n\nSmall.";
        let chunks = splitter.split_text(text);
        assert_eq!(chunks.len(), 1);
    }

    // ===================================================================
    // CodeTextSplitter tests
    // ===================================================================

    #[test]
    fn test_code_splitter_rust() {
        let splitter = CodeTextSplitter::new()
            .with_language(CodeLanguage::Rust)
            .with_chunk_size(30)
            .with_chunk_overlap(0);
        let text = "fn foo() {\n    let x = 1;\n}\n\nfn bar() {\n    let y = 2;\n}";
        let chunks = splitter.split_text(text);
        assert!(
            chunks.len() >= 2,
            "Expected Rust function-based splits, got {:?}",
            chunks
        );
    }

    #[test]
    fn test_code_splitter_python() {
        let splitter = CodeTextSplitter::new()
            .with_language(CodeLanguage::Python)
            .with_chunk_size(30)
            .with_chunk_overlap(0);
        let text = "class Foo:\n    pass\n\ndef bar():\n    return 1\n\ndef baz():\n    return 2";
        let chunks = splitter.split_text(text);
        assert!(
            chunks.len() >= 2,
            "Expected Python function-based splits, got {:?}",
            chunks
        );
    }

    #[test]
    fn test_code_splitter_javascript() {
        let splitter = CodeTextSplitter::new()
            .with_language(CodeLanguage::JavaScript)
            .with_chunk_size(40)
            .with_chunk_overlap(0);
        let text = "function hello() {\n  return 1;\n}\n\nconst x = 42;\n\nclass Foo {\n  constructor() {}\n}";
        let chunks = splitter.split_text(text);
        assert!(
            chunks.len() >= 2,
            "Expected JS-based splits, got {:?}",
            chunks
        );
    }

    #[test]
    fn test_code_splitter_empty_text() {
        let splitter = CodeTextSplitter::new();
        let chunks = splitter.split_text("");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_code_splitter_small_code() {
        let splitter = CodeTextSplitter::new()
            .with_language(CodeLanguage::Rust)
            .with_chunk_size(1000)
            .with_chunk_overlap(0);
        let text = "fn main() {}";
        let chunks = splitter.split_text(text);
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_code_splitter_rust_struct_enum() {
        let splitter = CodeTextSplitter::new()
            .with_language(CodeLanguage::Rust)
            .with_chunk_size(40)
            .with_chunk_overlap(0);
        let text = "struct Point {\n    x: f64,\n    y: f64,\n}\n\nenum Color {\n    Red,\n    Green,\n    Blue,\n}\n\nimpl Point {\n    fn new() -> Self {\n        Self { x: 0.0, y: 0.0 }\n    }\n}";
        let chunks = splitter.split_text(text);
        assert!(
            chunks.len() >= 2,
            "Expected struct/enum/impl splits, got {:?}",
            chunks
        );
    }

    // ===================================================================
    // Document splitting tests
    // ===================================================================

    #[test]
    fn test_split_documents_preserves_metadata() {
        let splitter = CharacterTextSplitter::new()
            .with_separator(" ")
            .with_chunk_size(10)
            .with_chunk_overlap(0);
        let docs = vec![
            doc_with_meta("hello world foo bar baz", "source", "test.txt"),
        ];
        let result = splitter.split_documents(docs);
        assert!(result.len() > 1);
        for d in &result {
            assert_eq!(
                d.metadata.get("source"),
                Some(&json!("test.txt")),
                "Metadata should be preserved on all chunks"
            );
        }
    }

    #[test]
    fn test_split_documents_empty_vec() {
        let splitter = CharacterTextSplitter::new();
        let result = splitter.split_documents(vec![]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_split_documents_multiple_docs() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(10)
            .with_chunk_overlap(0);
        let docs = vec![
            doc_with_meta("first document here with text", "id", "1"),
            doc_with_meta("second document also has content", "id", "2"),
        ];
        let result = splitter.split_documents(docs);
        assert!(result.len() >= 4);
        // Check both metadata values appear.
        let ids: Vec<_> = result
            .iter()
            .map(|d| d.metadata.get("id").unwrap().as_str().unwrap())
            .collect();
        assert!(ids.contains(&"1"));
        assert!(ids.contains(&"2"));
    }

    #[test]
    fn test_token_splitter_split_documents() {
        let splitter = TokenTextSplitter::new()
            .with_chunk_size(3)
            .with_chunk_overlap(0);
        let docs = vec![doc("one two three four five six")];
        let result = splitter.split_documents(docs);
        assert!(result.len() >= 2);
    }

    #[test]
    fn test_markdown_splitter_split_documents() {
        let splitter = MarkdownTextSplitter::new()
            .with_chunk_size(30)
            .with_chunk_overlap(0);
        let docs = vec![doc("# Title\n\nText.\n\n## Section\n\nMore text here.")];
        let result = splitter.split_documents(docs);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_code_splitter_split_documents() {
        let splitter = CodeTextSplitter::new()
            .with_language(CodeLanguage::Python)
            .with_chunk_size(25)
            .with_chunk_overlap(0);
        let docs = vec![doc("def foo():\n    pass\n\ndef bar():\n    pass")];
        let result = splitter.split_documents(docs);
        assert!(!result.is_empty());
    }

    // ===================================================================
    // Edge case tests
    // ===================================================================

    #[test]
    fn test_overlap_equal_to_chunk_size_character() {
        // overlap >= chunk_size should not cause infinite loops.
        let splitter = CharacterTextSplitter::new()
            .with_separator(" ")
            .with_chunk_size(5)
            .with_chunk_overlap(5);
        let chunks = splitter.split_text("a b c d e f g");
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_recursive_splitter_all_same_text() {
        let splitter = RecursiveCharacterTextSplitter::new()
            .with_chunk_size(5)
            .with_chunk_overlap(0);
        let text = "aaaaaaaaaa"; // 10 a's, no separators
        let chunks = splitter.split_text(text);
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_code_language_separators_not_empty() {
        assert!(!CodeLanguage::Rust.separators().is_empty());
        assert!(!CodeLanguage::Python.separators().is_empty());
        assert!(!CodeLanguage::JavaScript.separators().is_empty());
    }

    #[test]
    fn test_length_fn_debug() {
        let f = LengthFn::Chars;
        assert_eq!(format!("{:?}", f), "LengthFn::Chars");
        let f = LengthFn::Words;
        assert_eq!(format!("{:?}", f), "LengthFn::Words");
        let f = LengthFn::Custom(Arc::new(|s: &str| s.len()));
        assert_eq!(format!("{:?}", f), "LengthFn::Custom(...)");
    }
}
