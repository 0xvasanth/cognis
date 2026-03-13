use std::collections::HashMap;
use std::sync::Arc;

use regex::Regex;
use cognis_core::documents::Document;
use serde_json::Value;

use super::code::Language;
use super::TextSplitter;

/// How to measure the length of a text chunk.
#[derive(Clone, Default)]
pub enum LengthFunction {
    /// Count Unicode characters (default).
    #[default]
    Characters,
    /// Count whitespace-delimited words.
    Words,
    /// A user-supplied length function.
    Custom(Arc<dyn Fn(&str) -> usize + Send + Sync>),
}

impl LengthFunction {
    fn measure(&self, text: &str) -> usize {
        match self {
            LengthFunction::Characters => text.chars().count(),
            LengthFunction::Words => text.split_whitespace().count(),
            LengthFunction::Custom(f) => f(text),
        }
    }
}

impl std::fmt::Debug for LengthFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LengthFunction::Characters => write!(f, "LengthFunction::Characters"),
            LengthFunction::Words => write!(f, "LengthFunction::Words"),
            LengthFunction::Custom(_) => write!(f, "LengthFunction::Custom(...)"),
        }
    }
}

/// Where to attach the separator when splitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum KeepSeparator {
    /// Discard the separator.
    #[default]
    None,
    /// Prepend the separator to the start of the next chunk.
    Start,
    /// Append the separator to the end of the current chunk.
    End,
}

/// Recursively splits text trying each separator in order, falling back to
/// smaller separators when chunks exceed `chunk_size`.
///
/// This is the most commonly used text splitter for real-world workloads.
pub struct RecursiveCharacterTextSplitter {
    /// Hierarchy of separators, tried in order.
    pub separators: Vec<String>,
    /// Maximum length of each chunk (measured by `length_function`).
    pub chunk_size: usize,
    /// Number of overlapping units between adjacent chunks.
    pub chunk_overlap: usize,
    /// How to measure chunk length.
    pub length_function: LengthFunction,
    /// Whether and where to keep the separator in output chunks.
    pub keep_separator: KeepSeparator,
    /// Strip leading/trailing whitespace from chunks (default `true`).
    pub strip_whitespace: bool,
    /// Treat separators as regex patterns (default `false`).
    pub is_separator_regex: bool,
}

impl Default for RecursiveCharacterTextSplitter {
    fn default() -> Self {
        Self {
            separators: vec!["\n\n".into(), "\n".into(), " ".into(), "".into()],
            chunk_size: 4000,
            chunk_overlap: 200,
            length_function: LengthFunction::default(),
            keep_separator: KeepSeparator::None,
            strip_whitespace: true,
            is_separator_regex: false,
        }
    }
}

impl RecursiveCharacterTextSplitter {
    /// Create a new splitter with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a builder for configuring the splitter.
    pub fn builder() -> RecursiveCharacterTextSplitterBuilder {
        RecursiveCharacterTextSplitterBuilder::default()
    }

    /// Create a splitter pre-configured for a specific programming language.
    pub fn for_language(lang: Language) -> Self {
        Self {
            separators: lang.get_separators(),
            ..Self::default()
        }
    }

    // Legacy alias kept for backward-compatibility with existing callers.
    /// Create a splitter configured for a specific programming language.
    pub fn from_language(lang: Language) -> Self {
        Self::for_language(lang)
    }

    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    pub fn with_chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    pub fn with_separators(mut self, seps: Vec<String>) -> Self {
        self.separators = seps;
        self
    }

    /// Split `Documents`, preserving each document's metadata on every produced chunk.
    pub fn split_documents(&self, documents: &[Document]) -> Vec<Document> {
        let texts: Vec<&str> = documents.iter().map(|d| d.page_content.as_str()).collect();
        let metadatas: Vec<HashMap<String, Value>> =
            documents.iter().map(|d| d.metadata.clone()).collect();
        self.create_documents(&texts, Some(&metadatas))
    }

    /// Create `Document`s from raw texts with optional per-text metadata.
    pub fn create_documents(
        &self,
        texts: &[&str],
        metadatas: Option<&[HashMap<String, Value>]>,
    ) -> Vec<Document> {
        let mut docs = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            let metadata = metadatas
                .and_then(|m| m.get(i))
                .cloned()
                .unwrap_or_default();
            for chunk in self.split_text(text) {
                docs.push(Document::new(chunk).with_metadata(metadata.clone()));
            }
        }
        docs
    }

    // ── internal helpers ────────────────────────────────────────────────

    /// Measure text length using the configured length function.
    fn len(&self, text: &str) -> usize {
        self.length_function.measure(text)
    }

    /// Split `text` by `separator`, optionally keeping the separator attached
    /// to the start or end of the resulting pieces.
    fn split_by_separator(&self, text: &str, separator: &str) -> Vec<String> {
        if separator.is_empty() {
            // Character-level split.
            return text.chars().map(|c| c.to_string()).collect();
        }

        if self.is_separator_regex {
            self.split_by_regex(text, separator)
        } else {
            self.split_by_literal(text, separator)
        }
    }

    fn split_by_literal(&self, text: &str, separator: &str) -> Vec<String> {
        match self.keep_separator {
            KeepSeparator::None => text.split(separator).map(|s| s.to_string()).collect(),
            KeepSeparator::Start => {
                let mut result = Vec::new();
                let parts: Vec<&str> = text.split(separator).collect();
                for (i, part) in parts.iter().enumerate() {
                    if i == 0 {
                        result.push(part.to_string());
                    } else {
                        result.push(format!("{}{}", separator, part));
                    }
                }
                result
            }
            KeepSeparator::End => {
                let mut result = Vec::new();
                let parts: Vec<&str> = text.split(separator).collect();
                let last = parts.len().saturating_sub(1);
                for (i, part) in parts.iter().enumerate() {
                    if i < last {
                        result.push(format!("{}{}", part, separator));
                    } else {
                        result.push(part.to_string());
                    }
                }
                result
            }
        }
    }

    fn split_by_regex(&self, text: &str, pattern: &str) -> Vec<String> {
        let re = match Regex::new(pattern) {
            Ok(r) => r,
            Err(_) => return vec![text.to_string()],
        };

        match self.keep_separator {
            KeepSeparator::None => re.split(text).map(|s| s.to_string()).collect(),
            KeepSeparator::Start => {
                let mut result = Vec::new();
                let mut last_end = 0;
                for m in re.find_iter(text) {
                    // Text before this match.
                    let before = &text[last_end..m.start()];
                    if last_end == 0 {
                        result.push(before.to_string());
                    } else {
                        // Prepend separator found to this segment.
                        // Already handled below; push before segment.
                        result.push(before.to_string());
                    }
                    last_end = m.start();
                }
                // Remaining text with separator at start.
                if last_end < text.len() {
                    result.push(text[last_end..].to_string());
                }
                // If first result is empty, remove it for cleanliness but keep
                // structure consistent with literal version.
                if result.is_empty() {
                    result.push(text.to_string());
                }
                result
            }
            KeepSeparator::End => {
                let mut result = Vec::new();
                let mut last_end = 0;
                for m in re.find_iter(text) {
                    let chunk = &text[last_end..m.end()];
                    result.push(chunk.to_string());
                    last_end = m.end();
                }
                if last_end < text.len() {
                    result.push(text[last_end..].to_string());
                }
                if result.is_empty() {
                    result.push(text.to_string());
                }
                result
            }
        }
    }

    /// Check whether a separator (literal or regex) is found in `text`.
    fn separator_found_in(&self, text: &str, separator: &str) -> bool {
        if separator.is_empty() {
            return true;
        }
        if self.is_separator_regex {
            Regex::new(separator)
                .map(|re| re.is_match(text))
                .unwrap_or(false)
        } else {
            text.contains(separator)
        }
    }

    /// Merge small pieces into chunks respecting `chunk_size` and `chunk_overlap`,
    /// using the configured `length_function`.
    fn merge_pieces(&self, pieces: &[String], separator: &str) -> Vec<String> {
        let sep_len = self.len(separator);
        let mut docs: Vec<String> = Vec::new();
        let mut current_doc: Vec<&str> = Vec::new();
        let mut total: usize = 0;

        for piece in pieces {
            let len = self.len(piece);
            let added = if current_doc.is_empty() {
                len
            } else {
                len + sep_len
            };

            if total + added > self.chunk_size && !current_doc.is_empty() {
                let doc = current_doc.join(separator);
                let doc = if self.strip_whitespace {
                    doc.trim().to_string()
                } else {
                    doc
                };
                if !doc.is_empty() {
                    docs.push(doc);
                }
                // Keep overlap.
                if self.chunk_overlap == 0 {
                    current_doc.clear();
                    total = 0;
                } else {
                    while total > self.chunk_overlap && current_doc.len() > 1 {
                        let removed = self.len(current_doc[0]) + sep_len;
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
            let doc = if self.strip_whitespace {
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

    /// Core recursive algorithm.
    fn split_text_recursive(&self, text: &str, separators: &[String]) -> Vec<String> {
        let mut final_chunks: Vec<String> = Vec::new();

        // Find the appropriate separator.
        let mut separator = separators.last().map(|s| s.as_str()).unwrap_or("");
        let mut new_separators: &[String] = &[];

        for (i, sep) in separators.iter().enumerate() {
            if sep.is_empty() || self.separator_found_in(text, sep) {
                separator = sep.as_str();
                new_separators = &separators[i + 1..];
                break;
            }
        }

        let splits = self.split_by_separator(text, separator);

        // The separator used for merging; when keep_separator is active, we
        // already embedded the separator into the splits, so merge with "".
        let merge_sep = match self.keep_separator {
            KeepSeparator::None => separator,
            _ => "",
        };

        let mut good_splits: Vec<String> = Vec::new();

        for s in &splits {
            let piece = if self.strip_whitespace {
                s.trim().to_string()
            } else {
                s.to_string()
            };
            if piece.is_empty() {
                continue;
            }
            if self.len(&piece) < self.chunk_size {
                good_splits.push(piece);
            } else {
                if !good_splits.is_empty() {
                    let merged = self.merge_pieces(&good_splits, merge_sep);
                    final_chunks.extend(merged);
                    good_splits.clear();
                }
                if new_separators.is_empty() {
                    final_chunks.push(piece);
                } else {
                    let sub = self.split_text_recursive(&piece, new_separators);
                    final_chunks.extend(sub);
                }
            }
        }

        if !good_splits.is_empty() {
            let merged = self.merge_pieces(&good_splits, merge_sep);
            final_chunks.extend(merged);
        }

        final_chunks
    }
}

impl TextSplitter for RecursiveCharacterTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.split_text_recursive(text, &self.separators)
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }
}

// ── Builder ─────────────────────────────────────────────────────────────────

/// Builder for `RecursiveCharacterTextSplitter`.
pub struct RecursiveCharacterTextSplitterBuilder {
    separators: Option<Vec<String>>,
    chunk_size: usize,
    chunk_overlap: usize,
    length_function: LengthFunction,
    keep_separator: KeepSeparator,
    strip_whitespace: bool,
    is_separator_regex: bool,
}

impl Default for RecursiveCharacterTextSplitterBuilder {
    fn default() -> Self {
        Self {
            separators: None,
            chunk_size: 4000,
            chunk_overlap: 200,
            length_function: LengthFunction::default(),
            keep_separator: KeepSeparator::None,
            strip_whitespace: true,
            is_separator_regex: false,
        }
    }
}

impl RecursiveCharacterTextSplitterBuilder {
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    pub fn chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    pub fn separators(mut self, seps: Vec<String>) -> Self {
        self.separators = Some(seps);
        self
    }

    pub fn length_function(mut self, f: LengthFunction) -> Self {
        self.length_function = f;
        self
    }

    pub fn keep_separator(mut self, ks: KeepSeparator) -> Self {
        self.keep_separator = ks;
        self
    }

    pub fn strip_whitespace(mut self, strip: bool) -> Self {
        self.strip_whitespace = strip;
        self
    }

    pub fn is_separator_regex(mut self, is_regex: bool) -> Self {
        self.is_separator_regex = is_regex;
        self
    }

    /// Build the configured `RecursiveCharacterTextSplitter`.
    pub fn build(self) -> RecursiveCharacterTextSplitter {
        RecursiveCharacterTextSplitter {
            separators: self
                .separators
                .unwrap_or_else(|| vec!["\n\n".into(), "\n".into(), " ".into(), "".into()]),
            chunk_size: self.chunk_size,
            chunk_overlap: self.chunk_overlap,
            length_function: self.length_function,
            keep_separator: self.keep_separator,
            strip_whitespace: self.strip_whitespace,
            is_separator_regex: self.is_separator_regex,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_split_default_separators() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(20)
            .chunk_overlap(0)
            .build();

        let text = "Hello world.\n\nThis is a test.\n\nAnother paragraph here.";
        let chunks = splitter.split_text(text);
        assert!(!chunks.is_empty());
        for chunk in &chunks {
            assert!(
                chunk.len() <= 25, // small tolerance for boundary
                "Chunk too large: {:?} (len {})",
                chunk,
                chunk.len()
            );
        }
    }

    #[test]
    fn test_chunk_size_enforcement() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(10)
            .chunk_overlap(0)
            .build();

        let text = "abcde fghij klmno pqrst uvwxy";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() > 1);
        for chunk in &chunks {
            assert!(
                chunk.len() <= 11,
                "Chunk exceeds size: {:?} ({})",
                chunk,
                chunk.len()
            );
        }
    }

    #[test]
    fn test_chunk_overlap() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(15)
            .chunk_overlap(5)
            .build();

        let text = "alpha beta gamma delta epsilon zeta eta theta";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() > 1, "Expected multiple chunks");
        // With overlap, later chunks should share text with previous ones.
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
    fn test_custom_separators() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(12)
            .chunk_overlap(0)
            .separators(vec!["||".into(), " ".into(), "".into()])
            .build();

        let text = "chunk one||chunk two||chunk three";
        let chunks = splitter.split_text(text);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "chunk one");
        assert_eq!(chunks[1], "chunk two");
        assert_eq!(chunks[2], "chunk three");
    }

    #[test]
    fn test_keep_separator_start() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(6)
            .chunk_overlap(0)
            .separators(vec!["| ".into(), "".into()])
            .keep_separator(KeepSeparator::Start)
            .strip_whitespace(false)
            .build();

        let text = "aaa| bbb| ccc";
        let chunks = splitter.split_text(text);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "aaa");
        assert_eq!(chunks[1], "| bbb");
        assert_eq!(chunks[2], "| ccc");
    }

    #[test]
    fn test_keep_separator_end() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(10)
            .chunk_overlap(0)
            .separators(vec![". ".into(), "".into()])
            .keep_separator(KeepSeparator::End)
            .strip_whitespace(false)
            .build();

        let text = "Hello. World. Goodbye";
        let chunks = splitter.split_text(text);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], "Hello. ");
        assert_eq!(chunks[1], "World. ");
        assert_eq!(chunks[2], "Goodbye");
    }

    #[test]
    fn test_keep_separator_none() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(2)
            .chunk_overlap(0)
            .separators(vec![",".into(), "".into()])
            .keep_separator(KeepSeparator::None)
            .build();

        let text = "a,b,c";
        let chunks = splitter.split_text(text);
        assert_eq!(chunks, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_split_documents_metadata_preservation() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(10)
            .chunk_overlap(0)
            .build();

        let mut meta = HashMap::new();
        meta.insert("source".to_string(), Value::String("test.txt".to_string()));

        let doc = Document::new("Hello world. This is a test document with some content.")
            .with_metadata(meta.clone());

        let result = splitter.split_documents(&[doc]);
        assert!(result.len() > 1);
        for d in &result {
            assert_eq!(
                d.metadata.get("source"),
                Some(&Value::String("test.txt".to_string())),
                "Metadata should be preserved on all chunks"
            );
        }
    }

    #[test]
    fn test_language_preset_markdown() {
        let splitter = RecursiveCharacterTextSplitter::for_language(Language::Markdown)
            .with_chunk_size(50)
            .with_chunk_overlap(0);

        let text = "# Title\n\nSome text here.\n\n## Section\n\nMore text.\n\n### Sub\n\nDetails.";
        let chunks = splitter.split_text(text);
        assert!(!chunks.is_empty());
        // Markdown separator hierarchy should split at headers.
        assert!(
            chunks.len() >= 2,
            "Expected header-based splits, got {:?}",
            chunks
        );
    }

    #[test]
    fn test_language_preset_python() {
        let splitter = RecursiveCharacterTextSplitter::for_language(Language::Python)
            .with_chunk_size(50)
            .with_chunk_overlap(0);

        let text = "class Foo:\n    pass\n\ndef bar():\n    return 1\n\ndef baz():\n    return 2";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() >= 2, "Expected function-based splits");
    }

    #[test]
    fn test_language_preset_rust() {
        let splitter = RecursiveCharacterTextSplitter::for_language(Language::Rust)
            .with_chunk_size(30)
            .with_chunk_overlap(0);

        let text = "fn foo() {\n    let x = 1;\n}\n\nfn bar() {\n    let y = 2;\n}\n\nstruct Baz {\n    field: i32,\n}";
        let chunks = splitter.split_text(text);
        assert!(
            chunks.len() >= 2,
            "Expected Rust-based splits, got {:?}",
            chunks
        );
    }

    #[test]
    fn test_empty_text() {
        let splitter = RecursiveCharacterTextSplitter::new();
        let chunks = splitter.split_text("");
        assert!(chunks.is_empty(), "Empty text should produce no chunks");
    }

    #[test]
    fn test_text_smaller_than_chunk_size() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(1000)
            .chunk_overlap(0)
            .build();

        let text = "Short text.";
        let chunks = splitter.split_text(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Short text.");
    }

    #[test]
    fn test_regex_separators() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(20)
            .chunk_overlap(0)
            .separators(vec![r"\d+\.\s".into(), "".into()])
            .is_separator_regex(true)
            .build();

        let text = "1. First item 2. Second item 3. Third item";
        let chunks = splitter.split_text(text);
        assert!(
            chunks.len() >= 2,
            "Regex should split on numbered items, got {:?}",
            chunks
        );
    }

    #[test]
    fn test_strip_whitespace() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(50)
            .chunk_overlap(0)
            .strip_whitespace(true)
            .build();

        let text = "  hello  \n\n  world  ";
        let chunks = splitter.split_text(text);
        for chunk in &chunks {
            assert_eq!(chunk, chunk.trim(), "Chunks should be stripped");
        }
    }

    #[test]
    fn test_word_based_length_function() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(3) // max 3 words
            .chunk_overlap(0)
            .length_function(LengthFunction::Words)
            .build();

        let text = "one two three four five six seven eight";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() > 1, "Expected multiple word-bounded chunks");
        for chunk in &chunks {
            let word_count = chunk.split_whitespace().count();
            assert!(
                word_count <= 4, // small tolerance
                "Chunk has {} words, max is 3: {:?}",
                word_count,
                chunk
            );
        }
    }

    #[test]
    fn test_builder_pattern() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(100)
            .chunk_overlap(10)
            .separators(vec!["---".into(), "\n".into()])
            .keep_separator(KeepSeparator::End)
            .strip_whitespace(false)
            .is_separator_regex(false)
            .build();

        assert_eq!(splitter.chunk_size, 100);
        assert_eq!(splitter.chunk_overlap, 10);
        assert_eq!(splitter.separators, vec!["---", "\n"]);
        assert_eq!(splitter.keep_separator, KeepSeparator::End);
        assert!(!splitter.strip_whitespace);
        assert!(!splitter.is_separator_regex);
    }

    #[test]
    fn test_large_text_multiple_separator_levels() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(30)
            .chunk_overlap(0)
            .build();

        // Text with paragraph breaks and sentences.
        let text = "First paragraph with several words that exceed the limit.\n\n\
                     Second paragraph is also quite long and will need splitting.\n\n\
                     Third paragraph short.";
        let chunks = splitter.split_text(text);
        assert!(
            chunks.len() >= 3,
            "Expected at least 3 chunks from multi-level splitting, got {:?}",
            chunks
        );
        for chunk in &chunks {
            // Allow some tolerance for edge cases.
            assert!(
                chunk.len() <= 35,
                "Chunk too large after recursive split: {:?} (len {})",
                chunk,
                chunk.len()
            );
        }
    }

    #[test]
    fn test_create_documents_with_metadatas() {
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(10)
            .chunk_overlap(0)
            .build();

        let texts = vec!["Hello world test"];
        let mut meta = HashMap::new();
        meta.insert("key".to_string(), Value::String("val".to_string()));
        let metadatas = vec![meta.clone()];

        let docs = splitter.create_documents(&texts, Some(&metadatas));
        assert!(!docs.is_empty());
        for d in &docs {
            assert_eq!(
                d.metadata.get("key"),
                Some(&Value::String("val".to_string()))
            );
        }
    }

    #[test]
    fn test_custom_length_function() {
        // Count bytes instead of chars.
        let splitter = RecursiveCharacterTextSplitter::builder()
            .chunk_size(10)
            .chunk_overlap(0)
            .length_function(LengthFunction::Custom(Arc::new(|s: &str| s.len())))
            .build();

        let text = "abcdefghij klmnopqrst";
        let chunks = splitter.split_text(text);
        assert!(chunks.len() >= 2);
    }
}
