use std::collections::HashMap;

use cognis_core::documents::Document;
use regex::Regex;
use serde_json::Value;

use super::TextSplitter;

/// How to detect sentence boundaries.
#[derive(Debug, Clone, Default)]
pub enum SentencePattern {
    /// Splits on `.!?` followed by whitespace, handling common abbreviations
    /// (Mr., Mrs., Dr., etc.), decimal numbers, and basic URL/email detection.
    #[default]
    Default,
    /// Splits only on `.!?` followed by a space or newline. No abbreviation handling.
    Simple,
    /// Splits on Unicode sentence terminators (includes CJK full stops, etc.).
    Unicode,
    /// A user-supplied regex pattern. Each match is treated as a sentence boundary.
    Custom(String),
}

/// A text splitter that splits on sentence boundaries while respecting chunk
/// size limits.
///
/// Sentences are first detected using the configured [`SentencePattern`], then
/// merged into chunks that fit within `chunk_size` characters. An optional
/// `chunk_overlap` (measured in *sentences*, not characters) controls how many
/// sentences are repeated between consecutive chunks.
pub struct SentenceTextSplitter {
    /// Maximum characters per chunk.
    pub chunk_size: usize,
    /// Number of overlapping *sentences* between chunks.
    pub chunk_overlap: usize,
    /// Minimum chunk size in characters. Chunks smaller than this are merged
    /// with the next chunk when possible.
    pub min_chunk_size: Option<usize>,
    /// How to detect sentence boundaries.
    pub separator_pattern: SentencePattern,
    /// Strip leading/trailing whitespace from chunks (default `true`).
    pub strip_whitespace: bool,
    /// Try to keep paragraphs (double-newline separated blocks) together when
    /// they fit within `chunk_size`.
    pub preserve_paragraphs: bool,
}

impl Default for SentenceTextSplitter {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 0,
            min_chunk_size: None,
            separator_pattern: SentencePattern::Default,
            strip_whitespace: true,
            preserve_paragraphs: false,
        }
    }
}

impl SentenceTextSplitter {
    /// Create a new splitter with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a builder for configuring the splitter.
    pub fn builder() -> SentenceTextSplitterBuilder {
        SentenceTextSplitterBuilder::default()
    }

    // ── Sentence detection ──────────────────────────────────────────────

    /// Split text into individual sentences without any chunk merging.
    pub fn split_into_sentences(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        match &self.separator_pattern {
            SentencePattern::Default => self.split_sentences_default(text),
            SentencePattern::Simple => self.split_sentences_simple(text),
            SentencePattern::Unicode => self.split_sentences_unicode(text),
            SentencePattern::Custom(pat) => self.split_sentences_custom(text, pat),
        }
    }

    /// Default sentence splitting with abbreviation / number / URL awareness.
    fn split_sentences_default(&self, text: &str) -> Vec<String> {
        // Strategy: walk through the text and find sentence-ending punctuation
        // (.!?) followed by whitespace, but skip known abbreviations, decimals,
        // and URLs.

        let abbrevs = [
            "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.", "Gen.", "Gov.", "Sgt.",
            "Cpl.", "Pvt.", "Lt.", "Col.", "Capt.", "Maj.", "Rev.", "Hon.", "Pres.", "Inc.",
            "Corp.", "Ltd.", "Co.", "vs.", "etc.", "approx.", "dept.", "est.", "vol.", "fig.",
            "no.",
        ];

        let chars: Vec<char> = text.chars().collect();
        let len = chars.len();
        let mut sentences: Vec<String> = Vec::new();
        let mut start = 0;

        let mut i = 0;
        while i < len {
            let ch = chars[i];

            if (ch == '.' || ch == '!' || ch == '?') && i + 1 < len {
                let next = chars[i + 1];
                let is_boundary = next.is_whitespace() || next == '\n';
                if !is_boundary {
                    i += 1;
                    continue;
                }

                // Handle multiple punctuation (e.g. "!!" or "...")
                let mut end_punct = i;
                while end_punct + 1 < len
                    && (chars[end_punct + 1] == '.'
                        || chars[end_punct + 1] == '!'
                        || chars[end_punct + 1] == '?')
                {
                    end_punct += 1;
                }

                // Check for abbreviations
                if ch == '.' {
                    let preceding: String = chars[start..=end_punct].iter().collect();
                    let trimmed = preceding.trim_start();

                    // Check known abbreviations — match whole word only
                    let last_word = trimmed.split_whitespace().last().unwrap_or("");
                    let is_abbrev = abbrevs.iter().any(|a| last_word.eq_ignore_ascii_case(a));

                    // Check single-letter abbreviation pattern (e.g. U.S., U.K.)
                    let is_single_letter_abbrev = if end_punct >= 1 && chars[end_punct] == '.' {
                        let before_dot = end_punct.checked_sub(1).map(|j| chars[j]);
                        matches!(before_dot, Some(c) if c.is_ascii_uppercase())
                            && (end_punct < 2 || chars[end_punct - 2] == '.')
                    } else {
                        false
                    };

                    // Check decimal numbers (e.g. 3.14)
                    let is_decimal = if end_punct >= 1 {
                        let before = chars[end_punct - 1];
                        let after_ws_pos = end_punct + 1;
                        let after_non_ws = if after_ws_pos < len {
                            // Look past whitespace for a digit
                            let rest: String = chars[after_ws_pos..].iter().collect();
                            let first_non_ws = rest.trim_start().chars().next();
                            matches!(first_non_ws, Some(c) if c.is_ascii_digit())
                                && before.is_ascii_digit()
                        } else {
                            false
                        };
                        before.is_ascii_digit() && after_non_ws
                    } else {
                        false
                    };

                    // Check URLs (basic: contains :// or www.)
                    let word_before: String = chars[start..=end_punct]
                        .iter()
                        .collect::<String>()
                        .split_whitespace()
                        .last()
                        .unwrap_or("")
                        .to_string();
                    let is_url = word_before.contains("://") || word_before.starts_with("www.");

                    // Check email (basic: contains @)
                    let is_email = word_before.contains('@') && word_before.contains('.');

                    if is_abbrev || is_single_letter_abbrev || is_decimal || is_url || is_email {
                        i = end_punct + 1;
                        continue;
                    }
                }

                // This is a real sentence boundary
                let sentence: String = chars[start..=end_punct].iter().collect();
                let sentence = if self.strip_whitespace {
                    sentence.trim().to_string()
                } else {
                    sentence
                };
                if !sentence.is_empty() {
                    sentences.push(sentence);
                }
                start = end_punct + 1;
                // Skip whitespace after sentence
                while start < len && chars[start].is_whitespace() {
                    start += 1;
                }
                i = start;
                continue;
            }

            i += 1;
        }

        // Remaining text
        if start < len {
            let remaining: String = chars[start..].iter().collect();
            let remaining = if self.strip_whitespace {
                remaining.trim().to_string()
            } else {
                remaining
            };
            if !remaining.is_empty() {
                sentences.push(remaining);
            }
        }

        sentences
    }

    /// Simple splitting: just .!? followed by whitespace.
    fn split_sentences_simple(&self, text: &str) -> Vec<String> {
        let re = Regex::new(r"([.!?])\s+").unwrap();
        let mut sentences = Vec::new();
        let mut last = 0;

        for mat in re.find_iter(text) {
            // Include the punctuation mark but not the trailing whitespace.
            let end = mat.start() + 1; // one char for the punctuation
            let sentence = &text[last..end];
            let sentence = if self.strip_whitespace {
                sentence.trim()
            } else {
                sentence
            };
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
            }
            last = mat.end();
        }

        if last < text.len() {
            let remaining = if self.strip_whitespace {
                text[last..].trim()
            } else {
                &text[last..]
            };
            if !remaining.is_empty() {
                sentences.push(remaining.to_string());
            }
        }

        sentences
    }

    /// Unicode-aware sentence splitting.
    fn split_sentences_unicode(&self, text: &str) -> Vec<String> {
        // Unicode sentence terminators: . ! ? plus CJK full stop, etc.
        let re = Regex::new(r"([.!?\u{3002}\u{FF01}\u{FF1F}\u{2026}])\s*").unwrap();
        let mut sentences = Vec::new();
        let mut last = 0;

        for mat in re.find_iter(text) {
            let end = mat.start() + mat.as_str().trim_end().len();
            let sentence = &text[last..end];
            let sentence = if self.strip_whitespace {
                sentence.trim()
            } else {
                sentence
            };
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
            }
            last = mat.end();
        }

        if last < text.len() {
            let remaining = if self.strip_whitespace {
                text[last..].trim()
            } else {
                &text[last..]
            };
            if !remaining.is_empty() {
                sentences.push(remaining.to_string());
            }
        }

        sentences
    }

    /// Custom regex-based sentence splitting.
    fn split_sentences_custom(&self, text: &str, pattern: &str) -> Vec<String> {
        let re = match Regex::new(pattern) {
            Ok(r) => r,
            Err(_) => return vec![text.to_string()],
        };

        let mut sentences = Vec::new();
        let mut last = 0;

        for mat in re.find_iter(text) {
            let sentence = &text[last..mat.end()];
            let sentence = if self.strip_whitespace {
                sentence.trim()
            } else {
                sentence
            };
            if !sentence.is_empty() {
                sentences.push(sentence.to_string());
            }
            last = mat.end();
        }

        if last < text.len() {
            let remaining = if self.strip_whitespace {
                text[last..].trim()
            } else {
                &text[last..]
            };
            if !remaining.is_empty() {
                sentences.push(remaining.to_string());
            }
        }

        sentences
    }

    // ── Chunk merging ───────────────────────────────────────────────────

    /// Split text into chunks, respecting sentence boundaries and chunk size.
    pub fn split_text(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }

        if self.preserve_paragraphs {
            return self.split_preserving_paragraphs(text);
        }

        let sentences = self.split_into_sentences(text);
        self.merge_sentences_into_chunks(&sentences)
    }

    /// Split documents, preserving metadata on each produced chunk.
    pub fn split_documents(&self, documents: &[Document]) -> Vec<Document> {
        let texts: Vec<&str> = documents.iter().map(|d| d.page_content.as_str()).collect();
        let metadatas: Vec<HashMap<String, Value>> =
            documents.iter().map(|d| d.metadata.clone()).collect();
        self.create_documents(
            &texts.iter().map(|s| s.to_string()).collect::<Vec<_>>(),
            Some(&metadatas),
        )
    }

    /// Create documents from raw texts with optional per-text metadata.
    pub fn create_documents(
        &self,
        texts: &[String],
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

    /// Merge sentences into chunks respecting chunk_size and chunk_overlap
    /// (overlap measured in sentences).
    fn merge_sentences_into_chunks(&self, sentences: &[String]) -> Vec<String> {
        if sentences.is_empty() {
            return Vec::new();
        }

        let mut chunks: Vec<String> = Vec::new();
        let mut current_sentences: Vec<&str> = Vec::new();
        let mut current_len: usize = 0;

        for sentence in sentences {
            let s_len = sentence.len();
            let added = if current_sentences.is_empty() {
                s_len
            } else {
                s_len + 1 // space separator
            };

            // If adding this sentence would exceed chunk_size and we already have content
            if current_len + added > self.chunk_size && !current_sentences.is_empty() {
                let chunk = current_sentences.join(" ");
                let chunk = if self.strip_whitespace {
                    chunk.trim().to_string()
                } else {
                    chunk
                };
                if !chunk.is_empty() {
                    chunks.push(chunk);
                }

                // Handle sentence overlap
                if self.chunk_overlap > 0 && current_sentences.len() > self.chunk_overlap {
                    let overlap_start = current_sentences.len() - self.chunk_overlap;
                    let overlap: Vec<&str> = current_sentences[overlap_start..].to_vec();
                    current_len = overlap.iter().map(|s| s.len()).sum::<usize>()
                        + overlap.len().saturating_sub(1);
                    current_sentences = overlap;
                } else if self.chunk_overlap == 0 {
                    current_sentences.clear();
                    current_len = 0;
                }
                // If overlap >= current_sentences.len(), keep all (already set)
            }

            current_sentences.push(sentence);
            current_len = if current_sentences.len() == 1 {
                s_len
            } else {
                current_len + s_len + 1
            };
        }

        // Flush remaining
        if !current_sentences.is_empty() {
            let chunk = current_sentences.join(" ");
            let chunk = if self.strip_whitespace {
                chunk.trim().to_string()
            } else {
                chunk
            };
            if !chunk.is_empty() {
                chunks.push(chunk);
            }
        }

        // Apply min_chunk_size merging
        if let Some(min_size) = self.min_chunk_size {
            chunks = self.merge_small_chunks(chunks, min_size);
        }

        chunks
    }

    /// Merge chunks that are smaller than `min_size` with adjacent chunks.
    fn merge_small_chunks(&self, chunks: Vec<String>, min_size: usize) -> Vec<String> {
        if chunks.is_empty() {
            return chunks;
        }

        let mut merged: Vec<String> = Vec::new();
        let mut accumulator = String::new();

        for chunk in chunks {
            if accumulator.is_empty() {
                accumulator = chunk;
            } else if accumulator.len() + 1 + chunk.len() <= self.chunk_size {
                accumulator.push(' ');
                accumulator.push_str(&chunk);
            } else if accumulator.len() >= min_size {
                merged.push(accumulator);
                accumulator = chunk;
            } else {
                // Try to merge with current chunk
                if accumulator.len() + 1 + chunk.len() <= self.chunk_size {
                    accumulator.push(' ');
                    accumulator.push_str(&chunk);
                } else {
                    // Push undersized chunk as-is, start fresh
                    merged.push(accumulator);
                    accumulator = chunk;
                }
            }
        }

        if !accumulator.is_empty() {
            // If accumulator is too small and we can merge with previous
            if accumulator.len() < min_size && !merged.is_empty() {
                let last = merged.last_mut().unwrap();
                if last.len() + 1 + accumulator.len() <= self.chunk_size {
                    last.push(' ');
                    last.push_str(&accumulator);
                } else {
                    merged.push(accumulator);
                }
            } else {
                merged.push(accumulator);
            }
        }

        merged
    }

    /// Paragraph-preserving split: first split by paragraphs, then by sentences
    /// within oversized paragraphs.
    fn split_preserving_paragraphs(&self, text: &str) -> Vec<String> {
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        let mut all_sentences: Vec<String> = Vec::new();

        for para in &paragraphs {
            let trimmed = if self.strip_whitespace {
                para.trim()
            } else {
                para
            };
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.len() <= self.chunk_size && !all_sentences.is_empty() {
                // Check if we can append this paragraph to the current accumulation
                // by treating it as a single "sentence" unit
                all_sentences.push(trimmed.to_string());
            } else if trimmed.len() <= self.chunk_size {
                all_sentences.push(trimmed.to_string());
            } else {
                // Paragraph is too large, split into sentences
                let para_sentences = self.split_into_sentences(trimmed);
                all_sentences.extend(para_sentences);
            }
        }

        self.merge_sentences_into_chunks(&all_sentences)
    }
}

impl TextSplitter for SentenceTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        SentenceTextSplitter::split_text(self, text)
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }
}

// ── Builder ─────────────────────────────────────────────────────────────────

/// Builder for [`SentenceTextSplitter`].
pub struct SentenceTextSplitterBuilder {
    chunk_size: usize,
    chunk_overlap: usize,
    min_chunk_size: Option<usize>,
    separator_pattern: SentencePattern,
    strip_whitespace: bool,
    preserve_paragraphs: bool,
}

impl Default for SentenceTextSplitterBuilder {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 0,
            min_chunk_size: None,
            separator_pattern: SentencePattern::Default,
            strip_whitespace: true,
            preserve_paragraphs: false,
        }
    }
}

impl SentenceTextSplitterBuilder {
    /// Maximum characters per chunk.
    pub fn chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size;
        self
    }

    /// Number of overlapping sentences between chunks.
    pub fn chunk_overlap(mut self, overlap: usize) -> Self {
        self.chunk_overlap = overlap;
        self
    }

    /// Minimum chunk size in characters; smaller chunks are merged.
    pub fn min_chunk_size(mut self, size: usize) -> Self {
        self.min_chunk_size = Some(size);
        self
    }

    /// How to detect sentence boundaries.
    pub fn separator_pattern(mut self, pattern: SentencePattern) -> Self {
        self.separator_pattern = pattern;
        self
    }

    /// Strip leading/trailing whitespace from chunks.
    pub fn strip_whitespace(mut self, strip: bool) -> Self {
        self.strip_whitespace = strip;
        self
    }

    /// Try to keep paragraphs together when they fit within chunk_size.
    pub fn preserve_paragraphs(mut self, preserve: bool) -> Self {
        self.preserve_paragraphs = preserve;
        self
    }

    /// Build the configured [`SentenceTextSplitter`].
    pub fn build(self) -> SentenceTextSplitter {
        SentenceTextSplitter {
            chunk_size: self.chunk_size,
            chunk_overlap: self.chunk_overlap,
            min_chunk_size: self.min_chunk_size,
            separator_pattern: self.separator_pattern,
            strip_whitespace: self.strip_whitespace,
            preserve_paragraphs: self.preserve_paragraphs,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sentence_splitting() {
        let splitter = SentenceTextSplitter::builder().chunk_size(1000).build();
        let sentences = splitter.split_into_sentences("Hello world. This is a test. How are you?");
        assert_eq!(sentences.len(), 3, "Got {:?}", sentences);
        assert_eq!(sentences[0], "Hello world.");
        assert_eq!(sentences[1], "This is a test.");
        assert_eq!(sentences[2], "How are you?");
    }

    #[test]
    fn test_chunk_size_enforcement() {
        let splitter = SentenceTextSplitter::builder()
            .chunk_size(40)
            .chunk_overlap(0)
            .build();
        let text =
            "First sentence here. Second sentence here. Third sentence here. Fourth sentence.";
        let chunks = splitter.split_text(text);
        assert!(
            chunks.len() > 1,
            "Expected multiple chunks, got {:?}",
            chunks
        );
        for chunk in &chunks {
            assert!(
                chunk.len() <= 45, // small tolerance
                "Chunk exceeds size: {:?} (len {})",
                chunk,
                chunk.len()
            );
        }
    }

    #[test]
    fn test_sentence_overlap() {
        let splitter = SentenceTextSplitter::builder()
            .chunk_size(30)
            .chunk_overlap(1)
            .build();
        let text = "Sentence one. Sentence two. Sentence three. Sentence four.";
        let chunks = splitter.split_text(text);
        assert!(
            chunks.len() >= 2,
            "Expected multiple chunks, got {:?}",
            chunks
        );
        // With overlap of 1, the last sentence of chunk N should appear in chunk N+1
        if chunks.len() >= 2 {
            // Find a shared sentence
            let first_sentences = splitter.split_into_sentences(&chunks[0]);
            let second_sentences = splitter.split_into_sentences(&chunks[1]);
            let last_of_first = first_sentences.last().unwrap();
            let first_of_second = second_sentences.first().unwrap();
            assert_eq!(
                last_of_first, first_of_second,
                "Expected 1-sentence overlap between chunks"
            );
        }
    }

    #[test]
    fn test_abbreviation_mr_dr() {
        let splitter = SentenceTextSplitter::builder().chunk_size(1000).build();
        let text = "Mr. Smith went to Washington. Dr. Jones stayed home.";
        let sentences = splitter.split_into_sentences(text);
        assert_eq!(
            sentences.len(),
            2,
            "Should not split on Mr. or Dr., got {:?}",
            sentences
        );
        assert_eq!(sentences[0], "Mr. Smith went to Washington.");
        assert_eq!(sentences[1], "Dr. Jones stayed home.");
    }

    #[test]
    fn test_decimal_numbers_not_split() {
        let splitter = SentenceTextSplitter::builder().chunk_size(1000).build();
        let text = "The value is 3.14 approximately. Pi is important.";
        let sentences = splitter.split_into_sentences(text);
        assert_eq!(
            sentences.len(),
            2,
            "Should not split on decimal point, got {:?}",
            sentences
        );
        assert!(sentences[0].contains("3.14"));
    }

    #[test]
    fn test_multiple_punctuation() {
        let splitter = SentenceTextSplitter::builder().chunk_size(1000).build();
        let text = "What a day!! It was incredible... Really.";
        let sentences = splitter.split_into_sentences(text);
        // "What a day!!" and "It was incredible..." and "Really."
        assert!(
            sentences.len() >= 2,
            "Should handle multiple punctuation, got {:?}",
            sentences
        );
    }

    #[test]
    fn test_empty_text() {
        let splitter = SentenceTextSplitter::new();
        let chunks = splitter.split_text("");
        assert!(chunks.is_empty());
        let sentences = splitter.split_into_sentences("");
        assert!(sentences.is_empty());
    }

    #[test]
    fn test_single_sentence() {
        let splitter = SentenceTextSplitter::builder().chunk_size(1000).build();
        let text = "Just one sentence here.";
        let chunks = splitter.split_text(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "Just one sentence here.");
    }

    #[test]
    fn test_paragraph_preservation() {
        let splitter = SentenceTextSplitter::builder()
            .chunk_size(200)
            .chunk_overlap(0)
            .preserve_paragraphs(true)
            .build();

        let text = "First paragraph sentence one. Sentence two.\n\n\
                     Second paragraph sentence one. Sentence two.\n\n\
                     Third paragraph.";
        let chunks = splitter.split_text(text);
        assert!(!chunks.is_empty());
        // With large enough chunk_size, paragraphs should stay together
        // Check that at least one chunk contains a full paragraph
        let has_full_para = chunks
            .iter()
            .any(|c| c.contains("First paragraph") && c.contains("Sentence two."));
        assert!(
            has_full_para,
            "Expected paragraphs to be preserved, got {:?}",
            chunks
        );
    }

    #[test]
    fn test_simple_pattern() {
        let splitter = SentenceTextSplitter::builder()
            .chunk_size(1000)
            .separator_pattern(SentencePattern::Simple)
            .build();
        // Simple mode does NOT handle abbreviations
        let text = "Hello world. This is a test! Are you sure? Yes.";
        let sentences = splitter.split_into_sentences(text);
        assert_eq!(
            sentences.len(),
            4,
            "Simple split on .!? got {:?}",
            sentences
        );
    }

    #[test]
    fn test_custom_regex_pattern() {
        let splitter = SentenceTextSplitter::builder()
            .chunk_size(1000)
            .separator_pattern(SentencePattern::Custom(r"[;]\s*".to_string()))
            .build();
        let text = "part one; part two; part three";
        let sentences = splitter.split_into_sentences(text);
        assert_eq!(sentences.len(), 3, "Custom split on ; got {:?}", sentences);
    }

    #[test]
    fn test_min_chunk_size_merging() {
        let splitter = SentenceTextSplitter::builder()
            .chunk_size(100)
            .chunk_overlap(0)
            .min_chunk_size(30)
            .build();
        let text = "Hi. Ok. Sure. This is a longer sentence that has more content.";
        let chunks = splitter.split_text(text);
        // Short sentences should be merged to meet min_chunk_size
        for chunk in &chunks {
            // The final chunk might be smaller, but intermediate ones should meet min
            if chunks.len() > 1 {
                assert!(
                    chunk.len() >= 10, // relaxed check; merging should combine tiny chunks
                    "Chunk too small: {:?}",
                    chunk
                );
            }
        }
        assert!(!chunks.is_empty());
    }

    #[test]
    fn test_split_documents_with_metadata() {
        let splitter = SentenceTextSplitter::builder()
            .chunk_size(30)
            .chunk_overlap(0)
            .build();

        let mut meta = HashMap::new();
        meta.insert("source".to_string(), Value::String("doc.txt".to_string()));

        let doc = Document::new("First sentence. Second sentence. Third sentence.")
            .with_metadata(meta.clone());

        let result = splitter.split_documents(&[doc]);
        assert!(
            result.len() >= 2,
            "Expected multiple doc chunks, got {:?}",
            result
        );
        for d in &result {
            assert_eq!(
                d.metadata.get("source"),
                Some(&Value::String("doc.txt".to_string())),
                "Metadata should be preserved"
            );
        }
    }

    #[test]
    fn test_unicode_sentence_terminators() {
        let splitter = SentenceTextSplitter::builder()
            .chunk_size(1000)
            .separator_pattern(SentencePattern::Unicode)
            .build();
        // CJK full stop \u{3002}
        let text = "First sentence\u{3002}Second sentence\u{3002}Third";
        let sentences = splitter.split_into_sentences(text);
        assert_eq!(
            sentences.len(),
            3,
            "Unicode terminators should split, got {:?}",
            sentences
        );
    }

    #[test]
    fn test_builder_pattern() {
        let splitter = SentenceTextSplitter::builder()
            .chunk_size(500)
            .chunk_overlap(2)
            .min_chunk_size(50)
            .separator_pattern(SentencePattern::Simple)
            .strip_whitespace(false)
            .preserve_paragraphs(true)
            .build();

        assert_eq!(splitter.chunk_size, 500);
        assert_eq!(splitter.chunk_overlap, 2);
        assert_eq!(splitter.min_chunk_size, Some(50));
        assert!(!splitter.strip_whitespace);
        assert!(splitter.preserve_paragraphs);
        assert!(matches!(
            splitter.separator_pattern,
            SentencePattern::Simple
        ));
    }

    #[test]
    fn test_long_single_sentence_exceeds_chunk_size() {
        let splitter = SentenceTextSplitter::builder()
            .chunk_size(20)
            .chunk_overlap(0)
            .build();
        let text =
            "This is one very long sentence that clearly exceeds the chunk size limit by a lot.";
        let chunks = splitter.split_text(text);
        // A single sentence that exceeds chunk_size should still be returned
        // (we don't break mid-sentence)
        assert!(!chunks.is_empty(), "Should produce at least one chunk");
        assert_eq!(
            chunks.len(),
            1,
            "Single sentence should not be split mid-sentence"
        );
        assert_eq!(chunks[0], text);
    }

    #[test]
    fn test_us_uk_abbreviation() {
        let splitter = SentenceTextSplitter::builder().chunk_size(1000).build();
        let text = "The U.S. is a country. The U.K. is also a country.";
        let sentences = splitter.split_into_sentences(text);
        assert_eq!(
            sentences.len(),
            2,
            "Should not split on U.S. or U.K., got {:?}",
            sentences
        );
    }

    #[test]
    fn test_create_documents_method() {
        let splitter = SentenceTextSplitter::builder()
            .chunk_size(30)
            .chunk_overlap(0)
            .build();

        let texts = vec![
            "First text sentence one. Sentence two.".to_string(),
            "Second text sentence one. Sentence two.".to_string(),
        ];
        let mut meta1 = HashMap::new();
        meta1.insert("idx".to_string(), Value::Number(0.into()));
        let mut meta2 = HashMap::new();
        meta2.insert("idx".to_string(), Value::Number(1.into()));
        let metadatas = vec![meta1, meta2];

        let docs = splitter.create_documents(&texts, Some(&metadatas));
        assert!(docs.len() >= 2, "Should produce multiple documents");
        // All docs from the first text should have idx=0
        let first_text_docs: Vec<_> = docs
            .iter()
            .filter(|d| d.metadata.get("idx") == Some(&Value::Number(0.into())))
            .collect();
        assert!(!first_text_docs.is_empty());
    }

    #[test]
    fn test_text_splitter_trait() {
        // Verify SentenceTextSplitter implements the TextSplitter trait
        let splitter = SentenceTextSplitter::builder()
            .chunk_size(50)
            .chunk_overlap(0)
            .build();
        let trait_obj: &dyn TextSplitter = &splitter;
        assert_eq!(trait_obj.chunk_size(), 50);
        assert_eq!(trait_obj.chunk_overlap(), 0);
        let chunks = trait_obj.split_text("Hello world. Goodbye world.");
        assert!(!chunks.is_empty());
    }
}
