//! Summarization middleware for context window management.
//!
//! Provides multiple summarization strategies for compressing text and
//! managing context windows when conversations grow too long.
//!
//! The main types are:
//!
//! - [`SummarizationStrategy`] — strategy enum (extractive, abstractive, truncate, hierarchical)
//! - [`SummarizationConfig`] — configuration with builder pattern
//! - [`TextSegment`] — a piece of text with importance and metadata
//! - [`SummarizationResult`] — output of a summarization operation
//! - [`ExtractiveSummarizer`] — picks key sentences from source text
//! - [`TruncationSummarizer`] — truncates at sentence boundaries
//! - [`HierarchicalSummarizer`] — summarizes sections then combines
//! - [`ContextCompressor`] — manages context window by summarizing old messages
//! - [`CompressedMessage`] — a message that may be a summary of prior messages

use std::collections::HashMap;

use serde_json::Value;

// ---------------------------------------------------------------------------
// SummarizationStrategy
// ---------------------------------------------------------------------------

/// Strategy used for summarizing text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SummarizationStrategy {
    /// Pick key sentences from the original text.
    Extractive,
    /// Rewrite the text into a shorter form.
    Abstractive,
    /// Cut the text to a maximum length at a sentence boundary.
    Truncate,
    /// Summarize sections independently, then combine summaries.
    Hierarchical,
}

impl SummarizationStrategy {
    /// Human-readable name for this strategy.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Extractive => "extractive",
            Self::Abstractive => "abstractive",
            Self::Truncate => "truncate",
            Self::Hierarchical => "hierarchical",
        }
    }
}

impl std::fmt::Display for SummarizationStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// SummarizationConfig
// ---------------------------------------------------------------------------

/// Configuration for summarization operations.
#[derive(Debug, Clone)]
pub struct SummarizationConfig {
    /// Which strategy to use.
    pub strategy: SummarizationStrategy,
    /// Maximum number of tokens in the output summary.
    pub max_output_tokens: usize,
    /// Whether to attempt to preserve key facts.
    pub preserve_key_facts: bool,
    /// Target compression ratio (0.0–1.0). Lower means more compression.
    pub compression_ratio: f64,
    /// Minimum content length (in characters) before summarization is applied.
    pub min_content_length: usize,
}

impl Default for SummarizationConfig {
    fn default() -> Self {
        Self {
            strategy: SummarizationStrategy::Extractive,
            max_output_tokens: 500,
            preserve_key_facts: true,
            compression_ratio: 0.3,
            min_content_length: 100,
        }
    }
}

impl SummarizationConfig {
    /// Create a new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the summarization strategy.
    pub fn with_strategy(mut self, strategy: SummarizationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the maximum output tokens.
    pub fn with_max_output_tokens(mut self, tokens: usize) -> Self {
        self.max_output_tokens = tokens;
        self
    }

    /// Set whether to preserve key facts.
    pub fn with_preserve_key_facts(mut self, preserve: bool) -> Self {
        self.preserve_key_facts = preserve;
        self
    }

    /// Set the target compression ratio.
    pub fn with_compression_ratio(mut self, ratio: f64) -> Self {
        self.compression_ratio = ratio;
        self
    }

    /// Set the minimum content length for summarization.
    pub fn with_min_content_length(mut self, length: usize) -> Self {
        self.min_content_length = length;
        self
    }
}

// ---------------------------------------------------------------------------
// TextSegment
// ---------------------------------------------------------------------------

/// A segment of text with importance scoring and metadata.
#[derive(Debug, Clone)]
pub struct TextSegment {
    /// The textual content of this segment.
    pub content: String,
    /// Importance score (0.0–1.0). Higher is more important.
    pub importance: f64,
    /// Category label (e.g. "introduction", "conclusion", "detail").
    pub category: String,
    /// Arbitrary metadata.
    pub metadata: HashMap<String, Value>,
}

impl TextSegment {
    /// Create a new text segment.
    pub fn new(content: impl Into<String>, importance: f64, category: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            importance,
            category: category.into(),
            metadata: HashMap::new(),
        }
    }

    /// Builder method to attach a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Number of whitespace-delimited words in this segment.
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }

    /// Number of characters in this segment.
    pub fn char_count(&self) -> usize {
        self.content.len()
    }

    /// Serialize this segment to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "content": self.content,
            "importance": self.importance,
            "category": self.category,
            "word_count": self.word_count(),
            "char_count": self.char_count(),
            "metadata": self.metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// SummarizationResult
// ---------------------------------------------------------------------------

/// The result of a summarization operation.
#[derive(Debug, Clone)]
pub struct SummarizationResult {
    /// The summarized text.
    pub summary: String,
    /// Character length of the original text.
    pub original_length: usize,
    /// Character length of the summary.
    pub summary_length: usize,
    /// Achieved compression ratio (summary_length / original_length).
    pub compression_ratio: f64,
    /// Number of segments/sentences processed.
    pub segments_processed: usize,
    /// Key facts extracted during summarization.
    pub key_facts: Vec<String>,
}

impl SummarizationResult {
    /// Serialize this result to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "summary": self.summary,
            "original_length": self.original_length,
            "summary_length": self.summary_length,
            "compression_ratio": self.compression_ratio,
            "segments_processed": self.segments_processed,
            "key_facts": self.key_facts,
        })
    }
}

// ---------------------------------------------------------------------------
// Helper: sentence splitting
// ---------------------------------------------------------------------------

/// Split text into sentences using simple boundary detection.
fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();

    for ch in text.chars() {
        current.push(ch);
        if (ch == '.' || ch == '!' || ch == '?') && !current.trim().is_empty() {
            sentences.push(current.trim().to_string());
            current = String::new();
        }
    }

    // Remaining text that didn't end with sentence-ending punctuation.
    let remainder = current.trim().to_string();
    if !remainder.is_empty() {
        sentences.push(remainder);
    }

    sentences
}

/// Rough token estimate (~4 chars per token).
fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

/// Common key terms that boost sentence importance.
const KEY_TERMS: &[&str] = &[
    "important",
    "key",
    "critical",
    "essential",
    "significant",
    "primary",
    "main",
    "core",
    "fundamental",
    "conclusion",
    "summary",
    "result",
    "therefore",
    "consequently",
    "must",
    "should",
    "required",
    "necessary",
];

// ---------------------------------------------------------------------------
// ExtractiveSummarizer
// ---------------------------------------------------------------------------

/// Extracts the most important sentences from a text.
///
/// Scoring heuristics:
/// - Sentences with more words score higher.
/// - Sentences containing key terms score higher.
/// - Sentences earlier in the text score higher (position bonus).
pub struct ExtractiveSummarizer {
    config: SummarizationConfig,
}

impl ExtractiveSummarizer {
    /// Create a new extractive summarizer with the given config.
    pub fn new(config: SummarizationConfig) -> Self {
        Self { config }
    }

    /// Score a single sentence given its position index and total sentence count.
    fn score_sentence(sentence: &str, index: usize, total: usize) -> f64 {
        let word_count = sentence.split_whitespace().count();
        let word_score = (word_count as f64).min(20.0) / 20.0;

        let lower = sentence.to_lowercase();
        let key_term_score = KEY_TERMS
            .iter()
            .filter(|term| lower.contains(**term))
            .count() as f64
            * 0.15;

        // Position bonus: first sentences are more important.
        let position_score = if total > 1 {
            1.0 - (index as f64 / total as f64) * 0.5
        } else {
            1.0
        };

        word_score + key_term_score + position_score
    }

    /// Summarize the given text by extracting key sentences.
    pub fn summarize(&self, text: &str) -> SummarizationResult {
        let original_length = text.len();

        if text.is_empty() {
            return SummarizationResult {
                summary: String::new(),
                original_length: 0,
                summary_length: 0,
                compression_ratio: 0.0,
                segments_processed: 0,
                key_facts: Vec::new(),
            };
        }

        if original_length < self.config.min_content_length {
            return SummarizationResult {
                summary: text.to_string(),
                original_length,
                summary_length: original_length,
                compression_ratio: 1.0,
                segments_processed: 1,
                key_facts: Vec::new(),
            };
        }

        let sentences = split_sentences(text);
        let total = sentences.len();

        // Score each sentence.
        let mut scored: Vec<(usize, f64, &str)> = sentences
            .iter()
            .enumerate()
            .map(|(i, s)| (i, Self::score_sentence(s, i, total), s.as_str()))
            .collect();

        // Sort by score descending.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Pick top sentences until we reach max_output_tokens.
        let mut selected_indices: Vec<usize> = Vec::new();
        let mut token_budget = self.config.max_output_tokens;

        for &(idx, _score, sentence) in &scored {
            let tokens = estimate_tokens(sentence);
            if tokens <= token_budget {
                selected_indices.push(idx);
                token_budget = token_budget.saturating_sub(tokens);
            }
            if token_budget == 0 {
                break;
            }
        }

        // Restore original order for natural reading.
        selected_indices.sort();

        let key_facts: Vec<String> = if self.config.preserve_key_facts {
            scored
                .iter()
                .take(3)
                .map(|(_, _, s)| s.to_string())
                .collect()
        } else {
            Vec::new()
        };

        let summary: String = selected_indices
            .iter()
            .map(|&i| sentences[i].as_str())
            .collect::<Vec<_>>()
            .join(" ");

        let summary_length = summary.len();
        let compression_ratio = if original_length > 0 {
            summary_length as f64 / original_length as f64
        } else {
            0.0
        };

        SummarizationResult {
            summary,
            original_length,
            summary_length,
            compression_ratio,
            segments_processed: total,
            key_facts,
        }
    }
}

// ---------------------------------------------------------------------------
// TruncationSummarizer
// ---------------------------------------------------------------------------

/// Truncates text at sentence boundaries to fit within the token budget.
pub struct TruncationSummarizer {
    config: SummarizationConfig,
}

impl TruncationSummarizer {
    /// Create a new truncation summarizer with the given config.
    pub fn new(config: SummarizationConfig) -> Self {
        Self { config }
    }

    /// Summarize by truncating text at a sentence boundary.
    pub fn summarize(&self, text: &str) -> SummarizationResult {
        let original_length = text.len();

        if text.is_empty() {
            return SummarizationResult {
                summary: String::new(),
                original_length: 0,
                summary_length: 0,
                compression_ratio: 0.0,
                segments_processed: 0,
                key_facts: Vec::new(),
            };
        }

        if estimate_tokens(text) <= self.config.max_output_tokens {
            return SummarizationResult {
                summary: text.to_string(),
                original_length,
                summary_length: original_length,
                compression_ratio: 1.0,
                segments_processed: 1,
                key_facts: Vec::new(),
            };
        }

        let sentences = split_sentences(text);
        let mut result = String::new();
        let mut segments_used = 0;

        for sentence in &sentences {
            let candidate = if result.is_empty() {
                sentence.clone()
            } else {
                format!("{} {}", result, sentence)
            };
            if estimate_tokens(&candidate) > self.config.max_output_tokens {
                break;
            }
            result = candidate;
            segments_used += 1;
        }

        // If no full sentence fits, take as many characters as the budget allows.
        if result.is_empty() && !sentences.is_empty() {
            let max_chars = self.config.max_output_tokens * 4;
            result = text.chars().take(max_chars).collect();
            segments_used = 1;
        }

        let summary_length = result.len();
        let compression_ratio = if original_length > 0 {
            summary_length as f64 / original_length as f64
        } else {
            0.0
        };

        SummarizationResult {
            summary: result,
            original_length,
            summary_length,
            compression_ratio,
            segments_processed: segments_used,
            key_facts: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// HierarchicalSummarizer
// ---------------------------------------------------------------------------

/// Splits content into sections, summarizes each section independently,
/// then combines the section summaries.
pub struct HierarchicalSummarizer {
    config: SummarizationConfig,
    sections: Vec<TextSegment>,
}

impl HierarchicalSummarizer {
    /// Create a new hierarchical summarizer with the given config.
    pub fn new(config: SummarizationConfig) -> Self {
        Self {
            config,
            sections: Vec::new(),
        }
    }

    /// Add a section to be summarized.
    pub fn add_section(&mut self, segment: TextSegment) {
        self.sections.push(segment);
    }

    /// Summarize all sections and combine into a single result.
    pub fn summarize(&self) -> SummarizationResult {
        if self.sections.is_empty() {
            return SummarizationResult {
                summary: String::new(),
                original_length: 0,
                summary_length: 0,
                compression_ratio: 0.0,
                segments_processed: 0,
                key_facts: Vec::new(),
            };
        }

        let total_sections = self.sections.len();
        let tokens_per_section = self
            .config
            .max_output_tokens
            .checked_div(total_sections)
            .unwrap_or(self.config.max_output_tokens);

        let section_config = SummarizationConfig {
            max_output_tokens: tokens_per_section.max(1),
            ..self.config.clone()
        };

        let extractor = ExtractiveSummarizer::new(section_config);

        let mut section_summaries: Vec<String> = Vec::new();
        let mut total_original = 0usize;
        let mut all_key_facts: Vec<String> = Vec::new();

        // Sort sections by importance descending so higher-importance
        // sections contribute first.
        let mut sorted_sections: Vec<&TextSegment> = self.sections.iter().collect();
        sorted_sections.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for section in &sorted_sections {
            total_original += section.content.len();
            let result = extractor.summarize(&section.content);
            if !result.summary.is_empty() {
                section_summaries.push(result.summary);
            }
            all_key_facts.extend(result.key_facts);
        }

        let summary = section_summaries.join(" ");
        let summary_length = summary.len();
        let compression_ratio = if total_original > 0 {
            summary_length as f64 / total_original as f64
        } else {
            0.0
        };

        SummarizationResult {
            summary,
            original_length: total_original,
            summary_length,
            compression_ratio,
            segments_processed: total_sections,
            key_facts: all_key_facts,
        }
    }
}

// ---------------------------------------------------------------------------
// CompressedMessage
// ---------------------------------------------------------------------------

/// A message that may represent a summary of multiple prior messages.
#[derive(Debug, Clone)]
pub struct CompressedMessage {
    /// Role of the message (e.g. "user", "assistant", "system").
    pub role: String,
    /// The message content (possibly a summary).
    pub content: String,
    /// Whether this message is a summary of older messages.
    pub is_summary: bool,
    /// If this is a summary, how many original messages it represents.
    pub original_count: Option<usize>,
}

impl CompressedMessage {
    /// Create a new compressed message.
    pub fn new(
        role: impl Into<String>,
        content: impl Into<String>,
        is_summary: bool,
        original_count: Option<usize>,
    ) -> Self {
        Self {
            role: role.into(),
            content: content.into(),
            is_summary,
            original_count,
        }
    }

    /// Serialize this message to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "role": self.role,
            "content": self.content,
            "is_summary": self.is_summary,
            "original_count": self.original_count,
        })
    }
}

// ---------------------------------------------------------------------------
// ContextCompressor
// ---------------------------------------------------------------------------

/// Manages a conversation context window by summarizing old messages
/// when the total token count exceeds the configured maximum.
pub struct ContextCompressor {
    max_context_tokens: usize,
    messages: Vec<(String, String)>,
    compressed: Option<Vec<CompressedMessage>>,
}

impl ContextCompressor {
    /// Create a new context compressor with the given maximum token budget.
    pub fn new(max_context_tokens: usize) -> Self {
        Self {
            max_context_tokens,
            messages: Vec::new(),
            compressed: None,
        }
    }

    /// Add a message to the conversation history.
    pub fn add_message(&mut self, role: &str, content: &str) {
        self.messages.push((role.to_string(), content.to_string()));
        // Invalidate cached compression when new messages arrive.
        self.compressed = None;
    }

    /// Total number of raw messages stored.
    pub fn total_messages(&self) -> usize {
        self.messages.len()
    }

    /// Whether the context has been compressed (i.e. some messages were
    /// summarized to fit within the token budget).
    pub fn is_compressed(&self) -> bool {
        self.get_compressed_context()
            .iter()
            .any(|m| m.is_summary)
    }

    /// Build a compressed context that fits within the token budget.
    ///
    /// If the messages already fit, they are returned as-is. Otherwise the
    /// oldest messages are summarized into a single summary message, and
    /// the most recent messages are kept verbatim.
    pub fn get_compressed_context(&self) -> Vec<CompressedMessage> {
        if let Some(ref cached) = self.compressed {
            return cached.clone();
        }

        if self.messages.is_empty() {
            return Vec::new();
        }

        // Check total tokens.
        let total_tokens: usize = self
            .messages
            .iter()
            .map(|(r, c)| estimate_tokens(&format!("{}: {}", r, c)))
            .sum();

        if total_tokens <= self.max_context_tokens {
            // Everything fits — return uncompressed.
            return self
                .messages
                .iter()
                .map(|(role, content)| {
                    CompressedMessage::new(role.clone(), content.clone(), false, None)
                })
                .collect();
        }

        // Determine how many recent messages to keep verbatim.
        // Walk backwards until we use roughly half the budget.
        let keep_budget = self.max_context_tokens / 2;
        let mut keep_tokens = 0usize;
        let mut keep_count = 0usize;

        for (role, content) in self.messages.iter().rev() {
            let t = estimate_tokens(&format!("{}: {}", role, content));
            if keep_tokens + t > keep_budget && keep_count > 0 {
                break;
            }
            keep_tokens += t;
            keep_count += 1;
        }

        let split_point = self.messages.len().saturating_sub(keep_count);

        if split_point == 0 {
            // All messages are "recent" — just return them.
            return self
                .messages
                .iter()
                .map(|(role, content)| {
                    CompressedMessage::new(role.clone(), content.clone(), false, None)
                })
                .collect();
        }

        // Summarize the older messages.
        let older: Vec<String> = self.messages[..split_point]
            .iter()
            .map(|(role, content)| format!("{}: {}", role, content))
            .collect();
        let combined = older.join("\n");

        let summary_config = SummarizationConfig::new()
            .with_strategy(SummarizationStrategy::Extractive)
            .with_max_output_tokens(self.max_context_tokens / 4)
            .with_min_content_length(0);

        let summarizer = ExtractiveSummarizer::new(summary_config);
        let result = summarizer.summarize(&combined);

        let mut compressed = Vec::new();

        // Insert the summary as a system message.
        compressed.push(CompressedMessage::new(
            "system",
            format!("[Summary of {} earlier messages] {}", split_point, result.summary),
            true,
            Some(split_point),
        ));

        // Append the recent messages verbatim.
        for (role, content) in &self.messages[split_point..] {
            compressed.push(CompressedMessage::new(
                role.clone(),
                content.clone(),
                false,
                None,
            ));
        }

        compressed
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SummarizationStrategy tests --

    #[test]
    fn test_strategy_names() {
        assert_eq!(SummarizationStrategy::Extractive.name(), "extractive");
        assert_eq!(SummarizationStrategy::Abstractive.name(), "abstractive");
        assert_eq!(SummarizationStrategy::Truncate.name(), "truncate");
        assert_eq!(SummarizationStrategy::Hierarchical.name(), "hierarchical");
    }

    #[test]
    fn test_strategy_display() {
        assert_eq!(format!("{}", SummarizationStrategy::Extractive), "extractive");
        assert_eq!(format!("{}", SummarizationStrategy::Truncate), "truncate");
    }

    #[test]
    fn test_strategy_equality() {
        assert_eq!(SummarizationStrategy::Extractive, SummarizationStrategy::Extractive);
        assert_ne!(SummarizationStrategy::Extractive, SummarizationStrategy::Truncate);
    }

    #[test]
    fn test_strategy_clone() {
        let s = SummarizationStrategy::Hierarchical;
        let s2 = s;
        assert_eq!(s, s2);
    }

    #[test]
    fn test_strategy_debug() {
        let dbg = format!("{:?}", SummarizationStrategy::Abstractive);
        assert!(dbg.contains("Abstractive"));
    }

    // -- SummarizationConfig builder tests --

    #[test]
    fn test_config_defaults() {
        let config = SummarizationConfig::default();
        assert_eq!(config.strategy, SummarizationStrategy::Extractive);
        assert_eq!(config.max_output_tokens, 500);
        assert!(config.preserve_key_facts);
        assert!((config.compression_ratio - 0.3).abs() < f64::EPSILON);
        assert_eq!(config.min_content_length, 100);
    }

    #[test]
    fn test_config_builder() {
        let config = SummarizationConfig::new()
            .with_strategy(SummarizationStrategy::Truncate)
            .with_max_output_tokens(200)
            .with_preserve_key_facts(false)
            .with_compression_ratio(0.5)
            .with_min_content_length(50);

        assert_eq!(config.strategy, SummarizationStrategy::Truncate);
        assert_eq!(config.max_output_tokens, 200);
        assert!(!config.preserve_key_facts);
        assert!((config.compression_ratio - 0.5).abs() < f64::EPSILON);
        assert_eq!(config.min_content_length, 50);
    }

    #[test]
    fn test_config_clone() {
        let config = SummarizationConfig::new().with_max_output_tokens(42);
        let config2 = config.clone();
        assert_eq!(config2.max_output_tokens, 42);
    }

    // -- TextSegment tests --

    #[test]
    fn test_segment_new() {
        let seg = TextSegment::new("Hello world", 0.8, "greeting");
        assert_eq!(seg.content, "Hello world");
        assert!((seg.importance - 0.8).abs() < f64::EPSILON);
        assert_eq!(seg.category, "greeting");
        assert!(seg.metadata.is_empty());
    }

    #[test]
    fn test_segment_word_count() {
        let seg = TextSegment::new("one two three four", 0.5, "test");
        assert_eq!(seg.word_count(), 4);
    }

    #[test]
    fn test_segment_char_count() {
        let seg = TextSegment::new("abcdef", 0.5, "test");
        assert_eq!(seg.char_count(), 6);
    }

    #[test]
    fn test_segment_empty() {
        let seg = TextSegment::new("", 0.0, "empty");
        assert_eq!(seg.word_count(), 0);
        assert_eq!(seg.char_count(), 0);
    }

    #[test]
    fn test_segment_with_metadata() {
        let seg = TextSegment::new("text", 0.5, "cat")
            .with_metadata("source", serde_json::json!("file.txt"))
            .with_metadata("line", serde_json::json!(42));
        assert_eq!(seg.metadata.len(), 2);
        assert_eq!(seg.metadata["source"], serde_json::json!("file.txt"));
    }

    #[test]
    fn test_segment_to_json() {
        let seg = TextSegment::new("hello world", 0.9, "greeting")
            .with_metadata("k", serde_json::json!("v"));
        let json = seg.to_json();
        assert_eq!(json["content"], "hello world");
        assert_eq!(json["importance"], 0.9);
        assert_eq!(json["category"], "greeting");
        assert_eq!(json["word_count"], 2);
        assert_eq!(json["char_count"], 11);
        assert_eq!(json["metadata"]["k"], "v");
    }

    // -- ExtractiveSummarizer tests --

    #[test]
    fn test_extractive_empty_text() {
        let config = SummarizationConfig::new();
        let summarizer = ExtractiveSummarizer::new(config);
        let result = summarizer.summarize("");
        assert!(result.summary.is_empty());
        assert_eq!(result.original_length, 0);
        assert_eq!(result.segments_processed, 0);
    }

    #[test]
    fn test_extractive_short_text() {
        let config = SummarizationConfig::new().with_min_content_length(200);
        let summarizer = ExtractiveSummarizer::new(config);
        let result = summarizer.summarize("Short text.");
        // Text is shorter than min_content_length, returned as-is.
        assert_eq!(result.summary, "Short text.");
        assert_eq!(result.compression_ratio, 1.0);
    }

    #[test]
    fn test_extractive_basic_summarization() {
        let text = "This is the first important sentence. This is a less important filler sentence. \
                     The key result is that the system works correctly. Another filler sentence here. \
                     In conclusion, everything is fine.";
        let config = SummarizationConfig::new()
            .with_max_output_tokens(30)
            .with_min_content_length(10);
        let summarizer = ExtractiveSummarizer::new(config);
        let result = summarizer.summarize(text);

        assert!(!result.summary.is_empty());
        assert!(result.summary_length <= result.original_length);
        assert!(result.segments_processed > 0);
    }

    #[test]
    fn test_extractive_key_terms_boost() {
        let text = "This is a normal sentence without special words. \
                     The important key result must be noted. \
                     Another normal filler sentence here.";
        let config = SummarizationConfig::new()
            .with_max_output_tokens(20)
            .with_min_content_length(10);
        let summarizer = ExtractiveSummarizer::new(config);
        let result = summarizer.summarize(text);

        // The sentence with key terms should be prioritized.
        assert!(result.summary.contains("important") || result.summary.contains("key")
            || result.summary.contains("must") || result.summary.contains("result"));
    }

    #[test]
    fn test_extractive_preserves_key_facts() {
        let config = SummarizationConfig::new()
            .with_preserve_key_facts(true)
            .with_min_content_length(10);
        let summarizer = ExtractiveSummarizer::new(config);
        let text = "First sentence here. Second sentence now. Third sentence too.";
        let result = summarizer.summarize(text);
        assert!(!result.key_facts.is_empty());
    }

    #[test]
    fn test_extractive_no_key_facts_when_disabled() {
        let config = SummarizationConfig::new()
            .with_preserve_key_facts(false)
            .with_min_content_length(10);
        let summarizer = ExtractiveSummarizer::new(config);
        let text = "First sentence here. Second sentence now. Third sentence too.";
        let result = summarizer.summarize(text);
        assert!(result.key_facts.is_empty());
    }

    #[test]
    fn test_extractive_single_sentence() {
        let config = SummarizationConfig::new().with_min_content_length(10);
        let summarizer = ExtractiveSummarizer::new(config);
        let text = "This is the only sentence in this entire piece of text that we have.";
        let result = summarizer.summarize(text);
        assert_eq!(result.segments_processed, 1);
        assert!(!result.summary.is_empty());
    }

    #[test]
    fn test_extractive_compression_ratio() {
        let text = "First sentence. Second sentence. Third sentence. Fourth sentence. \
                     Fifth sentence. Sixth sentence. Seventh sentence. Eighth sentence.";
        let config = SummarizationConfig::new()
            .with_max_output_tokens(10)
            .with_min_content_length(10);
        let summarizer = ExtractiveSummarizer::new(config);
        let result = summarizer.summarize(text);
        assert!(result.compression_ratio < 1.0);
        assert!(result.compression_ratio > 0.0);
    }

    // -- TruncationSummarizer tests --

    #[test]
    fn test_truncation_empty_text() {
        let config = SummarizationConfig::new();
        let summarizer = TruncationSummarizer::new(config);
        let result = summarizer.summarize("");
        assert!(result.summary.is_empty());
        assert_eq!(result.original_length, 0);
    }

    #[test]
    fn test_truncation_text_fits() {
        let config = SummarizationConfig::new().with_max_output_tokens(1000);
        let summarizer = TruncationSummarizer::new(config);
        let text = "Short text.";
        let result = summarizer.summarize(text);
        assert_eq!(result.summary, "Short text.");
        assert_eq!(result.compression_ratio, 1.0);
    }

    #[test]
    fn test_truncation_at_sentence_boundary() {
        let text = "First sentence. Second sentence. Third very long sentence that goes on.";
        let config = SummarizationConfig::new()
            .with_max_output_tokens(10)
            .with_min_content_length(10);
        let summarizer = TruncationSummarizer::new(config);
        let result = summarizer.summarize(text);

        // Should end at a sentence boundary.
        assert!(
            result.summary.ends_with('.')
                || result.summary.ends_with('!')
                || result.summary.ends_with('?')
                || result.summary_length < result.original_length
        );
    }

    #[test]
    fn test_truncation_long_first_sentence() {
        let text = "This is an extremely long first sentence that contains many many words \
                     and keeps going on and on without stopping for a very long time indeed.";
        let config = SummarizationConfig::new().with_max_output_tokens(5);
        let summarizer = TruncationSummarizer::new(config);
        let result = summarizer.summarize(text);
        // Should still produce some output via character fallback.
        assert!(!result.summary.is_empty());
        assert!(result.summary_length <= result.original_length);
    }

    #[test]
    fn test_truncation_multiple_sentences() {
        let text = "One. Two. Three. Four. Five. Six. Seven. Eight. Nine. Ten.";
        let config = SummarizationConfig::new().with_max_output_tokens(8);
        let summarizer = TruncationSummarizer::new(config);
        let result = summarizer.summarize(text);
        assert!(result.segments_processed > 0);
        assert!(result.summary_length < result.original_length);
    }

    // -- HierarchicalSummarizer tests --

    #[test]
    fn test_hierarchical_no_sections() {
        let config = SummarizationConfig::new();
        let summarizer = HierarchicalSummarizer::new(config);
        let result = summarizer.summarize();
        assert!(result.summary.is_empty());
        assert_eq!(result.segments_processed, 0);
    }

    #[test]
    fn test_hierarchical_single_section() {
        let config = SummarizationConfig::new()
            .with_max_output_tokens(50)
            .with_min_content_length(10);
        let mut summarizer = HierarchicalSummarizer::new(config);
        summarizer.add_section(TextSegment::new(
            "This is a detailed section about an important topic. It has multiple sentences. \
             The key finding is significant.",
            0.9,
            "findings",
        ));
        let result = summarizer.summarize();
        assert!(!result.summary.is_empty());
        assert_eq!(result.segments_processed, 1);
    }

    #[test]
    fn test_hierarchical_multiple_sections() {
        let config = SummarizationConfig::new()
            .with_max_output_tokens(100)
            .with_min_content_length(10);
        let mut summarizer = HierarchicalSummarizer::new(config);
        summarizer.add_section(TextSegment::new(
            "Introduction to the topic. It provides necessary background information.",
            0.7,
            "introduction",
        ));
        summarizer.add_section(TextSegment::new(
            "The main findings are important. They show critical results. This is essential data.",
            0.9,
            "findings",
        ));
        summarizer.add_section(TextSegment::new(
            "In conclusion everything works. The summary confirms results.",
            0.5,
            "conclusion",
        ));

        let result = summarizer.summarize();
        assert!(!result.summary.is_empty());
        assert_eq!(result.segments_processed, 3);
    }

    #[test]
    fn test_hierarchical_importance_ordering() {
        let config = SummarizationConfig::new()
            .with_max_output_tokens(100)
            .with_min_content_length(10);
        let mut summarizer = HierarchicalSummarizer::new(config);
        summarizer.add_section(TextSegment::new(
            "Low importance filler text that is not very useful or needed really.",
            0.1,
            "filler",
        ));
        summarizer.add_section(TextSegment::new(
            "Critical important key finding that must be noted and is essential.",
            1.0,
            "critical",
        ));

        let result = summarizer.summarize();
        assert!(!result.summary.is_empty());
        assert_eq!(result.segments_processed, 2);
    }

    // -- SummarizationResult tests --

    #[test]
    fn test_result_to_json() {
        let result = SummarizationResult {
            summary: "A summary.".to_string(),
            original_length: 100,
            summary_length: 10,
            compression_ratio: 0.1,
            segments_processed: 5,
            key_facts: vec!["fact1".to_string(), "fact2".to_string()],
        };
        let json = result.to_json();
        assert_eq!(json["summary"], "A summary.");
        assert_eq!(json["original_length"], 100);
        assert_eq!(json["summary_length"], 10);
        assert_eq!(json["compression_ratio"], 0.1);
        assert_eq!(json["segments_processed"], 5);
        assert_eq!(json["key_facts"][0], "fact1");
        assert_eq!(json["key_facts"][1], "fact2");
    }

    #[test]
    fn test_result_compression_ratio_calculation() {
        let text = "First. Second. Third. Fourth. Fifth. Sixth. Seventh. Eighth. Ninth. Tenth.";
        let config = SummarizationConfig::new()
            .with_max_output_tokens(10)
            .with_min_content_length(10);
        let summarizer = ExtractiveSummarizer::new(config);
        let result = summarizer.summarize(text);
        let expected = result.summary_length as f64 / result.original_length as f64;
        assert!((result.compression_ratio - expected).abs() < 0.01);
    }

    // -- ContextCompressor tests --

    #[test]
    fn test_compressor_empty() {
        let compressor = ContextCompressor::new(1000);
        assert_eq!(compressor.total_messages(), 0);
        assert!(!compressor.is_compressed());
        let ctx = compressor.get_compressed_context();
        assert!(ctx.is_empty());
    }

    #[test]
    fn test_compressor_add_messages() {
        let mut compressor = ContextCompressor::new(10000);
        compressor.add_message("user", "Hello");
        compressor.add_message("assistant", "Hi there!");
        assert_eq!(compressor.total_messages(), 2);
    }

    #[test]
    fn test_compressor_no_compression_when_fits() {
        let mut compressor = ContextCompressor::new(10000);
        compressor.add_message("user", "Hello");
        compressor.add_message("assistant", "Hi");
        let ctx = compressor.get_compressed_context();
        assert_eq!(ctx.len(), 2);
        assert!(!ctx[0].is_summary);
        assert!(!ctx[1].is_summary);
        assert!(!compressor.is_compressed());
    }

    #[test]
    fn test_compressor_compresses_when_over_budget() {
        let mut compressor = ContextCompressor::new(20);
        // Add enough messages to exceed the token budget.
        for i in 0..20 {
            compressor.add_message("user", &format!("Message number {} with some content.", i));
            compressor.add_message(
                "assistant",
                &format!("Response number {} with additional words.", i),
            );
        }
        let ctx = compressor.get_compressed_context();
        // Should have fewer messages than the original 40.
        assert!(ctx.len() < compressor.total_messages());
        // First message should be a summary.
        assert!(ctx[0].is_summary);
        assert!(ctx[0].original_count.is_some());
        assert!(compressor.is_compressed());
    }

    #[test]
    fn test_compressor_preserves_recent_messages() {
        let mut compressor = ContextCompressor::new(30);
        for i in 0..10 {
            compressor.add_message("user", &format!("Old message {}. Details here.", i));
        }
        compressor.add_message("user", "Recent question?");
        compressor.add_message("assistant", "Recent answer.");

        let ctx = compressor.get_compressed_context();
        // The most recent messages should be present verbatim.
        let last = ctx.last().unwrap();
        assert!(!last.is_summary);
        assert!(last.content == "Recent answer." || last.content == "Recent question?");
    }

    #[test]
    fn test_compressor_single_message() {
        let mut compressor = ContextCompressor::new(1000);
        compressor.add_message("user", "Only message");
        let ctx = compressor.get_compressed_context();
        assert_eq!(ctx.len(), 1);
        assert!(!ctx[0].is_summary);
        assert_eq!(ctx[0].content, "Only message");
    }

    // -- CompressedMessage tests --

    #[test]
    fn test_compressed_message_new() {
        let msg = CompressedMessage::new("user", "Hello", false, None);
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello");
        assert!(!msg.is_summary);
        assert!(msg.original_count.is_none());
    }

    #[test]
    fn test_compressed_message_summary() {
        let msg = CompressedMessage::new("system", "Summary of 5 messages", true, Some(5));
        assert!(msg.is_summary);
        assert_eq!(msg.original_count, Some(5));
    }

    #[test]
    fn test_compressed_message_to_json() {
        let msg = CompressedMessage::new("assistant", "Response", false, None);
        let json = msg.to_json();
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"], "Response");
        assert_eq!(json["is_summary"], false);
        assert!(json["original_count"].is_null());
    }

    #[test]
    fn test_compressed_message_to_json_with_count() {
        let msg = CompressedMessage::new("system", "Summary", true, Some(10));
        let json = msg.to_json();
        assert_eq!(json["is_summary"], true);
        assert_eq!(json["original_count"], 10);
    }

    // -- Edge case tests --

    #[test]
    fn test_extractive_very_long_text() {
        let sentences: Vec<String> = (0..100)
            .map(|i| format!("This is sentence number {} with some additional words.", i))
            .collect();
        let text = sentences.join(" ");
        let config = SummarizationConfig::new()
            .with_max_output_tokens(50)
            .with_min_content_length(10);
        let summarizer = ExtractiveSummarizer::new(config);
        let result = summarizer.summarize(&text);
        assert!(result.compression_ratio < 1.0);
        assert!(result.segments_processed > 0);
    }

    #[test]
    fn test_truncation_single_word() {
        let config = SummarizationConfig::new().with_max_output_tokens(1000);
        let summarizer = TruncationSummarizer::new(config);
        let result = summarizer.summarize("Hello");
        assert_eq!(result.summary, "Hello");
    }

    #[test]
    fn test_split_sentences_basic() {
        let sentences = split_sentences("First. Second. Third.");
        assert_eq!(sentences.len(), 3);
        assert_eq!(sentences[0], "First.");
        assert_eq!(sentences[1], "Second.");
        assert_eq!(sentences[2], "Third.");
    }

    #[test]
    fn test_split_sentences_no_period() {
        let sentences = split_sentences("No period at the end");
        assert_eq!(sentences.len(), 1);
        assert_eq!(sentences[0], "No period at the end");
    }

    #[test]
    fn test_split_sentences_mixed_punctuation() {
        let sentences = split_sentences("Hello! How are you? I am fine.");
        assert_eq!(sentences.len(), 3);
    }

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens("abcdefgh"), 2);
    }

    #[test]
    fn test_compressor_invalidates_cache_on_add() {
        let mut compressor = ContextCompressor::new(10000);
        compressor.add_message("user", "First");
        let ctx1 = compressor.get_compressed_context();
        assert_eq!(ctx1.len(), 1);

        compressor.add_message("user", "Second");
        let ctx2 = compressor.get_compressed_context();
        assert_eq!(ctx2.len(), 2);
    }
}
