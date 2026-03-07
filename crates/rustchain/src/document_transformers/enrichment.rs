//! Enrichment transformers that add computed metadata to documents.
//!
//! These transformers analyze document content and attach useful metadata
//! such as word counts, character counts, language hints, keywords, and
//! summaries.

use std::collections::HashMap;

use async_trait::async_trait;
use serde_json::Value;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;

use super::DocumentTransformer;

// ─── WordCountEnricher ───

/// Adds a `word_count` metadata field to each document.
///
/// Words are delimited by whitespace.
pub struct WordCountEnricher;

impl WordCountEnricher {
    /// Create a new word count enricher.
    pub fn new() -> Self {
        Self
    }
}

impl Default for WordCountEnricher {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentTransformer for WordCountEnricher {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        Ok(documents
            .iter()
            .map(|doc| {
                let mut new_doc = doc.clone();
                let count = doc.page_content.split_whitespace().count();
                new_doc
                    .metadata
                    .insert("word_count".to_string(), Value::from(count as u64));
                new_doc
            })
            .collect())
    }

    fn name(&self) -> &str {
        "WordCountEnricher"
    }
}

// ─── CharCountEnricher ───

/// Adds a `char_count` metadata field to each document.
pub struct CharCountEnricher;

impl CharCountEnricher {
    /// Create a new character count enricher.
    pub fn new() -> Self {
        Self
    }
}

impl Default for CharCountEnricher {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentTransformer for CharCountEnricher {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        Ok(documents
            .iter()
            .map(|doc| {
                let mut new_doc = doc.clone();
                let count = doc.page_content.chars().count();
                new_doc
                    .metadata
                    .insert("char_count".to_string(), Value::from(count as u64));
                new_doc
            })
            .collect())
    }

    fn name(&self) -> &str {
        "CharCountEnricher"
    }
}

// ─── LanguageDetector ───

/// Adds a `language` metadata field using heuristic Unicode script analysis.
///
/// Detects the predominant script family: `latin`, `cjk`, `cyrillic`,
/// `arabic`, or `unknown`.
pub struct LanguageDetector;

impl LanguageDetector {
    /// Create a new language detector.
    pub fn new() -> Self {
        Self
    }
}

impl Default for LanguageDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple language detection based on Unicode script analysis.
fn detect_language(text: &str) -> &'static str {
    let mut latin = 0u32;
    let mut cjk = 0u32;
    let mut cyrillic = 0u32;
    let mut arabic = 0u32;

    for ch in text.chars() {
        if ch.is_ascii_alphabetic() || matches!(ch, '\u{00C0}'..='\u{024F}') {
            latin += 1;
        } else if matches!(ch, '\u{4E00}'..='\u{9FFF}' | '\u{3040}'..='\u{30FF}') {
            cjk += 1;
        } else if matches!(ch, '\u{0400}'..='\u{04FF}') {
            cyrillic += 1;
        } else if matches!(ch, '\u{0600}'..='\u{06FF}') {
            arabic += 1;
        }
    }

    let max = latin.max(cjk).max(cyrillic).max(arabic);
    if max == 0 {
        return "unknown";
    }
    if max == cjk {
        "cjk"
    } else if max == cyrillic {
        "cyrillic"
    } else if max == arabic {
        "arabic"
    } else {
        "latin"
    }
}

#[async_trait]
impl DocumentTransformer for LanguageDetector {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        Ok(documents
            .iter()
            .map(|doc| {
                let mut new_doc = doc.clone();
                let lang = detect_language(&doc.page_content);
                new_doc
                    .metadata
                    .insert("language".to_string(), Value::from(lang));
                new_doc
            })
            .collect())
    }

    fn name(&self) -> &str {
        "LanguageDetector"
    }
}

// ─── KeywordExtractor ───

/// Extracts top-N keywords from document content using term frequency scoring.
///
/// Words are lowercased, and common English stop words are filtered out.
/// The resulting keywords are stored as a JSON array in the `keywords`
/// metadata field.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::document_transformers::enrichment::KeywordExtractor;
///
/// let extractor = KeywordExtractor::new(5);
/// let docs = extractor.transform_documents(&docs).await?;
/// ```
pub struct KeywordExtractor {
    top_n: usize,
}

impl KeywordExtractor {
    /// Create a keyword extractor that keeps the top `n` keywords.
    pub fn new(top_n: usize) -> Self {
        Self { top_n }
    }
}

/// A small set of common English stop words.
const STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "my", "no", "not", "of", "on", "or", "our", "she", "so",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "to", "up", "us", "was", "we", "what", "when", "which", "who", "will",
    "with", "you", "your",
];

/// Extract top-N keywords from text using term frequency.
fn extract_keywords(text: &str, top_n: usize) -> Vec<String> {
    let stop_words: std::collections::HashSet<&str> = STOP_WORDS.iter().copied().collect();
    let mut freq: HashMap<String, usize> = HashMap::new();

    for word in text.split_whitespace() {
        let cleaned: String = word
            .chars()
            .filter(|c| c.is_alphanumeric())
            .collect::<String>()
            .to_lowercase();
        if cleaned.len() < 2 || stop_words.contains(cleaned.as_str()) {
            continue;
        }
        *freq.entry(cleaned).or_insert(0) += 1;
    }

    let mut entries: Vec<(String, usize)> = freq.into_iter().collect();
    entries.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    entries.into_iter().take(top_n).map(|(w, _)| w).collect()
}

#[async_trait]
impl DocumentTransformer for KeywordExtractor {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        Ok(documents
            .iter()
            .map(|doc| {
                let mut new_doc = doc.clone();
                let keywords = extract_keywords(&doc.page_content, self.top_n);
                let kw_value: Vec<Value> =
                    keywords.into_iter().map(Value::from).collect();
                new_doc
                    .metadata
                    .insert("keywords".to_string(), Value::from(kw_value));
                new_doc
            })
            .collect())
    }

    fn name(&self) -> &str {
        "KeywordExtractor"
    }
}

// ─── DocumentSummarizer ───

/// Adds a truncated summary of the document content to metadata.
///
/// The summary is simply the first `max_length` characters of the content,
/// with an ellipsis appended if truncated.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::document_transformers::enrichment::DocumentSummarizer;
///
/// let summarizer = DocumentSummarizer::new(100);
/// let docs = summarizer.transform_documents(&docs).await?;
/// ```
pub struct DocumentSummarizer {
    max_length: usize,
}

impl DocumentSummarizer {
    /// Create a summarizer that truncates content to the given maximum length.
    pub fn new(max_length: usize) -> Self {
        Self { max_length }
    }
}

#[async_trait]
impl DocumentTransformer for DocumentSummarizer {
    async fn transform_documents(&self, documents: &[Document]) -> Result<Vec<Document>> {
        Ok(documents
            .iter()
            .map(|doc| {
                let mut new_doc = doc.clone();
                let content = &doc.page_content;
                let summary = if content.chars().count() > self.max_length {
                    let truncated: String = content.chars().take(self.max_length).collect();
                    format!("{}...", truncated)
                } else {
                    content.clone()
                };
                new_doc
                    .metadata
                    .insert("summary".to_string(), Value::from(summary));
                new_doc
            })
            .collect())
    }

    fn name(&self) -> &str {
        "DocumentSummarizer"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(content: &str) -> Document {
        Document::new(content)
    }

    // ─── WordCountEnricher tests ───

    #[tokio::test]
    async fn test_word_count_enricher() {
        let enricher = WordCountEnricher::new();
        let docs = vec![make_doc("the quick brown fox")];
        let result = enricher.transform_documents(&docs).await.unwrap();
        assert_eq!(
            result[0]
                .metadata
                .get("word_count")
                .and_then(|v| v.as_u64()),
            Some(4)
        );
    }

    #[tokio::test]
    async fn test_word_count_empty_content() {
        let enricher = WordCountEnricher::new();
        let docs = vec![make_doc("")];
        let result = enricher.transform_documents(&docs).await.unwrap();
        assert_eq!(
            result[0]
                .metadata
                .get("word_count")
                .and_then(|v| v.as_u64()),
            Some(0)
        );
    }

    // ─── CharCountEnricher tests ───

    #[tokio::test]
    async fn test_char_count_enricher() {
        let enricher = CharCountEnricher::new();
        let docs = vec![make_doc("hello")];
        let result = enricher.transform_documents(&docs).await.unwrap();
        assert_eq!(
            result[0]
                .metadata
                .get("char_count")
                .and_then(|v| v.as_u64()),
            Some(5)
        );
    }

    #[tokio::test]
    async fn test_char_count_unicode() {
        let enricher = CharCountEnricher::new();
        let docs = vec![make_doc("\u{4F60}\u{597D}")]; // 2 CJK characters
        let result = enricher.transform_documents(&docs).await.unwrap();
        assert_eq!(
            result[0]
                .metadata
                .get("char_count")
                .and_then(|v| v.as_u64()),
            Some(2)
        );
    }

    // ─── LanguageDetector tests ───

    #[tokio::test]
    async fn test_language_detector_latin() {
        let detector = LanguageDetector::new();
        let docs = vec![make_doc("Hello world, this is English text")];
        let result = detector.transform_documents(&docs).await.unwrap();
        assert_eq!(
            result[0].metadata.get("language").and_then(|v| v.as_str()),
            Some("latin")
        );
    }

    #[tokio::test]
    async fn test_language_detector_cjk() {
        let detector = LanguageDetector::new();
        let docs = vec![make_doc("\u{4F60}\u{597D}\u{4E16}\u{754C}")];
        let result = detector.transform_documents(&docs).await.unwrap();
        assert_eq!(
            result[0].metadata.get("language").and_then(|v| v.as_str()),
            Some("cjk")
        );
    }

    #[tokio::test]
    async fn test_language_detector_unknown() {
        let detector = LanguageDetector::new();
        let docs = vec![make_doc("123 456 789")];
        let result = detector.transform_documents(&docs).await.unwrap();
        assert_eq!(
            result[0].metadata.get("language").and_then(|v| v.as_str()),
            Some("unknown")
        );
    }

    // ─── KeywordExtractor tests ───

    #[tokio::test]
    async fn test_keyword_extractor() {
        let extractor = KeywordExtractor::new(3);
        let docs = vec![make_doc(
            "rust rust rust programming programming language",
        )];
        let result = extractor.transform_documents(&docs).await.unwrap();
        let keywords = result[0]
            .metadata
            .get("keywords")
            .and_then(|v| v.as_array())
            .unwrap();
        // "rust" has freq 3, "programming" has freq 2, "language" has freq 1
        assert_eq!(keywords.len(), 3);
        assert_eq!(keywords[0].as_str().unwrap(), "rust");
        assert_eq!(keywords[1].as_str().unwrap(), "programming");
        assert_eq!(keywords[2].as_str().unwrap(), "language");
    }

    #[tokio::test]
    async fn test_keyword_extractor_filters_stop_words() {
        let extractor = KeywordExtractor::new(5);
        let docs = vec![make_doc("the the the and and or but is are was")];
        let result = extractor.transform_documents(&docs).await.unwrap();
        let keywords = result[0]
            .metadata
            .get("keywords")
            .and_then(|v| v.as_array())
            .unwrap();
        assert!(keywords.is_empty());
    }

    #[tokio::test]
    async fn test_keyword_extractor_top_n_limit() {
        let extractor = KeywordExtractor::new(2);
        let docs = vec![make_doc("alpha beta gamma delta epsilon")];
        let result = extractor.transform_documents(&docs).await.unwrap();
        let keywords = result[0]
            .metadata
            .get("keywords")
            .and_then(|v| v.as_array())
            .unwrap();
        assert_eq!(keywords.len(), 2);
    }

    // ─── DocumentSummarizer tests ───

    #[tokio::test]
    async fn test_summarizer_truncates() {
        let summarizer = DocumentSummarizer::new(10);
        let docs = vec![make_doc("This is a long document that should be truncated")];
        let result = summarizer.transform_documents(&docs).await.unwrap();
        let summary = result[0]
            .metadata
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap();
        assert!(summary.ends_with("..."));
        // 10 chars + "..."
        assert_eq!(summary.len(), 13);
    }

    #[tokio::test]
    async fn test_summarizer_short_content_no_truncation() {
        let summarizer = DocumentSummarizer::new(100);
        let docs = vec![make_doc("short")];
        let result = summarizer.transform_documents(&docs).await.unwrap();
        let summary = result[0]
            .metadata
            .get("summary")
            .and_then(|v| v.as_str())
            .unwrap();
        assert_eq!(summary, "short");
        assert!(!summary.ends_with("..."));
    }

    #[tokio::test]
    async fn test_summarizer_preserves_existing_metadata() {
        let summarizer = DocumentSummarizer::new(5);
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), Value::from("test"));
        let docs = vec![Document::new("hello world").with_metadata(meta)];
        let result = summarizer.transform_documents(&docs).await.unwrap();
        assert_eq!(
            result[0].metadata.get("source").and_then(|v| v.as_str()),
            Some("test")
        );
        assert!(result[0].metadata.contains_key("summary"));
    }
}
