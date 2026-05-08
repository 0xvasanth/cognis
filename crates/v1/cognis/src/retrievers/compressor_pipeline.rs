//! Document compressor pipeline that chains multiple compressors in sequence.
//!
//! Provides built-in compressors for common filtering and transformation tasks:
//! - [`LengthFilter`] — filter by document content length
//! - [`RelevanceScoreFilter`] — filter by relevance score in metadata
//! - [`DuplicateFilter`] — remove near-duplicate documents by content similarity
//! - [`MetadataFilter`] — filter by metadata key presence or key-value match
//! - [`ContentTruncator`] — truncate document content to a maximum length
//! - [`KeywordExtractor`] — keep only documents containing specified keywords
//!
//! Use [`CompressorPipelineBuilder`] for ergonomic construction.

use std::collections::HashSet;

use async_trait::async_trait;
use serde_json::Value;

use cognis_core::documents::Document;
use cognis_core::error::Result;

use super::contextual_compression::DocumentCompressor;

// ---------------------------------------------------------------------------
// CompressorPipeline
// ---------------------------------------------------------------------------

/// Chains multiple [`DocumentCompressor`] implementations in sequence.
///
/// Documents flow through each compressor in order — the output of one becomes
/// the input of the next. If any stage produces an empty list the pipeline
/// short-circuits and returns immediately.
///
/// # Example
///
/// ```rust,ignore
/// use cognis::retrievers::compressor_pipeline::*;
///
/// let pipeline = CompressorPipeline::new()
///     .with(LengthFilter::new().with_min(10).with_max(5000))
///     .with(RelevanceScoreFilter::new(0.5))
///     .with(DuplicateFilter::new(0.95));
///
/// let filtered = pipeline.compress_documents(&docs, "query").await?;
/// ```
pub struct CompressorPipeline {
    /// Ordered list of compressors to apply.
    compressors: Vec<Box<dyn DocumentCompressor>>,
}

impl CompressorPipeline {
    /// Create an empty pipeline.
    pub fn new() -> Self {
        Self {
            compressors: Vec::new(),
        }
    }

    /// Add a compressor to the end of the pipeline (mutating).
    pub fn add(&mut self, compressor: Box<dyn DocumentCompressor>) {
        self.compressors.push(compressor);
    }

    /// Add a compressor to the end of the pipeline (builder-style).
    pub fn with(mut self, compressor: impl DocumentCompressor + 'static) -> Self {
        self.compressors.push(Box::new(compressor));
        self
    }
}

impl Default for CompressorPipeline {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentCompressor for CompressorPipeline {
    fn name(&self) -> &str {
        "CompressorPipeline"
    }

    async fn compress_documents(
        &self,
        documents: &[Document],
        query: &str,
    ) -> Result<Vec<Document>> {
        let mut docs = documents.to_vec();
        for compressor in &self.compressors {
            if docs.is_empty() {
                return Ok(docs);
            }
            docs = compressor.compress_documents(&docs, query).await?;
        }
        Ok(docs)
    }
}

// ---------------------------------------------------------------------------
// LengthFilter
// ---------------------------------------------------------------------------

/// Filters documents by content length (number of characters).
pub struct LengthFilter {
    /// Minimum content length (inclusive). `None` means no lower bound.
    min_length: Option<usize>,
    /// Maximum content length (inclusive). `None` means no upper bound.
    max_length: Option<usize>,
}

impl LengthFilter {
    /// Create a new `LengthFilter` with no bounds.
    pub fn new() -> Self {
        Self {
            min_length: None,
            max_length: None,
        }
    }

    /// Set the minimum content length.
    pub fn with_min(mut self, min: usize) -> Self {
        self.min_length = Some(min);
        self
    }

    /// Set the maximum content length.
    pub fn with_max(mut self, max: usize) -> Self {
        self.max_length = Some(max);
        self
    }
}

impl Default for LengthFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentCompressor for LengthFilter {
    fn name(&self) -> &str {
        "LengthFilter"
    }

    async fn compress_documents(
        &self,
        documents: &[Document],
        _query: &str,
    ) -> Result<Vec<Document>> {
        Ok(documents
            .iter()
            .filter(|doc| {
                let len = doc.page_content.len();
                if let Some(min) = self.min_length {
                    if len < min {
                        return false;
                    }
                }
                if let Some(max) = self.max_length {
                    if len > max {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect())
    }
}

// ---------------------------------------------------------------------------
// RelevanceScoreFilter
// ---------------------------------------------------------------------------

/// Filters documents by a relevance score stored in metadata.
///
/// Expects a `"relevance_score"` key in each document's metadata whose value
/// is a JSON number. Documents without the key or with a score below the
/// threshold are removed.
pub struct RelevanceScoreFilter {
    /// Minimum relevance score to keep a document.
    threshold: f32,
}

impl RelevanceScoreFilter {
    /// Create a new filter with the given threshold.
    pub fn new(threshold: f32) -> Self {
        Self { threshold }
    }
}

#[async_trait]
impl DocumentCompressor for RelevanceScoreFilter {
    fn name(&self) -> &str {
        "RelevanceScoreFilter"
    }

    async fn compress_documents(
        &self,
        documents: &[Document],
        _query: &str,
    ) -> Result<Vec<Document>> {
        Ok(documents
            .iter()
            .filter(|doc| {
                doc.metadata
                    .get("relevance_score")
                    .and_then(|v| v.as_f64())
                    .map(|score| score as f32 >= self.threshold)
                    .unwrap_or(false)
            })
            .cloned()
            .collect())
    }
}

// ---------------------------------------------------------------------------
// DuplicateFilter
// ---------------------------------------------------------------------------

/// Removes near-duplicate documents based on character-level similarity.
///
/// Uses a simple Jaccard similarity on character bigrams. The first occurrence
/// of each group of near-duplicates is kept.
pub struct DuplicateFilter {
    /// Similarity threshold above which documents are considered duplicates.
    similarity_threshold: f32,
}

impl DuplicateFilter {
    /// Create a new filter with the given similarity threshold.
    pub fn new(similarity_threshold: f32) -> Self {
        Self {
            similarity_threshold,
        }
    }
}

/// Compute bigram-based Jaccard similarity between two strings.
fn bigram_similarity(a: &str, b: &str) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.len() < 2 || b.len() < 2 {
        return if a == b { 1.0 } else { 0.0 };
    }

    let bigrams_a: HashSet<(char, char)> = a.chars().zip(a.chars().skip(1)).collect();
    let bigrams_b: HashSet<(char, char)> = b.chars().zip(b.chars().skip(1)).collect();

    let intersection = bigrams_a.intersection(&bigrams_b).count();
    let union = bigrams_a.union(&bigrams_b).count();

    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

#[async_trait]
impl DocumentCompressor for DuplicateFilter {
    fn name(&self) -> &str {
        "DuplicateFilter"
    }

    async fn compress_documents(
        &self,
        documents: &[Document],
        _query: &str,
    ) -> Result<Vec<Document>> {
        let mut kept: Vec<Document> = Vec::new();

        for doc in documents {
            let is_duplicate = kept.iter().any(|existing| {
                bigram_similarity(&existing.page_content, &doc.page_content)
                    >= self.similarity_threshold
            });
            if !is_duplicate {
                kept.push(doc.clone());
            }
        }

        Ok(kept)
    }
}

// ---------------------------------------------------------------------------
// MetadataFilter
// ---------------------------------------------------------------------------

/// Filters documents based on metadata key presence and key-value matching.
pub struct MetadataFilter {
    /// Documents must have all of these metadata keys.
    required_keys: Vec<String>,
    /// Documents must NOT have any of these metadata keys.
    forbidden_keys: Vec<String>,
    /// Documents must match all of these key-value pairs.
    key_value_filters: Vec<(String, Value)>,
}

impl MetadataFilter {
    /// Create a new metadata filter with no constraints.
    pub fn new() -> Self {
        Self {
            required_keys: Vec::new(),
            forbidden_keys: Vec::new(),
            key_value_filters: Vec::new(),
        }
    }

    /// Set required metadata keys.
    pub fn with_required_keys(mut self, keys: Vec<String>) -> Self {
        self.required_keys = keys;
        self
    }

    /// Set forbidden metadata keys.
    pub fn with_forbidden_keys(mut self, keys: Vec<String>) -> Self {
        self.forbidden_keys = keys;
        self
    }

    /// Set key-value filters. Documents must have these exact key-value pairs.
    pub fn with_key_value_filters(mut self, filters: Vec<(String, Value)>) -> Self {
        self.key_value_filters = filters;
        self
    }
}

impl Default for MetadataFilter {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentCompressor for MetadataFilter {
    fn name(&self) -> &str {
        "MetadataFilter"
    }

    async fn compress_documents(
        &self,
        documents: &[Document],
        _query: &str,
    ) -> Result<Vec<Document>> {
        Ok(documents
            .iter()
            .filter(|doc| {
                // Check required keys
                for key in &self.required_keys {
                    if !doc.metadata.contains_key(key) {
                        return false;
                    }
                }
                // Check forbidden keys
                for key in &self.forbidden_keys {
                    if doc.metadata.contains_key(key) {
                        return false;
                    }
                }
                // Check key-value filters
                for (key, value) in &self.key_value_filters {
                    match doc.metadata.get(key) {
                        Some(v) if v == value => {}
                        _ => return false,
                    }
                }
                true
            })
            .cloned()
            .collect())
    }
}

// ---------------------------------------------------------------------------
// ContentTruncator
// ---------------------------------------------------------------------------

/// Truncates document content to a maximum number of characters.
///
/// If the content exceeds `max_chars`, it is truncated and a suffix (default
/// `"..."`) is appended.
pub struct ContentTruncator {
    /// Maximum number of characters to keep.
    max_chars: usize,
    /// Suffix appended to truncated content.
    truncation_suffix: String,
}

impl ContentTruncator {
    /// Create a new truncator with the given max character limit.
    pub fn new(max_chars: usize) -> Self {
        Self {
            max_chars,
            truncation_suffix: "...".to_string(),
        }
    }

    /// Set a custom truncation suffix.
    pub fn with_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.truncation_suffix = suffix.into();
        self
    }
}

#[async_trait]
impl DocumentCompressor for ContentTruncator {
    fn name(&self) -> &str {
        "ContentTruncator"
    }

    async fn compress_documents(
        &self,
        documents: &[Document],
        _query: &str,
    ) -> Result<Vec<Document>> {
        Ok(documents
            .iter()
            .map(|doc| {
                if doc.page_content.len() <= self.max_chars {
                    doc.clone()
                } else {
                    let truncated: String = doc.page_content.chars().take(self.max_chars).collect();
                    let mut new_doc = doc.clone();
                    new_doc.page_content = format!("{}{}", truncated, self.truncation_suffix);
                    new_doc
                }
            })
            .collect())
    }
}

// ---------------------------------------------------------------------------
// KeywordExtractor
// ---------------------------------------------------------------------------

/// Whether to require any or all keywords to match.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeywordMode {
    /// Document must contain at least one keyword.
    Any,
    /// Document must contain all keywords.
    All,
}

/// Keeps only documents that contain specified keywords.
pub struct KeywordExtractor {
    /// Keywords to look for (case-insensitive).
    keywords: Vec<String>,
    /// Whether to require any or all keywords.
    mode: KeywordMode,
}

impl KeywordExtractor {
    /// Create a new keyword filter.
    pub fn new(keywords: Vec<String>, mode: KeywordMode) -> Self {
        Self {
            keywords: keywords.into_iter().map(|k| k.to_lowercase()).collect(),
            mode,
        }
    }
}

#[async_trait]
impl DocumentCompressor for KeywordExtractor {
    fn name(&self) -> &str {
        "KeywordExtractor"
    }

    async fn compress_documents(
        &self,
        documents: &[Document],
        _query: &str,
    ) -> Result<Vec<Document>> {
        if self.keywords.is_empty() {
            return Ok(documents.to_vec());
        }

        Ok(documents
            .iter()
            .filter(|doc| {
                let content_lower = doc.page_content.to_lowercase();
                match self.mode {
                    KeywordMode::Any => self
                        .keywords
                        .iter()
                        .any(|kw| content_lower.contains(kw.as_str())),
                    KeywordMode::All => self
                        .keywords
                        .iter()
                        .all(|kw| content_lower.contains(kw.as_str())),
                }
            })
            .cloned()
            .collect())
    }
}

// ---------------------------------------------------------------------------
// CompressorPipelineBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`CompressorPipeline`] with convenience methods
/// for adding built-in compressors.
pub struct CompressorPipelineBuilder {
    compressors: Vec<Box<dyn DocumentCompressor>>,
}

impl CompressorPipelineBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        Self {
            compressors: Vec::new(),
        }
    }

    /// Add a length filter.
    pub fn add_length_filter(mut self, min: Option<usize>, max: Option<usize>) -> Self {
        let mut filter = LengthFilter::new();
        if let Some(min) = min {
            filter = filter.with_min(min);
        }
        if let Some(max) = max {
            filter = filter.with_max(max);
        }
        self.compressors.push(Box::new(filter));
        self
    }

    /// Add a relevance score filter.
    pub fn add_relevance_filter(mut self, threshold: f32) -> Self {
        self.compressors
            .push(Box::new(RelevanceScoreFilter::new(threshold)));
        self
    }

    /// Add a duplicate filter.
    pub fn add_duplicate_filter(mut self, threshold: f32) -> Self {
        self.compressors
            .push(Box::new(DuplicateFilter::new(threshold)));
        self
    }

    /// Add a metadata filter.
    pub fn add_metadata_filter(mut self, required: Vec<String>, forbidden: Vec<String>) -> Self {
        self.compressors.push(Box::new(
            MetadataFilter::new()
                .with_required_keys(required)
                .with_forbidden_keys(forbidden),
        ));
        self
    }

    /// Add a content truncator.
    pub fn add_truncator(mut self, max_chars: usize) -> Self {
        self.compressors
            .push(Box::new(ContentTruncator::new(max_chars)));
        self
    }

    /// Add a keyword filter.
    pub fn add_keyword_filter(mut self, keywords: Vec<String>, mode: KeywordMode) -> Self {
        self.compressors
            .push(Box::new(KeywordExtractor::new(keywords, mode)));
        self
    }

    /// Build the pipeline.
    pub fn build(self) -> CompressorPipeline {
        CompressorPipeline {
            compressors: self.compressors,
        }
    }
}

impl Default for CompressorPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn doc(content: &str) -> Document {
        Document::new(content)
    }

    fn doc_with_meta(content: &str, meta: Vec<(&str, Value)>) -> Document {
        let mut metadata = HashMap::new();
        for (k, v) in meta {
            metadata.insert(k.to_string(), v);
        }
        Document::new(content).with_metadata(metadata)
    }

    // -- LengthFilter --

    #[tokio::test]
    async fn test_length_filter_min() {
        let filter = LengthFilter::new().with_min(10);
        let docs = vec![doc("short"), doc("this is long enough")];
        let result = filter.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "this is long enough");
    }

    #[tokio::test]
    async fn test_length_filter_max() {
        let filter = LengthFilter::new().with_max(10);
        let docs = vec![doc("short"), doc("this is way too long for the filter")];
        let result = filter.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "short");
    }

    #[tokio::test]
    async fn test_length_filter_min_and_max() {
        let filter = LengthFilter::new().with_min(5).with_max(15);
        let docs = vec![
            doc("hi"),                          // too short (2)
            doc("hello world"),                 // ok (11)
            doc("this is way too long string"), // too long (27)
        ];
        let result = filter.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "hello world");
    }

    // -- RelevanceScoreFilter --

    #[tokio::test]
    async fn test_relevance_score_filter_threshold() {
        let docs = vec![
            doc_with_meta("high", vec![("relevance_score", Value::from(0.9))]),
            doc_with_meta("low", vec![("relevance_score", Value::from(0.3))]),
            doc_with_meta("mid", vec![("relevance_score", Value::from(0.6))]),
            doc("no score"),
        ];
        let filter = RelevanceScoreFilter::new(0.5);
        let result = filter.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].page_content, "high");
        assert_eq!(result[1].page_content, "mid");
    }

    // -- DuplicateFilter --

    #[tokio::test]
    async fn test_duplicate_filter_removes_near_duplicates() {
        let docs = vec![
            doc("the quick brown fox jumps over the lazy dog"),
            doc("the quick brown fox jumps over the lazy cat"), // near duplicate
            doc("machine learning is a branch of artificial intelligence"),
        ];
        let filter = DuplicateFilter::new(0.7);
        let result = filter.compress_documents(&docs, "q").await.unwrap();
        // The first and second are very similar, so second should be removed.
        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0].page_content,
            "the quick brown fox jumps over the lazy dog"
        );
        assert_eq!(
            result[1].page_content,
            "machine learning is a branch of artificial intelligence"
        );
    }

    #[tokio::test]
    async fn test_duplicate_filter_keeps_unique() {
        let docs = vec![doc("alpha beta gamma"), doc("delta epsilon zeta")];
        let filter = DuplicateFilter::new(0.9);
        let result = filter.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 2);
    }

    // -- MetadataFilter --

    #[tokio::test]
    async fn test_metadata_filter_required_keys() {
        let docs = vec![
            doc_with_meta("has source", vec![("source", Value::from("a.txt"))]),
            doc("no metadata"),
        ];
        let filter = MetadataFilter::new().with_required_keys(vec!["source".to_string()]);
        let result = filter.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "has source");
    }

    #[tokio::test]
    async fn test_metadata_filter_forbidden_keys() {
        let docs = vec![
            doc_with_meta("has draft", vec![("draft", Value::from(true))]),
            doc_with_meta("published", vec![("source", Value::from("b.txt"))]),
        ];
        let filter = MetadataFilter::new().with_forbidden_keys(vec!["draft".to_string()]);
        let result = filter.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "published");
    }

    #[tokio::test]
    async fn test_metadata_filter_key_value_match() {
        let docs = vec![
            doc_with_meta("pdf", vec![("format", Value::from("pdf"))]),
            doc_with_meta("html", vec![("format", Value::from("html"))]),
        ];
        let filter = MetadataFilter::new()
            .with_key_value_filters(vec![("format".to_string(), Value::from("pdf"))]);
        let result = filter.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "pdf");
    }

    // -- ContentTruncator --

    #[tokio::test]
    async fn test_content_truncator() {
        let truncator = ContentTruncator::new(10);
        let docs = vec![
            doc("short"),
            doc("this is a really long document that should be truncated"),
        ];
        let result = truncator.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].page_content, "short");
        assert_eq!(result[1].page_content, "this is a ...");
    }

    #[tokio::test]
    async fn test_content_truncator_custom_suffix() {
        let truncator = ContentTruncator::new(5).with_suffix(" [truncated]");
        let docs = vec![doc("hello world")];
        let result = truncator.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result[0].page_content, "hello [truncated]");
    }

    // -- KeywordExtractor --

    #[tokio::test]
    async fn test_keyword_extractor_any_mode() {
        let filter = KeywordExtractor::new(
            vec!["rust".to_string(), "python".to_string()],
            KeywordMode::Any,
        );
        let docs = vec![
            doc("Rust is a systems language"),
            doc("Python is great for scripting"),
            doc("Java is also popular"),
        ];
        let result = filter.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].page_content, "Rust is a systems language");
        assert_eq!(result[1].page_content, "Python is great for scripting");
    }

    #[tokio::test]
    async fn test_keyword_extractor_all_mode() {
        let filter = KeywordExtractor::new(
            vec!["rust".to_string(), "systems".to_string()],
            KeywordMode::All,
        );
        let docs = vec![
            doc("Rust is a systems language"),
            doc("Rust is great"),
            doc("Systems programming is fun"),
        ];
        let result = filter.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "Rust is a systems language");
    }

    // -- Pipeline --

    #[tokio::test]
    async fn test_pipeline_chains_compressors_in_order() {
        // First filter by length (min 5), then truncate to 10 chars.
        let pipeline = CompressorPipeline::new()
            .with(LengthFilter::new().with_min(5))
            .with(ContentTruncator::new(10));

        let docs = vec![
            doc("hi"),                         // filtered out by length
            doc("hello world, this is great"), // kept, then truncated
        ];
        let result = pipeline.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "hello worl...");
    }

    #[tokio::test]
    async fn test_pipeline_short_circuits_on_empty() {
        // First compressor filters everything, second should not run.
        let pipeline = CompressorPipeline::new()
            .with(LengthFilter::new().with_min(1000))
            .with(ContentTruncator::new(5));

        let docs = vec![doc("short")];
        let result = pipeline.compress_documents(&docs, "q").await.unwrap();
        assert!(result.is_empty());
    }

    // -- Builder --

    #[tokio::test]
    async fn test_builder_pattern() {
        let pipeline = CompressorPipelineBuilder::new()
            .add_length_filter(Some(5), Some(100))
            .add_keyword_filter(vec!["rust".to_string()], KeywordMode::Any)
            .add_truncator(15)
            .build();

        let docs = vec![
            doc("hi"),                       // too short
            doc("Rust is a great language"), // kept, has keyword, truncated
            doc("Python is also nice"),      // no keyword
        ];
        let result = pipeline.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "Rust is a great...");
    }

    // -- Edge cases --

    #[tokio::test]
    async fn test_empty_document_list() {
        let pipeline = CompressorPipeline::new().with(LengthFilter::new().with_min(10));
        let result = pipeline.compress_documents(&[], "q").await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_single_compressor_pipeline() {
        let pipeline = CompressorPipeline::new().with(LengthFilter::new().with_max(5));
        let docs = vec![doc("hi"), doc("hello world")];
        let result = pipeline.compress_documents(&docs, "q").await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "hi");
    }

    #[tokio::test]
    async fn test_all_docs_filtered_out() {
        let filter = RelevanceScoreFilter::new(0.9);
        let docs = vec![
            doc_with_meta("low", vec![("relevance_score", Value::from(0.1))]),
            doc_with_meta("also low", vec![("relevance_score", Value::from(0.2))]),
        ];
        let result = filter.compress_documents(&docs, "q").await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn test_compressor_name_method() {
        assert_eq!(LengthFilter::new().name(), "LengthFilter");
        assert_eq!(
            RelevanceScoreFilter::new(0.5).name(),
            "RelevanceScoreFilter"
        );
        assert_eq!(DuplicateFilter::new(0.9).name(), "DuplicateFilter");
        assert_eq!(MetadataFilter::new().name(), "MetadataFilter");
        assert_eq!(ContentTruncator::new(10).name(), "ContentTruncator");
        assert_eq!(
            KeywordExtractor::new(vec![], KeywordMode::Any).name(),
            "KeywordExtractor"
        );
        assert_eq!(CompressorPipeline::new().name(), "CompressorPipeline");
    }
}
