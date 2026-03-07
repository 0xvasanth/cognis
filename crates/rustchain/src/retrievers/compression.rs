//! Document compression retrievers that reduce document content to only the
//! most relevant parts before returning results.
//!
//! Provides a [`DocumentCompressor`] trait and several built-in implementations:
//!
//! - [`LengthCompressor`] — truncates documents to a maximum character length.
//! - [`SentenceExtractor`] — extracts sentences containing query keywords.
//! - [`RedundancyFilter`] — removes near-duplicate documents via Jaccard similarity.
//! - [`RelevanceScorer`] — scores documents by keyword overlap and filters below a threshold.
//! - [`MetadataFilter`] — filters documents based on metadata field conditions.
//! - [`CompressorPipeline`] — chains multiple compressors in sequence.
//! - [`ContextualCompressionRetriever`] — wraps a base document set with a compressor.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;

// ---------------------------------------------------------------------------
// DocumentCompressor trait
// ---------------------------------------------------------------------------

/// Trait for compressing or filtering documents based on a query.
///
/// Implementations may shorten document content, remove irrelevant documents,
/// or both.
pub trait DocumentCompressor: Send + Sync {
    /// Compress or filter the given documents with respect to `query`.
    fn compress(&self, documents: &[Document], query: &str) -> Result<Vec<Document>>;
}

// ---------------------------------------------------------------------------
// LengthCompressor
// ---------------------------------------------------------------------------

/// Truncates each document's content to a maximum character length,
/// preserving the beginning of the text.
pub struct LengthCompressor {
    max_length: usize,
}

impl LengthCompressor {
    /// Create a new `LengthCompressor` that truncates content to `max_length` characters.
    pub fn new(max_length: usize) -> Self {
        Self { max_length }
    }
}

impl DocumentCompressor for LengthCompressor {
    fn compress(&self, documents: &[Document], _query: &str) -> Result<Vec<Document>> {
        let mut result = Vec::with_capacity(documents.len());
        for doc in documents {
            let mut compressed = doc.clone();
            if compressed.page_content.len() > self.max_length {
                // Truncate at char boundary.
                let truncated: String = compressed
                    .page_content
                    .chars()
                    .take(self.max_length)
                    .collect();
                compressed.page_content = truncated;
            }
            result.push(compressed);
        }
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// SentenceExtractor
// ---------------------------------------------------------------------------

/// Extracts sentences from documents that contain query keywords.
///
/// If fewer than `min_sentences` match, the first `min_sentences` sentences
/// are returned to ensure some content is always present.
pub struct SentenceExtractor {
    min_sentences: usize,
}

impl SentenceExtractor {
    /// Create a new `SentenceExtractor` with a default minimum of 1 sentence.
    pub fn new() -> Self {
        Self { min_sentences: 1 }
    }

    /// Set the minimum number of sentences to keep even when no keywords match.
    pub fn with_min_sentences(mut self, n: usize) -> Self {
        self.min_sentences = n;
        self
    }

    /// Split text into sentences (simple heuristic: split on `. `, `! `, `? ` or line-ending punctuation).
    fn split_sentences(text: &str) -> Vec<String> {
        let mut sentences = Vec::new();
        let mut current = String::new();

        for ch in text.chars() {
            current.push(ch);
            if ch == '.' || ch == '!' || ch == '?' {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    sentences.push(trimmed);
                }
                current.clear();
            }
        }

        let trimmed = current.trim().to_string();
        if !trimmed.is_empty() {
            sentences.push(trimmed);
        }

        sentences
    }

    /// Extract keywords from query (lowercased, split on whitespace).
    fn extract_keywords(query: &str) -> HashSet<String> {
        query
            .split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|w| !w.is_empty())
            .collect()
    }
}

impl Default for SentenceExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentCompressor for SentenceExtractor {
    fn compress(&self, documents: &[Document], query: &str) -> Result<Vec<Document>> {
        let keywords = Self::extract_keywords(query);
        let mut result = Vec::with_capacity(documents.len());

        for doc in documents {
            let sentences = Self::split_sentences(&doc.page_content);

            if sentences.is_empty() {
                result.push(doc.clone());
                continue;
            }

            // Find sentences containing any keyword.
            let matching: Vec<&String> = sentences
                .iter()
                .filter(|s| {
                    let lower = s.to_lowercase();
                    keywords.iter().any(|kw| lower.contains(kw.as_str()))
                })
                .collect();

            let selected = if matching.len() >= self.min_sentences {
                matching.into_iter().cloned().collect::<Vec<_>>()
            } else {
                // Fall back to first min_sentences sentences.
                sentences
                    .iter()
                    .take(self.min_sentences)
                    .cloned()
                    .collect::<Vec<_>>()
            };

            let mut compressed = doc.clone();
            compressed.page_content = selected.join(" ");
            result.push(compressed);
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// RedundancyFilter
// ---------------------------------------------------------------------------

/// Removes near-duplicate documents based on Jaccard similarity of word sets.
///
/// For each document, if its Jaccard similarity to any previously accepted
/// document exceeds the threshold, it is discarded.
pub struct RedundancyFilter {
    similarity_threshold: f64,
}

impl RedundancyFilter {
    /// Create a new `RedundancyFilter` with the given similarity threshold.
    ///
    /// Default threshold is 0.8 if you want a reasonable default — use
    /// `RedundancyFilter::new(0.8)`.
    pub fn new(similarity_threshold: f64) -> Self {
        Self {
            similarity_threshold,
        }
    }

    /// Compute the Jaccard similarity between two sets of words.
    fn jaccard_similarity(a: &HashSet<String>, b: &HashSet<String>) -> f64 {
        if a.is_empty() && b.is_empty() {
            return 1.0;
        }
        let intersection = a.intersection(b).count() as f64;
        let union = a.union(b).count() as f64;
        if union == 0.0 {
            return 0.0;
        }
        intersection / union
    }

    /// Convert text to a set of lowercased words.
    fn word_set(text: &str) -> HashSet<String> {
        text.split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|w| !w.is_empty())
            .collect()
    }
}

impl DocumentCompressor for RedundancyFilter {
    fn compress(&self, documents: &[Document], _query: &str) -> Result<Vec<Document>> {
        let mut accepted: Vec<(Document, HashSet<String>)> = Vec::new();

        for doc in documents {
            let word_set = Self::word_set(&doc.page_content);
            let is_duplicate = accepted.iter().any(|(_, existing_set)| {
                Self::jaccard_similarity(&word_set, existing_set) >= self.similarity_threshold
            });

            if !is_duplicate {
                accepted.push((doc.clone(), word_set));
            }
        }

        Ok(accepted.into_iter().map(|(doc, _)| doc).collect())
    }
}

// ---------------------------------------------------------------------------
// RelevanceScorer
// ---------------------------------------------------------------------------

/// Scores documents by keyword overlap with the query and removes those
/// scoring below a minimum threshold.
///
/// The score is computed as: `matching_keywords / total_query_keywords`.
pub struct RelevanceScorer {
    min_score: f64,
}

impl RelevanceScorer {
    /// Create a new `RelevanceScorer` that filters documents scoring below `min_score`.
    pub fn new(min_score: f64) -> Self {
        Self { min_score }
    }

    /// Compute a relevance score based on what fraction of query keywords
    /// appear in the document text.
    fn score(doc_text: &str, query_keywords: &[String]) -> f64 {
        if query_keywords.is_empty() {
            return 1.0;
        }
        let lower = doc_text.to_lowercase();
        let matches = query_keywords
            .iter()
            .filter(|kw| lower.contains(kw.as_str()))
            .count();
        matches as f64 / query_keywords.len() as f64
    }
}

impl DocumentCompressor for RelevanceScorer {
    fn compress(&self, documents: &[Document], query: &str) -> Result<Vec<Document>> {
        let keywords: Vec<String> = query
            .split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|w| !w.is_empty())
            .collect();

        let result = documents
            .iter()
            .filter(|doc| Self::score(&doc.page_content, &keywords) >= self.min_score)
            .cloned()
            .collect();

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// MetadataFilter
// ---------------------------------------------------------------------------

/// Condition for filtering documents by metadata.
#[derive(Debug, Clone)]
enum MetadataCondition {
    /// The document must have this field present.
    RequireField(String),
    /// The document must have this field set to this value.
    RequireValue(String, Value),
    /// The document must not have this field set to this value.
    ExcludeValue(String, Value),
}

/// Filters documents based on metadata field conditions.
///
/// Supports requiring field presence, requiring specific values, and excluding
/// specific values. All conditions must be satisfied for a document to pass.
pub struct MetadataFilter {
    conditions: Vec<MetadataCondition>,
}

impl MetadataFilter {
    /// Create a new empty `MetadataFilter` (passes all documents).
    pub fn new() -> Self {
        Self {
            conditions: Vec::new(),
        }
    }

    /// Require that documents have the given metadata field.
    pub fn require_field(mut self, field: &str) -> Self {
        self.conditions
            .push(MetadataCondition::RequireField(field.to_string()));
        self
    }

    /// Require that documents have the given metadata field set to the given value.
    pub fn require_value(mut self, field: &str, value: Value) -> Self {
        self.conditions
            .push(MetadataCondition::RequireValue(field.to_string(), value));
        self
    }

    /// Exclude documents that have the given metadata field set to the given value.
    pub fn exclude_value(mut self, field: &str, value: Value) -> Self {
        self.conditions
            .push(MetadataCondition::ExcludeValue(field.to_string(), value));
        self
    }

    /// Check whether a document satisfies all conditions.
    fn satisfies(&self, metadata: &HashMap<String, Value>) -> bool {
        self.conditions.iter().all(|cond| match cond {
            MetadataCondition::RequireField(field) => metadata.contains_key(field),
            MetadataCondition::RequireValue(field, value) => metadata.get(field) == Some(value),
            MetadataCondition::ExcludeValue(field, value) => metadata.get(field) != Some(value),
        })
    }
}

impl Default for MetadataFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentCompressor for MetadataFilter {
    fn compress(&self, documents: &[Document], _query: &str) -> Result<Vec<Document>> {
        Ok(documents
            .iter()
            .filter(|doc| self.satisfies(&doc.metadata))
            .cloned()
            .collect())
    }
}

// ---------------------------------------------------------------------------
// CompressorPipeline
// ---------------------------------------------------------------------------

/// Chains multiple [`DocumentCompressor`] implementations in sequence.
///
/// Documents flow through each compressor in order — the output of one
/// becomes the input of the next.
pub struct CompressorPipeline {
    compressors: Vec<Box<dyn DocumentCompressor>>,
}

impl CompressorPipeline {
    /// Create a new empty pipeline.
    pub fn new() -> Self {
        Self {
            compressors: Vec::new(),
        }
    }

    /// Add a compressor to the end of the pipeline.
    #[allow(clippy::should_implement_trait)]
    pub fn add(mut self, compressor: Box<dyn DocumentCompressor>) -> Self {
        self.compressors.push(compressor);
        self
    }

    /// Return the number of compressors in the pipeline.
    pub fn len(&self) -> usize {
        self.compressors.len()
    }

    /// Return whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.compressors.is_empty()
    }
}

impl Default for CompressorPipeline {
    fn default() -> Self {
        Self::new()
    }
}

impl DocumentCompressor for CompressorPipeline {
    fn compress(&self, documents: &[Document], query: &str) -> Result<Vec<Document>> {
        let mut docs = documents.to_vec();
        for compressor in &self.compressors {
            docs = compressor.compress(&docs, query)?;
            if docs.is_empty() {
                return Ok(docs);
            }
        }
        Ok(docs)
    }
}

// ---------------------------------------------------------------------------
// ContextualCompressionRetriever
// ---------------------------------------------------------------------------

/// Wraps a base document set with a compressor, applying compression on retrieval.
pub struct ContextualCompressionRetriever {
    documents: Vec<Document>,
    compressor: Box<dyn DocumentCompressor>,
}

impl ContextualCompressionRetriever {
    /// Create a new retriever wrapping the given documents and compressor.
    pub fn new(documents: Vec<Document>, compressor: Box<dyn DocumentCompressor>) -> Self {
        Self {
            documents,
            compressor,
        }
    }

    /// Retrieve at most `k` compressed documents for the given query.
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let mut docs = self.compressor.compress(&self.documents, query)?;
        docs.truncate(k);
        Ok(docs)
    }

    /// Retrieve all compressed documents for the given query.
    pub fn retrieve_all(&self, query: &str) -> Result<Vec<Document>> {
        self.compressor.compress(&self.documents, query)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- Helpers --

    fn doc(content: &str) -> Document {
        Document::new(content)
    }

    fn doc_with_meta(content: &str, meta: Vec<(&str, Value)>) -> Document {
        let metadata: HashMap<String, Value> =
            meta.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        Document::new(content).with_metadata(metadata)
    }

    // -----------------------------------------------------------------------
    // LengthCompressor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_length_compressor_truncates_long_document() {
        let compressor = LengthCompressor::new(10);
        let docs = vec![doc("This is a long document that should be truncated")];
        let result = compressor.compress(&docs, "query").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content.len(), 10);
        assert_eq!(result[0].page_content, "This is a ");
    }

    #[test]
    fn test_length_compressor_keeps_short_document() {
        let compressor = LengthCompressor::new(100);
        let docs = vec![doc("Short")];
        let result = compressor.compress(&docs, "query").unwrap();
        assert_eq!(result[0].page_content, "Short");
    }

    #[test]
    fn test_length_compressor_exact_length() {
        let compressor = LengthCompressor::new(5);
        let docs = vec![doc("Hello")];
        let result = compressor.compress(&docs, "query").unwrap();
        assert_eq!(result[0].page_content, "Hello");
    }

    #[test]
    fn test_length_compressor_zero_length() {
        let compressor = LengthCompressor::new(0);
        let docs = vec![doc("Hello")];
        let result = compressor.compress(&docs, "query").unwrap();
        assert_eq!(result[0].page_content, "");
    }

    #[test]
    fn test_length_compressor_empty_docs() {
        let compressor = LengthCompressor::new(10);
        let result = compressor.compress(&[], "query").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_length_compressor_preserves_metadata() {
        let compressor = LengthCompressor::new(5);
        let docs = vec![doc_with_meta(
            "Hello World",
            vec![("source", json!("test.pdf"))],
        )];
        let result = compressor.compress(&docs, "query").unwrap();
        assert_eq!(result[0].metadata.get("source"), Some(&json!("test.pdf")));
    }

    // -----------------------------------------------------------------------
    // SentenceExtractor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sentence_extractor_matches_keywords() {
        let extractor = SentenceExtractor::new();
        let docs = vec![doc(
            "The cat sat on the mat. The dog ran in the park. Rust is great.",
        )];
        let result = extractor.compress(&docs, "dog park").unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].page_content.contains("dog"));
        assert!(result[0].page_content.contains("park"));
    }

    #[test]
    fn test_sentence_extractor_no_matches_returns_min_sentences() {
        let extractor = SentenceExtractor::new().with_min_sentences(2);
        let docs = vec![doc("First sentence. Second sentence. Third sentence.")];
        let result = extractor.compress(&docs, "nonexistent").unwrap();
        assert_eq!(result.len(), 1);
        // Should contain first 2 sentences.
        assert!(result[0].page_content.contains("First sentence."));
        assert!(result[0].page_content.contains("Second sentence."));
    }

    #[test]
    fn test_sentence_extractor_case_insensitive() {
        let extractor = SentenceExtractor::new();
        let docs = vec![doc("Rust is awesome. Python is nice.")];
        let result = extractor.compress(&docs, "RUST").unwrap();
        assert!(result[0].page_content.contains("Rust is awesome."));
    }

    #[test]
    fn test_sentence_extractor_empty_query() {
        let extractor = SentenceExtractor::new().with_min_sentences(1);
        let docs = vec![doc("First. Second. Third.")];
        let result = extractor.compress(&docs, "").unwrap();
        // No keywords to match, falls back to min_sentences.
        assert_eq!(result.len(), 1);
        assert!(result[0].page_content.contains("First."));
    }

    #[test]
    fn test_sentence_extractor_empty_docs() {
        let extractor = SentenceExtractor::new();
        let result = extractor.compress(&[], "query").unwrap();
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // RedundancyFilter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_redundancy_filter_removes_duplicates() {
        let filter = RedundancyFilter::new(0.8);
        let docs = vec![
            doc("the quick brown fox jumps over the lazy dog"),
            doc("the quick brown fox jumps over the lazy dog"), // exact duplicate
            doc("completely different content about something else"),
        ];
        let result = filter.compress(&docs, "query").unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0].page_content.contains("fox"));
        assert!(result[1].page_content.contains("different"));
    }

    #[test]
    fn test_redundancy_filter_near_duplicates() {
        let filter = RedundancyFilter::new(0.7);
        let docs = vec![
            doc("the quick brown fox jumps over the lazy dog"),
            doc("the quick brown fox leaps over the lazy dog"), // very similar
        ];
        let result = filter.compress(&docs, "query").unwrap();
        // High Jaccard similarity — should filter the second one.
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_redundancy_filter_low_threshold_keeps_all() {
        let filter = RedundancyFilter::new(1.0);
        let docs = vec![doc("the quick brown fox"), doc("the quick brown fox jumps")];
        let result = filter.compress(&docs, "query").unwrap();
        // Threshold 1.0 means only exact Jaccard=1.0 is filtered.
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_redundancy_filter_empty_docs() {
        let filter = RedundancyFilter::new(0.8);
        let result = filter.compress(&[], "query").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_redundancy_filter_single_doc() {
        let filter = RedundancyFilter::new(0.8);
        let docs = vec![doc("only document")];
        let result = filter.compress(&docs, "query").unwrap();
        assert_eq!(result.len(), 1);
    }

    // -----------------------------------------------------------------------
    // RelevanceScorer tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_relevance_scorer_filters_irrelevant() {
        let scorer = RelevanceScorer::new(0.5);
        let docs = vec![
            doc("rust programming language is fast and safe"),
            doc("cooking recipes for pasta and pizza"),
            doc("rust compiler and borrow checker"),
        ];
        let result = scorer.compress(&docs, "rust programming").unwrap();
        // First and third docs contain "rust", first also has "programming".
        assert_eq!(result.len(), 2);
        assert!(result[0].page_content.contains("rust"));
        assert!(result[1].page_content.contains("rust"));
    }

    #[test]
    fn test_relevance_scorer_all_relevant() {
        let scorer = RelevanceScorer::new(0.0);
        let docs = vec![doc("anything"), doc("goes")];
        let result = scorer.compress(&docs, "query").unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_relevance_scorer_none_relevant() {
        let scorer = RelevanceScorer::new(1.0);
        let docs = vec![doc("no matching keywords here")];
        let result = scorer.compress(&docs, "rust programming").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_relevance_scorer_empty_query() {
        let scorer = RelevanceScorer::new(0.5);
        let docs = vec![doc("some document")];
        let result = scorer.compress(&docs, "").unwrap();
        // Empty query => score is 1.0, everything passes.
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_relevance_scorer_empty_docs() {
        let scorer = RelevanceScorer::new(0.5);
        let result = scorer.compress(&[], "query").unwrap();
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // MetadataFilter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_metadata_filter_require_field() {
        let filter = MetadataFilter::new().require_field("source");
        let docs = vec![
            doc_with_meta("has source", vec![("source", json!("file.pdf"))]),
            doc("no source"),
        ];
        let result = filter.compress(&docs, "query").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "has source");
    }

    #[test]
    fn test_metadata_filter_require_value() {
        let filter = MetadataFilter::new().require_value("type", json!("article"));
        let docs = vec![
            doc_with_meta("article", vec![("type", json!("article"))]),
            doc_with_meta("blog", vec![("type", json!("blog"))]),
            doc("no type"),
        ];
        let result = filter.compress(&docs, "query").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "article");
    }

    #[test]
    fn test_metadata_filter_exclude_value() {
        let filter = MetadataFilter::new().exclude_value("status", json!("draft"));
        let docs = vec![
            doc_with_meta("published", vec![("status", json!("published"))]),
            doc_with_meta("draft", vec![("status", json!("draft"))]),
            doc("no status"),
        ];
        let result = filter.compress(&docs, "query").unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].page_content, "published");
        assert_eq!(result[1].page_content, "no status");
    }

    #[test]
    fn test_metadata_filter_combined_conditions() {
        let filter = MetadataFilter::new()
            .require_field("source")
            .exclude_value("status", json!("draft"));
        let docs = vec![
            doc_with_meta(
                "good",
                vec![("source", json!("a")), ("status", json!("published"))],
            ),
            doc_with_meta(
                "draft",
                vec![("source", json!("b")), ("status", json!("draft"))],
            ),
            doc_with_meta("no source", vec![("status", json!("published"))]),
            doc("nothing"),
        ];
        let result = filter.compress(&docs, "query").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "good");
    }

    #[test]
    fn test_metadata_filter_no_conditions() {
        let filter = MetadataFilter::new();
        let docs = vec![doc("a"), doc("b")];
        let result = filter.compress(&docs, "query").unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_metadata_filter_empty_docs() {
        let filter = MetadataFilter::new().require_field("source");
        let result = filter.compress(&[], "query").unwrap();
        assert!(result.is_empty());
    }

    // -----------------------------------------------------------------------
    // CompressorPipeline tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_chains_compressors() {
        let pipeline = CompressorPipeline::new()
            .add(Box::new(RelevanceScorer::new(0.5)))
            .add(Box::new(LengthCompressor::new(20)));
        let docs = vec![
            doc("rust programming language documentation"),
            doc("cooking recipes for beginners"),
        ];
        let result = pipeline.compress(&docs, "rust programming").unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].page_content.len() <= 20);
    }

    #[test]
    fn test_pipeline_empty_pipeline() {
        let pipeline = CompressorPipeline::new();
        assert!(pipeline.is_empty());
        assert_eq!(pipeline.len(), 0);
        let docs = vec![doc("unchanged")];
        let result = pipeline.compress(&docs, "query").unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].page_content, "unchanged");
    }

    #[test]
    fn test_pipeline_short_circuits_on_empty() {
        let pipeline = CompressorPipeline::new()
            .add(Box::new(RelevanceScorer::new(1.0)))
            .add(Box::new(LengthCompressor::new(5)));
        let docs = vec![doc("no matching keywords at all")];
        let result = pipeline.compress(&docs, "nonexistent").unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_pipeline_len() {
        let pipeline = CompressorPipeline::new()
            .add(Box::new(LengthCompressor::new(10)))
            .add(Box::new(RedundancyFilter::new(0.8)));
        assert_eq!(pipeline.len(), 2);
        assert!(!pipeline.is_empty());
    }

    // -----------------------------------------------------------------------
    // ContextualCompressionRetriever tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_retriever_retrieve_with_k() {
        let docs = vec![
            doc("rust is fast"),
            doc("rust is safe"),
            doc("python is dynamic"),
        ];
        let compressor = Box::new(RelevanceScorer::new(0.5));
        let retriever = ContextualCompressionRetriever::new(docs, compressor);
        let result = retriever.retrieve("rust", 1).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_retriever_retrieve_all() {
        let docs = vec![
            doc("rust is fast"),
            doc("rust is safe"),
            doc("python is dynamic"),
        ];
        let compressor = Box::new(RelevanceScorer::new(0.5));
        let retriever = ContextualCompressionRetriever::new(docs, compressor);
        let result = retriever.retrieve_all("rust").unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_retriever_empty_docs() {
        let compressor = Box::new(LengthCompressor::new(100));
        let retriever = ContextualCompressionRetriever::new(vec![], compressor);
        let result = retriever.retrieve("query", 5).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_retriever_k_larger_than_results() {
        let docs = vec![doc("only one")];
        let compressor = Box::new(LengthCompressor::new(100));
        let retriever = ContextualCompressionRetriever::new(docs, compressor);
        let result = retriever.retrieve("query", 10).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_retriever_end_to_end_with_pipeline() {
        let docs = vec![
            doc_with_meta(
                "rust is a great programming language. It is fast.",
                vec![("source", json!("docs"))],
            ),
            doc_with_meta(
                "cooking pasta is easy. Boil water first.",
                vec![("source", json!("recipes"))],
            ),
            doc("no metadata here"),
        ];
        let pipeline = CompressorPipeline::new()
            .add(Box::new(MetadataFilter::new().require_field("source")))
            .add(Box::new(RelevanceScorer::new(0.5)))
            .add(Box::new(LengthCompressor::new(30)));

        let retriever = ContextualCompressionRetriever::new(docs, Box::new(pipeline));
        let result = retriever.retrieve_all("rust programming").unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].page_content.len() <= 30);
    }

    #[test]
    fn test_retriever_all_filtered_out() {
        let docs = vec![doc("nothing relevant")];
        let compressor = Box::new(RelevanceScorer::new(1.0));
        let retriever = ContextualCompressionRetriever::new(docs, compressor);
        let result = retriever.retrieve("nonexistent keywords here", 5).unwrap();
        assert!(result.is_empty());
    }
}
