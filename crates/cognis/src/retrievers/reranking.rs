//! Reranking retrievers that take initial retrieval results and reorder them
//! by relevance using various scoring strategies.
//!
//! Provides pluggable rerankers (keyword overlap, TF-IDF, cross-encoder,
//! length-based, metadata-based) that can be composed via [`CascadeReranker`]
//! or sequentially via [`RerankerPipeline`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use cognis_core::documents::Document;
use cognis_core::error::Result;

// ---------------------------------------------------------------------------
// Reranker trait
// ---------------------------------------------------------------------------

/// Trait for reranking a set of documents given a query.
///
/// Implementations return documents paired with relevance scores, sorted in
/// descending score order.
pub trait Reranker: Send + Sync {
    /// Score and sort `documents` by relevance to `query`.
    ///
    /// Returns `(Document, score)` pairs sorted by score descending.
    fn rerank(&self, query: &str, documents: &[Document]) -> Result<Vec<(Document, f64)>>;
}

// ---------------------------------------------------------------------------
// KeywordReranker
// ---------------------------------------------------------------------------

/// Scores documents by keyword overlap with the query.
///
/// Score = (number of query terms appearing in document) / (total query terms).
/// Terms are lowercased and split on whitespace.
pub struct KeywordReranker;

impl KeywordReranker {
    /// Create a new keyword reranker.
    pub fn new() -> Self {
        Self
    }
}

impl Default for KeywordReranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker for KeywordReranker {
    fn rerank(&self, query: &str, documents: &[Document]) -> Result<Vec<(Document, f64)>> {
        let query_terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();

        if query_terms.is_empty() {
            let mut results: Vec<(Document, f64)> =
                documents.iter().map(|d| (d.clone(), 0.0)).collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return Ok(results);
        }

        let total = query_terms.len() as f64;
        let query_set: HashSet<&str> = query_terms.iter().map(|s| s.as_str()).collect();

        let mut results: Vec<(Document, f64)> = documents
            .iter()
            .map(|doc| {
                let doc_lower = doc.page_content.to_lowercase();
                let doc_words: HashSet<String> =
                    doc_lower.split_whitespace().map(String::from).collect();
                let overlap = query_set
                    .iter()
                    .filter(|qt| doc_words.contains(**qt))
                    .count();
                (doc.clone(), overlap as f64 / total)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// TfIdfReranker
// ---------------------------------------------------------------------------

/// Scores documents using TF-IDF: term frequency * inverse document frequency
/// across the candidate set.
///
/// TF(t, d) = count(t in d) / |d|
/// IDF(t) = ln(N / df(t))  where df(t) = number of docs containing t
/// Score = sum over query terms of TF * IDF
pub struct TfIdfReranker;

impl TfIdfReranker {
    /// Create a new TF-IDF reranker.
    pub fn new() -> Self {
        Self
    }
}

impl Default for TfIdfReranker {
    fn default() -> Self {
        Self::new()
    }
}

impl Reranker for TfIdfReranker {
    fn rerank(&self, query: &str, documents: &[Document]) -> Result<Vec<(Document, f64)>> {
        let query_terms: Vec<String> = query
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();

        if documents.is_empty() || query_terms.is_empty() {
            let mut results: Vec<(Document, f64)> =
                documents.iter().map(|d| (d.clone(), 0.0)).collect();
            results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            return Ok(results);
        }

        let n = documents.len() as f64;

        // Pre-tokenize all documents.
        let doc_tokens: Vec<Vec<String>> = documents
            .iter()
            .map(|doc| {
                doc.page_content
                    .to_lowercase()
                    .split_whitespace()
                    .map(String::from)
                    .collect()
            })
            .collect();

        // Compute document frequency for each query term.
        let mut df: HashMap<&str, usize> = HashMap::new();
        for qt in &query_terms {
            let count = doc_tokens
                .iter()
                .filter(|tokens| tokens.iter().any(|t| t == qt))
                .count();
            df.insert(qt.as_str(), count);
        }

        let mut results: Vec<(Document, f64)> = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| {
                let tokens = &doc_tokens[i];
                let doc_len = tokens.len() as f64;
                if doc_len == 0.0 {
                    return (doc.clone(), 0.0);
                }

                let score: f64 = query_terms
                    .iter()
                    .map(|qt| {
                        let tf = tokens.iter().filter(|t| t.as_str() == qt.as_str()).count() as f64
                            / doc_len;
                        let doc_freq = *df.get(qt.as_str()).unwrap_or(&0);
                        if doc_freq == 0 {
                            0.0
                        } else {
                            let idf = (n / doc_freq as f64).ln();
                            tf * idf
                        }
                    })
                    .sum();

                (doc.clone(), score)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// CrossEncoderReranker
// ---------------------------------------------------------------------------

/// Type alias for a cross-encoder scoring function: `(query, document) -> score`.
type ScorerFn = Arc<dyn Fn(&str, &str) -> f64 + Send + Sync>;

/// A pluggable reranker that accepts a custom scoring function.
///
/// This is useful for integrating external cross-encoder models or any
/// arbitrary relevance scoring logic.
pub struct CrossEncoderReranker {
    scorer: ScorerFn,
}

impl CrossEncoderReranker {
    /// Create a new cross-encoder reranker with the given scoring function.
    ///
    /// The function takes `(query, document_content)` and returns a score.
    pub fn new(scorer: ScorerFn) -> Self {
        Self { scorer }
    }
}

impl Reranker for CrossEncoderReranker {
    fn rerank(&self, query: &str, documents: &[Document]) -> Result<Vec<(Document, f64)>> {
        let mut results: Vec<(Document, f64)> = documents
            .iter()
            .map(|doc| {
                let score = (self.scorer)(query, &doc.page_content);
                (doc.clone(), score)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// LengthReranker
// ---------------------------------------------------------------------------

/// Scores documents inversely proportional to the distance from an ideal
/// document length (in characters).
///
/// Score = 1.0 / (1.0 + |doc_length - ideal_length|)
pub struct LengthReranker {
    /// The ideal document length in characters.
    pub ideal_length: usize,
}

impl LengthReranker {
    /// Create a new length reranker with the given ideal length.
    pub fn new(ideal_length: usize) -> Self {
        Self { ideal_length }
    }
}

impl Reranker for LengthReranker {
    fn rerank(&self, _query: &str, documents: &[Document]) -> Result<Vec<(Document, f64)>> {
        let mut results: Vec<(Document, f64)> = documents
            .iter()
            .map(|doc| {
                let diff =
                    (doc.page_content.len() as isize - self.ideal_length as isize).unsigned_abs();
                let score = 1.0 / (1.0 + diff as f64);
                (doc.clone(), score)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// MetadataReranker
// ---------------------------------------------------------------------------

/// Scores documents based on a numeric metadata field value.
///
/// If the field is missing or not numeric, the document receives a score of 0.
pub struct MetadataReranker {
    /// The metadata key to read the score from.
    pub field: String,
}

impl MetadataReranker {
    /// Create a new metadata reranker that reads scores from the given field.
    pub fn new(field: impl Into<String>) -> Self {
        Self {
            field: field.into(),
        }
    }
}

impl Reranker for MetadataReranker {
    fn rerank(&self, _query: &str, documents: &[Document]) -> Result<Vec<(Document, f64)>> {
        let mut results: Vec<(Document, f64)> = documents
            .iter()
            .map(|doc| {
                let score = doc
                    .metadata
                    .get(&self.field)
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                (doc.clone(), score)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// CascadeReranker
// ---------------------------------------------------------------------------

/// Chains multiple rerankers with configurable weights and computes a weighted
/// sum of their scores for each document.
pub struct CascadeReranker {
    /// (reranker, weight) pairs.
    rerankers: Vec<(Box<dyn Reranker>, f64)>,
}

impl CascadeReranker {
    /// Create a new cascade reranker with the given (reranker, weight) pairs.
    pub fn new(rerankers: Vec<(Box<dyn Reranker>, f64)>) -> Self {
        Self { rerankers }
    }
}

impl Reranker for CascadeReranker {
    fn rerank(&self, query: &str, documents: &[Document]) -> Result<Vec<(Document, f64)>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Collect scores from each reranker indexed by document position.
        let mut combined_scores: Vec<f64> = vec![0.0; documents.len()];

        for (reranker, weight) in &self.rerankers {
            let scored = reranker.rerank(query, documents)?;

            // Build a map from document content+id to score for lookup.
            // We match by index since the reranker may reorder.
            let score_map: HashMap<usize, f64> = documents
                .iter()
                .enumerate()
                .map(|(i, doc)| {
                    let score = scored
                        .iter()
                        .find(|(d, _)| d.page_content == doc.page_content && d.id == doc.id)
                        .map(|(_, s)| *s)
                        .unwrap_or(0.0);
                    (i, score)
                })
                .collect();

            for (i, score) in score_map {
                combined_scores[i] += score * weight;
            }
        }

        let mut results: Vec<(Document, f64)> = documents
            .iter()
            .enumerate()
            .map(|(i, doc)| (doc.clone(), combined_scores[i]))
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// RerankingRetriever
// ---------------------------------------------------------------------------

/// A retriever that wraps a base document set with a reranker, returning
/// top-k results after reranking.
pub struct RerankingRetriever {
    /// The base documents to search over.
    documents: Vec<Document>,
    /// The reranker to apply.
    reranker: Box<dyn Reranker>,
    /// Maximum number of results to return.
    top_k: usize,
    /// Minimum score threshold (documents below this are filtered out).
    min_score: Option<f64>,
}

impl RerankingRetriever {
    /// Create a new reranking retriever with the given documents and reranker.
    pub fn new(documents: Vec<Document>, reranker: Box<dyn Reranker>) -> Self {
        Self {
            documents,
            reranker,
            top_k: 10,
            min_score: None,
        }
    }

    /// Set the reranker.
    pub fn with_reranker(mut self, reranker: Box<dyn Reranker>) -> Self {
        self.reranker = reranker;
        self
    }

    /// Set the maximum number of results to return.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set a minimum score threshold. Documents scoring below this are excluded.
    pub fn with_min_score(mut self, threshold: f64) -> Self {
        self.min_score = Some(threshold);
        self
    }

    /// Retrieve the top-k documents after reranking, without scores.
    pub fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let scored = self.retrieve_with_scores(query, k)?;
        Ok(scored.into_iter().map(|(doc, _)| doc).collect())
    }

    /// Retrieve the top-k documents after reranking, with scores.
    pub fn retrieve_with_scores(&self, query: &str, k: usize) -> Result<Vec<(Document, f64)>> {
        let mut scored = self.reranker.rerank(query, &self.documents)?;

        // Apply min_score filter.
        if let Some(threshold) = self.min_score {
            scored.retain(|(_, score)| *score >= threshold);
        }

        // Take top-k.
        scored.truncate(k);
        Ok(scored)
    }
}

// ---------------------------------------------------------------------------
// RerankerPipeline
// ---------------------------------------------------------------------------

/// A sequential pipeline of rerankers. Each stage reranks the results of the
/// previous stage and keeps only the top-n for that stage.
pub struct RerankerPipeline {
    /// (reranker, top_n) pairs applied in order.
    stages: Vec<(Box<dyn Reranker>, usize)>,
}

impl RerankerPipeline {
    /// Create a new reranker pipeline with the given stages.
    ///
    /// Each stage is a `(reranker, top_n)` pair. The reranker is applied and
    /// only the top `top_n` results are passed to the next stage.
    pub fn new(stages: Vec<(Box<dyn Reranker>, usize)>) -> Self {
        Self { stages }
    }

    /// Run the pipeline on the given documents for the given query.
    pub fn run(&self, query: &str, documents: &[Document]) -> Result<Vec<(Document, f64)>> {
        let mut current: Vec<Document> = documents.to_vec();

        let mut final_scores: Vec<(Document, f64)> = Vec::new();

        for (reranker, top_n) in &self.stages {
            let mut scored = reranker.rerank(query, &current)?;
            scored.truncate(*top_n);
            current = scored.iter().map(|(d, _)| d.clone()).collect();
            final_scores = scored;
        }

        Ok(final_scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_doc(content: &str) -> Document {
        Document::new(content)
    }

    fn make_doc_with_meta(content: &str, key: &str, value: f64) -> Document {
        let mut meta = HashMap::new();
        meta.insert(key.to_string(), json!(value));
        Document::new(content).with_metadata(meta)
    }

    fn make_docs(contents: &[&str]) -> Vec<Document> {
        contents.iter().map(|c| make_doc(c)).collect()
    }

    // -----------------------------------------------------------------------
    // KeywordReranker tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_keyword_reranker_basic() {
        let reranker = KeywordReranker::new();
        let docs = make_docs(&["hello world", "foo bar", "hello foo"]);
        let results = reranker.rerank("hello", &docs).unwrap();

        // "hello world" and "hello foo" both contain "hello"
        assert_eq!(results[0].1, 1.0);
        assert_eq!(results[1].1, 1.0);
        assert_eq!(results[2].1, 0.0);
    }

    #[test]
    fn test_keyword_reranker_partial_overlap() {
        let reranker = KeywordReranker::new();
        let docs = make_docs(&["the cat sat", "the dog ran", "a bird flew"]);
        let results = reranker.rerank("the cat", &docs).unwrap();

        // "the cat sat" has 2/2 = 1.0
        assert_eq!(results[0].0.page_content, "the cat sat");
        assert_eq!(results[0].1, 1.0);

        // "the dog ran" has 1/2 = 0.5
        assert_eq!(results[1].0.page_content, "the dog ran");
        assert_eq!(results[1].1, 0.5);

        // "a bird flew" has 0/2 = 0.0
        assert_eq!(results[2].1, 0.0);
    }

    #[test]
    fn test_keyword_reranker_case_insensitive() {
        let reranker = KeywordReranker::new();
        let docs = make_docs(&["Hello World", "HELLO world"]);
        let results = reranker.rerank("hello", &docs).unwrap();

        assert_eq!(results[0].1, 1.0);
        assert_eq!(results[1].1, 1.0);
    }

    #[test]
    fn test_keyword_reranker_no_overlap() {
        let reranker = KeywordReranker::new();
        let docs = make_docs(&["alpha beta", "gamma delta"]);
        let results = reranker.rerank("xyz", &docs).unwrap();

        assert_eq!(results[0].1, 0.0);
        assert_eq!(results[1].1, 0.0);
    }

    #[test]
    fn test_keyword_reranker_empty_docs() {
        let reranker = KeywordReranker::new();
        let results = reranker.rerank("hello", &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_keyword_reranker_empty_query() {
        let reranker = KeywordReranker::new();
        let docs = make_docs(&["hello world"]);
        let results = reranker.rerank("", &docs).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 0.0);
    }

    // -----------------------------------------------------------------------
    // TfIdfReranker tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_tfidf_reranker_basic() {
        let reranker = TfIdfReranker::new();
        let docs = make_docs(&[
            "the cat sat on the mat",
            "the dog chased the cat",
            "a bird flew over",
        ]);
        let results = reranker.rerank("cat", &docs).unwrap();

        // "cat" appears in docs 0 and 1 but not 2
        assert!(results[2].1 == 0.0);
        // Both docs with "cat" should have positive scores
        assert!(results[0].1 > 0.0);
        assert!(results[1].1 > 0.0);
    }

    #[test]
    fn test_tfidf_reranker_rare_term_scores_higher() {
        let reranker = TfIdfReranker::new();
        let docs = make_docs(&["common rare", "common ordinary", "common typical"]);
        // "rare" only appears in doc 0, "common" in all docs
        let results = reranker.rerank("rare common", &docs).unwrap();

        // Doc 0 should score highest (has the rare term)
        assert_eq!(results[0].0.page_content, "common rare");
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_tfidf_reranker_empty_docs() {
        let reranker = TfIdfReranker::new();
        let results = reranker.rerank("hello", &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_tfidf_reranker_empty_query() {
        let reranker = TfIdfReranker::new();
        let docs = make_docs(&["hello world"]);
        let results = reranker.rerank("", &docs).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 0.0);
    }

    #[test]
    fn test_tfidf_single_doc() {
        let reranker = TfIdfReranker::new();
        let docs = make_docs(&["hello world hello"]);
        let results = reranker.rerank("hello", &docs).unwrap();
        // IDF = ln(1/1) = 0, so score should be 0
        // (with only one doc, IDF for any present term is ln(1) = 0)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 0.0);
    }

    #[test]
    fn test_tfidf_varying_frequencies() {
        let reranker = TfIdfReranker::new();
        let docs = make_docs(&[
            "rust rust rust python",
            "rust python",
            "python python python",
        ]);
        let results = reranker.rerank("rust", &docs).unwrap();
        // "rust" appears in doc 0 (3/4 TF) and doc 1 (1/2 TF), not in doc 2
        // IDF = ln(3/2)
        assert!(results[0].1 > results[1].1);
        assert_eq!(results[2].1, 0.0);
    }

    // -----------------------------------------------------------------------
    // CrossEncoderReranker tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cross_encoder_custom_function() {
        // Simple scorer: score = length of query + length of doc
        let scorer = Arc::new(|_query: &str, doc: &str| doc.len() as f64);
        let reranker = CrossEncoderReranker::new(scorer);
        let docs = make_docs(&["short", "a longer document", "mid"]);

        let results = reranker.rerank("test", &docs).unwrap();

        // Longest doc should score highest
        assert_eq!(results[0].0.page_content, "a longer document");
        assert_eq!(results[1].0.page_content, "short");
        assert_eq!(results[2].0.page_content, "mid");
    }

    #[test]
    fn test_cross_encoder_query_dependent() {
        // Score based on whether doc contains query
        let scorer = Arc::new(
            |query: &str, doc: &str| {
                if doc.contains(query) {
                    1.0
                } else {
                    0.0
                }
            },
        );
        let reranker = CrossEncoderReranker::new(scorer);
        let docs = make_docs(&["hello world", "goodbye world", "hello there"]);

        let results = reranker.rerank("hello", &docs).unwrap();
        assert_eq!(results[0].1, 1.0);
        assert_eq!(results[1].1, 1.0);
        assert_eq!(results[2].1, 0.0);
    }

    #[test]
    fn test_cross_encoder_empty_docs() {
        let scorer = Arc::new(|_: &str, _: &str| 0.5);
        let reranker = CrossEncoderReranker::new(scorer);
        let results = reranker.rerank("test", &[]).unwrap();
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // LengthReranker tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_length_reranker_exact_match() {
        let reranker = LengthReranker::new(5);
        let docs = make_docs(&["12345", "1234567890", "12"]);
        let results = reranker.rerank("query", &docs).unwrap();

        // "12345" is exactly 5 chars -> score = 1.0/(1+0) = 1.0
        assert_eq!(results[0].0.page_content, "12345");
        assert_eq!(results[0].1, 1.0);
    }

    #[test]
    fn test_length_reranker_ordering() {
        let reranker = LengthReranker::new(10);
        let docs = make_docs(&["ab", "abcdefghij", "abcde"]);
        let results = reranker.rerank("query", &docs).unwrap();

        // "abcdefghij" (10 chars) should be first
        assert_eq!(results[0].0.page_content, "abcdefghij");
        assert_eq!(results[0].1, 1.0);
    }

    #[test]
    fn test_length_reranker_different_ideal() {
        let reranker = LengthReranker::new(100);
        let docs = make_docs(&["short", "a bit longer text here"]);
        let results = reranker.rerank("query", &docs).unwrap();

        // "a bit longer text here" (22 chars) is closer to 100 than "short" (5 chars)
        assert_eq!(results[0].0.page_content, "a bit longer text here");
        assert!(results[0].1 > results[1].1);
    }

    // -----------------------------------------------------------------------
    // MetadataReranker tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_metadata_reranker_numeric_field() {
        let reranker = MetadataReranker::new("boost");
        let docs = vec![
            make_doc_with_meta("doc a", "boost", 0.5),
            make_doc_with_meta("doc b", "boost", 2.0),
            make_doc_with_meta("doc c", "boost", 1.0),
        ];
        let results = reranker.rerank("query", &docs).unwrap();

        assert_eq!(results[0].0.page_content, "doc b");
        assert_eq!(results[0].1, 2.0);
        assert_eq!(results[1].0.page_content, "doc c");
        assert_eq!(results[1].1, 1.0);
        assert_eq!(results[2].0.page_content, "doc a");
        assert_eq!(results[2].1, 0.5);
    }

    #[test]
    fn test_metadata_reranker_missing_field() {
        let reranker = MetadataReranker::new("score");
        let docs = vec![
            make_doc("no metadata"),
            make_doc_with_meta("has score", "score", 5.0),
        ];
        let results = reranker.rerank("query", &docs).unwrap();

        assert_eq!(results[0].0.page_content, "has score");
        assert_eq!(results[0].1, 5.0);
        assert_eq!(results[1].0.page_content, "no metadata");
        assert_eq!(results[1].1, 0.0);
    }

    #[test]
    fn test_metadata_reranker_non_numeric_value() {
        let reranker = MetadataReranker::new("tag");
        let mut meta = HashMap::new();
        meta.insert("tag".to_string(), json!("not a number"));
        let docs = vec![Document::new("doc").with_metadata(meta)];
        let results = reranker.rerank("query", &docs).unwrap();

        // Non-numeric metadata should give score 0
        assert_eq!(results[0].1, 0.0);
    }

    // -----------------------------------------------------------------------
    // CascadeReranker tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cascade_reranker_weighted_combination() {
        // Combine keyword reranker (weight 0.7) with metadata reranker (weight 0.3)
        let rerankers: Vec<(Box<dyn Reranker>, f64)> = vec![
            (Box::new(KeywordReranker::new()), 0.7),
            (Box::new(MetadataReranker::new("boost")), 0.3),
        ];
        let cascade = CascadeReranker::new(rerankers);

        let docs = vec![
            make_doc_with_meta("hello world", "boost", 1.0),
            make_doc_with_meta("foo bar", "boost", 10.0),
        ];

        let results = cascade.rerank("hello", &docs).unwrap();

        // "hello world": keyword=1.0*0.7 + meta=1.0*0.3 = 1.0
        // "foo bar": keyword=0.0*0.7 + meta=10.0*0.3 = 3.0
        assert_eq!(results[0].0.page_content, "foo bar");
        assert!((results[0].1 - 3.0).abs() < 1e-10);
        assert_eq!(results[1].0.page_content, "hello world");
        assert!((results[1].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cascade_reranker_empty_docs() {
        let rerankers: Vec<(Box<dyn Reranker>, f64)> =
            vec![(Box::new(KeywordReranker::new()), 1.0)];
        let cascade = CascadeReranker::new(rerankers);
        let results = cascade.rerank("hello", &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_cascade_reranker_single_reranker() {
        let rerankers: Vec<(Box<dyn Reranker>, f64)> =
            vec![(Box::new(KeywordReranker::new()), 1.0)];
        let cascade = CascadeReranker::new(rerankers);
        let docs = make_docs(&["hello world", "foo bar"]);
        let results = cascade.rerank("hello", &docs).unwrap();

        assert_eq!(results[0].0.page_content, "hello world");
        assert_eq!(results[0].1, 1.0);
    }

    // -----------------------------------------------------------------------
    // RerankingRetriever tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_reranking_retriever_end_to_end() {
        let docs = make_docs(&["rust programming", "python scripting", "rust and python"]);
        let retriever = RerankingRetriever::new(docs, Box::new(KeywordReranker::new()));

        let results = retriever.retrieve("rust", 2).unwrap();
        assert_eq!(results.len(), 2);
        // Both docs containing "rust" should be returned
        assert!(results[0].page_content.contains("rust"));
        assert!(results[1].page_content.contains("rust"));
    }

    #[test]
    fn test_reranking_retriever_with_scores() {
        let docs = make_docs(&["alpha beta", "gamma delta"]);
        let retriever = RerankingRetriever::new(docs, Box::new(KeywordReranker::new()));

        let results = retriever.retrieve_with_scores("alpha", 5).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.page_content, "alpha beta");
        assert_eq!(results[0].1, 1.0);
        assert_eq!(results[1].1, 0.0);
    }

    #[test]
    fn test_reranking_retriever_min_score() {
        let docs = make_docs(&["hello world", "foo bar", "hello foo"]);
        let retriever =
            RerankingRetriever::new(docs, Box::new(KeywordReranker::new())).with_min_score(0.5);

        let results = retriever.retrieve("hello", 10).unwrap();
        // Only docs with score >= 0.5 should be returned
        assert_eq!(results.len(), 2);
        for doc in &results {
            assert!(doc.page_content.contains("hello"));
        }
    }

    #[test]
    fn test_reranking_retriever_top_k_limits() {
        let docs = make_docs(&["a", "b", "c", "d", "e"]);
        let retriever = RerankingRetriever::new(docs, Box::new(KeywordReranker::new()));

        let results = retriever.retrieve("test", 3).unwrap();
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_reranking_retriever_builder_pattern() {
        let docs = make_docs(&["hello", "world"]);
        let retriever = RerankingRetriever::new(docs, Box::new(KeywordReranker::new()))
            .with_top_k(5)
            .with_min_score(0.1)
            .with_reranker(Box::new(KeywordReranker::new()));

        let results = retriever.retrieve("hello", 1).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_reranking_retriever_empty_docs() {
        let retriever = RerankingRetriever::new(vec![], Box::new(KeywordReranker::new()));

        let results = retriever.retrieve("hello", 5).unwrap();
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // RerankerPipeline tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pipeline_sequential_execution() {
        // Stage 1: keyword reranker, keep top 2
        // Stage 2: length reranker (ideal 5 chars), keep top 1
        let stages: Vec<(Box<dyn Reranker>, usize)> = vec![
            (Box::new(KeywordReranker::new()), 2),
            (Box::new(LengthReranker::new(5)), 1),
        ];
        let pipeline = RerankerPipeline::new(stages);

        let docs = make_docs(&["hello", "hello world is great", "nothing here"]);
        let results = pipeline.run("hello", &docs).unwrap();

        assert_eq!(results.len(), 1);
        // "hello" (5 chars) should be preferred by LengthReranker(5) over "hello world is great"
        assert_eq!(results[0].0.page_content, "hello");
    }

    #[test]
    fn test_pipeline_empty_docs() {
        let stages: Vec<(Box<dyn Reranker>, usize)> = vec![(Box::new(KeywordReranker::new()), 5)];
        let pipeline = RerankerPipeline::new(stages);
        let results = pipeline.run("hello", &[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_pipeline_single_stage() {
        let stages: Vec<(Box<dyn Reranker>, usize)> = vec![(Box::new(KeywordReranker::new()), 2)];
        let pipeline = RerankerPipeline::new(stages);

        let docs = make_docs(&["hello world", "foo bar", "hello there"]);
        let results = pipeline.run("hello", &docs).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].1 > 0.0);
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_document_keyword() {
        let reranker = KeywordReranker::new();
        let docs = make_docs(&["only one"]);
        let results = reranker.rerank("one", &docs).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].1, 1.0);
    }

    #[test]
    fn test_single_document_tfidf() {
        let reranker = TfIdfReranker::new();
        let docs = make_docs(&["only one"]);
        let results = reranker.rerank("one", &docs).unwrap();
        assert_eq!(results.len(), 1);
        // With single doc, IDF = ln(1/1) = 0
        assert_eq!(results[0].1, 0.0);
    }

    #[test]
    fn test_min_score_filters_all() {
        let docs = make_docs(&["foo bar", "baz qux"]);
        let retriever =
            RerankingRetriever::new(docs, Box::new(KeywordReranker::new())).with_min_score(0.5);

        // Query has no overlap with any doc
        let results = retriever.retrieve("xyz", 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_cascade_equal_weights() {
        let rerankers: Vec<(Box<dyn Reranker>, f64)> = vec![
            (Box::new(KeywordReranker::new()), 0.5),
            (Box::new(KeywordReranker::new()), 0.5),
        ];
        let cascade = CascadeReranker::new(rerankers);
        let docs = make_docs(&["hello world", "foo bar"]);
        let results = cascade.rerank("hello", &docs).unwrap();

        // Same reranker twice with equal weights = same as single reranker
        assert_eq!(results[0].0.page_content, "hello world");
        assert!((results[0].1 - 1.0).abs() < 1e-10);
    }
}
