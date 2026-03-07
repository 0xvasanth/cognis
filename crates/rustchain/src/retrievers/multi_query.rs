//! Multi-query retriever that generates multiple query variations to improve
//! retrieval recall.
//!
//! The core idea: a single user query may miss relevant documents due to phrasing.
//! By generating multiple reformulations and retrieving for each, we cast a wider
//! net and then merge/deduplicate the results.
//!
//! This module provides:
//! - [`QueryGenerator`] trait for producing query variations.
//! - [`SimpleQueryGenerator`] that creates variations via rephrasing heuristics.
//! - [`TemplateQueryGenerator`] that uses configurable prompt templates.
//! - [`MultiQueryRetriever`] that fans out to a base retriever with multiple queries.
//! - [`FusionRetriever`] that merges results using Reciprocal Rank Fusion (RRF).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use rustchain_core::retrievers::BaseRetriever;

// ---------------------------------------------------------------------------
// QueryGenerator trait
// ---------------------------------------------------------------------------

/// Trait for generating multiple query variations from a single input query.
#[async_trait]
pub trait QueryGenerator: Send + Sync {
    /// Generate alternative phrasings of the given query.
    async fn generate_queries(&self, query: &str) -> Result<Vec<String>>;
}

// ---------------------------------------------------------------------------
// SimpleQueryGenerator
// ---------------------------------------------------------------------------

/// Generates query variations using simple heuristic transformations:
/// - Rephrasing with synonyms for common question words
/// - Adding specificity ("detailed", "comprehensive")
/// - Changing perspective ("What is X?" -> "Define X", "Explain X")
///
/// This generator works entirely offline with no LLM calls.
pub struct SimpleQueryGenerator {
    /// Number of query variations to produce (default: 3).
    pub num_queries: usize,
}

impl Default for SimpleQueryGenerator {
    fn default() -> Self {
        Self { num_queries: 3 }
    }
}

impl SimpleQueryGenerator {
    /// Create a new `SimpleQueryGenerator` with the specified number of variations.
    pub fn new(num_queries: usize) -> Self {
        Self { num_queries }
    }

    /// Apply perspective transformations to a query.
    fn perspective_variations(query: &str) -> Vec<String> {
        let lower = query.to_lowercase();
        let trimmed = query.trim().trim_end_matches('?');
        let mut variations = Vec::new();

        // "What is X?" -> "Define X", "Explain X"
        if lower.starts_with("what is ") || lower.starts_with("what are ") {
            let subject = if lower.starts_with("what is ") {
                &trimmed["what is ".len()..]
            } else {
                &trimmed["what are ".len()..]
            };
            variations.push(format!("Define {}", subject));
            variations.push(format!("Explain {}", subject));
            variations.push(format!("Describe {}", subject));
        }

        // "How does X work?" -> "Explain how X works", "Mechanism of X"
        if lower.starts_with("how does ") || lower.starts_with("how do ") {
            let subject = if lower.starts_with("how does ") {
                &trimmed["how does ".len()..]
            } else {
                &trimmed["how do ".len()..]
            };
            variations.push(format!("Explain how {} works", subject));
            variations.push(format!("Mechanism of {}", subject));
        }

        // "Why" questions -> "Reason for X", "Explain why X"
        if lower.starts_with("why ") {
            let rest = &trimmed["why ".len()..];
            variations.push(format!("Reason for {}", rest));
            variations.push(format!("Explain why {}", rest));
        }

        // Generic transformations for any query
        variations.push(format!("Tell me about {}", trimmed));
        variations.push(format!("{} overview", trimmed));
        variations.push(format!("Detailed information on {}", trimmed));

        variations
    }

    /// Apply specificity variations.
    fn specificity_variations(query: &str) -> Vec<String> {
        let trimmed = query.trim().trim_end_matches('?');
        vec![
            format!("{} in detail", trimmed),
            format!("comprehensive guide to {}", trimmed),
            format!("{} summary", trimmed),
            format!("brief explanation of {}", trimmed),
        ]
    }
}

#[async_trait]
impl QueryGenerator for SimpleQueryGenerator {
    async fn generate_queries(&self, query: &str) -> Result<Vec<String>> {
        let mut all_variations = Vec::new();
        all_variations.extend(Self::perspective_variations(query));
        all_variations.extend(Self::specificity_variations(query));

        // Deduplicate and remove variations identical to original
        let original_lower = query.to_lowercase();
        let mut seen = HashSet::new();
        let mut unique = Vec::new();
        for v in all_variations {
            let key = v.to_lowercase();
            if key != original_lower && seen.insert(key) {
                unique.push(v);
            }
        }

        unique.truncate(self.num_queries);
        Ok(unique)
    }
}

// ---------------------------------------------------------------------------
// TemplateQueryGenerator
// ---------------------------------------------------------------------------

/// Generates query variations by formatting configurable template strings.
///
/// Each template should contain `{query}` as a placeholder that will be
/// replaced with the original query.
///
/// # Example
///
/// ```rust,ignore
/// let gen = TemplateQueryGenerator::new(vec![
///     "Rephrase '{query}' as a question".to_string(),
///     "What are the key aspects of {query}".to_string(),
/// ]);
/// ```
pub struct TemplateQueryGenerator {
    /// Template strings containing `{query}` placeholders.
    templates: Vec<String>,
}

impl TemplateQueryGenerator {
    /// Create a new `TemplateQueryGenerator` with the given templates.
    pub fn new(templates: Vec<String>) -> Self {
        Self { templates }
    }
}

#[async_trait]
impl QueryGenerator for TemplateQueryGenerator {
    async fn generate_queries(&self, query: &str) -> Result<Vec<String>> {
        let variations = self
            .templates
            .iter()
            .map(|t| t.replace("{query}", query))
            .collect();
        Ok(variations)
    }
}

// ---------------------------------------------------------------------------
// MultiQueryRetriever
// ---------------------------------------------------------------------------

/// Retriever that generates multiple query variations, retrieves documents for
/// each, and merges the results with optional deduplication.
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::retrievers::multi_query::{MultiQueryRetriever, SimpleQueryGenerator};
///
/// let retriever = MultiQueryRetriever::builder(base_retriever)
///     .query_generator(Arc::new(SimpleQueryGenerator::default()))
///     .k(5)
///     .build();
///
/// let docs = retriever.get_relevant_documents("my query").await?;
/// ```
pub struct MultiQueryRetriever {
    /// The base retriever to query with each variation.
    inner: Arc<dyn BaseRetriever>,
    /// The query generator for producing variations.
    query_generator: Arc<dyn QueryGenerator>,
    /// Maximum number of documents to return (default: 4).
    k: usize,
    /// Whether to deduplicate documents across query results (default: true).
    deduplicate: bool,
    /// Whether to include the original query in the retrieval set (default: true).
    include_original: bool,
}

impl MultiQueryRetriever {
    /// Start building a `MultiQueryRetriever` with the given base retriever.
    pub fn builder(inner: Arc<dyn BaseRetriever>) -> MultiQueryRetrieverBuilder {
        MultiQueryRetrieverBuilder {
            inner,
            query_generator: Arc::new(SimpleQueryGenerator::default()),
            k: 4,
            deduplicate: true,
            include_original: true,
        }
    }

    /// Deduplicate documents by ID (if present) or by content hash.
    fn deduplicate_docs(docs: Vec<Document>) -> Vec<Document> {
        let mut seen_ids: HashSet<String> = HashSet::new();
        let mut seen_content: HashSet<String> = HashSet::new();
        let mut result = Vec::new();

        for doc in docs {
            // Prefer deduplication by ID if available
            if let Some(ref id) = doc.id {
                if !seen_ids.insert(id.clone()) {
                    continue;
                }
            }
            // Fall back to content-based dedup
            if seen_content.insert(doc.page_content.clone()) {
                result.push(doc);
            }
        }

        result
    }
}

/// Builder for [`MultiQueryRetriever`].
pub struct MultiQueryRetrieverBuilder {
    inner: Arc<dyn BaseRetriever>,
    query_generator: Arc<dyn QueryGenerator>,
    k: usize,
    deduplicate: bool,
    include_original: bool,
}

impl MultiQueryRetrieverBuilder {
    /// Set the query generator.
    pub fn query_generator(mut self, gen: Arc<dyn QueryGenerator>) -> Self {
        self.query_generator = gen;
        self
    }

    /// Set the maximum number of documents to return.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set whether to deduplicate documents (default: true).
    pub fn deduplicate(mut self, deduplicate: bool) -> Self {
        self.deduplicate = deduplicate;
        self
    }

    /// Set whether to include the original query (default: true).
    pub fn include_original(mut self, include: bool) -> Self {
        self.include_original = include;
        self
    }

    /// Build the `MultiQueryRetriever`.
    pub fn build(self) -> MultiQueryRetriever {
        MultiQueryRetriever {
            inner: self.inner,
            query_generator: self.query_generator,
            k: self.k,
            deduplicate: self.deduplicate,
            include_original: self.include_original,
        }
    }
}

#[async_trait]
impl BaseRetriever for MultiQueryRetriever {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
        // Generate query variations
        let mut queries = self.query_generator.generate_queries(query).await?;

        // Optionally include the original query
        if self.include_original {
            queries.insert(0, query.to_string());
        }

        // Retrieve documents for all queries concurrently
        let futures: Vec<_> = queries
            .iter()
            .map(|q| self.inner.get_relevant_documents(q))
            .collect();

        let all_results = join_all(futures).await;

        // Flatten results, propagating errors
        let mut all_docs = Vec::new();
        for result in all_results {
            all_docs.extend(result?);
        }

        // Deduplicate if enabled
        if self.deduplicate {
            all_docs = MultiQueryRetriever::deduplicate_docs(all_docs);
        }

        // Limit to k results
        all_docs.truncate(self.k);
        Ok(all_docs)
    }
}

// ---------------------------------------------------------------------------
// FusionRetriever
// ---------------------------------------------------------------------------

/// A multi-query retriever variant that uses Reciprocal Rank Fusion (RRF) to
/// merge results from multiple query variations.
///
/// Unlike `MultiQueryRetriever` which simply concatenates and deduplicates,
/// `FusionRetriever` scores each document using RRF:
///
/// ```text
/// score(doc) = sum over queries: 1.0 / (rrf_k + rank)
/// ```
///
/// where `rank` is the 1-indexed position of the document in that query's results.
/// Documents appearing in more result lists and at higher ranks receive higher scores.
///
/// # Example
///
/// ```rust,ignore
/// let fusion = FusionRetriever::builder(base_retriever)
///     .rrf_k(60)
///     .k(5)
///     .build();
///
/// let docs = fusion.get_relevant_documents("my query").await?;
/// ```
pub struct FusionRetriever {
    /// The base retriever to query with each variation.
    inner: Arc<dyn BaseRetriever>,
    /// The query generator for producing variations.
    query_generator: Arc<dyn QueryGenerator>,
    /// Maximum number of documents to return (default: 4).
    k: usize,
    /// RRF constant (default: 60). Higher values reduce the impact of rank position.
    rrf_k: usize,
    /// Whether to include the original query in the retrieval set (default: true).
    include_original: bool,
}

impl FusionRetriever {
    /// Start building a `FusionRetriever` with the given base retriever.
    pub fn builder(inner: Arc<dyn BaseRetriever>) -> FusionRetrieverBuilder {
        FusionRetrieverBuilder {
            inner,
            query_generator: Arc::new(SimpleQueryGenerator::default()),
            k: 4,
            rrf_k: 60,
            include_original: true,
        }
    }

    /// Apply Reciprocal Rank Fusion across multiple result sets.
    fn reciprocal_rank_fusion(
        result_sets: &[Vec<Document>],
        rrf_k: usize,
    ) -> Vec<(Document, f64)> {
        let mut score_map: HashMap<String, (Document, f64)> = HashMap::new();

        for docs in result_sets {
            for (rank, doc) in docs.iter().enumerate() {
                let score = 1.0 / (rrf_k as f64 + (rank + 1) as f64);
                let key = if let Some(ref id) = doc.id {
                    id.clone()
                } else {
                    doc.page_content.clone()
                };
                let entry = score_map
                    .entry(key)
                    .or_insert_with(|| (doc.clone(), 0.0));
                entry.1 += score;
            }
        }

        let mut scored: Vec<(Document, f64)> = score_map.into_values().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored
    }
}

/// Builder for [`FusionRetriever`].
pub struct FusionRetrieverBuilder {
    inner: Arc<dyn BaseRetriever>,
    query_generator: Arc<dyn QueryGenerator>,
    k: usize,
    rrf_k: usize,
    include_original: bool,
}

impl FusionRetrieverBuilder {
    /// Set the query generator.
    pub fn query_generator(mut self, gen: Arc<dyn QueryGenerator>) -> Self {
        self.query_generator = gen;
        self
    }

    /// Set the maximum number of documents to return.
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the RRF constant (default: 60).
    pub fn rrf_k(mut self, rrf_k: usize) -> Self {
        self.rrf_k = rrf_k;
        self
    }

    /// Set whether to include the original query (default: true).
    pub fn include_original(mut self, include: bool) -> Self {
        self.include_original = include;
        self
    }

    /// Build the `FusionRetriever`.
    pub fn build(self) -> FusionRetriever {
        FusionRetriever {
            inner: self.inner,
            query_generator: self.query_generator,
            k: self.k,
            rrf_k: self.rrf_k,
            include_original: self.include_original,
        }
    }
}

#[async_trait]
impl BaseRetriever for FusionRetriever {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
        // Generate query variations
        let mut queries = self.query_generator.generate_queries(query).await?;

        if self.include_original {
            queries.insert(0, query.to_string());
        }

        // Retrieve documents for all queries concurrently
        let futures: Vec<_> = queries
            .iter()
            .map(|q| self.inner.get_relevant_documents(q))
            .collect();

        let all_results = join_all(futures).await;

        // Collect results, propagating errors
        let mut result_sets = Vec::with_capacity(all_results.len());
        for result in all_results {
            result_sets.push(result?);
        }

        // Apply RRF scoring
        let scored = Self::reciprocal_rank_fusion(&result_sets, self.rrf_k);

        Ok(scored
            .into_iter()
            .take(self.k)
            .map(|(doc, _)| doc)
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::error::RustChainError;

    /// Mock retriever that returns fixed documents.
    struct MockRetriever {
        docs: Vec<Document>,
    }

    impl MockRetriever {
        fn new(contents: &[&str]) -> Self {
            Self {
                docs: contents.iter().map(|c| Document::new(*c)).collect(),
            }
        }

        fn with_ids(pairs: &[(&str, &str)]) -> Self {
            Self {
                docs: pairs
                    .iter()
                    .map(|(id, content)| Document::new(*content).with_id(*id))
                    .collect(),
            }
        }
    }

    #[async_trait]
    impl BaseRetriever for MockRetriever {
        async fn get_relevant_documents(&self, _query: &str) -> Result<Vec<Document>> {
            Ok(self.docs.clone())
        }
    }

    /// Mock retriever that echoes the query as a document.
    struct EchoRetriever;

    #[async_trait]
    impl BaseRetriever for EchoRetriever {
        async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
            Ok(vec![Document::new(format!("result for: {}", query))])
        }
    }

    /// Mock retriever that always fails.
    struct FailingRetriever;

    #[async_trait]
    impl BaseRetriever for FailingRetriever {
        async fn get_relevant_documents(&self, _query: &str) -> Result<Vec<Document>> {
            Err(RustChainError::Other("retriever failed".into()))
        }
    }

    /// Mock query generator with fixed output.
    struct FixedQueryGenerator {
        queries: Vec<String>,
    }

    impl FixedQueryGenerator {
        fn new(queries: Vec<&str>) -> Self {
            Self {
                queries: queries.into_iter().map(String::from).collect(),
            }
        }
    }

    #[async_trait]
    impl QueryGenerator for FixedQueryGenerator {
        async fn generate_queries(&self, _query: &str) -> Result<Vec<String>> {
            Ok(self.queries.clone())
        }
    }

    // -----------------------------------------------------------------------
    // SimpleQueryGenerator tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_simple_generator_produces_correct_count() {
        let gen = SimpleQueryGenerator::new(3);
        let queries = gen.generate_queries("What is Rust?").await.unwrap();
        assert_eq!(queries.len(), 3);
    }

    #[tokio::test]
    async fn test_simple_generator_what_is_variations() {
        let gen = SimpleQueryGenerator::new(10);
        let queries = gen.generate_queries("What is machine learning?").await.unwrap();
        // Should include perspective changes like "Define machine learning"
        assert!(
            queries.iter().any(|q| q.contains("Define")),
            "Expected a 'Define' variation, got: {:?}",
            queries
        );
        assert!(
            queries.iter().any(|q| q.contains("Explain")),
            "Expected an 'Explain' variation, got: {:?}",
            queries
        );
    }

    #[tokio::test]
    async fn test_simple_generator_excludes_original() {
        let gen = SimpleQueryGenerator::new(10);
        let original = "What is Rust?";
        let queries = gen.generate_queries(original).await.unwrap();
        // None of the variations should be identical to the original (case-insensitive)
        let original_lower = original.to_lowercase();
        for q in &queries {
            assert_ne!(q.to_lowercase(), original_lower);
        }
    }

    #[tokio::test]
    async fn test_simple_generator_deduplicates() {
        let gen = SimpleQueryGenerator::new(20);
        let queries = gen.generate_queries("test query").await.unwrap();
        let unique: HashSet<String> = queries.iter().map(|q| q.to_lowercase()).collect();
        assert_eq!(unique.len(), queries.len(), "Variations should be unique");
    }

    #[tokio::test]
    async fn test_simple_generator_why_variations() {
        let gen = SimpleQueryGenerator::new(10);
        let queries = gen.generate_queries("Why is the sky blue?").await.unwrap();
        assert!(
            queries.iter().any(|q| q.contains("Reason for")),
            "Expected a 'Reason for' variation, got: {:?}",
            queries
        );
    }

    // -----------------------------------------------------------------------
    // TemplateQueryGenerator tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_template_generator_basic() {
        let gen = TemplateQueryGenerator::new(vec![
            "Rephrase: {query}".to_string(),
            "Summarize: {query}".to_string(),
        ]);
        let queries = gen.generate_queries("What is Rust?").await.unwrap();
        assert_eq!(queries.len(), 2);
        assert_eq!(queries[0], "Rephrase: What is Rust?");
        assert_eq!(queries[1], "Summarize: What is Rust?");
    }

    #[tokio::test]
    async fn test_template_generator_multiple_placeholders() {
        let gen = TemplateQueryGenerator::new(vec![
            "Given '{query}', find documents about {query}".to_string(),
        ]);
        let queries = gen.generate_queries("neural networks").await.unwrap();
        assert_eq!(
            queries[0],
            "Given 'neural networks', find documents about neural networks"
        );
    }

    #[tokio::test]
    async fn test_template_generator_empty_templates() {
        let gen = TemplateQueryGenerator::new(vec![]);
        let queries = gen.generate_queries("test").await.unwrap();
        assert!(queries.is_empty());
    }

    // -----------------------------------------------------------------------
    // MultiQueryRetriever tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_multi_query_retriever_basic() {
        let base = Arc::new(MockRetriever::new(&["doc1", "doc2", "doc3"]));
        let gen = Arc::new(FixedQueryGenerator::new(vec!["q1", "q2"]));
        let retriever = MultiQueryRetriever::builder(base)
            .query_generator(gen)
            .k(10)
            .build();
        let docs = retriever.get_relevant_documents("original").await.unwrap();
        // 3 queries (original + 2 generated) * 3 docs each = 9, but dedup reduces to 3
        assert_eq!(docs.len(), 3);
    }

    #[tokio::test]
    async fn test_multi_query_retriever_with_echo() {
        let base: Arc<dyn BaseRetriever> = Arc::new(EchoRetriever);
        let gen = Arc::new(FixedQueryGenerator::new(vec!["alt query"]));
        let retriever = MultiQueryRetriever::builder(base)
            .query_generator(gen)
            .k(10)
            .build();
        let docs = retriever.get_relevant_documents("main query").await.unwrap();
        // original + 1 generated = 2 unique docs
        assert_eq!(docs.len(), 2);
        assert!(docs.iter().any(|d| d.page_content == "result for: main query"));
        assert!(docs.iter().any(|d| d.page_content == "result for: alt query"));
    }

    #[tokio::test]
    async fn test_multi_query_retriever_k_limits_output() {
        let base = Arc::new(MockRetriever::new(&["a", "b", "c", "d", "e"]));
        let gen = Arc::new(FixedQueryGenerator::new(vec![]));
        let retriever = MultiQueryRetriever::builder(base)
            .query_generator(gen)
            .k(2)
            .build();
        let docs = retriever.get_relevant_documents("q").await.unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[tokio::test]
    async fn test_multi_query_retriever_dedup_by_id() {
        let base = Arc::new(MockRetriever::with_ids(&[
            ("id1", "content A"),
            ("id2", "content B"),
        ]));
        let gen = Arc::new(FixedQueryGenerator::new(vec!["q1"]));
        let retriever = MultiQueryRetriever::builder(base)
            .query_generator(gen)
            .k(10)
            .build();
        let docs = retriever.get_relevant_documents("q").await.unwrap();
        // Same IDs from 2 queries -> deduplicated to 2
        assert_eq!(docs.len(), 2);
    }

    #[tokio::test]
    async fn test_multi_query_retriever_no_dedup() {
        let base = Arc::new(MockRetriever::new(&["doc1"]));
        let gen = Arc::new(FixedQueryGenerator::new(vec!["q1"]));
        let retriever = MultiQueryRetriever::builder(base)
            .query_generator(gen)
            .k(10)
            .deduplicate(false)
            .build();
        let docs = retriever.get_relevant_documents("q").await.unwrap();
        // 2 queries * 1 doc = 2 (no dedup)
        assert_eq!(docs.len(), 2);
    }

    #[tokio::test]
    async fn test_multi_query_retriever_exclude_original() {
        let base: Arc<dyn BaseRetriever> = Arc::new(EchoRetriever);
        let gen = Arc::new(FixedQueryGenerator::new(vec!["alt"]));
        let retriever = MultiQueryRetriever::builder(base)
            .query_generator(gen)
            .k(10)
            .include_original(false)
            .build();
        let docs = retriever.get_relevant_documents("original").await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "result for: alt");
    }

    #[tokio::test]
    async fn test_multi_query_retriever_propagates_error() {
        let base: Arc<dyn BaseRetriever> = Arc::new(FailingRetriever);
        let gen = Arc::new(FixedQueryGenerator::new(vec![]));
        let retriever = MultiQueryRetriever::builder(base)
            .query_generator(gen)
            .k(10)
            .build();
        let result = retriever.get_relevant_documents("q").await;
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // FusionRetriever tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_fusion_retriever_basic() {
        let base = Arc::new(MockRetriever::new(&["doc1", "doc2"]));
        let gen = Arc::new(FixedQueryGenerator::new(vec!["q1"]));
        let retriever = FusionRetriever::builder(base)
            .query_generator(gen)
            .k(10)
            .build();
        let docs = retriever.get_relevant_documents("q").await.unwrap();
        // doc1 and doc2 appear in both query results -> deduplicated by RRF
        assert_eq!(docs.len(), 2);
    }

    #[tokio::test]
    async fn test_fusion_retriever_ranks_by_frequency() {
        // EchoRetriever returns unique docs per query, so all have equal RRF score.
        // Use a mock that returns overlapping results to test ranking.
        let base = Arc::new(MockRetriever::new(&["common", "unique_to_each"]));
        let gen = Arc::new(FixedQueryGenerator::new(vec!["q1", "q2"]));
        let retriever = FusionRetriever::builder(base)
            .query_generator(gen)
            .k(10)
            .build();
        let docs = retriever.get_relevant_documents("q").await.unwrap();
        // "common" appears at rank 1 in all 3 results -> highest score
        assert_eq!(docs[0].page_content, "common");
    }

    #[tokio::test]
    async fn test_fusion_retriever_k_limits_output() {
        let base = Arc::new(MockRetriever::new(&["a", "b", "c", "d", "e"]));
        let gen = Arc::new(FixedQueryGenerator::new(vec![]));
        let retriever = FusionRetriever::builder(base)
            .query_generator(gen)
            .k(3)
            .build();
        let docs = retriever.get_relevant_documents("q").await.unwrap();
        assert_eq!(docs.len(), 3);
    }

    #[tokio::test]
    async fn test_fusion_retriever_custom_rrf_k() {
        let base = Arc::new(MockRetriever::new(&["doc1"]));
        let gen = Arc::new(FixedQueryGenerator::new(vec![]));
        let retriever = FusionRetriever::builder(base)
            .query_generator(gen)
            .rrf_k(10)
            .k(10)
            .build();
        let docs = retriever.get_relevant_documents("q").await.unwrap();
        assert_eq!(docs.len(), 1);
    }

    #[tokio::test]
    async fn test_fusion_retriever_propagates_error() {
        let base: Arc<dyn BaseRetriever> = Arc::new(FailingRetriever);
        let gen = Arc::new(FixedQueryGenerator::new(vec![]));
        let retriever = FusionRetriever::builder(base)
            .query_generator(gen)
            .k(10)
            .build();
        let result = retriever.get_relevant_documents("q").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_fusion_rrf_scoring_correctness() {
        // Directly test the RRF scoring function
        let docs_a = vec![Document::new("X"), Document::new("Y")];
        let docs_b = vec![Document::new("Y"), Document::new("Z")];
        let result_sets = vec![docs_a, docs_b];

        let scored = FusionRetriever::reciprocal_rank_fusion(&result_sets, 60);

        // Y appears in both: rank 2 in set A -> 1/(60+2) and rank 1 in set B -> 1/(60+1)
        // X appears once: rank 1 in set A -> 1/(60+1)
        // Z appears once: rank 2 in set B -> 1/(60+2)
        assert_eq!(scored[0].0.page_content, "Y");
        assert_eq!(scored.len(), 3);

        // Verify Y's score > X's score
        let y_score = scored.iter().find(|(d, _)| d.page_content == "Y").unwrap().1;
        let x_score = scored.iter().find(|(d, _)| d.page_content == "X").unwrap().1;
        assert!(y_score > x_score);
    }

    #[tokio::test]
    async fn test_fusion_retriever_exclude_original() {
        let base: Arc<dyn BaseRetriever> = Arc::new(EchoRetriever);
        let gen = Arc::new(FixedQueryGenerator::new(vec!["alt"]));
        let retriever = FusionRetriever::builder(base)
            .query_generator(gen)
            .k(10)
            .include_original(false)
            .build();
        let docs = retriever.get_relevant_documents("original").await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "result for: alt");
    }

    // -----------------------------------------------------------------------
    // Deduplication tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_dedup_by_content() {
        let docs = vec![
            Document::new("same content"),
            Document::new("same content"),
            Document::new("different"),
        ];
        let deduped = MultiQueryRetriever::deduplicate_docs(docs);
        assert_eq!(deduped.len(), 2);
    }

    #[tokio::test]
    async fn test_dedup_by_id_takes_priority() {
        let docs = vec![
            Document::new("content A").with_id("id1"),
            Document::new("content B").with_id("id1"), // same ID, different content
            Document::new("content C").with_id("id2"),
        ];
        let deduped = MultiQueryRetriever::deduplicate_docs(docs);
        // id1 appears twice but dedup by ID keeps only the first
        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[0].page_content, "content A");
        assert_eq!(deduped[1].page_content, "content C");
    }
}
