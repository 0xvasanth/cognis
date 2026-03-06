//! Ensemble retriever that combines results from multiple retrievers using
//! Reciprocal Rank Fusion (RRF).
//!
//! RRF is a simple yet effective method for combining ranked lists from
//! different sources. Each document receives a score of `weight / (rrf_k + rank)`
//! where `rank` is 1-indexed, and scores are summed across all retrievers.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use rustchain_core::retrievers::BaseRetriever;

/// A retriever that combines results from multiple retrievers using
/// Reciprocal Rank Fusion (RRF).
///
/// # Example
///
/// ```rust,ignore
/// use rustchain::retrievers::ensemble::EnsembleRetriever;
/// use std::sync::Arc;
///
/// let ensemble = EnsembleRetriever::new(vec![retriever_a, retriever_b])
///     .with_weights(vec![0.7, 0.3])
///     .with_k(5);
///
/// let docs = ensemble.get_relevant_documents("query").await?;
/// ```
pub struct EnsembleRetriever {
    /// The retrievers to combine.
    retrievers: Vec<Arc<dyn BaseRetriever>>,
    /// Weight for each retriever. Must have the same length as `retrievers`.
    weights: Vec<f32>,
    /// Number of documents to return (default: 4).
    k: usize,
    /// RRF constant (default: 60). Higher values reduce the impact of rank position.
    rrf_k: usize,
}

impl EnsembleRetriever {
    /// Create a new `EnsembleRetriever` with equal weights for each retriever.
    pub fn new(retrievers: Vec<Arc<dyn BaseRetriever>>) -> Self {
        let n = retrievers.len();
        let weight = if n > 0 { 1.0 / n as f32 } else { 0.0 };
        let weights = vec![weight; n];
        Self {
            retrievers,
            weights,
            k: 4,
            rrf_k: 60,
        }
    }

    /// Set custom weights for each retriever.
    ///
    /// # Panics
    ///
    /// Panics if `weights.len() != retrievers.len()`.
    pub fn with_weights(mut self, weights: Vec<f32>) -> Self {
        assert_eq!(
            weights.len(),
            self.retrievers.len(),
            "weights length must match retrievers length"
        );
        self.weights = weights;
        self
    }

    /// Set the number of documents to return.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the RRF constant.
    pub fn with_rrf_k(mut self, rrf_k: usize) -> Self {
        self.rrf_k = rrf_k;
        self
    }
}

#[async_trait]
impl BaseRetriever for EnsembleRetriever {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
        // Query all retrievers concurrently.
        let futures: Vec<_> = self
            .retrievers
            .iter()
            .map(|r| r.get_relevant_documents(query))
            .collect();

        let all_results = join_all(futures).await;

        // Collect successful results, propagate first error if any retriever fails.
        let mut result_sets: Vec<Vec<Document>> = Vec::with_capacity(all_results.len());
        for result in all_results {
            result_sets.push(result?);
        }

        let scored = reciprocal_rank_fusion(&result_sets, &self.weights, self.rrf_k);

        Ok(scored
            .into_iter()
            .take(self.k)
            .map(|(doc, _)| doc)
            .collect())
    }
}

/// Compute Reciprocal Rank Fusion scores across multiple ranked result lists.
///
/// For each retriever `i`, every document at rank `r` (1-indexed) receives a
/// score of `weights[i] / (rrf_k + r)`. Scores are summed for documents that
/// appear in multiple result lists (matched by `page_content`).
///
/// Returns documents sorted by descending aggregate score.
pub fn reciprocal_rank_fusion(
    results: &[Vec<Document>],
    weights: &[f32],
    rrf_k: usize,
) -> Vec<(Document, f32)> {
    // Map from page_content -> (best Document copy, aggregate score)
    let mut score_map: HashMap<String, (Document, f32)> = HashMap::new();

    for (i, docs) in results.iter().enumerate() {
        let weight = weights.get(i).copied().unwrap_or(1.0);
        for (rank, doc) in docs.iter().enumerate() {
            let score = weight / (rrf_k as f32 + (rank + 1) as f32);
            let entry = score_map
                .entry(doc.page_content.clone())
                .or_insert_with(|| (doc.clone(), 0.0));
            entry.1 += score;
        }
    }

    let mut scored: Vec<(Document, f32)> = score_map.into_values().collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A mock retriever that returns a fixed list of documents.
    struct MockRetriever {
        docs: Vec<Document>,
    }

    impl MockRetriever {
        fn new(contents: &[&str]) -> Self {
            Self {
                docs: contents.iter().map(|c| Document::new(*c)).collect(),
            }
        }
    }

    #[async_trait]
    impl BaseRetriever for MockRetriever {
        async fn get_relevant_documents(&self, _query: &str) -> Result<Vec<Document>> {
            Ok(self.docs.clone())
        }
    }

    /// A mock retriever that always returns an empty list.
    struct EmptyRetriever;

    #[async_trait]
    impl BaseRetriever for EmptyRetriever {
        async fn get_relevant_documents(&self, _query: &str) -> Result<Vec<Document>> {
            Ok(vec![])
        }
    }

    #[tokio::test]
    async fn test_single_retriever_returns_its_results() {
        let r = Arc::new(MockRetriever::new(&["doc1", "doc2", "doc3"]));
        let ensemble = EnsembleRetriever::new(vec![r]).with_k(10);
        let docs = ensemble.get_relevant_documents("query").await.unwrap();

        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].page_content, "doc1");
        assert_eq!(docs[1].page_content, "doc2");
        assert_eq!(docs[2].page_content, "doc3");
    }

    #[tokio::test]
    async fn test_two_retrievers_combine_via_rrf() {
        let r1 = Arc::new(MockRetriever::new(&["A", "B", "C"]));
        let r2 = Arc::new(MockRetriever::new(&["D", "A", "E"]));
        let ensemble = EnsembleRetriever::new(vec![r1, r2]).with_k(10);
        let docs = ensemble.get_relevant_documents("query").await.unwrap();

        // "A" appears in both retrievers, so it should have the highest score.
        assert_eq!(docs[0].page_content, "A");
        // All 5 unique documents should be present.
        assert_eq!(docs.len(), 5);
    }

    #[tokio::test]
    async fn test_weighted_retrievers_favor_higher_weight() {
        // r1 has "X" at rank 1, r2 has "Y" at rank 1.
        // Give r2 a much higher weight so "Y" should come first.
        let r1 = Arc::new(MockRetriever::new(&["X"]));
        let r2 = Arc::new(MockRetriever::new(&["Y"]));
        let ensemble = EnsembleRetriever::new(vec![r1, r2])
            .with_weights(vec![0.1, 0.9])
            .with_k(10);
        let docs = ensemble.get_relevant_documents("query").await.unwrap();

        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].page_content, "Y");
        assert_eq!(docs[1].page_content, "X");
    }

    #[tokio::test]
    async fn test_deduplication_across_retrievers() {
        // Both retrievers return "shared" but it should appear only once.
        let r1 = Arc::new(MockRetriever::new(&["shared", "only_r1"]));
        let r2 = Arc::new(MockRetriever::new(&["shared", "only_r2"]));
        let ensemble = EnsembleRetriever::new(vec![r1, r2]).with_k(10);
        let docs = ensemble.get_relevant_documents("query").await.unwrap();

        let contents: Vec<&str> = docs.iter().map(|d| d.page_content.as_str()).collect();
        // "shared" appears exactly once despite being in both retrievers.
        assert_eq!(contents.iter().filter(|&&c| c == "shared").count(), 1);
        assert_eq!(docs.len(), 3); // shared, only_r1, only_r2
    }

    #[tokio::test]
    async fn test_custom_k_limits_output() {
        let r = Arc::new(MockRetriever::new(&["a", "b", "c", "d", "e"]));
        let ensemble = EnsembleRetriever::new(vec![r]).with_k(2);
        let docs = ensemble.get_relevant_documents("query").await.unwrap();

        assert_eq!(docs.len(), 2);
    }

    #[tokio::test]
    async fn test_empty_results_from_one_retriever() {
        let r1 = Arc::new(MockRetriever::new(&["real_doc"]));
        let r2: Arc<dyn BaseRetriever> = Arc::new(EmptyRetriever);
        let ensemble = EnsembleRetriever::new(vec![r1, r2]).with_k(10);
        let docs = ensemble.get_relevant_documents("query").await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "real_doc");
    }

    #[tokio::test]
    async fn test_rrf_scoring_standalone() {
        let docs_a = vec![Document::new("X"), Document::new("Y")];
        let docs_b = vec![Document::new("Y"), Document::new("Z")];
        let results = vec![docs_a, docs_b];
        let weights = vec![0.5, 0.5];

        let scored = reciprocal_rank_fusion(&results, &weights, 60);

        // "Y" appears in both at ranks 2 and 1 respectively.
        assert_eq!(scored[0].0.page_content, "Y");
        assert_eq!(scored.len(), 3);
    }
}
