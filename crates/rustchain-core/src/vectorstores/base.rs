use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::documents::Document;
use crate::error::Result;
use crate::retrievers::BaseRetriever;

/// The type of search to perform against a vector store.
#[derive(Debug, Clone, Default)]
pub enum SearchType {
    /// Standard similarity search returning the most similar documents.
    #[default]
    Similarity,
    /// Similarity search with a minimum score threshold.
    SimilarityScoreThreshold {
        score_threshold: f32,
    },
    /// Maximal Marginal Relevance search for diverse results.
    Mmr {
        fetch_k: usize,
        lambda_mult: f32,
    },
}


/// Trait defining the interface for vector store implementations.
///
/// This trait mirrors the Python `VectorStore` ABC, providing methods for
/// adding, deleting, and searching documents by text or embedding vector.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Add raw texts with optional metadata, returning their assigned IDs.
    ///
    /// Implementations should embed the texts and store them along with the
    /// provided metadata. If `metadatas` is `Some`, its length must equal
    /// the length of `texts`.
    async fn add_texts(
        &self,
        texts: &[String],
        metadatas: Option<&[HashMap<String, Value>]>,
        ids: Option<&[String]>,
    ) -> Result<Vec<String>>;

    /// Add `Document` objects to the vector store, returning their IDs.
    ///
    /// This is a convenience method that extracts text and metadata from
    /// documents and delegates to the underlying storage.
    async fn add_documents(
        &self,
        documents: Vec<Document>,
        ids: Option<Vec<String>>,
    ) -> Result<Vec<String>>;

    /// Delete documents by their IDs.
    ///
    /// Returns `true` if the deletion was successful.
    async fn delete(&self, ids: Option<&[String]>) -> Result<bool>;

    /// Retrieve documents by their IDs.
    ///
    /// Documents that are not found are silently skipped.
    async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Document>>;

    /// Search for documents similar to the query string.
    ///
    /// Returns the top `k` most similar documents.
    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<Document>>;

    /// Search for documents similar to the query string, returning scores.
    ///
    /// Returns tuples of `(Document, score)` sorted by descending similarity.
    async fn similarity_search_with_score(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<(Document, f32)>>;

    /// Search for documents similar to the given embedding vector.
    ///
    /// Returns the top `k` most similar documents.
    async fn similarity_search_by_vector(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<Document>>;

    /// Perform Maximal Marginal Relevance (MMR) search.
    ///
    /// MMR balances relevance to the query with diversity among results.
    ///
    /// # Arguments
    ///
    /// * `query` - The search query text.
    /// * `k` - Number of documents to return.
    /// * `fetch_k` - Number of candidates to fetch before applying MMR.
    /// * `lambda_mult` - Trade-off between relevance (1.0) and diversity (0.0).
    async fn max_marginal_relevance_search(
        &self,
        query: &str,
        k: usize,
        fetch_k: usize,
        lambda_mult: f32,
    ) -> Result<Vec<Document>>;
}

/// A retriever backed by a `VectorStore`.
///
/// Wraps a vector store and delegates retrieval to the configured search type.
pub struct VectorStoreRetriever {
    vectorstore: Arc<dyn VectorStore>,
    search_type: SearchType,
    k: usize,
}

impl VectorStoreRetriever {
    /// Create a new retriever wrapping the given vector store.
    pub fn new(vectorstore: Arc<dyn VectorStore>, search_type: SearchType, k: usize) -> Self {
        Self {
            vectorstore,
            search_type,
            k,
        }
    }

    /// Create a retriever with default settings (similarity search, k=4).
    pub fn from_vectorstore(vectorstore: Arc<dyn VectorStore>) -> Self {
        Self {
            vectorstore,
            search_type: SearchType::Similarity,
            k: 4,
        }
    }
}

#[async_trait]
impl BaseRetriever for VectorStoreRetriever {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
        match &self.search_type {
            SearchType::Similarity => {
                self.vectorstore.similarity_search(query, self.k).await
            }
            SearchType::SimilarityScoreThreshold { score_threshold } => {
                let threshold = *score_threshold;
                let results = self
                    .vectorstore
                    .similarity_search_with_score(query, self.k)
                    .await?;
                Ok(results
                    .into_iter()
                    .filter(|(_, score)| *score >= threshold)
                    .map(|(doc, _)| doc)
                    .collect())
            }
            SearchType::Mmr {
                fetch_k,
                lambda_mult,
            } => {
                self.vectorstore
                    .max_marginal_relevance_search(
                        query,
                        self.k,
                        *fetch_k,
                        *lambda_mult,
                    )
                    .await
            }
        }
    }
}

/// Compute cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Convert euclidean distance to a relevance score in [0, 1].
pub fn euclidean_relevance_score(distance: f32) -> f32 {
    1.0 - distance / 2f32.sqrt()
}

/// Convert cosine distance to a relevance score in [0, 1].
pub fn cosine_relevance_score(distance: f32) -> f32 {
    1.0 - distance
}

/// Convert max inner product distance to a relevance score.
pub fn max_inner_product_relevance_score(distance: f32) -> f32 {
    if distance > 0.0 {
        1.0 - distance
    } else {
        -distance
    }
}
