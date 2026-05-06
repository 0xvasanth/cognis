//! Vector store trait + SearchResult.

use std::collections::HashMap;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use cognis2_core::Result;

mod in_memory;
pub use in_memory::InMemoryVectorStore;

/// One document returned by a similarity search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    /// Document ID assigned by the store.
    pub id: String,
    /// Original text content.
    pub text: String,
    /// Similarity score — higher = more similar (per the store's
    /// configured Distance metric).
    pub score: f32,
    /// User-supplied metadata stored with the document.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// A vector store: holds documents + their embeddings, supports
/// add + similarity search + delete.
#[async_trait]
pub trait VectorStore: Send + Sync {
    /// Add documents (text + optional metadata). The store is responsible
    /// for embedding them. Returns the IDs assigned.
    async fn add_texts(
        &mut self,
        texts: Vec<String>,
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>>;

    /// Add pre-embedded vectors directly. Useful when the caller has
    /// already paid the embedding cost.
    async fn add_vectors(
        &mut self,
        vectors: Vec<Vec<f32>>,
        texts: Vec<String>,
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>>;

    /// Similarity search: embed the query, return top-k matches.
    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>>;

    /// Similarity search by pre-computed query vector.
    async fn similarity_search_by_vector(
        &self,
        query_vector: Vec<f32>,
        k: usize,
    ) -> Result<Vec<SearchResult>>;

    /// Delete documents by ID. IDs not found are silently ignored.
    async fn delete(&mut self, ids: Vec<String>) -> Result<()>;

    /// Number of documents currently stored.
    fn len(&self) -> usize;

    /// True if no documents are stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
