//! Vector store trait + SearchResult + Filter.

use std::collections::HashMap;

use async_trait::async_trait;
use cognis_core::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};

use cognis_core::Result;

mod in_memory;
pub use in_memory::InMemoryVectorStore;

#[cfg(feature = "vectorstore-chroma")]
pub mod chroma;
#[cfg(feature = "vectorstore-chroma")]
pub use chroma::{ChromaBuilder, ChromaProvider};

#[cfg(feature = "vectorstore-qdrant")]
pub mod qdrant;
#[cfg(feature = "vectorstore-qdrant")]
pub use qdrant::{QdrantBuilder, QdrantProvider};

#[cfg(feature = "vectorstore-pinecone")]
pub mod pinecone;
#[cfg(feature = "vectorstore-pinecone")]
pub use pinecone::{PineconeBuilder, PineconeProvider};

#[cfg(feature = "vectorstore-weaviate")]
pub mod weaviate;
#[cfg(feature = "vectorstore-weaviate")]
pub use weaviate::{WeaviateBuilder, WeaviateProvider};

/// Metadata filter applied to similarity search.
///
/// All conditions on a filter combine with AND semantics. An empty filter
/// matches everything (the same as no filter at all).
#[derive(Debug, Default, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Filter {
    /// Metadata key/value pairs every result must match exactly.
    /// Compared with serde_json `==` after both sides are coerced to
    /// `serde_json::Value`.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub equals: HashMap<String, serde_json::Value>,
    /// Metadata keys whose values must appear in the listed set.
    /// Equivalent to `metadata[key] in values`.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub r#in: HashMap<String, Vec<serde_json::Value>>,
    /// Numeric metadata keys with `>=` lower bound.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub gte: HashMap<String, f64>,
    /// Numeric metadata keys with `<=` upper bound.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub lte: HashMap<String, f64>,
}

impl Filter {
    /// Empty filter (matches everything).
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: require `metadata[key] == value`.
    pub fn equals(mut self, key: impl Into<String>, value: impl Into<serde_json::Value>) -> Self {
        self.equals.insert(key.into(), value.into());
        self
    }

    /// Builder: require `metadata[key]` to be one of `values`.
    pub fn one_of<I, V>(mut self, key: impl Into<String>, values: I) -> Self
    where
        I: IntoIterator<Item = V>,
        V: Into<serde_json::Value>,
    {
        self.r#in
            .insert(key.into(), values.into_iter().map(Into::into).collect());
        self
    }

    /// Builder: require `metadata[key] >= n` (numeric).
    pub fn gte(mut self, key: impl Into<String>, n: f64) -> Self {
        self.gte.insert(key.into(), n);
        self
    }

    /// Builder: require `metadata[key] <= n` (numeric).
    pub fn lte(mut self, key: impl Into<String>, n: f64) -> Self {
        self.lte.insert(key.into(), n);
        self
    }

    /// True if no conditions are set.
    pub fn is_empty(&self) -> bool {
        self.equals.is_empty() && self.r#in.is_empty() && self.gte.is_empty() && self.lte.is_empty()
    }

    /// Whether `metadata` satisfies every condition on this filter.
    pub fn matches(&self, metadata: &HashMap<String, serde_json::Value>) -> bool {
        for (k, v) in &self.equals {
            match metadata.get(k) {
                Some(actual) if actual == v => {}
                _ => return false,
            }
        }
        for (k, allowed) in &self.r#in {
            match metadata.get(k) {
                Some(actual) if allowed.iter().any(|v| v == actual) => {}
                _ => return false,
            }
        }
        for (k, lo) in &self.gte {
            match metadata.get(k).and_then(|v| v.as_f64()) {
                Some(n) if n >= *lo => {}
                _ => return false,
            }
        }
        for (k, hi) in &self.lte {
            match metadata.get(k).and_then(|v| v.as_f64()) {
                Some(n) if n <= *hi => {}
                _ => return false,
            }
        }
        true
    }
}

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

    /// Similarity search with a metadata filter.
    ///
    /// Default impl runs `similarity_search` with `k * 4` candidates and
    /// post-filters in the caller. Stores with native filter support
    /// (Qdrant, Pinecone, ...) override for efficiency.
    async fn similarity_search_with_filter(
        &self,
        query: &str,
        k: usize,
        filter: &Filter,
    ) -> Result<Vec<SearchResult>> {
        if filter.is_empty() {
            return self.similarity_search(query, k).await;
        }
        let candidates = self.similarity_search(query, k.saturating_mul(4)).await?;
        Ok(candidates
            .into_iter()
            .filter(|r| filter.matches(&r.metadata))
            .take(k)
            .collect())
    }

    /// Delete documents by ID. IDs not found are silently ignored.
    async fn delete(&mut self, ids: Vec<String>) -> Result<()>;

    /// Number of documents currently stored.
    fn len(&self) -> usize;

    /// True if no documents are stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
