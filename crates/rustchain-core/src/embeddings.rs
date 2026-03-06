use async_trait::async_trait;

use crate::error::Result;

/// Interface for embedding models.
#[async_trait]
pub trait Embeddings: Send + Sync {
    /// Embed a list of documents.
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;

    /// Embed a single query text.
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>>;
}
