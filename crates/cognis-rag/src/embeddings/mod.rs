//! Embeddings trait + implementations.

use async_trait::async_trait;

use cognis_core::{CognisError, Result};

mod fake;
pub use fake::FakeEmbeddings;

mod batched;
mod cached;
mod router;
pub use batched::BatchedEmbeddings;
pub use cached::CachedEmbeddings;
pub use router::{EmbeddingRouter, EmbeddingsRouter, FnRouter, LengthRouter};

#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "openai")]
pub use openai::OpenAIEmbeddings;

#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "ollama")]
pub use ollama::OllamaEmbeddings;

/// A model that converts text into vector embeddings. Implementations
/// MUST return vectors of consistent dimensionality across calls — the
/// `dimensions()` accessor (when known) reports it.
#[async_trait]
pub trait Embeddings: Send + Sync {
    /// Embed a batch of documents. Order of returned vectors matches
    /// the input order.
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>>;

    /// Embed a single query string. Default impl wraps `embed_documents`
    /// with a single-element batch.
    async fn embed_query(&self, text: String) -> Result<Vec<f32>> {
        let mut vecs = self.embed_documents(vec![text]).await?;
        vecs.pop()
            .ok_or_else(|| CognisError::Internal("embed_documents returned empty vec".into()))
    }

    /// Dimensionality of the embedding vectors, if known statically.
    fn dimensions(&self) -> Option<usize> {
        None
    }

    /// Model identifier (e.g. "text-embedding-3-small", "nomic-embed-text").
    fn model(&self) -> &str;
}
