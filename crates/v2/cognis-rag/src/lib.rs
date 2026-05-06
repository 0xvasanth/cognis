//! # cognis2-rag
//!
//! v2-beta embeddings + vector store + (slice 3) RAG primitives.
//! Slice 2b ships the foundation: Embeddings trait + 3 impls,
//! VectorStore trait + InMemoryVectorStore. Slice 3 will add document
//! loaders, text splitters, retrievers, and query transformers.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod distance;
pub mod embeddings;
pub mod vectorstore;

pub use distance::Distance;
#[cfg(feature = "ollama")]
pub use embeddings::OllamaEmbeddings;
#[cfg(feature = "openai")]
pub use embeddings::OpenAIEmbeddings;
pub use embeddings::{Embeddings, FakeEmbeddings};
pub use vectorstore::{InMemoryVectorStore, SearchResult, VectorStore};

/// Common imports for v2 RAG user code.
pub mod prelude {
    pub use crate::{Distance, Embeddings, InMemoryVectorStore, SearchResult, VectorStore};
    pub use cognis2_core::prelude::*;
}
