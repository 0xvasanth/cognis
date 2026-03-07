//! Retriever implementations that compose and extend `BaseRetriever`.
//!
//! - [`ensemble`] -- Ensemble retriever using Reciprocal Rank Fusion.
//! - [`docstore`] -- In-memory document store for parent-document and multi-vector retrievers.
//! - [`parent_document`] -- Parent-document retriever that indexes child chunks but returns full parents.
//! - [`multi_vector`] -- Multi-vector retriever that searches summaries but returns originals.
//! - [`contextual_compression`] -- Contextual compression retriever with LLM and embeddings compressors.

pub mod compressor_pipeline;
pub mod contextual_compression;
pub mod docstore;
pub mod ensemble;
pub mod multi_vector;
pub mod parent_document;
pub mod query_translator;
pub mod self_query;
