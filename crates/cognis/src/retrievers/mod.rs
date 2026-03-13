//! Retriever implementations that compose and extend `BaseRetriever`.
//!
//! - [`ensemble`] -- Ensemble retriever using Reciprocal Rank Fusion.
//! - [`docstore`] -- In-memory document store for parent-document and multi-vector retrievers.
//! - [`parent_document`] -- Parent-document retriever that indexes child chunks but returns full parents.
//! - [`multi_vector`] -- Multi-vector retriever that searches summaries but returns originals.
//! - [`contextual_compression`] -- Contextual compression retriever with LLM and embeddings compressors.
//! - [`compressor_pipeline`] -- Pipeline of document compressors applied in sequence.
//! - [`multi_query`] -- Multi-query retriever with query variation generation and reciprocal rank fusion.
//! - [`self_query`] -- Self-querying retriever that translates natural language into structured filters.
//! - [`caching`] -- Caching retriever that avoids redundant lookups with TTL and LRU eviction.
//! - [`time_weighted`] -- Time-weighted retriever scoring documents by recency + relevance.
//! - [`query_translator`] -- Structured query translator with rule-based (`RuleBasedTranslator`) and LLM-powered (`TranslatorChain`) strategies for vector store metadata filtering.
//! - [`reranking`] -- Reranking retrievers with keyword, TF-IDF, cross-encoder, and cascade support.

pub mod caching;
pub mod compression;
pub mod compressor_pipeline;
pub mod contextual_compression;
pub mod docstore;
pub mod ensemble;
pub mod multi_query;
pub mod multi_vector;
pub mod parent_document;
pub mod query_translator;
pub mod reranking;
pub mod self_query;
pub mod time_weighted;
