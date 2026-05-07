//! Retrievers — `Runnable<String, Vec<Document>>`.
//!
//! A retriever takes a free-form query string and returns the top-k most
//! relevant documents. They compose into chains via `RunnableExt::pipe`.

pub mod bm25;
pub mod caching;
pub mod compressor_pipeline;
pub mod ensemble;
pub mod multi_vector;
pub mod parent_document;
pub mod query_translator;
pub mod vector;

pub use bm25::BM25Retriever;
pub use caching::CachingRetriever;
pub use compressor_pipeline::CompressorPipeline;
pub use ensemble::EnsembleRetriever;
pub use multi_vector::MultiVectorRetriever;
pub use parent_document::ParentDocumentRetriever;
pub use query_translator::QueryTranslatorRetriever;
pub use vector::VectorRetriever;
