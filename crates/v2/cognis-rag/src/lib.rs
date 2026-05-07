//! # cognis2-rag
//!
//! v2-beta RAG primitives: embeddings, vector stores, document loaders,
//! text splitters, retrievers, and an indexing pipeline.
//!
//! Top-level modules:
//! - [`document`] — the universal `Document` type.
//! - [`embeddings`] — `Embeddings` trait + Fake/OpenAI/Ollama impls.
//! - [`vectorstore`] — `VectorStore` trait + `InMemoryVectorStore`.
//! - [`loaders`] — text/markdown/json/directory/csv/html loaders.
//! - [`splitters`] — recursive-char + markdown-aware splitters.
//! - [`retrievers`] — vector / BM25 / ensemble retrievers (each is a `Runnable`).
//! - [`indexing`] — wire load → split → embed → store with one call.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

pub mod cross_encoder;
pub mod distance;
pub mod docstore;
pub mod document;
pub mod embeddings;
pub mod indexing;
pub mod loaders;
pub mod multi_vector;
pub mod record_manager;
pub mod retrievers;
pub mod splitters;
pub mod transformers;
pub mod vectorstore;

pub use cross_encoder::{CrossEncoder, CrossEncoderReranker, FnCrossEncoder};
pub use distance::Distance;
pub use docstore::{Docstore, InMemoryDocstore};
pub use document::Document;
#[cfg(feature = "ollama")]
pub use embeddings::OllamaEmbeddings;
#[cfg(feature = "openai")]
pub use embeddings::OpenAIEmbeddings;
pub use embeddings::{Embeddings, FakeEmbeddings};
pub use indexing::{IncrementalReport, IndexingPipeline};
#[cfg(feature = "csv-loader")]
pub use loaders::CsvLoader;
#[cfg(feature = "html-loader")]
pub use loaders::HtmlLoader;
#[cfg(feature = "pdf-loader")]
pub use loaders::PdfLoader;
#[cfg(feature = "toml-loader")]
pub use loaders::TomlLoader;
#[cfg(feature = "web-loader")]
pub use loaders::WebLoader;
#[cfg(feature = "yaml-loader")]
pub use loaders::YamlLoader;
pub use loaders::{
    DirectoryLoader, DocumentLoader, DocumentStream, JsonLoader, MarkdownLoader, TextLoader,
};
pub use multi_vector::MultiVectorIndexer;
pub use record_manager::{fingerprint, InMemoryRecordManager, RecordManager};
pub use retrievers::{
    BM25Retriever, CachingRetriever, CompressorPipeline, EnsembleRetriever, MultiVectorRetriever,
    ParentDocumentRetriever, QueryTranslatorRetriever, VectorRetriever,
};
pub use splitters::{
    CharTokenizer, CodeLanguage, CodeSplitter, FnTokenizer, JsonSplitter, MarkdownSplitter,
    RecursiveCharSplitter, SentenceSplitter, TextSplitter, TokenAwareSplitter, Tokenizer,
};
pub use transformers::LongContextReorder;
pub use vectorstore::{Filter, InMemoryVectorStore, SearchResult, VectorStore};

/// Common imports for v2 RAG user code.
pub mod prelude {
    pub use crate::{
        BM25Retriever, DirectoryLoader, Distance, Document, DocumentLoader, Embeddings,
        EnsembleRetriever, InMemoryVectorStore, IndexingPipeline, JsonLoader, MarkdownLoader,
        MarkdownSplitter, RecursiveCharSplitter, SearchResult, TextLoader, TextSplitter,
        VectorRetriever, VectorStore,
    };
    pub use cognis2_core::prelude::*;
}
