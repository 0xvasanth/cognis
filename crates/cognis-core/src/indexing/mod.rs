//! Indexing infrastructure for managing document ingestion into vectorstores.
//!
//! Mirrors Python `langchain_core.indexing`.

pub mod api;
pub mod base;
pub mod in_memory;

pub use api::{index, CleanupMode, IndexingResult};
pub use base::{
    DeleteResponse, DocumentIndex, InMemoryRecordManager, RecordManager, UpsertResponse,
};
pub use in_memory::InMemoryDocumentIndex;
