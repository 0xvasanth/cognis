//! Abstract interfaces for document loader implementations.
//!
//! Mirrors Python `langchain_core.document_loaders`.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::documents::{Blob, Document};
use crate::error::Result;

/// Stream of documents yielded lazily.
pub type DocumentStream = Pin<Box<dyn Stream<Item = Result<Document>> + Send>>;

/// Stream of blobs yielded lazily.
pub type BlobStream = Pin<Box<dyn Stream<Item = Result<Blob>> + Send>>;

/// Abstract interface for document loaders.
///
/// Implementations should implement `lazy_load` using streams to avoid
/// loading all documents into memory at once.
///
/// `load` is provided for convenience and should not be overridden.
///
/// # Example
/// ```ignore
/// struct MyLoader { path: String }
///
/// #[async_trait]
/// impl BaseLoader for MyLoader {
///     async fn lazy_load(&self) -> Result<DocumentStream> {
///         let docs = vec![Document::new("content")];
///         Ok(Box::pin(futures::stream::iter(docs.into_iter().map(Ok))))
///     }
/// }
/// ```
#[async_trait]
pub trait BaseLoader: Send + Sync {
    /// Lazily load documents as a stream.
    ///
    /// This is the primary method implementors should override.
    async fn lazy_load(&self) -> Result<DocumentStream>;

    /// Eagerly load all documents into memory.
    ///
    /// Default implementation collects from `lazy_load`.
    async fn load(&self) -> Result<Vec<Document>> {
        use futures::StreamExt;
        let mut stream = self.lazy_load().await?;
        let mut docs = Vec::new();
        while let Some(doc_result) = stream.next().await {
            docs.push(doc_result?);
        }
        Ok(docs)
    }
}

/// Abstract interface for blob parsers.
///
/// A blob parser converts raw data stored in a `Blob` into one or more
/// `Document` objects. Parsers can be composed with blob loaders, making
/// it easy to reuse a parser independent of how the blob was loaded.
#[async_trait]
pub trait BaseBlobParser: Send + Sync {
    /// Lazily parse a blob into documents.
    ///
    /// This is the primary method implementors should override.
    async fn lazy_parse(&self, blob: &Blob) -> Result<DocumentStream>;

    /// Eagerly parse a blob into documents.
    ///
    /// Default implementation collects from `lazy_parse`.
    async fn parse(&self, blob: &Blob) -> Result<Vec<Document>> {
        use futures::StreamExt;
        let mut stream = self.lazy_parse(blob).await?;
        let mut docs = Vec::new();
        while let Some(doc_result) = stream.next().await {
            docs.push(doc_result?);
        }
        Ok(docs)
    }
}

/// Abstract interface for loading Blob objects from a source.
///
/// Blob loaders yield Blob objects rather than Documents. They're used
/// in conjunction with BlobParsers to separate loading from parsing.
#[async_trait]
pub trait BlobLoader: Send + Sync {
    /// Lazily load blobs from the source.
    async fn lazy_load(&self) -> Result<BlobStream>;

    /// Eagerly load all blobs into memory.
    async fn load(&self) -> Result<Vec<Blob>> {
        use futures::StreamExt;
        let mut stream = self.lazy_load().await?;
        let mut blobs = Vec::new();
        while let Some(blob_result) = stream.next().await {
            blobs.push(blob_result?);
        }
        Ok(blobs)
    }
}
