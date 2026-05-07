//! Document loaders — read sources into [`Document`]s.
//!
//! Each loader returns either:
//! - `Result<Vec<Document>>` — for small, bounded sources (one file).
//! - `RunnableStream<Document>` — for unbounded / large sources (a directory walk).
//!
//! The unifying trait is [`DocumentLoader`]: `load() -> Stream<Item=Result<Document>>`,
//! so callers can compose with `futures` combinators (e.g. `.take(100)`,
//! `.filter`).

use async_trait::async_trait;
use futures::Stream;

use cognis2_core::Result;

use crate::document::Document;

#[cfg(feature = "csv-loader")]
pub mod csv_loader;
pub mod directory;
#[cfg(feature = "html-loader")]
pub mod html;
pub mod json;
pub mod markdown;
#[cfg(feature = "pdf-loader")]
pub mod pdf;
pub mod text;
#[cfg(feature = "toml-loader")]
pub mod toml_loader;
#[cfg(feature = "web-loader")]
pub mod web;
#[cfg(feature = "yaml-loader")]
pub mod yaml;

#[cfg(feature = "csv-loader")]
pub use csv_loader::CsvLoader;
pub use directory::DirectoryLoader;
#[cfg(feature = "html-loader")]
pub use html::HtmlLoader;
pub use json::JsonLoader;
pub use markdown::MarkdownLoader;
#[cfg(feature = "pdf-loader")]
pub use pdf::PdfLoader;
pub use text::TextLoader;
#[cfg(feature = "toml-loader")]
pub use toml_loader::TomlLoader;
#[cfg(feature = "web-loader")]
pub use web::WebLoader;
#[cfg(feature = "yaml-loader")]
pub use yaml::YamlLoader;

/// A document loader.
///
/// Implementations stream documents — preferred over collecting into a
/// `Vec` so callers can early-terminate large sources.
#[async_trait]
pub trait DocumentLoader: Send + Sync {
    /// Stream of documents. Errors are per-item so a partial-failure source
    /// can still yield successful items.
    async fn load(&self) -> Result<DocumentStream>;

    /// Convenience: collect every yielded document. Stops at the first error.
    async fn load_all(&self) -> Result<Vec<Document>> {
        use futures::StreamExt;
        let mut s = self.load().await?;
        let mut out = Vec::new();
        while let Some(doc) = s.next().await {
            out.push(doc?);
        }
        Ok(out)
    }
}

/// Boxed stream returned by [`DocumentLoader::load`].
pub type DocumentStream = std::pin::Pin<Box<dyn Stream<Item = Result<Document>> + Send>>;
