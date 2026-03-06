//! Concrete document loader implementations.
//!
//! Provides loaders for common file formats: plain text, JSON, CSV,
//! and a directory loader that dispatches to the appropriate loader
//! based on file extension.

pub mod csv;
pub mod directory;
pub mod html;
pub mod json;
pub mod markdown;
#[cfg(feature = "pdf")]
pub mod pdf;
pub mod text;

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure",
))]
pub mod web;

pub use self::csv::CsvLoader;
pub use directory::DirectoryLoader;
pub use html::HTMLLoader;
pub use json::JsonLoader;
pub use markdown::MarkdownLoader;
#[cfg(feature = "pdf")]
pub use pdf::PdfLoader;
pub use text::TextLoader;

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure",
))]
pub use web::WebBaseLoader;
