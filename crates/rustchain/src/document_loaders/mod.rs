//! Concrete document loader implementations.
//!
//! Provides loaders for common file formats: plain text, JSON, CSV,
//! and a directory loader that dispatches to the appropriate loader
//! based on file extension.

pub mod csv;
pub mod directory;
pub mod json;
pub mod text;

pub use self::csv::CsvLoader;
pub use directory::DirectoryLoader;
pub use json::JsonLoader;
pub use text::TextLoader;
