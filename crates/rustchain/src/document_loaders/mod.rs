//! Document loader implementations for ingesting data from various sources.
//!
//! This module provides both file-based loaders (via submodules) and in-memory
//! loaders for common data formats. All loaders implement [`BaseLoader`] from
//! `rustchain-core` for async streaming, and many also implement the synchronous
//! [`DocumentLoader`] trait defined here, which integrates with
//! [`TextSplitter`](crate::text_splitters::TextSplitter) for chunking.
//!
//! ## Submodule Loaders (file-based, async)
//!
//! - [`text::TextLoader`] -- loads a plain text file as a single document.
//! - [`json::JsonLoader`] -- loads a JSON file with optional jq-path extraction.
//! - [`csv::CsvLoader`] -- loads a CSV file, one document per row.
//! - [`directory::DirectoryLoader`] -- loads all files in a directory.
//! - [`html::HTMLLoader`] -- loads an HTML file, extracting text.
//! - [`markdown::MarkdownLoader`] -- loads a markdown file with header-based sections.
//!
//! ## Inline Loaders (in-memory, synchronous)
//!
//! - [`InMemoryTextLoader`] -- loads from a plain text string.
//! - [`StringLoader`] -- loads from a string with configurable metadata.
//! - [`InMemoryJsonLoader`] -- parses JSON strings with optional dot-path extraction.
//! - [`InMemoryCsvLoader`] -- parses CSV text into documents.
//! - [`InMemoryDirectoryLoader`] -- loads files from a directory on disk.
//! - [`SimulatedWebLoader`] -- simulated web loader using pre-fetched content.
//! - [`LoaderConfig`] -- shared configuration for loaders.

// ---------------------------------------------------------------------------
// Submodules (existing file-based async loaders)
// ---------------------------------------------------------------------------

pub mod csv;
pub mod directory;
pub mod html;
pub mod json;
pub mod markdown;
#[cfg(feature = "pdf")]
pub mod pdf;
pub mod text;
#[cfg(feature = "toml-loader")]
pub mod toml_loader;
#[cfg(feature = "yaml")]
pub mod yaml;

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure",
))]
pub mod web;

// Re-exports from submodules
pub use self::csv::CsvLoader;
pub use directory::DirectoryLoader;
pub use html::HTMLLoader;
pub use json::JsonLoader;
pub use markdown::MarkdownLoader;
#[cfg(feature = "pdf")]
pub use pdf::PdfLoader;
pub use text::TextLoader;
#[cfg(feature = "toml-loader")]
pub use toml_loader::TomlDocumentLoader;
#[cfg(feature = "yaml")]
pub use yaml::YamlDocumentLoader;

#[cfg(any(
    feature = "openai",
    feature = "anthropic",
    feature = "google",
    feature = "ollama",
    feature = "azure",
))]
pub use web::{WebBaseLoader, WebCrawler, WebLoader};

// ---------------------------------------------------------------------------
// Imports for inline loaders
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::path::PathBuf;

use rustchain_core::documents::Document;
use rustchain_core::error::{RustChainError, Result};
use serde_json::Value;

use crate::text_splitters::TextSplitter;

// ---------------------------------------------------------------------------
// DocumentLoader trait (synchronous, with TextSplitter integration)
// ---------------------------------------------------------------------------

/// Synchronous document loading trait with built-in text splitter integration.
///
/// Unlike [`BaseLoader`](rustchain_core::document_loaders::BaseLoader) which is
/// async and stream-based, `DocumentLoader` provides a simpler synchronous API
/// suitable for in-memory data and small files.
pub trait DocumentLoader: Send + Sync {
    /// Load documents from the source.
    fn load(&self) -> Result<Vec<Document>>;

    /// Load documents and then split them using the given text splitter.
    ///
    /// The default implementation calls [`load`](DocumentLoader::load) and then
    /// delegates to [`TextSplitter::split_documents`].
    fn load_and_split(&self, splitter: &dyn TextSplitter) -> Result<Vec<Document>> {
        let docs = self.load()?;
        Ok(splitter.split_documents(docs))
    }
}

// ---------------------------------------------------------------------------
// LoaderConfig
// ---------------------------------------------------------------------------

/// Shared configuration for document loaders.
///
/// Controls encoding, maximum file size, and which metadata keys to include
/// in loaded documents.
#[derive(Debug, Clone)]
pub struct LoaderConfig {
    /// Text encoding (default: `"utf-8"`).
    pub encoding: String,
    /// Maximum file size in bytes (default: 10 MB).
    pub max_file_size: usize,
    /// Metadata keys to include in loaded documents. If empty, all metadata
    /// is included.
    pub metadata_keys: Vec<String>,
}

impl Default for LoaderConfig {
    fn default() -> Self {
        Self {
            encoding: "utf-8".to_string(),
            max_file_size: 10 * 1024 * 1024, // 10 MB
            metadata_keys: Vec::new(),
        }
    }
}

impl LoaderConfig {
    /// Create a new `LoaderConfig` with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the text encoding.
    pub fn with_encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = encoding.into();
        self
    }

    /// Set the maximum file size in bytes.
    pub fn with_max_file_size(mut self, max_size: usize) -> Self {
        self.max_file_size = max_size;
        self
    }

    /// Set the metadata keys to include. If empty, all metadata is included.
    pub fn with_metadata_keys(mut self, keys: Vec<impl Into<String>>) -> Self {
        self.metadata_keys = keys.into_iter().map(|k| k.into()).collect();
        self
    }

    /// Filter metadata according to configured keys.
    ///
    /// If `metadata_keys` is empty, all metadata is returned unchanged.
    /// Otherwise, only keys present in the list are retained.
    pub fn filter_metadata(&self, metadata: HashMap<String, Value>) -> HashMap<String, Value> {
        if self.metadata_keys.is_empty() {
            return metadata;
        }
        metadata
            .into_iter()
            .filter(|(k, _)| self.metadata_keys.contains(k))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// InMemoryTextLoader
// ---------------------------------------------------------------------------

/// Loads a plain text string as a single [`Document`].
///
/// Sets `"source"` metadata to `"text"`.
///
/// # Example
/// ```
/// use rustchain::document_loaders::{InMemoryTextLoader, DocumentLoader};
///
/// let loader = InMemoryTextLoader::new("Hello, world!");
/// let docs = loader.load().unwrap();
/// assert_eq!(docs.len(), 1);
/// assert_eq!(docs[0].page_content, "Hello, world!");
/// ```
pub struct InMemoryTextLoader {
    text: String,
}

impl InMemoryTextLoader {
    /// Create a new `InMemoryTextLoader` from the given text.
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

impl DocumentLoader for InMemoryTextLoader {
    fn load(&self) -> Result<Vec<Document>> {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::String("text".to_string()));
        Ok(vec![Document::new(&self.text).with_metadata(metadata)])
    }
}

// ---------------------------------------------------------------------------
// StringLoader
// ---------------------------------------------------------------------------

/// Loads a string as a single [`Document`] with configurable metadata.
///
/// # Example
/// ```
/// use rustchain::document_loaders::{StringLoader, DocumentLoader};
/// use std::collections::HashMap;
/// use serde_json::Value;
///
/// let mut meta = HashMap::new();
/// meta.insert("author".to_string(), Value::String("Alice".to_string()));
///
/// let loader = StringLoader::new("Document content", meta);
/// let docs = loader.load().unwrap();
/// assert_eq!(docs[0].metadata.get("author").unwrap(), &Value::String("Alice".to_string()));
/// ```
pub struct StringLoader {
    text: String,
    metadata: HashMap<String, Value>,
}

impl StringLoader {
    /// Create a new `StringLoader` with the given text and metadata.
    pub fn new(text: impl Into<String>, metadata: HashMap<String, Value>) -> Self {
        Self {
            text: text.into(),
            metadata,
        }
    }
}

impl DocumentLoader for StringLoader {
    fn load(&self) -> Result<Vec<Document>> {
        Ok(vec![
            Document::new(&self.text).with_metadata(self.metadata.clone())
        ])
    }
}

// ---------------------------------------------------------------------------
// InMemoryJsonLoader
// ---------------------------------------------------------------------------

/// Parses a JSON string and creates documents.
///
/// Supports optional dot-notation path extraction (e.g. `"data.items"`) to
/// navigate into the JSON structure. When the target is an array, each element
/// becomes a separate document. Otherwise the value is serialized as a single
/// document.
///
/// # Example
/// ```
/// use rustchain::document_loaders::{InMemoryJsonLoader, DocumentLoader};
///
/// let json = r#"{"data": {"items": ["one", "two", "three"]}}"#;
/// let loader = InMemoryJsonLoader::new(json).with_jq_path("data.items");
/// let docs = loader.load().unwrap();
/// assert_eq!(docs.len(), 3);
/// ```
pub struct InMemoryJsonLoader {
    json_str: String,
    jq_path: Option<String>,
}

impl InMemoryJsonLoader {
    /// Create a new `InMemoryJsonLoader` from the given JSON string.
    pub fn new(json_str: impl Into<String>) -> Self {
        Self {
            json_str: json_str.into(),
            jq_path: None,
        }
    }

    /// Set a dot-notation path for extracting specific fields.
    ///
    /// Example paths: `"data.items"`, `"results"`, `"response.body.entries"`.
    /// Leading dots are stripped automatically.
    pub fn with_jq_path(mut self, path: impl Into<String>) -> Self {
        self.jq_path = Some(path.into());
        self
    }

    /// Navigate into a JSON value using dot-separated keys.
    fn navigate<'a>(&self, root: &'a Value) -> Result<&'a Value> {
        let path = match &self.jq_path {
            Some(p) => p,
            None => return Ok(root),
        };

        let keys: Vec<&str> = path
            .trim_start_matches('.')
            .split('.')
            .filter(|k| !k.is_empty())
            .collect();

        let mut current = root;
        for key in &keys {
            current = current.get(*key).ok_or_else(|| {
                RustChainError::Other(format!("JSON path key '{}' not found", key))
            })?;
        }
        Ok(current)
    }
}

impl DocumentLoader for InMemoryJsonLoader {
    fn load(&self) -> Result<Vec<Document>> {
        let root: Value = serde_json::from_str(&self.json_str)?;
        let target = self.navigate(&root)?;

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::String("json".to_string()));

        match target {
            Value::Array(arr) => {
                let mut docs = Vec::with_capacity(arr.len());
                for (i, item) in arr.iter().enumerate() {
                    let content = match item {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    };
                    let mut meta = metadata.clone();
                    meta.insert("index".to_string(), Value::Number(i.into()));
                    docs.push(Document::new(content).with_metadata(meta));
                }
                Ok(docs)
            }
            Value::Object(map) => {
                let content = serde_json::to_string_pretty(&Value::Object(map.clone()))
                    .unwrap_or_default();
                Ok(vec![Document::new(content).with_metadata(metadata)])
            }
            other => {
                let content = match other {
                    Value::String(s) => s.clone(),
                    v => v.to_string(),
                };
                Ok(vec![Document::new(content).with_metadata(metadata)])
            }
        }
    }
}

// ---------------------------------------------------------------------------
// InMemoryCsvLoader
// ---------------------------------------------------------------------------

/// Parses CSV text into documents (one document per row).
///
/// Supports column selection and custom separators.
///
/// # Example
/// ```
/// use rustchain::document_loaders::{InMemoryCsvLoader, DocumentLoader};
///
/// let csv_text = "name,age,city\nAlice,30,NYC\nBob,25,LA";
/// let loader = InMemoryCsvLoader::new(csv_text);
/// let docs = loader.load().unwrap();
/// assert_eq!(docs.len(), 2);
/// ```
pub struct InMemoryCsvLoader {
    csv_text: String,
    columns: Option<Vec<String>>,
    separator: u8,
}

impl InMemoryCsvLoader {
    /// Create a new `InMemoryCsvLoader` from the given CSV text.
    pub fn new(csv_text: impl Into<String>) -> Self {
        Self {
            csv_text: csv_text.into(),
            columns: None,
            separator: b',',
        }
    }

    /// Select specific columns to include in the document content.
    ///
    /// When set, only the specified columns are included in `page_content`.
    /// All other columns are stored as metadata.
    pub fn with_columns(mut self, cols: Vec<impl Into<String>>) -> Self {
        self.columns = Some(cols.into_iter().map(|c| c.into()).collect());
        self
    }

    /// Set a custom field separator (default: `','`).
    pub fn with_separator(mut self, sep: char) -> Self {
        self.separator = sep as u8;
        self
    }
}

impl DocumentLoader for InMemoryCsvLoader {
    fn load(&self) -> Result<Vec<Document>> {
        let mut reader = ::csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(self.separator)
            .from_reader(self.csv_text.as_bytes());

        let headers: Vec<String> = reader
            .headers()
            .map_err(|e| RustChainError::Other(format!("CSV header error: {}", e)))?
            .iter()
            .map(|h| h.to_string())
            .collect();

        let mut docs = Vec::new();

        for (row_idx, result) in reader.records().enumerate() {
            let record =
                result.map_err(|e| RustChainError::Other(format!("CSV row error: {}", e)))?;

            let row_map: HashMap<&str, &str> = headers
                .iter()
                .zip(record.iter())
                .map(|(h, v)| (h.as_str(), v))
                .collect();

            // Determine which columns go into content vs metadata.
            let (content_cols, metadata_cols): (Vec<&str>, Vec<&str>) = match &self.columns {
                Some(selected) => {
                    let sel_set: std::collections::HashSet<&str> =
                        selected.iter().map(|s| s.as_str()).collect();
                    let content: Vec<&str> = headers
                        .iter()
                        .map(|h| h.as_str())
                        .filter(|h| sel_set.contains(h))
                        .collect();
                    let meta: Vec<&str> = headers
                        .iter()
                        .map(|h| h.as_str())
                        .filter(|h| !sel_set.contains(h))
                        .collect();
                    (content, meta)
                }
                None => (headers.iter().map(|h| h.as_str()).collect(), Vec::new()),
            };

            let content = content_cols
                .iter()
                .filter_map(|col| row_map.get(col).map(|v| format!("{}: {}", col, v)))
                .collect::<Vec<_>>()
                .join("\n");

            let mut metadata = HashMap::new();
            metadata.insert("source".to_string(), Value::String("csv".to_string()));
            metadata.insert("row".to_string(), Value::Number(row_idx.into()));

            for col in &metadata_cols {
                if let Some(val) = row_map.get(col) {
                    metadata.insert(col.to_string(), Value::String(val.to_string()));
                }
            }

            docs.push(Document::new(content).with_metadata(metadata));
        }

        Ok(docs)
    }
}

// ---------------------------------------------------------------------------
// InMemoryDirectoryLoader
// ---------------------------------------------------------------------------

/// Loads all files from a directory path on disk.
///
/// Files are read synchronously and each file becomes a single document.
/// Supports glob filtering and recursive traversal.
///
/// # Example
/// ```no_run
/// use rustchain::document_loaders::{InMemoryDirectoryLoader, DocumentLoader};
///
/// let loader = InMemoryDirectoryLoader::new("/tmp/data")
///     .with_glob("*.txt")
///     .with_recursive(true);
/// let docs = loader.load().unwrap();
/// ```
pub struct InMemoryDirectoryLoader {
    path: PathBuf,
    glob_pattern: Option<String>,
    recursive: bool,
}

impl InMemoryDirectoryLoader {
    /// Create a new `InMemoryDirectoryLoader` for the given directory path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            glob_pattern: None,
            recursive: false,
        }
    }

    /// Set a glob pattern to filter files (e.g. `"*.txt"`).
    pub fn with_glob(mut self, pattern: impl Into<String>) -> Self {
        self.glob_pattern = Some(pattern.into());
        self
    }

    /// Enable or disable recursive directory traversal.
    pub fn with_recursive(mut self, recursive: bool) -> Self {
        self.recursive = recursive;
        self
    }

    /// Collect matching file paths.
    fn collect_files(&self) -> Result<Vec<PathBuf>> {
        let pattern = self.glob_pattern.as_deref().unwrap_or("*");
        let glob_str = if self.recursive {
            format!("{}/**/{}", self.path.display(), pattern)
        } else {
            format!("{}/{}", self.path.display(), pattern)
        };

        let entries = glob::glob(&glob_str)
            .map_err(|e| RustChainError::Other(format!("Invalid glob pattern: {}", e)))?;

        let mut files = Vec::new();
        for entry in entries {
            let path =
                entry.map_err(|e| RustChainError::Other(format!("Glob entry error: {}", e)))?;
            if path.is_file() {
                files.push(path);
            }
        }
        files.sort();
        Ok(files)
    }
}

impl DocumentLoader for InMemoryDirectoryLoader {
    fn load(&self) -> Result<Vec<Document>> {
        let files = self.collect_files()?;
        let mut docs = Vec::new();

        for file_path in &files {
            let content = std::fs::read_to_string(file_path).map_err(|e| {
                RustChainError::Other(format!(
                    "Failed to read '{}': {}",
                    file_path.display(),
                    e
                ))
            })?;

            let mut metadata = HashMap::new();
            metadata.insert(
                "source".to_string(),
                Value::String(file_path.display().to_string()),
            );

            if let Some(ext) = file_path.extension().and_then(|e| e.to_str()) {
                metadata.insert(
                    "file_type".to_string(),
                    Value::String(ext.to_string()),
                );
            }

            docs.push(Document::new(content).with_metadata(metadata));
        }

        Ok(docs)
    }
}

// ---------------------------------------------------------------------------
// SimulatedWebLoader
// ---------------------------------------------------------------------------

/// Simulated web page loader that works with pre-fetched content.
///
/// This loader does **not** perform actual HTTP requests. Instead, it accepts
/// pre-fetched content and creates a document with the URL as source metadata.
///
/// Supports optional text extraction between markers using
/// [`with_selector`](SimulatedWebLoader::with_selector).
///
/// # Example
/// ```
/// use rustchain::document_loaders::{SimulatedWebLoader, DocumentLoader};
///
/// let loader = SimulatedWebLoader::new(
///     "https://example.com",
///     "<html><body>Hello world</body></html>",
/// );
/// let docs = loader.load().unwrap();
/// assert_eq!(docs.len(), 1);
/// ```
pub struct SimulatedWebLoader {
    url: String,
    content: String,
    selector: Option<String>,
}

impl SimulatedWebLoader {
    /// Create a new `SimulatedWebLoader` with pre-fetched content.
    pub fn new(url: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            content: content.into(),
            selector: None,
        }
    }

    /// Set a selector to extract text between markers.
    ///
    /// The selector is treated as a pair of start/end markers. The loader will
    /// extract text between `<selector>` and `</selector>` tags (or the first
    /// occurrence of the selector string as a section delimiter).
    ///
    /// For HTML-like content, use tag names such as `"article"`, `"main"`, etc.
    pub fn with_selector(mut self, selector: impl Into<String>) -> Self {
        self.selector = Some(selector.into());
        self
    }

    /// Extract content between `<tag>` and `</tag>` markers.
    fn extract_between_tags(content: &str, tag: &str) -> Option<String> {
        let open = format!("<{}", tag);
        let close = format!("</{}>", tag);

        let start_tag_pos = content.find(&open)?;
        // Find the end of the opening tag (the `>` after attributes).
        let tag_content_start = content[start_tag_pos..].find('>')? + start_tag_pos + 1;
        let end_pos = content[tag_content_start..].find(&close)? + tag_content_start;

        Some(content[tag_content_start..end_pos].to_string())
    }
}

impl DocumentLoader for SimulatedWebLoader {
    fn load(&self) -> Result<Vec<Document>> {
        let page_content = match &self.selector {
            Some(selector) => {
                Self::extract_between_tags(&self.content, selector)
                    .unwrap_or_else(|| self.content.clone())
            }
            None => self.content.clone(),
        };

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::String(self.url.clone()));
        metadata.insert(
            "content_type".to_string(),
            Value::String("text/html".to_string()),
        );

        Ok(vec![Document::new(page_content).with_metadata(metadata)])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod inline_tests {
    use super::*;
    use crate::text_splitters::CharacterTextSplitter;
    use std::fs;
    use tempfile::TempDir;

    // -----------------------------------------------------------------------
    // DocumentLoader trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_document_loader_trait_load() {
        let loader = InMemoryTextLoader::new("hello");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "hello");
    }

    #[test]
    fn test_document_loader_trait_load_and_split() {
        let loader = InMemoryTextLoader::new("aaa\n\nbbb\n\nccc");
        let splitter = CharacterTextSplitter::new()
            .with_chunk_size(5)
            .with_chunk_overlap(0);
        let docs = loader.load_and_split(&splitter).unwrap();
        assert!(docs.len() >= 2);
    }

    // -----------------------------------------------------------------------
    // InMemoryTextLoader
    // -----------------------------------------------------------------------

    #[test]
    fn test_text_loader_basic() {
        let loader = InMemoryTextLoader::new("Hello, world!");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "Hello, world!");
    }

    #[test]
    fn test_text_loader_source_metadata() {
        let loader = InMemoryTextLoader::new("test");
        let docs = loader.load().unwrap();
        assert_eq!(
            docs[0].metadata.get("source").unwrap(),
            &Value::String("text".to_string())
        );
    }

    #[test]
    fn test_text_loader_empty_string() {
        let loader = InMemoryTextLoader::new("");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "");
    }

    #[test]
    fn test_text_loader_multiline() {
        let loader = InMemoryTextLoader::new("line1\nline2\nline3");
        let docs = loader.load().unwrap();
        assert_eq!(docs[0].page_content, "line1\nline2\nline3");
    }

    #[test]
    fn test_text_loader_unicode() {
        let loader = InMemoryTextLoader::new("Hello, world!");
        let docs = loader.load().unwrap();
        assert_eq!(docs[0].page_content, "Hello, world!");
    }

    // -----------------------------------------------------------------------
    // StringLoader
    // -----------------------------------------------------------------------

    #[test]
    fn test_string_loader_basic() {
        let mut meta = HashMap::new();
        meta.insert("author".to_string(), Value::String("Alice".to_string()));
        let loader = StringLoader::new("content", meta);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "content");
        assert_eq!(
            docs[0].metadata.get("author").unwrap(),
            &Value::String("Alice".to_string())
        );
    }

    #[test]
    fn test_string_loader_empty_metadata() {
        let loader = StringLoader::new("text", HashMap::new());
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].metadata.is_empty());
    }

    #[test]
    fn test_string_loader_multiple_metadata() {
        let mut meta = HashMap::new();
        meta.insert("key1".to_string(), Value::String("val1".to_string()));
        meta.insert("key2".to_string(), Value::Number(42.into()));
        meta.insert("key3".to_string(), Value::Bool(true));
        let loader = StringLoader::new("data", meta);
        let docs = loader.load().unwrap();
        assert_eq!(docs[0].metadata.len(), 3);
    }

    #[test]
    fn test_string_loader_preserves_metadata() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), Value::String("custom_source".to_string()));
        let loader = StringLoader::new("text", meta);
        let docs = loader.load().unwrap();
        assert_eq!(
            docs[0].metadata.get("source").unwrap(),
            &Value::String("custom_source".to_string())
        );
    }

    // -----------------------------------------------------------------------
    // InMemoryJsonLoader
    // -----------------------------------------------------------------------

    #[test]
    fn test_json_loader_single_object() {
        let loader = InMemoryJsonLoader::new(r#"{"name": "Alice", "age": 30}"#);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].page_content.contains("Alice"));
    }

    #[test]
    fn test_json_loader_array() {
        let loader = InMemoryJsonLoader::new(r#"["one", "two", "three"]"#);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].page_content, "one");
        assert_eq!(docs[1].page_content, "two");
        assert_eq!(docs[2].page_content, "three");
    }

    #[test]
    fn test_json_loader_with_jq_path() {
        let json = r#"{"data": {"items": ["alpha", "beta"]}}"#;
        let loader = InMemoryJsonLoader::new(json).with_jq_path("data.items");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].page_content, "alpha");
        assert_eq!(docs[1].page_content, "beta");
    }

    #[test]
    fn test_json_loader_with_leading_dot_path() {
        let json = r#"{"results": [1, 2, 3]}"#;
        let loader = InMemoryJsonLoader::new(json).with_jq_path(".results");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 3);
    }

    #[test]
    fn test_json_loader_nested_objects() {
        let json = r#"{"data": {"items": [{"text": "hello"}, {"text": "world"}]}}"#;
        let loader = InMemoryJsonLoader::new(json).with_jq_path("data.items");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs[0].page_content.contains("hello"));
    }

    #[test]
    fn test_json_loader_invalid_json() {
        let loader = InMemoryJsonLoader::new("not valid json");
        let result = loader.load();
        assert!(result.is_err());
    }

    #[test]
    fn test_json_loader_invalid_path() {
        let json = r#"{"a": {"b": 1}}"#;
        let loader = InMemoryJsonLoader::new(json).with_jq_path("a.nonexistent");
        let result = loader.load();
        assert!(result.is_err());
    }

    #[test]
    fn test_json_loader_scalar_value() {
        let loader = InMemoryJsonLoader::new(r#""just a string""#);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "just a string");
    }

    #[test]
    fn test_json_loader_number_value() {
        let loader = InMemoryJsonLoader::new("42");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "42");
    }

    #[test]
    fn test_json_loader_index_metadata() {
        let loader = InMemoryJsonLoader::new(r#"["a", "b"]"#);
        let docs = loader.load().unwrap();
        assert_eq!(
            docs[0].metadata.get("index").unwrap(),
            &Value::Number(0.into())
        );
        assert_eq!(
            docs[1].metadata.get("index").unwrap(),
            &Value::Number(1.into())
        );
    }

    #[test]
    fn test_json_loader_empty_array() {
        let loader = InMemoryJsonLoader::new("[]");
        let docs = loader.load().unwrap();
        assert!(docs.is_empty());
    }

    #[test]
    fn test_json_loader_empty_object() {
        let loader = InMemoryJsonLoader::new("{}");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
    }

    // -----------------------------------------------------------------------
    // InMemoryCsvLoader
    // -----------------------------------------------------------------------

    #[test]
    fn test_csv_loader_basic() {
        let loader = InMemoryCsvLoader::new("name,age\nAlice,30\nBob,25");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs[0].page_content.contains("Alice"));
        assert!(docs[0].page_content.contains("30"));
    }

    #[test]
    fn test_csv_loader_column_selection() {
        let loader = InMemoryCsvLoader::new("id,name,bio\n1,Alice,Engineer\n2,Bob,Designer")
            .with_columns(vec!["name", "bio"]);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs[0].page_content.contains("name: Alice"));
        assert!(docs[0].page_content.contains("bio: Engineer"));
        assert!(!docs[0].page_content.contains("id"));
        // id should be in metadata
        assert_eq!(
            docs[0].metadata.get("id").unwrap(),
            &Value::String("1".to_string())
        );
    }

    #[test]
    fn test_csv_loader_custom_separator() {
        let loader = InMemoryCsvLoader::new("name\tage\nAlice\t30")
            .with_separator('\t');
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].page_content.contains("Alice"));
        assert!(docs[0].page_content.contains("30"));
    }

    #[test]
    fn test_csv_loader_semicolon_separator() {
        let loader = InMemoryCsvLoader::new("name;age\nAlice;30")
            .with_separator(';');
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].page_content.contains("Alice"));
    }

    #[test]
    fn test_csv_loader_row_metadata() {
        let loader = InMemoryCsvLoader::new("x\n1\n2\n3");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 3);
        for (i, doc) in docs.iter().enumerate() {
            assert_eq!(
                doc.metadata.get("row").unwrap(),
                &Value::Number(i.into())
            );
        }
    }

    #[test]
    fn test_csv_loader_source_metadata() {
        let loader = InMemoryCsvLoader::new("a\n1");
        let docs = loader.load().unwrap();
        assert_eq!(
            docs[0].metadata.get("source").unwrap(),
            &Value::String("csv".to_string())
        );
    }

    #[test]
    fn test_csv_loader_empty_csv() {
        let loader = InMemoryCsvLoader::new("name,age");
        let docs = loader.load().unwrap();
        assert!(docs.is_empty());
    }

    #[test]
    fn test_csv_loader_missing_columns() {
        // Requesting columns that don't exist
        let loader = InMemoryCsvLoader::new("name,age\nAlice,30")
            .with_columns(vec!["nonexistent"]);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        // Content should be empty since the column doesn't exist
        assert_eq!(docs[0].page_content, "");
    }

    // -----------------------------------------------------------------------
    // InMemoryDirectoryLoader
    // -----------------------------------------------------------------------

    #[test]
    fn test_directory_loader_basic() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("file1.txt"), "Hello").unwrap();
        fs::write(dir.path().join("file2.txt"), "World").unwrap();

        let loader = InMemoryDirectoryLoader::new(dir.path());
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_directory_loader_with_glob() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("notes.txt"), "Notes content").unwrap();
        fs::write(dir.path().join("data.json"), r#"{"key": "value"}"#).unwrap();

        let loader = InMemoryDirectoryLoader::new(dir.path()).with_glob("*.txt");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "Notes content");
    }

    #[test]
    fn test_directory_loader_recursive() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("subdir");
        fs::create_dir(&sub).unwrap();
        fs::write(dir.path().join("top.txt"), "Top level").unwrap();
        fs::write(sub.join("nested.txt"), "Nested level").unwrap();

        let loader = InMemoryDirectoryLoader::new(dir.path())
            .with_glob("*.txt")
            .with_recursive(true);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_directory_loader_non_recursive_skips_nested() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("subdir");
        fs::create_dir(&sub).unwrap();
        fs::write(dir.path().join("top.txt"), "Top").unwrap();
        fs::write(sub.join("nested.txt"), "Nested").unwrap();

        let loader = InMemoryDirectoryLoader::new(dir.path()).with_glob("*.txt");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
    }

    #[test]
    fn test_directory_loader_source_metadata() {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("test.txt");
        fs::write(&file_path, "content").unwrap();

        let loader = InMemoryDirectoryLoader::new(dir.path());
        let docs = loader.load().unwrap();
        assert_eq!(
            docs[0].metadata.get("source").unwrap(),
            &Value::String(file_path.display().to_string())
        );
    }

    #[test]
    fn test_directory_loader_file_type_metadata() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("test.txt"), "content").unwrap();

        let loader = InMemoryDirectoryLoader::new(dir.path());
        let docs = loader.load().unwrap();
        assert_eq!(
            docs[0].metadata.get("file_type").unwrap(),
            &Value::String("txt".to_string())
        );
    }

    #[test]
    fn test_directory_loader_empty_directory() {
        let dir = TempDir::new().unwrap();
        let loader = InMemoryDirectoryLoader::new(dir.path());
        let docs = loader.load().unwrap();
        assert!(docs.is_empty());
    }

    // -----------------------------------------------------------------------
    // SimulatedWebLoader
    // -----------------------------------------------------------------------

    #[test]
    fn test_web_loader_basic() {
        let loader = SimulatedWebLoader::new(
            "https://example.com",
            "Page content here",
        );
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "Page content here");
    }

    #[test]
    fn test_web_loader_url_metadata() {
        let loader = SimulatedWebLoader::new(
            "https://example.com/page",
            "content",
        );
        let docs = loader.load().unwrap();
        assert_eq!(
            docs[0].metadata.get("source").unwrap(),
            &Value::String("https://example.com/page".to_string())
        );
    }

    #[test]
    fn test_web_loader_content_type_metadata() {
        let loader = SimulatedWebLoader::new("https://example.com", "content");
        let docs = loader.load().unwrap();
        assert_eq!(
            docs[0].metadata.get("content_type").unwrap(),
            &Value::String("text/html".to_string())
        );
    }

    #[test]
    fn test_web_loader_with_selector() {
        let html = r#"<html><body><nav>Menu</nav><article>Important text</article></body></html>"#;
        let loader = SimulatedWebLoader::new("https://example.com", html)
            .with_selector("article");
        let docs = loader.load().unwrap();
        assert_eq!(docs[0].page_content, "Important text");
    }

    #[test]
    fn test_web_loader_with_selector_and_attributes() {
        let html = r#"<div><main class="content">Main content here</main></div>"#;
        let loader = SimulatedWebLoader::new("https://example.com", html)
            .with_selector("main");
        let docs = loader.load().unwrap();
        assert_eq!(docs[0].page_content, "Main content here");
    }

    #[test]
    fn test_web_loader_selector_not_found() {
        let html = "<p>Just a paragraph</p>";
        let loader = SimulatedWebLoader::new("https://example.com", html)
            .with_selector("article");
        let docs = loader.load().unwrap();
        // Falls back to full content
        assert_eq!(docs[0].page_content, html);
    }

    #[test]
    fn test_web_loader_empty_content() {
        let loader = SimulatedWebLoader::new("https://example.com", "");
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "");
    }

    // -----------------------------------------------------------------------
    // LoaderConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_loader_config_defaults() {
        let config = LoaderConfig::default();
        assert_eq!(config.encoding, "utf-8");
        assert_eq!(config.max_file_size, 10 * 1024 * 1024);
        assert!(config.metadata_keys.is_empty());
    }

    #[test]
    fn test_loader_config_custom_encoding() {
        let config = LoaderConfig::new().with_encoding("latin-1");
        assert_eq!(config.encoding, "latin-1");
    }

    #[test]
    fn test_loader_config_custom_max_file_size() {
        let config = LoaderConfig::new().with_max_file_size(1024);
        assert_eq!(config.max_file_size, 1024);
    }

    #[test]
    fn test_loader_config_metadata_keys() {
        let config = LoaderConfig::new().with_metadata_keys(vec!["source", "author"]);
        assert_eq!(config.metadata_keys, vec!["source", "author"]);
    }

    #[test]
    fn test_loader_config_filter_metadata_all() {
        let config = LoaderConfig::new(); // empty keys = include all
        let mut meta = HashMap::new();
        meta.insert("a".to_string(), Value::String("1".to_string()));
        meta.insert("b".to_string(), Value::String("2".to_string()));

        let filtered = config.filter_metadata(meta);
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_loader_config_filter_metadata_selective() {
        let config = LoaderConfig::new().with_metadata_keys(vec!["source"]);
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), Value::String("test".to_string()));
        meta.insert("extra".to_string(), Value::String("removed".to_string()));

        let filtered = config.filter_metadata(meta);
        assert_eq!(filtered.len(), 1);
        assert!(filtered.contains_key("source"));
        assert!(!filtered.contains_key("extra"));
    }

    // -----------------------------------------------------------------------
    // TextSplitter integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_load_and_split_preserves_metadata() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), Value::String("test".to_string()));
        let loader = StringLoader::new("aaa\n\nbbb\n\nccc", meta);
        let splitter = CharacterTextSplitter::new()
            .with_chunk_size(5)
            .with_chunk_overlap(0);
        let docs = loader.load_and_split(&splitter).unwrap();
        for doc in &docs {
            assert_eq!(
                doc.metadata.get("source").unwrap(),
                &Value::String("test".to_string())
            );
        }
    }

    #[test]
    fn test_load_and_split_csv() {
        let loader = InMemoryCsvLoader::new("text\nThis is a very long text that should be split into multiple chunks for testing purposes");
        let splitter = CharacterTextSplitter::new()
            .with_chunk_size(20)
            .with_chunk_overlap(0);
        let docs = loader.load_and_split(&splitter).unwrap();
        // The CSV produces one doc; splitting may produce multiple chunks
        assert!(docs.len() >= 1);
    }

    #[test]
    fn test_load_and_split_json_array() {
        let json = r#"["first chunk of text", "second chunk of text"]"#;
        let loader = InMemoryJsonLoader::new(json);
        let splitter = CharacterTextSplitter::new()
            .with_chunk_size(100)
            .with_chunk_overlap(0);
        let docs = loader.load_and_split(&splitter).unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_load_and_split_directory() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.txt"), "aaa\n\nbbb\n\nccc").unwrap();

        let loader = InMemoryDirectoryLoader::new(dir.path());
        let splitter = CharacterTextSplitter::new()
            .with_chunk_size(5)
            .with_chunk_overlap(0);
        let docs = loader.load_and_split(&splitter).unwrap();
        assert!(docs.len() >= 2);
    }

    #[test]
    fn test_load_and_split_web() {
        let loader = SimulatedWebLoader::new(
            "https://example.com",
            "aaa\n\nbbb\n\nccc",
        );
        let splitter = CharacterTextSplitter::new()
            .with_chunk_size(5)
            .with_chunk_overlap(0);
        let docs = loader.load_and_split(&splitter).unwrap();
        assert!(docs.len() >= 2);
    }
}
