//! Plain text file document loader.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use futures::stream;
use rustchain_core::document_loaders::BaseLoader;
use rustchain_core::document_loaders::DocumentStream;
use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use serde_json::Value;

/// Loads a plain text file as a single [`Document`].
///
/// The entire file content becomes the document's `page_content`.
/// Metadata includes the source file path.
///
/// # Example
/// ```no_run
/// use rustchain::document_loaders::text::TextLoader;
/// use rustchain_core::document_loaders::BaseLoader;
///
/// # async fn example() -> rustchain_core::error::Result<()> {
/// let loader = TextLoader::new("data/notes.txt");
/// let docs = loader.load().await?;
/// assert_eq!(docs.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct TextLoader {
    path: PathBuf,
    encoding: Option<String>,
}

impl TextLoader {
    /// Create a new `TextLoader` for the given file path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            encoding: None,
        }
    }

    /// Set the encoding hint (currently only UTF-8 is supported).
    pub fn with_encoding(mut self, encoding: impl Into<String>) -> Self {
        self.encoding = Some(encoding.into());
        self
    }
}

#[async_trait]
impl BaseLoader for TextLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let content = tokio::fs::read_to_string(&self.path).await?;
        let mut metadata = HashMap::new();
        metadata.insert(
            "source".to_string(),
            Value::String(self.path.display().to_string()),
        );
        let doc = Document::new(content).with_metadata(metadata);
        Ok(Box::pin(stream::iter(vec![Ok(doc)])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_text_loader() {
        let mut tmp = NamedTempFile::new().unwrap();
        write!(tmp, "Hello, world!\nSecond line.").unwrap();

        let loader = TextLoader::new(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "Hello, world!\nSecond line.");
        assert_eq!(
            docs[0].metadata.get("source").unwrap(),
            &Value::String(tmp.path().display().to_string())
        );
    }
}
