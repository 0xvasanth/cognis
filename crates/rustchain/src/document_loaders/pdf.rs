//! PDF file document loader.
//!
//! Requires the `pdf` feature to be enabled. Uses `pdf-extract` for text
//! extraction from PDF files.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use futures::stream;
use rustchain_core::document_loaders::BaseLoader;
use rustchain_core::document_loaders::DocumentStream;
use rustchain_core::documents::Document;
use rustchain_core::error::{Result, RustChainError};
use serde_json::Value;

/// Loads a PDF file and extracts its text content as a single [`Document`].
///
/// Text extraction is performed using the `pdf-extract` crate. The extracted
/// text becomes the document's `page_content`. Metadata includes the source
/// file path and content type.
///
/// # Example
/// ```no_run
/// use rustchain::document_loaders::pdf::PdfLoader;
/// use rustchain_core::document_loaders::BaseLoader;
///
/// # async fn example() -> rustchain_core::error::Result<()> {
/// let loader = PdfLoader::new("report.pdf");
/// let docs = loader.load().await?;
/// assert_eq!(docs.len(), 1);
/// # Ok(())
/// # }
/// ```
pub struct PdfLoader {
    path: PathBuf,
}

impl PdfLoader {
    /// Create a new `PdfLoader` for the given file path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

#[async_trait]
impl BaseLoader for PdfLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let path = self.path.clone();

        // pdf-extract is synchronous, so run in a blocking task
        let content = tokio::task::spawn_blocking(move || -> Result<String> {
            let bytes =
                std::fs::read(&path).map_err(|e| RustChainError::Other(e.to_string()))?;
            pdf_extract::extract_text_from_mem(&bytes)
                .map_err(|e| RustChainError::Other(format!("PDF extraction failed: {e}")))
        })
        .await
        .map_err(|e| RustChainError::Other(format!("Task join error: {e}")))??;

        let mut metadata = HashMap::new();
        metadata.insert(
            "source".to_string(),
            Value::String(self.path.display().to_string()),
        );
        metadata.insert(
            "content_type".to_string(),
            Value::String("application/pdf".to_string()),
        );

        let doc = Document::new(content).with_metadata(metadata);
        Ok(Box::pin(stream::iter(vec![Ok(doc)])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_loader_creation() {
        let loader = PdfLoader::new("/tmp/test.pdf");
        assert_eq!(loader.path, PathBuf::from("/tmp/test.pdf"));
    }

    #[tokio::test]
    async fn test_pdf_loader_nonexistent_file() {
        let loader = PdfLoader::new("/tmp/nonexistent_rustchain_test_file.pdf");
        let result = loader.load().await;
        assert!(result.is_err());
    }
}
