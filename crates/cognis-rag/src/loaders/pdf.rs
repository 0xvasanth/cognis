//! PDF loader (feature `pdf-loader`).
//!
//! Extracts a PDF's textual content via `pdf-extract`. Pulls heavy native-ish
//! deps so it's gated behind a feature flag — only enable when you actually
//! load PDFs.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use futures::stream;

use cognis_core::{CognisError, Result};

use crate::document::Document;

use super::{DocumentLoader, DocumentStream};

/// Loads a PDF file as one [`Document`] containing its extracted text.
/// `metadata.format = "pdf"`.
pub struct PdfLoader {
    path: PathBuf,
}

impl PdfLoader {
    /// Construct.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

#[async_trait]
impl DocumentLoader for PdfLoader {
    async fn load(&self) -> Result<DocumentStream> {
        let path = self.path.clone();
        // pdf-extract is sync — push it to a blocking thread.
        let text = tokio::task::spawn_blocking(move || pdf_extract::extract_text(&path))
            .await
            .map_err(|e| CognisError::Internal(format!("PdfLoader join: {e}")))?
            .map_err(|e| {
                CognisError::Configuration(format!(
                    "PdfLoader: extract `{}`: {e}",
                    self.path.display()
                ))
            })?;

        let doc = Document::new(text)
            .with_metadata("source", self.path.display().to_string())
            .with_metadata("format", "pdf");
        Ok(Box::pin(stream::iter(vec![Ok(doc)])))
    }
}
