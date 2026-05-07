//! Markdown loader — same as [`super::TextLoader`] but tags `metadata.format`.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use futures::stream;

use cognis2_core::{CognisError, Result};

use crate::document::Document;

use super::{DocumentLoader, DocumentStream};

/// Loads a Markdown file as one [`Document`]. Identical to [`super::TextLoader`]
/// but tags `metadata.format = "markdown"` so downstream splitters can apply
/// markdown-aware chunking.
pub struct MarkdownLoader {
    path: PathBuf,
}

impl MarkdownLoader {
    /// Construct a loader for the file at `path`.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

#[async_trait]
impl DocumentLoader for MarkdownLoader {
    async fn load(&self) -> Result<DocumentStream> {
        let content = tokio::fs::read_to_string(&self.path).await.map_err(|e| {
            CognisError::Configuration(format!(
                "MarkdownLoader: read `{}`: {e}",
                self.path.display()
            ))
        })?;

        let doc = Document::new(content)
            .with_metadata("source", self.path.display().to_string())
            .with_metadata("format", "markdown");
        Ok(Box::pin(stream::iter(vec![Ok(doc)])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn loads_markdown_with_format_tag() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "# Heading\n\ntext").unwrap();
        let docs = MarkdownLoader::new(f.path()).load_all().await.unwrap();
        assert_eq!(docs[0].metadata["format"], "markdown");
        assert!(docs[0].content.contains("# Heading"));
    }
}
