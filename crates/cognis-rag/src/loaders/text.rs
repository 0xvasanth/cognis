//! Plain-text file loader.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use futures::stream;

use cognis_core::{CognisError, Result};

use crate::document::Document;

use super::{DocumentLoader, DocumentStream};

/// Loads a single UTF-8 text file as one [`Document`].
///
/// `metadata.source` is set to the absolute (or as-given) path.
pub struct TextLoader {
    path: PathBuf,
    encoding_check: bool,
}

impl TextLoader {
    /// Construct a loader for the file at `path`.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            encoding_check: true,
        }
    }

    /// Skip the UTF-8 validity check (loader will lossily decode).
    pub fn lossy(mut self) -> Self {
        self.encoding_check = false;
        self
    }
}

#[async_trait]
impl DocumentLoader for TextLoader {
    async fn load(&self) -> Result<DocumentStream> {
        let bytes = tokio::fs::read(&self.path).await.map_err(|e| {
            CognisError::Configuration(format!("TextLoader: read `{}`: {e}", self.path.display()))
        })?;
        let content = if self.encoding_check {
            String::from_utf8(bytes).map_err(|e| {
                CognisError::Serialization(format!(
                    "TextLoader: `{}` is not valid UTF-8: {e}",
                    self.path.display()
                ))
            })?
        } else {
            String::from_utf8_lossy(&bytes).into_owned()
        };

        let doc = Document::new(content).with_metadata("source", self.path.display().to_string());
        Ok(Box::pin(stream::iter(vec![Ok(doc)])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn loads_text_file() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "hello").unwrap();
        let loader = TextLoader::new(f.path());
        let docs = loader.load_all().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.contains("hello"));
        assert!(docs[0].metadata.contains_key("source"));
    }

    #[tokio::test]
    async fn missing_file_errors() {
        let loader = TextLoader::new("/no/such/file/__test__");
        assert!(loader.load_all().await.is_err());
    }
}
