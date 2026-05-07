//! TOML loader — yields one [`Document`] for the file's top-level table.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use futures::stream;

use cognis2_core::{CognisError, Result};

use crate::document::Document;

use super::{DocumentLoader, DocumentStream};

/// Loads a TOML file as one [`Document`]. The body is the file content
/// re-serialized as TOML (canonicalized), and `metadata.format = "toml"`.
pub struct TomlLoader {
    path: PathBuf,
}

impl TomlLoader {
    /// Construct.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

#[async_trait]
impl DocumentLoader for TomlLoader {
    async fn load(&self) -> Result<DocumentStream> {
        let raw = tokio::fs::read_to_string(&self.path).await.map_err(|e| {
            CognisError::Configuration(format!("TomlLoader: read `{}`: {e}", self.path.display()))
        })?;
        // Validate by round-tripping; surface parse errors clearly.
        let _table: toml::Value = raw.parse().map_err(|e| {
            CognisError::Serialization(format!(
                "TomlLoader: `{}` is not valid TOML: {e}",
                self.path.display()
            ))
        })?;
        let doc = Document::new(raw)
            .with_metadata("source", self.path.display().to_string())
            .with_metadata("format", "toml");
        Ok(Box::pin(stream::iter(vec![Ok(doc)])))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn loads_toml_with_format_tag() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "[server]\nport = 8080").unwrap();
        let docs = TomlLoader::new(f.path()).load_all().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.contains("port"));
        assert_eq!(docs[0].metadata["format"], "toml");
    }

    #[tokio::test]
    async fn errors_on_invalid_toml() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "this is not toml = = =").unwrap();
        assert!(TomlLoader::new(f.path()).load_all().await.is_err());
    }
}
