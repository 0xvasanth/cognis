//! YAML loader — yields one [`Document`] per top-level array element, or a
//! single document for any other root.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use futures::stream;

use cognis_core::{CognisError, Result};

use crate::document::Document;

use super::{DocumentLoader, DocumentStream};

/// Loads a YAML file. Same shape rules as [`super::JsonLoader`]:
/// - Sequence root → one document per element.
/// - Mapping/scalar root → exactly one document.
/// - Each document's content is the YAML element re-serialized as YAML.
pub struct YamlLoader {
    path: PathBuf,
}

impl YamlLoader {
    /// Construct a loader.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }
}

#[async_trait]
impl DocumentLoader for YamlLoader {
    async fn load(&self) -> Result<DocumentStream> {
        let bytes = tokio::fs::read(&self.path).await.map_err(|e| {
            CognisError::Configuration(format!("YamlLoader: read `{}`: {e}", self.path.display()))
        })?;
        let value: serde_yaml::Value = serde_yaml::from_slice(&bytes).map_err(|e| {
            CognisError::Serialization(format!(
                "YamlLoader: `{}` is not valid YAML: {e}",
                self.path.display()
            ))
        })?;
        let source = self.path.display().to_string();
        let docs: Vec<Document> = match value {
            serde_yaml::Value::Sequence(items) => items
                .into_iter()
                .map(|v| build_doc(v, &source))
                .collect::<Result<Vec<_>>>()?,
            other => vec![build_doc(other, &source)?],
        };
        Ok(Box::pin(stream::iter(docs.into_iter().map(Ok))))
    }
}

fn build_doc(v: serde_yaml::Value, source: &str) -> Result<Document> {
    let content = serde_yaml::to_string(&v)
        .map_err(|e| CognisError::Serialization(format!("YamlLoader: serialize: {e}")))?;
    Ok(Document::new(content)
        .with_metadata("source", source.to_string())
        .with_metadata("format", "yaml"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn loads_sequence_root_one_doc_per_element() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "- name: a\n- name: b\n").unwrap();
        let docs = YamlLoader::new(f.path()).load_all().await.unwrap();
        assert_eq!(docs.len(), 2);
        assert!(docs[0].content.contains("name: a"));
        assert!(docs[1].content.contains("name: b"));
    }

    #[tokio::test]
    async fn loads_mapping_root_one_doc() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "name: world\nver: 1").unwrap();
        let docs = YamlLoader::new(f.path()).load_all().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.contains("name: world"));
        assert_eq!(docs[0].metadata["format"], "yaml");
    }
}
