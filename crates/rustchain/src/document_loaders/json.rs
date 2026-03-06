//! JSON file document loader.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use futures::stream;
use rustchain_core::document_loaders::BaseLoader;
use rustchain_core::document_loaders::DocumentStream;
use rustchain_core::documents::Document;
use rustchain_core::error::{Result, RustChainError};
use serde_json::Value;

/// Loads a JSON file as one or more [`Document`]s.
///
/// Supports three modes:
/// - **Full document**: without `jq_path`, the entire JSON becomes a single document.
/// - **Array extraction**: with `jq_path` (e.g. `".data.items"`), navigates to an array
///   and creates one document per element.
/// - **Text key**: with `text_key`, uses that key's value as `page_content` and the
///   remaining keys as metadata.
///
/// # Example
/// ```no_run
/// use rustchain::document_loaders::json::JsonLoader;
/// use rustchain_core::document_loaders::BaseLoader;
///
/// # async fn example() -> rustchain_core::error::Result<()> {
/// let loader = JsonLoader::new("data/records.json")
///     .with_jq_path(".records")
///     .with_text_key("body");
/// let docs = loader.load().await?;
/// # Ok(())
/// # }
/// ```
pub struct JsonLoader {
    path: PathBuf,
    /// Dot-separated path to navigate into the JSON (e.g. `.data.items`).
    jq_path: Option<String>,
    /// Key whose value should be used as `page_content`.
    text_key: Option<String>,
}

impl JsonLoader {
    /// Create a new `JsonLoader` for the given file path.
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            jq_path: None,
            text_key: None,
        }
    }

    /// Set a dot-path to navigate into the JSON structure.
    ///
    /// The path should start with `.` and use `.` to separate keys,
    /// e.g. `".data.items"`.
    pub fn with_jq_path(mut self, jq_path: impl Into<String>) -> Self {
        self.jq_path = Some(jq_path.into());
        self
    }

    /// Set the key whose value becomes `page_content`.
    pub fn with_text_key(mut self, text_key: impl Into<String>) -> Self {
        self.text_key = Some(text_key.into());
        self
    }

    /// Navigate into a JSON value using a dot-separated path.
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

    /// Convert a JSON value into a `Document`, optionally extracting `text_key`.
    fn value_to_document(&self, value: &Value, source: &str) -> Document {
        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::String(source.to_string()));

        match &self.text_key {
            Some(key) => {
                let content = value
                    .get(key)
                    .and_then(|v| match v {
                        Value::String(s) => Some(s.clone()),
                        other => Some(other.to_string()),
                    })
                    .unwrap_or_default();

                // Put remaining keys into metadata.
                if let Value::Object(map) = value {
                    for (k, v) in map {
                        if k != key {
                            metadata.insert(k.clone(), v.clone());
                        }
                    }
                }

                Document::new(content).with_metadata(metadata)
            }
            None => {
                let content = match value {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                Document::new(content).with_metadata(metadata)
            }
        }
    }
}

#[async_trait]
impl BaseLoader for JsonLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let raw = tokio::fs::read_to_string(&self.path).await?;
        let root: Value = serde_json::from_str(&raw)?;
        let source = self.path.display().to_string();

        let target = self.navigate(&root)?;

        let docs: Vec<Result<Document>> = match target {
            Value::Array(arr) => arr
                .iter()
                .map(|v| Ok(self.value_to_document(v, &source)))
                .collect(),
            other => vec![Ok(self.value_to_document(other, &source))],
        };

        Ok(Box::pin(stream::iter(docs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_json_loader_single() {
        let mut tmp = NamedTempFile::with_suffix(".json").unwrap();
        write!(tmp, r#"{{"name":"Alice","age":30}}"#).unwrap();

        let loader = JsonLoader::new(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert!(docs[0].page_content.contains("Alice"));
    }

    #[tokio::test]
    async fn test_json_loader_with_path() {
        let mut tmp = NamedTempFile::with_suffix(".json").unwrap();
        let data = r#"{"data":{"items":[{"text":"hello","id":1},{"text":"world","id":2}]}}"#;
        write!(tmp, "{}", data).unwrap();

        let loader = JsonLoader::new(tmp.path())
            .with_jq_path(".data.items")
            .with_text_key("text");
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].page_content, "hello");
        assert_eq!(docs[1].page_content, "world");
        // id should be in metadata
        assert_eq!(docs[0].metadata.get("id").unwrap(), &Value::Number(1.into()));
    }
}
