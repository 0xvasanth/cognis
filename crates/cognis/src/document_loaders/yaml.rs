//! YAML file document loader.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use cognis_core::document_loaders::BaseLoader;
use cognis_core::document_loaders::DocumentStream;
use cognis_core::documents::Document;
use cognis_core::error::{CognisError, Result};
use futures::stream;
use serde::Deserialize;
use serde_json::Value;

/// Source for the YAML content: either a file path or an inline string.
#[derive(Debug, Clone)]
enum YamlSource {
    File(PathBuf),
    String(String),
}

/// Loads a YAML file (or string) as one or more [`Document`]s.
///
/// Supports single-document and multi-document YAML (separated by `---`).
///
/// # Options
///
/// - **`content_key`**: If set, uses the value at this key as `page_content`
///   and puts remaining keys into metadata.
/// - **`metadata_keys`**: If set, only these keys are included in metadata.
///
/// # Example
/// ```no_run
/// use cognis::document_loaders::yaml::YamlDocumentLoader;
/// use cognis_core::document_loaders::BaseLoader;
///
/// # async fn example() -> cognis_core::error::Result<()> {
/// let loader = YamlDocumentLoader::from_file("data/config.yaml")
///     .with_content_key("description");
/// let docs = loader.load().await?;
/// # Ok(())
/// # }
/// ```
pub struct YamlDocumentLoader {
    source: YamlSource,
    /// Key whose value should be used as `page_content`.
    content_key: Option<String>,
    /// Specific keys to include in metadata.
    metadata_keys: Option<Vec<String>>,
}

impl YamlDocumentLoader {
    /// Create a new `YamlDocumentLoader` from a string path or content.
    ///
    /// If the source string ends with `.yaml` or `.yml`, it is treated as a
    /// file path; otherwise it is treated as inline YAML content.
    pub fn new(source: impl Into<String>) -> Self {
        let s: String = source.into();
        let yaml_source = if s.ends_with(".yaml") || s.ends_with(".yml") {
            YamlSource::File(PathBuf::from(&s))
        } else {
            YamlSource::String(s)
        };
        Self {
            source: yaml_source,
            content_key: None,
            metadata_keys: None,
        }
    }

    /// Create a `YamlDocumentLoader` from a file path.
    pub fn from_file(path: impl Into<PathBuf>) -> Self {
        Self {
            source: YamlSource::File(path.into()),
            content_key: None,
            metadata_keys: None,
        }
    }

    /// Create a `YamlDocumentLoader` from an inline YAML string.
    pub fn from_string(content: impl Into<String>) -> Self {
        Self {
            source: YamlSource::String(content.into()),
            content_key: None,
            metadata_keys: None,
        }
    }

    /// Set the key whose value becomes `page_content`.
    pub fn with_content_key(mut self, key: impl Into<String>) -> Self {
        self.content_key = Some(key.into());
        self
    }

    /// Set specific keys to include in metadata.
    pub fn with_metadata_keys(mut self, keys: Vec<String>) -> Self {
        self.metadata_keys = Some(keys);
        self
    }

    /// Read the raw YAML content from the source.
    async fn read_content(&self) -> Result<String> {
        match &self.source {
            YamlSource::File(path) => tokio::fs::read_to_string(path).await.map_err(Into::into),
            YamlSource::String(s) => Ok(s.clone()),
        }
    }

    /// Get a source label for metadata.
    fn source_label(&self) -> String {
        match &self.source {
            YamlSource::File(path) => path.display().to_string(),
            YamlSource::String(_) => "<string>".to_string(),
        }
    }

    /// Convert a serde_yaml Value to a serde_json Value.
    fn yaml_to_json(yaml_val: &serde_yaml::Value) -> Value {
        match yaml_val {
            serde_yaml::Value::Null => Value::Null,
            serde_yaml::Value::Bool(b) => Value::Bool(*b),
            serde_yaml::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    Value::Number(i.into())
                } else if let Some(u) = n.as_u64() {
                    Value::Number(u.into())
                } else if let Some(f) = n.as_f64() {
                    serde_json::Number::from_f64(f)
                        .map(Value::Number)
                        .unwrap_or(Value::Null)
                } else {
                    Value::Null
                }
            }
            serde_yaml::Value::String(s) => Value::String(s.clone()),
            serde_yaml::Value::Sequence(seq) => {
                Value::Array(seq.iter().map(Self::yaml_to_json).collect())
            }
            serde_yaml::Value::Mapping(map) => {
                let mut json_map = serde_json::Map::new();
                for (k, v) in map {
                    let key = match k {
                        serde_yaml::Value::String(s) => s.clone(),
                        other => format!("{:?}", other),
                    };
                    json_map.insert(key, Self::yaml_to_json(v));
                }
                Value::Object(json_map)
            }
            serde_yaml::Value::Tagged(tagged) => Self::yaml_to_json(&tagged.value),
        }
    }

    /// Convert a YAML value into a `Document`.
    fn value_to_document(
        &self,
        yaml_val: &serde_yaml::Value,
        source: &str,
        doc_index: usize,
    ) -> Document {
        let json_val = Self::yaml_to_json(yaml_val);

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::String(source.to_string()));
        metadata.insert("doc_index".to_string(), Value::Number(doc_index.into()));

        match &self.content_key {
            Some(key) => {
                // Extract content_key value as page_content.
                let content = json_val
                    .get(key)
                    .map(|v| match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_default();

                // Build metadata from remaining keys.
                if let Value::Object(map) = &json_val {
                    for (k, v) in map {
                        if k == key {
                            continue;
                        }
                        // Apply metadata_keys filter if set.
                        if let Some(ref allowed) = self.metadata_keys {
                            if !allowed.contains(k) {
                                continue;
                            }
                        }
                        metadata.insert(k.clone(), v.clone());
                    }
                }

                Document::new(content).with_metadata(metadata)
            }
            None => {
                // No content_key: serialize the whole value as page_content.
                let content = match &json_val {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };

                // If metadata_keys is set and the value is an object, extract them.
                if let (Some(ref keys), Value::Object(map)) = (&self.metadata_keys, &json_val) {
                    for k in keys {
                        if let Some(v) = map.get(k) {
                            metadata.insert(k.clone(), v.clone());
                        }
                    }
                }

                Document::new(content).with_metadata(metadata)
            }
        }
    }
}

#[async_trait]
impl BaseLoader for YamlDocumentLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let raw = self.read_content().await?;
        let source = self.source_label();

        let mut docs: Vec<Result<Document>> = Vec::new();

        // Use serde_yaml's multi-document deserializer.
        let mut doc_index = 0usize;
        for document in serde_yaml::Deserializer::from_str(&raw) {
            let yaml_val = serde_yaml::Value::deserialize(document).map_err(|e| {
                CognisError::Other(format!(
                    "Failed to parse YAML document {}: {}",
                    doc_index, e
                ))
            })?;

            // Skip null documents (empty docs between --- separators).
            if yaml_val.is_null() {
                continue;
            }

            docs.push(Ok(self.value_to_document(&yaml_val, &source, doc_index)));
            doc_index += 1;
        }

        Ok(Box::pin(stream::iter(docs)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn test_yaml_single_document() {
        let mut tmp = NamedTempFile::with_suffix(".yaml").unwrap();
        write!(tmp, "name: Alice\nage: 30\n").unwrap();

        let loader = YamlDocumentLoader::from_file(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert!(docs[0].page_content.contains("Alice"));
    }

    #[tokio::test]
    async fn test_yaml_multi_document() {
        let content = "name: Alice\nage: 30\n---\nname: Bob\nage: 25\n---\nname: Carol\nage: 35\n";
        let mut tmp = NamedTempFile::with_suffix(".yaml").unwrap();
        write!(tmp, "{}", content).unwrap();

        let loader = YamlDocumentLoader::from_file(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 3);
    }

    #[tokio::test]
    async fn test_yaml_content_key() {
        let content = "title: Hello World\nbody: This is the content\nauthor: Alice\n";
        let loader = YamlDocumentLoader::from_string(content).with_content_key("body");
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "This is the content");
        assert_eq!(
            docs[0].metadata.get("title").unwrap(),
            &Value::String("Hello World".to_string())
        );
        assert_eq!(
            docs[0].metadata.get("author").unwrap(),
            &Value::String("Alice".to_string())
        );
    }

    #[tokio::test]
    async fn test_yaml_metadata_keys() {
        let content = "title: Test\nbody: Content\nauthor: Bob\ndate: 2024-01-01\n";
        let loader = YamlDocumentLoader::from_string(content)
            .with_content_key("body")
            .with_metadata_keys(vec!["author".to_string()]);
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "Content");
        assert!(docs[0].metadata.contains_key("author"));
        // title and date should not be in metadata (not in allowed keys).
        assert!(!docs[0].metadata.contains_key("title"));
        assert!(!docs[0].metadata.contains_key("date"));
    }

    #[tokio::test]
    async fn test_yaml_from_string() {
        let content = "key: value\nnested:\n  inner: data\n";
        let loader = YamlDocumentLoader::from_string(content);
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].metadata.get("source").unwrap(),
            &Value::String("<string>".to_string())
        );
    }

    #[tokio::test]
    async fn test_yaml_empty_document() {
        let content = "";
        let loader = YamlDocumentLoader::from_string(content);
        let docs = loader.load().await.unwrap();

        // Empty content produces no documents (null is skipped).
        assert_eq!(docs.len(), 0);
    }

    #[tokio::test]
    async fn test_yaml_nested_structures() {
        let content = "database:\n  host: localhost\n  port: 5432\n  credentials:\n    user: admin\n    password: secret\n";
        let loader = YamlDocumentLoader::from_string(content).with_content_key("database");
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        // content_key on a nested object serializes it as JSON.
        assert!(docs[0].page_content.contains("localhost"));
        assert!(docs[0].page_content.contains("5432"));
    }

    #[tokio::test]
    async fn test_yaml_new_auto_detect_file() {
        let mut tmp = NamedTempFile::with_suffix(".yaml").unwrap();
        write!(tmp, "key: value\n").unwrap();

        let loader = YamlDocumentLoader::new(tmp.path().display().to_string());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
    }

    #[tokio::test]
    async fn test_yaml_new_auto_detect_string() {
        let loader = YamlDocumentLoader::new("key: value\nnumber: 42\n");
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert!(docs[0].page_content.contains("42"));
    }

    #[tokio::test]
    async fn test_yaml_multi_doc_with_content_key() {
        let content = "name: Alice\nrole: admin\n---\nname: Bob\nrole: user\n";
        let loader = YamlDocumentLoader::from_string(content).with_content_key("name");
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].page_content, "Alice");
        assert_eq!(docs[1].page_content, "Bob");
        assert_eq!(
            docs[0].metadata.get("role").unwrap(),
            &Value::String("admin".to_string())
        );
    }
}
