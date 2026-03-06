//! TOML file document loader.

use std::collections::HashMap;
use std::path::PathBuf;

use async_trait::async_trait;
use futures::stream;
use rustchain_core::document_loaders::BaseLoader;
use rustchain_core::document_loaders::DocumentStream;
use rustchain_core::documents::Document;
use rustchain_core::error::{Result, RustChainError};
use serde_json::Value;

/// Source for the TOML content: either a file path or an inline string.
#[derive(Debug, Clone)]
enum TomlSource {
    File(PathBuf),
    String(String),
}

/// Loads a TOML file (or string) as one or more [`Document`]s.
///
/// # Modes
///
/// - **Single document** (default): The entire TOML file becomes one `Document`.
/// - **Section mode** (`section_mode = true`): Each top-level table becomes a
///   separate `Document`, with the table name included in metadata.
///
/// # Options
///
/// - **`content_key`**: If set, uses the value at this key as `page_content`.
/// - **`section_mode`**: If true, splits by top-level tables.
///
/// # Example
/// ```no_run
/// use rustchain::document_loaders::toml_loader::TomlDocumentLoader;
/// use rustchain_core::document_loaders::BaseLoader;
///
/// # async fn example() -> rustchain_core::error::Result<()> {
/// let loader = TomlDocumentLoader::from_file("Cargo.toml")
///     .with_section_mode(true);
/// let docs = loader.load().await?;
/// # Ok(())
/// # }
/// ```
pub struct TomlDocumentLoader {
    source: TomlSource,
    /// Key whose value should be used as `page_content`.
    content_key: Option<String>,
    /// If true, each top-level table becomes a separate Document.
    section_mode: bool,
}

impl TomlDocumentLoader {
    /// Create a `TomlDocumentLoader` from a file path.
    pub fn from_file(path: impl Into<PathBuf>) -> Self {
        Self {
            source: TomlSource::File(path.into()),
            content_key: None,
            section_mode: false,
        }
    }

    /// Create a `TomlDocumentLoader` from an inline TOML string.
    pub fn from_string(content: impl Into<String>) -> Self {
        Self {
            source: TomlSource::String(content.into()),
            content_key: None,
            section_mode: false,
        }
    }

    /// Set the key whose value becomes `page_content`.
    pub fn with_content_key(mut self, key: impl Into<String>) -> Self {
        self.content_key = Some(key.into());
        self
    }

    /// Enable or disable section mode.
    ///
    /// When enabled, each top-level table in the TOML becomes a separate
    /// `Document` with the table name in metadata.
    pub fn with_section_mode(mut self, enabled: bool) -> Self {
        self.section_mode = enabled;
        self
    }

    /// Read the raw TOML content from the source.
    async fn read_content(&self) -> Result<String> {
        match &self.source {
            TomlSource::File(path) => {
                tokio::fs::read_to_string(path).await.map_err(Into::into)
            }
            TomlSource::String(s) => Ok(s.clone()),
        }
    }

    /// Get a source label for metadata.
    fn source_label(&self) -> String {
        match &self.source {
            TomlSource::File(path) => path.display().to_string(),
            TomlSource::String(_) => "<string>".to_string(),
        }
    }

    /// Convert a toml::Value to a serde_json::Value.
    fn toml_to_json(toml_val: &toml::Value) -> Value {
        match toml_val {
            toml::Value::String(s) => Value::String(s.clone()),
            toml::Value::Integer(i) => Value::Number((*i).into()),
            toml::Value::Float(f) => {
                serde_json::Number::from_f64(*f)
                    .map(Value::Number)
                    .unwrap_or(Value::Null)
            }
            toml::Value::Boolean(b) => Value::Bool(*b),
            toml::Value::Datetime(dt) => Value::String(dt.to_string()),
            toml::Value::Array(arr) => {
                Value::Array(arr.iter().map(Self::toml_to_json).collect())
            }
            toml::Value::Table(table) => {
                let mut map = serde_json::Map::new();
                for (k, v) in table {
                    map.insert(k.clone(), Self::toml_to_json(v));
                }
                Value::Object(map)
            }
        }
    }

    /// Convert a toml::Value into a `Document`, optionally extracting `content_key`.
    fn value_to_document(
        &self,
        toml_val: &toml::Value,
        source: &str,
        section_name: Option<&str>,
    ) -> Document {
        let json_val = Self::toml_to_json(toml_val);

        let mut metadata = HashMap::new();
        metadata.insert("source".to_string(), Value::String(source.to_string()));
        if let Some(name) = section_name {
            metadata.insert("section".to_string(), Value::String(name.to_string()));
        }

        match &self.content_key {
            Some(key) => {
                let content = json_val
                    .get(key)
                    .map(|v| match v {
                        Value::String(s) => s.clone(),
                        other => other.to_string(),
                    })
                    .unwrap_or_default();

                // Put remaining keys into metadata.
                if let Value::Object(map) = &json_val {
                    for (k, v) in map {
                        if k != key {
                            metadata.insert(k.clone(), v.clone());
                        }
                    }
                }

                Document::new(content).with_metadata(metadata)
            }
            None => {
                let content = match &json_val {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                Document::new(content).with_metadata(metadata)
            }
        }
    }
}

#[async_trait]
impl BaseLoader for TomlDocumentLoader {
    async fn lazy_load(&self) -> Result<DocumentStream> {
        let raw = self.read_content().await?;
        let source = self.source_label();

        let root: toml::Value = raw.parse().map_err(|e: toml::de::Error| {
            RustChainError::Other(format!("Failed to parse TOML: {}", e))
        })?;

        let docs: Vec<Result<Document>> = if self.section_mode {
            // Each top-level table becomes a separate document.
            match &root {
                toml::Value::Table(table) => {
                    table
                        .iter()
                        .map(|(key, val)| {
                            Ok(self.value_to_document(val, &source, Some(key)))
                        })
                        .collect()
                }
                _ => {
                    vec![Ok(self.value_to_document(&root, &source, None))]
                }
            }
        } else {
            vec![Ok(self.value_to_document(&root, &source, None))]
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
    async fn test_toml_basic_load() {
        let content = r#"
title = "My Config"
version = "1.0"
"#;
        let mut tmp = NamedTempFile::with_suffix(".toml").unwrap();
        write!(tmp, "{}", content).unwrap();

        let loader = TomlDocumentLoader::from_file(tmp.path());
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert!(docs[0].page_content.contains("My Config"));
    }

    #[tokio::test]
    async fn test_toml_section_mode() {
        let content = r#"
[package]
name = "myapp"
version = "0.1.0"

[dependencies]
serde = "1.0"
tokio = "1.0"

[dev-dependencies]
tempfile = "3"
"#;
        let loader = TomlDocumentLoader::from_string(content)
            .with_section_mode(true);
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 3);

        // Check that section names are in metadata.
        let sections: Vec<&str> = docs
            .iter()
            .map(|d| d.metadata.get("section").unwrap().as_str().unwrap())
            .collect();
        assert!(sections.contains(&"package"));
        assert!(sections.contains(&"dependencies"));
        assert!(sections.contains(&"dev-dependencies"));
    }

    #[tokio::test]
    async fn test_toml_content_key() {
        let content = r#"
name = "test-project"
description = "A test project"
version = "1.0.0"
"#;
        let loader = TomlDocumentLoader::from_string(content)
            .with_content_key("description");
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "A test project");
        assert_eq!(
            docs[0].metadata.get("name").unwrap(),
            &Value::String("test-project".to_string())
        );
    }

    #[tokio::test]
    async fn test_toml_from_string() {
        let content = "key = \"value\"\n";
        let loader = TomlDocumentLoader::from_string(content);
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].metadata.get("source").unwrap(),
            &Value::String("<string>".to_string())
        );
    }

    #[tokio::test]
    async fn test_toml_nested_tables() {
        let content = r#"
[server]
host = "localhost"
port = 8080

[server.tls]
enabled = true
cert = "/path/to/cert"
"#;
        let loader = TomlDocumentLoader::from_string(content)
            .with_section_mode(true);
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1); // "server" is the only top-level table.
        let doc = &docs[0];
        assert!(doc.page_content.contains("localhost"));
        assert!(doc.page_content.contains("8080"));
        // Nested tls should be within the server section.
        assert!(doc.page_content.contains("tls"));
    }

    #[tokio::test]
    async fn test_toml_empty_file() {
        let content = "";
        let loader = TomlDocumentLoader::from_string(content);
        let docs = loader.load().await.unwrap();

        // Empty TOML parses as an empty table.
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "{}");
    }

    #[tokio::test]
    async fn test_toml_metadata_extraction() {
        let content = r#"
[package]
name = "example"
version = "2.0"
edition = "2021"
"#;
        let loader = TomlDocumentLoader::from_string(content)
            .with_section_mode(true)
            .with_content_key("name");
        let docs = loader.load().await.unwrap();

        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "example");
        assert_eq!(
            docs[0].metadata.get("version").unwrap(),
            &Value::String("2.0".to_string())
        );
        assert_eq!(
            docs[0].metadata.get("edition").unwrap(),
            &Value::String("2021".to_string())
        );
        assert_eq!(
            docs[0].metadata.get("section").unwrap(),
            &Value::String("package".to_string())
        );
    }

    #[tokio::test]
    async fn test_toml_section_mode_with_scalars() {
        // Top-level scalars mixed with tables in section mode.
        let content = r#"
title = "root value"

[settings]
debug = true
"#;
        let loader = TomlDocumentLoader::from_string(content)
            .with_section_mode(true);
        let docs = loader.load().await.unwrap();

        // "title" and "settings" are both top-level keys.
        assert_eq!(docs.len(), 2);
    }
}
