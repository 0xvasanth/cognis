//! JSON loader — yields one [`Document`] per top-level array element, or a
//! single document for a non-array root.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use futures::stream;
use serde_json::Value;

use cognis2_core::{CognisError, Result};

use crate::document::Document;

use super::{DocumentLoader, DocumentStream};

/// Loads a JSON file. Behaviour:
///
/// - **Array root**: yields one [`Document`] per element.
///   - If `content_pointer` is set, extracts the doc text from that JSON
///     pointer (RFC 6901, e.g. `/text`); the rest of the element becomes
///     `metadata`.
///   - Otherwise, the element is serialized to a string and used as content.
/// - **Object root**: same rule, but yields exactly one document.
/// - **Scalar root**: one document with the scalar's string form.
pub struct JsonLoader {
    path: PathBuf,
    content_pointer: Option<String>,
}

impl JsonLoader {
    /// Construct a loader for the file at `path`.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            content_pointer: None,
        }
    }

    /// Pull document content from a JSON pointer (e.g. `/text` or `/body/0`).
    /// Required only when the JSON has a structured shape with a designated
    /// content field.
    pub fn with_content_pointer(mut self, pointer: impl Into<String>) -> Self {
        self.content_pointer = Some(pointer.into());
        self
    }
}

#[async_trait]
impl DocumentLoader for JsonLoader {
    async fn load(&self) -> Result<DocumentStream> {
        let bytes = tokio::fs::read(&self.path).await.map_err(|e| {
            CognisError::Configuration(format!("JsonLoader: read `{}`: {e}", self.path.display()))
        })?;
        let root: Value = serde_json::from_slice(&bytes).map_err(|e| {
            CognisError::Serialization(format!(
                "JsonLoader: `{}` is not valid JSON: {e}",
                self.path.display()
            ))
        })?;

        let source = self.path.display().to_string();
        let docs = match root {
            Value::Array(items) => items
                .into_iter()
                .map(|v| build_doc(v, self.content_pointer.as_deref(), &source))
                .collect::<Result<Vec<_>>>()?,
            other => vec![build_doc(other, self.content_pointer.as_deref(), &source)?],
        };

        Ok(Box::pin(stream::iter(docs.into_iter().map(Ok))))
    }
}

fn build_doc(v: Value, pointer: Option<&str>, source: &str) -> Result<Document> {
    let (content, mut metadata_value) = match (pointer, v) {
        (Some(p), Value::Object(mut obj)) => {
            // RFC 6901 pointer descent.
            let extracted = take_at_pointer(&mut obj, p).ok_or_else(|| {
                CognisError::Configuration(format!(
                    "JsonLoader: pointer `{p}` did not resolve in object"
                ))
            })?;
            let content = match extracted {
                Value::String(s) => s,
                v => v.to_string(),
            };
            (content, Value::Object(obj))
        }
        (Some(_), other) => (other.to_string(), Value::Null),
        (None, Value::String(s)) => (s, Value::Null),
        (None, other) => (other.to_string(), Value::Null),
    };

    let mut doc = Document::new(content).with_metadata("source", source.to_string());
    if let Value::Object(map) = std::mem::take(&mut metadata_value) {
        for (k, v) in map {
            doc.metadata.insert(k, v);
        }
    }
    Ok(doc)
}

/// Walk a `serde_json::Map` along an RFC 6901-style `/a/b/c` pointer,
/// removing and returning the leaf. Only handles the simple case of
/// object descent; we ignore `~0`/`~1` escape sequences.
fn take_at_pointer(obj: &mut serde_json::Map<String, Value>, pointer: &str) -> Option<Value> {
    let mut parts = pointer.trim_start_matches('/').split('/');
    let first = parts.next()?;
    let mut cur_key = first.to_string();
    let mut cur_obj = obj;
    for next in parts {
        let v = cur_obj.get_mut(&cur_key)?;
        match v {
            Value::Object(m) => {
                cur_obj = m;
                cur_key = next.to_string();
            }
            _ => return None,
        }
    }
    cur_obj.remove(&cur_key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn loads_array_root_one_doc_per_element() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(
            f,
            r#"[{{"text":"hi","author":"a"}}, {{"text":"hello","author":"b"}}]"#
        )
        .unwrap();
        let docs = JsonLoader::new(f.path())
            .with_content_pointer("/text")
            .load_all()
            .await
            .unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].content, "hi");
        assert_eq!(docs[0].metadata["author"], "a");
        assert_eq!(docs[1].content, "hello");
    }

    #[tokio::test]
    async fn loads_scalar_root() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "\"hello\"").unwrap();
        let docs = JsonLoader::new(f.path()).load_all().await.unwrap();
        assert_eq!(docs[0].content, "hello");
    }

    #[tokio::test]
    async fn errors_on_invalid_json() {
        let mut f = NamedTempFile::new().unwrap();
        writeln!(f, "not json").unwrap();
        assert!(JsonLoader::new(f.path()).load_all().await.is_err());
    }
}
