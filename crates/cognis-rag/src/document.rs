//! `Document` — the unit of RAG: a piece of text plus typed metadata.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A piece of text that flows through loaders → splitters → embeddings →
/// vector store → retrievers.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Document {
    /// Optional ID — set by stores once persisted, by loaders that have a
    /// natural key (e.g. file path), or left `None` for ephemeral chunks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,

    /// The text content. Loaders fill this from the source; splitters carve
    /// it into chunks.
    pub content: String,

    /// Free-form metadata (e.g. `{ "source": "file.txt", "page": 3 }`).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
}

impl Document {
    /// Construct from text only.
    pub fn new(content: impl Into<String>) -> Self {
        Self {
            id: None,
            content: content.into(),
            metadata: HashMap::new(),
        }
    }

    /// Set the ID (builder-style).
    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Insert one metadata key (builder-style).
    pub fn with_metadata(
        mut self,
        key: impl Into<String>,
        value: impl Into<serde_json::Value>,
    ) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Replace metadata wholesale (builder-style).
    pub fn with_metadata_map(mut self, m: HashMap<String, serde_json::Value>) -> Self {
        self.metadata = m;
        self
    }
}

impl From<String> for Document {
    fn from(s: String) -> Self {
        Self::new(s)
    }
}

impl From<&str> for Document {
    fn from(s: &str) -> Self {
        Self::new(s.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_chains() {
        let d = Document::new("hello")
            .with_id("doc-1")
            .with_metadata("source", "file.txt");
        assert_eq!(d.id.as_deref(), Some("doc-1"));
        assert_eq!(d.content, "hello");
        assert_eq!(d.metadata["source"], "file.txt");
    }

    #[test]
    fn from_str_works() {
        let d: Document = "hello".into();
        assert_eq!(d.content, "hello");
        assert!(d.metadata.is_empty());
    }

    #[test]
    fn serde_skips_optional_when_empty() {
        let d = Document::new("x");
        let s = serde_json::to_string(&d).unwrap();
        assert!(!s.contains("\"id\""));
        assert!(!s.contains("\"metadata\""));
    }
}
