use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::path::PathBuf;

use crate::error::Result;

/// A document for retrieval workflows (RAG, vector stores, semantic search).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Document {
    pub page_content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, Value>,
    /// Optional document type (e.g., "Document", "ChatDocument").
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_type: Option<String>,
}

impl Document {
    pub fn new(page_content: impl Into<String>) -> Self {
        Self {
            page_content: page_content.into(),
            id: None,
            metadata: HashMap::new(),
            doc_type: None,
        }
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, Value>) -> Self {
        self.metadata = metadata;
        self
    }

    pub fn with_doc_type(mut self, doc_type: impl Into<String>) -> Self {
        self.doc_type = Some(doc_type.into());
        self
    }
}

impl fmt::Display for Document {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.metadata.is_empty() {
            write!(f, "page_content='{}'", self.page_content)
        } else {
            write!(
                f,
                "page_content='{}' metadata={:?}",
                self.page_content, self.metadata
            )
        }
    }
}

/// Trait for compressing documents (e.g., extracting relevant passages for a query).
#[async_trait]
pub trait BaseDocumentCompressor: Send + Sync {
    async fn compress_documents(
        &self,
        documents: &[Document],
        query: &str,
    ) -> Result<Vec<Document>>;
}

/// Trait for transforming documents (e.g., splitting, translating).
#[async_trait]
pub trait BaseDocumentTransformer: Send + Sync {
    async fn transform_documents(&self, documents: Vec<Document>) -> Result<Vec<Document>>;
}

/// The data payload for a Blob.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum BlobData {
    Bytes(Vec<u8>),
    Text(String),
}

/// Raw data abstraction for document loading and file processing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Blob {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data: Option<BlobData>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mimetype: Option<String>,
    #[serde(default = "default_encoding")]
    pub encoding: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, Value>,
}

fn default_encoding() -> String {
    "utf-8".into()
}

impl Blob {
    /// Create a Blob from in-memory string data.
    pub fn from_string(data: impl Into<String>) -> Self {
        Self {
            data: Some(BlobData::Text(data.into())),
            path: None,
            mimetype: None,
            encoding: "utf-8".into(),
            id: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a Blob from in-memory byte data.
    pub fn from_bytes(data: Vec<u8>) -> Self {
        Self {
            data: Some(BlobData::Bytes(data)),
            path: None,
            mimetype: None,
            encoding: "utf-8".into(),
            id: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a Blob referencing a file path.
    pub fn from_path(path: impl Into<PathBuf>) -> Self {
        Self {
            data: None,
            path: Some(path.into()),
            mimetype: None,
            encoding: "utf-8".into(),
            id: None,
            metadata: HashMap::new(),
        }
    }

    pub fn with_mimetype(mut self, mimetype: impl Into<String>) -> Self {
        self.mimetype = Some(mimetype.into());
        self
    }

    /// Read data as a string. Returns an error if no data or path is available.
    pub fn as_string(&self) -> crate::error::Result<String> {
        match &self.data {
            Some(BlobData::Text(s)) => Ok(s.clone()),
            Some(BlobData::Bytes(b)) => String::from_utf8(b.clone())
                .map_err(|e| crate::error::CognisError::Other(e.to_string())),
            None => {
                if let Some(path) = &self.path {
                    std::fs::read_to_string(path)
                        .map_err(|e| crate::error::CognisError::Other(e.to_string()))
                } else {
                    Err(crate::error::CognisError::Other(
                        "Blob has no data or path".into(),
                    ))
                }
            }
        }
    }

    /// Read data as bytes. Returns an error if no data or path is available.
    pub fn as_bytes(&self) -> crate::error::Result<Vec<u8>> {
        match &self.data {
            Some(BlobData::Bytes(b)) => Ok(b.clone()),
            Some(BlobData::Text(s)) => Ok(s.as_bytes().to_vec()),
            None => {
                if let Some(path) = &self.path {
                    std::fs::read(path).map_err(|e| crate::error::CognisError::Other(e.to_string()))
                } else {
                    Err(crate::error::CognisError::Other(
                        "Blob has no data or path".into(),
                    ))
                }
            }
        }
    }

    /// Get the source location as a string.
    pub fn source(&self) -> Option<String> {
        if let Some(src) = self.metadata.get("source") {
            src.as_str().map(|s| s.to_string())
        } else {
            self.path.as_ref().map(|p| p.display().to_string())
        }
    }
}
