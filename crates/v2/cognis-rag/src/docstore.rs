//! `Docstore` — keyed [`Document`] storage by stable id.
//!
//! Used by the parent-document retriever pattern: small chunks live in a
//! `VectorStore` (similarity search), full parent docs live in a
//! `Docstore` (id lookup).

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use cognis2_core::Result;

use crate::document::Document;

/// Pluggable id-keyed document store.
#[async_trait]
pub trait Docstore: Send + Sync {
    /// Persist `(id, doc)` pairs. Replaces existing entries.
    async fn put(&self, items: Vec<(String, Document)>) -> Result<()>;
    /// Look up by id; missing ids are simply absent from the result.
    async fn get(&self, ids: &[String]) -> Result<Vec<Document>>;
    /// Delete by id; missing ids are silently ignored.
    async fn delete(&self, ids: &[String]) -> Result<()>;
}

/// In-process [`Docstore`] backed by a `Mutex<HashMap>`.
#[derive(Default)]
pub struct InMemoryDocstore {
    inner: Mutex<HashMap<String, Document>>,
}

impl InMemoryDocstore {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl Docstore for InMemoryDocstore {
    async fn put(&self, items: Vec<(String, Document)>) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| cognis2_core::CognisError::Internal(format!("docstore mutex: {e}")))?;
        for (id, d) in items {
            inner.insert(id, d);
        }
        Ok(())
    }
    async fn get(&self, ids: &[String]) -> Result<Vec<Document>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| cognis2_core::CognisError::Internal(format!("docstore mutex: {e}")))?;
        Ok(ids.iter().filter_map(|id| inner.get(id).cloned()).collect())
    }
    async fn delete(&self, ids: &[String]) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| cognis2_core::CognisError::Internal(format!("docstore mutex: {e}")))?;
        for id in ids {
            inner.remove(id);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn put_then_get() {
        let s = InMemoryDocstore::new();
        s.put(vec![
            ("a".into(), Document::new("alpha")),
            ("b".into(), Document::new("beta")),
        ])
        .await
        .unwrap();
        let docs = s
            .get(&["a".into(), "missing".into(), "b".into()])
            .await
            .unwrap();
        let contents: Vec<_> = docs.iter().map(|d| d.content.clone()).collect();
        assert!(contents.contains(&"alpha".to_string()));
        assert!(contents.contains(&"beta".to_string()));
        assert_eq!(contents.len(), 2);
    }

    #[tokio::test]
    async fn delete_removes() {
        let s = InMemoryDocstore::new();
        s.put(vec![("a".into(), Document::new("alpha"))])
            .await
            .unwrap();
        s.delete(&["a".into()]).await.unwrap();
        assert!(s.get(&["a".into()]).await.unwrap().is_empty());
    }
}
