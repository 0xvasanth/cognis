//! In-memory document store for use with parent-document and multi-vector retrievers.

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::RwLock;

use rustchain_core::documents::Document;

/// A simple in-memory document store backed by a `HashMap`.
///
/// Thread-safe via `Arc<RwLock<...>>`. Intended for use with retrievers that
/// need to store full documents separately from vector-store chunks (e.g.,
/// [`ParentDocumentRetriever`](super::parent_document::ParentDocumentRetriever),
/// [`MultiVectorRetriever`](super::multi_vector::MultiVectorRetriever)).
#[derive(Clone)]
pub struct InMemoryDocStore {
    store: Arc<RwLock<HashMap<String, Document>>>,
}

impl InMemoryDocStore {
    /// Create an empty document store.
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Add a document with the given ID.
    ///
    /// If a document with the same ID already exists, it is overwritten.
    pub async fn add(&self, id: impl Into<String>, doc: Document) {
        let mut store = self.store.write().await;
        store.insert(id.into(), doc);
    }

    /// Retrieve a document by its ID.
    ///
    /// Returns `None` if no document with that ID exists.
    pub async fn get(&self, id: &str) -> Option<Document> {
        let store = self.store.read().await;
        store.get(id).cloned()
    }

    /// Delete a document by its ID.
    ///
    /// Returns `true` if a document was removed.
    pub async fn delete(&self, id: &str) -> bool {
        let mut store = self.store.write().await;
        store.remove(id).is_some()
    }

    /// Retrieve multiple documents by their IDs.
    ///
    /// Missing IDs are returned as `None` in the output vector.
    pub async fn mget(&self, ids: &[String]) -> Vec<Option<Document>> {
        let store = self.store.read().await;
        ids.iter().map(|id| store.get(id).cloned()).collect()
    }

    /// List all document IDs in the store.
    pub async fn list_keys(&self) -> Vec<String> {
        let store = self.store.read().await;
        store.keys().cloned().collect()
    }
}

impl Default for InMemoryDocStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_add_and_get() {
        let store = InMemoryDocStore::new();
        let doc = Document::new("hello world");
        store.add("doc1", doc.clone()).await;

        let retrieved = store.get("doc1").await;
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().page_content, "hello world");
    }

    #[tokio::test]
    async fn test_get_missing() {
        let store = InMemoryDocStore::new();
        assert!(store.get("nonexistent").await.is_none());
    }

    #[tokio::test]
    async fn test_delete() {
        let store = InMemoryDocStore::new();
        store.add("doc1", Document::new("content")).await;

        assert!(store.delete("doc1").await);
        assert!(store.get("doc1").await.is_none());
        assert!(!store.delete("doc1").await);
    }

    #[tokio::test]
    async fn test_mget() {
        let store = InMemoryDocStore::new();
        store.add("a", Document::new("alpha")).await;
        store.add("b", Document::new("beta")).await;

        let results = store
            .mget(&["a".into(), "missing".into(), "b".into()])
            .await;
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().unwrap().page_content, "alpha");
        assert!(results[1].is_none());
        assert_eq!(results[2].as_ref().unwrap().page_content, "beta");
    }

    #[tokio::test]
    async fn test_list_keys() {
        let store = InMemoryDocStore::new();
        store.add("x", Document::new("x")).await;
        store.add("y", Document::new("y")).await;

        let mut keys = store.list_keys().await;
        keys.sort();
        assert_eq!(keys, vec!["x", "y"]);
    }
}
