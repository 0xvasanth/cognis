//! In-memory document index.
//!
//! Mirrors Python `langchain_core.indexing.in_memory`.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::documents::Document;
use crate::error::Result;

use super::base::{DeleteResponse, DocumentIndex, UpsertResponse};

/// In-memory document index for testing and lightweight usage.
///
/// Stores documents in a `HashMap` keyed by document ID.
/// Provides a simple search API that returns documents by the number of
/// times the given query appears in the document content.
pub struct InMemoryDocumentIndex {
    store: Mutex<HashMap<String, Document>>,
    top_k: usize,
}

impl InMemoryDocumentIndex {
    /// Create a new in-memory document index.
    pub fn new() -> Self {
        Self {
            store: Mutex::new(HashMap::new()),
            top_k: 4,
        }
    }

    /// Create a new in-memory document index with a custom top_k.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Return the number of documents in the index.
    pub fn len(&self) -> usize {
        self.store.lock().unwrap().len()
    }

    /// Return whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.store.lock().unwrap().is_empty()
    }

    /// Search documents by counting occurrences of the query string.
    ///
    /// Returns up to `top_k` documents sorted by relevance (occurrence count).
    pub fn search(&self, query: &str) -> Vec<Document> {
        let store = self.store.lock().unwrap();
        let mut counts: Vec<(Document, usize)> = store
            .values()
            .map(|doc| {
                let count = doc.page_content.matches(query).count();
                (doc.clone(), count)
            })
            .collect();

        counts.sort_by(|a, b| b.1.cmp(&a.1));
        counts
            .into_iter()
            .take(self.top_k)
            .map(|(doc, _)| doc)
            .collect()
    }
}

impl Default for InMemoryDocumentIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl DocumentIndex for InMemoryDocumentIndex {
    /// Upsert documents into the index.
    ///
    /// If a document has no ID, a UUID is generated for it.
    async fn upsert(&self, docs: Vec<Document>) -> Result<UpsertResponse> {
        let mut store = self.store.lock().unwrap();
        let mut succeeded = Vec::new();

        for doc in docs {
            let id = match &doc.id {
                Some(id) => id.clone(),
                None => uuid::Uuid::new_v4().to_string(),
            };

            let mut stored_doc = doc;
            if stored_doc.id.is_none() {
                stored_doc.id = Some(id.clone());
            }

            store.insert(id.clone(), stored_doc);
            succeeded.push(id);
        }

        Ok(UpsertResponse {
            succeeded,
            failed: vec![],
        })
    }

    /// Delete documents by IDs.
    async fn delete(&self, ids: &[String]) -> Result<DeleteResponse> {
        let mut store = self.store.lock().unwrap();
        let mut ok_ids = Vec::new();

        for id in ids {
            if store.remove(id).is_some() {
                ok_ids.push(id.clone());
            }
        }

        let num_deleted = ok_ids.len();
        Ok(DeleteResponse {
            num_deleted: Some(num_deleted),
            succeeded: Some(ok_ids),
            failed: None,
        })
    }

    /// Get documents by IDs.
    async fn get(&self, ids: &[String]) -> Result<Vec<Document>> {
        let store = self.store.lock().unwrap();
        Ok(ids.iter().filter_map(|id| store.get(id).cloned()).collect())
    }
}
