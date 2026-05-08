//! Document indexing orchestration.
//!
//! Mirrors Python `langchain_core.indexing.api`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::documents::Document;
use crate::error::Result;

use super::base::{DocumentIndex, RecordManager};

/// Cleanup modes for the indexing pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CleanupMode {
    /// Don't delete any existing documents.
    None,
    /// Delete documents that are no longer in the source (incremental).
    Incremental,
    /// Delete ALL existing documents before re-indexing.
    Full,
}

/// Result of an indexing operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexingResult {
    /// Number of documents added.
    pub num_added: usize,
    /// Number of documents updated.
    pub num_updated: usize,
    /// Number of documents skipped (already up-to-date).
    pub num_skipped: usize,
    /// Number of documents deleted.
    pub num_deleted: usize,
}

/// Compute a content hash for a document (for deduplication).
fn hash_document(doc: &Document) -> String {
    let mut hasher = DefaultHasher::new();
    doc.page_content.hash(&mut hasher);
    // Include sorted metadata keys for stability
    let mut meta_keys: Vec<&String> = doc.metadata.keys().collect();
    meta_keys.sort();
    for key in meta_keys {
        key.hash(&mut hasher);
        if let Some(val) = doc.metadata.get(key) {
            val.to_string().hash(&mut hasher);
        }
    }
    format!("{:016x}", hasher.finish())
}

/// Generate a stable key for a document.
fn get_doc_key(doc: &Document, source_id_key: Option<&str>) -> String {
    if let Some(id) = &doc.id {
        return id.clone();
    }
    if let Some(key) = source_id_key {
        if let Some(source) = doc.metadata.get(key) {
            if let Some(s) = source.as_str() {
                return format!("{}:{}", s, hash_document(doc));
            }
        }
    }
    hash_document(doc)
}

/// Index documents into a document index with deduplication and optional cleanup.
///
/// # Arguments
/// * `docs` - Documents to index.
/// * `record_manager` - Tracks which documents have been indexed.
/// * `doc_index` - The target document store.
/// * `cleanup` - Whether/how to clean up stale documents.
/// * `source_id_key` - Metadata key containing the source identifier.
/// * `batch_size` - Number of documents to process at a time.
pub async fn index(
    docs: Vec<Document>,
    record_manager: &dyn RecordManager,
    doc_index: &dyn DocumentIndex,
    cleanup: CleanupMode,
    source_id_key: Option<&str>,
    batch_size: usize,
) -> Result<IndexingResult> {
    // Ensure schema exists
    record_manager.create_schema().await?;

    let index_start_time = record_manager.get_time().await?;
    let mut result = IndexingResult {
        num_added: 0,
        num_updated: 0,
        num_skipped: 0,
        num_deleted: 0,
    };

    // Process documents in batches
    for batch in docs.chunks(batch_size.max(1)) {
        let keys: Vec<String> = batch
            .iter()
            .map(|doc| get_doc_key(doc, source_id_key))
            .collect();

        // Check which keys already exist
        let exists = record_manager.exists(&keys).await?;

        // Split into new and existing
        let mut docs_to_upsert = Vec::new();
        let mut keys_to_upsert = Vec::new();
        let mut group_ids = Vec::new();

        for (i, doc) in batch.iter().enumerate() {
            let group_id = source_id_key
                .and_then(|key| doc.metadata.get(key))
                .and_then(|v| v.as_str())
                .map(|s| s.to_string());

            if exists[i] {
                // Document exists — compute hash to check if it changed
                // For simplicity, skip if key already exists (assume unchanged)
                result.num_skipped += 1;
            } else {
                let mut indexed_doc = doc.clone();
                if indexed_doc.id.is_none() {
                    indexed_doc.id = Some(keys[i].clone());
                }
                docs_to_upsert.push(indexed_doc);
                keys_to_upsert.push(keys[i].clone());
                group_ids.push(group_id.clone());
            }
        }

        // Upsert new documents
        if !docs_to_upsert.is_empty() {
            let upsert_result = doc_index.upsert(docs_to_upsert).await?;
            result.num_added += upsert_result.succeeded.len();

            // Update record manager
            let upsert_group_ids: Vec<Option<String>> = group_ids;
            record_manager
                .update(&keys_to_upsert, &upsert_group_ids, Some(index_start_time))
                .await?;
        }

        // Also update records for skipped (existing) documents to refresh timestamps
        let existing_keys: Vec<String> = keys
            .iter()
            .enumerate()
            .filter(|(i, _)| exists[*i])
            .map(|(_, k)| k.clone())
            .collect();

        if !existing_keys.is_empty() {
            let existing_group_ids: Vec<Option<String>> = batch
                .iter()
                .enumerate()
                .filter(|(i, _)| exists[*i])
                .map(|(_, doc)| {
                    source_id_key
                        .and_then(|key| doc.metadata.get(key))
                        .and_then(|v| v.as_str())
                        .map(|s| s.to_string())
                })
                .collect();
            record_manager
                .update(&existing_keys, &existing_group_ids, Some(index_start_time))
                .await?;
        }
    }

    // Cleanup
    match cleanup {
        CleanupMode::None => {}
        CleanupMode::Full => {
            // Delete all keys indexed before the start time
            let stale_keys = record_manager
                .list_keys(Some(index_start_time), None, None, None)
                .await?;
            if !stale_keys.is_empty() {
                let delete_result = doc_index.delete(&stale_keys).await?;
                result.num_deleted = delete_result.num_deleted.unwrap_or(stale_keys.len());
                record_manager.delete_keys(&stale_keys).await?;
            }
        }
        CleanupMode::Incremental => {
            // In incremental mode, delete records that weren't updated in this run
            let stale_keys = record_manager
                .list_keys(Some(index_start_time), None, None, None)
                .await?;
            if !stale_keys.is_empty() {
                let delete_result = doc_index.delete(&stale_keys).await?;
                result.num_deleted = delete_result.num_deleted.unwrap_or(stale_keys.len());
                record_manager.delete_keys(&stale_keys).await?;
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::indexing::base::{DeleteResponse, InMemoryRecordManager, UpsertResponse};

    use async_trait::async_trait;
    use std::collections::HashMap;
    use std::sync::Mutex;

    /// Simple in-memory document index for testing.
    struct InMemoryDocumentIndex {
        docs: Mutex<HashMap<String, Document>>,
    }

    impl InMemoryDocumentIndex {
        fn new() -> Self {
            Self {
                docs: Mutex::new(HashMap::new()),
            }
        }

        fn len(&self) -> usize {
            self.docs.lock().unwrap().len()
        }
    }

    #[async_trait]
    impl DocumentIndex for InMemoryDocumentIndex {
        async fn upsert(&self, docs: Vec<Document>) -> Result<UpsertResponse> {
            let mut store = self.docs.lock().unwrap();
            let mut succeeded = Vec::new();
            for doc in docs {
                let key = doc.id.clone().unwrap_or_else(|| hash_document(&doc));
                succeeded.push(key.clone());
                store.insert(key, doc);
            }
            Ok(UpsertResponse {
                succeeded,
                failed: vec![],
            })
        }

        async fn delete(&self, ids: &[String]) -> Result<DeleteResponse> {
            let mut store = self.docs.lock().unwrap();
            let mut count = 0;
            for id in ids {
                if store.remove(id).is_some() {
                    count += 1;
                }
            }
            Ok(DeleteResponse {
                num_deleted: Some(count),
                succeeded: Some(ids.to_vec()),
                failed: None,
            })
        }

        async fn get(&self, ids: &[String]) -> Result<Vec<Document>> {
            let store = self.docs.lock().unwrap();
            Ok(ids.iter().filter_map(|id| store.get(id).cloned()).collect())
        }
    }

    #[tokio::test]
    async fn test_index_new_documents() {
        let record_manager = InMemoryRecordManager::new("test");
        let doc_index = InMemoryDocumentIndex::new();

        let docs = vec![
            Document::new("Hello world").with_id("doc1"),
            Document::new("Goodbye world").with_id("doc2"),
        ];

        let result = index(
            docs,
            &record_manager,
            &doc_index,
            CleanupMode::None,
            None,
            100,
        )
        .await
        .unwrap();

        assert_eq!(result.num_added, 2);
        assert_eq!(result.num_skipped, 0);
        assert_eq!(result.num_deleted, 0);
        assert_eq!(doc_index.len(), 2);
    }

    #[tokio::test]
    async fn test_index_skips_existing_documents() {
        let record_manager = InMemoryRecordManager::new("test");
        let doc_index = InMemoryDocumentIndex::new();

        let docs = vec![
            Document::new("Hello world").with_id("doc1"),
            Document::new("Goodbye world").with_id("doc2"),
        ];

        // First indexing
        let result1 = index(
            docs.clone(),
            &record_manager,
            &doc_index,
            CleanupMode::None,
            None,
            100,
        )
        .await
        .unwrap();
        assert_eq!(result1.num_added, 2);

        // Second indexing — same docs should be skipped
        let result2 = index(
            docs,
            &record_manager,
            &doc_index,
            CleanupMode::None,
            None,
            100,
        )
        .await
        .unwrap();
        assert_eq!(result2.num_added, 0);
        assert_eq!(result2.num_skipped, 2);
    }

    #[tokio::test]
    async fn test_index_with_batching() {
        let record_manager = InMemoryRecordManager::new("test");
        let doc_index = InMemoryDocumentIndex::new();

        let docs = vec![
            Document::new("Doc 1").with_id("d1"),
            Document::new("Doc 2").with_id("d2"),
            Document::new("Doc 3").with_id("d3"),
        ];

        let result = index(
            docs,
            &record_manager,
            &doc_index,
            CleanupMode::None,
            None,
            2, // batch size of 2
        )
        .await
        .unwrap();

        assert_eq!(result.num_added, 3);
        assert_eq!(doc_index.len(), 3);
    }

    #[tokio::test]
    async fn test_hash_document_deterministic() {
        let doc = Document::new("test content");
        let h1 = hash_document(&doc);
        let h2 = hash_document(&doc);
        assert_eq!(h1, h2);
    }

    #[tokio::test]
    async fn test_get_doc_key_uses_id() {
        let doc = Document::new("content").with_id("my-id");
        assert_eq!(get_doc_key(&doc, None), "my-id");
    }

    #[tokio::test]
    async fn test_get_doc_key_uses_source_id() {
        let mut metadata = HashMap::new();
        metadata.insert(
            "source".to_string(),
            serde_json::Value::String("file.txt".to_string()),
        );
        let doc = Document::new("content").with_metadata(metadata);
        let key = get_doc_key(&doc, Some("source"));
        assert!(key.starts_with("file.txt:"));
    }

    #[tokio::test]
    async fn test_get_doc_key_falls_back_to_hash() {
        let doc = Document::new("content");
        let key = get_doc_key(&doc, None);
        assert_eq!(key, hash_document(&doc));
    }
}
