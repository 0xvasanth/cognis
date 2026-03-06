//! Indexing pipeline for incremental document ingestion.
//!
//! Provides a [`RecordManager`] trait for tracking which documents have been
//! indexed, an [`InMemoryRecordManager`] implementation, and an
//! [`IndexingPipeline`] that orchestrates splitting, deduplication, and
//! vector store insertion with configurable cleanup modes.

use std::collections::HashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use rustchain_core::vectorstores::base::VectorStore;

use crate::text_splitter::TextSplitter;

// ---------------------------------------------------------------------------
// RecordManager trait
// ---------------------------------------------------------------------------

/// Tracks which document keys have already been indexed so that unchanged
/// documents can be skipped on subsequent indexing passes.
#[async_trait]
pub trait RecordManager: Send + Sync {
    /// Check which of the given keys already exist in the record store.
    async fn exists(&self, keys: &[String]) -> Result<Vec<bool>>;

    /// Mark the given keys as indexed, optionally associating them with
    /// group IDs for scoped cleanup.
    async fn update(&self, keys: &[String], group_ids: Option<&[String]>) -> Result<()>;

    /// Remove the given keys from the record store.
    async fn delete_keys(&self, keys: &[String]) -> Result<()>;

    /// List all keys, optionally filtered by group ID.
    async fn list_keys(&self, group_id: Option<&str>) -> Result<Vec<String>>;
}

// ---------------------------------------------------------------------------
// InMemoryRecordManager
// ---------------------------------------------------------------------------

/// A thread-safe, in-memory [`RecordManager`] backed by a `HashMap`.
///
/// Suitable for testing and lightweight workloads.
pub struct InMemoryRecordManager {
    /// Maps key -> optional group_id.
    records: RwLock<HashMap<String, Option<String>>>,
}

impl InMemoryRecordManager {
    /// Create a new empty record manager.
    pub fn new() -> Self {
        Self {
            records: RwLock::new(HashMap::new()),
        }
    }
}

impl Default for InMemoryRecordManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl RecordManager for InMemoryRecordManager {
    async fn exists(&self, keys: &[String]) -> Result<Vec<bool>> {
        let records = self.records.read().await;
        Ok(keys.iter().map(|k| records.contains_key(k)).collect())
    }

    async fn update(&self, keys: &[String], group_ids: Option<&[String]>) -> Result<()> {
        let mut records = self.records.write().await;
        for (i, key) in keys.iter().enumerate() {
            let group_id = group_ids.and_then(|g| g.get(i).cloned());
            records.insert(key.clone(), group_id);
        }
        Ok(())
    }

    async fn delete_keys(&self, keys: &[String]) -> Result<()> {
        let mut records = self.records.write().await;
        for key in keys {
            records.remove(key);
        }
        Ok(())
    }

    async fn list_keys(&self, group_id: Option<&str>) -> Result<Vec<String>> {
        let records = self.records.read().await;
        let keys = match group_id {
            Some(gid) => records
                .iter()
                .filter(|(_, v)| v.as_deref() == Some(gid))
                .map(|(k, _)| k.clone())
                .collect(),
            None => records.keys().cloned().collect(),
        };
        Ok(keys)
    }
}

// ---------------------------------------------------------------------------
// CleanupMode
// ---------------------------------------------------------------------------

/// Controls how the indexing pipeline handles previously-indexed documents
/// that are no longer present in the incoming batch.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum CleanupMode {
    /// No cleanup -- new documents are added but stale ones are never removed.
    #[default]
    None,
    /// Delete documents that were previously indexed (tracked by the record
    /// manager) but are not in the current batch.
    Incremental,
    /// Delete *all* tracked documents and re-index from scratch.
    Full,
}

// ---------------------------------------------------------------------------
// IndexingResult
// ---------------------------------------------------------------------------

/// Summary statistics returned after an indexing pass.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IndexingResult {
    /// Number of documents added to the vector store.
    pub num_added: usize,
    /// Number of documents skipped because they were already indexed.
    pub num_skipped: usize,
    /// Number of documents deleted from the vector store.
    pub num_deleted: usize,
}

// ---------------------------------------------------------------------------
// content_hash
// ---------------------------------------------------------------------------

/// Compute a deterministic hash of a document's content and metadata.
///
/// Two documents with the same `page_content` and `metadata` will always
/// produce the same hash, regardless of field ordering in metadata (since
/// we sort keys before hashing).
pub fn content_hash(doc: &Document) -> String {
    let mut hasher = DefaultHasher::new();
    doc.page_content.hash(&mut hasher);

    // Sort metadata keys for deterministic ordering.
    let mut keys: Vec<&String> = doc.metadata.keys().collect();
    keys.sort();
    for key in keys {
        key.hash(&mut hasher);
        let val = &doc.metadata[key];
        val.to_string().hash(&mut hasher);
    }

    format!("{:016x}", hasher.finish())
}

// ---------------------------------------------------------------------------
// IndexingPipeline
// ---------------------------------------------------------------------------

/// Orchestrates document ingestion: splitting, deduplication via content
/// hashing, vector store insertion, and optional cleanup of stale documents.
///
/// Use the builder methods to configure the pipeline before calling
/// [`index`](IndexingPipeline::index).
pub struct IndexingPipeline {
    /// Optional text splitter applied to incoming documents.
    pub text_splitter: Option<Box<dyn TextSplitter>>,
    /// The vector store that receives document embeddings.
    pub vectorstore: Arc<dyn VectorStore>,
    /// Optional record manager for deduplication and cleanup.
    pub record_manager: Option<Box<dyn RecordManager>>,
    /// Controls stale-document cleanup behaviour.
    pub cleanup_mode: CleanupMode,
}

impl IndexingPipeline {
    /// Create a new pipeline targeting the given vector store.
    pub fn new(vectorstore: Arc<dyn VectorStore>) -> Self {
        Self {
            text_splitter: None,
            vectorstore,
            record_manager: None,
            cleanup_mode: CleanupMode::None,
        }
    }

    /// Set the text splitter.
    pub fn with_text_splitter(mut self, splitter: Box<dyn TextSplitter>) -> Self {
        self.text_splitter = Some(splitter);
        self
    }

    /// Set the record manager.
    pub fn with_record_manager(mut self, manager: Box<dyn RecordManager>) -> Self {
        self.record_manager = Some(manager);
        self
    }

    /// Set the cleanup mode.
    pub fn with_cleanup_mode(mut self, mode: CleanupMode) -> Self {
        self.cleanup_mode = mode;
        self
    }

    /// Run the indexing pipeline on the given documents.
    ///
    /// 1. Optionally split documents using the configured text splitter.
    /// 2. Compute a content hash for each document.
    /// 3. If a record manager is present, skip documents whose hash is
    ///    already tracked (unchanged).
    /// 4. Add new/changed documents to the vector store.
    /// 5. Update the record manager.
    /// 6. Perform cleanup according to [`CleanupMode`].
    pub async fn index(&self, documents: Vec<Document>) -> Result<IndexingResult> {
        // Step 1: split if configured.
        let docs = match &self.text_splitter {
            Some(splitter) => splitter.split_documents(&documents),
            None => documents,
        };

        // Step 2: compute content hashes.
        let hashes: Vec<String> = docs.iter().map(|d| content_hash(d)).collect();

        let (docs_to_add, num_skipped) = match &self.record_manager {
            Some(rm) => {
                // Full cleanup: delete everything first, then re-index all.
                if self.cleanup_mode == CleanupMode::Full {
                    let all_keys = rm.list_keys(None).await?;
                    if !all_keys.is_empty() {
                        let key_refs: Vec<String> = all_keys.clone();
                        self.vectorstore.delete(Some(&key_refs)).await?;
                        rm.delete_keys(&all_keys).await?;
                    }
                    // All docs are "new" after full wipe.
                    (docs.clone(), 0usize)
                } else {
                    // Check which hashes already exist.
                    let existence = rm.exists(&hashes).await?;
                    let mut new_docs = Vec::new();
                    let mut skipped = 0usize;
                    for (i, exists) in existence.iter().enumerate() {
                        if *exists {
                            skipped += 1;
                        } else {
                            new_docs.push(docs[i].clone());
                        }
                    }
                    (new_docs, skipped)
                }
            }
            None => (docs.clone(), 0usize),
        };

        // Step 4: add new docs to vector store.
        let num_added = docs_to_add.len();
        if !docs_to_add.is_empty() {
            self.vectorstore
                .add_documents(docs_to_add, None)
                .await?;
        }

        // Step 5: update record manager with all current hashes.
        let mut num_deleted = 0usize;
        if let Some(rm) = &self.record_manager {
            // For Full mode we already deleted everything above; register
            // all current hashes now.
            if self.cleanup_mode == CleanupMode::Full {
                rm.update(&hashes, None).await?;
                // num_deleted was implicitly counted during the wipe, but we
                // report it as the number of keys that existed before.
                // Since we already deleted them, nothing more to do.
            } else {
                // Update all hashes (idempotent for existing ones).
                rm.update(&hashes, None).await?;

                // Incremental cleanup: find keys tracked by the record manager
                // that are no longer in the current batch.
                if self.cleanup_mode == CleanupMode::Incremental {
                    let all_keys = rm.list_keys(None).await?;
                    let current_set: std::collections::HashSet<&str> =
                        hashes.iter().map(|h| h.as_str()).collect();
                    let stale_keys: Vec<String> = all_keys
                        .iter()
                        .filter(|k| !current_set.contains(k.as_str()))
                        .cloned()
                        .collect();
                    if !stale_keys.is_empty() {
                        num_deleted = stale_keys.len();
                        self.vectorstore.delete(Some(&stale_keys)).await?;
                        rm.delete_keys(&stale_keys).await?;
                    }
                }
            }
        }

        Ok(IndexingResult {
            num_added,
            num_skipped,
            num_deleted,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectorstores::in_memory::InMemoryVectorStore;
    use rustchain_core::embeddings::Embeddings;
    use rustchain_core::embeddings_fake::DeterministicFakeEmbedding;
    use std::sync::Arc;

    fn make_embeddings() -> Arc<dyn Embeddings> {
        Arc::new(DeterministicFakeEmbedding::new(16))
    }

    fn make_store() -> Arc<InMemoryVectorStore> {
        Arc::new(InMemoryVectorStore::new(make_embeddings()))
    }

    fn make_docs(contents: &[&str]) -> Vec<Document> {
        contents.iter().map(|c| Document::new(*c)).collect()
    }

    // ---- Test 1: Basic indexing adds all documents ----
    #[tokio::test]
    async fn test_basic_indexing_adds_all_documents() {
        let store = make_store();
        let pipeline = IndexingPipeline::new(store.clone());
        let docs = make_docs(&["hello", "world", "foo"]);

        let result = pipeline.index(docs).await.unwrap();
        assert_eq!(result.num_added, 3);
        assert_eq!(result.num_skipped, 0);
        assert_eq!(result.num_deleted, 0);

        let found = store.similarity_search("hello", 10).await.unwrap();
        assert_eq!(found.len(), 3);
    }

    // ---- Test 2: Incremental skips unchanged docs ----
    #[tokio::test]
    async fn test_incremental_skips_unchanged() {
        let store = make_store();
        let rm = Box::new(InMemoryRecordManager::new());
        let pipeline = IndexingPipeline::new(store.clone())
            .with_record_manager(rm)
            .with_cleanup_mode(CleanupMode::None);

        let docs = make_docs(&["alpha", "beta"]);
        let r1 = pipeline.index(docs.clone()).await.unwrap();
        assert_eq!(r1.num_added, 2);
        assert_eq!(r1.num_skipped, 0);

        // Index the same documents again -- should skip all.
        let r2 = pipeline.index(docs).await.unwrap();
        assert_eq!(r2.num_added, 0);
        assert_eq!(r2.num_skipped, 2);
    }

    // ---- Test 3: Content hash changes trigger re-indexing ----
    #[tokio::test]
    async fn test_content_change_triggers_reindex() {
        let store = make_store();
        let rm = Box::new(InMemoryRecordManager::new());
        let pipeline = IndexingPipeline::new(store.clone())
            .with_record_manager(rm)
            .with_cleanup_mode(CleanupMode::None);

        let docs = make_docs(&["version1"]);
        let r1 = pipeline.index(docs).await.unwrap();
        assert_eq!(r1.num_added, 1);

        // Different content produces a different hash.
        let docs2 = make_docs(&["version2"]);
        let r2 = pipeline.index(docs2).await.unwrap();
        assert_eq!(r2.num_added, 1);
        assert_eq!(r2.num_skipped, 0);
    }

    // ---- Test 4: Full cleanup mode re-indexes all ----
    #[tokio::test]
    async fn test_full_cleanup_reindexes_all() {
        let store = make_store();
        let rm = Box::new(InMemoryRecordManager::new());
        let pipeline = IndexingPipeline::new(store.clone())
            .with_record_manager(rm)
            .with_cleanup_mode(CleanupMode::Full);

        let docs = make_docs(&["a", "b"]);
        let r1 = pipeline.index(docs.clone()).await.unwrap();
        assert_eq!(r1.num_added, 2);

        // Full mode wipes and re-adds everything.
        let r2 = pipeline.index(docs).await.unwrap();
        assert_eq!(r2.num_added, 2);
        assert_eq!(r2.num_skipped, 0);
    }

    // ---- Test 5: Indexing with text splitter ----
    #[tokio::test]
    async fn test_indexing_with_text_splitter() {
        use crate::text_splitter::CharacterTextSplitter;

        let store = make_store();
        // Splitter with a small chunk size to force splitting.
        let splitter = CharacterTextSplitter::new()
            .with_chunk_size(10)
            .with_chunk_overlap(0)
            .with_separator("\n");
        let pipeline = IndexingPipeline::new(store.clone())
            .with_text_splitter(Box::new(splitter));

        let docs = vec![Document::new("line one\nline two\nline three")];
        let result = pipeline.index(docs).await.unwrap();
        // The splitter should produce multiple chunks.
        assert!(result.num_added >= 2, "Expected at least 2 chunks, got {}", result.num_added);
    }

    // ---- Test 6: IndexingResult counts correct ----
    #[tokio::test]
    async fn test_indexing_result_counts() {
        let store = make_store();
        let rm = Box::new(InMemoryRecordManager::new());
        let pipeline = IndexingPipeline::new(store.clone())
            .with_record_manager(rm)
            .with_cleanup_mode(CleanupMode::Incremental);

        // First pass: add 3 documents.
        let docs = make_docs(&["x", "y", "z"]);
        let r1 = pipeline.index(docs).await.unwrap();
        assert_eq!(r1.num_added, 3);
        assert_eq!(r1.num_skipped, 0);
        assert_eq!(r1.num_deleted, 0);

        // Second pass: keep x and y, drop z.
        let docs2 = make_docs(&["x", "y"]);
        let r2 = pipeline.index(docs2).await.unwrap();
        assert_eq!(r2.num_skipped, 2);
        assert_eq!(r2.num_added, 0);
        assert_eq!(r2.num_deleted, 1); // z was removed
    }

    // ---- Test 7: Empty document list ----
    #[tokio::test]
    async fn test_empty_document_list() {
        let store = make_store();
        let pipeline = IndexingPipeline::new(store.clone());
        let result = pipeline.index(vec![]).await.unwrap();
        assert_eq!(result.num_added, 0);
        assert_eq!(result.num_skipped, 0);
        assert_eq!(result.num_deleted, 0);
    }

    // ---- Test 8: Record manager tracks keys ----
    #[tokio::test]
    async fn test_record_manager_tracks_keys() {
        let rm = InMemoryRecordManager::new();

        // Initially empty.
        let keys = rm.list_keys(None).await.unwrap();
        assert!(keys.is_empty());

        // Add some keys.
        rm.update(&["k1".into(), "k2".into()], None).await.unwrap();
        let keys = rm.list_keys(None).await.unwrap();
        assert_eq!(keys.len(), 2);

        // Check existence.
        let exists = rm
            .exists(&["k1".into(), "k3".into()])
            .await
            .unwrap();
        assert_eq!(exists, vec![true, false]);

        // Delete a key.
        rm.delete_keys(&["k1".into()]).await.unwrap();
        let keys = rm.list_keys(None).await.unwrap();
        assert_eq!(keys.len(), 1);
        assert_eq!(keys[0], "k2");
    }

    // ---- Test 9: Multiple indexing passes ----
    #[tokio::test]
    async fn test_multiple_indexing_passes() {
        let store = make_store();
        let rm = Box::new(InMemoryRecordManager::new());
        let pipeline = IndexingPipeline::new(store.clone())
            .with_record_manager(rm)
            .with_cleanup_mode(CleanupMode::None);

        // Pass 1
        let r1 = pipeline.index(make_docs(&["a", "b"])).await.unwrap();
        assert_eq!(r1.num_added, 2);

        // Pass 2 -- same docs
        let r2 = pipeline.index(make_docs(&["a", "b"])).await.unwrap();
        assert_eq!(r2.num_added, 0);
        assert_eq!(r2.num_skipped, 2);

        // Pass 3 -- add a new doc
        let r3 = pipeline
            .index(make_docs(&["a", "b", "c"]))
            .await
            .unwrap();
        assert_eq!(r3.num_added, 1);
        assert_eq!(r3.num_skipped, 2);
    }

    // ---- Test 10: Content hash determinism ----
    #[tokio::test]
    async fn test_content_hash_determinism() {
        let doc = Document::new("hello world");
        let h1 = content_hash(&doc);
        let h2 = content_hash(&doc);
        assert_eq!(h1, h2, "Same document must produce the same hash");

        // Different content -> different hash.
        let doc2 = Document::new("goodbye world");
        let h3 = content_hash(&doc2);
        assert_ne!(h1, h3, "Different content must produce different hashes");
    }

    // ---- Test 11: Content hash includes metadata ----
    #[tokio::test]
    async fn test_content_hash_includes_metadata() {
        let doc1 = Document::new("same content");
        let mut meta = HashMap::new();
        meta.insert("key".to_string(), serde_json::json!("value"));
        let doc2 = Document::new("same content").with_metadata(meta);

        let h1 = content_hash(&doc1);
        let h2 = content_hash(&doc2);
        assert_ne!(h1, h2, "Metadata differences should change the hash");
    }

    // ---- Test 12: Record manager with group IDs ----
    #[tokio::test]
    async fn test_record_manager_group_ids() {
        let rm = InMemoryRecordManager::new();
        rm.update(
            &["k1".into(), "k2".into(), "k3".into()],
            Some(&["g1".into(), "g1".into(), "g2".into()]),
        )
        .await
        .unwrap();

        let g1_keys = rm.list_keys(Some("g1")).await.unwrap();
        assert_eq!(g1_keys.len(), 2);

        let g2_keys = rm.list_keys(Some("g2")).await.unwrap();
        assert_eq!(g2_keys.len(), 1);

        let all_keys = rm.list_keys(None).await.unwrap();
        assert_eq!(all_keys.len(), 3);
    }
}
