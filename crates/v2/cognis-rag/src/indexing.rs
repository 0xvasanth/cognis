//! Indexing pipeline — load → split → embed → store.
//!
//! Most RAG setups follow the same shape: pull documents from a source,
//! chunk them, write the chunks to a vector store. This module wraps that
//! flow into one ergonomic call.
//!
//! [`IndexingPipeline::run_incremental`] adds change-tracking via a
//! [`RecordManager`]: only new/changed docs get re-embedded; deleted ones
//! get removed from the store.

use std::sync::Arc;

use tokio::sync::RwLock;

use cognis2_core::Result;

use crate::document::Document;
use crate::loaders::DocumentLoader;
use crate::record_manager::{fingerprint as compute_fingerprint, RecordManager};
use crate::splitters::TextSplitter;
use crate::vectorstore::VectorStore;

/// Drives the load → split → store pipeline.
///
/// ```ignore
/// let pipeline = IndexingPipeline::new(loader, splitter, store);
/// let count = pipeline.run().await?;
/// ```
pub struct IndexingPipeline<L, T> {
    loader: L,
    splitter: T,
    store: Arc<RwLock<dyn VectorStore>>,
}

impl<L, T> IndexingPipeline<L, T>
where
    L: DocumentLoader,
    T: TextSplitter,
{
    /// Construct.
    pub fn new(loader: L, splitter: T, store: Arc<RwLock<dyn VectorStore>>) -> Self {
        Self {
            loader,
            splitter,
            store,
        }
    }

    /// Execute the pipeline. Returns the number of chunks written to the store.
    pub async fn run(&self) -> Result<usize> {
        let docs = self.loader.load_all().await?;
        let chunks: Vec<Document> = docs.iter().flat_map(|d| self.splitter.split(d)).collect();
        let count = chunks.len();

        let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
        let metadatas: Vec<_> = chunks.iter().map(|c| c.metadata.clone()).collect();
        self.store
            .write()
            .await
            .add_texts(texts, Some(metadatas))
            .await?;
        Ok(count)
    }

    /// Incremental indexing report.
    ///
    /// `key_fn` derives a stable identifier from each loaded document.
    /// Typical choices: `metadata["source"]` (file path) or a primary key.
    /// Only docs whose fingerprint differs from the record manager's
    /// stored value are re-embedded; docs absent from the new load are
    /// deleted from the record manager (and reported in [`IncrementalReport::deleted`]).
    ///
    /// **Note**: this drops *records*, not the underlying vector-store
    /// chunks. If you need the chunk store cleaned too, use a fresh group
    /// per index pass (or pair this with the parent-document pattern
    /// where chunks key off `parent_id`).
    pub async fn run_incremental(
        &self,
        record_manager: &dyn RecordManager,
        group: &str,
        key_fn: impl Fn(&Document) -> Option<String>,
    ) -> Result<IncrementalReport> {
        let docs = self.loader.load_all().await?;
        let mut report = IncrementalReport::default();

        let mut seen_keys = std::collections::HashSet::new();
        let mut new_chunks: Vec<Document> = Vec::new();

        for d in &docs {
            let Some(key) = key_fn(d) else {
                report.skipped_no_key += 1;
                continue;
            };
            seen_keys.insert(key.clone());
            let fp = compute_fingerprint(&d.content);
            let prev = record_manager.get_fingerprint(group, &key).await?;
            match prev {
                Some(p) if p == fp => {
                    report.unchanged += 1;
                    continue;
                }
                Some(_) => {
                    report.changed += 1;
                }
                None => {
                    report.added += 1;
                }
            }
            record_manager.set_fingerprint(group, &key, &fp).await?;
            new_chunks.extend(self.splitter.split(d));
        }

        // Detect deletions: previously-tracked keys not present in current load.
        let prev_keys = record_manager.list_keys(group).await?;
        let to_delete: Vec<String> = prev_keys
            .into_iter()
            .filter(|k| !seen_keys.contains(k))
            .collect();
        if !to_delete.is_empty() {
            report.deleted = to_delete.len();
            record_manager.delete(group, &to_delete).await?;
        }

        if !new_chunks.is_empty() {
            let texts: Vec<String> = new_chunks.iter().map(|c| c.content.clone()).collect();
            let metadatas: Vec<_> = new_chunks.iter().map(|c| c.metadata.clone()).collect();
            self.store
                .write()
                .await
                .add_texts(texts, Some(metadatas))
                .await?;
        }
        report.chunks_written = new_chunks.len();
        Ok(report)
    }
}

/// Result of an [`IndexingPipeline::run_incremental`] call.
#[derive(Debug, Default, Clone)]
pub struct IncrementalReport {
    /// Docs new to the record manager (re-embedded).
    pub added: usize,
    /// Docs whose fingerprint changed since last index (re-embedded).
    pub changed: usize,
    /// Docs unchanged from last index (skipped).
    pub unchanged: usize,
    /// Records removed because the doc no longer appears in the loader.
    pub deleted: usize,
    /// Docs we couldn't index because `key_fn` returned `None`.
    pub skipped_no_key: usize,
    /// Total chunks written to the vector store this run.
    pub chunks_written: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::FakeEmbeddings;
    use crate::record_manager::InMemoryRecordManager;
    use crate::splitters::RecursiveCharSplitter;
    use crate::vectorstore::InMemoryVectorStore;
    use std::io::Write;
    use tempfile::TempDir;

    use crate::loaders::DirectoryLoader;

    #[tokio::test]
    async fn incremental_skip_unchanged_and_detect_deletes() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "rust is fast").unwrap();
        std::fs::write(dir.path().join("b.txt"), "cooking").unwrap();
        let store = InMemoryVectorStore::new(Arc::new(FakeEmbeddings::new(8)));
        let store_arc: Arc<RwLock<dyn VectorStore>> = Arc::new(RwLock::new(store));
        let pipeline = IndexingPipeline::new(
            DirectoryLoader::new(dir.path()),
            RecursiveCharSplitter::new()
                .with_chunk_size(20)
                .with_overlap(0),
            store_arc.clone(),
        );
        let rm = InMemoryRecordManager::new();
        let key_fn = |d: &Document| {
            d.metadata
                .get("source")
                .and_then(|v| v.as_str().map(|s| s.to_string()))
        };

        // First pass — both new.
        let r1 = pipeline.run_incremental(&rm, "g1", key_fn).await.unwrap();
        assert_eq!(r1.added, 2);
        assert_eq!(r1.unchanged, 0);
        assert!(r1.chunks_written > 0);

        // Second pass — both unchanged.
        let r2 = pipeline.run_incremental(&rm, "g1", key_fn).await.unwrap();
        assert_eq!(r2.added, 0);
        assert_eq!(r2.unchanged, 2);
        assert_eq!(r2.chunks_written, 0);

        // Third pass — change `a.txt`, delete `b.txt`.
        std::fs::write(dir.path().join("a.txt"), "rust is FAST").unwrap();
        std::fs::remove_file(dir.path().join("b.txt")).unwrap();
        let r3 = pipeline.run_incremental(&rm, "g1", key_fn).await.unwrap();
        assert_eq!(r3.added, 0);
        assert_eq!(r3.changed, 1);
        assert_eq!(r3.unchanged, 0);
        assert_eq!(r3.deleted, 1);
    }

    #[tokio::test]
    async fn indexes_a_directory_through_to_search() {
        let dir = TempDir::new().unwrap();
        let mut f = std::fs::File::create(dir.path().join("a.txt")).unwrap();
        writeln!(f, "Rust is a systems programming language").unwrap();
        let mut f = std::fs::File::create(dir.path().join("b.txt")).unwrap();
        writeln!(f, "Cooking with cast iron").unwrap();

        let store = InMemoryVectorStore::new(Arc::new(FakeEmbeddings::new(8)));
        let store_arc: Arc<RwLock<dyn VectorStore>> = Arc::new(RwLock::new(store));

        let pipeline = IndexingPipeline::new(
            DirectoryLoader::new(dir.path()),
            RecursiveCharSplitter::new()
                .with_chunk_size(20)
                .with_overlap(0),
            store_arc.clone(),
        );

        let n = pipeline.run().await.unwrap();
        assert!(n >= 2);
        assert!(!store_arc.read().await.is_empty());
    }
}
