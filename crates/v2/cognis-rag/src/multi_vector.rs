//! `MultiVectorIndexer` — index many *representations* of one document
//! under a shared parent id.
//!
//! Why? RAG quality often improves when you index multiple views of the
//! same doc (summary + key questions + raw chunks). Retrieval then hits
//! whichever view matches best, but the parent doc is what gets returned
//! to the model.
//!
//! Pairs with [`crate::ParentDocumentRetriever`]: this writes the small
//! per-view chunks to the vector store and the parent to a docstore;
//! the retriever fans out, dedupes, and pulls parents.

use std::sync::Arc;

use tokio::sync::RwLock;

use cognis2_core::Result;

use crate::docstore::Docstore;
use crate::document::Document;
use crate::vectorstore::VectorStore;

/// Indexes a parent doc under N text representations.
pub struct MultiVectorIndexer {
    chunks: Arc<RwLock<dyn VectorStore>>,
    parents: Arc<dyn Docstore>,
    /// Metadata key on each chunk pointing to its parent id.
    parent_id_key: String,
}

impl MultiVectorIndexer {
    /// Build with a chunk vector store + a parent docstore.
    pub fn new(chunks: Arc<RwLock<dyn VectorStore>>, parents: Arc<dyn Docstore>) -> Self {
        Self {
            chunks,
            parents,
            parent_id_key: "parent_id".to_string(),
        }
    }

    /// Override the metadata key used to wire chunks back to their parent.
    pub fn with_parent_id_key(mut self, k: impl Into<String>) -> Self {
        self.parent_id_key = k.into();
        self
    }

    /// Index one parent under multiple text representations.
    ///
    /// Each representation gets its own row in the chunk vector store
    /// (carrying the `parent_id` metadata); the parent doc itself gets
    /// stored in the docstore under `parent_id`.
    pub async fn index(
        &self,
        parent_id: impl Into<String>,
        parent: Document,
        representations: Vec<String>,
    ) -> Result<()> {
        let parent_id = parent_id.into();
        let mut parent = parent;
        if parent.id.is_none() {
            parent.id = Some(parent_id.clone());
        }
        self.parents.put(vec![(parent_id.clone(), parent)]).await?;

        if representations.is_empty() {
            return Ok(());
        }
        let metadatas: Vec<_> = (0..representations.len())
            .map(|_| {
                let mut m = std::collections::HashMap::new();
                m.insert(
                    self.parent_id_key.clone(),
                    serde_json::Value::String(parent_id.clone()),
                );
                m
            })
            .collect();

        self.chunks
            .write()
            .await
            .add_texts(representations, Some(metadatas))
            .await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docstore::InMemoryDocstore;
    use crate::embeddings::FakeEmbeddings;
    use crate::vectorstore::InMemoryVectorStore;

    #[tokio::test]
    async fn indexes_parent_and_each_representation() {
        let chunks = InMemoryVectorStore::new(Arc::new(FakeEmbeddings::new(8)));
        let chunks_arc: Arc<RwLock<dyn VectorStore>> = Arc::new(RwLock::new(chunks));
        let parents: Arc<dyn Docstore> = Arc::new(InMemoryDocstore::new());

        let idx = MultiVectorIndexer::new(chunks_arc.clone(), parents.clone());
        idx.index(
            "doc1",
            Document::new("FULL TEXT"),
            vec![
                "summary of the doc".into(),
                "Q: what is X? A: ...".into(),
                "raw chunk one".into(),
            ],
        )
        .await
        .unwrap();

        // Parent retrievable by id.
        let got = parents.get(&["doc1".to_string()]).await.unwrap();
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].content, "FULL TEXT");

        // Three chunks each tagged with parent_id="doc1".
        let store = chunks_arc.read().await;
        assert_eq!(store.len(), 3);
    }
}
