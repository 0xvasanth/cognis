//! `MultiVectorRetriever` — façade over `MultiVectorIndexer` (write side)
//! + `ParentDocumentRetriever` (read side).
//!
//! V1 packages this as one class; V2's foundation already covers it via
//! the indexer + parent-doc retriever combo. This thin façade gives the
//! pattern a single name for discovery.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use cognis_core::{Result, Runnable, RunnableConfig};

use crate::docstore::Docstore;
use crate::document::Document;
use crate::multi_vector::MultiVectorIndexer;
use crate::retrievers::ParentDocumentRetriever;
use crate::vectorstore::VectorStore;

/// One type that handles both indexing (multi-rep per parent) and
/// retrieval (parent-doc).
pub struct MultiVectorRetriever {
    indexer: MultiVectorIndexer,
    retriever: ParentDocumentRetriever,
}

impl MultiVectorRetriever {
    /// Build with a chunk vector store + parent docstore + top-k.
    pub fn new(
        chunks: Arc<RwLock<dyn VectorStore>>,
        parents: Arc<dyn Docstore>,
        top_k: usize,
    ) -> Self {
        Self {
            indexer: MultiVectorIndexer::new(chunks.clone(), parents.clone()),
            retriever: ParentDocumentRetriever::new(chunks, parents, top_k),
        }
    }

    /// Index one parent under multiple text representations.
    pub async fn index(
        &self,
        parent_id: impl Into<String>,
        parent: Document,
        representations: Vec<String>,
    ) -> Result<()> {
        self.indexer.index(parent_id, parent, representations).await
    }
}

#[async_trait]
impl Runnable<String, Vec<Document>> for MultiVectorRetriever {
    async fn invoke(&self, query: String, config: RunnableConfig) -> Result<Vec<Document>> {
        self.retriever.invoke(query, config).await
    }
    fn name(&self) -> &str {
        "MultiVectorRetriever"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docstore::InMemoryDocstore;
    use crate::embeddings::FakeEmbeddings;
    use crate::vectorstore::InMemoryVectorStore;

    #[tokio::test]
    async fn end_to_end_index_then_retrieve() {
        let chunks = InMemoryVectorStore::new(Arc::new(FakeEmbeddings::new(8)));
        let chunks_arc: Arc<RwLock<dyn VectorStore>> = Arc::new(RwLock::new(chunks));
        let parents: Arc<dyn Docstore> = Arc::new(InMemoryDocstore::new());

        let mvr = MultiVectorRetriever::new(chunks_arc, parents, 5);
        mvr.index(
            "doc1",
            Document::new("FULL TEXT").with_id("doc1"),
            vec!["summary".into(), "detail".into()],
        )
        .await
        .unwrap();
        let out = mvr
            .invoke("summary".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert!(out.iter().any(|d| d.content == "FULL TEXT"));
    }
}
