//! Parent-document retriever — small chunks are indexed for similarity,
//! but the full parent document is what's returned to the model.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use cognis_core::{Result, Runnable, RunnableConfig};

use crate::docstore::Docstore;
use crate::document::Document;
use crate::vectorstore::VectorStore;

/// Wraps a vector index of small chunks + a doc-id keyed parent docstore.
///
/// On `invoke(query)`:
/// 1. similarity-search the chunk store, get top-N hits (each with a
///    `parent_id` metadata field).
/// 2. dedupe by `parent_id`.
/// 3. fetch parents from the docstore.
pub struct ParentDocumentRetriever {
    chunks: Arc<RwLock<dyn VectorStore>>,
    parents: Arc<dyn Docstore>,
    /// How many chunks to retrieve before deduping.
    candidate_k: usize,
    /// Final cap on parent count.
    top_k: usize,
    /// Metadata key on the chunk that points back to its parent.
    parent_id_key: String,
}

impl ParentDocumentRetriever {
    /// Build a parent-document retriever.
    pub fn new(
        chunks: Arc<RwLock<dyn VectorStore>>,
        parents: Arc<dyn Docstore>,
        top_k: usize,
    ) -> Self {
        Self {
            chunks,
            parents,
            candidate_k: top_k * 4,
            top_k,
            parent_id_key: "parent_id".to_string(),
        }
    }

    /// Override the metadata key used to find each chunk's parent id.
    pub fn with_parent_id_key(mut self, k: impl Into<String>) -> Self {
        self.parent_id_key = k.into();
        self
    }

    /// Override how many chunks to retrieve before deduping by parent.
    pub fn with_candidate_k(mut self, k: usize) -> Self {
        self.candidate_k = k;
        self
    }
}

#[async_trait]
impl Runnable<String, Vec<Document>> for ParentDocumentRetriever {
    async fn invoke(&self, query: String, _: RunnableConfig) -> Result<Vec<Document>> {
        let hits = self
            .chunks
            .read()
            .await
            .similarity_search(&query, self.candidate_k)
            .await?;
        let mut seen = std::collections::HashSet::new();
        let mut ordered_parent_ids: Vec<String> = Vec::new();
        for h in hits {
            if let Some(pid) = h.metadata.get(&self.parent_id_key).and_then(|v| v.as_str()) {
                if seen.insert(pid.to_string()) {
                    ordered_parent_ids.push(pid.to_string());
                    if ordered_parent_ids.len() >= self.top_k {
                        break;
                    }
                }
            }
        }
        if ordered_parent_ids.is_empty() {
            return Ok(Vec::new());
        }
        let parents = self.parents.get(&ordered_parent_ids).await?;
        // Preserve hit order rather than docstore-internal order.
        let mut by_id: std::collections::HashMap<String, Document> = parents
            .into_iter()
            .filter_map(|d| d.id.clone().map(|id| (id, d)))
            .collect();
        Ok(ordered_parent_ids
            .into_iter()
            .filter_map(|pid| by_id.remove(&pid))
            .collect())
    }

    fn name(&self) -> &str {
        "ParentDocumentRetriever"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::docstore::InMemoryDocstore;
    use crate::embeddings::FakeEmbeddings;
    use crate::vectorstore::InMemoryVectorStore;

    #[tokio::test]
    async fn dedupes_by_parent_id_and_fetches_parents() {
        let mut chunks = InMemoryVectorStore::new(Arc::new(FakeEmbeddings::new(8)));
        chunks
            .add_texts(
                vec![
                    "alpha chunk 1".into(),
                    "alpha chunk 2".into(),
                    "beta chunk 1".into(),
                ],
                Some(vec![
                    [("parent_id".into(), serde_json::json!("alpha"))]
                        .into_iter()
                        .collect(),
                    [("parent_id".into(), serde_json::json!("alpha"))]
                        .into_iter()
                        .collect(),
                    [("parent_id".into(), serde_json::json!("beta"))]
                        .into_iter()
                        .collect(),
                ]),
            )
            .await
            .unwrap();
        let chunks_arc: Arc<RwLock<dyn VectorStore>> = Arc::new(RwLock::new(chunks));

        let parents = InMemoryDocstore::new();
        parents
            .put(vec![
                ("alpha".into(), Document::new("FULL ALPHA").with_id("alpha")),
                ("beta".into(), Document::new("FULL BETA").with_id("beta")),
            ])
            .await
            .unwrap();
        let parents_arc: Arc<dyn Docstore> = Arc::new(parents);

        let r = ParentDocumentRetriever::new(chunks_arc, parents_arc, 2);
        let out = r
            .invoke("alpha".into(), RunnableConfig::default())
            .await
            .unwrap();
        let ids: Vec<_> = out.iter().filter_map(|d| d.id.clone()).collect();
        // We expect parents (not chunks), and only 2 even though there were
        // multiple alpha chunks.
        assert!(ids.contains(&"alpha".to_string()));
        assert!(out.len() <= 2);
    }
}
