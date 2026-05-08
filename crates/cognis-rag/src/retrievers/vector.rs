//! Vector retriever — wraps any [`VectorStore`] in a `Runnable`.

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use cognis_core::{Result, Runnable, RunnableConfig};

use crate::document::Document;
use crate::vectorstore::VectorStore;

/// Adapts a [`VectorStore`] to a `Runnable<String, Vec<Document>>`.
///
/// Holds the store behind an `Arc<RwLock>` so concurrent invocations can
/// share a single store instance.
pub struct VectorRetriever {
    store: Arc<RwLock<dyn VectorStore>>,
    k: usize,
}

impl VectorRetriever {
    /// Wrap a vector store with a target top-k.
    pub fn new(store: Arc<RwLock<dyn VectorStore>>, k: usize) -> Self {
        Self { store, k }
    }

    /// Override `k` (builder-style).
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }
}

#[async_trait]
impl Runnable<String, Vec<Document>> for VectorRetriever {
    async fn invoke(&self, query: String, _: RunnableConfig) -> Result<Vec<Document>> {
        let results = self
            .store
            .read()
            .await
            .similarity_search(&query, self.k)
            .await?;
        Ok(results
            .into_iter()
            .map(|r| Document {
                id: Some(r.id),
                content: r.text,
                metadata: r.metadata,
            })
            .collect())
    }

    fn name(&self) -> &str {
        "VectorRetriever"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::FakeEmbeddings;
    use crate::vectorstore::InMemoryVectorStore;

    #[tokio::test]
    async fn retrieves_via_vector_store() {
        let mut store = InMemoryVectorStore::new(Arc::new(FakeEmbeddings::new(8)));
        store
            .add_texts(vec!["hello world".into(), "rust programming".into()], None)
            .await
            .unwrap();
        let store_arc: Arc<RwLock<dyn VectorStore>> = Arc::new(RwLock::new(store));

        let retriever = VectorRetriever::new(store_arc, 2);
        let docs = retriever
            .invoke("hello".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(docs.len(), 2);
    }
}
