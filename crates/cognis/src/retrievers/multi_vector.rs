//! Multi-vector retriever that searches over document summaries (or other
//! representations) but returns full original documents.
//!
//! This is useful when you want to embed summaries, questions, or hypothetical
//! answers for search while returning the complete source documents to the LLM.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use cognis_core::documents::Document;
use cognis_core::error::Result;
use cognis_core::retrievers::BaseRetriever;
use cognis_core::vectorstores::base::VectorStore;

use super::docstore::InMemoryDocStore;

/// A retriever that searches over alternative representations (e.g., summaries)
/// but returns the original full documents.
///
/// # How it works
///
/// 1. **Indexing**: Original documents are stored in the `docstore`. Their
///    alternative representations (summaries, hypothetical questions, etc.)
///    are stored in the `vectorstore` with a metadata key linking back to the
///    original document.
///
/// 2. **Retrieval**: A similarity search finds matching summaries. The
///    retriever extracts the original document IDs from the summaries'
///    metadata, deduplicates them, and returns the full documents from the
///    docstore.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use cognis::retrievers::multi_vector::MultiVectorRetriever;
/// use cognis::retrievers::docstore::InMemoryDocStore;
///
/// let retriever = MultiVectorRetriever::new(vectorstore, docstore);
/// retriever.add_documents(docs, summaries).await?;
/// let originals = retriever.get_relevant_documents("query").await?;
/// ```
pub struct MultiVectorRetriever {
    /// Vector store for searchable representations (summaries, etc.).
    vectorstore: Arc<dyn VectorStore>,
    /// Document store for full original documents.
    docstore: Arc<InMemoryDocStore>,
    /// Metadata key used to link representations to their original document.
    id_key: String,
    /// Number of representations to retrieve from the vector store.
    k: usize,
}

impl MultiVectorRetriever {
    /// Create a new `MultiVectorRetriever`.
    pub fn new(vectorstore: Arc<dyn VectorStore>, docstore: Arc<InMemoryDocStore>) -> Self {
        Self {
            vectorstore,
            docstore,
            id_key: "doc_id".to_string(),
            k: 4,
        }
    }

    /// Set the metadata key used to link representations to original documents.
    pub fn with_id_key(mut self, key: impl Into<String>) -> Self {
        self.id_key = key.into();
        self
    }

    /// Set the number of representations to retrieve.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Add original documents and their searchable representations (e.g., summaries).
    ///
    /// Each entry in `docs` is paired with the corresponding entry in `summaries`.
    /// The original document is stored in the docstore, and the summary is stored
    /// in the vectorstore with a metadata key linking to the original.
    ///
    /// # Panics
    ///
    /// Panics if `docs.len() != summaries.len()`.
    pub async fn add_documents(&self, docs: Vec<Document>, summaries: Vec<Document>) -> Result<()> {
        assert_eq!(
            docs.len(),
            summaries.len(),
            "docs and summaries must have the same length"
        );

        for (doc, mut summary) in docs.into_iter().zip(summaries.into_iter()) {
            let doc_id = Uuid::new_v4().to_string();

            // Store the full original document.
            self.docstore.add(&doc_id, doc).await;

            // Tag the summary with the original document's ID.
            summary
                .metadata
                .insert(self.id_key.clone(), serde_json::Value::String(doc_id));

            // Add the summary to the vectorstore.
            self.vectorstore.add_documents(vec![summary], None).await?;
        }

        Ok(())
    }
}

#[async_trait]
impl BaseRetriever for MultiVectorRetriever {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
        // Search for matching representations in the vectorstore.
        let representations = self.vectorstore.similarity_search(query, self.k).await?;

        // Extract unique document IDs from the representations' metadata.
        let mut seen = HashSet::new();
        let mut doc_ids = Vec::new();
        for rep in &representations {
            if let Some(serde_json::Value::String(did)) = rep.metadata.get(&self.id_key) {
                if seen.insert(did.clone()) {
                    doc_ids.push(did.clone());
                }
            }
        }

        // Fetch full original documents from the docstore.
        let doc_opts = self.docstore.mget(&doc_ids).await;
        let docs: Vec<Document> = doc_opts.into_iter().flatten().collect();

        Ok(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vectorstores::in_memory::InMemoryVectorStore;
    use cognis_core::embeddings_fake::DeterministicFakeEmbedding;

    fn make_embeddings() -> Arc<dyn cognis_core::embeddings::Embeddings> {
        Arc::new(DeterministicFakeEmbedding::new(16))
    }

    #[tokio::test]
    async fn test_add_and_retrieve_by_summary() {
        let embeddings = make_embeddings();
        let vectorstore: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new(embeddings));
        let docstore = Arc::new(InMemoryDocStore::new());

        let retriever = MultiVectorRetriever::new(vectorstore, docstore).with_k(4);

        let original = Document::new("This is a very long and detailed document about quantum physics and the nature of reality.");
        let summary = Document::new("quantum physics");

        retriever
            .add_documents(vec![original.clone()], vec![summary])
            .await
            .unwrap();

        let results = retriever
            .get_relevant_documents("quantum physics")
            .await
            .unwrap();

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, original.page_content);
    }

    #[tokio::test]
    async fn test_multiple_docs_with_summaries() {
        let embeddings = make_embeddings();
        let vectorstore: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new(embeddings));
        let docstore = Arc::new(InMemoryDocStore::new());

        let retriever = MultiVectorRetriever::new(vectorstore, docstore).with_k(4);

        let docs = vec![
            Document::new("Full content about cats and their behavior in domestic settings."),
            Document::new("Full content about dogs and their training methods for obedience."),
        ];
        let summaries = vec![
            Document::new("cats behavior"),
            Document::new("dogs training"),
        ];

        retriever
            .add_documents(docs.clone(), summaries)
            .await
            .unwrap();

        // Search for cats - should return the cats document.
        let results = retriever
            .get_relevant_documents("cats behavior")
            .await
            .unwrap();

        assert!(!results.is_empty());
        assert_eq!(results[0].page_content, docs[0].page_content);
    }

    #[tokio::test]
    async fn test_empty_vectorstore() {
        let embeddings = make_embeddings();
        let vectorstore: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new(embeddings));
        let docstore = Arc::new(InMemoryDocStore::new());

        let retriever = MultiVectorRetriever::new(vectorstore, docstore);

        let results = retriever.get_relevant_documents("anything").await.unwrap();

        assert!(results.is_empty());
    }
}
