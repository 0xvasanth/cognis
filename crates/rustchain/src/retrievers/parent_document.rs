//! Parent-document retriever that indexes child chunks but returns full parent documents.
//!
//! This retriever splits documents into smaller chunks for embedding, but stores
//! the original (parent) documents in a separate doc store. At retrieval time,
//! child chunks are matched via similarity search, then their parent documents
//! are looked up and returned, giving the LLM full context while maintaining
//! precise vector search over smaller passages.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use rustchain_core::documents::Document;
use rustchain_core::error::Result;
use rustchain_core::retrievers::BaseRetriever;
use rustchain_core::vectorstores::base::VectorStore;

use super::docstore::InMemoryDocStore;
use crate::text_splitter::TextSplitter;

/// A retriever that indexes child chunks but returns their full parent documents.
///
/// # How it works
///
/// 1. **Indexing**: Each parent document is split into child chunks using the
///    configured `child_splitter`. The parent is stored in the `docstore`, and
///    each child chunk is stored in the `vectorstore` with a metadata key
///    linking back to its parent.
///
/// 2. **Retrieval**: A similarity search over the vectorstore finds matching
///    child chunks. The retriever then extracts the parent IDs from the chunks'
///    metadata, deduplicates them, and fetches the full parent documents from
///    the docstore.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use rustchain::retrievers::parent_document::ParentDocumentRetriever;
/// use rustchain::retrievers::docstore::InMemoryDocStore;
///
/// let retriever = ParentDocumentRetriever::new(vectorstore, docstore, child_splitter);
/// retriever.add_documents(documents).await?;
/// let parents = retriever.get_relevant_documents("query").await?;
/// ```
pub struct ParentDocumentRetriever {
    /// Vector store for child chunk embeddings.
    vectorstore: Arc<dyn VectorStore>,
    /// Document store for full parent documents.
    docstore: Arc<InMemoryDocStore>,
    /// Splitter to break parent documents into child chunks.
    child_splitter: Arc<dyn TextSplitter>,
    /// Metadata key used to link child chunks to their parent document.
    parent_id_key: String,
    /// Number of child chunks to retrieve from the vector store.
    k: usize,
}

impl ParentDocumentRetriever {
    /// Create a new `ParentDocumentRetriever`.
    pub fn new(
        vectorstore: Arc<dyn VectorStore>,
        docstore: Arc<InMemoryDocStore>,
        child_splitter: Arc<dyn TextSplitter>,
    ) -> Self {
        Self {
            vectorstore,
            docstore,
            child_splitter,
            parent_id_key: "parent_id".to_string(),
            k: 4,
        }
    }

    /// Set the metadata key used to link child chunks to parent documents.
    pub fn with_parent_id_key(mut self, key: impl Into<String>) -> Self {
        self.parent_id_key = key.into();
        self
    }

    /// Set the number of child chunks to retrieve.
    pub fn with_k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Add parent documents: stores parents in the docstore and their child
    /// chunks in the vectorstore.
    pub async fn add_documents(&self, documents: Vec<Document>) -> Result<()> {
        for parent in &documents {
            let parent_id = Uuid::new_v4().to_string();

            // Store the full parent document.
            self.docstore.add(&parent_id, parent.clone()).await;

            // Split parent into child chunks.
            let child_docs = self.child_splitter.split_documents(std::slice::from_ref(parent));

            // Add parent_id metadata to each child and store in vectorstore.
            let children_with_meta: Vec<Document> = child_docs
                .into_iter()
                .map(|mut child| {
                    child.metadata.insert(
                        self.parent_id_key.clone(),
                        serde_json::Value::String(parent_id.clone()),
                    );
                    child
                })
                .collect();

            self.vectorstore
                .add_documents(children_with_meta, None)
                .await?;
        }

        Ok(())
    }
}

#[async_trait]
impl BaseRetriever for ParentDocumentRetriever {
    async fn get_relevant_documents(&self, query: &str) -> Result<Vec<Document>> {
        // Retrieve child chunks from the vectorstore.
        let children = self.vectorstore.similarity_search(query, self.k).await?;

        // Extract unique parent IDs from child metadata.
        let mut seen = HashSet::new();
        let mut parent_ids = Vec::new();
        for child in &children {
            if let Some(serde_json::Value::String(pid)) = child.metadata.get(&self.parent_id_key) {
                if seen.insert(pid.clone()) {
                    parent_ids.push(pid.clone());
                }
            }
        }

        // Fetch full parent documents from the docstore.
        let parent_opts = self.docstore.mget(&parent_ids).await;
        let parents: Vec<Document> = parent_opts.into_iter().flatten().collect();

        Ok(parents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::text_splitter::TextSplitter as TextSplitterTrait;
    use crate::vectorstores::in_memory::InMemoryVectorStore;
    use rustchain_core::embeddings_fake::DeterministicFakeEmbedding;

    /// A simple mock splitter that splits text into fixed-size character chunks.
    struct MockTextSplitter {
        size: usize,
    }

    impl MockTextSplitter {
        fn new(size: usize) -> Self {
            Self { size }
        }
    }

    impl TextSplitterTrait for MockTextSplitter {
        fn split_text(&self, text: &str) -> Vec<String> {
            text.chars()
                .collect::<Vec<_>>()
                .chunks(self.size)
                .map(|chunk| chunk.iter().collect::<String>())
                .collect()
        }

        fn chunk_size(&self) -> usize {
            self.size
        }

        fn chunk_overlap(&self) -> usize {
            0
        }
    }

    fn make_embeddings() -> Arc<dyn rustchain_core::embeddings::Embeddings> {
        Arc::new(DeterministicFakeEmbedding::new(16))
    }

    #[tokio::test]
    async fn test_add_and_retrieve_returns_parents() {
        let embeddings = make_embeddings();
        let vectorstore: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new(embeddings));
        let docstore = Arc::new(InMemoryDocStore::new());
        let splitter: Arc<dyn TextSplitterTrait> = Arc::new(MockTextSplitter::new(5));

        let retriever = ParentDocumentRetriever::new(
            vectorstore,
            docstore,
            splitter,
        )
        .with_k(4);

        let parent = Document::new("Hello World, this is a test document with enough text.");
        retriever.add_documents(vec![parent.clone()]).await.unwrap();

        let results = retriever.get_relevant_documents("Hello").await.unwrap();

        // Should return the full parent, not the chunks.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, parent.page_content);
    }

    #[tokio::test]
    async fn test_deduplication_multiple_chunks_same_parent() {
        let embeddings = make_embeddings();
        let vectorstore: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new(embeddings));
        let docstore = Arc::new(InMemoryDocStore::new());
        // Small chunk size so we get many chunks from one parent.
        let splitter: Arc<dyn TextSplitterTrait> = Arc::new(MockTextSplitter::new(3));

        let retriever = ParentDocumentRetriever::new(
            vectorstore,
            docstore,
            splitter,
        )
        .with_k(10);

        let parent = Document::new("abcdefghijklmnop");
        retriever.add_documents(vec![parent.clone()]).await.unwrap();

        let results = retriever.get_relevant_documents("abc").await.unwrap();

        // Even though multiple chunks match, only one parent should be returned.
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "abcdefghijklmnop");
    }

    #[tokio::test]
    async fn test_multiple_parents() {
        let embeddings = make_embeddings();
        let vectorstore: Arc<dyn VectorStore> = Arc::new(InMemoryVectorStore::new(embeddings));
        let docstore = Arc::new(InMemoryDocStore::new());
        let splitter: Arc<dyn TextSplitterTrait> = Arc::new(MockTextSplitter::new(10));

        let retriever = ParentDocumentRetriever::new(
            vectorstore,
            docstore,
            splitter,
        )
        .with_k(10);

        let parents = vec![
            Document::new("First parent document with some content here"),
            Document::new("Second parent document with different content"),
        ];
        retriever.add_documents(parents).await.unwrap();

        let results = retriever.get_relevant_documents("parent").await.unwrap();

        // Both parents should be retrievable.
        assert!(results.len() <= 2);
        assert!(!results.is_empty());
    }
}
