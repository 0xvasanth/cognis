//! In-memory vector store implementation.
//!
//! Stores documents and their embeddings in memory using `tokio::sync::RwLock`
//! for thread-safe concurrent access. Supports similarity search, scored search,
//! search by vector, and Maximal Marginal Relevance (MMR) search.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::RwLock;
use uuid::Uuid;

use cognis_core::documents::Document;
use cognis_core::embeddings::Embeddings;
use cognis_core::error::Result;
use cognis_core::vectorstores::base::{cosine_similarity, VectorStore};
use cognis_core::vectorstores::utils::maximal_marginal_relevance;

/// An entry in the in-memory store, pairing a document with its embedding.
#[derive(Debug, Clone)]
struct StoredEntry {
    document: Document,
    embedding: Vec<f32>,
}

/// In-memory vector store backed by a simple list of documents and embeddings.
///
/// Thread-safe via `Arc<RwLock<...>>`, suitable for testing, prototyping, and
/// small-to-medium workloads that fit in memory.
///
/// # Examples
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use cognis::vectorstores::in_memory::InMemoryVectorStore;
/// use cognis_core::embeddings_fake::DeterministicFakeEmbedding;
///
/// let embeddings = Arc::new(DeterministicFakeEmbedding::new(128));
/// let store = InMemoryVectorStore::new(embeddings);
/// store.add_texts(&["hello".into()], None, None).await.unwrap();
/// let results = store.similarity_search("hello", 1).await.unwrap();
/// ```
pub struct InMemoryVectorStore {
    embeddings: Arc<dyn Embeddings>,
    entries: Arc<RwLock<Vec<StoredEntry>>>,
}

impl InMemoryVectorStore {
    /// Create an empty in-memory vector store with the given embedding model.
    pub fn new(embeddings: Arc<dyn Embeddings>) -> Self {
        Self {
            embeddings,
            entries: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Create an in-memory vector store pre-populated with the given documents.
    pub async fn from_documents(
        documents: Vec<Document>,
        embeddings: Arc<dyn Embeddings>,
    ) -> Result<Self> {
        let store = Self::new(embeddings);
        store.add_documents(documents, None).await?;
        Ok(store)
    }

    /// Create an in-memory vector store from raw texts and optional metadata.
    pub async fn from_texts(
        texts: &[String],
        metadatas: Option<&[HashMap<String, Value>]>,
        embeddings: Arc<dyn Embeddings>,
    ) -> Result<Self> {
        let store = Self::new(embeddings);
        store.add_texts(texts, metadatas, None).await?;
        Ok(store)
    }

    /// Internal helper: search by a pre-computed embedding vector, returning
    /// `(Document, score)` pairs sorted by descending cosine similarity.
    async fn search_by_vector_with_score(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<(Document, f32)>> {
        let entries = self.entries.read().await;
        let mut scored: Vec<(Document, f32)> = entries
            .iter()
            .map(|e| {
                let score = cosine_similarity(embedding, &e.embedding);
                (e.document.clone(), score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn add_texts(
        &self,
        texts: &[String],
        metadatas: Option<&[HashMap<String, Value>]>,
        ids: Option<&[String]>,
    ) -> Result<Vec<String>> {
        let embeddings_vec = self.embeddings.embed_documents(texts.to_vec()).await?;

        let mut entries = self.entries.write().await;
        let mut result_ids = Vec::with_capacity(texts.len());

        for (i, text) in texts.iter().enumerate() {
            let id = ids
                .and_then(|id_list| id_list.get(i).cloned())
                .unwrap_or_else(|| Uuid::new_v4().to_string());

            let metadata = metadatas
                .and_then(|m| m.get(i).cloned())
                .unwrap_or_default();

            let doc = Document::new(text.clone())
                .with_id(id.clone())
                .with_metadata(metadata);

            entries.push(StoredEntry {
                document: doc,
                embedding: embeddings_vec[i].clone(),
            });

            result_ids.push(id);
        }

        Ok(result_ids)
    }

    async fn add_documents(
        &self,
        documents: Vec<Document>,
        ids: Option<Vec<String>>,
    ) -> Result<Vec<String>> {
        let texts: Vec<String> = documents.iter().map(|d| d.page_content.clone()).collect();
        let metadatas: Vec<HashMap<String, Value>> =
            documents.iter().map(|d| d.metadata.clone()).collect();
        let id_refs: Option<Vec<String>> = ids.or_else(|| {
            let doc_ids: Vec<String> = documents.iter().filter_map(|d| d.id.clone()).collect();
            if doc_ids.len() == documents.len() {
                Some(doc_ids)
            } else {
                None
            }
        });
        let id_slice_ref: Option<&[String]> = id_refs.as_deref();
        self.add_texts(&texts, Some(&metadatas), id_slice_ref).await
    }

    async fn delete(&self, ids: Option<&[String]>) -> Result<bool> {
        let Some(ids) = ids else {
            return Ok(false);
        };
        let mut entries = self.entries.write().await;
        let before = entries.len();
        entries.retain(|e| {
            e.document
                .id
                .as_ref()
                .map(|id| !ids.contains(id))
                .unwrap_or(true)
        });
        Ok(entries.len() < before)
    }

    async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Document>> {
        let entries = self.entries.read().await;
        let docs: Vec<Document> = entries
            .iter()
            .filter(|e| {
                e.document
                    .id
                    .as_ref()
                    .map(|id| ids.contains(id))
                    .unwrap_or(false)
            })
            .map(|e| e.document.clone())
            .collect();
        Ok(docs)
    }

    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let results = self.similarity_search_with_score(query, k).await?;
        Ok(results.into_iter().map(|(doc, _)| doc).collect())
    }

    async fn similarity_search_with_score(
        &self,
        query: &str,
        k: usize,
    ) -> Result<Vec<(Document, f32)>> {
        let query_embedding = self.embeddings.embed_query(query).await?;
        self.search_by_vector_with_score(&query_embedding, k).await
    }

    async fn similarity_search_by_vector(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<Document>> {
        let results = self.search_by_vector_with_score(embedding, k).await?;
        Ok(results.into_iter().map(|(doc, _)| doc).collect())
    }

    async fn max_marginal_relevance_search(
        &self,
        query: &str,
        k: usize,
        fetch_k: usize,
        lambda_mult: f32,
    ) -> Result<Vec<Document>> {
        let query_embedding = self.embeddings.embed_query(query).await?;
        let entries = self.entries.read().await;

        if entries.is_empty() {
            return Ok(vec![]);
        }

        // Get top fetch_k candidates by similarity first.
        let mut scored: Vec<(usize, f32)> = entries
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let score = cosine_similarity(&query_embedding, &e.embedding);
                (i, score)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(fetch_k);

        // Convert to f64 for the MMR utility function.
        let query_emb_f64: Vec<f64> = query_embedding.iter().map(|&v| v as f64).collect();
        let candidate_embeddings: Vec<Vec<f64>> = scored
            .iter()
            .map(|&(idx, _)| entries[idx].embedding.iter().map(|&v| v as f64).collect())
            .collect();

        let mmr_indices = maximal_marginal_relevance(
            &query_emb_f64,
            &candidate_embeddings,
            lambda_mult as f64,
            k,
        );

        let docs = mmr_indices
            .into_iter()
            .map(|mmr_idx| entries[scored[mmr_idx].0].document.clone())
            .collect();

        Ok(docs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::embeddings_fake::DeterministicFakeEmbedding;

    fn make_embeddings() -> Arc<dyn Embeddings> {
        Arc::new(DeterministicFakeEmbedding::new(16))
    }

    #[tokio::test]
    async fn test_add_texts_and_similarity_search() {
        let store = InMemoryVectorStore::new(make_embeddings());
        let texts = vec!["cat".into(), "dog".into(), "fish".into()];
        let ids = store.add_texts(&texts, None, None).await.unwrap();
        assert_eq!(ids.len(), 3);

        let results = store.similarity_search("cat", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "cat");
    }

    #[tokio::test]
    async fn test_add_texts_with_custom_ids() {
        let store = InMemoryVectorStore::new(make_embeddings());
        let texts = vec!["hello".into()];
        let custom_ids = vec!["my-id-1".to_string()];
        let ids = store
            .add_texts(&texts, None, Some(&custom_ids))
            .await
            .unwrap();
        assert_eq!(ids, vec!["my-id-1"]);

        let docs = store.get_by_ids(&["my-id-1".into()]).await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].page_content, "hello");
    }

    #[tokio::test]
    async fn test_add_texts_with_metadata() {
        let store = InMemoryVectorStore::new(make_embeddings());
        let texts = vec!["hello".into()];
        let mut meta = HashMap::new();
        meta.insert("source".into(), Value::String("test".into()));
        let metadatas = vec![meta.clone()];

        store
            .add_texts(&texts, Some(&metadatas), None)
            .await
            .unwrap();
        let results = store.similarity_search("hello", 1).await.unwrap();
        assert_eq!(results[0].metadata.get("source").unwrap(), "test");
    }

    #[tokio::test]
    async fn test_add_documents() {
        let store = InMemoryVectorStore::new(make_embeddings());
        let docs = vec![
            Document::new("alpha").with_id("a1"),
            Document::new("beta").with_id("b1"),
        ];
        let ids = store.add_documents(docs, None).await.unwrap();
        assert_eq!(ids, vec!["a1", "b1"]);

        let retrieved = store.get_by_ids(&["a1".into(), "b1".into()]).await.unwrap();
        assert_eq!(retrieved.len(), 2);
    }

    #[tokio::test]
    async fn test_delete() {
        let store = InMemoryVectorStore::new(make_embeddings());
        let texts = vec!["a".into(), "b".into(), "c".into()];
        let ids = store.add_texts(&texts, None, None).await.unwrap();

        let deleted = store.delete(Some(&[ids[1].clone()])).await.unwrap();
        assert!(deleted);

        let remaining = store.similarity_search("a", 10).await.unwrap();
        assert_eq!(remaining.len(), 2);
        assert!(remaining.iter().all(|d| d.page_content != "b"));
    }

    #[tokio::test]
    async fn test_delete_nonexistent() {
        let store = InMemoryVectorStore::new(make_embeddings());
        let texts = vec!["a".into()];
        store.add_texts(&texts, None, None).await.unwrap();

        let deleted = store.delete(Some(&["nonexistent".into()])).await.unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_delete_none() {
        let store = InMemoryVectorStore::new(make_embeddings());
        let deleted = store.delete(None).await.unwrap();
        assert!(!deleted);
    }

    #[tokio::test]
    async fn test_get_by_ids_missing() {
        let store = InMemoryVectorStore::new(make_embeddings());
        let texts = vec!["a".into()];
        store.add_texts(&texts, None, None).await.unwrap();

        let docs = store.get_by_ids(&["nonexistent".into()]).await.unwrap();
        assert!(docs.is_empty());
    }

    #[tokio::test]
    async fn test_similarity_search_with_score() {
        let store = InMemoryVectorStore::new(make_embeddings());
        let texts = vec!["cat".into(), "dog".into(), "fish".into()];
        store.add_texts(&texts, None, None).await.unwrap();

        let results = store.similarity_search_with_score("cat", 3).await.unwrap();
        assert_eq!(results.len(), 3);
        // The first result should be "cat" itself with the highest score.
        assert_eq!(results[0].0.page_content, "cat");
        // Scores should be in descending order.
        assert!(results[0].1 >= results[1].1);
        assert!(results[1].1 >= results[2].1);
    }

    #[tokio::test]
    async fn test_similarity_search_by_vector() {
        let embeddings = make_embeddings();
        let store = InMemoryVectorStore::new(embeddings.clone());
        let texts = vec!["sun".into(), "moon".into()];
        store.add_texts(&texts, None, None).await.unwrap();

        let query_vec = embeddings.embed_query("sun").await.unwrap();
        let results = store
            .similarity_search_by_vector(&query_vec, 1)
            .await
            .unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "sun");
    }

    #[tokio::test]
    async fn test_max_marginal_relevance_search() {
        let store = InMemoryVectorStore::new(make_embeddings());
        let texts: Vec<String> = vec![
            "apple".into(),
            "banana".into(),
            "cherry".into(),
            "date".into(),
        ];
        store.add_texts(&texts, None, None).await.unwrap();

        let results = store
            .max_marginal_relevance_search("apple", 2, 4, 0.5)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
        // First result should be most similar to query.
        assert_eq!(results[0].page_content, "apple");
    }

    #[tokio::test]
    async fn test_mmr_empty_store() {
        let store = InMemoryVectorStore::new(make_embeddings());
        let results = store
            .max_marginal_relevance_search("anything", 2, 4, 0.5)
            .await
            .unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_from_documents() {
        let docs = vec![Document::new("hello world"), Document::new("goodbye world")];
        let store = InMemoryVectorStore::from_documents(docs, make_embeddings())
            .await
            .unwrap();
        let results = store.similarity_search("hello", 1).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].page_content, "hello world");
    }

    #[tokio::test]
    async fn test_from_texts() {
        let texts = vec!["foo".into(), "bar".into()];
        let store = InMemoryVectorStore::from_texts(&texts, None, make_embeddings())
            .await
            .unwrap();
        let results = store.similarity_search("foo", 2).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_k_larger_than_store() {
        let store = InMemoryVectorStore::new(make_embeddings());
        let texts = vec!["only".into()];
        store.add_texts(&texts, None, None).await.unwrap();

        let results = store.similarity_search("only", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }
}
