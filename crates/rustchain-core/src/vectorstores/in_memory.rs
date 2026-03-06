use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::RwLock;

use crate::documents::Document;
use crate::embeddings::Embeddings;
use crate::error::Result;

use super::base::{cosine_similarity, VectorStore, VectorStoreRetriever, SearchType};
use super::utils::maximal_marginal_relevance;

/// An entry stored in the in-memory vector store.
struct VectorEntry {
    id: String,
    vector: Vec<f32>,
    text: String,
    metadata: HashMap<String, Value>,
}

/// A simple in-memory vector store backed by a `HashMap` and cosine similarity.
///
/// Uses the provided `Embeddings` implementation to embed texts and performs
/// cosine similarity for search. Supports similarity search, search with
/// scores, search by vector, and maximal marginal relevance search.
pub struct InMemoryVectorStore {
    store: RwLock<HashMap<String, VectorEntry>>,
    embedding: Arc<dyn Embeddings>,
}

impl InMemoryVectorStore {
    /// Create a new empty in-memory vector store with the given embedding model.
    pub fn new(embedding: Arc<dyn Embeddings>) -> Self {
        Self {
            store: RwLock::new(HashMap::new()),
            embedding,
        }
    }

    /// Create an in-memory vector store from a list of texts, optionally with metadata.
    pub async fn from_texts(
        texts: Vec<String>,
        embedding: Arc<dyn Embeddings>,
        metadatas: Option<Vec<HashMap<String, Value>>>,
    ) -> Result<Self> {
        let store = Self::new(embedding);
        let docs: Vec<Document> = texts
            .into_iter()
            .enumerate()
            .map(|(i, text)| {
                let mut doc = Document::new(text);
                if let Some(ref metas) = metadatas {
                    if let Some(meta) = metas.get(i) {
                        doc.metadata = meta.clone();
                    }
                }
                doc
            })
            .collect();
        store.add_documents(docs, None).await?;
        Ok(store)
    }

    /// Convert this vector store into a `VectorStoreRetriever` with default settings.
    pub fn as_retriever(self: &Arc<Self>) -> VectorStoreRetriever {
        VectorStoreRetriever::from_vectorstore(self.clone())
    }

    /// Convert this vector store into a `VectorStoreRetriever` with custom settings.
    pub fn as_retriever_with(
        self: &Arc<Self>,
        search_type: SearchType,
        k: usize,
    ) -> VectorStoreRetriever {
        VectorStoreRetriever::new(self.clone(), search_type, k)
    }

    /// Internal helper: search by vector and return documents with scores and vectors.
    async fn similarity_search_by_vector_with_score(
        &self,
        query_vec: &[f32],
        k: usize,
    ) -> Result<Vec<(Document, f32, Vec<f32>)>> {
        let store = self.store.read().await;
        let mut scored: Vec<(Document, f32, Vec<f32>)> = store
            .values()
            .map(|entry| {
                let score = cosine_similarity(query_vec, &entry.vector);
                let mut doc = Document::new(&entry.text);
                doc.id = Some(entry.id.clone());
                doc.metadata = entry.metadata.clone();
                (doc, score, entry.vector.clone())
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
        let documents: Vec<Document> = texts
            .iter()
            .enumerate()
            .map(|(i, text)| {
                let mut doc = Document::new(text.clone());
                if let Some(metas) = metadatas {
                    if let Some(meta) = metas.get(i) {
                        doc.metadata = meta.clone();
                    }
                }
                if let Some(id_list) = ids {
                    if let Some(id) = id_list.get(i) {
                        doc.id = Some(id.clone());
                    }
                }
                doc
            })
            .collect();
        let explicit_ids = ids.map(|s| s.to_vec());
        self.add_documents(documents, explicit_ids).await
    }

    async fn add_documents(
        &self,
        documents: Vec<Document>,
        ids: Option<Vec<String>>,
    ) -> Result<Vec<String>> {
        let texts: Vec<String> = documents.iter().map(|d| d.page_content.clone()).collect();
        let vectors = self.embedding.embed_documents(texts).await?;
        let mut store = self.store.write().await;
        let mut result_ids = Vec::with_capacity(documents.len());
        for (i, doc) in documents.into_iter().enumerate() {
            let id = ids
                .as_ref()
                .and_then(|ids| ids.get(i).cloned())
                .or_else(|| doc.id.clone())
                .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            store.insert(
                id.clone(),
                VectorEntry {
                    id: id.clone(),
                    vector: vectors[i].clone(),
                    text: doc.page_content,
                    metadata: doc.metadata,
                },
            );
            result_ids.push(id);
        }
        Ok(result_ids)
    }

    async fn delete(&self, ids: Option<&[String]>) -> Result<bool> {
        if let Some(ids) = ids {
            let mut store = self.store.write().await;
            for id in ids {
                store.remove(id);
            }
        }
        Ok(true)
    }

    async fn get_by_ids(&self, ids: &[String]) -> Result<Vec<Document>> {
        let store = self.store.read().await;
        Ok(ids
            .iter()
            .filter_map(|id| {
                store.get(id).map(|entry| {
                    let mut doc = Document::new(&entry.text);
                    doc.id = Some(entry.id.clone());
                    doc.metadata = entry.metadata.clone();
                    doc
                })
            })
            .collect())
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
        let query_vec = self.embedding.embed_query(query).await?;
        let results = self
            .similarity_search_by_vector_with_score(&query_vec, k)
            .await?;
        Ok(results.into_iter().map(|(doc, score, _)| (doc, score)).collect())
    }

    async fn similarity_search_by_vector(
        &self,
        embedding: &[f32],
        k: usize,
    ) -> Result<Vec<Document>> {
        let results = self
            .similarity_search_by_vector_with_score(embedding, k)
            .await?;
        Ok(results.into_iter().map(|(doc, _, _)| doc).collect())
    }

    async fn max_marginal_relevance_search(
        &self,
        query: &str,
        k: usize,
        fetch_k: usize,
        lambda_mult: f32,
    ) -> Result<Vec<Document>> {
        let query_vec = self.embedding.embed_query(query).await?;

        // Fetch more candidates than we need for MMR re-ranking.
        let prefetch = self
            .similarity_search_by_vector_with_score(&query_vec, fetch_k)
            .await?;

        if prefetch.is_empty() {
            return Ok(vec![]);
        }

        // Convert f32 vectors to f64 for the MMR utility function.
        let query_f64: Vec<f64> = query_vec.iter().map(|&x| x as f64).collect();
        let candidate_embeddings: Vec<Vec<f64>> = prefetch
            .iter()
            .map(|(_, _, vec)| vec.iter().map(|&x| x as f64).collect())
            .collect();

        let selected_indices = maximal_marginal_relevance(
            &query_f64,
            &candidate_embeddings,
            lambda_mult as f64,
            k,
        );

        Ok(selected_indices
            .into_iter()
            .map(|idx| prefetch[idx].0.clone())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings_fake::DeterministicFakeEmbedding;
    use crate::retrievers::BaseRetriever;
    use crate::vectorstores::base::{cosine_similarity, SearchType};

    const EPSILON: f32 = 1e-6;

    // --- cosine_similarity tests (f32 version from base.rs) ---

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0_f32, 0.0, 0.0];
        let b = vec![1.0_f32, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        // cos(45 degrees) = 1/sqrt(2) ~= 0.7071
        assert!((sim - (1.0_f32 / 2.0_f32.sqrt())).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0_f32, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0_f32, 0.0];
        let b = vec![0.0_f32, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < EPSILON);
    }

    // --- InMemoryVectorStore tests ---

    fn make_embedding() -> Arc<dyn Embeddings> {
        Arc::new(DeterministicFakeEmbedding::new(16))
    }

    #[tokio::test]
    async fn test_add_documents() {
        let emb = make_embedding();
        let store = InMemoryVectorStore::new(emb);
        let docs = vec![
            Document::new("hello world"),
            Document::new("foo bar"),
            Document::new("baz qux"),
        ];
        let ids = store.add_documents(docs, None).await.unwrap();
        assert_eq!(ids.len(), 3);

        // Verify documents can be retrieved by their IDs.
        let retrieved = store.get_by_ids(&ids).await.unwrap();
        assert_eq!(retrieved.len(), 3);
        let contents: Vec<&str> = retrieved.iter().map(|d| d.page_content.as_str()).collect();
        assert!(contents.contains(&"hello world"));
        assert!(contents.contains(&"foo bar"));
        assert!(contents.contains(&"baz qux"));
    }

    #[tokio::test]
    async fn test_similarity_search() {
        let emb = make_embedding();
        let store = InMemoryVectorStore::new(emb);
        let docs = vec![
            Document::new("the cat sat on the mat"),
            Document::new("the dog played in the park"),
            Document::new("a fish swam in the sea"),
        ];
        store.add_documents(docs, None).await.unwrap();

        // Search should return results (exact ordering depends on hash-based embeddings).
        let results = store.similarity_search("cat on mat", 2).await.unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_similarity_search_top_k() {
        let emb = make_embedding();
        let store = InMemoryVectorStore::new(emb);
        let docs = vec![
            Document::new("alpha"),
            Document::new("beta"),
            Document::new("gamma"),
            Document::new("delta"),
            Document::new("epsilon"),
        ];
        store.add_documents(docs, None).await.unwrap();

        // k=3 should return exactly 3 results.
        let results = store.similarity_search("alpha", 3).await.unwrap();
        assert_eq!(results.len(), 3);

        // Results with scores should be sorted descending by score.
        let scored = store
            .similarity_search_with_score("alpha", 5)
            .await
            .unwrap();
        assert_eq!(scored.len(), 5);
        for i in 0..scored.len() - 1 {
            assert!(
                scored[i].1 >= scored[i + 1].1,
                "Results must be sorted by descending score: {} < {}",
                scored[i].1,
                scored[i + 1].1,
            );
        }
    }

    #[tokio::test]
    async fn test_similarity_search_by_text() {
        let emb = make_embedding();
        let store = InMemoryVectorStore::new(emb);
        let docs = vec![
            Document::new("rust programming language"),
            Document::new("python programming language"),
            Document::new("unrelated topic about cooking"),
        ];
        store.add_documents(docs, None).await.unwrap();

        // Text-based search embeds the query and searches.
        let results = store
            .similarity_search("programming", 2)
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_similarity_search_with_score_values() {
        let emb = make_embedding();
        let store = InMemoryVectorStore::new(emb);

        // Add a single document and query with the same text.
        let docs = vec![Document::new("exact match test")];
        store.add_documents(docs, None).await.unwrap();

        let scored = store
            .similarity_search_with_score("exact match test", 1)
            .await
            .unwrap();
        assert_eq!(scored.len(), 1);
        // Same text should produce identical embedding, so cosine similarity = 1.0.
        assert!(
            (scored[0].1 - 1.0).abs() < EPSILON,
            "Identical text should have similarity ~1.0, got {}",
            scored[0].1,
        );
    }

    #[tokio::test]
    async fn test_vector_store_retriever() {
        let emb = make_embedding();
        let store = Arc::new(InMemoryVectorStore::new(emb));
        let docs = vec![
            Document::new("document one"),
            Document::new("document two"),
            Document::new("document three"),
        ];
        store.add_documents(docs, None).await.unwrap();

        // Create a retriever with k=2.
        let retriever = store.as_retriever_with(SearchType::Similarity, 2);
        let results = retriever
            .get_relevant_documents("document")
            .await
            .unwrap();
        assert_eq!(results.len(), 2);
    }

    #[tokio::test]
    async fn test_from_texts() {
        let emb = make_embedding();
        let texts = vec![
            "hello".to_string(),
            "world".to_string(),
        ];
        let store = InMemoryVectorStore::from_texts(texts, emb, None)
            .await
            .unwrap();
        let results = store.similarity_search("hello", 1).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn test_delete_documents() {
        let emb = make_embedding();
        let store = InMemoryVectorStore::new(emb);
        let docs = vec![Document::new("to delete"), Document::new("to keep")];
        let ids = store.add_documents(docs, None).await.unwrap();

        // Delete the first document.
        store.delete(Some(&[ids[0].clone()])).await.unwrap();

        // Only one document should remain.
        let all = store.get_by_ids(&ids).await.unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].page_content, "to keep");
    }
}
