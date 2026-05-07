//! In-process vector store. Linear scan for similarity search — fine
//! for prototyping and small corpora (< ~10k docs); not for production
//! at scale.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use uuid::Uuid;

use cognis_core::{CognisError, Result};

use crate::distance::Distance;
use crate::embeddings::Embeddings;

use super::{SearchResult, VectorStore};

#[derive(Clone)]
struct StoredDoc {
    id: String,
    text: String,
    vector: Vec<f32>,
    metadata: HashMap<String, serde_json::Value>,
}

/// Linear-scan in-memory vector store.
pub struct InMemoryVectorStore {
    embedder: Arc<dyn Embeddings>,
    distance: Distance,
    docs: Vec<StoredDoc>,
}

impl InMemoryVectorStore {
    /// New empty store with the given embedder and Cosine distance.
    pub fn new(embedder: Arc<dyn Embeddings>) -> Self {
        Self::with_distance(embedder, Distance::Cosine)
    }

    /// New empty store with explicit distance.
    pub fn with_distance(embedder: Arc<dyn Embeddings>, distance: Distance) -> Self {
        Self {
            embedder,
            distance,
            docs: Vec::new(),
        }
    }

    /// Currently configured distance metric.
    pub fn distance(&self) -> Distance {
        self.distance
    }
}

#[async_trait]
impl VectorStore for InMemoryVectorStore {
    async fn add_texts(
        &mut self,
        texts: Vec<String>,
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let vectors = self.embedder.embed_documents(texts.clone()).await?;
        self.add_vectors(vectors, texts, metadata).await
    }

    async fn add_vectors(
        &mut self,
        vectors: Vec<Vec<f32>>,
        texts: Vec<String>,
        metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    ) -> Result<Vec<String>> {
        if vectors.len() != texts.len() {
            return Err(CognisError::Configuration(format!(
                "vectors.len() ({}) must equal texts.len() ({})",
                vectors.len(),
                texts.len()
            )));
        }
        if let Some(m) = &metadata {
            if m.len() != texts.len() {
                return Err(CognisError::Configuration(format!(
                    "metadata.len() ({}) must equal texts.len() ({})",
                    m.len(),
                    texts.len()
                )));
            }
        }

        let mut ids = Vec::with_capacity(texts.len());
        for (i, (text, vector)) in texts.into_iter().zip(vectors).enumerate() {
            let id = Uuid::new_v4().to_string();
            let md = metadata.as_ref().map(|m| m[i].clone()).unwrap_or_default();
            ids.push(id.clone());
            self.docs.push(StoredDoc {
                id,
                text,
                vector,
                metadata: md,
            });
        }
        Ok(ids)
    }

    async fn similarity_search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        let qv = self.embedder.embed_query(query.to_string()).await?;
        self.similarity_search_by_vector(qv, k).await
    }

    async fn similarity_search_by_vector(
        &self,
        query_vector: Vec<f32>,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        if self.docs.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        // Compute scores.
        let mut scored: Vec<(f32, &StoredDoc)> = self
            .docs
            .iter()
            .map(|d| (self.distance.similarity(&query_vector, &d.vector), d))
            .collect();

        // Sort descending by score.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored
            .into_iter()
            .take(k)
            .map(|(score, d)| SearchResult {
                id: d.id.clone(),
                text: d.text.clone(),
                score,
                metadata: d.metadata.clone(),
            })
            .collect())
    }

    async fn delete(&mut self, ids: Vec<String>) -> Result<()> {
        let to_delete: std::collections::HashSet<String> = ids.into_iter().collect();
        self.docs.retain(|d| !to_delete.contains(&d.id));
        Ok(())
    }

    fn len(&self) -> usize {
        self.docs.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::FakeEmbeddings;

    fn fake_embedder(dim: usize) -> Arc<dyn Embeddings> {
        Arc::new(FakeEmbeddings::new(dim))
    }

    #[tokio::test]
    async fn add_texts_assigns_ids() {
        let mut store = InMemoryVectorStore::new(fake_embedder(8));
        let ids = store
            .add_texts(vec!["a".into(), "b".into(), "c".into()], None)
            .await
            .unwrap();
        assert_eq!(ids.len(), 3);
        assert_eq!(store.len(), 3);
        // IDs should be unique.
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(unique.len(), 3);
    }

    #[tokio::test]
    async fn search_returns_matches_in_order() {
        let mut store = InMemoryVectorStore::new(fake_embedder(8));
        store
            .add_texts(vec!["dog".into(), "cat".into(), "fish".into()], None)
            .await
            .unwrap();

        let results = store.similarity_search("dog", 2).await.unwrap();
        assert_eq!(results.len(), 2);
        // The exact-match query "dog" should be the top result (FakeEmbeddings
        // is deterministic, so the same input yields the same vector).
        assert_eq!(results[0].text, "dog");
    }

    #[tokio::test]
    async fn search_respects_k() {
        let mut store = InMemoryVectorStore::new(fake_embedder(8));
        store
            .add_texts((0..10).map(|i| format!("doc {i}")).collect(), None)
            .await
            .unwrap();
        let r1 = store.similarity_search("doc 5", 1).await.unwrap();
        let r5 = store.similarity_search("doc 5", 5).await.unwrap();
        assert_eq!(r1.len(), 1);
        assert_eq!(r5.len(), 5);
    }

    #[tokio::test]
    async fn metadata_roundtrip() {
        let mut store = InMemoryVectorStore::new(fake_embedder(8));
        let mut md = HashMap::new();
        md.insert("source".into(), serde_json::json!("wiki"));
        md.insert("year".into(), serde_json::json!(2024));
        store
            .add_texts(vec!["hello".into()], Some(vec![md.clone()]))
            .await
            .unwrap();
        let r = store.similarity_search("hello", 1).await.unwrap();
        assert_eq!(r[0].metadata.get("source").unwrap(), "wiki");
        assert_eq!(r[0].metadata.get("year").unwrap(), 2024);
    }

    #[tokio::test]
    async fn add_vectors_dimension_mismatch_errors() {
        let mut store = InMemoryVectorStore::new(fake_embedder(8));
        let err = store
            .add_vectors(vec![vec![0.1; 8], vec![0.2; 8]], vec!["one".into()], None)
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("must equal"));
    }

    #[tokio::test]
    async fn delete_removes_docs() {
        let mut store = InMemoryVectorStore::new(fake_embedder(8));
        let ids = store
            .add_texts(vec!["a".into(), "b".into(), "c".into()], None)
            .await
            .unwrap();
        store.delete(vec![ids[1].clone()]).await.unwrap();
        assert_eq!(store.len(), 2);
        let r = store.similarity_search("b", 5).await.unwrap();
        // "b" has been deleted, so it shouldn't appear in results.
        assert!(!r.iter().any(|s| s.text == "b"));
    }

    #[tokio::test]
    async fn delete_unknown_ids_silent() {
        let mut store = InMemoryVectorStore::new(fake_embedder(8));
        store.add_texts(vec!["a".into()], None).await.unwrap();
        // No error even though the ID doesn't exist.
        store.delete(vec!["nonexistent".into()]).await.unwrap();
        assert_eq!(store.len(), 1);
    }

    #[tokio::test]
    async fn empty_store_search_returns_empty() {
        let store = InMemoryVectorStore::new(fake_embedder(8));
        let r = store.similarity_search("anything", 5).await.unwrap();
        assert!(r.is_empty());
    }

    #[tokio::test]
    async fn k_zero_returns_empty() {
        let mut store = InMemoryVectorStore::new(fake_embedder(8));
        store.add_texts(vec!["a".into()], None).await.unwrap();
        let r = store.similarity_search("a", 0).await.unwrap();
        assert!(r.is_empty());
    }
}
