//! Deterministic fake embeddings based on text hashing. Useful for tests
//! and offline development. NOT semantically meaningful — semantic
//! similarity won't transfer.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use async_trait::async_trait;

use cognis2_core::Result;

use super::Embeddings;

/// Produces deterministic `dim`-length f32 vectors via repeated hashing
/// of the input text. Same input always yields the same vector.
pub struct FakeEmbeddings {
    dim: usize,
    model_name: String,
}

impl FakeEmbeddings {
    /// New fake embedder with the given dimensionality. Default model
    /// name is "fake-embeddings".
    pub fn new(dim: usize) -> Self {
        Self {
            dim: dim.max(1),
            model_name: "fake-embeddings".to_string(),
        }
    }

    /// Override the model name (defaults to "fake-embeddings").
    pub fn with_model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name = name.into();
        self
    }

    fn hash_to_vec(&self, text: &str) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.dim);
        for i in 0..self.dim {
            let mut hasher = DefaultHasher::new();
            (text, i).hash(&mut hasher);
            // Map u64 → f32 in [-1, 1].
            let raw = hasher.finish();
            let normalized = (raw as f64 / u64::MAX as f64) * 2.0 - 1.0;
            out.push(normalized as f32);
        }
        out
    }
}

#[async_trait]
impl Embeddings for FakeEmbeddings {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| self.hash_to_vec(t)).collect())
    }

    fn dimensions(&self) -> Option<usize> {
        Some(self.dim)
    }

    fn model(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn deterministic_same_input_same_vector() {
        let e = FakeEmbeddings::new(8);
        let v1 = e.embed_query("hello".into()).await.unwrap();
        let v2 = e.embed_query("hello".into()).await.unwrap();
        assert_eq!(v1, v2);
    }

    #[tokio::test]
    async fn different_input_different_vector() {
        let e = FakeEmbeddings::new(8);
        let v1 = e.embed_query("hello".into()).await.unwrap();
        let v2 = e.embed_query("world".into()).await.unwrap();
        assert_ne!(v1, v2);
    }

    #[tokio::test]
    async fn batch_preserves_order() {
        let e = FakeEmbeddings::new(4);
        let vecs = e
            .embed_documents(vec!["a".into(), "b".into(), "c".into()])
            .await
            .unwrap();
        assert_eq!(vecs.len(), 3);
        // Each vector should match what embed_query would produce.
        let sa = e.embed_query("a".into()).await.unwrap();
        let sb = e.embed_query("b".into()).await.unwrap();
        let sc = e.embed_query("c".into()).await.unwrap();
        assert_eq!(vecs[0], sa);
        assert_eq!(vecs[1], sb);
        assert_eq!(vecs[2], sc);
    }

    #[tokio::test]
    async fn dimensions_match_constructor() {
        let e = FakeEmbeddings::new(64);
        assert_eq!(e.dimensions(), Some(64));
        let v = e.embed_query("x".into()).await.unwrap();
        assert_eq!(v.len(), 64);
    }

    #[tokio::test]
    async fn vector_values_in_range() {
        let e = FakeEmbeddings::new(16);
        let v = e.embed_query("test".into()).await.unwrap();
        for x in v {
            assert!(
                (-1.0..=1.0).contains(&x),
                "vector value out of [-1, 1]: {x}"
            );
        }
    }

    #[test]
    fn model_name_default_and_override() {
        let e = FakeEmbeddings::new(4);
        assert_eq!(e.model(), "fake-embeddings");
        let e2 = FakeEmbeddings::new(4).with_model_name("custom");
        assert_eq!(e2.model(), "custom");
    }

    #[test]
    fn min_dim_one_when_zero_passed() {
        let e = FakeEmbeddings::new(0);
        assert_eq!(e.dimensions(), Some(1));
    }
}
