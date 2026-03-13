//! Fake embedding models for testing.
//!
//! Mirrors Python `langchain_core.embeddings.fake`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use async_trait::async_trait;

use crate::embeddings::Embeddings;
use crate::error::Result;

/// Fake embedding model that returns random-looking (but deterministic per-text)
/// embeddings based on hashing.
///
/// Useful for testing without requiring a real embedding service.
pub struct DeterministicFakeEmbedding {
    pub size: usize,
}

impl DeterministicFakeEmbedding {
    pub fn new(size: usize) -> Self {
        Self { size }
    }

    fn get_seed(text: &str) -> u64 {
        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        hasher.finish()
    }

    fn get_embedding(&self, seed: u64) -> Vec<f32> {
        // Simple deterministic pseudo-random using the seed.
        let mut state = seed;
        (0..self.size)
            .map(|_| {
                // xorshift64
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                // Map to [-1, 1] range
                let normalized = (state as f64) / (u64::MAX as f64);
                (normalized * 2.0 - 1.0) as f32
            })
            .collect()
    }
}

#[async_trait]
impl Embeddings for DeterministicFakeEmbedding {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Ok(texts
            .iter()
            .map(|t| self.get_embedding(Self::get_seed(t)))
            .collect())
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.get_embedding(Self::get_seed(text)))
    }
}

/// Fake embedding model that returns constant embeddings (all zeros).
///
/// Useful when you don't care about embedding values at all.
pub struct FakeConstantEmbedding {
    pub size: usize,
}

impl FakeConstantEmbedding {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

#[async_trait]
impl Embeddings for FakeConstantEmbedding {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0_f32; self.size]).collect())
    }

    async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(vec![0.0_f32; self.size])
    }
}
