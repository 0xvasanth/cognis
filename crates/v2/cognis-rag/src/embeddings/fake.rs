//! stub — filled in Task 4
use async_trait::async_trait;
use cognis2_core::Result;
use super::Embeddings;

/// Placeholder — real impl in Task 4.
pub struct FakeEmbeddings;

#[async_trait]
impl Embeddings for FakeEmbeddings {
    async fn embed_documents(&self, _texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        Ok(Vec::new())
    }
    fn model(&self) -> &str {
        "fake-embeddings"
    }
}
