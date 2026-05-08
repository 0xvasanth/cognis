//! Cross-encoder scoring trait + cross-encoder-based reranker.
//!
//! Distinct from LLM-as-judge reranking: a cross-encoder is a model that
//! takes `(query, doc)` and emits a relevance score in one forward pass.
//! Common production choice for first-stage RAG reranking.
//!
//! cognis doesn't bundle a model — implement [`CrossEncoder`] against
//! whatever you actually run (Cohere rerank, hosted bge-reranker, local
//! `tch-rs`, …).

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{Result, Runnable, RunnableConfig};

use crate::document::Document;

/// Scores `(query, doc)` pairs.
#[async_trait]
pub trait CrossEncoder: Send + Sync {
    /// Score every `doc` against `query`. Higher = more relevant.
    /// Implementations should batch when possible.
    async fn score(&self, query: &str, docs: &[Document]) -> Result<Vec<f32>>;
}

/// Closure-backed cross-encoder. Useful for tests; in production use a
/// real impl that calls a hosted scorer.
pub struct FnCrossEncoder<F>
where
    F: Fn(&str, &Document) -> f32 + Send + Sync,
{
    /// Per-doc scorer; runs concurrently across docs.
    pub f: F,
}

#[async_trait]
impl<F> CrossEncoder for FnCrossEncoder<F>
where
    F: Fn(&str, &Document) -> f32 + Send + Sync,
{
    async fn score(&self, query: &str, docs: &[Document]) -> Result<Vec<f32>> {
        Ok(docs.iter().map(|d| (self.f)(query, d)).collect())
    }
}

/// Wraps an inner retriever, then reranks its hits via a [`CrossEncoder`].
pub struct CrossEncoderReranker {
    inner: Arc<dyn Runnable<String, Vec<Document>>>,
    encoder: Arc<dyn CrossEncoder>,
    top_k: usize,
}

impl CrossEncoderReranker {
    /// Build with an inner retriever + cross-encoder + post-rerank top-k.
    pub fn new(
        inner: Arc<dyn Runnable<String, Vec<Document>>>,
        encoder: Arc<dyn CrossEncoder>,
        top_k: usize,
    ) -> Self {
        Self {
            inner,
            encoder,
            top_k,
        }
    }
}

#[async_trait]
impl Runnable<String, Vec<Document>> for CrossEncoderReranker {
    async fn invoke(&self, query: String, config: RunnableConfig) -> Result<Vec<Document>> {
        let docs = self.inner.invoke(query.clone(), config).await?;
        if docs.is_empty() {
            return Ok(docs);
        }
        let scores = self.encoder.score(&query, &docs).await?;
        let mut paired: Vec<(f32, Document)> = scores.into_iter().zip(docs).collect();
        paired.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        Ok(paired
            .into_iter()
            .take(self.top_k)
            .map(|(_, d)| d)
            .collect())
    }
    fn name(&self) -> &str {
        "CrossEncoderReranker"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticInner(Vec<Document>);
    #[async_trait]
    impl Runnable<String, Vec<Document>> for StaticInner {
        async fn invoke(&self, _: String, _: RunnableConfig) -> Result<Vec<Document>> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn reranks_by_score() {
        let inner: Arc<dyn Runnable<String, Vec<Document>>> = Arc::new(StaticInner(vec![
            Document::new("apple pie").with_id("a"),
            Document::new("rust crab").with_id("b"),
            Document::new("rust ferris").with_id("c"),
        ]));
        // Score by count of "rust" in content.
        let enc: Arc<dyn CrossEncoder> = Arc::new(FnCrossEncoder {
            f: |_q: &str, d: &Document| d.content.matches("rust").count() as f32,
        });
        let r = CrossEncoderReranker::new(inner, enc, 2);
        let out = r
            .invoke("rust".into(), RunnableConfig::default())
            .await
            .unwrap();
        let ids: Vec<_> = out.iter().filter_map(|d| d.id.clone()).collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&"b".to_string()) || ids.contains(&"c".to_string()));
        assert!(!ids.contains(&"a".to_string()));
    }
}
