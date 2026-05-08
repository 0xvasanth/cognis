//! Ensemble retriever — combines multiple retrievers via Reciprocal Rank
//! Fusion.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use futures::future::join_all;

use cognis_core::{Result, Runnable, RunnableConfig};

use crate::document::Document;

/// One retriever inside an [`EnsembleRetriever`] paired with its weight.
type WeightedRetriever = (Arc<dyn Runnable<String, Vec<Document>>>, f32);

/// Combines results from N retrievers using Reciprocal Rank Fusion (RRF).
///
/// For each retriever's ranked list, each document earns a score of
/// `weight / (rrf_k + rank)`. Scores across retrievers sum; final results
/// are sorted descending and truncated to `top_k`.
///
/// Common pattern: combine a [`super::VectorRetriever`] (semantic) with a
/// [`super::BM25Retriever`] (lexical) to capture both meaning and exact-term
/// matches.
pub struct EnsembleRetriever {
    retrievers: Vec<WeightedRetriever>,
    top_k: usize,
    rrf_k: f32,
}

impl EnsembleRetriever {
    /// Build an empty ensemble.
    pub fn new() -> Self {
        Self {
            retrievers: Vec::new(),
            top_k: 4,
            rrf_k: 60.0,
        }
    }

    /// Add a retriever with a contribution weight (default 1.0).
    pub fn with_retriever(
        mut self,
        retriever: Arc<dyn Runnable<String, Vec<Document>>>,
        weight: f32,
    ) -> Self {
        self.retrievers.push((retriever, weight));
        self
    }

    /// Override the final top-k.
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Override the RRF `k` constant (default 60.0 — RRF paper).
    pub fn with_rrf_k(mut self, k: f32) -> Self {
        self.rrf_k = k;
        self
    }
}

impl Default for EnsembleRetriever {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Runnable<String, Vec<Document>> for EnsembleRetriever {
    async fn invoke(&self, query: String, config: RunnableConfig) -> Result<Vec<Document>> {
        let calls = self.retrievers.iter().map(|(r, w)| {
            let r = r.clone();
            let q = query.clone();
            let cfg = config.clone();
            let weight = *w;
            async move {
                let docs = r.invoke(q, cfg).await?;
                Ok::<(Vec<Document>, f32), cognis_core::CognisError>((docs, weight))
            }
        });
        let lists = join_all(calls)
            .await
            .into_iter()
            .collect::<Result<Vec<_>>>()?;

        // Fuse via RRF.
        let mut scores: HashMap<String, (f32, Document)> = HashMap::new();
        for (docs, weight) in lists {
            for (rank, doc) in docs.into_iter().enumerate() {
                let key = doc.id.clone().unwrap_or_else(|| doc.content.clone());
                let contribution = weight / (self.rrf_k + rank as f32 + 1.0);
                scores
                    .entry(key)
                    .and_modify(|(s, _)| *s += contribution)
                    .or_insert((contribution, doc));
            }
        }

        let mut all: Vec<(String, f32, Document)> =
            scores.into_iter().map(|(k, (s, d))| (k, s, d)).collect();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(all
            .into_iter()
            .take(self.top_k)
            .map(|(_, _, d)| d)
            .collect())
    }

    fn name(&self) -> &str {
        "EnsembleRetriever"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticRetriever(Vec<Document>);

    #[async_trait]
    impl Runnable<String, Vec<Document>> for StaticRetriever {
        async fn invoke(&self, _q: String, _: RunnableConfig) -> Result<Vec<Document>> {
            Ok(self.0.clone())
        }
    }

    #[tokio::test]
    async fn fuses_two_retrievers_by_rrf() {
        let r1: Arc<dyn Runnable<String, Vec<Document>>> = Arc::new(StaticRetriever(vec![
            Document::new("a").with_id("a"),
            Document::new("b").with_id("b"),
            Document::new("c").with_id("c"),
        ]));
        let r2: Arc<dyn Runnable<String, Vec<Document>>> = Arc::new(StaticRetriever(vec![
            Document::new("c").with_id("c"),
            Document::new("a").with_id("a"),
            Document::new("d").with_id("d"),
        ]));

        let ens = EnsembleRetriever::new()
            .with_retriever(r1, 1.0)
            .with_retriever(r2, 1.0)
            .with_top_k(3);
        let out = ens
            .invoke("query".into(), RunnableConfig::default())
            .await
            .unwrap();
        let ids: Vec<_> = out.iter().filter_map(|d| d.id.clone()).collect();
        // `a` appears in both lists at high ranks → should rank first.
        assert_eq!(ids[0], "a");
        // `c` also appears in both → should rank well.
        assert!(ids.contains(&"c".to_string()));
    }

    #[tokio::test]
    async fn empty_ensemble_returns_empty() {
        let ens = EnsembleRetriever::new();
        let out = ens
            .invoke("q".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert!(out.is_empty());
    }
}
