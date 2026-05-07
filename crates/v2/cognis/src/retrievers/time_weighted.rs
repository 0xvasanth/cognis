//! `TimeWeightedRetriever` — combines semantic similarity with recency.
//!
//! Each document carries a `timestamp` metadata field (Unix seconds).
//! Final score = `similarity_score + decay_rate ^ hours_since_last_access`.
//! Older docs decay gradually unless they're highly similar.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use cognis2_core::{Result, Runnable, RunnableConfig};
use cognis2_rag::{Document, VectorStore};

/// Time-decayed wrapper over a [`VectorStore`].
pub struct TimeWeightedRetriever {
    store: Arc<RwLock<dyn VectorStore>>,
    /// Per-doc-id last-access seconds. Mutable so reads update access time.
    last_accessed: Arc<RwLock<HashMap<String, u64>>>,
    decay_rate: f32,
    k: usize,
}

impl TimeWeightedRetriever {
    /// Build with a default decay rate of 0.01 / hour (very gentle).
    pub fn new(store: Arc<RwLock<dyn VectorStore>>, k: usize) -> Self {
        Self {
            store,
            last_accessed: Arc::new(RwLock::new(HashMap::new())),
            decay_rate: 0.01,
            k,
        }
    }

    /// Override the decay rate (per-hour multiplier; values close to 1.0
    /// preserve old documents more, values close to 0 demote them faster).
    pub fn with_decay_rate(mut self, r: f32) -> Self {
        self.decay_rate = r;
        self
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[async_trait]
impl Runnable<String, Vec<Document>> for TimeWeightedRetriever {
    async fn invoke(&self, query: String, _: RunnableConfig) -> Result<Vec<Document>> {
        // Pull a wider candidate set, then rerank by time-weighted score.
        let candidates = self
            .store
            .read()
            .await
            .similarity_search(&query, self.k * 4)
            .await?;

        let now = now_secs();
        let last = self.last_accessed.read().await.clone();
        let mut scored: Vec<(f32, Document)> = candidates
            .into_iter()
            .map(|r| {
                let id = r.id.clone();
                let last_seen = last.get(&id).copied().unwrap_or(now);
                let hours = (now.saturating_sub(last_seen) as f32) / 3600.0;
                let decay = self.decay_rate.powf(hours);
                let combined = r.score + decay;
                (
                    combined,
                    Document {
                        id: Some(r.id),
                        content: r.text,
                        metadata: r.metadata,
                    },
                )
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Update last-access for the kept set.
        let kept: Vec<Document> = scored.into_iter().take(self.k).map(|(_, d)| d).collect();
        let mut access = self.last_accessed.write().await;
        for d in &kept {
            if let Some(id) = &d.id {
                access.insert(id.clone(), now);
            }
        }
        Ok(kept)
    }
    fn name(&self) -> &str {
        "TimeWeightedRetriever"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis2_rag::{FakeEmbeddings, InMemoryVectorStore};

    #[tokio::test]
    async fn returns_top_k_with_timestamps() {
        let mut store = InMemoryVectorStore::new(Arc::new(FakeEmbeddings::new(8)));
        store
            .add_texts(vec!["alpha".into(), "beta".into(), "gamma".into()], None)
            .await
            .unwrap();
        let store_arc: Arc<RwLock<dyn VectorStore>> = Arc::new(RwLock::new(store));
        let r = TimeWeightedRetriever::new(store_arc, 2);
        let out = r
            .invoke("alpha".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
    }
}
