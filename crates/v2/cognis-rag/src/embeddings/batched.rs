//! Batched embeddings — automatic micro-batching for embedding APIs
//! that have a per-request size cap (or that benefit from larger
//! batches up to some limit).
//!
//! Wraps any [`Embeddings`] and chunks an `embed_documents` call into
//! multiple inner calls of at most `max_batch_size` items each. Calls
//! are issued concurrently up to `max_concurrency`.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, StreamExt};

use cognis2_core::Result;

use super::Embeddings;

/// Auto-batching wrapper.
pub struct BatchedEmbeddings {
    inner: Arc<dyn Embeddings>,
    max_batch_size: usize,
    max_concurrency: usize,
}

impl BatchedEmbeddings {
    /// Wrap with the given per-call batch cap. Concurrency defaults to 4.
    pub fn new(inner: Arc<dyn Embeddings>, max_batch_size: usize) -> Self {
        Self {
            inner,
            max_batch_size: max_batch_size.max(1),
            max_concurrency: 4,
        }
    }

    /// Override the maximum number of concurrent inner calls.
    pub fn with_max_concurrency(mut self, n: usize) -> Self {
        self.max_concurrency = n.max(1);
        self
    }

    /// Current configuration: `(max_batch_size, max_concurrency)`.
    pub fn config(&self) -> (usize, usize) {
        (self.max_batch_size, self.max_concurrency)
    }
}

#[async_trait]
impl Embeddings for BatchedEmbeddings {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        if texts.len() <= self.max_batch_size {
            return self.inner.embed_documents(texts).await;
        }
        // Tag every input with its index, chunk, dispatch concurrently,
        // then re-assemble in original order.
        let chunks: Vec<(Vec<usize>, Vec<String>)> = texts
            .into_iter()
            .enumerate()
            .collect::<Vec<(usize, String)>>()
            .chunks(self.max_batch_size)
            .map(|c| {
                let (idxs, ts): (Vec<usize>, Vec<String>) = c.iter().cloned().unzip();
                (idxs, ts)
            })
            .collect();

        let results: Vec<Result<(Vec<usize>, Vec<Vec<f32>>)>> = stream::iter(chunks)
            .map(|(idxs, ts)| {
                let inner = self.inner.clone();
                async move {
                    let v = inner.embed_documents(ts).await?;
                    Ok((idxs, v))
                }
            })
            .buffer_unordered(self.max_concurrency)
            .collect()
            .await;

        // Reassemble in original input order.
        let mut total = 0usize;
        let collected: Vec<(Vec<usize>, Vec<Vec<f32>>)> = results
            .into_iter()
            .map(|r| {
                r.map(|(i, v)| {
                    total += i.len();
                    (i, v)
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut out: Vec<Option<Vec<f32>>> = vec![None; total];
        for (idxs, vecs) in collected {
            for (i, v) in idxs.into_iter().zip(vecs) {
                out[i] = Some(v);
            }
        }
        Ok(out.into_iter().map(|o| o.unwrap_or_default()).collect())
    }

    async fn embed_query(&self, text: String) -> Result<Vec<f32>> {
        // Single-query path doesn't need batching.
        self.inner.embed_query(text).await
    }

    fn dimensions(&self) -> Option<usize> {
        self.inner.dimensions()
    }

    fn model(&self) -> &str {
        self.inner.model()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::FakeEmbeddings;

    /// Records every batch call's size for assertions.
    struct Recording {
        inner: Arc<dyn Embeddings>,
        sizes: tokio::sync::Mutex<Vec<usize>>,
    }
    #[async_trait]
    impl Embeddings for Recording {
        async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
            self.sizes.lock().await.push(texts.len());
            self.inner.embed_documents(texts).await
        }
        fn model(&self) -> &str {
            "recording"
        }
    }

    fn recorded(dim: usize) -> Arc<Recording> {
        Arc::new(Recording {
            inner: Arc::new(FakeEmbeddings::new(dim)),
            sizes: tokio::sync::Mutex::new(Vec::new()),
        })
    }

    #[tokio::test]
    async fn small_batch_passes_through_unchunked() {
        let inner = recorded(4);
        let bw = BatchedEmbeddings::new(inner.clone() as Arc<dyn Embeddings>, 10);
        let texts: Vec<String> = (0..3).map(|i| format!("t{i}")).collect();
        let _ = bw.embed_documents(texts).await.unwrap();
        let sizes = inner.sizes.lock().await.clone();
        assert_eq!(sizes, vec![3]);
    }

    #[tokio::test]
    async fn large_batch_is_chunked() {
        let inner = recorded(4);
        let bw = BatchedEmbeddings::new(inner.clone() as Arc<dyn Embeddings>, 4);
        let texts: Vec<String> = (0..10).map(|i| format!("t{i}")).collect();
        let out = bw.embed_documents(texts).await.unwrap();
        assert_eq!(out.len(), 10);
        let sizes = inner.sizes.lock().await.clone();
        // 10 / 4 = 3 batches: 4 + 4 + 2.
        assert_eq!(sizes.iter().sum::<usize>(), 10);
        assert!(sizes.iter().all(|&s| s <= 4));
    }

    #[tokio::test]
    async fn output_order_preserved_across_chunking() {
        // FakeEmbeddings is text-deterministic, so each input has a
        // unique vector. We verify ordering by spot-checking that
        // texts[0]'s vector equals embed_query(texts[0]).
        let inner: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(8));
        let bw = BatchedEmbeddings::new(inner.clone(), 3);
        let texts: Vec<String> = (0..7).map(|i| format!("t{i}")).collect();
        let batched = bw.embed_documents(texts.clone()).await.unwrap();
        for (i, t) in texts.iter().enumerate() {
            let single = inner.embed_query(t.clone()).await.unwrap();
            assert_eq!(batched[i], single, "mismatch at index {i}");
        }
    }
}
