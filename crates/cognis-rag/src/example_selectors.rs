//! Embedding-driven example selectors for few-shot prompts.
//!
//! Implements [`cognis_core::prompts::ExampleSelector`] for example
//! pools where similarity to the input determines which examples to
//! include.
//!
//! Two strategies are provided:
//! - [`SemanticSimilarityExampleSelector`] — top-k by similarity to the
//!   input. Cheap and effective when examples are diverse.
//! - [`MmrExampleSelector`] — Maximal Marginal Relevance: balances
//!   relevance to the input with novelty among already-picked examples.
//!   Useful when the pool contains near-duplicates.
//!
//! Both delegate the actual example→string conversion to a user-supplied
//! closure, so the selector works for any `E` type the user can describe.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::prompts::ExampleSelector;
use cognis_core::{CognisError, Result};

use crate::distance::Distance;
use crate::embeddings::Embeddings;

/// Function that turns an example into the text we embed when selecting.
/// Often the same renderer used to inject the example into the prompt.
pub type ExampleTextFn<E> = Arc<dyn Fn(&E) -> String + Send + Sync>;

/// Cache mode for embedded examples. The pool's embeddings are the same
/// across calls, so most users want `Cached` (the default).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedMode {
    /// Re-embed the pool on every `select` call. Slower but always
    /// reflects a possibly-mutated pool.
    Fresh,
    /// Embed the pool once on first call and reuse. Use when the pool is
    /// stable for the lifetime of the selector.
    Cached,
}

// ---------------------------------------------------------------------------
// SemanticSimilarityExampleSelector
// ---------------------------------------------------------------------------

/// Pick the top-`k` examples whose embeddings are most similar to the
/// input.
pub struct SemanticSimilarityExampleSelector<E> {
    embeddings: Arc<dyn Embeddings>,
    k: usize,
    distance: Distance,
    text_of: ExampleTextFn<E>,
    mode: EmbedMode,
    // Cached pool embeddings + original index. Wrapped in tokio::Mutex
    // because we may need to embed inside the async `select`.
    cache: Arc<tokio::sync::Mutex<Option<Vec<Vec<f32>>>>>,
}

impl<E> SemanticSimilarityExampleSelector<E>
where
    E: Send + Sync + 'static,
{
    /// Build a selector that picks the top-`k` examples by similarity.
    pub fn new<F>(embeddings: Arc<dyn Embeddings>, k: usize, text_of: F) -> Self
    where
        F: Fn(&E) -> String + Send + Sync + 'static,
    {
        Self {
            embeddings,
            k,
            distance: Distance::Cosine,
            text_of: Arc::new(text_of),
            mode: EmbedMode::Cached,
            cache: Arc::new(tokio::sync::Mutex::new(None)),
        }
    }

    /// Override the distance metric (default: Cosine).
    pub fn with_distance(mut self, d: Distance) -> Self {
        self.distance = d;
        self
    }

    /// Override the embed mode (default: Cached).
    pub fn with_embed_mode(mut self, m: EmbedMode) -> Self {
        self.mode = m;
        // Reset the cache when switching modes.
        self.cache = Arc::new(tokio::sync::Mutex::new(None));
        self
    }

    /// Embed the pool, optionally reading or populating the cache.
    async fn embed_pool(&self, examples: &[E]) -> Result<Vec<Vec<f32>>> {
        if matches!(self.mode, EmbedMode::Cached) {
            let mut guard = self.cache.lock().await;
            if let Some(cached) = guard.as_ref() {
                if cached.len() == examples.len() {
                    return Ok(cached.clone());
                }
            }
            let texts: Vec<String> = examples.iter().map(|e| (self.text_of)(e)).collect();
            let vecs = self.embeddings.embed_documents(texts).await?;
            *guard = Some(vecs.clone());
            return Ok(vecs);
        }
        let texts: Vec<String> = examples.iter().map(|e| (self.text_of)(e)).collect();
        self.embeddings.embed_documents(texts).await
    }

    /// Async select. Trait method must be sync; this is the underlying
    /// implementation.
    async fn select_async(&self, input: &str, examples: &[E]) -> Result<Vec<E>>
    where
        E: Clone,
    {
        if examples.is_empty() {
            return Ok(Vec::new());
        }
        let q = self.embeddings.embed_query(input.to_string()).await?;
        let pool_vecs = self.embed_pool(examples).await?;
        let mut scored: Vec<(usize, f32)> = pool_vecs
            .iter()
            .enumerate()
            .map(|(i, v)| (i, self.distance.similarity(&q, v)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scored
            .into_iter()
            .take(self.k.min(examples.len()))
            .map(|(i, _)| examples[i].clone())
            .collect())
    }
}

#[async_trait]
impl<E> AsyncExampleSelector<E> for SemanticSimilarityExampleSelector<E>
where
    E: Clone + Send + Sync + 'static,
{
    async fn select_async(&self, input: &str, examples: &[E]) -> Result<Vec<E>> {
        SemanticSimilarityExampleSelector::select_async(self, input, examples).await
    }
}

impl<E> ExampleSelector<E> for SemanticSimilarityExampleSelector<E>
where
    E: Clone + Send + Sync + 'static,
{
    fn select(&self, input: &str, examples: &[E]) -> Result<Vec<E>> {
        // The trait method is sync; we run the async impl on the current
        // tokio runtime. This is the same pattern V1's `LengthBasedExampleSelector`
        // uses to bridge async embeddings into a sync trait.
        let handle = tokio::runtime::Handle::try_current().map_err(|_| {
            CognisError::Configuration(
                "SemanticSimilarityExampleSelector::select called outside a tokio runtime; \
                 use AsyncExampleSelector::select_async for explicit await"
                    .into(),
            )
        })?;
        tokio::task::block_in_place(|| handle.block_on(self.select_async(input, examples)))
    }
}

// ---------------------------------------------------------------------------
// MmrExampleSelector
// ---------------------------------------------------------------------------

/// Maximal Marginal Relevance selector: trades relevance to the input
/// against novelty among already-selected examples. `lambda` controls
/// the trade-off — `1.0` is pure similarity (equivalent to the semantic
/// selector); `0.0` is pure diversity.
pub struct MmrExampleSelector<E> {
    embeddings: Arc<dyn Embeddings>,
    k: usize,
    lambda: f32,
    distance: Distance,
    text_of: ExampleTextFn<E>,
}

impl<E> MmrExampleSelector<E>
where
    E: Send + Sync + 'static,
{
    /// Build with `lambda` clamped to `[0, 1]`. `k` is the number of
    /// examples returned.
    pub fn new<F>(embeddings: Arc<dyn Embeddings>, k: usize, lambda: f32, text_of: F) -> Self
    where
        F: Fn(&E) -> String + Send + Sync + 'static,
    {
        Self {
            embeddings,
            k,
            lambda: lambda.clamp(0.0, 1.0),
            distance: Distance::Cosine,
            text_of: Arc::new(text_of),
        }
    }

    /// Override the distance metric.
    pub fn with_distance(mut self, d: Distance) -> Self {
        self.distance = d;
        self
    }

    async fn select_async(&self, input: &str, examples: &[E]) -> Result<Vec<E>>
    where
        E: Clone,
    {
        if examples.is_empty() {
            return Ok(Vec::new());
        }
        let q = self.embeddings.embed_query(input.to_string()).await?;
        let texts: Vec<String> = examples.iter().map(|e| (self.text_of)(e)).collect();
        let pool_vecs = self.embeddings.embed_documents(texts).await?;
        let n = examples.len();
        let take = self.k.min(n);
        let mut chosen: Vec<usize> = Vec::with_capacity(take);
        let mut available: Vec<usize> = (0..n).collect();

        for _ in 0..take {
            let mut best_idx: Option<usize> = None;
            let mut best_score = f32::NEG_INFINITY;
            for &i in &available {
                let sim_to_query = self.distance.similarity(&q, &pool_vecs[i]);
                let max_sim_to_chosen = chosen
                    .iter()
                    .map(|&j| self.distance.similarity(&pool_vecs[i], &pool_vecs[j]))
                    .fold(f32::NEG_INFINITY, f32::max);
                let novelty = if chosen.is_empty() {
                    0.0
                } else {
                    max_sim_to_chosen
                };
                let score = self.lambda * sim_to_query - (1.0 - self.lambda) * novelty;
                if score > best_score {
                    best_score = score;
                    best_idx = Some(i);
                }
            }
            let pick = match best_idx {
                Some(i) => i,
                None => break,
            };
            chosen.push(pick);
            available.retain(|&i| i != pick);
        }

        Ok(chosen.into_iter().map(|i| examples[i].clone()).collect())
    }
}

#[async_trait]
impl<E> AsyncExampleSelector<E> for MmrExampleSelector<E>
where
    E: Clone + Send + Sync + 'static,
{
    async fn select_async(&self, input: &str, examples: &[E]) -> Result<Vec<E>> {
        MmrExampleSelector::select_async(self, input, examples).await
    }
}

impl<E> ExampleSelector<E> for MmrExampleSelector<E>
where
    E: Clone + Send + Sync + 'static,
{
    fn select(&self, input: &str, examples: &[E]) -> Result<Vec<E>> {
        let handle = tokio::runtime::Handle::try_current().map_err(|_| {
            CognisError::Configuration(
                "MmrExampleSelector::select called outside a tokio runtime; \
                 use AsyncExampleSelector::select_async for explicit await"
                    .into(),
            )
        })?;
        tokio::task::block_in_place(|| handle.block_on(self.select_async(input, examples)))
    }
}

// ---------------------------------------------------------------------------
// AsyncExampleSelector — explicit-async parallel trait.
// ---------------------------------------------------------------------------

/// Async-first variant of [`cognis_core::prompts::ExampleSelector`].
/// Use when you want to call selection from within an async context
/// without going through `block_in_place`.
#[async_trait]
pub trait AsyncExampleSelector<E>: Send + Sync
where
    E: Send + Sync + 'static,
{
    /// Select examples to include for `input`.
    async fn select_async(&self, input: &str, examples: &[E]) -> Result<Vec<E>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::FakeEmbeddings;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn semantic_selector_picks_topk() {
        // FakeEmbeddings is deterministic — the same text → the same vector.
        let embeddings: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(8));
        let sel =
            SemanticSimilarityExampleSelector::new(embeddings.clone(), 2, |s: &String| s.clone());
        let pool: Vec<String> = vec![
            "completely different".into(),
            "rust programming".into(),
            "python programming".into(),
        ];
        let picked = sel.select_async("rust programming", &pool).await.unwrap();
        assert_eq!(picked.len(), 2);
        // The exact match should always be in the top 2.
        assert!(picked.iter().any(|s| s == "rust programming"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn semantic_selector_handles_empty_pool() {
        let embeddings: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(4));
        let sel = SemanticSimilarityExampleSelector::new(embeddings, 3, |s: &String| s.clone());
        let picked = sel.select_async("anything", &[]).await.unwrap();
        assert!(picked.is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn mmr_selector_returns_k_distinct_picks() {
        let embeddings: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(8));
        let sel = MmrExampleSelector::new(embeddings, 2, 0.5, |s: &String| s.clone());
        let pool: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        let picked = sel.select_async("query", &pool).await.unwrap();
        assert_eq!(picked.len(), 2);
        assert_ne!(picked[0], picked[1]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn semantic_selector_caches_pool_embeddings() {
        let embeddings: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(4));
        let sel = SemanticSimilarityExampleSelector::new(embeddings, 1, |s: &String| s.clone())
            .with_embed_mode(EmbedMode::Cached);
        let pool: Vec<String> = vec!["one".into(), "two".into()];
        // First call populates cache.
        let _ = sel.select_async("one", &pool).await.unwrap();
        // Second call must reuse cache (we just check it doesn't error).
        let picked = sel.select_async("one", &pool).await.unwrap();
        assert_eq!(picked.len(), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn semantic_selector_sync_select_works_in_runtime() {
        let embeddings: Arc<dyn Embeddings> = Arc::new(FakeEmbeddings::new(4));
        let sel = SemanticSimilarityExampleSelector::new(embeddings, 2, |s: &String| s.clone());
        let pool: Vec<String> = vec!["a".into(), "b".into(), "c".into()];
        // Call the sync trait method from inside a tokio runtime.
        let picked = ExampleSelector::select(&sel, "a", &pool).unwrap();
        assert_eq!(picked.len(), 2);
    }
}
