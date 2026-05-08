//! Embeddings router — dispatch each call to one of N inner providers
//! based on a pluggable strategy.
//!
//! Use cases:
//! - Long-text → high-context embedder; short-text → cheaper embedder.
//! - Per-tenant routing for billing isolation.
//! - Failover during a provider outage (paired with [`super::Embeddings`]
//!   wrappers like CachedEmbeddings).

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{CognisError, Result};

use super::Embeddings;

/// Pluggable routing decision: an index into the configured provider list.
pub trait EmbeddingRouter: Send + Sync {
    /// Pick a provider for this batch (`embed_documents`) call.
    fn pick_documents(&self, texts: &[String]) -> usize;
    /// Pick a provider for this single-query call.
    fn pick_query(&self, text: &str) -> usize;
}

/// Closure-based router. Both methods use the same closure.
pub struct FnRouter<F> {
    f: F,
}

impl<F> FnRouter<F>
where
    F: Fn(&[String]) -> usize + Send + Sync,
{
    /// Build with a single closure that picks based on a slice of inputs.
    /// `pick_query` wraps the query text in a single-element slice
    /// before delegating.
    pub fn new(f: F) -> Self {
        Self { f }
    }
}

impl<F> EmbeddingRouter for FnRouter<F>
where
    F: Fn(&[String]) -> usize + Send + Sync,
{
    fn pick_documents(&self, texts: &[String]) -> usize {
        (self.f)(texts)
    }
    fn pick_query(&self, text: &str) -> usize {
        let v = vec![text.to_string()];
        (self.f)(&v)
    }
}

/// Stock router: route by length. Inputs whose total chars across the
/// batch exceed `threshold` go to `long_idx`; otherwise `short_idx`.
pub struct LengthRouter {
    /// Total-chars threshold for "long" routing.
    pub threshold: usize,
    /// Provider index used for "short" inputs.
    pub short_idx: usize,
    /// Provider index used for "long" inputs.
    pub long_idx: usize,
}

impl EmbeddingRouter for LengthRouter {
    fn pick_documents(&self, texts: &[String]) -> usize {
        let total: usize = texts.iter().map(|t| t.chars().count()).sum();
        if total > self.threshold {
            self.long_idx
        } else {
            self.short_idx
        }
    }
    fn pick_query(&self, text: &str) -> usize {
        if text.chars().count() > self.threshold {
            self.long_idx
        } else {
            self.short_idx
        }
    }
}

/// Embedding-provider router.
pub struct EmbeddingsRouter {
    providers: Vec<Arc<dyn Embeddings>>,
    router: Arc<dyn EmbeddingRouter>,
    /// Reported model id (used as the union name).
    name: String,
}

impl EmbeddingsRouter {
    /// Build with a list of providers and a routing strategy.
    pub fn new<R: EmbeddingRouter + 'static>(
        name: impl Into<String>,
        providers: Vec<Arc<dyn Embeddings>>,
        router: R,
    ) -> Result<Self> {
        if providers.is_empty() {
            return Err(CognisError::Configuration(
                "EmbeddingsRouter requires at least one provider".into(),
            ));
        }
        Ok(Self {
            providers,
            router: Arc::new(router),
            name: name.into(),
        })
    }

    /// Borrow the configured providers (read-only).
    pub fn providers(&self) -> &[Arc<dyn Embeddings>] {
        &self.providers
    }

    fn pick_documents(&self, texts: &[String]) -> &Arc<dyn Embeddings> {
        let mut idx = self.router.pick_documents(texts);
        if idx >= self.providers.len() {
            idx = 0;
        }
        &self.providers[idx]
    }

    fn pick_query(&self, text: &str) -> &Arc<dyn Embeddings> {
        let mut idx = self.router.pick_query(text);
        if idx >= self.providers.len() {
            idx = 0;
        }
        &self.providers[idx]
    }
}

#[async_trait]
impl Embeddings for EmbeddingsRouter {
    async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let p = self.pick_documents(&texts).clone();
        p.embed_documents(texts).await
    }
    async fn embed_query(&self, text: String) -> Result<Vec<f32>> {
        let p = self.pick_query(&text).clone();
        p.embed_query(text).await
    }
    fn dimensions(&self) -> Option<usize> {
        // Unknown — providers may have different dims.
        None
    }
    fn model(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::FakeEmbeddings;

    /// Tag-bearing fake embedder so we can identify which provider ran.
    struct Tagged {
        tag: &'static str,
        inner: Arc<dyn Embeddings>,
        seen: std::sync::atomic::AtomicUsize,
    }
    #[async_trait]
    impl Embeddings for Tagged {
        async fn embed_documents(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
            self.seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            self.inner.embed_documents(texts).await
        }
        fn model(&self) -> &str {
            self.tag
        }
    }
    fn tagged(tag: &'static str) -> Arc<Tagged> {
        Arc::new(Tagged {
            tag,
            inner: Arc::new(FakeEmbeddings::new(4)),
            seen: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    #[tokio::test]
    async fn length_router_dispatches_short_and_long() {
        let short = tagged("short");
        let long = tagged("long");
        let r = EmbeddingsRouter::new(
            "router",
            vec![
                short.clone() as Arc<dyn Embeddings>,
                long.clone() as Arc<dyn Embeddings>,
            ],
            LengthRouter {
                threshold: 50,
                short_idx: 0,
                long_idx: 1,
            },
        )
        .unwrap();
        let _ = r.embed_documents(vec!["abc".into()]).await.unwrap();
        let _ = r.embed_documents(vec!["x".repeat(100)]).await.unwrap();
        assert_eq!(short.seen.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(long.seen.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn closure_router_works() {
        let a = tagged("a");
        let b = tagged("b");
        let r = EmbeddingsRouter::new(
            "router",
            vec![
                a.clone() as Arc<dyn Embeddings>,
                b.clone() as Arc<dyn Embeddings>,
            ],
            FnRouter::new(|texts: &[String]| {
                if texts.iter().any(|t| t.starts_with('!')) {
                    1
                } else {
                    0
                }
            }),
        )
        .unwrap();
        let _ = r.embed_documents(vec!["plain".into()]).await.unwrap();
        let _ = r.embed_documents(vec!["!special".into()]).await.unwrap();
        assert_eq!(a.seen.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(b.seen.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn out_of_range_index_clamps_to_zero() {
        let a = tagged("a");
        let r = EmbeddingsRouter::new(
            "router",
            vec![a.clone() as Arc<dyn Embeddings>],
            FnRouter::new(|_| 99usize),
        )
        .unwrap();
        let _ = r.embed_documents(vec!["x".into()]).await.unwrap();
        assert_eq!(a.seen.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn empty_providers_errors() {
        let r = EmbeddingsRouter::new(
            "router",
            Vec::<Arc<dyn Embeddings>>::new(),
            LengthRouter {
                threshold: 0,
                short_idx: 0,
                long_idx: 0,
            },
        );
        assert!(r.is_err());
    }
}
