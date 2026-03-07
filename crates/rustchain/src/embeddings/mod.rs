//! Embeddings factory and provider registry.
//!
//! Provides `init_embeddings` for creating embedding models by provider name,
//! and utilities for parsing model strings.
//!
//! ## Submodules
//!
//! - [`openai`] -- OpenAI embedding provider (feature-gated).
//! - [`anthropic`] -- Anthropic embedding provider (feature-gated).
//! - [`google`] -- Google embedding provider (feature-gated).
//! - [`ollama`] -- Ollama embedding provider (feature-gated).
//! - [`cached`] -- Caching wrapper for any embedding provider.
//! - [`distance`] -- Distance metrics (cosine, euclidean, dot product).
//! - [`router`] -- Embedding router for dispatching to different providers.
//! - [`batch`] -- Batch embeddings processor with configurable concurrency, rate limiting,
//!   and chunked processing for large document sets.

#[cfg(feature = "anthropic")]
pub mod anthropic;
pub mod base;
pub mod batch;
pub mod cache;
pub mod cached;
pub mod distance;
#[cfg(feature = "google")]
pub mod google;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "openai")]
pub mod openai;
pub mod router;

pub use base::*;
