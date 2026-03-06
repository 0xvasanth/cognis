//! Embeddings factory and provider registry.
//!
//! Provides `init_embeddings` for creating embedding models by provider name,
//! and utilities for parsing model strings.

#[cfg(feature = "anthropic")]
pub mod anthropic;
pub mod base;
pub mod cached;
#[cfg(feature = "google")]
pub mod google;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "openai")]
pub mod openai;

pub use base::*;
