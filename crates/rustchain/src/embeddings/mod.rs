//! Embeddings factory and provider registry.
//!
//! Provides `init_embeddings` for creating embedding models by provider name,
//! and utilities for parsing model strings.

pub mod base;
pub mod cached;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "google")]
pub mod google;
#[cfg(feature = "anthropic")]
pub mod anthropic;

pub use base::*;
