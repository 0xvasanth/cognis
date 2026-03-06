//! Chat model factory and provider registry.
//!
//! Provides `init_chat_model` for creating chat models by provider name,
//! and utilities for parsing model strings.

pub mod base;
pub mod cached;
pub mod circuit_breaker;
pub mod graceful;
pub mod rate_limited;
pub mod retrying;
pub mod structured;
pub mod token_counting;
#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "google")]
pub mod google;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "azure")]
pub mod azure;

pub use base::*;
pub use structured::*;
