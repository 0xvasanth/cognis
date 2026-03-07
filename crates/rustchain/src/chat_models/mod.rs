//! Chat model factory and provider registry.
//!
//! Provides `init_chat_model` for creating chat models by provider name,
//! and utilities for parsing model strings.

#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "azure")]
pub mod azure;
pub mod base;
pub mod cached;
pub mod circuit_breaker;
#[cfg(feature = "google")]
pub mod google;
pub mod graceful;
pub mod interceptor;
#[cfg(feature = "ollama")]
pub mod ollama;
#[cfg(feature = "openai")]
pub mod openai;
pub mod rate_limited;
pub mod retrying;
pub mod structured;
pub mod token_counting;

pub use base::*;
pub use structured::*;
