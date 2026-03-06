//! Chat model factory and provider registry.
//!
//! Provides `init_chat_model` for creating chat models by provider name,
//! and utilities for parsing model strings.

pub mod base;
pub mod structured;
#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "openai")]
pub mod openai;
#[cfg(feature = "google")]
pub mod google;
#[cfg(feature = "ollama")]
pub mod ollama;

pub use base::*;
pub use structured::*;
