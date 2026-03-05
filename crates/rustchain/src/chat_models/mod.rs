//! Chat model factory and provider registry.
//!
//! Provides `init_chat_model` for creating chat models by provider name,
//! and utilities for parsing model strings.

pub mod base;
#[cfg(feature = "anthropic")]
pub mod anthropic;
#[cfg(feature = "openai")]
pub mod openai;

pub use base::*;
