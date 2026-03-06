//! Higher-level prompt management for RustChain.
//!
//! This module provides [`PromptHub`] for versioned template storage and
//! [`PromptTemplate`] as a convenience wrapper around core prompt formatting
//! with support for partial variables and `Runnable` composition.

pub mod hub;
pub mod template;

pub use hub::{PromptEntry, PromptHub};
pub use template::PromptTemplate;
