//! Higher-level prompt management for RustChain.
//!
//! This module provides [`PromptHub`] for versioned template storage and
//! [`PromptTemplate`] as a convenience wrapper around core prompt formatting
//! with support for partial variables and `Runnable` composition.
//!
//! ## Submodules
//!
//! - [`chat`] -- Chat prompt templates with message-level formatting, variable extraction,
//!   and `messages_placeholder` support for dynamic message insertion.
//! - [`few_shot`] -- Few-shot prompt templates with length-based and semantic similarity
//!   example selectors.
//! - [`hub`] -- Prompt hub for storing and retrieving versioned prompt entries.
//! - [`registry`] -- Prompt registry with composition and export utilities.
//! - [`template`] -- Base prompt template with variable substitution.

pub mod chat;
pub mod few_shot;
pub mod hub;
pub mod registry;
pub mod template;
pub mod template_engine;

pub use chat::{
    extract_template_variables, messages_placeholder, ChatPromptTemplate, MessagePromptTemplate,
};
pub use few_shot::{
    Example, ExampleSelector, FewShotPromptTemplate, FewShotPromptTemplateBuilder,
    LengthBasedSelector, LengthBasedSelectorBuilder, SemanticSimilaritySelector,
    SemanticSimilaritySelectorBuilder,
};
pub use hub::{PromptEntry, PromptHub};
pub use registry::{PromptComposer, PromptExporter, PromptRegistry, RegisteredPrompt};
pub use template::PromptTemplate;
pub use template_engine::{
    AdvancedPromptTemplate, TemplateBlock, TemplateContext, TemplateEngine, TemplateVariable,
};
