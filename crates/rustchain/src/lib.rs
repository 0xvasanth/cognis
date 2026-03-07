//! # rustchain
//!
//! Implementation layer for the RustChain LLM framework. This crate provides
//! concrete chat model integrations, agent execution, chains, memory strategies,
//! document loaders, text splitters, embedding providers, and built-in tools.
//!
//! ## Chat Model Providers
//!
//! Each provider is gated behind a feature flag:
//!
//! | Feature | Provider |
//! |---------|----------|
//! | `anthropic` | Anthropic Claude |
//! | `openai` | OpenAI GPT |
//! | `google` | Google Gemini |
//! | `ollama` | Ollama (local) |
//! | `azure` | Azure OpenAI |
//! | `all-providers` | All of the above |
//!
//! ## Quick Example
//!
//! ```rust,ignore
//! use rustchain::chat_models::anthropic::ChatAnthropic;
//! use rustchain_core::runnables::Runnable;
//! use serde_json::json;
//!
//! let model = ChatAnthropic::new("claude-sonnet-4-20250514");
//! let result = model.invoke(json!({"messages": []}), None).await.unwrap();
//! ```
//!
//! ## Modules
//!
//! - [`chat_models`] -- Chat model implementations for each provider.
//! - [`embeddings`] -- OpenAI and Ollama embedding providers.
//! - [`agents`] -- Agent executor with a pluggable middleware pipeline.
//! - [`chains`] -- LLM chain, conversation chain, and sequential chain.
//! - [`memory`] -- Buffer, window, and summary memory strategies.
//! - [`document_loaders`] -- Text, CSV, JSON, and directory document loaders.
//! - [`text_splitter`] -- Character, recursive, markdown, HTML, JSON, code, and token splitters.
//! - [`tools`] -- Calculator, shell, and JSON query tools.

pub mod agents;
pub mod cache;
pub mod chains;
pub mod chat_models;
pub mod document_loaders;
pub mod document_transformers;
pub mod embeddings;
pub mod evaluation;
pub mod indexing;
pub mod memory;
pub mod output_parsers;
pub mod prompts;
pub mod retrievers;
pub mod text_splitter;
pub mod stores;
pub mod tools;
pub mod vectorstores;

// Re-export core for convenience
pub use rustchain_core as core;
