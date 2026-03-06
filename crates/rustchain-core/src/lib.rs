//! # rustchain-core
//!
//! Foundation layer for the RustChain LLM framework. This crate defines the base
//! traits, types, and abstractions that all other crates in the workspace depend on.
//! It has **zero** dependencies on other workspace crates.
//!
//! ## Key Modules
//!
//! - [`messages`] -- `Message` enum (Human, AI, System, Tool) and utilities for
//!   conversion, filtering, trimming, and merging message sequences.
//! - [`language_models`] -- `BaseChatModel` and `BaseLLM` traits that chat model
//!   providers implement, plus fake models for testing.
//! - [`runnables`] -- The composable `Runnable` trait and combinators: sequence,
//!   parallel, branch, lambda, retry, fallbacks, passthrough.
//! - [`tools`] -- `BaseTool` trait for agent tool calling with JSON schema support.
//! - [`prompts`] -- Chat prompt templates, few-shot example selectors, and
//!   structured prompt builders.
//! - [`output_parsers`] -- Parsers for JSON, string, list, XML, and tool-call outputs.
//! - [`callbacks`] -- Extensible callback system with handler traits and run managers.
//! - [`vectorstores`] -- `VectorStore` trait and in-memory implementation.
//! - [`embeddings`] -- `Embeddings` trait for vector embedding providers.
//! - [`documents`] -- `Document` type used across loaders, splitters, and retrievers.
//!
//! ## Quick Example
//!
//! ```rust
//! use rustchain_core::messages::Message;
//!
//! let msg = Message::human("What is the capital of France?");
//! assert_eq!(msg.message_type(), rustchain_core::messages::MessageType::Human);
//! ```

pub mod error;
pub mod utils;
pub mod messages;
pub mod documents;
pub mod document_loaders;
pub mod outputs;
pub mod callbacks;
pub mod embeddings;
pub mod embeddings_fake;
pub mod prompt_values;
pub mod stores;
pub mod caches;
pub mod retrievers;
pub mod tools;
pub mod language_models;
pub mod agents;
pub mod chat_history;
pub mod chat_loaders;
pub mod chat_sessions;
pub mod runnables;
pub mod prompts;
pub mod output_parsers;
pub mod vectorstores;
pub mod rate_limiters;
pub mod structured_query;
pub mod indexing;
pub mod tracers;
pub mod load;
pub mod globals;
