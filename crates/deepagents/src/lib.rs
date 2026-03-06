//! # deepagents
//!
//! Batteries-included, high-level agent framework built on `rustchain` and `langgraph`.
//! Provides zero-boilerplate agent creation with pluggable middleware and backends.
//!
//! ## Overview
//!
//! The primary entry point is [`create_deep_agent`], which constructs a compiled
//! LangGraph [`CompiledStateGraph`](langgraph::graph::state::CompiledStateGraph) with
//! middleware hooks around model and tool invocations.
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use deepagents::config::DeepAgentConfig;
//! use deepagents::create_deep_agent;
//!
//! let config = DeepAgentConfig::default();
//! let graph = create_deep_agent(config).unwrap();
//! let result = graph.invoke(serde_json::json!({"messages": []})).await.unwrap();
//! ```
//!
//! ## Middleware
//!
//! The [`middleware::Middleware`] trait provides before/after hooks for model calls
//! and tool executions. Built-in middleware:
//!
//! - [`middleware::filesystem::FilesystemMiddleware`] -- file read, write, list, glob, grep
//! - [`middleware::memory::MemoryMiddleware`] -- inject persistent memory into context
//!
//! ## Backends
//!
//! The [`backends::Backend`] trait abstracts session state persistence:
//!
//! - [`backends::StateBackend`] -- in-memory (default)
//! - [`backends::FilesystemBackend`] -- local disk storage

pub mod agent;
pub mod backends;
pub mod config;
pub mod middleware;

pub use agent::create_deep_agent;
