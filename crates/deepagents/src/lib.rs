//! Deep Agents — batteries-included agent framework built on rustchain and langgraph.
//!
//! This crate provides a high-level, opinionated API for building LLM-powered agents
//! with pluggable middleware, backends, and zero-boilerplate setup.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use deepagents::config::DeepAgentConfig;
//! use deepagents::create_deep_agent;
//!
//! // let config = DeepAgentConfig::default();
//! // let graph = create_deep_agent(config).unwrap();
//! ```

pub mod agent;
pub mod backends;
pub mod config;
pub mod middleware;

pub use agent::create_deep_agent;
