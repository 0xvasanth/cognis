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
//! - [`middleware::subagent::SubAgentMiddleware`] -- isolated sub-agent delegation
//! - [`middleware::summarization::SummarizationMiddleware`] -- context window management
//! - [`middleware::skills::SkillsMiddleware`] -- custom skill loading
//! - [`middleware::patch_tool_calls::PatchToolCallsMiddleware`] -- tool call correction
//! - [`middleware::rate_limiter::RateLimiterMiddleware`] -- token bucket rate limiting
//! - [`middleware::logging::LoggingMiddleware`] -- structured logging with redaction
//! - [`middleware::context::ContextMiddleware`] -- dynamic context injection
//! - [`middleware::planning::PlanningMiddleware`] -- plan-then-execute with step tracking
//!
//! ## Tool Registry
//!
//! The [`tool_registry::ToolRegistry`] provides centralized tool management with
//! permission levels, call counting, enable/disable control, and filtering.
//!
//! ## Backends
//!
//! The [`backends::Backend`] trait abstracts session state persistence:
//!
//! - [`backends::StateBackend`] -- in-memory (default)
//! - [`backends::FilesystemBackend`] -- local disk storage
//! - [`backends::SandboxBackend`] -- isolated execution with resource limits
//!
//! ## Events
//!
//! The [`events`] module provides a typed event bus system with handlers and
//! lifecycle tracking for observing and reacting to agent execution events.
//!
//! ## Conversation
//!
//! The [`conversation`] module provides a conversation manager with context
//! windowing, message history management, and export/serialization support.
//!
//! ## Presets
//!
//! The [`presets`] module provides ready-made agent configurations via a preset
//! registry and customizer pattern for common agent types.

pub mod agent;
pub mod backends;
pub mod builder;
pub mod communication;
pub mod config;
pub mod context;
pub mod conversation;
pub mod events;
pub mod factory;
pub mod health;
pub mod memory;
pub mod middleware;
pub mod orchestrator;
pub mod plugins;
pub mod presets;
pub mod recovery;
pub mod sandbox;
pub mod session;
pub mod skills;
pub mod summarization;
pub mod telemetry;
pub mod tool_registry;
pub mod workflow;

pub use agent::create_deep_agent;
