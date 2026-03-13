//! Utility modules for LangGraph.
//!
//! Provides configuration helpers, runnable utilities, and other shared
//! functionality used across the LangGraph crate.
//!
//! ## Submodules
//!
//! - [`config`] -- Configuration helpers for runnable execution (`RunnableConfig`, `merge_configs`).
//! - [`profiler`] -- Execution profiler with bottleneck detection for graph runs.
//! - [`timeout`] -- Node execution timeout, cancellation tokens, and budget management
//!   for enforcing per-node and per-graph time limits.

pub mod config;
pub mod profiler;
pub mod timeout;

pub use config::{
    ensure_config, merge_configs, patch_configurable, RunnableConfig, CONFIG_KEY_RUNTIME,
};
