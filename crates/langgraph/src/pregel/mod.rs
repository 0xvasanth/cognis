//! Pregel execution engine for LangGraph.
//!
//! The Pregel module implements a step-based, channel-driven execution model
//! inspired by Google's Pregel and Apache Beam. It provides:
//!
//! - [`executor::PregelExecutor`] — the core execution engine
//! - [`protocol::PregelProtocol`] — the abstract trait for graph execution
//! - [`types`] — Pregel-specific types (task descriptions, configs, etc.)
//! - Channel I/O helpers ([`read`], [`write`], [`io`])
//! - [`retry`] — retry logic with exponential backoff
//! - [`validate`] — graph structure validation
//! - [`checkpoint`] — checkpoint management

pub mod checkpoint;
pub mod executor;
pub mod io;
pub mod protocol;
pub mod read;
pub mod retry;
pub mod types;
pub mod utils;
pub mod validate;
pub mod write;

pub use executor::PregelExecutor;
pub use protocol::PregelProtocol;
pub use utils::{get_new_channel_versions, ChannelVersion, ChannelVersions};
pub use types::{
    All, ChannelWrite, InterruptConfig, PregelConfig, PregelNodeAction, PregelNodeSpec,
    PregelStatus, PregelTaskDescription, StreamChunk, TaskResult,
};
