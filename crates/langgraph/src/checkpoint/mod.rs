//! Checkpoint persistence backends for LangGraph.
//!
//! This module provides the [`CheckpointSaver`] trait and concrete
//! implementations for persisting graph state across restarts.
//!
//! The core checkpoint types and the `CheckpointSaver` trait are defined in
//! [`crate::pregel::checkpoint`]. This module re-exports them and adds
//! additional backends:
//!
//! - [`InMemoryCheckpointSaver`] (re-exported from pregel) — ephemeral, for testing
//! - [`SqliteCheckpointSaver`] (behind the `sqlite` feature) — durable SQLite storage

pub mod memory;
pub mod serialization;

#[cfg(feature = "sqlite")]
pub mod sqlite;

#[cfg(feature = "postgres")]
pub mod postgres;

// Re-export the core checkpoint types and trait so users can import from
// `langgraph::checkpoint` directly.
pub use crate::pregel::checkpoint::{
    Checkpoint, CheckpointEntry, CheckpointMetadata, CheckpointSaver, CheckpointTuple,
    InMemoryCheckpointSaver,
};

#[cfg(feature = "sqlite")]
pub use sqlite::SqliteCheckpointSaver;

#[cfg(feature = "postgres")]
pub use postgres::PostgresCheckpointSaver;
