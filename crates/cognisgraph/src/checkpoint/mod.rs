//! Checkpointing — persist graph state across supersteps for resume,
//! time-travel, and human-in-the-loop interrupts.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use cognis_core::Result;

use crate::state::GraphState;

mod in_memory;
pub use in_memory::InMemoryCheckpointer;

pub mod serializer;
#[cfg(feature = "serializer-cbor")]
pub use serializer::CborSerializer;
pub use serializer::{CheckpointSerializer, JsonSerializer};

#[cfg(feature = "sqlite")]
pub mod sqlite;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteCheckpointer;

#[cfg(feature = "postgres")]
pub mod postgres;
#[cfg(feature = "postgres")]
pub use postgres::PostgresCheckpointer;

/// Snapshot of one active task at an interrupt boundary. Used for
/// point-of-interrupt resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveSnapshot {
    /// The node name to dispatch on resume.
    pub node_name: String,
    /// Per-target payload from `Goto::Send`, if any.
    pub payload: Option<serde_json::Value>,
}

/// Trait for storing & retrieving graph state at superstep boundaries.
#[async_trait]
pub trait Checkpointer<S: GraphState>: Send + Sync {
    /// Save state at `step` for `run_id`.
    async fn save(&self, run_id: Uuid, step: u64, state: &S) -> Result<()>;

    /// Load state for `run_id` at `step` (or the latest if `step` is None).
    async fn load(&self, run_id: Uuid, step: Option<u64>) -> Result<Option<S>>;

    /// List all saved step numbers for `run_id`.
    async fn list(&self, run_id: Uuid) -> Result<Vec<u64>>;

    /// Save the engine's active task snapshot alongside state. Default
    /// is no-op — older checkpointers won't persist the active set, and
    /// `engine::resume` falls back to the start node when `load_active`
    /// returns empty.
    async fn save_active(&self, run_id: Uuid, step: u64, active: &[ActiveSnapshot]) -> Result<()> {
        let _ = (run_id, step, active);
        Ok(())
    }

    /// Load the active task snapshot for `(run_id, step)`. Default is empty.
    async fn load_active(&self, run_id: Uuid, step: u64) -> Result<Vec<ActiveSnapshot>> {
        let _ = (run_id, step);
        Ok(Vec::new())
    }
}
