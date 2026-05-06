//! Checkpointing — persist graph state across supersteps for resume,
//! time-travel, and human-in-the-loop interrupts.

use async_trait::async_trait;
use uuid::Uuid;

use cognis2_core::Result;

use crate::state::GraphState;

mod in_memory;
pub use in_memory::InMemoryCheckpointer;

/// Trait for storing & retrieving graph state at superstep boundaries.
///
/// Slice 1 ships [`InMemoryCheckpointer`]. SqliteCheckpointer lives in
/// a separate (deferred) plan.
#[async_trait]
pub trait Checkpointer<S: GraphState>: Send + Sync {
    /// Save state at `step` for `run_id`.
    async fn save(&self, run_id: Uuid, step: u64, state: &S) -> Result<()>;

    /// Load state for `run_id` at `step` (or the latest if `step` is None).
    async fn load(&self, run_id: Uuid, step: Option<u64>) -> Result<Option<S>>;

    /// List all saved step numbers for `run_id`.
    async fn list(&self, run_id: Uuid) -> Result<Vec<u64>>;
}
