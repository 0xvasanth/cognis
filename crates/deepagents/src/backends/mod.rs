//! Backend system for persisting agent state.
//!
//! Backends allow saving and loading agent session state so that
//! conversations can be resumed or inspected later.

pub mod filesystem;
pub mod sandbox;
pub mod state;
pub mod storage;

pub use self::filesystem::FilesystemBackend;
pub use self::sandbox::SandboxBackend;
pub use self::state::StateBackend;
pub use self::storage::{
    BackendConfig, BackendFactory, BackendStats, CachedBackend, FilesystemStorageBackend,
    InMemoryBackend, NamespacedBackend, StorageBackend,
};

use async_trait::async_trait;
use serde_json::Value;

use crate::agent::DeepAgentError;

/// Result type for backend operations.
pub type Result<T> = std::result::Result<T, DeepAgentError>;

/// Trait for agent state persistence backends.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Save the state for a given session.
    async fn save_state(&self, session_id: &str, state: &Value) -> Result<()>;

    /// Load the state for a given session. Returns `None` if no state exists.
    async fn load_state(&self, session_id: &str) -> Result<Option<Value>>;

    /// List all known session IDs.
    async fn list_sessions(&self) -> Result<Vec<String>>;
}
