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

/// Metadata about a file or directory entry.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileInfo {
    /// Relative path of the file.
    pub path: String,
    /// Whether this entry is a directory.
    pub is_dir: bool,
    /// File size in bytes (0 for directories).
    pub size: u64,
}

/// Trait for agent state persistence backends.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Save the state for a given session.
    async fn save_state(&self, session_id: &str, state: &Value) -> Result<()>;

    /// Load the state for a given session. Returns `None` if no state exists.
    async fn load_state(&self, session_id: &str) -> Result<Option<Value>>;

    /// List all known session IDs.
    async fn list_sessions(&self) -> Result<Vec<String>>;

    /// Read the contents of a file at the given path.
    async fn read_file(&self, _path: &str) -> Result<String> {
        Err(DeepAgentError::Other(
            "read_file not supported by this backend".into(),
        ))
    }

    /// Write content to a file at the given path, creating directories as needed.
    async fn write_file(&self, _path: &str, _content: &str) -> Result<()> {
        Err(DeepAgentError::Other(
            "write_file not supported by this backend".into(),
        ))
    }

    /// Replace the first occurrence of `old` with `new` in the file at the given path.
    async fn edit_file(&self, _path: &str, _old: &str, _new: &str) -> Result<()> {
        Err(DeepAgentError::Other(
            "edit_file not supported by this backend".into(),
        ))
    }

    /// List entries in the directory at the given path.
    async fn list_dir(&self, _path: &str) -> Result<Vec<FileInfo>> {
        Err(DeepAgentError::Other(
            "list_dir not supported by this backend".into(),
        ))
    }
}
