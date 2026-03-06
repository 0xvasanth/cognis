//! Filesystem-based state backend.
//!
//! Each session is stored as a JSON file in a configurable directory.

use std::path::PathBuf;

use async_trait::async_trait;
use serde_json::Value;

use super::{Backend, Result};
use crate::agent::DeepAgentError;

/// Backend that persists session state as JSON files on disk.
pub struct FilesystemBackend {
    /// Directory where session files are stored.
    dir: PathBuf,
}

impl FilesystemBackend {
    /// Create a new `FilesystemBackend` that stores files in `dir`.
    ///
    /// The directory will be created on the first write if it does not exist.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.dir.join(format!("{session_id}.json"))
    }
}

#[async_trait]
impl Backend for FilesystemBackend {
    async fn save_state(&self, session_id: &str, state: &Value) -> Result<()> {
        tokio::fs::create_dir_all(&self.dir)
            .await
            .map_err(|e| DeepAgentError::BackendError(e.to_string()))?;

        let data = serde_json::to_string_pretty(state)
            .map_err(|e| DeepAgentError::BackendError(e.to_string()))?;

        tokio::fs::write(self.session_path(session_id), data)
            .await
            .map_err(|e| DeepAgentError::BackendError(e.to_string()))?;

        Ok(())
    }

    async fn load_state(&self, session_id: &str) -> Result<Option<Value>> {
        let path = self.session_path(session_id);
        if !path.exists() {
            return Ok(None);
        }

        let data = tokio::fs::read_to_string(&path)
            .await
            .map_err(|e| DeepAgentError::BackendError(e.to_string()))?;

        let value: Value =
            serde_json::from_str(&data).map_err(|e| DeepAgentError::BackendError(e.to_string()))?;

        Ok(Some(value))
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        if !self.dir.exists() {
            return Ok(Vec::new());
        }

        let mut sessions = Vec::new();
        let mut dir = tokio::fs::read_dir(&self.dir)
            .await
            .map_err(|e| DeepAgentError::BackendError(e.to_string()))?;

        while let Some(entry) = dir
            .next_entry()
            .await
            .map_err(|e| DeepAgentError::BackendError(e.to_string()))?
        {
            let name = entry.file_name().to_string_lossy().to_string();
            if let Some(session_id) = name.strip_suffix(".json") {
                sessions.push(session_id.to_string());
            }
        }

        sessions.sort();
        Ok(sessions)
    }
}
