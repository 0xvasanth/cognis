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

    fn session_path(&self, session_id: &str) -> Result<PathBuf> {
        // Reject session IDs that could escape the storage directory.
        if session_id.contains('/')
            || session_id.contains('\\')
            || session_id.contains("..")
            || session_id.is_empty()
        {
            return Err(DeepAgentError::Other(format!(
                "invalid session_id '{}': must not contain path separators or '..'",
                session_id
            )));
        }
        Ok(self.dir.join(format!("{session_id}.json")))
    }

    /// Validate that a path does not escape the base directory.
    ///
    /// Rejects absolute paths and paths containing `..` components.
    /// For existing files, canonicalizes and checks the prefix.
    fn validate_path(&self, path: &str) -> Result<PathBuf> {
        // Reject absolute paths.
        if std::path::Path::new(path).is_absolute() {
            return Err(DeepAgentError::Other(format!(
                "path '{}': absolute paths are not allowed",
                path
            )));
        }
        // Reject .. components.
        for component in std::path::Path::new(path).components() {
            if matches!(component, std::path::Component::ParentDir) {
                return Err(DeepAgentError::Other(format!(
                    "path '{}': path traversal ('..') is not allowed",
                    path
                )));
            }
        }
        let full_path = self.dir.join(path);
        // If the file exists, also verify via canonicalization.
        if full_path.exists() {
            let canonical = full_path
                .canonicalize()
                .map_err(|e| DeepAgentError::Other(format!("path '{}': {}", path, e)))?;
            let base = self
                .dir
                .canonicalize()
                .map_err(|e| DeepAgentError::Other(format!("base dir: {}", e)))?;
            if !canonical.starts_with(&base) {
                return Err(DeepAgentError::Other(format!(
                    "path '{}': escapes base directory",
                    path
                )));
            }
        }
        Ok(full_path)
    }

    /// Validate a path for write operations (file may not exist yet).
    ///
    /// Rejects `..` and absolute paths. If the target already exists (e.g.
    /// overwriting or following a symlink), canonicalizes to ensure the
    /// resolved path stays inside the base directory.
    fn validate_path_for_write(&self, path: &str) -> Result<PathBuf> {
        if std::path::Path::new(path).is_absolute() {
            return Err(DeepAgentError::Other(format!(
                "path '{}': absolute paths are not allowed",
                path
            )));
        }
        for component in std::path::Path::new(path).components() {
            if matches!(component, std::path::Component::ParentDir) {
                return Err(DeepAgentError::Other(format!(
                    "path '{}': path traversal ('..') is not allowed",
                    path
                )));
            }
        }
        let full_path = self.dir.join(path);
        // If the target already exists (e.g. a symlink), verify it resolves
        // inside the base directory to prevent symlink escapes.
        if full_path.exists() {
            let canonical = full_path
                .canonicalize()
                .map_err(|e| DeepAgentError::Other(format!("path '{}': {}", path, e)))?;
            let base = self
                .dir
                .canonicalize()
                .map_err(|e| DeepAgentError::Other(format!("base dir: {}", e)))?;
            if !canonical.starts_with(&base) {
                return Err(DeepAgentError::Other(format!(
                    "path '{}': escapes base directory via symlink",
                    path
                )));
            }
        }
        Ok(full_path)
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

        tokio::fs::write(self.session_path(session_id)?, data)
            .await
            .map_err(|e| DeepAgentError::BackendError(e.to_string()))?;

        Ok(())
    }

    async fn load_state(&self, session_id: &str) -> Result<Option<Value>> {
        let path = self.session_path(session_id)?;
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

    async fn read_file(&self, path: &str) -> Result<String> {
        let full_path = self.validate_path(path)?;
        tokio::fs::read_to_string(&full_path)
            .await
            .map_err(|e| DeepAgentError::Other(format!("read_file '{}': {}", path, e)))
    }

    async fn write_file(&self, path: &str, content: &str) -> Result<()> {
        let full_path = self.validate_path_for_write(path)?;
        if let Some(parent) = full_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| DeepAgentError::Other(format!("create dirs for '{}': {}", path, e)))?;
        }
        tokio::fs::write(&full_path, content)
            .await
            .map_err(|e| DeepAgentError::Other(format!("write_file '{}': {}", path, e)))
    }

    async fn edit_file(&self, path: &str, old: &str, new: &str) -> Result<()> {
        let content = self.read_file(path).await?;
        if !content.contains(old) {
            return Err(DeepAgentError::Other(format!(
                "edit_file '{}': old string not found",
                path
            )));
        }
        let updated = content.replacen(old, new, 1);
        self.write_file(path, &updated).await
    }

    async fn list_dir(&self, path: &str) -> Result<Vec<super::FileInfo>> {
        let full_path = self.validate_path(path)?;
        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(&full_path)
            .await
            .map_err(|e| DeepAgentError::Other(format!("list_dir '{}': {}", path, e)))?;
        while let Some(entry) = dir
            .next_entry()
            .await
            .map_err(|e| DeepAgentError::Other(format!("list_dir entry: {}", e)))?
        {
            let metadata = entry
                .metadata()
                .await
                .map_err(|e| DeepAgentError::Other(format!("metadata: {}", e)))?;
            let name = entry.file_name().to_string_lossy().to_string();
            entries.push(super::FileInfo {
                path: name,
                is_dir: metadata.is_dir(),
                size: metadata.len(),
            });
        }
        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn test_filesystem_backend_write_and_read() {
        let dir = TempDir::new().unwrap();
        let backend = FilesystemBackend::new(dir.path());
        backend.write_file("test.txt", "hello world").await.unwrap();
        let content = backend.read_file("test.txt").await.unwrap();
        assert_eq!(content, "hello world");
    }

    #[tokio::test]
    async fn test_filesystem_backend_edit_file() {
        let dir = TempDir::new().unwrap();
        let backend = FilesystemBackend::new(dir.path());
        backend.write_file("edit.txt", "hello world").await.unwrap();
        backend
            .edit_file("edit.txt", "world", "rust")
            .await
            .unwrap();
        let content = backend.read_file("edit.txt").await.unwrap();
        assert_eq!(content, "hello rust");
    }

    #[tokio::test]
    async fn test_filesystem_backend_list_dir() {
        let dir = TempDir::new().unwrap();
        let backend = FilesystemBackend::new(dir.path());
        backend.write_file("a.txt", "aaa").await.unwrap();
        backend.write_file("b.txt", "bbb").await.unwrap();
        let entries = backend.list_dir(".").await.unwrap();
        assert!(entries.len() >= 2);
    }

    #[tokio::test]
    async fn test_filesystem_backend_read_nonexistent() {
        let dir = TempDir::new().unwrap();
        let backend = FilesystemBackend::new(dir.path());
        let result = backend.read_file("missing.txt").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_filesystem_backend_edit_missing_string() {
        let dir = TempDir::new().unwrap();
        let backend = FilesystemBackend::new(dir.path());
        backend.write_file("edit.txt", "hello world").await.unwrap();
        let result = backend
            .edit_file("edit.txt", "missing", "replacement")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_path_traversal_rejected() {
        let dir = TempDir::new().unwrap();
        let backend = FilesystemBackend::new(dir.path());

        // .. traversal
        let result = backend.read_file("../../etc/passwd").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("path traversal"));

        // Absolute path
        let result = backend.read_file("/etc/passwd").await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("absolute"));

        // write should also reject
        let result = backend.write_file("../escape.txt", "bad").await;
        assert!(result.is_err());
    }
}
