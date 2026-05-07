//! Blob/object storage backend — bytes keyed by string.
//!
//! Distinct from the text-shaped [`Backend`](super::Backend): operates on
//! arbitrary bytes (e.g. rendered artifacts, downloaded files). Local-FS
//! impl ships; an S3-shaped impl belongs in a downstream crate behind a
//! feature flag.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use async_trait::async_trait;

use cognis_core::{CognisError, Result};

/// One stored blob.
#[derive(Debug, Clone)]
pub struct Blob {
    /// Raw bytes.
    pub bytes: Vec<u8>,
    /// MIME type (e.g. `"image/png"`).
    pub content_type: String,
}

/// Pluggable blob store.
#[async_trait]
pub trait StorageBackend: Send + Sync {
    /// Store bytes under `key`. Replaces existing.
    async fn put_blob(&self, key: &str, bytes: Vec<u8>, content_type: &str) -> Result<()>;
    /// Read bytes by key.
    async fn get_blob(&self, key: &str) -> Result<Option<Blob>>;
    /// Delete by key (silent if absent).
    async fn delete_blob(&self, key: &str) -> Result<()>;
    /// List all keys with the given prefix.
    async fn list_blobs(&self, prefix: &str) -> Result<Vec<String>>;
}

/// In-process blob store.
#[derive(Default)]
pub struct InMemoryStorageBackend {
    inner: Mutex<HashMap<String, Blob>>,
}

impl InMemoryStorageBackend {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl StorageBackend for InMemoryStorageBackend {
    async fn put_blob(&self, key: &str, bytes: Vec<u8>, content_type: &str) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| CognisError::Internal(format!("storage mutex: {e}")))?;
        inner.insert(
            key.to_string(),
            Blob {
                bytes,
                content_type: content_type.to_string(),
            },
        );
        Ok(())
    }
    async fn get_blob(&self, key: &str) -> Result<Option<Blob>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| CognisError::Internal(format!("storage mutex: {e}")))?;
        Ok(inner.get(key).cloned())
    }
    async fn delete_blob(&self, key: &str) -> Result<()> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| CognisError::Internal(format!("storage mutex: {e}")))?;
        inner.remove(key);
        Ok(())
    }
    async fn list_blobs(&self, prefix: &str) -> Result<Vec<String>> {
        let inner = self
            .inner
            .lock()
            .map_err(|e| CognisError::Internal(format!("storage mutex: {e}")))?;
        let mut out: Vec<String> = inner
            .keys()
            .filter(|k| k.starts_with(prefix))
            .cloned()
            .collect();
        out.sort();
        Ok(out)
    }
}

/// Local-filesystem blob store rooted at a directory. Keys map directly
/// to relative file paths; `..` traversal is refused.
pub struct LocalFsStorageBackend {
    root: PathBuf,
}

impl LocalFsStorageBackend {
    /// Build at `root` (created if missing). Path is canonicalized.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root).map_err(|e| {
            CognisError::Configuration(format!("storage root `{}`: {e}", root.display()))
        })?;
        Ok(Self {
            root: root
                .canonicalize()
                .map_err(|e| CognisError::Configuration(format!("canonicalize: {e}")))?,
        })
    }

    fn resolve(&self, key: &str) -> Result<PathBuf> {
        if key.starts_with('/') || key.split('/').any(|s| s == "..") {
            return Err(CognisError::Configuration(format!(
                "storage: refusing key `{key}`"
            )));
        }
        let p = self.root.join(key.trim_start_matches("./"));
        Ok(p)
    }
}

#[async_trait]
impl StorageBackend for LocalFsStorageBackend {
    async fn put_blob(&self, key: &str, bytes: Vec<u8>, content_type: &str) -> Result<()> {
        let p = self.resolve(key)?;
        if let Some(parent) = p.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(|e| {
                CognisError::Configuration(format!("mkdir `{}`: {e}", parent.display()))
            })?;
        }
        tokio::fs::write(&p, &bytes)
            .await
            .map_err(|e| CognisError::Configuration(format!("write `{}`: {e}", p.display())))?;
        // Sidecar file for content-type so we can reconstruct on read.
        let meta = p.with_extension("__ct");
        tokio::fs::write(&meta, content_type)
            .await
            .map_err(|e| CognisError::Configuration(format!("write content-type: {e}")))?;
        Ok(())
    }
    async fn get_blob(&self, key: &str) -> Result<Option<Blob>> {
        let p = self.resolve(key)?;
        if !p.exists() {
            return Ok(None);
        }
        let bytes = tokio::fs::read(&p)
            .await
            .map_err(|e| CognisError::Configuration(format!("read `{}`: {e}", p.display())))?;
        let meta = p.with_extension("__ct");
        let content_type = tokio::fs::read_to_string(&meta)
            .await
            .unwrap_or_else(|_| "application/octet-stream".into());
        Ok(Some(Blob {
            bytes,
            content_type,
        }))
    }
    async fn delete_blob(&self, key: &str) -> Result<()> {
        let p = self.resolve(key)?;
        let _ = tokio::fs::remove_file(&p).await;
        let _ = tokio::fs::remove_file(p.with_extension("__ct")).await;
        Ok(())
    }
    async fn list_blobs(&self, prefix: &str) -> Result<Vec<String>> {
        // Walk root, return relative paths matching prefix and not __ct sidecars.
        let mut out = Vec::new();
        let mut stack = vec![self.root.clone()];
        while let Some(d) = stack.pop() {
            let mut entries = tokio::fs::read_dir(&d).await.map_err(|e| {
                CognisError::Configuration(format!("read_dir `{}`: {e}", d.display()))
            })?;
            while let Some(entry) = entries
                .next_entry()
                .await
                .map_err(|e| CognisError::Configuration(format!("read_dir entry: {e}")))?
            {
                let p = entry.path();
                if p.is_dir() {
                    stack.push(p);
                } else if p.extension().and_then(|s| s.to_str()) == Some("__ct") {
                    continue;
                } else if let Ok(rel) = p.strip_prefix(&self.root) {
                    let s = rel.to_string_lossy().replace('\\', "/");
                    if s.starts_with(prefix) {
                        out.push(s);
                    }
                }
            }
        }
        out.sort();
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn in_memory_roundtrip() {
        let s = InMemoryStorageBackend::new();
        s.put_blob("img/a.png", vec![1, 2, 3], "image/png")
            .await
            .unwrap();
        let got = s.get_blob("img/a.png").await.unwrap().unwrap();
        assert_eq!(got.bytes, vec![1, 2, 3]);
        assert_eq!(got.content_type, "image/png");
        assert_eq!(s.list_blobs("img/").await.unwrap(), vec!["img/a.png"]);
    }

    #[tokio::test]
    async fn local_fs_roundtrip() {
        let dir = TempDir::new().unwrap();
        let s = LocalFsStorageBackend::new(dir.path()).unwrap();
        s.put_blob("a.bin", vec![9, 9, 9], "application/octet-stream")
            .await
            .unwrap();
        let got = s.get_blob("a.bin").await.unwrap().unwrap();
        assert_eq!(got.bytes, vec![9, 9, 9]);
        assert!(s
            .list_blobs("")
            .await
            .unwrap()
            .contains(&"a.bin".to_string()));
    }

    #[tokio::test]
    async fn local_fs_refuses_traversal() {
        let dir = TempDir::new().unwrap();
        let s = LocalFsStorageBackend::new(dir.path()).unwrap();
        assert!(s
            .put_blob("../escape", vec![1], "text/plain")
            .await
            .is_err());
    }
}
