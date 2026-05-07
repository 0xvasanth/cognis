//! Sandboxed real-filesystem backend rooted at a directory.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use tokio::fs;

use cognis_core::{CognisError, Result};

use super::memory::glob_match;
use super::{Backend, GrepHit};

/// Rooted-filesystem backend. Every path the agent supplies is resolved
/// relative to `root`; any resolution that escapes `root` is refused.
pub struct SandboxedFsBackend {
    root: PathBuf,
}

impl SandboxedFsBackend {
    /// Create a sandbox rooted at `root`. The directory is created if it
    /// doesn't exist.
    pub fn new(root: impl AsRef<Path>) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        std::fs::create_dir_all(&root).map_err(|e| {
            CognisError::Configuration(format!(
                "SandboxedFsBackend: create root `{}`: {e}",
                root.display()
            ))
        })?;
        Ok(Self {
            root: root
                .canonicalize()
                .map_err(|e| CognisError::Configuration(format!("canonicalize root: {e}")))?,
        })
    }

    /// Resolve `rel` against the sandbox root, refusing escapes.
    fn resolve(&self, rel: &str) -> Result<PathBuf> {
        if rel.starts_with('/') {
            return Err(CognisError::Configuration(format!(
                "SandboxedFsBackend: absolute paths not allowed: `{rel}`"
            )));
        }
        if rel.split('/').any(|seg| seg == "..") {
            return Err(CognisError::Configuration(format!(
                "SandboxedFsBackend: path traversal not allowed: `{rel}`"
            )));
        }
        let trimmed = rel.trim_start_matches("./");
        let candidate = self.root.join(trimmed);
        // Canonicalize when the path exists; otherwise compose-and-check.
        let canon = if candidate.exists() {
            candidate
                .canonicalize()
                .map_err(|e| CognisError::Configuration(format!("canonicalize `{rel}`: {e}")))?
        } else {
            candidate
        };
        if !canon.starts_with(&self.root) {
            return Err(CognisError::Configuration(format!(
                "SandboxedFsBackend: path `{rel}` escapes sandbox root"
            )));
        }
        Ok(canon)
    }

    /// Walk the sandbox and collect all file paths (relative to the root).
    fn walk(&self) -> Result<Vec<String>> {
        let mut out = Vec::new();
        self.walk_dir(&self.root, &mut out)?;
        Ok(out)
    }

    fn walk_dir(&self, dir: &Path, out: &mut Vec<String>) -> Result<()> {
        let read = std::fs::read_dir(dir).map_err(|e| {
            CognisError::Configuration(format!("read_dir `{}`: {e}", dir.display()))
        })?;
        for entry in read.flatten() {
            let p = entry.path();
            if p.is_dir() {
                self.walk_dir(&p, out)?;
            } else {
                let rel = p
                    .strip_prefix(&self.root)
                    .map_err(|e| CognisError::Internal(format!("strip_prefix: {e}")))?;
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Backend for SandboxedFsBackend {
    async fn read(&self, path: &str) -> Result<String> {
        let p = self.resolve(path)?;
        fs::read_to_string(&p)
            .await
            .map_err(|e| CognisError::Configuration(format!("read `{}`: {e}", p.display())))
    }

    async fn write(&self, path: &str, contents: &str) -> Result<()> {
        let p = self.resolve(path)?;
        if let Some(parent) = p.parent() {
            fs::create_dir_all(parent).await.map_err(|e| {
                CognisError::Configuration(format!("mkdir `{}`: {e}", parent.display()))
            })?;
        }
        fs::write(&p, contents)
            .await
            .map_err(|e| CognisError::Configuration(format!("write `{}`: {e}", p.display())))
    }

    async fn edit(
        &self,
        path: &str,
        find: &str,
        replace: &str,
        max_occurrences: usize,
    ) -> Result<usize> {
        let body = self.read(path).await?;
        let count = body.matches(find).count();
        if count == 0 {
            return Err(CognisError::Configuration(format!(
                "edit: `find` not present in `{path}`"
            )));
        }
        if count > max_occurrences {
            return Err(CognisError::Configuration(format!(
                "edit: `find` occurs {count} times, exceeds max_occurrences={max_occurrences}"
            )));
        }
        let new_body = body.replacen(find, replace, max_occurrences);
        self.write(path, &new_body).await?;
        Ok(count)
    }

    async fn ls(&self, dir: &str) -> Result<Vec<String>> {
        let resolved = if dir.is_empty() || dir == "." {
            self.root.clone()
        } else {
            self.resolve(dir)?
        };
        let mut entries = fs::read_dir(&resolved).await.map_err(|e| {
            CognisError::Configuration(format!("read_dir `{}`: {e}", resolved.display()))
        })?;
        let mut out = Vec::new();
        while let Some(entry) = entries
            .next_entry()
            .await
            .map_err(|e| CognisError::Configuration(format!("read_dir entry: {e}")))?
        {
            let p = entry.path();
            if p.is_file() {
                let rel = p
                    .strip_prefix(&self.root)
                    .map_err(|e| CognisError::Internal(format!("strip_prefix: {e}")))?;
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
        out.sort();
        Ok(out)
    }

    async fn glob(&self, pattern: &str) -> Result<Vec<String>> {
        let mut all = self.walk()?;
        all.retain(|p| glob_match(pattern, p));
        all.sort();
        Ok(all)
    }

    async fn grep(&self, pattern: &str) -> Result<Vec<GrepHit>> {
        let paths = self.walk()?;
        let mut out = Vec::new();
        for path in paths {
            let body = self.read(&path).await.unwrap_or_default();
            for (i, line) in body.lines().enumerate() {
                if line.contains(pattern) {
                    out.push(GrepHit {
                        path: path.clone(),
                        line: (i + 1) as u64,
                        text: line.to_string(),
                    });
                }
            }
        }
        Ok(out)
    }

    async fn exists(&self, path: &str) -> Result<bool> {
        let p = match self.resolve(path) {
            Ok(p) => p,
            Err(_) => return Ok(false),
        };
        Ok(p.exists())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[tokio::test]
    async fn write_then_read() {
        let dir = TempDir::new().unwrap();
        let b = SandboxedFsBackend::new(dir.path()).unwrap();
        b.write("a.txt", "hello").await.unwrap();
        assert_eq!(b.read("a.txt").await.unwrap(), "hello");
    }

    #[tokio::test]
    async fn refuses_absolute_and_traversal() {
        let dir = TempDir::new().unwrap();
        let b = SandboxedFsBackend::new(dir.path()).unwrap();
        assert!(b.read("/etc/passwd").await.is_err());
        assert!(b.read("../outside").await.is_err());
    }

    #[tokio::test]
    async fn ls_root() {
        let dir = TempDir::new().unwrap();
        let b = SandboxedFsBackend::new(dir.path()).unwrap();
        b.write("a.txt", "1").await.unwrap();
        b.write("sub/b.txt", "2").await.unwrap();
        let top = b.ls(".").await.unwrap();
        assert_eq!(top, vec!["a.txt"]);
    }

    #[tokio::test]
    async fn glob_recursive() {
        let dir = TempDir::new().unwrap();
        let b = SandboxedFsBackend::new(dir.path()).unwrap();
        b.write("a.txt", "1").await.unwrap();
        b.write("sub/b.txt", "2").await.unwrap();
        let mut all = b.glob("**/*.txt").await.unwrap();
        all.sort();
        assert_eq!(all, vec!["a.txt", "sub/b.txt"]);
    }

    #[tokio::test]
    async fn grep_across_files() {
        let dir = TempDir::new().unwrap();
        let b = SandboxedFsBackend::new(dir.path()).unwrap();
        b.write("a.txt", "hi\nbye\n").await.unwrap();
        b.write("b.txt", "bye\n").await.unwrap();
        let hits = b.grep("bye").await.unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[tokio::test]
    async fn edit_replaces_unique() {
        let dir = TempDir::new().unwrap();
        let b = SandboxedFsBackend::new(dir.path()).unwrap();
        b.write("a.txt", "hello world").await.unwrap();
        b.edit("a.txt", "world", "rust", 1).await.unwrap();
        assert_eq!(b.read("a.txt").await.unwrap(), "hello rust");
    }
}
