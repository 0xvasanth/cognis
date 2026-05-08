//! Directory walker — recursively yields one [`Document`] per file matching
//! a glob pattern.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use futures::stream;

use cognis_core::{CognisError, Result};

use crate::document::Document;

use super::{DocumentLoader, DocumentStream};

/// Recursively walks `root`, yielding one [`Document`] per file whose name
/// ends with one of the configured `suffixes` (default: `.txt`, `.md`).
///
/// Symlinks are **skipped by default** to avoid filesystem cycles
/// (a symlink to an ancestor would loop forever). Opt back in with
/// [`DirectoryLoader::follow_symlinks`] — at your own risk.
pub struct DirectoryLoader {
    root: PathBuf,
    suffixes: Vec<String>,
    recursive: bool,
    follow_symlinks: bool,
}

impl DirectoryLoader {
    /// Construct with the root directory.
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            suffixes: vec![".txt".into(), ".md".into()],
            recursive: true,
            follow_symlinks: false,
        }
    }

    /// Replace the suffix allowlist (e.g. `[".txt"]` or `[".md", ".txt"]`).
    pub fn with_suffixes<I, S>(mut self, suffixes: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.suffixes = suffixes.into_iter().map(Into::into).collect();
        self
    }

    /// Disable recursion (default: enabled).
    pub fn non_recursive(mut self) -> Self {
        self.recursive = false;
        self
    }

    /// Follow symlinks during traversal. **Off by default** — enabling
    /// this can cause unbounded recursion if any descendant points back
    /// at an ancestor. Only flip this on for trees you control.
    pub fn follow_symlinks(mut self) -> Self {
        self.follow_symlinks = true;
        self
    }

    fn collect(&self) -> Result<Vec<PathBuf>> {
        let mut out = Vec::new();
        self.walk(&self.root, &mut out)?;
        Ok(out)
    }

    fn walk(&self, dir: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        let read = std::fs::read_dir(dir).map_err(|e| {
            CognisError::Configuration(format!(
                "DirectoryLoader: read_dir `{}`: {e}",
                dir.display()
            ))
        })?;
        for entry in read.flatten() {
            let path = entry.path();
            // Inspect the entry without traversing the link target.
            let meta = match std::fs::symlink_metadata(&path) {
                Ok(m) => m,
                Err(_) => continue, // racing with deletion etc. — skip silently
            };
            if meta.file_type().is_symlink() && !self.follow_symlinks {
                continue;
            }
            if meta.is_dir() {
                if self.recursive {
                    self.walk(&path, out)?;
                }
            } else if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if self.suffixes.iter().any(|sfx| name.ends_with(sfx)) {
                    out.push(path);
                }
            }
        }
        Ok(())
    }
}

#[async_trait]
impl DocumentLoader for DirectoryLoader {
    async fn load(&self) -> Result<DocumentStream> {
        let paths = self.collect()?;
        let docs: Result<Vec<Document>> = paths
            .into_iter()
            .map(|p| {
                let content = std::fs::read_to_string(&p).map_err(|e| {
                    CognisError::Configuration(format!("read `{}`: {e}", p.display()))
                })?;
                Ok(Document::new(content).with_metadata("source", p.display().to_string()))
            })
            .collect();
        Ok(Box::pin(stream::iter(docs?.into_iter().map(Ok))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn write(dir: &Path, name: &str, content: &str) {
        let p = dir.join(name);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        let mut f = std::fs::File::create(p).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    #[tokio::test]
    async fn finds_txt_and_md_recursively() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "a.txt", "alpha");
        write(dir.path(), "sub/b.md", "beta");
        write(dir.path(), "skip.bin", "ignored");

        let docs = DirectoryLoader::new(dir.path()).load_all().await.unwrap();
        assert_eq!(docs.len(), 2);
        let contents: Vec<_> = docs.iter().map(|d| d.content.clone()).collect();
        assert!(contents.iter().any(|c| c == "alpha"));
        assert!(contents.iter().any(|c| c == "beta"));
    }

    #[tokio::test]
    async fn non_recursive_skips_subdirs() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "top.txt", "top");
        write(dir.path(), "sub/inner.txt", "inner");

        let docs = DirectoryLoader::new(dir.path())
            .non_recursive()
            .load_all()
            .await
            .unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "top");
    }

    #[tokio::test]
    async fn suffix_filter() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "a.txt", "1");
        write(dir.path(), "b.md", "2");

        let docs = DirectoryLoader::new(dir.path())
            .with_suffixes([".md"])
            .load_all()
            .await
            .unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "2");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn symlink_cycle_does_not_recurse() {
        let dir = TempDir::new().unwrap();
        write(dir.path(), "real.txt", "real");
        // Create a symlink loop: dir/loop -> dir
        std::os::unix::fs::symlink(dir.path(), dir.path().join("loop")).unwrap();

        // Default behavior: symlinks skipped; should terminate and find
        // exactly the one real file.
        let docs = DirectoryLoader::new(dir.path()).load_all().await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "real");
    }
}
