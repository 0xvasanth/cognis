//! Agent workspace backend: file ops + sandboxing.
//!
//! [`Backend`] models the agent's "filesystem" — the surface tools dispatch
//! through when the agent reads, writes, edits, lists, globs, or greps files.
//! Two impls ship:
//!
//! - [`MemoryBackend`] — in-process `HashMap<PathBuf, String>`. Default for
//!   tests; ergonomic when you want a sandbox you fully control.
//! - [`SandboxedFsBackend`] — real filesystem rooted at a directory; refuses
//!   any path that escapes the root.
//!
//! Implementations are async because real backends will fan out to network
//! storage (S3, etc.) in a future stage.

pub mod memory;
pub mod sandbox;
pub mod state;
pub mod storage;

pub use memory::MemoryBackend;
pub use sandbox::SandboxedFsBackend;
pub use state::{InMemoryStateBackend, StateBackend};
pub use storage::{Blob, InMemoryStorageBackend, LocalFsStorageBackend, StorageBackend};

use async_trait::async_trait;

use cognis_core::Result;

/// Filesystem-shaped operations the agent dispatches through.
///
/// All paths are *backend-relative*. Implementations that touch the real
/// filesystem are responsible for sandboxing — never expose `..` traversal
/// or absolute paths that escape the configured root.
#[async_trait]
pub trait Backend: Send + Sync {
    /// Read the full contents of a file.
    async fn read(&self, path: &str) -> Result<String>;

    /// Write (overwrite) the full contents of a file. Creates parent
    /// directories as needed.
    async fn write(&self, path: &str, contents: &str) -> Result<()>;

    /// Replace `find` with `replace` within `path`. Errors if `find` is
    /// not present or appears more than `max_occurrences` times (default 1
    /// — the safest target for a literal in-place edit).
    async fn edit(
        &self,
        path: &str,
        find: &str,
        replace: &str,
        max_occurrences: usize,
    ) -> Result<usize>;

    /// List files (non-recursive) under `dir`.
    async fn ls(&self, dir: &str) -> Result<Vec<String>>;

    /// Match files against a glob pattern. The pattern is shell-style:
    /// `*` matches any number of non-`/` characters; `**` matches across
    /// directories; `?` matches a single character.
    async fn glob(&self, pattern: &str) -> Result<Vec<String>>;

    /// Search for `pattern` (literal substring) across all files; returns
    /// `(path, line_number, line)` tuples. `line_number` is 1-based.
    async fn grep(&self, pattern: &str) -> Result<Vec<GrepHit>>;

    /// Whether a path exists.
    async fn exists(&self, path: &str) -> Result<bool>;
}

/// A single grep match.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GrepHit {
    /// Backend-relative path of the file.
    pub path: String,
    /// 1-based line number.
    pub line: u64,
    /// Full line text (without trailing newline).
    pub text: String,
}
