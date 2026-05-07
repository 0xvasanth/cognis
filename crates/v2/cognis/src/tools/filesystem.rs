//! Filesystem tools backed by [`Backend`](crate::Backend).
//!
//! Each tool is a thin wrapper over one method on a shared
//! `Arc<dyn Backend>` — the agent's "workspace". The backend decides
//! whether operations hit memory, a sandboxed real FS, or a remote store.
//!
//! [`register_filesystem_tools`] registers all six in one shot.

use std::sync::Arc;

use async_trait::async_trait;
use cognis2_core::schemars::{self, JsonSchema};
use serde::Deserialize;

use cognis2_core::{CognisError, Result};
use cognis2_llm::tools::{Tool, ToolInput, ToolOutput, ToolRegistry};

use crate::backend::Backend;

// ---------------------------------------------------------------------------
// fs_read
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct FileReadInput {
    /// Backend-relative path of the file to read.
    path: String,
}

/// Read the full contents of a file from the agent's backend.
pub struct FileReadTool {
    backend: Arc<dyn Backend>,
}

impl FileReadTool {
    /// Wrap a backend.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Tool for FileReadTool {
    fn name(&self) -> &str {
        "fs_read"
    }
    fn description(&self) -> &str {
        "Read the full contents of a file from the workspace."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(FileReadInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: FileReadInput = serde_json::from_value(input.into_json())
            .map_err(|e| CognisError::ToolValidationError(format!("fs_read: {e}")))?;
        let s = self.backend.read(&parsed.path).await?;
        Ok(ToolOutput::Text(s))
    }
}

// ---------------------------------------------------------------------------
// fs_write
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct FileWriteInput {
    /// Backend-relative path to write.
    path: String,
    /// Full file contents (overwrites any existing file).
    contents: String,
}

/// Overwrite a file with the supplied contents.
pub struct FileWriteTool {
    backend: Arc<dyn Backend>,
}

impl FileWriteTool {
    /// Wrap a backend.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Tool for FileWriteTool {
    fn name(&self) -> &str {
        "fs_write"
    }
    fn description(&self) -> &str {
        "Overwrite (or create) a file with the supplied contents."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(FileWriteInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: FileWriteInput = serde_json::from_value(input.into_json())
            .map_err(|e| CognisError::ToolValidationError(format!("fs_write: {e}")))?;
        self.backend.write(&parsed.path, &parsed.contents).await?;
        Ok(ToolOutput::Empty)
    }
}

// ---------------------------------------------------------------------------
// fs_edit
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct FileEditInput {
    /// File to edit.
    path: String,
    /// Literal string to find.
    find: String,
    /// Replacement.
    replace: String,
    /// Maximum number of occurrences `find` is allowed to match. Edit is
    /// refused if the actual count exceeds this — guards against accidental
    /// global replacement.
    #[serde(default = "default_max_occurrences")]
    max_occurrences: usize,
}

fn default_max_occurrences() -> usize {
    1
}

/// Literal find-and-replace within a file.
pub struct FileEditTool {
    backend: Arc<dyn Backend>,
}

impl FileEditTool {
    /// Wrap a backend.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Tool for FileEditTool {
    fn name(&self) -> &str {
        "fs_edit"
    }
    fn description(&self) -> &str {
        "Replace a literal string in a file. Refuses if `find` occurs more \
         times than `max_occurrences` (default 1)."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(FileEditInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: FileEditInput = serde_json::from_value(input.into_json())
            .map_err(|e| CognisError::ToolValidationError(format!("fs_edit: {e}")))?;
        let n = self
            .backend
            .edit(
                &parsed.path,
                &parsed.find,
                &parsed.replace,
                parsed.max_occurrences,
            )
            .await?;
        Ok(ToolOutput::Text(format!("replaced {n} occurrence(s)")))
    }
}

// ---------------------------------------------------------------------------
// fs_ls
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct FileListInput {
    /// Directory to list (relative to backend root). Use `"."` for root.
    #[serde(default = "default_dir")]
    dir: String,
}

fn default_dir() -> String {
    ".".into()
}

/// List files (non-recursive) under a directory.
pub struct FileListTool {
    backend: Arc<dyn Backend>,
}

impl FileListTool {
    /// Wrap a backend.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Tool for FileListTool {
    fn name(&self) -> &str {
        "fs_ls"
    }
    fn description(&self) -> &str {
        "List files (non-recursive) under a directory in the workspace."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(FileListInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: FileListInput = serde_json::from_value(input.into_json())
            .map_err(|e| CognisError::ToolValidationError(format!("fs_ls: {e}")))?;
        let names = self.backend.ls(&parsed.dir).await?;
        Ok(ToolOutput::Content(serde_json::json!(names)))
    }
}

// ---------------------------------------------------------------------------
// fs_glob
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct FileGlobInput {
    /// Glob pattern (e.g. `**/*.rs`). Supports `*`, `**`, `?`.
    pattern: String,
}

/// Find files matching a glob pattern.
pub struct FileGlobTool {
    backend: Arc<dyn Backend>,
}

impl FileGlobTool {
    /// Wrap a backend.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Tool for FileGlobTool {
    fn name(&self) -> &str {
        "fs_glob"
    }
    fn description(&self) -> &str {
        "Find files in the workspace matching a glob pattern."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(FileGlobInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: FileGlobInput = serde_json::from_value(input.into_json())
            .map_err(|e| CognisError::ToolValidationError(format!("fs_glob: {e}")))?;
        let names = self.backend.glob(&parsed.pattern).await?;
        Ok(ToolOutput::Content(serde_json::json!(names)))
    }
}

// ---------------------------------------------------------------------------
// fs_grep
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct FileGrepInput {
    /// Literal substring to search for.
    pattern: String,
}

/// Search for a literal substring across all files in the workspace.
pub struct FileGrepTool {
    backend: Arc<dyn Backend>,
}

impl FileGrepTool {
    /// Wrap a backend.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Tool for FileGrepTool {
    fn name(&self) -> &str {
        "fs_grep"
    }
    fn description(&self) -> &str {
        "Search for a literal substring across all files in the workspace. \
         Returns `[{path, line, text}]`."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(FileGrepInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: FileGrepInput = serde_json::from_value(input.into_json())
            .map_err(|e| CognisError::ToolValidationError(format!("fs_grep: {e}")))?;
        let hits = self.backend.grep(&parsed.pattern).await?;
        let val: Vec<_> = hits
            .into_iter()
            .map(|h| serde_json::json!({"path": h.path, "line": h.line, "text": h.text}))
            .collect();
        Ok(ToolOutput::Content(serde_json::json!(val)))
    }
}

// ---------------------------------------------------------------------------
// fs_exists
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, JsonSchema)]
struct FileExistsInput {
    /// Path to check.
    path: String,
}

/// Check whether a file exists in the workspace.
pub struct FileExistsTool {
    backend: Arc<dyn Backend>,
}

impl FileExistsTool {
    /// Wrap a backend.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }
}

#[async_trait]
impl Tool for FileExistsTool {
    fn name(&self) -> &str {
        "fs_exists"
    }
    fn description(&self) -> &str {
        "Return whether a file exists in the workspace."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(FileExistsInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: FileExistsInput = serde_json::from_value(input.into_json())
            .map_err(|e| CognisError::ToolValidationError(format!("fs_exists: {e}")))?;
        let exists = self.backend.exists(&parsed.path).await?;
        Ok(ToolOutput::Content(serde_json::json!(exists)))
    }
}

// ---------------------------------------------------------------------------
// One-shot registration helper
// ---------------------------------------------------------------------------

/// Register all six filesystem tools (`fs_read`, `fs_write`, `fs_edit`,
/// `fs_ls`, `fs_glob`, `fs_grep`, `fs_exists`) against a single shared
/// backend.
pub fn register_filesystem_tools(reg: &mut ToolRegistry, backend: Arc<dyn Backend>) {
    reg.register(Arc::new(FileReadTool::new(backend.clone())));
    reg.register(Arc::new(FileWriteTool::new(backend.clone())));
    reg.register(Arc::new(FileEditTool::new(backend.clone())));
    reg.register(Arc::new(FileListTool::new(backend.clone())));
    reg.register(Arc::new(FileGlobTool::new(backend.clone())));
    reg.register(Arc::new(FileGrepTool::new(backend.clone())));
    reg.register(Arc::new(FileExistsTool::new(backend)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::MemoryBackend;
    use serde_json::json;

    fn backend() -> Arc<dyn Backend> {
        Arc::new(MemoryBackend::new().with_files([("a.txt", "hello world"), ("sub/b.txt", "x")]))
    }

    #[tokio::test]
    async fn read_returns_contents() {
        let t = FileReadTool::new(backend());
        let mut a = std::collections::HashMap::new();
        a.insert("path".into(), json!("a.txt"));
        let out = t._run(ToolInput::Structured(a)).await.unwrap();
        assert_eq!(out.as_string(), "hello world");
    }

    #[tokio::test]
    async fn write_then_read_roundtrip() {
        let b = backend();
        let mut a = std::collections::HashMap::new();
        a.insert("path".into(), json!("c.txt"));
        a.insert("contents".into(), json!("new file"));
        FileWriteTool::new(b.clone())
            ._run(ToolInput::Structured(a))
            .await
            .unwrap();
        assert_eq!(b.read("c.txt").await.unwrap(), "new file");
    }

    #[tokio::test]
    async fn edit_replaces_unique() {
        let b = backend();
        let mut a = std::collections::HashMap::new();
        a.insert("path".into(), json!("a.txt"));
        a.insert("find".into(), json!("world"));
        a.insert("replace".into(), json!("rust"));
        FileEditTool::new(b.clone())
            ._run(ToolInput::Structured(a))
            .await
            .unwrap();
        assert_eq!(b.read("a.txt").await.unwrap(), "hello rust");
    }

    #[tokio::test]
    async fn glob_returns_paths() {
        let b = backend();
        let mut a = std::collections::HashMap::new();
        a.insert("pattern".into(), json!("**/*.txt"));
        let out = FileGlobTool::new(b)
            ._run(ToolInput::Structured(a))
            .await
            .unwrap();
        let v: Vec<String> = serde_json::from_value(match out {
            ToolOutput::Content(v) => v,
            _ => panic!(),
        })
        .unwrap();
        let mut sorted = v;
        sorted.sort();
        assert_eq!(sorted, vec!["a.txt", "sub/b.txt"]);
    }

    #[tokio::test]
    async fn grep_returns_hits() {
        let b = backend();
        let mut a = std::collections::HashMap::new();
        a.insert("pattern".into(), json!("hello"));
        let out = FileGrepTool::new(b)
            ._run(ToolInput::Structured(a))
            .await
            .unwrap();
        let v: serde_json::Value = match out {
            ToolOutput::Content(v) => v,
            _ => panic!(),
        };
        assert_eq!(v[0]["path"], "a.txt");
    }

    #[tokio::test]
    async fn register_helper_adds_all_tools() {
        let mut reg = ToolRegistry::new();
        register_filesystem_tools(&mut reg, backend());
        for name in [
            "fs_read",
            "fs_write",
            "fs_edit",
            "fs_ls",
            "fs_glob",
            "fs_grep",
            "fs_exists",
        ] {
            assert!(reg.contains(name), "missing {name}");
        }
    }
}
