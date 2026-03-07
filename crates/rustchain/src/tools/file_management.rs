//! File management tools for agent filesystem interaction.
//!
//! Provides tools for reading, writing, listing, searching, moving, and
//! inspecting files within a sandboxed root directory. All operations validate
//! that resolved paths remain under the configured `root_dir` to prevent path
//! traversal attacks.

use async_trait::async_trait;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::tools::base::{BaseTool, BaseToolkit};
use rustchain_core::tools::types::{ToolInput, ToolOutput};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

/// Configuration shared by all file management tools.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSystemConfig {
    /// Base directory for all operations. All paths are resolved relative to this.
    pub root_dir: PathBuf,
    /// Optional whitelist of allowed file extensions (e.g. `["rs", "toml"]`).
    /// When `None`, all extensions are allowed.
    pub allowed_extensions: Option<Vec<String>>,
    /// Maximum file size in bytes for read/write operations (default: 1 MB).
    #[serde(default = "default_max_file_size")]
    pub max_file_size: usize,
    /// When `true`, write and move operations are rejected.
    #[serde(default)]
    pub read_only: bool,
}

fn default_max_file_size() -> usize {
    1_048_576 // 1 MB
}

impl Default for FileSystemConfig {
    fn default() -> Self {
        Self {
            root_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            allowed_extensions: None,
            max_file_size: default_max_file_size(),
            read_only: false,
        }
    }
}

impl FileSystemConfig {
    /// Create a new config with the given root directory.
    pub fn new(root_dir: impl Into<PathBuf>) -> Self {
        Self {
            root_dir: root_dir.into(),
            ..Default::default()
        }
    }

    /// Resolve and validate a path, ensuring it stays within `root_dir`.
    ///
    /// Returns the canonicalized absolute path on success, or an error if the
    /// path escapes the root directory.
    fn resolve_path(&self, path: &str) -> Result<PathBuf> {
        let candidate = if Path::new(path).is_absolute() {
            PathBuf::from(path)
        } else {
            self.root_dir.join(path)
        };

        // We need to canonicalize to resolve `..` segments.
        // For paths that don't exist yet, we canonicalize the parent.
        let resolved = if candidate.exists() {
            candidate.canonicalize().map_err(|e| {
                RustChainError::ToolException(format!(
                    "Failed to resolve path '{}': {e}",
                    candidate.display()
                ))
            })?
        } else {
            let parent = candidate.parent().ok_or_else(|| {
                RustChainError::ToolException(format!(
                    "Invalid path: no parent for '{}'",
                    candidate.display()
                ))
            })?;
            if !parent.exists() {
                // Walk up until we find an existing ancestor to canonicalize.
                let mut ancestor = parent.to_path_buf();
                while !ancestor.exists() {
                    ancestor = ancestor
                        .parent()
                        .ok_or_else(|| {
                            RustChainError::ToolException(format!(
                                "No existing ancestor for '{}'",
                                candidate.display()
                            ))
                        })?
                        .to_path_buf();
                }
                let canonical_ancestor = ancestor.canonicalize().map_err(|e| {
                    RustChainError::ToolException(format!(
                        "Failed to resolve ancestor '{}': {e}",
                        ancestor.display()
                    ))
                })?;
                let remainder = candidate
                    .strip_prefix(&ancestor)
                    .unwrap_or(candidate.as_path());
                // Ensure no `..` in the remainder
                for component in remainder.components() {
                    if matches!(component, std::path::Component::ParentDir) {
                        return Err(RustChainError::ToolException(
                            "Path traversal detected: '..' in path".into(),
                        ));
                    }
                }
                canonical_ancestor.join(remainder)
            } else {
                let canonical_parent = parent.canonicalize().map_err(|e| {
                    RustChainError::ToolException(format!(
                        "Failed to resolve parent '{}': {e}",
                        parent.display()
                    ))
                })?;
                let file_name = candidate.file_name().ok_or_else(|| {
                    RustChainError::ToolException("Path has no file name".into())
                })?;
                canonical_parent.join(file_name)
            }
        };

        let canonical_root = self.root_dir.canonicalize().map_err(|e| {
            RustChainError::ToolException(format!(
                "Failed to resolve root_dir '{}': {e}",
                self.root_dir.display()
            ))
        })?;

        if !resolved.starts_with(&canonical_root) {
            return Err(RustChainError::ToolException(format!(
                "Path '{}' is outside the root directory '{}'",
                resolved.display(),
                canonical_root.display()
            )));
        }

        Ok(resolved)
    }

    /// Check if a file extension is allowed by the config.
    fn check_extension(&self, path: &Path) -> Result<()> {
        if let Some(ref allowed) = self.allowed_extensions {
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !allowed.iter().any(|a| a == ext) {
                return Err(RustChainError::ToolException(format!(
                    "File extension '.{ext}' is not in the allowed list: {allowed:?}"
                )));
            }
        }
        Ok(())
    }

    /// Assert that the config is not read-only.
    fn check_writable(&self) -> Result<()> {
        if self.read_only {
            return Err(RustChainError::ToolException(
                "Filesystem is configured as read-only".into(),
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helper: extract a string field from ToolInput
// ---------------------------------------------------------------------------

fn extract_string(input: &ToolInput, field: &str) -> Result<String> {
    match input {
        ToolInput::Text(s) => Ok(s.clone()),
        ToolInput::Structured(map) => {
            if let Some(Value::String(v)) = map.get(field) {
                Ok(v.clone())
            } else {
                Err(RustChainError::ToolValidationError(format!(
                    "Missing required field '{field}'"
                )))
            }
        }
        ToolInput::ToolCall(tc) => {
            if let Some(Value::String(v)) = tc.args.get(field) {
                Ok(v.clone())
            } else {
                Err(RustChainError::ToolValidationError(format!(
                    "Missing required field '{field}'"
                )))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ReadFileTool
// ---------------------------------------------------------------------------

/// Reads the content of a file within the configured root directory.
pub struct ReadFileTool {
    /// Shared filesystem configuration.
    pub config: FileSystemConfig,
}

impl ReadFileTool {
    /// Create a new `ReadFileTool` with the given configuration.
    pub fn new(config: FileSystemConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl BaseTool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Input is the relative path to the file."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Relative path to the file" }
            },
            "required": ["path"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let path_str = extract_string(&input, "path")?;
        let resolved = self.config.resolve_path(&path_str)?;
        self.config.check_extension(&resolved)?;

        let metadata = std::fs::metadata(&resolved).map_err(|e| {
            RustChainError::ToolException(format!(
                "Cannot read '{}': {e}",
                resolved.display()
            ))
        })?;

        if metadata.len() as usize > self.config.max_file_size {
            return Err(RustChainError::ToolException(format!(
                "File size {} bytes exceeds maximum allowed {} bytes",
                metadata.len(),
                self.config.max_file_size
            )));
        }

        let content = std::fs::read_to_string(&resolved).map_err(|e| {
            RustChainError::ToolException(format!(
                "Failed to read '{}': {e}",
                resolved.display()
            ))
        })?;

        Ok(ToolOutput::Content(Value::String(content)))
    }
}

// ---------------------------------------------------------------------------
// WriteFileTool
// ---------------------------------------------------------------------------

/// Writes content to a file within the configured root directory.
///
/// Creates parent directories automatically if they do not exist.
pub struct WriteFileTool {
    /// Shared filesystem configuration.
    pub config: FileSystemConfig,
}

impl WriteFileTool {
    /// Create a new `WriteFileTool` with the given configuration.
    pub fn new(config: FileSystemConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl BaseTool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates parent directories if needed."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Relative path to the file" },
                "content": { "type": "string", "description": "Content to write" }
            },
            "required": ["path", "content"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        self.config.check_writable()?;
        let path_str = extract_string(&input, "path")?;
        let content = extract_string(&input, "content")?;

        if content.len() > self.config.max_file_size {
            return Err(RustChainError::ToolException(format!(
                "Content size {} bytes exceeds maximum allowed {} bytes",
                content.len(),
                self.config.max_file_size
            )));
        }

        let resolved = self.config.resolve_path(&path_str)?;
        self.config.check_extension(&resolved)?;

        if let Some(parent) = resolved.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                RustChainError::ToolException(format!(
                    "Failed to create directories for '{}': {e}",
                    resolved.display()
                ))
            })?;
        }

        std::fs::write(&resolved, &content).map_err(|e| {
            RustChainError::ToolException(format!(
                "Failed to write '{}': {e}",
                resolved.display()
            ))
        })?;

        Ok(ToolOutput::Content(Value::String(format!(
            "Successfully wrote {} bytes to '{}'",
            content.len(),
            path_str
        ))))
    }
}

// ---------------------------------------------------------------------------
// ListDirectoryTool
// ---------------------------------------------------------------------------

/// Lists files and directories at a given path with file sizes.
pub struct ListDirectoryTool {
    /// Shared filesystem configuration.
    pub config: FileSystemConfig,
}

impl ListDirectoryTool {
    /// Create a new `ListDirectoryTool` with the given configuration.
    pub fn new(config: FileSystemConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl BaseTool for ListDirectoryTool {
    fn name(&self) -> &str {
        "list_directory"
    }

    fn description(&self) -> &str {
        "List files and directories at a path. Returns names and sizes."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Relative directory path (default: root)" }
            }
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let path_str = extract_string(&input, "path").unwrap_or_else(|_| ".".to_string());
        let resolved = self.config.resolve_path(&path_str)?;

        if !resolved.is_dir() {
            return Err(RustChainError::ToolException(format!(
                "'{}' is not a directory",
                resolved.display()
            )));
        }

        let mut entries = Vec::new();
        let read_dir = std::fs::read_dir(&resolved).map_err(|e| {
            RustChainError::ToolException(format!(
                "Failed to list '{}': {e}",
                resolved.display()
            ))
        })?;

        for entry in read_dir {
            let entry = entry.map_err(|e| {
                RustChainError::ToolException(format!("Error reading directory entry: {e}"))
            })?;
            let meta = entry.metadata().map_err(|e| {
                RustChainError::ToolException(format!(
                    "Error reading metadata for '{}': {e}",
                    entry.path().display()
                ))
            })?;
            let name = entry.file_name().to_string_lossy().to_string();
            let kind = if meta.is_dir() { "dir" } else { "file" };
            let size = if meta.is_file() { meta.len() } else { 0 };
            entries.push(format!("{name} [{kind}, {size} bytes]"));
        }

        entries.sort();
        let output = if entries.is_empty() {
            "Directory is empty".to_string()
        } else {
            entries.join("\n")
        };

        Ok(ToolOutput::Content(Value::String(output)))
    }
}

// ---------------------------------------------------------------------------
// SearchFilesTool
// ---------------------------------------------------------------------------

/// Searches for files matching a glob pattern within the root directory.
pub struct SearchFilesTool {
    /// Shared filesystem configuration.
    pub config: FileSystemConfig,
}

impl SearchFilesTool {
    /// Create a new `SearchFilesTool` with the given configuration.
    pub fn new(config: FileSystemConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl BaseTool for SearchFilesTool {
    fn name(&self) -> &str {
        "search_files"
    }

    fn description(&self) -> &str {
        "Search for files matching a glob pattern under the root directory."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Glob pattern (e.g. '*.rs', '**/*.toml')" }
            },
            "required": ["pattern"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let pattern_str = extract_string(&input, "pattern")?;

        let full_pattern = self.config.root_dir.join(&pattern_str);
        let full_pattern_str = full_pattern.to_string_lossy().to_string();

        let mut matches = Vec::new();
        let canonical_root = self.config.root_dir.canonicalize().map_err(|e| {
            RustChainError::ToolException(format!("Failed to resolve root_dir: {e}"))
        })?;

        for entry in glob::glob(&full_pattern_str).map_err(|e| {
            RustChainError::ToolException(format!("Invalid glob pattern: {e}"))
        })? {
            match entry {
                Ok(path) => {
                    let canonical = path.canonicalize().unwrap_or(path.clone());
                    if canonical.starts_with(&canonical_root) {
                        let relative = canonical
                            .strip_prefix(&canonical_root)
                            .unwrap_or(&canonical);
                        matches.push(relative.display().to_string());
                    }
                }
                Err(e) => {
                    // Skip unreadable entries
                    eprintln!("Glob error: {e}");
                }
            }
        }

        matches.sort();
        let output = if matches.is_empty() {
            format!("No files matching pattern '{pattern_str}'")
        } else {
            matches.join("\n")
        };

        Ok(ToolOutput::Content(Value::String(output)))
    }
}

// ---------------------------------------------------------------------------
// MoveFileTool
// ---------------------------------------------------------------------------

/// Moves or renames a file within the root directory.
pub struct MoveFileTool {
    /// Shared filesystem configuration.
    pub config: FileSystemConfig,
}

impl MoveFileTool {
    /// Create a new `MoveFileTool` with the given configuration.
    pub fn new(config: FileSystemConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl BaseTool for MoveFileTool {
    fn name(&self) -> &str {
        "move_file"
    }

    fn description(&self) -> &str {
        "Move or rename a file. Both source and destination must be within the root directory."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "source": { "type": "string", "description": "Source file path" },
                "destination": { "type": "string", "description": "Destination file path" }
            },
            "required": ["source", "destination"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        self.config.check_writable()?;
        let source_str = extract_string(&input, "source")?;
        let dest_str = extract_string(&input, "destination")?;

        let source = self.config.resolve_path(&source_str)?;
        let dest = self.config.resolve_path(&dest_str)?;

        if !source.exists() {
            return Err(RustChainError::ToolException(format!(
                "Source '{}' does not exist",
                source_str
            )));
        }

        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| {
                RustChainError::ToolException(format!(
                    "Failed to create directories for destination: {e}"
                ))
            })?;
        }

        std::fs::rename(&source, &dest).map_err(|e| {
            RustChainError::ToolException(format!(
                "Failed to move '{}' to '{}': {e}",
                source_str, dest_str
            ))
        })?;

        Ok(ToolOutput::Content(Value::String(format!(
            "Successfully moved '{source_str}' to '{dest_str}'"
        ))))
    }
}

// ---------------------------------------------------------------------------
// FileInfoTool
// ---------------------------------------------------------------------------

/// Returns metadata about a file: size, modified time, and permissions.
pub struct FileInfoTool {
    /// Shared filesystem configuration.
    pub config: FileSystemConfig,
}

impl FileInfoTool {
    /// Create a new `FileInfoTool` with the given configuration.
    pub fn new(config: FileSystemConfig) -> Self {
        Self { config }
    }
}

#[async_trait]
impl BaseTool for FileInfoTool {
    fn name(&self) -> &str {
        "file_info"
    }

    fn description(&self) -> &str {
        "Get file metadata: size, modification time, and permissions."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Relative path to the file" }
            },
            "required": ["path"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let path_str = extract_string(&input, "path")?;
        let resolved = self.config.resolve_path(&path_str)?;

        let metadata = std::fs::metadata(&resolved).map_err(|e| {
            RustChainError::ToolException(format!(
                "Cannot stat '{}': {e}",
                resolved.display()
            ))
        })?;

        let size = metadata.len();
        let is_dir = metadata.is_dir();
        let is_file = metadata.is_file();
        let is_symlink = metadata.file_type().is_symlink();
        let readonly = metadata.permissions().readonly();

        let modified = metadata
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let info = json!({
            "path": path_str,
            "size": size,
            "is_file": is_file,
            "is_dir": is_dir,
            "is_symlink": is_symlink,
            "readonly": readonly,
            "modified_epoch": modified,
        });

        Ok(ToolOutput::Content(info))
    }
}

// ---------------------------------------------------------------------------
// FileToolkit
// ---------------------------------------------------------------------------

/// A toolkit containing all file management tools.
///
/// Use [`create_file_toolkit`] as a convenience constructor.
pub struct FileToolkit {
    config: FileSystemConfig,
}

impl FileToolkit {
    /// Create a new `FileToolkit` with the given configuration.
    pub fn new(config: FileSystemConfig) -> Self {
        Self { config }
    }
}

impl BaseToolkit for FileToolkit {
    fn get_tools(&self) -> Vec<Box<dyn BaseTool>> {
        vec![
            Box::new(ReadFileTool::new(self.config.clone())),
            Box::new(WriteFileTool::new(self.config.clone())),
            Box::new(ListDirectoryTool::new(self.config.clone())),
            Box::new(SearchFilesTool::new(self.config.clone())),
            Box::new(MoveFileTool::new(self.config.clone())),
            Box::new(FileInfoTool::new(self.config.clone())),
        ]
    }
}

/// Convenience function to create a [`FileToolkit`] with all file management tools.
pub fn create_file_toolkit(config: FileSystemConfig) -> FileToolkit {
    FileToolkit::new(config)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn make_config(dir: &TempDir) -> FileSystemConfig {
        FileSystemConfig::new(dir.path())
    }

    fn text_input(s: &str) -> ToolInput {
        ToolInput::Text(s.to_string())
    }

    fn structured_input(pairs: &[(&str, &str)]) -> ToolInput {
        let map: HashMap<String, Value> = pairs
            .iter()
            .map(|(k, v)| (k.to_string(), Value::String(v.to_string())))
            .collect();
        ToolInput::Structured(map)
    }

    // ---- ReadFileTool ----

    #[tokio::test]
    async fn test_read_file_basic() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("hello.txt"), "hello world").unwrap();
        let tool = ReadFileTool::new(make_config(&dir));
        let result = tool._run(text_input("hello.txt")).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => assert_eq!(s, "hello world"),
            other => panic!("Expected string content, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_read_file_structured_input() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("data.txt"), "some data").unwrap();
        let tool = ReadFileTool::new(make_config(&dir));
        let result = tool
            ._run(structured_input(&[("path", "data.txt")]))
            .await
            .unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => assert_eq!(s, "some data"),
            other => panic!("Expected string, got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_read_file_not_found() {
        let dir = TempDir::new().unwrap();
        let tool = ReadFileTool::new(make_config(&dir));
        let result = tool._run(text_input("nonexistent.txt")).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_read_file_too_large() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.max_file_size = 10;
        std::fs::write(dir.path().join("big.txt"), "this is more than 10 bytes").unwrap();
        let tool = ReadFileTool::new(config);
        let result = tool._run(text_input("big.txt")).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds maximum"));
    }

    #[tokio::test]
    async fn test_read_file_path_traversal() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("safe.txt"), "ok").unwrap();
        let tool = ReadFileTool::new(make_config(&dir));
        let result = tool._run(text_input("../../../etc/passwd")).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("outside the root"));
    }

    #[tokio::test]
    async fn test_read_file_extension_filter_allowed() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("code.rs"), "fn main(){}").unwrap();
        let mut config = make_config(&dir);
        config.allowed_extensions = Some(vec!["rs".to_string()]);
        let tool = ReadFileTool::new(config);
        let result = tool._run(text_input("code.rs")).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_read_file_extension_filter_blocked() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("data.txt"), "text").unwrap();
        let mut config = make_config(&dir);
        config.allowed_extensions = Some(vec!["rs".to_string()]);
        let tool = ReadFileTool::new(config);
        let result = tool._run(text_input("data.txt")).await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("not in the allowed list"));
    }

    // ---- WriteFileTool ----

    #[tokio::test]
    async fn test_write_file_basic() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool::new(make_config(&dir));
        let input = structured_input(&[("path", "output.txt"), ("content", "written content")]);
        let result = tool._run(input).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => assert!(s.contains("Successfully wrote")),
            other => panic!("Unexpected: {other:?}"),
        }
        let written = std::fs::read_to_string(dir.path().join("output.txt")).unwrap();
        assert_eq!(written, "written content");
    }

    #[tokio::test]
    async fn test_write_file_creates_parent_dirs() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool::new(make_config(&dir));
        let input = structured_input(&[("path", "sub/dir/file.txt"), ("content", "nested")]);
        let result = tool._run(input).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => assert!(s.contains("Successfully wrote")),
            other => panic!("Unexpected: {other:?}"),
        }
        let written = std::fs::read_to_string(dir.path().join("sub/dir/file.txt")).unwrap();
        assert_eq!(written, "nested");
    }

    #[tokio::test]
    async fn test_write_file_read_only() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.read_only = true;
        let tool = WriteFileTool::new(config);
        let input = structured_input(&[("path", "file.txt"), ("content", "data")]);
        let result = tool._run(input).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("read-only"));
    }

    #[tokio::test]
    async fn test_write_file_exceeds_max_size() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.max_file_size = 5;
        let tool = WriteFileTool::new(config);
        let input = structured_input(&[("path", "file.txt"), ("content", "too long content")]);
        let result = tool._run(input).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("exceeds maximum"));
    }

    #[tokio::test]
    async fn test_write_file_path_traversal() {
        let dir = TempDir::new().unwrap();
        let tool = WriteFileTool::new(make_config(&dir));
        let input = structured_input(&[("path", "../../evil.txt"), ("content", "bad")]);
        let result = tool._run(input).await;
        assert!(result.is_err());
    }

    // ---- ListDirectoryTool ----

    #[tokio::test]
    async fn test_list_directory_basic() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("a.txt"), "aaa").unwrap();
        std::fs::write(dir.path().join("b.txt"), "bb").unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();
        let tool = ListDirectoryTool::new(make_config(&dir));
        let result = tool._run(text_input(".")).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert!(s.contains("a.txt"));
                assert!(s.contains("b.txt"));
                assert!(s.contains("subdir"));
                assert!(s.contains("[dir"));
                assert!(s.contains("[file"));
            }
            other => panic!("Unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_list_directory_empty() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("empty");
        std::fs::create_dir(&sub).unwrap();
        let tool = ListDirectoryTool::new(make_config(&dir));
        let result = tool._run(text_input("empty")).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => assert_eq!(s, "Directory is empty"),
            other => panic!("Unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_list_directory_not_a_dir() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "data").unwrap();
        let tool = ListDirectoryTool::new(make_config(&dir));
        let result = tool._run(text_input("file.txt")).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a directory"));
    }

    // ---- SearchFilesTool ----

    #[tokio::test]
    async fn test_search_files_basic() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("main.rs"), "fn main(){}").unwrap();
        std::fs::write(dir.path().join("lib.rs"), "pub mod lib;").unwrap();
        std::fs::write(dir.path().join("data.txt"), "text").unwrap();
        let tool = SearchFilesTool::new(make_config(&dir));
        let result = tool
            ._run(structured_input(&[("pattern", "*.rs")]))
            .await
            .unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert!(s.contains("main.rs"));
                assert!(s.contains("lib.rs"));
                assert!(!s.contains("data.txt"));
            }
            other => panic!("Unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_search_files_no_matches() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "data").unwrap();
        let tool = SearchFilesTool::new(make_config(&dir));
        let result = tool
            ._run(structured_input(&[("pattern", "*.xyz")]))
            .await
            .unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => assert!(s.contains("No files matching")),
            other => panic!("Unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_search_files_recursive() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir_all(dir.path().join("sub")).unwrap();
        std::fs::write(dir.path().join("top.rs"), "").unwrap();
        std::fs::write(dir.path().join("sub/nested.rs"), "").unwrap();
        let tool = SearchFilesTool::new(make_config(&dir));
        let result = tool
            ._run(structured_input(&[("pattern", "**/*.rs")]))
            .await
            .unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => {
                assert!(s.contains("nested.rs"));
            }
            other => panic!("Unexpected: {other:?}"),
        }
    }

    // ---- MoveFileTool ----

    #[tokio::test]
    async fn test_move_file_basic() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("old.txt"), "content").unwrap();
        let tool = MoveFileTool::new(make_config(&dir));
        let input = structured_input(&[("source", "old.txt"), ("destination", "new.txt")]);
        let result = tool._run(input).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => assert!(s.contains("Successfully moved")),
            other => panic!("Unexpected: {other:?}"),
        }
        assert!(!dir.path().join("old.txt").exists());
        assert_eq!(
            std::fs::read_to_string(dir.path().join("new.txt")).unwrap(),
            "content"
        );
    }

    #[tokio::test]
    async fn test_move_file_read_only() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("file.txt"), "data").unwrap();
        let mut config = make_config(&dir);
        config.read_only = true;
        let tool = MoveFileTool::new(config);
        let input = structured_input(&[("source", "file.txt"), ("destination", "moved.txt")]);
        let result = tool._run(input).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("read-only"));
    }

    #[tokio::test]
    async fn test_move_file_source_not_found() {
        let dir = TempDir::new().unwrap();
        let tool = MoveFileTool::new(make_config(&dir));
        let input = structured_input(&[("source", "missing.txt"), ("destination", "dest.txt")]);
        let result = tool._run(input).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    // ---- FileInfoTool ----

    #[tokio::test]
    async fn test_file_info_basic() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("info.txt"), "12345").unwrap();
        let tool = FileInfoTool::new(make_config(&dir));
        let result = tool._run(text_input("info.txt")).await.unwrap();
        match result {
            ToolOutput::Content(val) => {
                assert_eq!(val["size"], 5);
                assert_eq!(val["is_file"], true);
                assert_eq!(val["is_dir"], false);
            }
            other => panic!("Unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_file_info_directory() {
        let dir = TempDir::new().unwrap();
        std::fs::create_dir(dir.path().join("subdir")).unwrap();
        let tool = FileInfoTool::new(make_config(&dir));
        let result = tool._run(text_input("subdir")).await.unwrap();
        match result {
            ToolOutput::Content(val) => {
                assert_eq!(val["is_dir"], true);
                assert_eq!(val["is_file"], false);
            }
            other => panic!("Unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_file_info_not_found() {
        let dir = TempDir::new().unwrap();
        let tool = FileInfoTool::new(make_config(&dir));
        let result = tool._run(text_input("ghost.txt")).await;
        assert!(result.is_err());
    }

    // ---- FileToolkit ----

    #[tokio::test]
    async fn test_toolkit_returns_all_tools() {
        let dir = TempDir::new().unwrap();
        let toolkit = create_file_toolkit(make_config(&dir));
        let tools = toolkit.get_tools();
        assert_eq!(tools.len(), 6);
        let names: Vec<&str> = tools.iter().map(|t| t.name()).collect();
        assert!(names.contains(&"read_file"));
        assert!(names.contains(&"write_file"));
        assert!(names.contains(&"list_directory"));
        assert!(names.contains(&"search_files"));
        assert!(names.contains(&"move_file"));
        assert!(names.contains(&"file_info"));
    }

    // ---- Tool metadata tests ----

    #[tokio::test]
    async fn test_tool_names_and_descriptions() {
        let dir = TempDir::new().unwrap();
        let config = make_config(&dir);
        let read = ReadFileTool::new(config.clone());
        assert_eq!(read.name(), "read_file");
        assert!(!read.description().is_empty());
        assert!(read.args_schema().is_some());

        let write = WriteFileTool::new(config.clone());
        assert_eq!(write.name(), "write_file");

        let list = ListDirectoryTool::new(config.clone());
        assert_eq!(list.name(), "list_directory");

        let search = SearchFilesTool::new(config.clone());
        assert_eq!(search.name(), "search_files");

        let mv = MoveFileTool::new(config.clone());
        assert_eq!(mv.name(), "move_file");

        let info = FileInfoTool::new(config);
        assert_eq!(info.name(), "file_info");
    }

    #[tokio::test]
    async fn test_config_default() {
        let config = FileSystemConfig::default();
        assert_eq!(config.max_file_size, 1_048_576);
        assert!(!config.read_only);
        assert!(config.allowed_extensions.is_none());
    }

    // ---- run_json integration ----

    #[tokio::test]
    async fn test_read_file_via_run_json() {
        let dir = TempDir::new().unwrap();
        std::fs::write(dir.path().join("test.txt"), "json test").unwrap();
        let tool = ReadFileTool::new(make_config(&dir));
        let input = json!({"path": "test.txt"});
        let result = tool.run_json(&input).await.unwrap();
        assert_eq!(result, Value::String("json test".to_string()));
    }

    #[tokio::test]
    async fn test_write_file_extension_filter() {
        let dir = TempDir::new().unwrap();
        let mut config = make_config(&dir);
        config.allowed_extensions = Some(vec!["rs".to_string()]);
        let tool = WriteFileTool::new(config);

        // Allowed
        let input = structured_input(&[("path", "code.rs"), ("content", "fn main(){}")]);
        assert!(tool._run(input).await.is_ok());

        // Blocked
        let input = structured_input(&[("path", "data.txt"), ("content", "text")]);
        let err = tool._run(input).await.unwrap_err();
        assert!(err.to_string().contains("not in the allowed list"));
    }
}
