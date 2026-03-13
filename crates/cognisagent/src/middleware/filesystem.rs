//! Filesystem middleware — provides file read/write/list/glob/grep tools to agents.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use cognis_core::error::Result as CoreResult;
use cognis_core::tools::base::BaseTool;
use cognis_core::tools::types::{ToolInput, ToolOutput};

use crate::middleware::{AgentState, Middleware, Result};

/// Middleware that provides filesystem tools (read, write, list, glob, grep).
pub struct FilesystemMiddleware {
    /// The base working directory for filesystem operations.
    pub working_dir: PathBuf,
}

impl FilesystemMiddleware {
    /// Create a new `FilesystemMiddleware` rooted at the given directory.
    pub fn new(working_dir: impl Into<PathBuf>) -> Self {
        Self {
            working_dir: working_dir.into(),
        }
    }

    /// Return the set of tools this middleware provides.
    pub fn tools(&self) -> Vec<Arc<dyn BaseTool>> {
        let wd = self.working_dir.clone();
        vec![
            Arc::new(ReadFileTool {
                working_dir: wd.clone(),
            }),
            Arc::new(WriteFileTool {
                working_dir: wd.clone(),
            }),
            Arc::new(ListDirTool {
                working_dir: wd.clone(),
            }),
            Arc::new(GlobTool {
                working_dir: wd.clone(),
            }),
            Arc::new(GrepTool { working_dir: wd }),
        ]
    }
}

#[async_trait]
impl Middleware for FilesystemMiddleware {
    fn name(&self) -> &str {
        "filesystem"
    }

    async fn before_tool(&self, _state: &mut AgentState, _tool_name: &str) -> Result<()> {
        // Could add logging or permission checks here.
        Ok(())
    }

    async fn after_tool(
        &self,
        _state: &mut AgentState,
        _tool_name: &str,
        _result: &str,
    ) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tool implementations
// ---------------------------------------------------------------------------

/// Tool that reads a file and returns its contents.
pub struct ReadFileTool {
    working_dir: PathBuf,
}

#[async_trait]
impl BaseTool for ReadFileTool {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a file at the given path"
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Relative or absolute path to the file" }
            },
            "required": ["path"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> CoreResult<ToolOutput> {
        let path = extract_string_arg(&input, "path")?;
        let full = resolve_path(&self.working_dir, &path);
        let content = tokio::fs::read_to_string(&full)
            .await
            .map_err(|e| cognis_core::error::CognisError::ToolException(e.to_string()))?;
        Ok(ToolOutput::Content(Value::String(content)))
    }
}

/// Tool that writes content to a file.
pub struct WriteFileTool {
    working_dir: PathBuf,
}

#[async_trait]
impl BaseTool for WriteFileTool {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write content to a file at the given path"
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Path to write to" },
                "content": { "type": "string", "description": "Content to write" }
            },
            "required": ["path", "content"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> CoreResult<ToolOutput> {
        let path = extract_string_arg(&input, "path")?;
        let content = extract_string_arg(&input, "content")?;
        let full = resolve_path(&self.working_dir, &path);

        if let Some(parent) = full.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| cognis_core::error::CognisError::ToolException(e.to_string()))?;
        }

        tokio::fs::write(&full, &content)
            .await
            .map_err(|e| cognis_core::error::CognisError::ToolException(e.to_string()))?;
        Ok(ToolOutput::Content(Value::String(format!(
            "Wrote {} bytes to {}",
            content.len(),
            full.display()
        ))))
    }
}

/// Tool that lists the entries of a directory.
pub struct ListDirTool {
    working_dir: PathBuf,
}

#[async_trait]
impl BaseTool for ListDirTool {
    fn name(&self) -> &str {
        "list_dir"
    }

    fn description(&self) -> &str {
        "List files and directories in the given path"
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "path": { "type": "string", "description": "Directory path to list (defaults to working dir)" }
            }
        }))
    }

    async fn _run(&self, input: ToolInput) -> CoreResult<ToolOutput> {
        let path = extract_string_arg(&input, "path").unwrap_or_else(|_| ".".to_string());
        let full = resolve_path(&self.working_dir, &path);

        let mut entries = Vec::new();
        let mut dir = tokio::fs::read_dir(&full)
            .await
            .map_err(|e| cognis_core::error::CognisError::ToolException(e.to_string()))?;

        while let Some(entry) = dir
            .next_entry()
            .await
            .map_err(|e| cognis_core::error::CognisError::ToolException(e.to_string()))?
        {
            let name = entry.file_name().to_string_lossy().to_string();
            let ft = entry.file_type().await.ok();
            let suffix = if ft.map(|t| t.is_dir()).unwrap_or(false) {
                "/"
            } else {
                ""
            };
            entries.push(format!("{name}{suffix}"));
        }

        entries.sort();
        Ok(ToolOutput::Content(Value::String(entries.join("\n"))))
    }
}

/// Tool that performs glob pattern matching on files.
pub struct GlobTool {
    working_dir: PathBuf,
}

#[async_trait]
impl BaseTool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }

    fn description(&self) -> &str {
        "Find files matching a glob pattern"
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Glob pattern (e.g. **/*.rs)" }
            },
            "required": ["pattern"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> CoreResult<ToolOutput> {
        let pattern = extract_string_arg(&input, "pattern")?;
        let full_pattern = self.working_dir.join(&pattern);
        let pattern_str = full_pattern.to_string_lossy().to_string();

        let matches: Vec<String> = glob::glob(&pattern_str)
            .map_err(|e| cognis_core::error::CognisError::ToolException(e.to_string()))?
            .filter_map(|r| r.ok())
            .map(|p| p.to_string_lossy().to_string())
            .collect();

        Ok(ToolOutput::Content(Value::String(matches.join("\n"))))
    }
}

/// Tool that searches file contents for a pattern.
pub struct GrepTool {
    working_dir: PathBuf,
}

#[async_trait]
impl BaseTool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for a text pattern in files"
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "pattern": { "type": "string", "description": "Text pattern to search for" },
                "path": { "type": "string", "description": "File or directory to search in" }
            },
            "required": ["pattern"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> CoreResult<ToolOutput> {
        let pattern = extract_string_arg(&input, "pattern")?;
        let path = extract_string_arg(&input, "path").unwrap_or_else(|_| ".".to_string());
        let full = resolve_path(&self.working_dir, &path);

        let output = tokio::process::Command::new("grep")
            .args(["-rn", &pattern, &full.to_string_lossy()])
            .output()
            .await
            .map_err(|e| cognis_core::error::CognisError::ToolException(e.to_string()))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        Ok(ToolOutput::Content(Value::String(stdout)))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn resolve_path(working_dir: &Path, path: &str) -> PathBuf {
    let p = PathBuf::from(path);
    if p.is_absolute() {
        p
    } else {
        working_dir.join(p)
    }
}

fn extract_string_arg(input: &ToolInput, key: &str) -> CoreResult<String> {
    match input {
        ToolInput::Text(s) => Ok(s.clone()),
        ToolInput::Structured(map) => map
            .get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                cognis_core::error::CognisError::ToolException(format!(
                    "Missing required argument: {key}"
                ))
            }),
        ToolInput::ToolCall(tc) => tc
            .args
            .get(key)
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                cognis_core::error::CognisError::ToolException(format!(
                    "Missing required argument: {key}"
                ))
            }),
    }
}
