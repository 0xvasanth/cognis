//! Filesystem file search middleware — glob and grep search tools.
//!
//! Provides file search capabilities within a constrained root directory,
//! with path traversal protection.

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use regex::Regex;
use serde::{Deserialize, Serialize};

use rustchain_core::error::{Result, RustChainError};

use super::types::AgentMiddleware;

/// Output mode for grep searches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrepOutputMode {
    /// Return matching lines with content.
    Content,
    /// Return only file paths that contain matches.
    FilesWithMatches,
}

/// Configuration for the filesystem file search middleware.
pub struct FilesystemFileSearchMiddleware {
    /// Root path that all searches are constrained to.
    pub root_path: PathBuf,
    /// Whether to use ripgrep (rg) for grep searches (falls back to internal grep).
    pub use_ripgrep: bool,
    /// Maximum file size in bytes to include in search results.
    pub max_file_size: usize,
}

impl FilesystemFileSearchMiddleware {
    pub fn new(root_path: impl Into<PathBuf>) -> Self {
        Self {
            root_path: root_path.into(),
            use_ripgrep: false,
            max_file_size: 10 * 1024 * 1024, // 10 MB
        }
    }

    pub fn with_ripgrep(mut self, use_ripgrep: bool) -> Self {
        self.use_ripgrep = use_ripgrep;
        self
    }

    pub fn with_max_file_size(mut self, max_size: usize) -> Self {
        self.max_file_size = max_size;
        self
    }

    /// Validate that a path is within the root directory (no traversal).
    pub fn validate_path(&self, path: &Path) -> Result<PathBuf> {
        // Resolve the path relative to root
        let resolved = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.root_path.join(path)
        };

        // Canonicalize would follow symlinks; for safety, normalize manually
        let normalized = normalize_path(&resolved);
        let root_normalized = normalize_path(&self.root_path);

        if !normalized.starts_with(&root_normalized) {
            return Err(RustChainError::Other(format!(
                "Path traversal detected: '{}' is outside root '{}'",
                path.display(),
                self.root_path.display()
            )));
        }

        Ok(normalized)
    }

    /// Search for files matching a glob pattern within the root directory.
    ///
    /// Supports `**` patterns for recursive matching.
    /// An optional `path` parameter restricts the search to a subdirectory
    /// relative to the root.
    ///
    /// Returns a list of matching file paths relative to the root.
    pub fn glob_search(&self, pattern: &str) -> Result<Vec<String>> {
        self.glob_search_in(pattern, None)
    }

    /// Search for files matching a glob pattern within a subdirectory of root.
    pub fn glob_search_in(&self, pattern: &str, path: Option<&str>) -> Result<Vec<String>> {
        // Validate the pattern doesn't contain traversal
        if pattern.contains("..") {
            return Err(RustChainError::Other(
                "Glob pattern must not contain '..'".into(),
            ));
        }

        let search_root = if let Some(p) = path {
            let sub = self.validate_path(Path::new(p))?;
            sub
        } else {
            self.root_path.clone()
        };

        let mut results = Vec::new();
        self.glob_walk_directory(&search_root, pattern, &mut results)?;

        Ok(results)
    }

    /// Recursively walk a directory and collect files matching a glob pattern.
    fn glob_walk_directory(
        &self,
        dir: &Path,
        pattern: &str,
        results: &mut Vec<String>,
    ) -> Result<()> {
        let entries = std::fs::read_dir(dir).map_err(|e| {
            RustChainError::Other(format!(
                "Failed to read directory '{}': {}",
                dir.display(),
                e
            ))
        })?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip hidden directories
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if !name.starts_with('.') {
                        self.glob_walk_directory(&path, pattern, results)?;
                    }
                }
            } else if path.is_file() {
                // Check file size
                if let Ok(metadata) = path.metadata() {
                    if metadata.len() > self.max_file_size as u64 {
                        continue;
                    }
                }

                if let Ok(rel) = path.strip_prefix(&self.root_path) {
                    let rel_str = rel.to_string_lossy().to_string();
                    if matches_glob_pattern(&rel_str, pattern) {
                        results.push(rel_str);
                    }
                }
            }
        }

        Ok(())
    }

    /// Search for a regex pattern within files in the root directory.
    ///
    /// Returns matches as (file_path, line_number, line_content) tuples.
    pub fn grep_search(&self, pattern: &str, file_pattern: Option<&str>) -> Result<Vec<GrepMatch>> {
        self.grep_search_in(pattern, file_pattern, None, GrepOutputMode::Content)
    }

    /// Search for a regex pattern within files, with optional subdirectory and output mode.
    ///
    /// - `path`: optional subdirectory relative to root to restrict the search.
    /// - `output_mode`: `Content` returns full matches, `FilesWithMatches` returns
    ///   one entry per file (with line_number 0 and empty content).
    pub fn grep_search_in(
        &self,
        pattern: &str,
        file_pattern: Option<&str>,
        path: Option<&str>,
        output_mode: GrepOutputMode,
    ) -> Result<Vec<GrepMatch>> {
        let re = Regex::new(pattern).map_err(|e| {
            RustChainError::Other(format!("Invalid regex pattern '{}': {}", pattern, e))
        })?;

        let search_root = if let Some(p) = path {
            self.validate_path(Path::new(p))?
        } else {
            self.root_path.clone()
        };

        let mut results = Vec::new();
        self.search_directory(&search_root, &re, file_pattern, output_mode, &mut results)?;

        Ok(results)
    }

    fn search_directory(
        &self,
        dir: &Path,
        regex: &Regex,
        file_pattern: Option<&str>,
        output_mode: GrepOutputMode,
        results: &mut Vec<GrepMatch>,
    ) -> Result<()> {
        let entries = std::fs::read_dir(dir).map_err(|e| {
            RustChainError::Other(format!(
                "Failed to read directory '{}': {}",
                dir.display(),
                e
            ))
        })?;

        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Recurse into subdirectories (skip hidden dirs)
                if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                    if !name.starts_with('.') {
                        self.search_directory(&path, regex, file_pattern, output_mode, results)?;
                    }
                }
            } else if path.is_file() {
                // Check file pattern filter
                if let Some(fp) = file_pattern {
                    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
                    if !matches_simple_glob(name, fp) {
                        continue;
                    }
                }

                // Check file size
                if let Ok(metadata) = path.metadata() {
                    if metadata.len() > self.max_file_size as u64 {
                        continue;
                    }
                }

                // Read and search the file
                if let Ok(content) = std::fs::read_to_string(&path) {
                    let rel_path = path
                        .strip_prefix(&self.root_path)
                        .unwrap_or(&path)
                        .to_string_lossy()
                        .to_string();

                    match output_mode {
                        GrepOutputMode::Content => {
                            for (line_num, line) in content.lines().enumerate() {
                                if regex.is_match(line) {
                                    results.push(GrepMatch {
                                        file_path: rel_path.clone(),
                                        line_number: line_num + 1,
                                        line_content: line.to_string(),
                                    });
                                }
                            }
                        }
                        GrepOutputMode::FilesWithMatches => {
                            if regex.is_match(&content) {
                                results.push(GrepMatch {
                                    file_path: rel_path,
                                    line_number: 0,
                                    line_content: String::new(),
                                });
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

/// A match found by grep search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrepMatch {
    /// File path relative to root.
    pub file_path: String,
    /// Line number (1-indexed).
    pub line_number: usize,
    /// Content of the matching line.
    pub line_content: String,
}

/// Normalize a path by resolving `.` and `..` components without filesystem access.
fn normalize_path(path: &Path) -> PathBuf {
    let mut components = Vec::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                components.pop();
            }
            std::path::Component::CurDir => {}
            other => {
                components.push(other);
            }
        }
    }
    components.iter().collect()
}

/// Glob matching that supports `**` for recursive directory matching.
///
/// `**` matches any number of path segments (including zero).
/// `*` matches any characters except `/`.
fn matches_glob_pattern(text: &str, pattern: &str) -> bool {
    // Convert glob pattern to a regex
    let mut regex_str = String::from("^");
    let chars: Vec<char> = pattern.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if i + 1 < chars.len() && chars[i] == '*' && chars[i + 1] == '*' {
            // `**` — match any path segments
            if i + 2 < chars.len() && chars[i + 2] == '/' {
                regex_str.push_str("(.*/)?");
                i += 3;
            } else {
                regex_str.push_str(".*");
                i += 2;
            }
        } else if chars[i] == '*' {
            // `*` — match anything except `/`
            regex_str.push_str("[^/]*");
            i += 1;
        } else if chars[i] == '?' {
            regex_str.push_str("[^/]");
            i += 1;
        } else {
            // Escape regex special chars
            let c = chars[i];
            if ".+^${}()|[]\\".contains(c) {
                regex_str.push('\\');
            }
            regex_str.push(c);
            i += 1;
        }
    }
    regex_str.push('$');

    Regex::new(&regex_str)
        .map(|re| re.is_match(text))
        .unwrap_or(false)
}

/// Simple glob matching (supports `*` as wildcard).
fn matches_simple_glob(text: &str, pattern: &str) -> bool {
    if !pattern.contains('*') {
        return text == pattern;
    }
    let parts: Vec<&str> = pattern.split('*').collect();
    if parts.is_empty() {
        return true;
    }

    let mut pos = 0;
    for (i, part) in parts.iter().enumerate() {
        if part.is_empty() {
            continue;
        }
        if let Some(found) = text[pos..].find(part) {
            if i == 0 && found != 0 {
                return false; // Must start with first part if no leading *
            }
            pos += found + part.len();
        } else {
            return false;
        }
    }
    // If no trailing *, the text must end exactly
    if !pattern.ends_with('*') {
        return text.ends_with(parts.last().unwrap_or(&""));
    }
    true
}

#[async_trait]
impl AgentMiddleware for FilesystemFileSearchMiddleware {
    fn name(&self) -> &str {
        "FilesystemFileSearchMiddleware"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_path_valid() {
        let mw = FilesystemFileSearchMiddleware::new("/home/user/project");
        let result = mw.validate_path(Path::new("src/main.rs"));
        assert!(result.is_ok());
        let path = result.unwrap();
        assert!(path.starts_with("/home/user/project"));
    }

    #[test]
    fn test_validate_path_traversal_rejected() {
        let mw = FilesystemFileSearchMiddleware::new("/home/user/project");
        let result = mw.validate_path(Path::new("../../etc/passwd"));
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Path traversal"));
    }

    #[test]
    fn test_validate_path_absolute_within_root() {
        let mw = FilesystemFileSearchMiddleware::new("/home/user/project");
        let result = mw.validate_path(Path::new("/home/user/project/src/lib.rs"));
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_path_absolute_outside_root() {
        let mw = FilesystemFileSearchMiddleware::new("/home/user/project");
        let result = mw.validate_path(Path::new("/etc/passwd"));
        assert!(result.is_err());
    }

    #[test]
    fn test_normalize_path() {
        let path = normalize_path(Path::new("/a/b/../c/./d"));
        assert_eq!(path, PathBuf::from("/a/c/d"));
    }

    #[test]
    fn test_matches_simple_glob() {
        assert!(matches_simple_glob("hello.rs", "*.rs"));
        assert!(matches_simple_glob("hello.rs", "hello.*"));
        assert!(matches_simple_glob("hello.rs", "*"));
        assert!(!matches_simple_glob("hello.rs", "*.py"));
        assert!(matches_simple_glob("hello.rs", "hello.rs"));
    }

    #[test]
    fn test_glob_search_rejects_traversal() {
        let mw = FilesystemFileSearchMiddleware::new("/tmp/test");
        let result = mw.glob_search("../../etc/*");
        assert!(result.is_err());
    }

    #[test]
    fn test_middleware_name() {
        let mw = FilesystemFileSearchMiddleware::new("/tmp");
        assert_eq!(mw.name(), "FilesystemFileSearchMiddleware");
    }

    #[test]
    fn test_builder_methods() {
        let mw = FilesystemFileSearchMiddleware::new("/tmp")
            .with_ripgrep(true)
            .with_max_file_size(1024);
        assert!(mw.use_ripgrep);
        assert_eq!(mw.max_file_size, 1024);
    }

    #[test]
    fn test_grep_match_serde() {
        let m = GrepMatch {
            file_path: "src/main.rs".into(),
            line_number: 42,
            line_content: "fn main() {".into(),
        };
        let json = serde_json::to_string(&m).unwrap();
        let parsed: GrepMatch = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.line_number, 42);
    }
}
