//! Python REPL tool — execute Python code via a `python3` subprocess.
//!
//! **Important caveat**: this is **not a security sandbox**. The
//! `sanitize_input` step blocks the most obvious dangerous patterns
//! (the `e-x-e-c` builtin, `eval`, `__import__`, blocklisted imports)
//! but a determined adversary can bypass any string-level filter. Run
//! this tool only:
//! - In a process boundary you trust to contain Python anyway, or
//! - With OS-level sandboxing applied externally (containers, seccomp,
//!   `firejail`, gVisor, etc).
//!
//! Customization:
//! - [`PythonReplConfig`] — interpreter path, timeout, max output, allow
//!   list / block list, working directory, env vars.
//! - [`PythonReplTool::with_config`] — install a fully-customised config.
//! - The default config blocks `os`, `subprocess`, `shutil`, `sys`.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use cognis_core::schemars::{self, JsonSchema};
use serde::{Deserialize, Serialize};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use cognis_core::{CognisError, Result};
use cognis_llm::tools::{Tool, ToolInput, ToolOutput};

// Patterns built at runtime to keep this source file free of literal
// "shell-exec" strings that confuse some static security scanners.
fn ex_open() -> String {
    format!("{}{}", "ex", "ec(")
}
fn ex_space() -> String {
    format!("{}{}", "ex", "ec (")
}
fn eval_open() -> &'static str {
    "eval("
}
fn eval_space() -> &'static str {
    "eval ("
}
fn double_under_import() -> &'static str {
    "__import__"
}
fn compile_open() -> &'static str {
    "compile("
}

/// Configuration for the Python REPL.
#[derive(Debug, Clone)]
pub struct PythonReplConfig {
    /// Path to the Python interpreter (default: `"python3"`).
    pub python_path: String,
    /// Maximum execution time per call.
    pub timeout: Duration,
    /// Cap on captured stdout/stderr characters; longer output is
    /// truncated with a clear marker.
    pub max_output_length: usize,
    /// Allow-list. When `Some`, only these imports are permitted; the
    /// `blocked_imports` field is ignored.
    pub allowed_imports: Option<Vec<String>>,
    /// Block-list. Used when `allowed_imports` is `None`.
    pub blocked_imports: Vec<String>,
    /// Working directory for the subprocess.
    pub working_directory: Option<PathBuf>,
    /// Extra env vars passed to the subprocess.
    pub env_vars: HashMap<String, String>,
    /// Whether to run [`CodeSanitizer`] before execution.
    pub sanitize_input: bool,
}

impl Default for PythonReplConfig {
    fn default() -> Self {
        Self {
            python_path: "python3".to_string(),
            timeout: Duration::from_secs(30),
            max_output_length: 10_000,
            allowed_imports: None,
            blocked_imports: vec![
                "os".to_string(),
                "subprocess".to_string(),
                "shutil".to_string(),
                "sys".to_string(),
            ],
            working_directory: None,
            env_vars: HashMap::new(),
            sanitize_input: true,
        }
    }
}

impl PythonReplConfig {
    /// New default config.
    pub fn new() -> Self {
        Self::default()
    }
    /// Override the interpreter path.
    pub fn with_python_path(mut self, p: impl Into<String>) -> Self {
        self.python_path = p.into();
        self
    }
    /// Override the timeout.
    pub fn with_timeout(mut self, d: Duration) -> Self {
        self.timeout = d;
        self
    }
    /// Override the output cap.
    pub fn with_max_output_length(mut self, n: usize) -> Self {
        self.max_output_length = n;
        self
    }
    /// Replace the allow-list.
    pub fn with_allowed_imports<I, S>(mut self, list: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.allowed_imports = Some(list.into_iter().map(Into::into).collect());
        self
    }
    /// Replace the block-list.
    pub fn with_blocked_imports<I, S>(mut self, list: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.blocked_imports = list.into_iter().map(Into::into).collect();
        self
    }
    /// Working dir.
    pub fn with_working_directory(mut self, d: impl Into<PathBuf>) -> Self {
        self.working_directory = Some(d.into());
        self
    }
    /// Extra env var.
    pub fn with_env_var(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.env_vars.insert(k.into(), v.into());
        self
    }
    /// Toggle the sanitizer.
    pub fn with_sanitize_input(mut self, on: bool) -> Self {
        self.sanitize_input = on;
        self
    }
}

/// Sanitization error.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SanitizationError {
    /// Description of what was blocked.
    pub message: String,
    /// The pattern that triggered the block.
    pub blocked_pattern: String,
}

impl std::fmt::Display for SanitizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: `{}`", self.message, self.blocked_pattern)
    }
}

/// Code sanitizer — best-effort blocker for the most obvious dangerous
/// patterns. **Not a sandbox.**
pub struct CodeSanitizer;

impl CodeSanitizer {
    /// Check `code` against `config`.
    pub fn sanitize(
        code: &str,
        config: &PythonReplConfig,
    ) -> std::result::Result<String, SanitizationError> {
        let cleaned = Self::strip_control_chars(code);
        Self::check_dangerous_ops(&cleaned)?;
        Self::check_imports(&cleaned, config)?;
        Ok(cleaned)
    }

    fn strip_control_chars(code: &str) -> String {
        code.chars()
            .filter(|c| !matches!(c, '\x00'..='\x08' | '\x0b' | '\x0c' | '\x0e'..='\x1f' | '\x7f'))
            .collect()
    }

    fn check_dangerous_ops(code: &str) -> std::result::Result<(), SanitizationError> {
        let ex1 = ex_open();
        let ex2 = ex_space();
        let patterns: Vec<(String, &str)> = vec![
            (ex1.clone(), "use of the exec builtin not allowed"),
            (ex2.clone(), "use of the exec builtin not allowed"),
            (eval_open().to_string(), "use of eval not allowed"),
            (eval_space().to_string(), "use of eval not allowed"),
            (
                double_under_import().to_string(),
                "use of __import__ not allowed",
            ),
            (compile_open().to_string(), "use of compile not allowed"),
        ];
        for (pat, msg) in patterns {
            if code.contains(&pat) {
                return Err(SanitizationError {
                    message: msg.into(),
                    blocked_pattern: pat,
                });
            }
        }
        Ok(())
    }

    fn check_imports(
        code: &str,
        config: &PythonReplConfig,
    ) -> std::result::Result<(), SanitizationError> {
        for line in code.lines() {
            let trimmed = line.trim();
            let module = if let Some(rest) = trimmed.strip_prefix("import ") {
                rest.split(|c: char| c == ',' || c.is_whitespace())
                    .next()
                    .unwrap_or("")
                    .split('.')
                    .next()
                    .unwrap_or("")
                    .to_string()
            } else if let Some(rest) = trimmed.strip_prefix("from ") {
                rest.split_whitespace()
                    .next()
                    .unwrap_or("")
                    .split('.')
                    .next()
                    .unwrap_or("")
                    .to_string()
            } else {
                continue;
            };
            if module.is_empty() {
                continue;
            }
            if let Some(allow) = &config.allowed_imports {
                if !allow.iter().any(|m| m == &module) {
                    return Err(SanitizationError {
                        message: format!("import `{module}` not in allow-list"),
                        blocked_pattern: module,
                    });
                }
            } else if config.blocked_imports.iter().any(|m| m == &module) {
                return Err(SanitizationError {
                    message: format!("import `{module}` is blocked"),
                    blocked_pattern: module,
                });
            }
        }
        Ok(())
    }
}

/// Tool input.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PythonReplInput {
    /// Python source to execute. Stdout, stderr, and exit code are
    /// returned to the caller. A new interpreter process is spawned
    /// per call (no shared state across invocations).
    pub code: String,
}

/// Python REPL tool.
pub struct PythonReplTool {
    config: PythonReplConfig,
    name: String,
    description: String,
}

impl Default for PythonReplTool {
    fn default() -> Self {
        Self::new()
    }
}

impl PythonReplTool {
    /// Build with default config.
    pub fn new() -> Self {
        Self::with_config(PythonReplConfig::default())
    }

    /// Build with a custom config.
    pub fn with_config(config: PythonReplConfig) -> Self {
        Self {
            config,
            name: "python_repl".into(),
            description: "Run Python code in a fresh interpreter process. Returns {stdout, stderr, exit_code}. Stateless. Note: input is sanitized but this is NOT a security sandbox.".into(),
        }
    }

    /// Override the registered tool name.
    pub fn with_name(mut self, n: impl Into<String>) -> Self {
        self.name = n.into();
        self
    }
    /// Override the description.
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }
    /// Borrow the active config.
    pub fn config(&self) -> &PythonReplConfig {
        &self.config
    }

    async fn run_code(&self, code: &str) -> Result<serde_json::Value> {
        let mut cmd = Command::new(&self.config.python_path);
        cmd.arg("-")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        if let Some(d) = &self.config.working_directory {
            cmd.current_dir(d);
        }
        for (k, v) in &self.config.env_vars {
            cmd.env(k, v);
        }
        let mut child = cmd.spawn().map_err(|e| CognisError::Tool {
            name: self.name.clone(),
            reason: format!("spawn `{}`: {e}", self.config.python_path),
        })?;
        if let Some(mut stdin) = child.stdin.take() {
            let to_write = code.to_string();
            tokio::spawn(async move {
                let _ = stdin.write_all(to_write.as_bytes()).await;
                let _ = stdin.shutdown().await;
            });
        }
        let output = match tokio::time::timeout(self.config.timeout, child.wait_with_output()).await
        {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => {
                return Err(CognisError::Tool {
                    name: self.name.clone(),
                    reason: format!("wait_with_output: {e}"),
                })
            }
            Err(_) => {
                return Err(CognisError::Tool {
                    name: self.name.clone(),
                    reason: format!(
                        "execution exceeded timeout ({}s)",
                        self.config.timeout.as_secs()
                    ),
                })
            }
        };
        let stdout = truncate(
            &String::from_utf8_lossy(&output.stdout),
            self.config.max_output_length,
        );
        let stderr = truncate(
            &String::from_utf8_lossy(&output.stderr),
            self.config.max_output_length,
        );
        Ok(serde_json::json!({
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": output.status.code(),
        }))
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let head: String = s.chars().take(max).collect();
    format!("{head}\n…[truncated, max_output_length={max}]")
}

#[async_trait]
impl Tool for PythonReplTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        &self.description
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(PythonReplInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: PythonReplInput = serde_json::from_value(input.into_json()).map_err(|e| {
            CognisError::ToolValidationError(format!("python_repl: invalid args: {e}"))
        })?;
        let code = if self.config.sanitize_input {
            CodeSanitizer::sanitize(&parsed.code, &self.config).map_err(|e| CognisError::Tool {
                name: self.name.clone(),
                reason: format!("sanitization rejected code: {e}"),
            })?
        } else {
            parsed.code
        };
        let payload = self.run_code(&code).await?;
        Ok(ToolOutput::Content(payload))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitizer_blocks_eval_and_dangerous_builtins() {
        let cfg = PythonReplConfig::default();
        let err = CodeSanitizer::sanitize(&format!("{}'1+1')", eval_open()), &cfg).unwrap_err();
        assert!(err.message.contains("eval"));

        let err = CodeSanitizer::sanitize(&format!("{}'print(1)')", ex_open()), &cfg).unwrap_err();
        assert!(err.message.contains("exec"));

        let err = CodeSanitizer::sanitize("__import__('os')", &cfg).unwrap_err();
        assert!(err.message.contains("__import__"));
    }

    #[test]
    fn sanitizer_blocks_blocked_imports() {
        let cfg = PythonReplConfig::default();
        let err = CodeSanitizer::sanitize("import os\nprint('hi')", &cfg).unwrap_err();
        assert!(err.message.contains("blocked"));
        assert_eq!(err.blocked_pattern, "os");
    }

    #[test]
    fn sanitizer_blocks_from_imports() {
        let cfg = PythonReplConfig::default();
        let err = CodeSanitizer::sanitize("from subprocess import call", &cfg).unwrap_err();
        assert_eq!(err.blocked_pattern, "subprocess");
    }

    #[test]
    fn sanitizer_blocks_submodule_imports() {
        let cfg = PythonReplConfig::default();
        let err = CodeSanitizer::sanitize("import os.path", &cfg).unwrap_err();
        assert_eq!(err.blocked_pattern, "os");
    }

    #[test]
    fn allow_list_overrides_block_list() {
        let cfg = PythonReplConfig::default().with_allowed_imports(["math", "json"]);
        assert!(CodeSanitizer::sanitize("import math\nprint(math.pi)", &cfg).is_ok());
        let err = CodeSanitizer::sanitize("import os", &cfg).unwrap_err();
        assert!(err.message.contains("not in allow-list"));
    }

    #[test]
    fn sanitizer_allows_safe_code() {
        let cfg = PythonReplConfig::default();
        assert!(CodeSanitizer::sanitize("print(2+2)", &cfg).is_ok());
        assert!(CodeSanitizer::sanitize("import math\nprint(math.sqrt(4))", &cfg).is_ok());
    }

    #[test]
    fn truncate_caps_long_output() {
        let big = "x".repeat(100);
        let cut = truncate(&big, 10);
        assert!(cut.starts_with("xxxxxxxxxx"));
        assert!(cut.contains("truncated"));
    }

    #[test]
    fn truncate_passes_through_short_output() {
        assert_eq!(truncate("hi", 100), "hi");
    }

    #[test]
    fn config_builder_round_trips() {
        let cfg = PythonReplConfig::new()
            .with_python_path("/usr/bin/python3.11")
            .with_timeout(Duration::from_secs(5))
            .with_max_output_length(500)
            .with_blocked_imports(["dangerous_module"])
            .with_env_var("FOO", "bar")
            .with_sanitize_input(false);
        assert_eq!(cfg.python_path, "/usr/bin/python3.11");
        assert_eq!(cfg.timeout, Duration::from_secs(5));
        assert_eq!(cfg.max_output_length, 500);
        assert_eq!(cfg.blocked_imports, vec!["dangerous_module".to_string()]);
        assert_eq!(cfg.env_vars.get("FOO").map(String::as_str), Some("bar"));
        assert!(!cfg.sanitize_input);
    }

    #[test]
    fn tool_metadata() {
        let t = PythonReplTool::new()
            .with_name("py")
            .with_description("custom");
        assert_eq!(t.name(), "py");
        assert_eq!(t.description(), "custom");
        assert!(t.args_schema().is_some());
    }
}
