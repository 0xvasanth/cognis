//! Python REPL tool for executing Python code snippets.
//!
//! Executes Python code via a `python3` subprocess with safety controls
//! including import blocking, dangerous operation detection, and output
//! truncation.

use async_trait::async_trait;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::tools::base::BaseTool;
use rustchain_core::tools::types::{ToolInput, ToolOutput};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the Python REPL tool.
#[derive(Debug, Clone)]
pub struct PythonREPLConfig {
    /// Path to the Python interpreter (default: "python3").
    pub python_path: String,
    /// Maximum execution time for a single run (default: 30s).
    pub timeout: Duration,
    /// Maximum number of characters in output before truncation (default: 10000).
    pub max_output_length: usize,
    /// Optional whitelist of allowed imports. When `Some`, only these imports
    /// are permitted. When `None`, the blocked list is used instead.
    pub allowed_imports: Option<Vec<String>>,
    /// Imports that are blocked by default.
    pub blocked_imports: Vec<String>,
    /// Optional working directory for the subprocess.
    pub working_directory: Option<PathBuf>,
    /// Extra environment variables passed to the subprocess.
    pub env_vars: HashMap<String, String>,
    /// Whether to sanitize input before execution (default: true).
    pub sanitize_input: bool,
}

impl Default for PythonREPLConfig {
    fn default() -> Self {
        Self {
            python_path: "python3".to_string(),
            timeout: Duration::from_secs(30),
            max_output_length: 10000,
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

impl PythonREPLConfig {
    /// Create a new builder for `PythonREPLConfig`.
    pub fn builder() -> PythonREPLConfigBuilder {
        PythonREPLConfigBuilder::default()
    }
}

/// Builder for `PythonREPLConfig`.
#[derive(Debug, Default)]
pub struct PythonREPLConfigBuilder {
    python_path: Option<String>,
    timeout: Option<Duration>,
    max_output_length: Option<usize>,
    allowed_imports: Option<Vec<String>>,
    blocked_imports: Option<Vec<String>>,
    working_directory: Option<PathBuf>,
    env_vars: Option<HashMap<String, String>>,
    sanitize_input: Option<bool>,
}

impl PythonREPLConfigBuilder {
    pub fn python_path(mut self, path: impl Into<String>) -> Self {
        self.python_path = Some(path.into());
        self
    }

    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn max_output_length(mut self, len: usize) -> Self {
        self.max_output_length = Some(len);
        self
    }

    pub fn allowed_imports(mut self, imports: Vec<String>) -> Self {
        self.allowed_imports = Some(imports);
        self
    }

    pub fn blocked_imports(mut self, imports: Vec<String>) -> Self {
        self.blocked_imports = Some(imports);
        self
    }

    pub fn working_directory(mut self, dir: impl Into<PathBuf>) -> Self {
        self.working_directory = Some(dir.into());
        self
    }

    pub fn env_vars(mut self, vars: HashMap<String, String>) -> Self {
        self.env_vars = Some(vars);
        self
    }

    pub fn sanitize_input(mut self, sanitize: bool) -> Self {
        self.sanitize_input = Some(sanitize);
        self
    }

    pub fn build(self) -> PythonREPLConfig {
        let defaults = PythonREPLConfig::default();
        PythonREPLConfig {
            python_path: self.python_path.unwrap_or(defaults.python_path),
            timeout: self.timeout.unwrap_or(defaults.timeout),
            max_output_length: self.max_output_length.unwrap_or(defaults.max_output_length),
            allowed_imports: self.allowed_imports,
            blocked_imports: self.blocked_imports.unwrap_or(defaults.blocked_imports),
            working_directory: self.working_directory,
            env_vars: self.env_vars.unwrap_or_default(),
            sanitize_input: self.sanitize_input.unwrap_or(defaults.sanitize_input),
        }
    }
}

// ---------------------------------------------------------------------------
// Code sanitizer
// ---------------------------------------------------------------------------

/// Error returned when code sanitization fails.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SanitizationError {
    /// Human-readable description of what was blocked.
    pub message: String,
    /// The specific pattern that triggered the block.
    pub blocked_pattern: String,
}

impl std::fmt::Display for SanitizationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: '{}'", self.message, self.blocked_pattern)
    }
}

/// Sanitizes Python code according to the provided configuration.
pub struct CodeSanitizer;

impl CodeSanitizer {
    /// Sanitize the given code, returning the cleaned code or an error.
    pub fn sanitize(
        code: &str,
        config: &PythonREPLConfig,
    ) -> std::result::Result<String, SanitizationError> {
        let code = Self::strip_shell_escapes(code);

        // Check for dangerous built-in operations
        Self::check_dangerous_operations(&code)?;

        // Check imports
        Self::check_imports(&code, config)?;

        Ok(code)
    }

    /// Strip shell escape characters from the code.
    fn strip_shell_escapes(code: &str) -> String {
        code.chars()
            .filter(|c| {
                // Remove common shell escape / control characters (except
                // normal whitespace characters like \n, \r, \t).
                !matches!(c, '\x00'..='\x08' | '\x0b' | '\x0c' | '\x0e'..='\x1f' | '\x7f')
            })
            .collect()
    }

    /// Check for dangerous built-in operations: exec, eval, __import__, open with write.
    fn check_dangerous_operations(code: &str) -> std::result::Result<(), SanitizationError> {
        let dangerous_patterns = [
            ("exec(", "Use of exec() is not allowed"),
            ("exec (", "Use of exec() is not allowed"),
            ("eval(", "Use of eval() is not allowed"),
            ("eval (", "Use of eval() is not allowed"),
            ("__import__", "Use of __import__ is not allowed"),
        ];

        for (pattern, message) in &dangerous_patterns {
            if code.contains(pattern) {
                return Err(SanitizationError {
                    message: message.to_string(),
                    blocked_pattern: pattern.to_string(),
                });
            }
        }

        // Check for open() with write modes
        Self::check_open_write(code)?;

        Ok(())
    }

    /// Check for `open(...)` calls that include a write mode.
    fn check_open_write(code: &str) -> std::result::Result<(), SanitizationError> {
        let write_modes = [
            "\"w\"", "'w'", "\"a\"", "'a'", "\"w+\"", "'w+'", "\"a+\"", "'a+'", "\"wb\"", "'wb'",
            "\"ab\"", "'ab'",
        ];
        for line in code.lines() {
            let trimmed = line.trim();
            if trimmed.contains("open(") || trimmed.contains("open (") {
                for mode in &write_modes {
                    if trimmed.contains(mode) {
                        return Err(SanitizationError {
                            message: "Writing files via open() is not allowed".to_string(),
                            blocked_pattern: format!("open with mode {}", mode),
                        });
                    }
                }
            }
        }
        Ok(())
    }

    /// Check imports against allowed/blocked lists.
    fn check_imports(
        code: &str,
        config: &PythonREPLConfig,
    ) -> std::result::Result<(), SanitizationError> {
        let imports = Self::extract_imports(code);

        if let Some(ref allowed) = config.allowed_imports {
            // Whitelist mode
            for imp in &imports {
                let root = imp.split('.').next().unwrap_or(imp);
                if !allowed.iter().any(|a| a == root) {
                    return Err(SanitizationError {
                        message: format!("Import '{}' is not in the allowed list", imp),
                        blocked_pattern: imp.clone(),
                    });
                }
            }
        } else {
            // Blocklist mode
            for imp in &imports {
                let root = imp.split('.').next().unwrap_or(imp);
                if config.blocked_imports.iter().any(|b| b == root) {
                    return Err(SanitizationError {
                        message: format!("Import '{}' is blocked", imp),
                        blocked_pattern: imp.clone(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Extract import names from Python code (simple line-based parsing).
    fn extract_imports(code: &str) -> Vec<String> {
        let mut imports = Vec::new();
        for line in code.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("import ") {
                // `import foo, bar` or `import foo.bar`
                for part in rest.split(',') {
                    let module = part.trim().split_whitespace().next().unwrap_or("");
                    if !module.is_empty() {
                        imports.push(module.to_string());
                    }
                }
            } else if let Some(rest) = trimmed.strip_prefix("from ") {
                // `from foo import bar`
                if let Some(module) = rest.split_whitespace().next() {
                    imports.push(module.to_string());
                }
            }
        }
        imports
    }
}

// ---------------------------------------------------------------------------
// Execution result
// ---------------------------------------------------------------------------

/// Result of a Python code execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonREPLResult {
    /// Standard output from the subprocess.
    pub stdout: String,
    /// Standard error from the subprocess.
    pub stderr: String,
    /// Process exit code.
    pub exit_code: i32,
    /// How long the execution took.
    pub execution_time: Duration,
    /// Whether the output was truncated to fit `max_output_length`.
    pub truncated: bool,
}

// ---------------------------------------------------------------------------
// Python REPL Tool
// ---------------------------------------------------------------------------

/// A tool that executes Python code snippets via a subprocess.
pub struct PythonREPLTool {
    /// Configuration for the tool.
    pub config: PythonREPLConfig,
}

impl PythonREPLTool {
    /// Create a new `PythonREPLTool` with default configuration.
    pub fn new() -> Self {
        Self {
            config: PythonREPLConfig::default(),
        }
    }

    /// Create a new `PythonREPLTool` with the given configuration.
    pub fn with_config(config: PythonREPLConfig) -> Self {
        Self { config }
    }

    /// Validate the given code without executing it.
    pub fn validate(&self, code: &str) -> Result<()> {
        if code.trim().is_empty() {
            return Err(RustChainError::ToolValidationError(
                "Code must not be empty".to_string(),
            ));
        }

        if self.config.sanitize_input {
            CodeSanitizer::sanitize(code, &self.config)
                .map_err(|e| RustChainError::ToolValidationError(e.to_string()))?;
        }

        Ok(())
    }

    /// Execute Python code and return a detailed result.
    pub async fn run_code(&self, code: &str) -> Result<PythonREPLResult> {
        self.run_with_timeout(code, self.config.timeout).await
    }

    /// Execute Python code with a specific timeout.
    pub async fn run_with_timeout(
        &self,
        code: &str,
        timeout: Duration,
    ) -> Result<PythonREPLResult> {
        self.validate(code)?;

        let sanitized = if self.config.sanitize_input {
            CodeSanitizer::sanitize(code, &self.config)
                .map_err(|e| RustChainError::ToolValidationError(e.to_string()))?
        } else {
            code.to_string()
        };

        let start = std::time::Instant::now();

        let mut cmd = tokio::process::Command::new(&self.config.python_path);
        cmd.arg("-c").arg(&sanitized);

        if let Some(ref dir) = self.config.working_directory {
            cmd.current_dir(dir);
        }

        for (key, value) in &self.config.env_vars {
            cmd.env(key, value);
        }

        let output = tokio::time::timeout(timeout, cmd.output())
            .await
            .map_err(|_| {
                RustChainError::ToolException(format!(
                    "Python execution timed out after {:?}",
                    timeout
                ))
            })?
            .map_err(|e| RustChainError::ToolException(format!("Failed to run Python: {}", e)))?;

        let execution_time = start.elapsed();
        let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let mut stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let mut truncated = false;

        let max_len = self.config.max_output_length;
        if stdout.len() + stderr.len() > max_len {
            truncated = true;
            if stdout.len() > max_len / 2 {
                stdout.truncate(max_len / 2);
            }
            let remaining = max_len.saturating_sub(stdout.len());
            if stderr.len() > remaining {
                stderr.truncate(remaining);
            }
        }

        Ok(PythonREPLResult {
            stdout,
            stderr,
            exit_code: output.status.code().unwrap_or(-1),
            execution_time,
            truncated,
        })
    }
}

impl Default for PythonREPLTool {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseTool for PythonREPLTool {
    fn name(&self) -> &str {
        "python_repl"
    }

    fn description(&self) -> &str {
        "Execute Python code snippets. Input should be valid Python code."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let code = extract_code(&input)?;
        let result = self.run_code(&code).await?;

        let output = if result.stderr.is_empty() {
            result.stdout.clone()
        } else if result.stdout.is_empty() {
            result.stderr.clone()
        } else {
            format!("{}{}", result.stdout, result.stderr)
        };

        Ok(ToolOutput::Content(Value::String(output)))
    }
}

/// Extract the code string from various input formats.
fn extract_code(input: &ToolInput) -> Result<String> {
    match input {
        ToolInput::Text(s) => Ok(s.clone()),
        ToolInput::Structured(map) => {
            if let Some(Value::String(code)) = map.get("code") {
                Ok(code.clone())
            } else {
                Err(RustChainError::ToolValidationError(
                    "Missing required field 'code'".into(),
                ))
            }
        }
        ToolInput::ToolCall(tc) => {
            if let Some(Value::String(code)) = tc.args.get("code") {
                Ok(code.clone())
            } else {
                Err(RustChainError::ToolValidationError(
                    "Missing required field 'code'".into(),
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mock for testing
// ---------------------------------------------------------------------------

/// A mock Python REPL that returns canned responses without executing code.
pub struct MockPythonREPL {
    responses: Vec<PythonREPLResult>,
    call_index: std::sync::atomic::AtomicUsize,
    /// Configuration used for validation.
    pub config: PythonREPLConfig,
}

impl MockPythonREPL {
    /// Create a new mock with the given canned responses.
    pub fn new(responses: Vec<PythonREPLResult>) -> Self {
        Self {
            responses,
            call_index: std::sync::atomic::AtomicUsize::new(0),
            config: PythonREPLConfig::default(),
        }
    }

    /// Validate the given code without executing it.
    pub fn validate(&self, code: &str) -> Result<()> {
        if code.trim().is_empty() {
            return Err(RustChainError::ToolValidationError(
                "Code must not be empty".to_string(),
            ));
        }

        if self.config.sanitize_input {
            CodeSanitizer::sanitize(code, &self.config)
                .map_err(|e| RustChainError::ToolValidationError(e.to_string()))?;
        }

        Ok(())
    }

    /// Return the next canned response.
    pub async fn run_code(&self, code: &str) -> Result<PythonREPLResult> {
        self.validate(code)?;

        let idx = self
            .call_index
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        if idx < self.responses.len() {
            Ok(self.responses[idx].clone())
        } else {
            Err(RustChainError::ToolException(
                "MockPythonREPL: no more canned responses".to_string(),
            ))
        }
    }

    /// Return the next canned response with a timeout parameter (ignored).
    pub async fn run_with_timeout(
        &self,
        code: &str,
        _timeout: Duration,
    ) -> Result<PythonREPLResult> {
        self.run_code(code).await
    }
}

#[async_trait]
impl BaseTool for MockPythonREPL {
    fn name(&self) -> &str {
        "python_repl"
    }

    fn description(&self) -> &str {
        "Execute Python code snippets. Input should be valid Python code."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute"
                }
            },
            "required": ["code"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let code = extract_code(&input)?;
        let result = self.run_code(&code).await?;

        let output = if result.stderr.is_empty() {
            result.stdout.clone()
        } else if result.stdout.is_empty() {
            result.stderr.clone()
        } else {
            format!("{}{}", result.stdout, result.stderr)
        };

        Ok(ToolOutput::Content(Value::String(output)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Config defaults --

    #[test]
    fn test_config_defaults() {
        let config = PythonREPLConfig::default();
        assert_eq!(config.python_path, "python3");
        assert_eq!(config.timeout, Duration::from_secs(30));
        assert_eq!(config.max_output_length, 10000);
        assert!(config.allowed_imports.is_none());
        assert_eq!(
            config.blocked_imports,
            vec!["os", "subprocess", "shutil", "sys"]
        );
        assert!(config.working_directory.is_none());
        assert!(config.env_vars.is_empty());
        assert!(config.sanitize_input);
    }

    // -- Code sanitizer: blocks dangerous imports --

    #[test]
    fn test_sanitizer_blocks_os_import() {
        let config = PythonREPLConfig::default();
        let result = CodeSanitizer::sanitize("import os\nos.system('ls')", &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("blocked"), "Error: {}", err);
        assert_eq!(err.blocked_pattern, "os");
    }

    #[test]
    fn test_sanitizer_blocks_subprocess_import() {
        let config = PythonREPLConfig::default();
        let result = CodeSanitizer::sanitize("import subprocess", &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("blocked"));
        assert_eq!(err.blocked_pattern, "subprocess");
    }

    // -- Code sanitizer: allows safe imports --

    #[test]
    fn test_sanitizer_allows_math_import() {
        let config = PythonREPLConfig::default();
        let result = CodeSanitizer::sanitize("import math\nprint(math.pi)", &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sanitizer_allows_json_import() {
        let config = PythonREPLConfig::default();
        let result = CodeSanitizer::sanitize("import json\nprint(json.dumps({'a': 1}))", &config);
        assert!(result.is_ok());
    }

    // -- Code sanitizer: blocks exec/eval --

    #[test]
    fn test_sanitizer_blocks_exec() {
        let config = PythonREPLConfig::default();
        let result = CodeSanitizer::sanitize("exec('print(1)')", &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("exec"));
    }

    #[test]
    fn test_sanitizer_blocks_eval() {
        let config = PythonREPLConfig::default();
        let result = CodeSanitizer::sanitize("x = eval('1+1')", &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("eval"));
    }

    // -- Code sanitizer: blocks __import__ --

    #[test]
    fn test_sanitizer_blocks_dunder_import() {
        let config = PythonREPLConfig::default();
        let result = CodeSanitizer::sanitize("os = __import__('os')", &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("__import__"));
    }

    // -- Allowed imports whitelist --

    #[test]
    fn test_allowed_imports_whitelist() {
        let config = PythonREPLConfig {
            allowed_imports: Some(vec!["math".to_string(), "json".to_string()]),
            ..Default::default()
        };

        // Allowed
        let result = CodeSanitizer::sanitize("import math", &config);
        assert!(result.is_ok());

        // Not in whitelist
        let result = CodeSanitizer::sanitize("import collections", &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("not in the allowed list"));
    }

    // -- MockPythonREPL returns canned responses --

    #[tokio::test]
    async fn test_mock_returns_canned_responses() {
        let responses = vec![
            PythonREPLResult {
                stdout: "42\n".to_string(),
                stderr: String::new(),
                exit_code: 0,
                execution_time: Duration::from_millis(10),
                truncated: false,
            },
            PythonREPLResult {
                stdout: "hello\n".to_string(),
                stderr: String::new(),
                exit_code: 0,
                execution_time: Duration::from_millis(5),
                truncated: false,
            },
        ];

        let mock = MockPythonREPL::new(responses);

        let r1 = mock.run_code("print(42)").await.unwrap();
        assert_eq!(r1.stdout, "42\n");
        assert_eq!(r1.exit_code, 0);

        let r2 = mock.run_code("print('hello')").await.unwrap();
        assert_eq!(r2.stdout, "hello\n");

        // Exhausted
        let r3 = mock.run_code("print('gone')").await;
        assert!(r3.is_err());
    }

    // -- Result structure --

    #[test]
    fn test_result_structure() {
        let result = PythonREPLResult {
            stdout: "output".to_string(),
            stderr: "warning".to_string(),
            exit_code: 1,
            execution_time: Duration::from_millis(123),
            truncated: false,
        };

        assert_eq!(result.stdout, "output");
        assert_eq!(result.stderr, "warning");
        assert_eq!(result.exit_code, 1);
        assert_eq!(result.execution_time, Duration::from_millis(123));
        assert!(!result.truncated);
    }

    // -- Output truncation --

    #[test]
    fn test_output_truncation_flag() {
        let result = PythonREPLResult {
            stdout: "a".repeat(8000),
            stderr: "b".repeat(8000),
            exit_code: 0,
            execution_time: Duration::from_millis(10),
            truncated: true,
        };

        assert!(result.truncated);
    }

    // -- Validate method --

    #[test]
    fn test_validate_empty_code() {
        let tool = PythonREPLTool::new();
        let result = tool.validate("");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn test_validate_safe_code() {
        let tool = PythonREPLTool::new();
        let result = tool.validate("print('hello')");
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_dangerous_code() {
        let tool = PythonREPLTool::new();
        let result = tool.validate("import os");
        assert!(result.is_err());
    }

    // -- Tool schema generation --

    #[test]
    fn test_tool_schema() {
        let tool = PythonREPLTool::new();
        assert_eq!(tool.name(), "python_repl");
        assert!(tool.description().contains("Python"));

        let schema = tool.args_schema().unwrap();
        let props = schema.get("properties").unwrap();
        assert!(props.get("code").is_some());

        let required = schema.get("required").unwrap().as_array().unwrap();
        assert!(required.contains(&Value::String("code".to_string())));
    }

    // -- Builder pattern --

    #[test]
    fn test_builder_pattern() {
        let config = PythonREPLConfig::builder()
            .python_path("/usr/bin/python3.11")
            .timeout(Duration::from_secs(60))
            .max_output_length(5000)
            .blocked_imports(vec!["os".to_string()])
            .sanitize_input(false)
            .build();

        assert_eq!(config.python_path, "/usr/bin/python3.11");
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert_eq!(config.max_output_length, 5000);
        assert_eq!(config.blocked_imports, vec!["os"]);
        assert!(!config.sanitize_input);
    }

    // -- Multiple blocked patterns in one snippet --

    #[test]
    fn test_multiple_blocked_patterns() {
        let config = PythonREPLConfig::default();
        // Contains both exec and an os import — should fail on the first check
        let result = CodeSanitizer::sanitize("exec('import os')", &config);
        assert!(result.is_err());
    }

    // -- Empty code handling --

    #[tokio::test]
    async fn test_empty_code_handling() {
        let mock = MockPythonREPL::new(vec![]);
        let result = mock.run_code("").await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.to_string().contains("empty"));
    }

    // -- Whitespace-only code --

    #[tokio::test]
    async fn test_whitespace_only_code() {
        let mock = MockPythonREPL::new(vec![]);
        let result = mock.run_code("   \n  \t  ").await;
        assert!(result.is_err());
    }

    // -- from import syntax --

    #[test]
    fn test_sanitizer_blocks_from_import() {
        let config = PythonREPLConfig::default();
        let result = CodeSanitizer::sanitize("from os import path", &config);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.message.contains("blocked"));
    }
}
