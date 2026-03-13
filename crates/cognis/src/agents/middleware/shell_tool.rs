//! Shell tool middleware — simplified shell command execution.
//!
//! Provides configuration for executing shell commands within an agent,
//! with workspace isolation and startup/shutdown hooks.
//! Note: actual subprocess management is simplified — this module
//! stores configuration and provides a placeholder execution method.

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use cognis_core::error::{Result, CognisError};

use super::types::{AgentMiddleware, AgentState};

/// Result of a command execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandExecutionResult {
    /// The exit code of the command (0 = success).
    pub exit_code: i32,
    /// Standard output.
    pub stdout: String,
    /// Standard error output.
    pub stderr: String,
    /// Whether the command timed out.
    pub timed_out: bool,
    /// Duration the command took to execute, in milliseconds.
    pub duration_ms: u64,
}

impl CommandExecutionResult {
    /// Check if the command succeeded (exit code 0).
    pub fn success(&self) -> bool {
        self.exit_code == 0 && !self.timed_out
    }
}

/// Configuration for the shell tool middleware.
pub struct ShellToolMiddleware {
    /// Workspace root directory for command execution.
    pub workspace_root: PathBuf,
    /// Commands to run on startup (e.g., environment setup).
    pub startup_commands: Vec<String>,
    /// Commands to run on shutdown (e.g., cleanup).
    pub shutdown_commands: Vec<String>,
    /// The name of the tool as exposed to the agent.
    pub tool_name: String,
    /// Maximum command execution time.
    pub timeout: Duration,
    /// Environment variables to set for commands.
    pub env_vars: HashMap<String, String>,
    /// Commands that are not allowed to be executed.
    pub blocked_commands: Vec<String>,
    /// Maximum number of output lines to capture.
    pub max_output_lines: usize,
}

impl ShellToolMiddleware {
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        let workspace: PathBuf = workspace_root.into();
        // If workspace_root doesn't exist, create a temp directory
        let actual_root = if workspace.exists() {
            workspace
        } else {
            let tmp = std::env::temp_dir().join(format!("cognis-shell-{}", std::process::id()));
            let _ = std::fs::create_dir_all(&tmp);
            tmp
        };
        Self {
            workspace_root: actual_root,
            startup_commands: Vec::new(),
            shutdown_commands: Vec::new(),
            tool_name: "shell".into(),
            timeout: Duration::from_secs(120),
            env_vars: HashMap::new(),
            blocked_commands: vec!["rm -rf /".into(), "mkfs".into(), "dd if=/dev/zero".into()],
            max_output_lines: 100,
        }
    }

    pub fn with_tool_name(mut self, name: impl Into<String>) -> Self {
        self.tool_name = name.into();
        self
    }

    pub fn with_startup_command(mut self, cmd: impl Into<String>) -> Self {
        self.startup_commands.push(cmd.into());
        self
    }

    pub fn with_shutdown_command(mut self, cmd: impl Into<String>) -> Self {
        self.shutdown_commands.push(cmd.into());
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_env(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.env_vars.insert(key.into(), value.into());
        self
    }

    pub fn with_blocked_command(mut self, cmd: impl Into<String>) -> Self {
        self.blocked_commands.push(cmd.into());
        self
    }

    pub fn with_max_output_lines(mut self, max_lines: usize) -> Self {
        self.max_output_lines = max_lines;
        self
    }

    /// Check if a command is blocked.
    pub fn is_blocked(&self, command: &str) -> bool {
        let cmd_lower = command.to_lowercase();
        self.blocked_commands
            .iter()
            .any(|blocked| cmd_lower.contains(&blocked.to_lowercase()))
    }

    /// Execute a shell command using `tokio::process::Command`.
    ///
    /// Runs the command with proper timeout, environment, and working directory.
    /// Captures stdout and stderr separately and handles exit codes.
    pub async fn execute(&self, command: &str) -> Result<CommandExecutionResult> {
        if self.is_blocked(command) {
            return Err(CognisError::Other(format!(
                "Command is blocked: {}",
                command
            )));
        }

        let start = std::time::Instant::now();
        let result = tokio::time::timeout(
            self.timeout,
            tokio::process::Command::new("sh")
                .arg("-c")
                .arg(command)
                .current_dir(&self.workspace_root)
                .envs(&self.env_vars)
                .output(),
        )
        .await;

        let duration_ms = start.elapsed().as_millis() as u64;

        match result {
            Ok(Ok(output)) => {
                let mut stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();

                // Truncate stdout to max_output_lines
                let lines: Vec<&str> = stdout.lines().collect();
                if lines.len() > self.max_output_lines {
                    let truncated: Vec<&str> = lines[..self.max_output_lines].to_vec();
                    stdout = format!(
                        "{}\n... ({} lines truncated)",
                        truncated.join("\n"),
                        lines.len() - self.max_output_lines
                    );
                }

                Ok(CommandExecutionResult {
                    exit_code: output.status.code().unwrap_or(-1),
                    stdout,
                    stderr,
                    timed_out: false,
                    duration_ms,
                })
            }
            Ok(Err(e)) => Err(CognisError::Other(format!(
                "Failed to execute command: {}",
                e
            ))),
            Err(_) => Ok(CommandExecutionResult {
                exit_code: -1,
                stdout: String::new(),
                stderr: "Command timed out".to_string(),
                timed_out: true,
                duration_ms,
            }),
        }
    }
}

#[async_trait]
impl AgentMiddleware for ShellToolMiddleware {
    fn name(&self) -> &str {
        "ShellToolMiddleware"
    }

    async fn before_agent(&self, _state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        // Run startup commands
        for cmd in &self.startup_commands {
            let _ = self.execute(cmd).await?;
        }
        Ok(None)
    }

    async fn after_agent(&self, _state: &AgentState) -> Result<Option<HashMap<String, Value>>> {
        // Run shutdown commands
        for cmd in &self.shutdown_commands {
            let _ = self.execute(cmd).await?;
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shell_tool_new() {
        // Use /tmp which always exists
        let mw = ShellToolMiddleware::new("/tmp");
        assert_eq!(mw.tool_name, "shell");
        assert_eq!(mw.workspace_root, PathBuf::from("/tmp"));
    }

    #[test]
    fn test_shell_tool_new_nonexistent_uses_temp() {
        let mw = ShellToolMiddleware::new("/nonexistent/path/that/does/not/exist");
        assert_eq!(mw.tool_name, "shell");
        // Should have created a temp dir instead
        assert_ne!(
            mw.workspace_root,
            PathBuf::from("/nonexistent/path/that/does/not/exist")
        );
        assert!(mw.workspace_root.exists());
    }

    #[test]
    fn test_shell_tool_builder() {
        let mw = ShellToolMiddleware::new("/tmp")
            .with_tool_name("bash")
            .with_startup_command("echo setup")
            .with_shutdown_command("echo teardown")
            .with_timeout(Duration::from_secs(60))
            .with_env("PATH", "/usr/bin")
            .with_max_output_lines(50);
        assert_eq!(mw.tool_name, "bash");
        assert_eq!(mw.startup_commands.len(), 1);
        assert_eq!(mw.shutdown_commands.len(), 1);
        assert_eq!(mw.timeout, Duration::from_secs(60));
        assert_eq!(mw.env_vars.get("PATH"), Some(&"/usr/bin".to_string()));
        assert_eq!(mw.max_output_lines, 50);
    }

    #[test]
    fn test_is_blocked() {
        let mw = ShellToolMiddleware::new("/tmp");
        assert!(mw.is_blocked("rm -rf /"));
        assert!(mw.is_blocked("sudo rm -rf / --no-preserve-root"));
        assert!(!mw.is_blocked("ls -la"));
        assert!(!mw.is_blocked("echo hello"));
    }

    #[tokio::test]
    async fn test_execute_allowed() {
        let mw = ShellToolMiddleware::new("/tmp");
        let result = mw.execute("echo hello").await.unwrap();
        assert!(result.success());
        assert!(result.stdout.contains("hello"));
    }

    #[tokio::test]
    async fn test_execute_blocked() {
        let mw = ShellToolMiddleware::new("/tmp");
        let result = mw.execute("rm -rf /").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_execute_captures_stderr() {
        let mw = ShellToolMiddleware::new("/tmp");
        let result = mw.execute("echo err >&2").await.unwrap();
        assert!(result.stderr.contains("err"));
    }

    #[tokio::test]
    async fn test_execute_nonzero_exit() {
        let mw = ShellToolMiddleware::new("/tmp");
        let result = mw.execute("exit 42").await.unwrap();
        assert_eq!(result.exit_code, 42);
        assert!(!result.success());
    }

    #[tokio::test]
    async fn test_execute_output_truncation() {
        let mw = ShellToolMiddleware::new("/tmp").with_max_output_lines(3);
        let result = mw.execute("seq 1 10").await.unwrap();
        assert!(result.stdout.contains("truncated"));
    }

    #[tokio::test]
    async fn test_execute_timeout() {
        let mw = ShellToolMiddleware::new("/tmp").with_timeout(Duration::from_millis(100));
        let result = mw.execute("sleep 10").await.unwrap();
        assert!(result.timed_out);
        assert!(!result.success());
    }

    #[test]
    fn test_command_execution_result_success() {
        let result = CommandExecutionResult {
            exit_code: 0,
            stdout: "output".into(),
            stderr: String::new(),
            timed_out: false,
            duration_ms: 100,
        };
        assert!(result.success());
    }

    #[test]
    fn test_command_execution_result_failure() {
        let result = CommandExecutionResult {
            exit_code: 1,
            stdout: String::new(),
            stderr: "error".into(),
            timed_out: false,
            duration_ms: 100,
        };
        assert!(!result.success());
    }

    #[test]
    fn test_command_execution_result_timeout() {
        let result = CommandExecutionResult {
            exit_code: 0,
            stdout: String::new(),
            stderr: String::new(),
            timed_out: true,
            duration_ms: 120_000,
        };
        assert!(!result.success());
    }

    #[test]
    fn test_command_execution_result_serde() {
        let result = CommandExecutionResult {
            exit_code: 0,
            stdout: "hello".into(),
            stderr: String::new(),
            timed_out: false,
            duration_ms: 50,
        };
        let json = serde_json::to_string(&result).unwrap();
        let parsed: CommandExecutionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.exit_code, 0);
        assert_eq!(parsed.stdout, "hello");
    }

    #[test]
    fn test_middleware_name() {
        let mw = ShellToolMiddleware::new("/tmp");
        assert_eq!(mw.name(), "ShellToolMiddleware");
    }
}
