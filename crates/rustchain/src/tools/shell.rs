//! Shell command execution tool.
//!
//! Executes shell commands via `tokio::process::Command` with optional command
//! whitelisting and configurable timeout.

use async_trait::async_trait;
use rustchain_core::error::{Result, RustChainError};
use rustchain_core::tools::base::BaseTool;
use rustchain_core::tools::types::{ToolInput, ToolOutput};
use serde_json::{json, Value};
use std::time::Duration;

/// Execute shell commands safely.
pub struct ShellTool {
    /// Optional whitelist of allowed command prefixes. `None` means allow all.
    pub allowed_commands: Option<Vec<String>>,
    /// Optional working directory for command execution.
    pub working_dir: Option<String>,
    /// Timeout in seconds (default: 30).
    pub timeout_secs: u64,
}

impl Default for ShellTool {
    fn default() -> Self {
        Self {
            allowed_commands: None,
            working_dir: None,
            timeout_secs: 30,
        }
    }
}

#[async_trait]
impl BaseTool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }

    fn description(&self) -> &str {
        "Execute shell commands. Use with caution."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string"
                }
            },
            "required": ["command"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let command = extract_command(&input)?;

        // Validate against allowed commands whitelist
        if let Some(ref allowed) = self.allowed_commands {
            let cmd_trimmed = command.trim();
            let is_allowed = allowed.iter().any(|prefix| {
                cmd_trimmed == prefix.as_str() || cmd_trimmed.starts_with(&format!("{} ", prefix))
            });
            if !is_allowed {
                return Err(RustChainError::ToolException(format!(
                    "Command not allowed: '{command}'. Allowed commands: {allowed:?}"
                )));
            }
        }

        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg(&command);

        if let Some(ref dir) = self.working_dir {
            cmd.current_dir(dir);
        }

        let output = tokio::time::timeout(Duration::from_secs(self.timeout_secs), cmd.output())
            .await
            .map_err(|_| {
                RustChainError::ToolException(format!(
                    "Command timed out after {} seconds",
                    self.timeout_secs
                ))
            })?
            .map_err(|e| {
                RustChainError::ToolException(format!("Failed to execute command: {e}"))
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let combined = if stderr.is_empty() {
            stdout.to_string()
        } else if stdout.is_empty() {
            stderr.to_string()
        } else {
            format!("{stdout}{stderr}")
        };

        Ok(ToolOutput::Content(Value::String(combined)))
    }
}

/// Extract the command string from various input formats.
fn extract_command(input: &ToolInput) -> Result<String> {
    match input {
        ToolInput::Text(s) => Ok(s.clone()),
        ToolInput::Structured(map) => {
            if let Some(Value::String(cmd)) = map.get("command") {
                Ok(cmd.clone())
            } else {
                Err(RustChainError::ToolValidationError(
                    "Missing required field 'command'".into(),
                ))
            }
        }
        ToolInput::ToolCall(tc) => {
            if let Some(Value::String(cmd)) = tc.args.get("command") {
                Ok(cmd.clone())
            } else {
                Err(RustChainError::ToolValidationError(
                    "Missing required field 'command'".into(),
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_shell_echo() {
        let tool = ShellTool::default();
        let input = ToolInput::Structured(
            [(
                "command".to_string(),
                Value::String("echo hello".to_string()),
            )]
            .into_iter()
            .collect(),
        );
        let result = tool._run(input).await.unwrap();
        match result {
            ToolOutput::Content(Value::String(s)) => assert_eq!(s, "hello\n"),
            other => panic!("Expected Content(String), got: {other:?}"),
        }
    }

    #[tokio::test]
    async fn test_shell_with_allowed_commands() {
        let tool = ShellTool {
            allowed_commands: Some(vec!["echo".to_string()]),
            ..Default::default()
        };

        // Allowed command should succeed
        let input = ToolInput::Structured(
            [(
                "command".to_string(),
                Value::String("echo hello".to_string()),
            )]
            .into_iter()
            .collect(),
        );
        let result = tool._run(input).await;
        assert!(result.is_ok());

        // Blocked command should fail
        let input = ToolInput::Structured(
            [("command".to_string(), Value::String("rm -rf /".to_string()))]
                .into_iter()
                .collect(),
        );
        let result = tool._run(input).await;
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("not allowed"),
            "Expected 'not allowed' in error: {err}"
        );
    }

    #[tokio::test]
    async fn test_shell_via_run_json() {
        let tool = ShellTool::default();
        let input = serde_json::json!({"command": "echo test"});
        let result = tool.run_json(&input).await.unwrap();
        assert_eq!(result, Value::String("test\n".to_string()));
    }
}
