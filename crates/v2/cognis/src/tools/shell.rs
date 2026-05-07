//! Sandboxed shell tool.
//!
//! Refuses any command whose program (the first token) isn't on a
//! caller-supplied allowlist. The agent never sees the raw shell — every
//! invocation goes through `Command` with explicit args, `cwd`, and timeout.
//!
//! # Why no shell parsing?
//!
//! Shell-string parsing is a footgun: quoting, globbing, and pipelines all
//! invite injection. The tool takes a structured `program + args` JSON
//! payload so the LLM can't slip a `; rm -rf /` past us.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use cognis2_core::schemars::{self, JsonSchema};
use serde::Deserialize;
use tokio::process::Command;

use cognis2_core::{CognisError, Result};
use cognis2_llm::tools::{Tool, ToolInput, ToolOutput};

/// Input shape: `{ "program": "ls", "args": ["-la"] }`.
#[derive(Debug, Deserialize, JsonSchema)]
struct ShellInput {
    /// Program name (must appear on the tool's allowlist).
    program: String,
    /// Arguments passed to the program. No shell parsing is performed.
    #[serde(default)]
    args: Vec<String>,
}

/// Sandboxed shell tool.
///
/// Construction requires:
/// - An explicit allowlist of program names.
/// - A working directory (relative to which `cwd` is resolved).
/// - A per-call timeout.
pub struct ShellTool {
    allowed: Vec<String>,
    cwd: PathBuf,
    timeout: Duration,
}

impl ShellTool {
    /// Build a shell tool. `allowed` is the *exact* set of program names
    /// the agent may run — typically things like `["ls", "cat", "rg"]`.
    /// Programs are not resolved against `$PATH` for the allowlist check —
    /// the value `"ls"` matches both `/usr/bin/ls` and `/bin/ls` because
    /// the program field is compared as-is.
    pub fn new(
        allowed: impl IntoIterator<Item = impl Into<String>>,
        cwd: impl Into<PathBuf>,
        timeout: Duration,
    ) -> Self {
        Self {
            allowed: allowed.into_iter().map(Into::into).collect(),
            cwd: cwd.into(),
            timeout,
        }
    }

    /// Wrap behind an `Arc<dyn Tool>` for registration.
    pub fn into_arc(self) -> Arc<dyn Tool> {
        Arc::new(self)
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn name(&self) -> &str {
        "shell"
    }
    fn description(&self) -> &str {
        "Run a sandboxed shell command. The `program` must be on the tool's \
         allowlist. Arguments are passed without shell parsing."
    }
    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::to_value(schemars::schema_for!(ShellInput)).unwrap_or_default())
    }
    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let parsed: ShellInput = serde_json::from_value(input.into_json())
            .map_err(|e| CognisError::ToolValidationError(format!("shell: {e}")))?;

        if !self.allowed.iter().any(|p| p == &parsed.program) {
            return Err(CognisError::Tool {
                name: "shell".into(),
                reason: format!("program `{}` not on allowlist", parsed.program),
            });
        }

        let mut cmd = Command::new(&parsed.program);
        cmd.args(&parsed.args);
        cmd.current_dir(&self.cwd);
        cmd.kill_on_drop(true);

        let fut = cmd.output();
        let output = tokio::time::timeout(self.timeout, fut)
            .await
            .map_err(|_| CognisError::Timeout {
                operation: format!("shell:{}", parsed.program),
                timeout_ms: self.timeout.as_millis() as u64,
            })?
            .map_err(|e| CognisError::Tool {
                name: "shell".into(),
                reason: format!("spawn `{}`: {e}", parsed.program),
            })?;

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();

        Ok(ToolOutput::Content(serde_json::json!({
            "exit_code": output.status.code(),
            "stdout": stdout,
            "stderr": stderr,
        })))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn refuses_program_not_on_allowlist() {
        let t = ShellTool::new(["echo"], ".", Duration::from_secs(2));
        let mut a = std::collections::HashMap::new();
        a.insert("program".into(), json!("rm"));
        a.insert("args".into(), json!(["-rf", "/"]));
        let err = t._run(ToolInput::Structured(a)).await.unwrap_err();
        assert!(matches!(err, CognisError::Tool { .. }));
    }

    #[tokio::test]
    async fn runs_allowed_program() {
        let t = ShellTool::new(["echo"], ".", Duration::from_secs(5));
        let mut a = std::collections::HashMap::new();
        a.insert("program".into(), json!("echo"));
        a.insert("args".into(), json!(["hello"]));
        let out = t._run(ToolInput::Structured(a)).await.unwrap();
        let v: serde_json::Value = match out {
            ToolOutput::Content(v) => v,
            _ => panic!(),
        };
        assert!(v["stdout"].as_str().unwrap().contains("hello"));
        assert_eq!(v["exit_code"], 0);
    }

    #[tokio::test]
    async fn times_out_on_long_running_command() {
        let t = ShellTool::new(["sleep"], ".", Duration::from_millis(50));
        let mut a = std::collections::HashMap::new();
        a.insert("program".into(), json!("sleep"));
        a.insert("args".into(), json!(["5"]));
        let err = t._run(ToolInput::Structured(a)).await.unwrap_err();
        assert!(matches!(err, CognisError::Timeout { .. }));
    }
}
