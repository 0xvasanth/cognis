//! Execution policies for persistent shell sessions.
//!
//! Mirrors Python `langchain.agents.middleware._execution`.
//!
//! Defines how a shell process is launched and constrained.
//! - `HostExecutionPolicy` — run on the host with optional resource limits.
//! - `CodexSandboxExecutionPolicy` — run through the Codex CLI sandbox.
//! - `DockerExecutionPolicy` — run inside a Docker container.

use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// Prefix used for temporary shell workspace directories.
pub const SHELL_TEMP_PREFIX: &str = "langchain-shell-";

/// Base execution policy configuration shared by all policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionPolicyConfig {
    /// Maximum time (seconds) for a single command.
    pub command_timeout: f64,
    /// Maximum time (seconds) for shell startup.
    pub startup_timeout: f64,
    /// Maximum time (seconds) to wait for graceful termination.
    pub termination_timeout: f64,
    /// Maximum number of output lines to capture.
    pub max_output_lines: usize,
    /// Optional maximum output size in bytes.
    pub max_output_bytes: Option<usize>,
}

impl Default for ExecutionPolicyConfig {
    fn default() -> Self {
        Self {
            command_timeout: 30.0,
            startup_timeout: 30.0,
            termination_timeout: 10.0,
            max_output_lines: 100,
            max_output_bytes: None,
        }
    }
}

impl ExecutionPolicyConfig {
    /// Validate that the configuration is consistent.
    pub fn validate(&self) -> Result<(), String> {
        if self.max_output_lines == 0 {
            return Err("max_output_lines must be positive".into());
        }
        Ok(())
    }
}

/// Trait for execution policies.
///
/// Each policy defines how a shell process is spawned (command construction,
/// environment, resource limits). Actual process spawning uses `tokio::process::Command`.
#[async_trait]
pub trait ExecutionPolicy: Send + Sync {
    /// The base configuration for this policy.
    fn config(&self) -> &ExecutionPolicyConfig;

    /// Build the full command line for launching the shell.
    ///
    /// Takes the user-provided command and returns the fully-qualified
    /// command arguments (which may wrap the original command in a sandbox
    /// or container invocation).
    fn build_command(
        &self,
        command: &[String],
        workspace: &std::path::Path,
        env: &HashMap<String, String>,
    ) -> Result<Vec<String>, String>;

    /// Spawn a command using `tokio::process::Command`.
    ///
    /// Builds the full command via `build_command` then executes it.
    /// The default implementation works for all policies.
    async fn spawn_command(
        &self,
        command: &[String],
        workspace: &std::path::Path,
        env: &HashMap<String, String>,
    ) -> Result<std::process::Output, String> {
        let full_cmd = self.build_command(command, workspace, env)?;
        if full_cmd.is_empty() {
            return Err("Empty command".into());
        }
        let output = tokio::process::Command::new(&full_cmd[0])
            .args(&full_cmd[1..])
            .current_dir(workspace)
            .envs(env)
            .output()
            .await
            .map_err(|e| e.to_string())?;
        Ok(output)
    }
}

/// Run the shell directly on the host process.
///
/// Best suited for trusted or single-tenant environments (CI jobs,
/// developer workstations). Enforces optional CPU and memory limits
/// but offers no filesystem or network sandboxing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostExecutionPolicy {
    /// Base config.
    #[serde(flatten)]
    pub config: ExecutionPolicyConfig,
    /// Optional CPU time limit in seconds.
    pub cpu_time_seconds: Option<u64>,
    /// Optional memory limit in bytes.
    pub memory_bytes: Option<u64>,
    /// Whether to create a new process group.
    pub create_process_group: bool,
}

impl Default for HostExecutionPolicy {
    fn default() -> Self {
        Self {
            config: ExecutionPolicyConfig::default(),
            cpu_time_seconds: None,
            memory_bytes: None,
            create_process_group: true,
        }
    }
}

impl HostExecutionPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_cpu_limit(mut self, seconds: u64) -> Self {
        self.cpu_time_seconds = Some(seconds);
        self
    }

    pub fn with_memory_limit(mut self, bytes: u64) -> Self {
        self.memory_bytes = Some(bytes);
        self
    }

    pub fn with_config(mut self, config: ExecutionPolicyConfig) -> Self {
        self.config = config;
        self
    }

    /// Validate policy-specific constraints.
    pub fn validate(&self) -> Result<(), String> {
        self.config.validate()?;
        if let Some(cpu) = self.cpu_time_seconds {
            if cpu == 0 {
                return Err("cpu_time_seconds must be positive if provided".into());
            }
        }
        if let Some(mem) = self.memory_bytes {
            if mem == 0 {
                return Err("memory_bytes must be positive if provided".into());
            }
        }
        Ok(())
    }
}

#[async_trait]
impl ExecutionPolicy for HostExecutionPolicy {
    fn config(&self) -> &ExecutionPolicyConfig {
        &self.config
    }

    fn build_command(
        &self,
        command: &[String],
        _workspace: &std::path::Path,
        _env: &HashMap<String, String>,
    ) -> Result<Vec<String>, String> {
        // Host policy runs commands directly
        Ok(command.to_vec())
    }
}

/// Launch the shell through the Codex CLI sandbox.
///
/// Uses Anthropic's Seatbelt (macOS) or Landlock/seccomp (Linux) profiles
/// for additional syscall and filesystem restrictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodexSandboxExecutionPolicy {
    /// Base config.
    #[serde(flatten)]
    pub config: ExecutionPolicyConfig,
    /// Path to the codex binary.
    pub binary: String,
    /// Platform selection: "auto", "macos", or "linux".
    pub platform: String,
    /// Additional config overrides passed to the sandbox.
    pub config_overrides: HashMap<String, String>,
}

impl Default for CodexSandboxExecutionPolicy {
    fn default() -> Self {
        Self {
            config: ExecutionPolicyConfig::default(),
            binary: "codex".into(),
            platform: "auto".into(),
            config_overrides: HashMap::new(),
        }
    }
}

impl CodexSandboxExecutionPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_binary(mut self, binary: impl Into<String>) -> Self {
        self.binary = binary.into();
        self
    }

    pub fn with_platform(mut self, platform: impl Into<String>) -> Self {
        self.platform = platform.into();
        self
    }

    pub fn with_config_override(
        mut self,
        key: impl Into<String>,
        value: impl Into<String>,
    ) -> Self {
        self.config_overrides.insert(key.into(), value.into());
        self
    }

    /// Determine the platform argument.
    fn determine_platform(&self) -> Result<String, String> {
        if self.platform != "auto" {
            return Ok(self.platform.clone());
        }
        #[cfg(target_os = "linux")]
        return Ok("linux".into());
        #[cfg(target_os = "macos")]
        return Ok("macos".into());
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        return Err(
            "Codex sandbox policy could not determine a supported platform; set 'platform' explicitly".into(),
        );
    }
}

#[async_trait]
impl ExecutionPolicy for CodexSandboxExecutionPolicy {
    fn config(&self) -> &ExecutionPolicyConfig {
        &self.config
    }

    fn build_command(
        &self,
        command: &[String],
        _workspace: &std::path::Path,
        _env: &HashMap<String, String>,
    ) -> Result<Vec<String>, String> {
        let platform_arg = self.determine_platform()?;
        let mut full_command = vec![
            self.binary.clone(),
            "sandbox".into(),
            platform_arg,
        ];
        // Add config overrides in sorted order
        let mut sorted_overrides: Vec<_> = self.config_overrides.iter().collect();
        sorted_overrides.sort_by_key(|(k, _)| (*k).clone());
        for (key, value) in sorted_overrides {
            full_command.push("-c".into());
            full_command.push(format!("{key}={value}"));
        }
        full_command.push("--".into());
        full_command.extend(command.iter().cloned());
        Ok(full_command)
    }
}

/// Run the shell inside a dedicated Docker container.
///
/// Choose this policy when commands originate from untrusted users or you require
/// strong isolation between sessions. The container's network is disabled by default.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DockerExecutionPolicy {
    /// Base config.
    #[serde(flatten)]
    pub config: ExecutionPolicyConfig,
    /// Path to the docker binary.
    pub binary: String,
    /// Docker image to use.
    pub image: String,
    /// Remove container on exit.
    pub remove_container_on_exit: bool,
    /// Whether to enable networking.
    pub network_enabled: bool,
    /// Additional docker run arguments.
    pub extra_run_args: Vec<String>,
    /// Optional memory limit in bytes.
    pub memory_bytes: Option<u64>,
    /// Optional CPU limit string (e.g., "0.5").
    pub cpus: Option<String>,
    /// Whether to make the root filesystem read-only.
    pub read_only_rootfs: bool,
    /// Optional user to run as inside the container.
    pub user: Option<String>,
}

impl Default for DockerExecutionPolicy {
    fn default() -> Self {
        Self {
            config: ExecutionPolicyConfig::default(),
            binary: "docker".into(),
            image: "python:3.12-alpine3.19".into(),
            remove_container_on_exit: true,
            network_enabled: false,
            extra_run_args: Vec::new(),
            memory_bytes: None,
            cpus: None,
            read_only_rootfs: false,
            user: None,
        }
    }
}

impl DockerExecutionPolicy {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_image(mut self, image: impl Into<String>) -> Self {
        self.image = image.into();
        self
    }

    pub fn with_network(mut self, enabled: bool) -> Self {
        self.network_enabled = enabled;
        self
    }

    pub fn with_memory_limit(mut self, bytes: u64) -> Self {
        self.memory_bytes = Some(bytes);
        self
    }

    pub fn with_cpus(mut self, cpus: impl Into<String>) -> Self {
        self.cpus = Some(cpus.into());
        self
    }

    pub fn with_read_only_rootfs(mut self, read_only: bool) -> Self {
        self.read_only_rootfs = read_only;
        self
    }

    pub fn with_user(mut self, user: impl Into<String>) -> Self {
        self.user = Some(user.into());
        self
    }

    pub fn with_extra_run_arg(mut self, arg: impl Into<String>) -> Self {
        self.extra_run_args.push(arg.into());
        self
    }

    /// Validate policy-specific constraints.
    pub fn validate(&self) -> Result<(), String> {
        self.config.validate()?;
        if let Some(mem) = self.memory_bytes {
            if mem == 0 {
                return Err("memory_bytes must be positive if provided".into());
            }
        }
        if let Some(ref cpus) = self.cpus {
            if cpus.trim().is_empty() {
                return Err("cpus must be a non-empty string when provided".into());
            }
        }
        if let Some(ref user) = self.user {
            if user.trim().is_empty() {
                return Err("user must be a non-empty string when provided".into());
            }
        }
        Ok(())
    }

    /// Check whether the workspace should be mounted into the container.
    fn should_mount_workspace(workspace: &std::path::Path) -> bool {
        workspace
            .file_name()
            .and_then(|n| n.to_str())
            .map(|n| !n.starts_with(SHELL_TEMP_PREFIX))
            .unwrap_or(false)
    }
}

#[async_trait]
impl ExecutionPolicy for DockerExecutionPolicy {
    fn config(&self) -> &ExecutionPolicyConfig {
        &self.config
    }

    fn build_command(
        &self,
        command: &[String],
        workspace: &std::path::Path,
        env: &HashMap<String, String>,
    ) -> Result<Vec<String>, String> {
        let mut full_command = vec![self.binary.clone(), "run".into(), "-i".into()];
        if self.remove_container_on_exit {
            full_command.push("--rm".into());
        }
        if !self.network_enabled {
            full_command.extend(["--network".into(), "none".into()]);
        }
        if let Some(mem) = self.memory_bytes {
            full_command.extend(["--memory".into(), mem.to_string()]);
        }
        if Self::should_mount_workspace(workspace) {
            let host_path = workspace.to_string_lossy().to_string();
            full_command.extend(["-v".into(), format!("{host_path}:{host_path}")]);
            full_command.extend(["-w".into(), host_path]);
        } else {
            full_command.extend(["-w".into(), "/".into()]);
        }
        if self.read_only_rootfs {
            full_command.push("--read-only".into());
        }
        for (key, value) in env {
            full_command.extend(["-e".into(), format!("{key}={value}")]);
        }
        if let Some(ref cpus) = self.cpus {
            full_command.extend(["--cpus".into(), cpus.clone()]);
        }
        if let Some(ref user) = self.user {
            full_command.extend(["--user".into(), user.clone()]);
        }
        for arg in &self.extra_run_args {
            full_command.push(arg.clone());
        }
        full_command.push(self.image.clone());
        full_command.extend(command.iter().cloned());
        Ok(full_command)
    }
}

/// Enum wrapping all execution policy types for convenience.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExecutionPolicyKind {
    Host(HostExecutionPolicy),
    CodexSandbox(CodexSandboxExecutionPolicy),
    Docker(DockerExecutionPolicy),
}

impl ExecutionPolicyKind {
    /// Get a reference to the inner policy as a trait object.
    pub fn as_policy(&self) -> &dyn ExecutionPolicy {
        match self {
            ExecutionPolicyKind::Host(p) => p,
            ExecutionPolicyKind::CodexSandbox(p) => p,
            ExecutionPolicyKind::Docker(p) => p,
        }
    }

    /// Validate the policy.
    pub fn validate(&self) -> Result<(), String> {
        match self {
            ExecutionPolicyKind::Host(p) => p.validate(),
            ExecutionPolicyKind::CodexSandbox(p) => p.config.validate(),
            ExecutionPolicyKind::Docker(p) => p.validate(),
        }
    }
}

impl Default for ExecutionPolicyKind {
    fn default() -> Self {
        ExecutionPolicyKind::Host(HostExecutionPolicy::default())
    }
}

/// Helper to integrate an execution policy with the `ShellToolMiddleware`.
///
/// Given a policy and a workspace, produces the full spawn command.
pub fn build_spawn_command(
    policy: &dyn ExecutionPolicy,
    shell_command: &[String],
    workspace: &std::path::Path,
    env: &HashMap<String, String>,
) -> Result<Vec<String>, String> {
    policy.build_command(shell_command, workspace, env)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_policy_config_default() {
        let config = ExecutionPolicyConfig::default();
        assert_eq!(config.command_timeout, 30.0);
        assert_eq!(config.max_output_lines, 100);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_execution_policy_config_invalid() {
        let config = ExecutionPolicyConfig {
            max_output_lines: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_host_policy_default() {
        let policy = HostExecutionPolicy::default();
        assert!(policy.cpu_time_seconds.is_none());
        assert!(policy.memory_bytes.is_none());
        assert!(policy.create_process_group);
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn test_host_policy_builder() {
        let policy = HostExecutionPolicy::new()
            .with_cpu_limit(60)
            .with_memory_limit(1024 * 1024 * 512);
        assert_eq!(policy.cpu_time_seconds, Some(60));
        assert_eq!(policy.memory_bytes, Some(512 * 1024 * 1024));
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn test_host_policy_invalid_cpu() {
        let policy = HostExecutionPolicy {
            cpu_time_seconds: Some(0),
            ..Default::default()
        };
        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_host_policy_invalid_memory() {
        let policy = HostExecutionPolicy {
            memory_bytes: Some(0),
            ..Default::default()
        };
        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_host_policy_build_command() {
        let policy = HostExecutionPolicy::default();
        let cmd = vec!["bash".into(), "-c".into(), "echo hello".into()];
        let result = policy
            .build_command(&cmd, std::path::Path::new("/tmp"), &HashMap::new())
            .unwrap();
        assert_eq!(result, cmd);
    }

    #[test]
    fn test_codex_sandbox_default() {
        let policy = CodexSandboxExecutionPolicy::default();
        assert_eq!(policy.binary, "codex");
        assert_eq!(policy.platform, "auto");
    }

    #[test]
    fn test_codex_sandbox_builder() {
        let policy = CodexSandboxExecutionPolicy::new()
            .with_binary("codex-cli")
            .with_platform("linux")
            .with_config_override("key", "value");
        assert_eq!(policy.binary, "codex-cli");
        assert_eq!(policy.platform, "linux");
        assert_eq!(policy.config_overrides.get("key"), Some(&"value".into()));
    }

    #[test]
    fn test_codex_sandbox_build_command() {
        let policy = CodexSandboxExecutionPolicy::new().with_platform("linux");
        let cmd = vec!["bash".into()];
        let result = policy
            .build_command(&cmd, std::path::Path::new("/tmp"), &HashMap::new())
            .unwrap();
        assert_eq!(result[0], "codex");
        assert_eq!(result[1], "sandbox");
        assert_eq!(result[2], "linux");
        assert!(result.contains(&"--".to_string()));
        assert!(result.contains(&"bash".to_string()));
    }

    #[test]
    fn test_codex_sandbox_build_command_with_overrides() {
        let policy = CodexSandboxExecutionPolicy::new()
            .with_platform("macos")
            .with_config_override("network", "false")
            .with_config_override("allow_write", "true");
        let cmd = vec!["sh".into()];
        let result = policy
            .build_command(&cmd, std::path::Path::new("/tmp"), &HashMap::new())
            .unwrap();
        // Overrides are sorted by key
        let c_positions: Vec<usize> = result
            .iter()
            .enumerate()
            .filter(|(_, v)| *v == "-c")
            .map(|(i, _)| i)
            .collect();
        assert_eq!(c_positions.len(), 2);
        assert_eq!(result[c_positions[0] + 1], "allow_write=true");
        assert_eq!(result[c_positions[1] + 1], "network=false");
    }

    #[test]
    fn test_docker_policy_default() {
        let policy = DockerExecutionPolicy::default();
        assert_eq!(policy.image, "python:3.12-alpine3.19");
        assert!(policy.remove_container_on_exit);
        assert!(!policy.network_enabled);
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn test_docker_policy_builder() {
        let policy = DockerExecutionPolicy::new()
            .with_image("ubuntu:24.04")
            .with_network(true)
            .with_memory_limit(1024 * 1024 * 256)
            .with_cpus("0.5")
            .with_read_only_rootfs(true)
            .with_user("nobody")
            .with_extra_run_arg("--cap-drop=ALL");
        assert_eq!(policy.image, "ubuntu:24.04");
        assert!(policy.network_enabled);
        assert_eq!(policy.memory_bytes, Some(256 * 1024 * 1024));
        assert_eq!(policy.cpus.as_deref(), Some("0.5"));
        assert!(policy.read_only_rootfs);
        assert_eq!(policy.user.as_deref(), Some("nobody"));
        assert_eq!(policy.extra_run_args, vec!["--cap-drop=ALL"]);
        assert!(policy.validate().is_ok());
    }

    #[test]
    fn test_docker_policy_invalid_memory() {
        let policy = DockerExecutionPolicy {
            memory_bytes: Some(0),
            ..Default::default()
        };
        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_docker_policy_invalid_cpus() {
        let policy = DockerExecutionPolicy {
            cpus: Some("  ".into()),
            ..Default::default()
        };
        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_docker_policy_invalid_user() {
        let policy = DockerExecutionPolicy {
            user: Some("".into()),
            ..Default::default()
        };
        assert!(policy.validate().is_err());
    }

    #[test]
    fn test_docker_policy_build_command() {
        let policy = DockerExecutionPolicy::default();
        let cmd = vec!["bash".into()];
        let result = policy
            .build_command(&cmd, std::path::Path::new("/workspace"), &HashMap::new())
            .unwrap();
        assert!(result.contains(&"docker".to_string()));
        assert!(result.contains(&"run".to_string()));
        assert!(result.contains(&"--rm".to_string()));
        assert!(result.contains(&"--network".to_string()));
        assert!(result.contains(&"none".to_string()));
        assert!(result.contains(&"python:3.12-alpine3.19".to_string()));
        assert!(result.contains(&"bash".to_string()));
    }

    #[test]
    fn test_docker_should_mount_workspace() {
        assert!(DockerExecutionPolicy::should_mount_workspace(
            std::path::Path::new("/my/workspace")
        ));
        assert!(!DockerExecutionPolicy::should_mount_workspace(
            std::path::Path::new("/tmp/langchain-shell-abc123")
        ));
    }

    #[test]
    fn test_docker_policy_with_env() {
        let policy = DockerExecutionPolicy::default();
        let cmd = vec!["bash".into()];
        let mut env = HashMap::new();
        env.insert("FOO".into(), "bar".into());
        let result = policy
            .build_command(&cmd, std::path::Path::new("/workspace"), &env)
            .unwrap();
        assert!(result.contains(&"-e".to_string()));
        assert!(result.contains(&"FOO=bar".to_string()));
    }

    #[test]
    fn test_docker_policy_read_only() {
        let policy = DockerExecutionPolicy::new().with_read_only_rootfs(true);
        let cmd = vec!["bash".into()];
        let result = policy
            .build_command(&cmd, std::path::Path::new("/workspace"), &HashMap::new())
            .unwrap();
        assert!(result.contains(&"--read-only".to_string()));
    }

    #[test]
    fn test_execution_policy_kind_default() {
        let kind = ExecutionPolicyKind::default();
        matches!(kind, ExecutionPolicyKind::Host(_));
        assert!(kind.validate().is_ok());
    }

    #[test]
    fn test_execution_policy_kind_serde() {
        let kind = ExecutionPolicyKind::Docker(DockerExecutionPolicy::default());
        let json = serde_json::to_string(&kind).unwrap();
        assert!(json.contains("\"type\":\"docker\""));
        let parsed: ExecutionPolicyKind = serde_json::from_str(&json).unwrap();
        matches!(parsed, ExecutionPolicyKind::Docker(_));
    }

    #[test]
    fn test_build_spawn_command() {
        let policy = HostExecutionPolicy::default();
        let cmd = vec!["bash".into()];
        let workspace = std::path::Path::new("/tmp");
        let result = build_spawn_command(&policy, &cmd, workspace, &HashMap::new()).unwrap();
        assert_eq!(result, vec!["bash"]);
    }

    #[tokio::test]
    async fn test_host_spawn_command() {
        let policy = HostExecutionPolicy::default();
        let cmd = vec!["echo".into(), "hello".into()];
        let workspace = std::path::Path::new("/tmp");
        let output = policy
            .spawn_command(&cmd, workspace, &HashMap::new())
            .await
            .unwrap();
        assert!(output.status.success());
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.contains("hello"));
    }

    #[tokio::test]
    async fn test_spawn_command_empty() {
        let policy = HostExecutionPolicy::default();
        let cmd: Vec<String> = vec![];
        let workspace = std::path::Path::new("/tmp");
        let result = policy
            .spawn_command(&cmd, workspace, &HashMap::new())
            .await;
        assert!(result.is_err());
    }
}
