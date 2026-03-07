//! Sandbox backend for isolated agent execution.
//!
//! Provides resource limits, permission controls, and execution isolation
//! for running agent tasks in a controlled environment.
//!
//! # Key Types
//!
//! - [`SandboxPermission`] — granular permission flags
//! - [`ResourceLimits`] — memory, CPU, file, and network caps
//! - [`SandboxPolicy`] — combines permissions, path rules, and resource limits
//! - [`Sandbox`] — an isolated execution environment governed by a policy
//! - [`SandboxManager`] — manages multiple named sandboxes

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::Instant;

use serde_json::Value;

// ---------------------------------------------------------------------------
// SandboxPermission
// ---------------------------------------------------------------------------

/// Granular permission flag for sandbox operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SandboxPermission {
    /// Permission to read files.
    FileRead,
    /// Permission to write files.
    FileWrite,
    /// Permission to make network requests.
    NetworkAccess,
    /// Permission to execute shell commands.
    ShellExec,
    /// Permission to access environment variables.
    EnvAccess,
    /// A custom, user-defined permission.
    Custom(String),
}

impl fmt::Display for SandboxPermission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FileRead => write!(f, "FileRead"),
            Self::FileWrite => write!(f, "FileWrite"),
            Self::NetworkAccess => write!(f, "NetworkAccess"),
            Self::ShellExec => write!(f, "ShellExec"),
            Self::EnvAccess => write!(f, "EnvAccess"),
            Self::Custom(name) => write!(f, "Custom({name})"),
        }
    }
}

impl SandboxPermission {
    /// Returns `true` if this permission is considered dangerous.
    ///
    /// Dangerous permissions are those that can modify the system or
    /// exfiltrate data: [`FileWrite`](Self::FileWrite),
    /// [`ShellExec`](Self::ShellExec), and [`NetworkAccess`](Self::NetworkAccess).
    pub fn is_dangerous(&self) -> bool {
        matches!(
            self,
            Self::FileWrite | Self::ShellExec | Self::NetworkAccess
        )
    }
}

// ---------------------------------------------------------------------------
// ResourceLimits
// ---------------------------------------------------------------------------

/// Resource caps for a sandbox execution environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResourceLimits {
    /// Maximum memory in megabytes.
    pub max_memory_mb: Option<u64>,
    /// Maximum CPU time in seconds.
    pub max_cpu_seconds: Option<u64>,
    /// Maximum individual file size in megabytes.
    pub max_file_size_mb: Option<u64>,
    /// Maximum number of concurrently open files.
    pub max_open_files: Option<u32>,
    /// Maximum number of concurrent network connections.
    pub max_network_connections: Option<u32>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: None,
            max_cpu_seconds: None,
            max_file_size_mb: None,
            max_open_files: None,
            max_network_connections: None,
        }
    }
}

impl ResourceLimits {
    /// Set the maximum memory limit in megabytes.
    pub fn with_memory(mut self, mb: u64) -> Self {
        self.max_memory_mb = Some(mb);
        self
    }

    /// Set the maximum CPU time limit in seconds.
    pub fn with_cpu(mut self, seconds: u64) -> Self {
        self.max_cpu_seconds = Some(seconds);
        self
    }

    /// Set the maximum file size limit in megabytes.
    pub fn with_file_size(mut self, mb: u64) -> Self {
        self.max_file_size_mb = Some(mb);
        self
    }

    /// Set the maximum number of open files.
    pub fn with_open_files(mut self, n: u32) -> Self {
        self.max_open_files = Some(n);
        self
    }

    /// Set the maximum number of network connections.
    pub fn with_network(mut self, n: u32) -> Self {
        self.max_network_connections = Some(n);
        self
    }

    /// Returns `true` if all limits are `None` (unlimited).
    pub fn is_unlimited(&self) -> bool {
        self.max_memory_mb.is_none()
            && self.max_cpu_seconds.is_none()
            && self.max_file_size_mb.is_none()
            && self.max_open_files.is_none()
            && self.max_network_connections.is_none()
    }

    /// Serialize the resource limits to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "max_memory_mb": self.max_memory_mb,
            "max_cpu_seconds": self.max_cpu_seconds,
            "max_file_size_mb": self.max_file_size_mb,
            "max_open_files": self.max_open_files,
            "max_network_connections": self.max_network_connections,
        })
    }
}

// ---------------------------------------------------------------------------
// SandboxPolicy
// ---------------------------------------------------------------------------

/// Policy governing what a sandbox is allowed to do.
///
/// By default, a new policy is *restrictive*: no permissions are granted,
/// no paths are allowed, and resource limits are unlimited.
#[derive(Debug, Clone)]
pub struct SandboxPolicy {
    /// Set of granted permissions.
    pub allowed_permissions: HashSet<SandboxPermission>,
    /// Path prefixes that are explicitly denied.
    pub denied_paths: Vec<String>,
    /// Path prefixes that are explicitly allowed.
    pub allowed_paths: Vec<String>,
    /// Resource limits for the sandbox.
    pub resource_limits: ResourceLimits,
}

impl SandboxPolicy {
    /// Create a new restrictive policy with no permissions.
    pub fn new() -> Self {
        Self {
            allowed_permissions: HashSet::new(),
            denied_paths: Vec::new(),
            allowed_paths: Vec::new(),
            resource_limits: ResourceLimits::default(),
        }
    }

    /// Create a permissive policy with all standard permissions granted.
    pub fn permissive() -> Self {
        let mut perms = HashSet::new();
        perms.insert(SandboxPermission::FileRead);
        perms.insert(SandboxPermission::FileWrite);
        perms.insert(SandboxPermission::NetworkAccess);
        perms.insert(SandboxPermission::ShellExec);
        perms.insert(SandboxPermission::EnvAccess);
        Self {
            allowed_permissions: perms,
            denied_paths: Vec::new(),
            allowed_paths: Vec::new(),
            resource_limits: ResourceLimits::default(),
        }
    }

    /// Grant a permission.
    pub fn allow(&mut self, perm: SandboxPermission) {
        self.allowed_permissions.insert(perm);
    }

    /// Revoke a permission.
    pub fn deny(&mut self, perm: SandboxPermission) {
        self.allowed_permissions.remove(&perm);
    }

    /// Add a path prefix to the allowed list.
    pub fn allow_path(&mut self, path: &str) {
        self.allowed_paths.push(path.to_string());
    }

    /// Add a path prefix to the denied list.
    pub fn deny_path(&mut self, path: &str) {
        self.denied_paths.push(path.to_string());
    }

    /// Check whether a permission is granted.
    pub fn check_permission(&self, perm: &SandboxPermission) -> bool {
        self.allowed_permissions.contains(perm)
    }

    /// Check whether a path is accessible.
    ///
    /// A path is accessible if:
    /// 1. It is not under any denied prefix, AND
    /// 2. Either there are no allowed paths (open by default) OR it is under
    ///    an allowed prefix.
    ///
    /// Denied paths always take precedence over allowed paths.
    pub fn check_path(&self, path: &str) -> bool {
        // Denied paths take precedence.
        for denied in &self.denied_paths {
            if path.starts_with(denied.as_str()) {
                return false;
            }
        }
        // If there are allowed paths, the path must match one.
        if self.allowed_paths.is_empty() {
            return true;
        }
        for allowed in &self.allowed_paths {
            if path.starts_with(allowed.as_str()) {
                return true;
            }
        }
        false
    }

    /// Set the resource limits on this policy (builder-style).
    pub fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.resource_limits = limits;
        self
    }
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SandboxStatus
// ---------------------------------------------------------------------------

/// Status of a sandbox or a sandbox execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxStatus {
    /// Ready to accept work.
    Ready,
    /// Currently executing.
    Running,
    /// Execution paused.
    Paused,
    /// Execution completed successfully.
    Completed,
    /// Execution ended with an error.
    Error(String),
    /// Execution was forcefully terminated.
    Terminated,
}

impl SandboxStatus {
    /// Returns `true` if the status represents an active (non-terminal) state.
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Ready | Self::Running | Self::Paused)
    }

    /// Returns `true` if the status represents a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Error(_) | Self::Terminated)
    }
}

impl fmt::Display for SandboxStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ready => write!(f, "Ready"),
            Self::Running => write!(f, "Running"),
            Self::Paused => write!(f, "Paused"),
            Self::Completed => write!(f, "Completed"),
            Self::Error(msg) => write!(f, "Error: {msg}"),
            Self::Terminated => write!(f, "Terminated"),
        }
    }
}

// ---------------------------------------------------------------------------
// SandboxExecution
// ---------------------------------------------------------------------------

/// Tracks a single execution within a sandbox.
#[derive(Debug, Clone)]
pub struct SandboxExecution {
    id: String,
    status: SandboxStatus,
    output: Option<Value>,
    started_at: Option<Instant>,
    completed_at: Option<Instant>,
}

impl SandboxExecution {
    /// Create a new execution with the given identifier.
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            status: SandboxStatus::Ready,
            output: None,
            started_at: None,
            completed_at: None,
        }
    }

    /// Mark the execution as started.
    pub fn start(&mut self) {
        self.status = SandboxStatus::Running;
        self.started_at = Some(Instant::now());
    }

    /// Mark the execution as completed with the given output.
    pub fn complete(&mut self, output: Value) {
        self.status = SandboxStatus::Completed;
        self.output = Some(output);
        self.completed_at = Some(Instant::now());
    }

    /// Mark the execution as failed with an error message.
    pub fn fail(&mut self, error: &str) {
        self.status = SandboxStatus::Error(error.to_string());
        self.completed_at = Some(Instant::now());
    }

    /// Forcefully terminate the execution.
    pub fn terminate(&mut self) {
        self.status = SandboxStatus::Terminated;
        self.completed_at = Some(Instant::now());
    }

    /// Get the current status.
    pub fn status(&self) -> &SandboxStatus {
        &self.status
    }

    /// Get the output, if the execution completed successfully.
    pub fn output(&self) -> Option<&Value> {
        self.output.as_ref()
    }

    /// Get the elapsed duration if the execution has started.
    ///
    /// If the execution is still running, returns the time since start.
    /// If completed/failed/terminated, returns the total duration.
    pub fn duration(&self) -> Option<std::time::Duration> {
        let start = self.started_at?;
        let end = self.completed_at.unwrap_or_else(Instant::now);
        Some(end.duration_since(start))
    }

    /// Get the execution identifier.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Serialize the execution state to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "status": self.status.to_string(),
            "output": self.output,
            "has_started": self.started_at.is_some(),
            "has_completed": self.completed_at.is_some(),
            "duration_ms": self.duration().map(|d| d.as_millis() as u64),
        })
    }
}

// ---------------------------------------------------------------------------
// SandboxError
// ---------------------------------------------------------------------------

/// Errors produced by sandbox operations.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    /// A required permission was not granted.
    #[error("permission denied: {0}")]
    PermissionDenied(String),
    /// A path access was denied by the policy.
    #[error("path access denied: {0}")]
    PathDenied(String),
    /// The sandbox is in an invalid state for the requested operation.
    #[error("invalid sandbox state: {0}")]
    InvalidState(String),
}

/// Convenience alias for sandbox results.
pub type Result<T> = std::result::Result<T, SandboxError>;

// ---------------------------------------------------------------------------
// Sandbox
// ---------------------------------------------------------------------------

/// An isolated execution environment governed by a [`SandboxPolicy`].
#[derive(Debug)]
pub struct Sandbox {
    name: String,
    policy: SandboxPolicy,
    status: SandboxStatus,
    executions: Vec<SandboxExecution>,
    next_execution_id: u64,
}

impl Sandbox {
    /// Create a new sandbox with the given name and policy.
    pub fn new(name: &str, policy: SandboxPolicy) -> Self {
        Self {
            name: name.to_string(),
            policy,
            status: SandboxStatus::Ready,
            executions: Vec::new(),
            next_execution_id: 1,
        }
    }

    /// Execute a task within the sandbox.
    ///
    /// This is currently a mock implementation that immediately completes
    /// and echoes the input back as output alongside the task description.
    pub fn execute(&mut self, task: &str, input: Value) -> Result<SandboxExecution> {
        if !self.status.is_active() {
            return Err(SandboxError::InvalidState(format!(
                "sandbox '{}' is in state: {}",
                self.name, self.status
            )));
        }

        let exec_id = format!("exec-{}", self.next_execution_id);
        self.next_execution_id += 1;

        let mut execution = SandboxExecution::new(&exec_id);
        execution.start();

        // Mock: immediately complete with echoed input.
        let output = serde_json::json!({
            "task": task,
            "input": input,
            "result": "completed",
        });
        execution.complete(output);

        self.executions.push(execution.clone());
        Ok(execution)
    }

    /// Check whether the given permission is granted by this sandbox's policy.
    ///
    /// Returns `Ok(())` if granted, or `Err(SandboxError::PermissionDenied)`.
    pub fn check_permission(&self, perm: &SandboxPermission) -> Result<()> {
        if self.policy.check_permission(perm) {
            Ok(())
        } else {
            Err(SandboxError::PermissionDenied(perm.to_string()))
        }
    }

    /// Validate that a path is accessible according to this sandbox's policy.
    ///
    /// Returns `Ok(())` if accessible, or `Err(SandboxError::PathDenied)`.
    pub fn validate_path(&self, path: &str) -> Result<()> {
        if self.policy.check_path(path) {
            Ok(())
        } else {
            Err(SandboxError::PathDenied(path.to_string()))
        }
    }

    /// Get all executions that have occurred in this sandbox.
    pub fn executions(&self) -> &[SandboxExecution] {
        &self.executions
    }

    /// Get the most recent execution, if any.
    pub fn last_execution(&self) -> Option<&SandboxExecution> {
        self.executions.last()
    }

    /// Get the total number of executions.
    pub fn execution_count(&self) -> usize {
        self.executions.len()
    }

    /// Get the sandbox status.
    pub fn status(&self) -> &SandboxStatus {
        &self.status
    }

    /// Get a reference to the sandbox policy.
    pub fn policy(&self) -> &SandboxPolicy {
        &self.policy
    }

    /// Get the sandbox name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Reset the sandbox, clearing all executions.
    pub fn reset(&mut self) {
        self.executions.clear();
        self.next_execution_id = 1;
        self.status = SandboxStatus::Ready;
    }

    /// Serialize the sandbox state to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "status": self.status.to_string(),
            "execution_count": self.executions.len(),
            "resource_limits": self.policy.resource_limits.to_json(),
        })
    }
}

// ---------------------------------------------------------------------------
// SandboxManager
// ---------------------------------------------------------------------------

/// Manages multiple named sandbox environments.
#[derive(Debug)]
pub struct SandboxManager {
    sandboxes: HashMap<String, Sandbox>,
}

impl SandboxManager {
    /// Create a new, empty sandbox manager.
    pub fn new() -> Self {
        Self {
            sandboxes: HashMap::new(),
        }
    }

    /// Create a new sandbox with the given name and policy.
    ///
    /// If a sandbox with the same name already exists, it is replaced.
    pub fn create(&mut self, name: &str, policy: SandboxPolicy) -> &Sandbox {
        let sandbox = Sandbox::new(name, policy);
        self.sandboxes.insert(name.to_string(), sandbox);
        self.sandboxes.get(name).unwrap()
    }

    /// Get an immutable reference to a sandbox by name.
    pub fn get(&self, name: &str) -> Option<&Sandbox> {
        self.sandboxes.get(name)
    }

    /// Get a mutable reference to a sandbox by name.
    pub fn get_mut(&mut self, name: &str) -> Option<&mut Sandbox> {
        self.sandboxes.get_mut(name)
    }

    /// Remove a sandbox by name, returning `true` if it existed.
    pub fn remove(&mut self, name: &str) -> bool {
        self.sandboxes.remove(name).is_some()
    }

    /// List the names of all managed sandboxes.
    pub fn list(&self) -> Vec<&str> {
        self.sandboxes.keys().map(|s| s.as_str()).collect()
    }

    /// Get the number of managed sandboxes.
    pub fn len(&self) -> usize {
        self.sandboxes.len()
    }

    /// Returns `true` if no sandboxes are managed.
    pub fn is_empty(&self) -> bool {
        self.sandboxes.is_empty()
    }
}

impl Default for SandboxManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SandboxPermission tests --

    #[test]
    fn test_permission_display_file_read() {
        assert_eq!(SandboxPermission::FileRead.to_string(), "FileRead");
    }

    #[test]
    fn test_permission_display_file_write() {
        assert_eq!(SandboxPermission::FileWrite.to_string(), "FileWrite");
    }

    #[test]
    fn test_permission_display_network() {
        assert_eq!(
            SandboxPermission::NetworkAccess.to_string(),
            "NetworkAccess"
        );
    }

    #[test]
    fn test_permission_display_shell() {
        assert_eq!(SandboxPermission::ShellExec.to_string(), "ShellExec");
    }

    #[test]
    fn test_permission_display_env() {
        assert_eq!(SandboxPermission::EnvAccess.to_string(), "EnvAccess");
    }

    #[test]
    fn test_permission_display_custom() {
        let perm = SandboxPermission::Custom("gpu_access".to_string());
        assert_eq!(perm.to_string(), "Custom(gpu_access)");
    }

    #[test]
    fn test_permission_is_dangerous_file_write() {
        assert!(SandboxPermission::FileWrite.is_dangerous());
    }

    #[test]
    fn test_permission_is_dangerous_shell_exec() {
        assert!(SandboxPermission::ShellExec.is_dangerous());
    }

    #[test]
    fn test_permission_is_dangerous_network() {
        assert!(SandboxPermission::NetworkAccess.is_dangerous());
    }

    #[test]
    fn test_permission_is_not_dangerous_file_read() {
        assert!(!SandboxPermission::FileRead.is_dangerous());
    }

    #[test]
    fn test_permission_is_not_dangerous_env() {
        assert!(!SandboxPermission::EnvAccess.is_dangerous());
    }

    #[test]
    fn test_permission_is_not_dangerous_custom() {
        assert!(!SandboxPermission::Custom("anything".to_string()).is_dangerous());
    }

    // -- ResourceLimits tests --

    #[test]
    fn test_resource_limits_default_is_unlimited() {
        let limits = ResourceLimits::default();
        assert!(limits.is_unlimited());
    }

    #[test]
    fn test_resource_limits_builder_memory() {
        let limits = ResourceLimits::default().with_memory(512);
        assert_eq!(limits.max_memory_mb, Some(512));
        assert!(!limits.is_unlimited());
    }

    #[test]
    fn test_resource_limits_builder_cpu() {
        let limits = ResourceLimits::default().with_cpu(60);
        assert_eq!(limits.max_cpu_seconds, Some(60));
        assert!(!limits.is_unlimited());
    }

    #[test]
    fn test_resource_limits_builder_file_size() {
        let limits = ResourceLimits::default().with_file_size(100);
        assert_eq!(limits.max_file_size_mb, Some(100));
    }

    #[test]
    fn test_resource_limits_builder_open_files() {
        let limits = ResourceLimits::default().with_open_files(256);
        assert_eq!(limits.max_open_files, Some(256));
    }

    #[test]
    fn test_resource_limits_builder_network() {
        let limits = ResourceLimits::default().with_network(10);
        assert_eq!(limits.max_network_connections, Some(10));
    }

    #[test]
    fn test_resource_limits_builder_chain() {
        let limits = ResourceLimits::default()
            .with_memory(1024)
            .with_cpu(120)
            .with_file_size(50)
            .with_open_files(128)
            .with_network(5);
        assert_eq!(limits.max_memory_mb, Some(1024));
        assert_eq!(limits.max_cpu_seconds, Some(120));
        assert_eq!(limits.max_file_size_mb, Some(50));
        assert_eq!(limits.max_open_files, Some(128));
        assert_eq!(limits.max_network_connections, Some(5));
        assert!(!limits.is_unlimited());
    }

    #[test]
    fn test_resource_limits_to_json() {
        let limits = ResourceLimits::default().with_memory(256).with_cpu(30);
        let json = limits.to_json();
        assert_eq!(json["max_memory_mb"], 256);
        assert_eq!(json["max_cpu_seconds"], 30);
        assert!(json["max_file_size_mb"].is_null());
    }

    // -- SandboxPolicy tests --

    #[test]
    fn test_policy_new_is_restrictive() {
        let policy = SandboxPolicy::new();
        assert!(policy.allowed_permissions.is_empty());
        assert!(policy.denied_paths.is_empty());
        assert!(policy.allowed_paths.is_empty());
        assert!(policy.resource_limits.is_unlimited());
    }

    #[test]
    fn test_policy_permissive_has_all_standard_permissions() {
        let policy = SandboxPolicy::permissive();
        assert!(policy.check_permission(&SandboxPermission::FileRead));
        assert!(policy.check_permission(&SandboxPermission::FileWrite));
        assert!(policy.check_permission(&SandboxPermission::NetworkAccess));
        assert!(policy.check_permission(&SandboxPermission::ShellExec));
        assert!(policy.check_permission(&SandboxPermission::EnvAccess));
    }

    #[test]
    fn test_policy_permissive_does_not_include_custom() {
        let policy = SandboxPolicy::permissive();
        assert!(!policy.check_permission(&SandboxPermission::Custom("x".to_string())));
    }

    #[test]
    fn test_policy_allow_and_check() {
        let mut policy = SandboxPolicy::new();
        assert!(!policy.check_permission(&SandboxPermission::FileRead));
        policy.allow(SandboxPermission::FileRead);
        assert!(policy.check_permission(&SandboxPermission::FileRead));
    }

    #[test]
    fn test_policy_deny_revokes() {
        let mut policy = SandboxPolicy::permissive();
        assert!(policy.check_permission(&SandboxPermission::ShellExec));
        policy.deny(SandboxPermission::ShellExec);
        assert!(!policy.check_permission(&SandboxPermission::ShellExec));
    }

    #[test]
    fn test_policy_duplicate_allow() {
        let mut policy = SandboxPolicy::new();
        policy.allow(SandboxPermission::FileRead);
        policy.allow(SandboxPermission::FileRead);
        assert!(policy.check_permission(&SandboxPermission::FileRead));
        assert_eq!(policy.allowed_permissions.len(), 1);
    }

    #[test]
    fn test_policy_path_no_rules_allows_all() {
        let policy = SandboxPolicy::new();
        assert!(policy.check_path("/any/path"));
        assert!(policy.check_path(""));
    }

    #[test]
    fn test_policy_allowed_paths_restrict() {
        let mut policy = SandboxPolicy::new();
        policy.allow_path("/home/user");
        assert!(policy.check_path("/home/user/file.txt"));
        assert!(!policy.check_path("/etc/passwd"));
    }

    #[test]
    fn test_policy_denied_paths_block() {
        let mut policy = SandboxPolicy::new();
        policy.deny_path("/etc");
        assert!(!policy.check_path("/etc/passwd"));
        assert!(policy.check_path("/home/user"));
    }

    #[test]
    fn test_policy_denied_takes_precedence() {
        let mut policy = SandboxPolicy::new();
        policy.allow_path("/home");
        policy.deny_path("/home/secret");
        assert!(policy.check_path("/home/user/file.txt"));
        assert!(!policy.check_path("/home/secret/key"));
    }

    #[test]
    fn test_policy_empty_path() {
        let policy = SandboxPolicy::new();
        assert!(policy.check_path(""));
    }

    #[test]
    fn test_policy_with_limits() {
        let limits = ResourceLimits::default().with_memory(128);
        let policy = SandboxPolicy::new().with_limits(limits);
        assert_eq!(policy.resource_limits.max_memory_mb, Some(128));
    }

    // -- SandboxStatus tests --

    #[test]
    fn test_status_is_active() {
        assert!(SandboxStatus::Ready.is_active());
        assert!(SandboxStatus::Running.is_active());
        assert!(SandboxStatus::Paused.is_active());
        assert!(!SandboxStatus::Completed.is_active());
        assert!(!SandboxStatus::Error("fail".to_string()).is_active());
        assert!(!SandboxStatus::Terminated.is_active());
    }

    #[test]
    fn test_status_is_terminal() {
        assert!(!SandboxStatus::Ready.is_terminal());
        assert!(!SandboxStatus::Running.is_terminal());
        assert!(!SandboxStatus::Paused.is_terminal());
        assert!(SandboxStatus::Completed.is_terminal());
        assert!(SandboxStatus::Error("fail".to_string()).is_terminal());
        assert!(SandboxStatus::Terminated.is_terminal());
    }

    #[test]
    fn test_status_display() {
        assert_eq!(SandboxStatus::Ready.to_string(), "Ready");
        assert_eq!(SandboxStatus::Running.to_string(), "Running");
        assert_eq!(SandboxStatus::Paused.to_string(), "Paused");
        assert_eq!(SandboxStatus::Completed.to_string(), "Completed");
        assert_eq!(
            SandboxStatus::Error("boom".to_string()).to_string(),
            "Error: boom"
        );
        assert_eq!(SandboxStatus::Terminated.to_string(), "Terminated");
    }

    // -- SandboxExecution tests --

    #[test]
    fn test_execution_new() {
        let exec = SandboxExecution::new("test-1");
        assert_eq!(exec.id(), "test-1");
        assert_eq!(*exec.status(), SandboxStatus::Ready);
        assert!(exec.output().is_none());
        assert!(exec.duration().is_none());
    }

    #[test]
    fn test_execution_start() {
        let mut exec = SandboxExecution::new("e1");
        exec.start();
        assert_eq!(*exec.status(), SandboxStatus::Running);
        assert!(exec.duration().is_some());
    }

    #[test]
    fn test_execution_complete() {
        let mut exec = SandboxExecution::new("e2");
        exec.start();
        exec.complete(serde_json::json!({"result": "ok"}));
        assert_eq!(*exec.status(), SandboxStatus::Completed);
        assert!(exec.output().is_some());
        assert_eq!(exec.output().unwrap()["result"], "ok");
    }

    #[test]
    fn test_execution_fail() {
        let mut exec = SandboxExecution::new("e3");
        exec.start();
        exec.fail("something went wrong");
        assert_eq!(
            *exec.status(),
            SandboxStatus::Error("something went wrong".to_string())
        );
        assert!(exec.output().is_none());
        assert!(exec.duration().is_some());
    }

    #[test]
    fn test_execution_terminate() {
        let mut exec = SandboxExecution::new("e4");
        exec.start();
        exec.terminate();
        assert_eq!(*exec.status(), SandboxStatus::Terminated);
        assert!(exec.duration().is_some());
    }

    #[test]
    fn test_execution_to_json() {
        let mut exec = SandboxExecution::new("j1");
        exec.start();
        exec.complete(serde_json::json!("done"));
        let json = exec.to_json();
        assert_eq!(json["id"], "j1");
        assert_eq!(json["status"], "Completed");
        assert_eq!(json["has_started"], true);
        assert_eq!(json["has_completed"], true);
    }

    // -- Sandbox tests --

    #[test]
    fn test_sandbox_new() {
        let sb = Sandbox::new("test-sb", SandboxPolicy::new());
        assert_eq!(sb.name(), "test-sb");
        assert_eq!(*sb.status(), SandboxStatus::Ready);
        assert_eq!(sb.execution_count(), 0);
        assert!(sb.last_execution().is_none());
    }

    #[test]
    fn test_sandbox_execute_mock() {
        let mut sb = Sandbox::new("runner", SandboxPolicy::new());
        let exec = sb
            .execute("greet", serde_json::json!({"name": "world"}))
            .unwrap();
        assert_eq!(*exec.status(), SandboxStatus::Completed);
        let out = exec.output().unwrap();
        assert_eq!(out["task"], "greet");
        assert_eq!(out["input"]["name"], "world");
        assert_eq!(sb.execution_count(), 1);
    }

    #[test]
    fn test_sandbox_multiple_executions() {
        let mut sb = Sandbox::new("multi", SandboxPolicy::new());
        sb.execute("a", serde_json::json!(1)).unwrap();
        sb.execute("b", serde_json::json!(2)).unwrap();
        sb.execute("c", serde_json::json!(3)).unwrap();
        assert_eq!(sb.execution_count(), 3);
        assert_eq!(sb.last_execution().unwrap().id(), "exec-3");
    }

    #[test]
    fn test_sandbox_check_permission_granted() {
        let mut policy = SandboxPolicy::new();
        policy.allow(SandboxPermission::FileRead);
        let sb = Sandbox::new("perm", policy);
        assert!(sb.check_permission(&SandboxPermission::FileRead).is_ok());
    }

    #[test]
    fn test_sandbox_check_permission_denied() {
        let sb = Sandbox::new("noperm", SandboxPolicy::new());
        let result = sb.check_permission(&SandboxPermission::FileWrite);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("permission denied"));
    }

    #[test]
    fn test_sandbox_validate_path_allowed() {
        let mut policy = SandboxPolicy::new();
        policy.allow_path("/tmp");
        let sb = Sandbox::new("pathcheck", policy);
        assert!(sb.validate_path("/tmp/file.txt").is_ok());
    }

    #[test]
    fn test_sandbox_validate_path_denied() {
        let mut policy = SandboxPolicy::new();
        policy.allow_path("/tmp");
        let sb = Sandbox::new("pathcheck2", policy);
        let result = sb.validate_path("/etc/secret");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("path access denied"));
    }

    #[test]
    fn test_sandbox_reset() {
        let mut sb = Sandbox::new("reset-test", SandboxPolicy::new());
        sb.execute("task", serde_json::json!(null)).unwrap();
        assert_eq!(sb.execution_count(), 1);
        sb.reset();
        assert_eq!(sb.execution_count(), 0);
        assert!(sb.last_execution().is_none());
        assert_eq!(*sb.status(), SandboxStatus::Ready);
    }

    #[test]
    fn test_sandbox_to_json() {
        let limits = ResourceLimits::default().with_memory(256);
        let policy = SandboxPolicy::new().with_limits(limits);
        let sb = Sandbox::new("json-sb", policy);
        let json = sb.to_json();
        assert_eq!(json["name"], "json-sb");
        assert_eq!(json["status"], "Ready");
        assert_eq!(json["execution_count"], 0);
        assert_eq!(json["resource_limits"]["max_memory_mb"], 256);
    }

    #[test]
    fn test_sandbox_executions_slice() {
        let mut sb = Sandbox::new("slice", SandboxPolicy::new());
        sb.execute("x", serde_json::json!(null)).unwrap();
        let execs = sb.executions();
        assert_eq!(execs.len(), 1);
        assert_eq!(execs[0].id(), "exec-1");
    }

    // -- SandboxManager tests --

    #[test]
    fn test_manager_new_is_empty() {
        let mgr = SandboxManager::new();
        assert!(mgr.is_empty());
        assert_eq!(mgr.len(), 0);
    }

    #[test]
    fn test_manager_create_and_get() {
        let mut mgr = SandboxManager::new();
        mgr.create("sb1", SandboxPolicy::new());
        assert_eq!(mgr.len(), 1);
        assert!(!mgr.is_empty());
        let sb = mgr.get("sb1").unwrap();
        assert_eq!(sb.name(), "sb1");
    }

    #[test]
    fn test_manager_get_mut() {
        let mut mgr = SandboxManager::new();
        mgr.create("sb2", SandboxPolicy::new());
        let sb = mgr.get_mut("sb2").unwrap();
        sb.execute("test", serde_json::json!(null)).unwrap();
        assert_eq!(mgr.get("sb2").unwrap().execution_count(), 1);
    }

    #[test]
    fn test_manager_remove() {
        let mut mgr = SandboxManager::new();
        mgr.create("to-remove", SandboxPolicy::new());
        assert!(mgr.remove("to-remove"));
        assert!(mgr.get("to-remove").is_none());
        assert!(!mgr.remove("to-remove"));
    }

    #[test]
    fn test_manager_list() {
        let mut mgr = SandboxManager::new();
        mgr.create("alpha", SandboxPolicy::new());
        mgr.create("beta", SandboxPolicy::new());
        let mut names = mgr.list();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn test_manager_get_nonexistent() {
        let mgr = SandboxManager::new();
        assert!(mgr.get("nope").is_none());
    }

    #[test]
    fn test_manager_replace_existing() {
        let mut mgr = SandboxManager::new();
        let mut policy = SandboxPolicy::new();
        policy.allow(SandboxPermission::FileRead);
        mgr.create("dup", policy);

        // Replace with a different policy.
        mgr.create("dup", SandboxPolicy::permissive());
        assert_eq!(mgr.len(), 1);
        let sb = mgr.get("dup").unwrap();
        assert!(sb.policy().check_permission(&SandboxPermission::ShellExec));
    }
}
