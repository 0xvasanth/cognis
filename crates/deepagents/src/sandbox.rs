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
#[derive(Debug, Clone, Default, PartialEq, Eq)]
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
// PathPattern (glob-like path matching)
// ---------------------------------------------------------------------------

/// Glob-like path pattern supporting `*` (single segment) and `**` (recursive).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PathPattern {
    pattern: String,
}

impl PathPattern {
    /// Create a new path pattern from a glob string.
    pub fn new(pattern: &str) -> Self {
        Self {
            pattern: pattern.to_string(),
        }
    }

    /// Return the underlying pattern string.
    pub fn pattern(&self) -> &str {
        &self.pattern
    }

    /// Test whether `path` matches this pattern.
    ///
    /// Matching rules:
    /// - `*` matches exactly one path segment (no `/`)
    /// - `**` matches zero or more segments (including `/`)
    /// - Literal segments must match exactly
    pub fn matches(&self, path: &str) -> bool {
        let pat_parts: Vec<&str> = self.pattern.split('/').filter(|s| !s.is_empty()).collect();
        let path_parts: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
        Self::match_parts(&pat_parts, &path_parts)
    }

    fn match_parts(pat: &[&str], path: &[&str]) -> bool {
        if pat.is_empty() {
            return path.is_empty();
        }
        if pat[0] == "**" {
            let rest_pat = &pat[1..];
            for i in 0..=path.len() {
                if Self::match_parts(rest_pat, &path[i..]) {
                    return true;
                }
            }
            return false;
        }
        if path.is_empty() {
            return false;
        }
        let seg_match = pat[0] == "*" || pat[0] == path[0];
        seg_match && Self::match_parts(&pat[1..], &path[1..])
    }
}

impl fmt::Display for PathPattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.pattern)
    }
}

// ---------------------------------------------------------------------------
// Permission (fine-grained)
// ---------------------------------------------------------------------------

/// A fine-grained permission controlling a specific kind of resource access.
#[derive(Debug, Clone, PartialEq)]
pub enum Permission {
    /// Permission to read files matching a path pattern.
    FileRead(PathPattern),
    /// Permission to write files matching a path pattern.
    FileWrite(PathPattern),
    /// Permission to access specific network hosts.
    NetworkAccess { hosts: Vec<String> },
    /// Permission to execute specific shell commands.
    ShellExec { commands: Vec<String> },
    /// Permission to use specific tools.
    ToolUse { tools: Vec<String> },
    /// Permission to spawn sub-agents.
    SubAgentSpawn,
    /// Unrestricted access to everything.
    All,
}

impl Permission {
    /// Human-readable name for this permission variant.
    pub fn name(&self) -> &str {
        match self {
            Permission::FileRead(_) => "file_read",
            Permission::FileWrite(_) => "file_write",
            Permission::NetworkAccess { .. } => "network_access",
            Permission::ShellExec { .. } => "shell_exec",
            Permission::ToolUse { .. } => "tool_use",
            Permission::SubAgentSpawn => "sub_agent_spawn",
            Permission::All => "all",
        }
    }

    /// For file-based permissions, test whether `path` matches the pattern.
    /// Returns `false` for non-file permissions.
    pub fn matches_path(&self, path: &str) -> bool {
        match self {
            Permission::FileRead(pat) | Permission::FileWrite(pat) => pat.matches(path),
            Permission::All => true,
            _ => false,
        }
    }

    /// Serialize this permission to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "type": self.name(),
            "display": self.to_string(),
        })
    }
}

impl fmt::Display for Permission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Permission::FileRead(pat) => write!(f, "FileRead({})", pat),
            Permission::FileWrite(pat) => write!(f, "FileWrite({})", pat),
            Permission::NetworkAccess { hosts } => write!(f, "NetworkAccess({:?})", hosts),
            Permission::ShellExec { commands } => write!(f, "ShellExec({:?})", commands),
            Permission::ToolUse { tools } => write!(f, "ToolUse({:?})", tools),
            Permission::SubAgentSpawn => write!(f, "SubAgentSpawn"),
            Permission::All => write!(f, "All"),
        }
    }
}

// ---------------------------------------------------------------------------
// PermissionSet
// ---------------------------------------------------------------------------

/// A collection of allowed and denied fine-grained permissions.
#[derive(Debug, Clone)]
pub struct PermissionSet {
    allowed: Vec<Permission>,
    denied: Vec<Permission>,
}

impl PermissionSet {
    /// Create an empty permission set.
    pub fn new() -> Self {
        Self {
            allowed: Vec::new(),
            denied: Vec::new(),
        }
    }

    /// Add an allowed permission.
    pub fn allow(&mut self, permission: Permission) {
        self.allowed.push(permission);
    }

    /// Add a denied permission.
    pub fn deny(&mut self, permission: Permission) {
        self.denied.push(permission);
    }

    /// Check whether `permission` is allowed by this set.
    ///
    /// Denied rules take precedence. If an explicit deny matches, the
    /// permission is rejected. Otherwise it is allowed if any allow rule matches.
    pub fn is_allowed(&self, permission: &Permission) -> bool {
        for d in &self.denied {
            if Self::permission_matches(d, permission) {
                return false;
            }
        }
        for a in &self.allowed {
            if Self::permission_matches(a, permission) {
                return true;
            }
        }
        false
    }

    /// Return all tool names that are explicitly allowed.
    pub fn allowed_tools(&self) -> Vec<String> {
        let mut tools = Vec::new();
        for perm in &self.allowed {
            match perm {
                Permission::ToolUse { tools: t } => tools.extend(t.clone()),
                Permission::All => {
                    tools.push("*".to_string());
                    break;
                }
                _ => {}
            }
        }
        tools
    }

    /// Check whether reading the given path is allowed.
    pub fn can_read(&self, path: &str) -> bool {
        self.is_allowed(&Permission::FileRead(PathPattern::new(path)))
    }

    /// Check whether writing to the given path is allowed.
    pub fn can_write(&self, path: &str) -> bool {
        self.is_allowed(&Permission::FileWrite(PathPattern::new(path)))
    }

    /// Merge another permission set into this one.
    pub fn merge(&mut self, other: &PermissionSet) {
        self.allowed.extend(other.allowed.clone());
        self.denied.extend(other.denied.clone());
    }

    /// Number of rules (allowed + denied).
    pub fn len(&self) -> usize {
        self.allowed.len() + self.denied.len()
    }

    /// Whether the set has no rules.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "allowed": self.allowed.iter().map(|p| p.to_json()).collect::<Vec<_>>(),
            "denied": self.denied.iter().map(|p| p.to_json()).collect::<Vec<_>>(),
        })
    }

    fn permission_matches(rule: &Permission, request: &Permission) -> bool {
        match (rule, request) {
            (Permission::All, _) => true,
            (Permission::SubAgentSpawn, Permission::SubAgentSpawn) => true,
            (Permission::FileRead(rule_pat), Permission::FileRead(req_pat)) => {
                rule_pat.matches(req_pat.pattern())
            }
            (Permission::FileWrite(rule_pat), Permission::FileWrite(req_pat)) => {
                rule_pat.matches(req_pat.pattern())
            }
            (
                Permission::NetworkAccess { hosts: rule_hosts },
                Permission::NetworkAccess { hosts: req_hosts },
            ) => req_hosts.iter().all(|h| rule_hosts.contains(h)),
            (
                Permission::ShellExec {
                    commands: rule_cmds,
                },
                Permission::ShellExec { commands: req_cmds },
            ) => req_cmds.iter().all(|c| rule_cmds.contains(c)),
            (
                Permission::ToolUse {
                    tools: rule_tools,
                },
                Permission::ToolUse { tools: req_tools },
            ) => req_tools.iter().all(|t| rule_tools.contains(t)),
            _ => false,
        }
    }
}

impl Default for PermissionSet {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SandboxViolation
// ---------------------------------------------------------------------------

/// Records a sandbox policy violation.
#[derive(Debug, Clone)]
pub struct SandboxViolation {
    /// Description of the denied permission.
    pub permission_denied: String,
    /// Timestamp string.
    pub timestamp: String,
    /// Optional additional context.
    pub context: Option<String>,
}

impl SandboxViolation {
    /// Create a new violation record.
    pub fn new(denied: &str) -> Self {
        Self {
            permission_denied: denied.to_string(),
            timestamp: format!("{:?}", std::time::SystemTime::now()),
            context: None,
        }
    }

    /// Attach context information.
    pub fn with_context(mut self, ctx: &str) -> Self {
        self.context = Some(ctx.to_string());
        self
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "permission_denied": self.permission_denied,
            "timestamp": self.timestamp,
            "context": self.context,
        })
    }
}

impl fmt::Display for SandboxViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SandboxViolation: {} at {}",
            self.permission_denied, self.timestamp
        )?;
        if let Some(ctx) = &self.context {
            write!(f, " ({})", ctx)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SandboxConfig (for the enforcing sandbox)
// ---------------------------------------------------------------------------

/// Configuration for constructing an [`EnforcingSandbox`].
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Human-readable name for this sandbox.
    pub name: String,
    /// The permission set governing allowed/denied actions.
    pub permissions: PermissionSet,
    /// Maximum memory in megabytes (advisory).
    pub max_memory_mb: Option<usize>,
    /// Maximum wall-clock duration in seconds.
    pub max_duration_secs: Option<u64>,
    /// Maximum number of tool calls allowed.
    pub max_tool_calls: Option<usize>,
}

impl SandboxConfig {
    /// Create a new sandbox config with the given name and empty permissions.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            permissions: PermissionSet::new(),
            max_memory_mb: None,
            max_duration_secs: None,
            max_tool_calls: None,
        }
    }

    /// Add a permission and return self for chaining.
    pub fn with_permission(mut self, perm: Permission) -> Self {
        self.permissions.allow(perm);
        self
    }

    /// Set the memory limit.
    pub fn with_memory_limit(mut self, mb: usize) -> Self {
        self.max_memory_mb = Some(mb);
        self
    }

    /// Set the duration limit.
    pub fn with_duration_limit(mut self, secs: u64) -> Self {
        self.max_duration_secs = Some(secs);
        self
    }

    /// Set the tool call limit.
    pub fn with_tool_limit(mut self, n: usize) -> Self {
        self.max_tool_calls = Some(n);
        self
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "permissions": self.permissions.to_json(),
            "max_memory_mb": self.max_memory_mb,
            "max_duration_secs": self.max_duration_secs,
            "max_tool_calls": self.max_tool_calls,
        })
    }
}

// ---------------------------------------------------------------------------
// EnforcingSandbox
// ---------------------------------------------------------------------------

/// A live sandbox that enforces constraints from its [`SandboxConfig`].
///
/// Tracks tool call counts, records violations, and checks expiration.
pub struct EnforcingSandbox {
    config: SandboxConfig,
    tool_call_count: std::sync::atomic::AtomicUsize,
    violations: std::sync::Mutex<Vec<SandboxViolation>>,
    created_at: Instant,
}

impl EnforcingSandbox {
    /// Create a new enforcing sandbox from the given configuration.
    pub fn new(config: SandboxConfig) -> Self {
        Self {
            config,
            tool_call_count: std::sync::atomic::AtomicUsize::new(0),
            violations: std::sync::Mutex::new(Vec::new()),
            created_at: Instant::now(),
        }
    }

    /// Check whether a permission is allowed. Records a violation on denial.
    pub fn check_permission(
        &self,
        permission: &Permission,
    ) -> std::result::Result<(), SandboxViolation> {
        if self.is_expired() {
            let v = SandboxViolation::new("sandbox_expired")
                .with_context("Sandbox duration limit exceeded");
            self.violations.lock().unwrap().push(v.clone());
            return Err(v);
        }
        if self.config.permissions.is_allowed(permission) {
            Ok(())
        } else {
            let v = SandboxViolation::new(&format!("denied: {}", permission));
            self.violations.lock().unwrap().push(v.clone());
            Err(v)
        }
    }

    /// Record a tool call. Returns `Err` if the tool call limit is exceeded.
    pub fn record_tool_call(&self) -> std::result::Result<(), SandboxViolation> {
        let count = self
            .tool_call_count
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst)
            + 1;
        if let Some(max) = self.config.max_tool_calls {
            if count > max {
                let v = SandboxViolation::new("tool_call_limit_exceeded")
                    .with_context(&format!("Limit: {}, attempted: {}", max, count));
                self.violations.lock().unwrap().push(v.clone());
                return Err(v);
            }
        }
        Ok(())
    }

    /// Number of remaining tool calls, or `None` if unlimited.
    pub fn remaining_tool_calls(&self) -> Option<usize> {
        self.config.max_tool_calls.map(|max| {
            let used = self
                .tool_call_count
                .load(std::sync::atomic::Ordering::SeqCst);
            max.saturating_sub(used)
        })
    }

    /// Whether the sandbox has exceeded its duration limit.
    pub fn is_expired(&self) -> bool {
        if let Some(secs) = self.config.max_duration_secs {
            self.created_at.elapsed().as_secs() >= secs
        } else {
            false
        }
    }

    /// Return all recorded violations.
    pub fn violations(&self) -> Vec<SandboxViolation> {
        self.violations.lock().unwrap().clone()
    }

    /// Number of recorded violations.
    pub fn violation_count(&self) -> usize {
        self.violations.lock().unwrap().len()
    }

    /// Serialize sandbox state to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "config": self.config.to_json(),
            "tool_calls": self.tool_call_count.load(std::sync::atomic::Ordering::SeqCst),
            "violations": self.violations.lock().unwrap().iter().map(|v| v.to_json()).collect::<Vec<_>>(),
            "elapsed_secs": self.created_at.elapsed().as_secs(),
        })
    }
}

// ---------------------------------------------------------------------------
// SandboxPreset
// ---------------------------------------------------------------------------

/// Predefined sandbox configurations for common use cases.
pub struct SandboxPreset;

impl SandboxPreset {
    /// Minimal permissions — no access to anything.
    pub fn restrictive() -> SandboxConfig {
        SandboxConfig::new("restrictive")
            .with_memory_limit(256)
            .with_duration_limit(60)
            .with_tool_limit(10)
    }

    /// Most permissions allowed.
    pub fn permissive() -> SandboxConfig {
        SandboxConfig::new("permissive").with_permission(Permission::All)
    }

    /// Only file access for the given path patterns.
    pub fn file_only(paths: Vec<String>) -> SandboxConfig {
        let mut config = SandboxConfig::new("file_only");
        for p in &paths {
            config
                .permissions
                .allow(Permission::FileRead(PathPattern::new(p)));
            config
                .permissions
                .allow(Permission::FileWrite(PathPattern::new(p)));
        }
        config
    }

    /// Only network access for the given hosts.
    pub fn network_only(hosts: Vec<String>) -> SandboxConfig {
        SandboxConfig::new("network_only").with_permission(Permission::NetworkAccess { hosts })
    }
}

// ---------------------------------------------------------------------------
// SandboxAuditLog
// ---------------------------------------------------------------------------

/// Entry in the audit log.
#[derive(Debug, Clone)]
pub struct AuditEntry {
    /// Description of the checked permission.
    pub permission: String,
    /// Whether the check was allowed.
    pub allowed: bool,
    /// Timestamp string.
    pub timestamp: String,
}

/// Logs all permission checks for auditing.
#[derive(Debug, Clone, Default)]
pub struct SandboxAuditLog {
    entries: Vec<AuditEntry>,
}

impl SandboxAuditLog {
    /// Create an empty audit log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Record a permission check.
    pub fn record_check(&mut self, permission: &Permission, allowed: bool) {
        self.entries.push(AuditEntry {
            permission: format!("{}", permission),
            allowed,
            timestamp: format!("{:?}", std::time::SystemTime::now()),
        });
    }

    /// Return all entries.
    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    /// Return only denied entries.
    pub fn denied_entries(&self) -> Vec<&AuditEntry> {
        self.entries.iter().filter(|e| !e.allowed).collect()
    }

    /// Total number of checks recorded.
    pub fn total_checks(&self) -> usize {
        self.entries.len()
    }

    /// Fraction of checks that were denied (0.0–1.0). Returns 0.0 if empty.
    pub fn denial_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let denied = self.entries.iter().filter(|e| !e.allowed).count();
        denied as f64 / self.entries.len() as f64
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "total_checks": self.total_checks(),
            "denial_rate": self.denial_rate(),
            "entries": self.entries.iter().map(|e| serde_json::json!({
                "permission": e.permission,
                "allowed": e.allowed,
                "timestamp": e.timestamp,
            })).collect::<Vec<_>>(),
        })
    }
}

// ---------------------------------------------------------------------------
// ResourceTracker
// ---------------------------------------------------------------------------

/// Tracks resource usage within a sandbox.
#[derive(Debug)]
pub struct ResourceTracker {
    peak_memory_bytes: std::sync::atomic::AtomicUsize,
    current_memory_bytes: std::sync::atomic::AtomicUsize,
    total_cpu_ms: std::sync::atomic::AtomicUsize,
    total_io_bytes: std::sync::atomic::AtomicUsize,
}

impl ResourceTracker {
    /// Create a new resource tracker with zero counters.
    pub fn new() -> Self {
        Self {
            peak_memory_bytes: std::sync::atomic::AtomicUsize::new(0),
            current_memory_bytes: std::sync::atomic::AtomicUsize::new(0),
            total_cpu_ms: std::sync::atomic::AtomicUsize::new(0),
            total_io_bytes: std::sync::atomic::AtomicUsize::new(0),
        }
    }

    /// Record a memory usage sample in bytes.
    pub fn record_memory_usage(&self, bytes: usize) {
        self.current_memory_bytes
            .store(bytes, std::sync::atomic::Ordering::SeqCst);
        loop {
            let peak = self
                .peak_memory_bytes
                .load(std::sync::atomic::Ordering::SeqCst);
            if bytes <= peak {
                break;
            }
            if self
                .peak_memory_bytes
                .compare_exchange(
                    peak,
                    bytes,
                    std::sync::atomic::Ordering::SeqCst,
                    std::sync::atomic::Ordering::SeqCst,
                )
                .is_ok()
            {
                break;
            }
        }
    }

    /// Record CPU time used in milliseconds.
    pub fn record_cpu_time_ms(&self, ms: u64) {
        self.total_cpu_ms
            .fetch_add(ms as usize, std::sync::atomic::Ordering::SeqCst);
    }

    /// Record I/O bytes transferred.
    pub fn record_io_bytes(&self, bytes: usize) {
        self.total_io_bytes
            .fetch_add(bytes, std::sync::atomic::Ordering::SeqCst);
    }

    /// Peak memory usage observed in bytes.
    pub fn peak_memory(&self) -> usize {
        self.peak_memory_bytes
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Total CPU time recorded in milliseconds.
    pub fn total_cpu_ms(&self) -> u64 {
        self.total_cpu_ms
            .load(std::sync::atomic::Ordering::SeqCst) as u64
    }

    /// Total I/O bytes recorded.
    pub fn total_io_bytes(&self) -> usize {
        self.total_io_bytes
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "peak_memory_bytes": self.peak_memory(),
            "total_cpu_ms": self.total_cpu_ms(),
            "total_io_bytes": self.total_io_bytes(),
        })
    }
}

impl Default for ResourceTracker {
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

    // ===================================================================
    // New sandbox system tests: PathPattern, Permission, PermissionSet,
    // SandboxConfig, EnforcingSandbox, SandboxViolation, SandboxPreset,
    // SandboxAuditLog, ResourceTracker
    // ===================================================================

    // -- PathPattern tests --

    #[test]
    fn test_path_pattern_exact_match() {
        let p = PathPattern::new("src/main.rs");
        assert!(p.matches("src/main.rs"));
        assert!(!p.matches("src/lib.rs"));
    }

    #[test]
    fn test_path_pattern_single_wildcard() {
        // `*` matches exactly one path segment
        let p = PathPattern::new("src/*");
        assert!(p.matches("src/main.rs"));
        assert!(p.matches("src/lib.rs"));
        assert!(!p.matches("src/sub/main.rs"));
    }

    #[test]
    fn test_path_pattern_double_wildcard() {
        let p = PathPattern::new("src/**");
        assert!(p.matches("src/main.rs"));
        assert!(p.matches("src/sub/main.rs"));
        assert!(p.matches("src/a/b/c/main.rs"));
    }

    #[test]
    fn test_path_pattern_double_wildcard_prefix() {
        let p = PathPattern::new("**/test.rs");
        assert!(p.matches("test.rs"));
        assert!(p.matches("src/test.rs"));
        assert!(p.matches("a/b/c/test.rs"));
        assert!(!p.matches("a/b/c/other.rs"));
    }

    #[test]
    fn test_path_pattern_all_recursive() {
        let p = PathPattern::new("**");
        assert!(p.matches("anything"));
        assert!(p.matches("a/b/c/d"));
    }

    #[test]
    fn test_path_pattern_no_match_empty() {
        let p = PathPattern::new("src/main.rs");
        assert!(!p.matches(""));
    }

    #[test]
    fn test_path_pattern_display() {
        let p = PathPattern::new("src/**");
        assert_eq!(format!("{}", p), "src/**");
    }

    #[test]
    fn test_path_pattern_getter() {
        let p = PathPattern::new("foo/bar");
        assert_eq!(p.pattern(), "foo/bar");
    }

    #[test]
    fn test_path_pattern_leading_slash() {
        let p = PathPattern::new("/usr/local/bin");
        assert!(p.matches("/usr/local/bin"));
    }

    // -- Permission (fine-grained) tests --

    #[test]
    fn test_fg_permission_name_variants() {
        assert_eq!(Permission::FileRead(PathPattern::new("*")).name(), "file_read");
        assert_eq!(Permission::FileWrite(PathPattern::new("*")).name(), "file_write");
        assert_eq!(Permission::NetworkAccess { hosts: vec![] }.name(), "network_access");
        assert_eq!(Permission::ShellExec { commands: vec![] }.name(), "shell_exec");
        assert_eq!(Permission::ToolUse { tools: vec![] }.name(), "tool_use");
        assert_eq!(Permission::SubAgentSpawn.name(), "sub_agent_spawn");
        assert_eq!(Permission::All.name(), "all");
    }

    #[test]
    fn test_fg_permission_matches_path_file_read() {
        let perm = Permission::FileRead(PathPattern::new("src/**"));
        assert!(perm.matches_path("src/main.rs"));
        assert!(!perm.matches_path("tests/test.rs"));
    }

    #[test]
    fn test_fg_permission_matches_path_file_write() {
        let perm = Permission::FileWrite(PathPattern::new("output/*"));
        assert!(perm.matches_path("output/result.txt"));
        assert!(!perm.matches_path("output/sub/result.txt"));
    }

    #[test]
    fn test_fg_permission_matches_path_all() {
        assert!(Permission::All.matches_path("anything/at/all"));
    }

    #[test]
    fn test_fg_permission_matches_path_non_file() {
        let perm = Permission::NetworkAccess {
            hosts: vec!["example.com".into()],
        };
        assert!(!perm.matches_path("some/path"));
    }

    #[test]
    fn test_fg_permission_to_json() {
        let perm = Permission::SubAgentSpawn;
        let json = perm.to_json();
        assert_eq!(json["type"], "sub_agent_spawn");
    }

    #[test]
    fn test_fg_permission_display() {
        let perm = Permission::FileRead(PathPattern::new("src/**"));
        let s = format!("{}", perm);
        assert!(s.contains("FileRead"));
        assert!(s.contains("src/**"));
    }

    #[test]
    fn test_fg_permission_display_network() {
        let perm = Permission::NetworkAccess {
            hosts: vec!["a.com".into()],
        };
        let s = format!("{}", perm);
        assert!(s.contains("NetworkAccess"));
    }

    // -- PermissionSet tests --

    #[test]
    fn test_pset_empty_denies_all() {
        let ps = PermissionSet::new();
        assert!(!ps.is_allowed(&Permission::SubAgentSpawn));
        assert!(!ps.can_read("anything"));
    }

    #[test]
    fn test_pset_allow_and_check() {
        let mut ps = PermissionSet::new();
        ps.allow(Permission::SubAgentSpawn);
        assert!(ps.is_allowed(&Permission::SubAgentSpawn));
    }

    #[test]
    fn test_pset_deny_overrides_allow() {
        let mut ps = PermissionSet::new();
        ps.allow(Permission::All);
        ps.deny(Permission::SubAgentSpawn);
        assert!(!ps.is_allowed(&Permission::SubAgentSpawn));
    }

    #[test]
    fn test_pset_can_read() {
        let mut ps = PermissionSet::new();
        ps.allow(Permission::FileRead(PathPattern::new("src/**")));
        assert!(ps.can_read("src/main.rs"));
        assert!(!ps.can_read("tests/test.rs"));
    }

    #[test]
    fn test_pset_can_write() {
        let mut ps = PermissionSet::new();
        ps.allow(Permission::FileWrite(PathPattern::new("output/**")));
        assert!(ps.can_write("output/result.txt"));
        assert!(!ps.can_write("src/main.rs"));
    }

    #[test]
    fn test_pset_allowed_tools() {
        let mut ps = PermissionSet::new();
        ps.allow(Permission::ToolUse {
            tools: vec!["read".into(), "write".into()],
        });
        let tools = ps.allowed_tools();
        assert_eq!(tools, vec!["read", "write"]);
    }

    #[test]
    fn test_pset_allowed_tools_with_all() {
        let mut ps = PermissionSet::new();
        ps.allow(Permission::All);
        assert_eq!(ps.allowed_tools(), vec!["*"]);
    }

    #[test]
    fn test_pset_merge() {
        let mut ps1 = PermissionSet::new();
        ps1.allow(Permission::SubAgentSpawn);

        let mut ps2 = PermissionSet::new();
        ps2.allow(Permission::NetworkAccess {
            hosts: vec!["example.com".into()],
        });

        ps1.merge(&ps2);
        assert!(ps1.is_allowed(&Permission::SubAgentSpawn));
        assert!(ps1.is_allowed(&Permission::NetworkAccess {
            hosts: vec!["example.com".into()],
        }));
    }

    #[test]
    fn test_pset_len() {
        let mut ps = PermissionSet::new();
        assert_eq!(ps.len(), 0);
        assert!(ps.is_empty());
        ps.allow(Permission::SubAgentSpawn);
        ps.deny(Permission::All);
        assert_eq!(ps.len(), 2);
        assert!(!ps.is_empty());
    }

    #[test]
    fn test_pset_to_json() {
        let ps = PermissionSet::new();
        let json = ps.to_json();
        assert!(json["allowed"].is_array());
        assert!(json["denied"].is_array());
    }

    #[test]
    fn test_pset_network_access_check() {
        let mut ps = PermissionSet::new();
        ps.allow(Permission::NetworkAccess {
            hosts: vec!["a.com".into(), "b.com".into()],
        });
        assert!(ps.is_allowed(&Permission::NetworkAccess {
            hosts: vec!["a.com".into()],
        }));
        assert!(!ps.is_allowed(&Permission::NetworkAccess {
            hosts: vec!["c.com".into()],
        }));
    }

    #[test]
    fn test_pset_shell_exec_check() {
        let mut ps = PermissionSet::new();
        ps.allow(Permission::ShellExec {
            commands: vec!["ls".into(), "cat".into()],
        });
        assert!(ps.is_allowed(&Permission::ShellExec {
            commands: vec!["ls".into()],
        }));
        assert!(!ps.is_allowed(&Permission::ShellExec {
            commands: vec!["rm".into()],
        }));
    }

    #[test]
    fn test_pset_deny_specific_path() {
        let mut ps = PermissionSet::new();
        ps.allow(Permission::FileRead(PathPattern::new("**")));
        ps.deny(Permission::FileRead(PathPattern::new("secret/**")));
        assert!(ps.can_read("src/main.rs"));
        assert!(!ps.can_read("secret/key.pem"));
    }

    #[test]
    fn test_pset_default() {
        let ps = PermissionSet::default();
        assert!(ps.is_empty());
    }

    // -- SandboxViolation tests --

    #[test]
    fn test_violation_new() {
        let v = SandboxViolation::new("test_denied");
        assert_eq!(v.permission_denied, "test_denied");
        assert!(v.context.is_none());
    }

    #[test]
    fn test_violation_with_context() {
        let v = SandboxViolation::new("denied").with_context("extra info");
        assert_eq!(v.context.as_deref(), Some("extra info"));
    }

    #[test]
    fn test_violation_display() {
        let v = SandboxViolation::new("denied").with_context("ctx");
        let s = format!("{}", v);
        assert!(s.contains("denied"));
        assert!(s.contains("ctx"));
    }

    #[test]
    fn test_violation_to_json() {
        let v = SandboxViolation::new("test");
        let json = v.to_json();
        assert_eq!(json["permission_denied"], "test");
    }

    #[test]
    fn test_violation_timestamp_populated() {
        let v = SandboxViolation::new("test");
        assert!(!v.timestamp.is_empty());
    }

    // -- SandboxConfig tests --

    #[test]
    fn test_sandbox_config_new() {
        let cfg = SandboxConfig::new("test");
        assert_eq!(cfg.name, "test");
        assert!(cfg.max_memory_mb.is_none());
        assert!(cfg.max_duration_secs.is_none());
        assert!(cfg.max_tool_calls.is_none());
    }

    #[test]
    fn test_sandbox_config_builder_chain() {
        let cfg = SandboxConfig::new("test")
            .with_permission(Permission::SubAgentSpawn)
            .with_memory_limit(512)
            .with_duration_limit(120)
            .with_tool_limit(50);
        assert_eq!(cfg.max_memory_mb, Some(512));
        assert_eq!(cfg.max_duration_secs, Some(120));
        assert_eq!(cfg.max_tool_calls, Some(50));
        assert!(cfg.permissions.is_allowed(&Permission::SubAgentSpawn));
    }

    #[test]
    fn test_sandbox_config_to_json() {
        let cfg = SandboxConfig::new("json_test").with_memory_limit(128);
        let json = cfg.to_json();
        assert_eq!(json["name"], "json_test");
        assert_eq!(json["max_memory_mb"], 128);
    }

    // -- EnforcingSandbox tests --

    #[test]
    fn test_enforcing_sandbox_check_allowed() {
        let cfg = SandboxConfig::new("test").with_permission(Permission::SubAgentSpawn);
        let sb = EnforcingSandbox::new(cfg);
        assert!(sb.check_permission(&Permission::SubAgentSpawn).is_ok());
    }

    #[test]
    fn test_enforcing_sandbox_check_denied() {
        let cfg = SandboxConfig::new("test");
        let sb = EnforcingSandbox::new(cfg);
        assert!(sb.check_permission(&Permission::SubAgentSpawn).is_err());
        assert_eq!(sb.violation_count(), 1);
    }

    #[test]
    fn test_enforcing_sandbox_tool_calls() {
        let cfg = SandboxConfig::new("test").with_tool_limit(2);
        let sb = EnforcingSandbox::new(cfg);
        assert!(sb.record_tool_call().is_ok());
        assert_eq!(sb.remaining_tool_calls(), Some(1));
        assert!(sb.record_tool_call().is_ok());
        assert_eq!(sb.remaining_tool_calls(), Some(0));
        assert!(sb.record_tool_call().is_err());
    }

    #[test]
    fn test_enforcing_sandbox_unlimited_tool_calls() {
        let cfg = SandboxConfig::new("test");
        let sb = EnforcingSandbox::new(cfg);
        assert_eq!(sb.remaining_tool_calls(), None);
    }

    #[test]
    fn test_enforcing_sandbox_not_expired() {
        let cfg = SandboxConfig::new("test");
        let sb = EnforcingSandbox::new(cfg);
        assert!(!sb.is_expired());
    }

    #[test]
    fn test_enforcing_sandbox_violations_accumulate() {
        let cfg = SandboxConfig::new("test");
        let sb = EnforcingSandbox::new(cfg);
        let _ = sb.check_permission(&Permission::SubAgentSpawn);
        let _ = sb.check_permission(&Permission::SubAgentSpawn);
        assert_eq!(sb.violation_count(), 2);
        assert_eq!(sb.violations().len(), 2);
    }

    #[test]
    fn test_enforcing_sandbox_to_json() {
        let cfg = SandboxConfig::new("json_test").with_tool_limit(5);
        let sb = EnforcingSandbox::new(cfg);
        let json = sb.to_json();
        assert_eq!(json["tool_calls"], 0);
        assert!(json["config"].is_object());
    }

    #[test]
    fn test_enforcing_sandbox_file_permissions() {
        let cfg = SandboxConfig::new("file_sb")
            .with_permission(Permission::FileRead(PathPattern::new("src/**")))
            .with_permission(Permission::FileWrite(PathPattern::new("output/**")));
        let sb = EnforcingSandbox::new(cfg);
        assert!(sb
            .check_permission(&Permission::FileRead(PathPattern::new("src/main.rs")))
            .is_ok());
        assert!(sb
            .check_permission(&Permission::FileWrite(PathPattern::new("output/out.txt")))
            .is_ok());
        assert!(sb
            .check_permission(&Permission::FileRead(PathPattern::new("secret/key.pem")))
            .is_err());
    }

    #[test]
    fn test_enforcing_sandbox_tool_use_permission() {
        let cfg = SandboxConfig::new("tool_sb").with_permission(Permission::ToolUse {
            tools: vec!["read".into(), "write".into()],
        });
        let sb = EnforcingSandbox::new(cfg);
        assert!(sb
            .check_permission(&Permission::ToolUse {
                tools: vec!["read".into()],
            })
            .is_ok());
        assert!(sb
            .check_permission(&Permission::ToolUse {
                tools: vec!["delete".into()],
            })
            .is_err());
    }

    // -- SandboxPreset tests --

    #[test]
    fn test_preset_restrictive() {
        let cfg = SandboxPreset::restrictive();
        assert_eq!(cfg.name, "restrictive");
        assert_eq!(cfg.max_memory_mb, Some(256));
        assert_eq!(cfg.max_duration_secs, Some(60));
        assert_eq!(cfg.max_tool_calls, Some(10));
        assert!(!cfg.permissions.is_allowed(&Permission::SubAgentSpawn));
    }

    #[test]
    fn test_preset_permissive() {
        let cfg = SandboxPreset::permissive();
        assert_eq!(cfg.name, "permissive");
        assert!(cfg.permissions.is_allowed(&Permission::SubAgentSpawn));
        assert!(cfg.permissions.is_allowed(&Permission::NetworkAccess {
            hosts: vec!["any.com".into()],
        }));
    }

    #[test]
    fn test_preset_file_only() {
        let cfg = SandboxPreset::file_only(vec!["src/**".into(), "tests/**".into()]);
        assert_eq!(cfg.name, "file_only");
        assert!(cfg.permissions.can_read("src/main.rs"));
        assert!(cfg.permissions.can_write("tests/test.rs"));
        assert!(!cfg.permissions.can_read("data/file.txt"));
        assert!(!cfg.permissions.is_allowed(&Permission::SubAgentSpawn));
    }

    #[test]
    fn test_preset_network_only() {
        let cfg = SandboxPreset::network_only(vec!["api.example.com".into()]);
        assert_eq!(cfg.name, "network_only");
        assert!(cfg.permissions.is_allowed(&Permission::NetworkAccess {
            hosts: vec!["api.example.com".into()],
        }));
        assert!(!cfg.permissions.can_read("anything"));
    }

    // -- SandboxAuditLog tests --

    #[test]
    fn test_audit_log_empty() {
        let log = SandboxAuditLog::new();
        assert_eq!(log.total_checks(), 0);
        assert_eq!(log.denial_rate(), 0.0);
        assert!(log.entries().is_empty());
    }

    #[test]
    fn test_audit_log_record_checks() {
        let mut log = SandboxAuditLog::new();
        log.record_check(&Permission::SubAgentSpawn, true);
        log.record_check(&Permission::SubAgentSpawn, false);
        assert_eq!(log.total_checks(), 2);
    }

    #[test]
    fn test_audit_log_denied_entries() {
        let mut log = SandboxAuditLog::new();
        log.record_check(&Permission::SubAgentSpawn, true);
        log.record_check(&Permission::SubAgentSpawn, false);
        log.record_check(&Permission::All, false);
        assert_eq!(log.denied_entries().len(), 2);
    }

    #[test]
    fn test_audit_log_denial_rate() {
        let mut log = SandboxAuditLog::new();
        log.record_check(&Permission::SubAgentSpawn, true);
        log.record_check(&Permission::SubAgentSpawn, false);
        assert!((log.denial_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audit_log_to_json() {
        let mut log = SandboxAuditLog::new();
        log.record_check(&Permission::All, true);
        let json = log.to_json();
        assert_eq!(json["total_checks"], 1);
    }

    #[test]
    fn test_audit_log_entry_fields() {
        let mut log = SandboxAuditLog::new();
        log.record_check(&Permission::SubAgentSpawn, true);
        let entry = &log.entries()[0];
        assert!(entry.permission.contains("SubAgentSpawn"));
        assert!(entry.allowed);
    }

    #[test]
    fn test_audit_log_all_denied() {
        let mut log = SandboxAuditLog::new();
        log.record_check(&Permission::All, false);
        log.record_check(&Permission::All, false);
        assert!((log.denial_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audit_log_all_allowed() {
        let mut log = SandboxAuditLog::new();
        log.record_check(&Permission::All, true);
        log.record_check(&Permission::All, true);
        assert!((log.denial_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_audit_log_default() {
        let log = SandboxAuditLog::default();
        assert_eq!(log.total_checks(), 0);
    }

    // -- ResourceTracker tests --

    #[test]
    fn test_resource_tracker_new_zeroes() {
        let rt = ResourceTracker::new();
        assert_eq!(rt.peak_memory(), 0);
        assert_eq!(rt.total_cpu_ms(), 0);
        assert_eq!(rt.total_io_bytes(), 0);
    }

    #[test]
    fn test_resource_tracker_memory_peak() {
        let rt = ResourceTracker::new();
        rt.record_memory_usage(1000);
        rt.record_memory_usage(5000);
        rt.record_memory_usage(3000);
        assert_eq!(rt.peak_memory(), 5000);
    }

    #[test]
    fn test_resource_tracker_cpu_accumulates() {
        let rt = ResourceTracker::new();
        rt.record_cpu_time_ms(100);
        rt.record_cpu_time_ms(200);
        assert_eq!(rt.total_cpu_ms(), 300);
    }

    #[test]
    fn test_resource_tracker_io_accumulates() {
        let rt = ResourceTracker::new();
        rt.record_io_bytes(1024);
        rt.record_io_bytes(2048);
        assert_eq!(rt.total_io_bytes(), 3072);
    }

    #[test]
    fn test_resource_tracker_to_json() {
        let rt = ResourceTracker::new();
        rt.record_memory_usage(4096);
        rt.record_cpu_time_ms(50);
        rt.record_io_bytes(512);
        let json = rt.to_json();
        assert_eq!(json["peak_memory_bytes"], 4096);
        assert_eq!(json["total_cpu_ms"], 50);
        assert_eq!(json["total_io_bytes"], 512);
    }

    #[test]
    fn test_resource_tracker_default() {
        let rt = ResourceTracker::default();
        assert_eq!(rt.peak_memory(), 0);
    }

    #[test]
    fn test_resource_tracker_memory_monotonic_decrease() {
        let rt = ResourceTracker::new();
        rt.record_memory_usage(10000);
        rt.record_memory_usage(5000);
        rt.record_memory_usage(1000);
        assert_eq!(rt.peak_memory(), 10000);
    }
}
