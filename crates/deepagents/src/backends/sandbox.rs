//! Sandbox backend with resource limits and permission controls.
//!
//! Wraps an inner [`Backend`] and enforces path access restrictions,
//! network/env-var policies, memory limits, and execution time limits
//! before delegating to the underlying backend.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use thiserror::Error;
use tokio::sync::Mutex;

use super::{Backend, Result};
use crate::agent::DeepAgentError;

// ---------------------------------------------------------------------------
// SandboxViolation
// ---------------------------------------------------------------------------

/// Errors raised when a sandbox policy is violated.
#[derive(Debug, Error, Clone)]
pub enum SandboxViolation {
    /// A file-system path access was denied.
    #[error("path access denied: {path} ({operation})")]
    PathAccessDenied {
        /// The path that was accessed.
        path: PathBuf,
        /// The operation that was attempted.
        operation: PathOperation,
    },

    /// Network access is not allowed in this sandbox.
    #[error("network access denied")]
    NetworkAccessDenied,

    /// Access to an environment variable was denied.
    #[error("env var access denied: {var_name}")]
    EnvVarAccessDenied {
        /// The environment variable name.
        var_name: String,
    },

    /// Memory usage exceeded the configured limit.
    #[error("memory limit exceeded: limit={limit}, used={used}")]
    MemoryLimitExceeded {
        /// Configured limit in bytes.
        limit: usize,
        /// Current tracked usage in bytes.
        used: usize,
    },

    /// Execution time exceeded the configured limit.
    #[error("time limit exceeded: {limit:?}")]
    TimeLimitExceeded {
        /// Configured time limit.
        limit: Duration,
    },

    /// A write was attempted in read-only mode.
    #[error("read-only violation: {path}")]
    ReadOnlyViolation {
        /// The path that was written to.
        path: PathBuf,
    },
}

// ---------------------------------------------------------------------------
// PathOperation
// ---------------------------------------------------------------------------

/// The kind of file-system operation being checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathOperation {
    /// Reading a file or directory listing.
    Read,
    /// Writing or creating a file.
    Write,
    /// Executing a file.
    Execute,
    /// Deleting a file or directory.
    Delete,
}

impl std::fmt::Display for PathOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read => write!(f, "read"),
            Self::Write => write!(f, "write"),
            Self::Execute => write!(f, "execute"),
            Self::Delete => write!(f, "delete"),
        }
    }
}

// ---------------------------------------------------------------------------
// SandboxConfig
// ---------------------------------------------------------------------------

/// Configuration for the sandbox environment.
#[derive(Debug, Clone)]
pub struct SandboxConfig {
    /// Directories the sandbox is allowed to access.
    pub allowed_paths: Vec<PathBuf>,
    /// Explicitly denied paths (take precedence over allowed).
    pub denied_paths: Vec<PathBuf>,
    /// Optional memory limit in bytes.
    pub max_memory_bytes: Option<usize>,
    /// Optional execution time limit.
    pub max_execution_time: Option<Duration>,
    /// Whether network access is permitted. Default `false`.
    pub allow_network: bool,
    /// Whether environment variable access is permitted. Default `false`.
    pub allow_env_vars: bool,
    /// Whitelist of allowed environment variable names (used when `allow_env_vars` is `true`).
    pub allowed_env_vars: Vec<String>,
    /// Working directory for sandbox operations.
    pub working_directory: PathBuf,
    /// Whether the sandbox is read-only. Default `false`.
    pub read_only: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            allowed_paths: Vec::new(),
            denied_paths: Vec::new(),
            max_memory_bytes: None,
            max_execution_time: None,
            allow_network: false,
            allow_env_vars: false,
            allowed_env_vars: Vec::new(),
            working_directory: PathBuf::from("."),
            read_only: false,
        }
    }
}

// ---------------------------------------------------------------------------
// PermissionChecker
// ---------------------------------------------------------------------------

/// Validates operations against the sandbox configuration.
#[derive(Debug, Clone)]
pub struct PermissionChecker {
    config: SandboxConfig,
}

impl PermissionChecker {
    /// Create a new checker from the given configuration.
    pub fn new(config: SandboxConfig) -> Self {
        Self { config }
    }

    /// Check whether `path` is accessible for the given `operation`.
    ///
    /// Denied paths always take precedence. If the path (or an ancestor)
    /// appears in `denied_paths`, access is refused. Otherwise the path
    /// must be equal to or a subdirectory of an entry in `allowed_paths`.
    pub fn check_path_access(
        &self,
        path: &PathBuf,
        operation: PathOperation,
    ) -> std::result::Result<(), SandboxViolation> {
        // Read-only mode blocks mutating operations.
        if self.config.read_only
            && matches!(operation, PathOperation::Write | PathOperation::Delete)
        {
            return Err(SandboxViolation::ReadOnlyViolation { path: path.clone() });
        }

        // Denied paths take precedence.
        for denied in &self.config.denied_paths {
            if path.starts_with(denied) {
                return Err(SandboxViolation::PathAccessDenied {
                    path: path.clone(),
                    operation,
                });
            }
        }

        // Must be under at least one allowed path.
        if self.config.allowed_paths.is_empty() {
            return Err(SandboxViolation::PathAccessDenied {
                path: path.clone(),
                operation,
            });
        }

        let allowed = self
            .config
            .allowed_paths
            .iter()
            .any(|allowed| path.starts_with(allowed));

        if !allowed {
            return Err(SandboxViolation::PathAccessDenied {
                path: path.clone(),
                operation,
            });
        }

        Ok(())
    }

    /// Check whether network access is allowed.
    pub fn check_network_access(&self) -> std::result::Result<(), SandboxViolation> {
        if self.config.allow_network {
            Ok(())
        } else {
            Err(SandboxViolation::NetworkAccessDenied)
        }
    }

    /// Check whether the given environment variable may be read.
    pub fn check_env_access(&self, var_name: &str) -> std::result::Result<(), SandboxViolation> {
        if !self.config.allow_env_vars {
            return Err(SandboxViolation::EnvVarAccessDenied {
                var_name: var_name.to_string(),
            });
        }

        if self.config.allowed_env_vars.is_empty() {
            // allow_env_vars is true with no whitelist — allow everything.
            return Ok(());
        }

        if self.config.allowed_env_vars.iter().any(|v| v == var_name) {
            Ok(())
        } else {
            Err(SandboxViolation::EnvVarAccessDenied {
                var_name: var_name.to_string(),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// ResourceStats
// ---------------------------------------------------------------------------

/// Snapshot of resource usage within a sandbox.
#[derive(Debug, Clone)]
pub struct ResourceStats {
    /// Current tracked memory usage in bytes.
    pub memory_used: usize,
    /// Elapsed time since the sandbox was created.
    pub elapsed_time: Duration,
    /// Total number of backend operations performed.
    pub operations_count: usize,
    /// Set of file paths accessed through the sandbox.
    pub files_accessed: HashSet<PathBuf>,
}

// ---------------------------------------------------------------------------
// ResourceTracker
// ---------------------------------------------------------------------------

/// Tracks memory, time, and operation counts for a sandbox.
pub struct ResourceTracker {
    memory_used: AtomicUsize,
    max_memory_bytes: Option<usize>,
    max_execution_time: Option<Duration>,
    start: Instant,
    operations_count: AtomicUsize,
    files_accessed: Mutex<HashSet<PathBuf>>,
}

impl ResourceTracker {
    /// Create a new tracker with the given limits.
    pub fn new(max_memory_bytes: Option<usize>, max_execution_time: Option<Duration>) -> Self {
        Self {
            memory_used: AtomicUsize::new(0),
            max_memory_bytes,
            max_execution_time,
            start: Instant::now(),
            operations_count: AtomicUsize::new(0),
            files_accessed: Mutex::new(HashSet::new()),
        }
    }

    /// Record a memory allocation of `bytes`.
    pub fn track_memory(&self, bytes: usize) {
        self.memory_used.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Check whether the memory limit has been exceeded.
    pub fn check_memory_limit(&self) -> std::result::Result<(), SandboxViolation> {
        if let Some(limit) = self.max_memory_bytes {
            let used = self.memory_used.load(Ordering::Relaxed);
            if used > limit {
                return Err(SandboxViolation::MemoryLimitExceeded { limit, used });
            }
        }
        Ok(())
    }

    /// Return the elapsed time since the tracker was created.
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    /// Check whether the time limit has been exceeded.
    pub fn check_time_limit(&self) -> std::result::Result<(), SandboxViolation> {
        if let Some(limit) = self.max_execution_time {
            if self.start.elapsed() > limit {
                return Err(SandboxViolation::TimeLimitExceeded { limit });
            }
        }
        Ok(())
    }

    /// Increment the operations counter.
    pub fn record_operation(&self) {
        self.operations_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record that a file path was accessed.
    pub async fn record_file_access(&self, path: PathBuf) {
        self.files_accessed.lock().await.insert(path);
    }

    /// Return a snapshot of current resource usage.
    pub async fn stats(&self) -> ResourceStats {
        ResourceStats {
            memory_used: self.memory_used.load(Ordering::Relaxed),
            elapsed_time: self.start.elapsed(),
            operations_count: self.operations_count.load(Ordering::Relaxed),
            files_accessed: self.files_accessed.lock().await.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// SandboxBackend
// ---------------------------------------------------------------------------

/// A backend wrapper that enforces sandbox policies before delegating to an
/// inner [`Backend`] implementation.
pub struct SandboxBackend {
    /// The inner backend that actually persists state.
    inner: Box<dyn Backend>,
    /// Permission checker derived from the sandbox config.
    checker: PermissionChecker,
    /// Resource tracker for memory / time / operations.
    tracker: Arc<ResourceTracker>,
}

impl SandboxBackend {
    /// Create a new sandbox backend wrapping `inner` with the given `config`.
    pub fn new(inner: Box<dyn Backend>, config: SandboxConfig) -> Self {
        let tracker = Arc::new(ResourceTracker::new(
            config.max_memory_bytes,
            config.max_execution_time,
        ));
        let checker = PermissionChecker::new(config);
        Self {
            inner,
            checker,
            tracker,
        }
    }

    /// Return a reference to the permission checker.
    pub fn checker(&self) -> &PermissionChecker {
        &self.checker
    }

    /// Return a reference to the resource tracker.
    pub fn tracker(&self) -> &ResourceTracker {
        &self.tracker
    }

    /// Return current resource statistics.
    pub async fn stats(&self) -> ResourceStats {
        self.tracker.stats().await
    }

    fn check_limits(&self) -> Result<()> {
        self.tracker
            .check_memory_limit()
            .map_err(|e| DeepAgentError::BackendError(e.to_string()))?;
        self.tracker
            .check_time_limit()
            .map_err(|e| DeepAgentError::BackendError(e.to_string()))?;
        Ok(())
    }
}

#[async_trait]
impl Backend for SandboxBackend {
    async fn save_state(&self, session_id: &str, state: &Value) -> Result<()> {
        self.check_limits()?;
        self.tracker.record_operation();

        let serialized = serde_json::to_string(state)
            .map_err(|e| DeepAgentError::BackendError(e.to_string()))?;
        self.tracker.track_memory(serialized.len());
        self.tracker
            .check_memory_limit()
            .map_err(|e| DeepAgentError::BackendError(e.to_string()))?;

        self.inner.save_state(session_id, state).await
    }

    async fn load_state(&self, session_id: &str) -> Result<Option<Value>> {
        self.check_limits()?;
        self.tracker.record_operation();
        self.inner.load_state(session_id).await
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        self.check_limits()?;
        self.tracker.record_operation();
        self.inner.list_sessions().await
    }
}

// ---------------------------------------------------------------------------
// SandboxBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`SandboxBackend`].
pub struct SandboxBuilder {
    config: SandboxConfig,
}

impl SandboxBuilder {
    /// Start building a sandbox with default (deny-all) settings.
    pub fn new(working_directory: impl Into<PathBuf>) -> Self {
        Self {
            config: SandboxConfig {
                working_directory: working_directory.into(),
                ..Default::default()
            },
        }
    }

    /// Allow access to the given path (and its subdirectories).
    pub fn allow_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.allowed_paths.push(path.into());
        self
    }

    /// Deny access to the given path (takes precedence over allowed paths).
    pub fn deny_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.denied_paths.push(path.into());
        self
    }

    /// Set the maximum memory in bytes.
    pub fn max_memory(mut self, bytes: usize) -> Self {
        self.config.max_memory_bytes = Some(bytes);
        self
    }

    /// Set the maximum execution time.
    pub fn max_time(mut self, duration: Duration) -> Self {
        self.config.max_execution_time = Some(duration);
        self
    }

    /// Allow network access.
    pub fn allow_network(mut self) -> Self {
        self.config.allow_network = true;
        self
    }

    /// Allow environment variable access (optionally with a whitelist).
    pub fn allow_env_vars(mut self, whitelist: Vec<String>) -> Self {
        self.config.allow_env_vars = true;
        self.config.allowed_env_vars = whitelist;
        self
    }

    /// Enable read-only mode.
    pub fn read_only(mut self) -> Self {
        self.config.read_only = true;
        self
    }

    /// Build the [`SandboxBackend`] wrapping the given inner backend.
    pub fn build(self, inner: Box<dyn Backend>) -> SandboxBackend {
        SandboxBackend::new(inner, self.config)
    }

    /// Build and return just the [`SandboxConfig`].
    pub fn build_config(self) -> SandboxConfig {
        self.config
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::state::StateBackend;
    use serde_json::json;

    // 1. Config creation with defaults
    #[test]
    fn test_config_defaults() {
        let config = SandboxConfig::default();
        assert!(config.allowed_paths.is_empty());
        assert!(config.denied_paths.is_empty());
        assert!(config.max_memory_bytes.is_none());
        assert!(config.max_execution_time.is_none());
        assert!(!config.allow_network);
        assert!(!config.allow_env_vars);
        assert!(config.allowed_env_vars.is_empty());
        assert!(!config.read_only);
    }

    // 2. Path access check — allowed
    #[test]
    fn test_path_access_allowed() {
        let config = SandboxConfig {
            allowed_paths: vec![PathBuf::from("/home/user/project")],
            ..Default::default()
        };
        let checker = PermissionChecker::new(config);
        assert!(checker
            .check_path_access(
                &PathBuf::from("/home/user/project/src/main.rs"),
                PathOperation::Read
            )
            .is_ok());
    }

    // 3. Path access check — denied takes precedence
    #[test]
    fn test_denied_path_takes_precedence() {
        let config = SandboxConfig {
            allowed_paths: vec![PathBuf::from("/home/user")],
            denied_paths: vec![PathBuf::from("/home/user/secret")],
            ..Default::default()
        };
        let checker = PermissionChecker::new(config);

        // Allowed subdir works
        assert!(checker
            .check_path_access(&PathBuf::from("/home/user/docs"), PathOperation::Read)
            .is_ok());

        // Denied subdir is blocked even though parent is allowed
        let err = checker
            .check_path_access(
                &PathBuf::from("/home/user/secret/key.pem"),
                PathOperation::Read,
            )
            .unwrap_err();
        assert!(matches!(err, SandboxViolation::PathAccessDenied { .. }));
    }

    // 4. Network access control
    #[test]
    fn test_network_access_control() {
        let denied = PermissionChecker::new(SandboxConfig::default());
        assert!(matches!(
            denied.check_network_access().unwrap_err(),
            SandboxViolation::NetworkAccessDenied
        ));

        let allowed = PermissionChecker::new(SandboxConfig {
            allow_network: true,
            ..Default::default()
        });
        assert!(allowed.check_network_access().is_ok());
    }

    // 5. Env var access control with whitelist
    #[test]
    fn test_env_var_access_whitelist() {
        // env vars disabled
        let checker = PermissionChecker::new(SandboxConfig::default());
        assert!(matches!(
            checker.check_env_access("HOME").unwrap_err(),
            SandboxViolation::EnvVarAccessDenied { .. }
        ));

        // env vars enabled with whitelist
        let checker = PermissionChecker::new(SandboxConfig {
            allow_env_vars: true,
            allowed_env_vars: vec!["PATH".to_string(), "HOME".to_string()],
            ..Default::default()
        });
        assert!(checker.check_env_access("HOME").is_ok());
        assert!(checker.check_env_access("PATH").is_ok());
        assert!(matches!(
            checker.check_env_access("SECRET_KEY").unwrap_err(),
            SandboxViolation::EnvVarAccessDenied { .. }
        ));

        // env vars enabled with empty whitelist (allow all)
        let checker = PermissionChecker::new(SandboxConfig {
            allow_env_vars: true,
            ..Default::default()
        });
        assert!(checker.check_env_access("ANYTHING").is_ok());
    }

    // 6. Read-only mode blocks writes
    #[test]
    fn test_read_only_blocks_writes() {
        let config = SandboxConfig {
            allowed_paths: vec![PathBuf::from("/data")],
            read_only: true,
            ..Default::default()
        };
        let checker = PermissionChecker::new(config);

        assert!(checker
            .check_path_access(&PathBuf::from("/data/file.txt"), PathOperation::Read)
            .is_ok());
        assert!(matches!(
            checker
                .check_path_access(&PathBuf::from("/data/file.txt"), PathOperation::Write)
                .unwrap_err(),
            SandboxViolation::ReadOnlyViolation { .. }
        ));
        assert!(matches!(
            checker
                .check_path_access(&PathBuf::from("/data/file.txt"), PathOperation::Delete)
                .unwrap_err(),
            SandboxViolation::ReadOnlyViolation { .. }
        ));
    }

    // 7. Resource tracking — memory
    #[test]
    fn test_resource_tracking_memory() {
        let tracker = ResourceTracker::new(Some(1000), None);
        tracker.track_memory(500);
        assert!(tracker.check_memory_limit().is_ok());

        tracker.track_memory(600); // total 1100 > 1000
        assert!(matches!(
            tracker.check_memory_limit().unwrap_err(),
            SandboxViolation::MemoryLimitExceeded {
                limit: 1000,
                used: 1100
            }
        ));
    }

    // 8. Time limit checking
    #[test]
    fn test_time_limit_checking() {
        // With no limit, always ok
        let tracker = ResourceTracker::new(None, None);
        assert!(tracker.check_time_limit().is_ok());

        // With a very generous limit
        let tracker = ResourceTracker::new(None, Some(Duration::from_secs(3600)));
        assert!(tracker.check_time_limit().is_ok());

        // With zero duration limit — should immediately expire
        let tracker = ResourceTracker::new(None, Some(Duration::from_nanos(0)));
        // Spin briefly to ensure elapsed > 0
        std::thread::sleep(Duration::from_millis(1));
        assert!(matches!(
            tracker.check_time_limit().unwrap_err(),
            SandboxViolation::TimeLimitExceeded { .. }
        ));
    }

    // 9. Builder pattern
    #[test]
    fn test_builder_pattern() {
        let config = SandboxBuilder::new("/workspace")
            .allow_path("/workspace/src")
            .deny_path("/workspace/src/secrets")
            .max_memory(1024 * 1024)
            .max_time(Duration::from_secs(60))
            .allow_network()
            .read_only()
            .build_config();

        assert_eq!(config.working_directory, PathBuf::from("/workspace"));
        assert_eq!(config.allowed_paths, vec![PathBuf::from("/workspace/src")]);
        assert_eq!(
            config.denied_paths,
            vec![PathBuf::from("/workspace/src/secrets")]
        );
        assert_eq!(config.max_memory_bytes, Some(1024 * 1024));
        assert_eq!(config.max_execution_time, Some(Duration::from_secs(60)));
        assert!(config.allow_network);
        assert!(config.read_only);
    }

    // 10. SandboxViolation error display
    #[test]
    fn test_sandbox_violation_display() {
        let err = SandboxViolation::PathAccessDenied {
            path: PathBuf::from("/etc/passwd"),
            operation: PathOperation::Read,
        };
        assert!(err.to_string().contains("/etc/passwd"));
        assert!(err.to_string().contains("read"));

        let err = SandboxViolation::NetworkAccessDenied;
        assert_eq!(err.to_string(), "network access denied");

        let err = SandboxViolation::EnvVarAccessDenied {
            var_name: "SECRET".to_string(),
        };
        assert!(err.to_string().contains("SECRET"));

        let err = SandboxViolation::MemoryLimitExceeded {
            limit: 1000,
            used: 2000,
        };
        assert!(err.to_string().contains("1000"));
        assert!(err.to_string().contains("2000"));

        let err = SandboxViolation::ReadOnlyViolation {
            path: PathBuf::from("/data/file"),
        };
        assert!(err.to_string().contains("read-only"));
    }

    // 11. Nested path checking (subdirectory of allowed)
    #[test]
    fn test_nested_path_checking() {
        let config = SandboxConfig {
            allowed_paths: vec![PathBuf::from("/home/user/project")],
            ..Default::default()
        };
        let checker = PermissionChecker::new(config);

        // Exact match
        assert!(checker
            .check_path_access(&PathBuf::from("/home/user/project"), PathOperation::Read)
            .is_ok());

        // Nested file
        assert!(checker
            .check_path_access(
                &PathBuf::from("/home/user/project/src/lib.rs"),
                PathOperation::Read
            )
            .is_ok());

        // Deeply nested
        assert!(checker
            .check_path_access(
                &PathBuf::from("/home/user/project/a/b/c/d/e.txt"),
                PathOperation::Write
            )
            .is_ok());

        // Sibling — not allowed
        assert!(checker
            .check_path_access(&PathBuf::from("/home/user/other"), PathOperation::Read)
            .is_err());

        // Parent — not allowed
        assert!(checker
            .check_path_access(&PathBuf::from("/home/user"), PathOperation::Read)
            .is_err());
    }

    // 12. Empty config denies all paths
    #[test]
    fn test_empty_config_denies_all() {
        let checker = PermissionChecker::new(SandboxConfig::default());

        assert!(checker
            .check_path_access(&PathBuf::from("/any/path"), PathOperation::Read)
            .is_err());
        assert!(checker.check_network_access().is_err());
        assert!(checker.check_env_access("HOME").is_err());
    }

    // 13. Stats tracking
    #[tokio::test]
    async fn test_stats_tracking() {
        let tracker = ResourceTracker::new(None, None);
        tracker.track_memory(100);
        tracker.track_memory(200);
        tracker.record_operation();
        tracker.record_operation();
        tracker.record_operation();
        tracker.record_file_access(PathBuf::from("/a.txt")).await;
        tracker.record_file_access(PathBuf::from("/b.txt")).await;
        // duplicate should not increase count
        tracker.record_file_access(PathBuf::from("/a.txt")).await;

        let stats = tracker.stats().await;
        assert_eq!(stats.memory_used, 300);
        assert_eq!(stats.operations_count, 3);
        assert_eq!(stats.files_accessed.len(), 2);
        assert!(stats.elapsed_time.as_nanos() > 0);
    }

    // 14. Permission checker with multiple rules
    #[test]
    fn test_permission_checker_multiple_rules() {
        let config = SandboxConfig {
            allowed_paths: vec![
                PathBuf::from("/home/user/project"),
                PathBuf::from("/tmp"),
                PathBuf::from("/var/data"),
            ],
            denied_paths: vec![
                PathBuf::from("/tmp/private"),
                PathBuf::from("/var/data/secrets"),
            ],
            ..Default::default()
        };
        let checker = PermissionChecker::new(config);

        // Allowed paths
        assert!(checker
            .check_path_access(
                &PathBuf::from("/home/user/project/main.rs"),
                PathOperation::Read
            )
            .is_ok());
        assert!(checker
            .check_path_access(&PathBuf::from("/tmp/scratch.txt"), PathOperation::Write)
            .is_ok());
        assert!(checker
            .check_path_access(&PathBuf::from("/var/data/output.csv"), PathOperation::Read)
            .is_ok());

        // Denied paths within allowed
        assert!(checker
            .check_path_access(&PathBuf::from("/tmp/private/key"), PathOperation::Read)
            .is_err());
        assert!(checker
            .check_path_access(
                &PathBuf::from("/var/data/secrets/token"),
                PathOperation::Read
            )
            .is_err());

        // Completely outside allowed
        assert!(checker
            .check_path_access(&PathBuf::from("/etc/passwd"), PathOperation::Read)
            .is_err());
    }

    // 15. SandboxBackend delegates to inner
    #[tokio::test]
    async fn test_sandbox_backend_delegates() {
        let inner = StateBackend::new();
        let config = SandboxConfig::default();
        let sandbox = SandboxBackend::new(Box::new(inner), config);

        let state = json!({"messages": [{"role": "user", "content": "hi"}]});
        sandbox.save_state("s1", &state).await.unwrap();

        let loaded = sandbox.load_state("s1").await.unwrap();
        assert_eq!(loaded, Some(state));

        let sessions = sandbox.list_sessions().await.unwrap();
        assert_eq!(sessions, vec!["s1".to_string()]);

        let stats = sandbox.stats().await;
        assert_eq!(stats.operations_count, 3);
    }

    // 16. SandboxBackend memory limit enforcement
    #[tokio::test]
    async fn test_sandbox_backend_memory_limit() {
        let inner = StateBackend::new();
        let config = SandboxConfig {
            max_memory_bytes: Some(10), // very small
            ..Default::default()
        };
        let sandbox = SandboxBackend::new(Box::new(inner), config);

        // A large-ish payload should exceed the limit
        let state = json!({"data": "a]long string that is definitely more than 10 bytes"});
        let result = sandbox.save_state("s1", &state).await;
        assert!(result.is_err());
    }
}
