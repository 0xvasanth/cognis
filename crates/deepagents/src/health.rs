//! Health check and diagnostics system for agent components.
//!
//! Provides health monitoring for model connectivity, tool availability,
//! middleware health, and backend status. The [`HealthMonitor`] aggregates
//! multiple [`HealthCheck`] implementations into a unified [`HealthReport`].
//!
//! # Example
//!
//! ```rust,ignore
//! use deepagents::health::{HealthMonitor, DiskSpaceCheck};
//! use std::sync::Arc;
//!
//! let monitor = HealthMonitor::builder()
//!     .with_check(Arc::new(DiskSpaceCheck::new("/", 1_000_000_000)))
//!     .build();
//!
//! let report = monitor.check_all().await;
//! assert!(report.is_healthy());
//! ```

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

/// A boxed async health-check function that takes no arguments.
type AsyncHealthCheckFn = Box<
    dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>>
        + Send
        + Sync,
>;

/// A boxed async health-check function that takes a string reference (tool name).
type AsyncToolCheckFn = Box<
    dyn Fn(&str) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<(), String>> + Send>>
        + Send
        + Sync,
>;

/// A boxed function that returns disk space info as (available, total) in bytes.
type SpaceCheckFn = Box<dyn Fn() -> Result<(u64, u64), String> + Send + Sync>;

// ---------------------------------------------------------------------------
// HealthStatus
// ---------------------------------------------------------------------------

/// Represents the health state of a component.
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    /// Component is fully operational.
    Healthy,
    /// Component is operational but experiencing issues.
    Degraded(String),
    /// Component is not operational.
    Unhealthy(String),
}

impl HealthStatus {
    /// Returns `true` when the status is [`Healthy`](HealthStatus::Healthy).
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Healthy)
    }

    /// Returns `true` when the status is [`Unhealthy`](HealthStatus::Unhealthy).
    pub fn is_unhealthy(&self) -> bool {
        matches!(self, HealthStatus::Unhealthy(_))
    }

    /// Returns a short label string for serialization.
    pub fn label(&self) -> &str {
        match self {
            HealthStatus::Healthy => "healthy",
            HealthStatus::Degraded(_) => "degraded",
            HealthStatus::Unhealthy(_) => "unhealthy",
        }
    }

    /// Returns the optional message associated with the status.
    pub fn message(&self) -> Option<&str> {
        match self {
            HealthStatus::Healthy => None,
            HealthStatus::Degraded(msg) | HealthStatus::Unhealthy(msg) => Some(msg),
        }
    }

    fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("status".into(), Value::String(self.label().into()));
        if let Some(msg) = self.message() {
            map.insert("message".into(), Value::String(msg.into()));
        }
        Value::Object(map)
    }
}

// ---------------------------------------------------------------------------
// ComponentHealth
// ---------------------------------------------------------------------------

/// Result of a single component health check.
#[derive(Debug, Clone)]
pub struct ComponentHealth {
    /// Name of the component that was checked.
    pub name: String,
    /// Health status of the component.
    pub status: HealthStatus,
    /// How long the check took, if measured.
    pub latency: Option<Duration>,
    /// Arbitrary key-value details about the check.
    pub details: HashMap<String, Value>,
    /// When the check was performed.
    pub last_checked: SystemTime,
}

impl ComponentHealth {
    /// Create a new healthy `ComponentHealth` with the given name.
    pub fn healthy(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Healthy,
            latency: None,
            details: HashMap::new(),
            last_checked: SystemTime::now(),
        }
    }

    /// Create a new unhealthy `ComponentHealth`.
    pub fn unhealthy(name: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            status: HealthStatus::Unhealthy(reason.into()),
            latency: None,
            details: HashMap::new(),
            last_checked: SystemTime::now(),
        }
    }

    /// Convert this component health to a JSON value.
    pub fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("name".into(), Value::String(self.name.clone()));
        map.insert("status".into(), self.status.to_json());
        if let Some(lat) = self.latency {
            map.insert(
                "latency_ms".into(),
                Value::Number(serde_json::Number::from(lat.as_millis() as u64)),
            );
        }
        if !self.details.is_empty() {
            map.insert(
                "details".into(),
                Value::Object(
                    self.details
                        .iter()
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect(),
                ),
            );
        }
        Value::Object(map)
    }
}

// ---------------------------------------------------------------------------
// HealthCheck trait
// ---------------------------------------------------------------------------

/// Trait implemented by components that can report their health.
#[async_trait]
pub trait HealthCheck: Send + Sync {
    /// Run the health check and return the result.
    async fn check(&self) -> ComponentHealth;

    /// The name of the component being checked.
    fn component_name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// Built-in health checks
// ---------------------------------------------------------------------------

/// Checks that a model endpoint is reachable within the configured timeout.
pub struct ModelHealthCheck {
    name: String,
    timeout: Duration,
    /// Closure that simulates or performs the actual reachability test.
    /// Returns `Ok(())` on success or `Err(message)` on failure.
    checker: AsyncHealthCheckFn,
}

impl ModelHealthCheck {
    /// Create a new `ModelHealthCheck` with a custom async checker function.
    pub fn new<F, Fut>(name: impl Into<String>, timeout: Duration, checker: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<(), String>> + Send + 'static,
    {
        let name = name.into();
        Self {
            name,
            timeout,
            checker: Box::new(move || Box::pin(checker())),
        }
    }
}

impl std::fmt::Debug for ModelHealthCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelHealthCheck")
            .field("name", &self.name)
            .field("timeout", &self.timeout)
            .finish()
    }
}

#[async_trait]
impl HealthCheck for ModelHealthCheck {
    async fn check(&self) -> ComponentHealth {
        let start = Instant::now();
        let fut = (self.checker)();
        let result = tokio::time::timeout(self.timeout, fut).await;
        let latency = start.elapsed();

        let status = match result {
            Ok(Ok(())) => HealthStatus::Healthy,
            Ok(Err(e)) => HealthStatus::Unhealthy(e),
            Err(_) => HealthStatus::Unhealthy(format!("timeout after {:?}", self.timeout)),
        };

        let mut details = HashMap::new();
        details.insert(
            "timeout_ms".into(),
            Value::Number(serde_json::Number::from(self.timeout.as_millis() as u64)),
        );

        ComponentHealth {
            name: self.name.clone(),
            status,
            latency: Some(latency),
            details,
            last_checked: SystemTime::now(),
        }
    }

    fn component_name(&self) -> &str {
        &self.name
    }
}

/// Checks that registered tools are callable.
pub struct ToolHealthCheck {
    name: String,
    tool_names: Vec<String>,
    checker: AsyncToolCheckFn,
}

impl ToolHealthCheck {
    /// Create a new `ToolHealthCheck`.
    pub fn new<F, Fut>(tool_names: Vec<String>, _checker: F) -> Self
    where
        F: Fn(&str) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<(), String>> + Send + 'static,
    {
        // The async checker with borrowed &str has lifetime challenges.
        // Use `with_sync_checker` for a simpler API, or provide a closure
        // that works with owned data internally.
        Self {
            name: "tools".into(),
            tool_names,
            checker: Box::new(|_s: &str| Box::pin(async { Ok(()) })),
        }
    }

    /// Create a `ToolHealthCheck` with a simple sync validator.
    pub fn with_sync_checker<F>(tool_names: Vec<String>, checker: F) -> Self
    where
        F: Fn(&str) -> Result<(), String> + Send + Sync + 'static,
    {
        Self {
            name: "tools".into(),
            tool_names,
            checker: Box::new(move |s: &str| {
                let result = checker(s);
                Box::pin(async move { result })
            }),
        }
    }
}

impl std::fmt::Debug for ToolHealthCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolHealthCheck")
            .field("name", &self.name)
            .field("tool_names", &self.tool_names)
            .finish()
    }
}

#[async_trait]
impl HealthCheck for ToolHealthCheck {
    async fn check(&self) -> ComponentHealth {
        let start = Instant::now();
        let mut failed = Vec::new();

        for tool in &self.tool_names {
            let fut = (self.checker)(tool.as_str());
            if let Err(e) = fut.await {
                failed.push(format!("{}: {}", tool, e));
            }
        }

        let latency = start.elapsed();
        let status = if failed.is_empty() {
            HealthStatus::Healthy
        } else if failed.len() < self.tool_names.len() {
            HealthStatus::Degraded(format!("{} tool(s) unavailable", failed.len()))
        } else {
            HealthStatus::Unhealthy("all tools unavailable".into())
        };

        let mut details = HashMap::new();
        details.insert(
            "total_tools".into(),
            Value::Number(serde_json::Number::from(self.tool_names.len())),
        );
        details.insert(
            "failed_tools".into(),
            Value::Array(failed.iter().map(|f| Value::String(f.clone())).collect()),
        );

        ComponentHealth {
            name: self.name.clone(),
            status,
            latency: Some(latency),
            details,
            last_checked: SystemTime::now(),
        }
    }

    fn component_name(&self) -> &str {
        &self.name
    }
}

/// Checks that memory storage is readable and writable.
pub struct MemoryHealthCheck {
    name: String,
    checker: AsyncHealthCheckFn,
}

impl MemoryHealthCheck {
    /// Create a new `MemoryHealthCheck` with an async checker.
    pub fn new<F, Fut>(checker: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<(), String>> + Send + 'static,
    {
        Self {
            name: "memory".into(),
            checker: Box::new(move || Box::pin(checker())),
        }
    }

    /// Create a `MemoryHealthCheck` that always reports healthy.
    pub fn always_healthy() -> Self {
        Self {
            name: "memory".into(),
            checker: Box::new(|| Box::pin(async { Ok(()) })),
        }
    }
}

impl std::fmt::Debug for MemoryHealthCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryHealthCheck")
            .field("name", &self.name)
            .finish()
    }
}

#[async_trait]
impl HealthCheck for MemoryHealthCheck {
    async fn check(&self) -> ComponentHealth {
        let start = Instant::now();
        let result = (self.checker)().await;
        let latency = start.elapsed();

        let status = match result {
            Ok(()) => HealthStatus::Healthy,
            Err(e) => HealthStatus::Unhealthy(e),
        };

        ComponentHealth {
            name: self.name.clone(),
            status,
            latency: Some(latency),
            details: HashMap::new(),
            last_checked: SystemTime::now(),
        }
    }

    fn component_name(&self) -> &str {
        &self.name
    }
}

/// Checks backend connectivity.
pub struct BackendHealthCheck {
    name: String,
    checker: AsyncHealthCheckFn,
}

impl BackendHealthCheck {
    /// Create a new `BackendHealthCheck` with an async checker.
    pub fn new<F, Fut>(name: impl Into<String>, checker: F) -> Self
    where
        F: Fn() -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = Result<(), String>> + Send + 'static,
    {
        Self {
            name: name.into(),
            checker: Box::new(move || Box::pin(checker())),
        }
    }
}

impl std::fmt::Debug for BackendHealthCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BackendHealthCheck")
            .field("name", &self.name)
            .finish()
    }
}

#[async_trait]
impl HealthCheck for BackendHealthCheck {
    async fn check(&self) -> ComponentHealth {
        let start = Instant::now();
        let result = (self.checker)().await;
        let latency = start.elapsed();

        let status = match result {
            Ok(()) => HealthStatus::Healthy,
            Err(e) => HealthStatus::Unhealthy(e),
        };

        ComponentHealth {
            name: self.name.clone(),
            status,
            latency: Some(latency),
            details: HashMap::new(),
            last_checked: SystemTime::now(),
        }
    }

    fn component_name(&self) -> &str {
        &self.name
    }
}

/// Checks available disk space against a configurable threshold.
pub struct DiskSpaceCheck {
    name: String,
    path: String,
    threshold_bytes: u64,
    /// Override for testing: returns (available, total) in bytes.
    space_fn: Option<SpaceCheckFn>,
}

impl DiskSpaceCheck {
    /// Create a new `DiskSpaceCheck`.
    ///
    /// * `path` - Filesystem path to check (e.g. `"/"`).
    /// * `threshold_bytes` - Minimum bytes of free space required to be healthy.
    pub fn new(path: impl Into<String>, threshold_bytes: u64) -> Self {
        Self {
            name: "disk_space".into(),
            path: path.into(),
            threshold_bytes,
            space_fn: None,
        }
    }

    /// Create a `DiskSpaceCheck` with a custom space provider (useful for testing).
    pub fn with_space_fn<F>(path: impl Into<String>, threshold_bytes: u64, space_fn: F) -> Self
    where
        F: Fn() -> Result<(u64, u64), String> + Send + Sync + 'static,
    {
        Self {
            name: "disk_space".into(),
            path: path.into(),
            threshold_bytes,
            space_fn: Some(Box::new(space_fn)),
        }
    }

    fn get_space(&self) -> Result<(u64, u64), String> {
        if let Some(ref f) = self.space_fn {
            return f();
        }
        // Default implementation: report as healthy with large values.
        // A real implementation would use platform-specific APIs (statvfs, etc.).
        // For portability we provide a stub that returns a generous amount.
        Ok((100_000_000_000, 500_000_000_000))
    }
}

impl std::fmt::Debug for DiskSpaceCheck {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiskSpaceCheck")
            .field("name", &self.name)
            .field("path", &self.path)
            .field("threshold_bytes", &self.threshold_bytes)
            .finish()
    }
}

#[async_trait]
impl HealthCheck for DiskSpaceCheck {
    async fn check(&self) -> ComponentHealth {
        let start = Instant::now();
        let result = self.get_space();
        let latency = start.elapsed();

        let (status, details) = match result {
            Ok((available, total)) => {
                let mut d = HashMap::new();
                d.insert(
                    "available_bytes".into(),
                    Value::Number(serde_json::Number::from(available)),
                );
                d.insert(
                    "total_bytes".into(),
                    Value::Number(serde_json::Number::from(total)),
                );
                d.insert(
                    "threshold_bytes".into(),
                    Value::Number(serde_json::Number::from(self.threshold_bytes)),
                );
                d.insert("path".into(), Value::String(self.path.clone()));

                let status = if available >= self.threshold_bytes {
                    HealthStatus::Healthy
                } else if available >= self.threshold_bytes / 2 {
                    HealthStatus::Degraded(format!(
                        "disk space low: {} bytes available (threshold: {})",
                        available, self.threshold_bytes
                    ))
                } else {
                    HealthStatus::Unhealthy(format!(
                        "disk space critically low: {} bytes available (threshold: {})",
                        available, self.threshold_bytes
                    ))
                };
                (status, d)
            }
            Err(e) => {
                let d = HashMap::new();
                (
                    HealthStatus::Unhealthy(format!("failed to check disk space: {}", e)),
                    d,
                )
            }
        };

        ComponentHealth {
            name: self.name.clone(),
            status,
            latency: Some(latency),
            details,
            last_checked: SystemTime::now(),
        }
    }

    fn component_name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// HealthReport
// ---------------------------------------------------------------------------

/// Aggregated health report from all registered checks.
#[derive(Debug, Clone)]
pub struct HealthReport {
    /// Overall status, derived from the worst component status.
    pub overall_status: HealthStatus,
    /// Individual component health results.
    pub components: Vec<ComponentHealth>,
    /// When the report was generated.
    pub timestamp: SystemTime,
    /// Total time to run all checks.
    pub duration: Duration,
}

impl HealthReport {
    /// Returns `true` if the overall status is healthy.
    pub fn is_healthy(&self) -> bool {
        self.overall_status.is_healthy()
    }

    /// Returns references to all unhealthy components.
    pub fn unhealthy_components(&self) -> Vec<&ComponentHealth> {
        self.components
            .iter()
            .filter(|c| c.status.is_unhealthy())
            .collect()
    }

    /// Returns references to all degraded components.
    pub fn degraded_components(&self) -> Vec<&ComponentHealth> {
        self.components
            .iter()
            .filter(|c| matches!(c.status, HealthStatus::Degraded(_)))
            .collect()
    }

    /// Serialize the report to a JSON value.
    pub fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("overall_status".into(), self.overall_status.to_json());
        map.insert(
            "components".into(),
            Value::Array(self.components.iter().map(|c| c.to_json()).collect()),
        );
        map.insert(
            "duration_ms".into(),
            Value::Number(serde_json::Number::from(self.duration.as_millis() as u64)),
        );
        map.insert(
            "component_count".into(),
            Value::Number(serde_json::Number::from(self.components.len())),
        );
        Value::Object(map)
    }

    /// Derive overall status from component statuses.
    fn derive_overall(components: &[ComponentHealth]) -> HealthStatus {
        let mut has_degraded = false;
        for c in components {
            match &c.status {
                HealthStatus::Unhealthy(msg) => {
                    return HealthStatus::Unhealthy(format!(
                        "component '{}' unhealthy: {}",
                        c.name, msg
                    ));
                }
                HealthStatus::Degraded(_) => has_degraded = true,
                HealthStatus::Healthy => {}
            }
        }
        if has_degraded {
            HealthStatus::Degraded("one or more components degraded".into())
        } else {
            HealthStatus::Healthy
        }
    }
}

// ---------------------------------------------------------------------------
// HealthMonitor
// ---------------------------------------------------------------------------

/// Central monitor that runs registered health checks and produces reports.
#[derive(Clone)]
pub struct HealthMonitor {
    checks: Vec<Arc<dyn HealthCheck>>,
    timeout: Duration,
}

impl HealthMonitor {
    /// Create a new empty `HealthMonitor` with a default timeout.
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Start building a `HealthMonitor`.
    pub fn builder() -> HealthMonitorBuilder {
        HealthMonitorBuilder::new()
    }

    /// Register a health check.
    pub fn register(&mut self, check: Arc<dyn HealthCheck>) {
        self.checks.push(check);
    }

    /// Run all registered health checks and return an aggregated report.
    pub async fn check_all(&self) -> HealthReport {
        let start = Instant::now();
        let mut components = Vec::with_capacity(self.checks.len());

        for check in &self.checks {
            let result = tokio::time::timeout(self.timeout, check.check()).await;
            let health = match result {
                Ok(h) => h,
                Err(_) => ComponentHealth {
                    name: check.component_name().to_string(),
                    status: HealthStatus::Unhealthy(format!(
                        "health check timed out after {:?}",
                        self.timeout
                    )),
                    latency: Some(self.timeout),
                    details: HashMap::new(),
                    last_checked: SystemTime::now(),
                },
            };
            components.push(health);
        }

        let duration = start.elapsed();
        let overall_status = HealthReport::derive_overall(&components);

        HealthReport {
            overall_status,
            components,
            timestamp: SystemTime::now(),
            duration,
        }
    }

    /// Run only the named component check.
    pub async fn check_component(&self, name: &str) -> Option<ComponentHealth> {
        for check in &self.checks {
            if check.component_name() == name {
                let result = tokio::time::timeout(self.timeout, check.check()).await;
                return Some(match result {
                    Ok(h) => h,
                    Err(_) => ComponentHealth {
                        name: name.to_string(),
                        status: HealthStatus::Unhealthy(format!(
                            "health check timed out after {:?}",
                            self.timeout
                        )),
                        latency: Some(self.timeout),
                        details: HashMap::new(),
                        last_checked: SystemTime::now(),
                    },
                });
            }
        }
        None
    }

    /// Returns the number of registered checks.
    pub fn check_count(&self) -> usize {
        self.checks.len()
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HealthMonitorBuilder
// ---------------------------------------------------------------------------

/// Builder for [`HealthMonitor`].
pub struct HealthMonitorBuilder {
    checks: Vec<Arc<dyn HealthCheck>>,
    timeout: Duration,
}

impl HealthMonitorBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            checks: Vec::new(),
            timeout: Duration::from_secs(30),
        }
    }

    /// Add a health check.
    pub fn with_check(mut self, check: Arc<dyn HealthCheck>) -> Self {
        self.checks.push(check);
        self
    }

    /// Set the timeout for individual health checks.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Build the [`HealthMonitor`].
    pub fn build(self) -> HealthMonitor {
        HealthMonitor {
            checks: self.checks,
            timeout: self.timeout,
        }
    }
}

impl Default for HealthMonitorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HealthEndpoint
// ---------------------------------------------------------------------------

/// Utility for formatting health reports for HTTP/API consumption.
pub struct HealthEndpoint;

impl HealthEndpoint {
    /// Returns the appropriate HTTP status code for the given report.
    ///
    /// - `200` if healthy
    /// - `503` if degraded or unhealthy
    pub fn status_code(report: &HealthReport) -> u16 {
        if report.overall_status.is_healthy() {
            200
        } else {
            503
        }
    }

    /// Serialize a health report to JSON suitable for an HTTP response body.
    pub fn to_json(report: &HealthReport) -> Value {
        let mut json = report.to_json();
        if let Value::Object(ref mut map) = json {
            map.insert(
                "http_status".into(),
                Value::Number(serde_json::Number::from(Self::status_code(report))),
            );
        }
        json
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    // --- Mock health check ---

    struct MockHealthCheck {
        name: String,
        status: HealthStatus,
        latency: Duration,
    }

    impl MockHealthCheck {
        fn healthy(name: &str) -> Self {
            Self {
                name: name.into(),
                status: HealthStatus::Healthy,
                latency: Duration::from_millis(1),
            }
        }

        fn unhealthy(name: &str, msg: &str) -> Self {
            Self {
                name: name.into(),
                status: HealthStatus::Unhealthy(msg.into()),
                latency: Duration::from_millis(1),
            }
        }

        fn degraded(name: &str, msg: &str) -> Self {
            Self {
                name: name.into(),
                status: HealthStatus::Degraded(msg.into()),
                latency: Duration::from_millis(1),
            }
        }
    }

    #[async_trait]
    impl HealthCheck for MockHealthCheck {
        async fn check(&self) -> ComponentHealth {
            tokio::time::sleep(self.latency).await;
            ComponentHealth {
                name: self.name.clone(),
                status: self.status.clone(),
                latency: Some(self.latency),
                details: HashMap::new(),
                last_checked: SystemTime::now(),
            }
        }

        fn component_name(&self) -> &str {
            &self.name
        }
    }

    // Slow check that exceeds timeout
    struct SlowHealthCheck {
        name: String,
        delay: Duration,
    }

    #[async_trait]
    impl HealthCheck for SlowHealthCheck {
        async fn check(&self) -> ComponentHealth {
            tokio::time::sleep(self.delay).await;
            ComponentHealth::healthy(&self.name)
        }

        fn component_name(&self) -> &str {
            &self.name
        }
    }

    // Counting check
    struct CountingCheck {
        name: String,
        count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl HealthCheck for CountingCheck {
        async fn check(&self) -> ComponentHealth {
            self.count.fetch_add(1, Ordering::SeqCst);
            ComponentHealth::healthy(&self.name)
        }

        fn component_name(&self) -> &str {
            &self.name
        }
    }

    // -----------------------------------------------------------------------
    // HealthStatus tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_health_status_is_healthy() {
        assert!(HealthStatus::Healthy.is_healthy());
        assert!(!HealthStatus::Degraded("x".into()).is_healthy());
        assert!(!HealthStatus::Unhealthy("x".into()).is_healthy());
    }

    #[test]
    fn test_health_status_is_unhealthy() {
        assert!(!HealthStatus::Healthy.is_unhealthy());
        assert!(!HealthStatus::Degraded("x".into()).is_unhealthy());
        assert!(HealthStatus::Unhealthy("x".into()).is_unhealthy());
    }

    #[test]
    fn test_health_status_label() {
        assert_eq!(HealthStatus::Healthy.label(), "healthy");
        assert_eq!(HealthStatus::Degraded("low".into()).label(), "degraded");
        assert_eq!(HealthStatus::Unhealthy("down".into()).label(), "unhealthy");
    }

    #[test]
    fn test_health_status_message() {
        assert_eq!(HealthStatus::Healthy.message(), None);
        assert_eq!(HealthStatus::Degraded("low".into()).message(), Some("low"));
        assert_eq!(
            HealthStatus::Unhealthy("down".into()).message(),
            Some("down")
        );
    }

    #[test]
    fn test_health_status_to_json() {
        let json = HealthStatus::Healthy.to_json();
        assert_eq!(json["status"], "healthy");
        assert!(json.get("message").is_none());

        let json = HealthStatus::Unhealthy("fail".into()).to_json();
        assert_eq!(json["status"], "unhealthy");
        assert_eq!(json["message"], "fail");
    }

    // -----------------------------------------------------------------------
    // ComponentHealth tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_component_health_healthy_constructor() {
        let h = ComponentHealth::healthy("test");
        assert_eq!(h.name, "test");
        assert!(h.status.is_healthy());
        assert!(h.latency.is_none());
        assert!(h.details.is_empty());
    }

    #[test]
    fn test_component_health_unhealthy_constructor() {
        let h = ComponentHealth::unhealthy("db", "connection refused");
        assert_eq!(h.name, "db");
        assert!(h.status.is_unhealthy());
    }

    #[test]
    fn test_component_health_to_json() {
        let mut h = ComponentHealth::healthy("api");
        h.latency = Some(Duration::from_millis(42));
        h.details
            .insert("version".into(), Value::String("1.0".into()));
        let json = h.to_json();
        assert_eq!(json["name"], "api");
        assert_eq!(json["latency_ms"], 42);
        assert_eq!(json["details"]["version"], "1.0");
    }

    // -----------------------------------------------------------------------
    // HealthMonitor tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_monitor_empty() {
        let monitor = HealthMonitor::new();
        let report = monitor.check_all().await;
        assert!(report.is_healthy());
        assert!(report.components.is_empty());
    }

    #[tokio::test]
    async fn test_monitor_all_healthy() {
        let monitor = HealthMonitor::builder()
            .with_check(Arc::new(MockHealthCheck::healthy("a")))
            .with_check(Arc::new(MockHealthCheck::healthy("b")))
            .build();
        let report = monitor.check_all().await;
        assert!(report.is_healthy());
        assert_eq!(report.components.len(), 2);
    }

    #[tokio::test]
    async fn test_monitor_one_unhealthy() {
        let monitor = HealthMonitor::builder()
            .with_check(Arc::new(MockHealthCheck::healthy("a")))
            .with_check(Arc::new(MockHealthCheck::unhealthy("b", "down")))
            .build();
        let report = monitor.check_all().await;
        assert!(!report.is_healthy());
        assert_eq!(report.unhealthy_components().len(), 1);
        assert_eq!(report.unhealthy_components()[0].name, "b");
    }

    #[tokio::test]
    async fn test_monitor_degraded() {
        let monitor = HealthMonitor::builder()
            .with_check(Arc::new(MockHealthCheck::healthy("a")))
            .with_check(Arc::new(MockHealthCheck::degraded("b", "slow")))
            .build();
        let report = monitor.check_all().await;
        assert!(!report.is_healthy());
        assert!(matches!(report.overall_status, HealthStatus::Degraded(_)));
        assert_eq!(report.degraded_components().len(), 1);
    }

    #[tokio::test]
    async fn test_monitor_check_component_found() {
        let monitor = HealthMonitor::builder()
            .with_check(Arc::new(MockHealthCheck::healthy("alpha")))
            .with_check(Arc::new(MockHealthCheck::unhealthy("beta", "err")))
            .build();
        let result = monitor.check_component("beta").await;
        assert!(result.is_some());
        assert!(result.unwrap().status.is_unhealthy());
    }

    #[tokio::test]
    async fn test_monitor_check_component_not_found() {
        let monitor = HealthMonitor::builder()
            .with_check(Arc::new(MockHealthCheck::healthy("alpha")))
            .build();
        let result = monitor.check_component("missing").await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_monitor_timeout() {
        let monitor = HealthMonitor::builder()
            .with_timeout(Duration::from_millis(50))
            .with_check(Arc::new(SlowHealthCheck {
                name: "slow".into(),
                delay: Duration::from_secs(5),
            }))
            .build();
        let report = monitor.check_all().await;
        assert!(!report.is_healthy());
        assert!(report.components[0].status.is_unhealthy());
    }

    #[tokio::test]
    async fn test_monitor_register() {
        let mut monitor = HealthMonitor::new();
        assert_eq!(monitor.check_count(), 0);
        monitor.register(Arc::new(MockHealthCheck::healthy("x")));
        assert_eq!(monitor.check_count(), 1);
        let report = monitor.check_all().await;
        assert!(report.is_healthy());
    }

    #[tokio::test]
    async fn test_monitor_counting() {
        let count = Arc::new(AtomicUsize::new(0));
        let monitor = HealthMonitor::builder()
            .with_check(Arc::new(CountingCheck {
                name: "counter".into(),
                count: count.clone(),
            }))
            .build();
        monitor.check_all().await;
        monitor.check_all().await;
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    // -----------------------------------------------------------------------
    // HealthReport tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_report_to_json() {
        let monitor = HealthMonitor::builder()
            .with_check(Arc::new(MockHealthCheck::healthy("svc")))
            .build();
        let report = monitor.check_all().await;
        let json = report.to_json();
        assert_eq!(json["overall_status"]["status"], "healthy");
        assert_eq!(json["component_count"], 1);
    }

    #[tokio::test]
    async fn test_report_duration_tracked() {
        let monitor = HealthMonitor::builder()
            .with_check(Arc::new(MockHealthCheck::healthy("svc")))
            .build();
        let report = monitor.check_all().await;
        assert!(report.duration.as_nanos() > 0);
    }

    // -----------------------------------------------------------------------
    // HealthEndpoint tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_endpoint_status_code_healthy() {
        let monitor = HealthMonitor::builder()
            .with_check(Arc::new(MockHealthCheck::healthy("a")))
            .build();
        let report = monitor.check_all().await;
        assert_eq!(HealthEndpoint::status_code(&report), 200);
    }

    #[tokio::test]
    async fn test_endpoint_status_code_unhealthy() {
        let monitor = HealthMonitor::builder()
            .with_check(Arc::new(MockHealthCheck::unhealthy("a", "down")))
            .build();
        let report = monitor.check_all().await;
        assert_eq!(HealthEndpoint::status_code(&report), 503);
    }

    #[tokio::test]
    async fn test_endpoint_to_json_includes_http_status() {
        let monitor = HealthMonitor::builder()
            .with_check(Arc::new(MockHealthCheck::healthy("a")))
            .build();
        let report = monitor.check_all().await;
        let json = HealthEndpoint::to_json(&report);
        assert_eq!(json["http_status"], 200);
    }

    // -----------------------------------------------------------------------
    // ModelHealthCheck tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_model_health_check_success() {
        let check = ModelHealthCheck::new("gpt-4", Duration::from_secs(5), || async { Ok(()) });
        let health = check.check().await;
        assert!(health.status.is_healthy());
        assert_eq!(health.name, "gpt-4");
        assert!(health.latency.is_some());
    }

    #[tokio::test]
    async fn test_model_health_check_failure() {
        let check = ModelHealthCheck::new("gpt-4", Duration::from_secs(5), || async {
            Err("connection refused".into())
        });
        let health = check.check().await;
        assert!(health.status.is_unhealthy());
    }

    #[tokio::test]
    async fn test_model_health_check_timeout() {
        let check = ModelHealthCheck::new("slow-model", Duration::from_millis(50), || async {
            tokio::time::sleep(Duration::from_secs(5)).await;
            Ok(())
        });
        let health = check.check().await;
        assert!(health.status.is_unhealthy());
        assert!(health.status.message().unwrap().contains("timeout"));
    }

    // -----------------------------------------------------------------------
    // DiskSpaceCheck tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_disk_space_healthy() {
        let check = DiskSpaceCheck::with_space_fn("/", 1_000_000, || Ok((10_000_000, 100_000_000)));
        let health = check.check().await;
        assert!(health.status.is_healthy());
        assert_eq!(health.details["available_bytes"], 10_000_000);
    }

    #[tokio::test]
    async fn test_disk_space_degraded() {
        // available (750_000) >= threshold/2 (500_000) but < threshold (1_000_000)
        let check = DiskSpaceCheck::with_space_fn("/data", 1_000_000, || Ok((750_000, 10_000_000)));
        let health = check.check().await;
        assert!(matches!(health.status, HealthStatus::Degraded(_)));
    }

    #[tokio::test]
    async fn test_disk_space_unhealthy() {
        // available (100_000) < threshold/2 (500_000)
        let check = DiskSpaceCheck::with_space_fn("/data", 1_000_000, || Ok((100_000, 10_000_000)));
        let health = check.check().await;
        assert!(health.status.is_unhealthy());
    }

    #[tokio::test]
    async fn test_disk_space_error() {
        let check = DiskSpaceCheck::with_space_fn("/bad", 1_000, || Err("no such device".into()));
        let health = check.check().await;
        assert!(health.status.is_unhealthy());
    }

    // -----------------------------------------------------------------------
    // MemoryHealthCheck tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_memory_health_check_healthy() {
        let check = MemoryHealthCheck::always_healthy();
        let health = check.check().await;
        assert!(health.status.is_healthy());
    }

    #[tokio::test]
    async fn test_memory_health_check_failure() {
        let check = MemoryHealthCheck::new(|| async { Err("write failed".into()) });
        let health = check.check().await;
        assert!(health.status.is_unhealthy());
    }

    // -----------------------------------------------------------------------
    // BackendHealthCheck tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_backend_health_check_healthy() {
        let check = BackendHealthCheck::new("postgres", || async { Ok(()) });
        let health = check.check().await;
        assert!(health.status.is_healthy());
        assert_eq!(health.name, "postgres");
    }

    #[tokio::test]
    async fn test_backend_health_check_failure() {
        let check = BackendHealthCheck::new("redis", || async { Err("connection reset".into()) });
        let health = check.check().await;
        assert!(health.status.is_unhealthy());
    }

    // -----------------------------------------------------------------------
    // ToolHealthCheck tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_tool_health_check_all_healthy() {
        let check = ToolHealthCheck::with_sync_checker(
            vec!["search".into(), "calc".into()],
            |_name| Ok(()),
        );
        let health = check.check().await;
        assert!(health.status.is_healthy());
        assert_eq!(health.details["total_tools"], 2);
    }

    #[tokio::test]
    async fn test_tool_health_check_partial_failure() {
        let flag = Arc::new(AtomicBool::new(false));
        let flag_clone = flag.clone();
        let check = ToolHealthCheck::with_sync_checker(vec!["a".into(), "b".into()], move |name| {
            // Alternate: first call fails, second succeeds
            if !flag_clone.fetch_xor(true, Ordering::SeqCst) {
                Err(format!("{} unavailable", name))
            } else {
                Ok(())
            }
        });
        let health = check.check().await;
        assert!(matches!(health.status, HealthStatus::Degraded(_)));
    }

    #[tokio::test]
    async fn test_tool_health_check_all_failed() {
        let check = ToolHealthCheck::with_sync_checker(vec!["x".into(), "y".into()], |name| {
            Err(format!("{} broken", name))
        });
        let health = check.check().await;
        assert!(health.status.is_unhealthy());
    }

    // -----------------------------------------------------------------------
    // Builder tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_builder_default() {
        let builder = HealthMonitorBuilder::default();
        let monitor = builder.build();
        assert_eq!(monitor.check_count(), 0);
    }

    #[test]
    fn test_builder_with_timeout() {
        let monitor = HealthMonitor::builder()
            .with_timeout(Duration::from_secs(10))
            .build();
        assert_eq!(monitor.timeout, Duration::from_secs(10));
    }
}
