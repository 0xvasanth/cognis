//! Observability and diagnostics for agent execution.
//!
//! Complements the [`telemetry`](crate::telemetry) module (which focuses on raw metric
//! collection, spans, and token usage) by providing higher-level diagnostic events,
//! per-agent profiling, threshold-based alerting, and a unified dashboard view.
//!
//! # Key types
//!
//! - [`DiagnosticEvent`] -- typed events emitted during agent execution
//! - [`DiagnosticSink`] -- pluggable consumer of diagnostic events
//! - [`InMemoryDiagnosticSink`] -- queryable in-memory event store
//! - [`AgentProfiler`] -- per-agent timing breakdown (LLM, tools, middleware)
//! - [`ProfileReport`] -- summary of profiling data with percentages and counts
//! - [`AlertRule`] / [`AlertManager`] -- threshold-based alert evaluation
//! - [`DiagnosticDashboard`] -- aggregates sinks, profilers, and alerts

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant, SystemTime};

// ---------------------------------------------------------------------------
// Severity
// ---------------------------------------------------------------------------

/// Severity level for alerts and diagnostic events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Severity {
    /// Informational notice.
    Info,
    /// Something may need attention.
    Warning,
    /// A serious problem requiring immediate attention.
    Critical,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "info"),
            Self::Warning => write!(f, "warning"),
            Self::Critical => write!(f, "critical"),
        }
    }
}

// ---------------------------------------------------------------------------
// DiagnosticEvent
// ---------------------------------------------------------------------------

/// A typed diagnostic event emitted during agent execution.
#[derive(Debug, Clone)]
pub enum DiagnosticEvent {
    /// An agent started executing.
    AgentStarted {
        /// Agent identifier.
        agent_id: String,
        /// When the agent started.
        timestamp: SystemTime,
    },
    /// An agent completed execution.
    AgentCompleted {
        /// Agent identifier.
        agent_id: String,
        /// Total wall-clock duration.
        duration: Duration,
        /// Whether the execution was successful.
        success: bool,
        /// When the agent completed.
        timestamp: SystemTime,
    },
    /// A tool was called during execution.
    ToolCalled {
        /// Agent identifier.
        agent_id: String,
        /// Tool name.
        tool_name: String,
        /// How long the tool call took.
        duration: Duration,
        /// Whether the call succeeded.
        success: bool,
        /// When the tool was called.
        timestamp: SystemTime,
    },
    /// An error occurred during execution.
    ErrorOccurred {
        /// Agent identifier.
        agent_id: String,
        /// Error message.
        message: String,
        /// Error severity.
        severity: Severity,
        /// When the error occurred.
        timestamp: SystemTime,
    },
    /// A configured threshold was exceeded.
    ThresholdExceeded {
        /// Name of the metric that exceeded its threshold.
        metric_name: String,
        /// The threshold value.
        threshold: f64,
        /// The actual observed value.
        actual: f64,
        /// When the threshold was exceeded.
        timestamp: SystemTime,
    },
    /// Memory pressure detected.
    MemoryPressure {
        /// Agent identifier.
        agent_id: String,
        /// Current memory usage in bytes.
        usage_bytes: u64,
        /// Configured limit in bytes.
        limit_bytes: u64,
        /// When the pressure was detected.
        timestamp: SystemTime,
    },
    /// A middleware executed.
    MiddlewareExecuted {
        /// Agent identifier.
        agent_id: String,
        /// Middleware name.
        middleware_name: String,
        /// How long it took.
        duration: Duration,
        /// When it was executed.
        timestamp: SystemTime,
    },
    /// An LLM call was made.
    LlmCallCompleted {
        /// Agent identifier.
        agent_id: String,
        /// Model name.
        model: String,
        /// How long the call took.
        duration: Duration,
        /// When the call completed.
        timestamp: SystemTime,
    },
}

impl DiagnosticEvent {
    /// Return the agent ID associated with this event, if any.
    pub fn agent_id(&self) -> Option<&str> {
        match self {
            Self::AgentStarted { agent_id, .. }
            | Self::AgentCompleted { agent_id, .. }
            | Self::ToolCalled { agent_id, .. }
            | Self::ErrorOccurred { agent_id, .. }
            | Self::MemoryPressure { agent_id, .. }
            | Self::MiddlewareExecuted { agent_id, .. }
            | Self::LlmCallCompleted { agent_id, .. } => Some(agent_id),
            Self::ThresholdExceeded { .. } => None,
        }
    }

    /// Return the timestamp of this event.
    pub fn timestamp(&self) -> SystemTime {
        match self {
            Self::AgentStarted { timestamp, .. }
            | Self::AgentCompleted { timestamp, .. }
            | Self::ToolCalled { timestamp, .. }
            | Self::ErrorOccurred { timestamp, .. }
            | Self::ThresholdExceeded { timestamp, .. }
            | Self::MemoryPressure { timestamp, .. }
            | Self::MiddlewareExecuted { timestamp, .. }
            | Self::LlmCallCompleted { timestamp, .. } => *timestamp,
        }
    }

    /// Return a short label describing the event kind.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::AgentStarted { .. } => "agent_started",
            Self::AgentCompleted { .. } => "agent_completed",
            Self::ToolCalled { .. } => "tool_called",
            Self::ErrorOccurred { .. } => "error_occurred",
            Self::ThresholdExceeded { .. } => "threshold_exceeded",
            Self::MemoryPressure { .. } => "memory_pressure",
            Self::MiddlewareExecuted { .. } => "middleware_executed",
            Self::LlmCallCompleted { .. } => "llm_call_completed",
        }
    }
}

// ---------------------------------------------------------------------------
// DiagnosticSink
// ---------------------------------------------------------------------------

/// Trait for pluggable diagnostic event consumers.
///
/// Implementations receive events and can store, forward, or process them
/// as needed.
pub trait DiagnosticSink {
    /// Receive a diagnostic event.
    fn receive(&mut self, event: DiagnosticEvent);

    /// Flush any buffered events. Default implementation is a no-op.
    fn flush(&mut self) {}
}

// ---------------------------------------------------------------------------
// InMemoryDiagnosticSink
// ---------------------------------------------------------------------------

/// An in-memory diagnostic sink that stores events for querying and filtering.
#[derive(Debug, Default)]
pub struct InMemoryDiagnosticSink {
    events: Vec<DiagnosticEvent>,
    capacity: Option<usize>,
}

impl InMemoryDiagnosticSink {
    /// Create a new unbounded sink.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            capacity: None,
        }
    }

    /// Create a sink with a maximum event capacity. When full, oldest events
    /// are evicted.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            events: Vec::with_capacity(capacity),
            capacity: Some(capacity),
        }
    }

    /// Return all stored events.
    pub fn events(&self) -> &[DiagnosticEvent] {
        &self.events
    }

    /// Return the number of stored events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Return whether the sink is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Filter events by kind label.
    pub fn filter_by_kind(&self, kind: &str) -> Vec<&DiagnosticEvent> {
        self.events.iter().filter(|e| e.kind() == kind).collect()
    }

    /// Filter events by agent ID.
    pub fn filter_by_agent(&self, agent_id: &str) -> Vec<&DiagnosticEvent> {
        self.events
            .iter()
            .filter(|e| e.agent_id() == Some(agent_id))
            .collect()
    }

    /// Count events matching a given kind.
    pub fn count_by_kind(&self, kind: &str) -> usize {
        self.events.iter().filter(|e| e.kind() == kind).count()
    }

    /// Return only error events.
    pub fn errors(&self) -> Vec<&DiagnosticEvent> {
        self.filter_by_kind("error_occurred")
    }

    /// Clear all stored events.
    pub fn clear(&mut self) {
        self.events.clear();
    }
}

impl DiagnosticSink for InMemoryDiagnosticSink {
    fn receive(&mut self, event: DiagnosticEvent) {
        if let Some(cap) = self.capacity {
            if self.events.len() >= cap && cap > 0 {
                self.events.remove(0);
            }
        }
        self.events.push(event);
    }
}

// ---------------------------------------------------------------------------
// ProfileCategory
// ---------------------------------------------------------------------------

/// Category of work tracked by the profiler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProfileCategory {
    /// Time spent in LLM calls.
    Llm,
    /// Time spent in tool execution.
    Tool,
    /// Time spent in middleware.
    Middleware,
    /// Time spent in other/overhead.
    Other,
}

impl fmt::Display for ProfileCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Llm => write!(f, "llm"),
            Self::Tool => write!(f, "tool"),
            Self::Middleware => write!(f, "middleware"),
            Self::Other => write!(f, "other"),
        }
    }
}

// ---------------------------------------------------------------------------
// ProfileEntry
// ---------------------------------------------------------------------------

/// A single profiling measurement.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct ProfileEntry {
    category: ProfileCategory,
    duration: Duration,
    label: String,
}

// ---------------------------------------------------------------------------
// AgentProfiler
// ---------------------------------------------------------------------------

/// Tracks per-agent timing breakdown across categories (LLM, tools, middleware).
#[derive(Debug)]
pub struct AgentProfiler {
    agent_id: String,
    entries: Vec<ProfileEntry>,
    wall_start: Option<Instant>,
    wall_end: Option<Instant>,
    active_timers: HashMap<String, (ProfileCategory, Instant)>,
}

impl AgentProfiler {
    /// Create a new profiler for the given agent.
    pub fn new(agent_id: impl Into<String>) -> Self {
        Self {
            agent_id: agent_id.into(),
            entries: Vec::new(),
            wall_start: None,
            wall_end: None,
            active_timers: HashMap::new(),
        }
    }

    /// Return the agent ID this profiler is tracking.
    pub fn agent_id(&self) -> &str {
        &self.agent_id
    }

    /// Mark the start of the agent execution (wall clock).
    pub fn start(&mut self) {
        self.wall_start = Some(Instant::now());
    }

    /// Mark the end of the agent execution (wall clock).
    pub fn stop(&mut self) {
        self.wall_end = Some(Instant::now());
    }

    /// Record a completed duration for a category.
    pub fn record(&mut self, category: ProfileCategory, label: impl Into<String>, duration: Duration) {
        self.entries.push(ProfileEntry {
            category,
            duration,
            label: label.into(),
        });
    }

    /// Start a named timer. Returns the timer key.
    pub fn start_timer(&mut self, category: ProfileCategory, label: impl Into<String>) -> String {
        let label = label.into();
        let key = format!("{}:{}", category, label);
        self.active_timers.insert(key.clone(), (category, Instant::now()));
        key
    }

    /// Stop a named timer and record the elapsed duration.
    pub fn stop_timer(&mut self, key: &str) -> Option<Duration> {
        if let Some((category, start)) = self.active_timers.remove(key) {
            let duration = start.elapsed();
            let label = key.splitn(2, ':').nth(1).unwrap_or(key).to_string();
            self.entries.push(ProfileEntry {
                category,
                duration,
                label,
            });
            Some(duration)
        } else {
            None
        }
    }

    /// Return the total wall-clock duration (between start and stop).
    pub fn wall_duration(&self) -> Option<Duration> {
        match (self.wall_start, self.wall_end) {
            (Some(start), Some(end)) => Some(end.duration_since(start)),
            (Some(start), None) => Some(start.elapsed()),
            _ => None,
        }
    }

    /// Total time spent in a given category.
    pub fn total_for_category(&self, category: ProfileCategory) -> Duration {
        self.entries
            .iter()
            .filter(|e| e.category == category)
            .map(|e| e.duration)
            .sum()
    }

    /// Number of recorded entries for a given category.
    pub fn count_for_category(&self, category: ProfileCategory) -> usize {
        self.entries
            .iter()
            .filter(|e| e.category == category)
            .count()
    }

    /// Generate a profile report.
    pub fn report(&self) -> ProfileReport {
        let wall = self.wall_duration().unwrap_or(Duration::ZERO);
        let wall_secs = wall.as_secs_f64();

        let mut categories = HashMap::new();
        for cat in &[
            ProfileCategory::Llm,
            ProfileCategory::Tool,
            ProfileCategory::Middleware,
            ProfileCategory::Other,
        ] {
            let total = self.total_for_category(*cat);
            let count = self.count_for_category(*cat);
            let pct = if wall_secs > 0.0 {
                (total.as_secs_f64() / wall_secs) * 100.0
            } else {
                0.0
            };
            categories.insert(
                *cat,
                CategoryBreakdown {
                    total_duration: total,
                    call_count: count,
                    percentage: pct,
                },
            );
        }

        ProfileReport {
            agent_id: self.agent_id.clone(),
            wall_duration: wall,
            categories,
            entry_count: self.entries.len(),
        }
    }
}

// ---------------------------------------------------------------------------
// CategoryBreakdown
// ---------------------------------------------------------------------------

/// Breakdown of time spent in a single category.
#[derive(Debug, Clone)]
pub struct CategoryBreakdown {
    /// Total time in this category.
    pub total_duration: Duration,
    /// Number of calls / entries.
    pub call_count: usize,
    /// Percentage of wall-clock time.
    pub percentage: f64,
}

// ---------------------------------------------------------------------------
// ProfileReport
// ---------------------------------------------------------------------------

/// Summary report produced by [`AgentProfiler::report`].
#[derive(Debug)]
pub struct ProfileReport {
    /// The agent this report is for.
    pub agent_id: String,
    /// Total wall-clock duration.
    pub wall_duration: Duration,
    /// Per-category breakdown.
    pub categories: HashMap<ProfileCategory, CategoryBreakdown>,
    /// Total number of profiling entries.
    pub entry_count: usize,
}

impl ProfileReport {
    /// Get the breakdown for a specific category.
    pub fn get(&self, category: ProfileCategory) -> Option<&CategoryBreakdown> {
        self.categories.get(&category)
    }

    /// Format the report as a human-readable string.
    pub fn to_summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Profile Report for agent '{}'\n",
            self.agent_id
        ));
        out.push_str(&format!(
            "Wall duration: {:.3}ms\n",
            self.wall_duration.as_secs_f64() * 1000.0
        ));
        out.push_str(&format!("Total entries: {}\n", self.entry_count));

        for cat in &[
            ProfileCategory::Llm,
            ProfileCategory::Tool,
            ProfileCategory::Middleware,
            ProfileCategory::Other,
        ] {
            if let Some(bd) = self.categories.get(cat) {
                out.push_str(&format!(
                    "  {}: {:.3}ms ({:.1}%), {} calls\n",
                    cat,
                    bd.total_duration.as_secs_f64() * 1000.0,
                    bd.percentage,
                    bd.call_count,
                ));
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Alert
// ---------------------------------------------------------------------------

/// A fired alert.
#[derive(Debug, Clone)]
pub struct Alert {
    /// The rule that triggered this alert.
    pub rule_name: String,
    /// Alert severity.
    pub severity: Severity,
    /// Human-readable message.
    pub message: String,
    /// When the alert was fired.
    pub timestamp: SystemTime,
}

impl Alert {
    /// Create a new alert.
    pub fn new(
        rule_name: impl Into<String>,
        severity: Severity,
        message: impl Into<String>,
    ) -> Self {
        Self {
            rule_name: rule_name.into(),
            severity,
            message: message.into(),
            timestamp: SystemTime::now(),
        }
    }
}

impl fmt::Display for Alert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}] {}: {}",
            self.severity, self.rule_name, self.message
        )
    }
}

// ---------------------------------------------------------------------------
// AlertCondition
// ---------------------------------------------------------------------------

/// The condition that triggers an alert rule.
#[derive(Debug, Clone)]
pub enum AlertCondition {
    /// The metric exceeds a threshold (greater than).
    GreaterThan(f64),
    /// The metric falls below a threshold (less than).
    LessThan(f64),
    /// The metric equals a value.
    Equals(f64),
}

impl AlertCondition {
    /// Evaluate this condition against an actual value.
    pub fn evaluate(&self, actual: f64) -> bool {
        match self {
            Self::GreaterThan(threshold) => actual > *threshold,
            Self::LessThan(threshold) => actual < *threshold,
            Self::Equals(expected) => (actual - expected).abs() < f64::EPSILON,
        }
    }

    /// Return the threshold value.
    pub fn threshold(&self) -> f64 {
        match self {
            Self::GreaterThan(v) | Self::LessThan(v) | Self::Equals(v) => *v,
        }
    }
}

// ---------------------------------------------------------------------------
// AlertRule
// ---------------------------------------------------------------------------

/// A rule defining when an alert should fire.
#[derive(Debug, Clone)]
pub struct AlertRule {
    /// Unique name for this rule.
    pub name: String,
    /// The metric name to evaluate.
    pub metric_name: String,
    /// The condition to check.
    pub condition: AlertCondition,
    /// The severity of alerts produced by this rule.
    pub severity: Severity,
    /// A template for the alert message. `{value}` and `{threshold}` are
    /// replaced with actual values.
    pub message_template: String,
    /// Whether this rule is enabled.
    pub enabled: bool,
}

impl AlertRule {
    /// Create a new enabled alert rule.
    pub fn new(
        name: impl Into<String>,
        metric_name: impl Into<String>,
        condition: AlertCondition,
        severity: Severity,
        message_template: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            metric_name: metric_name.into(),
            condition,
            severity,
            message_template: message_template.into(),
            enabled: true,
        }
    }

    /// Disable this rule.
    pub fn disable(&mut self) {
        self.enabled = false;
    }

    /// Enable this rule.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Evaluate this rule against a metric value and return an alert if triggered.
    pub fn evaluate(&self, value: f64) -> Option<Alert> {
        if !self.enabled {
            return None;
        }
        if self.condition.evaluate(value) {
            let message = self
                .message_template
                .replace("{value}", &format!("{:.4}", value))
                .replace("{threshold}", &format!("{:.4}", self.condition.threshold()));
            Some(Alert::new(&self.name, self.severity, message))
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// AlertManager
// ---------------------------------------------------------------------------

/// Manages a set of alert rules and evaluates them against incoming metrics.
#[derive(Debug, Default)]
pub struct AlertManager {
    rules: Vec<AlertRule>,
    fired_alerts: Vec<Alert>,
}

impl AlertManager {
    /// Create a new empty alert manager.
    pub fn new() -> Self {
        Self {
            rules: Vec::new(),
            fired_alerts: Vec::new(),
        }
    }

    /// Add an alert rule.
    pub fn add_rule(&mut self, rule: AlertRule) {
        self.rules.push(rule);
    }

    /// Remove a rule by name. Returns whether a rule was removed.
    pub fn remove_rule(&mut self, name: &str) -> bool {
        let before = self.rules.len();
        self.rules.retain(|r| r.name != name);
        self.rules.len() < before
    }

    /// Return all configured rules.
    pub fn rules(&self) -> &[AlertRule] {
        &self.rules
    }

    /// Evaluate all rules against a set of named metric values.
    /// Returns newly fired alerts.
    pub fn evaluate(&mut self, metrics: &HashMap<String, f64>) -> Vec<Alert> {
        let mut new_alerts = Vec::new();
        for rule in &self.rules {
            if let Some(value) = metrics.get(&rule.metric_name) {
                if let Some(alert) = rule.evaluate(*value) {
                    new_alerts.push(alert);
                }
            }
        }
        self.fired_alerts.extend(new_alerts.clone());
        new_alerts
    }

    /// Evaluate a single named metric against all matching rules.
    pub fn evaluate_metric(&mut self, metric_name: &str, value: f64) -> Vec<Alert> {
        let mut metrics = HashMap::new();
        metrics.insert(metric_name.to_string(), value);
        self.evaluate(&metrics)
    }

    /// Return all alerts fired so far.
    pub fn fired_alerts(&self) -> &[Alert] {
        &self.fired_alerts
    }

    /// Return alerts filtered by severity.
    pub fn alerts_by_severity(&self, severity: Severity) -> Vec<&Alert> {
        self.fired_alerts
            .iter()
            .filter(|a| a.severity == severity)
            .collect()
    }

    /// Clear all fired alerts.
    pub fn clear_alerts(&mut self) {
        self.fired_alerts.clear();
    }

    /// Return the number of fired alerts.
    pub fn alert_count(&self) -> usize {
        self.fired_alerts.len()
    }
}

// ---------------------------------------------------------------------------
// HealthStatus
// ---------------------------------------------------------------------------

/// Overall health status derived from active alerts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// No alerts active.
    Healthy,
    /// Info-level alerts only.
    Informational,
    /// Warning-level alerts present.
    Degraded,
    /// Critical-level alerts present.
    Unhealthy,
}

impl fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Healthy => write!(f, "healthy"),
            Self::Informational => write!(f, "informational"),
            Self::Degraded => write!(f, "degraded"),
            Self::Unhealthy => write!(f, "unhealthy"),
        }
    }
}

// ---------------------------------------------------------------------------
// DiagnosticDashboard
// ---------------------------------------------------------------------------

/// Aggregates diagnostic sinks, profilers, and alerts into a unified view.
#[derive(Debug)]
pub struct DiagnosticDashboard {
    sink: InMemoryDiagnosticSink,
    profilers: HashMap<String, AgentProfiler>,
    alert_manager: AlertManager,
}

impl DiagnosticDashboard {
    /// Create a new empty dashboard.
    pub fn new() -> Self {
        Self {
            sink: InMemoryDiagnosticSink::new(),
            profilers: HashMap::new(),
            alert_manager: AlertManager::new(),
        }
    }

    /// Return a reference to the diagnostic sink.
    pub fn sink(&self) -> &InMemoryDiagnosticSink {
        &self.sink
    }

    /// Return a mutable reference to the diagnostic sink.
    pub fn sink_mut(&mut self) -> &mut InMemoryDiagnosticSink {
        &mut self.sink
    }

    /// Send a diagnostic event to the sink.
    pub fn emit(&mut self, event: DiagnosticEvent) {
        self.sink.receive(event);
    }

    /// Get or create a profiler for the given agent.
    pub fn profiler(&mut self, agent_id: &str) -> &mut AgentProfiler {
        self.profilers
            .entry(agent_id.to_string())
            .or_insert_with(|| AgentProfiler::new(agent_id))
    }

    /// Return a reference to a profiler if it exists.
    pub fn get_profiler(&self, agent_id: &str) -> Option<&AgentProfiler> {
        self.profilers.get(agent_id)
    }

    /// Return a mutable reference to the alert manager.
    pub fn alert_manager_mut(&mut self) -> &mut AlertManager {
        &mut self.alert_manager
    }

    /// Return a reference to the alert manager.
    pub fn alert_manager(&self) -> &AlertManager {
        &self.alert_manager
    }

    /// Add an alert rule to the manager.
    pub fn add_alert_rule(&mut self, rule: AlertRule) {
        self.alert_manager.add_rule(rule);
    }

    /// Evaluate metrics against alert rules and return new alerts.
    pub fn evaluate_alerts(&mut self, metrics: &HashMap<String, f64>) -> Vec<Alert> {
        self.alert_manager.evaluate(metrics)
    }

    /// Derive the overall health status from currently fired alerts.
    pub fn health_status(&self) -> HealthStatus {
        let alerts = self.alert_manager.fired_alerts();
        if alerts.is_empty() {
            return HealthStatus::Healthy;
        }
        let max_severity = alerts.iter().map(|a| a.severity).max().unwrap();
        match max_severity {
            Severity::Info => HealthStatus::Informational,
            Severity::Warning => HealthStatus::Degraded,
            Severity::Critical => HealthStatus::Unhealthy,
        }
    }

    /// Generate profile reports for all tracked agents.
    pub fn all_reports(&self) -> Vec<ProfileReport> {
        self.profilers.values().map(|p| p.report()).collect()
    }

    /// Generate a text summary of the dashboard state.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str("=== Diagnostic Dashboard ===\n\n");
        out.push_str(&format!("Health: {}\n", self.health_status()));
        out.push_str(&format!("Events: {}\n", self.sink.len()));
        out.push_str(&format!("Agents profiled: {}\n", self.profilers.len()));
        out.push_str(&format!(
            "Alerts fired: {}\n",
            self.alert_manager.alert_count()
        ));
        out.push_str(&format!(
            "Alert rules: {}\n",
            self.alert_manager.rules().len()
        ));

        if !self.alert_manager.fired_alerts().is_empty() {
            out.push_str("\nActive Alerts:\n");
            for alert in self.alert_manager.fired_alerts() {
                out.push_str(&format!("  {}\n", alert));
            }
        }

        for report in self.all_reports() {
            out.push_str(&format!("\n{}", report.to_summary()));
        }

        out
    }

    /// Return total event count.
    pub fn event_count(&self) -> usize {
        self.sink.len()
    }

    /// Return error event count.
    pub fn error_count(&self) -> usize {
        self.sink.count_by_kind("error_occurred")
    }
}

impl Default for DiagnosticDashboard {
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

    // -- Severity --

    #[test]
    fn test_severity_display() {
        assert_eq!(Severity::Info.to_string(), "info");
        assert_eq!(Severity::Warning.to_string(), "warning");
        assert_eq!(Severity::Critical.to_string(), "critical");
    }

    #[test]
    fn test_severity_ordering() {
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Critical);
        assert!(Severity::Info < Severity::Critical);
    }

    #[test]
    fn test_severity_equality() {
        assert_eq!(Severity::Info, Severity::Info);
        assert_ne!(Severity::Info, Severity::Warning);
    }

    // -- DiagnosticEvent --

    #[test]
    fn test_event_agent_started() {
        let event = DiagnosticEvent::AgentStarted {
            agent_id: "a1".into(),
            timestamp: SystemTime::now(),
        };
        assert_eq!(event.agent_id(), Some("a1"));
        assert_eq!(event.kind(), "agent_started");
    }

    #[test]
    fn test_event_agent_completed() {
        let event = DiagnosticEvent::AgentCompleted {
            agent_id: "a2".into(),
            duration: Duration::from_secs(5),
            success: true,
            timestamp: SystemTime::now(),
        };
        assert_eq!(event.agent_id(), Some("a2"));
        assert_eq!(event.kind(), "agent_completed");
    }

    #[test]
    fn test_event_tool_called() {
        let event = DiagnosticEvent::ToolCalled {
            agent_id: "a1".into(),
            tool_name: "search".into(),
            duration: Duration::from_millis(100),
            success: true,
            timestamp: SystemTime::now(),
        };
        assert_eq!(event.kind(), "tool_called");
        assert_eq!(event.agent_id(), Some("a1"));
    }

    #[test]
    fn test_event_error_occurred() {
        let event = DiagnosticEvent::ErrorOccurred {
            agent_id: "a1".into(),
            message: "timeout".into(),
            severity: Severity::Critical,
            timestamp: SystemTime::now(),
        };
        assert_eq!(event.kind(), "error_occurred");
    }

    #[test]
    fn test_event_threshold_exceeded_no_agent() {
        let event = DiagnosticEvent::ThresholdExceeded {
            metric_name: "latency".into(),
            threshold: 5.0,
            actual: 7.0,
            timestamp: SystemTime::now(),
        };
        assert_eq!(event.agent_id(), None);
        assert_eq!(event.kind(), "threshold_exceeded");
    }

    #[test]
    fn test_event_memory_pressure() {
        let event = DiagnosticEvent::MemoryPressure {
            agent_id: "a1".into(),
            usage_bytes: 900,
            limit_bytes: 1000,
            timestamp: SystemTime::now(),
        };
        assert_eq!(event.kind(), "memory_pressure");
        assert_eq!(event.agent_id(), Some("a1"));
    }

    #[test]
    fn test_event_middleware_executed() {
        let event = DiagnosticEvent::MiddlewareExecuted {
            agent_id: "a1".into(),
            middleware_name: "rate_limiter".into(),
            duration: Duration::from_millis(2),
            timestamp: SystemTime::now(),
        };
        assert_eq!(event.kind(), "middleware_executed");
    }

    #[test]
    fn test_event_llm_call_completed() {
        let event = DiagnosticEvent::LlmCallCompleted {
            agent_id: "a1".into(),
            model: "gpt-4".into(),
            duration: Duration::from_millis(500),
            timestamp: SystemTime::now(),
        };
        assert_eq!(event.kind(), "llm_call_completed");
    }

    #[test]
    fn test_event_timestamp() {
        let now = SystemTime::now();
        let event = DiagnosticEvent::AgentStarted {
            agent_id: "a1".into(),
            timestamp: now,
        };
        assert_eq!(event.timestamp(), now);
    }

    // -- InMemoryDiagnosticSink --

    #[test]
    fn test_sink_new_empty() {
        let sink = InMemoryDiagnosticSink::new();
        assert!(sink.is_empty());
        assert_eq!(sink.len(), 0);
    }

    #[test]
    fn test_sink_receive_event() {
        let mut sink = InMemoryDiagnosticSink::new();
        sink.receive(DiagnosticEvent::AgentStarted {
            agent_id: "a1".into(),
            timestamp: SystemTime::now(),
        });
        assert_eq!(sink.len(), 1);
        assert!(!sink.is_empty());
    }

    #[test]
    fn test_sink_capacity_eviction() {
        let mut sink = InMemoryDiagnosticSink::with_capacity(2);
        for i in 0..3 {
            sink.receive(DiagnosticEvent::AgentStarted {
                agent_id: format!("a{}", i),
                timestamp: SystemTime::now(),
            });
        }
        assert_eq!(sink.len(), 2);
        // First event (a0) should have been evicted
        assert_eq!(sink.events()[0].agent_id(), Some("a1"));
        assert_eq!(sink.events()[1].agent_id(), Some("a2"));
    }

    #[test]
    fn test_sink_filter_by_kind() {
        let mut sink = InMemoryDiagnosticSink::new();
        sink.receive(DiagnosticEvent::AgentStarted {
            agent_id: "a1".into(),
            timestamp: SystemTime::now(),
        });
        sink.receive(DiagnosticEvent::ErrorOccurred {
            agent_id: "a1".into(),
            message: "fail".into(),
            severity: Severity::Warning,
            timestamp: SystemTime::now(),
        });
        sink.receive(DiagnosticEvent::AgentStarted {
            agent_id: "a2".into(),
            timestamp: SystemTime::now(),
        });

        let started = sink.filter_by_kind("agent_started");
        assert_eq!(started.len(), 2);
        let errors = sink.filter_by_kind("error_occurred");
        assert_eq!(errors.len(), 1);
    }

    #[test]
    fn test_sink_filter_by_agent() {
        let mut sink = InMemoryDiagnosticSink::new();
        sink.receive(DiagnosticEvent::AgentStarted {
            agent_id: "a1".into(),
            timestamp: SystemTime::now(),
        });
        sink.receive(DiagnosticEvent::AgentStarted {
            agent_id: "a2".into(),
            timestamp: SystemTime::now(),
        });
        sink.receive(DiagnosticEvent::ToolCalled {
            agent_id: "a1".into(),
            tool_name: "search".into(),
            duration: Duration::from_millis(10),
            success: true,
            timestamp: SystemTime::now(),
        });

        let a1_events = sink.filter_by_agent("a1");
        assert_eq!(a1_events.len(), 2);
        let a2_events = sink.filter_by_agent("a2");
        assert_eq!(a2_events.len(), 1);
    }

    #[test]
    fn test_sink_count_by_kind() {
        let mut sink = InMemoryDiagnosticSink::new();
        for _ in 0..5 {
            sink.receive(DiagnosticEvent::AgentStarted {
                agent_id: "a1".into(),
                timestamp: SystemTime::now(),
            });
        }
        assert_eq!(sink.count_by_kind("agent_started"), 5);
        assert_eq!(sink.count_by_kind("error_occurred"), 0);
    }

    #[test]
    fn test_sink_errors() {
        let mut sink = InMemoryDiagnosticSink::new();
        sink.receive(DiagnosticEvent::ErrorOccurred {
            agent_id: "a1".into(),
            message: "bad".into(),
            severity: Severity::Critical,
            timestamp: SystemTime::now(),
        });
        sink.receive(DiagnosticEvent::AgentStarted {
            agent_id: "a1".into(),
            timestamp: SystemTime::now(),
        });
        assert_eq!(sink.errors().len(), 1);
    }

    #[test]
    fn test_sink_clear() {
        let mut sink = InMemoryDiagnosticSink::new();
        sink.receive(DiagnosticEvent::AgentStarted {
            agent_id: "a1".into(),
            timestamp: SystemTime::now(),
        });
        assert_eq!(sink.len(), 1);
        sink.clear();
        assert!(sink.is_empty());
    }

    #[test]
    fn test_sink_flush_is_noop() {
        let mut sink = InMemoryDiagnosticSink::new();
        sink.flush(); // should not panic
    }

    // -- ProfileCategory --

    #[test]
    fn test_profile_category_display() {
        assert_eq!(ProfileCategory::Llm.to_string(), "llm");
        assert_eq!(ProfileCategory::Tool.to_string(), "tool");
        assert_eq!(ProfileCategory::Middleware.to_string(), "middleware");
        assert_eq!(ProfileCategory::Other.to_string(), "other");
    }

    // -- AgentProfiler --

    #[test]
    fn test_profiler_new() {
        let p = AgentProfiler::new("agent-1");
        assert_eq!(p.agent_id(), "agent-1");
        assert!(p.wall_duration().is_none());
    }

    #[test]
    fn test_profiler_start_stop() {
        let mut p = AgentProfiler::new("a1");
        p.start();
        std::thread::sleep(Duration::from_millis(5));
        p.stop();
        let dur = p.wall_duration().unwrap();
        assert!(dur >= Duration::from_millis(5));
    }

    #[test]
    fn test_profiler_record() {
        let mut p = AgentProfiler::new("a1");
        p.record(ProfileCategory::Llm, "call-1", Duration::from_millis(100));
        p.record(ProfileCategory::Llm, "call-2", Duration::from_millis(200));
        p.record(ProfileCategory::Tool, "search", Duration::from_millis(50));

        assert_eq!(p.total_for_category(ProfileCategory::Llm), Duration::from_millis(300));
        assert_eq!(p.count_for_category(ProfileCategory::Llm), 2);
        assert_eq!(p.count_for_category(ProfileCategory::Tool), 1);
        assert_eq!(p.count_for_category(ProfileCategory::Middleware), 0);
    }

    #[test]
    fn test_profiler_timer() {
        let mut p = AgentProfiler::new("a1");
        let key = p.start_timer(ProfileCategory::Tool, "search");
        std::thread::sleep(Duration::from_millis(5));
        let dur = p.stop_timer(&key);
        assert!(dur.is_some());
        assert!(dur.unwrap() >= Duration::from_millis(5));
        assert_eq!(p.count_for_category(ProfileCategory::Tool), 1);
    }

    #[test]
    fn test_profiler_stop_timer_unknown_key() {
        let mut p = AgentProfiler::new("a1");
        assert!(p.stop_timer("nonexistent").is_none());
    }

    #[test]
    fn test_profiler_wall_duration_while_running() {
        let mut p = AgentProfiler::new("a1");
        p.start();
        let dur = p.wall_duration();
        assert!(dur.is_some());
    }

    #[test]
    fn test_profiler_report() {
        let mut p = AgentProfiler::new("a1");
        p.start();
        p.record(ProfileCategory::Llm, "call", Duration::from_millis(100));
        p.record(ProfileCategory::Tool, "tool", Duration::from_millis(50));
        p.record(ProfileCategory::Middleware, "mw", Duration::from_millis(10));
        p.stop();

        let report = p.report();
        assert_eq!(report.agent_id, "a1");
        assert_eq!(report.entry_count, 3);
        assert!(report.wall_duration >= Duration::from_millis(0));

        let llm = report.get(ProfileCategory::Llm).unwrap();
        assert_eq!(llm.call_count, 1);
        assert_eq!(llm.total_duration, Duration::from_millis(100));
    }

    #[test]
    fn test_profile_report_summary_format() {
        let mut p = AgentProfiler::new("test-agent");
        p.start();
        p.record(ProfileCategory::Llm, "c", Duration::from_millis(50));
        p.stop();

        let report = p.report();
        let summary = report.to_summary();
        assert!(summary.contains("test-agent"));
        assert!(summary.contains("llm"));
    }

    #[test]
    fn test_profiler_report_zero_wall() {
        let p = AgentProfiler::new("a1");
        let report = p.report();
        // All percentages should be 0 when wall is zero
        for bd in report.categories.values() {
            assert_eq!(bd.percentage, 0.0);
        }
    }

    // -- AlertCondition --

    #[test]
    fn test_condition_greater_than() {
        let cond = AlertCondition::GreaterThan(5.0);
        assert!(cond.evaluate(6.0));
        assert!(!cond.evaluate(5.0));
        assert!(!cond.evaluate(4.0));
        assert_eq!(cond.threshold(), 5.0);
    }

    #[test]
    fn test_condition_less_than() {
        let cond = AlertCondition::LessThan(3.0);
        assert!(cond.evaluate(2.0));
        assert!(!cond.evaluate(3.0));
        assert!(!cond.evaluate(4.0));
    }

    #[test]
    fn test_condition_equals() {
        let cond = AlertCondition::Equals(10.0);
        assert!(cond.evaluate(10.0));
        assert!(!cond.evaluate(10.1));
    }

    // -- AlertRule --

    #[test]
    fn test_alert_rule_fires() {
        let rule = AlertRule::new(
            "high_latency",
            "latency_ms",
            AlertCondition::GreaterThan(5000.0),
            Severity::Warning,
            "Latency {value}ms exceeds {threshold}ms",
        );
        let alert = rule.evaluate(6000.0);
        assert!(alert.is_some());
        let alert = alert.unwrap();
        assert_eq!(alert.rule_name, "high_latency");
        assert_eq!(alert.severity, Severity::Warning);
        assert!(alert.message.contains("6000"));
    }

    #[test]
    fn test_alert_rule_does_not_fire() {
        let rule = AlertRule::new(
            "high_latency",
            "latency_ms",
            AlertCondition::GreaterThan(5000.0),
            Severity::Warning,
            "too slow",
        );
        assert!(rule.evaluate(3000.0).is_none());
    }

    #[test]
    fn test_alert_rule_disabled() {
        let mut rule = AlertRule::new(
            "r1",
            "m1",
            AlertCondition::GreaterThan(0.0),
            Severity::Critical,
            "msg",
        );
        rule.disable();
        assert!(rule.evaluate(100.0).is_none());
    }

    #[test]
    fn test_alert_rule_enable_disable() {
        let mut rule = AlertRule::new(
            "r1",
            "m1",
            AlertCondition::GreaterThan(0.0),
            Severity::Info,
            "msg",
        );
        assert!(rule.enabled);
        rule.disable();
        assert!(!rule.enabled);
        rule.enable();
        assert!(rule.enabled);
    }

    // -- Alert --

    #[test]
    fn test_alert_display() {
        let alert = Alert::new("rule1", Severity::Critical, "system overloaded");
        let s = format!("{}", alert);
        assert!(s.contains("critical"));
        assert!(s.contains("rule1"));
        assert!(s.contains("system overloaded"));
    }

    #[test]
    fn test_alert_new() {
        let alert = Alert::new("test", Severity::Info, "all good");
        assert_eq!(alert.rule_name, "test");
        assert_eq!(alert.severity, Severity::Info);
        assert_eq!(alert.message, "all good");
    }

    // -- AlertManager --

    #[test]
    fn test_alert_manager_new() {
        let am = AlertManager::new();
        assert!(am.rules().is_empty());
        assert!(am.fired_alerts().is_empty());
        assert_eq!(am.alert_count(), 0);
    }

    #[test]
    fn test_alert_manager_add_rule() {
        let mut am = AlertManager::new();
        am.add_rule(AlertRule::new(
            "r1",
            "m1",
            AlertCondition::GreaterThan(10.0),
            Severity::Warning,
            "exceeded",
        ));
        assert_eq!(am.rules().len(), 1);
    }

    #[test]
    fn test_alert_manager_remove_rule() {
        let mut am = AlertManager::new();
        am.add_rule(AlertRule::new(
            "r1",
            "m1",
            AlertCondition::GreaterThan(10.0),
            Severity::Warning,
            "exceeded",
        ));
        assert!(am.remove_rule("r1"));
        assert!(am.rules().is_empty());
        assert!(!am.remove_rule("nonexistent"));
    }

    #[test]
    fn test_alert_manager_evaluate() {
        let mut am = AlertManager::new();
        am.add_rule(AlertRule::new(
            "high_err",
            "error_rate",
            AlertCondition::GreaterThan(0.1),
            Severity::Critical,
            "Error rate {value} > {threshold}",
        ));
        am.add_rule(AlertRule::new(
            "low_throughput",
            "throughput",
            AlertCondition::LessThan(100.0),
            Severity::Warning,
            "Low throughput",
        ));

        let mut metrics = HashMap::new();
        metrics.insert("error_rate".to_string(), 0.15);
        metrics.insert("throughput".to_string(), 200.0);

        let alerts = am.evaluate(&metrics);
        // Only error_rate should fire
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].rule_name, "high_err");
        assert_eq!(am.alert_count(), 1);
    }

    #[test]
    fn test_alert_manager_evaluate_metric() {
        let mut am = AlertManager::new();
        am.add_rule(AlertRule::new(
            "r1",
            "latency",
            AlertCondition::GreaterThan(1000.0),
            Severity::Warning,
            "slow",
        ));
        let alerts = am.evaluate_metric("latency", 2000.0);
        assert_eq!(alerts.len(), 1);
    }

    #[test]
    fn test_alert_manager_no_match() {
        let mut am = AlertManager::new();
        am.add_rule(AlertRule::new(
            "r1",
            "latency",
            AlertCondition::GreaterThan(1000.0),
            Severity::Warning,
            "slow",
        ));
        let alerts = am.evaluate_metric("latency", 500.0);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_alert_manager_alerts_by_severity() {
        let mut am = AlertManager::new();
        am.add_rule(AlertRule::new(
            "r1",
            "m1",
            AlertCondition::GreaterThan(0.0),
            Severity::Warning,
            "w",
        ));
        am.add_rule(AlertRule::new(
            "r2",
            "m2",
            AlertCondition::GreaterThan(0.0),
            Severity::Critical,
            "c",
        ));
        let mut metrics = HashMap::new();
        metrics.insert("m1".to_string(), 1.0);
        metrics.insert("m2".to_string(), 1.0);
        am.evaluate(&metrics);

        assert_eq!(am.alerts_by_severity(Severity::Warning).len(), 1);
        assert_eq!(am.alerts_by_severity(Severity::Critical).len(), 1);
        assert_eq!(am.alerts_by_severity(Severity::Info).len(), 0);
    }

    #[test]
    fn test_alert_manager_clear() {
        let mut am = AlertManager::new();
        am.add_rule(AlertRule::new(
            "r1",
            "m1",
            AlertCondition::GreaterThan(0.0),
            Severity::Info,
            "msg",
        ));
        am.evaluate_metric("m1", 1.0);
        assert_eq!(am.alert_count(), 1);
        am.clear_alerts();
        assert_eq!(am.alert_count(), 0);
    }

    // -- HealthStatus --

    #[test]
    fn test_health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "healthy");
        assert_eq!(HealthStatus::Informational.to_string(), "informational");
        assert_eq!(HealthStatus::Degraded.to_string(), "degraded");
        assert_eq!(HealthStatus::Unhealthy.to_string(), "unhealthy");
    }

    #[test]
    fn test_health_status_equality() {
        assert_eq!(HealthStatus::Healthy, HealthStatus::Healthy);
        assert_ne!(HealthStatus::Healthy, HealthStatus::Unhealthy);
    }

    // -- DiagnosticDashboard --

    #[test]
    fn test_dashboard_new() {
        let d = DiagnosticDashboard::new();
        assert_eq!(d.event_count(), 0);
        assert_eq!(d.error_count(), 0);
        assert_eq!(d.health_status(), HealthStatus::Healthy);
    }

    #[test]
    fn test_dashboard_default() {
        let d = DiagnosticDashboard::default();
        assert_eq!(d.event_count(), 0);
    }

    #[test]
    fn test_dashboard_emit_events() {
        let mut d = DiagnosticDashboard::new();
        d.emit(DiagnosticEvent::AgentStarted {
            agent_id: "a1".into(),
            timestamp: SystemTime::now(),
        });
        d.emit(DiagnosticEvent::ErrorOccurred {
            agent_id: "a1".into(),
            message: "oops".into(),
            severity: Severity::Warning,
            timestamp: SystemTime::now(),
        });
        assert_eq!(d.event_count(), 2);
        assert_eq!(d.error_count(), 1);
    }

    #[test]
    fn test_dashboard_profiler() {
        let mut d = DiagnosticDashboard::new();
        {
            let p = d.profiler("a1");
            p.start();
            p.record(ProfileCategory::Llm, "call", Duration::from_millis(100));
            p.stop();
        }
        assert!(d.get_profiler("a1").is_some());
        assert!(d.get_profiler("a2").is_none());
    }

    #[test]
    fn test_dashboard_all_reports() {
        let mut d = DiagnosticDashboard::new();
        {
            let p = d.profiler("a1");
            p.start();
            p.stop();
        }
        {
            let p = d.profiler("a2");
            p.start();
            p.stop();
        }
        let reports = d.all_reports();
        assert_eq!(reports.len(), 2);
    }

    #[test]
    fn test_dashboard_health_healthy() {
        let d = DiagnosticDashboard::new();
        assert_eq!(d.health_status(), HealthStatus::Healthy);
    }

    #[test]
    fn test_dashboard_health_info() {
        let mut d = DiagnosticDashboard::new();
        d.add_alert_rule(AlertRule::new(
            "r1",
            "m1",
            AlertCondition::GreaterThan(0.0),
            Severity::Info,
            "info",
        ));
        d.evaluate_alerts(&{
            let mut m = HashMap::new();
            m.insert("m1".to_string(), 1.0);
            m
        });
        assert_eq!(d.health_status(), HealthStatus::Informational);
    }

    #[test]
    fn test_dashboard_health_degraded() {
        let mut d = DiagnosticDashboard::new();
        d.add_alert_rule(AlertRule::new(
            "r1",
            "m1",
            AlertCondition::GreaterThan(0.0),
            Severity::Warning,
            "warn",
        ));
        d.evaluate_alerts(&{
            let mut m = HashMap::new();
            m.insert("m1".to_string(), 1.0);
            m
        });
        assert_eq!(d.health_status(), HealthStatus::Degraded);
    }

    #[test]
    fn test_dashboard_health_unhealthy() {
        let mut d = DiagnosticDashboard::new();
        d.add_alert_rule(AlertRule::new(
            "r1",
            "m1",
            AlertCondition::GreaterThan(0.0),
            Severity::Critical,
            "crit",
        ));
        d.evaluate_alerts(&{
            let mut m = HashMap::new();
            m.insert("m1".to_string(), 1.0);
            m
        });
        assert_eq!(d.health_status(), HealthStatus::Unhealthy);
    }

    #[test]
    fn test_dashboard_summary() {
        let mut d = DiagnosticDashboard::new();
        d.emit(DiagnosticEvent::AgentStarted {
            agent_id: "a1".into(),
            timestamp: SystemTime::now(),
        });
        let s = d.summary();
        assert!(s.contains("Diagnostic Dashboard"));
        assert!(s.contains("healthy"));
        assert!(s.contains("Events: 1"));
    }

    #[test]
    fn test_dashboard_summary_with_alerts() {
        let mut d = DiagnosticDashboard::new();
        d.add_alert_rule(AlertRule::new(
            "test_rule",
            "metric",
            AlertCondition::GreaterThan(0.0),
            Severity::Warning,
            "alert fired",
        ));
        d.evaluate_alerts(&{
            let mut m = HashMap::new();
            m.insert("metric".to_string(), 1.0);
            m
        });
        let s = d.summary();
        assert!(s.contains("Active Alerts"));
        assert!(s.contains("test_rule"));
    }

    #[test]
    fn test_dashboard_sink_access() {
        let mut d = DiagnosticDashboard::new();
        d.sink_mut().receive(DiagnosticEvent::AgentStarted {
            agent_id: "a1".into(),
            timestamp: SystemTime::now(),
        });
        assert_eq!(d.sink().len(), 1);
    }

    #[test]
    fn test_dashboard_alert_manager_access() {
        let mut d = DiagnosticDashboard::new();
        d.alert_manager_mut().add_rule(AlertRule::new(
            "r1",
            "m1",
            AlertCondition::GreaterThan(0.0),
            Severity::Info,
            "msg",
        ));
        assert_eq!(d.alert_manager().rules().len(), 1);
    }
}
