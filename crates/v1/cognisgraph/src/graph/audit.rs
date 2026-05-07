//! Graph execution audit log.
//!
//! Provides [`AuditLog`] for recording a detailed trail of every action during
//! graph execution — useful for debugging, compliance, and observability.
//! [`AuditTrail`] adds higher-level queries that correlate related events, and
//! [`AuditReport`] produces aggregated summaries.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// AuditSeverity
// ---------------------------------------------------------------------------

/// Severity level for an audit event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum AuditSeverity {
    /// Informational event.
    Info,
    /// Warning — something unexpected but not fatal.
    Warning,
    /// Error — an operation failed.
    Error,
    /// Critical — a severe failure.
    Critical,
}

impl std::fmt::Display for AuditSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditSeverity::Info => write!(f, "INFO"),
            AuditSeverity::Warning => write!(f, "WARNING"),
            AuditSeverity::Error => write!(f, "ERROR"),
            AuditSeverity::Critical => write!(f, "CRITICAL"),
        }
    }
}

// ---------------------------------------------------------------------------
// AuditEventType
// ---------------------------------------------------------------------------

/// The kind of action that an [`AuditEvent`] records.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditEventType {
    /// Graph execution started.
    GraphStart,
    /// Graph execution ended.
    GraphEnd,
    /// A node began executing.
    NodeEnter,
    /// A node finished executing.
    NodeExit,
    /// An edge was traversed between two nodes.
    EdgeTraversal,
    /// A conditional branch was evaluated.
    ConditionalBranch {
        /// The condition expression that was evaluated.
        condition: String,
        /// The branch that was selected.
        result: String,
    },
    /// A state key was updated.
    StateUpdate {
        /// The key that changed.
        key: String,
    },
    /// An error occurred.
    Error {
        /// Human-readable error description.
        message: String,
    },
    /// Execution was interrupted.
    Interrupt {
        /// The reason for the interrupt.
        reason: String,
    },
    /// A checkpoint was saved.
    CheckpointSave,
    /// A checkpoint was restored.
    CheckpointRestore,
    /// A user-defined event type.
    Custom {
        /// Name of the custom event.
        name: String,
    },
}

impl AuditEventType {
    /// Return a short string label for this event type (used in reports and
    /// filtering).
    pub fn label(&self) -> String {
        match self {
            AuditEventType::GraphStart => "GraphStart".to_string(),
            AuditEventType::GraphEnd => "GraphEnd".to_string(),
            AuditEventType::NodeEnter => "NodeEnter".to_string(),
            AuditEventType::NodeExit => "NodeExit".to_string(),
            AuditEventType::EdgeTraversal => "EdgeTraversal".to_string(),
            AuditEventType::ConditionalBranch { .. } => "ConditionalBranch".to_string(),
            AuditEventType::StateUpdate { .. } => "StateUpdate".to_string(),
            AuditEventType::Error { .. } => "Error".to_string(),
            AuditEventType::Interrupt { .. } => "Interrupt".to_string(),
            AuditEventType::CheckpointSave => "CheckpointSave".to_string(),
            AuditEventType::CheckpointRestore => "CheckpointRestore".to_string(),
            AuditEventType::Custom { name } => format!("Custom({name})"),
        }
    }
}

// ---------------------------------------------------------------------------
// AuditEvent
// ---------------------------------------------------------------------------

/// A single recorded event in the audit log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event identifier (UUID v4).
    pub id: String,
    /// ISO-8601 timestamp of when the event was recorded.
    pub timestamp: String,
    /// The type of action this event represents.
    pub event_type: AuditEventType,
    /// The node associated with this event (if any).
    pub node_name: Option<String>,
    /// Free-form details about the event.
    pub details: String,
    /// Graph state before this event (captured only when configured).
    pub state_before: Option<Value>,
    /// Graph state after this event (captured only when configured).
    pub state_after: Option<Value>,
    /// Duration of the action (if applicable).
    pub duration: Option<Duration>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, Value>,
    /// Severity level.
    pub severity: AuditSeverity,
}

// ---------------------------------------------------------------------------
// AuditLogConfig
// ---------------------------------------------------------------------------

/// Configuration for an [`AuditLog`].
#[derive(Debug, Clone)]
pub struct AuditLogConfig {
    /// Whether the audit log is enabled. When `false`, [`AuditLog::record`]
    /// is a no-op.
    pub enabled: bool,
    /// Maximum number of events to keep. When set, the oldest events are
    /// dropped to stay within the limit (rolling buffer).
    pub max_events: Option<usize>,
    /// Minimum severity level. Events below this level are discarded.
    pub min_severity: AuditSeverity,
    /// Whether to capture `state_before` / `state_after` snapshots on
    /// events.
    pub capture_state: bool,
    /// Whether to capture duration on events that support it.
    pub capture_duration: bool,
}

impl Default for AuditLogConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_events: None,
            min_severity: AuditSeverity::Info,
            capture_state: false,
            capture_duration: true,
        }
    }
}

// ---------------------------------------------------------------------------
// AuditLog
// ---------------------------------------------------------------------------

/// Thread-safe audit log that records [`AuditEvent`]s during graph execution.
///
/// # Example
///
/// ```
/// use cognisgraph::graph::audit::{AuditLog, AuditLogConfig, AuditEvent, AuditEventType, AuditSeverity};
/// use std::collections::HashMap;
///
/// let log = AuditLog::new(AuditLogConfig::default());
/// log.record(AuditEvent {
///     id: uuid::Uuid::new_v4().to_string(),
///     timestamp: String::new(),
///     event_type: AuditEventType::GraphStart,
///     node_name: None,
///     details: "graph started".to_string(),
///     state_before: None,
///     state_after: None,
///     duration: None,
///     metadata: HashMap::new(),
///     severity: AuditSeverity::Info,
/// });
/// assert_eq!(log.size(), 1);
/// ```
#[derive(Debug, Clone)]
pub struct AuditLog {
    config: AuditLogConfig,
    events: Arc<Mutex<Vec<AuditEvent>>>,
}

impl AuditLog {
    /// Create a new audit log with the given configuration.
    pub fn new(config: AuditLogConfig) -> Self {
        Self {
            config,
            events: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Record an event. If the log is disabled or the event severity is below
    /// the configured minimum, the event is silently dropped. If
    /// `max_events` is set and the buffer is full, the oldest event is
    /// removed first.
    pub fn record(&self, event: AuditEvent) {
        if !self.config.enabled {
            return;
        }
        if event.severity < self.config.min_severity {
            return;
        }
        let mut events = self.events.lock().unwrap();
        if let Some(max) = self.config.max_events {
            if events.len() >= max {
                events.remove(0);
            }
        }
        events.push(event);
    }

    /// Return all recorded events.
    pub fn get_events(&self) -> Vec<AuditEvent> {
        self.events.lock().unwrap().clone()
    }

    /// Return events matching the given event type label.
    pub fn get_events_by_type(&self, event_type: &AuditEventType) -> Vec<AuditEvent> {
        let label = event_type.label();
        let events = self.events.lock().unwrap();
        events
            .iter()
            .filter(|e| e.event_type.label() == label)
            .cloned()
            .collect()
    }

    /// Return events associated with a specific node.
    pub fn get_events_for_node(&self, node_name: &str) -> Vec<AuditEvent> {
        let events = self.events.lock().unwrap();
        events
            .iter()
            .filter(|e| e.node_name.as_deref() == Some(node_name))
            .cloned()
            .collect()
    }

    /// Return events whose timestamp falls within the given range
    /// (inclusive, ISO-8601 string comparison).
    pub fn get_events_in_range(&self, start: &str, end: &str) -> Vec<AuditEvent> {
        let events = self.events.lock().unwrap();
        events
            .iter()
            .filter(|e| e.timestamp.as_str() >= start && e.timestamp.as_str() <= end)
            .cloned()
            .collect()
    }

    /// Full-text search in the `details` field of all events.
    pub fn search(&self, query: &str) -> Vec<AuditEvent> {
        let query_lower = query.to_lowercase();
        let events = self.events.lock().unwrap();
        events
            .iter()
            .filter(|e| e.details.to_lowercase().contains(&query_lower))
            .cloned()
            .collect()
    }

    /// Remove all recorded events.
    pub fn clear(&self) {
        self.events.lock().unwrap().clear();
    }

    /// Serialize all events to a JSON string.
    pub fn export_json(&self) -> String {
        let events = self.events.lock().unwrap();
        serde_json::to_string_pretty(&*events).unwrap_or_else(|_| "[]".to_string())
    }

    /// Return the number of recorded events.
    pub fn size(&self) -> usize {
        self.events.lock().unwrap().len()
    }
}

// ---------------------------------------------------------------------------
// AuditReport
// ---------------------------------------------------------------------------

/// Aggregated summary report derived from an [`AuditLog`].
#[derive(Debug, Clone)]
pub struct AuditReport {
    /// Total number of events in the log.
    pub total_events: usize,
    /// Number of events per event-type label.
    pub event_counts: HashMap<String, usize>,
    /// Number of events per node name.
    pub node_activity: HashMap<String, usize>,
    /// All events with severity >= Error.
    pub errors: Vec<AuditEvent>,
    /// Total duration from first GraphStart to last GraphEnd (if both
    /// exist and have durations).
    pub total_duration: Option<Duration>,
}

impl AuditReport {
    /// Generate a report from the given audit log.
    pub fn generate(log: &AuditLog) -> Self {
        let events = log.get_events();
        let total_events = events.len();

        let mut event_counts: HashMap<String, usize> = HashMap::new();
        let mut node_activity: HashMap<String, usize> = HashMap::new();
        let mut errors: Vec<AuditEvent> = Vec::new();
        let mut total_duration: Option<Duration> = None;

        // Attempt to derive total_duration from GraphStart → GraphEnd
        // timestamps.
        for event in &events {
            *event_counts.entry(event.event_type.label()).or_insert(0) += 1;

            if let Some(ref name) = event.node_name {
                *node_activity.entry(name.clone()).or_insert(0) += 1;
            }

            if event.severity >= AuditSeverity::Error {
                errors.push(event.clone());
            }

            if let AuditEventType::GraphEnd = &event.event_type {
                if let Some(dur) = event.duration {
                    total_duration = Some(dur);
                }
            }
        }

        // If we have both start/end events with durations, sum up all
        // recorded durations as a fallback.
        if total_duration.is_none() {
            let sum: Duration = events.iter().filter_map(|e| e.duration).sum();
            if sum > Duration::ZERO {
                total_duration = Some(sum);
            }
        }

        Self {
            total_events,
            event_counts,
            node_activity,
            errors,
            total_duration,
        }
    }

    /// Return a human-readable summary string.
    pub fn summary(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "Audit Report: {} total events\n",
            self.total_events
        ));

        if !self.event_counts.is_empty() {
            out.push_str("Event counts:\n");
            let mut counts: Vec<_> = self.event_counts.iter().collect();
            counts.sort_by_key(|(a, _)| *a);
            for (label, count) in counts {
                out.push_str(&format!("  {label}: {count}\n"));
            }
        }

        if !self.node_activity.is_empty() {
            out.push_str("Node activity:\n");
            let mut activity: Vec<_> = self.node_activity.iter().collect();
            activity.sort_by_key(|(a, _)| *a);
            for (node, count) in activity {
                out.push_str(&format!("  {node}: {count}\n"));
            }
        }

        out.push_str(&format!("Errors: {}\n", self.errors.len()));

        if let Some(dur) = self.total_duration {
            out.push_str(&format!("Total duration: {:?}\n", dur));
        }

        out
    }
}

// ---------------------------------------------------------------------------
// AuditTrail
// ---------------------------------------------------------------------------

/// Higher-level queries that correlate related audit events into logical
/// traces (e.g. matching NodeEnter with NodeExit for a given node).
pub struct AuditTrail<'a> {
    log: &'a AuditLog,
}

impl<'a> AuditTrail<'a> {
    /// Create a new trail view over the given audit log.
    pub fn new(log: &'a AuditLog) -> Self {
        Self { log }
    }

    /// Return matched NodeEnter/NodeExit pairs for the given node, in order.
    pub fn trace_node_execution(&self, node_name: &str) -> Vec<AuditEvent> {
        let events = self.log.get_events();
        events
            .into_iter()
            .filter(|e| {
                e.node_name.as_deref() == Some(node_name)
                    && matches!(
                        e.event_type,
                        AuditEventType::NodeEnter | AuditEventType::NodeExit
                    )
            })
            .collect()
    }

    /// Return GraphStart and GraphEnd events — the outer execution boundary.
    pub fn trace_graph_execution(&self) -> Vec<AuditEvent> {
        let events = self.log.get_events();
        events
            .into_iter()
            .filter(|e| {
                matches!(
                    e.event_type,
                    AuditEventType::GraphStart | AuditEventType::GraphEnd
                )
            })
            .collect()
    }

    /// Return all error events (event type `Error` or severity >= Error).
    pub fn find_errors(&self) -> Vec<AuditEvent> {
        let events = self.log.get_events();
        events
            .into_iter()
            .filter(|e| {
                matches!(e.event_type, AuditEventType::Error { .. })
                    || e.severity >= AuditSeverity::Error
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Helper: build an AuditEvent conveniently
// ---------------------------------------------------------------------------

/// Build an [`AuditEvent`] with sensible defaults, filling in UUID and
/// timestamp automatically.
pub fn make_event(
    event_type: AuditEventType,
    node_name: Option<&str>,
    details: &str,
    severity: AuditSeverity,
) -> AuditEvent {
    AuditEvent {
        id: Uuid::new_v4().to_string(),
        timestamp: iso_now(),
        event_type,
        node_name: node_name.map(|s| s.to_string()),
        details: details.to_string(),
        state_before: None,
        state_after: None,
        duration: None,
        metadata: HashMap::new(),
        severity,
    }
}

/// Return an ISO-8601 UTC timestamp string using the same algorithm as
/// the snapshot module.
fn iso_now() -> String {
    use std::time::SystemTime;
    let dur = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    let days = secs / 86400;
    let time_secs = secs % 86400;
    let hours = time_secs / 3600;
    let minutes = (time_secs % 3600) / 60;
    let seconds = time_secs % 60;
    let (year, month, day) = days_to_ymd(days);
    format!("{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z")
}

/// Convert days since 1970-01-01 to (year, month, day).
fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y as u64, m, d)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Convenience: create an event with the given type, node, details, and
    /// severity, using [`make_event`].
    fn evt(
        event_type: AuditEventType,
        node: Option<&str>,
        details: &str,
        severity: AuditSeverity,
    ) -> AuditEvent {
        make_event(event_type, node, details, severity)
    }

    /// Like [`evt`] but allows setting a custom timestamp.
    fn evt_at(
        event_type: AuditEventType,
        node: Option<&str>,
        details: &str,
        severity: AuditSeverity,
        timestamp: &str,
    ) -> AuditEvent {
        let mut e = evt(event_type, node, details, severity);
        e.timestamp = timestamp.to_string();
        e
    }

    // 1. Record and retrieve events
    #[test]
    fn test_record_and_retrieve_events() {
        let log = AuditLog::new(AuditLogConfig::default());
        log.record(evt(
            AuditEventType::GraphStart,
            None,
            "start",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("agent"),
            "entering agent",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::GraphEnd,
            None,
            "end",
            AuditSeverity::Info,
        ));

        let events = log.get_events();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].event_type, AuditEventType::GraphStart);
        assert_eq!(events[1].node_name, Some("agent".to_string()));
        assert_eq!(events[2].event_type, AuditEventType::GraphEnd);
    }

    // 2. Filter by event type
    #[test]
    fn test_filter_by_event_type() {
        let log = AuditLog::new(AuditLogConfig::default());
        log.record(evt(
            AuditEventType::GraphStart,
            None,
            "s",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("a"),
            "n",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("b"),
            "n",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::GraphEnd,
            None,
            "e",
            AuditSeverity::Info,
        ));

        let enters = log.get_events_by_type(&AuditEventType::NodeEnter);
        assert_eq!(enters.len(), 2);

        let starts = log.get_events_by_type(&AuditEventType::GraphStart);
        assert_eq!(starts.len(), 1);
    }

    // 3. Filter by node name
    #[test]
    fn test_filter_by_node_name() {
        let log = AuditLog::new(AuditLogConfig::default());
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("agent"),
            "enter",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeExit,
            Some("agent"),
            "exit",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("tools"),
            "enter",
            AuditSeverity::Info,
        ));

        let agent_events = log.get_events_for_node("agent");
        assert_eq!(agent_events.len(), 2);

        let tools_events = log.get_events_for_node("tools");
        assert_eq!(tools_events.len(), 1);

        let missing = log.get_events_for_node("unknown");
        assert!(missing.is_empty());
    }

    // 4. Time range filtering
    #[test]
    fn test_time_range_filtering() {
        let log = AuditLog::new(AuditLogConfig::default());
        log.record(evt_at(
            AuditEventType::GraphStart,
            None,
            "s",
            AuditSeverity::Info,
            "2026-01-01T00:00:00Z",
        ));
        log.record(evt_at(
            AuditEventType::NodeEnter,
            Some("a"),
            "n",
            AuditSeverity::Info,
            "2026-01-01T01:00:00Z",
        ));
        log.record(evt_at(
            AuditEventType::NodeExit,
            Some("a"),
            "n",
            AuditSeverity::Info,
            "2026-01-01T02:00:00Z",
        ));
        log.record(evt_at(
            AuditEventType::GraphEnd,
            None,
            "e",
            AuditSeverity::Info,
            "2026-01-01T03:00:00Z",
        ));

        let range = log.get_events_in_range("2026-01-01T00:30:00Z", "2026-01-01T02:30:00Z");
        assert_eq!(range.len(), 2); // 01:00 and 02:00
    }

    // 5. Text search in event details
    #[test]
    fn test_text_search() {
        let log = AuditLog::new(AuditLogConfig::default());
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("a"),
            "Entering node agent",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeExit,
            Some("a"),
            "Exiting node agent",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::Error {
                message: "boom".into(),
            },
            Some("b"),
            "Error in tools: timeout occurred",
            AuditSeverity::Error,
        ));

        let results = log.search("agent");
        assert_eq!(results.len(), 2);

        let results = log.search("timeout");
        assert_eq!(results.len(), 1);

        let results = log.search("ENTERING"); // case-insensitive
        assert_eq!(results.len(), 1);

        let results = log.search("nonexistent");
        assert!(results.is_empty());
    }

    // 6. Export JSON
    #[test]
    fn test_export_json() {
        let log = AuditLog::new(AuditLogConfig::default());
        log.record(evt(
            AuditEventType::GraphStart,
            None,
            "s",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::GraphEnd,
            None,
            "e",
            AuditSeverity::Info,
        ));

        let json_str = log.export_json();
        let parsed: Vec<AuditEvent> = serde_json::from_str(&json_str).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].event_type, AuditEventType::GraphStart);
    }

    // 7. Config: max events rolling buffer
    #[test]
    fn test_max_events_rolling_buffer() {
        let config = AuditLogConfig {
            max_events: Some(3),
            ..Default::default()
        };
        let log = AuditLog::new(config);

        for i in 0..5 {
            log.record(evt(
                AuditEventType::NodeEnter,
                Some("n"),
                &format!("event {i}"),
                AuditSeverity::Info,
            ));
        }

        assert_eq!(log.size(), 3);
        let events = log.get_events();
        // The oldest two (0, 1) should have been evicted.
        assert_eq!(events[0].details, "event 2");
        assert_eq!(events[1].details, "event 3");
        assert_eq!(events[2].details, "event 4");
    }

    // 8. Config: min severity filtering
    #[test]
    fn test_min_severity_filtering() {
        let config = AuditLogConfig {
            min_severity: AuditSeverity::Warning,
            ..Default::default()
        };
        let log = AuditLog::new(config);

        log.record(evt(
            AuditEventType::GraphStart,
            None,
            "info event",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("a"),
            "warn event",
            AuditSeverity::Warning,
        ));
        log.record(evt(
            AuditEventType::Error {
                message: "err".into(),
            },
            Some("b"),
            "error event",
            AuditSeverity::Error,
        ));
        log.record(evt(
            AuditEventType::Error {
                message: "crit".into(),
            },
            None,
            "critical event",
            AuditSeverity::Critical,
        ));

        // Info event should be filtered out.
        assert_eq!(log.size(), 3);
        let events = log.get_events();
        assert!(events.iter().all(|e| e.severity >= AuditSeverity::Warning));
    }

    // 9. AuditReport generation
    #[test]
    fn test_audit_report_generation() {
        let log = AuditLog::new(AuditLogConfig::default());
        log.record(evt(
            AuditEventType::GraphStart,
            None,
            "start",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("agent"),
            "enter",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeExit,
            Some("agent"),
            "exit",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::Error {
                message: "oops".into(),
            },
            Some("agent"),
            "something broke",
            AuditSeverity::Error,
        ));
        let mut end_evt = evt(AuditEventType::GraphEnd, None, "end", AuditSeverity::Info);
        end_evt.duration = Some(Duration::from_millis(500));
        log.record(end_evt);

        let report = AuditReport::generate(&log);
        assert_eq!(report.total_events, 5);
        assert_eq!(report.errors.len(), 1);
        assert_eq!(report.total_duration, Some(Duration::from_millis(500)));

        let summary = report.summary();
        assert!(summary.contains("5 total events"));
        assert!(summary.contains("Errors: 1"));
    }

    // 10. Event counts by type
    #[test]
    fn test_event_counts_by_type() {
        let log = AuditLog::new(AuditLogConfig::default());
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("a"),
            "e",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("b"),
            "e",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeExit,
            Some("a"),
            "x",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::EdgeTraversal,
            None,
            "edge",
            AuditSeverity::Info,
        ));

        let report = AuditReport::generate(&log);
        assert_eq!(report.event_counts.get("NodeEnter"), Some(&2));
        assert_eq!(report.event_counts.get("NodeExit"), Some(&1));
        assert_eq!(report.event_counts.get("EdgeTraversal"), Some(&1));
    }

    // 11. AuditTrail trace node execution
    #[test]
    fn test_audit_trail_trace_node() {
        let log = AuditLog::new(AuditLogConfig::default());
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("agent"),
            "enter",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::EdgeTraversal,
            Some("agent"),
            "edge",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeExit,
            Some("agent"),
            "exit",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("tools"),
            "enter",
            AuditSeverity::Info,
        ));

        let trail = AuditTrail::new(&log);
        let agent_trace = trail.trace_node_execution("agent");
        assert_eq!(agent_trace.len(), 2); // Enter + Exit, not EdgeTraversal
        assert_eq!(agent_trace[0].event_type, AuditEventType::NodeEnter);
        assert_eq!(agent_trace[1].event_type, AuditEventType::NodeExit);
    }

    // 12. AuditTrail find errors
    #[test]
    fn test_audit_trail_find_errors() {
        let log = AuditLog::new(AuditLogConfig::default());
        log.record(evt(
            AuditEventType::GraphStart,
            None,
            "s",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::Error {
                message: "timeout".into(),
            },
            Some("agent"),
            "timeout",
            AuditSeverity::Error,
        ));
        log.record(evt(
            AuditEventType::Error {
                message: "crash".into(),
            },
            Some("tools"),
            "crash",
            AuditSeverity::Critical,
        ));
        log.record(evt(
            AuditEventType::GraphEnd,
            None,
            "e",
            AuditSeverity::Info,
        ));

        let trail = AuditTrail::new(&log);
        let errors = trail.find_errors();
        assert_eq!(errors.len(), 2);
    }

    // 13. Clear log
    #[test]
    fn test_clear_log() {
        let log = AuditLog::new(AuditLogConfig::default());
        log.record(evt(
            AuditEventType::GraphStart,
            None,
            "s",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::GraphEnd,
            None,
            "e",
            AuditSeverity::Info,
        ));
        assert_eq!(log.size(), 2);

        log.clear();
        assert_eq!(log.size(), 0);
        assert!(log.get_events().is_empty());
    }

    // 14. State capture in events
    #[test]
    fn test_state_capture_in_events() {
        let log = AuditLog::new(AuditLogConfig::default());

        let mut event = evt(
            AuditEventType::NodeEnter,
            Some("agent"),
            "entering with state",
            AuditSeverity::Info,
        );
        event.state_before = Some(json!({"counter": 0}));
        event.state_after = Some(json!({"counter": 1}));
        event.duration = Some(Duration::from_millis(42));
        event.metadata.insert("user".to_string(), json!("alice"));
        log.record(event);

        let events = log.get_events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].state_before, Some(json!({"counter": 0})));
        assert_eq!(events[0].state_after, Some(json!({"counter": 1})));
        assert_eq!(events[0].duration, Some(Duration::from_millis(42)));
        assert_eq!(events[0].metadata.get("user"), Some(&json!("alice")));
    }

    // 15. AuditTrail trace graph execution
    #[test]
    fn test_audit_trail_trace_graph_execution() {
        let log = AuditLog::new(AuditLogConfig::default());
        log.record(evt(
            AuditEventType::GraphStart,
            None,
            "start",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("a"),
            "n",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeExit,
            Some("a"),
            "n",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::GraphEnd,
            None,
            "end",
            AuditSeverity::Info,
        ));

        let trail = AuditTrail::new(&log);
        let trace = trail.trace_graph_execution();
        assert_eq!(trace.len(), 2);
        assert_eq!(trace[0].event_type, AuditEventType::GraphStart);
        assert_eq!(trace[1].event_type, AuditEventType::GraphEnd);
    }

    // 16. Disabled log records nothing
    #[test]
    fn test_disabled_log() {
        let config = AuditLogConfig {
            enabled: false,
            ..Default::default()
        };
        let log = AuditLog::new(config);
        log.record(evt(
            AuditEventType::GraphStart,
            None,
            "s",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::GraphEnd,
            None,
            "e",
            AuditSeverity::Critical,
        ));
        assert_eq!(log.size(), 0);
    }

    // 17. Node activity in report
    #[test]
    fn test_node_activity_in_report() {
        let log = AuditLog::new(AuditLogConfig::default());
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("agent"),
            "e",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeExit,
            Some("agent"),
            "x",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("agent"),
            "e",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeExit,
            Some("agent"),
            "x",
            AuditSeverity::Info,
        ));
        log.record(evt(
            AuditEventType::NodeEnter,
            Some("tools"),
            "e",
            AuditSeverity::Info,
        ));

        let report = AuditReport::generate(&log);
        assert_eq!(report.node_activity.get("agent"), Some(&4));
        assert_eq!(report.node_activity.get("tools"), Some(&1));
    }
}
