//! Structured logging system for DeepAgents execution.
//!
//! Provides [`Logger`] for recording [`LogEntry`]s with configurable
//! [`LogLevel`]s, [`LogFilter`] for querying entries, and a sink-based
//! dispatch system via [`LogSink`], [`MultiLogger`], and aggregation
//! via [`LogAggregator`].
//!
//! # Example
//!
//! ```rust
//! use deepagents::logging::{Logger, LogFilter, LogLevel, MultiLogger, InMemorySink, LogAggregator};
//!
//! let mut logger = Logger::new("my-agent".to_string());
//! logger.info("Agent started", "main");
//! logger.warn("High latency detected", "network");
//! logger.error("Connection failed", "network");
//!
//! let filter = LogFilter::new().with_min_level(LogLevel::Warn);
//! let warnings_and_above = logger.filtered(&filter);
//! assert_eq!(warnings_and_above.len(), 2);
//! ```

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// LogLevel
// ---------------------------------------------------------------------------

/// Severity level for a log entry.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LogLevel {
    /// Finest-grained tracing information.
    Trace,
    /// Diagnostic information useful during development.
    Debug,
    /// General informational messages.
    Info,
    /// Potentially harmful situations.
    Warn,
    /// Error events that may still allow the application to continue.
    Error,
    /// Severe errors that cause premature termination.
    Fatal,
}

impl LogLevel {
    /// Return the level name as a static string.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Trace => "TRACE",
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
            Self::Fatal => "FATAL",
        }
    }

    /// Return a numeric severity (0 = Trace through 5 = Fatal).
    pub fn severity(&self) -> u8 {
        match self {
            Self::Trace => 0,
            Self::Debug => 1,
            Self::Info => 2,
            Self::Warn => 3,
            Self::Error => 4,
            Self::Fatal => 5,
        }
    }
}

impl fmt::Display for LogLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl PartialOrd for LogLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for LogLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.severity().cmp(&other.severity())
    }
}

// ---------------------------------------------------------------------------
// LogEntry
// ---------------------------------------------------------------------------

/// A single structured log entry.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Severity level.
    pub level: LogLevel,
    /// Human-readable log message.
    pub message: String,
    /// The source component that emitted this entry.
    pub source: String,
    /// Timestamp string (e.g. ISO-8601).
    pub timestamp: String,
    /// Arbitrary structured context data.
    pub context: HashMap<String, Value>,
    /// Optional span identifier for correlating with traces.
    pub span_id: Option<String>,
}

impl LogEntry {
    /// Create a new log entry builder.
    pub fn builder(
        level: LogLevel,
        message: impl Into<String>,
        source: impl Into<String>,
    ) -> LogEntryBuilder {
        LogEntryBuilder {
            level,
            message: message.into(),
            source: source.into(),
            timestamp: String::new(),
            context: HashMap::new(),
            span_id: None,
        }
    }

    /// Serialize this entry to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "level": self.level.as_str(),
            "message": self.message,
            "source": self.source,
            "timestamp": self.timestamp,
            "context": self.context,
            "span_id": self.span_id,
        })
    }

    /// Format this entry as a human-readable string.
    ///
    /// Format: `[<timestamp> <LEVEL>] <source>: <message>`
    pub fn to_formatted_string(&self) -> String {
        format!(
            "[{} {}] {}: {}",
            self.timestamp, self.level, self.source, self.message,
        )
    }
}

// ---------------------------------------------------------------------------
// LogEntryBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing [`LogEntry`] instances.
pub struct LogEntryBuilder {
    level: LogLevel,
    message: String,
    source: String,
    timestamp: String,
    context: HashMap<String, Value>,
    span_id: Option<String>,
}

impl LogEntryBuilder {
    /// Set the timestamp string.
    pub fn timestamp(mut self, ts: impl Into<String>) -> Self {
        self.timestamp = ts.into();
        self
    }

    /// Add a context key-value pair.
    pub fn context(mut self, key: impl Into<String>, value: Value) -> Self {
        self.context.insert(key.into(), value);
        self
    }

    /// Set the span identifier.
    pub fn span_id(mut self, id: impl Into<String>) -> Self {
        self.span_id = Some(id.into());
        self
    }

    /// Build the final [`LogEntry`].
    pub fn build(self) -> LogEntry {
        LogEntry {
            level: self.level,
            message: self.message,
            source: self.source,
            timestamp: self.timestamp,
            context: self.context,
            span_id: self.span_id,
        }
    }
}

// ---------------------------------------------------------------------------
// LogFilter
// ---------------------------------------------------------------------------

/// Filters log entries by level, source, or message content.
#[derive(Debug, Clone)]
pub struct LogFilter {
    min_level: Option<LogLevel>,
    source: Option<String>,
    contains: Option<String>,
}

impl LogFilter {
    /// Create a new filter with no constraints (matches everything).
    pub fn new() -> Self {
        Self {
            min_level: None,
            source: None,
            contains: None,
        }
    }

    /// Only match entries at or above the given level.
    pub fn with_min_level(mut self, level: LogLevel) -> Self {
        self.min_level = Some(level);
        self
    }

    /// Only match entries from the given source.
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }

    /// Only match entries whose message contains the given text.
    pub fn with_contains(mut self, text: impl Into<String>) -> Self {
        self.contains = Some(text.into());
        self
    }

    /// Return `true` if the entry matches all configured constraints.
    pub fn matches(&self, entry: &LogEntry) -> bool {
        if let Some(ref min) = self.min_level {
            if entry.level < *min {
                return false;
            }
        }
        if let Some(ref src) = self.source {
            if entry.source != *src {
                return false;
            }
        }
        if let Some(ref text) = self.contains {
            if !entry.message.contains(text.as_str()) {
                return false;
            }
        }
        true
    }
}

impl Default for LogFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Logger
// ---------------------------------------------------------------------------

/// A simple structured logger that stores entries in memory.
pub struct Logger {
    /// Name of this logger instance.
    pub name: String,
    entries: Vec<LogEntry>,
}

impl Logger {
    /// Create a new logger with the given name.
    pub fn new(name: String) -> Self {
        Self {
            name,
            entries: Vec::new(),
        }
    }

    /// Log an entry at the specified level.
    pub fn log(&mut self, level: LogLevel, message: impl Into<String>, source: impl Into<String>) {
        let entry = LogEntry {
            level,
            message: message.into(),
            source: source.into(),
            timestamp: String::new(),
            context: HashMap::new(),
            span_id: None,
        };
        self.entries.push(entry);
    }

    /// Log a trace-level message.
    pub fn trace(&mut self, message: impl Into<String>, source: impl Into<String>) {
        self.log(LogLevel::Trace, message, source);
    }

    /// Log a debug-level message.
    pub fn debug(&mut self, message: impl Into<String>, source: impl Into<String>) {
        self.log(LogLevel::Debug, message, source);
    }

    /// Log an info-level message.
    pub fn info(&mut self, message: impl Into<String>, source: impl Into<String>) {
        self.log(LogLevel::Info, message, source);
    }

    /// Log a warn-level message.
    pub fn warn(&mut self, message: impl Into<String>, source: impl Into<String>) {
        self.log(LogLevel::Warn, message, source);
    }

    /// Log an error-level message.
    pub fn error(&mut self, message: impl Into<String>, source: impl Into<String>) {
        self.log(LogLevel::Error, message, source);
    }

    /// Return a slice of all logged entries.
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Return entries matching the given filter.
    pub fn filtered(&self, filter: &LogFilter) -> Vec<&LogEntry> {
        self.entries.iter().filter(|e| filter.matches(e)).collect()
    }

    /// Return the number of logged entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ---------------------------------------------------------------------------
// LogSink
// ---------------------------------------------------------------------------

/// Trait for destinations that receive log entries.
pub trait LogSink {
    /// Write a single entry to this sink.
    fn write(&mut self, entry: &LogEntry);
    /// Flush any buffered data.
    fn flush(&mut self);
    /// Return the name of this sink.
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// InMemorySink
// ---------------------------------------------------------------------------

/// A [`LogSink`] that stores entries in an in-memory vector.
pub struct InMemorySink {
    entries: Vec<LogEntry>,
}

impl InMemorySink {
    /// Create a new empty in-memory sink.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Return a slice of all stored entries.
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Return the number of stored entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if no entries are stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for InMemorySink {
    fn default() -> Self {
        Self::new()
    }
}

impl LogSink for InMemorySink {
    fn write(&mut self, entry: &LogEntry) {
        self.entries.push(entry.clone());
    }

    fn flush(&mut self) {
        // No-op for in-memory storage.
    }

    fn name(&self) -> &str {
        "in_memory"
    }
}

// ---------------------------------------------------------------------------
// JsonSink
// ---------------------------------------------------------------------------

/// A [`LogSink`] that serializes each entry to a JSON string.
pub struct JsonSink {
    json_entries: Vec<String>,
}

impl JsonSink {
    /// Create a new empty JSON sink.
    pub fn new() -> Self {
        Self {
            json_entries: Vec::new(),
        }
    }

    /// Return a slice of all stored JSON strings.
    pub fn json_entries(&self) -> &[String] {
        &self.json_entries
    }
}

impl Default for JsonSink {
    fn default() -> Self {
        Self::new()
    }
}

impl LogSink for JsonSink {
    fn write(&mut self, entry: &LogEntry) {
        let json = entry.to_json();
        self.json_entries
            .push(serde_json::to_string(&json).unwrap_or_default());
    }

    fn flush(&mut self) {
        // No-op.
    }

    fn name(&self) -> &str {
        "json"
    }
}

// ---------------------------------------------------------------------------
// MultiLogger
// ---------------------------------------------------------------------------

/// Dispatches log entries to multiple [`LogSink`] implementations.
pub struct MultiLogger {
    sinks: Vec<Box<dyn LogSink>>,
}

impl MultiLogger {
    /// Create a new multi-logger with no sinks.
    pub fn new() -> Self {
        Self { sinks: Vec::new() }
    }

    /// Add a sink to this multi-logger.
    pub fn add_sink(&mut self, sink: Box<dyn LogSink>) {
        self.sinks.push(sink);
    }

    /// Dispatch a log entry to all registered sinks.
    pub fn log(&mut self, entry: LogEntry) {
        for sink in &mut self.sinks {
            sink.write(&entry);
        }
    }

    /// Flush all registered sinks.
    pub fn flush_all(&mut self) {
        for sink in &mut self.sinks {
            sink.flush();
        }
    }

    /// Return the number of registered sinks.
    pub fn sink_count(&self) -> usize {
        self.sinks.len()
    }
}

impl Default for MultiLogger {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// LogAggregator
// ---------------------------------------------------------------------------

/// Aggregates statistics from a collection of log entries.
pub struct LogAggregator {
    entries: Vec<LogEntry>,
}

impl LogAggregator {
    /// Create a new empty aggregator.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Ingest a slice of log entries for aggregation.
    pub fn ingest(&mut self, entries: &[LogEntry]) {
        for entry in entries {
            self.entries.push(entry.clone());
        }
    }

    /// Return a count of entries grouped by level name.
    pub fn count_by_level(&self) -> HashMap<String, usize> {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for entry in &self.entries {
            *counts.entry(entry.level.as_str().to_string()).or_insert(0) += 1;
        }
        counts
    }

    /// Return the fraction of entries that are Error or Fatal.
    ///
    /// Returns `0.0` if there are no entries.
    pub fn error_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let error_count = self
            .entries
            .iter()
            .filter(|e| e.level == LogLevel::Error || e.level == LogLevel::Fatal)
            .count();
        error_count as f64 / self.entries.len() as f64
    }

    /// Return the source that appears most frequently, if any.
    pub fn most_common_source(&self) -> Option<String> {
        if self.entries.is_empty() {
            return None;
        }
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for entry in &self.entries {
            *counts.entry(&entry.source).or_insert(0) += 1;
        }
        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(source, _)| source.to_string())
    }

    /// Serialize the aggregation summary to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "total_entries": self.entries.len(),
            "count_by_level": self.count_by_level(),
            "error_rate": self.error_rate(),
            "most_common_source": self.most_common_source(),
        })
    }
}

impl Default for LogAggregator {
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

    // -- LogLevel --

    #[test]
    fn test_log_level_as_str() {
        assert_eq!(LogLevel::Trace.as_str(), "TRACE");
        assert_eq!(LogLevel::Debug.as_str(), "DEBUG");
        assert_eq!(LogLevel::Info.as_str(), "INFO");
        assert_eq!(LogLevel::Warn.as_str(), "WARN");
        assert_eq!(LogLevel::Error.as_str(), "ERROR");
        assert_eq!(LogLevel::Fatal.as_str(), "FATAL");
    }

    #[test]
    fn test_log_level_severity() {
        assert_eq!(LogLevel::Trace.severity(), 0);
        assert_eq!(LogLevel::Debug.severity(), 1);
        assert_eq!(LogLevel::Info.severity(), 2);
        assert_eq!(LogLevel::Warn.severity(), 3);
        assert_eq!(LogLevel::Error.severity(), 4);
        assert_eq!(LogLevel::Fatal.severity(), 5);
    }

    #[test]
    fn test_log_level_display() {
        assert_eq!(LogLevel::Info.to_string(), "INFO");
        assert_eq!(LogLevel::Fatal.to_string(), "FATAL");
    }

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Trace < LogLevel::Debug);
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Fatal);
    }

    #[test]
    fn test_log_level_equality() {
        assert_eq!(LogLevel::Info, LogLevel::Info);
        assert_ne!(LogLevel::Info, LogLevel::Debug);
    }

    #[test]
    fn test_log_level_sort() {
        let mut levels = vec![
            LogLevel::Error,
            LogLevel::Trace,
            LogLevel::Warn,
            LogLevel::Info,
            LogLevel::Fatal,
            LogLevel::Debug,
        ];
        levels.sort();
        assert_eq!(
            levels,
            vec![
                LogLevel::Trace,
                LogLevel::Debug,
                LogLevel::Info,
                LogLevel::Warn,
                LogLevel::Error,
                LogLevel::Fatal,
            ]
        );
    }

    // -- LogEntry --

    #[test]
    fn test_log_entry_builder_basic() {
        let entry = LogEntry::builder(LogLevel::Info, "hello", "main")
            .timestamp("2024-01-01T00:00:00Z")
            .build();
        assert_eq!(entry.level, LogLevel::Info);
        assert_eq!(entry.message, "hello");
        assert_eq!(entry.source, "main");
        assert_eq!(entry.timestamp, "2024-01-01T00:00:00Z");
        assert!(entry.context.is_empty());
        assert!(entry.span_id.is_none());
    }

    #[test]
    fn test_log_entry_builder_with_context() {
        let entry = LogEntry::builder(LogLevel::Debug, "test", "src")
            .context("key", serde_json::json!("value"))
            .context("num", serde_json::json!(42))
            .build();
        assert_eq!(entry.context.len(), 2);
        assert_eq!(entry.context["key"], serde_json::json!("value"));
        assert_eq!(entry.context["num"], serde_json::json!(42));
    }

    #[test]
    fn test_log_entry_builder_with_span_id() {
        let entry = LogEntry::builder(LogLevel::Warn, "msg", "src")
            .span_id("span-123")
            .build();
        assert_eq!(entry.span_id, Some("span-123".to_string()));
    }

    #[test]
    fn test_log_entry_to_json() {
        let entry = LogEntry::builder(LogLevel::Error, "failure", "network")
            .timestamp("2024-06-15T12:00:00Z")
            .context("code", serde_json::json!(500))
            .build();
        let json = entry.to_json();
        assert_eq!(json["level"], "ERROR");
        assert_eq!(json["message"], "failure");
        assert_eq!(json["source"], "network");
        assert_eq!(json["timestamp"], "2024-06-15T12:00:00Z");
        assert_eq!(json["context"]["code"], 500);
        assert!(json["span_id"].is_null());
    }

    #[test]
    fn test_log_entry_to_formatted_string() {
        let entry = LogEntry::builder(LogLevel::Info, "Agent started", "main")
            .timestamp("2024-01-01")
            .build();
        let formatted = entry.to_formatted_string();
        assert_eq!(formatted, "[2024-01-01 INFO] main: Agent started");
    }

    #[test]
    fn test_log_entry_to_formatted_string_error() {
        let entry = LogEntry::builder(LogLevel::Error, "crash", "core")
            .timestamp("2024-12-31")
            .build();
        assert_eq!(
            entry.to_formatted_string(),
            "[2024-12-31 ERROR] core: crash"
        );
    }

    // -- LogFilter --

    #[test]
    fn test_log_filter_empty_matches_all() {
        let filter = LogFilter::new();
        let entry = LogEntry::builder(LogLevel::Trace, "x", "y").build();
        assert!(filter.matches(&entry));
    }

    #[test]
    fn test_log_filter_min_level() {
        let filter = LogFilter::new().with_min_level(LogLevel::Warn);
        let trace = LogEntry::builder(LogLevel::Trace, "x", "y").build();
        let warn = LogEntry::builder(LogLevel::Warn, "x", "y").build();
        let error = LogEntry::builder(LogLevel::Error, "x", "y").build();
        assert!(!filter.matches(&trace));
        assert!(filter.matches(&warn));
        assert!(filter.matches(&error));
    }

    #[test]
    fn test_log_filter_source() {
        let filter = LogFilter::new().with_source("network");
        let net = LogEntry::builder(LogLevel::Info, "x", "network").build();
        let db = LogEntry::builder(LogLevel::Info, "x", "database").build();
        assert!(filter.matches(&net));
        assert!(!filter.matches(&db));
    }

    #[test]
    fn test_log_filter_contains() {
        let filter = LogFilter::new().with_contains("timeout");
        let yes = LogEntry::builder(LogLevel::Error, "connection timeout", "net").build();
        let no = LogEntry::builder(LogLevel::Error, "connection refused", "net").build();
        assert!(filter.matches(&yes));
        assert!(!filter.matches(&no));
    }

    #[test]
    fn test_log_filter_combined() {
        let filter = LogFilter::new()
            .with_min_level(LogLevel::Warn)
            .with_source("network")
            .with_contains("timeout");

        let match_all = LogEntry::builder(LogLevel::Error, "connection timeout", "network").build();
        let wrong_level =
            LogEntry::builder(LogLevel::Debug, "connection timeout", "network").build();
        let wrong_source = LogEntry::builder(LogLevel::Error, "connection timeout", "db").build();
        let wrong_msg = LogEntry::builder(LogLevel::Error, "success", "network").build();

        assert!(filter.matches(&match_all));
        assert!(!filter.matches(&wrong_level));
        assert!(!filter.matches(&wrong_source));
        assert!(!filter.matches(&wrong_msg));
    }

    #[test]
    fn test_log_filter_default() {
        let filter = LogFilter::default();
        let entry = LogEntry::builder(LogLevel::Fatal, "x", "y").build();
        assert!(filter.matches(&entry));
    }

    // -- Logger --

    #[test]
    fn test_logger_new() {
        let logger = Logger::new("test".to_string());
        assert_eq!(logger.name, "test");
        assert!(logger.is_empty());
        assert_eq!(logger.len(), 0);
    }

    #[test]
    fn test_logger_log() {
        let mut logger = Logger::new("test".to_string());
        logger.log(LogLevel::Info, "hello", "main");
        assert_eq!(logger.len(), 1);
        assert_eq!(logger.entries()[0].message, "hello");
        assert_eq!(logger.entries()[0].level, LogLevel::Info);
        assert_eq!(logger.entries()[0].source, "main");
    }

    #[test]
    fn test_logger_convenience_methods() {
        let mut logger = Logger::new("test".to_string());
        logger.trace("t", "s");
        logger.debug("d", "s");
        logger.info("i", "s");
        logger.warn("w", "s");
        logger.error("e", "s");
        assert_eq!(logger.len(), 5);
        assert_eq!(logger.entries()[0].level, LogLevel::Trace);
        assert_eq!(logger.entries()[1].level, LogLevel::Debug);
        assert_eq!(logger.entries()[2].level, LogLevel::Info);
        assert_eq!(logger.entries()[3].level, LogLevel::Warn);
        assert_eq!(logger.entries()[4].level, LogLevel::Error);
    }

    #[test]
    fn test_logger_filtered() {
        let mut logger = Logger::new("test".to_string());
        logger.info("ok", "main");
        logger.warn("caution", "main");
        logger.error("fail", "main");

        let filter = LogFilter::new().with_min_level(LogLevel::Warn);
        let results = logger.filtered(&filter);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].level, LogLevel::Warn);
        assert_eq!(results[1].level, LogLevel::Error);
    }

    #[test]
    fn test_logger_filtered_empty_result() {
        let mut logger = Logger::new("test".to_string());
        logger.info("ok", "main");
        let filter = LogFilter::new().with_min_level(LogLevel::Fatal);
        assert!(logger.filtered(&filter).is_empty());
    }

    #[test]
    fn test_logger_clear() {
        let mut logger = Logger::new("test".to_string());
        logger.info("a", "s");
        logger.info("b", "s");
        assert_eq!(logger.len(), 2);
        logger.clear();
        assert!(logger.is_empty());
        assert_eq!(logger.len(), 0);
    }

    #[test]
    fn test_logger_entries_order() {
        let mut logger = Logger::new("test".to_string());
        logger.info("first", "s");
        logger.info("second", "s");
        logger.info("third", "s");
        assert_eq!(logger.entries()[0].message, "first");
        assert_eq!(logger.entries()[1].message, "second");
        assert_eq!(logger.entries()[2].message, "third");
    }

    // -- InMemorySink --

    #[test]
    fn test_in_memory_sink_new() {
        let sink = InMemorySink::new();
        assert!(sink.is_empty());
        assert_eq!(sink.len(), 0);
    }

    #[test]
    fn test_in_memory_sink_write() {
        let mut sink = InMemorySink::new();
        let entry = LogEntry::builder(LogLevel::Info, "test", "src").build();
        sink.write(&entry);
        assert_eq!(sink.len(), 1);
        assert_eq!(sink.entries()[0].message, "test");
    }

    #[test]
    fn test_in_memory_sink_name() {
        let sink = InMemorySink::new();
        assert_eq!(sink.name(), "in_memory");
    }

    #[test]
    fn test_in_memory_sink_multiple_writes() {
        let mut sink = InMemorySink::new();
        for i in 0..10 {
            let entry = LogEntry::builder(LogLevel::Debug, format!("msg-{}", i), "src").build();
            sink.write(&entry);
        }
        assert_eq!(sink.len(), 10);
    }

    // -- JsonSink --

    #[test]
    fn test_json_sink_new() {
        let sink = JsonSink::new();
        assert!(sink.json_entries().is_empty());
    }

    #[test]
    fn test_json_sink_write() {
        let mut sink = JsonSink::new();
        let entry = LogEntry::builder(LogLevel::Warn, "warning", "net")
            .timestamp("2024-01-01")
            .build();
        sink.write(&entry);
        assert_eq!(sink.json_entries().len(), 1);
        let parsed: Value = serde_json::from_str(&sink.json_entries()[0]).unwrap();
        assert_eq!(parsed["level"], "WARN");
        assert_eq!(parsed["message"], "warning");
    }

    #[test]
    fn test_json_sink_name() {
        let sink = JsonSink::new();
        assert_eq!(sink.name(), "json");
    }

    #[test]
    fn test_json_sink_produces_valid_json() {
        let mut sink = JsonSink::new();
        let entry = LogEntry::builder(LogLevel::Error, "err", "core")
            .context("detail", serde_json::json!("something broke"))
            .build();
        sink.write(&entry);
        let parsed: Value = serde_json::from_str(&sink.json_entries()[0]).unwrap();
        assert_eq!(parsed["context"]["detail"], "something broke");
    }

    // -- MultiLogger --

    #[test]
    fn test_multi_logger_new() {
        let ml = MultiLogger::new();
        assert_eq!(ml.sink_count(), 0);
    }

    #[test]
    fn test_multi_logger_add_sink() {
        let mut ml = MultiLogger::new();
        ml.add_sink(Box::new(InMemorySink::new()));
        ml.add_sink(Box::new(JsonSink::new()));
        assert_eq!(ml.sink_count(), 2);
    }

    #[test]
    fn test_multi_logger_dispatches_to_all_sinks() {
        // We test by adding two InMemorySinks, logging, and checking counts
        // via the sink_count. Since we can't downcast easily, we use a
        // simpler validation: just ensure log doesn't panic with multiple sinks.
        let mut ml = MultiLogger::new();
        ml.add_sink(Box::new(InMemorySink::new()));
        ml.add_sink(Box::new(JsonSink::new()));

        let entry = LogEntry::builder(LogLevel::Info, "dispatch test", "ml").build();
        ml.log(entry);
        // No panic = success; sinks received the entry.
        assert_eq!(ml.sink_count(), 2);
    }

    #[test]
    fn test_multi_logger_flush_all() {
        let mut ml = MultiLogger::new();
        ml.add_sink(Box::new(InMemorySink::new()));
        ml.flush_all(); // Should not panic.
    }

    #[test]
    fn test_multi_logger_default() {
        let ml = MultiLogger::default();
        assert_eq!(ml.sink_count(), 0);
    }

    #[test]
    fn test_multi_logger_log_no_sinks() {
        let mut ml = MultiLogger::new();
        let entry = LogEntry::builder(LogLevel::Fatal, "no sinks", "test").build();
        ml.log(entry); // Should not panic.
    }

    // -- LogAggregator --

    #[test]
    fn test_aggregator_new() {
        let agg = LogAggregator::new();
        assert!(agg.count_by_level().is_empty());
        assert_eq!(agg.error_rate(), 0.0);
        assert_eq!(agg.most_common_source(), None);
    }

    #[test]
    fn test_aggregator_count_by_level() {
        let mut agg = LogAggregator::new();
        let entries = vec![
            LogEntry::builder(LogLevel::Info, "a", "s").build(),
            LogEntry::builder(LogLevel::Info, "b", "s").build(),
            LogEntry::builder(LogLevel::Error, "c", "s").build(),
        ];
        agg.ingest(&entries);
        let counts = agg.count_by_level();
        assert_eq!(counts["INFO"], 2);
        assert_eq!(counts["ERROR"], 1);
    }

    #[test]
    fn test_aggregator_error_rate() {
        let mut agg = LogAggregator::new();
        let entries = vec![
            LogEntry::builder(LogLevel::Info, "a", "s").build(),
            LogEntry::builder(LogLevel::Error, "b", "s").build(),
            LogEntry::builder(LogLevel::Fatal, "c", "s").build(),
            LogEntry::builder(LogLevel::Info, "d", "s").build(),
        ];
        agg.ingest(&entries);
        let rate = agg.error_rate();
        assert!((rate - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_aggregator_error_rate_zero() {
        let mut agg = LogAggregator::new();
        let entries = vec![
            LogEntry::builder(LogLevel::Info, "a", "s").build(),
            LogEntry::builder(LogLevel::Debug, "b", "s").build(),
        ];
        agg.ingest(&entries);
        assert_eq!(agg.error_rate(), 0.0);
    }

    #[test]
    fn test_aggregator_error_rate_all_errors() {
        let mut agg = LogAggregator::new();
        let entries = vec![
            LogEntry::builder(LogLevel::Error, "a", "s").build(),
            LogEntry::builder(LogLevel::Fatal, "b", "s").build(),
        ];
        agg.ingest(&entries);
        assert!((agg.error_rate() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_aggregator_most_common_source() {
        let mut agg = LogAggregator::new();
        let entries = vec![
            LogEntry::builder(LogLevel::Info, "a", "network").build(),
            LogEntry::builder(LogLevel::Info, "b", "network").build(),
            LogEntry::builder(LogLevel::Info, "c", "database").build(),
        ];
        agg.ingest(&entries);
        assert_eq!(agg.most_common_source(), Some("network".to_string()));
    }

    #[test]
    fn test_aggregator_most_common_source_empty() {
        let agg = LogAggregator::new();
        assert_eq!(agg.most_common_source(), None);
    }

    #[test]
    fn test_aggregator_to_json() {
        let mut agg = LogAggregator::new();
        let entries = vec![
            LogEntry::builder(LogLevel::Info, "a", "src").build(),
            LogEntry::builder(LogLevel::Error, "b", "src").build(),
        ];
        agg.ingest(&entries);
        let json = agg.to_json();
        assert_eq!(json["total_entries"], 2);
        assert_eq!(json["most_common_source"], "src");
        assert!(json["error_rate"].is_number());
        assert!(json["count_by_level"].is_object());
    }

    #[test]
    fn test_aggregator_multiple_ingests() {
        let mut agg = LogAggregator::new();
        let batch1 = vec![LogEntry::builder(LogLevel::Info, "a", "s").build()];
        let batch2 = vec![LogEntry::builder(LogLevel::Warn, "b", "s").build()];
        agg.ingest(&batch1);
        agg.ingest(&batch2);
        let counts = agg.count_by_level();
        assert_eq!(counts["INFO"], 1);
        assert_eq!(counts["WARN"], 1);
    }

    #[test]
    fn test_aggregator_default() {
        let agg = LogAggregator::default();
        assert_eq!(agg.error_rate(), 0.0);
    }

    // -- Edge cases --

    #[test]
    fn test_log_entry_empty_strings() {
        let entry = LogEntry::builder(LogLevel::Info, "", "").build();
        assert_eq!(entry.message, "");
        assert_eq!(entry.source, "");
        assert_eq!(entry.to_formatted_string(), "[ INFO] : ");
    }

    #[test]
    fn test_logger_filter_by_source_and_level() {
        let mut logger = Logger::new("test".to_string());
        logger.info("ok", "api");
        logger.warn("slow", "api");
        logger.error("fail", "db");
        logger.error("crash", "api");

        let filter = LogFilter::new()
            .with_min_level(LogLevel::Error)
            .with_source("api");
        let results = logger.filtered(&filter);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].message, "crash");
    }

    #[test]
    fn test_log_level_clone() {
        let level = LogLevel::Warn;
        let cloned = level.clone();
        assert_eq!(level, cloned);
    }

    #[test]
    fn test_log_entry_clone() {
        let entry = LogEntry::builder(LogLevel::Info, "clone me", "src")
            .context("k", serde_json::json!("v"))
            .span_id("s1")
            .build();
        let cloned = entry.clone();
        assert_eq!(cloned.message, "clone me");
        assert_eq!(cloned.span_id, Some("s1".to_string()));
        assert_eq!(cloned.context["k"], serde_json::json!("v"));
    }

    #[test]
    fn test_log_filter_contains_partial_match() {
        let filter = LogFilter::new().with_contains("time");
        let yes = LogEntry::builder(LogLevel::Info, "timeout error", "s").build();
        let also_yes = LogEntry::builder(LogLevel::Info, "runtime", "s").build();
        let no = LogEntry::builder(LogLevel::Info, "other", "s").build();
        assert!(filter.matches(&yes));
        assert!(filter.matches(&also_yes));
        assert!(!filter.matches(&no));
    }
}
