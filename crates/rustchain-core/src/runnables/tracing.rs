//! Tracing and observability system for runnables.
//!
//! Provides structured logging and span tracking for debugging runnable
//! pipelines. The core types are [`TraceCollector`] for gathering trace data,
//! [`TraceEntry`] for log entries, [`TraceSpan`] for span-based timing, and
//! [`TraceFormatter`] / [`TraceExporter`] for output formatting.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// TraceLevel
// ---------------------------------------------------------------------------

/// Severity level for a trace entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TraceLevel {
    /// Verbose diagnostic information.
    Debug,
    /// General informational messages.
    Info,
    /// Potential issues that do not prevent execution.
    Warn,
    /// Errors that may affect execution.
    Error,
}

impl TraceLevel {
    /// Return the level as a static string slice.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Debug => "DEBUG",
            Self::Info => "INFO",
            Self::Warn => "WARN",
            Self::Error => "ERROR",
        }
    }

    /// Numeric priority for ordering (higher = more severe).
    fn priority(&self) -> u8 {
        match self {
            Self::Debug => 0,
            Self::Info => 1,
            Self::Warn => 2,
            Self::Error => 3,
        }
    }
}

impl PartialOrd for TraceLevel {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TraceLevel {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority().cmp(&other.priority())
    }
}

impl fmt::Display for TraceLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// TraceEntry
// ---------------------------------------------------------------------------

/// A single log entry within the tracing system.
#[derive(Debug, Clone)]
pub struct TraceEntry {
    /// Severity level of this entry.
    pub level: TraceLevel,
    /// Human-readable message.
    pub message: String,
    /// The source component that produced this entry.
    pub source: String,
    /// When this entry was created.
    pub timestamp: Instant,
    /// Optional structured data payload.
    pub data: Option<Value>,
    /// Optional span ID to associate this entry with a span.
    pub span_id: Option<String>,
}

impl TraceEntry {
    /// Create a new trace entry with the given level, message, and source.
    pub fn new(level: TraceLevel, message: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            level,
            message: message.into(),
            source: source.into(),
            timestamp: Instant::now(),
            data: None,
            span_id: None,
        }
    }

    /// Attach structured data to this entry.
    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }

    /// Associate this entry with a span.
    pub fn with_span(mut self, span_id: impl Into<String>) -> Self {
        self.span_id = Some(span_id.into());
        self
    }

    /// Serialize this entry to a JSON [`Value`].
    ///
    /// Because `Instant` cannot be serialized directly, the timestamp is
    /// represented as elapsed nanoseconds since the entry was created (i.e.,
    /// the age of the entry at the time of serialization).
    pub fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("level".into(), Value::String(self.level.as_str().into()));
        map.insert("message".into(), Value::String(self.message.clone()));
        map.insert("source".into(), Value::String(self.source.clone()));
        map.insert(
            "elapsed_ns".into(),
            Value::Number(serde_json::Number::from(self.timestamp.elapsed().as_nanos() as u64)),
        );
        if let Some(ref data) = self.data {
            map.insert("data".into(), data.clone());
        }
        if let Some(ref span_id) = self.span_id {
            map.insert("span_id".into(), Value::String(span_id.clone()));
        }
        Value::Object(map)
    }
}

// ---------------------------------------------------------------------------
// SpanResult
// ---------------------------------------------------------------------------

/// The outcome status of a [`TraceSpan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpanResult {
    /// The span completed successfully.
    Ok,
    /// The span completed with an error.
    Error(String),
    /// The span has not yet finished.
    Pending,
}

impl fmt::Display for SpanResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ok => write!(f, "OK"),
            Self::Error(msg) => write!(f, "ERROR: {}", msg),
            Self::Pending => write!(f, "PENDING"),
        }
    }
}

// ---------------------------------------------------------------------------
// TraceSpan
// ---------------------------------------------------------------------------

/// A span representing a unit of work with timing and metadata.
#[derive(Debug, Clone)]
pub struct TraceSpan {
    /// Unique identifier for this span.
    pub id: String,
    /// Human-readable name.
    pub name: String,
    /// Optional parent span ID for establishing hierarchy.
    pub parent_id: Option<String>,
    /// When the span started.
    pub start: Instant,
    /// When the span ended (if finished).
    pub end: Option<Instant>,
    /// Arbitrary key-value attributes.
    pub attributes: HashMap<String, Value>,
    /// The outcome status.
    pub status: SpanResult,
}

impl TraceSpan {
    /// Create a new root span with the given name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            parent_id: None,
            start: Instant::now(),
            end: None,
            attributes: HashMap::new(),
            status: SpanResult::Pending,
        }
    }

    /// Create a child span under the given parent.
    pub fn child(parent_id: impl Into<String>, name: impl Into<String>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            name: name.into(),
            parent_id: Some(parent_id.into()),
            start: Instant::now(),
            end: None,
            attributes: HashMap::new(),
            status: SpanResult::Pending,
        }
    }

    /// Mark this span as finished, recording the end time.
    pub fn finish(&mut self) {
        self.end = Some(Instant::now());
        if self.status == SpanResult::Pending {
            self.status = SpanResult::Ok;
        }
    }

    /// Return the duration of this span, or `None` if it has not finished.
    pub fn duration(&self) -> Option<Duration> {
        self.end.map(|end| end.duration_since(self.start))
    }

    /// Set an attribute on this span.
    pub fn set_attribute(&mut self, key: impl Into<String>, value: Value) {
        self.attributes.insert(key.into(), value);
    }

    /// Set the status of this span.
    pub fn set_status(&mut self, status: SpanResult) {
        self.status = status;
    }

    /// Serialize this span to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let mut map = serde_json::Map::new();
        map.insert("id".into(), Value::String(self.id.clone()));
        map.insert("name".into(), Value::String(self.name.clone()));
        if let Some(ref parent_id) = self.parent_id {
            map.insert("parent_id".into(), Value::String(parent_id.clone()));
        }
        if let Some(duration) = self.duration() {
            map.insert(
                "duration_ms".into(),
                serde_json::json!(duration.as_secs_f64() * 1000.0),
            );
        }
        if !self.attributes.is_empty() {
            map.insert("attributes".into(), serde_json::json!(self.attributes));
        }
        map.insert("status".into(), Value::String(self.status.to_string()));
        Value::Object(map)
    }
}

// ---------------------------------------------------------------------------
// TraceCollector
// ---------------------------------------------------------------------------

/// A thread-safe collector for trace entries and spans.
///
/// Internally uses `Arc<Mutex<...>>` so cloning shares the same underlying
/// storage.
#[derive(Debug, Clone)]
pub struct TraceCollector {
    entries: Arc<Mutex<Vec<TraceEntry>>>,
    spans: Arc<Mutex<Vec<TraceSpan>>>,
    min_level: TraceLevel,
}

impl TraceCollector {
    /// Create a new collector that accepts all levels (starting from `Debug`).
    pub fn new() -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
            spans: Arc::new(Mutex::new(Vec::new())),
            min_level: TraceLevel::Debug,
        }
    }

    /// Create a new collector that only records entries at or above the given level.
    pub fn with_min_level(level: TraceLevel) -> Self {
        Self {
            entries: Arc::new(Mutex::new(Vec::new())),
            spans: Arc::new(Mutex::new(Vec::new())),
            min_level: level,
        }
    }

    /// Log a trace entry. Entries below the minimum level are silently dropped.
    pub fn log(&self, entry: TraceEntry) {
        if entry.level >= self.min_level {
            self.entries.lock().unwrap().push(entry);
        }
    }

    /// Start a new root span with the given name and return it.
    pub fn start_span(&self, name: &str) -> TraceSpan {
        TraceSpan::new(name)
    }

    /// Record a finished (or in-progress) span in the collector.
    pub fn finish_span(&self, span: TraceSpan) {
        self.spans.lock().unwrap().push(span);
    }

    /// Return a snapshot of all collected entries.
    pub fn entries(&self) -> Vec<TraceEntry> {
        self.entries.lock().unwrap().clone()
    }

    /// Return a snapshot of all collected spans.
    pub fn spans(&self) -> Vec<TraceSpan> {
        self.spans.lock().unwrap().clone()
    }

    /// Return entries matching the given level.
    pub fn entries_by_level(&self, level: TraceLevel) -> Vec<TraceEntry> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.level == level)
            .cloned()
            .collect()
    }

    /// Return entries associated with the given span ID.
    pub fn entries_for_span(&self, span_id: &str) -> Vec<TraceEntry> {
        self.entries
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.span_id.as_deref() == Some(span_id))
            .cloned()
            .collect()
    }

    /// Remove all entries and spans.
    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
        self.spans.lock().unwrap().clear();
    }

    /// Return the number of collected entries.
    pub fn entry_count(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    /// Return the number of collected spans.
    pub fn span_count(&self) -> usize {
        self.spans.lock().unwrap().len()
    }
}

impl Default for TraceCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TraceFormatter trait + implementations
// ---------------------------------------------------------------------------

/// A formatter that converts trace entries into a string representation.
pub trait TraceFormatter: Send + Sync {
    /// Format the given entries into a single string.
    fn format(&self, entries: &[TraceEntry]) -> String;
}

/// Formats trace entries as multi-line human-readable text.
#[derive(Debug, Default)]
pub struct TextTraceFormatter;

impl TraceFormatter for TextTraceFormatter {
    fn format(&self, entries: &[TraceEntry]) -> String {
        let mut output = String::new();
        for entry in entries {
            let span_str = entry
                .span_id
                .as_deref()
                .map(|s| format!(" [span:{}]", s))
                .unwrap_or_default();
            output.push_str(&format!(
                "[{}] {}{} - {}\n",
                entry.level.as_str(),
                entry.source,
                span_str,
                entry.message,
            ));
        }
        output
    }
}

/// Formats trace entries as a JSON array.
#[derive(Debug, Default)]
pub struct JsonTraceFormatter;

impl TraceFormatter for JsonTraceFormatter {
    fn format(&self, entries: &[TraceEntry]) -> String {
        let arr: Vec<Value> = entries.iter().map(|e| e.to_json()).collect();
        serde_json::to_string_pretty(&Value::Array(arr)).unwrap_or_else(|_| "[]".to_string())
    }
}

/// Formats each trace entry as a single compact line.
#[derive(Debug, Default)]
pub struct CompactTraceFormatter;

impl TraceFormatter for CompactTraceFormatter {
    fn format(&self, entries: &[TraceEntry]) -> String {
        let mut output = String::new();
        for entry in entries {
            output.push_str(&format!(
                "{} {} {}\n",
                entry.level.as_str(),
                entry.source,
                entry.message,
            ));
        }
        output
    }
}

// ---------------------------------------------------------------------------
// TraceExporter
// ---------------------------------------------------------------------------

/// Exports collected trace data using a [`TraceFormatter`].
pub struct TraceExporter {
    formatter: Box<dyn TraceFormatter>,
}

impl TraceExporter {
    /// Create a new exporter with the given formatter.
    pub fn new(formatter: Box<dyn TraceFormatter>) -> Self {
        Self { formatter }
    }

    /// Export all entries from the collector.
    pub fn export(&self, collector: &TraceCollector) -> String {
        let entries = collector.entries();
        self.formatter.format(&entries)
    }

    /// Export all spans from the collector as JSON.
    pub fn export_spans(&self, collector: &TraceCollector) -> String {
        let spans = collector.spans();
        let arr: Vec<Value> = spans.iter().map(|s| s.to_json()).collect();
        serde_json::to_string_pretty(&Value::Array(arr)).unwrap_or_else(|_| "[]".to_string())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // TraceLevel tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_trace_level_as_str() {
        assert_eq!(TraceLevel::Debug.as_str(), "DEBUG");
        assert_eq!(TraceLevel::Info.as_str(), "INFO");
        assert_eq!(TraceLevel::Warn.as_str(), "WARN");
        assert_eq!(TraceLevel::Error.as_str(), "ERROR");
    }

    #[test]
    fn test_trace_level_ordering() {
        assert!(TraceLevel::Debug < TraceLevel::Info);
        assert!(TraceLevel::Info < TraceLevel::Warn);
        assert!(TraceLevel::Warn < TraceLevel::Error);
        assert!(TraceLevel::Debug < TraceLevel::Error);
    }

    #[test]
    fn test_trace_level_equality() {
        assert_eq!(TraceLevel::Debug, TraceLevel::Debug);
        assert_ne!(TraceLevel::Debug, TraceLevel::Info);
    }

    #[test]
    fn test_trace_level_display() {
        assert_eq!(format!("{}", TraceLevel::Info), "INFO");
        assert_eq!(format!("{}", TraceLevel::Error), "ERROR");
    }

    #[test]
    fn test_trace_level_sort() {
        let mut levels = vec![
            TraceLevel::Error,
            TraceLevel::Debug,
            TraceLevel::Warn,
            TraceLevel::Info,
        ];
        levels.sort();
        assert_eq!(
            levels,
            vec![
                TraceLevel::Debug,
                TraceLevel::Info,
                TraceLevel::Warn,
                TraceLevel::Error
            ]
        );
    }

    // -----------------------------------------------------------------------
    // TraceEntry tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_trace_entry_new() {
        let entry = TraceEntry::new(TraceLevel::Info, "hello", "test_source");
        assert_eq!(entry.level, TraceLevel::Info);
        assert_eq!(entry.message, "hello");
        assert_eq!(entry.source, "test_source");
        assert!(entry.data.is_none());
        assert!(entry.span_id.is_none());
    }

    #[test]
    fn test_trace_entry_with_data() {
        let entry =
            TraceEntry::new(TraceLevel::Debug, "msg", "src").with_data(json!({"key": "val"}));
        assert_eq!(entry.data, Some(json!({"key": "val"})));
    }

    #[test]
    fn test_trace_entry_with_span() {
        let entry = TraceEntry::new(TraceLevel::Warn, "msg", "src").with_span("span-123");
        assert_eq!(entry.span_id, Some("span-123".to_string()));
    }

    #[test]
    fn test_trace_entry_chained_builders() {
        let entry = TraceEntry::new(TraceLevel::Error, "fail", "engine")
            .with_data(json!(42))
            .with_span("s1");
        assert_eq!(entry.level, TraceLevel::Error);
        assert_eq!(entry.data, Some(json!(42)));
        assert_eq!(entry.span_id, Some("s1".to_string()));
    }

    #[test]
    fn test_trace_entry_to_json() {
        let entry = TraceEntry::new(TraceLevel::Info, "test msg", "my_source")
            .with_data(json!({"x": 1}))
            .with_span("span-abc");
        let json = entry.to_json();
        assert_eq!(json["level"], "INFO");
        assert_eq!(json["message"], "test msg");
        assert_eq!(json["source"], "my_source");
        assert_eq!(json["data"], json!({"x": 1}));
        assert_eq!(json["span_id"], "span-abc");
        assert!(json["elapsed_ns"].is_number());
    }

    #[test]
    fn test_trace_entry_to_json_minimal() {
        let entry = TraceEntry::new(TraceLevel::Debug, "m", "s");
        let json = entry.to_json();
        assert!(json.get("data").is_none());
        assert!(json.get("span_id").is_none());
    }

    // -----------------------------------------------------------------------
    // SpanResult tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_span_result_display() {
        assert_eq!(SpanResult::Ok.to_string(), "OK");
        assert_eq!(SpanResult::Pending.to_string(), "PENDING");
        assert_eq!(
            SpanResult::Error("bad".into()).to_string(),
            "ERROR: bad"
        );
    }

    #[test]
    fn test_span_result_equality() {
        assert_eq!(SpanResult::Ok, SpanResult::Ok);
        assert_eq!(SpanResult::Pending, SpanResult::Pending);
        assert_eq!(
            SpanResult::Error("x".into()),
            SpanResult::Error("x".into())
        );
        assert_ne!(SpanResult::Ok, SpanResult::Pending);
        assert_ne!(
            SpanResult::Error("a".into()),
            SpanResult::Error("b".into())
        );
    }

    // -----------------------------------------------------------------------
    // TraceSpan tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_span_new() {
        let span = TraceSpan::new("my-span");
        assert_eq!(span.name, "my-span");
        assert!(span.parent_id.is_none());
        assert!(!span.id.is_empty());
        assert!(span.end.is_none());
        assert_eq!(span.status, SpanResult::Pending);
        assert!(span.attributes.is_empty());
    }

    #[test]
    fn test_span_child() {
        let parent = TraceSpan::new("parent");
        let child = TraceSpan::child(&parent.id, "child");
        assert_eq!(child.parent_id, Some(parent.id.clone()));
        assert_eq!(child.name, "child");
        assert_ne!(child.id, parent.id);
    }

    #[test]
    fn test_span_finish() {
        let mut span = TraceSpan::new("work");
        assert!(span.duration().is_none());
        assert_eq!(span.status, SpanResult::Pending);

        span.finish();
        assert!(span.end.is_some());
        assert_eq!(span.status, SpanResult::Ok);
        assert!(span.duration().is_some());
    }

    #[test]
    fn test_span_finish_preserves_error_status() {
        let mut span = TraceSpan::new("work");
        span.set_status(SpanResult::Error("failed".into()));
        span.finish();
        // finish should NOT overwrite an Error status.
        assert_eq!(span.status, SpanResult::Error("failed".into()));
        assert!(span.end.is_some());
    }

    #[test]
    fn test_span_duration() {
        let mut span = TraceSpan::new("timed");
        std::thread::sleep(Duration::from_millis(10));
        span.finish();
        let dur = span.duration().unwrap();
        assert!(dur >= Duration::from_millis(5));
    }

    #[test]
    fn test_span_set_attribute() {
        let mut span = TraceSpan::new("attrs");
        span.set_attribute("model", json!("gpt-4"));
        span.set_attribute("tokens", json!(100));
        assert_eq!(span.attributes.get("model"), Some(&json!("gpt-4")));
        assert_eq!(span.attributes.get("tokens"), Some(&json!(100)));
    }

    #[test]
    fn test_span_to_json() {
        let mut span = TraceSpan::new("json-span");
        span.set_attribute("key", json!("value"));
        span.finish();

        let json = span.to_json();
        assert_eq!(json["name"], "json-span");
        assert_eq!(json["status"], "OK");
        assert!(json["duration_ms"].is_number());
        assert!(json["attributes"].is_object());
        assert_eq!(json["id"], span.id.as_str());
    }

    #[test]
    fn test_span_to_json_unfinished() {
        let span = TraceSpan::new("pending");
        let json = span.to_json();
        assert_eq!(json["status"], "PENDING");
        assert!(json.get("duration_ms").is_none());
    }

    #[test]
    fn test_span_to_json_with_parent() {
        let child = TraceSpan::child("parent-id-123", "child-span");
        let json = child.to_json();
        assert_eq!(json["parent_id"], "parent-id-123");
    }

    // -----------------------------------------------------------------------
    // TraceCollector tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_collector_new_is_empty() {
        let collector = TraceCollector::new();
        assert_eq!(collector.entry_count(), 0);
        assert_eq!(collector.span_count(), 0);
        assert!(collector.entries().is_empty());
        assert!(collector.spans().is_empty());
    }

    #[test]
    fn test_collector_log_and_entries() {
        let collector = TraceCollector::new();
        collector.log(TraceEntry::new(TraceLevel::Info, "first", "src"));
        collector.log(TraceEntry::new(TraceLevel::Warn, "second", "src"));
        assert_eq!(collector.entry_count(), 2);
        let entries = collector.entries();
        assert_eq!(entries[0].message, "first");
        assert_eq!(entries[1].message, "second");
    }

    #[test]
    fn test_collector_min_level_filtering() {
        let collector = TraceCollector::with_min_level(TraceLevel::Warn);
        collector.log(TraceEntry::new(TraceLevel::Debug, "dropped", "src"));
        collector.log(TraceEntry::new(TraceLevel::Info, "dropped too", "src"));
        collector.log(TraceEntry::new(TraceLevel::Warn, "kept", "src"));
        collector.log(TraceEntry::new(TraceLevel::Error, "also kept", "src"));
        assert_eq!(collector.entry_count(), 2);
        let entries = collector.entries();
        assert_eq!(entries[0].message, "kept");
        assert_eq!(entries[1].message, "also kept");
    }

    #[test]
    fn test_collector_start_and_finish_span() {
        let collector = TraceCollector::new();
        let mut span = collector.start_span("my-op");
        assert_eq!(span.name, "my-op");
        span.finish();
        collector.finish_span(span);
        assert_eq!(collector.span_count(), 1);
        let spans = collector.spans();
        assert_eq!(spans[0].name, "my-op");
        assert_eq!(spans[0].status, SpanResult::Ok);
    }

    #[test]
    fn test_collector_entries_by_level() {
        let collector = TraceCollector::new();
        collector.log(TraceEntry::new(TraceLevel::Info, "a", "s"));
        collector.log(TraceEntry::new(TraceLevel::Error, "b", "s"));
        collector.log(TraceEntry::new(TraceLevel::Info, "c", "s"));
        collector.log(TraceEntry::new(TraceLevel::Debug, "d", "s"));

        let infos = collector.entries_by_level(TraceLevel::Info);
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].message, "a");
        assert_eq!(infos[1].message, "c");

        let errors = collector.entries_by_level(TraceLevel::Error);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].message, "b");
    }

    #[test]
    fn test_collector_entries_for_span() {
        let collector = TraceCollector::new();
        collector.log(TraceEntry::new(TraceLevel::Info, "no span", "s"));
        collector.log(
            TraceEntry::new(TraceLevel::Info, "in span A", "s").with_span("span-A"),
        );
        collector.log(
            TraceEntry::new(TraceLevel::Warn, "also span A", "s").with_span("span-A"),
        );
        collector.log(
            TraceEntry::new(TraceLevel::Info, "in span B", "s").with_span("span-B"),
        );

        let span_a = collector.entries_for_span("span-A");
        assert_eq!(span_a.len(), 2);
        assert_eq!(span_a[0].message, "in span A");
        assert_eq!(span_a[1].message, "also span A");

        let span_b = collector.entries_for_span("span-B");
        assert_eq!(span_b.len(), 1);

        let span_c = collector.entries_for_span("span-C");
        assert!(span_c.is_empty());
    }

    #[test]
    fn test_collector_clear() {
        let collector = TraceCollector::new();
        collector.log(TraceEntry::new(TraceLevel::Info, "msg", "s"));
        let mut span = collector.start_span("sp");
        span.finish();
        collector.finish_span(span);
        assert_eq!(collector.entry_count(), 1);
        assert_eq!(collector.span_count(), 1);

        collector.clear();
        assert_eq!(collector.entry_count(), 0);
        assert_eq!(collector.span_count(), 0);
    }

    #[test]
    fn test_collector_thread_safety() {
        let collector = TraceCollector::new();
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let c = collector.clone();
                std::thread::spawn(move || {
                    c.log(TraceEntry::new(
                        TraceLevel::Info,
                        format!("thread-{}", i),
                        "test",
                    ));
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(collector.entry_count(), 10);
    }

    #[test]
    fn test_collector_default() {
        let collector = TraceCollector::default();
        assert_eq!(collector.entry_count(), 0);
    }

    // -----------------------------------------------------------------------
    // TextTraceFormatter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_text_formatter_empty() {
        let fmt = TextTraceFormatter;
        assert_eq!(fmt.format(&[]), "");
    }

    #[test]
    fn test_text_formatter_output() {
        let fmt = TextTraceFormatter;
        let entries = vec![
            TraceEntry::new(TraceLevel::Info, "starting up", "engine"),
            TraceEntry::new(TraceLevel::Error, "kaboom", "tool").with_span("s1"),
        ];
        let output = fmt.format(&entries);
        assert!(output.contains("[INFO] engine - starting up"));
        assert!(output.contains("[ERROR] tool [span:s1] - kaboom"));
    }

    // -----------------------------------------------------------------------
    // JsonTraceFormatter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_json_formatter_empty() {
        let fmt = JsonTraceFormatter;
        let output = fmt.format(&[]);
        let parsed: Value = serde_json::from_str(&output).unwrap();
        assert_eq!(parsed, json!([]));
    }

    #[test]
    fn test_json_formatter_output() {
        let fmt = JsonTraceFormatter;
        let entries = vec![TraceEntry::new(TraceLevel::Warn, "caution", "mod_a")];
        let output = fmt.format(&entries);
        let parsed: Value = serde_json::from_str(&output).unwrap();
        assert!(parsed.is_array());
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["level"], "WARN");
        assert_eq!(arr[0]["message"], "caution");
    }

    // -----------------------------------------------------------------------
    // CompactTraceFormatter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_compact_formatter_empty() {
        let fmt = CompactTraceFormatter;
        assert_eq!(fmt.format(&[]), "");
    }

    #[test]
    fn test_compact_formatter_output() {
        let fmt = CompactTraceFormatter;
        let entries = vec![
            TraceEntry::new(TraceLevel::Debug, "detail", "core"),
            TraceEntry::new(TraceLevel::Info, "ok", "core"),
        ];
        let output = fmt.format(&entries);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "DEBUG core detail");
        assert_eq!(lines[1], "INFO core ok");
    }

    // -----------------------------------------------------------------------
    // TraceExporter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_exporter_export_entries() {
        let collector = TraceCollector::new();
        collector.log(TraceEntry::new(TraceLevel::Info, "hello", "src"));
        collector.log(TraceEntry::new(TraceLevel::Warn, "world", "src"));

        let exporter = TraceExporter::new(Box::new(TextTraceFormatter));
        let output = exporter.export(&collector);
        assert!(output.contains("[INFO] src - hello"));
        assert!(output.contains("[WARN] src - world"));
    }

    #[test]
    fn test_exporter_export_spans() {
        let collector = TraceCollector::new();
        let mut span = TraceSpan::new("op1");
        span.finish();
        collector.finish_span(span);

        let exporter = TraceExporter::new(Box::new(JsonTraceFormatter));
        let output = exporter.export_spans(&collector);
        let parsed: Value = serde_json::from_str(&output).unwrap();
        assert!(parsed.is_array());
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["name"], "op1");
        assert_eq!(arr[0]["status"], "OK");
    }

    #[test]
    fn test_exporter_empty_collector() {
        let collector = TraceCollector::new();
        let exporter = TraceExporter::new(Box::new(CompactTraceFormatter));
        assert_eq!(exporter.export(&collector), "");
        let spans_output = exporter.export_spans(&collector);
        let parsed: Value = serde_json::from_str(&spans_output).unwrap();
        assert_eq!(parsed, json!([]));
    }

    #[test]
    fn test_exporter_with_json_formatter() {
        let collector = TraceCollector::new();
        collector.log(
            TraceEntry::new(TraceLevel::Error, "fail", "engine").with_data(json!({"code": 500})),
        );

        let exporter = TraceExporter::new(Box::new(JsonTraceFormatter));
        let output = exporter.export(&collector);
        let parsed: Value = serde_json::from_str(&output).unwrap();
        let arr = parsed.as_array().unwrap();
        assert_eq!(arr[0]["data"]["code"], 500);
    }

    // -----------------------------------------------------------------------
    // Edge case tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unfinished_span_has_no_duration() {
        let span = TraceSpan::new("pending");
        assert!(span.duration().is_none());
        assert_eq!(span.status, SpanResult::Pending);
    }

    #[test]
    fn test_collector_entries_for_nonexistent_span() {
        let collector = TraceCollector::new();
        collector.log(TraceEntry::new(TraceLevel::Info, "msg", "s"));
        assert!(collector.entries_for_span("nonexistent").is_empty());
    }

    #[test]
    fn test_collector_entries_by_level_no_matches() {
        let collector = TraceCollector::new();
        collector.log(TraceEntry::new(TraceLevel::Info, "only info", "s"));
        assert!(collector.entries_by_level(TraceLevel::Error).is_empty());
    }

    #[test]
    fn test_multiple_child_spans() {
        let parent = TraceSpan::new("parent");
        let child1 = TraceSpan::child(&parent.id, "child1");
        let child2 = TraceSpan::child(&parent.id, "child2");
        assert_eq!(child1.parent_id, Some(parent.id.clone()));
        assert_eq!(child2.parent_id, Some(parent.id.clone()));
        assert_ne!(child1.id, child2.id);
    }

    #[test]
    fn test_span_set_status_explicitly() {
        let mut span = TraceSpan::new("op");
        span.set_status(SpanResult::Error("timeout".into()));
        assert_eq!(span.status, SpanResult::Error("timeout".into()));
    }
}
