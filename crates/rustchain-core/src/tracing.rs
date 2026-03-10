//! Tracing and observability for the LLM framework.
//!
//! This module provides structured tracing for debugging and monitoring LLM
//! pipelines. It tracks execution spans, latency, and provides export
//! capabilities for analysis and visualization.
//!
//! # Key Types
//!
//! - [`TraceId`] — Unique identifier for a trace (a complete request/operation).
//! - [`SpanId`] — Unique identifier for a span within a trace.
//! - [`Span`] — A unit of work with timing, status, attributes, and events.
//! - [`Trace`] — A collection of related spans forming a tree.
//! - [`TraceCollector`] — Collects and manages traces during execution.
//! - [`TraceExporter`] — Exports traces in JSON, summary, or flamegraph formats.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// TraceId
// ---------------------------------------------------------------------------

/// Unique identifier for a trace.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceId(String);

impl TraceId {
    /// Generate a new unique trace id (UUID v4-like using random bytes).
    pub fn new() -> Self {
        Self(generate_uuid_v4())
    }

    /// Create a `TraceId` from an existing string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Return the id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for TraceId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// SpanId
// ---------------------------------------------------------------------------

/// Unique identifier for a span within a trace.
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpanId(String);

impl SpanId {
    /// Generate a new unique span id.
    pub fn new() -> Self {
        Self(generate_uuid_v4())
    }

    /// Create a `SpanId` from an existing string.
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Return the id as a string slice.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for SpanId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for SpanId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// SpanStatus
// ---------------------------------------------------------------------------

/// Status of a span.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SpanStatus {
    /// The span completed successfully.
    Ok,
    /// The span ended with an error.
    Error(String),
    /// Status has not been set.
    Unset,
}

impl fmt::Display for SpanStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpanStatus::Ok => write!(f, "OK"),
            SpanStatus::Error(msg) => write!(f, "ERROR: {}", msg),
            SpanStatus::Unset => write!(f, "UNSET"),
        }
    }
}

// ---------------------------------------------------------------------------
// SpanEvent
// ---------------------------------------------------------------------------

/// An event recorded within a span.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanEvent {
    /// Name of the event.
    pub name: String,
    /// ISO-8601 timestamp of when the event occurred.
    pub timestamp: String,
    /// Arbitrary key-value attributes attached to the event.
    pub attributes: HashMap<String, Value>,
}

impl SpanEvent {
    /// Create a new event with the current timestamp.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            timestamp: current_timestamp(),
            attributes: HashMap::new(),
        }
    }

    /// Add an attribute to this event (builder style).
    pub fn with_attribute(mut self, key: impl Into<String>, value: Value) -> Self {
        self.attributes.insert(key.into(), value);
        self
    }

    /// Serialize this event to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": self.attributes,
        })
    }
}

// ---------------------------------------------------------------------------
// Span
// ---------------------------------------------------------------------------

/// A unit of work within a trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// The trace this span belongs to.
    pub trace_id: TraceId,
    /// Unique identifier for this span.
    pub span_id: SpanId,
    /// Optional parent span id (root spans have `None`).
    pub parent_id: Option<SpanId>,
    /// Human-readable name describing this span.
    pub name: String,
    /// ISO-8601 start timestamp.
    pub start_time: String,
    /// ISO-8601 end timestamp (set when the span finishes).
    pub end_time: Option<String>,
    /// Completion status.
    pub status: SpanStatus,
    /// Arbitrary key-value attributes.
    pub attributes: HashMap<String, Value>,
    /// Events recorded during the span's lifetime.
    pub events: Vec<SpanEvent>,
}

impl Span {
    /// Create a new in-progress span.
    pub fn new(trace_id: TraceId, name: impl Into<String>) -> Self {
        Self {
            trace_id,
            span_id: SpanId::new(),
            parent_id: None,
            name: name.into(),
            start_time: current_timestamp(),
            end_time: None,
            status: SpanStatus::Unset,
            attributes: HashMap::new(),
            events: Vec::new(),
        }
    }

    /// Set the parent span id (builder style).
    pub fn with_parent(mut self, parent_id: SpanId) -> Self {
        self.parent_id = Some(parent_id);
        self
    }

    /// Attach an attribute to this span.
    pub fn set_attribute(&mut self, key: impl Into<String>, value: Value) {
        self.attributes.insert(key.into(), value);
    }

    /// Record an event on this span.
    pub fn add_event(&mut self, event: SpanEvent) {
        self.events.push(event);
    }

    /// Mark this span as finished with the current timestamp and status `Ok`
    /// (if the status is still `Unset`).
    pub fn finish(&mut self) {
        self.end_time = Some(current_timestamp());
        if self.status == SpanStatus::Unset {
            self.status = SpanStatus::Ok;
        }
    }

    /// Compute the duration in milliseconds between start and end.
    ///
    /// Returns `None` if the span has not finished or timestamps cannot be
    /// parsed.
    pub fn duration_ms(&self) -> Option<u64> {
        let end = self.end_time.as_ref()?;
        let start_ms = parse_timestamp_ms(&self.start_time)?;
        let end_ms = parse_timestamp_ms(end)?;
        Some(end_ms.saturating_sub(start_ms))
    }

    /// Returns `true` if the span has been finished.
    pub fn is_finished(&self) -> bool {
        self.end_time.is_some()
    }

    /// Serialize this span to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "trace_id": self.trace_id.as_str(),
            "span_id": self.span_id.as_str(),
            "parent_id": self.parent_id.as_ref().map(|p| p.as_str()),
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "status": format!("{}", self.status),
            "attributes": self.attributes,
            "events": self.events.iter().map(|e| e.to_json()).collect::<Vec<_>>(),
        })
    }
}

// ---------------------------------------------------------------------------
// Trace
// ---------------------------------------------------------------------------

/// A collection of related spans forming an execution tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trace {
    /// The unique id shared by all spans in this trace.
    pub trace_id: TraceId,
    /// Ordered list of spans (insertion order).
    pub spans: Vec<Span>,
}

impl Trace {
    /// Create a new empty trace.
    pub fn new() -> Self {
        Self {
            trace_id: TraceId::new(),
            spans: Vec::new(),
        }
    }

    /// Add a span to this trace.
    pub fn add_span(&mut self, span: Span) {
        self.spans.push(span);
    }

    /// Return the root span (the first span without a parent), if any.
    pub fn root_span(&self) -> Option<&Span> {
        self.spans.iter().find(|s| s.parent_id.is_none())
    }

    /// Look up a span by its id.
    pub fn get_span(&self, span_id: &SpanId) -> Option<&Span> {
        self.spans.iter().find(|s| &s.span_id == span_id)
    }

    /// Return all direct children of the given span.
    pub fn children_of(&self, span_id: &SpanId) -> Vec<&Span> {
        self.spans
            .iter()
            .filter(|s| s.parent_id.as_ref() == Some(span_id))
            .collect()
    }

    /// Number of spans in this trace.
    pub fn span_count(&self) -> usize {
        self.spans.len()
    }

    /// Total wall-clock duration from the earliest start to the latest end, in
    /// milliseconds. Returns `None` if there are no finished spans.
    pub fn total_duration_ms(&self) -> Option<u64> {
        let min_start = self
            .spans
            .iter()
            .filter_map(|s| parse_timestamp_ms(&s.start_time))
            .min()?;
        let max_end = self
            .spans
            .iter()
            .filter_map(|s| s.end_time.as_ref().and_then(|e| parse_timestamp_ms(e)))
            .max()?;
        Some(max_end.saturating_sub(min_start))
    }

    /// Serialize the entire trace to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "trace_id": self.trace_id.as_str(),
            "spans": self.spans.iter().map(|s| s.to_json()).collect::<Vec<_>>(),
        })
    }
}

impl Default for Trace {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TraceCollector
// ---------------------------------------------------------------------------

/// Collects and manages traces during execution.
pub struct TraceCollector {
    traces: HashMap<String, Trace>,
}

impl TraceCollector {
    /// Create an empty collector.
    pub fn new() -> Self {
        Self {
            traces: HashMap::new(),
        }
    }

    /// Start a new trace with a root span of the given name. Returns the
    /// [`TraceId`].
    pub fn start_trace(&mut self, name: impl Into<String>) -> TraceId {
        let mut trace = Trace::new();
        let trace_id = trace.trace_id.clone();
        let span = Span::new(trace_id.clone(), name);
        trace.add_span(span);
        self.traces.insert(trace_id.as_str().to_string(), trace);
        trace_id
    }

    /// Start a new root-level span inside an existing trace. Returns the
    /// [`SpanId`] of the new span, or `None` if the trace does not exist.
    pub fn start_span(&mut self, trace_id: &TraceId, name: impl Into<String>) -> Option<SpanId> {
        let trace = self.traces.get_mut(trace_id.as_str())?;
        let span = Span::new(trace_id.clone(), name);
        let span_id = span.span_id.clone();
        trace.add_span(span);
        Some(span_id)
    }

    /// Start a child span under the given parent. Returns the [`SpanId`] of
    /// the new span, or `None` if the trace does not exist.
    pub fn start_child_span(
        &mut self,
        trace_id: &TraceId,
        parent_id: &SpanId,
        name: impl Into<String>,
    ) -> Option<SpanId> {
        let trace = self.traces.get_mut(trace_id.as_str())?;
        let span = Span::new(trace_id.clone(), name).with_parent(parent_id.clone());
        let span_id = span.span_id.clone();
        trace.add_span(span);
        Some(span_id)
    }

    /// Finish a span (sets end time and status).
    pub fn finish_span(&mut self, trace_id: &TraceId, span_id: &SpanId) {
        if let Some(trace) = self.traces.get_mut(trace_id.as_str()) {
            if let Some(span) = trace.spans.iter_mut().find(|s| &s.span_id == span_id) {
                span.finish();
            }
        }
    }

    /// Set the status of a span.
    pub fn set_span_status(&mut self, trace_id: &TraceId, span_id: &SpanId, status: SpanStatus) {
        if let Some(trace) = self.traces.get_mut(trace_id.as_str()) {
            if let Some(span) = trace.spans.iter_mut().find(|s| &s.span_id == span_id) {
                span.status = status;
            }
        }
    }

    /// Attach an attribute to a span.
    pub fn set_span_attribute(
        &mut self,
        trace_id: &TraceId,
        span_id: &SpanId,
        key: impl Into<String>,
        value: Value,
    ) {
        if let Some(trace) = self.traces.get_mut(trace_id.as_str()) {
            if let Some(span) = trace.spans.iter_mut().find(|s| &s.span_id == span_id) {
                span.set_attribute(key, value);
            }
        }
    }

    /// Retrieve a trace by its id.
    pub fn get_trace(&self, trace_id: &TraceId) -> Option<&Trace> {
        self.traces.get(trace_id.as_str())
    }

    /// Return references to all collected traces.
    pub fn all_traces(&self) -> Vec<&Trace> {
        self.traces.values().collect()
    }

    /// Remove all collected traces.
    pub fn clear(&mut self) {
        self.traces.clear();
    }
}

impl Default for TraceCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// FlameEntry
// ---------------------------------------------------------------------------

/// A single entry suitable for flamegraph rendering.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlameEntry {
    /// The span name.
    pub name: String,
    /// Depth in the call tree (root = 0).
    pub depth: usize,
    /// Duration in milliseconds.
    pub duration_ms: u64,
}

// ---------------------------------------------------------------------------
// TraceExporter
// ---------------------------------------------------------------------------

/// Exports traces in various formats for analysis and visualization.
pub struct TraceExporter;

impl TraceExporter {
    /// Export a trace as a JSON [`Value`].
    pub fn to_json(trace: &Trace) -> Value {
        trace.to_json()
    }

    /// Produce a human-readable summary of a trace.
    pub fn to_summary(trace: &Trace) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Trace: {}", trace.trace_id));
        lines.push(format!("Spans: {}", trace.span_count()));
        if let Some(dur) = trace.total_duration_ms() {
            lines.push(format!("Total duration: {}ms", dur));
        }
        lines.push(String::new());

        for span in &trace.spans {
            let depth = Self::span_depth(trace, span);
            let indent = "  ".repeat(depth);
            let dur_str = span
                .duration_ms()
                .map(|d| format!(" ({}ms)", d))
                .unwrap_or_default();
            let status_str = format!(" [{}]", span.status);
            lines.push(format!(
                "{}{}{}{}", indent, span.name, dur_str, status_str
            ));
        }
        lines.join("\n")
    }

    /// Convert a trace into flamegraph entries.
    pub fn to_flamegraph_data(trace: &Trace) -> Vec<FlameEntry> {
        let mut entries = Vec::new();
        for span in &trace.spans {
            let depth = Self::span_depth(trace, span);
            let duration_ms = span.duration_ms().unwrap_or(0);
            entries.push(FlameEntry {
                name: span.name.clone(),
                depth,
                duration_ms,
            });
        }
        entries
    }

    /// Compute the depth of a span in its trace tree.
    fn span_depth(trace: &Trace, span: &Span) -> usize {
        let mut depth = 0;
        let mut current_parent = span.parent_id.as_ref();
        while let Some(pid) = current_parent {
            depth += 1;
            if let Some(parent_span) = trace.get_span(pid) {
                current_parent = parent_span.parent_id.as_ref();
            } else {
                break;
            }
        }
        depth
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Generate a UUID v4-like string using random bytes from the standard library.
fn generate_uuid_v4() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    // Combine several entropy sources to produce a pseudo-random 128-bit value.
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let state = RandomState::new();
    let mut h1 = state.build_hasher();
    h1.write_u128(now.as_nanos());
    let part1 = h1.finish();

    let state2 = RandomState::new();
    let mut h2 = state2.build_hasher();
    h2.write_u64(part1.wrapping_add(now.subsec_nanos() as u64));
    let part2 = h2.finish();

    let bytes: [u8; 16] = {
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&part1.to_le_bytes());
        b[8..].copy_from_slice(&part2.to_le_bytes());
        // Set version 4 and variant bits.
        b[6] = (b[6] & 0x0f) | 0x40;
        b[8] = (b[8] & 0x3f) | 0x80;
        b
    };

    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    )
}

/// Return the current time as an ISO-8601 timestamp string (millisecond
/// precision, UTC).
fn current_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let dur = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let total_ms = dur.as_millis() as u64;
    let secs = total_ms / 1000;
    let millis = total_ms % 1000;

    // Manual UTC breakdown (no chrono).
    let (year, month, day, hour, minute, second) = epoch_secs_to_utc(secs);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year, month, day, hour, minute, second, millis
    )
}

/// Convert epoch seconds to (year, month, day, hour, minute, second) in UTC.
fn epoch_secs_to_utc(secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let second = secs % 60;
    let minute = (secs / 60) % 60;
    let hour = (secs / 3600) % 24;

    let mut days = secs / 86400;
    let mut year = 1970u64;
    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    let month_days: [u64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 0u64;
    for (i, &md) in month_days.iter().enumerate() {
        if days < md {
            month = i as u64 + 1;
            break;
        }
        days -= md;
    }

    let day = days + 1;
    (year, month, day, hour, minute, second)
}

fn is_leap_year(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

/// Parse an ISO-8601 timestamp (the format produced by [`current_timestamp`])
/// into epoch milliseconds. Returns `None` on parse failure.
fn parse_timestamp_ms(s: &str) -> Option<u64> {
    // Expected format: "YYYY-MM-DDTHH:MM:SS.mmmZ"
    if s.len() < 24 {
        return None;
    }
    let year: u64 = s.get(0..4)?.parse().ok()?;
    let month: u64 = s.get(5..7)?.parse().ok()?;
    let day: u64 = s.get(8..10)?.parse().ok()?;
    let hour: u64 = s.get(11..13)?.parse().ok()?;
    let minute: u64 = s.get(14..16)?.parse().ok()?;
    let second: u64 = s.get(17..19)?.parse().ok()?;
    let millis: u64 = s.get(20..23)?.parse().ok()?;

    // Convert to epoch seconds.
    let mut total_days: u64 = 0;
    for y in 1970..year {
        total_days += if is_leap_year(y) { 366 } else { 365 };
    }
    let month_days: [u64; 12] = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    for i in 0..(month as usize).saturating_sub(1) {
        total_days += month_days[i];
    }
    total_days += day.saturating_sub(1);

    let total_secs = total_days * 86400 + hour * 3600 + minute * 60 + second;
    Some(total_secs * 1000 + millis)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- TraceId tests --

    #[test]
    fn test_trace_id_new_is_unique() {
        let a = TraceId::new();
        let b = TraceId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn test_trace_id_from_string() {
        let id = TraceId::from_string("abc-123");
        assert_eq!(id.as_str(), "abc-123");
    }

    #[test]
    fn test_trace_id_display() {
        let id = TraceId::from_string("trace-1");
        assert_eq!(format!("{}", id), "trace-1");
    }

    #[test]
    fn test_trace_id_clone_eq() {
        let id = TraceId::from_string("t1");
        let cloned = id.clone();
        assert_eq!(id, cloned);
    }

    #[test]
    fn test_trace_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TraceId::from_string("a"));
        set.insert(TraceId::from_string("a"));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn test_trace_id_default() {
        let id = TraceId::default();
        assert!(!id.as_str().is_empty());
    }

    #[test]
    fn test_trace_id_serialize() {
        let id = TraceId::from_string("ser-test");
        let json = serde_json::to_string(&id).unwrap();
        assert!(json.contains("ser-test"));
    }

    // -- SpanId tests --

    #[test]
    fn test_span_id_new_is_unique() {
        let a = SpanId::new();
        let b = SpanId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn test_span_id_from_string() {
        let id = SpanId::from_string("span-1");
        assert_eq!(id.as_str(), "span-1");
    }

    #[test]
    fn test_span_id_display() {
        let id = SpanId::from_string("s-display");
        assert_eq!(format!("{}", id), "s-display");
    }

    #[test]
    fn test_span_id_clone_eq() {
        let id = SpanId::from_string("s1");
        assert_eq!(id.clone(), id);
    }

    #[test]
    fn test_span_id_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(SpanId::from_string("x"));
        set.insert(SpanId::from_string("y"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_span_id_default() {
        let id = SpanId::default();
        assert!(!id.as_str().is_empty());
    }

    // -- SpanStatus tests --

    #[test]
    fn test_span_status_display_ok() {
        assert_eq!(format!("{}", SpanStatus::Ok), "OK");
    }

    #[test]
    fn test_span_status_display_error() {
        let s = SpanStatus::Error("timeout".into());
        assert_eq!(format!("{}", s), "ERROR: timeout");
    }

    #[test]
    fn test_span_status_display_unset() {
        assert_eq!(format!("{}", SpanStatus::Unset), "UNSET");
    }

    #[test]
    fn test_span_status_eq() {
        assert_eq!(SpanStatus::Ok, SpanStatus::Ok);
        assert_ne!(SpanStatus::Ok, SpanStatus::Unset);
    }

    // -- SpanEvent tests --

    #[test]
    fn test_span_event_new() {
        let evt = SpanEvent::new("my-event");
        assert_eq!(evt.name, "my-event");
        assert!(!evt.timestamp.is_empty());
        assert!(evt.attributes.is_empty());
    }

    #[test]
    fn test_span_event_with_attribute() {
        let evt = SpanEvent::new("evt")
            .with_attribute("key", json!("value"))
            .with_attribute("num", json!(42));
        assert_eq!(evt.attributes.len(), 2);
        assert_eq!(evt.attributes["key"], json!("value"));
    }

    #[test]
    fn test_span_event_to_json() {
        let evt = SpanEvent::new("e1").with_attribute("a", json!(1));
        let j = evt.to_json();
        assert_eq!(j["name"], "e1");
        assert_eq!(j["attributes"]["a"], 1);
        assert!(j["timestamp"].is_string());
    }

    // -- Span tests --

    #[test]
    fn test_span_new() {
        let tid = TraceId::from_string("t1");
        let span = Span::new(tid.clone(), "root");
        assert_eq!(span.trace_id, tid);
        assert_eq!(span.name, "root");
        assert!(span.parent_id.is_none());
        assert_eq!(span.status, SpanStatus::Unset);
        assert!(!span.is_finished());
    }

    #[test]
    fn test_span_with_parent() {
        let tid = TraceId::from_string("t");
        let pid = SpanId::from_string("parent");
        let span = Span::new(tid, "child").with_parent(pid.clone());
        assert_eq!(span.parent_id.unwrap(), pid);
    }

    #[test]
    fn test_span_set_attribute() {
        let mut span = Span::new(TraceId::from_string("t"), "s");
        span.set_attribute("model", json!("gpt-4"));
        assert_eq!(span.attributes["model"], json!("gpt-4"));
    }

    #[test]
    fn test_span_add_event() {
        let mut span = Span::new(TraceId::from_string("t"), "s");
        span.add_event(SpanEvent::new("start"));
        span.add_event(SpanEvent::new("end"));
        assert_eq!(span.events.len(), 2);
    }

    #[test]
    fn test_span_finish() {
        let mut span = Span::new(TraceId::from_string("t"), "s");
        assert!(!span.is_finished());
        span.finish();
        assert!(span.is_finished());
        assert_eq!(span.status, SpanStatus::Ok);
        assert!(span.end_time.is_some());
    }

    #[test]
    fn test_span_finish_preserves_error_status() {
        let mut span = Span::new(TraceId::from_string("t"), "s");
        span.status = SpanStatus::Error("fail".into());
        span.finish();
        assert!(matches!(span.status, SpanStatus::Error(_)));
    }

    #[test]
    fn test_span_duration_ms_unfinished() {
        let span = Span::new(TraceId::from_string("t"), "s");
        assert!(span.duration_ms().is_none());
    }

    #[test]
    fn test_span_duration_ms_with_known_timestamps() {
        let mut span = Span::new(TraceId::from_string("t"), "s");
        span.start_time = "2026-01-01T00:00:00.000Z".to_string();
        span.end_time = Some("2026-01-01T00:00:01.500Z".to_string());
        assert_eq!(span.duration_ms(), Some(1500));
    }

    #[test]
    fn test_span_duration_ms_same_time() {
        let mut span = Span::new(TraceId::from_string("t"), "s");
        span.start_time = "2026-06-15T12:00:00.000Z".to_string();
        span.end_time = Some("2026-06-15T12:00:00.000Z".to_string());
        assert_eq!(span.duration_ms(), Some(0));
    }

    #[test]
    fn test_span_to_json() {
        let mut span = Span::new(TraceId::from_string("t1"), "myspan");
        span.set_attribute("k", json!("v"));
        span.add_event(SpanEvent::new("e"));
        let j = span.to_json();
        assert_eq!(j["trace_id"], "t1");
        assert_eq!(j["name"], "myspan");
        assert_eq!(j["attributes"]["k"], "v");
        assert_eq!(j["events"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_span_to_json_parent() {
        let span = Span::new(TraceId::from_string("t"), "s")
            .with_parent(SpanId::from_string("p1"));
        let j = span.to_json();
        assert_eq!(j["parent_id"], "p1");
    }

    #[test]
    fn test_span_to_json_no_parent() {
        let span = Span::new(TraceId::from_string("t"), "s");
        let j = span.to_json();
        assert!(j["parent_id"].is_null());
    }

    // -- Trace tests --

    #[test]
    fn test_trace_new() {
        let trace = Trace::new();
        assert_eq!(trace.span_count(), 0);
        assert!(trace.root_span().is_none());
    }

    #[test]
    fn test_trace_add_span() {
        let mut trace = Trace::new();
        let span = Span::new(trace.trace_id.clone(), "root");
        trace.add_span(span);
        assert_eq!(trace.span_count(), 1);
    }

    #[test]
    fn test_trace_root_span() {
        let mut trace = Trace::new();
        let root = Span::new(trace.trace_id.clone(), "root");
        let child = Span::new(trace.trace_id.clone(), "child")
            .with_parent(root.span_id.clone());
        trace.add_span(root);
        trace.add_span(child);
        assert_eq!(trace.root_span().unwrap().name, "root");
    }

    #[test]
    fn test_trace_get_span() {
        let mut trace = Trace::new();
        let span = Span::new(trace.trace_id.clone(), "lookup");
        let sid = span.span_id.clone();
        trace.add_span(span);
        assert_eq!(trace.get_span(&sid).unwrap().name, "lookup");
    }

    #[test]
    fn test_trace_get_span_not_found() {
        let trace = Trace::new();
        assert!(trace.get_span(&SpanId::from_string("missing")).is_none());
    }

    #[test]
    fn test_trace_children_of() {
        let mut trace = Trace::new();
        let root = Span::new(trace.trace_id.clone(), "root");
        let root_id = root.span_id.clone();
        let c1 = Span::new(trace.trace_id.clone(), "c1").with_parent(root_id.clone());
        let c2 = Span::new(trace.trace_id.clone(), "c2").with_parent(root_id.clone());
        trace.add_span(root);
        trace.add_span(c1);
        trace.add_span(c2);
        let children = trace.children_of(&root_id);
        assert_eq!(children.len(), 2);
    }

    #[test]
    fn test_trace_children_of_empty() {
        let mut trace = Trace::new();
        let span = Span::new(trace.trace_id.clone(), "leaf");
        let sid = span.span_id.clone();
        trace.add_span(span);
        assert!(trace.children_of(&sid).is_empty());
    }

    #[test]
    fn test_trace_total_duration_ms() {
        let mut trace = Trace::new();
        let mut s1 = Span::new(trace.trace_id.clone(), "s1");
        s1.start_time = "2026-01-01T00:00:00.000Z".into();
        s1.end_time = Some("2026-01-01T00:00:02.000Z".into());
        let mut s2 = Span::new(trace.trace_id.clone(), "s2");
        s2.start_time = "2026-01-01T00:00:01.000Z".into();
        s2.end_time = Some("2026-01-01T00:00:03.000Z".into());
        trace.add_span(s1);
        trace.add_span(s2);
        assert_eq!(trace.total_duration_ms(), Some(3000));
    }

    #[test]
    fn test_trace_total_duration_no_finished() {
        let mut trace = Trace::new();
        trace.add_span(Span::new(trace.trace_id.clone(), "unfinished"));
        assert!(trace.total_duration_ms().is_none());
    }

    #[test]
    fn test_trace_to_json() {
        let mut trace = Trace::new();
        trace.add_span(Span::new(trace.trace_id.clone(), "a"));
        let j = trace.to_json();
        assert!(j["trace_id"].is_string());
        assert_eq!(j["spans"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_trace_default() {
        let trace = Trace::default();
        assert_eq!(trace.span_count(), 0);
    }

    // -- TraceCollector tests --

    #[test]
    fn test_collector_new() {
        let c = TraceCollector::new();
        assert!(c.all_traces().is_empty());
    }

    #[test]
    fn test_collector_start_trace() {
        let mut c = TraceCollector::new();
        let tid = c.start_trace("root");
        let trace = c.get_trace(&tid).unwrap();
        assert_eq!(trace.span_count(), 1);
        assert_eq!(trace.root_span().unwrap().name, "root");
    }

    #[test]
    fn test_collector_start_span() {
        let mut c = TraceCollector::new();
        let tid = c.start_trace("root");
        let sid = c.start_span(&tid, "extra").unwrap();
        let trace = c.get_trace(&tid).unwrap();
        assert_eq!(trace.span_count(), 2);
        assert!(trace.get_span(&sid).is_some());
    }

    #[test]
    fn test_collector_start_span_missing_trace() {
        let mut c = TraceCollector::new();
        let fake = TraceId::from_string("nope");
        assert!(c.start_span(&fake, "x").is_none());
    }

    #[test]
    fn test_collector_start_child_span() {
        let mut c = TraceCollector::new();
        let tid = c.start_trace("root");
        let root_sid = c.get_trace(&tid).unwrap().root_span().unwrap().span_id.clone();
        let child_sid = c.start_child_span(&tid, &root_sid, "child").unwrap();
        let trace = c.get_trace(&tid).unwrap();
        let child = trace.get_span(&child_sid).unwrap();
        assert_eq!(child.parent_id.as_ref().unwrap(), &root_sid);
    }

    #[test]
    fn test_collector_start_child_span_missing_trace() {
        let mut c = TraceCollector::new();
        let fake_tid = TraceId::from_string("nope");
        let fake_sid = SpanId::from_string("nope");
        assert!(c.start_child_span(&fake_tid, &fake_sid, "x").is_none());
    }

    #[test]
    fn test_collector_finish_span() {
        let mut c = TraceCollector::new();
        let tid = c.start_trace("root");
        let sid = c.get_trace(&tid).unwrap().root_span().unwrap().span_id.clone();
        c.finish_span(&tid, &sid);
        let span = c.get_trace(&tid).unwrap().get_span(&sid).unwrap();
        assert!(span.is_finished());
        assert_eq!(span.status, SpanStatus::Ok);
    }

    #[test]
    fn test_collector_set_span_status() {
        let mut c = TraceCollector::new();
        let tid = c.start_trace("root");
        let sid = c.get_trace(&tid).unwrap().root_span().unwrap().span_id.clone();
        c.set_span_status(&tid, &sid, SpanStatus::Error("boom".into()));
        let span = c.get_trace(&tid).unwrap().get_span(&sid).unwrap();
        assert!(matches!(span.status, SpanStatus::Error(ref s) if s == "boom"));
    }

    #[test]
    fn test_collector_set_span_attribute() {
        let mut c = TraceCollector::new();
        let tid = c.start_trace("root");
        let sid = c.get_trace(&tid).unwrap().root_span().unwrap().span_id.clone();
        c.set_span_attribute(&tid, &sid, "model", json!("claude"));
        let span = c.get_trace(&tid).unwrap().get_span(&sid).unwrap();
        assert_eq!(span.attributes["model"], json!("claude"));
    }

    #[test]
    fn test_collector_all_traces() {
        let mut c = TraceCollector::new();
        c.start_trace("a");
        c.start_trace("b");
        assert_eq!(c.all_traces().len(), 2);
    }

    #[test]
    fn test_collector_clear() {
        let mut c = TraceCollector::new();
        c.start_trace("a");
        c.clear();
        assert!(c.all_traces().is_empty());
    }

    #[test]
    fn test_collector_get_trace_missing() {
        let c = TraceCollector::new();
        assert!(c.get_trace(&TraceId::from_string("nope")).is_none());
    }

    #[test]
    fn test_collector_default() {
        let c = TraceCollector::default();
        assert!(c.all_traces().is_empty());
    }

    // -- TraceExporter tests --

    #[test]
    fn test_exporter_to_json() {
        let mut trace = Trace::new();
        trace.add_span(Span::new(trace.trace_id.clone(), "root"));
        let j = TraceExporter::to_json(&trace);
        assert!(j["trace_id"].is_string());
    }

    #[test]
    fn test_exporter_to_summary_basic() {
        let mut trace = Trace::new();
        let mut span = Span::new(trace.trace_id.clone(), "root");
        span.start_time = "2026-01-01T00:00:00.000Z".into();
        span.end_time = Some("2026-01-01T00:00:01.000Z".into());
        span.status = SpanStatus::Ok;
        trace.add_span(span);
        let summary = TraceExporter::to_summary(&trace);
        assert!(summary.contains("root"));
        assert!(summary.contains("1000ms"));
        assert!(summary.contains("Spans: 1"));
    }

    #[test]
    fn test_exporter_to_summary_nested() {
        let mut trace = Trace::new();
        let mut root = Span::new(trace.trace_id.clone(), "root");
        root.start_time = "2026-01-01T00:00:00.000Z".into();
        root.end_time = Some("2026-01-01T00:00:02.000Z".into());
        let root_id = root.span_id.clone();
        let mut child = Span::new(trace.trace_id.clone(), "child")
            .with_parent(root_id.clone());
        child.start_time = "2026-01-01T00:00:00.500Z".into();
        child.end_time = Some("2026-01-01T00:00:01.500Z".into());
        trace.add_span(root);
        trace.add_span(child);
        let summary = TraceExporter::to_summary(&trace);
        // child should be indented
        assert!(summary.contains("  child"));
    }

    #[test]
    fn test_exporter_to_flamegraph_data() {
        let mut trace = Trace::new();
        let mut root = Span::new(trace.trace_id.clone(), "root");
        root.start_time = "2026-01-01T00:00:00.000Z".into();
        root.end_time = Some("2026-01-01T00:00:01.000Z".into());
        let root_id = root.span_id.clone();
        let mut child = Span::new(trace.trace_id.clone(), "child")
            .with_parent(root_id);
        child.start_time = "2026-01-01T00:00:00.000Z".into();
        child.end_time = Some("2026-01-01T00:00:00.500Z".into());
        trace.add_span(root);
        trace.add_span(child);

        let entries = TraceExporter::to_flamegraph_data(&trace);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].name, "root");
        assert_eq!(entries[0].depth, 0);
        assert_eq!(entries[0].duration_ms, 1000);
        assert_eq!(entries[1].name, "child");
        assert_eq!(entries[1].depth, 1);
        assert_eq!(entries[1].duration_ms, 500);
    }

    #[test]
    fn test_exporter_flamegraph_unfinished_span() {
        let mut trace = Trace::new();
        trace.add_span(Span::new(trace.trace_id.clone(), "open"));
        let entries = TraceExporter::to_flamegraph_data(&trace);
        assert_eq!(entries[0].duration_ms, 0);
    }

    // -- Timestamp helper tests --

    #[test]
    fn test_parse_timestamp_roundtrip() {
        let ts = current_timestamp();
        assert!(parse_timestamp_ms(&ts).is_some());
    }

    #[test]
    fn test_parse_known_timestamp() {
        let ms = parse_timestamp_ms("2026-01-01T00:00:00.000Z").unwrap();
        // 2026-01-01 = days from 1970 to 2026: known value
        let expected_days: u64 = (1970..2026).map(|y| if is_leap_year(y) { 366u64 } else { 365 }).sum();
        assert_eq!(ms, expected_days * 86400 * 1000);
    }

    #[test]
    fn test_parse_timestamp_with_millis() {
        let ms = parse_timestamp_ms("2026-01-01T00:00:00.123Z").unwrap();
        let base = parse_timestamp_ms("2026-01-01T00:00:00.000Z").unwrap();
        assert_eq!(ms - base, 123);
    }

    #[test]
    fn test_parse_timestamp_invalid() {
        assert!(parse_timestamp_ms("not a time").is_none());
        assert!(parse_timestamp_ms("").is_none());
    }

    #[test]
    fn test_epoch_secs_to_utc_epoch() {
        let (y, m, d, h, min, s) = epoch_secs_to_utc(0);
        assert_eq!((y, m, d, h, min, s), (1970, 1, 1, 0, 0, 0));
    }

    #[test]
    fn test_epoch_secs_to_utc_known_date() {
        // 2026-03-10T00:00:00Z
        let days: u64 = (1970..2026).map(|y| if is_leap_year(y) { 366u64 } else { 365 }).sum();
        // Jan(31) + Feb(28) + 9 days (March 10 = day 69 of year, 0-indexed = 68)
        let day_of_year = 31 + 28 + 9; // = 68 days past Jan 1
        let secs = (days + day_of_year) * 86400;
        let (y, m, d, _, _, _) = epoch_secs_to_utc(secs);
        assert_eq!((y, m, d), (2026, 3, 10));
    }

    // -- FlameEntry tests --

    #[test]
    fn test_flame_entry_serialize() {
        let entry = FlameEntry {
            name: "test".into(),
            depth: 2,
            duration_ms: 100,
        };
        let j = serde_json::to_value(&entry).unwrap();
        assert_eq!(j["name"], "test");
        assert_eq!(j["depth"], 2);
        assert_eq!(j["duration_ms"], 100);
    }

    #[test]
    fn test_flame_entry_clone() {
        let entry = FlameEntry {
            name: "a".into(),
            depth: 0,
            duration_ms: 50,
        };
        let cloned = entry.clone();
        assert_eq!(cloned.name, "a");
    }

    // -- Integration-style tests --

    #[test]
    fn test_full_workflow() {
        let mut collector = TraceCollector::new();
        let tid = collector.start_trace("llm-call");
        let root_sid = collector.get_trace(&tid).unwrap().root_span().unwrap().span_id.clone();

        let child_sid = collector.start_child_span(&tid, &root_sid, "tokenize").unwrap();
        collector.set_span_attribute(&tid, &child_sid, "tokens", json!(128));
        collector.finish_span(&tid, &child_sid);

        let child2_sid = collector.start_child_span(&tid, &root_sid, "generate").unwrap();
        collector.set_span_status(&tid, &child2_sid, SpanStatus::Error("rate_limit".into()));
        collector.finish_span(&tid, &child2_sid);

        collector.finish_span(&tid, &root_sid);

        let trace = collector.get_trace(&tid).unwrap();
        assert_eq!(trace.span_count(), 3);
        assert!(trace.root_span().unwrap().is_finished());

        let children = trace.children_of(&root_sid);
        assert_eq!(children.len(), 2);

        let summary = TraceExporter::to_summary(trace);
        assert!(summary.contains("llm-call"));
        assert!(summary.contains("tokenize"));
        assert!(summary.contains("generate"));
    }

    #[test]
    fn test_multiple_traces() {
        let mut collector = TraceCollector::new();
        let t1 = collector.start_trace("trace-a");
        let t2 = collector.start_trace("trace-b");
        assert_ne!(t1, t2);
        assert_eq!(collector.all_traces().len(), 2);
    }

    #[test]
    fn test_deeply_nested_spans() {
        let mut trace = Trace::new();
        let mut root = Span::new(trace.trace_id.clone(), "level-0");
        root.start_time = "2026-01-01T00:00:00.000Z".into();
        root.end_time = Some("2026-01-01T00:00:05.000Z".into());
        let mut parent_id = root.span_id.clone();
        trace.add_span(root);

        for i in 1..=5 {
            let mut span = Span::new(trace.trace_id.clone(), format!("level-{}", i))
                .with_parent(parent_id.clone());
            span.start_time = "2026-01-01T00:00:00.000Z".into();
            span.end_time = Some("2026-01-01T00:00:05.000Z".into());
            parent_id = span.span_id.clone();
            trace.add_span(span);
        }

        let entries = TraceExporter::to_flamegraph_data(&trace);
        assert_eq!(entries.len(), 6);
        for i in 0..6 {
            assert_eq!(entries[i].depth, i);
            assert_eq!(entries[i].name, format!("level-{}", i));
        }
    }
}
