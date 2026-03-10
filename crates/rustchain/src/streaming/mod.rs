//! Streaming utilities for LLM responses.
//!
//! This module provides types for handling streaming LLM output, including
//! event buffering, tool call accumulation, stream transformation, statistics
//! tracking, and event routing.

use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// StreamEvent
// ---------------------------------------------------------------------------

/// A single event emitted during an LLM streaming response.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A text token produced by the model.
    Token(String),
    /// The start of a tool/function call.
    ToolCallStart {
        /// Tool name.
        name: String,
        /// Unique call id.
        id: String,
    },
    /// A fragment of the serialized arguments for the current tool call.
    ToolCallArg(String),
    /// Signals that the current tool call is complete.
    ToolCallEnd,
    /// Arbitrary key-value metadata attached to the stream.
    Metadata(HashMap<String, Value>),
    /// An error that occurred during streaming.
    Error(String),
    /// The stream has finished.
    Done,
}

impl StreamEvent {
    /// Returns `true` when the event signals the end of the stream or an error.
    pub fn is_terminal(&self) -> bool {
        matches!(self, StreamEvent::Done | StreamEvent::Error(_))
    }

    /// Serializes the event to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        match self {
            StreamEvent::Token(t) => serde_json::json!({
                "type": "token",
                "content": t,
            }),
            StreamEvent::ToolCallStart { name, id } => serde_json::json!({
                "type": "tool_call_start",
                "name": name,
                "id": id,
            }),
            StreamEvent::ToolCallArg(a) => serde_json::json!({
                "type": "tool_call_arg",
                "content": a,
            }),
            StreamEvent::ToolCallEnd => serde_json::json!({
                "type": "tool_call_end",
            }),
            StreamEvent::Metadata(m) => serde_json::json!({
                "type": "metadata",
                "data": m,
            }),
            StreamEvent::Error(e) => serde_json::json!({
                "type": "error",
                "message": e,
            }),
            StreamEvent::Done => serde_json::json!({
                "type": "done",
            }),
        }
    }
}

impl fmt::Display for StreamEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamEvent::Token(t) => write!(f, "{}", t),
            StreamEvent::ToolCallStart { name, id } => {
                write!(f, "[tool_call_start: {}({})]", name, id)
            }
            StreamEvent::ToolCallArg(a) => write!(f, "{}", a),
            StreamEvent::ToolCallEnd => write!(f, "[tool_call_end]"),
            StreamEvent::Metadata(_) => write!(f, "[metadata]"),
            StreamEvent::Error(e) => write!(f, "[error: {}]", e),
            StreamEvent::Done => write!(f, "[done]"),
        }
    }
}

// ---------------------------------------------------------------------------
// ToolCallAccumulator
// ---------------------------------------------------------------------------

/// Accumulates streaming chunks of a single tool call into a complete
/// representation.
#[derive(Debug, Clone)]
pub struct ToolCallAccumulator {
    /// The tool name.
    pub name: String,
    /// The unique call id.
    pub id: String,
    /// Accumulated serialized arguments.
    pub arguments: String,
    /// Whether `ToolCallEnd` has been received.
    pub is_complete: bool,
}

impl ToolCallAccumulator {
    /// Create a new accumulator for the given tool call.
    pub fn new(name: impl Into<String>, id: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            id: id.into(),
            arguments: String::new(),
            is_complete: false,
        }
    }

    /// Append an argument chunk.
    pub fn append_arg(&mut self, chunk: &str) {
        self.arguments.push_str(chunk);
    }

    /// Serialize the accumulated tool call to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "id": self.id,
            "arguments": self.arguments,
            "is_complete": self.is_complete,
        })
    }
}

// ---------------------------------------------------------------------------
// StreamBuffer
// ---------------------------------------------------------------------------

/// Collects [`StreamEvent`]s and provides accessors over the accumulated data.
#[derive(Debug)]
pub struct StreamBuffer {
    events: Vec<StreamEvent>,
    token_buf: String,
    tool_calls: Vec<ToolCallAccumulator>,
    complete: bool,
}

impl StreamBuffer {
    /// Create an empty buffer.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            token_buf: String::new(),
            tool_calls: Vec::new(),
            complete: false,
        }
    }

    /// Push an event into the buffer.
    pub fn push(&mut self, event: StreamEvent) {
        match &event {
            StreamEvent::Token(t) => self.token_buf.push_str(t),
            StreamEvent::ToolCallStart { name, id } => {
                self.tool_calls
                    .push(ToolCallAccumulator::new(name.clone(), id.clone()));
            }
            StreamEvent::ToolCallArg(a) => {
                if let Some(tc) = self.tool_calls.last_mut() {
                    tc.append_arg(a);
                }
            }
            StreamEvent::ToolCallEnd => {
                if let Some(tc) = self.tool_calls.last_mut() {
                    tc.is_complete = true;
                }
            }
            StreamEvent::Done => self.complete = true,
            StreamEvent::Error(_) => self.complete = true,
            StreamEvent::Metadata(_) => {}
        }
        self.events.push(event);
    }

    /// Return all accumulated tokens concatenated into a single string.
    pub fn tokens(&self) -> &str {
        &self.token_buf
    }

    /// Return accumulated tool calls.
    pub fn tool_calls(&self) -> &[ToolCallAccumulator] {
        &self.tool_calls
    }

    /// Number of events received.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Whether the stream has completed (received `Done` or `Error`).
    pub fn is_complete(&self) -> bool {
        self.complete
    }

    /// Reset the buffer to its initial state.
    pub fn clear(&mut self) {
        self.events.clear();
        self.token_buf.clear();
        self.tool_calls.clear();
        self.complete = false;
    }
}

impl Default for StreamBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StreamTransformer trait
// ---------------------------------------------------------------------------

/// Transforms a [`StreamEvent`], optionally filtering it out.
pub trait StreamTransformer {
    /// Transform or filter a single event. Return `None` to drop the event.
    fn transform(&self, event: StreamEvent) -> Option<StreamEvent>;
}

// ---------------------------------------------------------------------------
// FilterTransformer
// ---------------------------------------------------------------------------

/// Filters stream events, keeping only selected event types.
#[derive(Debug, Clone)]
pub struct FilterTransformer {
    keep_tokens: bool,
    keep_tool_calls: bool,
    keep_metadata: bool,
}

impl FilterTransformer {
    /// Create a filter that drops everything by default.
    pub fn new() -> Self {
        Self {
            keep_tokens: false,
            keep_tool_calls: false,
            keep_metadata: false,
        }
    }

    /// Keep `Token` events.
    pub fn keep_tokens(mut self) -> Self {
        self.keep_tokens = true;
        self
    }

    /// Keep `ToolCallStart`, `ToolCallArg`, and `ToolCallEnd` events.
    pub fn keep_tool_calls(mut self) -> Self {
        self.keep_tool_calls = true;
        self
    }

    /// Keep `Metadata` events.
    pub fn keep_metadata(mut self) -> Self {
        self.keep_metadata = true;
        self
    }
}

impl Default for FilterTransformer {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamTransformer for FilterTransformer {
    fn transform(&self, event: StreamEvent) -> Option<StreamEvent> {
        match &event {
            StreamEvent::Token(_) if self.keep_tokens => Some(event),
            StreamEvent::ToolCallStart { .. }
            | StreamEvent::ToolCallArg(_)
            | StreamEvent::ToolCallEnd
                if self.keep_tool_calls =>
            {
                Some(event)
            }
            StreamEvent::Metadata(_) if self.keep_metadata => Some(event),
            // Terminal events are always passed through.
            StreamEvent::Done | StreamEvent::Error(_) => Some(event),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// TokenAggregator
// ---------------------------------------------------------------------------

/// Batches individual token events into larger chunks.
#[derive(Debug)]
pub struct TokenAggregator {
    batch_size: usize,
    buffer: String,
    count: usize,
}

impl TokenAggregator {
    /// Create an aggregator that emits a combined string every `batch_size` tokens.
    pub fn new(batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");
        Self {
            batch_size,
            buffer: String::new(),
            count: 0,
        }
    }

    /// Push a token. Returns `Some(aggregated)` when the batch is full.
    pub fn push(&mut self, token: &str) -> Option<String> {
        self.buffer.push_str(token);
        self.count += 1;
        if self.count >= self.batch_size {
            Some(self.take())
        } else {
            None
        }
    }

    /// Flush any remaining tokens in the buffer.
    pub fn flush(&mut self) -> Option<String> {
        if self.buffer.is_empty() {
            None
        } else {
            Some(self.take())
        }
    }

    fn take(&mut self) -> String {
        self.count = 0;
        std::mem::take(&mut self.buffer)
    }
}

// ---------------------------------------------------------------------------
// StreamStats
// ---------------------------------------------------------------------------

/// Tracks streaming statistics such as token count, throughput, and latency.
#[derive(Debug)]
pub struct StreamStats {
    start: Instant,
    first_token_at: Option<Instant>,
    token_count: usize,
    event_count: usize,
    error_count: usize,
}

impl StreamStats {
    /// Create a new stats tracker. The clock starts now.
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
            first_token_at: None,
            token_count: 0,
            event_count: 0,
            error_count: 0,
        }
    }

    /// Record a single event.
    pub fn record_event(&mut self, event: &StreamEvent) {
        self.event_count += 1;
        match event {
            StreamEvent::Token(_) => {
                if self.first_token_at.is_none() {
                    self.first_token_at = Some(Instant::now());
                }
                self.token_count += 1;
            }
            StreamEvent::Error(_) => {
                self.error_count += 1;
            }
            _ => {}
        }
    }

    /// Tokens received per second (since creation).
    pub fn tokens_per_second(&self) -> f64 {
        let elapsed = self.start.elapsed().as_secs_f64();
        if elapsed == 0.0 {
            return 0.0;
        }
        self.token_count as f64 / elapsed
    }

    /// Total number of token events recorded.
    pub fn total_tokens(&self) -> usize {
        self.token_count
    }

    /// Duration since the stats tracker was created.
    pub fn total_duration(&self) -> Duration {
        self.start.elapsed()
    }

    /// Latency from creation to the first token event, if any.
    pub fn first_token_latency(&self) -> Option<Duration> {
        self.first_token_at.map(|t| t.duration_since(self.start))
    }

    /// Serialize statistics to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "total_tokens": self.token_count,
            "event_count": self.event_count,
            "error_count": self.error_count,
            "total_duration_ms": self.total_duration().as_millis(),
            "tokens_per_second": self.tokens_per_second(),
            "first_token_latency_ms": self.first_token_latency().map(|d| d.as_millis()),
        })
    }
}

impl Default for StreamStats {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StreamRouter
// ---------------------------------------------------------------------------

/// Routes stream events to registered handler closures based on event type.
pub struct StreamRouter {
    token_handlers: Vec<Box<dyn Fn(&str)>>,
    tool_call_handlers: Vec<Box<dyn Fn(&StreamEvent)>>,
    error_handlers: Vec<Box<dyn Fn(&str)>>,
}

impl StreamRouter {
    /// Create an empty router with no handlers.
    pub fn new() -> Self {
        Self {
            token_handlers: Vec::new(),
            tool_call_handlers: Vec::new(),
            error_handlers: Vec::new(),
        }
    }

    /// Register a handler invoked for every `Token` event.
    pub fn on_token(&mut self, handler: impl Fn(&str) + 'static) {
        self.token_handlers.push(Box::new(handler));
    }

    /// Register a handler invoked for `ToolCallStart`, `ToolCallArg`, and
    /// `ToolCallEnd` events.
    pub fn on_tool_call(&mut self, handler: impl Fn(&StreamEvent) + 'static) {
        self.tool_call_handlers.push(Box::new(handler));
    }

    /// Register a handler invoked for `Error` events.
    pub fn on_error(&mut self, handler: impl Fn(&str) + 'static) {
        self.error_handlers.push(Box::new(handler));
    }

    /// Route a single event to the appropriate handlers.
    pub fn route(&self, event: &StreamEvent) {
        match event {
            StreamEvent::Token(t) => {
                for h in &self.token_handlers {
                    h(t);
                }
            }
            StreamEvent::ToolCallStart { .. }
            | StreamEvent::ToolCallArg(_)
            | StreamEvent::ToolCallEnd => {
                for h in &self.tool_call_handlers {
                    h(event);
                }
            }
            StreamEvent::Error(e) => {
                for h in &self.error_handlers {
                    h(e);
                }
            }
            _ => {}
        }
    }
}

impl Default for StreamRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for StreamRouter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StreamRouter")
            .field("token_handlers", &self.token_handlers.len())
            .field("tool_call_handlers", &self.tool_call_handlers.len())
            .field("error_handlers", &self.error_handlers.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    // ---- StreamEvent ----

    #[test]
    fn token_is_not_terminal() {
        let e = StreamEvent::Token("hi".into());
        assert!(!e.is_terminal());
    }

    #[test]
    fn done_is_terminal() {
        assert!(StreamEvent::Done.is_terminal());
    }

    #[test]
    fn error_is_terminal() {
        let e = StreamEvent::Error("boom".into());
        assert!(e.is_terminal());
    }

    #[test]
    fn tool_call_start_is_not_terminal() {
        let e = StreamEvent::ToolCallStart {
            name: "calc".into(),
            id: "1".into(),
        };
        assert!(!e.is_terminal());
    }

    #[test]
    fn tool_call_arg_is_not_terminal() {
        assert!(!StreamEvent::ToolCallArg("arg".into()).is_terminal());
    }

    #[test]
    fn tool_call_end_is_not_terminal() {
        assert!(!StreamEvent::ToolCallEnd.is_terminal());
    }

    #[test]
    fn metadata_is_not_terminal() {
        let e = StreamEvent::Metadata(HashMap::new());
        assert!(!e.is_terminal());
    }

    #[test]
    fn token_to_json() {
        let j = StreamEvent::Token("hello".into()).to_json();
        assert_eq!(j["type"], "token");
        assert_eq!(j["content"], "hello");
    }

    #[test]
    fn tool_call_start_to_json() {
        let j = StreamEvent::ToolCallStart {
            name: "search".into(),
            id: "42".into(),
        }
        .to_json();
        assert_eq!(j["type"], "tool_call_start");
        assert_eq!(j["name"], "search");
        assert_eq!(j["id"], "42");
    }

    #[test]
    fn tool_call_arg_to_json() {
        let j = StreamEvent::ToolCallArg("{\"q\":".into()).to_json();
        assert_eq!(j["type"], "tool_call_arg");
        assert_eq!(j["content"], "{\"q\":");
    }

    #[test]
    fn tool_call_end_to_json() {
        let j = StreamEvent::ToolCallEnd.to_json();
        assert_eq!(j["type"], "tool_call_end");
    }

    #[test]
    fn metadata_to_json() {
        let mut m = HashMap::new();
        m.insert("model".into(), Value::String("gpt-4".into()));
        let j = StreamEvent::Metadata(m).to_json();
        assert_eq!(j["type"], "metadata");
        assert_eq!(j["data"]["model"], "gpt-4");
    }

    #[test]
    fn error_to_json() {
        let j = StreamEvent::Error("timeout".into()).to_json();
        assert_eq!(j["type"], "error");
        assert_eq!(j["message"], "timeout");
    }

    #[test]
    fn done_to_json() {
        let j = StreamEvent::Done.to_json();
        assert_eq!(j["type"], "done");
    }

    #[test]
    fn token_display() {
        assert_eq!(format!("{}", StreamEvent::Token("hi".into())), "hi");
    }

    #[test]
    fn tool_call_start_display() {
        let e = StreamEvent::ToolCallStart {
            name: "calc".into(),
            id: "1".into(),
        };
        assert_eq!(format!("{}", e), "[tool_call_start: calc(1)]");
    }

    #[test]
    fn done_display() {
        assert_eq!(format!("{}", StreamEvent::Done), "[done]");
    }

    #[test]
    fn error_display() {
        assert_eq!(
            format!("{}", StreamEvent::Error("fail".into())),
            "[error: fail]"
        );
    }

    #[test]
    fn metadata_display() {
        let e = StreamEvent::Metadata(HashMap::new());
        assert_eq!(format!("{}", e), "[metadata]");
    }

    #[test]
    fn tool_call_end_display() {
        assert_eq!(format!("{}", StreamEvent::ToolCallEnd), "[tool_call_end]");
    }

    #[test]
    fn tool_call_arg_display() {
        assert_eq!(
            format!("{}", StreamEvent::ToolCallArg("x".into())),
            "x"
        );
    }

    // ---- ToolCallAccumulator ----

    #[test]
    fn accumulator_new() {
        let acc = ToolCallAccumulator::new("search", "tc_1");
        assert_eq!(acc.name, "search");
        assert_eq!(acc.id, "tc_1");
        assert!(acc.arguments.is_empty());
        assert!(!acc.is_complete);
    }

    #[test]
    fn accumulator_append_arg() {
        let mut acc = ToolCallAccumulator::new("f", "1");
        acc.append_arg("{\"a\":");
        acc.append_arg("1}");
        assert_eq!(acc.arguments, "{\"a\":1}");
    }

    #[test]
    fn accumulator_to_json() {
        let mut acc = ToolCallAccumulator::new("search", "tc_1");
        acc.append_arg("{\"q\":\"rust\"}");
        acc.is_complete = true;
        let j = acc.to_json();
        assert_eq!(j["name"], "search");
        assert_eq!(j["id"], "tc_1");
        assert_eq!(j["arguments"], "{\"q\":\"rust\"}");
        assert_eq!(j["is_complete"], true);
    }

    #[test]
    fn accumulator_to_json_incomplete() {
        let acc = ToolCallAccumulator::new("f", "1");
        assert_eq!(acc.to_json()["is_complete"], false);
    }

    // ---- StreamBuffer ----

    #[test]
    fn buffer_new_is_empty() {
        let buf = StreamBuffer::new();
        assert_eq!(buf.tokens(), "");
        assert!(buf.tool_calls().is_empty());
        assert_eq!(buf.event_count(), 0);
        assert!(!buf.is_complete());
    }

    #[test]
    fn buffer_collects_tokens() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::Token("Hello".into()));
        buf.push(StreamEvent::Token(" world".into()));
        assert_eq!(buf.tokens(), "Hello world");
        assert_eq!(buf.event_count(), 2);
    }

    #[test]
    fn buffer_collects_tool_calls() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::ToolCallStart {
            name: "calc".into(),
            id: "1".into(),
        });
        buf.push(StreamEvent::ToolCallArg("{\"expr\":".into()));
        buf.push(StreamEvent::ToolCallArg("\"2+2\"}".into()));
        buf.push(StreamEvent::ToolCallEnd);
        assert_eq!(buf.tool_calls().len(), 1);
        let tc = &buf.tool_calls()[0];
        assert_eq!(tc.name, "calc");
        assert_eq!(tc.arguments, "{\"expr\":\"2+2\"}");
        assert!(tc.is_complete);
    }

    #[test]
    fn buffer_multiple_tool_calls() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::ToolCallStart {
            name: "a".into(),
            id: "1".into(),
        });
        buf.push(StreamEvent::ToolCallEnd);
        buf.push(StreamEvent::ToolCallStart {
            name: "b".into(),
            id: "2".into(),
        });
        buf.push(StreamEvent::ToolCallEnd);
        assert_eq!(buf.tool_calls().len(), 2);
    }

    #[test]
    fn buffer_done_completes() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::Done);
        assert!(buf.is_complete());
    }

    #[test]
    fn buffer_error_completes() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::Error("oops".into()));
        assert!(buf.is_complete());
    }

    #[test]
    fn buffer_clear() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::Token("hi".into()));
        buf.push(StreamEvent::Done);
        buf.clear();
        assert_eq!(buf.tokens(), "");
        assert_eq!(buf.event_count(), 0);
        assert!(!buf.is_complete());
        assert!(buf.tool_calls().is_empty());
    }

    #[test]
    fn buffer_default_trait() {
        let buf = StreamBuffer::default();
        assert_eq!(buf.event_count(), 0);
    }

    #[test]
    fn buffer_tool_call_arg_without_start_ignored() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::ToolCallArg("orphan".into()));
        assert!(buf.tool_calls().is_empty());
        assert_eq!(buf.event_count(), 1);
    }

    #[test]
    fn buffer_metadata_counted() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::Metadata(HashMap::new()));
        assert_eq!(buf.event_count(), 1);
    }

    // ---- FilterTransformer ----

    #[test]
    fn filter_default_drops_tokens() {
        let f = FilterTransformer::new();
        assert!(f.transform(StreamEvent::Token("x".into())).is_none());
    }

    #[test]
    fn filter_keep_tokens() {
        let f = FilterTransformer::new().keep_tokens();
        assert!(f.transform(StreamEvent::Token("x".into())).is_some());
    }

    #[test]
    fn filter_keep_tokens_drops_tool_calls() {
        let f = FilterTransformer::new().keep_tokens();
        assert!(f
            .transform(StreamEvent::ToolCallStart {
                name: "a".into(),
                id: "1".into()
            })
            .is_none());
    }

    #[test]
    fn filter_keep_tool_calls() {
        let f = FilterTransformer::new().keep_tool_calls();
        assert!(f
            .transform(StreamEvent::ToolCallStart {
                name: "a".into(),
                id: "1".into()
            })
            .is_some());
        assert!(f.transform(StreamEvent::ToolCallArg("x".into())).is_some());
        assert!(f.transform(StreamEvent::ToolCallEnd).is_some());
    }

    #[test]
    fn filter_keep_metadata() {
        let f = FilterTransformer::new().keep_metadata();
        assert!(f.transform(StreamEvent::Metadata(HashMap::new())).is_some());
    }

    #[test]
    fn filter_always_passes_done() {
        let f = FilterTransformer::new();
        assert!(f.transform(StreamEvent::Done).is_some());
    }

    #[test]
    fn filter_always_passes_error() {
        let f = FilterTransformer::new();
        assert!(f.transform(StreamEvent::Error("e".into())).is_some());
    }

    #[test]
    fn filter_default_trait() {
        let f = FilterTransformer::default();
        assert!(f.transform(StreamEvent::Token("x".into())).is_none());
    }

    // ---- TokenAggregator ----

    #[test]
    fn aggregator_batches_tokens() {
        let mut agg = TokenAggregator::new(3);
        assert!(agg.push("a").is_none());
        assert!(agg.push("b").is_none());
        let out = agg.push("c");
        assert_eq!(out, Some("abc".into()));
    }

    #[test]
    fn aggregator_flush_remaining() {
        let mut agg = TokenAggregator::new(5);
        agg.push("x");
        agg.push("y");
        assert_eq!(agg.flush(), Some("xy".into()));
    }

    #[test]
    fn aggregator_flush_empty() {
        let mut agg = TokenAggregator::new(3);
        assert_eq!(agg.flush(), None);
    }

    #[test]
    fn aggregator_resets_after_batch() {
        let mut agg = TokenAggregator::new(2);
        agg.push("a");
        agg.push("b"); // emits "ab"
        assert!(agg.push("c").is_none());
        assert_eq!(agg.push("d"), Some("cd".into()));
    }

    #[test]
    #[should_panic(expected = "batch_size must be > 0")]
    fn aggregator_zero_batch_panics() {
        TokenAggregator::new(0);
    }

    #[test]
    fn aggregator_batch_size_one() {
        let mut agg = TokenAggregator::new(1);
        assert_eq!(agg.push("hello"), Some("hello".into()));
    }

    // ---- StreamStats ----

    #[test]
    fn stats_initial_state() {
        let s = StreamStats::new();
        assert_eq!(s.total_tokens(), 0);
        assert!(s.first_token_latency().is_none());
    }

    #[test]
    fn stats_counts_tokens() {
        let mut s = StreamStats::new();
        s.record_event(&StreamEvent::Token("a".into()));
        s.record_event(&StreamEvent::Token("b".into()));
        assert_eq!(s.total_tokens(), 2);
    }

    #[test]
    fn stats_first_token_latency_set() {
        let mut s = StreamStats::new();
        s.record_event(&StreamEvent::Token("a".into()));
        assert!(s.first_token_latency().is_some());
    }

    #[test]
    fn stats_first_token_latency_not_reset() {
        let mut s = StreamStats::new();
        s.record_event(&StreamEvent::Token("a".into()));
        let first = s.first_token_latency().unwrap();
        s.record_event(&StreamEvent::Token("b".into()));
        assert_eq!(s.first_token_latency().unwrap(), first);
    }

    #[test]
    fn stats_total_duration_positive() {
        let s = StreamStats::new();
        // Duration should be >= 0 (just instantiated).
        assert!(s.total_duration().as_nanos() < 1_000_000_000);
    }

    #[test]
    fn stats_to_json_fields() {
        let mut s = StreamStats::new();
        s.record_event(&StreamEvent::Token("a".into()));
        s.record_event(&StreamEvent::Error("e".into()));
        let j = s.to_json();
        assert_eq!(j["total_tokens"], 1);
        assert_eq!(j["event_count"], 2);
        assert_eq!(j["error_count"], 1);
        assert!(j["first_token_latency_ms"].is_number());
    }

    #[test]
    fn stats_default_trait() {
        let s = StreamStats::default();
        assert_eq!(s.total_tokens(), 0);
    }

    #[test]
    fn stats_non_token_events_do_not_count_as_tokens() {
        let mut s = StreamStats::new();
        s.record_event(&StreamEvent::Done);
        s.record_event(&StreamEvent::Metadata(HashMap::new()));
        assert_eq!(s.total_tokens(), 0);
        assert!(s.first_token_latency().is_none());
    }

    #[test]
    fn stats_tokens_per_second_zero_when_no_tokens() {
        let s = StreamStats::new();
        // Should not panic, returns 0.0
        let _ = s.tokens_per_second();
    }

    // ---- StreamRouter ----

    #[test]
    fn router_routes_tokens() {
        let collected = Rc::new(RefCell::new(Vec::new()));
        let c = collected.clone();
        let mut router = StreamRouter::new();
        router.on_token(move |t| c.borrow_mut().push(t.to_string()));
        router.route(&StreamEvent::Token("hi".into()));
        assert_eq!(*collected.borrow(), vec!["hi"]);
    }

    #[test]
    fn router_routes_tool_calls() {
        let count = Rc::new(RefCell::new(0usize));
        let c = count.clone();
        let mut router = StreamRouter::new();
        router.on_tool_call(move |_| *c.borrow_mut() += 1);
        router.route(&StreamEvent::ToolCallStart {
            name: "a".into(),
            id: "1".into(),
        });
        router.route(&StreamEvent::ToolCallArg("x".into()));
        router.route(&StreamEvent::ToolCallEnd);
        assert_eq!(*count.borrow(), 3);
    }

    #[test]
    fn router_routes_errors() {
        let msgs = Rc::new(RefCell::new(Vec::new()));
        let m = msgs.clone();
        let mut router = StreamRouter::new();
        router.on_error(move |e| m.borrow_mut().push(e.to_string()));
        router.route(&StreamEvent::Error("fail".into()));
        assert_eq!(*msgs.borrow(), vec!["fail"]);
    }

    #[test]
    fn router_ignores_unhandled_events() {
        let router = StreamRouter::new();
        // Should not panic.
        router.route(&StreamEvent::Done);
        router.route(&StreamEvent::Metadata(HashMap::new()));
    }

    #[test]
    fn router_multiple_handlers() {
        let count = Rc::new(RefCell::new(0usize));
        let c1 = count.clone();
        let c2 = count.clone();
        let mut router = StreamRouter::new();
        router.on_token(move |_| *c1.borrow_mut() += 1);
        router.on_token(move |_| *c2.borrow_mut() += 1);
        router.route(&StreamEvent::Token("t".into()));
        assert_eq!(*count.borrow(), 2);
    }

    #[test]
    fn router_default_trait() {
        let r = StreamRouter::default();
        // Empty router should not panic on route.
        r.route(&StreamEvent::Token("t".into()));
    }

    #[test]
    fn router_debug() {
        let r = StreamRouter::new();
        let d = format!("{:?}", r);
        assert!(d.contains("StreamRouter"));
    }
}
