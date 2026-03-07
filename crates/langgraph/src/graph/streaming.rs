//! Streaming protocol for graph execution.
//!
//! Provides real-time output as nodes process, including Server-Sent Events (SSE)
//! formatting and state aggregation. This module complements the lower-level
//! [`super::stream_writer`] channel-based streaming with a higher-level protocol
//! that tracks node lifecycle events, edge traversals, and execution state.

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// StreamMode
// ---------------------------------------------------------------------------

/// The streaming mode that controls which events are emitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamMode {
    /// Stream the full state values after each node completes.
    Values,
    /// Stream only the incremental updates produced by each node.
    Updates,
    /// Stream debug-level events (node start/end, edge traversals, etc.).
    Debug,
    /// A custom, user-defined stream mode.
    Custom(String),
}

impl fmt::Display for StreamMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamMode::Values => write!(f, "values"),
            StreamMode::Updates => write!(f, "updates"),
            StreamMode::Debug => write!(f, "debug"),
            StreamMode::Custom(name) => write!(f, "custom:{}", name),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamEventType
// ---------------------------------------------------------------------------

/// The type of a streaming event emitted during graph execution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEventType {
    /// A node is about to start processing.
    NodeStart,
    /// A node has produced output.
    NodeOutput,
    /// A node has finished processing.
    NodeEnd,
    /// The execution engine is traversing an edge.
    EdgeTraversal,
    /// The graph state has been updated.
    StateUpdate,
    /// An error occurred during execution.
    Error,
    /// The graph execution has completed.
    Complete,
    /// A keep-alive heartbeat signal.
    Heartbeat,
}

impl fmt::Display for StreamEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StreamEventType::NodeStart => write!(f, "node_start"),
            StreamEventType::NodeOutput => write!(f, "node_output"),
            StreamEventType::NodeEnd => write!(f, "node_end"),
            StreamEventType::EdgeTraversal => write!(f, "edge_traversal"),
            StreamEventType::StateUpdate => write!(f, "state_update"),
            StreamEventType::Error => write!(f, "error"),
            StreamEventType::Complete => write!(f, "complete"),
            StreamEventType::Heartbeat => write!(f, "heartbeat"),
        }
    }
}

// ---------------------------------------------------------------------------
// StreamEvent
// ---------------------------------------------------------------------------

/// A single event emitted during streaming graph execution.
#[derive(Debug, Clone)]
pub struct StreamEvent {
    /// The kind of event.
    pub event_type: StreamEventType,
    /// The node associated with this event, if any.
    pub node: Option<String>,
    /// The data payload.
    pub data: Value,
    /// When the event was created.
    pub timestamp: Instant,
    /// Monotonically increasing sequence number.
    pub sequence: u64,
}

impl StreamEvent {
    /// Create a new event with the given type and data. The timestamp is set to
    /// [`Instant::now()`] and the sequence number defaults to 0.
    pub fn new(event_type: StreamEventType, data: Value) -> Self {
        Self {
            event_type,
            node: None,
            data,
            timestamp: Instant::now(),
            sequence: 0,
        }
    }

    /// Builder method: set the node associated with this event.
    pub fn with_node(mut self, node: &str) -> Self {
        self.node = Some(node.to_string());
        self
    }

    /// Serialize this event to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "event": self.event_type.to_string(),
            "node": self.node,
            "data": self.data,
            "sequence": self.sequence,
        })
    }

    /// Format this event as a Server-Sent Events (SSE) message.
    pub fn to_sse(&self) -> String {
        SSEFormatter::format(self)
    }
}

// ---------------------------------------------------------------------------
// StreamBuffer
// ---------------------------------------------------------------------------

/// A buffer that collects [`StreamEvent`] values with an optional capacity limit.
#[derive(Debug)]
pub struct StreamBuffer {
    events: Vec<StreamEvent>,
    capacity: Option<usize>,
}

impl StreamBuffer {
    /// Create a new buffer with no capacity limit.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            capacity: None,
        }
    }

    /// Create a new buffer that holds at most `n` events.
    /// When at capacity, the oldest event is removed on push.
    pub fn with_capacity(n: usize) -> Self {
        Self {
            events: Vec::with_capacity(n),
            capacity: Some(n),
        }
    }

    /// Push an event into the buffer. If a capacity limit is set and the buffer
    /// is full, the oldest event is dropped.
    pub fn push(&mut self, event: StreamEvent) {
        if let Some(cap) = self.capacity {
            if self.events.len() >= cap {
                self.events.remove(0);
            }
        }
        self.events.push(event);
    }

    /// Remove and return all events in the buffer.
    pub fn drain(&mut self) -> Vec<StreamEvent> {
        std::mem::take(&mut self.events)
    }

    /// Peek at the most recently pushed event without removing it.
    pub fn peek(&self) -> Option<&StreamEvent> {
        self.events.last()
    }

    /// Return a slice of all buffered events.
    pub fn events(&self) -> &[StreamEvent] {
        &self.events
    }

    /// The number of events currently in the buffer.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Remove all events from the buffer.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Return references to events matching the given event type.
    pub fn filter_by_type(&self, event_type: &StreamEventType) -> Vec<&StreamEvent> {
        self.events
            .iter()
            .filter(|e| &e.event_type == event_type)
            .collect()
    }

    /// Return references to events associated with the given node.
    pub fn filter_by_node(&self, node: &str) -> Vec<&StreamEvent> {
        self.events
            .iter()
            .filter(|e| e.node.as_deref() == Some(node))
            .collect()
    }
}

impl Default for StreamBuffer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StreamConfig
// ---------------------------------------------------------------------------

/// Configuration for a streaming graph execution session.
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// The streaming mode.
    pub mode: StreamMode,
    /// Whether heartbeat events should be included.
    pub include_heartbeats: bool,
    /// Optional maximum buffer size.
    pub buffer_size: Option<usize>,
    /// Whether debug events should be included.
    pub include_debug: bool,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            mode: StreamMode::Values,
            include_heartbeats: false,
            buffer_size: None,
            include_debug: false,
        }
    }
}

impl StreamConfig {
    /// Builder: set the stream mode.
    pub fn with_mode(mut self, mode: StreamMode) -> Self {
        self.mode = mode;
        self
    }

    /// Builder: enable or disable heartbeats.
    pub fn with_heartbeats(mut self, enabled: bool) -> Self {
        self.include_heartbeats = enabled;
        self
    }

    /// Builder: set the buffer size limit.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = Some(size);
        self
    }

    /// Builder: enable or disable debug events.
    pub fn with_debug(mut self, enabled: bool) -> Self {
        self.include_debug = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// StreamController
// ---------------------------------------------------------------------------

/// Manages streaming state for a single graph execution, emitting events into
/// an internal [`StreamBuffer`] and filtering based on [`StreamConfig`].
pub struct StreamController {
    config: StreamConfig,
    buffer: StreamBuffer,
    sequence: u64,
    complete: bool,
}

impl StreamController {
    /// Create a new controller with the given configuration.
    pub fn new(config: StreamConfig) -> Self {
        let buffer = match config.buffer_size {
            Some(size) => StreamBuffer::with_capacity(size),
            None => StreamBuffer::new(),
        };
        Self {
            config,
            buffer,
            sequence: 0,
            complete: false,
        }
    }

    /// Emit a raw event. The event's sequence number is set automatically.
    /// Events that do not pass the current config filters are silently dropped.
    pub fn emit(&mut self, mut event: StreamEvent) {
        // Filter heartbeats if disabled.
        if event.event_type == StreamEventType::Heartbeat && !self.config.include_heartbeats {
            return;
        }
        // Filter debug-style events (NodeStart, NodeEnd, EdgeTraversal) unless debug is on.
        if !self.config.include_debug {
            match event.event_type {
                StreamEventType::NodeStart
                | StreamEventType::NodeEnd
                | StreamEventType::EdgeTraversal => return,
                _ => {}
            }
        }
        event.sequence = self.sequence;
        self.sequence += 1;
        if event.event_type == StreamEventType::Complete {
            self.complete = true;
        }
        self.buffer.push(event);
    }

    /// Emit a [`StreamEventType::NodeStart`] event.
    pub fn emit_node_start(&mut self, node: &str, state: &Value) {
        let event = StreamEvent::new(StreamEventType::NodeStart, state.clone()).with_node(node);
        self.emit(event);
    }

    /// Emit a [`StreamEventType::NodeOutput`] event.
    pub fn emit_node_output(&mut self, node: &str, output: &Value) {
        let event = StreamEvent::new(StreamEventType::NodeOutput, output.clone()).with_node(node);
        self.emit(event);
    }

    /// Emit a [`StreamEventType::NodeEnd`] event.
    pub fn emit_node_end(&mut self, node: &str) {
        let event = StreamEvent::new(StreamEventType::NodeEnd, Value::Null).with_node(node);
        self.emit(event);
    }

    /// Emit a [`StreamEventType::EdgeTraversal`] event.
    pub fn emit_edge(&mut self, from: &str, to: &str) {
        let data = json!({"from": from, "to": to});
        let event = StreamEvent::new(StreamEventType::EdgeTraversal, data);
        self.emit(event);
    }

    /// Emit a [`StreamEventType::Error`] event.
    pub fn emit_error(&mut self, error: &str) {
        let data = json!({"error": error});
        let event = StreamEvent::new(StreamEventType::Error, data);
        self.emit(event);
    }

    /// Emit a [`StreamEventType::Complete`] event.
    pub fn emit_complete(&mut self) {
        let event = StreamEvent::new(StreamEventType::Complete, Value::Null);
        self.emit(event);
    }

    /// Emit a [`StreamEventType::Heartbeat`] event.
    pub fn emit_heartbeat(&mut self) {
        let event = StreamEvent::new(StreamEventType::Heartbeat, Value::Null);
        self.emit(event);
    }

    /// Drain all buffered events.
    pub fn drain(&mut self) -> Vec<StreamEvent> {
        self.buffer.drain()
    }

    /// Access the underlying buffer.
    pub fn buffer(&self) -> &StreamBuffer {
        &self.buffer
    }

    /// The total number of events that have been assigned a sequence number.
    pub fn event_count(&self) -> u64 {
        self.sequence
    }

    /// Whether a [`StreamEventType::Complete`] event has been emitted.
    pub fn is_complete(&self) -> bool {
        self.complete
    }
}

// ---------------------------------------------------------------------------
// SSEFormatter
// ---------------------------------------------------------------------------

/// Formats [`StreamEvent`] values as
/// [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events).
pub struct SSEFormatter;

impl SSEFormatter {
    /// Format a single event as an SSE message.
    pub fn format(event: &StreamEvent) -> String {
        let data_str = serde_json::to_string(&event.to_json()).unwrap_or_default();
        format!(
            "event: {}\ndata: {}\nid: {}\n\n",
            event.event_type, data_str, event.sequence
        )
    }

    /// Format a batch of events as consecutive SSE messages.
    pub fn format_batch(events: &[StreamEvent]) -> String {
        let mut out = String::new();
        for event in events {
            out.push_str(&Self::format(event));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// StreamAggregator
// ---------------------------------------------------------------------------

/// Accumulates streaming events into a final aggregated state.
///
/// Tracks the latest output per node, the overall merged state, and the order
/// in which nodes executed.
pub struct StreamAggregator {
    state: Value,
    outputs: HashMap<String, Value>,
    order: Vec<String>,
}

impl StreamAggregator {
    /// Create a new, empty aggregator.
    pub fn new() -> Self {
        Self {
            state: json!({}),
            outputs: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Process a single event, updating the internal state as appropriate.
    ///
    /// - [`StreamEventType::NodeOutput`] and [`StreamEventType::StateUpdate`]
    ///   merge the event data into the current state and record the node output.
    /// - [`StreamEventType::NodeEnd`] records the execution order.
    pub fn process(&mut self, event: &StreamEvent) {
        match event.event_type {
            StreamEventType::NodeOutput | StreamEventType::StateUpdate => {
                // Merge data into state
                if let Value::Object(map) = &event.data {
                    if let Value::Object(state_map) = &mut self.state {
                        for (k, v) in map {
                            state_map.insert(k.clone(), v.clone());
                        }
                    }
                }
                // Record per-node output
                if let Some(ref node) = event.node {
                    self.outputs.insert(node.clone(), event.data.clone());
                }
            }
            StreamEventType::NodeEnd => {
                if let Some(ref node) = event.node {
                    if !self.order.contains(node) {
                        self.order.push(node.clone());
                    }
                }
            }
            _ => {}
        }
    }

    /// The current aggregated state.
    pub fn current_state(&self) -> &Value {
        &self.state
    }

    /// The latest output value for each node that produced output.
    pub fn node_outputs(&self) -> HashMap<String, Value> {
        self.outputs.clone()
    }

    /// The order in which nodes completed execution.
    pub fn execution_order(&self) -> Vec<String> {
        self.order.clone()
    }
}

impl Default for StreamAggregator {
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
    use serde_json::json;

    // -- StreamEvent creation -------------------------------------------------

    #[test]
    fn test_stream_event_new_defaults() {
        let event = StreamEvent::new(StreamEventType::NodeOutput, json!({"x": 1}));
        assert_eq!(event.event_type, StreamEventType::NodeOutput);
        assert!(event.node.is_none());
        assert_eq!(event.data, json!({"x": 1}));
        assert_eq!(event.sequence, 0);
    }

    #[test]
    fn test_stream_event_with_node() {
        let event = StreamEvent::new(StreamEventType::NodeStart, json!(null)).with_node("agent");
        assert_eq!(event.node.as_deref(), Some("agent"));
    }

    #[test]
    fn test_stream_event_to_json() {
        let mut event =
            StreamEvent::new(StreamEventType::NodeOutput, json!({"val": 42})).with_node("n1");
        event.sequence = 5;
        let j = event.to_json();
        assert_eq!(j["event"], "node_output");
        assert_eq!(j["node"], "n1");
        assert_eq!(j["data"]["val"], 42);
        assert_eq!(j["sequence"], 5);
    }

    #[test]
    fn test_stream_event_to_sse() {
        let mut event = StreamEvent::new(StreamEventType::Heartbeat, json!(null));
        event.sequence = 3;
        let sse = event.to_sse();
        assert!(sse.starts_with("event: heartbeat\n"));
        assert!(sse.contains("id: 3"));
        assert!(sse.ends_with("\n\n"));
    }

    // -- StreamMode Display ---------------------------------------------------

    #[test]
    fn test_stream_mode_display() {
        assert_eq!(StreamMode::Values.to_string(), "values");
        assert_eq!(StreamMode::Updates.to_string(), "updates");
        assert_eq!(StreamMode::Debug.to_string(), "debug");
        assert_eq!(
            StreamMode::Custom("my_mode".into()).to_string(),
            "custom:my_mode"
        );
    }

    // -- StreamEventType Display ----------------------------------------------

    #[test]
    fn test_stream_event_type_display() {
        assert_eq!(StreamEventType::NodeStart.to_string(), "node_start");
        assert_eq!(StreamEventType::NodeOutput.to_string(), "node_output");
        assert_eq!(StreamEventType::NodeEnd.to_string(), "node_end");
        assert_eq!(StreamEventType::EdgeTraversal.to_string(), "edge_traversal");
        assert_eq!(StreamEventType::StateUpdate.to_string(), "state_update");
        assert_eq!(StreamEventType::Error.to_string(), "error");
        assert_eq!(StreamEventType::Complete.to_string(), "complete");
        assert_eq!(StreamEventType::Heartbeat.to_string(), "heartbeat");
    }

    // -- StreamBuffer ---------------------------------------------------------

    #[test]
    fn test_buffer_new_is_empty() {
        let buf = StreamBuffer::new();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert!(buf.peek().is_none());
    }

    #[test]
    fn test_buffer_push_and_len() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::new(StreamEventType::Heartbeat, json!(null)));
        buf.push(StreamEvent::new(StreamEventType::Complete, json!(null)));
        assert_eq!(buf.len(), 2);
        assert!(!buf.is_empty());
    }

    #[test]
    fn test_buffer_peek_returns_last() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::new(StreamEventType::NodeStart, json!("first")));
        buf.push(StreamEvent::new(StreamEventType::NodeEnd, json!("second")));
        let peeked = buf.peek().unwrap();
        assert_eq!(peeked.event_type, StreamEventType::NodeEnd);
    }

    #[test]
    fn test_buffer_drain() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::new(StreamEventType::Heartbeat, json!(null)));
        buf.push(StreamEvent::new(StreamEventType::Heartbeat, json!(null)));
        let drained = buf.drain();
        assert_eq!(drained.len(), 2);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_clear() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::new(StreamEventType::Heartbeat, json!(null)));
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_with_capacity_evicts_oldest() {
        let mut buf = StreamBuffer::with_capacity(2);
        buf.push(
            StreamEvent::new(StreamEventType::NodeStart, json!("a")).with_node("first"),
        );
        buf.push(
            StreamEvent::new(StreamEventType::NodeOutput, json!("b")).with_node("second"),
        );
        buf.push(
            StreamEvent::new(StreamEventType::NodeEnd, json!("c")).with_node("third"),
        );
        assert_eq!(buf.len(), 2);
        // The first event should have been evicted.
        assert_eq!(buf.events()[0].node.as_deref(), Some("second"));
        assert_eq!(buf.events()[1].node.as_deref(), Some("third"));
    }

    #[test]
    fn test_buffer_filter_by_type() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::new(StreamEventType::NodeStart, json!(null)).with_node("a"));
        buf.push(StreamEvent::new(StreamEventType::NodeOutput, json!(1)).with_node("a"));
        buf.push(StreamEvent::new(StreamEventType::NodeStart, json!(null)).with_node("b"));
        let starts = buf.filter_by_type(&StreamEventType::NodeStart);
        assert_eq!(starts.len(), 2);
    }

    #[test]
    fn test_buffer_filter_by_node() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::new(StreamEventType::NodeOutput, json!(1)).with_node("a"));
        buf.push(StreamEvent::new(StreamEventType::NodeOutput, json!(2)).with_node("b"));
        buf.push(StreamEvent::new(StreamEventType::NodeEnd, json!(null)).with_node("a"));
        let node_a = buf.filter_by_node("a");
        assert_eq!(node_a.len(), 2);
    }

    #[test]
    fn test_buffer_events_returns_slice() {
        let mut buf = StreamBuffer::new();
        buf.push(StreamEvent::new(StreamEventType::Complete, json!(null)));
        assert_eq!(buf.events().len(), 1);
    }

    // -- StreamConfig ---------------------------------------------------------

    #[test]
    fn test_stream_config_default() {
        let cfg = StreamConfig::default();
        assert_eq!(cfg.mode, StreamMode::Values);
        assert!(!cfg.include_heartbeats);
        assert!(!cfg.include_debug);
        assert!(cfg.buffer_size.is_none());
    }

    #[test]
    fn test_stream_config_builder() {
        let cfg = StreamConfig::default()
            .with_mode(StreamMode::Debug)
            .with_heartbeats(true)
            .with_buffer_size(100)
            .with_debug(true);
        assert_eq!(cfg.mode, StreamMode::Debug);
        assert!(cfg.include_heartbeats);
        assert_eq!(cfg.buffer_size, Some(100));
        assert!(cfg.include_debug);
    }

    // -- StreamController -----------------------------------------------------

    #[test]
    fn test_controller_new_empty() {
        let ctrl = StreamController::new(StreamConfig::default());
        assert_eq!(ctrl.event_count(), 0);
        assert!(!ctrl.is_complete());
        assert!(ctrl.buffer().is_empty());
    }

    #[test]
    fn test_controller_emit_node_output() {
        let cfg = StreamConfig::default();
        let mut ctrl = StreamController::new(cfg);
        ctrl.emit_node_output("agent", &json!({"answer": 42}));
        assert_eq!(ctrl.event_count(), 1);
        let events = ctrl.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, StreamEventType::NodeOutput);
        assert_eq!(events[0].node.as_deref(), Some("agent"));
    }

    #[test]
    fn test_controller_filters_debug_events_by_default() {
        let cfg = StreamConfig::default(); // include_debug = false
        let mut ctrl = StreamController::new(cfg);
        ctrl.emit_node_start("n", &json!({}));
        ctrl.emit_node_end("n");
        ctrl.emit_edge("a", "b");
        // None of the debug events should have been buffered.
        assert_eq!(ctrl.event_count(), 0);
        assert!(ctrl.buffer().is_empty());
    }

    #[test]
    fn test_controller_includes_debug_events_when_enabled() {
        let cfg = StreamConfig::default().with_debug(true);
        let mut ctrl = StreamController::new(cfg);
        ctrl.emit_node_start("n", &json!({}));
        ctrl.emit_node_end("n");
        ctrl.emit_edge("a", "b");
        assert_eq!(ctrl.event_count(), 3);
    }

    #[test]
    fn test_controller_heartbeat_filtered_by_default() {
        let cfg = StreamConfig::default();
        let mut ctrl = StreamController::new(cfg);
        ctrl.emit_heartbeat();
        assert_eq!(ctrl.event_count(), 0);
    }

    #[test]
    fn test_controller_heartbeat_included_when_enabled() {
        let cfg = StreamConfig::default().with_heartbeats(true);
        let mut ctrl = StreamController::new(cfg);
        ctrl.emit_heartbeat();
        assert_eq!(ctrl.event_count(), 1);
        assert_eq!(
            ctrl.buffer().events()[0].event_type,
            StreamEventType::Heartbeat
        );
    }

    #[test]
    fn test_controller_emit_error() {
        let cfg = StreamConfig::default();
        let mut ctrl = StreamController::new(cfg);
        ctrl.emit_error("something went wrong");
        let events = ctrl.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event_type, StreamEventType::Error);
        assert_eq!(events[0].data["error"], "something went wrong");
    }

    #[test]
    fn test_controller_emit_complete() {
        let cfg = StreamConfig::default();
        let mut ctrl = StreamController::new(cfg);
        assert!(!ctrl.is_complete());
        ctrl.emit_complete();
        assert!(ctrl.is_complete());
    }

    #[test]
    fn test_controller_full_lifecycle() {
        let cfg = StreamConfig::default().with_debug(true).with_heartbeats(true);
        let mut ctrl = StreamController::new(cfg);

        ctrl.emit_node_start("agent", &json!({"input": "hello"}));
        ctrl.emit_heartbeat();
        ctrl.emit_node_output("agent", &json!({"response": "world"}));
        ctrl.emit_node_end("agent");
        ctrl.emit_edge("agent", "tool");
        ctrl.emit_node_start("tool", &json!({"query": "search"}));
        ctrl.emit_node_output("tool", &json!({"result": "found"}));
        ctrl.emit_node_end("tool");
        ctrl.emit_complete();

        assert_eq!(ctrl.event_count(), 9);
        assert!(ctrl.is_complete());

        let events = ctrl.drain();
        assert_eq!(events.len(), 9);
        assert_eq!(events[0].sequence, 0);
        assert_eq!(events[8].sequence, 8);
    }

    #[test]
    fn test_controller_drain_empties_buffer() {
        let cfg = StreamConfig::default();
        let mut ctrl = StreamController::new(cfg);
        ctrl.emit_node_output("n", &json!(1));
        assert_eq!(ctrl.drain().len(), 1);
        assert!(ctrl.buffer().is_empty());
    }

    #[test]
    fn test_controller_complete_without_events() {
        let cfg = StreamConfig::default();
        let mut ctrl = StreamController::new(cfg);
        ctrl.emit_complete();
        assert!(ctrl.is_complete());
        assert_eq!(ctrl.event_count(), 1);
    }

    // -- SSEFormatter ---------------------------------------------------------

    #[test]
    fn test_sse_formatter_format() {
        let mut event =
            StreamEvent::new(StreamEventType::NodeOutput, json!({"x": 1})).with_node("n1");
        event.sequence = 7;
        let sse = SSEFormatter::format(&event);
        assert!(sse.starts_with("event: node_output\n"));
        assert!(sse.contains("data: "));
        assert!(sse.contains("id: 7\n"));
        assert!(sse.ends_with("\n\n"));
    }

    #[test]
    fn test_sse_formatter_format_batch() {
        let mut e1 = StreamEvent::new(StreamEventType::NodeOutput, json!(1));
        e1.sequence = 0;
        let mut e2 = StreamEvent::new(StreamEventType::Complete, json!(null));
        e2.sequence = 1;
        let batch = SSEFormatter::format_batch(&[e1, e2]);
        // Two events, each terminated by double newline.
        assert_eq!(batch.matches("\n\n").count(), 2);
    }

    #[test]
    fn test_sse_formatter_empty_batch() {
        let batch = SSEFormatter::format_batch(&[]);
        assert!(batch.is_empty());
    }

    // -- StreamAggregator -----------------------------------------------------

    #[test]
    fn test_aggregator_new_empty_state() {
        let agg = StreamAggregator::new();
        assert_eq!(agg.current_state(), &json!({}));
        assert!(agg.node_outputs().is_empty());
        assert!(agg.execution_order().is_empty());
    }

    #[test]
    fn test_aggregator_accumulates_state() {
        let mut agg = StreamAggregator::new();
        let e1 = StreamEvent::new(StreamEventType::NodeOutput, json!({"a": 1})).with_node("n1");
        let e2 = StreamEvent::new(StreamEventType::NodeOutput, json!({"b": 2})).with_node("n2");
        agg.process(&e1);
        agg.process(&e2);
        assert_eq!(agg.current_state(), &json!({"a": 1, "b": 2}));
    }

    #[test]
    fn test_aggregator_overwrites_same_key() {
        let mut agg = StreamAggregator::new();
        let e1 = StreamEvent::new(StreamEventType::NodeOutput, json!({"x": 1})).with_node("n1");
        let e2 = StreamEvent::new(StreamEventType::NodeOutput, json!({"x": 99})).with_node("n2");
        agg.process(&e1);
        agg.process(&e2);
        assert_eq!(agg.current_state(), &json!({"x": 99}));
    }

    #[test]
    fn test_aggregator_node_outputs() {
        let mut agg = StreamAggregator::new();
        let e1 = StreamEvent::new(StreamEventType::NodeOutput, json!({"r": "hello"})).with_node("agent");
        agg.process(&e1);
        let outputs = agg.node_outputs();
        assert_eq!(outputs["agent"], json!({"r": "hello"}));
    }

    #[test]
    fn test_aggregator_execution_order() {
        let mut agg = StreamAggregator::new();
        let end_a = StreamEvent::new(StreamEventType::NodeEnd, json!(null)).with_node("a");
        let end_b = StreamEvent::new(StreamEventType::NodeEnd, json!(null)).with_node("b");
        let end_c = StreamEvent::new(StreamEventType::NodeEnd, json!(null)).with_node("c");
        agg.process(&end_a);
        agg.process(&end_b);
        agg.process(&end_c);
        assert_eq!(agg.execution_order(), vec!["a", "b", "c"]);
    }

    #[test]
    fn test_aggregator_no_duplicate_order() {
        let mut agg = StreamAggregator::new();
        let end1 = StreamEvent::new(StreamEventType::NodeEnd, json!(null)).with_node("a");
        let end2 = StreamEvent::new(StreamEventType::NodeEnd, json!(null)).with_node("a");
        agg.process(&end1);
        agg.process(&end2);
        assert_eq!(agg.execution_order(), vec!["a"]);
    }

    #[test]
    fn test_aggregator_ignores_irrelevant_events() {
        let mut agg = StreamAggregator::new();
        let hb = StreamEvent::new(StreamEventType::Heartbeat, json!(null));
        let err = StreamEvent::new(StreamEventType::Error, json!({"error": "oops"}));
        agg.process(&hb);
        agg.process(&err);
        assert_eq!(agg.current_state(), &json!({}));
        assert!(agg.execution_order().is_empty());
    }

    #[test]
    fn test_aggregator_state_update_event() {
        let mut agg = StreamAggregator::new();
        let su = StreamEvent::new(StreamEventType::StateUpdate, json!({"count": 5})).with_node("counter");
        agg.process(&su);
        assert_eq!(agg.current_state(), &json!({"count": 5}));
        assert_eq!(agg.node_outputs()["counter"], json!({"count": 5}));
    }
}
