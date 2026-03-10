//! Streaming execution for LangGraph.
//!
//! This module allows graph nodes to emit partial results during execution,
//! providing real-time visibility into graph processing through chunk-based
//! streaming with collection, filtering, replay, subscription, aggregation,
//! and metrics capabilities.

use std::collections::HashMap;
use std::fmt;

use serde_json::{json, Value};

// ---------------------------------------------------------------------------
// StreamMode
// ---------------------------------------------------------------------------

/// Controls which category of streaming output is emitted.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StreamMode {
    /// Stream full state values after each node completes.
    Values,
    /// Stream only incremental updates produced by each node.
    Updates,
    /// Stream lifecycle events (node start/end, edge traversals, etc.).
    Events,
    /// Stream chat-style messages.
    Messages,
    /// A custom, user-defined stream mode.
    Custom(String),
}

impl StreamMode {
    /// Return the canonical name of this mode.
    pub fn name(&self) -> &str {
        match self {
            StreamMode::Values => "values",
            StreamMode::Updates => "updates",
            StreamMode::Events => "events",
            StreamMode::Messages => "messages",
            StreamMode::Custom(n) => n.as_str(),
        }
    }

    /// Serialize this mode to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({ "mode": self.name() })
    }
}

impl fmt::Display for StreamMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// ChunkType
// ---------------------------------------------------------------------------

/// The kind of streaming chunk emitted during graph execution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ChunkType {
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
}

impl ChunkType {
    /// Return the canonical name of this chunk type.
    pub fn name(&self) -> &str {
        match self {
            ChunkType::NodeStart => "node_start",
            ChunkType::NodeOutput => "node_output",
            ChunkType::NodeEnd => "node_end",
            ChunkType::EdgeTraversal => "edge_traversal",
            ChunkType::StateUpdate => "state_update",
            ChunkType::Error => "error",
        }
    }
}

impl fmt::Display for ChunkType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// StreamChunk
// ---------------------------------------------------------------------------

/// A single piece of streaming output from graph execution.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// The node that produced this chunk.
    pub node_id: String,
    /// The kind of chunk.
    pub chunk_type: ChunkType,
    /// The data payload.
    pub data: Value,
    /// Monotonically increasing sequence number (assigned by the collector).
    pub sequence: u64,
    /// ISO-8601 timestamp string.
    pub timestamp: String,
}

impl StreamChunk {
    /// Create a new chunk. The sequence is initialised to 0 and the timestamp
    /// is set to a placeholder; callers (e.g. [`StreamCollector`]) may overwrite
    /// these fields.
    pub fn new(node_id: impl Into<String>, chunk_type: ChunkType, data: Value) -> Self {
        Self {
            node_id: node_id.into(),
            chunk_type,
            data,
            sequence: 0,
            timestamp: Self::now_iso8601(),
        }
    }

    /// Serialize this chunk to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "node_id": self.node_id,
            "chunk_type": self.chunk_type.name(),
            "data": self.data,
            "sequence": self.sequence,
            "timestamp": self.timestamp,
        })
    }

    /// Return a simple ISO-8601–ish timestamp using std only.
    fn now_iso8601() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let dur = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let secs = dur.as_secs();
        format!("1970-01-01T00:00:00+{secs:010}Z")
    }
}

// ---------------------------------------------------------------------------
// StreamCollector
// ---------------------------------------------------------------------------

/// Collects [`StreamChunk`] values emitted during graph execution.
#[derive(Debug)]
pub struct StreamCollector {
    chunks: Vec<StreamChunk>,
    next_seq: u64,
}

impl StreamCollector {
    /// Create a new, empty collector.
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            next_seq: 0,
        }
    }

    /// Push a chunk into the collector. The chunk's `sequence` field is set
    /// to the next monotonically increasing value.
    pub fn push(&mut self, mut chunk: StreamChunk) {
        chunk.sequence = self.next_seq;
        self.next_seq += 1;
        self.chunks.push(chunk);
    }

    /// Return a slice of all collected chunks.
    pub fn chunks(&self) -> &[StreamChunk] {
        &self.chunks
    }

    /// Return references to chunks produced by the given node.
    pub fn chunks_for_node(&self, node_id: &str) -> Vec<&StreamChunk> {
        self.chunks
            .iter()
            .filter(|c| c.node_id == node_id)
            .collect()
    }

    /// Return the most recent [`ChunkType::StateUpdate`] chunk, if any.
    pub fn latest_state(&self) -> Option<&StreamChunk> {
        self.chunks
            .iter()
            .rev()
            .find(|c| c.chunk_type == ChunkType::StateUpdate)
    }

    /// The number of collected chunks.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// Whether the collector is empty.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Remove all collected chunks.
    pub fn clear(&mut self) {
        self.chunks.clear();
        self.next_seq = 0;
    }

    /// Return a sorted, deduplicated list of node IDs that have produced chunks.
    pub fn node_ids(&self) -> Vec<String> {
        let mut ids: Vec<String> = self.chunks.iter().map(|c| c.node_id.clone()).collect();
        ids.sort();
        ids.dedup();
        ids
    }

    /// Return references to all error chunks.
    pub fn errors(&self) -> Vec<&StreamChunk> {
        self.chunks
            .iter()
            .filter(|c| c.chunk_type == ChunkType::Error)
            .collect()
    }
}

impl Default for StreamCollector {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StreamFilter
// ---------------------------------------------------------------------------

/// Filters a slice of [`StreamChunk`] values by node and/or type criteria.
#[derive(Debug, Clone)]
pub struct StreamFilter {
    node_ids: Vec<String>,
    include_types: Vec<ChunkType>,
    exclude_types: Vec<ChunkType>,
}

impl StreamFilter {
    /// Create a new filter with no criteria (passes everything).
    pub fn new() -> Self {
        Self {
            node_ids: Vec::new(),
            include_types: Vec::new(),
            exclude_types: Vec::new(),
        }
    }

    /// Restrict results to chunks from the given node.
    pub fn by_node(mut self, node_id: impl Into<String>) -> Self {
        self.node_ids.push(node_id.into());
        self
    }

    /// Restrict results to chunks of the given type.
    pub fn by_type(mut self, chunk_type: ChunkType) -> Self {
        self.include_types.push(chunk_type);
        self
    }

    /// Exclude chunks of the given type.
    pub fn exclude_type(mut self, chunk_type: ChunkType) -> Self {
        self.exclude_types.push(chunk_type);
        self
    }

    /// Apply this filter to a slice of chunks.
    pub fn apply<'a>(&self, chunks: &'a [StreamChunk]) -> Vec<&'a StreamChunk> {
        chunks
            .iter()
            .filter(|c| {
                if !self.node_ids.is_empty() && !self.node_ids.contains(&c.node_id) {
                    return false;
                }
                if !self.include_types.is_empty() && !self.include_types.contains(&c.chunk_type) {
                    return false;
                }
                if self.exclude_types.contains(&c.chunk_type) {
                    return false;
                }
                true
            })
            .collect()
    }
}

impl Default for StreamFilter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StreamReplay
// ---------------------------------------------------------------------------

/// Replays a recorded stream of chunks in order.
#[derive(Debug)]
pub struct StreamReplay {
    chunks: Vec<StreamChunk>,
    position: usize,
}

impl StreamReplay {
    /// Create a new replay from a vector of chunks.
    pub fn new(chunks: Vec<StreamChunk>) -> Self {
        Self {
            chunks,
            position: 0,
        }
    }

    /// Advance and return the next chunk, or `None` if exhausted.
    pub fn next_chunk(&mut self) -> Option<&StreamChunk> {
        if self.position < self.chunks.len() {
            let chunk = &self.chunks[self.position];
            self.position += 1;
            Some(chunk)
        } else {
            None
        }
    }

    /// Peek at the next chunk without advancing.
    pub fn peek(&self) -> Option<&StreamChunk> {
        self.chunks.get(self.position)
    }

    /// The number of remaining chunks.
    pub fn remaining(&self) -> usize {
        self.chunks.len().saturating_sub(self.position)
    }

    /// Reset the replay to the beginning.
    pub fn reset(&mut self) {
        self.position = 0;
    }

    /// Skip forward until the next chunk from the given node is reached.
    /// If no such chunk exists ahead, the position is unchanged.
    pub fn skip_to_node(&mut self, node_id: &str) {
        for i in self.position..self.chunks.len() {
            if self.chunks[i].node_id == node_id {
                self.position = i;
                return;
            }
        }
    }

    /// The current position in the replay (0-based).
    pub fn current_position(&self) -> usize {
        self.position
    }
}

// ---------------------------------------------------------------------------
// StreamSubscription
// ---------------------------------------------------------------------------

/// A subscription to specific stream events by node and/or chunk type.
#[derive(Debug, Clone)]
pub struct StreamSubscription {
    nodes: Vec<String>,
    types: Vec<ChunkType>,
}

impl StreamSubscription {
    /// Create a new subscription with no filters (nothing subscribed).
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            types: Vec::new(),
        }
    }

    /// Subscribe to chunks from a specific node.
    pub fn subscribe_node(mut self, node_id: impl Into<String>) -> Self {
        self.nodes.push(node_id.into());
        self
    }

    /// Subscribe to chunks of a specific type.
    pub fn subscribe_type(mut self, chunk_type: ChunkType) -> Self {
        self.types.push(chunk_type);
        self
    }

    /// Check whether a chunk matches this subscription.
    ///
    /// A chunk is considered subscribed if:
    /// - It matches at least one subscribed node **or** at least one subscribed type.
    /// - If no nodes and no types have been subscribed, nothing matches.
    pub fn is_subscribed(&self, chunk: &StreamChunk) -> bool {
        if self.nodes.is_empty() && self.types.is_empty() {
            return false;
        }
        let node_match = self.nodes.is_empty() || self.nodes.contains(&chunk.node_id);
        let type_match = self.types.is_empty() || self.types.contains(&chunk.chunk_type);
        node_match && type_match
    }

    /// Return the list of subscribed node IDs.
    pub fn subscribed_nodes(&self) -> &[String] {
        &self.nodes
    }

    /// Return the list of subscribed chunk types.
    pub fn subscribed_types(&self) -> &[ChunkType] {
        &self.types
    }
}

impl Default for StreamSubscription {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StreamAggregator
// ---------------------------------------------------------------------------

/// Aggregates streaming output into final results.
///
/// Tracks per-node outputs, the merged state, and execution order.
pub struct StreamAggregator {
    outputs: HashMap<String, Vec<Value>>,
    state: Option<Value>,
    order: Vec<String>,
}

impl StreamAggregator {
    /// Create a new, empty aggregator.
    pub fn new() -> Self {
        Self {
            outputs: HashMap::new(),
            state: None,
            order: Vec::new(),
        }
    }

    /// Process a single chunk, updating the internal state.
    ///
    /// - `NodeOutput` / `StateUpdate` — record the data.
    /// - `NodeStart` — record execution order.
    pub fn process(&mut self, chunk: &StreamChunk) {
        match chunk.chunk_type {
            ChunkType::NodeOutput => {
                self.outputs
                    .entry(chunk.node_id.clone())
                    .or_default()
                    .push(chunk.data.clone());
            }
            ChunkType::StateUpdate => {
                self.state = Some(chunk.data.clone());
            }
            ChunkType::NodeStart => {
                if !self.order.contains(&chunk.node_id) {
                    self.order.push(chunk.node_id.clone());
                }
            }
            _ => {}
        }
    }

    /// Return per-node output values.
    pub fn node_outputs(&self) -> HashMap<String, Vec<Value>> {
        self.outputs.clone()
    }

    /// Return the latest state update value.
    pub fn final_state(&self) -> Option<Value> {
        self.state.clone()
    }

    /// Return the order in which nodes started execution.
    pub fn execution_order(&self) -> Vec<String> {
        self.order.clone()
    }

    /// Serialize the aggregated results to JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "node_outputs": self.outputs,
            "final_state": self.state,
            "execution_order": self.order,
        })
    }
}

impl Default for StreamAggregator {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// StreamMetrics
// ---------------------------------------------------------------------------

/// Metrics about streaming execution.
pub struct StreamMetrics {
    chunks_per_node: HashMap<String, usize>,
    node_start_seq: HashMap<String, u64>,
    node_durations: HashMap<String, u64>,
    total: usize,
    first_seq: Option<u64>,
    last_seq: Option<u64>,
}

impl StreamMetrics {
    /// Create a new, empty metrics tracker.
    pub fn new() -> Self {
        Self {
            chunks_per_node: HashMap::new(),
            node_start_seq: HashMap::new(),
            node_durations: HashMap::new(),
            total: 0,
            first_seq: None,
            last_seq: None,
        }
    }

    /// Record a chunk, updating internal metrics.
    pub fn record(&mut self, chunk: &StreamChunk) {
        self.total += 1;
        *self
            .chunks_per_node
            .entry(chunk.node_id.clone())
            .or_insert(0) += 1;

        if self.first_seq.is_none() {
            self.first_seq = Some(chunk.sequence);
        }
        self.last_seq = Some(chunk.sequence);

        match chunk.chunk_type {
            ChunkType::NodeStart => {
                self.node_start_seq
                    .insert(chunk.node_id.clone(), chunk.sequence);
            }
            ChunkType::NodeEnd => {
                if let Some(start) = self.node_start_seq.get(&chunk.node_id) {
                    let dur = chunk.sequence.saturating_sub(*start);
                    self.node_durations.insert(chunk.node_id.clone(), dur);
                }
            }
            _ => {}
        }
    }

    /// Return the number of chunks recorded per node.
    pub fn chunks_per_node(&self) -> HashMap<String, usize> {
        self.chunks_per_node.clone()
    }

    /// Return the duration (in sequence-number ticks) for each node that
    /// had both a `NodeStart` and `NodeEnd` event.
    pub fn node_durations(&self) -> HashMap<String, u64> {
        self.node_durations.clone()
    }

    /// The total number of chunks recorded.
    pub fn total_chunks(&self) -> usize {
        self.total
    }

    /// The span between first and last sequence numbers (a proxy for total
    /// duration in sequence ticks).
    pub fn total_duration_ms(&self) -> u64 {
        match (self.first_seq, self.last_seq) {
            (Some(first), Some(last)) => last.saturating_sub(first),
            _ => 0,
        }
    }

    /// Serialize the metrics to JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "total_chunks": self.total,
            "chunks_per_node": self.chunks_per_node,
            "node_durations": self.node_durations,
            "total_duration_ms": self.total_duration_ms(),
        })
    }
}

impl Default for StreamMetrics {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- helpers --

    fn chunk(node: &str, ct: ChunkType, data: Value) -> StreamChunk {
        StreamChunk::new(node, ct, data)
    }

    fn sample_chunks() -> Vec<StreamChunk> {
        let mut collector = StreamCollector::new();
        collector.push(chunk("a", ChunkType::NodeStart, json!({})));
        collector.push(chunk("a", ChunkType::NodeOutput, json!({"x": 1})));
        collector.push(chunk("a", ChunkType::NodeEnd, json!({})));
        collector.push(chunk("b", ChunkType::NodeStart, json!({})));
        collector.push(chunk("b", ChunkType::NodeOutput, json!({"y": 2})));
        collector.push(chunk("b", ChunkType::StateUpdate, json!({"state": "ok"})));
        collector.push(chunk("b", ChunkType::NodeEnd, json!({})));
        collector.push(chunk("c", ChunkType::Error, json!({"err": "fail"})));
        collector.chunks.clone()
    }

    // -----------------------------------------------------------------------
    // StreamMode
    // -----------------------------------------------------------------------

    #[test]
    fn test_stream_mode_name() {
        assert_eq!(StreamMode::Values.name(), "values");
        assert_eq!(StreamMode::Updates.name(), "updates");
        assert_eq!(StreamMode::Events.name(), "events");
        assert_eq!(StreamMode::Messages.name(), "messages");
        assert_eq!(StreamMode::Custom("xyz".into()).name(), "xyz");
    }

    #[test]
    fn test_stream_mode_display() {
        assert_eq!(StreamMode::Values.to_string(), "values");
        assert_eq!(StreamMode::Custom("my".into()).to_string(), "my");
    }

    #[test]
    fn test_stream_mode_to_json() {
        let j = StreamMode::Events.to_json();
        assert_eq!(j["mode"], "events");
    }

    #[test]
    fn test_stream_mode_equality() {
        assert_eq!(StreamMode::Values, StreamMode::Values);
        assert_ne!(StreamMode::Values, StreamMode::Updates);
        assert_eq!(
            StreamMode::Custom("a".into()),
            StreamMode::Custom("a".into())
        );
        assert_ne!(
            StreamMode::Custom("a".into()),
            StreamMode::Custom("b".into())
        );
    }

    // -----------------------------------------------------------------------
    // ChunkType
    // -----------------------------------------------------------------------

    #[test]
    fn test_chunk_type_name() {
        assert_eq!(ChunkType::NodeStart.name(), "node_start");
        assert_eq!(ChunkType::NodeOutput.name(), "node_output");
        assert_eq!(ChunkType::NodeEnd.name(), "node_end");
        assert_eq!(ChunkType::EdgeTraversal.name(), "edge_traversal");
        assert_eq!(ChunkType::StateUpdate.name(), "state_update");
        assert_eq!(ChunkType::Error.name(), "error");
    }

    #[test]
    fn test_chunk_type_display() {
        assert_eq!(ChunkType::NodeStart.to_string(), "node_start");
        assert_eq!(ChunkType::Error.to_string(), "error");
    }

    // -----------------------------------------------------------------------
    // StreamChunk
    // -----------------------------------------------------------------------

    #[test]
    fn test_stream_chunk_new() {
        let c = StreamChunk::new("node1", ChunkType::NodeOutput, json!(42));
        assert_eq!(c.node_id, "node1");
        assert_eq!(c.chunk_type, ChunkType::NodeOutput);
        assert_eq!(c.data, json!(42));
        assert_eq!(c.sequence, 0);
        assert!(!c.timestamp.is_empty());
    }

    #[test]
    fn test_stream_chunk_to_json() {
        let mut c = StreamChunk::new("n", ChunkType::Error, json!("oops"));
        c.sequence = 7;
        let j = c.to_json();
        assert_eq!(j["node_id"], "n");
        assert_eq!(j["chunk_type"], "error");
        assert_eq!(j["data"], "oops");
        assert_eq!(j["sequence"], 7);
        assert!(j["timestamp"].is_string());
    }

    // -----------------------------------------------------------------------
    // StreamCollector
    // -----------------------------------------------------------------------

    #[test]
    fn test_collector_new_is_empty() {
        let c = StreamCollector::new();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn test_collector_push_assigns_sequence() {
        let mut c = StreamCollector::new();
        c.push(chunk("a", ChunkType::NodeStart, json!({})));
        c.push(chunk("b", ChunkType::NodeEnd, json!({})));
        assert_eq!(c.len(), 2);
        assert_eq!(c.chunks()[0].sequence, 0);
        assert_eq!(c.chunks()[1].sequence, 1);
    }

    #[test]
    fn test_collector_chunks_for_node() {
        let mut c = StreamCollector::new();
        c.push(chunk("a", ChunkType::NodeStart, json!({})));
        c.push(chunk("b", ChunkType::NodeOutput, json!(1)));
        c.push(chunk("a", ChunkType::NodeEnd, json!({})));
        assert_eq!(c.chunks_for_node("a").len(), 2);
        assert_eq!(c.chunks_for_node("b").len(), 1);
        assert_eq!(c.chunks_for_node("z").len(), 0);
    }

    #[test]
    fn test_collector_latest_state() {
        let mut c = StreamCollector::new();
        c.push(chunk("a", ChunkType::StateUpdate, json!({"v": 1})));
        c.push(chunk("a", ChunkType::NodeOutput, json!({})));
        c.push(chunk("b", ChunkType::StateUpdate, json!({"v": 2})));
        let s = c.latest_state().unwrap();
        assert_eq!(s.data, json!({"v": 2}));
    }

    #[test]
    fn test_collector_latest_state_none() {
        let c = StreamCollector::new();
        assert!(c.latest_state().is_none());
    }

    #[test]
    fn test_collector_clear() {
        let mut c = StreamCollector::new();
        c.push(chunk("a", ChunkType::NodeStart, json!({})));
        c.clear();
        assert!(c.is_empty());
        // Sequence resets.
        c.push(chunk("b", ChunkType::NodeStart, json!({})));
        assert_eq!(c.chunks()[0].sequence, 0);
    }

    #[test]
    fn test_collector_node_ids() {
        let mut c = StreamCollector::new();
        c.push(chunk("b", ChunkType::NodeStart, json!({})));
        c.push(chunk("a", ChunkType::NodeStart, json!({})));
        c.push(chunk("b", ChunkType::NodeEnd, json!({})));
        let ids = c.node_ids();
        assert_eq!(ids, vec!["a", "b"]);
    }

    #[test]
    fn test_collector_errors() {
        let mut c = StreamCollector::new();
        c.push(chunk("a", ChunkType::NodeOutput, json!(1)));
        c.push(chunk("a", ChunkType::Error, json!("e1")));
        c.push(chunk("b", ChunkType::Error, json!("e2")));
        assert_eq!(c.errors().len(), 2);
    }

    #[test]
    fn test_collector_errors_empty() {
        let c = StreamCollector::new();
        assert!(c.errors().is_empty());
    }

    #[test]
    fn test_collector_default() {
        let c = StreamCollector::default();
        assert!(c.is_empty());
    }

    // -----------------------------------------------------------------------
    // StreamFilter
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_no_criteria_passes_all() {
        let chunks = sample_chunks();
        let f = StreamFilter::new();
        assert_eq!(f.apply(&chunks).len(), chunks.len());
    }

    #[test]
    fn test_filter_by_node() {
        let chunks = sample_chunks();
        let f = StreamFilter::new().by_node("a");
        let result = f.apply(&chunks);
        assert!(result.iter().all(|c| c.node_id == "a"));
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_filter_by_type() {
        let chunks = sample_chunks();
        let f = StreamFilter::new().by_type(ChunkType::NodeOutput);
        let result = f.apply(&chunks);
        assert!(result.iter().all(|c| c.chunk_type == ChunkType::NodeOutput));
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_filter_exclude_type() {
        let chunks = sample_chunks();
        let f = StreamFilter::new().exclude_type(ChunkType::Error);
        let result = f.apply(&chunks);
        assert!(result.iter().all(|c| c.chunk_type != ChunkType::Error));
        assert_eq!(result.len(), 7);
    }

    #[test]
    fn test_filter_by_node_and_type() {
        let chunks = sample_chunks();
        let f = StreamFilter::new()
            .by_node("b")
            .by_type(ChunkType::NodeOutput);
        let result = f.apply(&chunks);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].node_id, "b");
    }

    #[test]
    fn test_filter_multiple_nodes() {
        let chunks = sample_chunks();
        let f = StreamFilter::new().by_node("a").by_node("c");
        let result = f.apply(&chunks);
        assert_eq!(result.len(), 4); // 3 from a + 1 from c
    }

    #[test]
    fn test_filter_exclude_multiple_types() {
        let chunks = sample_chunks();
        let f = StreamFilter::new()
            .exclude_type(ChunkType::NodeStart)
            .exclude_type(ChunkType::NodeEnd);
        let result = f.apply(&chunks);
        assert_eq!(result.len(), 4); // 2 outputs + 1 state + 1 error
    }

    #[test]
    fn test_filter_default() {
        let chunks = sample_chunks();
        let f = StreamFilter::default();
        assert_eq!(f.apply(&chunks).len(), chunks.len());
    }

    // -----------------------------------------------------------------------
    // StreamReplay
    // -----------------------------------------------------------------------

    #[test]
    fn test_replay_basic() {
        let chunks = vec![
            chunk("a", ChunkType::NodeStart, json!({})),
            chunk("a", ChunkType::NodeEnd, json!({})),
        ];
        let mut r = StreamReplay::new(chunks);
        assert_eq!(r.remaining(), 2);
        assert_eq!(r.current_position(), 0);

        let c1 = r.next_chunk().unwrap();
        assert_eq!(c1.node_id, "a");
        assert_eq!(c1.chunk_type, ChunkType::NodeStart);

        let c2 = r.next_chunk().unwrap();
        assert_eq!(c2.chunk_type, ChunkType::NodeEnd);

        assert!(r.next_chunk().is_none());
        assert_eq!(r.remaining(), 0);
    }

    #[test]
    fn test_replay_peek() {
        let chunks = vec![chunk("x", ChunkType::NodeOutput, json!(1))];
        let r = StreamReplay::new(chunks);
        let p = r.peek().unwrap();
        assert_eq!(p.node_id, "x");
        // Peek does not advance.
        assert_eq!(r.remaining(), 1);
    }

    #[test]
    fn test_replay_reset() {
        let chunks = vec![
            chunk("a", ChunkType::NodeStart, json!({})),
            chunk("b", ChunkType::NodeStart, json!({})),
        ];
        let mut r = StreamReplay::new(chunks);
        r.next_chunk();
        r.next_chunk();
        assert_eq!(r.remaining(), 0);
        r.reset();
        assert_eq!(r.remaining(), 2);
        assert_eq!(r.current_position(), 0);
    }

    #[test]
    fn test_replay_skip_to_node() {
        let chunks = vec![
            chunk("a", ChunkType::NodeStart, json!({})),
            chunk("a", ChunkType::NodeEnd, json!({})),
            chunk("b", ChunkType::NodeStart, json!({})),
            chunk("b", ChunkType::NodeEnd, json!({})),
        ];
        let mut r = StreamReplay::new(chunks);
        r.skip_to_node("b");
        assert_eq!(r.current_position(), 2);
        assert_eq!(r.peek().unwrap().node_id, "b");
    }

    #[test]
    fn test_replay_skip_to_node_not_found() {
        let chunks = vec![chunk("a", ChunkType::NodeStart, json!({}))];
        let mut r = StreamReplay::new(chunks);
        r.skip_to_node("z");
        // Position unchanged.
        assert_eq!(r.current_position(), 0);
    }

    #[test]
    fn test_replay_empty() {
        let mut r = StreamReplay::new(vec![]);
        assert!(r.next_chunk().is_none());
        assert!(r.peek().is_none());
        assert_eq!(r.remaining(), 0);
    }

    // -----------------------------------------------------------------------
    // StreamSubscription
    // -----------------------------------------------------------------------

    #[test]
    fn test_subscription_empty_matches_nothing() {
        let sub = StreamSubscription::new();
        let c = chunk("a", ChunkType::NodeOutput, json!(1));
        assert!(!sub.is_subscribed(&c));
    }

    #[test]
    fn test_subscription_by_node() {
        let sub = StreamSubscription::new().subscribe_node("agent");
        let c1 = chunk("agent", ChunkType::NodeOutput, json!(1));
        let c2 = chunk("tool", ChunkType::NodeOutput, json!(2));
        assert!(sub.is_subscribed(&c1));
        assert!(!sub.is_subscribed(&c2));
    }

    #[test]
    fn test_subscription_by_type() {
        let sub = StreamSubscription::new().subscribe_type(ChunkType::Error);
        let c1 = chunk("a", ChunkType::Error, json!("e"));
        let c2 = chunk("a", ChunkType::NodeOutput, json!(1));
        assert!(sub.is_subscribed(&c1));
        assert!(!sub.is_subscribed(&c2));
    }

    #[test]
    fn test_subscription_by_node_and_type() {
        let sub = StreamSubscription::new()
            .subscribe_node("agent")
            .subscribe_type(ChunkType::NodeOutput);
        let c1 = chunk("agent", ChunkType::NodeOutput, json!(1));
        let c2 = chunk("agent", ChunkType::NodeStart, json!({}));
        let c3 = chunk("tool", ChunkType::NodeOutput, json!(2));
        assert!(sub.is_subscribed(&c1)); // matches both
        assert!(!sub.is_subscribed(&c2)); // wrong type
        assert!(!sub.is_subscribed(&c3)); // wrong node
    }

    #[test]
    fn test_subscription_subscribed_nodes() {
        let sub = StreamSubscription::new()
            .subscribe_node("a")
            .subscribe_node("b");
        assert_eq!(sub.subscribed_nodes(), &["a", "b"]);
    }

    #[test]
    fn test_subscription_subscribed_types() {
        let sub = StreamSubscription::new()
            .subscribe_type(ChunkType::NodeStart)
            .subscribe_type(ChunkType::NodeEnd);
        assert_eq!(sub.subscribed_types().len(), 2);
    }

    #[test]
    fn test_subscription_default() {
        let sub = StreamSubscription::default();
        assert!(sub.subscribed_nodes().is_empty());
        assert!(sub.subscribed_types().is_empty());
    }

    // -----------------------------------------------------------------------
    // StreamAggregator
    // -----------------------------------------------------------------------

    #[test]
    fn test_aggregator_new() {
        let a = StreamAggregator::new();
        assert!(a.node_outputs().is_empty());
        assert!(a.final_state().is_none());
        assert!(a.execution_order().is_empty());
    }

    #[test]
    fn test_aggregator_process_node_output() {
        let mut a = StreamAggregator::new();
        a.process(&chunk("n1", ChunkType::NodeOutput, json!({"a": 1})));
        a.process(&chunk("n1", ChunkType::NodeOutput, json!({"b": 2})));
        let outputs = a.node_outputs();
        assert_eq!(outputs["n1"].len(), 2);
    }

    #[test]
    fn test_aggregator_process_state_update() {
        let mut a = StreamAggregator::new();
        a.process(&chunk("n1", ChunkType::StateUpdate, json!({"s": 1})));
        a.process(&chunk("n2", ChunkType::StateUpdate, json!({"s": 2})));
        assert_eq!(a.final_state(), Some(json!({"s": 2})));
    }

    #[test]
    fn test_aggregator_execution_order() {
        let mut a = StreamAggregator::new();
        a.process(&chunk("b", ChunkType::NodeStart, json!({})));
        a.process(&chunk("a", ChunkType::NodeStart, json!({})));
        a.process(&chunk("b", ChunkType::NodeStart, json!({}))); // duplicate
        assert_eq!(a.execution_order(), vec!["b", "a"]);
    }

    #[test]
    fn test_aggregator_to_json() {
        let mut a = StreamAggregator::new();
        a.process(&chunk("n", ChunkType::NodeOutput, json!(1)));
        a.process(&chunk("n", ChunkType::NodeStart, json!({})));
        let j = a.to_json();
        assert!(j["node_outputs"].is_object());
        assert!(j["execution_order"].is_array());
    }

    #[test]
    fn test_aggregator_default() {
        let a = StreamAggregator::default();
        assert!(a.final_state().is_none());
    }

    #[test]
    fn test_aggregator_ignores_irrelevant_types() {
        let mut a = StreamAggregator::new();
        a.process(&chunk("n", ChunkType::EdgeTraversal, json!({})));
        a.process(&chunk("n", ChunkType::Error, json!("e")));
        a.process(&chunk("n", ChunkType::NodeEnd, json!({})));
        assert!(a.node_outputs().is_empty());
        assert!(a.final_state().is_none());
        assert!(a.execution_order().is_empty());
    }

    // -----------------------------------------------------------------------
    // StreamMetrics
    // -----------------------------------------------------------------------

    #[test]
    fn test_metrics_new() {
        let m = StreamMetrics::new();
        assert_eq!(m.total_chunks(), 0);
        assert_eq!(m.total_duration_ms(), 0);
        assert!(m.chunks_per_node().is_empty());
        assert!(m.node_durations().is_empty());
    }

    #[test]
    fn test_metrics_record_counts() {
        let mut m = StreamMetrics::new();
        let chunks = sample_chunks();
        for c in &chunks {
            m.record(c);
        }
        assert_eq!(m.total_chunks(), 8);
        assert_eq!(m.chunks_per_node()["a"], 3);
        assert_eq!(m.chunks_per_node()["b"], 4);
        assert_eq!(m.chunks_per_node()["c"], 1);
    }

    #[test]
    fn test_metrics_node_durations() {
        let mut m = StreamMetrics::new();
        let mut c1 = chunk("n", ChunkType::NodeStart, json!({}));
        c1.sequence = 10;
        let mut c2 = chunk("n", ChunkType::NodeOutput, json!(1));
        c2.sequence = 11;
        let mut c3 = chunk("n", ChunkType::NodeEnd, json!({}));
        c3.sequence = 15;

        m.record(&c1);
        m.record(&c2);
        m.record(&c3);

        let durations = m.node_durations();
        assert_eq!(durations["n"], 5); // 15 - 10
    }

    #[test]
    fn test_metrics_total_duration() {
        let mut m = StreamMetrics::new();
        let mut c1 = chunk("a", ChunkType::NodeStart, json!({}));
        c1.sequence = 5;
        let mut c2 = chunk("b", ChunkType::NodeEnd, json!({}));
        c2.sequence = 25;
        m.record(&c1);
        m.record(&c2);
        assert_eq!(m.total_duration_ms(), 20);
    }

    #[test]
    fn test_metrics_to_json() {
        let mut m = StreamMetrics::new();
        m.record(&chunk("a", ChunkType::NodeOutput, json!(1)));
        let j = m.to_json();
        assert_eq!(j["total_chunks"], 1);
        assert!(j["chunks_per_node"].is_object());
    }

    #[test]
    fn test_metrics_default() {
        let m = StreamMetrics::default();
        assert_eq!(m.total_chunks(), 0);
    }

    #[test]
    fn test_metrics_no_duration_without_start() {
        let mut m = StreamMetrics::new();
        let mut c = chunk("n", ChunkType::NodeEnd, json!({}));
        c.sequence = 10;
        m.record(&c);
        assert!(m.node_durations().is_empty());
    }

    // -----------------------------------------------------------------------
    // Integration / cross-type tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_collector_filter_replay_integration() {
        let mut collector = StreamCollector::new();
        collector.push(chunk("a", ChunkType::NodeStart, json!({})));
        collector.push(chunk("a", ChunkType::NodeOutput, json!({"v": 1})));
        collector.push(chunk("b", ChunkType::NodeOutput, json!({"v": 2})));
        collector.push(chunk("a", ChunkType::NodeEnd, json!({})));

        let filter = StreamFilter::new().by_node("a");
        let filtered: Vec<StreamChunk> = filter
            .apply(collector.chunks())
            .into_iter()
            .cloned()
            .collect();

        let mut replay = StreamReplay::new(filtered);
        assert_eq!(replay.remaining(), 3);
        let first = replay.next_chunk().unwrap();
        assert_eq!(first.chunk_type, ChunkType::NodeStart);
        let second = replay.next_chunk().unwrap();
        assert_eq!(second.data, json!({"v": 1}));
    }

    #[test]
    fn test_subscription_with_collector() {
        let mut collector = StreamCollector::new();
        collector.push(chunk("agent", ChunkType::NodeOutput, json!(1)));
        collector.push(chunk("tool", ChunkType::NodeOutput, json!(2)));
        collector.push(chunk("agent", ChunkType::Error, json!("e")));

        let sub = StreamSubscription::new()
            .subscribe_node("agent")
            .subscribe_type(ChunkType::NodeOutput);

        let matched: Vec<&StreamChunk> = collector
            .chunks()
            .iter()
            .filter(|c| sub.is_subscribed(c))
            .collect();
        assert_eq!(matched.len(), 1); // only agent+NodeOutput
    }

    #[test]
    fn test_aggregator_from_collector() {
        let mut collector = StreamCollector::new();
        collector.push(chunk("a", ChunkType::NodeStart, json!({})));
        collector.push(chunk("a", ChunkType::NodeOutput, json!({"r": 1})));
        collector.push(chunk("a", ChunkType::StateUpdate, json!({"final": true})));

        let mut agg = StreamAggregator::new();
        for c in collector.chunks() {
            agg.process(c);
        }
        assert_eq!(agg.final_state(), Some(json!({"final": true})));
        assert_eq!(agg.execution_order(), vec!["a"]);
    }

    #[test]
    fn test_metrics_from_collector() {
        let mut collector = StreamCollector::new();
        collector.push(chunk("a", ChunkType::NodeStart, json!({})));
        collector.push(chunk("a", ChunkType::NodeOutput, json!(1)));
        collector.push(chunk("a", ChunkType::NodeEnd, json!({})));

        let mut metrics = StreamMetrics::new();
        for c in collector.chunks() {
            metrics.record(c);
        }
        assert_eq!(metrics.total_chunks(), 3);
        assert_eq!(metrics.chunks_per_node()["a"], 3);
        let durations = metrics.node_durations();
        assert_eq!(durations["a"], 2); // seq 2 - seq 0
    }
}
