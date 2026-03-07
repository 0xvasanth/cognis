//! Stream writer/reader utilities for graph execution.
//!
//! Provides [`StreamWriter`] and [`StreamReader`] for producing and consuming
//! [`StreamChunk`] values through a bounded async channel, along with a
//! [`StreamCollector`] for aggregating chunks after the fact.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::SystemTime;

use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;

use crate::types::StreamMode;

// ---------------------------------------------------------------------------
// StreamChunk
// ---------------------------------------------------------------------------

/// A single chunk of streaming output produced during graph execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// The node that produced this chunk.
    pub node: String,
    /// The execution step number.
    pub step: usize,
    /// The stream mode of this chunk.
    pub mode: StreamMode,
    /// The data payload.
    pub data: Value,
    /// When the chunk was created.
    #[serde(with = "system_time_serde")]
    pub timestamp: SystemTime,
}

impl PartialEq for StreamChunk {
    fn eq(&self, other: &Self) -> bool {
        self.node == other.node
            && self.step == other.step
            && self.mode == other.mode
            && self.data == other.data
    }
}

/// Serde helper for `SystemTime`.
mod system_time_serde {
    use serde::{self, Deserialize, Deserializer, Serializer};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    pub fn serialize<S>(time: &SystemTime, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let duration = time.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
        serializer.serialize_f64(duration.as_secs_f64())
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<SystemTime, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(UNIX_EPOCH + Duration::from_secs_f64(secs))
    }
}

// ---------------------------------------------------------------------------
// StreamWriter
// ---------------------------------------------------------------------------

/// Writes [`StreamChunk`] values into a bounded async channel.
///
/// Create a writer/reader pair with [`StreamWriter::new`].
#[derive(Clone)]
pub struct StreamWriter {
    tx: mpsc::Sender<StreamChunk>,
    closed: Arc<AtomicBool>,
}

impl StreamWriter {
    /// Create a new `(StreamWriter, StreamReader)` pair backed by a bounded
    /// channel of the given `capacity`.
    pub fn new(capacity: usize) -> (StreamWriter, StreamReader) {
        let (tx, rx) = mpsc::channel(capacity);
        let closed = Arc::new(AtomicBool::new(false));
        let writer = StreamWriter {
            tx,
            closed: Arc::clone(&closed),
        };
        let reader = StreamReader {
            rx,
            closed: Arc::clone(&closed),
        };
        (writer, reader)
    }

    /// Write a chunk to the stream. Returns an error if the stream is closed.
    pub async fn write(&self, chunk: StreamChunk) -> crate::errors::Result<()> {
        if self.is_closed() {
            return Err(crate::errors::LangGraphError::Other(
                "stream writer is closed".into(),
            ));
        }
        self.tx
            .send(chunk)
            .await
            .map_err(|_| crate::errors::LangGraphError::Other("stream receiver dropped".into()))
    }

    /// Convenience: write a chunk with [`StreamMode::Values`].
    pub async fn write_value(
        &self,
        node: &str,
        step: usize,
        data: Value,
    ) -> crate::errors::Result<()> {
        self.write(StreamChunk {
            node: node.to_string(),
            step,
            mode: StreamMode::Values,
            data,
            timestamp: SystemTime::now(),
        })
        .await
    }

    /// Convenience: write a chunk with [`StreamMode::Updates`].
    pub async fn write_update(
        &self,
        node: &str,
        step: usize,
        data: Value,
    ) -> crate::errors::Result<()> {
        self.write(StreamChunk {
            node: node.to_string(),
            step,
            mode: StreamMode::Updates,
            data,
            timestamp: SystemTime::now(),
        })
        .await
    }

    /// Signal that no more chunks will be written.
    pub fn close(&self) {
        self.closed.store(true, Ordering::SeqCst);
    }

    /// Returns `true` if [`close`](Self::close) has been called.
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }
}

// ---------------------------------------------------------------------------
// StreamReader
// ---------------------------------------------------------------------------

/// Reads [`StreamChunk`] values from the channel produced by a [`StreamWriter`].
///
/// Implements [`futures::Stream`] so it can be used with combinators like
/// `StreamExt::next()`.
pub struct StreamReader {
    rx: mpsc::Receiver<StreamChunk>,
    closed: Arc<AtomicBool>,
}

impl Stream for StreamReader {
    type Item = StreamChunk;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

impl StreamReader {
    /// Consume the stream and collect all chunks into a `Vec`.
    pub async fn collect_all(mut self) -> Vec<StreamChunk> {
        use futures::StreamExt;
        let mut chunks = Vec::new();
        while let Some(chunk) = self.next().await {
            chunks.push(chunk);
        }
        chunks
    }

    /// Return a new stream that only yields chunks matching the given mode.
    pub fn filter_by_mode(self, mode: StreamMode) -> FilteredStream {
        FilteredStream {
            inner: Box::pin(self),
            mode_filter: Some(mode),
            node_filter: None,
        }
    }

    /// Return a new stream that only yields chunks from the given node.
    pub fn filter_by_node(self, node: &str) -> FilteredStream {
        FilteredStream {
            inner: Box::pin(self),
            mode_filter: None,
            node_filter: Some(node.to_string()),
        }
    }

    /// Returns `true` if the writer has signalled close.
    pub fn is_writer_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }
}

// ---------------------------------------------------------------------------
// FilteredStream
// ---------------------------------------------------------------------------

/// A stream adapter that filters [`StreamChunk`] values by mode and/or node.
pub struct FilteredStream {
    inner: Pin<Box<dyn Stream<Item = StreamChunk> + Send>>,
    mode_filter: Option<StreamMode>,
    node_filter: Option<String>,
}

impl Stream for FilteredStream {
    type Item = StreamChunk;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            match self.inner.as_mut().poll_next(cx) {
                Poll::Ready(Some(chunk)) => {
                    if let Some(ref mode) = self.mode_filter {
                        if chunk.mode != *mode {
                            continue;
                        }
                    }
                    if let Some(ref node) = self.node_filter {
                        if chunk.node != *node {
                            continue;
                        }
                    }
                    return Poll::Ready(Some(chunk));
                }
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StreamCollector
// ---------------------------------------------------------------------------

/// Collects and aggregates [`StreamChunk`] values for later inspection.
#[derive(Debug, Default)]
pub struct StreamCollector {
    chunks: Vec<StreamChunk>,
}

impl StreamCollector {
    /// Create a new, empty collector.
    pub fn new() -> Self {
        Self { chunks: Vec::new() }
    }

    /// Add a chunk to the collector.
    pub fn add_chunk(&mut self, chunk: StreamChunk) {
        self.chunks.push(chunk);
    }

    /// Get the final state value — the data from the last `Values`-mode chunk.
    pub fn get_final_state(&self) -> Option<&Value> {
        self.chunks
            .iter()
            .rev()
            .find(|c| c.mode == StreamMode::Values)
            .map(|c| &c.data)
    }

    /// Get all output values grouped by node name.
    pub fn get_node_outputs(&self) -> HashMap<String, Vec<Value>> {
        let mut map: HashMap<String, Vec<Value>> = HashMap::new();
        for chunk in &self.chunks {
            map.entry(chunk.node.clone())
                .or_default()
                .push(chunk.data.clone());
        }
        map
    }

    /// Get all chunks produced at a given step.
    pub fn get_chunks_by_step(&self, step: usize) -> Vec<&StreamChunk> {
        self.chunks.iter().filter(|c| c.step == step).collect()
    }

    /// Total number of chunks collected.
    pub fn total_chunks(&self) -> usize {
        self.chunks.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use serde_json::json;

    fn make_chunk(node: &str, step: usize, mode: StreamMode, data: Value) -> StreamChunk {
        StreamChunk {
            node: node.to_string(),
            step,
            mode,
            data,
            timestamp: SystemTime::now(),
        }
    }

    // 1. Basic write and read
    #[tokio::test]
    async fn test_write_and_read_single_chunk() {
        let (writer, reader) = StreamWriter::new(8);
        let chunk = make_chunk("A", 0, StreamMode::Values, json!({"x": 1}));
        writer.write(chunk.clone()).await.unwrap();
        drop(writer);
        let collected = reader.collect_all().await;
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].node, "A");
    }

    // 2. Multiple chunks preserve order
    #[tokio::test]
    async fn test_multiple_chunks_order() {
        let (writer, reader) = StreamWriter::new(16);
        for i in 0..5 {
            writer
                .write(make_chunk("N", i, StreamMode::Updates, json!(i)))
                .await
                .unwrap();
        }
        drop(writer);
        let collected = reader.collect_all().await;
        assert_eq!(collected.len(), 5);
        for (i, c) in collected.iter().enumerate() {
            assert_eq!(c.step, i);
        }
    }

    // 3. write_value convenience
    #[tokio::test]
    async fn test_write_value_convenience() {
        let (writer, reader) = StreamWriter::new(8);
        writer
            .write_value("node1", 0, json!({"state": true}))
            .await
            .unwrap();
        drop(writer);
        let collected = reader.collect_all().await;
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].mode, StreamMode::Values);
        assert_eq!(collected[0].node, "node1");
    }

    // 4. write_update convenience
    #[tokio::test]
    async fn test_write_update_convenience() {
        let (writer, reader) = StreamWriter::new(8);
        writer
            .write_update("node2", 1, json!({"delta": 42}))
            .await
            .unwrap();
        drop(writer);
        let collected = reader.collect_all().await;
        assert_eq!(collected.len(), 1);
        assert_eq!(collected[0].mode, StreamMode::Updates);
    }

    // 5. close() marks writer as closed
    #[tokio::test]
    async fn test_close_marks_closed() {
        let (writer, _reader) = StreamWriter::new(8);
        assert!(!writer.is_closed());
        writer.close();
        assert!(writer.is_closed());
    }

    // 6. Writing after close returns error
    #[tokio::test]
    async fn test_write_after_close_errors() {
        let (writer, _reader) = StreamWriter::new(8);
        writer.close();
        let result = writer
            .write(make_chunk("A", 0, StreamMode::Values, json!(null)))
            .await;
        assert!(result.is_err());
    }

    // 7. Reader sees writer closed state
    #[tokio::test]
    async fn test_reader_sees_writer_closed() {
        let (writer, reader) = StreamWriter::new(8);
        assert!(!reader.is_writer_closed());
        writer.close();
        assert!(reader.is_writer_closed());
    }

    // 8. Reader stream terminates when writer is dropped
    #[tokio::test]
    async fn test_reader_terminates_on_writer_drop() {
        let (writer, reader) = StreamWriter::new(8);
        writer.write_value("A", 0, json!(1)).await.unwrap();
        drop(writer);
        let collected = reader.collect_all().await;
        assert_eq!(collected.len(), 1);
    }

    // 9. filter_by_mode only passes matching mode
    #[tokio::test]
    async fn test_filter_by_mode() {
        let (writer, reader) = StreamWriter::new(16);
        writer.write_value("A", 0, json!(1)).await.unwrap();
        writer.write_update("A", 1, json!(2)).await.unwrap();
        writer.write_value("B", 2, json!(3)).await.unwrap();
        writer.write_update("B", 3, json!(4)).await.unwrap();
        drop(writer);

        let mut filtered = reader.filter_by_mode(StreamMode::Values);
        let mut results = Vec::new();
        while let Some(c) = filtered.next().await {
            results.push(c);
        }
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|c| c.mode == StreamMode::Values));
    }

    // 10. filter_by_node only passes matching node
    #[tokio::test]
    async fn test_filter_by_node() {
        let (writer, reader) = StreamWriter::new(16);
        writer.write_value("A", 0, json!(1)).await.unwrap();
        writer.write_value("B", 1, json!(2)).await.unwrap();
        writer.write_value("A", 2, json!(3)).await.unwrap();
        drop(writer);

        let mut filtered = reader.filter_by_node("A");
        let mut results = Vec::new();
        while let Some(c) = filtered.next().await {
            results.push(c);
        }
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|c| c.node == "A"));
    }

    // 11. StreamCollector new is empty
    #[tokio::test]
    async fn test_collector_new_empty() {
        let collector = StreamCollector::new();
        assert_eq!(collector.total_chunks(), 0);
        assert!(collector.get_final_state().is_none());
    }

    // 12. StreamCollector add_chunk and total_chunks
    #[tokio::test]
    async fn test_collector_add_chunk() {
        let mut collector = StreamCollector::new();
        collector.add_chunk(make_chunk("A", 0, StreamMode::Values, json!(1)));
        collector.add_chunk(make_chunk("B", 1, StreamMode::Updates, json!(2)));
        assert_eq!(collector.total_chunks(), 2);
    }

    // 13. StreamCollector get_final_state returns last Values chunk
    #[tokio::test]
    async fn test_collector_get_final_state() {
        let mut collector = StreamCollector::new();
        collector.add_chunk(make_chunk("A", 0, StreamMode::Values, json!({"x": 1})));
        collector.add_chunk(make_chunk(
            "A",
            1,
            StreamMode::Updates,
            json!({"delta": true}),
        ));
        collector.add_chunk(make_chunk("B", 2, StreamMode::Values, json!({"x": 2})));

        let final_state = collector.get_final_state().unwrap();
        assert_eq!(final_state, &json!({"x": 2}));
    }

    // 14. StreamCollector get_final_state returns None when no Values chunks
    #[tokio::test]
    async fn test_collector_no_values_chunks() {
        let mut collector = StreamCollector::new();
        collector.add_chunk(make_chunk("A", 0, StreamMode::Updates, json!(1)));
        assert!(collector.get_final_state().is_none());
    }

    // 15. StreamCollector get_node_outputs
    #[tokio::test]
    async fn test_collector_get_node_outputs() {
        let mut collector = StreamCollector::new();
        collector.add_chunk(make_chunk("A", 0, StreamMode::Values, json!(1)));
        collector.add_chunk(make_chunk("B", 1, StreamMode::Values, json!(2)));
        collector.add_chunk(make_chunk("A", 2, StreamMode::Updates, json!(3)));

        let outputs = collector.get_node_outputs();
        assert_eq!(outputs.get("A").unwrap().len(), 2);
        assert_eq!(outputs.get("B").unwrap().len(), 1);
    }

    // 16. StreamCollector get_chunks_by_step
    #[tokio::test]
    async fn test_collector_get_chunks_by_step() {
        let mut collector = StreamCollector::new();
        collector.add_chunk(make_chunk("A", 0, StreamMode::Values, json!(1)));
        collector.add_chunk(make_chunk("B", 0, StreamMode::Updates, json!(2)));
        collector.add_chunk(make_chunk("C", 1, StreamMode::Values, json!(3)));

        let step0 = collector.get_chunks_by_step(0);
        assert_eq!(step0.len(), 2);
        let step1 = collector.get_chunks_by_step(1);
        assert_eq!(step1.len(), 1);
        let step2 = collector.get_chunks_by_step(2);
        assert_eq!(step2.len(), 0);
    }

    // 17. StreamChunk serialization roundtrip
    #[tokio::test]
    async fn test_chunk_serialization() {
        let chunk = make_chunk("A", 5, StreamMode::Debug, json!({"debug": true}));
        let serialized = serde_json::to_string(&chunk).unwrap();
        let deserialized: StreamChunk = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized.node, "A");
        assert_eq!(deserialized.step, 5);
        assert_eq!(deserialized.mode, StreamMode::Debug);
        assert_eq!(deserialized.data, json!({"debug": true}));
    }

    // 18. Writer clone sends to same channel
    #[tokio::test]
    async fn test_writer_clone_sends_to_same_channel() {
        let (writer, reader) = StreamWriter::new(16);
        let writer2 = writer.clone();

        writer.write_value("A", 0, json!(1)).await.unwrap();
        writer2.write_value("B", 1, json!(2)).await.unwrap();

        drop(writer);
        drop(writer2);

        let collected = reader.collect_all().await;
        assert_eq!(collected.len(), 2);
    }

    // 19. Cloned writer sees close state
    #[tokio::test]
    async fn test_cloned_writer_sees_close() {
        let (writer, _reader) = StreamWriter::new(8);
        let writer2 = writer.clone();

        writer.close();
        assert!(writer2.is_closed());
    }

    // 20. StreamReader implements Stream trait correctly
    #[tokio::test]
    async fn test_reader_as_stream() {
        let (writer, mut reader) = StreamWriter::new(8);
        writer.write_value("A", 0, json!("hello")).await.unwrap();
        drop(writer);

        let item = reader.next().await;
        assert!(item.is_some());
        assert_eq!(item.unwrap().data, json!("hello"));

        let item = reader.next().await;
        assert!(item.is_none());
    }

    // 21. FilteredStream with no matching chunks yields empty
    #[tokio::test]
    async fn test_filter_yields_empty_when_no_match() {
        let (writer, reader) = StreamWriter::new(8);
        writer.write_value("A", 0, json!(1)).await.unwrap();
        drop(writer);

        let mut filtered = reader.filter_by_node("Z");
        let item = filtered.next().await;
        assert!(item.is_none());
    }

    // 22. Collector with mixed modes and multiple nodes
    #[tokio::test]
    async fn test_collector_integration() {
        let (writer, reader) = StreamWriter::new(32);
        let mut collector = StreamCollector::new();

        writer
            .write_value("A", 0, json!({"state": "a"}))
            .await
            .unwrap();
        writer
            .write_update("A", 0, json!({"delta": "a"}))
            .await
            .unwrap();
        writer
            .write_value("B", 1, json!({"state": "b"}))
            .await
            .unwrap();
        writer
            .write_update("B", 1, json!({"delta": "b"}))
            .await
            .unwrap();
        writer
            .write_value("C", 2, json!({"state": "c"}))
            .await
            .unwrap();
        drop(writer);

        let chunks = reader.collect_all().await;
        for chunk in chunks {
            collector.add_chunk(chunk);
        }

        assert_eq!(collector.total_chunks(), 5);
        assert_eq!(collector.get_final_state().unwrap(), &json!({"state": "c"}));

        let outputs = collector.get_node_outputs();
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs["A"].len(), 2);
        assert_eq!(outputs["B"].len(), 2);
        assert_eq!(outputs["C"].len(), 1);
    }
}
