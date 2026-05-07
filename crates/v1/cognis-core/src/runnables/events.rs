//! Stream all callback events produced during execution of a runnable.
//!
//! This module provides the [`stream_events`] function, the Rust equivalent of
//! Python's `Runnable.astream_events(version="v2")`. It wires up an
//! [`EventStreamCallbackHandler`] into the runnable config, spawns the
//! invocation in a background task, and returns a stream of [`StreamEvent`]
//! values that the caller can consume in real time.
//!
//! Additionally, this module provides a typed event system for observing
//! runnable execution progress: [`RunEventType`], [`RunEvent`],
//! [`EventEmitter`], [`EventFilter`], and [`EventTrace`].

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::callbacks::CallbackHandler;
use crate::error::Result;
use crate::tracers::event_stream::{EventStreamCallbackHandler, RootEventFilter, StreamEvent};

use super::base::Runnable;
use super::config::{ensure_config, RunnableConfig};

/// Stream all events produced during execution of a runnable.
///
/// Creates an [`EventStreamCallbackHandler`], injects it into the config's
/// callback list, spawns `runnable.invoke()` in a background tokio task, and
/// returns a `Stream` of [`StreamEvent`] values. The stream yields events as
/// they are emitted by the callback handler and completes once the invoke task
/// finishes and the channel is drained.
///
/// This is the Rust equivalent of Python's `astream_events(version="v2")`.
///
/// # Arguments
///
/// * `runnable` - The runnable to execute, wrapped in `Arc` so it can be
///   shared with the spawned task.
/// * `input` - The input value to pass to `runnable.invoke()`.
/// * `config` - Optional configuration. The handler is appended to whatever
///   callbacks are already present.
///
/// # Examples
///
/// ```ignore
/// use std::sync::Arc;
/// use futures::StreamExt;
/// use cognis_core::runnables::{RunnableLambda, events::stream_events};
///
/// let runnable = Arc::new(RunnableLambda::new("double", |v| async move {
///     Ok(serde_json::json!(v.as_i64().unwrap() * 2))
/// }));
///
/// let mut stream = stream_events(runnable, serde_json::json!(5), None).await?;
/// while let Some(event) = stream.next().await {
///     println!("{:?}", event?);
/// }
/// ```
pub async fn stream_events(
    runnable: Arc<dyn Runnable>,
    input: Value,
    config: Option<RunnableConfig>,
) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
    stream_events_with_filter(runnable, input, config, RootEventFilter::default()).await
}

/// Like [`stream_events`] but with a custom [`RootEventFilter`] to control
/// which events are emitted.
pub async fn stream_events_with_filter(
    runnable: Arc<dyn Runnable>,
    input: Value,
    config: Option<RunnableConfig>,
    filter: RootEventFilter,
) -> Result<Pin<Box<dyn Stream<Item = Result<StreamEvent>> + Send>>> {
    // 1. Create the event stream handler and take its receiver.
    let handler = Arc::new(EventStreamCallbackHandler::new(256, filter));
    let receiver = handler
        .take_receiver()
        .expect("receiver should be available on a fresh handler");

    // 2. Build a config that includes this handler in callbacks.
    let mut cfg = ensure_config(config.as_ref());
    cfg.callbacks
        .push(handler.clone() as Arc<dyn CallbackHandler>);

    // Assign a run ID if one is not already set.
    let run_id = cfg.run_id.unwrap_or_else(Uuid::new_v4);
    cfg.run_id = Some(run_id);

    // 3. Capture the runnable name for the wrapping chain events.
    let runnable_name = runnable.name().to_string();
    let input_clone = input.clone();

    // 4. Spawn the invoke in a background task.
    //    We emit on_chain_start / on_chain_end / on_chain_error around the
    //    invoke so that every stream_events call produces at least a start
    //    and end (or error) event, matching Python's v2 behaviour.
    let handler_for_task = handler.clone();
    tokio::spawn(async move {
        let serialized =
            serde_json::json!({"name": runnable_name, "id": ["Runnable", &runnable_name]});

        // Emit on_chain_start.
        let _ = handler_for_task
            .on_chain_start(&serialized, &input_clone, run_id, None)
            .await;

        // Run the actual runnable.
        let result = runnable.invoke(input_clone.clone(), Some(&cfg)).await;

        match &result {
            Ok(output) => {
                let _ = handler_for_task.on_chain_end(output, run_id, None).await;
            }
            Err(e) => {
                let _ = handler_for_task
                    .on_chain_error(&e.to_string(), run_id, None)
                    .await;
            }
        }

        // Drop the handler reference so the sender side closes once all
        // references are gone. The `handler` Arc in the outer scope will be
        // dropped after this function returns, and the one inside `cfg.callbacks`
        // is dropped here when `cfg` goes out of scope.
        drop(handler_for_task);
        drop(cfg);
    });

    // 5. Convert the mpsc::Receiver into a Stream of Result<StreamEvent>.
    let stream = ReceiverStream::new(receiver);
    let mapped = futures::StreamExt::map(stream, Ok);

    Ok(Box::pin(mapped))
}

// ---------------------------------------------------------------------------
// Typed event streaming system
// ---------------------------------------------------------------------------

/// The type of a run event, describing what stage of execution it represents.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RunEventType {
    /// Execution has started.
    Start,
    /// Execution has completed successfully.
    End,
    /// A streaming chunk was produced.
    StreamChunk,
    /// An error occurred during execution.
    Error,
    /// A retry attempt is being made.
    Retry,
    /// A tool invocation has started.
    ToolStart,
    /// A tool invocation has completed.
    ToolEnd,
    /// A chain execution has started.
    ChainStart,
    /// A chain execution has completed.
    ChainEnd,
    /// A custom, user-defined event type.
    Custom(String),
}

/// A single event emitted during runnable execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunEvent {
    /// The type of this event.
    pub event_type: RunEventType,
    /// A human-readable name for this event.
    pub name: String,
    /// Arbitrary data payload associated with the event.
    pub data: Value,
    /// Unique identifier for this run.
    pub run_id: String,
    /// Optional identifier of the parent run, establishing a tree relationship.
    pub parent_run_id: Option<String>,
    /// The time at which this event was created.
    pub timestamp: SystemTime,
    /// Arbitrary metadata key-value pairs.
    pub metadata: HashMap<String, Value>,
}

impl RunEvent {
    /// Create a new `RunEvent` with an auto-generated `run_id` and the current
    /// timestamp. The `parent_run_id` defaults to `None` and `metadata` is empty.
    pub fn new(event_type: RunEventType, name: impl Into<String>, data: Value) -> Self {
        Self {
            event_type,
            name: name.into(),
            data,
            run_id: Uuid::new_v4().to_string(),
            parent_run_id: None,
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
        }
    }

    /// Set the `parent_run_id` on this event, returning the modified event.
    pub fn with_parent(mut self, parent_run_id: impl Into<String>) -> Self {
        self.parent_run_id = Some(parent_run_id.into());
        self
    }

    /// Insert a metadata key-value pair, returning the modified event.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Serialize this event to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Compute the duration elapsed between `other`'s timestamp and this event's
    /// timestamp. Returns `Duration::ZERO` if this event is earlier than `other`.
    pub fn elapsed_since(&self, other: &RunEvent) -> Duration {
        self.timestamp
            .duration_since(other.timestamp)
            .unwrap_or(Duration::ZERO)
    }
}

/// An event collector that buffers [`RunEvent`]s in a thread-safe manner.
///
/// Internally backed by `Arc<Mutex<Vec<RunEvent>>>`, so cloning the emitter
/// shares the same underlying buffer.
#[derive(Debug, Clone)]
pub struct EventEmitter {
    buffer: Arc<Mutex<Vec<RunEvent>>>,
}

impl EventEmitter {
    /// Create a new, empty `EventEmitter`.
    pub fn new() -> Self {
        Self {
            buffer: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Add an event to the internal buffer.
    pub fn emit(&self, event: RunEvent) {
        self.buffer.lock().unwrap().push(event);
    }

    /// Return a snapshot (clone) of all events collected so far.
    pub fn events(&self) -> Vec<RunEvent> {
        self.buffer.lock().unwrap().clone()
    }

    /// Return a snapshot of events matching the given `event_type`.
    pub fn events_by_type(&self, event_type: &RunEventType) -> Vec<RunEvent> {
        self.buffer
            .lock()
            .unwrap()
            .iter()
            .filter(|e| &e.event_type == event_type)
            .cloned()
            .collect()
    }

    /// Return a snapshot of events whose `name` matches the given string.
    pub fn events_by_name(&self, name: &str) -> Vec<RunEvent> {
        self.buffer
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.name == name)
            .cloned()
            .collect()
    }

    /// Clear all events from the buffer.
    pub fn clear(&self) {
        self.buffer.lock().unwrap().clear();
    }

    /// Return the number of events in the buffer.
    pub fn len(&self) -> usize {
        self.buffer.lock().unwrap().len()
    }

    /// Return `true` if the buffer contains no events.
    pub fn is_empty(&self) -> bool {
        self.buffer.lock().unwrap().is_empty()
    }
}

impl Default for EventEmitter {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for event filter predicate functions.
type EventPredicate = Box<dyn Fn(&RunEvent) -> bool + Send + Sync>;

/// A filter for selecting events by type, name, or a custom predicate.
pub struct EventFilter {
    types: Option<Vec<RunEventType>>,
    names: Option<Vec<String>>,
    predicate: Option<EventPredicate>,
}

impl EventFilter {
    /// Create a new `EventFilter` that matches all events.
    pub fn new() -> Self {
        Self {
            types: None,
            names: None,
            predicate: None,
        }
    }

    /// Only match events whose `event_type` is in the given list.
    pub fn with_types(mut self, types: Vec<RunEventType>) -> Self {
        self.types = Some(types);
        self
    }

    /// Only match events whose `name` is in the given list.
    pub fn with_names(mut self, names: Vec<String>) -> Self {
        self.names = Some(names);
        self
    }

    /// Only match events for which the given predicate returns `true`.
    pub fn with_predicate<F>(mut self, predicate: F) -> Self
    where
        F: Fn(&RunEvent) -> bool + Send + Sync + 'static,
    {
        self.predicate = Some(Box::new(predicate));
        self
    }

    /// Check whether a single event matches all active filter criteria.
    pub fn matches(&self, event: &RunEvent) -> bool {
        if let Some(ref types) = self.types {
            if !types.contains(&event.event_type) {
                return false;
            }
        }
        if let Some(ref names) = self.names {
            if !names.contains(&event.name) {
                return false;
            }
        }
        if let Some(ref pred) = self.predicate {
            if !pred(event) {
                return false;
            }
        }
        true
    }

    /// Return a new `Vec` containing only the events that match this filter.
    pub fn filter(&self, events: &[RunEvent]) -> Vec<RunEvent> {
        events.iter().filter(|e| self.matches(e)).cloned().collect()
    }
}

impl Default for EventFilter {
    fn default() -> Self {
        Self::new()
    }
}

/// A tree view of [`RunEvent`]s organized by parent/child relationships.
///
/// Events are linked through `run_id` and `parent_run_id`. An event with
/// `parent_run_id == None` is considered a root event.
pub struct EventTrace {
    events: Vec<RunEvent>,
}

impl EventTrace {
    /// Build an `EventTrace` from a slice of events.
    pub fn from_events(events: &[RunEvent]) -> Self {
        Self {
            events: events.to_vec(),
        }
    }

    /// Return references to all root events (those with no parent).
    pub fn roots(&self) -> Vec<&RunEvent> {
        self.events
            .iter()
            .filter(|e| e.parent_run_id.is_none())
            .collect()
    }

    /// Return references to all events whose `parent_run_id` matches the
    /// given `run_id`.
    pub fn children_of(&self, run_id: &str) -> Vec<&RunEvent> {
        self.events
            .iter()
            .filter(|e| e.parent_run_id.as_deref() == Some(run_id))
            .collect()
    }

    /// Compute the depth of the event with the given `run_id` in the tree.
    ///
    /// A root event has depth 0. Returns 0 if the `run_id` is not found.
    pub fn depth_of(&self, run_id: &str) -> usize {
        let mut depth = 0usize;
        let mut current_id = run_id;

        loop {
            let event = self.events.iter().find(|e| e.run_id == current_id);
            match event.and_then(|e| e.parent_run_id.as_deref()) {
                Some(parent_id) => {
                    depth += 1;
                    current_id = parent_id;
                }
                None => break,
            }
        }

        depth
    }

    /// Serialize the trace as a nested JSON tree.
    ///
    /// Each root event becomes a JSON object with a `"children"` array
    /// containing its nested descendants.
    pub fn to_json(&self) -> Value {
        let roots = self.roots();
        let nodes: Vec<Value> = roots.iter().map(|root| self.event_to_json(root)).collect();
        serde_json::json!(nodes)
    }

    fn event_to_json(&self, event: &RunEvent) -> Value {
        let children = self.children_of(&event.run_id);
        let child_nodes: Vec<Value> = children.iter().map(|c| self.event_to_json(c)).collect();

        let mut obj = event.to_json();
        if let Value::Object(ref mut map) = obj {
            map.insert("children".to_string(), serde_json::json!(child_nodes));
        }
        obj
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runnables::{RunnableLambda, RunnableSequence};
    use crate::tracers::EventType;
    use futures::StreamExt;
    use serde_json::json;

    #[tokio::test]
    async fn test_stream_events_produces_chain_start_and_end() {
        let runnable: Arc<dyn Runnable> =
            Arc::new(RunnableLambda::new("doubler", |v: Value| async move {
                let n = v.as_i64().unwrap();
                Ok(json!(n * 2))
            }));

        let mut stream = stream_events(runnable, json!(5), None).await.unwrap();

        let mut events = Vec::new();
        while let Some(evt) = stream.next().await {
            events.push(evt.unwrap());
        }

        // Should have at least on_chain_start and on_chain_end.
        assert!(
            events.len() >= 2,
            "expected at least 2 events, got {}",
            events.len()
        );

        assert_eq!(events.first().unwrap().event, EventType::OnChainStart);
        assert_eq!(events.last().unwrap().event, EventType::OnChainEnd);

        // The end event should carry the output.
        let end_event = events.last().unwrap();
        assert_eq!(end_event.data.output, Some(json!(10)));
    }

    #[tokio::test]
    async fn test_stream_events_correct_event_types() {
        let runnable: Arc<dyn Runnable> = Arc::new(RunnableLambda::new(
            "identity",
            |v: Value| async move { Ok(v) },
        ));

        let mut stream = stream_events(runnable, json!("hello"), None).await.unwrap();

        let mut event_types = Vec::new();
        while let Some(evt) = stream.next().await {
            let evt = evt.unwrap();
            event_types.push(evt.event.clone());
        }

        assert!(
            event_types.contains(&EventType::OnChainStart),
            "missing on_chain_start, got: {:?}",
            event_types
        );
        assert!(
            event_types.contains(&EventType::OnChainEnd),
            "missing on_chain_end, got: {:?}",
            event_types
        );
    }

    #[tokio::test]
    async fn test_stream_events_completes_after_invoke() {
        let runnable: Arc<dyn Runnable> =
            Arc::new(RunnableLambda::new("slow", |v: Value| async move {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                Ok(v)
            }));

        let mut stream = stream_events(runnable, json!(42), None).await.unwrap();

        let mut count = 0;
        while let Some(evt) = stream.next().await {
            evt.unwrap();
            count += 1;
        }

        // Stream must have completed (the while loop exited).
        assert!(
            count >= 2,
            "stream should have yielded events before completing"
        );
    }

    #[tokio::test]
    async fn test_stream_events_with_sequence() {
        let step1 = Arc::new(RunnableLambda::new("add_one", |v: Value| async move {
            let n = v.as_i64().unwrap();
            Ok(json!(n + 1))
        })) as Arc<dyn Runnable>;

        let step2 = Arc::new(RunnableLambda::new("double", |v: Value| async move {
            let n = v.as_i64().unwrap();
            Ok(json!(n * 2))
        })) as Arc<dyn Runnable>;

        let sequence =
            Arc::new(RunnableSequence::new(vec![step1, step2]).unwrap()) as Arc<dyn Runnable>;

        let mut stream = stream_events(sequence, json!(3), None).await.unwrap();

        let mut events = Vec::new();
        while let Some(evt) = stream.next().await {
            events.push(evt.unwrap());
        }

        // Should have start and end for the outer sequence.
        assert_eq!(events.first().unwrap().event, EventType::OnChainStart);
        assert_eq!(events.last().unwrap().event, EventType::OnChainEnd);

        // The final output should be (3 + 1) * 2 = 8.
        let end_event = events.last().unwrap();
        assert_eq!(end_event.data.output, Some(json!(8)));

        // The name should reflect the sequence.
        assert_eq!(events.first().unwrap().name, "RunnableSequence");
    }

    #[tokio::test]
    async fn test_stream_events_carries_input_in_start() {
        let runnable: Arc<dyn Runnable> =
            Arc::new(RunnableLambda::new("echo", |v: Value| async move { Ok(v) }));

        let input = json!({"query": "test"});
        let mut stream = stream_events(runnable, input.clone(), None).await.unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.event, EventType::OnChainStart);
        assert_eq!(first.data.input, Some(input));
    }

    #[tokio::test]
    async fn test_stream_events_error_produces_chain_error() {
        let runnable: Arc<dyn Runnable> =
            Arc::new(RunnableLambda::new("failing", |_v: Value| async move {
                Err(crate::error::CognisError::Other(
                    "deliberate failure".into(),
                ))
            }));

        let mut stream = stream_events(runnable, json!(1), None).await.unwrap();

        let mut events = Vec::new();
        while let Some(evt) = stream.next().await {
            events.push(evt.unwrap());
        }

        assert_eq!(events.first().unwrap().event, EventType::OnChainStart);
        assert_eq!(events.last().unwrap().event, EventType::OnChainError);
        assert!(events.last().unwrap().data.error.is_some());
    }

    // -----------------------------------------------------------------------
    // RunEvent tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_run_event_new_has_generated_run_id() {
        let event = RunEvent::new(RunEventType::Start, "test", json!({"key": "value"}));
        assert!(!event.run_id.is_empty());
        assert_eq!(event.name, "test");
        assert_eq!(event.event_type, RunEventType::Start);
        assert_eq!(event.data, json!({"key": "value"}));
        assert!(event.parent_run_id.is_none());
        assert!(event.metadata.is_empty());
    }

    #[test]
    fn test_run_event_with_parent() {
        let event =
            RunEvent::new(RunEventType::Start, "child", json!(null)).with_parent("parent-123");
        assert_eq!(event.parent_run_id, Some("parent-123".to_string()));
    }

    #[test]
    fn test_run_event_with_metadata() {
        let event = RunEvent::new(RunEventType::End, "done", json!(null))
            .with_metadata("model", json!("gpt-4"))
            .with_metadata("tokens", json!(150));
        assert_eq!(event.metadata.get("model"), Some(&json!("gpt-4")));
        assert_eq!(event.metadata.get("tokens"), Some(&json!(150)));
    }

    #[test]
    fn test_run_event_to_json_serialization() {
        let event = RunEvent::new(RunEventType::StreamChunk, "chunk", json!("data"))
            .with_metadata("idx", json!(0));
        let json_val = event.to_json();
        assert!(json_val.is_object());
        assert_eq!(json_val["name"], "chunk");
        assert_eq!(json_val["data"], "data");
        assert_eq!(json_val["event_type"], "StreamChunk");
    }

    #[test]
    fn test_run_event_elapsed_since() {
        let earlier = RunEvent::new(RunEventType::Start, "a", json!(null));
        std::thread::sleep(std::time::Duration::from_millis(10));
        let later = RunEvent::new(RunEventType::End, "b", json!(null));
        let elapsed = later.elapsed_since(&earlier);
        assert!(elapsed >= std::time::Duration::from_millis(5));
    }

    #[test]
    fn test_run_event_elapsed_since_same_event_is_zero_or_small() {
        let event = RunEvent::new(RunEventType::Start, "a", json!(null));
        let elapsed = event.elapsed_since(&event);
        assert!(elapsed <= std::time::Duration::from_millis(1));
    }

    #[test]
    fn test_run_event_type_all_variants() {
        let variants = vec![
            RunEventType::Start,
            RunEventType::End,
            RunEventType::StreamChunk,
            RunEventType::Error,
            RunEventType::Retry,
            RunEventType::ToolStart,
            RunEventType::ToolEnd,
            RunEventType::ChainStart,
            RunEventType::ChainEnd,
            RunEventType::Custom("my_event".to_string()),
        ];
        assert_eq!(variants.len(), 10);
        // Ensure they are all distinct.
        for (i, a) in variants.iter().enumerate() {
            for (j, b) in variants.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_run_event_type_custom_equality() {
        assert_eq!(
            RunEventType::Custom("x".into()),
            RunEventType::Custom("x".into())
        );
        assert_ne!(
            RunEventType::Custom("x".into()),
            RunEventType::Custom("y".into())
        );
    }

    // -----------------------------------------------------------------------
    // EventEmitter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_emitter_new_is_empty() {
        let emitter = EventEmitter::new();
        assert!(emitter.is_empty());
        assert_eq!(emitter.len(), 0);
        assert!(emitter.events().is_empty());
    }

    #[test]
    fn test_emitter_emit_and_events() {
        let emitter = EventEmitter::new();
        emitter.emit(RunEvent::new(RunEventType::Start, "a", json!(1)));
        emitter.emit(RunEvent::new(RunEventType::End, "b", json!(2)));
        assert_eq!(emitter.len(), 2);
        assert!(!emitter.is_empty());
        let events = emitter.events();
        assert_eq!(events[0].name, "a");
        assert_eq!(events[1].name, "b");
    }

    #[test]
    fn test_emitter_events_by_type() {
        let emitter = EventEmitter::new();
        emitter.emit(RunEvent::new(RunEventType::Start, "a", json!(null)));
        emitter.emit(RunEvent::new(RunEventType::End, "b", json!(null)));
        emitter.emit(RunEvent::new(RunEventType::Start, "c", json!(null)));

        let starts = emitter.events_by_type(&RunEventType::Start);
        assert_eq!(starts.len(), 2);
        assert_eq!(starts[0].name, "a");
        assert_eq!(starts[1].name, "c");
    }

    #[test]
    fn test_emitter_events_by_name() {
        let emitter = EventEmitter::new();
        emitter.emit(RunEvent::new(RunEventType::Start, "alpha", json!(null)));
        emitter.emit(RunEvent::new(RunEventType::End, "beta", json!(null)));
        emitter.emit(RunEvent::new(RunEventType::Error, "alpha", json!(null)));

        let alphas = emitter.events_by_name("alpha");
        assert_eq!(alphas.len(), 2);
    }

    #[test]
    fn test_emitter_clear() {
        let emitter = EventEmitter::new();
        emitter.emit(RunEvent::new(RunEventType::Start, "a", json!(null)));
        emitter.emit(RunEvent::new(RunEventType::End, "b", json!(null)));
        assert_eq!(emitter.len(), 2);

        emitter.clear();
        assert!(emitter.is_empty());
        assert_eq!(emitter.len(), 0);
    }

    #[test]
    fn test_emitter_thread_safety() {
        let emitter = EventEmitter::new();
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let em = emitter.clone();
                std::thread::spawn(move || {
                    em.emit(RunEvent::new(
                        RunEventType::StreamChunk,
                        &format!("thread-{}", i),
                        json!(i),
                    ));
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(emitter.len(), 10);
    }

    // -----------------------------------------------------------------------
    // EventFilter tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_default_matches_all() {
        let filter = EventFilter::new();
        let event = RunEvent::new(RunEventType::Start, "x", json!(null));
        assert!(filter.matches(&event));
    }

    #[test]
    fn test_filter_by_types() {
        let filter = EventFilter::new().with_types(vec![RunEventType::Start, RunEventType::End]);
        assert!(filter.matches(&RunEvent::new(RunEventType::Start, "a", json!(null))));
        assert!(filter.matches(&RunEvent::new(RunEventType::End, "b", json!(null))));
        assert!(!filter.matches(&RunEvent::new(RunEventType::Error, "c", json!(null))));
    }

    #[test]
    fn test_filter_by_names() {
        let filter = EventFilter::new().with_names(vec!["alpha".to_string(), "beta".to_string()]);
        assert!(filter.matches(&RunEvent::new(RunEventType::Start, "alpha", json!(null))));
        assert!(!filter.matches(&RunEvent::new(RunEventType::Start, "gamma", json!(null))));
    }

    #[test]
    fn test_filter_with_predicate() {
        let filter = EventFilter::new().with_predicate(|e: &RunEvent| e.data != Value::Null);

        assert!(filter.matches(&RunEvent::new(RunEventType::Start, "a", json!(42))));
        assert!(!filter.matches(&RunEvent::new(RunEventType::Start, "b", json!(null))));
    }

    #[test]
    fn test_filter_combined_criteria() {
        let filter = EventFilter::new()
            .with_types(vec![RunEventType::Start])
            .with_names(vec!["target".to_string()]);

        // Both match
        assert!(filter.matches(&RunEvent::new(RunEventType::Start, "target", json!(null))));
        // Type matches, name does not
        assert!(!filter.matches(&RunEvent::new(RunEventType::Start, "other", json!(null))));
        // Name matches, type does not
        assert!(!filter.matches(&RunEvent::new(RunEventType::End, "target", json!(null))));
    }

    #[test]
    fn test_filter_vec() {
        let events = vec![
            RunEvent::new(RunEventType::Start, "a", json!(null)),
            RunEvent::new(RunEventType::End, "b", json!(null)),
            RunEvent::new(RunEventType::Start, "c", json!(null)),
        ];
        let filter = EventFilter::new().with_types(vec![RunEventType::Start]);
        let filtered = filter.filter(&events);
        assert_eq!(filtered.len(), 2);
        assert_eq!(filtered[0].name, "a");
        assert_eq!(filtered[1].name, "c");
    }

    #[test]
    fn test_filter_empty_events() {
        let filter = EventFilter::new().with_types(vec![RunEventType::Error]);
        let filtered = filter.filter(&[]);
        assert!(filtered.is_empty());
    }

    // -----------------------------------------------------------------------
    // EventTrace tests
    // -----------------------------------------------------------------------

    fn make_event(
        event_type: RunEventType,
        name: &str,
        run_id: &str,
        parent: Option<&str>,
    ) -> RunEvent {
        let mut e = RunEvent::new(event_type, name, json!(null));
        e.run_id = run_id.to_string();
        e.parent_run_id = parent.map(|s| s.to_string());
        e
    }

    #[test]
    fn test_trace_roots() {
        let events = vec![
            make_event(RunEventType::ChainStart, "root1", "r1", None),
            make_event(RunEventType::ToolStart, "child1", "c1", Some("r1")),
            make_event(RunEventType::ChainStart, "root2", "r2", None),
        ];
        let trace = EventTrace::from_events(&events);
        let roots = trace.roots();
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0].name, "root1");
        assert_eq!(roots[1].name, "root2");
    }

    #[test]
    fn test_trace_children_of() {
        let events = vec![
            make_event(RunEventType::ChainStart, "root", "r1", None),
            make_event(RunEventType::ToolStart, "tool_a", "t1", Some("r1")),
            make_event(RunEventType::ToolStart, "tool_b", "t2", Some("r1")),
            make_event(RunEventType::ToolEnd, "tool_a_end", "t3", Some("t1")),
        ];
        let trace = EventTrace::from_events(&events);
        let children = trace.children_of("r1");
        assert_eq!(children.len(), 2);
        assert_eq!(children[0].name, "tool_a");
        assert_eq!(children[1].name, "tool_b");

        let grandchildren = trace.children_of("t1");
        assert_eq!(grandchildren.len(), 1);
        assert_eq!(grandchildren[0].name, "tool_a_end");
    }

    #[test]
    fn test_trace_depth_of() {
        let events = vec![
            make_event(RunEventType::ChainStart, "root", "r1", None),
            make_event(RunEventType::ToolStart, "child", "c1", Some("r1")),
            make_event(RunEventType::StreamChunk, "grandchild", "g1", Some("c1")),
        ];
        let trace = EventTrace::from_events(&events);
        assert_eq!(trace.depth_of("r1"), 0);
        assert_eq!(trace.depth_of("c1"), 1);
        assert_eq!(trace.depth_of("g1"), 2);
    }

    #[test]
    fn test_trace_depth_of_unknown_id() {
        let trace = EventTrace::from_events(&[]);
        assert_eq!(trace.depth_of("nonexistent"), 0);
    }

    #[test]
    fn test_trace_to_json_nested() {
        let events = vec![
            make_event(RunEventType::ChainStart, "root", "r1", None),
            make_event(RunEventType::ToolStart, "child", "c1", Some("r1")),
        ];
        let trace = EventTrace::from_events(&events);
        let json = trace.to_json();
        assert!(json.is_array());
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        let root = &arr[0];
        assert_eq!(root["name"], "root");
        let children = root["children"].as_array().unwrap();
        assert_eq!(children.len(), 1);
        assert_eq!(children[0]["name"], "child");
    }

    #[test]
    fn test_trace_empty() {
        let trace = EventTrace::from_events(&[]);
        assert!(trace.roots().is_empty());
        assert_eq!(trace.to_json(), json!([]));
    }

    #[test]
    fn test_emitter_default_trait() {
        let emitter = EventEmitter::default();
        assert!(emitter.is_empty());
    }
}
