//! Event system for the LLM framework.
//!
//! Provides pub/sub communication between components with typed events and hooks.
//!
//! # Key Types
//!
//! - [`EventType`] — categorises events (chain, LLM, tool, retriever, error, custom).
//! - [`Event`] — a single typed event carrying arbitrary JSON data.
//! - [`EventHandler`] — trait for objects that react to events.
//! - [`EventBus`] — publish/subscribe dispatcher.
//! - [`EventFilter`] — declarative predicate for filtering events.
//! - [`EventLog`] — append-only log for replay / debugging.
//! - [`EventHook`] — before/after transformation hooks.
//! - [`HookRegistry`] — named collection of hooks.
//! - [`EventMetrics`] — lightweight counters and timing stats.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// EventType
// ---------------------------------------------------------------------------

/// Describes the kind of event that occurred.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EventType {
    ChainStart,
    ChainEnd,
    LlmStart,
    LlmEnd,
    ToolStart,
    ToolEnd,
    RetrieverStart,
    RetrieverEnd,
    Error,
    Custom(String),
}

impl EventType {
    /// Returns a human-readable name for this event type.
    pub fn name(&self) -> &str {
        match self {
            EventType::ChainStart => "chain_start",
            EventType::ChainEnd => "chain_end",
            EventType::LlmStart => "llm_start",
            EventType::LlmEnd => "llm_end",
            EventType::ToolStart => "tool_start",
            EventType::ToolEnd => "tool_end",
            EventType::RetrieverStart => "retriever_start",
            EventType::RetrieverEnd => "retriever_end",
            EventType::Error => "error",
            EventType::Custom(name) => name.as_str(),
        }
    }

    /// Serialises this event type to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

impl fmt::Display for EventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

/// A typed event with associated data, timestamp, optional source, and metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Unique identifier for the event.
    pub id: String,
    /// The kind of event.
    pub event_type: EventType,
    /// Arbitrary JSON payload.
    pub data: Value,
    /// ISO-8601 timestamp (or any string representation).
    pub timestamp: String,
    /// Optional component name that emitted the event.
    pub source: Option<String>,
    /// Free-form metadata.
    pub metadata: HashMap<String, Value>,
}

impl Event {
    /// Creates a new event with an auto-generated id and the current timestamp.
    pub fn new(event_type: EventType, data: Value) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            event_type,
            data,
            timestamp: chrono_now(),
            source: None,
            metadata: HashMap::new(),
        }
    }

    /// Sets the source and returns `self` (builder pattern).
    pub fn with_source(mut self, src: &str) -> Self {
        self.source = Some(src.to_string());
        self
    }

    /// Adds a metadata entry and returns `self` (builder pattern).
    pub fn with_metadata(mut self, key: &str, value: Value) -> Self {
        self.metadata.insert(key.to_string(), value);
        self
    }

    /// Serialises the full event to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }
}

/// Simple timestamp helper – returns a fixed-format string without pulling in
/// `chrono`. Uses `std::time::SystemTime`.
fn chrono_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", d.as_secs())
}

// ---------------------------------------------------------------------------
// EventHandler trait
// ---------------------------------------------------------------------------

/// Trait for components that react to events.
pub trait EventHandler {
    /// Process an event.
    fn handle(&self, event: &Event);
    /// Which event types this handler is interested in.
    fn event_types(&self) -> Vec<EventType>;
}

// ---------------------------------------------------------------------------
// EventBus
// ---------------------------------------------------------------------------

/// A synchronous publish/subscribe event bus.
///
/// Subscribers are registered by name with a set of event types they care about
/// and a callback closure.
pub struct EventBus {
    subscribers: Vec<Subscriber>,
}

struct Subscriber {
    name: String,
    event_types: Vec<EventType>,
    callback: Box<dyn Fn(&Event)>,
}

impl EventBus {
    /// Creates an empty event bus.
    pub fn new() -> Self {
        Self {
            subscribers: Vec::new(),
        }
    }

    /// Registers a subscriber.
    pub fn subscribe(
        &mut self,
        handler_name: String,
        event_types: Vec<EventType>,
        callback: Box<dyn Fn(&Event)>,
    ) {
        self.subscribers.push(Subscriber {
            name: handler_name,
            event_types,
            callback,
        });
    }

    /// Publishes an event to all subscribers whose event-type list contains the
    /// event's type.
    pub fn publish(&self, event: &Event) {
        for sub in &self.subscribers {
            if sub.event_types.contains(&event.event_type) {
                (sub.callback)(event);
            }
        }
    }

    /// Removes all subscribers with the given name.
    pub fn unsubscribe(&mut self, handler_name: &str) {
        self.subscribers.retain(|s| s.name != handler_name);
    }

    /// Returns the number of active subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    /// Removes all subscribers.
    pub fn clear(&mut self) {
        self.subscribers.clear();
    }
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// EventFilter
// ---------------------------------------------------------------------------

/// Declarative filter that can match events by type, source, timestamp, or
/// metadata.
#[derive(Debug, Clone, Default)]
pub struct EventFilter {
    event_types: Vec<EventType>,
    sources: Vec<String>,
    since: Option<String>,
    metadata: Vec<(String, Value)>,
}

impl EventFilter {
    /// Creates an empty filter that matches everything.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds an event-type constraint (logical OR with other type constraints).
    pub fn by_type(mut self, event_type: EventType) -> Self {
        self.event_types.push(event_type);
        self
    }

    /// Adds a source constraint (logical OR with other source constraints).
    pub fn by_source(mut self, source: &str) -> Self {
        self.sources.push(source.to_string());
        self
    }

    /// Only match events whose timestamp is >= `timestamp` (lexicographic).
    pub fn since(mut self, timestamp: &str) -> Self {
        self.since = Some(timestamp.to_string());
        self
    }

    /// Adds a metadata key/value constraint (all metadata constraints must match).
    pub fn by_metadata(mut self, key: &str, value: Value) -> Self {
        self.metadata.push((key.to_string(), value));
        self
    }

    /// Returns `true` if the event matches **all** active constraints.
    pub fn matches(&self, event: &Event) -> bool {
        // Type filter (OR)
        if !self.event_types.is_empty() && !self.event_types.contains(&event.event_type) {
            return false;
        }
        // Source filter (OR)
        if !self.sources.is_empty() {
            match &event.source {
                Some(src) if self.sources.contains(src) => {}
                _ => return false,
            }
        }
        // Since filter
        if let Some(ref since) = self.since {
            if event.timestamp.as_str() < since.as_str() {
                return false;
            }
        }
        // Metadata filter (AND)
        for (key, val) in &self.metadata {
            match event.metadata.get(key) {
                Some(v) if v == val => {}
                _ => return false,
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// EventLog
// ---------------------------------------------------------------------------

/// An append-only event log for replay and debugging.
#[derive(Debug, Default)]
pub struct EventLog {
    events: Vec<Event>,
}

impl EventLog {
    /// Creates an empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records an event.
    pub fn record(&mut self, event: Event) {
        self.events.push(event);
    }

    /// Returns a slice of all recorded events.
    pub fn events(&self) -> &[Event] {
        &self.events
    }

    /// Returns events that match the given filter.
    pub fn filter(&self, filter: &EventFilter) -> Vec<&Event> {
        self.events.iter().filter(|e| filter.matches(e)).collect()
    }

    /// Number of recorded events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Clears all events.
    pub fn clear(&mut self) {
        self.events.clear();
    }

    /// Returns events that match the given event type.
    pub fn events_by_type(&self, event_type: &EventType) -> Vec<&Event> {
        self.events
            .iter()
            .filter(|e| &e.event_type == event_type)
            .collect()
    }

    /// Serialises the full log to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(&self.events).unwrap_or(Value::Null)
    }
}

// ---------------------------------------------------------------------------
// EventHook
// ---------------------------------------------------------------------------

/// Callback type for event hook transformations.
type HookCallback = Box<dyn Fn(&Value) -> Value>;

/// Pre/post transformation hooks around an operation.
pub struct EventHook {
    name: String,
    before_cb: Option<HookCallback>,
    after_cb: Option<HookCallback>,
}

impl EventHook {
    /// Creates a new hook with the given name.
    pub fn new(name: String) -> Self {
        Self {
            name,
            before_cb: None,
            after_cb: None,
        }
    }

    /// Registers a *before* callback.
    pub fn before(mut self, callback: Box<dyn Fn(&Value) -> Value>) -> Self {
        self.before_cb = Some(callback);
        self
    }

    /// Registers an *after* callback.
    pub fn after(mut self, callback: Box<dyn Fn(&Value) -> Value>) -> Self {
        self.after_cb = Some(callback);
        self
    }

    /// Runs the *before* callback (or returns the input unchanged).
    pub fn run_before(&self, input: &Value) -> Value {
        match &self.before_cb {
            Some(cb) => cb(input),
            None => input.clone(),
        }
    }

    /// Runs the *after* callback (or returns the output unchanged).
    pub fn run_after(&self, output: &Value) -> Value {
        match &self.after_cb {
            Some(cb) => cb(output),
            None => output.clone(),
        }
    }

    /// Returns the hook's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Serialises hook metadata to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "name": self.name,
            "has_before": self.before_cb.is_some(),
            "has_after": self.after_cb.is_some(),
        })
    }
}

// ---------------------------------------------------------------------------
// HookRegistry
// ---------------------------------------------------------------------------

/// A named collection of [`EventHook`]s.
pub struct HookRegistry {
    hooks: Vec<EventHook>,
}

impl HookRegistry {
    /// Creates an empty registry.
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Registers a hook. If a hook with the same name exists it is replaced.
    pub fn register(&mut self, hook: EventHook) {
        self.hooks.retain(|h| h.name != hook.name);
        self.hooks.push(hook);
    }

    /// Looks up a hook by name.
    pub fn get(&self, name: &str) -> Option<&EventHook> {
        self.hooks.iter().find(|h| h.name == name)
    }

    /// Removes a hook by name.
    pub fn remove(&mut self, name: &str) {
        self.hooks.retain(|h| h.name != name);
    }

    /// Runs every hook's *before* callback in registration order, threading the
    /// value through each one.
    pub fn run_all_before(&self, input: &Value) -> Value {
        let mut val = input.clone();
        for hook in &self.hooks {
            val = hook.run_before(&val);
        }
        val
    }

    /// Runs every hook's *after* callback in registration order, threading the
    /// value through each one.
    pub fn run_all_after(&self, output: &Value) -> Value {
        let mut val = output.clone();
        for hook in &self.hooks {
            val = hook.run_after(&val);
        }
        val
    }

    /// Number of registered hooks.
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }
}

impl Default for HookRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// EventMetrics
// ---------------------------------------------------------------------------

/// Lightweight counters and timing statistics for events.
#[derive(Debug, Default)]
pub struct EventMetrics {
    counts: HashMap<String, usize>,
    total_time: HashMap<String, u64>,
    time_samples: HashMap<String, u64>,
}

impl EventMetrics {
    /// Creates empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Increments the counter for the given event type.
    pub fn record_event(&mut self, event_type: &EventType) {
        let key = event_type.name().to_string();
        *self.counts.entry(key).or_insert(0) += 1;
    }

    /// Records a processing-time sample for the given event type.
    pub fn record_processing_time(&mut self, event_type: &EventType, duration_ms: u64) {
        let key = event_type.name().to_string();
        *self.total_time.entry(key.clone()).or_insert(0) += duration_ms;
        *self.time_samples.entry(key).or_insert(0) += 1;
    }

    /// Returns the event count for each event type that has been recorded.
    pub fn events_per_type(&self) -> HashMap<String, usize> {
        self.counts.clone()
    }

    /// Returns the average processing time (in ms) for the given event type, or
    /// `None` if no samples have been recorded.
    pub fn average_processing_time(&self, event_type: &EventType) -> Option<f64> {
        let key = event_type.name();
        let total = *self.total_time.get(key)?;
        let samples = *self.time_samples.get(key)?;
        if samples == 0 {
            return None;
        }
        Some(total as f64 / samples as f64)
    }

    /// Total number of events recorded across all types.
    pub fn total_events(&self) -> usize {
        self.counts.values().sum()
    }

    /// Serialises metrics to JSON.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "counts": self.counts,
            "total_events": self.total_events(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::cell::RefCell;
    use std::rc::Rc;

    // -- EventType tests --------------------------------------------------

    #[test]
    fn event_type_name_builtin() {
        assert_eq!(EventType::ChainStart.name(), "chain_start");
        assert_eq!(EventType::ChainEnd.name(), "chain_end");
        assert_eq!(EventType::LlmStart.name(), "llm_start");
        assert_eq!(EventType::LlmEnd.name(), "llm_end");
        assert_eq!(EventType::ToolStart.name(), "tool_start");
        assert_eq!(EventType::ToolEnd.name(), "tool_end");
        assert_eq!(EventType::RetrieverStart.name(), "retriever_start");
        assert_eq!(EventType::RetrieverEnd.name(), "retriever_end");
        assert_eq!(EventType::Error.name(), "error");
    }

    #[test]
    fn event_type_name_custom() {
        let et = EventType::Custom("my_event".into());
        assert_eq!(et.name(), "my_event");
    }

    #[test]
    fn event_type_display() {
        assert_eq!(format!("{}", EventType::LlmStart), "llm_start");
        assert_eq!(format!("{}", EventType::Custom("foo".into())), "foo");
    }

    #[test]
    fn event_type_to_json() {
        let j = EventType::ChainStart.to_json();
        assert!(j.is_string() || j.is_object()); // serialised enum
    }

    #[test]
    fn event_type_equality() {
        assert_eq!(EventType::ChainStart, EventType::ChainStart);
        assert_ne!(EventType::ChainStart, EventType::ChainEnd);
        assert_eq!(EventType::Custom("x".into()), EventType::Custom("x".into()));
        assert_ne!(EventType::Custom("x".into()), EventType::Custom("y".into()));
    }

    #[test]
    fn event_type_clone() {
        let et = EventType::ToolStart;
        let et2 = et.clone();
        assert_eq!(et, et2);
    }

    #[test]
    fn event_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(EventType::LlmStart);
        set.insert(EventType::LlmStart);
        assert_eq!(set.len(), 1);
    }

    // -- Event tests ------------------------------------------------------

    #[test]
    fn event_new_has_id_and_timestamp() {
        let e = Event::new(EventType::ChainStart, json!({"key": "val"}));
        assert!(!e.id.is_empty());
        assert!(!e.timestamp.is_empty());
        assert_eq!(e.event_type, EventType::ChainStart);
        assert_eq!(e.data, json!({"key": "val"}));
        assert!(e.source.is_none());
        assert!(e.metadata.is_empty());
    }

    #[test]
    fn event_with_source() {
        let e = Event::new(EventType::LlmStart, json!(null)).with_source("llm_provider");
        assert_eq!(e.source.as_deref(), Some("llm_provider"));
    }

    #[test]
    fn event_with_metadata() {
        let e = Event::new(EventType::ToolStart, json!(null))
            .with_metadata("run_id", json!("abc"))
            .with_metadata("attempt", json!(2));
        assert_eq!(e.metadata.get("run_id"), Some(&json!("abc")));
        assert_eq!(e.metadata.get("attempt"), Some(&json!(2)));
    }

    #[test]
    fn event_to_json_contains_fields() {
        let e = Event::new(EventType::Error, json!("oops")).with_source("test");
        let j = e.to_json();
        assert!(j.get("id").is_some());
        assert!(j.get("event_type").is_some());
        assert!(j.get("data").is_some());
        assert!(j.get("timestamp").is_some());
        assert!(j.get("source").is_some());
    }

    #[test]
    fn event_unique_ids() {
        let a = Event::new(EventType::ChainStart, json!(null));
        let b = Event::new(EventType::ChainStart, json!(null));
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn event_clone() {
        let e = Event::new(EventType::LlmEnd, json!(42));
        let e2 = e.clone();
        assert_eq!(e.id, e2.id);
        assert_eq!(e.data, e2.data);
    }

    // -- EventHandler trait test ------------------------------------------

    struct CountingHandler {
        count: Rc<RefCell<usize>>,
    }

    impl EventHandler for CountingHandler {
        fn handle(&self, _event: &Event) {
            *self.count.borrow_mut() += 1;
        }
        fn event_types(&self) -> Vec<EventType> {
            vec![EventType::LlmStart, EventType::LlmEnd]
        }
    }

    #[test]
    fn event_handler_trait() {
        let count = Rc::new(RefCell::new(0));
        let handler = CountingHandler {
            count: count.clone(),
        };
        assert_eq!(handler.event_types().len(), 2);
        handler.handle(&Event::new(EventType::LlmStart, json!(null)));
        assert_eq!(*count.borrow(), 1);
    }

    // -- EventBus tests ---------------------------------------------------

    #[test]
    fn event_bus_new_is_empty() {
        let bus = EventBus::new();
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn event_bus_subscribe_and_count() {
        let mut bus = EventBus::new();
        bus.subscribe("a".into(), vec![EventType::ChainStart], Box::new(|_| {}));
        bus.subscribe("b".into(), vec![EventType::ChainEnd], Box::new(|_| {}));
        assert_eq!(bus.subscriber_count(), 2);
    }

    #[test]
    fn event_bus_publish_delivers_to_matching() {
        let counter = Rc::new(RefCell::new(0));
        let c = counter.clone();
        let mut bus = EventBus::new();
        bus.subscribe(
            "sub".into(),
            vec![EventType::LlmStart],
            Box::new(move |_| {
                *c.borrow_mut() += 1;
            }),
        );
        bus.publish(&Event::new(EventType::LlmStart, json!(null)));
        bus.publish(&Event::new(EventType::LlmEnd, json!(null))); // should not match
        assert_eq!(*counter.borrow(), 1);
    }

    #[test]
    fn event_bus_publish_multiple_types() {
        let counter = Rc::new(RefCell::new(0));
        let c = counter.clone();
        let mut bus = EventBus::new();
        bus.subscribe(
            "sub".into(),
            vec![EventType::ToolStart, EventType::ToolEnd],
            Box::new(move |_| {
                *c.borrow_mut() += 1;
            }),
        );
        bus.publish(&Event::new(EventType::ToolStart, json!(null)));
        bus.publish(&Event::new(EventType::ToolEnd, json!(null)));
        assert_eq!(*counter.borrow(), 2);
    }

    #[test]
    fn event_bus_unsubscribe() {
        let mut bus = EventBus::new();
        bus.subscribe("x".into(), vec![EventType::Error], Box::new(|_| {}));
        assert_eq!(bus.subscriber_count(), 1);
        bus.unsubscribe("x");
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn event_bus_unsubscribe_nonexistent() {
        let mut bus = EventBus::new();
        bus.unsubscribe("nope");
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn event_bus_clear() {
        let mut bus = EventBus::new();
        bus.subscribe("a".into(), vec![EventType::ChainStart], Box::new(|_| {}));
        bus.subscribe("b".into(), vec![EventType::ChainEnd], Box::new(|_| {}));
        bus.clear();
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn event_bus_default() {
        let bus = EventBus::default();
        assert_eq!(bus.subscriber_count(), 0);
    }

    #[test]
    fn event_bus_multiple_subscribers_same_event() {
        let c1 = Rc::new(RefCell::new(0));
        let c2 = Rc::new(RefCell::new(0));
        let c1c = c1.clone();
        let c2c = c2.clone();
        let mut bus = EventBus::new();
        bus.subscribe(
            "a".into(),
            vec![EventType::ChainStart],
            Box::new(move |_| *c1c.borrow_mut() += 1),
        );
        bus.subscribe(
            "b".into(),
            vec![EventType::ChainStart],
            Box::new(move |_| *c2c.borrow_mut() += 1),
        );
        bus.publish(&Event::new(EventType::ChainStart, json!(null)));
        assert_eq!(*c1.borrow(), 1);
        assert_eq!(*c2.borrow(), 1);
    }

    // -- EventFilter tests ------------------------------------------------

    #[test]
    fn filter_empty_matches_everything() {
        let f = EventFilter::new();
        let e = Event::new(EventType::LlmStart, json!(null));
        assert!(f.matches(&e));
    }

    #[test]
    fn filter_by_type_matches() {
        let f = EventFilter::new().by_type(EventType::LlmStart);
        assert!(f.matches(&Event::new(EventType::LlmStart, json!(null))));
        assert!(!f.matches(&Event::new(EventType::LlmEnd, json!(null))));
    }

    #[test]
    fn filter_by_type_multiple() {
        let f = EventFilter::new()
            .by_type(EventType::LlmStart)
            .by_type(EventType::LlmEnd);
        assert!(f.matches(&Event::new(EventType::LlmStart, json!(null))));
        assert!(f.matches(&Event::new(EventType::LlmEnd, json!(null))));
        assert!(!f.matches(&Event::new(EventType::Error, json!(null))));
    }

    #[test]
    fn filter_by_source() {
        let f = EventFilter::new().by_source("agent");
        let e1 = Event::new(EventType::ChainStart, json!(null)).with_source("agent");
        let e2 = Event::new(EventType::ChainStart, json!(null)).with_source("tool");
        let e3 = Event::new(EventType::ChainStart, json!(null)); // no source
        assert!(f.matches(&e1));
        assert!(!f.matches(&e2));
        assert!(!f.matches(&e3));
    }

    #[test]
    fn filter_since() {
        let f = EventFilter::new().since("1000000100");
        let mut e1 = Event::new(EventType::ChainStart, json!(null));
        e1.timestamp = "1000000099".into();
        let mut e2 = Event::new(EventType::ChainStart, json!(null));
        e2.timestamp = "1000000100".into();
        let mut e3 = Event::new(EventType::ChainStart, json!(null));
        e3.timestamp = "1000000200".into();
        assert!(!f.matches(&e1));
        assert!(f.matches(&e2));
        assert!(f.matches(&e3));
    }

    #[test]
    fn filter_by_metadata() {
        let f = EventFilter::new().by_metadata("env", json!("prod"));
        let e1 = Event::new(EventType::ChainStart, json!(null)).with_metadata("env", json!("prod"));
        let e2 = Event::new(EventType::ChainStart, json!(null)).with_metadata("env", json!("dev"));
        assert!(f.matches(&e1));
        assert!(!f.matches(&e2));
    }

    #[test]
    fn filter_combined() {
        let f = EventFilter::new()
            .by_type(EventType::LlmStart)
            .by_source("provider")
            .by_metadata("model", json!("gpt4"));
        let e = Event::new(EventType::LlmStart, json!(null))
            .with_source("provider")
            .with_metadata("model", json!("gpt4"));
        assert!(f.matches(&e));

        // Wrong type
        let e2 = Event::new(EventType::LlmEnd, json!(null))
            .with_source("provider")
            .with_metadata("model", json!("gpt4"));
        assert!(!f.matches(&e2));
    }

    #[test]
    fn filter_metadata_missing_key() {
        let f = EventFilter::new().by_metadata("missing", json!(true));
        let e = Event::new(EventType::Error, json!(null));
        assert!(!f.matches(&e));
    }

    // -- EventLog tests ---------------------------------------------------

    #[test]
    fn event_log_new_is_empty() {
        let log = EventLog::new();
        assert_eq!(log.len(), 0);
        assert!(log.is_empty());
    }

    #[test]
    fn event_log_record_and_len() {
        let mut log = EventLog::new();
        log.record(Event::new(EventType::ChainStart, json!(null)));
        log.record(Event::new(EventType::ChainEnd, json!(null)));
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn event_log_events_slice() {
        let mut log = EventLog::new();
        log.record(Event::new(EventType::Error, json!("err")));
        assert_eq!(log.events().len(), 1);
        assert_eq!(log.events()[0].event_type, EventType::Error);
    }

    #[test]
    fn event_log_filter() {
        let mut log = EventLog::new();
        log.record(Event::new(EventType::LlmStart, json!(null)));
        log.record(Event::new(EventType::LlmEnd, json!(null)));
        log.record(Event::new(EventType::Error, json!(null)));
        let f = EventFilter::new().by_type(EventType::LlmStart);
        let results = log.filter(&f);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].event_type, EventType::LlmStart);
    }

    #[test]
    fn event_log_clear() {
        let mut log = EventLog::new();
        log.record(Event::new(EventType::ChainStart, json!(null)));
        log.clear();
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn event_log_events_by_type() {
        let mut log = EventLog::new();
        log.record(Event::new(EventType::ToolStart, json!(null)));
        log.record(Event::new(EventType::ToolEnd, json!(null)));
        log.record(Event::new(EventType::ToolStart, json!(null)));
        let starts = log.events_by_type(&EventType::ToolStart);
        assert_eq!(starts.len(), 2);
    }

    #[test]
    fn event_log_to_json() {
        let mut log = EventLog::new();
        log.record(Event::new(EventType::ChainStart, json!(null)));
        let j = log.to_json();
        assert!(j.is_array());
        assert_eq!(j.as_array().unwrap().len(), 1);
    }

    #[test]
    fn event_log_events_by_type_empty() {
        let log = EventLog::new();
        assert!(log.events_by_type(&EventType::Error).is_empty());
    }

    // -- EventHook tests --------------------------------------------------

    #[test]
    fn hook_new() {
        let h = EventHook::new("test_hook".into());
        assert_eq!(h.name(), "test_hook");
    }

    #[test]
    fn hook_run_before_no_callback() {
        let h = EventHook::new("h".into());
        let v = json!({"a": 1});
        assert_eq!(h.run_before(&v), v);
    }

    #[test]
    fn hook_run_after_no_callback() {
        let h = EventHook::new("h".into());
        let v = json!(42);
        assert_eq!(h.run_after(&v), v);
    }

    #[test]
    fn hook_before_callback() {
        let h = EventHook::new("h".into()).before(Box::new(|v| {
            let mut obj = v.clone();
            obj["injected"] = json!(true);
            obj
        }));
        let out = h.run_before(&json!({"x": 1}));
        assert_eq!(out["injected"], json!(true));
        assert_eq!(out["x"], json!(1));
    }

    #[test]
    fn hook_after_callback() {
        let h = EventHook::new("h".into()).after(Box::new(
            |v| json!({"result": v.clone(), "processed": true}),
        ));
        let out = h.run_after(&json!("data"));
        assert_eq!(out["processed"], json!(true));
    }

    #[test]
    fn hook_to_json() {
        let h = EventHook::new("my_hook".into()).before(Box::new(|v| v.clone()));
        let j = h.to_json();
        assert_eq!(j["name"], "my_hook");
        assert_eq!(j["has_before"], true);
        assert_eq!(j["has_after"], false);
    }

    #[test]
    fn hook_both_callbacks() {
        let h = EventHook::new("h".into())
            .before(Box::new(|_| json!("before")))
            .after(Box::new(|_| json!("after")));
        assert_eq!(h.run_before(&json!(null)), json!("before"));
        assert_eq!(h.run_after(&json!(null)), json!("after"));
    }

    // -- HookRegistry tests -----------------------------------------------

    #[test]
    fn registry_new_is_empty() {
        let r = HookRegistry::new();
        assert_eq!(r.len(), 0);
        assert!(r.is_empty());
    }

    #[test]
    fn registry_register_and_get() {
        let mut r = HookRegistry::new();
        r.register(EventHook::new("a".into()));
        assert_eq!(r.len(), 1);
        assert!(r.get("a").is_some());
        assert!(r.get("b").is_none());
    }

    #[test]
    fn registry_register_replaces_same_name() {
        let mut r = HookRegistry::new();
        r.register(EventHook::new("a".into()));
        r.register(EventHook::new("a".into()));
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn registry_remove() {
        let mut r = HookRegistry::new();
        r.register(EventHook::new("a".into()));
        r.remove("a");
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn registry_remove_nonexistent() {
        let mut r = HookRegistry::new();
        r.remove("nope"); // no panic
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn registry_run_all_before() {
        let mut r = HookRegistry::new();
        r.register(EventHook::new("add_a".into()).before(Box::new(|v| {
            let mut o = v.clone();
            o["a"] = json!(1);
            o
        })));
        r.register(EventHook::new("add_b".into()).before(Box::new(|v| {
            let mut o = v.clone();
            o["b"] = json!(2);
            o
        })));
        let out = r.run_all_before(&json!({}));
        assert_eq!(out["a"], json!(1));
        assert_eq!(out["b"], json!(2));
    }

    #[test]
    fn registry_run_all_after() {
        let mut r = HookRegistry::new();
        r.register(
            EventHook::new("wrap".into()).after(Box::new(|v| json!({"wrapped": v.clone()}))),
        );
        let out = r.run_all_after(&json!("data"));
        assert_eq!(out["wrapped"], json!("data"));
    }

    #[test]
    fn registry_default() {
        let r = HookRegistry::default();
        assert!(r.is_empty());
    }

    // -- EventMetrics tests -----------------------------------------------

    #[test]
    fn metrics_new_is_empty() {
        let m = EventMetrics::new();
        assert_eq!(m.total_events(), 0);
    }

    #[test]
    fn metrics_record_event() {
        let mut m = EventMetrics::new();
        m.record_event(&EventType::LlmStart);
        m.record_event(&EventType::LlmStart);
        m.record_event(&EventType::LlmEnd);
        assert_eq!(m.total_events(), 3);
        let per = m.events_per_type();
        assert_eq!(per.get("llm_start"), Some(&2));
        assert_eq!(per.get("llm_end"), Some(&1));
    }

    #[test]
    fn metrics_processing_time() {
        let mut m = EventMetrics::new();
        m.record_processing_time(&EventType::ToolStart, 100);
        m.record_processing_time(&EventType::ToolStart, 200);
        let avg = m.average_processing_time(&EventType::ToolStart).unwrap();
        assert!((avg - 150.0).abs() < f64::EPSILON);
    }

    #[test]
    fn metrics_average_no_samples() {
        let m = EventMetrics::new();
        assert!(m.average_processing_time(&EventType::Error).is_none());
    }

    #[test]
    fn metrics_to_json() {
        let mut m = EventMetrics::new();
        m.record_event(&EventType::ChainStart);
        let j = m.to_json();
        assert_eq!(j["total_events"], 1);
    }

    #[test]
    fn metrics_custom_event_type() {
        let mut m = EventMetrics::new();
        let ct = EventType::Custom("my_metric".into());
        m.record_event(&ct);
        m.record_processing_time(&ct, 50);
        assert_eq!(m.total_events(), 1);
        assert_eq!(m.average_processing_time(&ct), Some(50.0));
    }

    #[test]
    fn metrics_events_per_type_empty() {
        let m = EventMetrics::new();
        assert!(m.events_per_type().is_empty());
    }

    // -- Integration / combined tests -------------------------------------

    #[test]
    fn bus_and_log_integration() {
        let log = Rc::new(RefCell::new(EventLog::new()));
        let log_c = log.clone();
        let mut bus = EventBus::new();
        bus.subscribe(
            "logger".into(),
            vec![EventType::ChainStart, EventType::ChainEnd],
            Box::new(move |e| {
                log_c.borrow_mut().record(e.clone());
            }),
        );
        bus.publish(&Event::new(EventType::ChainStart, json!({"step": 1})));
        bus.publish(&Event::new(EventType::ChainEnd, json!({"step": 2})));
        bus.publish(&Event::new(EventType::Error, json!("ignored")));
        assert_eq!(log.borrow().len(), 2);
    }

    #[test]
    fn filter_and_log_integration() {
        let mut log = EventLog::new();
        log.record(
            Event::new(EventType::LlmStart, json!(null))
                .with_source("openai")
                .with_metadata("model", json!("gpt4")),
        );
        log.record(
            Event::new(EventType::LlmStart, json!(null))
                .with_source("anthropic")
                .with_metadata("model", json!("claude")),
        );
        log.record(Event::new(EventType::Error, json!(null)));

        let f = EventFilter::new()
            .by_type(EventType::LlmStart)
            .by_source("openai");
        let results = log.filter(&f);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].source.as_deref(), Some("openai"));
    }

    #[test]
    fn hook_registry_chaining() {
        let mut reg = HookRegistry::new();
        reg.register(
            EventHook::new("double".into())
                .before(Box::new(|v| {
                    let n = v.as_i64().unwrap_or(0);
                    json!(n * 2)
                }))
                .after(Box::new(|v| {
                    let n = v.as_i64().unwrap_or(0);
                    json!(n + 1)
                })),
        );
        reg.register(EventHook::new("add_ten".into()).before(Box::new(|v| {
            let n = v.as_i64().unwrap_or(0);
            json!(n + 10)
        })));
        // before: 5 -> double(5)=10 -> add_ten(10)=20
        let result = reg.run_all_before(&json!(5));
        assert_eq!(result, json!(20));

        // after: 100 -> double.after(100)=101 -> add_ten has no after -> 101
        let result = reg.run_all_after(&json!(100));
        assert_eq!(result, json!(101));
    }

    #[test]
    fn event_bus_custom_event_type() {
        let counter = Rc::new(RefCell::new(0));
        let c = counter.clone();
        let mut bus = EventBus::new();
        bus.subscribe(
            "custom_sub".into(),
            vec![EventType::Custom("my_custom".into())],
            Box::new(move |_| *c.borrow_mut() += 1),
        );
        bus.publish(&Event::new(
            EventType::Custom("my_custom".into()),
            json!(null),
        ));
        bus.publish(&Event::new(EventType::Custom("other".into()), json!(null)));
        assert_eq!(*counter.borrow(), 1);
    }

    #[test]
    fn event_serialization_roundtrip() {
        let e = Event::new(EventType::ToolEnd, json!({"result": "ok"}))
            .with_source("calculator")
            .with_metadata("duration_ms", json!(42));
        let json_str = serde_json::to_string(&e).unwrap();
        let e2: Event = serde_json::from_str(&json_str).unwrap();
        assert_eq!(e.id, e2.id);
        assert_eq!(e.event_type, e2.event_type);
        assert_eq!(e.data, e2.data);
        assert_eq!(e.source, e2.source);
    }

    #[test]
    fn filter_by_multiple_sources() {
        let f = EventFilter::new().by_source("a").by_source("b");
        let e1 = Event::new(EventType::ChainStart, json!(null)).with_source("a");
        let e2 = Event::new(EventType::ChainStart, json!(null)).with_source("b");
        let e3 = Event::new(EventType::ChainStart, json!(null)).with_source("c");
        assert!(f.matches(&e1));
        assert!(f.matches(&e2));
        assert!(!f.matches(&e3));
    }

    #[test]
    fn filter_multiple_metadata() {
        let f = EventFilter::new()
            .by_metadata("env", json!("prod"))
            .by_metadata("region", json!("us"));
        let e1 = Event::new(EventType::ChainStart, json!(null))
            .with_metadata("env", json!("prod"))
            .with_metadata("region", json!("us"));
        let e2 = Event::new(EventType::ChainStart, json!(null)).with_metadata("env", json!("prod"));
        assert!(f.matches(&e1));
        assert!(!f.matches(&e2)); // missing region
    }

    #[test]
    fn metrics_multiple_types_processing_time() {
        let mut m = EventMetrics::new();
        m.record_processing_time(&EventType::LlmStart, 10);
        m.record_processing_time(&EventType::LlmEnd, 20);
        m.record_processing_time(&EventType::LlmEnd, 40);
        assert_eq!(m.average_processing_time(&EventType::LlmStart), Some(10.0));
        assert_eq!(m.average_processing_time(&EventType::LlmEnd), Some(30.0));
    }
}
