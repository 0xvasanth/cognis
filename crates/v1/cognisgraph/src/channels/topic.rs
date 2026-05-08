//! Topic channel — pub/sub style channel that collects values into a list,
//! with support for topic-based message routing, filtering, and dead letter queues.
//!
//! The topic channel stores a list of values. If `accumulate` is `false` (default),
//! the list is cleared at the start of each step before new values are added.
//! If `accumulate` is `true`, values persist across steps.
//!
//! # Extended Topic Types
//!
//! Beyond the basic [`Topic`] channel, this module provides:
//!
//! - [`TopicMessage`] — A structured message with topic, payload, sender, and metadata.
//! - [`TopicFilter`] — Filter messages by exact topic, prefix, glob pattern, or match all.
//! - [`TopicChannel`] — A buffered pub/sub channel supporting publish, subscribe, and consume.
//! - [`TopicRouter`] — Routes messages to named handlers based on filter rules.
//! - [`DeadLetterQueue`] — Stores messages that could not be routed.
//! - [`TopicBus`] — Combines channel, router, and DLQ into a unified message bus.

use std::collections::HashSet;
use std::time::{Duration, Instant};

use serde_json::Value;

use crate::errors::LangGraphError;

use super::base::BaseChannel;

// ---------------------------------------------------------------------------
// Original Topic channel (BaseChannel implementation)
// ---------------------------------------------------------------------------

/// A pub/sub channel that collects values into a list.
///
/// When `accumulate` is `false`, the values list is cleared at each step via
/// [`consume`](BaseChannel::consume) before new values are added. When `true`,
/// values persist and accumulate across steps.
#[derive(Debug, Clone)]
pub struct Topic {
    /// The key name of this channel.
    key: String,
    /// The collected values.
    values: Vec<Value>,
    /// Whether to accumulate values across steps.
    accumulate: bool,
}

impl Topic {
    /// Create a new `Topic` channel with the given key.
    ///
    /// By default, `accumulate` is `false`, meaning the channel clears
    /// at each step.
    pub fn new(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            values: Vec::new(),
            accumulate: false,
        }
    }

    /// Create a new accumulating `Topic` channel.
    pub fn accumulating(key: impl Into<String>) -> Self {
        Self {
            key: key.into(),
            values: Vec::new(),
            accumulate: true,
        }
    }

    /// Set whether the channel should accumulate values.
    pub fn with_accumulate(mut self, accumulate: bool) -> Self {
        self.accumulate = accumulate;
        self
    }
}

impl BaseChannel for Topic {
    fn key(&self) -> &str {
        &self.key
    }

    fn box_clone(&self) -> Box<dyn BaseChannel> {
        Box::new(self.clone())
    }

    fn checkpoint(&self) -> Option<Value> {
        if self.values.is_empty() {
            None
        } else {
            Some(Value::Array(self.values.clone()))
        }
    }

    fn from_checkpoint(&self, checkpoint: Option<Value>) -> Box<dyn BaseChannel> {
        let values = match checkpoint {
            Some(Value::Array(arr)) => arr,
            Some(other) => vec![other],
            None => Vec::new(),
        };
        Box::new(Topic {
            key: self.key.clone(),
            values,
            accumulate: self.accumulate,
        })
    }

    fn get(&self) -> Result<Value, LangGraphError> {
        if self.values.is_empty() {
            Err(LangGraphError::EmptyChannelError)
        } else {
            Ok(Value::Array(self.values.clone()))
        }
    }

    fn is_available(&self) -> bool {
        !self.values.is_empty()
    }

    fn update(&mut self, values: Vec<Value>) -> Result<bool, LangGraphError> {
        if values.is_empty() {
            return Ok(false);
        }

        if !self.accumulate {
            self.values.clear();
        }

        // Flatten any arrays in the input values.
        for v in values {
            match v {
                Value::Array(arr) => self.values.extend(arr),
                other => self.values.push(other),
            }
        }
        Ok(true)
    }

    fn consume(&mut self) -> bool {
        if !self.accumulate && !self.values.is_empty() {
            self.values.clear();
            return true;
        }
        false
    }
}

// ---------------------------------------------------------------------------
// TopicMessage
// ---------------------------------------------------------------------------

/// A structured message for topic-based pub/sub communication between graph nodes.
#[derive(Debug, Clone)]
pub struct TopicMessage {
    /// The topic this message belongs to.
    pub topic: String,
    /// The message payload.
    pub payload: Value,
    /// Optional sender identifier.
    pub sender: Option<String>,
    /// When the message was created.
    pub timestamp: Instant,
    /// Monotonically increasing sequence number (assigned by the channel).
    pub sequence: u64,
}

impl TopicMessage {
    /// Create a new `TopicMessage` with the given topic and payload.
    pub fn new(topic: impl Into<String>, payload: Value) -> Self {
        Self {
            topic: topic.into(),
            payload,
            sender: None,
            timestamp: Instant::now(),
            sequence: 0,
        }
    }

    /// Set the sender on this message (builder pattern).
    pub fn with_sender(mut self, sender: impl Into<String>) -> Self {
        self.sender = Some(sender.into());
        self
    }

    /// Return the elapsed time since this message was created.
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }

    /// Serialize this message to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "topic": self.topic,
            "payload": self.payload,
            "sender": self.sender,
            "sequence": self.sequence,
        })
    }
}

// ---------------------------------------------------------------------------
// TopicFilter
// ---------------------------------------------------------------------------

/// Filter for selecting messages by topic name.
#[derive(Debug, Clone)]
pub enum TopicFilter {
    /// Match messages whose topic equals the given string exactly.
    Exact(String),
    /// Match messages whose topic starts with the given prefix.
    Prefix(String),
    /// Match messages whose topic matches a glob-style pattern (`*` and `?`).
    Pattern(String),
    /// Match all messages regardless of topic.
    All,
}

impl TopicFilter {
    /// Returns `true` if `topic` matches this filter.
    pub fn matches(&self, topic: &str) -> bool {
        match self {
            TopicFilter::Exact(s) => topic == s,
            TopicFilter::Prefix(p) => topic.starts_with(p.as_str()),
            TopicFilter::Pattern(pattern) => glob_match(pattern, topic),
            TopicFilter::All => true,
        }
    }
}

/// Simple glob matching supporting `*` (any sequence) and `?` (single char).
fn glob_match(pattern: &str, text: &str) -> bool {
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();
    glob_match_inner(&pat, &txt)
}

fn glob_match_inner(pat: &[char], txt: &[char]) -> bool {
    if pat.is_empty() {
        return txt.is_empty();
    }
    match pat[0] {
        '*' => {
            // '*' matches zero or more characters
            for i in 0..=txt.len() {
                if glob_match_inner(&pat[1..], &txt[i..]) {
                    return true;
                }
            }
            false
        }
        '?' => {
            if txt.is_empty() {
                false
            } else {
                glob_match_inner(&pat[1..], &txt[1..])
            }
        }
        c => {
            if txt.is_empty() || txt[0] != c {
                false
            } else {
                glob_match_inner(&pat[1..], &txt[1..])
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TopicChannel
// ---------------------------------------------------------------------------

/// Default maximum number of messages in a `TopicChannel`.
const DEFAULT_CHANNEL_CAPACITY: usize = 1024;

/// A buffered pub/sub channel that stores [`TopicMessage`]s and supports
/// filtered subscribe and consume operations.
#[derive(Debug)]
pub struct TopicChannel {
    messages: Vec<TopicMessage>,
    max_messages: usize,
    next_sequence: u64,
}

impl TopicChannel {
    /// Create a new `TopicChannel` with the default capacity.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
            max_messages: DEFAULT_CHANNEL_CAPACITY,
            next_sequence: 0,
        }
    }

    /// Create a new `TopicChannel` with the specified maximum capacity.
    pub fn with_capacity(max_messages: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_messages: max_messages.max(1),
            next_sequence: 0,
        }
    }

    /// Publish a message to the channel. If at capacity, the oldest message
    /// is evicted.
    pub fn publish(&mut self, mut message: TopicMessage) {
        message.sequence = self.next_sequence;
        self.next_sequence += 1;

        if self.messages.len() >= self.max_messages {
            self.messages.remove(0);
        }
        self.messages.push(message);
    }

    /// Return references to all messages matching the given filter.
    pub fn subscribe(&self, filter: TopicFilter) -> Vec<&TopicMessage> {
        self.messages
            .iter()
            .filter(|m| filter.matches(&m.topic))
            .collect()
    }

    /// Remove and return all messages matching the given filter.
    pub fn consume(&mut self, filter: TopicFilter) -> Vec<TopicMessage> {
        let mut consumed = Vec::new();
        let mut remaining = Vec::new();
        for msg in self.messages.drain(..) {
            if filter.matches(&msg.topic) {
                consumed.push(msg);
            } else {
                remaining.push(msg);
            }
        }
        self.messages = remaining;
        consumed
    }

    /// Peek at the latest message for a given topic.
    pub fn peek(&self, topic: &str) -> Option<&TopicMessage> {
        self.messages.iter().rev().find(|m| m.topic == topic)
    }

    /// Return the unique topic names present in the channel.
    pub fn topics(&self) -> Vec<String> {
        let set: HashSet<&str> = self.messages.iter().map(|m| m.topic.as_str()).collect();
        let mut topics: Vec<String> = set.into_iter().map(String::from).collect();
        topics.sort();
        topics
    }

    /// Return the total number of messages in the channel.
    pub fn message_count(&self) -> usize {
        self.messages.len()
    }

    /// Return `true` if the channel has no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Remove all messages from the channel.
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

impl Default for TopicChannel {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// TopicRouter
// ---------------------------------------------------------------------------

/// A routing rule mapping a filter to a handler name.
#[derive(Debug, Clone)]
struct Route {
    filter: TopicFilter,
    handler_name: String,
}

/// Routes [`TopicMessage`]s to named handlers based on registered filter rules.
#[derive(Debug, Clone)]
pub struct TopicRouter {
    routes: Vec<Route>,
}

impl TopicRouter {
    /// Create an empty router.
    pub fn new() -> Self {
        Self { routes: Vec::new() }
    }

    /// Register a routing rule: messages matching `filter` will be routed to `handler_name`.
    pub fn add_route(&mut self, filter: TopicFilter, handler_name: impl Into<String>) {
        self.routes.push(Route {
            filter,
            handler_name: handler_name.into(),
        });
    }

    /// Return the handler names whose filters match the given message's topic.
    pub fn route(&self, message: &TopicMessage) -> Vec<String> {
        self.routes
            .iter()
            .filter(|r| r.filter.matches(&message.topic))
            .map(|r| r.handler_name.clone())
            .collect()
    }

    /// Remove all routes for the given handler name.
    pub fn remove_route(&mut self, handler_name: &str) {
        self.routes.retain(|r| r.handler_name != handler_name);
    }

    /// Return the registered handler names (in order, may contain duplicates
    /// if the same handler has multiple rules).
    pub fn routes(&self) -> Vec<String> {
        self.routes.iter().map(|r| r.handler_name.clone()).collect()
    }

    /// Return the number of routing rules.
    pub fn len(&self) -> usize {
        self.routes.len()
    }

    /// Return `true` if no routing rules are registered.
    pub fn is_empty(&self) -> bool {
        self.routes.is_empty()
    }
}

impl Default for TopicRouter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// DeadLetterQueue
// ---------------------------------------------------------------------------

/// Stores messages that could not be routed to any handler.
#[derive(Debug)]
pub struct DeadLetterQueue {
    messages: Vec<TopicMessage>,
    capacity: usize,
}

impl DeadLetterQueue {
    /// Create a new `DeadLetterQueue` with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            messages: Vec::new(),
            capacity: capacity.max(1),
        }
    }

    /// Add a message to the dead letter queue. If at capacity, the oldest
    /// message is evicted.
    pub fn add(&mut self, message: TopicMessage) {
        if self.messages.len() >= self.capacity {
            self.messages.remove(0);
        }
        self.messages.push(message);
    }

    /// Drain all messages from the dead letter queue.
    pub fn drain(&mut self) -> Vec<TopicMessage> {
        self.messages.drain(..).collect()
    }

    /// Return the number of messages in the dead letter queue.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Return `true` if the dead letter queue is empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

// ---------------------------------------------------------------------------
// TopicBus
// ---------------------------------------------------------------------------

/// Statistics for a [`TopicBus`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopicBusStats {
    /// Total messages currently in the channel.
    pub message_count: usize,
    /// Number of distinct topics in the channel.
    pub topic_count: usize,
    /// Number of routing rules.
    pub route_count: usize,
    /// Number of messages in the dead letter queue.
    pub dlq_count: usize,
}

/// Default dead-letter-queue capacity.
const DEFAULT_DLQ_CAPACITY: usize = 256;

/// A unified message bus combining a [`TopicChannel`], [`TopicRouter`], and
/// [`DeadLetterQueue`].
///
/// When a message is published, the bus routes it through the router. If no
/// handler matches, the message is placed in the dead letter queue.
#[derive(Debug)]
pub struct TopicBus {
    channel: TopicChannel,
    router: TopicRouter,
    dlq: DeadLetterQueue,
}

impl TopicBus {
    /// Create a new `TopicBus` with default capacities.
    pub fn new() -> Self {
        Self {
            channel: TopicChannel::new(),
            router: TopicRouter::new(),
            dlq: DeadLetterQueue::new(DEFAULT_DLQ_CAPACITY),
        }
    }

    /// Create a new `TopicBus` with the given channel and DLQ capacities.
    pub fn with_config(channel_capacity: usize, dlq_capacity: usize) -> Self {
        Self {
            channel: TopicChannel::with_capacity(channel_capacity),
            router: TopicRouter::new(),
            dlq: DeadLetterQueue::new(dlq_capacity),
        }
    }

    /// Register a routing rule on the underlying router.
    pub fn add_route(&mut self, filter: TopicFilter, handler_name: impl Into<String>) {
        self.router.add_route(filter, handler_name);
    }

    /// Publish a message to the bus. The message is stored in the channel and
    /// routed through the router. Returns the list of handler names that
    /// matched. If no handler matched, the message is added to the DLQ.
    pub fn publish(&mut self, message: TopicMessage) -> Vec<String> {
        let handlers = self.router.route(&message);
        if handlers.is_empty() {
            self.dlq.add(message.clone());
        }
        self.channel.publish(message);
        handlers
    }

    /// Return references to messages matching the given filter.
    pub fn subscribe(&self, filter: TopicFilter) -> Vec<&TopicMessage> {
        self.channel.subscribe(filter)
    }

    /// Return a reference to the dead letter queue.
    pub fn dead_letters(&self) -> &DeadLetterQueue {
        &self.dlq
    }

    /// Return a mutable reference to the dead letter queue.
    pub fn dead_letters_mut(&mut self) -> &mut DeadLetterQueue {
        &mut self.dlq
    }

    /// Return current statistics for the bus.
    pub fn stats(&self) -> TopicBusStats {
        TopicBusStats {
            message_count: self.channel.message_count(),
            topic_count: self.channel.topics().len(),
            route_count: self.router.len(),
            dlq_count: self.dlq.len(),
        }
    }
}

impl Default for TopicBus {
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

    // -- Topic (BaseChannel) tests ------------------------------------------

    #[test]
    fn test_new_channel_is_empty() {
        let ch = Topic::new("events");
        assert!(!ch.is_available());
        assert!(ch.get().is_err());
    }

    #[test]
    fn test_update_single_value() {
        let mut ch = Topic::new("events");
        ch.update(vec![Value::from("event1")]).unwrap();
        assert_eq!(ch.get().unwrap(), Value::Array(vec![Value::from("event1")]));
    }

    #[test]
    fn test_update_multiple_values() {
        let mut ch = Topic::new("events");
        ch.update(vec![Value::from("a"), Value::from("b")]).unwrap();
        assert_eq!(
            ch.get().unwrap(),
            Value::Array(vec![Value::from("a"), Value::from("b")])
        );
    }

    #[test]
    fn test_flatten_arrays_in_input() {
        let mut ch = Topic::new("events");
        ch.update(vec![Value::Array(vec![Value::from(1), Value::from(2)])])
            .unwrap();
        assert_eq!(
            ch.get().unwrap(),
            Value::Array(vec![Value::from(1), Value::from(2)])
        );
    }

    #[test]
    fn test_non_accumulate_clears_on_update() {
        let mut ch = Topic::new("events");
        ch.update(vec![Value::from("first")]).unwrap();
        ch.update(vec![Value::from("second")]).unwrap();
        assert_eq!(ch.get().unwrap(), Value::Array(vec![Value::from("second")]));
    }

    #[test]
    fn test_accumulate_mode() {
        let mut ch = Topic::accumulating("events");
        ch.update(vec![Value::from("first")]).unwrap();
        ch.update(vec![Value::from("second")]).unwrap();
        assert_eq!(
            ch.get().unwrap(),
            Value::Array(vec![Value::from("first"), Value::from("second")])
        );
    }

    #[test]
    fn test_consume_clears_non_accumulate() {
        let mut ch = Topic::new("events");
        ch.update(vec![Value::from("data")]).unwrap();
        assert!(!ch.values.is_empty());

        let changed = ch.consume();
        assert!(changed);
        assert!(!ch.is_available());
        assert!(ch.get().is_err());
    }

    #[test]
    fn test_consume_noop_on_accumulate() {
        let mut ch = Topic::accumulating("events");
        ch.update(vec![Value::from("data")]).unwrap();

        let changed = ch.consume();
        assert!(!changed);
        assert_eq!(ch.get().unwrap(), Value::Array(vec![Value::from("data")]));
    }

    #[test]
    fn test_consume_noop_on_empty() {
        let mut ch = Topic::new("events");
        let changed = ch.consume();
        assert!(!changed);
    }

    #[test]
    fn test_empty_update() {
        let mut ch = Topic::new("events");
        let changed = ch.update(vec![]).unwrap();
        assert!(!changed);
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let mut ch = Topic::accumulating("events");
        ch.update(vec![Value::from("a"), Value::from("b")]).unwrap();

        let ckpt = ch.checkpoint();
        assert!(ckpt.is_some());

        let restored = ch.from_checkpoint(ckpt);
        assert_eq!(
            restored.get().unwrap(),
            Value::Array(vec![Value::from("a"), Value::from("b")])
        );
    }

    #[test]
    fn test_checkpoint_empty() {
        let ch = Topic::new("events");
        assert!(ch.checkpoint().is_none());
    }

    #[test]
    fn test_from_checkpoint_none() {
        let ch = Topic::new("events");
        let restored = ch.from_checkpoint(None);
        assert!(!restored.is_available());
        assert!(restored.get().is_err());
    }

    #[test]
    fn test_with_accumulate_builder() {
        let ch = Topic::new("events").with_accumulate(true);
        assert!(ch.accumulate);
    }

    // -- TopicMessage tests -------------------------------------------------

    #[test]
    fn test_topic_message_new() {
        let msg = TopicMessage::new("orders", Value::from(42));
        assert_eq!(msg.topic, "orders");
        assert_eq!(msg.payload, Value::from(42));
        assert!(msg.sender.is_none());
        assert_eq!(msg.sequence, 0);
    }

    #[test]
    fn test_topic_message_with_sender() {
        let msg = TopicMessage::new("orders", Value::from("data")).with_sender("node_a");
        assert_eq!(msg.sender, Some("node_a".to_string()));
    }

    #[test]
    fn test_topic_message_age() {
        let msg = TopicMessage::new("test", Value::Null);
        // Age should be very small (sub-millisecond) right after creation.
        assert!(msg.age() < Duration::from_secs(1));
    }

    #[test]
    fn test_topic_message_to_json() {
        let msg = TopicMessage::new("events", Value::from("hello")).with_sender("sender1");
        let json = msg.to_json();
        assert_eq!(json["topic"], "events");
        assert_eq!(json["payload"], "hello");
        assert_eq!(json["sender"], "sender1");
        assert_eq!(json["sequence"], 0);
    }

    #[test]
    fn test_topic_message_to_json_no_sender() {
        let msg = TopicMessage::new("events", Value::from(99));
        let json = msg.to_json();
        assert_eq!(json["topic"], "events");
        assert!(json["sender"].is_null());
    }

    // -- TopicFilter tests --------------------------------------------------

    #[test]
    fn test_filter_exact_match() {
        let f = TopicFilter::Exact("orders.created".to_string());
        assert!(f.matches("orders.created"));
        assert!(!f.matches("orders.updated"));
        assert!(!f.matches("orders"));
    }

    #[test]
    fn test_filter_prefix_match() {
        let f = TopicFilter::Prefix("orders.".to_string());
        assert!(f.matches("orders.created"));
        assert!(f.matches("orders.updated"));
        assert!(!f.matches("events.created"));
        assert!(!f.matches("orders"));
    }

    #[test]
    fn test_filter_pattern_star() {
        let f = TopicFilter::Pattern("orders.*".to_string());
        assert!(f.matches("orders.created"));
        assert!(f.matches("orders."));
        assert!(!f.matches("events.created"));
    }

    #[test]
    fn test_filter_pattern_question() {
        let f = TopicFilter::Pattern("log?".to_string());
        assert!(f.matches("logs"));
        assert!(f.matches("logx"));
        assert!(!f.matches("log"));
        assert!(!f.matches("logss"));
    }

    #[test]
    fn test_filter_pattern_complex() {
        let f = TopicFilter::Pattern("*.events.*".to_string());
        assert!(f.matches("us.events.created"));
        assert!(f.matches("eu.events.deleted"));
        assert!(!f.matches("events.created"));
    }

    #[test]
    fn test_filter_all() {
        let f = TopicFilter::All;
        assert!(f.matches("anything"));
        assert!(f.matches(""));
        assert!(f.matches("deeply.nested.topic.name"));
    }

    // -- TopicChannel tests -------------------------------------------------

    #[test]
    fn test_channel_new_is_empty() {
        let ch = TopicChannel::new();
        assert!(ch.is_empty());
        assert_eq!(ch.message_count(), 0);
        assert!(ch.topics().is_empty());
    }

    #[test]
    fn test_channel_publish_and_subscribe() {
        let mut ch = TopicChannel::new();
        ch.publish(TopicMessage::new("orders", Value::from(1)));
        ch.publish(TopicMessage::new("events", Value::from(2)));
        ch.publish(TopicMessage::new("orders", Value::from(3)));

        let orders = ch.subscribe(TopicFilter::Exact("orders".to_string()));
        assert_eq!(orders.len(), 2);
        assert_eq!(orders[0].payload, Value::from(1));
        assert_eq!(orders[1].payload, Value::from(3));
    }

    #[test]
    fn test_channel_subscribe_all() {
        let mut ch = TopicChannel::new();
        ch.publish(TopicMessage::new("a", Value::from(1)));
        ch.publish(TopicMessage::new("b", Value::from(2)));

        let all = ch.subscribe(TopicFilter::All);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn test_channel_consume_removes_messages() {
        let mut ch = TopicChannel::new();
        ch.publish(TopicMessage::new("orders", Value::from(1)));
        ch.publish(TopicMessage::new("events", Value::from(2)));
        ch.publish(TopicMessage::new("orders", Value::from(3)));

        let consumed = ch.consume(TopicFilter::Exact("orders".to_string()));
        assert_eq!(consumed.len(), 2);
        assert_eq!(ch.message_count(), 1);
        assert_eq!(ch.subscribe(TopicFilter::All)[0].payload, Value::from(2));
    }

    #[test]
    fn test_channel_consume_empties_buffer() {
        let mut ch = TopicChannel::new();
        ch.publish(TopicMessage::new("a", Value::from(1)));
        ch.publish(TopicMessage::new("a", Value::from(2)));

        let consumed = ch.consume(TopicFilter::All);
        assert_eq!(consumed.len(), 2);
        assert!(ch.is_empty());
    }

    #[test]
    fn test_channel_peek() {
        let mut ch = TopicChannel::new();
        ch.publish(TopicMessage::new("orders", Value::from(1)));
        ch.publish(TopicMessage::new("events", Value::from(2)));
        ch.publish(TopicMessage::new("orders", Value::from(3)));

        let latest = ch.peek("orders").unwrap();
        assert_eq!(latest.payload, Value::from(3));
    }

    #[test]
    fn test_channel_peek_missing_topic() {
        let ch = TopicChannel::new();
        assert!(ch.peek("nonexistent").is_none());
    }

    #[test]
    fn test_channel_topics() {
        let mut ch = TopicChannel::new();
        ch.publish(TopicMessage::new("b", Value::Null));
        ch.publish(TopicMessage::new("a", Value::Null));
        ch.publish(TopicMessage::new("b", Value::Null));

        let topics = ch.topics();
        assert_eq!(topics, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn test_channel_capacity_eviction() {
        let mut ch = TopicChannel::with_capacity(3);
        for i in 0..5 {
            ch.publish(TopicMessage::new("t", Value::from(i)));
        }
        assert_eq!(ch.message_count(), 3);
        let all = ch.subscribe(TopicFilter::All);
        // Oldest (0, 1) should have been evicted.
        assert_eq!(all[0].payload, Value::from(2));
        assert_eq!(all[1].payload, Value::from(3));
        assert_eq!(all[2].payload, Value::from(4));
    }

    #[test]
    fn test_channel_sequence_numbers() {
        let mut ch = TopicChannel::new();
        ch.publish(TopicMessage::new("a", Value::Null));
        ch.publish(TopicMessage::new("b", Value::Null));
        ch.publish(TopicMessage::new("c", Value::Null));

        let all = ch.subscribe(TopicFilter::All);
        assert_eq!(all[0].sequence, 0);
        assert_eq!(all[1].sequence, 1);
        assert_eq!(all[2].sequence, 2);
    }

    #[test]
    fn test_channel_clear() {
        let mut ch = TopicChannel::new();
        ch.publish(TopicMessage::new("t", Value::Null));
        ch.clear();
        assert!(ch.is_empty());
    }

    // -- TopicRouter tests --------------------------------------------------

    #[test]
    fn test_router_new_is_empty() {
        let r = TopicRouter::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }

    #[test]
    fn test_router_add_and_route() {
        let mut r = TopicRouter::new();
        r.add_route(TopicFilter::Exact("orders".to_string()), "order_handler");
        r.add_route(TopicFilter::Prefix("log".to_string()), "log_handler");

        let msg = TopicMessage::new("orders", Value::Null);
        let handlers = r.route(&msg);
        assert_eq!(handlers, vec!["order_handler".to_string()]);

        let msg2 = TopicMessage::new("logs", Value::Null);
        let handlers2 = r.route(&msg2);
        assert_eq!(handlers2, vec!["log_handler".to_string()]);
    }

    #[test]
    fn test_router_multi_route() {
        let mut r = TopicRouter::new();
        r.add_route(TopicFilter::All, "catch_all");
        r.add_route(TopicFilter::Exact("orders".to_string()), "order_handler");

        let msg = TopicMessage::new("orders", Value::Null);
        let handlers = r.route(&msg);
        assert_eq!(handlers.len(), 2);
        assert!(handlers.contains(&"catch_all".to_string()));
        assert!(handlers.contains(&"order_handler".to_string()));
    }

    #[test]
    fn test_router_no_match() {
        let mut r = TopicRouter::new();
        r.add_route(TopicFilter::Exact("orders".to_string()), "order_handler");

        let msg = TopicMessage::new("events", Value::Null);
        let handlers = r.route(&msg);
        assert!(handlers.is_empty());
    }

    #[test]
    fn test_router_remove_route() {
        let mut r = TopicRouter::new();
        r.add_route(TopicFilter::All, "handler_a");
        r.add_route(TopicFilter::All, "handler_b");
        assert_eq!(r.len(), 2);

        r.remove_route("handler_a");
        assert_eq!(r.len(), 1);
        assert_eq!(r.routes(), vec!["handler_b".to_string()]);
    }

    #[test]
    fn test_router_routes_list() {
        let mut r = TopicRouter::new();
        r.add_route(TopicFilter::All, "a");
        r.add_route(TopicFilter::All, "b");
        assert_eq!(r.routes(), vec!["a".to_string(), "b".to_string()]);
    }

    // -- DeadLetterQueue tests ----------------------------------------------

    #[test]
    fn test_dlq_new_is_empty() {
        let dlq = DeadLetterQueue::new(10);
        assert!(dlq.is_empty());
        assert_eq!(dlq.len(), 0);
    }

    #[test]
    fn test_dlq_add_and_drain() {
        let mut dlq = DeadLetterQueue::new(10);
        dlq.add(TopicMessage::new("t", Value::from(1)));
        dlq.add(TopicMessage::new("t", Value::from(2)));
        assert_eq!(dlq.len(), 2);

        let drained = dlq.drain();
        assert_eq!(drained.len(), 2);
        assert!(dlq.is_empty());
    }

    #[test]
    fn test_dlq_capacity_eviction() {
        let mut dlq = DeadLetterQueue::new(2);
        dlq.add(TopicMessage::new("t", Value::from(1)));
        dlq.add(TopicMessage::new("t", Value::from(2)));
        dlq.add(TopicMessage::new("t", Value::from(3)));

        assert_eq!(dlq.len(), 2);
        let drained = dlq.drain();
        assert_eq!(drained[0].payload, Value::from(2));
        assert_eq!(drained[1].payload, Value::from(3));
    }

    // -- TopicBus tests -----------------------------------------------------

    #[test]
    fn test_bus_new() {
        let bus = TopicBus::new();
        let stats = bus.stats();
        assert_eq!(stats.message_count, 0);
        assert_eq!(stats.topic_count, 0);
        assert_eq!(stats.route_count, 0);
        assert_eq!(stats.dlq_count, 0);
    }

    #[test]
    fn test_bus_publish_with_route() {
        let mut bus = TopicBus::new();
        bus.add_route(TopicFilter::Exact("orders".to_string()), "order_handler");

        let handlers = bus.publish(TopicMessage::new("orders", Value::from("new_order")));
        assert_eq!(handlers, vec!["order_handler".to_string()]);
        assert_eq!(bus.stats().message_count, 1);
        assert_eq!(bus.stats().dlq_count, 0);
    }

    #[test]
    fn test_bus_unrouted_goes_to_dlq() {
        let mut bus = TopicBus::new();
        bus.add_route(TopicFilter::Exact("orders".to_string()), "order_handler");

        let handlers = bus.publish(TopicMessage::new("events", Value::from("unrouted")));
        assert!(handlers.is_empty());
        assert_eq!(bus.stats().dlq_count, 1);
        assert_eq!(bus.dead_letters().len(), 1);
    }

    #[test]
    fn test_bus_subscribe() {
        let mut bus = TopicBus::new();
        bus.add_route(TopicFilter::All, "catch_all");
        bus.publish(TopicMessage::new("a", Value::from(1)));
        bus.publish(TopicMessage::new("b", Value::from(2)));

        let msgs = bus.subscribe(TopicFilter::Exact("a".to_string()));
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].payload, Value::from(1));
    }

    #[test]
    fn test_bus_stats_tracking() {
        let mut bus = TopicBus::new();
        bus.add_route(TopicFilter::Prefix("order".to_string()), "handler1");
        bus.add_route(TopicFilter::Exact("events".to_string()), "handler2");

        bus.publish(TopicMessage::new("orders", Value::from(1)));
        bus.publish(TopicMessage::new("events", Value::from(2)));
        bus.publish(TopicMessage::new("unknown", Value::from(3)));

        let stats = bus.stats();
        assert_eq!(stats.message_count, 3);
        assert_eq!(stats.topic_count, 3);
        assert_eq!(stats.route_count, 2);
        assert_eq!(stats.dlq_count, 1);
    }

    #[test]
    fn test_bus_with_config() {
        let mut bus = TopicBus::with_config(5, 3);
        // Publish 7 messages with no routes — all go to DLQ (cap 3) and channel (cap 5).
        for i in 0..7 {
            bus.publish(TopicMessage::new("t", Value::from(i)));
        }
        assert_eq!(bus.stats().message_count, 5); // channel cap
        assert_eq!(bus.stats().dlq_count, 3); // dlq cap
    }

    #[test]
    fn test_bus_dead_letters_drain() {
        let mut bus = TopicBus::new();
        bus.publish(TopicMessage::new("x", Value::from(1)));
        bus.publish(TopicMessage::new("y", Value::from(2)));

        let drained = bus.dead_letters_mut().drain();
        assert_eq!(drained.len(), 2);
        assert_eq!(bus.stats().dlq_count, 0);
    }

    #[test]
    fn test_bus_no_routes_empty() {
        let bus = TopicBus::new();
        let msgs = bus.subscribe(TopicFilter::All);
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_bus_end_to_end() {
        let mut bus = TopicBus::new();
        bus.add_route(TopicFilter::Prefix("order".to_string()), "order_service");
        bus.add_route(TopicFilter::Exact("logs".to_string()), "log_service");
        bus.add_route(TopicFilter::All, "audit");

        // Published to matching routes.
        let h1 = bus.publish(TopicMessage::new("orders.created", Value::from("o1")));
        assert_eq!(h1.len(), 2); // order_service + audit

        let h2 = bus.publish(TopicMessage::new("logs", Value::from("l1")));
        assert_eq!(h2.len(), 2); // log_service + audit

        // Still in channel.
        let all = bus.subscribe(TopicFilter::All);
        assert_eq!(all.len(), 2);

        // No DLQ entries since catch-all audit route matches everything.
        assert_eq!(bus.stats().dlq_count, 0);
    }
}
