//! Broadcast channel — pub/sub channel with topic filtering and history retention.
//!
//! The broadcast channel allows multiple publishers to send messages and multiple
//! subscribers to receive all published messages. It supports topic-based filtering,
//! configurable buffer sizes, delivery guarantees, and history retention.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use serde_json::Value;

use crate::errors::LangGraphError;

use super::base::BaseChannel;

/// Delivery guarantee for broadcast messages when the buffer is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeliveryGuarantee {
    /// Drop the message if the buffer is full.
    BestEffort,
    /// Block/wait until buffer has space (in sync context, acts like DropOldest).
    Guaranteed,
    /// Drop the oldest message when the buffer is full.
    DropOldest,
}

impl Default for DeliveryGuarantee {
    fn default() -> Self {
        Self::DropOldest
    }
}

/// Filter for broadcast subscriptions.
#[derive(Clone)]
pub struct BroadcastFilter {
    /// Only receive messages matching these topics. Empty means all topics.
    pub topics: Vec<String>,
    /// Custom filter function applied to the message payload.
    pub match_fn: Option<Arc<dyn Fn(&Value) -> bool + Send + Sync>>,
}

impl std::fmt::Debug for BroadcastFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BroadcastFilter")
            .field("topics", &self.topics)
            .field("match_fn", &self.match_fn.as_ref().map(|_| "<fn>"))
            .finish()
    }
}

impl BroadcastFilter {
    /// Create a filter that accepts messages on the given topics.
    pub fn topics(topics: Vec<String>) -> Self {
        Self {
            topics,
            match_fn: None,
        }
    }

    /// Create a filter with a custom match function.
    pub fn custom(f: Arc<dyn Fn(&Value) -> bool + Send + Sync>) -> Self {
        Self {
            topics: Vec::new(),
            match_fn: Some(f),
        }
    }

    /// Check if a message passes this filter.
    fn matches(&self, message: &BroadcastMessage) -> bool {
        // Check topic filter
        if !self.topics.is_empty() {
            match &message.topic {
                Some(topic) => {
                    if !self.topics.contains(topic) {
                        return false;
                    }
                }
                None => return false,
            }
        }

        // Check custom filter
        if let Some(ref f) = self.match_fn {
            if !f(&message.payload) {
                return false;
            }
        }

        true
    }
}

/// Configuration for a broadcast channel.
#[derive(Debug, Clone)]
pub struct BroadcastConfig {
    /// Maximum number of messages in each subscriber's buffer.
    pub buffer_size: usize,
    /// How many past messages to retain in history (0 = no history).
    pub history_size: usize,
    /// What to do when a subscriber's buffer is full.
    pub delivery_guarantee: DeliveryGuarantee,
    /// Optional default filter for new subscribers.
    pub filter: Option<BroadcastFilter>,
}

impl Default for BroadcastConfig {
    fn default() -> Self {
        Self {
            buffer_size: 100,
            history_size: 0,
            delivery_guarantee: DeliveryGuarantee::DropOldest,
            filter: None,
        }
    }
}

/// A message published through a broadcast channel.
#[derive(Debug, Clone)]
pub struct BroadcastMessage {
    /// Optional topic for topic-based filtering.
    pub topic: Option<String>,
    /// The message payload.
    pub payload: Value,
    /// The node/sender that published this message.
    pub sender: String,
    /// When the message was published.
    pub timestamp: Instant,
    /// Monotonically increasing sequence number.
    pub sequence: u64,
}

/// Internal shared state for the broadcast channel.
#[derive(Debug)]
struct BroadcastState {
    /// Per-subscriber message buffers, keyed by subscriber ID.
    subscriber_buffers: Vec<Arc<Mutex<VecDeque<BroadcastMessage>>>>,
    /// Per-subscriber filters.
    subscriber_filters: Vec<Option<BroadcastFilter>>,
    /// Message history for retention.
    history: VecDeque<BroadcastMessage>,
    /// Next sequence number.
    next_sequence: u64,
    /// Total messages published.
    message_count: u64,
    /// Configuration.
    config: BroadcastConfig,
}

/// A subscriber to a broadcast channel.
///
/// Receives messages published after subscription was created.
pub struct Subscriber {
    buffer: Arc<Mutex<VecDeque<BroadcastMessage>>>,
}

impl Subscriber {
    /// Receive the next message, removing it from the buffer.
    /// Returns `None` if no messages are available.
    pub fn receive(&self) -> Option<BroadcastMessage> {
        let mut buf = self.buffer.lock().unwrap();
        buf.pop_front()
    }

    /// Try to receive a message without blocking.
    /// Returns `None` if no messages are available.
    pub fn try_receive(&self) -> Option<BroadcastMessage> {
        self.receive()
    }

    /// Drain all pending messages from the buffer.
    pub fn drain(&self) -> Vec<BroadcastMessage> {
        let mut buf = self.buffer.lock().unwrap();
        buf.drain(..).collect()
    }
}

/// A pub/sub broadcast channel that allows multiple nodes to subscribe to and
/// receive values published by other nodes.
///
/// Unlike the step-based channels (`LastValue`, `Topic`, etc.), the broadcast
/// channel maintains independent subscriber buffers and supports topic-based
/// filtering, history retention, and configurable delivery guarantees.
#[derive(Debug, Clone)]
pub struct BroadcastChannel {
    /// The key name of this channel.
    key: String,
    /// Shared mutable state.
    state: Arc<Mutex<BroadcastState>>,
}

impl BroadcastChannel {
    /// Create a new broadcast channel with the given key and default configuration.
    pub fn new(key: impl Into<String>) -> Self {
        Self::with_config(key, BroadcastConfig::default())
    }

    /// Create a new broadcast channel with the given key and configuration.
    pub fn with_config(key: impl Into<String>, config: BroadcastConfig) -> Self {
        Self {
            key: key.into(),
            state: Arc::new(Mutex::new(BroadcastState {
                subscriber_buffers: Vec::new(),
                subscriber_filters: Vec::new(),
                history: VecDeque::new(),
                next_sequence: 0,
                message_count: 0,
                config,
            })),
        }
    }

    /// Publish a message to all subscribers.
    pub fn publish(&self, message: BroadcastMessage) -> Result<(), LangGraphError> {
        let mut state = self.state.lock().unwrap();

        let msg = BroadcastMessage {
            sequence: state.next_sequence,
            timestamp: message.timestamp,
            topic: message.topic,
            payload: message.payload,
            sender: message.sender,
        };
        state.next_sequence += 1;
        state.message_count += 1;

        // Retain in history if configured.
        if state.config.history_size > 0 {
            if state.history.len() >= state.config.history_size {
                state.history.pop_front();
            }
            state.history.push_back(msg.clone());
        }

        // Deliver to each subscriber.
        for (i, buf_arc) in state.subscriber_buffers.iter().enumerate() {
            // Check filter.
            if let Some(ref filter) = state.subscriber_filters[i] {
                if !filter.matches(&msg) {
                    continue;
                }
            }

            let mut buf = buf_arc.lock().unwrap();
            if buf.len() >= state.config.buffer_size {
                match state.config.delivery_guarantee {
                    DeliveryGuarantee::BestEffort => {
                        // Drop this message for this subscriber.
                        continue;
                    }
                    DeliveryGuarantee::Guaranteed | DeliveryGuarantee::DropOldest => {
                        // Drop the oldest message.
                        buf.pop_front();
                    }
                }
            }
            buf.push_back(msg.clone());
        }

        Ok(())
    }

    /// Publish a message with a specific topic.
    pub fn publish_to_topic(
        &self,
        topic: impl Into<String>,
        payload: Value,
        sender: impl Into<String>,
    ) -> Result<(), LangGraphError> {
        self.publish(BroadcastMessage {
            topic: Some(topic.into()),
            payload,
            sender: sender.into(),
            timestamp: Instant::now(),
            sequence: 0, // Will be overwritten by publish()
        })
    }

    /// Create a new subscription, optionally with a filter.
    pub fn subscribe(&self, filter: Option<BroadcastFilter>) -> Subscriber {
        let buf = Arc::new(Mutex::new(VecDeque::new()));
        let mut state = self.state.lock().unwrap();
        state.subscriber_buffers.push(buf.clone());
        state.subscriber_filters.push(filter);
        Subscriber { buffer: buf }
    }

    /// Get retained message history.
    pub fn get_history(&self) -> Vec<BroadcastMessage> {
        let state = self.state.lock().unwrap();
        state.history.iter().cloned().collect()
    }

    /// Get retained message history filtered by topic.
    pub fn get_history_by_topic(&self, topic: &str) -> Vec<BroadcastMessage> {
        let state = self.state.lock().unwrap();
        state
            .history
            .iter()
            .filter(|m| m.topic.as_deref() == Some(topic))
            .cloned()
            .collect()
    }

    /// Clear all retained history.
    pub fn clear_history(&self) {
        let mut state = self.state.lock().unwrap();
        state.history.clear();
    }

    /// Get the current number of subscribers.
    pub fn subscriber_count(&self) -> usize {
        let state = self.state.lock().unwrap();
        state.subscriber_buffers.len()
    }

    /// Get the total number of messages published.
    pub fn message_count(&self) -> u64 {
        let state = self.state.lock().unwrap();
        state.message_count
    }
}

impl BaseChannel for BroadcastChannel {
    fn key(&self) -> &str {
        &self.key
    }

    fn box_clone(&self) -> Box<dyn BaseChannel> {
        Box::new(self.clone())
    }

    fn checkpoint(&self) -> Option<Value> {
        let state = self.state.lock().unwrap();
        if state.history.is_empty() && state.message_count == 0 {
            return None;
        }
        let history: Vec<Value> = state
            .history
            .iter()
            .map(|m| {
                serde_json::json!({
                    "topic": m.topic,
                    "payload": m.payload,
                    "sender": m.sender,
                    "sequence": m.sequence,
                })
            })
            .collect();
        Some(serde_json::json!({
            "history": history,
            "message_count": state.message_count,
            "next_sequence": state.next_sequence,
        }))
    }

    fn from_checkpoint(&self, checkpoint: Option<Value>) -> Box<dyn BaseChannel> {
        let config = {
            let state = self.state.lock().unwrap();
            state.config.clone()
        };
        match checkpoint {
            Some(obj) => {
                let message_count = obj
                    .get("message_count")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let next_sequence = obj
                    .get("next_sequence")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                let history: VecDeque<BroadcastMessage> = obj
                    .get("history")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .map(|m| BroadcastMessage {
                                topic: m
                                    .get("topic")
                                    .and_then(|t| t.as_str())
                                    .map(|s| s.to_string()),
                                payload: m
                                    .get("payload")
                                    .cloned()
                                    .unwrap_or(Value::Null),
                                sender: m
                                    .get("sender")
                                    .and_then(|s| s.as_str())
                                    .unwrap_or("")
                                    .to_string(),
                                timestamp: Instant::now(),
                                sequence: m
                                    .get("sequence")
                                    .and_then(|s| s.as_u64())
                                    .unwrap_or(0),
                            })
                            .collect()
                    })
                    .unwrap_or_default();
                Box::new(BroadcastChannel {
                    key: self.key.clone(),
                    state: Arc::new(Mutex::new(BroadcastState {
                        subscriber_buffers: Vec::new(),
                        subscriber_filters: Vec::new(),
                        history,
                        next_sequence,
                        message_count,
                        config,
                    })),
                })
            }
            None => Box::new(BroadcastChannel::with_config(self.key.clone(), config)),
        }
    }

    fn get(&self) -> Result<Value, LangGraphError> {
        let state = self.state.lock().unwrap();
        if state.history.is_empty() {
            return Err(LangGraphError::EmptyChannelError);
        }
        let values: Vec<Value> = state
            .history
            .iter()
            .map(|m| {
                serde_json::json!({
                    "topic": m.topic,
                    "payload": m.payload,
                    "sender": m.sender,
                    "sequence": m.sequence,
                })
            })
            .collect();
        Ok(Value::Array(values))
    }

    fn is_available(&self) -> bool {
        let state = self.state.lock().unwrap();
        !state.history.is_empty() || state.message_count > 0
    }

    fn update(&mut self, values: Vec<Value>) -> Result<bool, LangGraphError> {
        if values.is_empty() {
            return Ok(false);
        }
        for v in values {
            let topic = v.get("topic").and_then(|t| t.as_str()).map(|s| s.to_string());
            let payload = v.get("payload").cloned().unwrap_or_else(|| v.clone());
            let sender = v
                .get("sender")
                .and_then(|s| s.as_str())
                .unwrap_or("unknown")
                .to_string();
            self.publish(BroadcastMessage {
                topic,
                payload,
                sender,
                timestamp: Instant::now(),
                sequence: 0,
            })?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_publish_and_receive_single_message() {
        let channel = BroadcastChannel::new("test");
        let sub = channel.subscribe(None);

        channel
            .publish(BroadcastMessage {
                topic: None,
                payload: Value::from("hello"),
                sender: "node_a".to_string(),
                timestamp: Instant::now(),
                sequence: 0,
            })
            .unwrap();

        let msg = sub.receive().unwrap();
        assert_eq!(msg.payload, Value::from("hello"));
        assert_eq!(msg.sender, "node_a");
        assert_eq!(msg.sequence, 0);
    }

    #[test]
    fn test_multiple_subscribers_receive_same_message() {
        let channel = BroadcastChannel::new("test");
        let sub1 = channel.subscribe(None);
        let sub2 = channel.subscribe(None);

        channel
            .publish(BroadcastMessage {
                topic: None,
                payload: Value::from(42),
                sender: "node_a".to_string(),
                timestamp: Instant::now(),
                sequence: 0,
            })
            .unwrap();

        let msg1 = sub1.receive().unwrap();
        let msg2 = sub2.receive().unwrap();
        assert_eq!(msg1.payload, Value::from(42));
        assert_eq!(msg2.payload, Value::from(42));
    }

    #[test]
    fn test_topic_filtering() {
        let channel = BroadcastChannel::new("test");
        let sub = channel.subscribe(Some(BroadcastFilter::topics(vec![
            "events".to_string(),
        ])));

        // This message has a non-matching topic.
        channel
            .publish(BroadcastMessage {
                topic: Some("logs".to_string()),
                payload: Value::from("log msg"),
                sender: "node_a".to_string(),
                timestamp: Instant::now(),
                sequence: 0,
            })
            .unwrap();

        // This message matches.
        channel
            .publish(BroadcastMessage {
                topic: Some("events".to_string()),
                payload: Value::from("event msg"),
                sender: "node_b".to_string(),
                timestamp: Instant::now(),
                sequence: 0,
            })
            .unwrap();

        let msg = sub.receive().unwrap();
        assert_eq!(msg.payload, Value::from("event msg"));
        assert!(sub.receive().is_none());
    }

    #[test]
    fn test_custom_filter_function() {
        let channel = BroadcastChannel::new("test");
        let filter = BroadcastFilter::custom(Arc::new(|v: &Value| {
            v.as_i64().map(|n| n > 10).unwrap_or(false)
        }));
        let sub = channel.subscribe(Some(filter));

        channel
            .publish(BroadcastMessage {
                topic: None,
                payload: Value::from(5),
                sender: "a".to_string(),
                timestamp: Instant::now(),
                sequence: 0,
            })
            .unwrap();

        channel
            .publish(BroadcastMessage {
                topic: None,
                payload: Value::from(20),
                sender: "b".to_string(),
                timestamp: Instant::now(),
                sequence: 0,
            })
            .unwrap();

        let msg = sub.receive().unwrap();
        assert_eq!(msg.payload, Value::from(20));
        assert!(sub.receive().is_none());
    }

    #[test]
    fn test_history_retention() {
        let config = BroadcastConfig {
            history_size: 3,
            ..Default::default()
        };
        let channel = BroadcastChannel::with_config("test", config);

        for i in 0..5 {
            channel
                .publish(BroadcastMessage {
                    topic: None,
                    payload: Value::from(i),
                    sender: "a".to_string(),
                    timestamp: Instant::now(),
                    sequence: 0,
                })
                .unwrap();
        }

        let history = channel.get_history();
        assert_eq!(history.len(), 3);
        // Should have messages 2, 3, 4 (oldest dropped).
        assert_eq!(history[0].payload, Value::from(2));
        assert_eq!(history[1].payload, Value::from(3));
        assert_eq!(history[2].payload, Value::from(4));
    }

    #[test]
    fn test_history_by_topic() {
        let config = BroadcastConfig {
            history_size: 10,
            ..Default::default()
        };
        let channel = BroadcastChannel::with_config("test", config);

        channel
            .publish_to_topic("events", Value::from("e1"), "a")
            .unwrap();
        channel
            .publish_to_topic("logs", Value::from("l1"), "a")
            .unwrap();
        channel
            .publish_to_topic("events", Value::from("e2"), "b")
            .unwrap();

        let events = channel.get_history_by_topic("events");
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].payload, Value::from("e1"));
        assert_eq!(events[1].payload, Value::from("e2"));
    }

    #[test]
    fn test_buffer_overflow_drop_oldest() {
        let config = BroadcastConfig {
            buffer_size: 3,
            delivery_guarantee: DeliveryGuarantee::DropOldest,
            ..Default::default()
        };
        let channel = BroadcastChannel::with_config("test", config);
        let sub = channel.subscribe(None);

        for i in 0..5 {
            channel
                .publish(BroadcastMessage {
                    topic: None,
                    payload: Value::from(i),
                    sender: "a".to_string(),
                    timestamp: Instant::now(),
                    sequence: 0,
                })
                .unwrap();
        }

        let messages = sub.drain();
        assert_eq!(messages.len(), 3);
        // Oldest dropped, should have 2, 3, 4.
        assert_eq!(messages[0].payload, Value::from(2));
        assert_eq!(messages[1].payload, Value::from(3));
        assert_eq!(messages[2].payload, Value::from(4));
    }

    #[test]
    fn test_buffer_overflow_best_effort() {
        let config = BroadcastConfig {
            buffer_size: 2,
            delivery_guarantee: DeliveryGuarantee::BestEffort,
            ..Default::default()
        };
        let channel = BroadcastChannel::with_config("test", config);
        let sub = channel.subscribe(None);

        for i in 0..5 {
            channel
                .publish(BroadcastMessage {
                    topic: None,
                    payload: Value::from(i),
                    sender: "a".to_string(),
                    timestamp: Instant::now(),
                    sequence: 0,
                })
                .unwrap();
        }

        let messages = sub.drain();
        // Only first 2 messages should be in buffer; rest dropped.
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].payload, Value::from(0));
        assert_eq!(messages[1].payload, Value::from(1));
    }

    #[test]
    fn test_sequence_number_monotonicity() {
        let channel = BroadcastChannel::new("test");
        let sub = channel.subscribe(None);

        for _ in 0..5 {
            channel
                .publish(BroadcastMessage {
                    topic: None,
                    payload: Value::from(1),
                    sender: "a".to_string(),
                    timestamp: Instant::now(),
                    sequence: 0,
                })
                .unwrap();
        }

        let messages = sub.drain();
        for (i, msg) in messages.iter().enumerate() {
            assert_eq!(msg.sequence, i as u64);
        }
    }

    #[test]
    fn test_subscriber_count_tracking() {
        let channel = BroadcastChannel::new("test");
        assert_eq!(channel.subscriber_count(), 0);

        let _sub1 = channel.subscribe(None);
        assert_eq!(channel.subscriber_count(), 1);

        let _sub2 = channel.subscribe(None);
        assert_eq!(channel.subscriber_count(), 2);
    }

    #[test]
    fn test_message_count_tracking() {
        let channel = BroadcastChannel::new("test");
        assert_eq!(channel.message_count(), 0);

        channel
            .publish(BroadcastMessage {
                topic: None,
                payload: Value::from(1),
                sender: "a".to_string(),
                timestamp: Instant::now(),
                sequence: 0,
            })
            .unwrap();
        assert_eq!(channel.message_count(), 1);

        channel
            .publish(BroadcastMessage {
                topic: None,
                payload: Value::from(2),
                sender: "b".to_string(),
                timestamp: Instant::now(),
                sequence: 0,
            })
            .unwrap();
        assert_eq!(channel.message_count(), 2);
    }

    #[test]
    fn test_clear_history() {
        let config = BroadcastConfig {
            history_size: 10,
            ..Default::default()
        };
        let channel = BroadcastChannel::with_config("test", config);

        channel
            .publish(BroadcastMessage {
                topic: None,
                payload: Value::from("data"),
                sender: "a".to_string(),
                timestamp: Instant::now(),
                sequence: 0,
            })
            .unwrap();

        assert!(!channel.get_history().is_empty());
        channel.clear_history();
        assert!(channel.get_history().is_empty());
    }

    #[test]
    fn test_config_defaults() {
        let config = BroadcastConfig::default();
        assert_eq!(config.buffer_size, 100);
        assert_eq!(config.history_size, 0);
        assert_eq!(config.delivery_guarantee, DeliveryGuarantee::DropOldest);
        assert!(config.filter.is_none());
    }

    #[test]
    fn test_drain_all_pending_messages() {
        let channel = BroadcastChannel::new("test");
        let sub = channel.subscribe(None);

        for i in 0..3 {
            channel
                .publish(BroadcastMessage {
                    topic: None,
                    payload: Value::from(i),
                    sender: "a".to_string(),
                    timestamp: Instant::now(),
                    sequence: 0,
                })
                .unwrap();
        }

        let messages = sub.drain();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].payload, Value::from(0));
        assert_eq!(messages[1].payload, Value::from(1));
        assert_eq!(messages[2].payload, Value::from(2));

        // Buffer should be empty after drain.
        assert!(sub.receive().is_none());
    }

    #[test]
    fn test_empty_channel_receive() {
        let channel = BroadcastChannel::new("test");
        let sub = channel.subscribe(None);

        assert!(sub.receive().is_none());
        assert!(sub.try_receive().is_none());
        assert!(sub.drain().is_empty());
    }
}
