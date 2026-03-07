//! Inter-agent communication system for message passing, shared state, and coordination.
//!
//! This module provides the building blocks for agents to communicate with each
//! other through direct messages, pub/sub channels, and shared key-value state.
//!
//! # Components
//!
//! - [`AgentMessage`] -- a structured message with priority, metadata, and reply tracking
//! - [`MessageFilter`] -- declarative filter for selecting messages by criteria
//! - [`Mailbox`] -- per-agent inbox for receiving messages
//! - [`SharedState`] -- shared key-value store with ownership tracking
//! - [`Channel`] -- named pub/sub channel for broadcasting to subscribers
//! - [`CommunicationHub`] -- central router that ties everything together
//!
//! # Example
//!
//! ```rust
//! use deepagents::communication::*;
//! use serde_json::json;
//!
//! let mut hub = CommunicationHub::new();
//! hub.register_agent("planner");
//! hub.register_agent("executor");
//!
//! let msg = AgentMessage::new("planner", "executor", "task", json!({"action": "search"}));
//! hub.send(msg).unwrap();
//!
//! let mailbox = hub.get_mailbox("executor").unwrap();
//! let messages = mailbox.read();
//! assert_eq!(messages.len(), 1);
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use serde_json::{json, Value};
use uuid::Uuid;

use crate::agent::DeepAgentError;

/// Result type alias for communication operations.
pub type Result<T> = std::result::Result<T, DeepAgentError>;

// ---------------------------------------------------------------------------
// MessagePriority
// ---------------------------------------------------------------------------

/// Priority level for inter-agent messages.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MessagePriority {
    /// Time-sensitive, should be processed immediately.
    Urgent,
    /// Default priority.
    Normal,
    /// Background or informational, can be deferred.
    Low,
}

impl MessagePriority {
    /// Returns a numeric rank for ordering (higher = more urgent).
    fn rank(self) -> u8 {
        match self {
            Self::Urgent => 2,
            Self::Normal => 1,
            Self::Low => 0,
        }
    }
}

impl PartialOrd for MessagePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for MessagePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank().cmp(&other.rank())
    }
}

impl fmt::Display for MessagePriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Urgent => write!(f, "Urgent"),
            Self::Normal => write!(f, "Normal"),
            Self::Low => write!(f, "Low"),
        }
    }
}

// ---------------------------------------------------------------------------
// AgentMessage
// ---------------------------------------------------------------------------

/// A message sent between agents.
#[derive(Debug, Clone)]
pub struct AgentMessage {
    /// Unique message identifier.
    pub id: String,
    /// Name of the sending agent.
    pub from: String,
    /// Name of the intended recipient agent.
    pub to: String,
    /// Short description of the message purpose.
    pub subject: String,
    /// Message payload.
    pub body: Value,
    /// Delivery priority.
    pub priority: MessagePriority,
    /// When the message was created.
    pub timestamp: Instant,
    /// If this is a reply, the id of the original message.
    pub reply_to: Option<String>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, Value>,
}

impl AgentMessage {
    /// Create a new message with default priority and auto-generated id.
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        subject: impl Into<String>,
        body: Value,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            from: from.into(),
            to: to.into(),
            subject: subject.into(),
            body,
            priority: MessagePriority::Normal,
            timestamp: Instant::now(),
            reply_to: None,
            metadata: HashMap::new(),
        }
    }

    /// Set the message priority (builder pattern).
    pub fn with_priority(mut self, priority: MessagePriority) -> Self {
        self.priority = priority;
        self
    }

    /// Mark this message as a reply to another message (builder pattern).
    pub fn with_reply_to(mut self, id: impl Into<String>) -> Self {
        self.reply_to = Some(id.into());
        self
    }

    /// Add a metadata entry (builder pattern).
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Returns `true` if this message is a reply to another.
    pub fn is_reply(&self) -> bool {
        self.reply_to.is_some()
    }

    /// Serialize the message to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "from": self.from,
            "to": self.to,
            "subject": self.subject,
            "body": self.body,
            "priority": self.priority.to_string(),
            "reply_to": self.reply_to,
            "metadata": self.metadata,
        })
    }
}

// ---------------------------------------------------------------------------
// MessageFilter
// ---------------------------------------------------------------------------

/// Declarative filter for selecting messages by various criteria.
///
/// All configured conditions must match (logical AND). An empty filter matches
/// everything.
#[derive(Debug, Clone, Default)]
pub struct MessageFilter {
    from: Option<String>,
    to: Option<String>,
    subject_contains: Option<String>,
    priority: Option<MessagePriority>,
}

impl MessageFilter {
    /// Create an empty filter that matches all messages.
    pub fn new() -> Self {
        Self::default()
    }

    /// Only match messages from the given agent.
    pub fn from_agent(mut self, name: impl Into<String>) -> Self {
        self.from = Some(name.into());
        self
    }

    /// Only match messages addressed to the given agent.
    pub fn to_agent(mut self, name: impl Into<String>) -> Self {
        self.to = Some(name.into());
        self
    }

    /// Only match messages whose subject contains the given text.
    pub fn subject_contains(mut self, text: impl Into<String>) -> Self {
        self.subject_contains = Some(text.into());
        self
    }

    /// Only match messages with the given priority.
    pub fn priority(mut self, p: MessagePriority) -> Self {
        self.priority = Some(p);
        self
    }

    /// Returns `true` if the message satisfies all configured conditions.
    pub fn matches(&self, msg: &AgentMessage) -> bool {
        if let Some(ref from) = self.from {
            if msg.from != *from {
                return false;
            }
        }
        if let Some(ref to) = self.to {
            if msg.to != *to {
                return false;
            }
        }
        if let Some(ref text) = self.subject_contains {
            if !msg.subject.contains(text.as_str()) {
                return false;
            }
        }
        if let Some(p) = self.priority {
            if msg.priority != p {
                return false;
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// Mailbox
// ---------------------------------------------------------------------------

/// Per-agent message inbox.
///
/// Messages are stored in delivery order and can be consumed (drained) or
/// peeked without removal.
#[derive(Debug)]
pub struct Mailbox {
    /// The agent that owns this mailbox.
    owner: String,
    /// Messages waiting to be read.
    inbox: Vec<AgentMessage>,
}

impl Mailbox {
    /// Create a new empty mailbox for the given agent.
    pub fn new(owner: &str) -> Self {
        Self {
            owner: owner.to_string(),
            inbox: Vec::new(),
        }
    }

    /// Deliver a message to this mailbox.
    pub fn deliver(&mut self, msg: AgentMessage) {
        self.inbox.push(msg);
    }

    /// Read and consume all messages, draining the inbox.
    pub fn read(&mut self) -> Vec<AgentMessage> {
        std::mem::take(&mut self.inbox)
    }

    /// View all messages without consuming them.
    pub fn peek(&self) -> Vec<&AgentMessage> {
        self.inbox.iter().collect()
    }

    /// Read and consume only messages that match the given filter.
    ///
    /// Non-matching messages remain in the inbox.
    pub fn read_filtered(&mut self, filter: &MessageFilter) -> Vec<AgentMessage> {
        let mut matched = Vec::new();
        let mut remaining = Vec::new();
        for msg in self.inbox.drain(..) {
            if filter.matches(&msg) {
                matched.push(msg);
            } else {
                remaining.push(msg);
            }
        }
        self.inbox = remaining;
        matched
    }

    /// Number of unread messages in the inbox.
    pub fn unread_count(&self) -> usize {
        self.inbox.len()
    }

    /// Returns `true` if the inbox has no messages.
    pub fn is_empty(&self) -> bool {
        self.inbox.is_empty()
    }

    /// The name of the agent that owns this mailbox.
    pub fn owner(&self) -> &str {
        &self.owner
    }
}

// ---------------------------------------------------------------------------
// SharedState
// ---------------------------------------------------------------------------

/// Shared key-value state store with ownership tracking.
///
/// Any agent can read all entries, but writes record which agent set the value.
#[derive(Debug, Default)]
pub struct SharedState {
    data: HashMap<String, Value>,
    owners: HashMap<String, String>,
}

impl SharedState {
    /// Create a new empty shared state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a key to a value, recording the owning agent.
    pub fn set(&mut self, key: &str, value: Value, owner: &str) {
        self.data.insert(key.to_string(), value);
        self.owners.insert(key.to_string(), owner.to_string());
    }

    /// Get the value for a key, if it exists.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }

    /// Get the agent that last set the given key.
    pub fn get_owner(&self, key: &str) -> Option<&str> {
        self.owners.get(key).map(|s| s.as_str())
    }

    /// Remove a key, returning its value if it existed.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.owners.remove(key);
        self.data.remove(key)
    }

    /// All keys currently in the store.
    pub fn keys(&self) -> Vec<&str> {
        self.data.keys().map(|s| s.as_str()).collect()
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Return all entries set by a particular agent.
    pub fn entries_by_owner(&self, owner: &str) -> Vec<(&str, &Value)> {
        self.owners
            .iter()
            .filter(|(_, o)| o.as_str() == owner)
            .filter_map(|(k, _)| self.data.get(k).map(|v| (k.as_str(), v)))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Channel
// ---------------------------------------------------------------------------

/// A named pub/sub channel for broadcasting messages to subscribed agents.
#[derive(Debug)]
pub struct Channel {
    /// Channel name.
    name: String,
    /// Set of subscribed agent names.
    subscribers: Vec<String>,
}

impl Channel {
    /// Create a new channel with no subscribers.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            subscribers: Vec::new(),
        }
    }

    /// Subscribe an agent to this channel. No-op if already subscribed.
    pub fn subscribe(&mut self, agent: &str) {
        if !self.subscribers.iter().any(|s| s == agent) {
            self.subscribers.push(agent.to_string());
        }
    }

    /// Unsubscribe an agent from this channel. No-op if not subscribed.
    pub fn unsubscribe(&mut self, agent: &str) {
        self.subscribers.retain(|s| s != agent);
    }

    /// List current subscribers.
    pub fn subscribers(&self) -> Vec<&str> {
        self.subscribers.iter().map(|s| s.as_str()).collect()
    }

    /// Number of subscribers.
    pub fn subscriber_count(&self) -> usize {
        self.subscribers.len()
    }

    /// The name of this channel.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Create a message for each subscriber (excluding the sender).
    ///
    /// Returns the list of generated messages ready for delivery.
    pub fn broadcast(&self, from: &str, subject: &str, body: Value) -> Vec<AgentMessage> {
        self.subscribers
            .iter()
            .filter(|s| s.as_str() != from)
            .map(|s| AgentMessage::new(from, s.as_str(), subject, body.clone()))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// CommunicationHub
// ---------------------------------------------------------------------------

/// Central message router that manages mailboxes, channels, and shared state.
///
/// Agents must be registered before they can send or receive messages.
#[derive(Debug)]
pub struct CommunicationHub {
    mailboxes: HashMap<String, Mailbox>,
    channels: HashMap<String, Channel>,
    shared_state: SharedState,
}

impl CommunicationHub {
    /// Create a new hub with no agents or channels.
    pub fn new() -> Self {
        Self {
            mailboxes: HashMap::new(),
            channels: HashMap::new(),
            shared_state: SharedState::new(),
        }
    }

    /// Register an agent, creating its mailbox.
    ///
    /// If the agent is already registered this is a no-op.
    pub fn register_agent(&mut self, name: &str) {
        self.mailboxes
            .entry(name.to_string())
            .or_insert_with(|| Mailbox::new(name));
    }

    /// Route a message to the recipient's mailbox.
    ///
    /// Returns an error if the recipient is not registered.
    pub fn send(&mut self, msg: AgentMessage) -> Result<()> {
        let to = msg.to.clone();
        match self.mailboxes.get_mut(&to) {
            Some(mailbox) => {
                mailbox.deliver(msg);
                Ok(())
            }
            None => Err(DeepAgentError::Other(format!("unknown recipient: {}", to))),
        }
    }

    /// Broadcast a message to all subscribers of a channel.
    ///
    /// Returns the number of messages delivered, or an error if the channel
    /// does not exist.
    pub fn broadcast(
        &mut self,
        channel: &str,
        from: &str,
        subject: &str,
        body: Value,
    ) -> Result<usize> {
        let messages = match self.channels.get(channel) {
            Some(ch) => ch.broadcast(from, subject, body),
            None => {
                return Err(DeepAgentError::Other(format!(
                    "unknown channel: {}",
                    channel
                )));
            }
        };

        let count = messages.len();
        for msg in messages {
            // Silently skip unregistered recipients (they may have been removed
            // after subscribing to the channel).
            if let Some(mailbox) = self.mailboxes.get_mut(&msg.to) {
                mailbox.deliver(msg);
            }
        }
        Ok(count)
    }

    /// Create a new named channel. No-op if it already exists.
    pub fn create_channel(&mut self, name: &str) {
        self.channels
            .entry(name.to_string())
            .or_insert_with(|| Channel::new(name));
    }

    /// Subscribe an agent to a channel, creating the channel if needed.
    pub fn subscribe(&mut self, channel: &str, agent: &str) {
        self.channels
            .entry(channel.to_string())
            .or_insert_with(|| Channel::new(channel))
            .subscribe(agent);
    }

    /// Get a mutable reference to an agent's mailbox.
    pub fn get_mailbox(&mut self, agent: &str) -> Option<&mut Mailbox> {
        self.mailboxes.get_mut(agent)
    }

    /// Read-only access to shared state.
    pub fn shared_state(&self) -> &SharedState {
        &self.shared_state
    }

    /// Mutable access to shared state.
    pub fn shared_state_mut(&mut self) -> &mut SharedState {
        &mut self.shared_state
    }

    /// Number of registered agents.
    pub fn agent_count(&self) -> usize {
        self.mailboxes.len()
    }

    /// Number of channels.
    pub fn channel_count(&self) -> usize {
        self.channels.len()
    }
}

impl Default for CommunicationHub {
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

    // -- AgentMessage tests --

    #[test]
    fn test_message_new_has_id() {
        let msg = AgentMessage::new("a", "b", "hello", json!({}));
        assert!(!msg.id.is_empty());
        assert_eq!(msg.from, "a");
        assert_eq!(msg.to, "b");
        assert_eq!(msg.subject, "hello");
        assert_eq!(msg.priority, MessagePriority::Normal);
        assert!(!msg.is_reply());
    }

    #[test]
    fn test_message_unique_ids() {
        let m1 = AgentMessage::new("a", "b", "s", json!({}));
        let m2 = AgentMessage::new("a", "b", "s", json!({}));
        assert_ne!(m1.id, m2.id);
    }

    #[test]
    fn test_message_with_priority() {
        let msg =
            AgentMessage::new("a", "b", "s", json!({})).with_priority(MessagePriority::Urgent);
        assert_eq!(msg.priority, MessagePriority::Urgent);
    }

    #[test]
    fn test_message_with_reply_to() {
        let original = AgentMessage::new("a", "b", "s", json!({}));
        let reply = AgentMessage::new("b", "a", "re: s", json!({})).with_reply_to(&original.id);
        assert!(reply.is_reply());
        assert_eq!(reply.reply_to.as_deref(), Some(original.id.as_str()));
    }

    #[test]
    fn test_message_with_metadata() {
        let msg = AgentMessage::new("a", "b", "s", json!({}))
            .with_metadata("key", json!("value"))
            .with_metadata("num", json!(42));
        assert_eq!(msg.metadata.get("key"), Some(&json!("value")));
        assert_eq!(msg.metadata.get("num"), Some(&json!(42)));
    }

    #[test]
    fn test_message_is_reply_false() {
        let msg = AgentMessage::new("a", "b", "s", json!({}));
        assert!(!msg.is_reply());
    }

    #[test]
    fn test_message_to_json() {
        let msg = AgentMessage::new("alice", "bob", "greet", json!({"text": "hi"}))
            .with_priority(MessagePriority::Urgent);
        let j = msg.to_json();
        assert_eq!(j["from"], "alice");
        assert_eq!(j["to"], "bob");
        assert_eq!(j["subject"], "greet");
        assert_eq!(j["body"]["text"], "hi");
        assert_eq!(j["priority"], "Urgent");
        assert!(j["reply_to"].is_null());
    }

    #[test]
    fn test_message_to_json_with_reply() {
        let msg = AgentMessage::new("a", "b", "s", json!({})).with_reply_to("orig-123");
        let j = msg.to_json();
        assert_eq!(j["reply_to"], "orig-123");
    }

    // -- MessagePriority tests --

    #[test]
    fn test_priority_ordering() {
        assert!(MessagePriority::Urgent > MessagePriority::Normal);
        assert!(MessagePriority::Normal > MessagePriority::Low);
        assert!(MessagePriority::Urgent > MessagePriority::Low);
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(format!("{}", MessagePriority::Urgent), "Urgent");
        assert_eq!(format!("{}", MessagePriority::Normal), "Normal");
        assert_eq!(format!("{}", MessagePriority::Low), "Low");
    }

    #[test]
    fn test_priority_sort() {
        let mut priorities = vec![
            MessagePriority::Low,
            MessagePriority::Urgent,
            MessagePriority::Normal,
        ];
        priorities.sort();
        assert_eq!(
            priorities,
            vec![
                MessagePriority::Low,
                MessagePriority::Normal,
                MessagePriority::Urgent,
            ]
        );
    }

    // -- MessageFilter tests --

    #[test]
    fn test_filter_empty_matches_all() {
        let filter = MessageFilter::new();
        let msg = AgentMessage::new("a", "b", "s", json!({}));
        assert!(filter.matches(&msg));
    }

    #[test]
    fn test_filter_from_agent() {
        let filter = MessageFilter::new().from_agent("alice");
        let yes = AgentMessage::new("alice", "bob", "s", json!({}));
        let no = AgentMessage::new("carol", "bob", "s", json!({}));
        assert!(filter.matches(&yes));
        assert!(!filter.matches(&no));
    }

    #[test]
    fn test_filter_to_agent() {
        let filter = MessageFilter::new().to_agent("bob");
        let yes = AgentMessage::new("alice", "bob", "s", json!({}));
        let no = AgentMessage::new("alice", "carol", "s", json!({}));
        assert!(filter.matches(&yes));
        assert!(!filter.matches(&no));
    }

    #[test]
    fn test_filter_subject_contains() {
        let filter = MessageFilter::new().subject_contains("urgent");
        let yes = AgentMessage::new("a", "b", "urgent task", json!({}));
        let no = AgentMessage::new("a", "b", "normal task", json!({}));
        assert!(filter.matches(&yes));
        assert!(!filter.matches(&no));
    }

    #[test]
    fn test_filter_priority() {
        let filter = MessageFilter::new().priority(MessagePriority::Urgent);
        let yes =
            AgentMessage::new("a", "b", "s", json!({})).with_priority(MessagePriority::Urgent);
        let no = AgentMessage::new("a", "b", "s", json!({}));
        assert!(filter.matches(&yes));
        assert!(!filter.matches(&no));
    }

    #[test]
    fn test_filter_combined() {
        let filter = MessageFilter::new()
            .from_agent("alice")
            .priority(MessagePriority::Urgent);
        let yes =
            AgentMessage::new("alice", "b", "s", json!({})).with_priority(MessagePriority::Urgent);
        let no_wrong_sender =
            AgentMessage::new("bob", "b", "s", json!({})).with_priority(MessagePriority::Urgent);
        let no_wrong_priority = AgentMessage::new("alice", "b", "s", json!({}));
        assert!(filter.matches(&yes));
        assert!(!filter.matches(&no_wrong_sender));
        assert!(!filter.matches(&no_wrong_priority));
    }

    // -- Mailbox tests --

    #[test]
    fn test_mailbox_new_is_empty() {
        let mb = Mailbox::new("agent1");
        assert!(mb.is_empty());
        assert_eq!(mb.unread_count(), 0);
        assert_eq!(mb.owner(), "agent1");
    }

    #[test]
    fn test_mailbox_deliver_and_read() {
        let mut mb = Mailbox::new("agent1");
        mb.deliver(AgentMessage::new("a", "agent1", "s", json!({})));
        mb.deliver(AgentMessage::new("b", "agent1", "s2", json!({})));
        assert_eq!(mb.unread_count(), 2);

        let messages = mb.read();
        assert_eq!(messages.len(), 2);
        assert!(mb.is_empty());
    }

    #[test]
    fn test_mailbox_peek_does_not_consume() {
        let mut mb = Mailbox::new("agent1");
        mb.deliver(AgentMessage::new("a", "agent1", "s", json!({})));
        let peeked = mb.peek();
        assert_eq!(peeked.len(), 1);
        assert_eq!(mb.unread_count(), 1); // still there
    }

    #[test]
    fn test_mailbox_read_filtered() {
        let mut mb = Mailbox::new("agent1");
        mb.deliver(
            AgentMessage::new("alice", "agent1", "task", json!({}))
                .with_priority(MessagePriority::Urgent),
        );
        mb.deliver(AgentMessage::new("bob", "agent1", "info", json!({})));
        mb.deliver(
            AgentMessage::new("alice", "agent1", "task2", json!({}))
                .with_priority(MessagePriority::Urgent),
        );

        let filter = MessageFilter::new().priority(MessagePriority::Urgent);
        let urgent = mb.read_filtered(&filter);
        assert_eq!(urgent.len(), 2);
        assert_eq!(mb.unread_count(), 1); // the Normal one remains
    }

    #[test]
    fn test_mailbox_read_empty() {
        let mut mb = Mailbox::new("agent1");
        let messages = mb.read();
        assert!(messages.is_empty());
    }

    // -- SharedState tests --

    #[test]
    fn test_shared_state_new_is_empty() {
        let state = SharedState::new();
        assert!(state.is_empty());
        assert_eq!(state.len(), 0);
    }

    #[test]
    fn test_shared_state_set_and_get() {
        let mut state = SharedState::new();
        state.set("key1", json!("value1"), "agent_a");
        assert_eq!(state.get("key1"), Some(&json!("value1")));
        assert_eq!(state.get_owner("key1"), Some("agent_a"));
    }

    #[test]
    fn test_shared_state_overwrite() {
        let mut state = SharedState::new();
        state.set("key1", json!("v1"), "a");
        state.set("key1", json!("v2"), "b");
        assert_eq!(state.get("key1"), Some(&json!("v2")));
        assert_eq!(state.get_owner("key1"), Some("b"));
    }

    #[test]
    fn test_shared_state_remove() {
        let mut state = SharedState::new();
        state.set("key1", json!("v"), "a");
        let removed = state.remove("key1");
        assert_eq!(removed, Some(json!("v")));
        assert!(state.get("key1").is_none());
        assert!(state.get_owner("key1").is_none());
    }

    #[test]
    fn test_shared_state_remove_nonexistent() {
        let mut state = SharedState::new();
        assert!(state.remove("nope").is_none());
    }

    #[test]
    fn test_shared_state_keys_and_len() {
        let mut state = SharedState::new();
        state.set("a", json!(1), "x");
        state.set("b", json!(2), "x");
        assert_eq!(state.len(), 2);
        let mut keys = state.keys();
        keys.sort();
        assert_eq!(keys, vec!["a", "b"]);
    }

    #[test]
    fn test_shared_state_entries_by_owner() {
        let mut state = SharedState::new();
        state.set("k1", json!(1), "alice");
        state.set("k2", json!(2), "bob");
        state.set("k3", json!(3), "alice");

        let alice_entries = state.entries_by_owner("alice");
        assert_eq!(alice_entries.len(), 2);

        let bob_entries = state.entries_by_owner("bob");
        assert_eq!(bob_entries.len(), 1);

        let nobody = state.entries_by_owner("nobody");
        assert!(nobody.is_empty());
    }

    // -- Channel tests --

    #[test]
    fn test_channel_new() {
        let ch = Channel::new("updates");
        assert_eq!(ch.name(), "updates");
        assert_eq!(ch.subscriber_count(), 0);
    }

    #[test]
    fn test_channel_subscribe() {
        let mut ch = Channel::new("ch");
        ch.subscribe("a");
        ch.subscribe("b");
        assert_eq!(ch.subscriber_count(), 2);
        assert_eq!(ch.subscribers(), vec!["a", "b"]);
    }

    #[test]
    fn test_channel_subscribe_idempotent() {
        let mut ch = Channel::new("ch");
        ch.subscribe("a");
        ch.subscribe("a");
        assert_eq!(ch.subscriber_count(), 1);
    }

    #[test]
    fn test_channel_unsubscribe() {
        let mut ch = Channel::new("ch");
        ch.subscribe("a");
        ch.subscribe("b");
        ch.unsubscribe("a");
        assert_eq!(ch.subscriber_count(), 1);
        assert_eq!(ch.subscribers(), vec!["b"]);
    }

    #[test]
    fn test_channel_unsubscribe_nonexistent() {
        let mut ch = Channel::new("ch");
        ch.unsubscribe("x"); // should not panic
        assert_eq!(ch.subscriber_count(), 0);
    }

    #[test]
    fn test_channel_broadcast() {
        let mut ch = Channel::new("ch");
        ch.subscribe("a");
        ch.subscribe("b");
        ch.subscribe("c");

        let msgs = ch.broadcast("a", "news", json!({"data": 1}));
        // sender "a" should be excluded
        assert_eq!(msgs.len(), 2);
        let recipients: Vec<&str> = msgs.iter().map(|m| m.to.as_str()).collect();
        assert!(recipients.contains(&"b"));
        assert!(recipients.contains(&"c"));
        assert_eq!(msgs[0].from, "a");
        assert_eq!(msgs[0].subject, "news");
    }

    #[test]
    fn test_channel_broadcast_no_subscribers() {
        let ch = Channel::new("ch");
        let msgs = ch.broadcast("sender", "s", json!({}));
        assert!(msgs.is_empty());
    }

    // -- CommunicationHub tests --

    #[test]
    fn test_hub_new() {
        let hub = CommunicationHub::new();
        assert_eq!(hub.agent_count(), 0);
        assert_eq!(hub.channel_count(), 0);
    }

    #[test]
    fn test_hub_register_agent() {
        let mut hub = CommunicationHub::new();
        hub.register_agent("a");
        hub.register_agent("b");
        assert_eq!(hub.agent_count(), 2);
    }

    #[test]
    fn test_hub_register_agent_idempotent() {
        let mut hub = CommunicationHub::new();
        hub.register_agent("a");
        hub.register_agent("a");
        assert_eq!(hub.agent_count(), 1);
    }

    #[test]
    fn test_hub_send_message() {
        let mut hub = CommunicationHub::new();
        hub.register_agent("alice");
        hub.register_agent("bob");

        let msg = AgentMessage::new("alice", "bob", "hello", json!({"text": "hi"}));
        hub.send(msg).unwrap();

        let mailbox = hub.get_mailbox("bob").unwrap();
        assert_eq!(mailbox.unread_count(), 1);
        let messages = mailbox.read();
        assert_eq!(messages[0].from, "alice");
        assert_eq!(messages[0].subject, "hello");
    }

    #[test]
    fn test_hub_send_unknown_recipient() {
        let mut hub = CommunicationHub::new();
        hub.register_agent("alice");

        let msg = AgentMessage::new("alice", "nobody", "s", json!({}));
        let result = hub.send(msg);
        assert!(result.is_err());
    }

    #[test]
    fn test_hub_broadcast_channel() {
        let mut hub = CommunicationHub::new();
        hub.register_agent("a");
        hub.register_agent("b");
        hub.register_agent("c");

        hub.create_channel("updates");
        hub.subscribe("updates", "a");
        hub.subscribe("updates", "b");
        hub.subscribe("updates", "c");

        let count = hub
            .broadcast("updates", "a", "news", json!({"v": 1}))
            .unwrap();
        assert_eq!(count, 2); // b and c (not a)

        assert_eq!(hub.get_mailbox("a").unwrap().unread_count(), 0);
        assert_eq!(hub.get_mailbox("b").unwrap().unread_count(), 1);
        assert_eq!(hub.get_mailbox("c").unwrap().unread_count(), 1);
    }

    #[test]
    fn test_hub_broadcast_unknown_channel() {
        let mut hub = CommunicationHub::new();
        let result = hub.broadcast("nonexistent", "a", "s", json!({}));
        assert!(result.is_err());
    }

    #[test]
    fn test_hub_shared_state() {
        let mut hub = CommunicationHub::new();
        hub.shared_state_mut()
            .set("plan", json!({"step": 1}), "planner");
        assert_eq!(hub.shared_state().get("plan"), Some(&json!({"step": 1})));
        assert_eq!(hub.shared_state().get_owner("plan"), Some("planner"));
    }

    #[test]
    fn test_hub_get_mailbox_nonexistent() {
        let mut hub = CommunicationHub::new();
        assert!(hub.get_mailbox("ghost").is_none());
    }

    #[test]
    fn test_hub_create_channel() {
        let mut hub = CommunicationHub::new();
        hub.create_channel("ch1");
        hub.create_channel("ch2");
        assert_eq!(hub.channel_count(), 2);
    }

    #[test]
    fn test_hub_subscribe_creates_channel() {
        let mut hub = CommunicationHub::new();
        hub.subscribe("auto-ch", "agent1");
        assert_eq!(hub.channel_count(), 1);
    }

    #[test]
    fn test_hub_end_to_end_conversation() {
        let mut hub = CommunicationHub::new();
        hub.register_agent("planner");
        hub.register_agent("executor");
        hub.register_agent("reviewer");

        // Planner sends task to executor
        let task = AgentMessage::new(
            "planner",
            "executor",
            "execute task",
            json!({"task": "build"}),
        )
        .with_priority(MessagePriority::Urgent);
        let task_id = task.id.clone();
        hub.send(task).unwrap();

        // Executor reads and replies
        let mailbox = hub.get_mailbox("executor").unwrap();
        let msgs = mailbox.read();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].priority, MessagePriority::Urgent);

        let reply = AgentMessage::new("executor", "planner", "task done", json!({"status": "ok"}))
            .with_reply_to(&task_id);
        hub.send(reply).unwrap();

        // Planner reads the reply
        let planner_msgs = hub.get_mailbox("planner").unwrap().read();
        assert_eq!(planner_msgs.len(), 1);
        assert!(planner_msgs[0].is_reply());

        // Broadcast result to all via channel
        hub.create_channel("results");
        hub.subscribe("results", "planner");
        hub.subscribe("results", "executor");
        hub.subscribe("results", "reviewer");

        let count = hub
            .broadcast("results", "planner", "result ready", json!({"done": true}))
            .unwrap();
        assert_eq!(count, 2); // executor and reviewer

        assert_eq!(hub.get_mailbox("reviewer").unwrap().unread_count(), 1);
    }

    #[test]
    fn test_hub_shared_state_multiple_agents() {
        let mut hub = CommunicationHub::new();
        hub.shared_state_mut().set("x", json!(1), "a");
        hub.shared_state_mut().set("y", json!(2), "b");
        hub.shared_state_mut().set("z", json!(3), "a");

        let a_entries = hub.shared_state().entries_by_owner("a");
        assert_eq!(a_entries.len(), 2);
        assert_eq!(hub.shared_state().len(), 3);
    }

    #[test]
    fn test_priority_equality() {
        assert_eq!(MessagePriority::Normal, MessagePriority::Normal);
        assert_ne!(MessagePriority::Urgent, MessagePriority::Low);
    }
}
