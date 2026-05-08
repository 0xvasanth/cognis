//! Topic-based pub/sub for inter-agent broadcast.
//!
//! Complements [`crate::multi_agent::MessageBus`], which is point-to-point
//! (publish-to-inbox + drain-by-recipient). The bus here is broadcast:
//! one publish reaches every subscriber on the topic.
//!
//! Built on `tokio::sync::broadcast`, so each subscriber owns a
//! [`Subscription`] handle backed by an independent receiver. Slow
//! subscribers fall behind; on overflow, the oldest unread messages are
//! dropped (per `broadcast` semantics) and `recv()` returns
//! [`SubscribeError::Lagged`] so the caller can decide.
//!
//! Topics are created lazily on first subscribe / publish. Empty topics
//! (no subscribers, no senders) are kept for the bus's lifetime — this
//! is fine for typical fleet sizes; if you churn through millions of
//! topic ids, prune via [`AgentBus::drop_topic`].

use std::collections::HashMap;
use std::sync::Arc;

use tokio::sync::{broadcast, RwLock};

use crate::multi_agent::AgentMessage;

/// Default per-topic channel capacity.
const DEFAULT_CAPACITY: usize = 256;

/// Topic-based pub/sub bus. Cheap to clone (`Arc` inside).
#[derive(Clone)]
pub struct AgentBus {
    topics: Arc<RwLock<HashMap<String, broadcast::Sender<AgentMessage>>>>,
    capacity: usize,
}

impl Default for AgentBus {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentBus {
    /// Empty bus with the default per-topic capacity.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY)
    }

    /// Empty bus with a custom per-topic channel capacity. Larger
    /// capacities tolerate slower subscribers but use more memory.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            topics: Arc::new(RwLock::new(HashMap::new())),
            capacity: capacity.max(1),
        }
    }

    /// Subscribe to `topic`. Returns a [`Subscription`] you `.recv()` on.
    /// Subsequent subscribers on the same topic each get an independent
    /// receiver — every published message reaches every subscriber.
    pub async fn subscribe(&self, topic: impl Into<String>) -> Subscription {
        let topic = topic.into();
        let mut topics = self.topics.write().await;
        let tx = topics
            .entry(topic.clone())
            .or_insert_with(|| broadcast::channel(self.capacity).0);
        Subscription {
            rx: tx.subscribe(),
            topic,
        }
    }

    /// Publish `msg` to `topic`. If no subscribers are listening, the
    /// message is dropped silently (broadcast channel semantics) — that
    /// is *not* an error.
    ///
    /// Returns the number of subscribers the message was delivered to.
    pub async fn publish(&self, topic: &str, msg: AgentMessage) -> usize {
        // Fast path: topic exists.
        if let Some(tx) = self.topics.read().await.get(topic) {
            return tx.send(msg).unwrap_or(0);
        }
        // Slow path: create the topic so future subscribers find it,
        // even though this publish has nowhere to land.
        let mut topics = self.topics.write().await;
        let tx = topics
            .entry(topic.to_string())
            .or_insert_with(|| broadcast::channel(self.capacity).0);
        tx.send(msg).unwrap_or(0)
    }

    /// Number of registered topics.
    pub async fn topic_count(&self) -> usize {
        self.topics.read().await.len()
    }

    /// Number of active subscribers on `topic` (0 if topic doesn't exist).
    pub async fn subscriber_count(&self, topic: &str) -> usize {
        self.topics
            .read()
            .await
            .get(topic)
            .map(broadcast::Sender::receiver_count)
            .unwrap_or(0)
    }

    /// Drop a topic's underlying channel. Existing subscribers on this
    /// topic will see [`SubscribeError::Closed`] on their next `recv()`.
    pub async fn drop_topic(&self, topic: &str) {
        self.topics.write().await.remove(topic);
    }
}

/// A subscriber handle. Drop the subscription to unsubscribe.
pub struct Subscription {
    rx: broadcast::Receiver<AgentMessage>,
    topic: String,
}

impl Subscription {
    /// The topic this subscription is bound to.
    pub fn topic(&self) -> &str {
        &self.topic
    }

    /// Wait for the next message. Returns [`SubscribeError::Closed`] if
    /// the bus dropped the topic, or [`SubscribeError::Lagged`] if this
    /// subscriber fell behind and the channel evicted unread messages
    /// — the inner `u64` is how many were dropped.
    pub async fn recv(&mut self) -> Result<AgentMessage, SubscribeError> {
        match self.rx.recv().await {
            Ok(m) => Ok(m),
            Err(broadcast::error::RecvError::Closed) => Err(SubscribeError::Closed),
            Err(broadcast::error::RecvError::Lagged(n)) => Err(SubscribeError::Lagged(n)),
        }
    }

    /// Non-blocking check for a pending message.
    pub fn try_recv(&mut self) -> Result<AgentMessage, SubscribeError> {
        match self.rx.try_recv() {
            Ok(m) => Ok(m),
            Err(broadcast::error::TryRecvError::Empty) => Err(SubscribeError::Empty),
            Err(broadcast::error::TryRecvError::Closed) => Err(SubscribeError::Closed),
            Err(broadcast::error::TryRecvError::Lagged(n)) => Err(SubscribeError::Lagged(n)),
        }
    }
}

/// Errors a subscriber may encounter.
#[derive(Debug)]
pub enum SubscribeError {
    /// The topic was dropped or all senders were closed.
    Closed,
    /// This subscriber fell behind by `n` messages — the channel
    /// evicted unread items. The next `recv` will return the
    /// freshest available message.
    Lagged(u64),
    /// `try_recv` only: no message currently waiting.
    Empty,
}

impl std::fmt::Display for SubscribeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubscribeError::Closed => write!(f, "topic closed"),
            SubscribeError::Lagged(n) => write!(f, "subscriber lagged by {n} messages"),
            SubscribeError::Empty => write!(f, "no message available"),
        }
    }
}

impl std::error::Error for SubscribeError {}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::Message;

    fn msg(text: &str) -> AgentMessage {
        AgentMessage {
            from: "user".into(),
            to: "broadcast".into(),
            content: Message::human(text),
            metadata: serde_json::Value::Null,
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn one_publisher_one_subscriber_roundtrip() {
        let bus = AgentBus::new();
        let mut sub = bus.subscribe("planning").await;
        let n = bus.publish("planning", msg("hello")).await;
        assert_eq!(n, 1);
        let got = sub.recv().await.unwrap();
        assert_eq!(got.content.content(), "hello");
    }

    #[tokio::test]
    async fn fanout_to_multiple_subscribers() {
        let bus = AgentBus::new();
        let mut a = bus.subscribe("alerts").await;
        let mut b = bus.subscribe("alerts").await;
        let mut c = bus.subscribe("alerts").await;
        let n = bus.publish("alerts", msg("fire")).await;
        assert_eq!(n, 3);
        assert_eq!(a.recv().await.unwrap().content.content(), "fire");
        assert_eq!(b.recv().await.unwrap().content.content(), "fire");
        assert_eq!(c.recv().await.unwrap().content.content(), "fire");
    }

    #[tokio::test]
    async fn topic_isolation() {
        let bus = AgentBus::new();
        let mut planning = bus.subscribe("planning").await;
        let mut alerts = bus.subscribe("alerts").await;
        bus.publish("planning", msg("plan-msg")).await;
        bus.publish("alerts", msg("alert-msg")).await;
        assert_eq!(planning.recv().await.unwrap().content.content(), "plan-msg");
        assert_eq!(alerts.recv().await.unwrap().content.content(), "alert-msg");
    }

    #[tokio::test]
    async fn publish_with_no_subscribers_drops_silently() {
        let bus = AgentBus::new();
        let n = bus.publish("ghost-topic", msg("nobody home")).await;
        assert_eq!(n, 0);
        // Topic now exists for future subscribers — but published msg is gone.
        assert_eq!(bus.topic_count().await, 1);
    }

    #[tokio::test]
    async fn try_recv_empty_when_no_message() {
        let bus = AgentBus::new();
        let mut sub = bus.subscribe("t").await;
        assert!(matches!(sub.try_recv(), Err(SubscribeError::Empty)));
    }

    #[tokio::test]
    async fn drop_topic_closes_subscribers() {
        let bus = AgentBus::new();
        let mut sub = bus.subscribe("t").await;
        bus.drop_topic("t").await;
        assert!(matches!(sub.recv().await, Err(SubscribeError::Closed)));
    }

    #[tokio::test]
    async fn subscriber_count_tracks_active() {
        let bus = AgentBus::new();
        let _a = bus.subscribe("t").await;
        let _b = bus.subscribe("t").await;
        assert_eq!(bus.subscriber_count("t").await, 2);
        drop(_a);
        // broadcast::Sender::receiver_count updates lazily; on next op.
        // But a fresh subscribe will reflect the drop:
        let _c = bus.subscribe("t").await;
        assert_eq!(bus.subscriber_count("t").await, 2);
    }

    #[tokio::test]
    async fn lagged_subscribers_get_lagged_error() {
        // Tiny capacity so we overflow quickly.
        let bus = AgentBus::with_capacity(2);
        let mut sub = bus.subscribe("t").await;
        for i in 0..10 {
            bus.publish("t", msg(&format!("m{i}"))).await;
        }
        // First recv should report Lagged.
        match sub.recv().await {
            Err(SubscribeError::Lagged(n)) => assert!(n > 0),
            other => panic!("expected Lagged, got {other:?}"),
        }
    }
}
