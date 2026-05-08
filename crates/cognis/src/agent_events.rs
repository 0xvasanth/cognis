//! Typed lifecycle events layered over [`AgentBus`].
//!
//! `AgentBus` is a generic topic pub/sub of `AgentMessage`s. This module
//! adds a typed [`AgentEvent`] envelope so subscribers can match on
//! variants (`Started`, `Finished`, `ToolCalled`, `ToolResult`, `Errored`,
//! `Custom`) without parsing free-form text.
//!
//! Use [`AgentEventBus`] when you want a single channel of typed agent
//! lifecycle events; use raw [`AgentBus`] for arbitrary topic routing.
//!
//! Wire format: events are serialized to JSON inside an [`AgentMessage`]
//! payload (`metadata.event = ...`). Receivers deserialize back into
//! `AgentEvent`. This means events survive the bus's broadcast
//! semantics — every subscriber sees every event.

use cognis_core::Message;
use serde::{Deserialize, Serialize};

use crate::agent_bus::{AgentBus, SubscribeError, Subscription};
use crate::multi_agent::AgentMessage;

/// Default topic for typed agent lifecycle events.
pub const DEFAULT_EVENTS_TOPIC: &str = "agent.events";

/// Typed agent-lifecycle event. Attach extra context via the `Custom`
/// variant if you need data the built-in variants don't cover.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AgentEvent {
    /// An agent run started.
    Started {
        /// Agent id.
        agent_id: String,
        /// First-turn input as plain text.
        input: String,
    },
    /// An agent run completed successfully.
    Finished {
        /// Agent id.
        agent_id: String,
        /// Final output content.
        output: String,
        /// How many ReAct iterations the run took.
        iterations: u32,
    },
    /// A tool call was issued by the agent.
    ToolCalled {
        /// Agent id.
        agent_id: String,
        /// Tool name (matches `Tool::name()`).
        tool_name: String,
        /// Tool input as JSON.
        args: serde_json::Value,
    },
    /// A tool returned a result.
    ToolResult {
        /// Agent id.
        agent_id: String,
        /// Tool name.
        tool_name: String,
        /// Tool output as JSON.
        result: serde_json::Value,
    },
    /// An agent run errored.
    Errored {
        /// Agent id.
        agent_id: String,
        /// Error message.
        error: String,
    },
    /// Custom event for application-specific signals not covered by the
    /// other variants.
    Custom {
        /// Agent id.
        agent_id: String,
        /// User-defined event tag.
        tag: String,
        /// Free-form payload.
        payload: serde_json::Value,
    },
}

impl AgentEvent {
    /// Agent id for any variant.
    pub fn agent_id(&self) -> &str {
        match self {
            AgentEvent::Started { agent_id, .. }
            | AgentEvent::Finished { agent_id, .. }
            | AgentEvent::ToolCalled { agent_id, .. }
            | AgentEvent::ToolResult { agent_id, .. }
            | AgentEvent::Errored { agent_id, .. }
            | AgentEvent::Custom { agent_id, .. } => agent_id,
        }
    }
}

/// Typed event bus on top of [`AgentBus`]. Cheap to clone (the inner bus
/// is `Arc`-backed).
#[derive(Clone)]
pub struct AgentEventBus {
    inner: AgentBus,
    topic: String,
}

impl Default for AgentEventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentEventBus {
    /// New typed bus on the default topic (`"agent.events"`) with a
    /// fresh underlying `AgentBus`.
    pub fn new() -> Self {
        Self::with_bus(AgentBus::new())
    }

    /// New typed bus reusing an existing [`AgentBus`].
    pub fn with_bus(bus: AgentBus) -> Self {
        Self {
            inner: bus,
            topic: DEFAULT_EVENTS_TOPIC.into(),
        }
    }

    /// Override the topic on which events are published.
    pub fn with_topic(mut self, topic: impl Into<String>) -> Self {
        self.topic = topic.into();
        self
    }

    /// The underlying generic [`AgentBus`]. Useful for mixing typed
    /// events with arbitrary point-to-point topics.
    pub fn bus(&self) -> &AgentBus {
        &self.inner
    }

    /// Publish an event. Returns the number of subscribers it reached.
    pub async fn publish(&self, event: AgentEvent) -> usize {
        let payload = serde_json::to_value(&event).unwrap_or(serde_json::Value::Null);
        let agent_id = event.agent_id().to_string();
        let envelope = AgentMessage {
            from: agent_id,
            to: self.topic.clone(),
            content: Message::system("agent-event"),
            metadata: serde_json::json!({ "event": payload }),
        };
        self.inner.publish(&self.topic, envelope).await
    }

    /// Subscribe to typed events.
    pub async fn subscribe(&self) -> EventSubscription {
        EventSubscription {
            inner: self.inner.subscribe(&self.topic).await,
        }
    }

    /// Total topics on the underlying bus.
    pub async fn topic_count(&self) -> usize {
        self.inner.topic_count().await
    }

    /// Subscriber count on the typed events topic.
    pub async fn subscriber_count(&self) -> usize {
        self.inner.subscriber_count(&self.topic).await
    }
}

/// Typed-event subscription handle. Drop to unsubscribe.
pub struct EventSubscription {
    inner: Subscription,
}

impl EventSubscription {
    /// Topic this subscription is bound to.
    pub fn topic(&self) -> &str {
        self.inner.topic()
    }

    /// Wait for the next event. Returns the deserialized `AgentEvent`,
    /// or a [`SubscribeError`] if the bus closed / lagged / payload
    /// failed to deserialize.
    pub async fn recv(&mut self) -> Result<AgentEvent, SubscribeError> {
        let envelope = self.inner.recv().await?;
        decode_event(&envelope).map_err(|_| SubscribeError::Closed)
    }
}

fn decode_event(env: &AgentMessage) -> Result<AgentEvent, serde_json::Error> {
    let v = env
        .metadata
        .get("event")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    serde_json::from_value(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn started_event_round_trip() {
        let bus = AgentEventBus::new();
        let mut sub = bus.subscribe().await;
        let n = bus
            .publish(AgentEvent::Started {
                agent_id: "writer".into(),
                input: "hello".into(),
            })
            .await;
        assert_eq!(n, 1, "subscriber should receive the event");
        match sub.recv().await.unwrap() {
            AgentEvent::Started { agent_id, input } => {
                assert_eq!(agent_id, "writer");
                assert_eq!(input, "hello");
            }
            other => panic!("expected Started, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn fanout_to_every_subscriber() {
        let bus = AgentEventBus::new();
        let mut a = bus.subscribe().await;
        let mut b = bus.subscribe().await;
        bus.publish(AgentEvent::Errored {
            agent_id: "x".into(),
            error: "boom".into(),
        })
        .await;
        for sub in [&mut a, &mut b] {
            match sub.recv().await.unwrap() {
                AgentEvent::Errored { error, .. } => assert_eq!(error, "boom"),
                _ => panic!("variant"),
            }
        }
    }

    #[tokio::test]
    async fn custom_topic_isolates_subscriptions() {
        let other = AgentEventBus::new().with_topic("alerts");
        let mut sub = other.subscribe().await;
        // Default-topic subscriber on a fresh bus shouldn't see this.
        let unrelated = AgentEventBus::new();
        let _unrelated_sub = unrelated.subscribe().await;
        other
            .publish(AgentEvent::Custom {
                agent_id: "watchdog".into(),
                tag: "alert".into(),
                payload: serde_json::json!({"severity": "high"}),
            })
            .await;
        match sub.recv().await.unwrap() {
            AgentEvent::Custom { tag, .. } => assert_eq!(tag, "alert"),
            _ => panic!("variant"),
        }
    }

    #[tokio::test]
    async fn shared_underlying_bus() {
        // Two AgentEventBus on the same AgentBus + same topic see each
        // other's events.
        let inner = AgentBus::new();
        let a = AgentEventBus::with_bus(inner.clone());
        let b = AgentEventBus::with_bus(inner);
        let mut sub = b.subscribe().await;
        a.publish(AgentEvent::Started {
            agent_id: "shared".into(),
            input: "ok".into(),
        })
        .await;
        match sub.recv().await.unwrap() {
            AgentEvent::Started { agent_id, .. } => assert_eq!(agent_id, "shared"),
            _ => panic!("variant"),
        }
    }

    #[tokio::test]
    async fn subscriber_count_tracks_active() {
        let bus = AgentEventBus::new();
        let _a = bus.subscribe().await;
        let _b = bus.subscribe().await;
        assert_eq!(bus.subscriber_count().await, 2);
    }
}
