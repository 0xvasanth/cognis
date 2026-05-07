//! Session — a long-lived conversation with metadata, optional expiry,
//! and a pluggable backing store.
//!
//! Distinct from [`crate::history::SessionKey`]: that's only an
//! identifier carried in `RunnableConfig::extras`. A `Session` is the
//! materialized object — id, history, metadata, lifetime — and is what
//! a SessionStore persists.
//!
//! Customization:
//! - Implement [`SessionStore`] for Redis/SQL/S3 backings.
//! - Override the time source via [`SessionStore::with_clock`]-style
//!   construction (the trait is async; users embed a clock in their
//!   impl directly).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use cognis_core::{Message, Result};

/// A long-lived conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    /// Stable identifier (caller-supplied).
    pub id: String,
    /// Conversation history.
    pub history: Vec<Message>,
    /// Free-form metadata. Examples: user-id, tenant-id, agent name.
    pub metadata: serde_json::Value,
    /// Unix-millis when the session was created.
    pub created_at_ms: u64,
    /// Unix-millis when the session was last touched.
    pub updated_at_ms: u64,
    /// Optional unix-millis after which the session is considered expired.
    pub expires_at_ms: Option<u64>,
}

impl Session {
    /// New session with empty history + null metadata.
    pub fn new(id: impl Into<String>) -> Self {
        let now = now_millis();
        Self {
            id: id.into(),
            history: Vec::new(),
            metadata: serde_json::Value::Null,
            created_at_ms: now,
            updated_at_ms: now,
            expires_at_ms: None,
        }
    }

    /// Set metadata (replaces any existing value).
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set a TTL relative to `created_at_ms`.
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.expires_at_ms = Some(self.created_at_ms + ttl.as_millis() as u64);
        self
    }

    /// True if the session has expired (relative to `now_millis`).
    pub fn is_expired(&self) -> bool {
        match self.expires_at_ms {
            Some(t) => now_millis() >= t,
            None => false,
        }
    }

    /// Append a message and bump `updated_at_ms`.
    pub fn push(&mut self, msg: Message) {
        self.history.push(msg);
        self.updated_at_ms = now_millis();
    }
}

fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Pluggable session storage. The default [`InMemorySessionStore`] keeps
/// everything in a hashmap; production users plug in Redis / Postgres /
/// DynamoDB.
#[async_trait]
pub trait SessionStore: Send + Sync {
    /// Read a session by id. Returns `None` if not found or expired.
    async fn get(&self, id: &str) -> Result<Option<Session>>;

    /// Persist a session (creates or replaces).
    async fn put(&self, session: Session) -> Result<()>;

    /// Delete a session.
    async fn delete(&self, id: &str) -> Result<()>;

    /// List all session ids (best-effort; backends may cap).
    async fn list_ids(&self) -> Result<Vec<String>>;
}

/// In-memory store. Single-process, thread-safe, no persistence.
#[derive(Default)]
pub struct InMemorySessionStore {
    inner: RwLock<HashMap<String, Session>>,
}

impl InMemorySessionStore {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn get(&self, id: &str) -> Result<Option<Session>> {
        let g = self.inner.read().await;
        Ok(g.get(id).filter(|s| !s.is_expired()).cloned())
    }
    async fn put(&self, session: Session) -> Result<()> {
        self.inner.write().await.insert(session.id.clone(), session);
        Ok(())
    }
    async fn delete(&self, id: &str) -> Result<()> {
        self.inner.write().await.remove(id);
        Ok(())
    }
    async fn list_ids(&self) -> Result<Vec<String>> {
        let mut ids: Vec<String> = self.inner.read().await.keys().cloned().collect();
        ids.sort();
        Ok(ids)
    }
}

/// Type alias for an `Arc<dyn SessionStore>` so users keep a shared
/// handle without sprinkling `dyn` everywhere.
pub type SessionStoreHandle = Arc<dyn SessionStore>;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn round_trip_basic() {
        let store = InMemorySessionStore::new();
        let mut s = Session::new("sess-1").with_metadata(serde_json::json!({"user": "alice"}));
        s.push(Message::human("hi"));
        s.push(Message::ai("hello"));
        store.put(s).await.unwrap();

        let read = store.get("sess-1").await.unwrap().unwrap();
        assert_eq!(read.id, "sess-1");
        assert_eq!(read.history.len(), 2);
        assert_eq!(read.metadata["user"], "alice");
    }

    #[tokio::test]
    async fn delete_removes() {
        let store = InMemorySessionStore::new();
        store.put(Session::new("a")).await.unwrap();
        assert!(store.get("a").await.unwrap().is_some());
        store.delete("a").await.unwrap();
        assert!(store.get("a").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn expired_sessions_are_filtered_on_read() {
        let store = InMemorySessionStore::new();
        let s = Session::new("expired").with_ttl(Duration::from_millis(0));
        // Sleep a touch to ensure now > expires_at_ms.
        tokio::time::sleep(Duration::from_millis(2)).await;
        store.put(s).await.unwrap();
        assert!(store.get("expired").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn list_ids_sorted() {
        let store = InMemorySessionStore::new();
        store.put(Session::new("zeta")).await.unwrap();
        store.put(Session::new("alpha")).await.unwrap();
        let ids = store.list_ids().await.unwrap();
        assert_eq!(ids, vec!["alpha".to_string(), "zeta".into()]);
    }
}
