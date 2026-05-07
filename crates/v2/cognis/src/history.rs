//! `RunnableWithMessageHistory` — wrap a `Runnable<Vec<Message>, Message>`
//! so it carries conversation history per session ID.
//!
//! This is the LangChain-equivalent message-history wrapper. The wrapper
//! holds an `Arc<dyn HistoryStore>` so different storage backends (memory,
//! Redis, sqlite, ...) plug in.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::RwLock;

use cognis2_core::{Message, Result, Runnable, RunnableConfig};

/// Pluggable conversation-history store.
#[async_trait]
pub trait HistoryStore: Send + Sync {
    /// Read the current history for `session_id`.
    async fn read(&self, session_id: &str) -> Result<Vec<Message>>;
    /// Append messages to the history for `session_id`.
    async fn append(&self, session_id: &str, msgs: Vec<Message>) -> Result<()>;
    /// Clear the history for `session_id`.
    async fn clear(&self, session_id: &str) -> Result<()>;
}

/// In-memory history store. Default for tests / single-process apps.
#[derive(Default)]
pub struct InMemoryHistory {
    sessions: RwLock<HashMap<String, Vec<Message>>>,
}

impl InMemoryHistory {
    /// Empty store.
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl HistoryStore for InMemoryHistory {
    async fn read(&self, session_id: &str) -> Result<Vec<Message>> {
        Ok(self
            .sessions
            .read()
            .await
            .get(session_id)
            .cloned()
            .unwrap_or_default())
    }
    async fn append(&self, session_id: &str, msgs: Vec<Message>) -> Result<()> {
        self.sessions
            .write()
            .await
            .entry(session_id.to_string())
            .or_default()
            .extend(msgs);
        Ok(())
    }
    async fn clear(&self, session_id: &str) -> Result<()> {
        self.sessions.write().await.remove(session_id);
        Ok(())
    }
}

/// Key inserted into `RunnableConfig::extras` to identify the session.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SessionKey {
    /// Session identifier (e.g. user id, conversation id).
    pub id: String,
}

impl SessionKey {
    /// Construct.
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

/// Wraps a `Runnable<Vec<Message>, Message>` with per-session history.
///
/// On each invoke:
/// 1. Read history for the session.
/// 2. Concatenate `[history, input]` and call the inner.
/// 3. Append `[input.last(), output]` to history.
pub struct RunnableWithMessageHistory<R> {
    inner: R,
    store: Arc<dyn HistoryStore>,
}

impl<R> RunnableWithMessageHistory<R>
where
    R: Runnable<Vec<Message>, Message>,
{
    /// Build a wrapper.
    pub fn new(inner: R, store: Arc<dyn HistoryStore>) -> Self {
        Self { inner, store }
    }
}

#[async_trait]
impl<R> Runnable<Vec<Message>, Message> for RunnableWithMessageHistory<R>
where
    R: Runnable<Vec<Message>, Message>,
{
    async fn invoke(&self, input: Vec<Message>, config: RunnableConfig) -> Result<Message> {
        let session_id = config
            .extras
            .get::<SessionKey>()
            .map(|k| k.id.clone())
            .unwrap_or_else(|| "default".to_string());
        let history = self.store.read(&session_id).await?;
        let mut combined = Vec::with_capacity(history.len() + input.len());
        combined.extend(history);
        combined.extend(input.iter().cloned());

        let out = self.inner.invoke(combined, config).await?;

        // Append the latest user input(s) and the produced output.
        let mut to_persist = input;
        to_persist.push(out.clone());
        self.store.append(&session_id, to_persist).await?;
        Ok(out)
    }
    fn name(&self) -> &str {
        "RunnableWithMessageHistory"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct EchoLast;

    #[async_trait]
    impl Runnable<Vec<Message>, Message> for EchoLast {
        async fn invoke(&self, input: Vec<Message>, _: RunnableConfig) -> Result<Message> {
            Ok(Message::ai(format!(
                "saw {} msgs, last: {}",
                input.len(),
                input
                    .last()
                    .map(|m| m.content().to_string())
                    .unwrap_or_default()
            )))
        }
    }

    fn cfg_for(session: &str) -> RunnableConfig {
        // RunnableConfig::clone deliberately drops extras (Any can't be
        // generically cloned), so callers build a fresh cfg per invoke
        // when they need to pass extras across runs.
        let mut c = RunnableConfig::default();
        c.extras.insert(SessionKey::new(session));
        c
    }

    #[tokio::test]
    async fn history_accumulates_across_calls() {
        let store: Arc<dyn HistoryStore> = Arc::new(InMemoryHistory::new());
        let r = RunnableWithMessageHistory::new(EchoLast, store.clone());

        let out1 = r
            .invoke(vec![Message::human("first")], cfg_for("s1"))
            .await
            .unwrap();
        assert!(out1.content().contains("saw 1 msgs"));

        let out2 = r
            .invoke(vec![Message::human("second")], cfg_for("s1"))
            .await
            .unwrap();
        // History now contains: human("first"), ai(out1), human("second") = 3
        assert!(out2.content().contains("saw 3 msgs"));
    }

    #[tokio::test]
    async fn sessions_are_isolated() {
        let store: Arc<dyn HistoryStore> = Arc::new(InMemoryHistory::new());
        let r = RunnableWithMessageHistory::new(EchoLast, store.clone());

        r.invoke(vec![Message::human("a1")], cfg_for("a"))
            .await
            .unwrap();
        r.invoke(vec![Message::human("b1")], cfg_for("b"))
            .await
            .unwrap();

        let out_a = r
            .invoke(vec![Message::human("a2")], cfg_for("a"))
            .await
            .unwrap();
        // a saw a1 + ai(a1) + a2 = 3
        assert!(out_a.content().contains("saw 3 msgs"));
    }
}
