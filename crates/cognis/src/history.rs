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

use cognis_core::{Message, Result, Runnable, RunnableConfig};

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

/// Closure used by [`RunnableWithMessageHistory`] to derive the
/// session-id for a call from `(input, config)`. Default: read
/// `SessionKey` from `config.extras` and fall back to `"default"`.
pub type SessionResolver = Arc<dyn Fn(&[Message], &RunnableConfig) -> String + Send + Sync>;

/// Closure used by [`RunnableWithMessageHistory`] to (optionally) trim
/// the merged `[history + input]` Vec before it's passed to the inner
/// runnable. Returning a shorter Vec is fine; the wrapper re-uses the
/// returned value as the inner input.
pub type HistoryTrimmer = Arc<dyn Fn(Vec<Message>) -> Vec<Message> + Send + Sync>;

/// Wraps a `Runnable<Vec<Message>, Message>` with per-session history.
///
/// On each invoke:
/// 1. Resolve session-id (default: from `SessionKey` in `config.extras`,
///    overridable via [`Self::with_session_resolver`]).
/// 2. Read history for the session, concat `[history, input]`, optionally
///    trim via [`Self::with_trimmer`], and call the inner runnable.
/// 3. Append `[…input, output]` to the store.
///
/// All side-effects can be swapped out:
/// - **store** — implement [`HistoryStore`] for Redis / SQL / S3.
/// - **session resolver** — derive ids from anywhere (URL path, JWT, …).
/// - **trimmer** — plug in `trim_messages` or any custom strategy to
///   keep the inner call within token budget.
pub struct RunnableWithMessageHistory<R> {
    inner: R,
    store: Arc<dyn HistoryStore>,
    session_resolver: Option<SessionResolver>,
    trimmer: Option<HistoryTrimmer>,
}

impl<R> RunnableWithMessageHistory<R>
where
    R: Runnable<Vec<Message>, Message>,
{
    /// Build a wrapper with default session-resolution and no trimming.
    pub fn new(inner: R, store: Arc<dyn HistoryStore>) -> Self {
        Self {
            inner,
            store,
            session_resolver: None,
            trimmer: None,
        }
    }

    /// Override session-id resolution. The closure receives the input
    /// messages and the active config; it must return the session id.
    pub fn with_session_resolver<F>(mut self, f: F) -> Self
    where
        F: Fn(&[Message], &RunnableConfig) -> String + Send + Sync + 'static,
    {
        self.session_resolver = Some(Arc::new(f));
        self
    }

    /// Install a trimmer that runs after merging `[history, input]` and
    /// before the inner invoke. Use to enforce a token budget.
    pub fn with_trimmer<F>(mut self, f: F) -> Self
    where
        F: Fn(Vec<Message>) -> Vec<Message> + Send + Sync + 'static,
    {
        self.trimmer = Some(Arc::new(f));
        self
    }

    /// Borrow the active history store.
    pub fn store(&self) -> &Arc<dyn HistoryStore> {
        &self.store
    }
}

#[async_trait]
impl<R> Runnable<Vec<Message>, Message> for RunnableWithMessageHistory<R>
where
    R: Runnable<Vec<Message>, Message>,
{
    async fn invoke(&self, input: Vec<Message>, config: RunnableConfig) -> Result<Message> {
        let session_id = match &self.session_resolver {
            Some(f) => f(&input, &config),
            None => config
                .extras
                .get::<SessionKey>()
                .map(|k| k.id.clone())
                .unwrap_or_else(|| "default".to_string()),
        };
        let history = self.store.read(&session_id).await?;
        let mut combined = Vec::with_capacity(history.len() + input.len());
        combined.extend(history);
        combined.extend(input.iter().cloned());
        if let Some(trimmer) = &self.trimmer {
            combined = trimmer(combined);
        }

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

    #[tokio::test]
    async fn custom_session_resolver_overrides_extras() {
        let store: Arc<dyn HistoryStore> = Arc::new(InMemoryHistory::new());
        let r = RunnableWithMessageHistory::new(EchoLast, store.clone()).with_session_resolver(
            |input, _| {
                // Resolve session from the first message's content.
                input
                    .first()
                    .map(|m| format!("derived-{}", m.content()))
                    .unwrap_or_else(|| "fallback".to_string())
            },
        );

        // With the resolver, no SessionKey in extras is needed —
        // session is derived from input.
        r.invoke(vec![Message::human("alpha")], RunnableConfig::default())
            .await
            .unwrap();
        r.invoke(vec![Message::human("alpha")], RunnableConfig::default())
            .await
            .unwrap();
        // Both calls hit the same session ("derived-alpha"); store now
        // has 4 messages there.
        let history = store.read("derived-alpha").await.unwrap();
        assert_eq!(history.len(), 4);
    }

    #[tokio::test]
    async fn trimmer_applies_before_inner_invoke() {
        let store: Arc<dyn HistoryStore> = Arc::new(InMemoryHistory::new());
        let r = RunnableWithMessageHistory::new(EchoLast, store.clone()).with_trimmer(|msgs| {
            // Keep at most 2 messages.
            let keep = msgs.len().min(2);
            msgs.into_iter().rev().take(keep).rev().collect()
        });

        // Pre-seed history with several messages.
        store
            .append(
                "trim-session",
                vec![
                    Message::human("h1"),
                    Message::ai("a1"),
                    Message::human("h2"),
                ],
            )
            .await
            .unwrap();

        let mut cfg = RunnableConfig::default();
        cfg.extras.insert(SessionKey::new("trim-session"));
        let out = r.invoke(vec![Message::human("query")], cfg).await.unwrap();
        // Inner saw exactly 2 messages (the trimmer kept only the last 2).
        assert!(out.content().contains("saw 2 msgs"));
    }
}
