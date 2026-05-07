//! Chat history store backends for managing session-based message histories.
//!
//! This module provides:
//!
//! - [`ChatHistoryStore`] -- trait for session-based message storage
//! - [`InMemoryChatHistory`] -- thread-safe in-memory implementation
//! - [`FileChatHistory`] -- file-system backed implementation (JSON per session)
//! - [`ChatHistoryMemory`] -- adapter bridging `ChatHistoryStore` to [`BaseMemory`]
//! - Pruning utilities for managing history size

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::RwLock;

use cognis_core::error::{CognisError, Result};
use cognis_core::messages::{count_tokens_approximately, Message};

use super::BaseMemory;

/// Trait for session-based chat message storage.
///
/// Implementations manage messages indexed by a session identifier string,
/// supporting multi-session use cases such as per-user or per-conversation
/// histories.
#[async_trait]
pub trait ChatHistoryStore: Send + Sync {
    /// Retrieve all messages for the given session.
    async fn get_messages(&self, session_id: &str) -> Result<Vec<Message>>;

    /// Append a single message to a session.
    async fn add_message(&self, session_id: &str, message: Message) -> Result<()>;

    /// Append multiple messages to a session.
    async fn add_messages(&self, session_id: &str, messages: &[Message]) -> Result<()>;

    /// Remove all messages from a session (session entry remains).
    async fn clear(&self, session_id: &str) -> Result<()>;

    /// List all known session identifiers.
    async fn list_sessions(&self) -> Result<Vec<String>>;

    /// Delete a session and all its messages entirely.
    async fn delete_session(&self, session_id: &str) -> Result<()>;
}

// ---------------------------------------------------------------------------
// InMemoryChatHistory
// ---------------------------------------------------------------------------

/// Thread-safe, in-memory chat history store.
///
/// Messages are stored in a `HashMap<String, Vec<Message>>` protected by an
/// `Arc<RwLock>`, making it safe for concurrent access from multiple tasks.
#[derive(Debug, Clone)]
pub struct InMemoryChatHistory {
    sessions: Arc<RwLock<HashMap<String, Vec<Message>>>>,
}

impl InMemoryChatHistory {
    /// Create a new, empty in-memory chat history store.
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl Default for InMemoryChatHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ChatHistoryStore for InMemoryChatHistory {
    async fn get_messages(&self, session_id: &str) -> Result<Vec<Message>> {
        let sessions = self.sessions.read().await;
        Ok(sessions.get(session_id).cloned().unwrap_or_default())
    }

    async fn add_message(&self, session_id: &str, message: Message) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions
            .entry(session_id.to_string())
            .or_default()
            .push(message);
        Ok(())
    }

    async fn add_messages(&self, session_id: &str, messages: &[Message]) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions
            .entry(session_id.to_string())
            .or_default()
            .extend(messages.iter().cloned());
        Ok(())
    }

    async fn clear(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        if let Some(msgs) = sessions.get_mut(session_id) {
            msgs.clear();
        }
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        let sessions = self.sessions.read().await;
        let mut keys: Vec<String> = sessions.keys().cloned().collect();
        keys.sort();
        Ok(keys)
    }

    async fn delete_session(&self, session_id: &str) -> Result<()> {
        let mut sessions = self.sessions.write().await;
        sessions.remove(session_id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FileChatHistory
// ---------------------------------------------------------------------------

/// File-system backed chat history store.
///
/// Each session is persisted as a JSON file at `{base_dir}/{session_id}.json`.
/// The directory is created automatically if it does not exist.
#[derive(Debug, Clone)]
pub struct FileChatHistory {
    base_dir: PathBuf,
}

impl FileChatHistory {
    /// Create a new file-based chat history store.
    ///
    /// The `base_dir` will be created if it does not already exist.
    pub fn new(base_dir: impl Into<PathBuf>) -> Result<Self> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(&base_dir).map_err(|e| {
            CognisError::Other(format!(
                "Failed to create chat history directory {:?}: {}",
                base_dir, e
            ))
        })?;
        Ok(Self { base_dir })
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.base_dir.join(format!("{}.json", session_id))
    }

    fn read_session_sync(&self, session_id: &str) -> Result<Vec<Message>> {
        let path = self.session_path(session_id);
        if !path.exists() {
            return Ok(Vec::new());
        }
        let data = std::fs::read_to_string(&path).map_err(|e| {
            CognisError::Other(format!("Failed to read session file {:?}: {}", path, e))
        })?;
        let messages: Vec<Message> = serde_json::from_str(&data)?;
        Ok(messages)
    }

    fn write_session_sync(&self, session_id: &str, messages: &[Message]) -> Result<()> {
        let path = self.session_path(session_id);
        let data = serde_json::to_string_pretty(messages)?;
        std::fs::write(&path, data).map_err(|e| {
            CognisError::Other(format!("Failed to write session file {:?}: {}", path, e))
        })?;
        Ok(())
    }
}

#[async_trait]
impl ChatHistoryStore for FileChatHistory {
    async fn get_messages(&self, session_id: &str) -> Result<Vec<Message>> {
        self.read_session_sync(session_id)
    }

    async fn add_message(&self, session_id: &str, message: Message) -> Result<()> {
        let mut messages = self.read_session_sync(session_id)?;
        messages.push(message);
        self.write_session_sync(session_id, &messages)
    }

    async fn add_messages(&self, session_id: &str, messages: &[Message]) -> Result<()> {
        let mut existing = self.read_session_sync(session_id)?;
        existing.extend(messages.iter().cloned());
        self.write_session_sync(session_id, &existing)
    }

    async fn clear(&self, session_id: &str) -> Result<()> {
        let path = self.session_path(session_id);
        if path.exists() {
            self.write_session_sync(session_id, &[])?;
        }
        Ok(())
    }

    async fn list_sessions(&self) -> Result<Vec<String>> {
        let mut sessions = Vec::new();
        let entries = std::fs::read_dir(&self.base_dir).map_err(|e| {
            CognisError::Other(format!(
                "Failed to read chat history directory {:?}: {}",
                self.base_dir, e
            ))
        })?;
        for entry in entries {
            let entry = entry.map_err(|e| {
                CognisError::Other(format!("Failed to read directory entry: {}", e))
            })?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("json") {
                if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                    sessions.push(stem.to_string());
                }
            }
        }
        sessions.sort();
        Ok(sessions)
    }

    async fn delete_session(&self, session_id: &str) -> Result<()> {
        let path = self.session_path(session_id);
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| {
                CognisError::Other(format!("Failed to delete session file {:?}: {}", path, e))
            })?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// ChatHistoryMemory -- bridges ChatHistoryStore to BaseMemory
// ---------------------------------------------------------------------------

/// Adapter that wraps a [`ChatHistoryStore`] + session ID and implements
/// [`BaseMemory`], allowing a chat history store to be used anywhere a
/// `BaseMemory` is expected (e.g., in chains).
pub struct ChatHistoryMemory {
    store: Arc<dyn ChatHistoryStore>,
    session_id: String,
    memory_key: String,
    /// If true, return messages as a JSON array; if false, return a formatted string.
    return_messages: bool,
}

impl ChatHistoryMemory {
    /// Create a new `ChatHistoryMemory` wrapping the given store and session.
    pub fn new(store: Arc<dyn ChatHistoryStore>, session_id: impl Into<String>) -> Self {
        Self {
            store,
            session_id: session_id.into(),
            memory_key: "history".to_string(),
            return_messages: true,
        }
    }

    /// Set the memory key used in chain context.
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// Set whether to return messages as structured data or formatted text.
    pub fn with_return_messages(mut self, return_messages: bool) -> Self {
        self.return_messages = return_messages;
        self
    }
}

#[async_trait]
impl BaseMemory for ChatHistoryMemory {
    async fn load_memory_variables(&self) -> Result<HashMap<String, Value>> {
        let messages = self.store.get_messages(&self.session_id).await?;
        let mut vars = HashMap::new();

        if self.return_messages {
            let serialized: Vec<Value> = messages
                .iter()
                .map(|m| serde_json::to_value(m).unwrap_or(Value::Null))
                .collect();
            vars.insert(self.memory_key.clone(), Value::Array(serialized));
        } else {
            let buffer = cognis_core::messages::get_buffer_string(&messages, "Human", "AI");
            vars.insert(self.memory_key.clone(), Value::String(buffer));
        }

        Ok(vars)
    }

    async fn save_context(&self, input: &Message, output: &Message) -> Result<()> {
        self.store
            .add_messages(&self.session_id, &[input.clone(), output.clone()])
            .await
    }

    async fn clear(&self) -> Result<()> {
        self.store.clear(&self.session_id).await
    }

    fn memory_key(&self) -> &str {
        &self.memory_key
    }
}

// ---------------------------------------------------------------------------
// Pruning utilities
// ---------------------------------------------------------------------------

/// Keep only the last `max_messages` messages in a session.
///
/// If the session has fewer messages than the limit, this is a no-op.
pub async fn prune_by_count(
    store: &dyn ChatHistoryStore,
    session_id: &str,
    max_messages: usize,
) -> Result<()> {
    let messages = store.get_messages(session_id).await?;
    if messages.len() <= max_messages {
        return Ok(());
    }
    let kept = messages[messages.len() - max_messages..].to_vec();
    store.clear(session_id).await?;
    store.add_messages(session_id, &kept).await?;
    Ok(())
}

/// Keep only the most recent messages that fit within a token budget.
///
/// Uses the approximate token counter from `cognis_core`. Messages are
/// kept from the end (most recent first) until adding another message would
/// exceed `max_tokens`.
pub async fn prune_by_token_count(
    store: &dyn ChatHistoryStore,
    session_id: &str,
    max_tokens: usize,
) -> Result<()> {
    let messages = store.get_messages(session_id).await?;
    if messages.is_empty() {
        return Ok(());
    }

    // Walk from the end, accumulating tokens.
    let mut kept: Vec<Message> = Vec::new();
    let mut total_tokens = 0usize;
    for msg in messages.iter().rev() {
        let msg_tokens = count_tokens_approximately(std::slice::from_ref(msg), 4.0, 3.0);
        if total_tokens + msg_tokens > max_tokens && !kept.is_empty() {
            break;
        }
        total_tokens += msg_tokens;
        kept.push(msg.clone());
    }
    kept.reverse();

    if kept.len() < messages.len() {
        store.clear(session_id).await?;
        store.add_messages(session_id, &kept).await?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // InMemoryChatHistory tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_inmemory_add_and_get_messages() {
        let store = InMemoryChatHistory::new();
        store
            .add_message("s1", Message::human("Hello"))
            .await
            .unwrap();
        store
            .add_message("s1", Message::ai("Hi there"))
            .await
            .unwrap();

        let msgs = store.get_messages("s1").await.unwrap();
        assert_eq!(msgs.len(), 2);
    }

    #[tokio::test]
    async fn test_inmemory_multiple_sessions() {
        let store = InMemoryChatHistory::new();
        store.add_message("s1", Message::human("A")).await.unwrap();
        store.add_message("s2", Message::human("B")).await.unwrap();

        assert_eq!(store.get_messages("s1").await.unwrap().len(), 1);
        assert_eq!(store.get_messages("s2").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_inmemory_clear_session() {
        let store = InMemoryChatHistory::new();
        store
            .add_message("s1", Message::human("Hello"))
            .await
            .unwrap();
        store.clear("s1").await.unwrap();

        let msgs = store.get_messages("s1").await.unwrap();
        assert!(msgs.is_empty());
    }

    #[tokio::test]
    async fn test_inmemory_list_sessions() {
        let store = InMemoryChatHistory::new();
        store
            .add_message("alpha", Message::human("A"))
            .await
            .unwrap();
        store
            .add_message("beta", Message::human("B"))
            .await
            .unwrap();

        let sessions = store.list_sessions().await.unwrap();
        assert_eq!(sessions, vec!["alpha", "beta"]);
    }

    #[tokio::test]
    async fn test_inmemory_delete_session() {
        let store = InMemoryChatHistory::new();
        store
            .add_message("s1", Message::human("Hello"))
            .await
            .unwrap();
        store.delete_session("s1").await.unwrap();

        let sessions = store.list_sessions().await.unwrap();
        assert!(sessions.is_empty());
        assert!(store.get_messages("s1").await.unwrap().is_empty());
    }

    // -----------------------------------------------------------------------
    // FileChatHistory tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_file_add_and_get_messages() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileChatHistory::new(dir.path()).unwrap();

        store
            .add_message("s1", Message::human("Hello"))
            .await
            .unwrap();
        store.add_message("s1", Message::ai("Hi")).await.unwrap();

        let msgs = store.get_messages("s1").await.unwrap();
        assert_eq!(msgs.len(), 2);
    }

    #[tokio::test]
    async fn test_file_list_sessions() {
        let dir = tempfile::tempdir().unwrap();
        let store = FileChatHistory::new(dir.path()).unwrap();

        store
            .add_message("alpha", Message::human("A"))
            .await
            .unwrap();
        store
            .add_message("beta", Message::human("B"))
            .await
            .unwrap();

        let sessions = store.list_sessions().await.unwrap();
        assert_eq!(sessions, vec!["alpha", "beta"]);
    }

    #[tokio::test]
    async fn test_file_persistence_across_instances() {
        let dir = tempfile::tempdir().unwrap();

        // First instance writes messages.
        {
            let store = FileChatHistory::new(dir.path()).unwrap();
            store
                .add_message("s1", Message::human("Persisted"))
                .await
                .unwrap();
        }

        // Second instance reads them back.
        {
            let store = FileChatHistory::new(dir.path()).unwrap();
            let msgs = store.get_messages("s1").await.unwrap();
            assert_eq!(msgs.len(), 1);
        }
    }

    // -----------------------------------------------------------------------
    // ChatHistoryMemory tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_chat_history_memory_load_save() {
        let store = Arc::new(InMemoryChatHistory::new());
        let mem = ChatHistoryMemory::new(store.clone(), "sess1");

        // Initially empty.
        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_array().unwrap();
        assert!(history.is_empty());

        // Save a turn.
        mem.save_context(&Message::human("Hi"), &Message::ai("Hello"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_array().unwrap();
        assert_eq!(history.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Pruning tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_prune_by_count() {
        let store = InMemoryChatHistory::new();
        for i in 0..10 {
            store
                .add_message("s1", Message::human(format!("msg {}", i)))
                .await
                .unwrap();
        }

        prune_by_count(&store, "s1", 3).await.unwrap();
        let msgs = store.get_messages("s1").await.unwrap();
        assert_eq!(msgs.len(), 3);
        // Verify we kept the last 3.
        assert_eq!(msgs[0].content().text(), "msg 7");
        assert_eq!(msgs[1].content().text(), "msg 8");
        assert_eq!(msgs[2].content().text(), "msg 9");
    }

    #[tokio::test]
    async fn test_prune_by_token_count() {
        let store = InMemoryChatHistory::new();
        // Add messages with known lengths.
        for _ in 0..20 {
            store
                .add_message("s1", Message::human("Hello"))
                .await
                .unwrap();
        }

        // Allow roughly 20 tokens -- should keep only a few messages.
        prune_by_token_count(&store, "s1", 20).await.unwrap();
        let msgs = store.get_messages("s1").await.unwrap();
        assert!(msgs.len() < 20, "should have pruned some messages");
        assert!(!msgs.is_empty(), "should have kept at least one message");
    }

    // -----------------------------------------------------------------------
    // Thread safety test
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_thread_safety() {
        let store = Arc::new(InMemoryChatHistory::new());
        let mut handles = Vec::new();

        for i in 0..10 {
            let store = store.clone();
            handles.push(tokio::spawn(async move {
                store
                    .add_message("shared", Message::human(format!("msg {}", i)))
                    .await
                    .unwrap();
            }));
        }

        for h in handles {
            h.await.unwrap();
        }

        let msgs = store.get_messages("shared").await.unwrap();
        assert_eq!(msgs.len(), 10);
    }
}
