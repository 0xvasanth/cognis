//! Chat session manager with persistence and lifecycle management.
//!
//! This module provides:
//!
//! - [`ChatSession`] -- a single conversation with metadata and lifecycle state
//! - [`SessionStatus`] -- lifecycle states (Active, Archived, Deleted)
//! - [`SessionSummary`] -- lightweight view of a session for listing
//! - [`SessionStorage`] -- trait for pluggable persistence backends
//! - [`InMemorySessionStorage`] -- thread-safe in-memory storage
//! - [`FileSessionStorage`] -- JSON-file-per-session storage
//! - [`SessionManager`] -- high-level API for managing multiple concurrent sessions
//! - [`SessionManagerBuilder`] -- builder for configuring a `SessionManager`

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::RwLock;

use cognis_core::error::{Result, CognisError};
use cognis_core::messages::Message;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the current time as an ISO-8601-ish UTC string without pulling in
/// the `chrono` crate.  Format: `YYYY-MM-DDTHH:MM:SSZ`.
fn now_iso() -> String {
    let d = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = d.as_secs();
    // Re-derive calendar fields from epoch seconds (UTC).
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Simple Gregorian calendar from day count since 1970-01-01.
    let (year, month, day) = days_to_ymd(days);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

fn days_to_ymd(mut days: u64) -> (u64, u64, u64) {
    let mut year = 1970u64;
    loop {
        let days_in_year = if is_leap(year) { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }
    let leap = is_leap(year);
    let month_days: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut month = 1u64;
    for &md in &month_days {
        if days < md {
            break;
        }
        days -= md;
        month += 1;
    }
    (year, month, days + 1)
}

fn is_leap(y: u64) -> bool {
    y.is_multiple_of(4) && (!y.is_multiple_of(100) || y.is_multiple_of(400))
}

// ---------------------------------------------------------------------------
// SessionStatus
// ---------------------------------------------------------------------------

/// Lifecycle state of a chat session.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionStatus {
    Active,
    Archived,
    Deleted,
}

impl std::fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Archived => write!(f, "archived"),
            Self::Deleted => write!(f, "deleted"),
        }
    }
}

// ---------------------------------------------------------------------------
// ChatSession
// ---------------------------------------------------------------------------

/// A single conversation with metadata, message history, and lifecycle state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    /// Unique session identifier.
    pub id: String,
    /// Optional human-readable title.
    pub title: Option<String>,
    /// Ordered conversation messages.
    pub messages: Vec<Message>,
    /// Arbitrary key-value metadata.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    /// ISO-8601 creation timestamp.
    pub created_at: String,
    /// ISO-8601 last-update timestamp.
    pub updated_at: String,
    /// Lifecycle status.
    pub status: SessionStatus,
    /// If set, automatically trim oldest messages when the count exceeds this.
    pub max_messages: Option<usize>,
}

impl ChatSession {
    /// Create a new active session with a generated UUID.
    pub fn new(title: Option<String>) -> Self {
        let now = now_iso();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            title,
            messages: Vec::new(),
            metadata: HashMap::new(),
            created_at: now.clone(),
            updated_at: now,
            status: SessionStatus::Active,
            max_messages: None,
        }
    }

    /// Build a [`SessionSummary`] from this session.
    pub fn summary(&self) -> SessionSummary {
        let last_message_preview = self.messages.last().map(|m| {
            let text = m.content().text();
            if text.len() > 100 {
                format!("{}...", &text[..100])
            } else {
                text
            }
        });

        SessionSummary {
            id: self.id.clone(),
            title: self.title.clone(),
            message_count: self.messages.len(),
            status: self.status,
            created_at: self.created_at.clone(),
            updated_at: self.updated_at.clone(),
            last_message_preview,
        }
    }

    /// Trim messages to `max_messages` if the limit is set and exceeded.
    fn auto_trim(&mut self) {
        if let Some(max) = self.max_messages {
            if self.messages.len() > max {
                let start = self.messages.len() - max;
                self.messages = self.messages.split_off(start);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SessionSummary
// ---------------------------------------------------------------------------

/// Lightweight view of a session for listing/searching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: String,
    pub title: Option<String>,
    pub message_count: usize,
    pub status: SessionStatus,
    pub created_at: String,
    pub updated_at: String,
    pub last_message_preview: Option<String>,
}

// ---------------------------------------------------------------------------
// SessionStorage trait
// ---------------------------------------------------------------------------

/// Pluggable persistence backend for chat sessions.
#[async_trait]
pub trait SessionStorage: Send + Sync {
    /// Persist a session (insert or update).
    async fn save(&self, session: &ChatSession) -> Result<()>;
    /// Load a session by ID, returning `None` if not found.
    async fn load(&self, id: &str) -> Result<Option<ChatSession>>;
    /// List all known session IDs.
    async fn list(&self) -> Result<Vec<String>>;
    /// Delete a session from the store.
    async fn delete(&self, id: &str) -> Result<()>;
}

// ---------------------------------------------------------------------------
// InMemorySessionStorage
// ---------------------------------------------------------------------------

/// Thread-safe, in-memory session storage backed by a `HashMap`.
#[derive(Debug, Clone, Default)]
pub struct InMemorySessionStorage {
    sessions: Arc<RwLock<HashMap<String, ChatSession>>>,
}

impl InMemorySessionStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

#[async_trait]
impl SessionStorage for InMemorySessionStorage {
    async fn save(&self, session: &ChatSession) -> Result<()> {
        self.sessions
            .write()
            .await
            .insert(session.id.clone(), session.clone());
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<ChatSession>> {
        Ok(self.sessions.read().await.get(id).cloned())
    }

    async fn list(&self) -> Result<Vec<String>> {
        let mut ids: Vec<String> = self.sessions.read().await.keys().cloned().collect();
        ids.sort();
        Ok(ids)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        self.sessions.write().await.remove(id);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FileSessionStorage
// ---------------------------------------------------------------------------

/// File-system session storage — each session is a JSON file at
/// `{base_dir}/{session_id}.json`.
#[derive(Debug, Clone)]
pub struct FileSessionStorage {
    base_dir: PathBuf,
}

impl FileSessionStorage {
    /// Create a new file-based session storage. The directory is created if it
    /// does not exist.
    pub fn new(base_dir: impl Into<PathBuf>) -> Result<Self> {
        let base_dir = base_dir.into();
        std::fs::create_dir_all(&base_dir).map_err(|e| {
            CognisError::Other(format!(
                "Failed to create session storage directory {:?}: {}",
                base_dir, e
            ))
        })?;
        Ok(Self { base_dir })
    }

    fn session_path(&self, id: &str) -> PathBuf {
        self.base_dir.join(format!("{}.json", id))
    }
}

#[async_trait]
impl SessionStorage for FileSessionStorage {
    async fn save(&self, session: &ChatSession) -> Result<()> {
        let path = self.session_path(&session.id);
        let data = serde_json::to_string_pretty(session)?;
        std::fs::write(&path, data).map_err(|e| {
            CognisError::Other(format!("Failed to write session file {:?}: {}", path, e))
        })?;
        Ok(())
    }

    async fn load(&self, id: &str) -> Result<Option<ChatSession>> {
        let path = self.session_path(id);
        if !path.exists() {
            return Ok(None);
        }
        let data = std::fs::read_to_string(&path).map_err(|e| {
            CognisError::Other(format!("Failed to read session file {:?}: {}", path, e))
        })?;
        let session: ChatSession = serde_json::from_str(&data)?;
        Ok(Some(session))
    }

    async fn list(&self) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        let entries = std::fs::read_dir(&self.base_dir).map_err(|e| {
            CognisError::Other(format!(
                "Failed to read session directory {:?}: {}",
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
                    ids.push(stem.to_string());
                }
            }
        }
        ids.sort();
        Ok(ids)
    }

    async fn delete(&self, id: &str) -> Result<()> {
        let path = self.session_path(id);
        if path.exists() {
            std::fs::remove_file(&path).map_err(|e| {
                CognisError::Other(format!("Failed to delete session file {:?}: {}", path, e))
            })?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------

/// High-level manager for multiple concurrent chat sessions.
///
/// Wraps a [`SessionStorage`] backend and provides the full session lifecycle
/// API: create, retrieve, list, archive, delete, message management, search,
/// and import/export.
pub struct SessionManager {
    storage: Arc<dyn SessionStorage>,
    max_sessions: Option<usize>,
    default_max_messages: Option<usize>,
}

impl SessionManager {
    /// Create a `SessionManager` with the given storage backend.
    pub fn new(storage: Arc<dyn SessionStorage>) -> Self {
        Self {
            storage,
            max_sessions: None,
            default_max_messages: None,
        }
    }

    /// Return a [`SessionManagerBuilder`].
    pub fn builder() -> SessionManagerBuilder {
        SessionManagerBuilder::default()
    }

    // -- Session lifecycle -------------------------------------------------

    /// Create a new active session, optionally with a title.
    pub async fn create_session(&self, title: Option<String>) -> Result<ChatSession> {
        // Check max_sessions limit.
        if let Some(max) = self.max_sessions {
            let ids = self.storage.list().await?;
            if ids.len() >= max {
                return Err(CognisError::Other(format!(
                    "Maximum number of sessions ({}) reached",
                    max
                )));
            }
        }

        let mut session = ChatSession::new(title);
        session.max_messages = self.default_max_messages;
        self.storage.save(&session).await?;
        Ok(session)
    }

    /// Retrieve a session by ID.
    pub async fn get_session(&self, id: &str) -> Result<Option<ChatSession>> {
        self.storage.load(id).await
    }

    /// List summaries of all sessions.
    pub async fn list_sessions(&self) -> Result<Vec<SessionSummary>> {
        let ids = self.storage.list().await?;
        let mut summaries = Vec::with_capacity(ids.len());
        for id in &ids {
            if let Some(session) = self.storage.load(id).await? {
                summaries.push(session.summary());
            }
        }
        Ok(summaries)
    }

    /// List summaries of active sessions only.
    pub async fn list_active_sessions(&self) -> Result<Vec<SessionSummary>> {
        let all = self.list_sessions().await?;
        Ok(all
            .into_iter()
            .filter(|s| s.status == SessionStatus::Active)
            .collect())
    }

    /// Delete a session. Returns `true` if the session existed.
    pub async fn delete_session(&self, id: &str) -> Result<bool> {
        let existed = self.storage.load(id).await?.is_some();
        if existed {
            self.storage.delete(id).await?;
        }
        Ok(existed)
    }

    /// Archive a session (set status to `Archived`).
    pub async fn archive_session(&self, id: &str) -> Result<()> {
        let mut session = self
            .storage
            .load(id)
            .await?
            .ok_or_else(|| CognisError::Other(format!("Session not found: {}", id)))?;
        session.status = SessionStatus::Archived;
        session.updated_at = now_iso();
        self.storage.save(&session).await
    }

    // -- Message management ------------------------------------------------

    /// Add a message to a session.
    pub async fn add_message(&self, session_id: &str, message: Message) -> Result<()> {
        let mut session =
            self.storage.load(session_id).await?.ok_or_else(|| {
                CognisError::Other(format!("Session not found: {}", session_id))
            })?;
        session.messages.push(message);
        session.auto_trim();
        session.updated_at = now_iso();
        self.storage.save(&session).await
    }

    /// Get all messages in a session.
    pub async fn get_messages(&self, session_id: &str) -> Result<Vec<Message>> {
        let session =
            self.storage.load(session_id).await?.ok_or_else(|| {
                CognisError::Other(format!("Session not found: {}", session_id))
            })?;
        Ok(session.messages)
    }

    /// Get the last `n` messages from a session.
    pub async fn get_messages_window(
        &self,
        session_id: &str,
        last_n: usize,
    ) -> Result<Vec<Message>> {
        let messages = self.get_messages(session_id).await?;
        if messages.len() <= last_n {
            Ok(messages)
        } else {
            Ok(messages[messages.len() - last_n..].to_vec())
        }
    }

    /// Clear all messages in a session (session itself remains).
    pub async fn clear_messages(&self, session_id: &str) -> Result<()> {
        let mut session =
            self.storage.load(session_id).await?.ok_or_else(|| {
                CognisError::Other(format!("Session not found: {}", session_id))
            })?;
        session.messages.clear();
        session.updated_at = now_iso();
        self.storage.save(&session).await
    }

    // -- Search ------------------------------------------------------------

    /// Search sessions by title or message content (case-insensitive substring).
    pub async fn search_sessions(&self, query: &str) -> Result<Vec<SessionSummary>> {
        let query_lower = query.to_lowercase();
        let ids = self.storage.list().await?;
        let mut results = Vec::new();
        for id in &ids {
            if let Some(session) = self.storage.load(id).await? {
                let title_match = session
                    .title
                    .as_ref()
                    .map(|t| t.to_lowercase().contains(&query_lower))
                    .unwrap_or(false);

                let content_match = session
                    .messages
                    .iter()
                    .any(|m| m.content().text().to_lowercase().contains(&query_lower));

                if title_match || content_match {
                    results.push(session.summary());
                }
            }
        }
        Ok(results)
    }

    // -- Import / Export ---------------------------------------------------

    /// Export a session as a pretty-printed JSON string.
    pub async fn export_session(&self, id: &str) -> Result<String> {
        let session = self
            .storage
            .load(id)
            .await?
            .ok_or_else(|| CognisError::Other(format!("Session not found: {}", id)))?;
        let json = serde_json::to_string_pretty(&session)?;
        Ok(json)
    }

    /// Import a session from a JSON string. The session is saved to storage
    /// and returned.
    pub async fn import_session(&self, json: &str) -> Result<ChatSession> {
        let session: ChatSession = serde_json::from_str(json)?;
        self.storage.save(&session).await?;
        Ok(session)
    }
}

// ---------------------------------------------------------------------------
// SessionManagerBuilder
// ---------------------------------------------------------------------------

/// Builder for constructing a [`SessionManager`] with optional configuration.
#[derive(Default)]
pub struct SessionManagerBuilder {
    storage: Option<Arc<dyn SessionStorage>>,
    max_sessions: Option<usize>,
    default_max_messages: Option<usize>,
}

impl SessionManagerBuilder {
    /// Set the storage backend.
    pub fn storage(mut self, storage: Arc<dyn SessionStorage>) -> Self {
        self.storage = Some(storage);
        self
    }

    /// Set the maximum number of sessions allowed.
    pub fn max_sessions(mut self, limit: usize) -> Self {
        self.max_sessions = Some(limit);
        self
    }

    /// Set the default maximum messages per session.
    pub fn default_max_messages(mut self, limit: usize) -> Self {
        self.default_max_messages = Some(limit);
        self
    }

    /// Build the `SessionManager`. Falls back to `InMemorySessionStorage` if
    /// no storage backend was specified.
    pub fn build(self) -> SessionManager {
        let storage = self
            .storage
            .unwrap_or_else(|| Arc::new(InMemorySessionStorage::new()));
        SessionManager {
            storage,
            max_sessions: self.max_sessions,
            default_max_messages: self.default_max_messages,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_manager() -> SessionManager {
        SessionManager::new(Arc::new(InMemorySessionStorage::new()))
    }

    // 1. Create session
    #[tokio::test]
    async fn test_create_session() {
        let mgr = make_manager();
        let session = mgr
            .create_session(Some("Test Session".into()))
            .await
            .unwrap();
        assert_eq!(session.title.as_deref(), Some("Test Session"));
        assert_eq!(session.status, SessionStatus::Active);
        assert!(session.messages.is_empty());
        assert!(!session.id.is_empty());
    }

    // 2. Get session
    #[tokio::test]
    async fn test_get_session() {
        let mgr = make_manager();
        let session = mgr.create_session(Some("Hello".into())).await.unwrap();
        let loaded = mgr.get_session(&session.id).await.unwrap().unwrap();
        assert_eq!(loaded.id, session.id);
        assert_eq!(loaded.title, session.title);
    }

    // 3. List sessions
    #[tokio::test]
    async fn test_list_sessions() {
        let mgr = make_manager();
        mgr.create_session(Some("A".into())).await.unwrap();
        mgr.create_session(Some("B".into())).await.unwrap();
        let list = mgr.list_sessions().await.unwrap();
        assert_eq!(list.len(), 2);
    }

    // 4. Delete session
    #[tokio::test]
    async fn test_delete_session() {
        let mgr = make_manager();
        let session = mgr.create_session(None).await.unwrap();
        let deleted = mgr.delete_session(&session.id).await.unwrap();
        assert!(deleted);
        assert!(mgr.get_session(&session.id).await.unwrap().is_none());

        // Deleting non-existent returns false.
        let deleted_again = mgr.delete_session(&session.id).await.unwrap();
        assert!(!deleted_again);
    }

    // 5. Archive session
    #[tokio::test]
    async fn test_archive_session() {
        let mgr = make_manager();
        let session = mgr.create_session(Some("To Archive".into())).await.unwrap();
        mgr.archive_session(&session.id).await.unwrap();
        let loaded = mgr.get_session(&session.id).await.unwrap().unwrap();
        assert_eq!(loaded.status, SessionStatus::Archived);
    }

    // 6. Add messages
    #[tokio::test]
    async fn test_add_messages() {
        let mgr = make_manager();
        let session = mgr.create_session(None).await.unwrap();
        mgr.add_message(&session.id, Message::human("Hi"))
            .await
            .unwrap();
        mgr.add_message(&session.id, Message::ai("Hello!"))
            .await
            .unwrap();
        let msgs = mgr.get_messages(&session.id).await.unwrap();
        assert_eq!(msgs.len(), 2);
    }

    // 7. Get messages window
    #[tokio::test]
    async fn test_get_messages_window() {
        let mgr = make_manager();
        let session = mgr.create_session(None).await.unwrap();
        for i in 0..10 {
            mgr.add_message(&session.id, Message::human(format!("msg {}", i)))
                .await
                .unwrap();
        }
        let window = mgr.get_messages_window(&session.id, 3).await.unwrap();
        assert_eq!(window.len(), 3);
        assert_eq!(window[0].content().text(), "msg 7");
        assert_eq!(window[2].content().text(), "msg 9");
    }

    // 8. Clear messages
    #[tokio::test]
    async fn test_clear_messages() {
        let mgr = make_manager();
        let session = mgr.create_session(None).await.unwrap();
        mgr.add_message(&session.id, Message::human("Hi"))
            .await
            .unwrap();
        mgr.clear_messages(&session.id).await.unwrap();
        let msgs = mgr.get_messages(&session.id).await.unwrap();
        assert!(msgs.is_empty());
    }

    // 9. Search sessions
    #[tokio::test]
    async fn test_search_sessions() {
        let mgr = make_manager();
        let s1 = mgr
            .create_session(Some("Weather chat".into()))
            .await
            .unwrap();
        let s2 = mgr
            .create_session(Some("Code review".into()))
            .await
            .unwrap();
        mgr.add_message(&s2.id, Message::human("Please review my weather code"))
            .await
            .unwrap();

        // Search by title.
        let results = mgr.search_sessions("weather").await.unwrap();
        assert_eq!(results.len(), 2); // title match + content match

        // Search by content only.
        let results = mgr.search_sessions("review").await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, s2.id);

        // No results.
        let results = mgr.search_sessions("nonexistent").await.unwrap();
        assert!(results.is_empty());

        // Ensure s1 is referenced so the compiler doesn't warn.
        let _ = s1;
    }

    // 10. Export/import roundtrip
    #[tokio::test]
    async fn test_export_import_roundtrip() {
        let mgr = make_manager();
        let session = mgr.create_session(Some("Roundtrip".into())).await.unwrap();
        mgr.add_message(&session.id, Message::human("Hello"))
            .await
            .unwrap();
        mgr.add_message(&session.id, Message::ai("World"))
            .await
            .unwrap();

        let json = mgr.export_session(&session.id).await.unwrap();

        // Import into a fresh manager.
        let mgr2 = make_manager();
        let imported = mgr2.import_session(&json).await.unwrap();
        assert_eq!(imported.id, session.id);
        assert_eq!(imported.title.as_deref(), Some("Roundtrip"));
        assert_eq!(imported.messages.len(), 2);
    }

    // 11. Session summary
    #[tokio::test]
    async fn test_session_summary() {
        let mgr = make_manager();
        let session = mgr
            .create_session(Some("Summary Test".into()))
            .await
            .unwrap();
        mgr.add_message(&session.id, Message::human("First"))
            .await
            .unwrap();
        mgr.add_message(&session.id, Message::ai("Second"))
            .await
            .unwrap();

        let loaded = mgr.get_session(&session.id).await.unwrap().unwrap();
        let summary = loaded.summary();
        assert_eq!(summary.id, session.id);
        assert_eq!(summary.title.as_deref(), Some("Summary Test"));
        assert_eq!(summary.message_count, 2);
        assert_eq!(summary.status, SessionStatus::Active);
        assert_eq!(summary.last_message_preview.as_deref(), Some("Second"));
    }

    // 12. Max messages auto-trim
    #[tokio::test]
    async fn test_max_messages_auto_trim() {
        let mgr = SessionManager::builder().default_max_messages(3).build();
        let session = mgr.create_session(None).await.unwrap();
        for i in 0..10 {
            mgr.add_message(&session.id, Message::human(format!("msg {}", i)))
                .await
                .unwrap();
        }
        let msgs = mgr.get_messages(&session.id).await.unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].content().text(), "msg 7");
        assert_eq!(msgs[1].content().text(), "msg 8");
        assert_eq!(msgs[2].content().text(), "msg 9");
    }

    // 13. InMemorySessionStorage CRUD
    #[tokio::test]
    async fn test_inmemory_storage_crud() {
        let storage = InMemorySessionStorage::new();
        let session = ChatSession::new(Some("Test".into()));
        let id = session.id.clone();

        // Save & load.
        storage.save(&session).await.unwrap();
        let loaded = storage.load(&id).await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().title.as_deref(), Some("Test"));

        // List.
        let ids = storage.list().await.unwrap();
        assert_eq!(ids.len(), 1);

        // Delete.
        storage.delete(&id).await.unwrap();
        assert!(storage.load(&id).await.unwrap().is_none());
        assert!(storage.list().await.unwrap().is_empty());
    }

    // 14. FileSessionStorage CRUD (tempdir)
    #[tokio::test]
    async fn test_file_storage_crud() {
        let dir = tempfile::tempdir().unwrap();
        let storage = FileSessionStorage::new(dir.path()).unwrap();
        let session = ChatSession::new(Some("File Test".into()));
        let id = session.id.clone();

        // Save & load.
        storage.save(&session).await.unwrap();
        let loaded = storage.load(&id).await.unwrap();
        assert!(loaded.is_some());
        assert_eq!(loaded.unwrap().title.as_deref(), Some("File Test"));

        // List.
        let ids = storage.list().await.unwrap();
        assert_eq!(ids.len(), 1);

        // Delete.
        storage.delete(&id).await.unwrap();
        assert!(storage.load(&id).await.unwrap().is_none());
        assert!(storage.list().await.unwrap().is_empty());

        // Persistence across instances.
        let s2 = ChatSession::new(Some("Persist".into()));
        let id2 = s2.id.clone();
        storage.save(&s2).await.unwrap();

        let storage2 = FileSessionStorage::new(dir.path()).unwrap();
        let loaded = storage2.load(&id2).await.unwrap();
        assert!(loaded.is_some());
    }

    // 15. Builder pattern
    #[tokio::test]
    async fn test_builder_pattern() {
        let storage = Arc::new(InMemorySessionStorage::new());
        let mgr = SessionManager::builder()
            .storage(storage.clone())
            .max_sessions(2)
            .default_max_messages(5)
            .build();

        mgr.create_session(None).await.unwrap();
        mgr.create_session(None).await.unwrap();

        // Third session should fail due to max_sessions limit.
        let result = mgr.create_session(None).await;
        assert!(result.is_err());
    }

    // 16. List active sessions filters archived
    #[tokio::test]
    async fn test_list_active_sessions() {
        let mgr = make_manager();
        let s1 = mgr.create_session(Some("Active".into())).await.unwrap();
        let s2 = mgr.create_session(Some("To Archive".into())).await.unwrap();
        mgr.archive_session(&s2.id).await.unwrap();

        let active = mgr.list_active_sessions().await.unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, s1.id);
    }
}
