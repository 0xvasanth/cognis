//! Session management system for tracking agent execution sessions.
//!
//! Provides session lifecycle tracking, event recording, and replay capability.
//!
//! - [`SessionStatus`] — lifecycle states for a session
//! - [`SessionEvent`] — a timestamped event within a session
//! - [`SessionConfig`] — configuration for session behavior
//! - [`Session`] — a single execution session with events and metadata
//! - [`SessionManager`] — manages multiple sessions
//! - [`SessionReplay`] — replays session events with timeline support

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use serde_json::Value;

// ---------------------------------------------------------------------------
// SessionStatus
// ---------------------------------------------------------------------------

/// Lifecycle status of a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionStatus {
    /// The session is currently running.
    Active,
    /// The session has been paused.
    Paused,
    /// The session completed successfully.
    Completed,
    /// The session failed with an error message.
    Failed(String),
    /// The session was cancelled by the user or system.
    Cancelled,
}

impl SessionStatus {
    /// Returns `true` if the session is in a terminal state
    /// (Completed, Failed, or Cancelled).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Failed(_) | Self::Cancelled)
    }
}

impl fmt::Display for SessionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Active => write!(f, "active"),
            Self::Paused => write!(f, "paused"),
            Self::Completed => write!(f, "completed"),
            Self::Failed(msg) => write!(f, "failed: {}", msg),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

// ---------------------------------------------------------------------------
// SessionEvent
// ---------------------------------------------------------------------------

/// A timestamped event recorded during a session.
#[derive(Debug, Clone)]
pub struct SessionEvent {
    /// The type/category of the event.
    pub event_type: String,
    /// Arbitrary JSON data associated with the event.
    pub data: Value,
    /// When the event was recorded.
    pub timestamp: Instant,
    /// Monotonically increasing sequence number within the session.
    pub sequence: u64,
}

impl SessionEvent {
    /// Create a new event with the given type and data.
    ///
    /// The sequence number is set to 0 and should be assigned by the
    /// owning [`Session`] when the event is added.
    pub fn new(event_type: impl Into<String>, data: Value) -> Self {
        Self {
            event_type: event_type.into(),
            data,
            timestamp: Instant::now(),
            sequence: 0,
        }
    }

    /// Returns the elapsed duration between the session start and this event.
    pub fn elapsed_since_start(&self, session_start: Instant) -> Duration {
        self.timestamp.duration_since(session_start)
    }
}

// ---------------------------------------------------------------------------
// SessionConfig
// ---------------------------------------------------------------------------

/// Configuration for a session's behavior.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Maximum number of events the session will store.
    pub max_events: usize,
    /// Whether the session should auto-save on each event.
    pub auto_save: bool,
    /// Optional timeout after which the session is considered stale.
    pub timeout: Option<Duration>,
    /// Arbitrary tags for categorization.
    pub tags: Vec<String>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            max_events: 10_000,
            auto_save: false,
            timeout: None,
            tags: Vec::new(),
        }
    }
}

impl SessionConfig {
    /// Set the maximum number of events.
    pub fn with_max_events(mut self, max: usize) -> Self {
        self.max_events = max;
        self
    }

    /// Enable or disable auto-save.
    pub fn with_auto_save(mut self, auto_save: bool) -> Self {
        self.auto_save = auto_save;
        self
    }

    /// Set the session timeout.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Add a tag to the session configuration.
    pub fn with_tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

/// A single agent execution session with events, metadata, and lifecycle
/// tracking.
#[derive(Debug)]
pub struct Session {
    /// Unique identifier for this session.
    pub id: String,
    /// Human-readable name for this session.
    pub name: String,
    /// Current lifecycle status.
    pub status: SessionStatus,
    /// Configuration controlling session behavior.
    pub config: SessionConfig,
    /// Ordered list of events recorded in this session.
    pub events: Vec<SessionEvent>,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, Value>,
    /// When the session was created.
    pub created_at: Instant,
    /// When the session was last modified.
    pub updated_at: Instant,
    /// Internal counter for assigning event sequence numbers.
    next_sequence: u64,
}

impl Session {
    /// Create a new session with the given name and default configuration.
    pub fn new(name: impl Into<String>) -> Self {
        Self::with_config(name, SessionConfig::default())
    }

    /// Create a new session with the given name and configuration.
    pub fn with_config(name: impl Into<String>, config: SessionConfig) -> Self {
        let now = Instant::now();
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.into(),
            status: SessionStatus::Active,
            config,
            events: Vec::new(),
            metadata: HashMap::new(),
            created_at: now,
            updated_at: now,
            next_sequence: 0,
        }
    }

    /// Record a new event in this session.
    ///
    /// If the number of events would exceed `config.max_events`, the oldest
    /// event is removed to make room.
    pub fn add_event(&mut self, event_type: &str, data: Value) {
        let mut event = SessionEvent::new(event_type, data);
        event.sequence = self.next_sequence;
        self.next_sequence += 1;

        if self.events.len() >= self.config.max_events {
            self.events.remove(0);
        }

        self.events.push(event);
        self.updated_at = Instant::now();
    }

    /// Set the session status.
    pub fn set_status(&mut self, status: SessionStatus) {
        self.status = status;
        self.updated_at = Instant::now();
    }

    /// Set a metadata key-value pair.
    pub fn set_metadata(&mut self, key: &str, value: Value) {
        self.metadata.insert(key.to_string(), value);
        self.updated_at = Instant::now();
    }

    /// Returns the duration since the session was created.
    pub fn duration(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Returns the number of events recorded.
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Returns all events matching the given event type.
    pub fn events_by_type(&self, event_type: &str) -> Vec<&SessionEvent> {
        self.events
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Returns the most recent event, if any.
    pub fn last_event(&self) -> Option<&SessionEvent> {
        self.events.last()
    }

    /// Returns `true` if the session status is `Active`.
    pub fn is_active(&self) -> bool {
        self.status == SessionStatus::Active
    }

    /// Returns `true` if the session is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        self.status.is_terminal()
    }

    /// Export the full session as a JSON value.
    pub fn to_json(&self) -> Value {
        let events: Vec<Value> = self
            .events
            .iter()
            .map(|e| {
                serde_json::json!({
                    "event_type": e.event_type,
                    "data": e.data,
                    "sequence": e.sequence,
                    "elapsed_secs": e.elapsed_since_start(self.created_at).as_secs_f64(),
                })
            })
            .collect();

        serde_json::json!({
            "id": self.id,
            "name": self.name,
            "status": self.status.to_string(),
            "config": {
                "max_events": self.config.max_events,
                "auto_save": self.config.auto_save,
                "timeout_secs": self.config.timeout.map(|d| d.as_secs_f64()),
                "tags": self.config.tags,
            },
            "events": events,
            "metadata": self.metadata,
            "duration_secs": self.duration().as_secs_f64(),
            "event_count": self.event_count(),
        })
    }

    /// Produce a compact summary of the session.
    pub fn summary(&self) -> Value {
        serde_json::json!({
            "id": self.id,
            "name": self.name,
            "status": self.status.to_string(),
            "event_count": self.event_count(),
            "duration_secs": self.duration().as_secs_f64(),
        })
    }
}

// ---------------------------------------------------------------------------
// SessionManager
// ---------------------------------------------------------------------------

/// Manages multiple sessions by ID.
#[derive(Debug, Default)]
pub struct SessionManager {
    sessions: HashMap<String, Session>,
}

impl SessionManager {
    /// Create a new, empty session manager.
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    /// Create a new session with default configuration and return a reference
    /// to it.
    pub fn create_session(&mut self, name: &str) -> &Session {
        self.create_session_with_config(name, SessionConfig::default())
    }

    /// Create a new session with the given configuration and return a
    /// reference to it.
    pub fn create_session_with_config(&mut self, name: &str, config: SessionConfig) -> &Session {
        let session = Session::with_config(name, config);
        let id = session.id.clone();
        self.sessions.insert(id.clone(), session);
        self.sessions.get(&id).unwrap()
    }

    /// Look up a session by ID.
    pub fn get_session(&self, id: &str) -> Option<&Session> {
        self.sessions.get(id)
    }

    /// Look up a session by ID (mutable).
    pub fn get_session_mut(&mut self, id: &str) -> Option<&mut Session> {
        self.sessions.get_mut(id)
    }

    /// Return references to all sessions with `Active` status.
    pub fn active_sessions(&self) -> Vec<&Session> {
        self.sessions.values().filter(|s| s.is_active()).collect()
    }

    /// Return references to all sessions with `Completed` status.
    pub fn completed_sessions(&self) -> Vec<&Session> {
        self.sessions
            .values()
            .filter(|s| s.status == SessionStatus::Completed)
            .collect()
    }

    /// Remove a session by ID and return it.
    pub fn remove_session(&mut self, id: &str) -> Option<Session> {
        self.sessions.remove(id)
    }

    /// Return the total number of managed sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }

    /// List all sessions as compact JSON summaries.
    pub fn list_sessions(&self) -> Vec<Value> {
        self.sessions.values().map(|s| s.summary()).collect()
    }
}

// ---------------------------------------------------------------------------
// SessionReplay
// ---------------------------------------------------------------------------

/// Provides replay and timeline queries over a recorded session's events.
#[derive(Debug)]
pub struct SessionReplay {
    events: Vec<SessionEvent>,
    session_start: Instant,
}

impl SessionReplay {
    /// Create a replay from an existing session.
    pub fn from_session(session: &Session) -> Self {
        Self {
            events: session.events.clone(),
            session_start: session.created_at,
        }
    }

    /// Return all events in the replay.
    pub fn events(&self) -> &[SessionEvent] {
        &self.events
    }

    /// Return events whose offset from session start falls within
    /// `[start, end)`.
    pub fn events_in_range(&self, start: Duration, end: Duration) -> Vec<&SessionEvent> {
        self.events
            .iter()
            .filter(|e| {
                let offset = e.elapsed_since_start(self.session_start);
                offset >= start && offset < end
            })
            .collect()
    }

    /// Return a timeline of `(offset, event)` pairs sorted by offset.
    pub fn event_timeline(&self) -> Vec<(Duration, &SessionEvent)> {
        self.events
            .iter()
            .map(|e| (e.elapsed_since_start(self.session_start), e))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- SessionStatus --

    #[test]
    fn test_session_status_is_terminal_active() {
        assert!(!SessionStatus::Active.is_terminal());
    }

    #[test]
    fn test_session_status_is_terminal_paused() {
        assert!(!SessionStatus::Paused.is_terminal());
    }

    #[test]
    fn test_session_status_is_terminal_completed() {
        assert!(SessionStatus::Completed.is_terminal());
    }

    #[test]
    fn test_session_status_is_terminal_failed() {
        assert!(SessionStatus::Failed("err".into()).is_terminal());
    }

    #[test]
    fn test_session_status_is_terminal_cancelled() {
        assert!(SessionStatus::Cancelled.is_terminal());
    }

    #[test]
    fn test_session_status_display_active() {
        assert_eq!(SessionStatus::Active.to_string(), "active");
    }

    #[test]
    fn test_session_status_display_paused() {
        assert_eq!(SessionStatus::Paused.to_string(), "paused");
    }

    #[test]
    fn test_session_status_display_completed() {
        assert_eq!(SessionStatus::Completed.to_string(), "completed");
    }

    #[test]
    fn test_session_status_display_failed() {
        assert_eq!(
            SessionStatus::Failed("timeout".into()).to_string(),
            "failed: timeout"
        );
    }

    #[test]
    fn test_session_status_display_cancelled() {
        assert_eq!(SessionStatus::Cancelled.to_string(), "cancelled");
    }

    // -- SessionEvent --

    #[test]
    fn test_session_event_new() {
        let event = SessionEvent::new("model_call", serde_json::json!({"model": "gpt-4"}));
        assert_eq!(event.event_type, "model_call");
        assert_eq!(event.data["model"], "gpt-4");
        assert_eq!(event.sequence, 0);
    }

    #[test]
    fn test_session_event_elapsed_since_start() {
        let start = Instant::now();
        std::thread::sleep(Duration::from_millis(10));
        let event = SessionEvent::new("test", Value::Null);
        let elapsed = event.elapsed_since_start(start);
        assert!(elapsed >= Duration::from_millis(10));
    }

    // -- SessionConfig --

    #[test]
    fn test_session_config_defaults() {
        let config = SessionConfig::default();
        assert_eq!(config.max_events, 10_000);
        assert!(!config.auto_save);
        assert!(config.timeout.is_none());
        assert!(config.tags.is_empty());
    }

    #[test]
    fn test_session_config_with_max_events() {
        let config = SessionConfig::default().with_max_events(500);
        assert_eq!(config.max_events, 500);
    }

    #[test]
    fn test_session_config_with_auto_save() {
        let config = SessionConfig::default().with_auto_save(true);
        assert!(config.auto_save);
    }

    #[test]
    fn test_session_config_with_timeout() {
        let config = SessionConfig::default().with_timeout(Duration::from_secs(60));
        assert_eq!(config.timeout, Some(Duration::from_secs(60)));
    }

    #[test]
    fn test_session_config_with_tag() {
        let config = SessionConfig::default()
            .with_tag("production")
            .with_tag("v2");
        assert_eq!(config.tags, vec!["production", "v2"]);
    }

    #[test]
    fn test_session_config_builder_chain() {
        let config = SessionConfig::default()
            .with_max_events(100)
            .with_auto_save(true)
            .with_timeout(Duration::from_secs(30))
            .with_tag("test");
        assert_eq!(config.max_events, 100);
        assert!(config.auto_save);
        assert_eq!(config.timeout, Some(Duration::from_secs(30)));
        assert_eq!(config.tags, vec!["test"]);
    }

    // -- Session lifecycle --

    #[test]
    fn test_session_new() {
        let session = Session::new("my session");
        assert_eq!(session.name, "my session");
        assert_eq!(session.status, SessionStatus::Active);
        assert!(session.events.is_empty());
        assert!(session.metadata.is_empty());
        assert!(!session.id.is_empty());
    }

    #[test]
    fn test_session_with_config() {
        let config = SessionConfig::default().with_max_events(50);
        let session = Session::with_config("test", config);
        assert_eq!(session.config.max_events, 50);
        assert_eq!(session.name, "test");
    }

    #[test]
    fn test_session_add_event() {
        let mut session = Session::new("test");
        session.add_event("start", serde_json::json!({"action": "init"}));
        assert_eq!(session.event_count(), 1);
        assert_eq!(session.events[0].event_type, "start");
        assert_eq!(session.events[0].sequence, 0);
    }

    #[test]
    fn test_session_add_multiple_events_sequence() {
        let mut session = Session::new("test");
        session.add_event("a", Value::Null);
        session.add_event("b", Value::Null);
        session.add_event("c", Value::Null);
        assert_eq!(session.events[0].sequence, 0);
        assert_eq!(session.events[1].sequence, 1);
        assert_eq!(session.events[2].sequence, 2);
    }

    #[test]
    fn test_session_max_events_enforcement() {
        let config = SessionConfig::default().with_max_events(3);
        let mut session = Session::with_config("test", config);
        for i in 0..5 {
            session.add_event("evt", serde_json::json!(i));
        }
        assert_eq!(session.event_count(), 3);
        // Oldest events should have been removed; remaining are sequence 2, 3, 4
        assert_eq!(session.events[0].sequence, 2);
        assert_eq!(session.events[1].sequence, 3);
        assert_eq!(session.events[2].sequence, 4);
    }

    #[test]
    fn test_session_set_status() {
        let mut session = Session::new("test");
        assert!(session.is_active());
        session.set_status(SessionStatus::Paused);
        assert_eq!(session.status, SessionStatus::Paused);
        assert!(!session.is_active());
        assert!(!session.is_terminal());
    }

    #[test]
    fn test_session_set_status_completed() {
        let mut session = Session::new("test");
        session.set_status(SessionStatus::Completed);
        assert!(session.is_terminal());
        assert!(!session.is_active());
    }

    #[test]
    fn test_session_set_metadata() {
        let mut session = Session::new("test");
        session.set_metadata("model", serde_json::json!("claude-sonnet-4"));
        assert_eq!(
            session.metadata.get("model"),
            Some(&serde_json::json!("claude-sonnet-4"))
        );
    }

    #[test]
    fn test_session_duration() {
        let session = Session::new("test");
        let dur = session.duration();
        // Should be very small but non-negative
        assert!(dur < Duration::from_secs(5));
    }

    #[test]
    fn test_session_events_by_type() {
        let mut session = Session::new("test");
        session.add_event("model_call", serde_json::json!({"id": 1}));
        session.add_event("tool_call", serde_json::json!({"id": 2}));
        session.add_event("model_call", serde_json::json!({"id": 3}));

        let model_events = session.events_by_type("model_call");
        assert_eq!(model_events.len(), 2);
        assert_eq!(model_events[0].data["id"], 1);
        assert_eq!(model_events[1].data["id"], 3);
    }

    #[test]
    fn test_session_events_by_type_no_match() {
        let session = Session::new("test");
        let events = session.events_by_type("nonexistent");
        assert!(events.is_empty());
    }

    #[test]
    fn test_session_last_event() {
        let mut session = Session::new("test");
        assert!(session.last_event().is_none());
        session.add_event("first", Value::Null);
        session.add_event("second", Value::Null);
        assert_eq!(session.last_event().unwrap().event_type, "second");
    }

    #[test]
    fn test_session_to_json() {
        let mut session = Session::new("json test");
        session.add_event("evt", serde_json::json!({"key": "val"}));
        session.set_metadata("env", serde_json::json!("prod"));

        let json = session.to_json();
        assert_eq!(json["id"], session.id);
        assert_eq!(json["name"], "json test");
        assert_eq!(json["status"], "active");
        assert_eq!(json["event_count"], 1);
        assert!(json["duration_secs"].is_number());
        assert!(json["events"].is_array());
        assert_eq!(json["events"].as_array().unwrap().len(), 1);
        assert_eq!(json["metadata"]["env"], "prod");
        assert_eq!(json["config"]["max_events"], 10_000);
    }

    #[test]
    fn test_session_summary() {
        let mut session = Session::new("summary test");
        session.add_event("a", Value::Null);
        session.add_event("b", Value::Null);

        let summary = session.summary();
        assert_eq!(summary["id"], session.id);
        assert_eq!(summary["name"], "summary test");
        assert_eq!(summary["status"], "active");
        assert_eq!(summary["event_count"], 2);
        assert!(summary["duration_secs"].is_number());
    }

    #[test]
    fn test_session_empty_to_json() {
        let session = Session::new("empty");
        let json = session.to_json();
        assert_eq!(json["event_count"], 0);
        assert!(json["events"].as_array().unwrap().is_empty());
    }

    // -- SessionManager --

    #[test]
    fn test_session_manager_new() {
        let mgr = SessionManager::new();
        assert_eq!(mgr.session_count(), 0);
    }

    #[test]
    fn test_session_manager_create_session() {
        let mut mgr = SessionManager::new();
        let session = mgr.create_session("test session");
        let id = session.id.clone();

        assert_eq!(session.name, "test session");
        assert_eq!(mgr.session_count(), 1);
        assert!(mgr.get_session(&id).is_some());
    }

    #[test]
    fn test_session_manager_create_session_with_config() {
        let mut mgr = SessionManager::new();
        let config = SessionConfig::default().with_max_events(100);
        let session = mgr.create_session_with_config("cfg session", config);
        assert_eq!(session.config.max_events, 100);
    }

    #[test]
    fn test_session_manager_get_session_not_found() {
        let mgr = SessionManager::new();
        assert!(mgr.get_session("nonexistent").is_none());
    }

    #[test]
    fn test_session_manager_get_session_mut() {
        let mut mgr = SessionManager::new();
        let id = mgr.create_session("test").id.to_string();

        let session = mgr.get_session_mut(&id).unwrap();
        session.add_event("mutated", Value::Null);
        assert_eq!(session.event_count(), 1);
    }

    #[test]
    fn test_session_manager_active_sessions() {
        let mut mgr = SessionManager::new();
        let id1 = mgr.create_session("s1").id.to_string();
        let _id2 = mgr.create_session("s2").id.to_string();

        mgr.get_session_mut(&id1)
            .unwrap()
            .set_status(SessionStatus::Completed);

        let active = mgr.active_sessions();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].name, "s2");
    }

    #[test]
    fn test_session_manager_completed_sessions() {
        let mut mgr = SessionManager::new();
        let id1 = mgr.create_session("s1").id.to_string();
        let _id2 = mgr.create_session("s2").id.to_string();

        mgr.get_session_mut(&id1)
            .unwrap()
            .set_status(SessionStatus::Completed);

        let completed = mgr.completed_sessions();
        assert_eq!(completed.len(), 1);
        assert_eq!(completed[0].name, "s1");
    }

    #[test]
    fn test_session_manager_remove_session() {
        let mut mgr = SessionManager::new();
        let id = mgr.create_session("removable").id.to_string();

        let removed = mgr.remove_session(&id);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().name, "removable");
        assert_eq!(mgr.session_count(), 0);
    }

    #[test]
    fn test_session_manager_remove_nonexistent() {
        let mut mgr = SessionManager::new();
        assert!(mgr.remove_session("nope").is_none());
    }

    #[test]
    fn test_session_manager_list_sessions() {
        let mut mgr = SessionManager::new();
        mgr.create_session("s1");
        mgr.create_session("s2");

        let list = mgr.list_sessions();
        assert_eq!(list.len(), 2);
        let names: Vec<&str> = list.iter().filter_map(|v| v["name"].as_str()).collect();
        assert!(names.contains(&"s1"));
        assert!(names.contains(&"s2"));
    }

    #[test]
    fn test_session_manager_duplicate_names() {
        let mut mgr = SessionManager::new();
        let s1 = mgr.create_session("same name");
        let id1 = s1.id.clone();
        let s2 = mgr.create_session("same name");
        let id2 = s2.id.clone();

        // Different IDs despite same name
        assert_ne!(id1, id2);
        assert_eq!(mgr.session_count(), 2);
    }

    // -- SessionReplay --

    #[test]
    fn test_session_replay_from_session() {
        let mut session = Session::new("replay test");
        session.add_event("a", serde_json::json!(1));
        session.add_event("b", serde_json::json!(2));

        let replay = SessionReplay::from_session(&session);
        assert_eq!(replay.events().len(), 2);
        assert_eq!(replay.events()[0].event_type, "a");
        assert_eq!(replay.events()[1].event_type, "b");
    }

    #[test]
    fn test_session_replay_event_timeline() {
        let mut session = Session::new("timeline test");
        session.add_event("first", Value::Null);
        session.add_event("second", Value::Null);

        let replay = SessionReplay::from_session(&session);
        let timeline = replay.event_timeline();
        assert_eq!(timeline.len(), 2);

        // Offsets should be non-negative and ordered
        assert!(timeline[0].0 <= timeline[1].0);
        assert_eq!(timeline[0].1.event_type, "first");
        assert_eq!(timeline[1].1.event_type, "second");
    }

    #[test]
    fn test_session_replay_events_in_range() {
        let mut session = Session::new("range test");
        session.add_event("immediate", Value::Null);

        let replay = SessionReplay::from_session(&session);
        // All events should be within a generous range
        let in_range = replay.events_in_range(Duration::ZERO, Duration::from_secs(10));
        assert_eq!(in_range.len(), 1);

        // No events in a far-future range
        let none = replay.events_in_range(Duration::from_secs(100), Duration::from_secs(200));
        assert!(none.is_empty());
    }

    #[test]
    fn test_session_replay_empty_session() {
        let session = Session::new("empty");
        let replay = SessionReplay::from_session(&session);
        assert!(replay.events().is_empty());
        assert!(replay.event_timeline().is_empty());
        assert!(replay
            .events_in_range(Duration::ZERO, Duration::from_secs(1))
            .is_empty());
    }

    #[test]
    fn test_session_failed_status_message() {
        let mut session = Session::new("fail test");
        session.set_status(SessionStatus::Failed("out of memory".into()));
        assert!(session.is_terminal());
        assert_eq!(session.status.to_string(), "failed: out of memory");
    }

    #[test]
    fn test_session_manager_default() {
        let mgr = SessionManager::default();
        assert_eq!(mgr.session_count(), 0);
    }
}
