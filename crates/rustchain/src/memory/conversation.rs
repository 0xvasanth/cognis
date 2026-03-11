//! Conversation memory for LLM chains — storing and retrieving conversation
//! history with different strategies.
//!
//! This module provides:
//!
//! - [`ConversationMessage`] / [`MessageRole`] — individual messages with metadata
//! - [`BufferMemory`] — unbounded message buffer
//! - [`WindowMemory`] — sliding window of the last N message pairs
//! - [`TokenBufferMemory`] — token-budget-aware buffer
//! - [`SummaryMemory`] — running summary of older messages plus recent history
//! - [`ConversationStore`] — persistent multi-session conversation storage
//! - [`MemorySearch`] — search utilities over conversation history
//! - [`MemoryStats`] — statistics about memory usage

use std::collections::HashMap;
use std::fmt;

use serde_json::{json, Value};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// MessageRole
// ---------------------------------------------------------------------------

/// The role of a participant in a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageRole {
    /// A human / user message.
    Human,
    /// An AI / assistant message.
    Ai,
    /// A system prompt.
    System,
    /// A function / tool result.
    Function {
        /// The name of the function that produced this message.
        name: String,
    },
}

impl MessageRole {
    /// Return a string representation of the role.
    pub fn as_str(&self) -> &str {
        match self {
            MessageRole::Human => "human",
            MessageRole::Ai => "ai",
            MessageRole::System => "system",
            MessageRole::Function { .. } => "function",
        }
    }
}

impl fmt::Display for MessageRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessageRole::Human => write!(f, "human"),
            MessageRole::Ai => write!(f, "ai"),
            MessageRole::System => write!(f, "system"),
            MessageRole::Function { name } => write!(f, "function({})", name),
        }
    }
}

// ---------------------------------------------------------------------------
// ConversationMessage
// ---------------------------------------------------------------------------

/// A single message in a conversation.
#[derive(Debug, Clone)]
pub struct ConversationMessage {
    /// The role of the message author.
    pub role: MessageRole,
    /// The textual content of the message.
    pub content: String,
    /// An ISO-8601 (or arbitrary) timestamp string.
    pub timestamp: String,
    /// Estimated token count for the message content.
    pub token_count: usize,
    /// Arbitrary key-value metadata.
    pub metadata: HashMap<String, Value>,
}

impl ConversationMessage {
    /// Create a new message with an auto-generated timestamp and estimated token count.
    pub fn new(role: MessageRole, content: impl Into<String>) -> Self {
        let content = content.into();
        let token_count = Self::estimate_tokens(&content);
        Self {
            role,
            content,
            timestamp: String::new(),
            token_count,
            metadata: HashMap::new(),
        }
    }

    /// Builder method — attach a metadata key-value pair.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Serialize the message to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "role": self.role.to_string(),
            "content": self.content,
            "timestamp": self.timestamp,
            "token_count": self.token_count,
            "metadata": self.metadata,
        })
    }

    /// Rough token estimate: split on whitespace.
    pub fn estimated_tokens(&self) -> usize {
        self.token_count
    }

    /// Internal helper for estimating tokens from a string.
    fn estimate_tokens(text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        text.split_whitespace().count()
    }
}

// ---------------------------------------------------------------------------
// BufferMemory
// ---------------------------------------------------------------------------

/// Stores all messages without any limit (unbounded).
#[derive(Debug, Default)]
pub struct BufferMemory {
    messages: Vec<ConversationMessage>,
}

impl BufferMemory {
    /// Create a new empty buffer memory.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Append a message to the buffer.
    pub fn add_message(&mut self, msg: ConversationMessage) {
        self.messages.push(msg);
    }

    /// Return all messages.
    pub fn messages(&self) -> &[ConversationMessage] {
        &self.messages
    }

    /// Remove all messages.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Number of stored messages.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Render all messages as a prompt string using the given prefixes for
    /// human and AI messages.
    pub fn to_prompt_string(&self, human_prefix: &str, ai_prefix: &str) -> String {
        let mut parts = Vec::new();
        for msg in &self.messages {
            let prefix = match &msg.role {
                MessageRole::Human => human_prefix,
                MessageRole::Ai => ai_prefix,
                MessageRole::System => "System",
                MessageRole::Function { name } => name.as_str(),
            };
            parts.push(format!("{}: {}", prefix, msg.content));
        }
        parts.join("\n")
    }
}

// ---------------------------------------------------------------------------
// WindowMemory
// ---------------------------------------------------------------------------

/// Keeps only the last `window_size` message pairs (i.e. `window_size * 2`
/// individual messages if strictly alternating human/AI).
///
/// When the number of messages exceeds `window_size * 2`, the oldest messages
/// are dropped.
#[derive(Debug)]
pub struct WindowMemory {
    messages: Vec<ConversationMessage>,
    window_size: usize,
}

impl WindowMemory {
    /// Create a window memory that retains the last `window_size` message pairs.
    pub fn new(window_size: usize) -> Self {
        Self {
            messages: Vec::new(),
            window_size,
        }
    }

    /// Add a message, trimming the oldest messages if the window is exceeded.
    pub fn add_message(&mut self, msg: ConversationMessage) {
        self.messages.push(msg);
        let max_messages = self.window_size * 2;
        if self.messages.len() > max_messages {
            let excess = self.messages.len() - max_messages;
            self.messages.drain(..excess);
        }
    }

    /// Return the current messages within the window.
    pub fn messages(&self) -> &[ConversationMessage] {
        &self.messages
    }

    /// The configured window size (in pairs).
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// Remove all messages.
    pub fn clear(&mut self) {
        self.messages.clear();
    }
}

// ---------------------------------------------------------------------------
// TokenBufferMemory
// ---------------------------------------------------------------------------

/// Stores messages up to a cumulative token budget. When a new message would
/// exceed the budget, the oldest messages are evicted until the budget is met.
#[derive(Debug)]
pub struct TokenBufferMemory {
    messages: Vec<ConversationMessage>,
    max_tokens: usize,
    current_tokens: usize,
}

impl TokenBufferMemory {
    /// Create a new token buffer memory with the given maximum token budget.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            messages: Vec::new(),
            max_tokens,
            current_tokens: 0,
        }
    }

    /// Add a message, evicting oldest messages if the token budget would be exceeded.
    pub fn add_message(&mut self, msg: ConversationMessage) {
        let msg_tokens = msg.token_count;
        // If a single message exceeds the budget, still store it (evict everything else).
        while self.current_tokens + msg_tokens > self.max_tokens && !self.messages.is_empty() {
            let removed = self.messages.remove(0);
            self.current_tokens = self.current_tokens.saturating_sub(removed.token_count);
        }
        self.current_tokens += msg_tokens;
        self.messages.push(msg);
    }

    /// Return the stored messages.
    pub fn messages(&self) -> &[ConversationMessage] {
        &self.messages
    }

    /// Current total token count across all stored messages.
    pub fn total_tokens(&self) -> usize {
        self.current_tokens
    }

    /// Remaining token budget.
    pub fn remaining_tokens(&self) -> usize {
        self.max_tokens.saturating_sub(self.current_tokens)
    }

    /// Remove all messages and reset the token count.
    pub fn clear(&mut self) {
        self.messages.clear();
        self.current_tokens = 0;
    }
}

// ---------------------------------------------------------------------------
// SummaryMemory
// ---------------------------------------------------------------------------

/// Keeps a running textual summary of older messages plus a small buffer of
/// recent messages. The caller is responsible for producing the summary (e.g.
/// via an LLM call) and feeding it back with [`set_summary`](SummaryMemory::set_summary).
#[derive(Debug, Default)]
pub struct SummaryMemory {
    summary: String,
    recent: Vec<ConversationMessage>,
}

impl SummaryMemory {
    /// Create a new empty summary memory.
    pub fn new() -> Self {
        Self {
            summary: String::new(),
            recent: Vec::new(),
        }
    }

    /// Add a message to the recent buffer.
    pub fn add_message(&mut self, msg: ConversationMessage) {
        self.recent.push(msg);
    }

    /// Replace the running summary text.
    pub fn set_summary(&mut self, summary: String) {
        self.summary = summary;
    }

    /// Return the current running summary.
    pub fn summary(&self) -> &str {
        &self.summary
    }

    /// Return the recent (un-summarised) messages.
    pub fn recent_messages(&self) -> &[ConversationMessage] {
        &self.recent
    }

    /// Number of recent messages.
    pub fn recent_count(&self) -> usize {
        self.recent.len()
    }

    /// Render a prompt string that combines the summary and recent messages.
    pub fn to_prompt_string(&self) -> String {
        let mut parts = Vec::new();
        if !self.summary.is_empty() {
            parts.push(format!("Summary: {}", self.summary));
        }
        for msg in &self.recent {
            parts.push(format!("{}: {}", msg.role, msg.content));
        }
        parts.join("\n")
    }
}

// ---------------------------------------------------------------------------
// ConversationStore
// ---------------------------------------------------------------------------

/// A session record inside [`ConversationStore`].
#[derive(Debug, Clone)]
struct Session {
    name: String,
    messages: Vec<ConversationMessage>,
}

/// Persistent conversation storage with multiple named sessions.
#[derive(Debug, Default)]
pub struct ConversationStore {
    sessions: HashMap<String, Session>,
}

impl ConversationStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self {
            sessions: HashMap::new(),
        }
    }

    /// Create a new session with the given name. Returns the session ID.
    pub fn create_session(&mut self, name: impl Into<String>) -> String {
        let id = Uuid::new_v4().to_string();
        self.sessions.insert(
            id.clone(),
            Session {
                name: name.into(),
                messages: Vec::new(),
            },
        );
        id
    }

    /// Get the messages for a session by ID.
    pub fn get_session(&self, id: &str) -> Option<&Vec<ConversationMessage>> {
        self.sessions.get(id).map(|s| &s.messages)
    }

    /// Add a message to a session.
    pub fn add_to_session(&mut self, id: &str, msg: ConversationMessage) {
        if let Some(session) = self.sessions.get_mut(id) {
            session.messages.push(msg);
        }
    }

    /// List all sessions as `(id, name)` pairs.
    pub fn list_sessions(&self) -> Vec<(String, String)> {
        self.sessions
            .iter()
            .map(|(id, s)| (id.clone(), s.name.clone()))
            .collect()
    }

    /// Delete a session by ID.
    pub fn delete_session(&mut self, id: &str) {
        self.sessions.remove(id);
    }

    /// Number of sessions.
    pub fn session_count(&self) -> usize {
        self.sessions.len()
    }
}

// ---------------------------------------------------------------------------
// MemorySearch
// ---------------------------------------------------------------------------

/// Utilities for searching through conversation history.
#[derive(Debug, Default)]
pub struct MemorySearch;

impl MemorySearch {
    /// Create a new search helper (stateless).
    pub fn new() -> Self {
        Self
    }

    /// Find messages whose content contains the query (case-insensitive).
    pub fn search_by_content<'a>(
        &self,
        messages: &'a [ConversationMessage],
        query: &str,
    ) -> Vec<&'a ConversationMessage> {
        let q = query.to_lowercase();
        messages
            .iter()
            .filter(|m| m.content.to_lowercase().contains(&q))
            .collect()
    }

    /// Find messages with a specific role.
    pub fn search_by_role<'a>(
        &self,
        messages: &'a [ConversationMessage],
        role: &MessageRole,
    ) -> Vec<&'a ConversationMessage> {
        messages.iter().filter(|m| &m.role == role).collect()
    }

    /// Find messages whose timestamp string falls within the inclusive range
    /// `[from, to]` using simple lexicographic comparison (works correctly for
    /// ISO-8601 timestamps).
    pub fn search_by_time_range<'a>(
        &self,
        messages: &'a [ConversationMessage],
        from: &str,
        to: &str,
    ) -> Vec<&'a ConversationMessage> {
        messages
            .iter()
            .filter(|m| m.timestamp.as_str() >= from && m.timestamp.as_str() <= to)
            .collect()
    }
}

// ---------------------------------------------------------------------------
// MemoryStats
// ---------------------------------------------------------------------------

/// Aggregate statistics about a set of conversation messages.
#[derive(Debug)]
pub struct MemoryStats {
    total_messages: usize,
    total_tokens: usize,
    role_counts: HashMap<String, usize>,
    total_content_length: usize,
}

impl MemoryStats {
    /// Compute statistics from a slice of messages.
    pub fn from_messages(messages: &[ConversationMessage]) -> Self {
        let mut role_counts: HashMap<String, usize> = HashMap::new();
        let mut total_tokens = 0;
        let mut total_content_length = 0;

        for msg in messages {
            *role_counts.entry(msg.role.to_string()).or_insert(0) += 1;
            total_tokens += msg.token_count;
            total_content_length += msg.content.len();
        }

        Self {
            total_messages: messages.len(),
            total_tokens,
            role_counts,
            total_content_length,
        }
    }

    /// Total number of messages.
    pub fn total_messages(&self) -> usize {
        self.total_messages
    }

    /// Total token count across all messages.
    pub fn total_tokens(&self) -> usize {
        self.total_tokens
    }

    /// Message counts keyed by role string.
    pub fn messages_by_role(&self) -> &HashMap<String, usize> {
        &self.role_counts
    }

    /// Average content length (in characters) per message, or 0 if empty.
    pub fn average_message_length(&self) -> usize {
        if self.total_messages == 0 {
            0
        } else {
            self.total_content_length / self.total_messages
        }
    }

    /// Serialize statistics to JSON.
    pub fn to_json(&self) -> Value {
        json!({
            "total_messages": self.total_messages,
            "total_tokens": self.total_tokens,
            "messages_by_role": self.role_counts,
            "average_message_length": self.average_message_length(),
        })
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────

    fn human(content: &str) -> ConversationMessage {
        ConversationMessage::new(MessageRole::Human, content)
    }

    fn ai(content: &str) -> ConversationMessage {
        ConversationMessage::new(MessageRole::Ai, content)
    }

    fn system(content: &str) -> ConversationMessage {
        ConversationMessage::new(MessageRole::System, content)
    }

    fn function_msg(name: &str, content: &str) -> ConversationMessage {
        ConversationMessage::new(
            MessageRole::Function {
                name: name.to_string(),
            },
            content,
        )
    }

    fn stamped(role: MessageRole, content: &str, ts: &str) -> ConversationMessage {
        let mut m = ConversationMessage::new(role, content);
        m.timestamp = ts.to_string();
        m
    }

    // ── MessageRole ─────────────────────────────────────────────────────

    #[test]
    fn test_role_as_str() {
        assert_eq!(MessageRole::Human.as_str(), "human");
        assert_eq!(MessageRole::Ai.as_str(), "ai");
        assert_eq!(MessageRole::System.as_str(), "system");
        assert_eq!(
            MessageRole::Function {
                name: "calc".into()
            }
            .as_str(),
            "function"
        );
    }

    #[test]
    fn test_role_display() {
        assert_eq!(format!("{}", MessageRole::Human), "human");
        assert_eq!(format!("{}", MessageRole::Ai), "ai");
        assert_eq!(format!("{}", MessageRole::System), "system");
        assert_eq!(
            format!(
                "{}",
                MessageRole::Function {
                    name: "calc".into()
                }
            ),
            "function(calc)"
        );
    }

    #[test]
    fn test_role_equality() {
        assert_eq!(MessageRole::Human, MessageRole::Human);
        assert_ne!(MessageRole::Human, MessageRole::Ai);
        assert_eq!(
            MessageRole::Function {
                name: "f".into()
            },
            MessageRole::Function {
                name: "f".into()
            }
        );
        assert_ne!(
            MessageRole::Function {
                name: "a".into()
            },
            MessageRole::Function {
                name: "b".into()
            }
        );
    }

    // ── ConversationMessage ─────────────────────────────────────────────

    #[test]
    fn test_message_new() {
        let msg = human("hello world");
        assert_eq!(msg.role, MessageRole::Human);
        assert_eq!(msg.content, "hello world");
        assert!(msg.metadata.is_empty());
    }

    #[test]
    fn test_message_token_count() {
        let msg = human("one two three four");
        assert_eq!(msg.token_count, 4);
        assert_eq!(msg.estimated_tokens(), 4);
    }

    #[test]
    fn test_message_empty_content_tokens() {
        let msg = human("");
        assert_eq!(msg.estimated_tokens(), 0);
    }

    #[test]
    fn test_message_with_metadata() {
        let msg = human("hi")
            .with_metadata("key", json!("value"))
            .with_metadata("num", json!(42));
        assert_eq!(msg.metadata.get("key").unwrap(), &json!("value"));
        assert_eq!(msg.metadata.get("num").unwrap(), &json!(42));
    }

    #[test]
    fn test_message_to_json() {
        let msg = human("test").with_metadata("k", json!("v"));
        let j = msg.to_json();
        assert_eq!(j["role"], "human");
        assert_eq!(j["content"], "test");
        assert_eq!(j["metadata"]["k"], "v");
        assert!(j["token_count"].is_number());
    }

    #[test]
    fn test_message_to_json_function_role() {
        let msg = function_msg("calculator", "42");
        let j = msg.to_json();
        assert_eq!(j["role"], "function(calculator)");
    }

    // ── BufferMemory ────────────────────────────────────────────────────

    #[test]
    fn test_buffer_new_empty() {
        let buf = BufferMemory::new();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert!(buf.messages().is_empty());
    }

    #[test]
    fn test_buffer_add_and_retrieve() {
        let mut buf = BufferMemory::new();
        buf.add_message(human("hi"));
        buf.add_message(ai("hello"));
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.messages()[0].content, "hi");
        assert_eq!(buf.messages()[1].content, "hello");
    }

    #[test]
    fn test_buffer_clear() {
        let mut buf = BufferMemory::new();
        buf.add_message(human("a"));
        buf.add_message(ai("b"));
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_to_prompt_string() {
        let mut buf = BufferMemory::new();
        buf.add_message(human("What is Rust?"));
        buf.add_message(ai("A systems language."));
        let prompt = buf.to_prompt_string("User", "Assistant");
        assert!(prompt.contains("User: What is Rust?"));
        assert!(prompt.contains("Assistant: A systems language."));
    }

    #[test]
    fn test_buffer_prompt_string_system_and_function() {
        let mut buf = BufferMemory::new();
        buf.add_message(system("You are helpful."));
        buf.add_message(function_msg("calc", "42"));
        let prompt = buf.to_prompt_string("H", "A");
        assert!(prompt.contains("System: You are helpful."));
        assert!(prompt.contains("calc: 42"));
    }

    #[test]
    fn test_buffer_default() {
        let buf = BufferMemory::default();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_buffer_many_messages() {
        let mut buf = BufferMemory::new();
        for i in 0..100 {
            buf.add_message(human(&format!("msg {}", i)));
        }
        assert_eq!(buf.len(), 100);
        assert_eq!(buf.messages()[99].content, "msg 99");
    }

    // ── WindowMemory ────────────────────────────────────────────────────

    #[test]
    fn test_window_new() {
        let win = WindowMemory::new(3);
        assert_eq!(win.window_size(), 3);
        assert!(win.messages().is_empty());
    }

    #[test]
    fn test_window_within_limit() {
        let mut win = WindowMemory::new(3);
        win.add_message(human("h1"));
        win.add_message(ai("a1"));
        win.add_message(human("h2"));
        win.add_message(ai("a2"));
        // 4 messages, window_size=3 => max 6 messages, so all fit
        assert_eq!(win.messages().len(), 4);
    }

    #[test]
    fn test_window_exceeds_limit() {
        let mut win = WindowMemory::new(2); // max 4 messages
        win.add_message(human("h1"));
        win.add_message(ai("a1"));
        win.add_message(human("h2"));
        win.add_message(ai("a2"));
        win.add_message(human("h3"));
        win.add_message(ai("a3"));
        // 6 messages added, max 4 kept
        assert_eq!(win.messages().len(), 4);
        assert_eq!(win.messages()[0].content, "h2");
        assert_eq!(win.messages()[3].content, "a3");
    }

    #[test]
    fn test_window_clear() {
        let mut win = WindowMemory::new(5);
        win.add_message(human("x"));
        win.clear();
        assert!(win.messages().is_empty());
    }

    #[test]
    fn test_window_size_one() {
        let mut win = WindowMemory::new(1); // max 2 messages
        win.add_message(human("h1"));
        win.add_message(ai("a1"));
        win.add_message(human("h2"));
        win.add_message(ai("a2"));
        assert_eq!(win.messages().len(), 2);
        assert_eq!(win.messages()[0].content, "h2");
    }

    // ── TokenBufferMemory ───────────────────────────────────────────────

    #[test]
    fn test_token_buffer_new() {
        let tb = TokenBufferMemory::new(100);
        assert!(tb.messages().is_empty());
        assert_eq!(tb.total_tokens(), 0);
        assert_eq!(tb.remaining_tokens(), 100);
    }

    #[test]
    fn test_token_buffer_add_within_budget() {
        let mut tb = TokenBufferMemory::new(100);
        tb.add_message(human("one two three")); // 3 tokens
        tb.add_message(ai("four five")); // 2 tokens
        assert_eq!(tb.messages().len(), 2);
        assert_eq!(tb.total_tokens(), 5);
        assert_eq!(tb.remaining_tokens(), 95);
    }

    #[test]
    fn test_token_buffer_eviction() {
        let mut tb = TokenBufferMemory::new(5);
        tb.add_message(human("one two three")); // 3 tokens
        tb.add_message(ai("four five")); // 2 tokens => total 5
        tb.add_message(human("six seven eight")); // 3 tokens => evict until fits
        // After eviction: "four five" (2) + "six seven eight" (3) = 5
        assert_eq!(tb.messages().len(), 2);
        assert_eq!(tb.messages()[0].content, "four five");
        assert_eq!(tb.messages()[1].content, "six seven eight");
        assert_eq!(tb.total_tokens(), 5);
    }

    #[test]
    fn test_token_buffer_single_message_exceeds_budget() {
        let mut tb = TokenBufferMemory::new(2);
        tb.add_message(human("one two three four five")); // 5 tokens > budget 2
        // Still stored — everything else is evicted
        assert_eq!(tb.messages().len(), 1);
        assert_eq!(tb.total_tokens(), 5);
    }

    #[test]
    fn test_token_buffer_clear() {
        let mut tb = TokenBufferMemory::new(50);
        tb.add_message(human("hello world"));
        tb.clear();
        assert!(tb.messages().is_empty());
        assert_eq!(tb.total_tokens(), 0);
        assert_eq!(tb.remaining_tokens(), 50);
    }

    // ── SummaryMemory ───────────────────────────────────────────────────

    #[test]
    fn test_summary_new() {
        let sm = SummaryMemory::new();
        assert!(sm.summary().is_empty());
        assert!(sm.recent_messages().is_empty());
        assert_eq!(sm.recent_count(), 0);
    }

    #[test]
    fn test_summary_add_messages() {
        let mut sm = SummaryMemory::new();
        sm.add_message(human("hi"));
        sm.add_message(ai("hello"));
        assert_eq!(sm.recent_count(), 2);
        assert_eq!(sm.recent_messages()[0].content, "hi");
    }

    #[test]
    fn test_summary_set_summary() {
        let mut sm = SummaryMemory::new();
        sm.set_summary("The user asked about Rust.".to_string());
        assert_eq!(sm.summary(), "The user asked about Rust.");
    }

    #[test]
    fn test_summary_to_prompt_string_with_summary() {
        let mut sm = SummaryMemory::new();
        sm.set_summary("Previous: user discussed Rust".to_string());
        sm.add_message(human("What about async?"));
        sm.add_message(ai("Async uses futures."));
        let prompt = sm.to_prompt_string();
        assert!(prompt.contains("Summary: Previous: user discussed Rust"));
        assert!(prompt.contains("human: What about async?"));
        assert!(prompt.contains("ai: Async uses futures."));
    }

    #[test]
    fn test_summary_to_prompt_string_without_summary() {
        let mut sm = SummaryMemory::new();
        sm.add_message(human("hello"));
        let prompt = sm.to_prompt_string();
        assert!(!prompt.contains("Summary:"));
        assert!(prompt.contains("human: hello"));
    }

    #[test]
    fn test_summary_default() {
        let sm = SummaryMemory::default();
        assert!(sm.summary().is_empty());
    }

    // ── ConversationStore ───────────────────────────────────────────────

    #[test]
    fn test_store_new() {
        let store = ConversationStore::new();
        assert_eq!(store.session_count(), 0);
        assert!(store.list_sessions().is_empty());
    }

    #[test]
    fn test_store_create_session() {
        let mut store = ConversationStore::new();
        let id = store.create_session("test session");
        assert!(!id.is_empty());
        assert_eq!(store.session_count(), 1);
    }

    #[test]
    fn test_store_get_session() {
        let mut store = ConversationStore::new();
        let id = store.create_session("my chat");
        let session = store.get_session(&id);
        assert!(session.is_some());
        assert!(session.unwrap().is_empty());
    }

    #[test]
    fn test_store_get_nonexistent_session() {
        let store = ConversationStore::new();
        assert!(store.get_session("fake-id").is_none());
    }

    #[test]
    fn test_store_add_to_session() {
        let mut store = ConversationStore::new();
        let id = store.create_session("chat");
        store.add_to_session(&id, human("hello"));
        store.add_to_session(&id, ai("hi there"));
        let msgs = store.get_session(&id).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].content, "hello");
        assert_eq!(msgs[1].content, "hi there");
    }

    #[test]
    fn test_store_add_to_nonexistent_session() {
        let mut store = ConversationStore::new();
        // Should silently do nothing
        store.add_to_session("nonexistent", human("msg"));
        assert_eq!(store.session_count(), 0);
    }

    #[test]
    fn test_store_list_sessions() {
        let mut store = ConversationStore::new();
        let id1 = store.create_session("first");
        let id2 = store.create_session("second");
        let sessions = store.list_sessions();
        assert_eq!(sessions.len(), 2);
        let ids: Vec<&str> = sessions.iter().map(|(id, _)| id.as_str()).collect();
        assert!(ids.contains(&id1.as_str()));
        assert!(ids.contains(&id2.as_str()));
    }

    #[test]
    fn test_store_delete_session() {
        let mut store = ConversationStore::new();
        let id = store.create_session("doomed");
        store.add_to_session(&id, human("msg"));
        store.delete_session(&id);
        assert_eq!(store.session_count(), 0);
        assert!(store.get_session(&id).is_none());
    }

    #[test]
    fn test_store_delete_nonexistent() {
        let mut store = ConversationStore::new();
        store.delete_session("nope"); // should not panic
        assert_eq!(store.session_count(), 0);
    }

    #[test]
    fn test_store_multiple_sessions_isolated() {
        let mut store = ConversationStore::new();
        let id1 = store.create_session("s1");
        let id2 = store.create_session("s2");
        store.add_to_session(&id1, human("in s1"));
        store.add_to_session(&id2, human("in s2"));
        assert_eq!(store.get_session(&id1).unwrap().len(), 1);
        assert_eq!(store.get_session(&id2).unwrap().len(), 1);
        assert_eq!(store.get_session(&id1).unwrap()[0].content, "in s1");
        assert_eq!(store.get_session(&id2).unwrap()[0].content, "in s2");
    }

    // ── MemorySearch ────────────────────────────────────────────────────

    #[test]
    fn test_search_by_content() {
        let msgs = vec![
            human("Tell me about Rust"),
            ai("Rust is great"),
            human("What about Python?"),
        ];
        let search = MemorySearch::new();
        let results = search.search_by_content(&msgs, "rust");
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_by_content_no_match() {
        let msgs = vec![human("hello"), ai("world")];
        let search = MemorySearch::new();
        let results = search.search_by_content(&msgs, "golang");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_by_content_case_insensitive() {
        let msgs = vec![human("HELLO WORLD")];
        let search = MemorySearch::new();
        let results = search.search_by_content(&msgs, "hello");
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn test_search_by_role() {
        let msgs = vec![
            human("q1"),
            ai("a1"),
            human("q2"),
            system("sys"),
        ];
        let search = MemorySearch::new();
        let humans = search.search_by_role(&msgs, &MessageRole::Human);
        assert_eq!(humans.len(), 2);
        let systems = search.search_by_role(&msgs, &MessageRole::System);
        assert_eq!(systems.len(), 1);
    }

    #[test]
    fn test_search_by_role_function() {
        let msgs = vec![
            function_msg("calc", "42"),
            function_msg("search", "results"),
            human("ok"),
        ];
        let search = MemorySearch::new();
        let results = search.search_by_role(
            &msgs,
            &MessageRole::Function {
                name: "calc".into(),
            },
        );
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "42");
    }

    #[test]
    fn test_search_by_time_range() {
        let msgs = vec![
            stamped(MessageRole::Human, "old", "2025-01-01T00:00:00Z"),
            stamped(MessageRole::Ai, "mid", "2025-06-15T12:00:00Z"),
            stamped(MessageRole::Human, "new", "2025-12-31T23:59:59Z"),
        ];
        let search = MemorySearch::new();
        let results =
            search.search_by_time_range(&msgs, "2025-06-01T00:00:00Z", "2025-12-31T23:59:59Z");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].content, "mid");
        assert_eq!(results[1].content, "new");
    }

    #[test]
    fn test_search_by_time_range_empty_timestamps() {
        let msgs = vec![human("no timestamp")];
        let search = MemorySearch::new();
        // Empty string < any ISO timestamp, so should not match "2025-..." range
        let results = search.search_by_time_range(&msgs, "2025-01-01", "2025-12-31");
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_by_time_range_no_match() {
        let msgs = vec![
            stamped(MessageRole::Human, "a", "2024-01-01T00:00:00Z"),
        ];
        let search = MemorySearch::new();
        let results = search.search_by_time_range(&msgs, "2025-01-01", "2025-12-31");
        assert!(results.is_empty());
    }

    // ── MemoryStats ─────────────────────────────────────────────────────

    #[test]
    fn test_stats_empty() {
        let stats = MemoryStats::from_messages(&[]);
        assert_eq!(stats.total_messages(), 0);
        assert_eq!(stats.total_tokens(), 0);
        assert_eq!(stats.average_message_length(), 0);
        assert!(stats.messages_by_role().is_empty());
    }

    #[test]
    fn test_stats_basic() {
        let msgs = vec![
            human("hello world"),       // 2 tokens, 11 chars
            ai("hi there friend"),      // 3 tokens, 15 chars
            human("bye"),               // 1 token, 3 chars
        ];
        let stats = MemoryStats::from_messages(&msgs);
        assert_eq!(stats.total_messages(), 3);
        assert_eq!(stats.total_tokens(), 6);
        assert_eq!(*stats.messages_by_role().get("human").unwrap(), 2);
        assert_eq!(*stats.messages_by_role().get("ai").unwrap(), 1);
        // avg length: (11 + 15 + 3) / 3 = 9
        assert_eq!(stats.average_message_length(), 9);
    }

    #[test]
    fn test_stats_to_json() {
        let msgs = vec![human("one two"), ai("three")];
        let stats = MemoryStats::from_messages(&msgs);
        let j = stats.to_json();
        assert_eq!(j["total_messages"], 2);
        assert!(j["total_tokens"].is_number());
        assert!(j["messages_by_role"].is_object());
        assert!(j["average_message_length"].is_number());
    }

    #[test]
    fn test_stats_function_role() {
        let msgs = vec![
            function_msg("calc", "42"),
            function_msg("search", "result"),
        ];
        let stats = MemoryStats::from_messages(&msgs);
        // Each function role has a different Display string
        assert_eq!(
            *stats.messages_by_role().get("function(calc)").unwrap(),
            1
        );
        assert_eq!(
            *stats.messages_by_role().get("function(search)").unwrap(),
            1
        );
    }

    #[test]
    fn test_stats_all_roles() {
        let msgs = vec![
            system("sys"),
            human("h"),
            ai("a"),
            function_msg("f", "r"),
        ];
        let stats = MemoryStats::from_messages(&msgs);
        assert_eq!(stats.messages_by_role().len(), 4);
    }
}
