//! Conversation manager for orchestrating multi-turn interactions.
//!
//! This module provides:
//!
//! - [`ConversationTurn`] — a single turn in a conversation
//! - [`Conversation`] — an ordered sequence of turns with token tracking
//! - [`ConversationSummary`] — summary statistics for a conversation
//! - [`ConversationManager`] — thread-safe manager for multiple conversations
//! - [`ConversationManagerBuilder`] — builder for configuring a manager

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use rustchain_core::messages::Message;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during conversation operations.
#[derive(Debug, thiserror::Error)]
pub enum ConversationError {
    /// The requested conversation was not found.
    #[error("conversation not found: {0}")]
    NotFound(String),
    /// The maximum number of conversations has been reached.
    #[error("maximum number of conversations reached: {0}")]
    MaxConversationsReached(usize),
    /// JSON serialization/deserialization error.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// The imported data is missing a required field.
    #[error("invalid import data: {0}")]
    InvalidImport(String),
    /// Lock poisoned (internal concurrency error).
    #[error("internal lock error")]
    LockPoisoned,
}

/// Result alias for conversation operations.
pub type Result<T> = std::result::Result<T, ConversationError>;

// ---------------------------------------------------------------------------
// TurnRole
// ---------------------------------------------------------------------------

/// The role of a participant in a conversation turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TurnRole {
    /// A human user.
    User,
    /// The AI assistant.
    Assistant,
    /// A system instruction.
    System,
    /// A tool invocation or result.
    Tool,
}

// ---------------------------------------------------------------------------
// ConversationTurn
// ---------------------------------------------------------------------------

/// A single turn in a conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationTurn {
    /// The role of the participant.
    pub role: TurnRole,
    /// The textual content of the turn.
    pub content: String,
    /// When the turn was recorded.
    pub timestamp: SystemTime,
    /// Arbitrary metadata attached to the turn.
    pub metadata: HashMap<String, Value>,
    /// Estimated token count for the content (if known).
    pub token_count: Option<usize>,
}

impl ConversationTurn {
    /// Create a new turn with the given role and content.
    pub fn new(role: TurnRole, content: impl Into<String>) -> Self {
        let content = content.into();
        let estimated_tokens = estimate_tokens(&content);
        Self {
            role,
            content,
            timestamp: SystemTime::now(),
            metadata: HashMap::new(),
            token_count: Some(estimated_tokens),
        }
    }

    /// Create a new turn with explicit token count.
    pub fn with_token_count(mut self, count: usize) -> Self {
        self.token_count = Some(count);
        self
    }

    /// Attach metadata to the turn.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

/// Rough token estimation: ~4 characters per token (common heuristic).
fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

// ---------------------------------------------------------------------------
// ConversationSummary
// ---------------------------------------------------------------------------

/// Summary statistics for a conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversationSummary {
    /// Total number of turns.
    pub total_turns: usize,
    /// Number of user messages.
    pub user_messages: usize,
    /// Number of assistant messages.
    pub assistant_messages: usize,
    /// Total estimated tokens across all turns.
    pub total_tokens: usize,
    /// Duration from first to last turn (zero if fewer than 2 turns).
    pub duration: Duration,
}

// ---------------------------------------------------------------------------
// Conversation
// ---------------------------------------------------------------------------

/// An ordered sequence of conversation turns with token tracking.
#[derive(Debug, Clone)]
pub struct Conversation {
    /// Unique identifier for this conversation.
    pub id: String,
    /// The turns in chronological order.
    pub turns: Vec<ConversationTurn>,
    /// Optional system prompt prepended to the conversation.
    pub system_prompt: Option<String>,
    /// Running total of estimated tokens.
    pub total_tokens: usize,
    /// Optional maximum token budget.
    pub max_tokens: Option<usize>,
    /// When the conversation was created.
    #[allow(dead_code)]
    created_at: SystemTime,
}

impl Conversation {
    /// Create a new empty conversation.
    pub fn new(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            turns: Vec::new(),
            system_prompt: None,
            total_tokens: 0,
            max_tokens: None,
            created_at: SystemTime::now(),
        }
    }

    /// Create a new conversation with a system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Set the maximum token budget.
    pub fn with_max_tokens(mut self, max: usize) -> Self {
        self.max_tokens = Some(max);
        self
    }

    /// Add a turn to the conversation.
    pub fn add_turn(&mut self, turn: ConversationTurn) {
        let tokens = turn.token_count.unwrap_or(0);
        self.total_tokens += tokens;
        self.turns.push(turn);
    }

    /// Get all turns as a slice.
    pub fn get_turns(&self) -> &[ConversationTurn] {
        &self.turns
    }

    /// Get the most recent turn, if any.
    pub fn last_turn(&self) -> Option<&ConversationTurn> {
        self.turns.last()
    }

    /// Get all turns from the user.
    pub fn user_turns(&self) -> Vec<&ConversationTurn> {
        self.turns
            .iter()
            .filter(|t| t.role == TurnRole::User)
            .collect()
    }

    /// Get all turns from the assistant.
    pub fn assistant_turns(&self) -> Vec<&ConversationTurn> {
        self.turns
            .iter()
            .filter(|t| t.role == TurnRole::Assistant)
            .collect()
    }

    /// Truncate the conversation to fit within the given token budget.
    ///
    /// Removes the oldest turns first until the total token count is within
    /// the budget. The system prompt tokens are not counted against the budget.
    pub fn truncate_to_tokens(&mut self, max: usize) {
        while self.total_tokens > max && !self.turns.is_empty() {
            let removed = self.turns.remove(0);
            let tokens = removed.token_count.unwrap_or(0);
            self.total_tokens = self.total_tokens.saturating_sub(tokens);
        }
    }

    /// Convert the conversation to a vector of [`Message`] values suitable
    /// for sending to an LLM.
    pub fn to_messages(&self) -> Vec<Message> {
        let mut messages = Vec::new();

        if let Some(ref prompt) = self.system_prompt {
            messages.push(Message::system(prompt.clone()));
        }

        for turn in &self.turns {
            let msg = match turn.role {
                TurnRole::User => Message::human(&turn.content),
                TurnRole::Assistant => Message::ai(&turn.content),
                TurnRole::System => Message::system(&turn.content),
                TurnRole::Tool => Message::tool(&turn.content, ""),
            };
            messages.push(msg);
        }

        messages
    }

    /// Produce a summary of the conversation.
    pub fn summary(&self) -> ConversationSummary {
        let user_messages = self
            .turns
            .iter()
            .filter(|t| t.role == TurnRole::User)
            .count();
        let assistant_messages = self
            .turns
            .iter()
            .filter(|t| t.role == TurnRole::Assistant)
            .count();

        let duration = if self.turns.len() >= 2 {
            let first = self.turns.first().unwrap().timestamp;
            let last = self.turns.last().unwrap().timestamp;
            last.duration_since(first).unwrap_or(Duration::ZERO)
        } else {
            Duration::ZERO
        };

        ConversationSummary {
            total_turns: self.turns.len(),
            user_messages,
            assistant_messages,
            total_tokens: self.total_tokens,
            duration,
        }
    }
}

// ---------------------------------------------------------------------------
// ConversationManager
// ---------------------------------------------------------------------------

/// Thread-safe manager for multiple conversations.
///
/// Wraps an inner map in `Arc<RwLock<...>>` so the manager can be shared
/// across threads.
#[derive(Clone)]
pub struct ConversationManager {
    conversations: Arc<RwLock<HashMap<String, Conversation>>>,
    max_conversations: Option<usize>,
    default_max_tokens: Option<usize>,
    default_system_prompt: Option<String>,
}

impl ConversationManager {
    /// Create a new, empty conversation manager.
    pub fn new() -> Self {
        Self {
            conversations: Arc::new(RwLock::new(HashMap::new())),
            max_conversations: None,
            default_max_tokens: None,
            default_system_prompt: None,
        }
    }

    /// Create a conversation and return its ID.
    pub fn create_conversation(&self, system_prompt: Option<&str>) -> Result<String> {
        let mut map = self
            .conversations
            .write()
            .map_err(|_| ConversationError::LockPoisoned)?;

        if let Some(max) = self.max_conversations {
            if map.len() >= max {
                return Err(ConversationError::MaxConversationsReached(max));
            }
        }

        let id = uuid::Uuid::new_v4().to_string();
        let prompt = system_prompt
            .map(|s| s.to_string())
            .or_else(|| self.default_system_prompt.clone());

        let mut conv = Conversation::new(id.clone());
        if let Some(p) = prompt {
            conv = conv.with_system_prompt(p);
        }
        if let Some(max) = self.default_max_tokens {
            conv = conv.with_max_tokens(max);
        }

        map.insert(id.clone(), conv);
        Ok(id)
    }

    /// Get a clone of a conversation by ID.
    ///
    /// Returns `None` if not found.
    pub fn get_conversation(&self, id: &str) -> Result<Option<Conversation>> {
        let map = self
            .conversations
            .read()
            .map_err(|_| ConversationError::LockPoisoned)?;
        Ok(map.get(id).cloned())
    }

    /// Add a user message to the specified conversation.
    pub fn add_user_message(&self, id: &str, content: &str) -> Result<()> {
        let mut map = self
            .conversations
            .write()
            .map_err(|_| ConversationError::LockPoisoned)?;
        let conv = map
            .get_mut(id)
            .ok_or_else(|| ConversationError::NotFound(id.to_string()))?;
        conv.add_turn(ConversationTurn::new(TurnRole::User, content));
        Ok(())
    }

    /// Add an assistant message to the specified conversation.
    pub fn add_assistant_message(&self, id: &str, content: &str) -> Result<()> {
        let mut map = self
            .conversations
            .write()
            .map_err(|_| ConversationError::LockPoisoned)?;
        let conv = map
            .get_mut(id)
            .ok_or_else(|| ConversationError::NotFound(id.to_string()))?;
        conv.add_turn(ConversationTurn::new(TurnRole::Assistant, content));
        Ok(())
    }

    /// Get the most recent turns that fit within the given token budget.
    ///
    /// Returns turns in chronological order, selecting from the most recent
    /// turns first.
    pub fn get_context_window(&self, id: &str, max_tokens: usize) -> Result<Vec<ConversationTurn>> {
        let map = self
            .conversations
            .read()
            .map_err(|_| ConversationError::LockPoisoned)?;
        let conv = map
            .get(id)
            .ok_or_else(|| ConversationError::NotFound(id.to_string()))?;

        let mut result: Vec<ConversationTurn> = Vec::new();
        let mut budget = max_tokens;

        // Walk backwards, collecting turns that fit.
        for turn in conv.turns.iter().rev() {
            let tokens = turn.token_count.unwrap_or(0);
            if tokens > budget {
                break;
            }
            budget -= tokens;
            result.push(turn.clone());
        }

        // Reverse to restore chronological order.
        result.reverse();
        Ok(result)
    }

    /// List all conversations with their summaries.
    pub fn list_conversations(&self) -> Result<Vec<(String, ConversationSummary)>> {
        let map = self
            .conversations
            .read()
            .map_err(|_| ConversationError::LockPoisoned)?;
        let list = map
            .iter()
            .map(|(id, conv)| (id.clone(), conv.summary()))
            .collect();
        Ok(list)
    }

    /// Delete a conversation by ID.
    pub fn delete_conversation(&self, id: &str) -> Result<()> {
        let mut map = self
            .conversations
            .write()
            .map_err(|_| ConversationError::LockPoisoned)?;
        if map.remove(id).is_none() {
            return Err(ConversationError::NotFound(id.to_string()));
        }
        Ok(())
    }

    /// Export a conversation as a JSON [`Value`].
    pub fn export_conversation(&self, id: &str) -> Result<Value> {
        let map = self
            .conversations
            .read()
            .map_err(|_| ConversationError::LockPoisoned)?;
        let conv = map
            .get(id)
            .ok_or_else(|| ConversationError::NotFound(id.to_string()))?;

        let turns: Vec<Value> = conv
            .turns
            .iter()
            .map(|t| serde_json::to_value(t).unwrap())
            .collect();

        let export = serde_json::json!({
            "id": conv.id,
            "system_prompt": conv.system_prompt,
            "total_tokens": conv.total_tokens,
            "max_tokens": conv.max_tokens,
            "turns": turns,
        });

        Ok(export)
    }

    /// Import a conversation from a JSON [`Value`] and return its ID.
    ///
    /// The JSON should have the same structure as produced by [`export_conversation`](Self::export_conversation).
    pub fn import_conversation(&self, data: &Value) -> Result<String> {
        let id = data
            .get("id")
            .and_then(|v| v.as_str())
            .ok_or_else(|| ConversationError::InvalidImport("missing 'id' field".to_string()))?
            .to_string();

        let system_prompt = data
            .get("system_prompt")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let max_tokens = data
            .get("max_tokens")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let turns_value = data
            .get("turns")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                ConversationError::InvalidImport("missing or invalid 'turns' field".to_string())
            })?;

        let mut conv = Conversation::new(id.clone());
        conv.system_prompt = system_prompt;
        conv.max_tokens = max_tokens;

        for turn_val in turns_value {
            let turn: ConversationTurn = serde_json::from_value(turn_val.clone())?;
            conv.add_turn(turn);
        }

        let mut map = self
            .conversations
            .write()
            .map_err(|_| ConversationError::LockPoisoned)?;

        if let Some(max) = self.max_conversations {
            if map.len() >= max && !map.contains_key(&id) {
                return Err(ConversationError::MaxConversationsReached(max));
            }
        }

        map.insert(id.clone(), conv);
        Ok(id)
    }
}

impl Default for ConversationManager {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ConversationManagerBuilder
// ---------------------------------------------------------------------------

/// Builder for configuring a [`ConversationManager`].
#[derive(Debug, Default)]
pub struct ConversationManagerBuilder {
    max_conversations: Option<usize>,
    default_max_tokens: Option<usize>,
    default_system_prompt: Option<String>,
}

impl ConversationManagerBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of conversations the manager can hold.
    pub fn max_conversations(mut self, max: usize) -> Self {
        self.max_conversations = Some(max);
        self
    }

    /// Set the default maximum token budget for new conversations.
    pub fn default_max_tokens(mut self, max: usize) -> Self {
        self.default_max_tokens = Some(max);
        self
    }

    /// Set the default system prompt for new conversations.
    pub fn default_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.default_system_prompt = Some(prompt.into());
        self
    }

    /// Build the [`ConversationManager`].
    pub fn build(self) -> ConversationManager {
        ConversationManager {
            conversations: Arc::new(RwLock::new(HashMap::new())),
            max_conversations: self.max_conversations,
            default_max_tokens: self.default_max_tokens,
            default_system_prompt: self.default_system_prompt,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- ConversationTurn tests --

    #[test]
    fn test_turn_new() {
        let turn = ConversationTurn::new(TurnRole::User, "Hello");
        assert_eq!(turn.role, TurnRole::User);
        assert_eq!(turn.content, "Hello");
        assert!(turn.token_count.is_some());
        assert!(turn.metadata.is_empty());
    }

    #[test]
    fn test_turn_with_metadata() {
        let turn = ConversationTurn::new(TurnRole::Assistant, "Hi there")
            .with_metadata("model", Value::String("gpt-4".to_string()));
        assert_eq!(
            turn.metadata.get("model"),
            Some(&Value::String("gpt-4".to_string()))
        );
    }

    #[test]
    fn test_turn_with_token_count() {
        let turn = ConversationTurn::new(TurnRole::User, "test").with_token_count(42);
        assert_eq!(turn.token_count, Some(42));
    }

    #[test]
    fn test_turn_role_serialization() {
        let role = TurnRole::User;
        let json = serde_json::to_string(&role).unwrap();
        assert_eq!(json, "\"user\"");

        let deserialized: TurnRole = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized, TurnRole::User);
    }

    #[test]
    fn test_turn_serialization_roundtrip() {
        let turn = ConversationTurn::new(TurnRole::Assistant, "response")
            .with_metadata("key", Value::Bool(true));
        let json = serde_json::to_value(&turn).unwrap();
        let deserialized: ConversationTurn = serde_json::from_value(json).unwrap();
        assert_eq!(deserialized.role, TurnRole::Assistant);
        assert_eq!(deserialized.content, "response");
        assert_eq!(deserialized.metadata.get("key"), Some(&Value::Bool(true)));
    }

    // -- Conversation tests --

    #[test]
    fn test_conversation_new() {
        let conv = Conversation::new("test-id");
        assert_eq!(conv.id, "test-id");
        assert!(conv.turns.is_empty());
        assert!(conv.system_prompt.is_none());
        assert_eq!(conv.total_tokens, 0);
        assert!(conv.max_tokens.is_none());
    }

    #[test]
    fn test_conversation_with_system_prompt() {
        let conv = Conversation::new("id").with_system_prompt("Be helpful");
        assert_eq!(conv.system_prompt, Some("Be helpful".to_string()));
    }

    #[test]
    fn test_conversation_add_turn() {
        let mut conv = Conversation::new("id");
        conv.add_turn(ConversationTurn::new(TurnRole::User, "Hello"));
        assert_eq!(conv.turns.len(), 1);
        assert!(conv.total_tokens > 0);
    }

    #[test]
    fn test_conversation_get_turns() {
        let mut conv = Conversation::new("id");
        conv.add_turn(ConversationTurn::new(TurnRole::User, "Hello"));
        conv.add_turn(ConversationTurn::new(TurnRole::Assistant, "Hi"));
        assert_eq!(conv.get_turns().len(), 2);
    }

    #[test]
    fn test_conversation_last_turn() {
        let mut conv = Conversation::new("id");
        assert!(conv.last_turn().is_none());

        conv.add_turn(ConversationTurn::new(TurnRole::User, "Hello"));
        conv.add_turn(ConversationTurn::new(TurnRole::Assistant, "Hi"));
        assert_eq!(conv.last_turn().unwrap().role, TurnRole::Assistant);
    }

    #[test]
    fn test_conversation_user_turns() {
        let mut conv = Conversation::new("id");
        conv.add_turn(ConversationTurn::new(TurnRole::User, "Q1"));
        conv.add_turn(ConversationTurn::new(TurnRole::Assistant, "A1"));
        conv.add_turn(ConversationTurn::new(TurnRole::User, "Q2"));

        let user = conv.user_turns();
        assert_eq!(user.len(), 2);
        assert_eq!(user[0].content, "Q1");
        assert_eq!(user[1].content, "Q2");
    }

    #[test]
    fn test_conversation_assistant_turns() {
        let mut conv = Conversation::new("id");
        conv.add_turn(ConversationTurn::new(TurnRole::User, "Q1"));
        conv.add_turn(ConversationTurn::new(TurnRole::Assistant, "A1"));
        conv.add_turn(ConversationTurn::new(TurnRole::Assistant, "A2"));

        let asst = conv.assistant_turns();
        assert_eq!(asst.len(), 2);
    }

    #[test]
    fn test_conversation_truncate_to_tokens() {
        let mut conv = Conversation::new("id");
        // Add turns with known token counts.
        conv.add_turn(ConversationTurn::new(TurnRole::User, "a").with_token_count(100));
        conv.add_turn(ConversationTurn::new(TurnRole::Assistant, "b").with_token_count(100));
        conv.add_turn(ConversationTurn::new(TurnRole::User, "c").with_token_count(100));

        assert_eq!(conv.total_tokens, 300);

        conv.truncate_to_tokens(200);
        assert_eq!(conv.turns.len(), 2);
        assert_eq!(conv.total_tokens, 200);
        assert_eq!(conv.turns[0].content, "b");
    }

    #[test]
    fn test_conversation_truncate_removes_all_if_needed() {
        let mut conv = Conversation::new("id");
        conv.add_turn(ConversationTurn::new(TurnRole::User, "a").with_token_count(500));
        conv.truncate_to_tokens(100);
        assert!(conv.turns.is_empty());
        assert_eq!(conv.total_tokens, 0);
    }

    #[test]
    fn test_conversation_to_messages_empty() {
        let conv = Conversation::new("id");
        let msgs = conv.to_messages();
        assert!(msgs.is_empty());
    }

    #[test]
    fn test_conversation_to_messages_with_system_prompt() {
        let conv = Conversation::new("id").with_system_prompt("Be helpful");
        let msgs = conv.to_messages();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content().text(), "Be helpful");
    }

    #[test]
    fn test_conversation_to_messages_full() {
        let mut conv = Conversation::new("id").with_system_prompt("System");
        conv.add_turn(ConversationTurn::new(TurnRole::User, "Hello"));
        conv.add_turn(ConversationTurn::new(TurnRole::Assistant, "Hi"));

        let msgs = conv.to_messages();
        assert_eq!(msgs.len(), 3);

        use rustchain_core::messages::base::MessageType;
        assert_eq!(msgs[0].message_type(), MessageType::System);
        assert_eq!(msgs[1].message_type(), MessageType::Human);
        assert_eq!(msgs[2].message_type(), MessageType::Ai);
    }

    #[test]
    fn test_conversation_summary() {
        let mut conv = Conversation::new("id");
        conv.add_turn(ConversationTurn::new(TurnRole::User, "Q1").with_token_count(10));
        conv.add_turn(ConversationTurn::new(TurnRole::Assistant, "A1").with_token_count(20));
        conv.add_turn(ConversationTurn::new(TurnRole::User, "Q2").with_token_count(15));

        let summary = conv.summary();
        assert_eq!(summary.total_turns, 3);
        assert_eq!(summary.user_messages, 2);
        assert_eq!(summary.assistant_messages, 1);
        assert_eq!(summary.total_tokens, 45);
    }

    #[test]
    fn test_conversation_summary_empty() {
        let conv = Conversation::new("id");
        let summary = conv.summary();
        assert_eq!(summary.total_turns, 0);
        assert_eq!(summary.duration, Duration::ZERO);
    }

    // -- ConversationManager tests --

    #[test]
    fn test_manager_create_conversation() {
        let mgr = ConversationManager::new();
        let id = mgr.create_conversation(None).unwrap();
        assert!(!id.is_empty());

        let conv = mgr.get_conversation(&id).unwrap().unwrap();
        assert_eq!(conv.id, id);
        assert!(conv.system_prompt.is_none());
    }

    #[test]
    fn test_manager_create_with_system_prompt() {
        let mgr = ConversationManager::new();
        let id = mgr.create_conversation(Some("Be concise")).unwrap();
        let conv = mgr.get_conversation(&id).unwrap().unwrap();
        assert_eq!(conv.system_prompt, Some("Be concise".to_string()));
    }

    #[test]
    fn test_manager_add_messages() {
        let mgr = ConversationManager::new();
        let id = mgr.create_conversation(None).unwrap();

        mgr.add_user_message(&id, "Hello").unwrap();
        mgr.add_assistant_message(&id, "Hi there").unwrap();

        let conv = mgr.get_conversation(&id).unwrap().unwrap();
        assert_eq!(conv.turns.len(), 2);
        assert_eq!(conv.turns[0].role, TurnRole::User);
        assert_eq!(conv.turns[1].role, TurnRole::Assistant);
    }

    #[test]
    fn test_manager_add_message_not_found() {
        let mgr = ConversationManager::new();
        let result = mgr.add_user_message("nonexistent", "Hello");
        assert!(result.is_err());
        match result.unwrap_err() {
            ConversationError::NotFound(id) => assert_eq!(id, "nonexistent"),
            other => panic!("expected NotFound, got {:?}", other),
        }
    }

    #[test]
    fn test_manager_get_context_window() {
        let mgr = ConversationManager::new();
        let id = mgr.create_conversation(None).unwrap();

        // Add turns with known token counts via the lower-level API.
        {
            let mut map = mgr.conversations.write().unwrap();
            let conv = map.get_mut(&id).unwrap();
            conv.add_turn(ConversationTurn::new(TurnRole::User, "a").with_token_count(50));
            conv.add_turn(ConversationTurn::new(TurnRole::Assistant, "b").with_token_count(50));
            conv.add_turn(ConversationTurn::new(TurnRole::User, "c").with_token_count(50));
        }

        // Budget of 100 should get last 2 turns.
        let window = mgr.get_context_window(&id, 100).unwrap();
        assert_eq!(window.len(), 2);
        assert_eq!(window[0].content, "b");
        assert_eq!(window[1].content, "c");
    }

    #[test]
    fn test_manager_get_context_window_all_fit() {
        let mgr = ConversationManager::new();
        let id = mgr.create_conversation(None).unwrap();

        mgr.add_user_message(&id, "short").unwrap();
        mgr.add_assistant_message(&id, "reply").unwrap();

        let window = mgr.get_context_window(&id, 100_000).unwrap();
        assert_eq!(window.len(), 2);
    }

    #[test]
    fn test_manager_get_context_window_not_found() {
        let mgr = ConversationManager::new();
        let result = mgr.get_context_window("missing", 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_manager_list_conversations() {
        let mgr = ConversationManager::new();
        let id1 = mgr.create_conversation(None).unwrap();
        let id2 = mgr.create_conversation(Some("prompt")).unwrap();

        mgr.add_user_message(&id1, "Hello").unwrap();

        let list = mgr.list_conversations().unwrap();
        assert_eq!(list.len(), 2);

        let ids: Vec<&String> = list.iter().map(|(id, _)| id).collect();
        assert!(ids.contains(&&id1));
        assert!(ids.contains(&&id2));
    }

    #[test]
    fn test_manager_delete_conversation() {
        let mgr = ConversationManager::new();
        let id = mgr.create_conversation(None).unwrap();

        mgr.delete_conversation(&id).unwrap();
        assert!(mgr.get_conversation(&id).unwrap().is_none());
    }

    #[test]
    fn test_manager_delete_not_found() {
        let mgr = ConversationManager::new();
        let result = mgr.delete_conversation("missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_manager_export_conversation() {
        let mgr = ConversationManager::new();
        let id = mgr.create_conversation(Some("System prompt")).unwrap();
        mgr.add_user_message(&id, "Hello").unwrap();
        mgr.add_assistant_message(&id, "Hi").unwrap();

        let exported = mgr.export_conversation(&id).unwrap();
        assert_eq!(exported["id"].as_str().unwrap(), id);
        assert_eq!(exported["system_prompt"].as_str().unwrap(), "System prompt");
        assert_eq!(exported["turns"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_manager_export_not_found() {
        let mgr = ConversationManager::new();
        let result = mgr.export_conversation("missing");
        assert!(result.is_err());
    }

    #[test]
    fn test_manager_import_conversation() {
        let mgr = ConversationManager::new();
        let id = mgr.create_conversation(Some("prompt")).unwrap();
        mgr.add_user_message(&id, "Hello").unwrap();

        let exported = mgr.export_conversation(&id).unwrap();

        // Import into a fresh manager.
        let mgr2 = ConversationManager::new();
        let imported_id = mgr2.import_conversation(&exported).unwrap();
        assert_eq!(imported_id, id);

        let conv = mgr2.get_conversation(&imported_id).unwrap().unwrap();
        assert_eq!(conv.system_prompt, Some("prompt".to_string()));
        assert_eq!(conv.turns.len(), 1);
        assert_eq!(conv.turns[0].content, "Hello");
    }

    #[test]
    fn test_manager_import_invalid_data() {
        let mgr = ConversationManager::new();
        let result = mgr.import_conversation(&serde_json::json!({"no_id": true}));
        assert!(result.is_err());
    }

    #[test]
    fn test_manager_import_missing_turns() {
        let mgr = ConversationManager::new();
        let result = mgr.import_conversation(&serde_json::json!({"id": "x"}));
        assert!(result.is_err());
    }

    // -- ConversationManagerBuilder tests --

    #[test]
    fn test_builder_defaults() {
        let mgr = ConversationManagerBuilder::new().build();
        assert!(mgr.max_conversations.is_none());
        assert!(mgr.default_max_tokens.is_none());
        assert!(mgr.default_system_prompt.is_none());
    }

    #[test]
    fn test_builder_max_conversations() {
        let mgr = ConversationManagerBuilder::new()
            .max_conversations(2)
            .build();

        let _id1 = mgr.create_conversation(None).unwrap();
        let _id2 = mgr.create_conversation(None).unwrap();
        let result = mgr.create_conversation(None);
        assert!(result.is_err());
        match result.unwrap_err() {
            ConversationError::MaxConversationsReached(2) => {}
            other => panic!("expected MaxConversationsReached(2), got {:?}", other),
        }
    }

    #[test]
    fn test_builder_default_max_tokens() {
        let mgr = ConversationManagerBuilder::new()
            .default_max_tokens(1000)
            .build();
        let id = mgr.create_conversation(None).unwrap();
        let conv = mgr.get_conversation(&id).unwrap().unwrap();
        assert_eq!(conv.max_tokens, Some(1000));
    }

    #[test]
    fn test_builder_default_system_prompt() {
        let mgr = ConversationManagerBuilder::new()
            .default_system_prompt("Default prompt")
            .build();
        let id = mgr.create_conversation(None).unwrap();
        let conv = mgr.get_conversation(&id).unwrap().unwrap();
        assert_eq!(conv.system_prompt, Some("Default prompt".to_string()));
    }

    #[test]
    fn test_builder_system_prompt_override() {
        let mgr = ConversationManagerBuilder::new()
            .default_system_prompt("Default")
            .build();
        let id = mgr.create_conversation(Some("Override")).unwrap();
        let conv = mgr.get_conversation(&id).unwrap().unwrap();
        assert_eq!(conv.system_prompt, Some("Override".to_string()));
    }

    // -- Token estimation --

    #[test]
    fn test_estimate_tokens() {
        assert_eq!(estimate_tokens(""), 0);
        assert!(estimate_tokens("Hello, world!") > 0);
        // "Hello, world!" is 13 chars => (13+3)/4 = 4
        assert_eq!(estimate_tokens("Hello, world!"), 4);
    }

    #[test]
    fn test_conversation_with_max_tokens() {
        let conv = Conversation::new("id").with_max_tokens(5000);
        assert_eq!(conv.max_tokens, Some(5000));
    }

    #[test]
    fn test_conversation_to_messages_tool_turn() {
        let mut conv = Conversation::new("id");
        conv.add_turn(ConversationTurn::new(TurnRole::Tool, "result"));
        let msgs = conv.to_messages();
        assert_eq!(msgs.len(), 1);
        use rustchain_core::messages::base::MessageType;
        assert_eq!(msgs[0].message_type(), MessageType::Tool);
    }

    #[test]
    fn test_manager_default_impl() {
        let mgr = ConversationManager::default();
        let id = mgr.create_conversation(None).unwrap();
        assert!(!id.is_empty());
    }
}
