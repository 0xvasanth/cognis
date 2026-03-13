//! Message history management for conversations.
//!
//! Provides ordered collections of messages with various eviction strategies
//! (sliding window, token budget) and conversation turn management.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::{Result, CognisError};

/// Generates a simple ISO-like timestamp string from the current system time.
fn now_timestamp() -> String {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();
    // Simple epoch-seconds string; no chrono dependency needed.
    format!("{}", secs)
}

/// Estimates the number of tokens in a string.
///
/// Uses a simple heuristic of ~4 characters per token, consistent with
/// common approximations for English text.
fn estimate_tokens(text: &str) -> usize {
    // Rough approximation: 1 token per 4 characters, minimum 1 for non-empty.
    if text.is_empty() {
        0
    } else {
        text.len().div_ceil(4)
    }
}

/// A single message stored in a history.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HistoryMessage {
    /// The role of the message sender (e.g., "human", "ai", "system").
    pub role: String,
    /// The text content of the message.
    pub content: String,
    /// Arbitrary metadata associated with the message.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, Value>,
    /// Timestamp when the message was created (epoch seconds as string).
    pub timestamp: String,
    /// Positional index in the history.
    pub index: usize,
}

impl HistoryMessage {
    /// Serialize this message to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::to_value(self).unwrap_or(Value::Null)
    }

    /// Returns `true` if this message has the "human" role.
    pub fn is_human(&self) -> bool {
        self.role == "human"
    }

    /// Returns `true` if this message has the "ai" role.
    pub fn is_ai(&self) -> bool {
        self.role == "ai"
    }

    /// Returns `true` if this message has the "system" role.
    pub fn is_system(&self) -> bool {
        self.role == "system"
    }

    /// Returns the number of whitespace-delimited words in the content.
    pub fn word_count(&self) -> usize {
        self.content.split_whitespace().count()
    }
}

/// An ordered collection of messages representing a conversation history.
#[derive(Debug, Clone, Default)]
pub struct MessageHistory {
    messages: Vec<HistoryMessage>,
}

impl MessageHistory {
    /// Create an empty message history.
    pub fn new() -> Self {
        Self {
            messages: Vec::new(),
        }
    }

    /// Add a message with the given role and content.
    pub fn add(&mut self, role: &str, content: &str) {
        self.add_with_metadata(role, content, HashMap::new());
    }

    /// Add a message with the given role, content, and metadata.
    pub fn add_with_metadata(
        &mut self,
        role: &str,
        content: &str,
        metadata: HashMap<String, Value>,
    ) {
        let index = self.messages.len();
        self.messages.push(HistoryMessage {
            role: role.to_string(),
            content: content.to_string(),
            metadata,
            timestamp: now_timestamp(),
            index,
        });
    }

    /// Returns a slice of all messages in order.
    pub fn messages(&self) -> &[HistoryMessage] {
        &self.messages
    }

    /// Returns the last message, if any.
    pub fn last(&self) -> Option<&HistoryMessage> {
        self.messages.last()
    }

    /// Returns all messages with the specified role.
    pub fn by_role(&self, role: &str) -> Vec<&HistoryMessage> {
        self.messages.iter().filter(|m| m.role == role).collect()
    }

    /// Returns the number of messages in the history.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Returns `true` if the history contains no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }

    /// Removes all messages from the history.
    pub fn clear(&mut self) {
        self.messages.clear();
    }

    /// Serialize the entire history to a JSON value.
    pub fn to_json(&self) -> Value {
        let arr: Vec<Value> = self.messages.iter().map(|m| m.to_json()).collect();
        Value::Array(arr)
    }
}

/// A message history that maintains a fixed-size sliding window.
///
/// When the number of messages exceeds `max_messages`, the oldest messages
/// are evicted to stay within the limit.
#[derive(Debug, Clone)]
pub struct SlidingWindowHistory {
    history: MessageHistory,
    max_messages: usize,
    total_added: usize,
}

impl SlidingWindowHistory {
    /// Create a new sliding window history with the given maximum size.
    pub fn new(max_messages: usize) -> Self {
        Self {
            history: MessageHistory::new(),
            max_messages,
            total_added: 0,
        }
    }

    /// Add a message, evicting the oldest if the window is full.
    pub fn add(&mut self, role: &str, content: &str) {
        self.add_with_metadata(role, content, HashMap::new());
    }

    /// Add a message with metadata, evicting the oldest if the window is full.
    pub fn add_with_metadata(
        &mut self,
        role: &str,
        content: &str,
        metadata: HashMap<String, Value>,
    ) {
        self.total_added += 1;
        self.history.add_with_metadata(role, content, metadata);
        while self.history.len() > self.max_messages {
            self.history.messages.remove(0);
        }
    }

    /// Returns the configured window size.
    pub fn window_size(&self) -> usize {
        self.max_messages
    }

    /// Returns the total number of messages ever added, including evicted ones.
    pub fn total_added(&self) -> usize {
        self.total_added
    }

    /// Returns a slice of all messages currently in the window.
    pub fn messages(&self) -> &[HistoryMessage] {
        self.history.messages()
    }

    /// Returns the last message, if any.
    pub fn last(&self) -> Option<&HistoryMessage> {
        self.history.last()
    }

    /// Returns the number of messages currently in the window.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Returns `true` if the window is empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Removes all messages and resets the total added counter.
    pub fn clear(&mut self) {
        self.history.clear();
        self.total_added = 0;
    }

    /// Serialize the current window to JSON.
    pub fn to_json(&self) -> Value {
        self.history.to_json()
    }
}

/// A message history that keeps messages within a token budget.
///
/// Uses a simple token estimation heuristic (~4 characters per token).
/// When adding a message would exceed the budget, the oldest non-system
/// messages are evicted until there is room.
#[derive(Debug, Clone)]
pub struct TokenBudgetHistory {
    history: MessageHistory,
    max_tokens: usize,
    current_tokens: usize,
}

impl TokenBudgetHistory {
    /// Create a new token budget history with the given maximum token count.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            history: MessageHistory::new(),
            max_tokens,
            current_tokens: 0,
        }
    }

    /// Add a message, evicting oldest non-system messages if the budget is exceeded.
    pub fn add(&mut self, role: &str, content: &str) {
        let new_tokens = estimate_tokens(content);
        self.history.add(role, content);
        self.current_tokens += new_tokens;

        // Evict oldest non-system messages until we are within budget.
        while self.current_tokens > self.max_tokens {
            let evict_idx = self
                .history
                .messages
                .iter()
                .position(|m| m.role != "system");
            match evict_idx {
                Some(idx) => {
                    let removed = self.history.messages.remove(idx);
                    self.current_tokens = self
                        .current_tokens
                        .saturating_sub(estimate_tokens(&removed.content));
                }
                None => break, // Only system messages remain; cannot evict further.
            }
        }
    }

    /// Returns the current estimated token count.
    pub fn current_tokens(&self) -> usize {
        self.current_tokens
    }

    /// Returns the configured token budget.
    pub fn budget(&self) -> usize {
        self.max_tokens
    }

    /// Returns the fraction of the budget currently used (0.0 to 1.0+).
    pub fn utilization(&self) -> f64 {
        if self.max_tokens == 0 {
            return 0.0;
        }
        self.current_tokens as f64 / self.max_tokens as f64
    }

    /// Returns a slice of all messages currently in the history.
    pub fn messages(&self) -> &[HistoryMessage] {
        self.history.messages()
    }

    /// Returns the last message, if any.
    pub fn last(&self) -> Option<&HistoryMessage> {
        self.history.last()
    }

    /// Returns the number of messages currently in the history.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Returns `true` if the history is empty.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Removes all messages and resets the token counter.
    pub fn clear(&mut self) {
        self.history.clear();
        self.current_tokens = 0;
    }

    /// Serialize the current history to JSON.
    pub fn to_json(&self) -> Value {
        self.history.to_json()
    }
}

/// Represents a single conversation turn: a human message followed by an AI response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ConversationTurn {
    human: HistoryMessage,
    ai: HistoryMessage,
}

impl ConversationTurn {
    /// Create a new conversation turn from a human message and an AI response.
    pub fn new(human: HistoryMessage, ai: HistoryMessage) -> Self {
        Self { human, ai }
    }

    /// Returns a reference to the human message.
    pub fn human(&self) -> &HistoryMessage {
        &self.human
    }

    /// Returns a reference to the AI message.
    pub fn ai(&self) -> &HistoryMessage {
        &self.ai
    }

    /// Returns the estimated total tokens for both messages in this turn.
    pub fn total_tokens(&self) -> usize {
        estimate_tokens(&self.human.content) + estimate_tokens(&self.ai.content)
    }

    /// Serialize this turn to a JSON value.
    pub fn to_json(&self) -> Value {
        serde_json::json!({
            "human": self.human.to_json(),
            "ai": self.ai.to_json(),
        })
    }
}

/// Manages conversation as a sequence of human-AI turns.
#[derive(Debug, Clone, Default)]
pub struct TurnHistory {
    turns: Vec<ConversationTurn>,
    next_index: usize,
}

impl TurnHistory {
    /// Create a new empty turn history.
    pub fn new() -> Self {
        Self {
            turns: Vec::new(),
            next_index: 0,
        }
    }

    /// Add a conversation turn with the given human and AI content.
    pub fn add_turn(&mut self, human_content: &str, ai_content: &str) {
        let ts = now_timestamp();
        let human = HistoryMessage {
            role: "human".to_string(),
            content: human_content.to_string(),
            metadata: HashMap::new(),
            timestamp: ts.clone(),
            index: self.next_index,
        };
        self.next_index += 1;
        let ai = HistoryMessage {
            role: "ai".to_string(),
            content: ai_content.to_string(),
            metadata: HashMap::new(),
            timestamp: ts,
            index: self.next_index,
        };
        self.next_index += 1;
        self.turns.push(ConversationTurn::new(human, ai));
    }

    /// Returns a slice of all conversation turns.
    pub fn turns(&self) -> &[ConversationTurn] {
        &self.turns
    }

    /// Returns the last conversation turn, if any.
    pub fn last_turn(&self) -> Option<&ConversationTurn> {
        self.turns.last()
    }

    /// Returns the number of turns.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    /// Returns `true` if there are no turns.
    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Serialize all turns to a JSON value.
    pub fn to_json(&self) -> Value {
        let arr: Vec<Value> = self.turns.iter().map(|t| t.to_json()).collect();
        Value::Array(arr)
    }
}

/// Utilities for serializing and deserializing message histories.
pub struct HistorySerializer;

impl HistorySerializer {
    /// Serialize a `MessageHistory` to a JSON value.
    pub fn to_json(history: &MessageHistory) -> Value {
        history.to_json()
    }

    /// Deserialize a `MessageHistory` from a JSON value.
    ///
    /// Expects a JSON array of objects, each with at least `role` and `content` fields.
    pub fn from_json(json: &Value) -> Result<MessageHistory> {
        let arr = json.as_array().ok_or_else(|| {
            CognisError::Other("Expected a JSON array for message history".to_string())
        })?;

        let mut history = MessageHistory::new();
        for (i, item) in arr.iter().enumerate() {
            let role = item.get("role").and_then(|v| v.as_str()).ok_or_else(|| {
                CognisError::Other(format!("Missing 'role' in message at index {}", i))
            })?;
            let content = item
                .get("content")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    CognisError::Other(format!("Missing 'content' in message at index {}", i))
                })?;
            let metadata: HashMap<String, Value> = item
                .get("metadata")
                .and_then(|v| serde_json::from_value(v.clone()).ok())
                .unwrap_or_default();

            history.add_with_metadata(role, content, metadata);
        }
        Ok(history)
    }

    /// Convert a `MessageHistory` to OpenAI chat completion format.
    ///
    /// Returns a `Vec<Value>` where each element is `{"role": "...", "content": "..."}`.
    pub fn to_chat_format(history: &MessageHistory) -> Vec<Value> {
        history
            .messages()
            .iter()
            .map(|m| {
                serde_json::json!({
                    "role": m.role,
                    "content": m.content,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── HistoryMessage tests ──

    #[test]
    fn test_history_message_is_human() {
        let msg = HistoryMessage {
            role: "human".into(),
            content: "hello".into(),
            metadata: HashMap::new(),
            timestamp: "0".into(),
            index: 0,
        };
        assert!(msg.is_human());
        assert!(!msg.is_ai());
        assert!(!msg.is_system());
    }

    #[test]
    fn test_history_message_is_ai() {
        let msg = HistoryMessage {
            role: "ai".into(),
            content: "response".into(),
            metadata: HashMap::new(),
            timestamp: "0".into(),
            index: 0,
        };
        assert!(msg.is_ai());
        assert!(!msg.is_human());
    }

    #[test]
    fn test_history_message_is_system() {
        let msg = HistoryMessage {
            role: "system".into(),
            content: "you are helpful".into(),
            metadata: HashMap::new(),
            timestamp: "0".into(),
            index: 0,
        };
        assert!(msg.is_system());
    }

    #[test]
    fn test_history_message_word_count() {
        let msg = HistoryMessage {
            role: "human".into(),
            content: "how many words are here".into(),
            metadata: HashMap::new(),
            timestamp: "0".into(),
            index: 0,
        };
        assert_eq!(msg.word_count(), 5);
    }

    #[test]
    fn test_history_message_word_count_empty() {
        let msg = HistoryMessage {
            role: "human".into(),
            content: "".into(),
            metadata: HashMap::new(),
            timestamp: "0".into(),
            index: 0,
        };
        assert_eq!(msg.word_count(), 0);
    }

    #[test]
    fn test_history_message_to_json() {
        let msg = HistoryMessage {
            role: "human".into(),
            content: "test".into(),
            metadata: HashMap::new(),
            timestamp: "123".into(),
            index: 0,
        };
        let json = msg.to_json();
        assert_eq!(json["role"], "human");
        assert_eq!(json["content"], "test");
        assert_eq!(json["timestamp"], "123");
        assert_eq!(json["index"], 0);
    }

    #[test]
    fn test_history_message_with_metadata() {
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), Value::String("test".to_string()));
        let msg = HistoryMessage {
            role: "ai".into(),
            content: "reply".into(),
            metadata: meta,
            timestamp: "0".into(),
            index: 1,
        };
        let json = msg.to_json();
        assert_eq!(json["metadata"]["source"], "test");
    }

    // ── MessageHistory tests ──

    #[test]
    fn test_message_history_new_is_empty() {
        let h = MessageHistory::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn test_message_history_add_and_len() {
        let mut h = MessageHistory::new();
        h.add("human", "hello");
        h.add("ai", "hi there");
        assert_eq!(h.len(), 2);
        assert!(!h.is_empty());
    }

    #[test]
    fn test_message_history_messages_order() {
        let mut h = MessageHistory::new();
        h.add("human", "first");
        h.add("ai", "second");
        h.add("human", "third");
        let msgs = h.messages();
        assert_eq!(msgs[0].content, "first");
        assert_eq!(msgs[1].content, "second");
        assert_eq!(msgs[2].content, "third");
    }

    #[test]
    fn test_message_history_last() {
        let mut h = MessageHistory::new();
        assert!(h.last().is_none());
        h.add("human", "only");
        assert_eq!(h.last().unwrap().content, "only");
    }

    #[test]
    fn test_message_history_by_role() {
        let mut h = MessageHistory::new();
        h.add("human", "q1");
        h.add("ai", "a1");
        h.add("human", "q2");
        h.add("system", "sys");
        let humans = h.by_role("human");
        assert_eq!(humans.len(), 2);
        assert_eq!(humans[0].content, "q1");
        assert_eq!(humans[1].content, "q2");
    }

    #[test]
    fn test_message_history_clear() {
        let mut h = MessageHistory::new();
        h.add("human", "a");
        h.add("ai", "b");
        h.clear();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn test_message_history_add_with_metadata() {
        let mut h = MessageHistory::new();
        let mut meta = HashMap::new();
        meta.insert("key".to_string(), Value::String("val".to_string()));
        h.add_with_metadata("human", "content", meta);
        assert_eq!(h.messages()[0].metadata["key"], "val");
    }

    #[test]
    fn test_message_history_to_json() {
        let mut h = MessageHistory::new();
        h.add("human", "hello");
        h.add("ai", "world");
        let json = h.to_json();
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert_eq!(arr[0]["role"], "human");
        assert_eq!(arr[1]["role"], "ai");
    }

    #[test]
    fn test_message_history_index_tracking() {
        let mut h = MessageHistory::new();
        h.add("human", "a");
        h.add("ai", "b");
        h.add("human", "c");
        assert_eq!(h.messages()[0].index, 0);
        assert_eq!(h.messages()[1].index, 1);
        assert_eq!(h.messages()[2].index, 2);
    }

    // ── SlidingWindowHistory tests ──

    #[test]
    fn test_sliding_window_basic() {
        let mut sw = SlidingWindowHistory::new(3);
        sw.add("human", "a");
        sw.add("ai", "b");
        sw.add("human", "c");
        assert_eq!(sw.len(), 3);
        assert_eq!(sw.total_added(), 3);
    }

    #[test]
    fn test_sliding_window_eviction() {
        let mut sw = SlidingWindowHistory::new(2);
        sw.add("human", "first");
        sw.add("ai", "second");
        sw.add("human", "third");
        assert_eq!(sw.len(), 2);
        assert_eq!(sw.messages()[0].content, "second");
        assert_eq!(sw.messages()[1].content, "third");
        assert_eq!(sw.total_added(), 3);
    }

    #[test]
    fn test_sliding_window_eviction_multiple() {
        let mut sw = SlidingWindowHistory::new(2);
        for i in 0..10 {
            sw.add("human", &format!("msg{}", i));
        }
        assert_eq!(sw.len(), 2);
        assert_eq!(sw.total_added(), 10);
        assert_eq!(sw.messages()[0].content, "msg8");
        assert_eq!(sw.messages()[1].content, "msg9");
    }

    #[test]
    fn test_sliding_window_size() {
        let sw = SlidingWindowHistory::new(5);
        assert_eq!(sw.window_size(), 5);
    }

    #[test]
    fn test_sliding_window_clear() {
        let mut sw = SlidingWindowHistory::new(3);
        sw.add("human", "a");
        sw.add("ai", "b");
        sw.clear();
        assert!(sw.is_empty());
        assert_eq!(sw.total_added(), 0);
    }

    #[test]
    fn test_sliding_window_last() {
        let mut sw = SlidingWindowHistory::new(2);
        assert!(sw.last().is_none());
        sw.add("human", "hello");
        assert_eq!(sw.last().unwrap().content, "hello");
    }

    #[test]
    fn test_sliding_window_to_json() {
        let mut sw = SlidingWindowHistory::new(2);
        sw.add("human", "a");
        sw.add("ai", "b");
        let json = sw.to_json();
        assert_eq!(json.as_array().unwrap().len(), 2);
    }

    // ── TokenBudgetHistory tests ──

    #[test]
    fn test_token_budget_basic() {
        let mut tb = TokenBudgetHistory::new(100);
        tb.add("human", "hello");
        assert!(tb.current_tokens() > 0);
        assert_eq!(tb.budget(), 100);
    }

    #[test]
    fn test_token_budget_eviction() {
        // Budget of 10 tokens (~40 chars).
        let mut tb = TokenBudgetHistory::new(10);
        tb.add("human", "short");
        tb.add("ai", "also short text here");
        tb.add("human", "another message with more text");
        // Should have evicted some messages to stay within budget.
        assert!(tb.current_tokens() <= tb.budget());
    }

    #[test]
    fn test_token_budget_preserves_system() {
        let mut tb = TokenBudgetHistory::new(10);
        tb.add("system", "you are helpful");
        tb.add(
            "human",
            "a long human message that exceeds the budget easily",
        );
        tb.add("ai", "another long ai message that also exceeds the budget");
        // System messages should be preserved.
        let has_system = tb.messages().iter().any(|m| m.role == "system");
        assert!(has_system);
    }

    #[test]
    fn test_token_budget_utilization() {
        let mut tb = TokenBudgetHistory::new(100);
        assert_eq!(tb.utilization(), 0.0);
        tb.add("human", "hello world");
        assert!(tb.utilization() > 0.0);
        assert!(tb.utilization() <= 1.0);
    }

    #[test]
    fn test_token_budget_utilization_zero_budget() {
        let tb = TokenBudgetHistory::new(0);
        assert_eq!(tb.utilization(), 0.0);
    }

    #[test]
    fn test_token_budget_clear() {
        let mut tb = TokenBudgetHistory::new(100);
        tb.add("human", "hello");
        tb.clear();
        assert!(tb.is_empty());
        assert_eq!(tb.current_tokens(), 0);
    }

    #[test]
    fn test_token_budget_len_and_empty() {
        let mut tb = TokenBudgetHistory::new(1000);
        assert!(tb.is_empty());
        tb.add("human", "a");
        assert_eq!(tb.len(), 1);
        assert!(!tb.is_empty());
    }

    // ── ConversationTurn tests ──

    #[test]
    fn test_conversation_turn_accessors() {
        let human = HistoryMessage {
            role: "human".into(),
            content: "question".into(),
            metadata: HashMap::new(),
            timestamp: "0".into(),
            index: 0,
        };
        let ai = HistoryMessage {
            role: "ai".into(),
            content: "answer".into(),
            metadata: HashMap::new(),
            timestamp: "0".into(),
            index: 1,
        };
        let turn = ConversationTurn::new(human, ai);
        assert_eq!(turn.human().content, "question");
        assert_eq!(turn.ai().content, "answer");
    }

    #[test]
    fn test_conversation_turn_total_tokens() {
        let human = HistoryMessage {
            role: "human".into(),
            content: "hello world".into(),
            metadata: HashMap::new(),
            timestamp: "0".into(),
            index: 0,
        };
        let ai = HistoryMessage {
            role: "ai".into(),
            content: "hi".into(),
            metadata: HashMap::new(),
            timestamp: "0".into(),
            index: 1,
        };
        let turn = ConversationTurn::new(human, ai);
        assert!(turn.total_tokens() > 0);
    }

    #[test]
    fn test_conversation_turn_to_json() {
        let human = HistoryMessage {
            role: "human".into(),
            content: "q".into(),
            metadata: HashMap::new(),
            timestamp: "0".into(),
            index: 0,
        };
        let ai = HistoryMessage {
            role: "ai".into(),
            content: "a".into(),
            metadata: HashMap::new(),
            timestamp: "0".into(),
            index: 1,
        };
        let turn = ConversationTurn::new(human, ai);
        let json = turn.to_json();
        assert!(json.get("human").is_some());
        assert!(json.get("ai").is_some());
        assert_eq!(json["human"]["content"], "q");
        assert_eq!(json["ai"]["content"], "a");
    }

    // ── TurnHistory tests ──

    #[test]
    fn test_turn_history_new_is_empty() {
        let th = TurnHistory::new();
        assert!(th.is_empty());
        assert_eq!(th.len(), 0);
    }

    #[test]
    fn test_turn_history_add_turn() {
        let mut th = TurnHistory::new();
        th.add_turn("hello", "hi");
        assert_eq!(th.len(), 1);
        assert!(!th.is_empty());
    }

    #[test]
    fn test_turn_history_turns_content() {
        let mut th = TurnHistory::new();
        th.add_turn("q1", "a1");
        th.add_turn("q2", "a2");
        let turns = th.turns();
        assert_eq!(turns[0].human().content, "q1");
        assert_eq!(turns[0].ai().content, "a1");
        assert_eq!(turns[1].human().content, "q2");
        assert_eq!(turns[1].ai().content, "a2");
    }

    #[test]
    fn test_turn_history_last_turn() {
        let mut th = TurnHistory::new();
        assert!(th.last_turn().is_none());
        th.add_turn("q1", "a1");
        th.add_turn("q2", "a2");
        assert_eq!(th.last_turn().unwrap().human().content, "q2");
    }

    #[test]
    fn test_turn_history_to_json() {
        let mut th = TurnHistory::new();
        th.add_turn("hello", "hi");
        let json = th.to_json();
        let arr = json.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert!(arr[0].get("human").is_some());
        assert!(arr[0].get("ai").is_some());
    }

    #[test]
    fn test_turn_history_index_tracking() {
        let mut th = TurnHistory::new();
        th.add_turn("q1", "a1");
        th.add_turn("q2", "a2");
        assert_eq!(th.turns()[0].human().index, 0);
        assert_eq!(th.turns()[0].ai().index, 1);
        assert_eq!(th.turns()[1].human().index, 2);
        assert_eq!(th.turns()[1].ai().index, 3);
    }

    // ── HistorySerializer tests ──

    #[test]
    fn test_serializer_to_json() {
        let mut h = MessageHistory::new();
        h.add("human", "hello");
        let json = HistorySerializer::to_json(&h);
        assert!(json.is_array());
        assert_eq!(json.as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_serializer_roundtrip() {
        let mut h = MessageHistory::new();
        h.add("human", "question");
        h.add("ai", "answer");
        h.add("system", "context");

        let json = HistorySerializer::to_json(&h);
        let restored = HistorySerializer::from_json(&json).unwrap();

        assert_eq!(restored.len(), h.len());
        for (orig, rest) in h.messages().iter().zip(restored.messages().iter()) {
            assert_eq!(orig.role, rest.role);
            assert_eq!(orig.content, rest.content);
        }
    }

    #[test]
    fn test_serializer_from_json_error_not_array() {
        let json = serde_json::json!({"not": "array"});
        let result = HistorySerializer::from_json(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_serializer_from_json_error_missing_role() {
        let json = serde_json::json!([{"content": "hello"}]);
        let result = HistorySerializer::from_json(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_serializer_from_json_error_missing_content() {
        let json = serde_json::json!([{"role": "human"}]);
        let result = HistorySerializer::from_json(&json);
        assert!(result.is_err());
    }

    #[test]
    fn test_serializer_to_chat_format() {
        let mut h = MessageHistory::new();
        h.add("system", "You are helpful.");
        h.add("human", "What is 2+2?");
        h.add("ai", "4");

        let chat = HistorySerializer::to_chat_format(&h);
        assert_eq!(chat.len(), 3);
        assert_eq!(chat[0]["role"], "system");
        assert_eq!(chat[0]["content"], "You are helpful.");
        assert_eq!(chat[1]["role"], "human");
        assert_eq!(chat[1]["content"], "What is 2+2?");
        assert_eq!(chat[2]["role"], "ai");
        assert_eq!(chat[2]["content"], "4");
    }

    #[test]
    fn test_serializer_to_chat_format_empty() {
        let h = MessageHistory::new();
        let chat = HistorySerializer::to_chat_format(&h);
        assert!(chat.is_empty());
    }

    #[test]
    fn test_serializer_roundtrip_with_metadata() {
        let mut h = MessageHistory::new();
        let mut meta = HashMap::new();
        meta.insert("source".to_string(), Value::String("test".to_string()));
        h.add_with_metadata("human", "hello", meta);

        let json = HistorySerializer::to_json(&h);
        let restored = HistorySerializer::from_json(&json).unwrap();
        assert_eq!(restored.messages()[0].metadata["source"], "test");
    }

    // ── estimate_tokens tests ──

    #[test]
    fn test_estimate_tokens_empty() {
        assert_eq!(estimate_tokens(""), 0);
    }

    #[test]
    fn test_estimate_tokens_nonempty() {
        // "hello" = 5 chars => (5+3)/4 = 2 tokens
        assert_eq!(estimate_tokens("hello"), 2);
        // "hi" = 2 chars => (2+3)/4 = 1 token
        assert_eq!(estimate_tokens("hi"), 1);
    }

    #[test]
    fn test_by_role_empty_result() {
        let h = MessageHistory::new();
        assert!(h.by_role("human").is_empty());
    }
}
