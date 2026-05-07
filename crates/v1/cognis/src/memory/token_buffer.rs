//! Token-count-aware buffer memory for conversation history.
//!
//! [`ConversationTokenBufferMemory`] works like a buffer memory but trims
//! messages from the front when the total estimated token count exceeds a
//! configurable limit. A pluggable [`TokenCounter`] trait allows callers to
//! supply their own tokenisation strategy.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::RwLock;

use cognis_core::error::Result;
use cognis_core::messages::{get_buffer_string, Message};

use super::BaseMemory;

/// Trait for estimating the number of tokens in a string.
pub trait TokenCounter: Send + Sync {
    /// Return the estimated token count for the given text.
    fn count_tokens(&self, text: &str) -> usize;
}

/// Simple token counter that approximates tokens from whitespace word count.
///
/// Uses the heuristic `words * 4 / 3` which is a rough approximation for
/// English text with most common tokenisers.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleTokenCounter;

impl SimpleTokenCounter {
    /// Create a new simple token counter.
    pub fn new() -> Self {
        Self
    }
}

impl TokenCounter for SimpleTokenCounter {
    fn count_tokens(&self, text: &str) -> usize {
        text.split_whitespace().count() * 4 / 3
    }
}

/// Character-based token counter that estimates tokens as `chars / ratio`.
#[derive(Debug, Clone, Copy)]
pub struct CharBasedTokenCounter {
    /// Average characters per token (typically ~4 for English).
    chars_per_token: f64,
}

impl CharBasedTokenCounter {
    /// Create a counter with the given characters-per-token ratio.
    pub fn new(chars_per_token: f64) -> Self {
        Self { chars_per_token }
    }
}

impl TokenCounter for CharBasedTokenCounter {
    fn count_tokens(&self, text: &str) -> usize {
        (text.chars().count() as f64 / self.chars_per_token).ceil() as usize
    }
}

/// Conversation memory that trims messages by token count.
///
/// When the estimated total token count of all stored messages exceeds
/// `max_token_limit`, the oldest messages are dropped until the total
/// fits within budget.
///
/// # Example
///
/// ```rust
/// use cognis::memory::token_buffer::{TokenBufferMemory, SimpleTokenCounter};
///
/// let mem = TokenBufferMemory::builder()
///     .max_tokens(500)
///     .counter(SimpleTokenCounter::new())
///     .build();
/// ```
pub struct TokenBufferMemory {
    messages: Arc<RwLock<Vec<Message>>>,
    max_token_limit: usize,
    token_counter: Box<dyn TokenCounter>,
    memory_key: String,
    return_messages: bool,
}

impl TokenBufferMemory {
    /// Create a new token buffer memory with default settings.
    ///
    /// Defaults: 2000 token limit, `SimpleTokenCounter`, key = "history".
    pub fn new() -> Self {
        Self {
            messages: Arc::new(RwLock::new(Vec::new())),
            max_token_limit: 2000,
            token_counter: Box::new(SimpleTokenCounter),
            memory_key: "history".to_string(),
            return_messages: true,
        }
    }

    /// Set the maximum token limit.
    pub fn with_max_tokens(mut self, limit: usize) -> Self {
        self.max_token_limit = limit;
        self
    }

    /// Set a custom token counter.
    pub fn with_counter(mut self, counter: impl TokenCounter + 'static) -> Self {
        self.token_counter = Box::new(counter);
        self
    }

    /// Set the memory key used in chain context.
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// Set whether to return messages as structured JSON or formatted text.
    pub fn with_return_messages(mut self, return_messages: bool) -> Self {
        self.return_messages = return_messages;
        self
    }

    /// Return a builder for this memory type.
    pub fn builder() -> TokenBufferMemoryBuilder {
        TokenBufferMemoryBuilder::default()
    }

    /// Add a message and trim if over budget.
    pub async fn add_message(&self, msg: Message) {
        let mut messages = self.messages.write().await;
        messages.push(msg);
        self.trim_messages(&mut messages);
    }

    /// Get a reference to stored messages.
    pub async fn get_messages(&self) -> Vec<Message> {
        self.messages.read().await.clone()
    }

    /// Compute the total estimated token count of all stored messages.
    pub async fn total_tokens(&self) -> usize {
        let messages = self.messages.read().await;
        self.count_messages_tokens(&messages)
    }

    /// Clear all stored messages.
    pub async fn clear_messages(&self) {
        let mut messages = self.messages.write().await;
        messages.clear();
    }

    /// Count tokens for a slice of messages.
    fn count_messages_tokens(&self, messages: &[Message]) -> usize {
        messages
            .iter()
            .map(|m| {
                let text = m.content().text();
                // Add a small overhead per message for role/framing tokens
                self.token_counter.count_tokens(&text) + 3
            })
            .sum()
    }

    /// Trim oldest messages until total tokens fit within the limit.
    fn trim_messages(&self, messages: &mut Vec<Message>) {
        while !messages.is_empty() && self.count_messages_tokens(messages) > self.max_token_limit {
            messages.remove(0);
        }
    }
}

impl Default for TokenBufferMemory {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for [`TokenBufferMemory`].
#[derive(Default)]
pub struct TokenBufferMemoryBuilder {
    max_tokens: Option<usize>,
    counter: Option<Box<dyn TokenCounter>>,
    memory_key: Option<String>,
    return_messages: Option<bool>,
}

impl TokenBufferMemoryBuilder {
    /// Set the maximum token limit.
    pub fn max_tokens(mut self, limit: usize) -> Self {
        self.max_tokens = Some(limit);
        self
    }

    /// Set a custom token counter.
    pub fn counter(mut self, counter: impl TokenCounter + 'static) -> Self {
        self.counter = Some(Box::new(counter));
        self
    }

    /// Set the memory key.
    pub fn memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = Some(key.into());
        self
    }

    /// Set whether to return messages as JSON or text.
    pub fn return_messages(mut self, val: bool) -> Self {
        self.return_messages = Some(val);
        self
    }

    /// Build the [`TokenBufferMemory`].
    pub fn build(self) -> TokenBufferMemory {
        TokenBufferMemory {
            messages: Arc::new(RwLock::new(Vec::new())),
            max_token_limit: self.max_tokens.unwrap_or(2000),
            token_counter: self.counter.unwrap_or_else(|| Box::new(SimpleTokenCounter)),
            memory_key: self.memory_key.unwrap_or_else(|| "history".to_string()),
            return_messages: self.return_messages.unwrap_or(true),
        }
    }
}

#[async_trait]
impl BaseMemory for TokenBufferMemory {
    async fn load_memory_variables(&self) -> Result<HashMap<String, Value>> {
        let messages = self.messages.read().await;
        let mut vars = HashMap::new();

        if self.return_messages {
            let serialized: Vec<Value> = messages
                .iter()
                .map(|m| serde_json::to_value(m).unwrap_or(Value::Null))
                .collect();
            vars.insert(self.memory_key.clone(), Value::Array(serialized));
        } else {
            let buffer = get_buffer_string(&messages, "Human", "AI");
            vars.insert(self.memory_key.clone(), Value::String(buffer));
        }

        Ok(vars)
    }

    async fn save_context(&self, input: &Message, output: &Message) -> Result<()> {
        let mut messages = self.messages.write().await;
        messages.push(input.clone());
        messages.push(output.clone());
        self.trim_messages(&mut messages);
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut messages = self.messages.write().await;
        messages.clear();
        Ok(())
    }

    fn memory_key(&self) -> &str {
        &self.memory_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cognis_core::messages::Message;

    // ─── TokenCounter tests ───

    #[test]
    fn test_simple_counter_empty() {
        let counter = SimpleTokenCounter::new();
        assert_eq!(counter.count_tokens(""), 0);
    }

    #[test]
    fn test_simple_counter_single_word() {
        let counter = SimpleTokenCounter::new();
        // 1 word * 4 / 3 = 1 (integer division)
        assert_eq!(counter.count_tokens("hello"), 1);
    }

    #[test]
    fn test_simple_counter_multiple_words() {
        let counter = SimpleTokenCounter::new();
        // 4 words * 4 / 3 = 5
        let tokens = counter.count_tokens("hello world foo bar");
        assert_eq!(tokens, 5);
    }

    #[test]
    fn test_char_based_counter() {
        let counter = CharBasedTokenCounter::new(4.0);
        // 5 chars / 4.0 = 1.25, ceil = 2
        assert_eq!(counter.count_tokens("hello"), 2);
    }

    #[test]
    fn test_char_based_counter_longer() {
        let counter = CharBasedTokenCounter::new(4.0);
        // 20 chars / 4.0 = 5
        assert_eq!(counter.count_tokens("12345678901234567890"), 5);
    }

    // ─── TokenBufferMemory basic tests ───

    #[tokio::test]
    async fn test_new_memory_empty() {
        let mem = TokenBufferMemory::new();
        let msgs = mem.get_messages().await;
        assert!(msgs.is_empty());
        assert_eq!(mem.total_tokens().await, 0);
    }

    #[tokio::test]
    async fn test_add_and_get_messages() {
        let mem = TokenBufferMemory::new().with_max_tokens(10000);
        mem.add_message(Message::human("Hello")).await;
        mem.add_message(Message::ai("Hi there")).await;
        let msgs = mem.get_messages().await;
        assert_eq!(msgs.len(), 2);
    }

    #[tokio::test]
    async fn test_save_context_adds_two() {
        let mem = TokenBufferMemory::new().with_max_tokens(10000);
        mem.save_context(&Message::human("Hey"), &Message::ai("Hello"))
            .await
            .unwrap();
        let msgs = mem.get_messages().await;
        assert_eq!(msgs.len(), 2);
    }

    #[tokio::test]
    async fn test_total_tokens_nonzero() {
        let mem = TokenBufferMemory::new().with_max_tokens(10000);
        mem.add_message(Message::human("Hello world")).await;
        assert!(mem.total_tokens().await > 0);
    }

    #[tokio::test]
    async fn test_trimming_by_token_count() {
        // Very small limit to force trimming
        let mem = TokenBufferMemory::new().with_max_tokens(10);

        mem.save_context(
            &Message::human("This is a fairly long message that should use many tokens"),
            &Message::ai("This is also a long response with many tokens in it"),
        )
        .await
        .unwrap();

        // After trimming, total tokens should be within budget
        assert!(mem.total_tokens().await <= 10);
    }

    #[tokio::test]
    async fn test_clear() {
        let mem = TokenBufferMemory::new();
        mem.save_context(&Message::human("Hi"), &Message::ai("Hello"))
            .await
            .unwrap();

        mem.clear().await.unwrap();

        let msgs = mem.get_messages().await;
        assert!(msgs.is_empty());
        assert_eq!(mem.total_tokens().await, 0);
    }

    #[tokio::test]
    async fn test_clear_messages_method() {
        let mem = TokenBufferMemory::new();
        mem.add_message(Message::human("Test")).await;
        mem.clear_messages().await;
        assert!(mem.get_messages().await.is_empty());
    }

    // ─── Builder tests ───

    #[tokio::test]
    async fn test_builder_default() {
        let mem = TokenBufferMemory::builder().build();
        assert_eq!(mem.max_token_limit, 2000);
        assert_eq!(mem.memory_key, "history");
        assert!(mem.return_messages);
    }

    #[tokio::test]
    async fn test_builder_custom() {
        let mem = TokenBufferMemory::builder()
            .max_tokens(500)
            .memory_key("chat")
            .return_messages(false)
            .counter(SimpleTokenCounter::new())
            .build();

        assert_eq!(mem.max_token_limit, 500);
        assert_eq!(mem.memory_key, "chat");
        assert!(!mem.return_messages);
    }

    #[tokio::test]
    async fn test_with_custom_counter() {
        let mem = TokenBufferMemory::new()
            .with_max_tokens(100)
            .with_counter(CharBasedTokenCounter::new(4.0));

        mem.add_message(Message::human("Hello")).await;
        assert!(mem.total_tokens().await > 0);
    }

    // ─── BaseMemory trait tests ───

    #[tokio::test]
    async fn test_load_as_json() {
        let mem = TokenBufferMemory::new().with_max_tokens(10000);
        mem.save_context(&Message::human("Hello"), &Message::ai("Hi"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_array().unwrap();
        assert_eq!(history.len(), 2);
    }

    #[tokio::test]
    async fn test_load_as_string() {
        let mem = TokenBufferMemory::new()
            .with_max_tokens(10000)
            .with_return_messages(false);

        mem.save_context(&Message::human("Hello"), &Message::ai("World"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.contains("Hello"));
        assert!(history.contains("World"));
    }

    #[tokio::test]
    async fn test_custom_memory_key() {
        let mem = TokenBufferMemory::new()
            .with_max_tokens(10000)
            .with_memory_key("chat_log");

        mem.save_context(&Message::human("Hi"), &Message::ai("Hey"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        assert!(vars.contains_key("chat_log"));
        assert!(!vars.contains_key("history"));
    }

    #[tokio::test]
    async fn test_memory_key_method() {
        let mem = TokenBufferMemory::new().with_memory_key("custom");
        assert_eq!(mem.memory_key(), "custom");
    }
}
