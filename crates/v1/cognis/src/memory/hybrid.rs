//! Hybrid memory combining window and summary strategies.
//!
//! This module provides two memory implementations:
//!
//! - [`HybridMemory`] — keeps the last N messages verbatim and summarizes older
//!   messages using an LLM, combining the benefits of window and summary memory.
//! - [`ConversationTokenBufferMemory`] — like window memory but uses an
//!   approximate token count instead of message count to decide when to trim.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::RwLock;

use cognis_core::error::Result;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{count_tokens_approximately, get_buffer_string, Message};

use super::BaseMemory;

/// Default summarization prompt template.
///
/// The placeholder `{existing}` is replaced with the current running summary
/// (or "N/A" if there is none) and `{new_lines}` with the formatted messages
/// to incorporate.
const DEFAULT_SUMMARY_PROMPT: &str = "\
Progressively summarize the lines of conversation provided, adding onto the \
previous summary returning a new summary.

Current summary:
{existing}

New lines of conversation:
{new_lines}

New summary:";

/// Hybrid memory that keeps recent messages verbatim and summarizes older ones.
///
/// When the total number of stored messages exceeds `window_size`, the oldest
/// messages that fall outside the window are summarized via an LLM call and
/// merged into a running summary. The `load_memory_variables` method returns
/// the running summary followed by the recent messages.
pub struct HybridMemory {
    inner: Arc<RwLock<HybridMemoryInner>>,
    model: Arc<dyn BaseChatModel>,
    window_size: usize,
    summary_prompt: String,
    memory_key: String,
}

struct HybridMemoryInner {
    messages: Vec<Message>,
    running_summary: String,
}

/// Builder for [`HybridMemory`].
pub struct HybridMemoryBuilder {
    model: Arc<dyn BaseChatModel>,
    window_size: usize,
    summary_prompt: String,
    memory_key: String,
}

impl HybridMemoryBuilder {
    /// Create a new builder with the required chat model.
    pub fn new(model: Arc<dyn BaseChatModel>) -> Self {
        Self {
            model,
            window_size: 10,
            summary_prompt: DEFAULT_SUMMARY_PROMPT.to_string(),
            memory_key: "history".to_string(),
        }
    }

    /// Set the number of recent messages to keep verbatim (default: 10).
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set a custom summarization prompt.
    ///
    /// The prompt should contain `{existing}` and `{new_lines}` placeholders.
    pub fn summary_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.summary_prompt = prompt.into();
        self
    }

    /// Set the memory key used in chain context (default: "history").
    pub fn memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// Build the [`HybridMemory`].
    pub fn build(self) -> HybridMemory {
        HybridMemory {
            inner: Arc::new(RwLock::new(HybridMemoryInner {
                messages: Vec::new(),
                running_summary: String::new(),
            })),
            model: self.model,
            window_size: self.window_size,
            summary_prompt: self.summary_prompt,
            memory_key: self.memory_key,
        }
    }
}

impl HybridMemory {
    /// Create a new hybrid memory with the given chat model and default settings.
    pub fn new(model: Arc<dyn BaseChatModel>) -> Self {
        HybridMemoryBuilder::new(model).build()
    }

    /// Return a builder for fine-grained configuration.
    pub fn builder(model: Arc<dyn BaseChatModel>) -> HybridMemoryBuilder {
        HybridMemoryBuilder::new(model)
    }

    /// Summarize the given messages, merging with the existing summary.
    async fn summarize_messages(
        &self,
        messages: &[Message],
        existing_summary: &str,
    ) -> Result<String> {
        let buffer = get_buffer_string(messages, "Human", "AI");
        let existing = if existing_summary.is_empty() {
            "N/A".to_string()
        } else {
            existing_summary.to_string()
        };
        let prompt = self
            .summary_prompt
            .replace("{existing}", &existing)
            .replace("{new_lines}", &buffer);

        let prompt_msg = Message::human(prompt);
        let response = self.model.invoke_messages(&[prompt_msg], None).await?;
        Ok(response.base.content.text())
    }
}

#[async_trait]
impl BaseMemory for HybridMemory {
    async fn load_memory_variables(&self) -> Result<HashMap<String, Value>> {
        let inner = self.inner.read().await;
        let mut parts = Vec::new();

        if !inner.running_summary.is_empty() {
            parts.push(format!(
                "Summary of earlier conversation:\n{}",
                inner.running_summary
            ));
        }

        if !inner.messages.is_empty() {
            let buffer = get_buffer_string(&inner.messages, "Human", "AI");
            parts.push(buffer);
        }

        let mut vars = HashMap::new();
        vars.insert(self.memory_key.clone(), Value::String(parts.join("\n\n")));
        Ok(vars)
    }

    async fn save_context(&self, input: &Message, output: &Message) -> Result<()> {
        // Add the new messages.
        {
            let mut inner = self.inner.write().await;
            inner.messages.push(input.clone());
            inner.messages.push(output.clone());
        }

        // Check if we need to summarize.
        let needs_summarization = {
            let inner = self.inner.read().await;
            inner.messages.len() > self.window_size
        };

        if needs_summarization {
            let (msgs_to_summarize, remaining, existing_summary) = {
                let inner = self.inner.read().await;
                // Keep the last `window_size` messages; summarize the rest.
                let split_at = inner.messages.len().saturating_sub(self.window_size);
                let to_summarize = inner.messages[..split_at].to_vec();
                let remaining = inner.messages[split_at..].to_vec();
                (to_summarize, remaining, inner.running_summary.clone())
            };

            let new_summary = self
                .summarize_messages(&msgs_to_summarize, &existing_summary)
                .await?;

            {
                let mut inner = self.inner.write().await;
                inner.running_summary = new_summary;
                inner.messages = remaining;
            }
        }

        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut inner = self.inner.write().await;
        inner.messages.clear();
        inner.running_summary.clear();
        Ok(())
    }

    fn memory_key(&self) -> &str {
        &self.memory_key
    }
}

// ─── ConversationTokenBufferMemory ───

/// Window-style memory that trims by approximate token count instead of message count.
///
/// When the estimated token count of all stored messages exceeds `max_tokens`,
/// the oldest messages are dropped until the total fits within budget.
pub struct ConversationTokenBufferMemory {
    messages: Arc<RwLock<Vec<Message>>>,
    /// Maximum token budget for history.
    max_tokens: usize,
    memory_key: String,
    /// If true, return messages as a JSON array; if false, return a formatted string.
    return_messages: bool,
}

impl ConversationTokenBufferMemory {
    /// Create a new token buffer memory with the given token budget.
    pub fn new(max_tokens: usize) -> Self {
        Self {
            messages: Arc::new(RwLock::new(Vec::new())),
            max_tokens,
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

    /// Estimate the token count of the given messages.
    ///
    /// Uses `count_tokens_approximately` with typical defaults:
    /// ~4 chars per token and 3 extra tokens per message for role overhead.
    fn estimate_tokens(messages: &[Message]) -> usize {
        count_tokens_approximately(messages, 4.0, 3.0)
    }

    /// Trim messages from the front until the total token count is within budget.
    fn trim(messages: &mut Vec<Message>, max_tokens: usize) {
        while !messages.is_empty() && Self::estimate_tokens(messages) > max_tokens {
            messages.remove(0);
        }
    }
}

#[async_trait]
impl BaseMemory for ConversationTokenBufferMemory {
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
        Self::trim(&mut messages, self.max_tokens);
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
    use cognis_core::language_models::fake::FakeListChatModel;
    use cognis_core::messages::Message;

    // ─── HybridMemory tests ───

    #[tokio::test]
    async fn test_hybrid_within_window_no_summary() {
        let model = Arc::new(FakeListChatModel::new(vec![
            "should not be called".to_string()
        ]));
        // window_size = 10, so 2 messages will not trigger summarization
        let mem = HybridMemory::builder(model).window_size(10).build();

        mem.save_context(&Message::human("Hello"), &Message::ai("Hi"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.contains("Hello"));
        assert!(history.contains("Hi"));
        // No summary prefix should appear
        assert!(!history.contains("Summary of earlier conversation"));
    }

    #[tokio::test]
    async fn test_hybrid_triggers_summarization() {
        let model = Arc::new(FakeListChatModel::new(vec![
            "User greeted and AI responded.".to_string(),
        ]));
        // window_size = 2, so after 4 messages (2 turns) the first 2 overflow
        let mem = HybridMemory::builder(model).window_size(2).build();

        mem.save_context(&Message::human("Hello"), &Message::ai("Hi there"))
            .await
            .unwrap();
        // 2 messages, at window_size, no summarization yet
        {
            let inner = mem.inner.read().await;
            assert_eq!(inner.messages.len(), 2);
            assert!(inner.running_summary.is_empty());
        }

        // Second turn pushes to 4 messages > 2, triggers summarization
        mem.save_context(&Message::human("How are you?"), &Message::ai("Fine"))
            .await
            .unwrap();

        {
            let inner = mem.inner.read().await;
            // After summarization, only the last 2 messages (window_size) remain
            assert_eq!(inner.messages.len(), 2);
            assert_eq!(inner.messages[0].content().text(), "How are you?");
            assert_eq!(inner.messages[1].content().text(), "Fine");
            assert_eq!(inner.running_summary, "User greeted and AI responded.");
        }
    }

    #[tokio::test]
    async fn test_hybrid_running_summary_accumulates() {
        let model = Arc::new(FakeListChatModel::new(vec![
            "Summary after turn 1".to_string(),
            "Summary after turn 1 and 2".to_string(),
        ]));
        let mem = HybridMemory::builder(model).window_size(2).build();

        // Turn 1: within window
        mem.save_context(&Message::human("A"), &Message::ai("B"))
            .await
            .unwrap();
        // Turn 2: triggers summarization (4 > 2)
        mem.save_context(&Message::human("C"), &Message::ai("D"))
            .await
            .unwrap();

        {
            let inner = mem.inner.read().await;
            assert_eq!(inner.running_summary, "Summary after turn 1");
        }

        // Turn 3: triggers summarization again (4 > 2)
        mem.save_context(&Message::human("E"), &Message::ai("F"))
            .await
            .unwrap();

        {
            let inner = mem.inner.read().await;
            assert_eq!(inner.running_summary, "Summary after turn 1 and 2");
            assert_eq!(inner.messages.len(), 2);
            assert_eq!(inner.messages[0].content().text(), "E");
        }
    }

    #[tokio::test]
    async fn test_hybrid_clear_resets_everything() {
        let model = Arc::new(FakeListChatModel::new(vec!["Some summary".to_string()]));
        let mem = HybridMemory::builder(model).window_size(2).build();

        mem.save_context(&Message::human("A"), &Message::ai("B"))
            .await
            .unwrap();
        mem.save_context(&Message::human("C"), &Message::ai("D"))
            .await
            .unwrap();

        mem.clear().await.unwrap();

        {
            let inner = mem.inner.read().await;
            assert!(inner.messages.is_empty());
            assert!(inner.running_summary.is_empty());
        }

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_hybrid_custom_summary_prompt() {
        let model = Arc::new(FakeListChatModel::new(vec![
            "custom summary result".to_string()
        ]));
        let mem = HybridMemory::builder(model)
            .window_size(2)
            .summary_prompt("Summarize: {new_lines}\nPrevious: {existing}")
            .build();

        mem.save_context(&Message::human("X"), &Message::ai("Y"))
            .await
            .unwrap();
        mem.save_context(&Message::human("Z"), &Message::ai("W"))
            .await
            .unwrap();

        {
            let inner = mem.inner.read().await;
            assert_eq!(inner.running_summary, "custom summary result");
        }
    }

    #[tokio::test]
    async fn test_hybrid_load_returns_summary_plus_recent() {
        let model = Arc::new(FakeListChatModel::new(vec![
            "Earlier they discussed greetings.".to_string(),
        ]));
        let mem = HybridMemory::builder(model).window_size(2).build();

        mem.save_context(&Message::human("Hello"), &Message::ai("Hi"))
            .await
            .unwrap();
        mem.save_context(&Message::human("Bye"), &Message::ai("See ya"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();

        assert!(history.contains("Summary of earlier conversation"));
        assert!(history.contains("Earlier they discussed greetings."));
        assert!(history.contains("Bye"));
        assert!(history.contains("See ya"));
    }

    #[tokio::test]
    async fn test_hybrid_empty_history() {
        let model = Arc::new(FakeListChatModel::new(vec!["unused".to_string()]));
        let mem = HybridMemory::new(model);

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_hybrid_builder_pattern() {
        let model = Arc::new(FakeListChatModel::new(vec!["summary".to_string()]));
        let mem = HybridMemory::builder(model)
            .window_size(4)
            .memory_key("chat_history")
            .summary_prompt("Custom: {existing} {new_lines}")
            .build();

        assert_eq!(mem.memory_key(), "chat_history");
        assert_eq!(mem.window_size, 4);
        assert!(mem.summary_prompt.starts_with("Custom:"));
    }

    #[tokio::test]
    async fn test_hybrid_memory_key() {
        let model = Arc::new(FakeListChatModel::new(vec!["unused".to_string()]));
        let mem = HybridMemory::builder(model).memory_key("my_key").build();

        mem.save_context(&Message::human("Hi"), &Message::ai("Hey"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        assert!(vars.contains_key("my_key"));
        assert!(!vars.contains_key("history"));
    }

    // ─── ConversationTokenBufferMemory tests ───

    #[tokio::test]
    async fn test_token_buffer_within_budget() {
        // Very large budget — nothing should be dropped
        let mem = ConversationTokenBufferMemory::new(10000);

        mem.save_context(&Message::human("Hello"), &Message::ai("Hi"))
            .await
            .unwrap();
        mem.save_context(&Message::human("How are you?"), &Message::ai("Fine"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_array().unwrap();
        assert_eq!(history.len(), 4);
    }

    #[tokio::test]
    async fn test_token_buffer_drops_old_messages() {
        // Very small budget to force trimming
        let mem = ConversationTokenBufferMemory::new(15);

        mem.save_context(
            &Message::human("First message that is somewhat long"),
            &Message::ai("First response that is also somewhat long"),
        )
        .await
        .unwrap();
        mem.save_context(&Message::human("Second"), &Message::ai("Reply"))
            .await
            .unwrap();

        let messages = mem.messages.read().await;
        // The old long messages should have been trimmed to fit budget
        let total_tokens = ConversationTokenBufferMemory::estimate_tokens(&messages);
        assert!(total_tokens <= 15);
    }

    #[tokio::test]
    async fn test_token_buffer_clear() {
        let mem = ConversationTokenBufferMemory::new(10000);

        mem.save_context(&Message::human("Hi"), &Message::ai("Hello"))
            .await
            .unwrap();

        mem.clear().await.unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_array().unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_token_buffer_as_string() {
        let mem = ConversationTokenBufferMemory::new(10000).with_return_messages(false);

        mem.save_context(&Message::human("Hello"), &Message::ai("World"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.contains("Hello"));
        assert!(history.contains("World"));
    }
}
