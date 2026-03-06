//! Summary memory that condenses older messages using an LLM.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use rustchain_core::error::Result;
use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{get_buffer_string, Message};

use super::BaseMemory;

/// Summarizes older messages using an LLM, keeping recent ones verbatim.
///
/// When the message buffer exceeds `max_messages`, the oldest messages are
/// condensed into a running summary via an LLM call. The summary is stored
/// separately and prepended to the memory output.
pub struct ConversationSummaryMemory {
    messages: Arc<Mutex<Vec<Message>>>,
    summary: Arc<Mutex<String>>,
    model: Arc<dyn BaseChatModel>,
    /// When the buffer exceeds this count, summarize the oldest messages.
    max_messages: usize,
    memory_key: String,
}

impl ConversationSummaryMemory {
    /// Create a new summary memory.
    ///
    /// - `model` — the chat model used to generate summaries.
    /// - `max_messages` — when the buffer exceeds this many messages, the
    ///   oldest messages are summarized and replaced with a summary.
    pub fn new(model: Arc<dyn BaseChatModel>, max_messages: usize) -> Self {
        Self {
            messages: Arc::new(Mutex::new(Vec::new())),
            summary: Arc::new(Mutex::new(String::new())),
            model,
            max_messages,
            memory_key: "history".to_string(),
        }
    }

    /// Set the memory key used in chain context.
    pub fn with_memory_key(mut self, key: impl Into<String>) -> Self {
        self.memory_key = key.into();
        self
    }

    /// Summarize the given messages and merge with the existing summary.
    async fn summarize_messages(
        &self,
        messages: &[Message],
        existing_summary: &str,
    ) -> Result<String> {
        let buffer = get_buffer_string(messages, "Human", "AI");
        let prompt = if existing_summary.is_empty() {
            format!(
                "Summarize the following conversation so far:\n{}\n\nSummary:",
                buffer
            )
        } else {
            format!(
                "Current summary:\n{}\n\nNew conversation:\n{}\n\nUpdated summary:",
                existing_summary, buffer
            )
        };

        let prompt_msg = Message::human(prompt);
        let response = self.model.invoke_messages(&[prompt_msg], None).await?;
        Ok(response.base.content.text())
    }
}

#[async_trait]
impl BaseMemory for ConversationSummaryMemory {
    async fn load_memory_variables(&self) -> Result<HashMap<String, Value>> {
        let messages = self.messages.lock().await;
        let summary = self.summary.lock().await;

        let mut vars = HashMap::new();

        // Build output: summary (if any) + current messages
        let mut parts = Vec::new();
        if !summary.is_empty() {
            parts.push(format!("Summary of earlier conversation:\n{}", *summary));
        }
        if !messages.is_empty() {
            let buffer = get_buffer_string(&messages, "Human", "AI");
            parts.push(buffer);
        }

        vars.insert(self.memory_key.clone(), Value::String(parts.join("\n\n")));
        Ok(vars)
    }

    async fn save_context(&self, input: &Message, output: &Message) -> Result<()> {
        {
            let mut messages = self.messages.lock().await;
            messages.push(input.clone());
            messages.push(output.clone());
        }

        // Check if we need to summarize
        let needs_summarization = {
            let messages = self.messages.lock().await;
            messages.len() > self.max_messages
        };

        if needs_summarization {
            let (msgs_to_summarize, remaining) = {
                let messages = self.messages.lock().await;
                // Summarize all but the last 2 messages (the most recent turn)
                let split_at = messages.len().saturating_sub(2);
                let to_summarize = messages[..split_at].to_vec();
                let remaining = messages[split_at..].to_vec();
                (to_summarize, remaining)
            };

            let existing_summary = {
                let summary = self.summary.lock().await;
                summary.clone()
            };

            let new_summary = self
                .summarize_messages(&msgs_to_summarize, &existing_summary)
                .await?;

            {
                let mut summary = self.summary.lock().await;
                *summary = new_summary;
            }
            {
                let mut messages = self.messages.lock().await;
                *messages = remaining;
            }
        }

        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut messages = self.messages.lock().await;
        let mut summary = self.summary.lock().await;
        messages.clear();
        summary.clear();
        Ok(())
    }

    fn memory_key(&self) -> &str {
        &self.memory_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustchain_core::language_models::fake::FakeListChatModel;
    use rustchain_core::messages::Message;

    #[tokio::test]
    async fn test_summary_under_limit() {
        let model = Arc::new(FakeListChatModel::new(vec![
            "This should not be called".to_string()
        ]));
        // max_messages = 10, so 2 messages won't trigger summarization
        let mem = ConversationSummaryMemory::new(model, 10);

        mem.save_context(&Message::human("Hello"), &Message::ai("Hi"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.contains("Hello"));
        assert!(history.contains("Hi"));
    }

    #[tokio::test]
    async fn test_summary_triggers() {
        let model = Arc::new(FakeListChatModel::new(vec![
            "User greeted and AI responded.".to_string(),
        ]));
        // max_messages = 2, so after adding 2 messages it will trigger summarization
        let mem = ConversationSummaryMemory::new(model, 2);

        mem.save_context(&Message::human("Hello"), &Message::ai("Hi there"))
            .await
            .unwrap();

        // After save, we had 2 messages (== max, not >), no summarization yet
        {
            let msgs = mem.messages.lock().await;
            assert_eq!(msgs.len(), 2);
        }

        // Add another turn, now we have 4 > 2, triggers summarization
        mem.save_context(&Message::human("How are you?"), &Message::ai("Fine"))
            .await
            .unwrap();

        // After summarization, only the latest turn should remain in messages
        {
            let msgs = mem.messages.lock().await;
            assert_eq!(msgs.len(), 2);
            assert_eq!(msgs[0].content().text(), "How are you?");
            assert_eq!(msgs[1].content().text(), "Fine");
        }

        // Summary should be populated
        {
            let summary = mem.summary.lock().await;
            assert_eq!(*summary, "User greeted and AI responded.");
        }

        // Load should include both summary and recent messages
        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.contains("User greeted and AI responded."));
        assert!(history.contains("How are you?"));
    }
}
