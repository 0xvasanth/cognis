//! Buffer memory that stores the complete conversation history.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use cognis_core::error::Result;
use cognis_core::messages::{get_buffer_string, Message};

use super::BaseMemory;

/// Stores all conversation messages in a Vec.
///
/// This is the simplest memory implementation — it keeps every message
/// that has been exchanged. When `return_messages` is true, the memory
/// variable contains a JSON array of serialized messages; when false,
/// it contains a formatted string with role prefixes.
pub struct ConversationBufferMemory {
    messages: Arc<Mutex<Vec<Message>>>,
    memory_key: String,
    /// If true, return messages as a JSON array; if false, return a formatted string.
    return_messages: bool,
}

impl ConversationBufferMemory {
    /// Create a new buffer memory with default settings.
    pub fn new() -> Self {
        Self {
            messages: Arc::new(Mutex::new(Vec::new())),
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

impl Default for ConversationBufferMemory {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl BaseMemory for ConversationBufferMemory {
    async fn load_memory_variables(&self) -> Result<HashMap<String, Value>> {
        let messages = self.messages.lock().await;
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
        let mut messages = self.messages.lock().await;
        messages.push(input.clone());
        messages.push(output.clone());
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let mut messages = self.messages.lock().await;
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

    #[tokio::test]
    async fn test_buffer_save_and_load() {
        let mem = ConversationBufferMemory::new();

        let human = Message::human("Hello");
        let ai = Message::ai("Hi there!");
        mem.save_context(&human, &ai).await.unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_array().unwrap();
        assert_eq!(history.len(), 2);
    }

    #[tokio::test]
    async fn test_buffer_multiple_turns() {
        let mem = ConversationBufferMemory::new();

        mem.save_context(&Message::human("Hi"), &Message::ai("Hello"))
            .await
            .unwrap();
        mem.save_context(&Message::human("How are you?"), &Message::ai("I'm fine"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_array().unwrap();
        assert_eq!(history.len(), 4);
    }

    #[tokio::test]
    async fn test_buffer_clear() {
        let mem = ConversationBufferMemory::new();

        mem.save_context(&Message::human("Hi"), &Message::ai("Hello"))
            .await
            .unwrap();
        mem.clear().await.unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_array().unwrap();
        assert!(history.is_empty());
    }

    #[tokio::test]
    async fn test_buffer_as_string() {
        let mem = ConversationBufferMemory::new().with_return_messages(false);

        mem.save_context(&Message::human("Hello"), &Message::ai("Hi there!"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_str().unwrap();
        assert!(history.contains("Human"));
        assert!(history.contains("Hello"));
        assert!(history.contains("AI"));
        assert!(history.contains("Hi there!"));
    }
}
