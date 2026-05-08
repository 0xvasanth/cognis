//! Window memory that keeps only the last K conversation turns.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use cognis_core::error::Result;
use cognis_core::messages::{get_buffer_string, Message};

use super::BaseMemory;

/// Keeps only the last K conversation turns (pairs of human + AI messages).
///
/// Each turn consists of two messages (input + output). When the number of
/// stored turns exceeds `k`, the oldest turns are discarded.
pub struct ConversationWindowMemory {
    messages: Arc<Mutex<Vec<Message>>>,
    /// Number of turns (pairs) to keep.
    k: usize,
    memory_key: String,
    /// If true, return messages as a JSON array; if false, return a formatted string.
    return_messages: bool,
}

impl ConversationWindowMemory {
    /// Create a new window memory that keeps the last `k` turns.
    pub fn new(k: usize) -> Self {
        Self {
            messages: Arc::new(Mutex::new(Vec::new())),
            k,
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

    /// Trim messages to keep only the last `k` turns (2*k messages).
    fn trim(messages: &mut Vec<Message>, k: usize) {
        let max_messages = k * 2;
        if messages.len() > max_messages {
            let drain_count = messages.len() - max_messages;
            messages.drain(..drain_count);
        }
    }
}

#[async_trait]
impl BaseMemory for ConversationWindowMemory {
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
        Self::trim(&mut messages, self.k);
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
    async fn test_window_keeps_k_turns() {
        let mem = ConversationWindowMemory::new(2); // keep last 2 turns

        // Add 3 turns
        mem.save_context(&Message::human("Turn 1"), &Message::ai("Response 1"))
            .await
            .unwrap();
        mem.save_context(&Message::human("Turn 2"), &Message::ai("Response 2"))
            .await
            .unwrap();
        mem.save_context(&Message::human("Turn 3"), &Message::ai("Response 3"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_array().unwrap();

        // Should only have 4 messages (2 turns * 2 messages each)
        assert_eq!(history.len(), 4);

        // The oldest turn (Turn 1) should be gone
        let messages = mem.messages.lock().await;
        assert_eq!(messages[0].content().text(), "Turn 2");
        assert_eq!(messages[1].content().text(), "Response 2");
        assert_eq!(messages[2].content().text(), "Turn 3");
        assert_eq!(messages[3].content().text(), "Response 3");
    }

    #[tokio::test]
    async fn test_window_under_limit() {
        let mem = ConversationWindowMemory::new(5); // keep last 5 turns

        // Add only 2 turns
        mem.save_context(&Message::human("Hi"), &Message::ai("Hello"))
            .await
            .unwrap();
        mem.save_context(&Message::human("How?"), &Message::ai("Fine"))
            .await
            .unwrap();

        let vars = mem.load_memory_variables().await.unwrap();
        let history = vars.get("history").unwrap().as_array().unwrap();

        // All 4 messages should be present
        assert_eq!(history.len(), 4);
    }
}
