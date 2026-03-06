//! Memory middleware — manages conversation memory and injects context.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;
use tokio::sync::Mutex;

use crate::middleware::{AgentState, Middleware, Result};

/// Middleware that manages a key-value memory store and injects
/// relevant memories into the conversation before each model call.
pub struct MemoryMiddleware {
    /// In-memory store of key-value pairs.
    store: Arc<Mutex<HashMap<String, String>>>,
    /// Maximum number of memory entries to inject.
    max_entries: usize,
}

impl MemoryMiddleware {
    /// Create a new `MemoryMiddleware` with the given maximum number of entries to inject.
    pub fn new(max_entries: usize) -> Self {
        Self {
            store: Arc::new(Mutex::new(HashMap::new())),
            max_entries,
        }
    }

    /// Store a memory entry.
    pub async fn remember(&self, key: impl Into<String>, value: impl Into<String>) {
        let mut store = self.store.lock().await;
        store.insert(key.into(), value.into());
    }

    /// Retrieve a memory entry.
    pub async fn recall(&self, key: &str) -> Option<String> {
        let store = self.store.lock().await;
        store.get(key).cloned()
    }

    /// List all memory keys.
    pub async fn keys(&self) -> Vec<String> {
        let store = self.store.lock().await;
        store.keys().cloned().collect()
    }

    /// Clear all memories.
    pub async fn clear(&self) {
        let mut store = self.store.lock().await;
        store.clear();
    }
}

#[async_trait]
impl Middleware for MemoryMiddleware {
    fn name(&self) -> &str {
        "memory"
    }

    /// Before the model is called, inject a summary of stored memories
    /// as a system message if any memories exist.
    async fn before_model(&self, state: &mut AgentState) -> Result<()> {
        let store = self.store.lock().await;
        if store.is_empty() {
            return Ok(());
        }

        // Build a memory context string from the most recent entries.
        let entries: Vec<String> = store
            .iter()
            .take(self.max_entries)
            .map(|(k, v)| format!("- {k}: {v}"))
            .collect();

        let memory_context = format!("## Remembered context\n{}", entries.join("\n"));

        // Prepend a system message with the memory context if messages exist.
        if let Some(messages) = state.get_mut("messages").and_then(|v| v.as_array_mut()) {
            let memory_msg = json!({
                "type": "system",
                "content": memory_context
            });
            // Insert memory as second message (after any existing system prompt).
            let insert_pos = if messages
                .first()
                .and_then(|m| m.get("type"))
                .and_then(|t| t.as_str())
                == Some("system")
            {
                1
            } else {
                0
            };
            messages.insert(insert_pos, memory_msg);
        }

        Ok(())
    }
}
