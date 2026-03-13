use async_trait::async_trait;

use crate::error::Result;
use crate::messages::Message;

/// Abstract base class for storing chat message history.
#[async_trait]
pub trait BaseChatMessageHistory: Send + Sync {
    /// Get all messages.
    async fn messages(&self) -> Result<Vec<Message>>;

    /// Add messages to the history.
    async fn add_messages(&self, messages: Vec<Message>) -> Result<()>;

    /// Clear all messages.
    async fn clear(&self) -> Result<()>;
}

/// In-memory chat message history.
#[derive(Debug, Default)]
pub struct InMemoryChatMessageHistory {
    messages: tokio::sync::RwLock<Vec<Message>>,
}

impl InMemoryChatMessageHistory {
    pub fn new() -> Self {
        Self {
            messages: tokio::sync::RwLock::new(Vec::new()),
        }
    }
}

#[async_trait]
impl BaseChatMessageHistory for InMemoryChatMessageHistory {
    async fn messages(&self) -> Result<Vec<Message>> {
        Ok(self.messages.read().await.clone())
    }

    async fn add_messages(&self, messages: Vec<Message>) -> Result<()> {
        self.messages.write().await.extend(messages);
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        self.messages.write().await.clear();
        Ok(())
    }
}
