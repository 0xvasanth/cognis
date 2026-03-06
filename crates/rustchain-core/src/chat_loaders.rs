//! Abstract interface for chat loaders.
//!
//! Mirrors Python `langchain_core.chat_loaders`.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::chat_sessions::ChatSession;
use crate::error::Result;

/// Stream of chat sessions yielded lazily.
pub type ChatSessionStream = Pin<Box<dyn Stream<Item = Result<ChatSession>> + Send>>;

/// Abstract interface for chat loaders.
///
/// A chat loader reads conversation data from an external source
/// (file, API, database, etc.) and yields `ChatSession` objects.
///
/// Implementations should override `lazy_load` using streams to avoid
/// loading all sessions into memory at once.
#[async_trait]
pub trait BaseChatLoader: Send + Sync {
    /// Lazily load chat sessions as a stream.
    async fn lazy_load(&self) -> Result<ChatSessionStream>;

    /// Eagerly load all chat sessions into memory.
    ///
    /// Default implementation collects from `lazy_load`.
    async fn load(&self) -> Result<Vec<ChatSession>> {
        use futures::StreamExt;
        let mut stream = self.lazy_load().await?;
        let mut sessions = Vec::new();
        while let Some(session_result) = stream.next().await {
            sessions.push(session_result?);
        }
        Ok(sessions)
    }
}
