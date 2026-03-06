//! Conversation memory systems for managing chat history in chains.
//!
//! This module provides several memory implementations:
//!
//! - [`ConversationBufferMemory`] — stores all messages in a buffer
//! - [`ConversationWindowMemory`] — keeps only the last K conversation turns
//! - [`ConversationSummaryMemory`] — summarizes older messages using an LLM

pub mod buffer;
pub mod summary;
pub mod vector;
pub mod window;

pub use buffer::ConversationBufferMemory;
pub use summary::ConversationSummaryMemory;
pub use vector::VectorStoreMemory;
pub use window::ConversationWindowMemory;

use std::collections::HashMap;

use async_trait::async_trait;
use rustchain_core::error::Result;
use rustchain_core::messages::Message;
use serde_json::Value;

/// Base trait for conversation memory implementations.
///
/// A memory stores conversation turns (human input + AI output) and provides
/// them as variables that can be injected into chain prompts.
#[async_trait]
pub trait BaseMemory: Send + Sync {
    /// Get memory variables (messages) to add to the chain context.
    async fn load_memory_variables(&self) -> Result<HashMap<String, Value>>;

    /// Save context from a conversation turn.
    async fn save_context(&self, input: &Message, output: &Message) -> Result<()>;

    /// Clear memory.
    async fn clear(&self) -> Result<()>;

    /// The key used in chain context (default: "history").
    fn memory_key(&self) -> &str {
        "history"
    }
}
