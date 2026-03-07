//! Conversation memory systems for managing chat history in chains.
//!
//! This module provides several memory implementations:
//!
//! - [`ConversationBufferMemory`] — stores all messages in a buffer
//! - [`ConversationWindowMemory`] — keeps only the last K conversation turns
//! - [`ConversationSummaryMemory`] — summarizes older messages using an LLM
//! - [`VectorStoreMemory`] — retrieves relevant past messages via vector similarity
//! - [`ChatHistoryMemory`] — pluggable chat history with file and in-memory backends
//! - [`HybridMemory`] — combines multiple memory strategies with a builder API
//! - [`EntityMemory`] — tracks named entities mentioned in conversation using regex extraction
//! - [`KnowledgeGraphMemory`] — extracts and stores knowledge triples (subject-predicate-object) from conversation
//! - [`TokenBufferMemory`] — token-count-aware buffer that trims oldest messages when a limit is exceeded, with a pluggable [`TokenCounter`] trait
//! - [`SummaryBufferMemory`] — combines buffer memory with periodic summarization when token threshold is exceeded

pub mod buffer;
pub mod chat_history;
pub mod entity;
pub mod hybrid;
pub mod knowledge_graph;
pub mod summary;
pub mod summary_buffer;
pub mod token_buffer;
pub mod vector;
pub mod window;

pub use buffer::ConversationBufferMemory;
pub use chat_history::{
    prune_by_count, prune_by_token_count, ChatHistoryMemory, ChatHistoryStore, FileChatHistory,
    InMemoryChatHistory,
};
pub use entity::{Entity, EntityMemory, EntityStore, InMemoryEntityStore};
pub use hybrid::{ConversationTokenBufferMemory, HybridMemory, HybridMemoryBuilder};
pub use knowledge_graph::{
    KnowledgeGraph, KnowledgeGraphMemory, KnowledgeGraphMemoryBuilder, KnowledgeTriple,
    RegexTripleExtractor, TripleExtractor,
};
pub use summary::ConversationSummaryMemory;
pub use summary_buffer::{
    SimpleSummarizer, SummaryBufferMemory, SummaryBufferMemoryBuilder, SummaryStrategy,
    Summarizer, TemplateSummarizer,
};
pub use token_buffer::{
    CharBasedTokenCounter, SimpleTokenCounter, TokenBufferMemory, TokenBufferMemoryBuilder,
    TokenCounter,
};
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
