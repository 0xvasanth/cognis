//! Chat sessions — a collection of messages and function call specs.
//!
//! Mirrors Python `langchain_core.chat_sessions`.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::messages::Message;

/// A single conversation, channel, or other group of messages.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatSession {
    /// Messages loaded from the source.
    pub messages: Vec<Message>,

    /// Function calling specs associated with the session.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub functions: Vec<Value>,
}

impl ChatSession {
    pub fn new(messages: Vec<Message>) -> Self {
        Self {
            messages,
            functions: Vec::new(),
        }
    }

    pub fn with_functions(mut self, functions: Vec<Value>) -> Self {
        self.functions = functions;
        self
    }
}
