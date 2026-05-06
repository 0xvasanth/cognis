//! # cognis2-llm
//!
//! v2-beta LLM client. Re-exports `error`, `schemars`, and `Message` from
//! cognis2-core so v2 user code can target a single `crate_path =
//! "cognis2_llm"` and have all macro-generated paths resolve.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

// Re-export the cognis2-core error module so macro-generated code that
// targets `cognis2_llm::error::*` resolves transparently.
pub mod error;

// Re-export schemars for the same reason.
pub use cognis2_core::schemars;
pub use cognis2_core::schemars::JsonSchema;
pub use cognis2_core::schema_for;

// Re-export Message + variants since they're the LLM-conversation primitive.
pub use cognis2_core::message;
pub use cognis2_core::{AiMessage, HumanMessage, Message, SystemMessage, ToolCall, ToolMessage};

pub mod schema;
pub use schema::schema_for_tool;

pub mod chat;
pub use chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, ToolCallDelta, Usage};

pub mod tools;
pub use tools::{BaseTool, Tool, ToolDefinition, ToolRegistry};
pub mod client;
pub use client::{Client, ClientBuilder};

pub mod provider;
pub use provider::{LLMProvider, Provider};

/// Common imports for v2 user code building agents and tools.
pub mod prelude {
    pub use crate::{
        AiMessage, HumanMessage, JsonSchema, Message, SystemMessage, ToolCall, ToolMessage,
    };
    pub use crate::schema_for;
    pub use cognis2_core::prelude::*;
}
