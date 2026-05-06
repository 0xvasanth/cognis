//! # cognis2
//!
//! v2-beta umbrella crate. Re-exports core/graph/llm and adds a thin
//! agent layer.

#![warn(missing_docs)]
#![warn(rust_2018_idioms)]

// Re-exports — what `use cognis2::*` gives users.
pub use cognis2_core;
pub use cognis2_graph;
pub use cognis2_llm;

pub use cognis2_core::{
    CognisError, Event, EventStream, Extensions, JsonSchema, Message, Observer, Result, Runnable,
    RunnableConfig, RunnableStream, ToolCall,
};
pub use cognis2_graph::{
    node_fn, Checkpointer, CompiledGraph, Goto, Graph, GraphState, InMemoryCheckpointer, Node,
    NodeCtx, NodeOut,
};
pub use cognis2_llm::{
    BaseTool, ChatOptions, ChatResponse, Client, ClientBuilder, LLMProvider, Provider,
    SchemaBasedTool, StreamChunk, Tool, ToolDefinition, ToolInput, ToolOutput, ToolRegistry, Usage,
};

// Filled in by subsequent tasks:
pub mod agent;
pub use agent::{
    default_react_graph, Agent, AgentBuilder, AgentResponse, AgentState, AgentStateUpdate,
    ConversationMode, Memory, ThinkNode, ToolDispatchNode, Window,
};

/// Common imports for v2 user code building agents.
pub mod prelude {
    pub use crate::*;
    pub use async_trait::async_trait;
}
