//! Prebuilt agent graphs.
//!
//! This module provides ready-to-use agent graph constructors,
//! starting with the classic ReAct (Reasoning + Acting) agent pattern
//! and the focused tool-calling agent.

pub mod agents;
pub mod chat_agent;
pub mod react_agent;
pub mod tool_agent;

pub use agents::{
    create_react_agent as create_standalone_react_agent, create_tool_node, AgentState,
    ReActAgent, ReactAgentConfig, ReactAgentConfigBuilder, ToolCall as StandaloneToolCall,
    ToolChoice, ToolDefinition, ToolNode,
};
pub use chat_agent::{create_chat_agent, ChatAgent, ChatAgentConfig};
pub use react_agent::create_react_agent;
pub use tool_agent::{create_tool_agent, ToolAgentConfig};
