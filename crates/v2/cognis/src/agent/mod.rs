//! Thin agent layer: AgentState + Memory + AgentBuilder + Agent.
//!
//! Each user-facing primitive composes the cognis2-core/graph/llm
//! foundations underneath.

#[allow(clippy::module_inception)]
pub mod agent;
pub mod builder;
pub mod default_graph;
pub mod memory;
pub mod state;
pub mod think_node;
pub mod tool_node;

pub use agent::{Agent, AgentResponse, ConversationMode};
pub use builder::AgentBuilder;
pub use default_graph::default_react_graph;
pub use memory::{Memory, Window};
pub use state::{AgentState, AgentStateUpdate};
pub use think_node::ThinkNode;
pub use tool_node::ToolDispatchNode;
