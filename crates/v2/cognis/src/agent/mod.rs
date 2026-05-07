//! Thin agent layer: AgentState + Memory + AgentBuilder + Agent.
//!
//! Each user-facing primitive composes the cognis2-core/graph/llm
//! foundations underneath.

#[allow(clippy::module_inception)]
pub mod agent;
pub mod builder;
pub mod default_graph;
pub mod health;
pub mod lifecycle;
pub mod memory;
pub mod plugin;
pub mod state;
pub mod think_node;
pub mod tool_node;
pub mod workflow;

pub use agent::{Agent, AgentResponse, ConversationMode};
pub use builder::AgentBuilder;
pub use default_graph::{default_react_graph, default_react_graph_with_limits};
pub use health::AgentHealth;
pub use lifecycle::{AgentLifecycle, OnStart};
pub use memory::{Buffer, Memory, SummaryMemory, VectorMemory, Window};
pub use plugin::{AgentPlugin, FnPlugin};
pub use state::{AgentState, AgentStateUpdate};
pub use think_node::ThinkNode;
pub use tool_node::ToolDispatchNode;
pub use workflow::{Workflow, WorkflowState, WorkflowStateUpdate};
