//! Middleware system for Deep Agents.
//!
//! Middleware hooks run before/after model calls and tool executions,
//! allowing pluggable behaviour such as filesystem access, memory
//! injection, and context summarization.

pub mod filesystem;
pub mod logging;
pub mod memory;
pub mod patch_tool_calls;
pub mod rate_limiter;
pub mod skills;
pub mod subagent;
pub mod summarization;

use async_trait::async_trait;
use serde_json::Value;

use crate::agent::DeepAgentError;

/// Result type for middleware operations.
pub type Result<T> = std::result::Result<T, DeepAgentError>;

/// A mutable reference to the agent state (a JSON object with at least a `"messages"` key).
pub type AgentState = Value;

/// Trait for pluggable middleware in the Deep Agent pipeline.
///
/// All methods have default no-op implementations so that concrete
/// middleware only needs to override the hooks it cares about.
#[async_trait]
pub trait Middleware: Send + Sync {
    /// A human-readable name for this middleware.
    fn name(&self) -> &str;

    /// Called before the model is invoked. Can mutate the state (e.g. inject context).
    async fn before_model(&self, _state: &mut AgentState) -> Result<()> {
        Ok(())
    }

    /// Called after the model responds. Can inspect or mutate the state.
    async fn after_model(&self, _state: &mut AgentState) -> Result<()> {
        Ok(())
    }

    /// Called before a tool is executed.
    async fn before_tool(&self, _state: &mut AgentState, _tool_name: &str) -> Result<()> {
        Ok(())
    }

    /// Called after a tool execution completes.
    async fn after_tool(
        &self,
        _state: &mut AgentState,
        _tool_name: &str,
        _result: &str,
    ) -> Result<()> {
        Ok(())
    }
}
