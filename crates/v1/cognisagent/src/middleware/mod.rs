//! Middleware system for Deep Agents.
//!
//! Middleware hooks run before/after model calls and tool executions,
//! allowing pluggable behaviour such as filesystem access, memory
//! injection, and context summarization.
//!
//! Built-in middleware:
//! - [`filesystem::FilesystemMiddleware`] — file read, write, edit, ls, glob, grep
//! - [`memory::MemoryMiddleware`] — inject persistent memory into context
//! - [`subagent::SubAgentMiddleware`] — isolated sub-agent delegation
//! - [`summarization::SummarizationMiddleware`] — context window management
//! - [`skills::SkillsMiddleware`] — custom skill loading
//! - [`patch_tool_calls::PatchToolCallsMiddleware`] — tool call correction
//! - [`rate_limiter::RateLimiterMiddleware`] — token bucket rate limiting with cost tracking
//! - [`logging::LoggingMiddleware`] — structured logging with redaction
//! - [`context::ContextMiddleware`] — dynamic context injection
//! - [`planning::PlanningMiddleware`] — plan-then-execute support with step tracking, dependencies, and status injection
//! - [`prompt_caching::PromptCachingMiddleware`] — Anthropic-style prompt caching with cache control injection
//! - [`todo::TodoListMiddleware`] — todo list tracking with status injection
//! - [`approval_gate::ApprovalGateMiddleware`] — async human-approval gate for tools that declare `requires_approval`

pub mod approval_gate;
pub mod context;
pub mod filesystem;
pub mod logging;
pub mod memory;
pub mod patch_tool_calls;
pub mod pipeline;
pub mod planning;
pub mod prompt_caching;
pub mod rate_limiter;
pub mod skills;
pub mod subagent;
pub mod summarization;
pub mod todo;

use async_trait::async_trait;
use serde_json::Value;

use crate::agent::DeepAgentError;

/// Result type for middleware operations.
pub type Result<T> = std::result::Result<T, DeepAgentError>;

/// A mutable reference to the agent state (a JSON object with at least a `"messages"` key).
pub type AgentState = Value;

/// The decision a middleware returns from [`Middleware::gate_tool`] to control
/// whether a tool invocation should proceed.
#[derive(Debug, Clone)]
pub enum ToolGateDecision {
    /// The tool invocation should proceed normally.
    Continue,
    /// The tool invocation should be skipped. The supplied `observation`
    /// replaces the would-be tool result (typically a user-rejection message).
    Reject { observation: String },
}

impl ToolGateDecision {
    /// Whether this decision short-circuits tool execution.
    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Reject { .. })
    }
}

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

    /// Called before a tool is executed, with an opportunity to short-circuit.
    ///
    /// Unlike [`before_tool`](Self::before_tool), this hook returns a
    /// [`ToolGateDecision`] that can reject the invocation and substitute an
    /// observation instead of running the tool. Used by approval-gate
    /// middleware to suspend execution until a human resolves the request.
    ///
    /// Default implementation is a no-op returning `Continue`.
    async fn gate_tool(
        &self,
        _state: &mut AgentState,
        _tool_name: &str,
        _tool_input: &Value,
    ) -> Result<ToolGateDecision> {
        Ok(ToolGateDecision::Continue)
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
