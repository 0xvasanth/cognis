//! Configuration for the Deep Agent.

use std::sync::Arc;

use rustchain_core::tools::BaseTool;

use crate::backends::{Backend, StateBackend};
use crate::middleware::Middleware;

/// Configuration for building a Deep Agent via [`create_deep_agent`](crate::create_deep_agent).
pub struct DeepAgentConfig {
    /// The model identifier to use (e.g. `"claude-sonnet-4-6"`).
    pub model_name: String,
    /// Maximum number of agent loop iterations before stopping.
    pub max_iterations: u32,
    /// Optional system prompt prepended to the conversation.
    pub system_prompt: Option<String>,
    /// Tools available to the agent.
    pub tools: Vec<Arc<dyn BaseTool>>,
    /// Middleware pipeline applied around model and tool calls.
    pub middleware: Vec<Arc<dyn Middleware>>,
    /// Backend for persisting agent state across sessions.
    pub backend: Box<dyn Backend>,
}

impl Default for DeepAgentConfig {
    fn default() -> Self {
        Self {
            model_name: "claude-sonnet-4-6".to_string(),
            max_iterations: 25,
            system_prompt: None,
            tools: Vec::new(),
            middleware: Vec::new(),
            backend: Box::new(StateBackend::new()),
        }
    }
}

impl DeepAgentConfig {
    /// Create a new config with the given model name.
    pub fn with_model(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = model_name.into();
        self
    }

    /// Set the maximum number of iterations.
    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Add a tool to the agent.
    pub fn with_tool(mut self, tool: Arc<dyn BaseTool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Set all tools at once.
    pub fn with_tools(mut self, tools: Vec<Arc<dyn BaseTool>>) -> Self {
        self.tools = tools;
        self
    }

    /// Add a middleware to the pipeline.
    pub fn with_middleware(mut self, mw: Arc<dyn Middleware>) -> Self {
        self.middleware.push(mw);
        self
    }

    /// Set the backend.
    pub fn with_backend(mut self, backend: Box<dyn Backend>) -> Self {
        self.backend = backend;
        self
    }
}
