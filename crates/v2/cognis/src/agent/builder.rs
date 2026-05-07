//! Fluent builder for `Agent`.

use std::sync::Arc;

use cognis2_core::{CognisError, Result};
use cognis2_graph::CompiledGraph;
use cognis2_llm::{Client, Tool};

use super::agent::{Agent, ConversationMode};
use super::default_graph::default_react_graph_with_limits;
use super::memory::{Memory, Window};
use super::state::AgentState;
use crate::backend::Backend;
use crate::tools::{
    ApprovalGatedTool, Approver, FileEditTool, FileExistsTool, FileGlobTool, FileGrepTool,
    FileListTool, FileReadTool, FileWriteTool,
};

const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful assistant. Use tools when needed. Be concise.";

/// Fluent builder for [`Agent`].
pub struct AgentBuilder {
    client: Option<Client>,
    tools: Vec<Arc<dyn Tool>>,
    system_prompt: Option<String>,
    memory: Option<Box<dyn Memory>>,
    max_iterations: u32,
    max_tool_calls: Option<u32>,
    mode: ConversationMode,
    custom_graph: Option<CompiledGraph<AgentState>>,
    approver: Option<Arc<dyn Approver>>,
}

impl Default for AgentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentBuilder {
    /// New builder with sensible defaults.
    pub fn new() -> Self {
        Self {
            client: None,
            tools: Vec::new(),
            system_prompt: None,
            memory: None,
            max_iterations: 10,
            max_tool_calls: None,
            mode: ConversationMode::Stateless,
            custom_graph: None,
            approver: None,
        }
    }

    /// LLM client.
    pub fn with_llm(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    /// Add a single tool.
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Add many tools.
    pub fn with_tools<I: IntoIterator<Item = Arc<dyn Tool>>>(mut self, tools: I) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Wire all seven filesystem tools (`fs_read` / `fs_write` / `fs_edit`
    /// / `fs_ls` / `fs_glob` / `fs_grep` / `fs_exists`) against `backend`.
    pub fn with_filesystem(mut self, backend: Arc<dyn Backend>) -> Self {
        self.tools
            .push(Arc::new(FileReadTool::new(backend.clone())));
        self.tools
            .push(Arc::new(FileWriteTool::new(backend.clone())));
        self.tools
            .push(Arc::new(FileEditTool::new(backend.clone())));
        self.tools
            .push(Arc::new(FileListTool::new(backend.clone())));
        self.tools
            .push(Arc::new(FileGlobTool::new(backend.clone())));
        self.tools
            .push(Arc::new(FileGrepTool::new(backend.clone())));
        self.tools.push(Arc::new(FileExistsTool::new(backend)));
        self
    }

    /// Override the system prompt.
    pub fn with_system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Custom memory backend (overrides the default Window).
    pub fn with_memory(mut self, mem: impl Memory + 'static) -> Self {
        self.memory = Some(Box::new(mem));
        self
    }

    /// Override max LLM iterations (default 10).
    pub fn with_max_iterations(mut self, n: u32) -> Self {
        self.max_iterations = n;
        self
    }

    /// Cap the number of tool messages this agent's loop may accumulate
    /// across a single run.
    pub fn with_max_tool_calls(mut self, n: u32) -> Self {
        self.max_tool_calls = Some(n);
        self
    }

    /// Wrap every registered tool in an [`ApprovalGatedTool`] backed by
    /// `approver`. Tools registered AFTER this call are NOT auto-wrapped —
    /// call `.with_approver(...)` last.
    pub fn with_approver(mut self, approver: Arc<dyn Approver>) -> Self {
        self.approver = Some(approver);
        self
    }

    /// Power-user override: supply your own graph instead of the default ReAct.
    pub fn with_graph(mut self, graph: CompiledGraph<AgentState>) -> Self {
        self.custom_graph = Some(graph);
        self
    }

    /// Mark the agent stateful — memory is read on each `run()` and updated.
    pub fn stateful(mut self) -> Self {
        self.mode = ConversationMode::Stateful;
        self
    }

    /// Mark the agent stateless — each `run()` is independent.
    pub fn stateless(mut self) -> Self {
        self.mode = ConversationMode::Stateless;
        self
    }

    /// Build the Agent.
    pub fn build(self) -> Result<Agent> {
        let system_prompt = self
            .system_prompt
            .unwrap_or_else(|| DEFAULT_SYSTEM_PROMPT.to_string());

        let graph = if let Some(g) = self.custom_graph {
            g
        } else {
            let client = self.client.ok_or_else(|| {
                CognisError::Configuration(
                    "AgentBuilder requires .with_llm(client) (or .with_graph for custom graphs)"
                        .into(),
                )
            })?;
            let tools: Vec<Arc<dyn Tool>> = if let Some(approver) = self.approver {
                self.tools
                    .into_iter()
                    .map(|t| Arc::new(ApprovalGatedTool::new(t, approver.clone())) as Arc<dyn Tool>)
                    .collect()
            } else {
                self.tools
            };
            default_react_graph_with_limits(
                client,
                tools,
                self.max_iterations,
                self.max_tool_calls,
            )?
        };

        let memory: Option<Box<dyn Memory>> = match (self.mode, self.memory) {
            (ConversationMode::Stateful, Some(m)) => Some(m),
            (ConversationMode::Stateful, None) => {
                Some(Box::new(Window::new(50).with_system(system_prompt.clone())))
            }
            (ConversationMode::Stateless, _) => None,
        };

        Ok(Agent::new(graph, memory, self.mode, system_prompt))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_client_errors() {
        let err = AgentBuilder::new().build().unwrap_err();
        assert!(format!("{err}").contains("with_llm"));
    }
}
