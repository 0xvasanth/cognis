//! Fluent builder for `Agent`.

use std::sync::Arc;

use cognis2_core::{CognisError, Result};
use cognis2_graph::CompiledGraph;
use cognis2_llm::{Client, Tool};

use super::agent::{Agent, ConversationMode};
use super::default_graph::default_react_graph;
use super::memory::{Memory, Window};
use super::state::AgentState;

const DEFAULT_SYSTEM_PROMPT: &str =
    "You are a helpful assistant. Use tools when needed. Be concise.";

/// Fluent builder for [`Agent`].
pub struct AgentBuilder {
    client: Option<Client>,
    tools: Vec<Arc<dyn Tool>>,
    system_prompt: Option<String>,
    memory: Option<Box<dyn Memory>>,
    max_iterations: u32,
    mode: ConversationMode,
    custom_graph: Option<CompiledGraph<AgentState>>,
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
            mode: ConversationMode::Stateless,
            custom_graph: None,
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

    /// Override max iterations (default 10).
    pub fn with_max_iterations(mut self, n: u32) -> Self {
        self.max_iterations = n;
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
            default_react_graph(client, self.tools, self.max_iterations)?
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
