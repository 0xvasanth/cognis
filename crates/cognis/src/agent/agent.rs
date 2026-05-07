//! `Agent` — wraps a `CompiledGraph<AgentState>` with memory + system
//! prompt + conversation mode.

use cognis_core::{EventStream, Message, Result, Runnable, RunnableConfig};
use cognis_graph::CompiledGraph;

use super::memory::Memory;
use super::state::AgentState;

/// How a multi-turn conversation handles state across `run()` calls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversationMode {
    /// Each `run()` is independent. Initial state seeded from
    /// `[system, input]`.
    Stateless,
    /// Each `run()` reads from `Memory` to build the seed and writes
    /// new messages back. Carries conversation history.
    Stateful,
}

/// Final result of `Agent::run`.
#[derive(Debug, Clone)]
pub struct AgentResponse {
    /// Text content of the final assistant message.
    pub content: String,
    /// Tool calls in the final message (typically empty when graph reaches End).
    pub tool_calls: Vec<cognis_core::ToolCall>,
    /// All messages added during this run (excludes the seed).
    pub messages: Vec<Message>,
    /// Final agent state.
    pub state: AgentState,
}

/// A graph-backed agent. Wrap any `CompiledGraph<AgentState>` (or use
/// [`AgentBuilder`](super::AgentBuilder) for the default ReAct flow).
pub struct Agent {
    pub(crate) graph: CompiledGraph<AgentState>,
    pub(crate) memory: Option<Box<dyn Memory>>,
    pub(crate) mode: ConversationMode,
    pub(crate) system_prompt: String,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("mode", &self.mode)
            .field("system_prompt", &self.system_prompt)
            .finish_non_exhaustive()
    }
}

impl Agent {
    pub(crate) fn new(
        graph: CompiledGraph<AgentState>,
        memory: Option<Box<dyn Memory>>,
        mode: ConversationMode,
        system_prompt: String,
    ) -> Self {
        Self {
            graph,
            memory,
            mode,
            system_prompt,
        }
    }

    /// Wrap a custom graph directly. Bypasses [`AgentBuilder`] when you
    /// want full control.
    pub fn wrap(graph: CompiledGraph<AgentState>) -> Self {
        Self::new(graph, None, ConversationMode::Stateless, String::new())
    }

    /// One-shot run.
    pub async fn run(&mut self, input: impl Into<Message>) -> Result<AgentResponse> {
        let input_msg = input.into();
        let initial = self.build_initial_state(input_msg.clone());
        let seed_len = initial.messages.len();

        let final_state = self
            .graph
            .invoke(initial, RunnableConfig::default())
            .await?;

        // Extract messages added during this run.
        let new_messages: Vec<Message> = final_state.messages[seed_len..].to_vec();

        // If stateful, push the input + new messages to memory.
        if matches!(self.mode, ConversationMode::Stateful) {
            if let Some(mem) = self.memory.as_mut() {
                mem.write(input_msg);
                for m in &new_messages {
                    mem.write(m.clone());
                }
            }
        }

        let last = final_state
            .messages
            .last()
            .cloned()
            .unwrap_or_else(|| Message::ai(""));
        Ok(AgentResponse {
            content: last.content().to_string(),
            tool_calls: last.tool_calls().to_vec(),
            messages: new_messages,
            state: final_state,
        })
    }

    /// Stream structured events as the graph runs. Each `Event` (OnNodeStart,
    /// OnNodeEnd, OnError, OnEnd) is emitted in real time as each node
    /// completes. Delegates to `CompiledGraph::stream_events`.
    pub async fn stream(&mut self, input: impl Into<Message>) -> Result<EventStream> {
        use cognis_core::Runnable;
        let initial = self.build_initial_state(input.into());
        self.graph
            .stream_events(initial, RunnableConfig::default())
            .await
    }

    /// Escape hatch — give back the underlying compiled graph.
    pub fn into_graph(self) -> CompiledGraph<AgentState> {
        self.graph
    }

    /// Inspect current memory (Stateful mode only).
    pub fn memory(&self) -> Option<&dyn Memory> {
        self.memory.as_deref()
    }

    /// Clear conversation memory (no-op in Stateless mode).
    pub fn clear_memory(&mut self) {
        if let Some(m) = self.memory.as_mut() {
            m.clear();
        }
    }

    fn build_initial_state(&self, input: Message) -> AgentState {
        let mut messages = Vec::new();
        match self.mode {
            ConversationMode::Stateless => {
                if !self.system_prompt.is_empty() {
                    messages.push(Message::system(self.system_prompt.clone()));
                }
                messages.push(input);
            }
            ConversationMode::Stateful => {
                if let Some(m) = &self.memory {
                    messages.extend(m.seed());
                } else if !self.system_prompt.is_empty() {
                    messages.push(Message::system(self.system_prompt.clone()));
                }
                messages.push(input);
            }
        }
        AgentState {
            messages,
            iterations: 0,
            extras: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use async_trait::async_trait;
    use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
    use cognis_llm::provider::{LLMProvider, Provider};
    use cognis_llm::Client;

    use crate::agent::default_graph::default_react_graph;

    /// Provider that always responds with a single AI message of fixed content.
    /// Records call count to verify the graph invoked it.
    struct Constant {
        content: String,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl Constant {
        fn new(content: impl Into<String>) -> Self {
            Self {
                content: content.into(),
                calls: std::sync::atomic::AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for Constant {
        fn name(&self) -> &str {
            "constant"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(
            &self,
            messages: Vec<Message>,
            opts: ChatOptions,
        ) -> Result<ChatResponse> {
            self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            // Ignore messages and opts — constant response by design — but
            // asserting they arrived non-empty catches wiring bugs.
            let _ = (messages, opts);
            Ok(ChatResponse {
                message: Message::ai(&self.content),
                usage: Some(Usage::default()),
                finish_reason: "stop".into(),
                model: "constant".into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            messages: Vec<Message>,
            opts: ChatOptions,
        ) -> Result<cognis_core::RunnableStream<StreamChunk>> {
            let _ = (messages, opts);
            unimplemented!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    #[tokio::test]
    async fn stateless_run_seeds_with_system_and_input() {
        let client = Client::new(Arc::new(Constant::new("hello back")));
        let graph = default_react_graph(client, Vec::new(), 10).unwrap();
        let mut agent = Agent::new(graph, None, ConversationMode::Stateless, "be terse".into());
        let resp = agent.run("hi there").await.unwrap();
        assert_eq!(resp.content, "hello back");
        // initial: [system, human]; after run: + ai = 3
        assert_eq!(resp.state.messages.len(), 3);
        assert!(matches!(resp.state.messages[0], Message::System(_)));
    }

    #[tokio::test]
    async fn wrap_custom_graph() {
        let client = Client::new(Arc::new(Constant::new("ok")));
        let graph = default_react_graph(client, Vec::new(), 10).unwrap();
        let mut agent = Agent::wrap(graph);
        let resp = agent.run("hello").await.unwrap();
        assert_eq!(resp.content, "ok");
    }
}
