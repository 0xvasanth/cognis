//! `Agent` — wraps a `CompiledGraph<AgentState>` with memory + system
//! prompt + conversation mode.

use cognis_core::{EventStream, Message, Result, Runnable, RunnableConfig};
use cognis_graph::CompiledGraph;
use cognis_llm::Client;

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
    /// Present only for builder-constructed agents; backs [`Agent::stream_elements`].
    pub(crate) client: Option<Client>,
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
        client: Option<Client>,
    ) -> Self {
        Self {
            graph,
            memory,
            mode,
            system_prompt,
            client,
        }
    }

    /// Wrap a custom graph directly. Bypasses [`AgentBuilder`] when you
    /// want full control.
    pub fn wrap(graph: CompiledGraph<AgentState>) -> Self {
        Self::new(
            graph,
            None,
            ConversationMode::Stateless,
            String::new(),
            None,
        )
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

    /// Stream the agent's structured list output element-by-element.
    ///
    /// Different from [`Agent::stream`], which emits graph events: this calls
    /// the LLM directly with the seeded conversation and yields each element
    /// of the JSON array as soon as it closes. It bypasses the tool loop and
    /// does not write to memory.
    ///
    /// Requires an agent built via [`AgentBuilder`](super::AgentBuilder)
    /// (the LLM client is needed); `wrap`-constructed agents and custom
    /// graphs return a `Configuration` error.
    pub async fn stream_elements<T: serde::de::DeserializeOwned + Send + 'static>(
        &mut self,
        input: impl Into<Message>,
    ) -> Result<cognis_core::RunnableStream<T>> {
        let client = self.client.clone().ok_or_else(|| {
            cognis_core::CognisError::Configuration(
                "stream_elements needs an AgentBuilder-built agent".into(),
            )
        })?;
        let state = self.build_initial_state(input.into());
        client.stream_array::<T>(state.messages).await
    }

    /// Stream the agent's response as a single object of type `T`, delivered
    /// as progressively-filled `T::Partial` snapshots.
    ///
    /// Different from [`Agent::stream_elements`], which yields whole list
    /// elements: this yields the same object repeatedly as more of it arrives,
    /// every field optional until streamed (see `#[derive(Partial)]`). Like
    /// `stream_elements`, it is a single LLM turn with no tool loop and does
    /// not write to memory, and needs an [`AgentBuilder`](super::AgentBuilder)
    /// built agent (`Configuration` error otherwise).
    pub async fn stream_partial<T: cognis_core::Partial>(
        &mut self,
        input: impl Into<Message>,
    ) -> Result<cognis_core::RunnableStream<T::Partial>> {
        let client = self.client.clone().ok_or_else(|| {
            cognis_core::CognisError::Configuration(
                "stream_partial needs an AgentBuilder-built agent".into(),
            )
        })?;
        let state = self.build_initial_state(input.into());
        client.stream_object_partial::<T>(state.messages).await
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
        let mut agent = Agent::new(
            graph,
            None,
            ConversationMode::Stateless,
            "be terse".into(),
            None,
        );
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

    #[derive(serde::Deserialize, Debug, PartialEq)]
    struct Step {
        id: u32,
    }

    #[derive(cognis_macros::Partial)]
    #[allow(dead_code)]
    struct Report {
        title: String,
        score: u32,
    }

    /// Streams scripted text pieces and records the messages it received.
    struct ArrayStreamer {
        seen: std::sync::Mutex<Vec<Message>>,
        pieces: &'static [&'static str],
    }

    #[async_trait]
    impl LLMProvider for ArrayStreamer {
        fn name(&self) -> &str {
            "array-streamer"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(
            &self,
            _messages: Vec<Message>,
            _opts: ChatOptions,
        ) -> Result<ChatResponse> {
            unimplemented!()
        }
        async fn chat_completion_stream(
            &self,
            _messages: Vec<Message>,
            _opts: ChatOptions,
        ) -> Result<cognis_core::RunnableStream<StreamChunk>> {
            unimplemented!()
        }
        async fn chat_completion_stream_with_tools(
            &self,
            messages: Vec<Message>,
            _tools: Vec<cognis_llm::ToolDefinition>,
            _opts: ChatOptions,
        ) -> Result<cognis_core::RunnableStream<StreamChunk>> {
            *self.seen.lock().unwrap() = messages;
            let chunks: Vec<Result<StreamChunk>> = self
                .pieces
                .iter()
                .map(|s| {
                    Ok(StreamChunk {
                        content: (*s).into(),
                        is_delta: true,
                        is_done: false,
                        finish_reason: None,
                        usage: None,
                        tool_calls_delta: vec![],
                    })
                })
                .collect();
            Ok(cognis_core::RunnableStream::new(futures::stream::iter(
                chunks,
            )))
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    #[tokio::test]
    async fn stream_elements_yields_typed_elements_with_system_prompt_seeded() {
        use futures::StreamExt;
        let provider = Arc::new(ArrayStreamer {
            seen: Default::default(),
            pieces: &["[{\"id\":1},", "{\"id\":2}]"],
        });
        let mut agent = crate::agent::AgentBuilder::new()
            .with_llm(Client::new(provider.clone()))
            .with_system_prompt("plan things")
            .build()
            .unwrap();
        let got: Vec<Step> = agent
            .stream_elements::<Step>("go")
            .await
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
            .await;
        assert_eq!(got, vec![Step { id: 1 }, Step { id: 2 }]);
        let seen = provider.seen.lock().unwrap();
        assert!(
            matches!(seen.first(), Some(Message::System(_))),
            "got: {seen:?}"
        );
        assert_eq!(seen.last().unwrap().content(), "go");
    }

    #[tokio::test]
    async fn stream_elements_errors_for_wrapped_agent_without_client() {
        let client = Client::new(Arc::new(Constant::new("ok")));
        let graph = default_react_graph(client, Vec::new(), 10).unwrap();
        let mut agent = Agent::wrap(graph);
        let err = match agent.stream_elements::<Step>("go").await {
            Ok(_) => panic!("expected Configuration error"),
            Err(e) => e,
        };
        assert!(
            matches!(err, cognis_core::CognisError::Configuration(_)),
            "got: {err:?}"
        );
    }

    #[tokio::test]
    async fn stream_partial_yields_typed_snapshots_with_system_prompt_seeded() {
        use futures::StreamExt;
        let provider = Arc::new(ArrayStreamer {
            seen: Default::default(),
            pieces: &["{\"title\":\"x\"", ",\"score\":5}"],
        });
        let mut agent = crate::agent::AgentBuilder::new()
            .with_llm(Client::new(provider.clone()))
            .with_system_prompt("report")
            .build()
            .unwrap();
        let got: Vec<ReportPartial> = agent
            .stream_partial::<Report>("go")
            .await
            .unwrap()
            .map(|r| r.unwrap())
            .collect()
            .await;
        let last = got.last().unwrap();
        assert_eq!(last.title.as_deref(), Some("x"));
        assert_eq!(last.score, Some(5));
        let seen = provider.seen.lock().unwrap();
        assert!(
            matches!(seen.first(), Some(Message::System(_))),
            "got: {seen:?}"
        );
        assert_eq!(seen.last().unwrap().content(), "go");
    }

    #[tokio::test]
    async fn stream_partial_errors_for_wrapped_agent_without_client() {
        let client = Client::new(Arc::new(Constant::new("ok")));
        let graph = default_react_graph(client, Vec::new(), 10).unwrap();
        let mut agent = Agent::wrap(graph);
        let err = match agent.stream_partial::<Report>("go").await {
            Ok(_) => panic!("expected Configuration error"),
            Err(e) => e,
        };
        assert!(
            matches!(err, cognis_core::CognisError::Configuration(_)),
            "got: {err:?}"
        );
    }
}
