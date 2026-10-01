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
    /// Present when the builder was given `with_llm`; backs the typed
    /// streams ([`Agent::stream_elements`], [`Agent::stream_partial`]).
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

    /// Stream structured events as the graph runs, in real time. Delegates
    /// to `CompiledGraph::stream_events`.
    ///
    /// Events you can expect from the default ReAct graph:
    /// - `OnStart` / `OnEnd` for the run, `OnError` if it fails;
    /// - `OnNodeStart` / `OnNodeEnd` around each `think` and `act` step;
    /// - `OnLlmToken` for each chunk of model text, between a `think`
    ///   step's `OnNodeStart` and `OnNodeEnd` — only when streaming is
    ///   enabled via [`AgentBuilder::with_streaming`](super::AgentBuilder::with_streaming);
    /// - `OnToolStart` / `OnToolEnd` around each tool call inside `act`.
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
    /// The prompt must ask the model for a JSON array — nothing is injected.
    /// If the model answers without one, the stream ends with a
    /// `Serialization` error (see [`Client::stream_array`]).
    ///
    /// Needs the LLM client: build the agent with
    /// [`AgentBuilder::with_llm`](super::AgentBuilder::with_llm) (also when
    /// supplying a custom graph). [`Agent::wrap`] agents have no client and
    /// return a `Configuration` error.
    pub async fn stream_elements<T: serde::de::DeserializeOwned + Send + 'static>(
        &mut self,
        input: impl Into<Message>,
    ) -> Result<cognis_core::RunnableStream<T>> {
        let client = self.typed_stream_client("stream_elements")?;
        let state = self.build_initial_state(input.into());
        client.stream_array::<T>(state.messages).await
    }

    /// Stream the agent's response as a single object of type `T`, delivered
    /// as progressively-filled `T::Partial` snapshots.
    ///
    /// Different from [`Agent::stream_elements`], which yields whole list
    /// elements: this yields the same object repeatedly as more of it arrives,
    /// every field optional until streamed (see `#[derive(Partial)]`). String
    /// fields may be prefixes of their final value until the stream ends;
    /// numbers appear only once complete.
    ///
    /// Like `stream_elements`, it is a single LLM turn with no tool loop,
    /// does not write to memory, injects nothing into the prompt (ask the
    /// model for a JSON object yourself), ends with a `Serialization` error
    /// if no usable object arrives (see [`Client::stream_object_partial`]),
    /// and needs an agent built with
    /// [`AgentBuilder::with_llm`](super::AgentBuilder::with_llm)
    /// (`Configuration` error otherwise).
    pub async fn stream_partial<T: cognis_core::Partial>(
        &mut self,
        input: impl Into<Message>,
    ) -> Result<cognis_core::RunnableStream<T::Partial>> {
        let client = self.typed_stream_client("stream_partial")?;
        let state = self.build_initial_state(input.into());
        client.stream_object_partial::<T>(state.messages).await
    }

    fn typed_stream_client(&self, method: &str) -> Result<Client> {
        self.client.clone().ok_or_else(|| {
            cognis_core::CognisError::Configuration(format!(
                "{method} calls the LLM directly, but this agent has no client: \
                 build it with AgentBuilder::with_llm(..) (Agent::wrap and a \
                 builder given only with_graph(..) do not carry one)"
            ))
        })
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

    /// Streams scripted text pieces over the plain streaming path and records
    /// the messages it received. `chat_completion` panics, so a typed stream
    /// that falls back to the non-streaming tool path fails the test.
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
            messages: Vec<Message>,
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

    struct WeatherTool;
    #[async_trait]
    impl cognis_llm::Tool for WeatherTool {
        fn name(&self) -> &str {
            "weather"
        }
        fn description(&self) -> &str {
            "look up the weather"
        }
        fn args_schema(&self) -> Option<serde_json::Value> {
            Some(serde_json::json!({"type": "object"}))
        }
        async fn _run(&self, input: cognis_llm::ToolInput) -> Result<cognis_llm::ToolOutput> {
            let args = input.into_json();
            Ok(cognis_llm::ToolOutput::Text(format!(
                "sunny in {}",
                args["city"].as_str().unwrap_or("?")
            )))
        }
    }

    fn text_chunk(content: &str) -> StreamChunk {
        StreamChunk {
            content: content.into(),
            is_delta: true,
            ..Default::default()
        }
    }

    fn args_chunk(id: Option<&str>, name: Option<&str>, args: &str) -> StreamChunk {
        StreamChunk {
            is_delta: true,
            tool_calls_delta: vec![cognis_llm::chat::ToolCallDelta {
                index: 0,
                id: id.map(String::from),
                name: name.map(String::from),
                arguments_delta: Some(args.into()),
            }],
            ..Default::default()
        }
    }

    fn finish_chunk(reason: &str) -> StreamChunk {
        StreamChunk {
            is_done: true,
            finish_reason: Some(reason.into()),
            ..Default::default()
        }
    }

    /// Streams one scripted chunk list per model call and records the
    /// messages each call received. The non-streaming entry point errors,
    /// so an agent that was not built for streaming produces no tokens.
    struct TurnStreamer {
        turns: std::sync::Mutex<std::collections::VecDeque<Vec<StreamChunk>>>,
        seen: std::sync::Mutex<Vec<Vec<Message>>>,
    }

    #[async_trait]
    impl LLMProvider for TurnStreamer {
        fn name(&self) -> &str {
            "turn-streamer"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(
            &self,
            _messages: Vec<Message>,
            _opts: ChatOptions,
        ) -> Result<ChatResponse> {
            Err(cognis_core::CognisError::Internal(
                "non-streaming path used by a streaming agent".into(),
            ))
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
            self.seen.lock().unwrap().push(messages);
            let chunks = self.turns.lock().unwrap().pop_front().unwrap_or_default();
            Ok(cognis_core::RunnableStream::new(futures::stream::iter(
                chunks.into_iter().map(Ok),
            )))
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    /// The events a streaming UI cares about, as compact labels.
    fn label(e: &cognis_core::Event) -> Option<String> {
        use cognis_core::Event;
        Some(match e {
            Event::OnNodeStart { node, .. } => format!("node_start:{node}"),
            Event::OnNodeEnd { node, .. } => format!("node_end:{node}"),
            Event::OnLlmToken { token, .. } => format!("token:{token}"),
            Event::OnToolStart { tool, args, .. } => format!("tool_start:{tool}:{args}"),
            Event::OnToolEnd { tool, result, .. } => format!("tool_end:{tool}:{result}"),
            _ => return None,
        })
    }

    #[tokio::test]
    async fn stream_emits_tokens_from_both_model_calls_around_tool_events_when_streaming_enabled() {
        use futures::StreamExt;
        let provider = Arc::new(TurnStreamer {
            turns: std::sync::Mutex::new(
                vec![
                    vec![
                        text_chunk("Checking"),
                        text_chunk(" the weather."),
                        args_chunk(Some("call_1"), Some("weather"), "{\"ci"),
                        args_chunk(None, None, "ty\":\"Par"),
                        args_chunk(None, None, "is\"}"),
                        finish_chunk("tool_calls"),
                    ],
                    vec![
                        text_chunk("It is"),
                        text_chunk(" sunny."),
                        finish_chunk("stop"),
                    ],
                ]
                .into(),
            ),
            seen: Default::default(),
        });
        let mut agent = crate::agent::AgentBuilder::new()
            .with_llm(Client::new(provider.clone()))
            .with_tool(Arc::new(WeatherTool))
            .with_streaming(true)
            .build()
            .unwrap();

        let events: Vec<cognis_core::Event> = agent
            .stream("Weather in Paris?")
            .await
            .unwrap()
            .collect()
            .await;
        let labels: Vec<String> = events.iter().filter_map(label).collect();

        assert_eq!(
            labels,
            vec![
                "node_start:think",
                "token:Checking",
                "token: the weather.",
                "node_end:think",
                "node_start:act",
                "tool_start:weather:{\"city\":\"Paris\"}",
                "tool_end:weather:\"sunny in Paris\"",
                "node_end:act",
                "node_start:think",
                "token:It is",
                "token: sunny.",
                "node_end:think",
            ],
            "got: {labels:#?}"
        );

        // The second model call saw the tool result produced from the
        // arguments assembled out of the first call's stream.
        let seen = provider.seen.lock().unwrap();
        assert_eq!(seen.len(), 2, "two model calls expected");
        let last = seen[1].last().unwrap();
        assert!(matches!(last, Message::Tool(_)), "got: {last:?}");
        assert_eq!(last.content(), "sunny in Paris");
    }

    #[tokio::test]
    async fn stream_emits_no_tokens_when_streaming_not_enabled() {
        use futures::StreamExt;
        let client = Client::new(Arc::new(Constant::new("hello back")));
        let mut agent = crate::agent::AgentBuilder::new()
            .with_llm(client)
            .build()
            .unwrap();
        let events: Vec<cognis_core::Event> = agent.stream("hi").await.unwrap().collect().await;
        let labels: Vec<String> = events.iter().filter_map(label).collect();
        assert_eq!(labels, vec!["node_start:think", "node_end:think"]);
    }

    #[tokio::test]
    async fn stream_elements_works_when_builder_has_both_custom_graph_and_llm() {
        use futures::StreamExt;
        let provider = Arc::new(ArrayStreamer {
            seen: Default::default(),
            pieces: &["[{\"id\":1},", "{\"id\":2}]"],
        });
        let graph_client = Client::new(Arc::new(Constant::new("ok")));
        let graph = default_react_graph(graph_client, Vec::new(), 10).unwrap();
        let mut agent = crate::agent::AgentBuilder::new()
            .with_graph(graph)
            .with_llm(Client::new(provider))
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
    }

    #[tokio::test]
    async fn typed_streams_name_with_llm_when_builder_agent_has_only_custom_graph() {
        let client = Client::new(Arc::new(Constant::new("ok")));
        let graph = default_react_graph(client, Vec::new(), 10).unwrap();
        let mut agent = crate::agent::AgentBuilder::new()
            .with_graph(graph)
            .build()
            .unwrap();
        let elements = match agent.stream_elements::<Step>("go").await {
            Ok(_) => panic!("expected Configuration error"),
            Err(e) => e.to_string(),
        };
        let partial = match agent.stream_partial::<Report>("go").await {
            Ok(_) => panic!("expected Configuration error"),
            Err(e) => e.to_string(),
        };
        for (method, msg) in [("stream_elements", elements), ("stream_partial", partial)] {
            assert!(msg.contains(method), "got: {msg}");
            assert!(msg.contains("with_llm"), "got: {msg}");
            assert!(
                !msg.contains("needs an AgentBuilder-built agent"),
                "a builder-built agent hit this error, so that wording is wrong: {msg}"
            );
        }
    }
}
