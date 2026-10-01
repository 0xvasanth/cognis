//! `ThinkNode` — calls the LLM and decides whether to invoke tools or end.

use async_trait::async_trait;
use futures::StreamExt;

use cognis_core::stream::Event;
use cognis_core::{Message, Result};
use cognis_graph::{Goto, Node, NodeCtx, NodeOut};
use cognis_llm::{ChatOptions, Client, StreamAggregator, ToolDefinition};

use super::state::{AgentState, AgentStateUpdate};

/// LLM call node. If the LLM emits tool calls, routes to "act"; otherwise
/// terminates the graph.
///
/// Two limits apply:
/// - `max_iterations` — how many LLM calls the loop is allowed to make.
/// - `max_tool_calls` — how many tool messages may accumulate in state.
///   Counted directly from `state.messages` so no extra bookkeeping is
///   needed. `None` means unlimited.
pub struct ThinkNode {
    client: Client,
    tool_defs: Vec<ToolDefinition>,
    max_iterations: u32,
    max_tool_calls: Option<u32>,
    streaming: bool,
}

impl ThinkNode {
    /// New `ThinkNode` from a Client + tool definitions + max_iterations.
    pub fn new(client: Client, tool_defs: Vec<ToolDefinition>, max_iterations: u32) -> Self {
        Self {
            client,
            tool_defs,
            max_iterations,
            max_tool_calls: None,
            streaming: false,
        }
    }

    /// Cap the number of tool messages this loop may accumulate.
    pub fn with_max_tool_calls(mut self, n: u32) -> Self {
        self.max_tool_calls = Some(n);
        self
    }

    /// Stream the model's output token-by-token, emitting
    /// [`Event::OnLlmToken`](cognis_core::stream::Event::OnLlmToken) per
    /// non-empty content chunk. The final message is identical to the
    /// non-streaming path. Cancellation is checked per chunk. Default off.
    pub fn with_streaming(mut self, on: bool) -> Self {
        self.streaming = on;
        self
    }
}

#[async_trait]
impl Node<AgentState> for ThinkNode {
    async fn execute(&self, state: &AgentState, ctx: &NodeCtx<'_>) -> Result<NodeOut<AgentState>> {
        if ctx.is_cancelled() {
            return Err(cognis_core::CognisError::Cancelled);
        }
        if state.iterations >= self.max_iterations {
            return Ok(NodeOut {
                update: AgentStateUpdate {
                    messages: vec![Message::ai(format!(
                        "[max_iterations={} reached]",
                        self.max_iterations
                    ))],
                    iterations: 0,
                },
                goto: Goto::end(),
            });
        }

        if let Some(limit) = self.max_tool_calls {
            let used = state
                .messages
                .iter()
                .filter(|m| matches!(m, Message::Tool(_)))
                .count() as u32;
            if used >= limit {
                return Ok(NodeOut {
                    update: AgentStateUpdate {
                        messages: vec![Message::ai(format!("[max_tool_calls={limit} reached]"))],
                        iterations: 0,
                    },
                    goto: Goto::end(),
                });
            }
        }

        let messages = state.messages.clone();
        let msg = if self.streaming {
            let mut stream = self
                .client
                .provider()
                .chat_completion_stream_with_tools(
                    messages,
                    self.tool_defs.clone(),
                    ChatOptions::default(),
                )
                .await?;
            let mut agg = StreamAggregator::new();
            while let Some(item) = stream.next().await {
                // A model turn can stream for a long time; honour
                // cancellation between chunks instead of only between nodes.
                if ctx.is_cancelled() {
                    return Err(cognis_core::CognisError::Cancelled);
                }
                let chunk = item?;
                if !chunk.content.is_empty() {
                    ctx.emit(&Event::OnLlmToken {
                        token: chunk.content.clone(),
                        run_id: ctx.run_id,
                    });
                }
                agg.push(chunk);
            }
            agg.finalize().message
        } else {
            self.client
                .provider()
                .chat_completion_with_tools(
                    messages,
                    self.tool_defs.clone(),
                    ChatOptions::default(),
                )
                .await?
                .message
        };
        let route_to_tools = msg.has_tool_calls();
        Ok(NodeOut {
            update: AgentStateUpdate {
                messages: vec![msg],
                iterations: 1,
            },
            goto: if route_to_tools {
                Goto::node("act")
            } else {
                Goto::end()
            },
        })
    }

    fn name(&self) -> &str {
        "think"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use async_trait::async_trait;
    use cognis_core::{AiMessage, RunnableConfig, RunnableStream, ToolCall};
    use cognis_llm::chat::{ChatResponse, HealthStatus, StreamChunk, Usage};
    use cognis_llm::provider::{LLMProvider, Provider};
    use uuid::Uuid;

    /// Provider that returns canned responses based on call index.
    /// Records all received (messages, opts) pairs for assertion.
    struct ScriptedProvider {
        responses: std::sync::Mutex<std::collections::VecDeque<Message>>,
        received: std::sync::Mutex<Vec<(Vec<Message>, ChatOptions)>>,
    }

    impl ScriptedProvider {
        fn new(responses: Vec<Message>) -> Self {
            Self {
                responses: std::sync::Mutex::new(responses.into()),
                received: std::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for ScriptedProvider {
        fn name(&self) -> &str {
            "scripted"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(
            &self,
            messages: Vec<Message>,
            opts: ChatOptions,
        ) -> Result<ChatResponse> {
            self.received.lock().unwrap().push((messages.clone(), opts));
            let mut q = self.responses.lock().unwrap();
            let msg = q
                .pop_front()
                .unwrap_or(Message::ai("(no more responses scripted)"));
            Ok(ChatResponse {
                message: msg,
                usage: Some(Usage::default()),
                finish_reason: "stop".into(),
                model: "scripted".into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            messages: Vec<Message>,
            opts: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            let _ = (messages, opts);
            unimplemented!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    fn ai_with_tool_call(name: &str) -> Message {
        Message::Ai(AiMessage {
            content: String::new(),
            tool_calls: vec![ToolCall {
                id: format!("call_{name}"),
                name: name.to_string(),
                arguments: serde_json::json!({}),
            }],
            parts: Vec::new(),
        })
    }

    #[tokio::test]
    async fn routes_to_act_when_tool_calls() {
        let provider = Arc::new(ScriptedProvider::new(vec![ai_with_tool_call("search")]));
        let client = Client::new(provider);
        let node = ThinkNode::new(client, Vec::new(), 10);
        let cfg = RunnableConfig::default();
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        let out = node.execute(&AgentState::default(), &ctx).await.unwrap();
        assert!(matches!(out.goto, Goto::Node(ref s) if s == "act"));
        assert_eq!(out.update.iterations, 1);
    }

    #[tokio::test]
    async fn ends_when_no_tool_calls() {
        let provider = Arc::new(ScriptedProvider::new(vec![Message::ai("done")]));
        let client = Client::new(provider);
        let node = ThinkNode::new(client, Vec::new(), 10);
        let cfg = RunnableConfig::default();
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        let out = node.execute(&AgentState::default(), &ctx).await.unwrap();
        assert!(matches!(out.goto, Goto::End));
        assert_eq!(out.update.iterations, 1);
    }

    #[tokio::test]
    async fn max_iterations_short_circuits() {
        let provider = Arc::new(ScriptedProvider::new(vec![]));
        let client = Client::new(provider);
        let node = ThinkNode::new(client, Vec::new(), 3);
        let state = AgentState {
            iterations: 3,
            ..Default::default()
        };
        let cfg = RunnableConfig::default();
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        let out = node.execute(&state, &ctx).await.unwrap();
        assert!(matches!(out.goto, Goto::End));
        assert!(out.update.messages[0]
            .content()
            .contains("max_iterations=3"));
    }

    #[tokio::test]
    async fn provider_receives_state_messages() {
        let provider = Arc::new(ScriptedProvider::new(vec![Message::ai("ok")]));
        let client = Client::new(Arc::clone(&provider) as Arc<dyn LLMProvider>);
        let node = ThinkNode::new(client, Vec::new(), 10);
        let state = AgentState {
            messages: vec![Message::human("hello from state")],
            ..Default::default()
        };
        let cfg = RunnableConfig::default();
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        node.execute(&state, &ctx).await.unwrap();
        let calls = provider.received.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0[0].content(), "hello from state");
    }

    fn delta(content: &str) -> StreamChunk {
        StreamChunk {
            content: content.into(),
            is_delta: true,
            is_done: false,
            finish_reason: None,
            usage: None,
            tool_calls_delta: vec![],
        }
    }

    fn done_chunk() -> StreamChunk {
        StreamChunk {
            content: String::new(),
            is_delta: false,
            is_done: true,
            finish_reason: Some("stop".into()),
            usage: None,
            tool_calls_delta: vec![],
        }
    }

    /// Provider whose streaming-with-tools path replays scripted chunk results.
    struct StreamingProvider {
        chunks: std::sync::Mutex<Option<Vec<Result<StreamChunk>>>>,
    }

    impl StreamingProvider {
        fn new(chunks: Vec<Result<StreamChunk>>) -> Self {
            Self {
                chunks: std::sync::Mutex::new(Some(chunks)),
            }
        }
    }

    #[async_trait]
    impl LLMProvider for StreamingProvider {
        fn name(&self) -> &str {
            "sp"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
            unreachable!("streaming ThinkNode must not call chat_completion")
        }
        async fn chat_completion_stream(
            &self,
            _m: Vec<Message>,
            _o: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unreachable!()
        }
        async fn chat_completion_stream_with_tools(
            &self,
            _m: Vec<Message>,
            _t: Vec<ToolDefinition>,
            _o: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            let chunks = self.chunks.lock().unwrap().take().unwrap_or_default();
            Ok(RunnableStream::new(futures::stream::iter(chunks)))
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    struct Rec(Arc<std::sync::Mutex<Vec<String>>>);
    impl cognis_core::Observer for Rec {
        fn on_event(&self, e: &cognis_core::stream::Event) {
            if let cognis_core::stream::Event::OnLlmToken { token, .. } = e {
                self.0.lock().unwrap().push(token.clone());
            }
        }
    }

    #[tokio::test]
    async fn streaming_think_emits_tokens_and_identical_message() {
        let tokens = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider =
            StreamingProvider::new(vec![Ok(delta("Hel")), Ok(delta("lo")), Ok(done_chunk())]);
        let client = Client::new(Arc::new(provider));
        let node = ThinkNode::new(client, Vec::new(), 10).with_streaming(true);
        let cfg = RunnableConfig::default().with_observer(Arc::new(Rec(tokens.clone())));
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        let out = node.execute(&AgentState::default(), &ctx).await.unwrap();

        assert_eq!(
            *tokens.lock().unwrap(),
            vec!["Hel".to_string(), "lo".to_string()]
        );
        assert_eq!(out.update.messages[0].content(), "Hello");
        assert_eq!(out.update.iterations, 1);
        assert!(matches!(out.goto, Goto::End));
    }

    #[tokio::test]
    async fn streaming_think_skips_empty_chunks_when_emitting_tokens() {
        let tokens = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider =
            StreamingProvider::new(vec![Ok(delta("")), Ok(delta("hi")), Ok(done_chunk())]);
        let client = Client::new(Arc::new(provider));
        let node = ThinkNode::new(client, Vec::new(), 10).with_streaming(true);
        let cfg = RunnableConfig::default().with_observer(Arc::new(Rec(tokens.clone())));
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        node.execute(&AgentState::default(), &ctx).await.unwrap();
        assert_eq!(*tokens.lock().unwrap(), vec!["hi".to_string()]);
    }

    #[tokio::test]
    async fn streaming_think_propagates_stream_error() {
        let provider = StreamingProvider::new(vec![
            Ok(delta("par")),
            Err(cognis_core::CognisError::Cancelled),
        ]);
        let client = Client::new(Arc::new(provider));
        let node = ThinkNode::new(client, Vec::new(), 10).with_streaming(true);
        let cfg = RunnableConfig::default();
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        let res = node.execute(&AgentState::default(), &ctx).await;
        assert!(matches!(res, Err(cognis_core::CognisError::Cancelled)));
    }

    fn tool_delta(index: u32, id: Option<&str>, name: Option<&str>, args: &str) -> StreamChunk {
        StreamChunk {
            content: String::new(),
            is_delta: true,
            is_done: false,
            finish_reason: None,
            usage: None,
            tool_calls_delta: vec![cognis_llm::chat::ToolCallDelta {
                index,
                id: id.map(String::from),
                name: name.map(String::from),
                arguments_delta: Some(args.into()),
            }],
        }
    }

    fn tool_calls_done() -> StreamChunk {
        StreamChunk {
            finish_reason: Some("tool_calls".into()),
            ..done_chunk()
        }
    }

    async fn run_streaming(chunks: Vec<Result<StreamChunk>>) -> NodeOut<AgentState> {
        let client = Client::new(Arc::new(StreamingProvider::new(chunks)));
        let node = ThinkNode::new(client, Vec::new(), 10).with_streaming(true);
        let cfg = RunnableConfig::default();
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        node.execute(&AgentState::default(), &ctx).await.unwrap()
    }

    #[tokio::test]
    async fn streaming_think_routes_to_act_when_tool_arguments_split_across_chunks() {
        let out = run_streaming(vec![
            Ok(tool_delta(0, Some("call_1"), Some("weather"), "")),
            Ok(tool_delta(0, None, None, "{\"ci")),
            Ok(tool_delta(0, None, None, "ty\":\"")),
            Ok(tool_delta(0, None, None, "San")),
            Ok(tool_delta(0, None, None, " Fran")),
            Ok(tool_delta(0, None, None, "cisco\"")),
            Ok(tool_delta(0, None, None, ",\"units\":\"c\"}")),
            Ok(tool_calls_done()),
        ])
        .await;

        assert!(
            matches!(out.goto, Goto::Node(ref s) if s == "act"),
            "got: {:?}",
            out.goto
        );
        assert_eq!(
            out.update.messages,
            vec![Message::Ai(AiMessage {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "call_1".into(),
                    name: "weather".into(),
                    arguments: serde_json::json!({"city": "San Francisco", "units": "c"}),
                }],
                parts: Vec::new(),
            })]
        );
    }

    #[tokio::test]
    async fn streaming_think_assembles_parallel_tool_calls_when_deltas_interleave() {
        // Index 1 opens first and the argument fragments alternate, so a
        // merge keyed on arrival order instead of `index` would scramble them.
        let out = run_streaming(vec![
            Ok(tool_delta(1, Some("call_b"), Some("time"), "{\"tz\":")),
            Ok(tool_delta(0, Some("call_a"), Some("weather"), "{\"city\":")),
            Ok(tool_delta(1, None, None, "\"Europe")),
            Ok(tool_delta(0, None, None, "\"Par")),
            Ok(tool_delta(1, None, None, "/Paris\"}")),
            Ok(tool_delta(0, None, None, "is\"}")),
            Ok(tool_calls_done()),
        ])
        .await;

        assert!(matches!(out.goto, Goto::Node(ref s) if s == "act"));
        assert_eq!(
            out.update.messages[0].tool_calls(),
            &[
                ToolCall {
                    id: "call_a".into(),
                    name: "weather".into(),
                    arguments: serde_json::json!({"city": "Paris"}),
                },
                ToolCall {
                    id: "call_b".into(),
                    name: "time".into(),
                    arguments: serde_json::json!({"tz": "Europe/Paris"}),
                },
            ]
        );
    }

    /// The wire shape a streaming provider would produce for `msg`: content
    /// and each tool call's argument JSON cut into small fragments.
    fn stream_chunks_for(msg: &Message) -> Vec<Result<StreamChunk>> {
        fn pieces(s: &str, n: usize) -> Vec<String> {
            let chars: Vec<char> = s.chars().collect();
            chars.chunks(n).map(|c| c.iter().collect()).collect()
        }
        let mut chunks = Vec::new();
        for piece in pieces(msg.content(), 4) {
            chunks.push(Ok(delta(&piece)));
        }
        for (i, tc) in msg.tool_calls().iter().enumerate() {
            let i = i as u32;
            chunks.push(Ok(tool_delta(i, Some(&tc.id), Some(&tc.name), "")));
            for piece in pieces(&tc.arguments.to_string(), 3) {
                chunks.push(Ok(tool_delta(i, None, None, &piece)));
            }
        }
        chunks.push(Ok(tool_calls_done()));
        chunks
    }

    #[tokio::test]
    async fn streaming_and_non_streaming_think_produce_equal_messages() {
        let scripted = Message::Ai(AiMessage {
            content: "Let me check both — un instant ☕.".into(),
            tool_calls: vec![
                ToolCall {
                    id: "call_a".into(),
                    name: "weather".into(),
                    arguments: serde_json::json!({"city": "San Francisco", "days": 3}),
                },
                ToolCall {
                    id: "call_b".into(),
                    name: "time".into(),
                    arguments: serde_json::json!({"tz": "Europe/Paris", "nested": {"a": [1, 2]}}),
                },
            ],
            parts: Vec::new(),
        });

        let streamed = run_streaming(stream_chunks_for(&scripted)).await;

        let provider = Arc::new(ScriptedProvider::new(vec![scripted.clone()]));
        let node = ThinkNode::new(Client::new(provider), Vec::new(), 10);
        let cfg = RunnableConfig::default();
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        let plain = node.execute(&AgentState::default(), &ctx).await.unwrap();

        assert_eq!(streamed.update.messages, plain.update.messages);
        assert_eq!(streamed.update.messages, vec![scripted]);
        assert_eq!(streamed.update.iterations, plain.update.iterations);
        assert!(matches!(streamed.goto, Goto::Node(ref s) if s == "act"));
        assert!(matches!(plain.goto, Goto::Node(ref s) if s == "act"));
    }

    /// Provider whose stream cancels the run's token while yielding the
    /// chunk at `cancel_at`, and counts how many chunks were pulled.
    struct CancellingProvider {
        token: tokio_util::sync::CancellationToken,
        cancel_at: usize,
        pulled: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl LLMProvider for CancellingProvider {
        fn name(&self) -> &str {
            "cancelling"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(&self, _m: Vec<Message>, _o: ChatOptions) -> Result<ChatResponse> {
            unreachable!()
        }
        async fn chat_completion_stream(
            &self,
            _m: Vec<Message>,
            _o: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unreachable!()
        }
        async fn chat_completion_stream_with_tools(
            &self,
            _m: Vec<Message>,
            _t: Vec<ToolDefinition>,
            _o: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            let token = self.token.clone();
            let cancel_at = self.cancel_at;
            let pulled = self.pulled.clone();
            let chunks = ["a", "b", "c", "d"]
                .into_iter()
                .enumerate()
                .map(move |(i, t)| {
                    pulled.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if i == cancel_at {
                        token.cancel();
                    }
                    Ok(delta(t))
                });
            Ok(RunnableStream::new(futures::stream::iter(chunks)))
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    #[tokio::test]
    async fn streaming_think_returns_cancelled_when_token_fires_mid_stream() {
        let token = tokio_util::sync::CancellationToken::new();
        let pulled = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let tokens = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = CancellingProvider {
            token: token.clone(),
            cancel_at: 1,
            pulled: pulled.clone(),
        };
        let node =
            ThinkNode::new(Client::new(Arc::new(provider)), Vec::new(), 10).with_streaming(true);
        let cfg = RunnableConfig::default()
            .with_cancel_token(token)
            .with_observer(Arc::new(Rec(tokens.clone())));
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);

        let res = node.execute(&AgentState::default(), &ctx).await;

        assert!(
            matches!(res, Err(cognis_core::CognisError::Cancelled)),
            "expected Cancelled"
        );
        assert_eq!(
            *tokens.lock().unwrap(),
            vec!["a".to_string()],
            "the chunk that arrived after cancellation must not be emitted"
        );
        assert_eq!(
            pulled.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "the stream must not be drained after cancellation"
        );
    }

    #[tokio::test]
    async fn non_streaming_think_emits_no_tokens() {
        let tokens = Arc::new(std::sync::Mutex::new(Vec::new()));
        let provider = Arc::new(ScriptedProvider::new(vec![Message::ai("done")]));
        let node = ThinkNode::new(Client::new(provider), Vec::new(), 10);
        let cfg = RunnableConfig::default().with_observer(Arc::new(Rec(tokens.clone())));
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        node.execute(&AgentState::default(), &ctx).await.unwrap();
        assert!(tokens.lock().unwrap().is_empty());
    }
}
