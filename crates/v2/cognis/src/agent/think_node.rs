//! `ThinkNode` — calls the LLM and decides whether to invoke tools or end.

use async_trait::async_trait;

use cognis2_core::{Message, Result};
use cognis2_graph::{Goto, Node, NodeCtx, NodeOut};
use cognis2_llm::{ChatOptions, Client, ToolDefinition};

use super::state::{AgentState, AgentStateUpdate};

/// LLM call node. If the LLM emits tool calls, routes to "act"; otherwise
/// terminates the graph.
pub struct ThinkNode {
    client: Client,
    tool_defs: Vec<ToolDefinition>,
    max_iterations: u32,
}

impl ThinkNode {
    /// New `ThinkNode` from a Client + tool definitions + max_iterations.
    pub fn new(client: Client, tool_defs: Vec<ToolDefinition>, max_iterations: u32) -> Self {
        Self {
            client,
            tool_defs,
            max_iterations,
        }
    }
}

#[async_trait]
impl Node<AgentState> for ThinkNode {
    async fn execute(&self, state: &AgentState, ctx: &NodeCtx<'_>) -> Result<NodeOut<AgentState>> {
        if ctx.is_cancelled() {
            return Err(cognis2_core::CognisError::Cancelled);
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

        let messages = state.messages.clone();
        let resp = self
            .client
            .provider()
            .chat_completion_with_tools(messages, self.tool_defs.clone(), ChatOptions::default())
            .await?;
        let msg = resp.message;
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
    use cognis2_core::{AiMessage, RunnableConfig, RunnableStream, ToolCall};
    use cognis2_llm::chat::{ChatResponse, HealthStatus, StreamChunk, Usage};
    use cognis2_llm::provider::{LLMProvider, Provider};
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
}
