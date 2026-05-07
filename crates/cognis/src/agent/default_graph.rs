//! Build the standard ReAct graph: think → (if tool calls) act → think → … → end.

use std::sync::Arc;

use cognis_core::Result;
use cognis_graph::{CompiledGraph, Graph};
use cognis_llm::{Client, Tool, ToolDefinition};

use super::state::AgentState;
use super::think_node::ThinkNode;
use super::tool_node::ToolDispatchNode;

/// Build the standard 2-node ReAct graph:
///
/// ```text
/// ┌──────────┐  has tool calls?  ┌──────────┐
/// │  think   │ ─────yes────────> │   act    │
/// │ (LLM)    │                   │ (tools)  │
/// └──────────┘ <──── loop ────── └──────────┘
///       │
///       │ no tool calls / max_iter
///       ▼
///      End
/// ```
pub fn default_react_graph(
    client: Client,
    tools: Vec<Arc<dyn Tool>>,
    max_iterations: u32,
) -> Result<CompiledGraph<AgentState>> {
    default_react_graph_with_limits(client, tools, max_iterations, None)
}

/// Same as [`default_react_graph`] but with an explicit tool-call cap.
pub fn default_react_graph_with_limits(
    client: Client,
    tools: Vec<Arc<dyn Tool>>,
    max_iterations: u32,
    max_tool_calls: Option<u32>,
) -> Result<CompiledGraph<AgentState>> {
    let tool_defs: Vec<ToolDefinition> = tools
        .iter()
        .map(|t| ToolDefinition::from_tool(t.as_ref()))
        .collect();

    let mut think = ThinkNode::new(client, tool_defs, max_iterations);
    if let Some(n) = max_tool_calls {
        think = think.with_max_tool_calls(n);
    }
    let act = ToolDispatchNode::new(tools);

    Graph::<AgentState>::new()
        .node("think", think)
        .node("act", act)
        .start_at("think")
        .compile()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use async_trait::async_trait;
    use cognis_core::{AiMessage, Message, ToolCall};
    use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk, Usage};
    use cognis_llm::provider::{LLMProvider, Provider};
    use cognis_llm::{Tool, ToolInput, ToolOutput};

    struct EchoTool;
    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echo"
        }
        fn args_schema(&self) -> Option<serde_json::Value> {
            Some(serde_json::json!({}))
        }
        async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
            Ok(ToolOutput::Content(input.into_json()))
        }
    }

    /// Two-shot scripted provider: first response has a tool call,
    /// second response is plain.
    struct TwoShot {
        idx: std::sync::atomic::AtomicUsize,
    }
    impl TwoShot {
        fn new() -> Self {
            Self { idx: 0.into() }
        }
    }
    #[async_trait]
    impl LLMProvider for TwoShot {
        fn name(&self) -> &str {
            "two-shot"
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(
            &self,
            messages: Vec<Message>,
            opts: ChatOptions,
        ) -> Result<ChatResponse> {
            let _ = (messages, opts);
            use std::sync::atomic::Ordering;
            let n = self.idx.fetch_add(1, Ordering::SeqCst);
            let message = if n == 0 {
                Message::Ai(AiMessage {
                    content: String::new(),
                    tool_calls: vec![ToolCall {
                        id: "c1".into(),
                        name: "echo".into(),
                        arguments: serde_json::json!({"x": 1}),
                    }],
                    parts: Vec::new(),
                })
            } else {
                Message::ai("done")
            };
            Ok(ChatResponse {
                message,
                usage: Some(Usage::default()),
                finish_reason: "stop".into(),
                model: "ts".into(),
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
    async fn full_react_loop_runs_to_end() {
        use cognis_core::{Runnable, RunnableConfig};
        let client = Client::new(Arc::new(TwoShot::new()));
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(EchoTool)];
        let graph = default_react_graph(client, tools, 10).unwrap();

        let initial = AgentState {
            messages: vec![Message::human("call echo then say done")],
            iterations: 0,
            extras: Default::default(),
        };
        let final_state = graph
            .invoke(initial, RunnableConfig::default())
            .await
            .unwrap();
        // Expected sequence: human, ai(tool_call), tool, ai(done) = 4 messages
        assert_eq!(final_state.messages.len(), 4);
        assert_eq!(final_state.iterations, 2);
        assert_eq!(final_state.messages.last().unwrap().content(), "done");
    }
}
