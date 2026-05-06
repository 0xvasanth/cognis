//! `ToolDispatchNode` — executes tool calls. The agent's bridge between
//! LLM-emitted `tool_calls` and the `Tool` trait registry.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

use cognis2_core::{CognisError, Message, Result};
use cognis2_graph::{Goto, Node, NodeCtx, NodeOut};
use cognis2_llm::{Tool, ToolInput};

use super::state::{AgentState, AgentStateUpdate};

/// A graph node that dispatches the tool calls in the most recent
/// assistant message. After dispatch, returns one `Message::tool(...)`
/// per call and routes back to "think".
pub struct ToolDispatchNode {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolDispatchNode {
    /// Build a dispatcher from a list of tools.
    pub fn new(tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        let map: HashMap<String, Arc<dyn Tool>> = tools
            .into_iter()
            .map(|t| (t.name().to_string(), t))
            .collect();
        Self { tools: map }
    }

    /// Tool count.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// True if no tools registered.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

#[async_trait]
impl Node<AgentState> for ToolDispatchNode {
    async fn execute(&self, state: &AgentState, ctx: &NodeCtx<'_>) -> Result<NodeOut<AgentState>> {
        if ctx.is_cancelled() {
            return Err(cognis2_core::CognisError::Cancelled);
        }
        let last = state.messages.last().ok_or_else(|| {
            CognisError::Internal("ToolDispatchNode invoked but state.messages is empty".into())
        })?;
        let calls = last.tool_calls();
        if calls.is_empty() {
            return Err(CognisError::Internal(
                "ToolDispatchNode invoked but last message has no tool_calls".into(),
            ));
        }

        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            let tool = self.tools.get(&call.name);
            let result_msg = match tool {
                Some(t) => match t._run(ToolInput::ToolCall(call.clone())).await {
                    Ok(out) => Message::tool(&call.id, out.as_string()),
                    Err(e) => Message::tool(&call.id, format!("error: {e}")),
                },
                None => Message::tool(
                    &call.id,
                    format!("error: tool `{}` not registered", call.name),
                ),
            };
            results.push(result_msg);
        }

        Ok(NodeOut {
            update: AgentStateUpdate {
                messages: results,
                iterations: 0,
            },
            goto: Goto::node("think"),
        })
    }

    fn name(&self) -> &str {
        "tools"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;

    use cognis2_core::{AiMessage, RunnableConfig, ToolCall};
    use cognis2_llm::{Tool, ToolOutput};
    use uuid::Uuid;

    /// Mock tool that echoes the args back as Content.
    struct EchoTool;
    #[async_trait]
    impl Tool for EchoTool {
        fn name(&self) -> &str {
            "echo"
        }
        fn description(&self) -> &str {
            "echoes input"
        }
        fn args_schema(&self) -> Option<serde_json::Value> {
            Some(json!({"type": "object"}))
        }
        async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
            Ok(ToolOutput::Content(input.into_json()))
        }
    }

    #[tokio::test]
    async fn dispatches_each_tool_call() {
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(EchoTool)];
        let node = ToolDispatchNode::new(tools);
        let state = AgentState {
            messages: vec![Message::Ai(AiMessage {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "echo".into(),
                    arguments: json!({"x": 42}),
                }],
            })],
            iterations: 0,
            extras: Default::default(),
        };
        let cfg = RunnableConfig::default();
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);

        let out = node.execute(&state, &ctx).await.unwrap();
        assert_eq!(out.update.messages.len(), 1);
        assert!(matches!(out.goto, Goto::Node(ref s) if s == "think"));
        // Tool result should be a Tool message with the call id
        if let Message::Tool(t) = &out.update.messages[0] {
            assert_eq!(t.tool_call_id, "c1");
        } else {
            panic!("expected Tool message");
        }
    }

    #[tokio::test]
    async fn unknown_tool_yields_error_message() {
        let node = ToolDispatchNode::new(Vec::<Arc<dyn Tool>>::new());
        let state = AgentState {
            messages: vec![Message::Ai(AiMessage {
                content: String::new(),
                tool_calls: vec![ToolCall {
                    id: "c1".into(),
                    name: "missing".into(),
                    arguments: json!({}),
                }],
            })],
            iterations: 0,
            extras: Default::default(),
        };
        let cfg = RunnableConfig::default();
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        let out = node.execute(&state, &ctx).await.unwrap();
        if let Message::Tool(t) = &out.update.messages[0] {
            assert!(t.content.contains("not registered"));
        }
    }

    #[tokio::test]
    async fn empty_tool_calls_errors() {
        let node = ToolDispatchNode::new(Vec::<Arc<dyn Tool>>::new());
        let state = AgentState {
            messages: vec![Message::ai("plain text")],
            iterations: 0,
            extras: Default::default(),
        };
        let cfg = RunnableConfig::default();
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        let result = node.execute(&state, &ctx).await;
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(format!("{err}").contains("no tool_calls"));
    }
}
