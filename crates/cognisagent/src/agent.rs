//! Deep Agent factory — builds a LangGraph `CompiledStateGraph` with middleware hooks.
//!
//! [`create_deep_agent`] is the primary entry point. It wraps `create_react_agent`
//! with middleware hooks around model and tool invocations.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};
use thiserror::Error;

use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::{Message, ToolCall, ToolMessage};
use cognis_core::tools::BaseTool;

use cognisgraph::constants::END;
use cognisgraph::errors::LangGraphError;
use cognisgraph::graph::branch::RouterResult;
use cognisgraph::graph::state::{AsyncNodeAction, CompiledStateGraph, StateGraph};

use crate::config::DeepAgentConfig;
use crate::middleware::Middleware;

/// Errors specific to the Deep Agent system.
#[derive(Debug, Error)]
pub enum DeepAgentError {
    /// An error originating from the underlying graph execution.
    #[error("graph error: {0}")]
    GraphError(#[from] LangGraphError),

    /// An error from the persistence backend.
    #[error("backend error: {0}")]
    BackendError(String),

    /// An error from middleware execution.
    #[error("middleware error: {0}")]
    MiddlewareError(String),

    /// A configuration error.
    #[error("config error: {0}")]
    ConfigError(String),

    /// A generic error.
    #[error("{0}")]
    Other(String),
}

/// Build a Deep Agent from the given configuration.
///
/// This constructs a LangGraph [`CompiledStateGraph`] with two nodes:
///
/// - **"agent"** — invokes the chat model with middleware `before_model`/`after_model` hooks
/// - **"tools"** — executes tool calls with middleware `before_tool`/`after_tool` hooks
///
/// The graph loops between the agent and tools nodes until the model
/// produces a response without tool calls, at which point it terminates.
///
/// # Arguments
///
/// * `model` — The chat model to use.
/// * `config` — A [`DeepAgentConfig`] specifying tools, middleware, etc.
///
/// # Returns
///
/// A compiled state graph ready for invocation.
pub fn create_deep_agent(
    model: Arc<dyn BaseChatModel>,
    config: DeepAgentConfig,
) -> Result<CompiledStateGraph, DeepAgentError> {
    let middleware: Arc<Vec<Arc<dyn Middleware>>> = Arc::new(config.middleware);

    // Collect tools: user-supplied + middleware-contributed tools.
    let mut all_tools = config.tools;
    // (Middleware tools would be added here in a full implementation.)

    let system_prompt = config.system_prompt.clone();

    // -----------------------------------------------------------------------
    // Agent node
    // -----------------------------------------------------------------------
    let agent_model = model.clone();
    let agent_mw = middleware.clone();
    let agent_system_prompt = system_prompt.clone();

    let agent_node: AsyncNodeAction = Arc::new(move |mut state: Value| {
        let model = agent_model.clone();
        let mw = agent_mw.clone();
        let sys_prompt = agent_system_prompt.clone();

        Box::pin(async move {
            // Run before_model middleware.
            for m in mw.iter() {
                m.before_model(&mut state).await.map_err(|e| {
                    LangGraphError::Other(format!("Middleware '{}' before_model: {e}", m.name()))
                })?;
            }

            let mut messages = extract_messages(&state)?;

            // Inject system prompt if configured and not already present.
            if let Some(ref prompt) = sys_prompt {
                let has_system = messages
                    .first()
                    .map(|m| matches!(m, Message::System(_)))
                    .unwrap_or(false);
                if !has_system {
                    messages.insert(
                        0,
                        Message::System(cognis_core::messages::SystemMessage::new(prompt)),
                    );
                }
            }

            let result = model
                ._generate(&messages, None)
                .await
                .map_err(|e| LangGraphError::Other(format!("Model error: {e}")))?;

            let generation = result
                .generations
                .into_iter()
                .next()
                .ok_or_else(|| LangGraphError::Other("No generations returned".into()))?;

            let ai_msg_value = serde_json::to_value(&generation.message)
                .map_err(|e| LangGraphError::Other(format!("Serialization error: {e}")))?;

            let mut msgs = state
                .get("messages")
                .cloned()
                .and_then(|v| v.as_array().cloned())
                .unwrap_or_default();
            msgs.push(ai_msg_value);

            state["messages"] = Value::Array(msgs);

            // Run after_model middleware.
            for m in mw.iter() {
                m.after_model(&mut state).await.map_err(|e| {
                    LangGraphError::Other(format!("Middleware '{}' after_model: {e}", m.name()))
                })?;
            }

            Ok(state)
        })
    });

    // -----------------------------------------------------------------------
    // Tools node
    // -----------------------------------------------------------------------
    let tools_map: HashMap<String, Arc<dyn BaseTool>> = all_tools
        .drain(..)
        .map(|t| (t.name().to_string(), t))
        .collect();

    let tools_mw = middleware.clone();
    let tools_node: AsyncNodeAction = Arc::new(move |mut state: Value| {
        let tools_map = tools_map.clone();
        let mw = tools_mw.clone();

        Box::pin(async move {
            let messages = extract_messages(&state)?;
            let tool_calls = get_last_ai_tool_calls(&messages)?;

            let mut msgs = state
                .get("messages")
                .cloned()
                .and_then(|v| v.as_array().cloned())
                .unwrap_or_default();

            for tc in &tool_calls {
                let tool = tools_map.get(&tc.name).ok_or_else(|| {
                    LangGraphError::Other(format!("Tool '{}' not found", tc.name))
                })?;

                // before_tool middleware
                for m in mw.iter() {
                    m.before_tool(&mut state, &tc.name).await.map_err(|e| {
                        LangGraphError::Other(format!("Middleware '{}' before_tool: {e}", m.name()))
                    })?;
                }

                let input = cognis_core::tools::types::ToolInput::Structured(tc.args.clone());
                let tool_call_id = tc.id.clone().unwrap_or_default();

                let result = match tool.run(input, Some(&tool_call_id)).await {
                    Ok(v) => v.to_string(),
                    Err(e) => format!("Error: {e}"),
                };

                // after_tool middleware
                for m in mw.iter() {
                    m.after_tool(&mut state, &tc.name, &result)
                        .await
                        .map_err(|e| {
                            LangGraphError::Other(format!(
                                "Middleware '{}' after_tool: {e}",
                                m.name()
                            ))
                        })?;
                }

                let tool_msg = Message::Tool(ToolMessage::new(&result, &tool_call_id));
                let tool_msg_value = serde_json::to_value(&tool_msg)
                    .map_err(|e| LangGraphError::Other(format!("Serialization error: {e}")))?;
                msgs.push(tool_msg_value);
            }

            Ok(json!({ "messages": msgs }))
        })
    });

    // -----------------------------------------------------------------------
    // Router
    // -----------------------------------------------------------------------
    let should_continue = Arc::new(|state: &Value| -> RouterResult {
        let messages = state
            .get("messages")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        if let Some(last) = messages.last() {
            if let Ok(Message::Ai(ai)) = serde_json::from_value::<Message>(last.clone()).as_ref() {
                if !ai.tool_calls.is_empty() {
                    return RouterResult::Single("tools".to_string());
                }
            }
        }
        RouterResult::Single(END.to_string())
    });

    // -----------------------------------------------------------------------
    // Build graph
    // -----------------------------------------------------------------------
    let graph = StateGraph::new()
        .add_node("agent", agent_node)
        .add_node("tools", tools_node)
        .set_entry_point("agent")
        .add_conditional_edges("agent", should_continue, None)
        .add_edge("tools", "agent")
        .compile()
        .map_err(DeepAgentError::GraphError)?;

    Ok(graph)
}

/// Extract messages from the state JSON.
fn extract_messages(state: &Value) -> Result<Vec<Message>, LangGraphError> {
    let msgs_value = state
        .get("messages")
        .ok_or_else(|| LangGraphError::Other("State missing 'messages' key".into()))?;

    serde_json::from_value(msgs_value.clone())
        .map_err(|e| LangGraphError::Other(format!("Failed to deserialize messages: {e}")))
}

/// Get tool calls from the last AI message.
fn get_last_ai_tool_calls(messages: &[Message]) -> Result<Vec<ToolCall>, LangGraphError> {
    for msg in messages.iter().rev() {
        if let Message::Ai(ai) = msg {
            return Ok(ai.tool_calls.clone());
        }
    }
    Err(LangGraphError::Other(
        "No AI message found in messages".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::filesystem::FilesystemBackend;
    use crate::backends::state::StateBackend;
    use crate::backends::Backend;
    use crate::middleware::Middleware;

    use cognis_core::language_models::fake::FakeMessagesListChatModel;
    use cognis_core::messages::AIMessage;
    use cognis_core::tools::types::{ToolInput, ToolOutput};
    use serde_json::json;

    /// A simple mock tool for testing.
    struct MockTool {
        name: String,
        result: String,
    }

    impl MockTool {
        fn new(name: &str, result: &str) -> Self {
            Self {
                name: name.to_string(),
                result: result.to_string(),
            }
        }
    }

    #[async_trait::async_trait]
    impl BaseTool for MockTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn description(&self) -> &str {
            "A mock tool for testing"
        }

        async fn _run(&self, _input: ToolInput) -> cognis_core::error::Result<ToolOutput> {
            Ok(ToolOutput::Content(Value::String(self.result.clone())))
        }
    }

    /// A no-op middleware for testing default implementations.
    struct NoopMiddleware;

    #[async_trait::async_trait]
    impl Middleware for NoopMiddleware {
        fn name(&self) -> &str {
            "noop"
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: create_deep_agent compiles successfully
    // -----------------------------------------------------------------------
    #[test]
    fn test_create_deep_agent_compiles() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("hello"),
        )]));
        let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("test_tool", "result"));
        let config = DeepAgentConfig {
            tools: vec![tool],
            ..Default::default()
        };

        let graph = create_deep_agent(model, config);
        assert!(graph.is_ok());
        let graph = graph.unwrap();
        let mut names = graph.node_names();
        names.sort();
        assert_eq!(names, vec!["agent", "tools"]);
    }

    // -----------------------------------------------------------------------
    // Test 2: Deep agent runs to completion without tool calls
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_deep_agent_simple_response() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("The answer is 42"),
        )]));
        let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("calculator", "42"));
        let config = DeepAgentConfig {
            tools: vec![tool],
            ..Default::default()
        };

        let graph = create_deep_agent(model, config).unwrap();
        let input = json!({
            "messages": [{"type": "human", "content": "What is the meaning of life?"}]
        });

        let result = graph.invoke(input).await.unwrap();
        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2);
        let last: Message = serde_json::from_value(messages.last().unwrap().clone()).unwrap();
        assert_eq!(last.content().text(), "The answer is 42");
    }

    // -----------------------------------------------------------------------
    // Test 3: StateBackend save/load round-trip
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_state_backend_save_load() {
        let backend = StateBackend::new();

        let state = json!({"messages": [{"type": "human", "content": "hello"}]});
        backend.save_state("session-1", &state).await.unwrap();

        let loaded = backend.load_state("session-1").await.unwrap();
        assert_eq!(loaded, Some(state));

        let missing = backend.load_state("nonexistent").await.unwrap();
        assert!(missing.is_none());

        let sessions = backend.list_sessions().await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert!(sessions.contains(&"session-1".to_string()));
    }

    // -----------------------------------------------------------------------
    // Test 4: FilesystemBackend save/load round-trip
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_filesystem_backend_save_load() {
        let tmpdir = tempfile::tempdir().unwrap();
        let backend = FilesystemBackend::new(tmpdir.path());

        let state = json!({"messages": [{"type": "human", "content": "hello"}], "count": 5});
        backend.save_state("session-abc", &state).await.unwrap();

        let loaded = backend.load_state("session-abc").await.unwrap();
        assert_eq!(loaded, Some(state));

        let missing = backend.load_state("no-such-session").await.unwrap();
        assert!(missing.is_none());

        let sessions = backend.list_sessions().await.unwrap();
        assert_eq!(sessions, vec!["session-abc".to_string()]);
    }

    // -----------------------------------------------------------------------
    // Test 5: Middleware default implementations are no-ops
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_middleware_defaults_are_noops() {
        let mw = NoopMiddleware;
        assert_eq!(mw.name(), "noop");

        let mut state = json!({"messages": []});
        assert!(mw.before_model(&mut state).await.is_ok());
        assert!(mw.after_model(&mut state).await.is_ok());
        assert!(mw.before_tool(&mut state, "some_tool").await.is_ok());
        assert!(mw
            .after_tool(&mut state, "some_tool", "result")
            .await
            .is_ok());
    }

    // -----------------------------------------------------------------------
    // Test 6: Deep agent with tool call loop
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_deep_agent_with_tool_call() {
        let tc = ToolCall {
            name: "calculator".to_string(),
            args: {
                let mut m = HashMap::new();
                m.insert("expression".to_string(), json!("6*7"));
                m
            },
            id: Some("call_1".to_string()),
        };
        let mut ai_with_tc = AIMessage::new("");
        ai_with_tc.tool_calls = vec![tc];
        let ai_final = AIMessage::new("The result is 42");

        let model = Arc::new(FakeMessagesListChatModel::new(vec![
            Message::Ai(ai_with_tc),
            Message::Ai(ai_final),
        ]));
        let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("calculator", "42"));
        let config = DeepAgentConfig {
            tools: vec![tool],
            ..Default::default()
        };

        let graph = create_deep_agent(model, config).unwrap();
        let input = json!({
            "messages": [{"type": "human", "content": "What is 6*7?"}]
        });

        let result = graph.invoke(input).await.unwrap();
        let messages = result["messages"].as_array().unwrap();
        // human + ai(tool_call) + tool_result + ai(final) = 4
        assert_eq!(messages.len(), 4);

        let final_msg: Message = serde_json::from_value(messages[3].clone()).unwrap();
        assert_eq!(final_msg.content().text(), "The result is 42");
    }

    // -----------------------------------------------------------------------
    // Test 7: MemoryMiddleware stores and retrieves
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_memory_middleware_remember_recall() {
        use crate::middleware::memory::MemoryMiddleware;

        let mw = MemoryMiddleware::new(10);
        mw.remember("user_name", "Alice").await;
        mw.remember("preference", "dark mode").await;

        assert_eq!(mw.recall("user_name").await, Some("Alice".to_string()));
        assert_eq!(mw.recall("unknown").await, None);

        let keys = mw.keys().await;
        assert_eq!(keys.len(), 2);

        mw.clear().await;
        assert_eq!(mw.keys().await.len(), 0);
    }
}
