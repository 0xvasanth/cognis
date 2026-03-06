//! Prebuilt ReAct (Reasoning + Acting) agent.
//!
//! The ReAct agent alternates between calling a chat model and executing
//! tool calls until the model produces a response with no tool calls.
//!
//! The graph has two nodes:
//! - **"agent"** — invokes the chat model with the current messages
//! - **"tools"** — executes any tool calls from the last AI message
//!
//! And the following edges:
//! - `START` -> `"agent"`
//! - `"agent"` -> conditional: if tool calls present -> `"tools"`, else -> `END`
//! - `"tools"` -> `"agent"` (loop back)

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};

use rustchain_core::language_models::chat_model::BaseChatModel;
use rustchain_core::messages::{Message, ToolCall, ToolMessage};
use rustchain_core::tools::BaseTool;

use crate::constants::END;
use crate::errors::LangGraphError;
use crate::graph::branch::RouterResult;
use crate::graph::state::{AsyncNodeAction, CompiledStateGraph, StateGraph};

/// Create a prebuilt ReAct agent graph.
///
/// The agent alternates between calling the model and executing tools
/// until the model stops requesting tool calls.
///
/// # State format
///
/// The state is a JSON object with a `"messages"` key containing an array
/// of serialized [`Message`] values.
///
/// # Arguments
///
/// * `model` - A chat model that supports tool calling.
/// * `tools` - A list of tools available to the agent.
///
/// # Returns
///
/// A compiled state graph ready for invocation.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use serde_json::json;
/// use langgraph::prebuilt::create_react_agent;
///
/// // Assuming you have a model and tools:
/// // let graph = create_react_agent(model, vec![tool]).unwrap();
/// // let result = graph.invoke(json!({"messages": [{"type": "human", "content": "Hello"}]})).await.unwrap();
/// ```
pub fn create_react_agent(
    model: Arc<dyn BaseChatModel>,
    tools: Vec<Arc<dyn BaseTool>>,
) -> Result<CompiledStateGraph, LangGraphError> {
    // Build the agent node: calls the model with current messages.
    let agent_model = model.clone();
    let agent_node: AsyncNodeAction = Arc::new(move |state: Value| {
        let model = agent_model.clone();
        Box::pin(async move {
            let messages = extract_messages(&state)?;
            let result = model
                ._generate(&messages, None)
                .await
                .map_err(|e| LangGraphError::Other(format!("Model error: {e}")))?;

            let generation = result
                .generations
                .into_iter()
                .next()
                .ok_or_else(|| LangGraphError::Other("No generations returned".into()))?;

            // Append the AI message to the messages array.
            let ai_msg_value = serde_json::to_value(&generation.message)
                .map_err(|e| LangGraphError::Other(format!("Serialization error: {e}")))?;

            let mut msgs = state
                .get("messages")
                .cloned()
                .and_then(|v| v.as_array().cloned())
                .unwrap_or_default();
            msgs.push(ai_msg_value);

            Ok(json!({ "messages": msgs }))
        })
    });

    // Build the tools node: executes tool calls from the last AI message.
    let tools_map: HashMap<String, Arc<dyn BaseTool>> = tools
        .iter()
        .map(|t| (t.name().to_string(), t.clone()))
        .collect();
    let tools_node: AsyncNodeAction = Arc::new(move |state: Value| {
        let tools_map = tools_map.clone();
        Box::pin(async move {
            let messages = extract_messages(&state)?;

            // Get the last AI message and its tool calls.
            let tool_calls = get_last_ai_tool_calls(&messages)?;

            let mut msgs = state
                .get("messages")
                .cloned()
                .and_then(|v| v.as_array().cloned())
                .unwrap_or_default();

            // Execute each tool call.
            for tc in &tool_calls {
                let tool = tools_map.get(&tc.name).ok_or_else(|| {
                    LangGraphError::Other(format!("Tool '{}' not found", tc.name))
                })?;

                let input = rustchain_core::tools::types::ToolInput::Structured(tc.args.clone());
                let tool_call_id = tc.id.clone().unwrap_or_default();

                let result = match tool.run(input, Some(&tool_call_id)).await {
                    Ok(v) => v.to_string(),
                    Err(e) => format!("Error: {e}"),
                };

                let tool_msg = Message::Tool(ToolMessage::new(&result, &tool_call_id));
                let tool_msg_value = serde_json::to_value(&tool_msg)
                    .map_err(|e| LangGraphError::Other(format!("Serialization error: {e}")))?;
                msgs.push(tool_msg_value);
            }

            Ok(json!({ "messages": msgs }))
        })
    });

    // Router: check if the last message has tool calls.
    let should_continue = Arc::new(|state: &Value| -> RouterResult {
        let messages = state
            .get("messages")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        if let Some(last) = messages.last() {
            // Check if the last message is an AI message with tool_calls.
            if let Ok(Message::Ai(ai)) = serde_json::from_value::<Message>(last.clone()).as_ref() {
                if !ai.tool_calls.is_empty() {
                    return RouterResult::Single("tools".to_string());
                }
            }
        }
        RouterResult::Single(END.to_string())
    });

    // Build the graph.
    StateGraph::new()
        .add_node("agent", agent_node)
        .add_node("tools", tools_node)
        .set_entry_point("agent")
        .add_conditional_edges("agent", should_continue, None)
        .add_edge("tools", "agent")
        .compile()
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
    use rustchain_core::language_models::fake::FakeMessagesListChatModel;
    use rustchain_core::messages::AIMessage;
    use rustchain_core::tools::types::{ToolInput, ToolOutput};
    use serde_json::json;
    use std::collections::HashMap;

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

        async fn _run(&self, _input: ToolInput) -> rustchain_core::error::Result<ToolOutput> {
            Ok(ToolOutput::Content(Value::String(self.result.clone())))
        }
    }

    fn make_ai_with_tool_calls(content: &str, tool_calls: Vec<ToolCall>) -> Message {
        let mut ai = AIMessage::new(content);
        ai.tool_calls = tool_calls;
        Message::Ai(ai)
    }

    #[test]
    fn test_create_react_agent_compiles() {
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("hello"),
        )]));
        let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("test_tool", "result"));
        let graph = create_react_agent(model, vec![tool]);
        assert!(graph.is_ok());
        let graph = graph.unwrap();
        let mut names = graph.node_names();
        names.sort();
        assert_eq!(names, vec!["agent", "tools"]);
    }

    #[tokio::test]
    async fn test_react_agent_simple_response() {
        // Model returns a plain text response (no tool calls) -> agent finishes immediately.
        let model = Arc::new(FakeMessagesListChatModel::new(vec![Message::Ai(
            AIMessage::new("The answer is 42"),
        )]));
        let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("calculator", "42"));
        let graph = create_react_agent(model, vec![tool]).unwrap();

        let input = json!({
            "messages": [
                {"type": "human", "content": "What is the meaning of life?"}
            ]
        });

        let result = graph.invoke(input).await.unwrap();
        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 2); // human + ai
                                       // Last message should be the AI response.
        let last: Message = serde_json::from_value(messages.last().unwrap().clone()).unwrap();
        assert_eq!(last.content().text(), "The answer is 42");
    }

    #[tokio::test]
    async fn test_react_agent_with_tool_call() {
        // First call: model returns tool call. Second call: model returns plain text.
        let tc = ToolCall {
            name: "calculator".to_string(),
            args: {
                let mut m = HashMap::new();
                m.insert("expression".to_string(), json!("6*7"));
                m
            },
            id: Some("call_1".to_string()),
        };
        let ai_with_tc = make_ai_with_tool_calls("", vec![tc]);
        let ai_final = Message::Ai(AIMessage::new("The result is 42"));

        let model = Arc::new(FakeMessagesListChatModel::new(vec![ai_with_tc, ai_final]));
        let tool: Arc<dyn BaseTool> = Arc::new(MockTool::new("calculator", "42"));
        let graph = create_react_agent(model, vec![tool]).unwrap();

        let input = json!({
            "messages": [
                {"type": "human", "content": "What is 6*7?"}
            ]
        });

        let result = graph.invoke(input).await.unwrap();
        let messages = result["messages"].as_array().unwrap();
        // human + ai(tool_call) + tool_result + ai(final)
        assert_eq!(messages.len(), 4);

        // Check the tool message
        let tool_msg: Message = serde_json::from_value(messages[2].clone()).unwrap();
        assert!(matches!(tool_msg, Message::Tool(_)));

        // Check the final AI message
        let final_msg: Message = serde_json::from_value(messages[3].clone()).unwrap();
        assert_eq!(final_msg.content().text(), "The result is 42");
    }

    #[tokio::test]
    async fn test_react_agent_multiple_tools() {
        // Model returns multiple tool calls in one response.
        let tc1 = ToolCall {
            name: "add".to_string(),
            args: {
                let mut m = HashMap::new();
                m.insert("a".to_string(), json!(1));
                m.insert("b".to_string(), json!(2));
                m
            },
            id: Some("call_1".to_string()),
        };
        let tc2 = ToolCall {
            name: "multiply".to_string(),
            args: {
                let mut m = HashMap::new();
                m.insert("a".to_string(), json!(3));
                m.insert("b".to_string(), json!(4));
                m
            },
            id: Some("call_2".to_string()),
        };
        let ai_with_tcs = make_ai_with_tool_calls("", vec![tc1, tc2]);
        let ai_final = Message::Ai(AIMessage::new("3 and 12"));

        let model = Arc::new(FakeMessagesListChatModel::new(vec![ai_with_tcs, ai_final]));
        let tools: Vec<Arc<dyn BaseTool>> = vec![
            Arc::new(MockTool::new("add", "3")),
            Arc::new(MockTool::new("multiply", "12")),
        ];
        let graph = create_react_agent(model, tools).unwrap();

        let input = json!({
            "messages": [
                {"type": "human", "content": "Add 1+2 and multiply 3*4"}
            ]
        });

        let result = graph.invoke(input).await.unwrap();
        let messages = result["messages"].as_array().unwrap();
        // human + ai(2 tool_calls) + tool_result_1 + tool_result_2 + ai(final)
        assert_eq!(messages.len(), 5);

        // Both tool messages should be present.
        let tool_msg1: Message = serde_json::from_value(messages[2].clone()).unwrap();
        let tool_msg2: Message = serde_json::from_value(messages[3].clone()).unwrap();
        assert!(matches!(tool_msg1, Message::Tool(_)));
        assert!(matches!(tool_msg2, Message::Tool(_)));

        let final_msg: Message = serde_json::from_value(messages[4].clone()).unwrap();
        assert_eq!(final_msg.content().text(), "3 and 12");
    }
}
