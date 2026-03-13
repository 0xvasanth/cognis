//! Tool-calling agent factory.
//!
//! Creates an LCEL-style agent that uses a chat model's `bind_tools` capability
//! to decide which tools to call. This mirrors Python's `create_tool_calling_agent`.

use std::collections::HashMap;

use serde_json::{json, Value};

use cognis_core::agents::{AgentAction, AgentFinish};
use cognis_core::error::Result;
use cognis_core::messages::Message;

/// Output from the tool-calling agent: either actions to take or a final answer.
#[derive(Debug, Clone)]
pub enum AgentOutput {
    /// The agent wants to call one or more tools.
    Actions(Vec<AgentAction>),
    /// The agent has produced a final answer.
    Finish(AgentFinish),
}

/// Parses an AI message's tool calls into `AgentOutput`.
///
/// If tool_calls are present, returns `AgentOutput::Actions`.
/// Otherwise, returns `AgentOutput::Finish` with the message content.
pub fn parse_ai_message_to_agent_output(ai_message: &Value) -> Result<AgentOutput> {
    // Try to extract tool_calls from the AI message
    let tool_calls = ai_message
        .get("tool_calls")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    if tool_calls.is_empty() {
        // No tool calls — this is the final answer
        let content = ai_message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let mut return_values = HashMap::new();
        return_values.insert("output".to_string(), Value::String(content));
        return Ok(AgentOutput::Finish(AgentFinish::new(return_values, "")));
    }

    // Parse tool calls into AgentActions
    let actions = tool_calls
        .into_iter()
        .map(|tc| {
            let tool_name = tc
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let tool_input = tc.get("args").cloned().unwrap_or(json!({}));
            let _tool_call_id = tc
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let log = format!(
                "Calling tool `{}` with args: {}",
                tool_name,
                serde_json::to_string(&tool_input).unwrap_or_default()
            );
            AgentAction::new(tool_name, tool_input, log)
        })
        .collect();

    Ok(AgentOutput::Actions(actions))
}

/// Formats intermediate agent steps into tool messages for the agent_scratchpad.
///
/// Each step becomes a pair of messages:
/// 1. An AI message with the tool call
/// 2. A Tool message with the observation
pub fn format_to_tool_messages(intermediate_steps: &[(AgentAction, String)]) -> Vec<Message> {
    let mut messages = Vec::new();
    for (action, observation) in intermediate_steps {
        // Add a tool message for the observation
        messages.push(Message::tool(observation.as_str(), &action.tool));
    }
    messages
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_parse_ai_message_no_tool_calls() {
        let msg = json!({
            "content": "The answer is 42",
            "tool_calls": []
        });
        match parse_ai_message_to_agent_output(&msg).unwrap() {
            AgentOutput::Finish(finish) => {
                assert_eq!(
                    finish.return_values.get("output"),
                    Some(&Value::String("The answer is 42".into()))
                );
            }
            _ => panic!("Expected Finish"),
        }
    }

    #[test]
    fn test_parse_ai_message_with_tool_calls() {
        let msg = json!({
            "content": "",
            "tool_calls": [
                {
                    "name": "search",
                    "args": {"query": "rust lang"},
                    "id": "call_123"
                }
            ]
        });
        match parse_ai_message_to_agent_output(&msg).unwrap() {
            AgentOutput::Actions(actions) => {
                assert_eq!(actions.len(), 1);
                assert_eq!(actions[0].tool, "search");
                assert_eq!(actions[0].tool_input, json!({"query": "rust lang"}));
            }
            _ => panic!("Expected Actions"),
        }
    }

    #[test]
    fn test_parse_ai_message_multiple_tool_calls() {
        let msg = json!({
            "content": "",
            "tool_calls": [
                {"name": "search", "args": {"q": "a"}, "id": "1"},
                {"name": "calc", "args": {"expr": "2+2"}, "id": "2"}
            ]
        });
        match parse_ai_message_to_agent_output(&msg).unwrap() {
            AgentOutput::Actions(actions) => {
                assert_eq!(actions.len(), 2);
                assert_eq!(actions[0].tool, "search");
                assert_eq!(actions[1].tool, "calc");
            }
            _ => panic!("Expected Actions"),
        }
    }

    #[test]
    fn test_format_to_tool_messages() {
        let steps = vec![(
            AgentAction::new("search", json!({"q": "test"}), "Calling search"),
            "Search result: found it".to_string(),
        )];
        let messages = format_to_tool_messages(&steps);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].content().text(), "Search result: found it");
    }
}
