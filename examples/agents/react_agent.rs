//! ReAct Agent Example
//!
//! Demonstrates a ReAct (Reasoning + Acting) agent that automatically decides
//! when to call tools based on the user's message. You just pass a natural
//! language request — the agent reasons about which tools to use, calls them,
//! and returns a final answer.
//!
//! Tools:
//! - `get_current_time` — returns the current date and time
//! - `calculator` — evaluates simple arithmetic (add, sub, mul, div)
//!
//! Auto-detects Ollama for real LLM reasoning. Falls back to a fake model
//! with canned tool-calling responses when Ollama is not available.
//!
//! Run with: `cargo run -p cognis-examples --example react_agent`

#[path = "../shared.rs"]
mod shared;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::language_models::FakeMessagesListChatModel;
use cognis_core::messages::tool_types::ToolCall;
use cognis_core::messages::{AIMessage, Message};
use cognis_core::tools::types::{ToolInput, ToolOutput};
use cognis_core::tools::BaseTool;
use cognisgraph::prebuilt::create_react_agent;

// ---------------------------------------------------------------------------
// Tool: get_current_time
// ---------------------------------------------------------------------------

/// Returns the current date and time. The LLM cannot know this on its own,
/// so it *must* call this tool to answer time-related questions.
struct GetCurrentTimeTool;

#[async_trait]
impl BaseTool for GetCurrentTimeTool {
    fn name(&self) -> &str {
        "get_current_time"
    }

    fn description(&self) -> &str {
        "Get the current date and time. No input required. Returns the current timestamp."
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(json!({
            "type": "object",
            "properties": {},
        }))
    }

    async fn _run(&self, _input: ToolInput) -> cognis_core::error::Result<ToolOutput> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap();
        let secs = now.as_secs();

        // Simple date formatting without chrono dependency
        let days = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;

        // Approximate date from epoch (good enough for a demo)
        let year = 1970 + (days / 365);
        let day_of_year = days % 365;
        let month = day_of_year / 30 + 1;
        let day = day_of_year % 30 + 1;

        let timestamp =
            format!("{year}-{month:02}-{day:02} {hours:02}:{minutes:02}:{seconds:02} UTC");

        println!("  [get_current_time] Returning: {timestamp}");
        Ok(ToolOutput::Content(json!({ "current_time": timestamp })))
    }
}

// ---------------------------------------------------------------------------
// Tool: calculator
// ---------------------------------------------------------------------------

/// A calculator tool for arithmetic. Has a proper args_schema so the LLM
/// knows exactly what arguments to pass.
struct CalculatorTool;

#[async_trait]
impl BaseTool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluate simple arithmetic between two numbers."
    }

    fn args_schema(&self) -> Option<serde_json::Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "a": { "type": "number", "description": "First operand" },
                "b": { "type": "number", "description": "Second operand" },
                "op": {
                    "type": "string",
                    "enum": ["add", "sub", "mul", "div"],
                    "description": "Operation to perform"
                }
            },
            "required": ["a", "b", "op"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> cognis_core::error::Result<ToolOutput> {
        let args = match &input {
            ToolInput::Structured(map) => map.clone(),
            ToolInput::ToolCall(tc) => tc.args.clone(),
            ToolInput::Text(s) => serde_json::from_str(s).unwrap_or_default(),
        };

        let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
        let op = args.get("op").and_then(|v| v.as_str()).unwrap_or("add");

        let result = match op {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            "div" if b != 0.0 => a / b,
            "div" => return Ok(ToolOutput::Content(json!("Error: division by zero"))),
            _ => return Ok(ToolOutput::Content(json!(format!("Unknown op: {op}")))),
        };

        println!("  [calculator] {a} {op} {b} = {result}");
        Ok(ToolOutput::Content(json!({ "result": result })))
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ReAct Agent Example ===\n");

    // Step 1: Get a chat model.
    //
    // With Ollama: the agent uses the real LLM to decide tool calls.
    // Without Ollama: uses a fake model that simulates tool-calling behavior.
    let model: Arc<dyn BaseChatModel> = if shared::is_ollama_available() {
        shared::get_chat_model(vec![])
    } else {
        println!("[Ollama not detected — using fake tool-calling model]\n");

        // Simulate the ReAct loop: first response has a tool call, second is final answer
        let mut ai_with_tool = AIMessage::new("");
        ai_with_tool.tool_calls.push(ToolCall {
            name: "get_current_time".to_string(),
            args: HashMap::new(),
            id: Some("call_1".to_string()),
        });

        let ai_final =
            AIMessage::new("The current time is shown above from the get_current_time tool.");

        Arc::new(FakeMessagesListChatModel::new(vec![
            Message::Ai(ai_with_tool),
            Message::Ai(ai_final),
        ]))
    };

    // Step 2: Define the tools.
    let tools: Vec<Arc<dyn BaseTool>> =
        vec![Arc::new(GetCurrentTimeTool), Arc::new(CalculatorTool)];

    println!("Available tools:");
    for tool in &tools {
        println!("  - {} : {}", tool.name(), tool.description());
    }

    // Step 3: Create the ReAct agent.
    //
    // create_react_agent binds the tool schemas to the model and builds a
    // graph that loops: agent -> tools -> agent until no more tool calls.
    let graph = create_react_agent(model, tools)?;
    println!("\nAgent ready (nodes: {:?})\n", graph.node_names());

    // Step 4: Ask a question that requires a tool call.
    //
    // "What time is it?" — the LLM cannot know the current time, so it
    // MUST call the get_current_time tool to answer.
    let question = "What time is it right now?";
    println!("User: {question}\n");

    let input = json!({
        "messages": [
            {"type": "human", "content": question}
        ]
    });

    println!("--- Agent Execution ---");
    let result = graph.invoke(input).await?;
    println!("--- Agent Finished ---\n");

    // Step 5: Print the conversation showing the full ReAct loop.
    let messages = result["messages"]
        .as_array()
        .expect("result should contain messages");

    println!("Conversation ({} messages):\n", messages.len());
    for (i, msg_value) in messages.iter().enumerate() {
        let msg: Message = serde_json::from_value(msg_value.clone())?;
        match &msg {
            Message::Human(_) => {
                println!("  [{}] Human: {}", i + 1, msg.content().text());
            }
            Message::Ai(ai) if !ai.tool_calls.is_empty() => {
                println!("  [{}] AI: (deciding to call tools)", i + 1);
                for tc in &ai.tool_calls {
                    println!("        -> calls {}({:?})", tc.name, tc.args);
                }
            }
            Message::Ai(_) => {
                println!("  [{}] AI: {}", i + 1, msg.content().text());
            }
            Message::Tool(_) => {
                println!("  [{}] Tool Result: {}", i + 1, msg.content().text());
            }
            _ => {
                println!("  [{}] {}", i + 1, msg.content().text());
            }
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
