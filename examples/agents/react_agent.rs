//! ReAct Agent Example
//!
//! Demonstrates a ReAct (Reasoning + Acting) agent that automatically decides
//! when to call tools based on the user's message. You just pass a natural
//! language request — the agent reasons about which tools to use, calls them,
//! and returns a final answer.
//!
//! Tools:
//! - `calculator` — evaluates simple arithmetic (add, sub, mul, div)
//! - `reverse` — reverses a string
//! - `uppercase` — converts a string to uppercase
//!
//! Auto-detects Ollama for real LLM reasoning. Falls back to a fake model
//! with canned responses when Ollama is not available.
//!
//! Run with: `cargo run -p cognis-examples --example react_agent`

#[path = "../shared.rs"]
mod shared;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use cognis_core::messages::Message;
use cognis_core::tools::types::{ToolInput, ToolOutput};
use cognis_core::tools::BaseTool;
use cognisgraph::prebuilt::create_react_agent;

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

/// A calculator tool that evaluates simple arithmetic.
struct CalculatorTool;

#[async_trait]
impl BaseTool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Evaluate simple arithmetic. Input: JSON with fields 'a' (number), 'b' (number), 'op' (one of: add, sub, mul, div). Returns the result."
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
            _ => {
                return Ok(ToolOutput::Content(json!(format!(
                    "Unknown operation: {op}"
                ))))
            }
        };

        println!("  [calculator] {a} {op} {b} = {result}");
        Ok(ToolOutput::Content(json!({ "result": result })))
    }
}

/// A tool that reverses a string.
struct ReverseTool;

#[async_trait]
impl BaseTool for ReverseTool {
    fn name(&self) -> &str {
        "reverse"
    }

    fn description(&self) -> &str {
        "Reverse a string. Input: JSON with field 'text' (string). Returns the reversed string."
    }

    async fn _run(&self, input: ToolInput) -> cognis_core::error::Result<ToolOutput> {
        let text = match &input {
            ToolInput::Structured(map) => map
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            ToolInput::ToolCall(tc) => tc
                .args
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            ToolInput::Text(s) => s.clone(),
        };

        let reversed: String = text.chars().rev().collect();
        println!("  [reverse] \"{text}\" -> \"{reversed}\"");
        Ok(ToolOutput::Content(json!(reversed)))
    }
}

/// A tool that converts a string to uppercase.
struct UppercaseTool;

#[async_trait]
impl BaseTool for UppercaseTool {
    fn name(&self) -> &str {
        "uppercase"
    }

    fn description(&self) -> &str {
        "Convert a string to uppercase. Input: JSON with field 'text' (string). Returns the uppercased string."
    }

    async fn _run(&self, input: ToolInput) -> cognis_core::error::Result<ToolOutput> {
        let text = match &input {
            ToolInput::Structured(map) => map
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            ToolInput::ToolCall(tc) => tc
                .args
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            ToolInput::Text(s) => s.clone(),
        };

        let upper = text.to_uppercase();
        println!("  [uppercase] \"{text}\" -> \"{upper}\"");
        Ok(ToolOutput::Content(json!(upper)))
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
    // With Ollama running, the agent will reason about which tools to call.
    // Without Ollama, a fake model provides canned responses.
    let model = shared::get_chat_model(vec!["100 + 200 = 300. The answer is 300.".into()]);

    // Step 2: Define the tools the agent can use.
    let tools: Vec<Arc<dyn BaseTool>> = vec![
        Arc::new(CalculatorTool),
        Arc::new(ReverseTool),
        Arc::new(UppercaseTool),
    ];

    println!("Available tools:");
    for tool in &tools {
        println!("  - {} : {}", tool.name(), tool.description());
    }
    println!();

    // Step 3: Create the ReAct agent.
    //
    // This builds a graph with two nodes:
    //   "agent" — calls the LLM with current messages
    //   "tools" — executes any tool calls the LLM requested
    //
    // The agent loops (agent -> tools -> agent) until the LLM responds
    // without requesting any tool calls.
    let graph = create_react_agent(model, tools)?;
    println!("Agent graph nodes: {:?}\n", graph.node_names());

    // Step 4: Ask the agent a question.
    //
    // Just pass a natural language message — the agent decides which tools
    // to call (if any) and returns a final answer.
    let question = "What is 100 + 200?";
    println!("User: {question}\n");

    let input = json!({
        "messages": [
            {"type": "human", "content": question}
        ]
    });

    println!("--- Agent Execution ---");
    let result = graph.invoke(input).await?;
    println!("--- Agent Finished ---\n");

    // Step 5: Print the conversation.
    let messages = result["messages"]
        .as_array()
        .expect("result should contain messages array");

    println!("Conversation ({} messages):\n", messages.len());
    for (i, msg_value) in messages.iter().enumerate() {
        let msg: Message = serde_json::from_value(msg_value.clone())?;
        let role = match &msg {
            Message::Human(_) => "Human",
            Message::Ai(ai) if !ai.tool_calls.is_empty() => "AI (tool call)",
            Message::Ai(_) => "AI",
            Message::Tool(_) => "Tool Result",
            Message::System(_) => "System",
            _ => "Other",
        };
        println!("  [{}] {}: {}", i + 1, role, msg.content().text());

        if let Message::Ai(ai) = &msg {
            for tc in &ai.tool_calls {
                println!("        -> calls {}({:?})", tc.name, tc.args);
            }
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
