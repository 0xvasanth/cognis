//! LangGraph ReAct Agent Example
//!
//! Demonstrates how to create a ReAct (Reasoning + Acting) agent using
//! LangGraph's create_react_agent. The agent alternates between calling a
//! chat model and executing tools until done.
//!
//! No API keys required -- auto-detects Ollama, falls back to fake model.

mod shared;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use cognisgraph::prebuilt::create_react_agent;
use cognis_core::messages::Message;
use cognis_core::tools::types::{ToolInput, ToolOutput};
use cognis_core::tools::BaseTool;

/// A simple lookup tool that returns a hardcoded answer.
///
/// In a real application, this could query a database, call an API,
/// or perform any computation.
struct LookupTool;

#[async_trait]
impl BaseTool for LookupTool {
    fn name(&self) -> &str {
        "lookup"
    }

    fn description(&self) -> &str {
        "Look up information about a topic"
    }

    async fn _run(&self, input: ToolInput) -> cognis_core::error::Result<ToolOutput> {
        // Extract the query from the tool input.
        let query = match &input {
            ToolInput::Text(s) => s.clone(),
            ToolInput::Structured(map) => map
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
            ToolInput::ToolCall(tc) => tc
                .args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string(),
        };
        println!("  [LookupTool] Looking up: {query}");
        Ok(ToolOutput::Content(serde_json::Value::String(format!(
            "The capital of France is Paris. It has a population of about 2.1 million."
        ))))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LangGraph ReAct Agent Example ===\n");

    // Step 1: Create the chat model.
    //
    // Auto-detects Ollama; falls back to a fake model with a canned response.
    let model = shared::get_chat_model(vec![
        "The capital of France is Paris, with a population of about 2.1 million people.".into(),
    ]);

    // Step 2: Create the tools.
    let tool: Arc<dyn BaseTool> = Arc::new(LookupTool);
    println!("Tool registered: {} - {}", tool.name(), tool.description());

    // Step 3: Create the ReAct agent graph.
    //
    // create_react_agent builds a StateGraph with two nodes:
    //   - "agent": calls the model with the current messages
    //   - "tools": executes any tool calls from the last AI message
    //
    // Edges:
    //   START -> agent
    //   agent -> conditional: if tool calls -> tools, else -> END
    //   tools -> agent (loop back)
    let graph = create_react_agent(model, vec![tool])?;

    let node_names = graph.node_names();
    println!("Graph nodes: {:?}\n", node_names);

    // Step 4: Invoke the graph with a user message.
    //
    // The state is a JSON object with a "messages" key.
    let input = json!({
        "messages": [
            {"type": "human", "content": "What is the capital of France?"}
        ]
    });

    println!("User: What is the capital of France?\n");
    println!("--- Agent Execution ---");

    let result = graph.invoke(input).await?;

    println!("--- Agent Finished ---\n");

    // Step 5: Print the resulting messages.
    let messages = result["messages"]
        .as_array()
        .expect("result should contain messages array");

    println!("=== Conversation ({} messages) ===\n", messages.len());

    for (i, msg_value) in messages.iter().enumerate() {
        let msg: Message = serde_json::from_value(msg_value.clone())?;
        let role = match &msg {
            Message::Human(_) => "Human",
            Message::Ai(ai) => {
                if ai.tool_calls.is_empty() {
                    "AI"
                } else {
                    "AI (tool call)"
                }
            }
            Message::Tool(_) => "Tool Result",
            Message::System(_) => "System",
            _ => "Other",
        };
        println!("  [{}] {}: {}", i + 1, role, msg.content().text());

        // Show tool call details if present.
        if let Message::Ai(ai) = &msg {
            for tc in &ai.tool_calls {
                println!("        -> calls {}({:?})", tc.name, tc.args);
            }
        }
    }

    println!("\nDone!");
    Ok(())
}
