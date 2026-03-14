//! LangGraph ReAct Agent Example
//!
//! Creates a ReAct agent using create_react_agent with a LookupTool.
//! The agent loops between calling the model and executing tools until done.

#[path = "../shared.rs"]
mod shared;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::json;

use cognis_core::messages::Message;
use cognis_core::tools::types::{ToolInput, ToolOutput};
use cognis_core::tools::BaseTool;
use cognisgraph::prebuilt::create_react_agent;

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
        let query = match &input {
            ToolInput::Text(s) => s.clone(),
            ToolInput::Structured(map) => map
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .into(),
            ToolInput::ToolCall(tc) => tc
                .args
                .get("query")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .into(),
        };
        println!("  [LookupTool] query: {query}");
        Ok(ToolOutput::Content(serde_json::Value::String(
            "The capital of France is Paris, population ~2.1 million.".into(),
        )))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LangGraph ReAct Agent ===\n");

    let model = shared::get_chat_model(vec![
        "The capital of France is Paris, with about 2.1 million people.".into(),
    ]);

    let tool: Arc<dyn BaseTool> = Arc::new(LookupTool);
    let graph = create_react_agent(model, vec![tool])?;
    println!("Graph nodes: {:?}\n", graph.node_names());

    let result = graph
        .invoke(json!({
            "messages": [{"type": "human", "content": "What is the capital of France?"}]
        }))
        .await?;

    let messages = result["messages"].as_array().expect("messages array");
    println!("Conversation ({} messages):\n", messages.len());
    for (i, val) in messages.iter().enumerate() {
        let msg: Message = serde_json::from_value(val.clone())?;
        let role = match &msg {
            Message::Human(_) => "Human",
            Message::Ai(ai) if ai.tool_calls.is_empty() => "AI",
            Message::Ai(_) => "AI (tool call)",
            Message::Tool(_) => "Tool",
            _ => "Other",
        };
        println!("  [{}] {}: {}", i + 1, role, msg.content().text());
        if let Message::Ai(ai) = &msg {
            for tc in &ai.tool_calls {
                println!("       -> {}({:?})", tc.name, tc.args);
            }
        }
    }

    Ok(())
}
