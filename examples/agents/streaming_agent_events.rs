//! Streaming Agent Events Example
//!
//! Demonstrates `AgentExecutor::astream_events()` — the high-level streaming
//! API that eliminates hand-rolled callback handlers + channels.
//!
//! Shows how to match on event types to build a UI-ready stream of agent
//! progress (LLM calls, tool invocations, final answer).
//!
//! Run with: `cargo run -p cognis-examples --example streaming_agent_events`

#[path = "../shared.rs"]
mod shared;

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use serde_json::{json, Value};

use cognis::agents::AgentExecutor;
use cognis_core::error::Result;
use cognis_core::messages::{HumanMessage, Message};
use cognis_core::tools::types::{ToolInput, ToolOutput};
use cognis_core::tools::BaseTool;
use cognis_core::tracers::event_stream::EventType;

/// A simple calculator tool that adds two numbers.
struct CalculatorTool;

#[async_trait]
impl BaseTool for CalculatorTool {
    fn name(&self) -> &str {
        "calculator"
    }

    fn description(&self) -> &str {
        "Add two numbers. Input: JSON object with keys 'a' and 'b' (numbers)."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"}
            },
            "required": ["a", "b"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let (a, b) = match &input {
            ToolInput::Structured(m) => (
                m.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0),
                m.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0),
            ),
            ToolInput::ToolCall(tc) => (
                tc.args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0),
                tc.args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0),
            ),
            ToolInput::Text(s) => {
                if let Ok(v) = serde_json::from_str::<Value>(s) {
                    (
                        v.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0),
                        v.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0),
                    )
                } else {
                    (0.0, 0.0)
                }
            }
        };
        println!("  [Calculator] {} + {} = {}", a, b, a + b);
        Ok(ToolOutput::Content(json!({ "result": a + b })))
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Streaming Agent Events Demo ===\n");

    // Use the tool-calling fallback helper: real Ollama if available, otherwise
    // a scripted fake model that issues a tool call then a final answer. This
    // ensures the example demonstrates OnAgentAction/OnToolStart/OnToolEnd
    // events even without a live LLM.
    let model = shared::get_tool_calling_model(vec![
        Message::ai_with_tool_calls(
            "",
            vec![json!({
                "name": "calculator",
                "args": {"a": 7, "b": 5},
                "id": "call_1"
            })],
        ),
        Message::ai("7 + 5 = 12"),
    ]);

    let executor = AgentExecutor::builder()
        .model(model)
        .tools(vec![Arc::new(CalculatorTool) as Arc<dyn BaseTool>])
        .max_iterations(3)
        .build();

    let messages = vec![Message::Human(HumanMessage::new("What is 7 + 5?"))];

    let mut stream = executor.astream_events(messages).await?;

    println!("Streaming events as they occur:\n");
    let mut event_count = 0usize;
    while let Some(event) = stream.next().await {
        let ev = event?;
        event_count += 1;
        match ev.event {
            EventType::OnChainStart => {
                println!("→ [{}] Agent started", ev.run_id);
            }
            EventType::OnChatModelStart | EventType::OnLlmStart => {
                println!("→ LLM call started ({})", ev.name);
            }
            EventType::OnChatModelEnd | EventType::OnLlmEnd => {
                println!("← LLM call finished");
            }
            EventType::OnAgentAction => {
                println!(
                    "→ Tool call: {} with input {:?}",
                    ev.name,
                    ev.data.input.as_ref().unwrap_or(&json!(null))
                );
            }
            EventType::OnToolStart => {
                println!("→ Tool '{}' starting", ev.name);
            }
            EventType::OnToolEnd => {
                println!(
                    "← Tool '{}' returned: {:?}",
                    ev.name,
                    ev.data.output.as_ref().unwrap_or(&json!(null))
                );
            }
            EventType::OnAgentFinish => {
                println!(
                    "✓ Final answer: {:?}",
                    ev.data.output.as_ref().unwrap_or(&json!(null))
                );
            }
            EventType::OnChainEnd => {
                println!("← Agent finished");
            }
            other => {
                println!("  (event: {:?})", other);
            }
        }
    }

    println!("\nTotal events received: {}", event_count);
    println!("\n=== Done ===");
    Ok(())
}
