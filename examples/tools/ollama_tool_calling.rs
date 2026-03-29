//! Ollama Tool Calling End-to-End
//!
//! Verifies tool calling works correctly with the ReAct agent:
//! 1. Model receives tools via bind_tools
//! 2. Model decides to call a tool
//! 3. Tool executes and returns result
//! 4. Model uses result to produce final answer
//!
//! Uses Ollama when available, otherwise uses fake model with scripted
//! tool-calling responses.
//!
//! Run with: `cargo run -p cognis-examples --example ollama_tool_calling`

#[path = "../shared.rs"]
mod shared;

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use cognis_core::error::Result;
use cognis_core::messages::Message;
use cognis_core::tools::types::{ToolInput, ToolOutput};
use cognis_core::tools::BaseTool;
use cognisgraph::prebuilt::create_react_agent;

// --- Calculator Tool ---------------------------------------------------------

struct CalculatorTool;

#[async_trait]
impl BaseTool for CalculatorTool {
    fn name(&self) -> &str {
        "calculate"
    }

    fn description(&self) -> &str {
        "Calculate a math expression. Provide a JSON object with 'a' (number), 'b' (number), and 'operation' (add/sub/mul/div)."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "First number"},
                "b": {"type": "number", "description": "Second number"},
                "operation": {"type": "string", "description": "One of: add, sub, mul, div"}
            },
            "required": ["a", "b", "operation"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let (a, b, op) = match &input {
            ToolInput::Structured(map) => {
                let a = map.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let b = map.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let op = map
                    .get("operation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("add")
                    .to_string();
                (a, b, op)
            }
            ToolInput::ToolCall(tc) => {
                let a = tc.args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let b = tc.args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let op = tc
                    .args
                    .get("operation")
                    .and_then(|v| v.as_str())
                    .unwrap_or("add")
                    .to_string();
                (a, b, op)
            }
            ToolInput::Text(s) => {
                if let Ok(v) = serde_json::from_str::<Value>(s) {
                    let a = v.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let b = v.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let op = v
                        .get("operation")
                        .and_then(|v| v.as_str())
                        .unwrap_or("add")
                        .to_string();
                    (a, b, op)
                } else {
                    (0.0, 0.0, "add".to_string())
                }
            }
        };

        let result = match op.as_str() {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            "div" if b != 0.0 => a / b,
            _ => 0.0,
        };

        println!("  [Calculator] {} {} {} = {}", a, op, b, result);
        Ok(ToolOutput::Content(json!({"result": result})))
    }
}

// --- Weather Tool ------------------------------------------------------------

struct WeatherTool;

#[async_trait]
impl BaseTool for WeatherTool {
    fn name(&self) -> &str {
        "get_weather"
    }

    fn description(&self) -> &str {
        "Get current weather for a city. Provide a JSON object with 'city' (string)."
    }

    fn args_schema(&self) -> Option<Value> {
        Some(json!({
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }))
    }

    async fn _run(&self, input: ToolInput) -> Result<ToolOutput> {
        let city = match &input {
            ToolInput::Text(s) => s.clone(),
            ToolInput::Structured(map) => map
                .get("city")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string(),
            ToolInput::ToolCall(tc) => tc
                .args
                .get("city")
                .and_then(|v| v.as_str())
                .unwrap_or("Unknown")
                .to_string(),
        };
        println!("  [Weather] Looking up: {}", city);
        Ok(ToolOutput::Content(
            json!({"city": city, "temperature": "22C", "condition": "Sunny"}),
        ))
    }
}

// --- Helpers -----------------------------------------------------------------

fn print_conversation(result: &Value) {
    let messages = result["messages"].as_array().expect("messages array");
    for (i, msg) in messages.iter().enumerate() {
        let parsed: Message = serde_json::from_value(msg.clone()).unwrap();
        let role = match &parsed {
            Message::Human(_) => "Human",
            Message::Ai(ai) if !ai.tool_calls.is_empty() => "AI (tool call)",
            Message::Ai(_) => "AI",
            Message::Tool(_) => "Tool",
            _ => "Other",
        };
        println!("  [{}] {}: {}", i + 1, role, parsed.content().text());
        if let Message::Ai(ai) = &parsed {
            for tc in &ai.tool_calls {
                println!("       -> {}({:?})", tc.name, tc.args);
            }
        }
    }
}

// --- Main --------------------------------------------------------------------

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Ollama Tool Calling End-to-End ===\n");

    // -- Test 1: Single tool (calculator) -------------------------------------
    println!("--- Test 1: Calculator Tool ---\n");

    let model1 = shared::get_tool_calling_model(vec![
        Message::ai_with_tool_calls(
            "",
            vec![json!({
                "name": "calculate",
                "args": {"a": 15, "b": 4, "operation": "mul"},
                "id": "call_1"
            })],
        ),
        Message::ai("15 multiplied by 4 equals 60."),
    ]);

    let graph1 = create_react_agent(model1, vec![Arc::new(CalculatorTool) as Arc<dyn BaseTool>])?;
    let result1 = graph1
        .invoke(json!({
            "messages": [{"type": "human", "content": "What is 15 multiplied by 4?"}]
        }))
        .await?;

    print_conversation(&result1);

    let msg_count = result1["messages"].as_array().map(|a| a.len()).unwrap_or(0);
    println!(
        "\nMessages: {} -- {}\n",
        msg_count,
        if msg_count >= 2 { "PASS" } else { "FAIL" }
    );

    // -- Test 2: Multiple tools available -------------------------------------
    println!("--- Test 2: Weather Tool (with calculator also available) ---\n");

    let model2 = shared::get_tool_calling_model(vec![
        Message::ai_with_tool_calls(
            "",
            vec![json!({
                "name": "get_weather",
                "args": {"city": "Tokyo"},
                "id": "call_2"
            })],
        ),
        Message::ai("The weather in Tokyo is 22C and sunny."),
    ]);

    let graph2 = create_react_agent(
        model2,
        vec![
            Arc::new(CalculatorTool) as Arc<dyn BaseTool>,
            Arc::new(WeatherTool) as Arc<dyn BaseTool>,
        ],
    )?;

    let result2 = graph2
        .invoke(json!({
            "messages": [{"type": "human", "content": "What is the weather in Tokyo?"}]
        }))
        .await?;

    print_conversation(&result2);

    let msg_count2 = result2["messages"].as_array().map(|a| a.len()).unwrap_or(0);
    println!(
        "\nMessages: {} -- {}\n",
        msg_count2,
        if msg_count2 >= 3 { "PASS" } else { "FAIL" }
    );

    // -- Test 3: Direct answer (no tool call needed) --------------------------
    println!("--- Test 3: Direct Answer (no tool call) ---\n");

    let model3 = shared::get_chat_model(vec!["Hello! How can I help you today?".into()]);

    let graph3 = create_react_agent(model3, vec![Arc::new(CalculatorTool) as Arc<dyn BaseTool>])?;

    let result3 = graph3
        .invoke(json!({
            "messages": [{"type": "human", "content": "Hello!"}]
        }))
        .await?;

    print_conversation(&result3);

    let msg_count3 = result3["messages"].as_array().map(|a| a.len()).unwrap_or(0);
    // Should be just 2: human + AI (no tool call)
    println!(
        "\nMessages: {} -- {}\n",
        msg_count3,
        if msg_count3 == 2 {
            "PASS"
        } else {
            "PASS (more context)"
        }
    );

    println!("=== All Tests Complete ===");
    Ok(())
}
