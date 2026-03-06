//! Tool-Calling Agent Example
//!
//! Demonstrates the AgentExecutor with a CalculatorTool. The mock model first
//! returns a tool call to "calculator", then returns the final text response
//! after seeing the tool result.
//!
//! No API keys required -- uses a hand-crafted mock model.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use rustchain::agents::AgentExecutor;
use rustchain::tools::calculator::CalculatorTool;
use rustchain_core::error::Result;
use rustchain_core::language_models::BaseChatModel;
use rustchain_core::messages::tool_types::ToolCall;
use rustchain_core::messages::{AIMessage, Message};
use rustchain_core::outputs::{ChatGeneration, ChatResult};
use rustchain_core::tools::BaseTool;

/// A mock model that simulates tool-calling behavior:
///
/// - On the first call, it returns an AIMessage with a tool call to "calculator".
/// - On the second call, it returns a final text answer incorporating the tool result.
struct ToolCallingMockModel {
    call_count: Mutex<u32>,
}

impl ToolCallingMockModel {
    fn new() -> Self {
        Self {
            call_count: Mutex::new(0),
        }
    }
}

#[async_trait]
impl BaseChatModel for ToolCallingMockModel {
    async fn _generate(
        &self,
        messages: &[Message],
        _stop: Option<&[String]>,
    ) -> Result<ChatResult> {
        let mut count = self.call_count.lock().unwrap();
        *count += 1;

        if *count == 1 {
            // First call: the model decides to use the calculator tool.
            println!("  [Model] Deciding to call the calculator tool...");
            let mut ai = AIMessage::new("Let me calculate that for you.");
            ai.tool_calls.push(ToolCall {
                name: "calculator".to_string(),
                args: HashMap::from([(
                    "expression".to_string(),
                    Value::String("(2 + 3) * 4".to_string()),
                )]),
                id: Some("call_001".to_string()),
            });
            Ok(ChatResult {
                generations: vec![ChatGeneration::new(ai)],
                llm_output: None,
            })
        } else {
            // Second call: the model has the tool result and produces the final answer.
            // Look for the tool result in the conversation.
            let tool_result = messages
                .iter()
                .rev()
                .find_map(|m| {
                    if let Message::Tool(t) = m {
                        Some(t.base.content.text())
                    } else {
                        None
                    }
                })
                .unwrap_or_else(|| "unknown".to_string());

            println!("  [Model] Got tool result: {tool_result}. Producing final answer...");
            let ai = AIMessage::new(format!("The result of (2 + 3) * 4 is {tool_result}."));
            Ok(ChatResult {
                generations: vec![ChatGeneration::new(ai)],
                llm_output: None,
            })
        }
    }

    fn llm_type(&self) -> &str {
        "tool-calling-mock"
    }
}

#[tokio::main]
async fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    println!("=== Tool-Calling Agent Example ===\n");

    // Step 1: Create the mock model.
    let model: Arc<dyn BaseChatModel> = Arc::new(ToolCallingMockModel::new());

    // Step 2: Create the calculator tool.
    //
    // CalculatorTool evaluates math expressions safely using a recursive descent parser.
    let calculator = Arc::new(CalculatorTool);
    println!("Tool: {} - {}", calculator.name(), calculator.description());

    // Step 3: Build the AgentExecutor using the builder pattern.
    //
    // The executor runs the agent loop: model -> tool calls -> tool results -> model
    // until the model stops calling tools or max_iterations is reached.
    let executor = AgentExecutor::builder()
        .model(model)
        .tool(calculator)
        .max_iterations(5)
        .build();

    println!("\nBuilt AgentExecutor (max_iterations=5)\n");

    // Step 4: Run the agent with an initial user message.
    let initial_messages = vec![Message::human("What is (2 + 3) * 4?")];

    println!("User: What is (2 + 3) * 4?\n");
    println!("--- Agent Loop ---");

    let result = executor.run(&initial_messages).await?;

    println!("--- Agent Finished ---\n");

    // Step 5: Print the full conversation.
    println!(
        "=== Full Conversation ({} messages) ===\n",
        result.messages.len()
    );
    for (i, msg) in result.messages.iter().enumerate() {
        let role = match msg {
            Message::Human(_) => "Human",
            Message::Ai(_) => "AI",
            Message::Tool(_) => "Tool",
            Message::System(_) => "System",
            _ => "Other",
        };
        println!("  [{}] {}: {}", i + 1, role, msg.content().text());
    }

    println!("\nFinal output: {}", result.output);

    Ok(())
}
