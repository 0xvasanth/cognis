//! Tool-Calling Agent Example
//!
//! Demonstrates AgentExecutor with a CalculatorTool. A mock model emits a tool
//! call on the first turn, then produces a final answer after seeing the result.

#[path = "../shared.rs"]
mod shared;

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use serde_json::Value;

use cognis::agents::AgentExecutor;
use cognis::tools::calculator::CalculatorTool;
use cognis_core::error::Result;
use cognis_core::language_models::BaseChatModel;
use cognis_core::messages::tool_types::ToolCall;
use cognis_core::messages::{AIMessage, Message};
use cognis_core::outputs::{ChatGeneration, ChatResult};
use cognis_core::tools::BaseTool;

/// Mock model: first call returns a tool call, second call returns a text answer.
struct ToolCallingMockModel {
    call_count: Mutex<u32>,
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
            let mut ai = AIMessage::new("Let me calculate that for you.");
            ai.tool_calls.push(ToolCall {
                name: "calculator".to_string(),
                args: HashMap::from([(
                    "expression".to_string(),
                    Value::String("(2 + 3) * 4".into()),
                )]),
                id: Some("call_001".to_string()),
            });
            Ok(ChatResult {
                generations: vec![ChatGeneration::new(ai)],
                llm_output: None,
            })
        } else {
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
                .unwrap_or_else(|| "unknown".into());
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
    println!("=== Tool-Calling Agent ===\n");

    let model: Arc<dyn BaseChatModel> = Arc::new(ToolCallingMockModel {
        call_count: Mutex::new(0),
    });
    let calculator = Arc::new(CalculatorTool);
    println!("Tool: {} - {}", calculator.name(), calculator.description());

    let executor = AgentExecutor::builder()
        .model(model)
        .tool(calculator)
        .max_iterations(5)
        .build();

    let result = executor
        .run(&[Message::human("What is (2 + 3) * 4?")])
        .await?;

    println!("\nConversation ({} messages):", result.messages.len());
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
    println!("\nFinal: {}", result.output);

    // Real LLM demo
    let real_model = shared::get_chat_model(vec!["The result of 2 + 2 is 4.".into()]);
    let real_result = real_model
        ._generate(&[Message::human("What is 2 + 2?")], None)
        .await?;
    if let Some(gen) = real_result.generations.first() {
        println!("\nLLM: {}", gen.message.content().text());
    }

    Ok(())
}
