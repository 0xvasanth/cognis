//! Tool-Calling Agent with SimpleTool, StructuredTool, and CachedTool
//!
//! Shows how to create tools with SimpleTool/StructuredTool, wrap them with
//! CachedTool for result caching, and run them through an AgentExecutor.

#[path = "../shared.rs"]
mod shared;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use cognis::agents::AgentExecutor;
use cognis::tools::cached::CachedTool;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognis_core::messages::tool_types::ToolCall;
use cognis_core::messages::{AIMessage, Message};
use cognis_core::tools::base::BaseTool;
use cognis_core::tools::simple::SimpleTool;
use cognis_core::tools::structured::StructuredTool;
use cognis_core::tools::types::ToolInput;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tool-Calling Agent with Caching ===\n");

    // SimpleTool: single string input
    let search_tool = SimpleTool::new(
        "search",
        "Search for information about a topic",
        |query: &str| {
            Ok(format!(
                "Results for '{}': Rust is a systems language focused on safety.",
                query
            ))
        },
    );

    // StructuredTool: named arguments with schema
    let calculator_tool = StructuredTool::new(
        "calculator",
        "Perform arithmetic calculations with two numbers",
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "number" },
                "b": { "type": "number" },
                "operation": { "type": "string", "enum": ["add", "subtract", "multiply", "divide"] }
            },
            "required": ["a", "b", "operation"]
        }),
        |args: HashMap<String, Value>| async move {
            let a = args.get("a").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let b = args.get("b").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let op = args
                .get("operation")
                .and_then(|v| v.as_str())
                .unwrap_or("add");
            let result = match op {
                "add" => a + b,
                "subtract" => a - b,
                "multiply" => a * b,
                "divide" if b != 0.0 => a / b,
                "divide" => return Ok(json!({"error": "Division by zero"})),
                _ => return Ok(json!({"error": format!("Unknown op: {}", op)})),
            };
            Ok(json!({"result": result, "expression": format!("{} {} {} = {}", a, op, b, result)}))
        },
    );

    // Wrap search with caching
    let cached_search = CachedTool::new(Arc::new(search_tool))
        .with_ttl(Duration::from_secs(300))
        .with_max_size(50);

    let input = ToolInput::Text("Rust ownership".into());
    let _ = cached_search._run(input.clone()).await?;
    let _ = cached_search._run(input).await?;
    let stats = cached_search.cache_stats();
    println!("Cache: hits={}, misses={}\n", stats.hits, stats.misses);

    // Build AgentExecutor with fake tool-calling model
    let mut ai_calc = AIMessage::new("Let me calculate that.");
    ai_calc.tool_calls.push(ToolCall {
        name: "calculator".into(),
        args: HashMap::from([
            ("a".into(), json!(42.0)),
            ("b".into(), json!(7.0)),
            ("operation".into(), json!("multiply")),
        ]),
        id: Some("call_001".into()),
    });
    let ai_final = AIMessage::new("42 multiplied by 7 equals 294.");
    let model: Arc<dyn BaseChatModel> =
        shared::get_tool_calling_model(vec![Message::Ai(ai_calc), Message::Ai(ai_final)]);

    let executor = AgentExecutor::builder()
        .model(model)
        .tool(Arc::new(cached_search) as Arc<dyn BaseTool>)
        .tool(Arc::new(calculator_tool) as Arc<dyn BaseTool>)
        .max_iterations(10)
        .build();

    let result = executor
        .run(&[Message::human("What is 42 multiplied by 7?")])
        .await?;

    println!("Conversation ({} messages):", result.messages.len());
    for (i, msg) in result.messages.iter().enumerate() {
        let label = match msg {
            Message::Human(_) => "Human".into(),
            Message::Ai(ai) if !ai.tool_calls.is_empty() => format!(
                "AI -> {:?}",
                ai.tool_calls.iter().map(|tc| &tc.name).collect::<Vec<_>>()
            ),
            Message::Ai(_) => "AI".into(),
            Message::Tool(t) => format!("Tool [{}]", t.tool_call_id),
            _ => "Other".into(),
        };
        println!(
            "  [{}] {}: {}",
            i + 1,
            label,
            &msg.content().text()[..msg.content().text().len().min(80)]
        );
    }
    println!("\nFinal: {}", result.output);
    Ok(())
}
