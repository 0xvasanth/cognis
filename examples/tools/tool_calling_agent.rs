//! Tool-Calling Agent with SimpleTool, StructuredTool, and CachedTool
//!
//! Demonstrates creating tools using SimpleTool (single string input) and
//! StructuredTool (named arguments with schema validation), wrapping them
//! with CachedTool for result caching, and running them through an
//! AgentExecutor with a real LLM that decides which tools to call.
//!
//! Auto-detects Ollama for real LLM reasoning. Falls back to a fake model
//! with canned tool-calling responses when Ollama is not available.
//!
//! Run with: cargo run -p cognis-examples --example tool_calling_agent

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Tool-Calling Agent with Caching Example ===\n");

    // -------------------------------------------------------------------------
    // Step 1: Create tools using SimpleTool and StructuredTool
    // -------------------------------------------------------------------------
    println!("--- Step 1: Creating tools ---\n");

    // SimpleTool: takes a single string input.
    let search_tool = SimpleTool::new(
        "search",
        "Search for information about a topic. Input: a search query string.",
        |query: &str| {
            Ok(format!(
                "Search results for '{}': Rust is a systems programming language focused on \
                 safety, speed, and concurrency. It achieves memory safety without garbage \
                 collection through its ownership system.",
                query
            ))
        },
    );

    println!(
        "  SimpleTool: {} - {}",
        search_tool.name(),
        search_tool.description()
    );

    // StructuredTool: takes named arguments validated against a schema.
    let calculator_tool = StructuredTool::new(
        "calculator",
        "Perform arithmetic calculations with two numbers",
        json!({
            "type": "object",
            "properties": {
                "a": { "type": "number", "description": "First number" },
                "b": { "type": "number", "description": "Second number" },
                "operation": {
                    "type": "string",
                    "description": "Operation to perform",
                    "enum": ["add", "subtract", "multiply", "divide"]
                }
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
                _ => return Ok(json!({"error": format!("Unknown operation: {}", op)})),
            };

            Ok(json!({
                "result": result,
                "expression": format!("{} {} {} = {}", a, op, b, result)
            }))
        },
    );

    println!(
        "  StructuredTool: {} - {}",
        calculator_tool.name(),
        calculator_tool.description()
    );
    println!();

    // -------------------------------------------------------------------------
    // Step 2: Wrap the search tool with CachedTool
    // -------------------------------------------------------------------------
    println!("--- Step 2: Wrapping search tool with CachedTool ---\n");

    let cached_search = CachedTool::new(Arc::new(search_tool))
        .with_ttl(Duration::from_secs(300))
        .with_max_size(50);

    println!(
        "  CachedTool wrapping: {} (TTL=300s, max_size=50)",
        cached_search.name()
    );

    // Demonstrate caching: same query returns cached result on second call.
    use cognis_core::tools::types::ToolInput;

    println!("  Running search twice with same input...");
    let input = ToolInput::Text("Rust ownership".to_string());
    let result1 = cached_search._run(input.clone()).await?;
    let stats1 = cached_search.cache_stats();
    println!(
        "    First call:  hits={}, misses={}",
        stats1.hits, stats1.misses
    );

    let _result2 = cached_search._run(input).await?;
    let stats2 = cached_search.cache_stats();
    println!(
        "    Second call: hits={}, misses={} (cache hit!)",
        stats2.hits, stats2.misses,
    );

    if let cognis_core::tools::types::ToolOutput::Content(v) = result1 {
        let preview = v.as_str().unwrap_or("");
        println!("    Result: {}...", &preview[..preview.len().min(60)]);
    }
    println!();

    // -------------------------------------------------------------------------
    // Step 3: Set up the AgentExecutor with tools
    // -------------------------------------------------------------------------
    println!("--- Step 3: Building AgentExecutor ---\n");

    // Get a model: Ollama if available, otherwise fake with tool-calling simulation.
    let mut ai_calc = AIMessage::new("Let me calculate that for you.");
    ai_calc.tool_calls.push(ToolCall {
        name: "calculator".to_string(),
        args: {
            let mut m = HashMap::new();
            m.insert("a".to_string(), json!(42.0));
            m.insert("b".to_string(), json!(7.0));
            m.insert("operation".to_string(), json!("multiply"));
            m
        },
        id: Some("call_001".to_string()),
    });
    let ai_final = AIMessage::new("42 multiplied by 7 equals 294.");

    let model: Arc<dyn BaseChatModel> =
        shared::get_tool_calling_model(vec![Message::Ai(ai_calc), Message::Ai(ai_final)]);

    let search_tool_arc: Arc<dyn BaseTool> = Arc::new(cached_search);
    let calc_tool_arc: Arc<dyn BaseTool> = Arc::new(calculator_tool);

    let executor = AgentExecutor::builder()
        .model(model)
        .tool(search_tool_arc)
        .tool(calc_tool_arc)
        .max_iterations(10)
        .build();

    println!("  Built AgentExecutor (max_iterations=10, 2 tools)\n");

    // -------------------------------------------------------------------------
    // Step 4: Run the agent with a user question
    // -------------------------------------------------------------------------
    println!("--- Step 4: Running the agent ---\n");

    let user_message = "What is 42 multiplied by 7?";
    println!("  User: {user_message}\n");
    println!("  --- Agent Think-Act-Observe Loop ---");

    let initial_messages = vec![Message::human(user_message)];
    let result = executor.run(&initial_messages).await?;

    println!("  --- Agent Finished ---\n");

    // -------------------------------------------------------------------------
    // Step 5: Display the conversation
    // -------------------------------------------------------------------------
    println!(
        "--- Step 5: Conversation ({} messages) ---\n",
        result.messages.len()
    );

    for (i, msg) in result.messages.iter().enumerate() {
        match msg {
            Message::Human(_) => {
                println!("  [{}] Human: {}", i + 1, msg.content().text());
            }
            Message::Ai(ai) if !ai.tool_calls.is_empty() => {
                let tool_names: Vec<_> = ai.tool_calls.iter().map(|tc| tc.name.as_str()).collect();
                println!(
                    "  [{}] AI: {} -> calls {:?}",
                    i + 1,
                    msg.content().text(),
                    tool_names
                );
            }
            Message::Ai(_) => {
                println!("  [{}] AI (final): {}", i + 1, msg.content().text());
            }
            Message::Tool(t) => {
                let text = t.base.content.text();
                let preview = &text[..text.len().min(80)];
                println!("  [{}] Tool [{}]: {}", i + 1, t.tool_call_id, preview);
            }
            _ => {
                println!("  [{}] {}", i + 1, msg.content().text());
            }
        }
    }

    println!("\n  Final answer: {}", result.output);
    println!("\nDone!");
    Ok(())
}
