//! Tool-Calling Agent with SimpleTool, StructuredTool, and CachedTool
//!
//! Demonstrates creating tools using SimpleTool (single string input) and
//! StructuredTool (named arguments with schema validation), wrapping them
//! with CachedTool for result caching, and running them through an
//! AgentExecutor think-act-observe loop.
//!
//! No API keys required -- uses FakeMessagesListChatModel.
//!
//! Run with: cargo run -p cognis-examples --example tool_calling_agent

mod shared;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

use cognis::agents::AgentExecutor;
use cognis::tools::cached::CachedTool;
use cognis_core::language_models::FakeMessagesListChatModel;
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
        "Search for information about a topic",
        |query: &str| {
            // Simulate a search result based on the query.
            Ok(format!(
                "Search results for '{}': Found 3 relevant articles about this topic. \
                 Key finding: This is a well-documented subject with extensive resources.",
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
                    "description": "Operation to perform: add, subtract, multiply, divide",
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
                "divide" => {
                    if b == 0.0 {
                        return Ok(json!({"error": "Division by zero"}));
                    }
                    a / b
                }
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
    if let Some(schema) = calculator_tool.args_schema() {
        println!("  Schema: {}", serde_json::to_string(&schema)?);
    }
    println!();

    // -------------------------------------------------------------------------
    // Step 2: Wrap the search tool with CachedTool
    // -------------------------------------------------------------------------
    println!("--- Step 2: Wrapping search tool with CachedTool ---\n");

    let cached_search = CachedTool::new(Arc::new(search_tool))
        .with_ttl(Duration::from_secs(300)) // 5-minute TTL
        .with_max_size(50); // Max 50 cached entries

    println!(
        "  CachedTool wrapping: {} (TTL=300s, max_size=50)",
        cached_search.name()
    );

    // Demonstrate caching behavior.
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
        "    Second call: hits={}, misses={} (hit rate: {:.0}%)",
        stats2.hits,
        stats2.misses,
        stats2.hit_rate * 100.0
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

    // Create a mock model that simulates the think-act-observe loop:
    // 1. First call: model decides to search.
    // 2. Second call: model decides to calculate.
    // 3. Third call: model produces the final answer.

    let mut ai_search = AIMessage::new("Let me search for information about Rust's memory model.");
    ai_search.tool_calls.push(ToolCall {
        name: "search".to_string(),
        args: {
            let mut m = HashMap::new();
            m.insert(
                "tool_input".to_string(),
                json!("Rust memory safety ownership"),
            );
            m
        },
        id: Some("call_001".to_string()),
    });

    let mut ai_calc = AIMessage::new("Now let me calculate something.");
    ai_calc.tool_calls.push(ToolCall {
        name: "calculator".to_string(),
        args: {
            let mut m = HashMap::new();
            m.insert("a".to_string(), json!(42.0));
            m.insert("b".to_string(), json!(7.0));
            m.insert("operation".to_string(), json!("multiply"));
            m
        },
        id: Some("call_002".to_string()),
    });

    let ai_final = AIMessage::new(
        "Based on my research and calculations: Rust achieves memory safety through \
         its ownership system without garbage collection. The system uses three rules: \
         each value has one owner, only one owner at a time, and values are dropped when \
         the owner goes out of scope. Also, 42 * 7 = 294.",
    );

    // Uses a custom mock model for deterministic tool-calling behavior.
    // When Ollama is available, see ollama_chain example for real LLM usage.
    let model = Arc::new(FakeMessagesListChatModel::new(vec![
        Message::Ai(ai_search),
        Message::Ai(ai_calc),
        Message::Ai(ai_final),
    ]));

    // Build the executor with both tools.
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
    // Step 4: Run the agent
    // -------------------------------------------------------------------------
    println!("--- Step 4: Running the agent ---\n");

    let user_message = "Tell me about Rust's memory model and calculate 42 * 7.";
    println!("  User: {user_message}\n");
    println!("  --- Agent Think-Act-Observe Loop ---");

    let initial_messages = vec![Message::human(user_message)];
    let result = executor.run(&initial_messages).await?;

    println!("  --- Agent Finished ---\n");

    // -------------------------------------------------------------------------
    // Step 5: Display the conversation
    // -------------------------------------------------------------------------
    println!(
        "--- Step 5: Full Conversation ({} messages) ---\n",
        result.messages.len()
    );

    for (i, msg) in result.messages.iter().enumerate() {
        let (role, content) = match msg {
            Message::Human(_) => ("Human", msg.content().text()),
            Message::Ai(ai) => {
                if ai.tool_calls.is_empty() {
                    ("AI (final)", msg.content().text())
                } else {
                    let tool_names: Vec<_> =
                        ai.tool_calls.iter().map(|tc| tc.name.as_str()).collect();
                    (
                        "AI (tool call)",
                        format!("{} -> calls {:?}", msg.content().text(), tool_names),
                    )
                }
            }
            Message::Tool(t) => (
                "Tool Result",
                format!(
                    "[{}] {}",
                    t.tool_call_id,
                    &t.base.content.text()[..t.base.content.text().len().min(80)]
                ),
            ),
            Message::System(_) => ("System", msg.content().text()),
            _ => ("Other", msg.content().text()),
        };
        println!("  [{}] {}: {}", i + 1, role, content);
    }

    println!("\n  Final output: {}", result.output);

    // --- Real LLM Demo ---
    println!("\n--- Real LLM Demo ---\n");
    let real_model = shared::get_chat_model(vec![
        "Rust achieves memory safety through its ownership system with zero-cost abstractions."
            .into(),
    ]);
    let simple_messages = vec![Message::human(
        "Explain Rust's ownership model in one sentence.",
    )];
    let real_result = real_model._generate(&simple_messages, None).await?;
    if let Some(gen) = real_result.generations.first() {
        println!("Question: Explain Rust's ownership model in one sentence.");
        println!("LLM Response: {}", gen.message.content().text());
    }

    println!("\nDone!");
    Ok(())
}
