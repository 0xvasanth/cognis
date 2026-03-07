//! ReAct Agent Example
//!
//! Demonstrates the ReAct (Reasoning + Acting) agent loop using a mock model
//! and simple tools. The agent reasons about which tool to call, executes it,
//! observes the result, and iterates until it produces a final answer.
//!
//! No API keys required -- uses a fake chat model with scripted responses.

use std::sync::Arc;

use rustchain::agents::{create_react_agent, ReActAgent};
use rustchain_core::language_models::FakeListChatModel;
use rustchain_core::tools::{BaseTool, SimpleTool};

/// Create a mock calculator tool that evaluates simple expressions.
fn make_calculator_tool() -> Arc<dyn BaseTool> {
    Arc::new(SimpleTool::new(
        "calculator",
        "Perform arithmetic calculations. Input should be a math expression.",
        |expr: &str| {
            // Simple mock: return a fixed result for demonstration purposes.
            match expr.trim() {
                "15 * 4" => Ok("60".to_string()),
                "60 + 7" => Ok("67".to_string()),
                _ => Ok(format!("Result of '{}' = 42", expr)),
            }
        },
    ))
}

/// Create a mock lookup tool that returns information about topics.
fn make_lookup_tool() -> Arc<dyn BaseTool> {
    Arc::new(SimpleTool::new(
        "lookup",
        "Look up factual information about a topic. Input is the topic name.",
        |topic: &str| match topic.trim().to_lowercase().as_str() {
            "rust" => Ok(
                "Rust is a systems programming language focused on safety, speed, and concurrency."
                    .to_string(),
            ),
            "ferris" => {
                Ok("Ferris is the unofficial mascot of Rust. Ferris is a crab.".to_string())
            }
            _ => Ok(format!(
                "Information about '{}': it is an interesting topic.",
                topic
            )),
        },
    ))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ReAct Agent Example ===\n");

    // -------------------------------------------------------------------------
    // Example 1: Single tool call then final answer
    // -------------------------------------------------------------------------
    println!("--- Example 1: Single tool call (lookup) ---\n");

    // The FakeListChatModel cycles through scripted responses.
    // First response triggers a tool call, second provides the final answer.
    let model_1 = Arc::new(FakeListChatModel::new(vec![
        "Thought: I should look up information about Rust.\n\
         Action: lookup\n\
         Action Input: rust"
            .into(),
        "Final Answer: Rust is a systems programming language focused on safety, \
         speed, and concurrency."
            .into(),
    ]));

    let tools: Vec<Arc<dyn BaseTool>> = vec![make_calculator_tool(), make_lookup_tool()];

    let agent = create_react_agent(model_1, tools.clone());

    let result = agent.run("What is Rust?").await?;
    println!("  Output: {}", result.output);
    println!("  Iterations: {}", result.iterations);
    println!("  Tool calls: {}", result.tool_calls.len());
    for (name, input, output) in &result.tool_calls {
        println!("    - Tool: {}, Input: {}, Output: {}", name, input, output);
    }

    // -------------------------------------------------------------------------
    // Example 2: Multiple tool calls with trace
    // -------------------------------------------------------------------------
    println!("\n--- Example 2: Multiple tool calls with trace ---\n");

    let model_2 = Arc::new(FakeListChatModel::new(vec![
        "Thought: I need to look up what Ferris is first.\n\
         Action: lookup\n\
         Action Input: ferris"
            .into(),
        "Thought: Now I need to calculate 15 * 4.\n\
         Action: calculator\n\
         Action Input: 15 * 4"
            .into(),
        "Final Answer: Ferris is the Rust mascot (a crab), and 15 * 4 = 60.".into(),
    ]));

    let agent = create_react_agent(model_2, tools.clone());

    let (output, trace) = agent
        .run_with_trace("Who is Ferris and what is 15 * 4?")
        .await?;
    println!("  Output: {}", output);
    println!("  Total steps: {}", trace.steps.len());
    println!("  Approximate tokens: {}", trace.total_tokens);
    println!("  Trace complete: {}", trace.is_complete());
    println!("\n  Step-by-step trace:");
    for (i, step) in trace.get_steps().iter().enumerate() {
        println!("    Step {}:", i + 1);
        println!("      Thought: {}", step.thought);
        if let Some(ref action) = step.action {
            println!("      Action: {}", action);
        }
        if let Some(ref input) = step.action_input {
            println!("      Action Input: {}", input);
        }
        if let Some(ref obs) = step.observation {
            println!("      Observation: {}", obs);
        }
        println!("      Is final: {}", step.is_final);
    }

    // -------------------------------------------------------------------------
    // Example 3: Using the builder with custom settings
    // -------------------------------------------------------------------------
    println!("\n--- Example 3: Builder pattern with max_iterations ---\n");

    let model_3 = Arc::new(FakeListChatModel::new(vec![
        "Thought: Let me calculate.\n\
         Action: calculator\n\
         Action Input: 60 + 7"
            .into(),
        "Final Answer: 60 + 7 = 67.".into(),
    ]));

    let agent = ReActAgent::builder()
        .model(model_3)
        .tool(make_calculator_tool())
        .tool(make_lookup_tool())
        .max_iterations(5)
        .system_prompt("You are a helpful math assistant.")
        .verbose(true) // Prints intermediate steps to stdout
        .build()?;

    println!("  Running agent with verbose=true...\n");
    let result = agent.run("What is 60 + 7?").await?;
    println!("\n  Final output: {}", result.output);
    println!("  Iterations used: {} / 5 max", result.iterations);

    // -------------------------------------------------------------------------
    // Example 4: Immediate final answer (no tool calls)
    // -------------------------------------------------------------------------
    println!("\n--- Example 4: Immediate final answer ---\n");

    let model_4 = Arc::new(FakeListChatModel::new(vec![
        "Final Answer: I already know that 2 + 2 = 4.".into(),
    ]));

    let agent = create_react_agent(model_4, tools);

    let result = agent.run("What is 2 + 2?").await?;
    println!("  Output: {}", result.output);
    println!("  Iterations: {}", result.iterations);
    println!("  Tool calls: {} (none needed)", result.tool_calls.len());

    println!("\n=== ReAct Agent Example Complete ===");
    Ok(())
}
