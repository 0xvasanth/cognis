//! ReAct Agent Example (Prebuilt Agents)
//!
//! Demonstrates the standalone prebuilt agent primitives from
//! `cognisgraph::prebuilt::agents`, which provide a lightweight tool-calling
//! workflow without requiring a full `BaseChatModel` implementation.
//!
//! Features shown:
//! - ToolDefinition creation with closure-based handlers
//! - ToolCall creation and resolution
//! - AgentState management (messages, tool calls, iteration tracking)
//! - ReactAgentConfig building with ToolChoice strategies
//! - create_react_agent factory and ReActAgent usage
//! - ToolNode executing pending tool calls
//! - Agent invocation with simple tools (calculator, string utils)
//!
//! No API keys required.
//!
//! Run with: `cargo run -p cognis-examples --example react_agent`

mod shared;
use cognis_core::language_models::chat_model::BaseChatModel;
use cognisgraph::prebuilt::agents::{
    create_react_agent, create_tool_node, AgentState, ReactAgentConfig, ToolCall, ToolChoice,
    ToolDefinition,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== ReAct Agent (Prebuilt Agents) Example ===\n");

    // -----------------------------------------------------------------------
    // 1. ToolDefinition Creation with Handlers
    // -----------------------------------------------------------------------
    println!("--- 1. ToolDefinition Creation ---");
    println!("Tools are closures wrapped in a name + description.\n");

    let calculator = ToolDefinition::new("calculator", "Evaluate simple arithmetic", |input| {
        let obj = input.as_object().ok_or("expected JSON object")?;
        let a = obj
            .get("a")
            .and_then(|v| v.as_f64())
            .ok_or("missing field 'a'")?;
        let b = obj
            .get("b")
            .and_then(|v| v.as_f64())
            .ok_or("missing field 'b'")?;
        let op = obj
            .get("op")
            .and_then(|v| v.as_str())
            .ok_or("missing field 'op'")?;
        let result = match op {
            "add" => a + b,
            "sub" => a - b,
            "mul" => a * b,
            "div" => {
                if b == 0.0 {
                    return Err("division by zero".into());
                }
                a / b
            }
            _ => return Err(format!("unknown op: {}", op)),
        };
        Ok(json!({ "result": result }))
    });

    let reverse = ToolDefinition::new("reverse", "Reverse a string", |input| {
        let s = input.as_str().ok_or("expected a string")?;
        let reversed: String = s.chars().rev().collect();
        Ok(json!(reversed))
    });

    let uppercase = ToolDefinition::new("uppercase", "Convert a string to uppercase", |input| {
        let s = input.as_str().ok_or("expected a string")?;
        Ok(json!(s.to_uppercase()))
    });

    println!("  calculator: {}", calculator.description());
    println!("  reverse:    {}", reverse.description());
    println!("  uppercase:  {}", uppercase.description());

    // Quick smoke-test the calculator handler directly.
    let calc_result = calculator
        .execute(json!({"a": 6, "b": 7, "op": "mul"}))
        .unwrap();
    println!("\n  Direct execute: 6 * 7 = {}", calc_result["result"]);

    // -----------------------------------------------------------------------
    // 2. ToolCall Creation and Resolution
    // -----------------------------------------------------------------------
    println!("\n--- 2. ToolCall Creation and Resolution ---");
    println!("ToolCalls track pending invocations and their results.\n");

    let call = ToolCall::new(
        "call-1",
        "calculator",
        json!({"a": 10, "b": 3, "op": "add"}),
    );
    println!("  Created: id={}, tool={}", call.id, call.tool_name);
    println!("  Resolved? {}", call.is_resolved());

    let resolved = call.with_result(json!({"result": 13}));
    println!("  After with_result: resolved={}", resolved.is_resolved());
    println!("  Result: {}", resolved.result.as_ref().unwrap());

    // Serialization round-trip.
    let call_json = resolved.to_json();
    println!("  JSON: {}", call_json);

    // -----------------------------------------------------------------------
    // 3. AgentState Management
    // -----------------------------------------------------------------------
    println!("\n--- 3. AgentState Management ---");
    println!("AgentState tracks messages, tool calls, and iteration count.\n");

    let mut state = AgentState::new(10);
    state.add_message(json!({"role": "human", "content": "What is 5 + 3?"}));
    state.add_message(json!({"role": "system", "content": "You are a math helper."}));
    state.add_tool_call(ToolCall::new(
        "tc-1",
        "calculator",
        json!({"a": 5, "b": 3, "op": "add"}),
    ));

    println!("  Messages:       {}", state.messages.len());
    println!("  Pending calls:  {}", state.tool_calls.len());
    println!("  Iteration:      {}", state.iteration);
    println!("  Max iterations: {}", state.max_iterations);
    println!("  Finished?       {}", state.is_finished);
    println!("  Over limit?     {}", state.is_over_limit());

    state.mark_finished();
    println!("  After mark_finished: {}", state.is_finished);

    // Serialize the full state.
    let state_json = state.to_json();
    println!(
        "  State JSON keys: {:?}",
        state_json.as_object().unwrap().keys().collect::<Vec<_>>()
    );

    // -----------------------------------------------------------------------
    // 4. ReactAgentConfig with ToolChoice
    // -----------------------------------------------------------------------
    println!("\n--- 4. ReactAgentConfig Building ---");
    println!("Configure max iterations, tool choice strategy, and system message.\n");

    // Default config.
    let default_config = ReactAgentConfig::default();
    println!(
        "  Default: max_iterations={}, tool_choice={}",
        default_config.max_iterations, default_config.tool_choice
    );

    // Builder with all options.
    let config = ReactAgentConfig::builder()
        .max_iterations(5)
        .tool_choice(ToolChoice::Required)
        .system_message("You are a helpful calculator and string utilities assistant.")
        .build();
    println!(
        "  Custom:  max_iterations={}, tool_choice={}",
        config.max_iterations, config.tool_choice
    );
    println!(
        "  System:  {:?}",
        config.system_message.as_deref().unwrap_or("(none)")
    );

    // ToolChoice variants.
    println!("\n  ToolChoice variants:");
    println!("    Auto:     {}", ToolChoice::Auto);
    println!("    Required: {}", ToolChoice::Required);
    println!("    None:     {}", ToolChoice::None);
    println!(
        "    Specific: {}",
        ToolChoice::Specific("calculator".into())
    );

    // -----------------------------------------------------------------------
    // 5. create_react_agent and ReActAgent Usage
    // -----------------------------------------------------------------------
    println!("\n--- 5. create_react_agent Factory ---");
    println!("Build an agent and invoke it with tool-call directives.\n");

    let tools = vec![
        ToolDefinition::new("calculator", "Evaluate simple arithmetic", |input| {
            let obj = input.as_object().ok_or("expected JSON object")?;
            let a = obj.get("a").and_then(|v| v.as_f64()).ok_or("missing 'a'")?;
            let b = obj.get("b").and_then(|v| v.as_f64()).ok_or("missing 'b'")?;
            let op = obj
                .get("op")
                .and_then(|v| v.as_str())
                .ok_or("missing 'op'")?;
            let result = match op {
                "add" => a + b,
                "sub" => a - b,
                "mul" => a * b,
                "div" if b != 0.0 => a / b,
                "div" => return Err("division by zero".into()),
                _ => return Err(format!("unknown op: {}", op)),
            };
            Ok(json!({ "result": result }))
        }),
        ToolDefinition::new("reverse", "Reverse a string", |input| {
            let s = input.as_str().ok_or("expected string")?;
            Ok(json!(s.chars().rev().collect::<String>()))
        }),
        ToolDefinition::new("uppercase", "Uppercase a string", |input| {
            let s = input.as_str().ok_or("expected string")?;
            Ok(json!(s.to_uppercase()))
        }),
    ];

    let agent_config = ReactAgentConfig::builder()
        .max_iterations(10)
        .system_message("You are a multi-tool assistant.")
        .build();
    let agent = create_react_agent(tools, agent_config);

    println!("  Available tools: {:?}", agent.available_tools());
    println!("  Config max_iterations: {}", agent.config().max_iterations);

    // Invoke with a calculator tool call.
    let input = json!({
        "messages": [{"role": "human", "content": "What is 12 * 5?"}],
        "tool_calls": [
            {"id": "tc-calc", "tool_name": "calculator", "arguments": {"a": 12, "b": 5, "op": "mul"}}
        ]
    });

    let result_state = agent.invoke(input).unwrap();
    println!("\n  Invocation result:");
    println!("    Finished:   {}", result_state.is_finished);
    println!("    Iterations: {}", result_state.iteration);
    println!("    Messages:   {}", result_state.messages.len());
    for msg in &result_state.messages {
        println!(
            "      [{}] {}",
            msg["role"].as_str().unwrap_or("?"),
            msg["content"]
        );
    }
    println!("    Tool results:");
    for tc in &result_state.tool_calls {
        println!(
            "      {} -> {}",
            tc.tool_name,
            tc.result.as_ref().unwrap_or(&json!(null))
        );
    }

    // -----------------------------------------------------------------------
    // 6. ToolNode Executing Pending Calls
    // -----------------------------------------------------------------------
    println!("\n--- 6. ToolNode Direct Usage ---");
    println!("Use ToolNode independently to batch-execute tool calls.\n");

    let node_tools = vec![
        ToolDefinition::new("reverse", "Reverse a string", |input| {
            let s = input.as_str().ok_or("expected string")?;
            Ok(json!(s.chars().rev().collect::<String>()))
        }),
        ToolDefinition::new("uppercase", "Uppercase a string", |input| {
            let s = input.as_str().ok_or("expected string")?;
            Ok(json!(s.to_uppercase()))
        }),
    ];
    let tool_node = create_tool_node(node_tools);

    println!("  Tool node tools: {:?}", tool_node.tool_names());

    let mut node_state = AgentState::new(5);
    node_state.add_tool_call(ToolCall::new("n1", "reverse", json!("hello")));
    node_state.add_tool_call(ToolCall::new("n2", "uppercase", json!("world")));
    node_state.add_tool_call(ToolCall::new("n3", "missing_tool", json!("oops")));

    tool_node.execute(&mut node_state).unwrap();

    println!("  After execution:");
    for tc in &node_state.tool_calls {
        let result = tc.result.as_ref().unwrap();
        println!("    {} ({}): {}", tc.tool_name, tc.id, result);
    }
    println!("  Messages added: {}", node_state.messages.len());

    // -----------------------------------------------------------------------
    // 7. Agent Invoke with Multiple Tools
    // -----------------------------------------------------------------------
    println!("\n--- 7. Multi-Tool Agent Invocation ---");
    println!("Invoke an agent with multiple tool calls in a single request.\n");

    let multi_tools = vec![
        ToolDefinition::new("calculator", "Arithmetic", |input| {
            let obj = input.as_object().ok_or("expected object")?;
            let a = obj.get("a").and_then(|v| v.as_f64()).ok_or("missing 'a'")?;
            let b = obj.get("b").and_then(|v| v.as_f64()).ok_or("missing 'b'")?;
            let op = obj
                .get("op")
                .and_then(|v| v.as_str())
                .ok_or("missing 'op'")?;
            let r = match op {
                "add" => a + b,
                "mul" => a * b,
                _ => return Err(format!("unsupported: {}", op)),
            };
            Ok(json!({"result": r}))
        }),
        ToolDefinition::new("reverse", "Reverse", |input| {
            let s = input.as_str().ok_or("expected string")?;
            Ok(json!(s.chars().rev().collect::<String>()))
        }),
    ];

    let multi_agent = create_react_agent(
        multi_tools,
        ReactAgentConfig::builder()
            .max_iterations(5)
            .system_message("Handle all requested tool calls.")
            .build(),
    );

    let multi_input = json!({
        "messages": [
            {"role": "human", "content": "Compute 100 + 200 and reverse 'cognis'"}
        ],
        "tool_calls": [
            {"id": "m1", "tool_name": "calculator", "arguments": {"a": 100, "b": 200, "op": "add"}},
            {"id": "m2", "tool_name": "reverse", "arguments": "cognis"}
        ]
    });

    let multi_state = multi_agent.invoke(multi_input).unwrap();
    println!("  Finished:   {}", multi_state.is_finished);
    println!("  Iterations: {}", multi_state.iteration);
    println!("  Results:");
    for tc in &multi_state.tool_calls {
        println!(
            "    {} -> {}",
            tc.tool_name,
            tc.result.as_ref().unwrap_or(&json!(null))
        );
    }

    // Edge case: invoke with no tool calls finishes immediately.
    println!("\n  Edge case -- no tool calls:");
    let empty_result = multi_agent
        .invoke(json!({"messages": [{"role": "human", "content": "hi"}]}))
        .unwrap();
    println!("    Finished immediately: {}", empty_result.is_finished);
    println!("    Iterations: {}", empty_result.iteration);

    // Edge case: max iterations reached.
    println!("\n  Edge case -- zero max iterations:");
    let zero_agent = create_react_agent(
        vec![ToolDefinition::new("t", "d", |_| Ok(json!("ok")))],
        ReactAgentConfig::builder().max_iterations(0).build(),
    );
    let zero_state = zero_agent
        .invoke(json!({"tool_calls": [{"id": "z1", "tool_name": "t", "arguments": null}]}))
        .unwrap();
    println!("    Finished: {}", zero_state.is_finished);
    println!(
        "    Tool resolved: {}",
        zero_state.tool_calls[0].is_resolved()
    );

    // -----------------------------------------------------------------------
    // 8. Real LLM Demo — LLM showing real agent behavior
    // -----------------------------------------------------------------------
    println!("\n--- 8. Real LLM Demo ---");
    println!("Use an LLM to demonstrate real reasoning in a ReAct agent context.\n");

    let model = shared::get_chat_model(vec![
        "To solve '15 * 7 + 23', I would use the calculator tool: first compute 15 * 7 = 105, then add 23 to get 128. The final answer is 128.".into(),
    ]);
    let messages = vec![
        cognis_core::messages::Message::human(
            "Using a ReAct approach, explain step by step how you would solve: 15 * 7 + 23"
        ),
    ];
    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        println!("  LLM Response: {}", gen.message.content().text());
    }

    println!("\n=== ReAct Agent (Prebuilt Agents) Example Complete ===");
    Ok(())
}
