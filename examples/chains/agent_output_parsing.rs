//! Agent Output Parsing Example
//!
//! Demonstrates JSON and ToolCall output parsers that convert raw LLM text
//! into structured `AgentOutput` values (actions or final answers).
//!
//! Run with: `cargo run -p cognis-examples --example agent_output_parsing`

#[path = "../shared.rs"]
mod shared;

use cognis::agents::{AgentOutputParser, JsonOutputParser, ToolCallOutputParser};
use cognis_core::messages::Message;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Agent Output Parsing ===\n");

    // -- 1. JSON Output Parser ------------------------------------------------
    println!("--- JSON Output Parser ---\n");
    let json_parser = JsonOutputParser::new();

    // Action: tool invocation
    let json_action = r#"{"action": "calculator", "action_input": {"expression": "2 + 2"}}"#;
    println!("Action → {:?}\n", json_parser.parse(json_action)?);

    // Action inside a Markdown code block (common LLM pattern)
    let json_code_block = r#"Here is my response:
```json
{"action": "web_search", "action_input": {"query": "latest Rust release"}}
```"#;
    println!("Code block → {:?}\n", json_parser.parse(json_code_block)?);

    // Final answer
    let json_final = r#"{"action": "Final Answer", "action_input": "The answer is 42."}"#;
    println!("Final answer → {:?}\n", json_parser.parse(json_final)?);

    // Error case: no JSON present
    let bad = "I don't know what tool to use, sorry.";
    println!("Malformed → {}\n", json_parser.parse(bad).unwrap_err());

    // -- 2. ToolCall Output Parser --------------------------------------------
    println!("--- ToolCall Output Parser ---\n");
    let tc_parser = ToolCallOutputParser::new();

    // AI message with a tool call
    let tc_action = r#"{
        "content": "",
        "tool_calls": [
            {"name": "get_weather", "args": {"city": "London"}, "id": "call_001"}
        ]
    }"#;
    println!("Tool call → {:?}\n", tc_parser.parse(tc_action)?);

    // AI message with no tool calls (final answer)
    let tc_final = r#"{"content": "It's 15°C in London.", "tool_calls": []}"#;
    println!("Final answer → {:?}\n", tc_parser.parse(tc_final)?);

    // -- 3. Real LLM Demo (JSON format) ---------------------------------------
    println!("--- Real LLM Demo ---\n");

    let model = shared::get_chat_model(vec![
        r#"```json
{"action": "Final Answer", "action_input": "Rust is a systems programming language focused on safety and performance."}
```"#
            .into(),
    ]);

    let messages = vec![
        Message::system(
            "Always respond in JSON: {\"action\": \"Final Answer\", \"action_input\": \"<your answer>\"}",
        ),
        Message::human("What is Rust?"),
    ];

    let result = model._generate(&messages, None).await?;
    if let Some(gen) = result.generations.first() {
        let text = gen.message.content().text();
        println!("Raw LLM output:\n{text}\n");
        match json_parser.parse(&text) {
            Ok(output) => println!("Parsed → {output:?}"),
            Err(e) => println!("Parse error: {e}"),
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
