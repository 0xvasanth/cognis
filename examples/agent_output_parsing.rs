//! Agent Output Parsing Example
//!
//! Demonstrates the ReAct, JSON, XML, and ToolCall output parsers that convert
//! raw LLM text into structured `AgentOutput` values (actions or final answers).
//!
//! No API keys required.
//!
//! Run with: `cargo run -p rustchain-examples --example agent_output_parsing`

use rustchain::agents::{
    AgentOutputParser, JsonOutputParser, ReActOutputParser, ToolCallOutputParser, XmlOutputParser,
};

fn main() {
    println!("=== Agent Output Parsing Example ===\n");

    // -----------------------------------------------------------------------
    // 1. ReAct Output Parser
    // -----------------------------------------------------------------------
    println!("--- 1. ReAct Output Parser ---\n");

    let react_parser = ReActOutputParser::new();

    // 1a. Parse a tool action
    let react_action = "\
Thought: I need to look up information about Rust programming.
Action: search
Action Input: {\"query\": \"Rust programming language\"}";

    println!("Input (action):\n{react_action}\n");
    match react_parser.parse(react_action) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error: {e}\n"),
    }

    // 1b. Parse a final answer
    let react_final = "\
Thought: I now have enough information to answer.
Final Answer: Rust is a systems programming language focused on safety, speed, and concurrency.";

    println!("Input (final answer):\n{react_final}\n");
    match react_parser.parse(react_final) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error: {e}\n"),
    }

    // 1c. Parse plain string input (non-JSON action input)
    let react_plain = "\
Thought: Let me search for this.
Action: wikipedia
Action Input: quantum computing basics";

    println!("Input (plain string input):\n{react_plain}\n");
    match react_parser.parse(react_plain) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error: {e}\n"),
    }

    // 1d. Error handling: missing Action field
    let react_bad = "Thought: I'm confused and don't know what to do next.";

    println!("Input (malformed - missing Action):\n{react_bad}\n");
    match react_parser.parse(react_bad) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error (expected): {e}\n"),
    }

    // -----------------------------------------------------------------------
    // 2. JSON Output Parser
    // -----------------------------------------------------------------------
    println!("--- 2. JSON Output Parser ---\n");

    let json_parser = JsonOutputParser::new();

    // 2a. Parse a raw JSON action
    let json_action = r#"{"action": "calculator", "action_input": {"expression": "2 + 2"}}"#;

    println!("Input (raw JSON):\n{json_action}\n");
    match json_parser.parse(json_action) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error: {e}\n"),
    }

    // 2b. Parse JSON inside a Markdown code block
    let json_code_block = r#"Here is my response:
```json
{"action": "web_search", "action_input": {"query": "latest Rust release"}}
```"#;

    println!("Input (Markdown code block):\n{json_code_block}\n");
    match json_parser.parse(json_code_block) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error: {e}\n"),
    }

    // 2c. Parse a JSON final answer
    let json_final = r#"{"action": "Final Answer", "action_input": "The answer is 42."}"#;

    println!("Input (final answer):\n{json_final}\n");
    match json_parser.parse(json_final) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error: {e}\n"),
    }

    // 2d. Case-insensitive final answer variant
    let json_final_snake = r#"{"action": "final_answer", "action_input": "done"}"#;

    println!("Input (final_answer, snake_case):\n{json_final_snake}\n");
    match json_parser.parse(json_final_snake) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error: {e}\n"),
    }

    // 2e. Error handling: no JSON found
    let json_bad = "I don't know what tool to use, sorry.";

    println!("Input (malformed - no JSON):\n{json_bad}\n");
    match json_parser.parse(json_bad) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error (expected): {e}\n"),
    }

    // 2f. Error handling: JSON without action field
    let json_missing_field = r#"{"tool": "search", "input": "rust"}"#;

    println!("Input (malformed - missing 'action' field):\n{json_missing_field}\n");
    match json_parser.parse(json_missing_field) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error (expected): {e}\n"),
    }

    // -----------------------------------------------------------------------
    // 3. XML Output Parser
    // -----------------------------------------------------------------------
    println!("--- 3. XML Output Parser ---\n");

    let xml_parser = XmlOutputParser::new();

    // 3a. Parse an XML action with JSON input
    let xml_action = r#"<tool>search</tool>
<tool_input>{"query": "Rust async programming"}</tool_input>"#;

    println!("Input (XML action with JSON):\n{xml_action}\n");
    match xml_parser.parse(xml_action) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error: {e}\n"),
    }

    // 3b. Parse an XML action with plain string input
    let xml_plain = "<tool>greet</tool><tool_input>hello world</tool_input>";

    println!("Input (XML action with plain string):\n{xml_plain}\n");
    match xml_parser.parse(xml_plain) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error: {e}\n"),
    }

    // 3c. Parse an XML final answer
    let xml_final = r#"<tool>final_answer</tool>
<tool_input>The capital of France is Paris.</tool_input>"#;

    println!("Input (XML final answer):\n{xml_final}\n");
    match xml_parser.parse(xml_final) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error: {e}\n"),
    }

    // 3d. Error handling: missing <tool> tag
    let xml_bad = "<tool_input>some input without a tool tag</tool_input>";

    println!("Input (malformed - missing <tool>):\n{xml_bad}\n");
    match xml_parser.parse(xml_bad) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error (expected): {e}\n"),
    }

    // -----------------------------------------------------------------------
    // 4. ToolCall Output Parser
    // -----------------------------------------------------------------------
    println!("--- 4. ToolCall Output Parser ---\n");

    let toolcall_parser = ToolCallOutputParser::new();

    // 4a. Parse an AI message with tool calls
    let toolcall_action = r#"{
        "content": "",
        "tool_calls": [
            {"name": "get_weather", "args": {"city": "London", "units": "celsius"}, "id": "call_001"}
        ]
    }"#;

    println!("Input (AI message with tool calls):\n{toolcall_action}\n");
    match toolcall_parser.parse(toolcall_action) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error: {e}\n"),
    }

    // 4b. Parse an AI message with no tool calls (final answer)
    let toolcall_final =
        r#"{"content": "The weather in London is 15 degrees Celsius.", "tool_calls": []}"#;

    println!("Input (AI message with no tool calls):\n{toolcall_final}\n");
    match toolcall_parser.parse(toolcall_final) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error: {e}\n"),
    }

    // 4c. Error handling: invalid JSON
    let toolcall_bad = "This is not a JSON AI message at all.";

    println!("Input (malformed - not JSON):\n{toolcall_bad}\n");
    match toolcall_parser.parse(toolcall_bad) {
        Ok(output) => println!("Parsed: {output:#?}\n"),
        Err(e) => println!("Error (expected): {e}\n"),
    }

    println!("=== Done ===");
}
