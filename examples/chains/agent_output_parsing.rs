//! Tool-call output handling. V2 carries tool calls natively in
//! `AiMessage::tool_calls`, so V1's `ToolCallOutputParser` /
//! `JsonOutputParser` collapse into "read the message".

use cognis::prelude::*;
use cognis_core::message::{AiMessage, ToolCall};

fn main() {
    println!("=== V2 Tool-Call Output Handling ===\n");

    // Imagine the model returned this assistant message.
    let assistant = Message::Ai(AiMessage {
        content: String::new(),
        tool_calls: vec![ToolCall {
            id: "call_1".into(),
            name: "calculator".into(),
            arguments: serde_json::json!({"expression": "23 * 17"}),
        }],
        parts: Vec::new(),
    });

    // V2 handling: the agent loop reads `tool_calls()` directly.
    if assistant.has_tool_calls() {
        for tc in assistant.tool_calls() {
            println!("dispatch: {} with {}", tc.name, tc.arguments);
        }
    } else {
        println!("plain answer: {}", assistant.content());
    }

    // For a "final answer" message, the same code falls through to
    // the else branch.
    let plain = Message::ai("Here is the answer.");
    if !plain.has_tool_calls() {
        println!("plain answer: {}", plain.content());
    }
}
