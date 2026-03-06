use serde_json::{json, Value};
use super::{Message, ToolCall};

fn get_openai_role(msg: &Message) -> &'static str {
    match msg {
        Message::Human(_) | Message::HumanChunk(_) => "user",
        Message::Ai(_) | Message::AiChunk(_) => "assistant",
        Message::System(_) | Message::SystemChunk(_) => "system",
        Message::Tool(_) | Message::ToolChunk(_) => "tool",
        Message::Function(_) | Message::FunctionChunk(_) => "function",
        Message::Chat(_) | Message::ChatChunk(_) => "assistant",
        Message::Remove(_) => "system",
    }
}

fn convert_tool_calls(tool_calls: &[ToolCall]) -> Vec<Value> {
    tool_calls
        .iter()
        .map(|tc| {
            let args_str = serde_json::to_string(&tc.args).unwrap_or_default();
            json!({
                "type": "function",
                "id": tc.id.as_deref().unwrap_or(""),
                "function": { "name": tc.name, "arguments": args_str }
            })
        })
        .collect()
}

/// Convert LangChain messages to OpenAI-format message dicts.
pub fn convert_to_openai_messages(messages: &[Message]) -> Vec<Value> {
    messages
        .iter()
        .map(|msg| {
            let role = get_openai_role(msg);
            let content = msg.content().text();
            let mut result = json!({"role": role, "content": content});

            if let Message::Ai(ai) = msg {
                if !ai.tool_calls.is_empty() {
                    result["tool_calls"] = Value::Array(convert_tool_calls(&ai.tool_calls));
                }
            }
            if let Message::Tool(tool) = msg {
                result["tool_call_id"] = Value::String(tool.tool_call_id.clone());
            }
            if let Some(base) = msg.base() {
                if let Some(name) = &base.name {
                    result["name"] = Value::String(name.clone());
                }
            }
            result
        })
        .collect()
}

/// Count tokens approximately using character-based heuristic.
pub fn count_tokens_approximately(
    messages: &[Message],
    chars_per_token: f64,
    extra_tokens_per_message: f64,
) -> usize {
    let mut total = 0.0;
    for msg in messages {
        let text = msg.content().text();
        total += (text.len() as f64) / chars_per_token + extra_tokens_per_message;
        if let Message::Ai(ai) = msg {
            for tc in &ai.tool_calls {
                let args_str = serde_json::to_string(&tc.args).unwrap_or_default();
                total += (tc.name.len() as f64 + args_str.len() as f64) / chars_per_token;
            }
        }
    }
    total.ceil() as usize
}
