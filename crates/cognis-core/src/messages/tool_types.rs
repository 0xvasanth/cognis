use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

/// Represents an AI's request to call a tool.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub args: HashMap<String, Value>,
    pub id: Option<String>,
}

/// A tool call that failed to parse.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InvalidToolCall {
    pub name: Option<String>,
    pub args: Option<String>,
    pub id: Option<String>,
    pub error: Option<String>,
}

/// A chunk of a tool call (yielded when streaming).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCallChunk {
    pub name: Option<String>,
    pub args: Option<String>,
    pub id: Option<String>,
    pub index: Option<usize>,
}

/// Create a ToolCall with validated fields.
pub fn tool_call(
    name: impl Into<String>,
    args: HashMap<String, Value>,
    id: Option<String>,
) -> ToolCall {
    ToolCall {
        name: name.into(),
        args,
        id,
    }
}

/// Create a ToolCallChunk.
pub fn tool_call_chunk(
    name: Option<String>,
    args: Option<String>,
    id: Option<String>,
    index: Option<usize>,
) -> ToolCallChunk {
    ToolCallChunk {
        name,
        args,
        id,
        index,
    }
}

/// Create an InvalidToolCall.
pub fn invalid_tool_call(
    name: Option<String>,
    args: Option<String>,
    id: Option<String>,
    error: Option<String>,
) -> InvalidToolCall {
    InvalidToolCall {
        name,
        args,
        id,
        error,
    }
}

/// Parse raw tool call dicts into validated ToolCalls and InvalidToolCalls.
pub fn default_tool_parser(raw_tool_calls: &[Value]) -> (Vec<ToolCall>, Vec<InvalidToolCall>) {
    let mut valid = Vec::new();
    let mut invalid = Vec::new();
    for raw in raw_tool_calls {
        let name = raw
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let id = raw
            .get("id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        match raw.get("args") {
            Some(Value::Object(map)) => {
                let args: HashMap<String, Value> =
                    map.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
                valid.push(ToolCall { name, args, id });
            }
            Some(Value::String(s)) => match serde_json::from_str::<HashMap<String, Value>>(s) {
                Ok(args) => valid.push(ToolCall { name, args, id }),
                Err(e) => invalid.push(InvalidToolCall {
                    name: Some(name),
                    args: Some(s.clone()),
                    id,
                    error: Some(e.to_string()),
                }),
            },
            _ => {
                invalid.push(InvalidToolCall {
                    name: Some(name),
                    args: raw.get("args").map(|v| v.to_string()),
                    id,
                    error: Some("args is not an object or string".into()),
                });
            }
        }
    }
    (valid, invalid)
}

/// Parse raw streaming tool call dicts into ToolCallChunks.
pub fn default_tool_chunk_parser(raw_tool_calls: &[Value]) -> Vec<ToolCallChunk> {
    raw_tool_calls
        .iter()
        .map(|raw| ToolCallChunk {
            name: raw
                .get("name")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            args: raw
                .get("args")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            id: raw
                .get("id")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string()),
            index: raw
                .get("index")
                .and_then(|v| v.as_u64())
                .map(|n| n as usize),
        })
        .collect()
}
