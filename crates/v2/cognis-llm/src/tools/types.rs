//! Tool I/O types.
//!
//! `ToolInput` is the input the macro-generated `_run` body deserializes
//! from. `ToolOutput` is what the body returns. Both are kept simple
//! — the framework handles the boundary translation between LLM
//! `tool_calls` and these types.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Input to a tool's `_run` method.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolInput {
    /// Raw text — used when the model passes a single string blob.
    Text(String),
    /// Structured object — the typical case for typed tools.
    Structured(HashMap<String, serde_json::Value>),
    /// Pre-parsed tool call (carries id + name + arguments).
    ToolCall(crate::ToolCall),
}

impl ToolInput {
    /// Convert this input into a `serde_json::Value` for downstream
    /// deserialization. Used by macro-generated `_run` bodies.
    pub fn into_json(self) -> serde_json::Value {
        match self {
            ToolInput::Text(s) => serde_json::Value::String(s),
            ToolInput::Structured(map) => {
                let mut m = serde_json::Map::new();
                for (k, v) in map {
                    m.insert(k, v);
                }
                serde_json::Value::Object(m)
            }
            ToolInput::ToolCall(tc) => tc.arguments,
        }
    }
}

/// Output from a tool's `_run` method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ToolOutput {
    /// Structured JSON content — the common case.
    Content(serde_json::Value),
    /// Plain string output.
    Text(String),
    /// An empty result (tool succeeded but has nothing to report).
    Empty,
}

impl ToolOutput {
    /// Render the output as a string for the LLM tool-result message.
    pub fn as_string(&self) -> String {
        match self {
            ToolOutput::Content(v) => v.to_string(),
            ToolOutput::Text(s) => s.clone(),
            ToolOutput::Empty => String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn text_input_into_json() {
        assert_eq!(ToolInput::Text("hi".into()).into_json(), json!("hi"));
    }

    #[test]
    fn structured_input_into_json() {
        let mut m = HashMap::new();
        m.insert("a".into(), json!(1));
        m.insert("b".into(), json!("two"));
        let v = ToolInput::Structured(m).into_json();
        assert_eq!(v["a"], json!(1));
        assert_eq!(v["b"], json!("two"));
    }

    #[test]
    fn output_as_string_variants() {
        assert_eq!(ToolOutput::Content(json!(42)).as_string(), "42");
        assert_eq!(ToolOutput::Text("hello".into()).as_string(), "hello");
        assert_eq!(ToolOutput::Empty.as_string(), "");
    }
}
