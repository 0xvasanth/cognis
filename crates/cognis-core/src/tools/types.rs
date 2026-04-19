use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ToolInput {
    Text(String),
    ToolCall(ToolCallInput),
    Structured(HashMap<String, Value>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallInput {
    pub id: String,
    pub name: String,
    pub args: HashMap<String, Value>,
}

impl ToolInput {
    /// Lower a `ToolInput` into a single `serde_json::Value` suitable for
    /// deserialization into a typed args struct (used by
    /// `#[cognis::tool]`-generated `_run` methods).
    ///
    /// - `Text(s)` → `Value::String(s)`
    /// - `Structured(map)` → `Value::Object(map)`
    /// - `ToolCall(call)` → `Value::Object(call.args)` (the `id` / `name`
    ///   fields are dropped — callers already have them via the surrounding
    ///   context)
    pub fn into_json(self) -> Value {
        match self {
            ToolInput::Text(s) => Value::String(s),
            ToolInput::Structured(m) => Value::Object(m.into_iter().collect()),
            ToolInput::ToolCall(call) => Value::Object(call.args.into_iter().collect()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResponseFormat {
    #[default]
    Content,
    ContentAndArtifact,
}

#[derive(Debug, Clone)]
pub enum ToolOutput {
    Content(Value),
    ContentAndArtifact { content: Value, artifact: Value },
}

#[derive(Clone, Default)]
pub enum ErrorHandler {
    #[default]
    Propagate,
    DefaultMessage,
    StaticMessage(String),
    Dynamic(Arc<dyn Fn(&str) -> String + Send + Sync>),
}

impl std::fmt::Debug for ErrorHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Propagate => write!(f, "Propagate"),
            Self::DefaultMessage => write!(f, "DefaultMessage"),
            Self::StaticMessage(s) => write!(f, "StaticMessage({:?})", s),
            Self::Dynamic(_) => write!(f, "Dynamic(...)"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn into_json_text_becomes_string() {
        let v = ToolInput::Text("hello".into()).into_json();
        assert_eq!(v, json!("hello"));
    }

    #[test]
    fn into_json_structured_becomes_object() {
        let mut m = HashMap::new();
        m.insert("k".to_string(), json!(1));
        let v = ToolInput::Structured(m).into_json();
        assert_eq!(v, json!({ "k": 1 }));
    }

    #[test]
    fn into_json_tool_call_uses_args() {
        let v = ToolInput::ToolCall(ToolCallInput {
            id: "1".into(),
            name: "x".into(),
            args: HashMap::from([("a".to_string(), json!(2))]),
        })
        .into_json();
        assert_eq!(v, json!({ "a": 2 }));
    }
}
