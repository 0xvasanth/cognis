use std::collections::HashMap;
use std::sync::Arc;
use serde::{Deserialize, Serialize};
use serde_json::Value;

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
