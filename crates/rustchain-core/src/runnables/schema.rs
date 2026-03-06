use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Data payload for a standard stream event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventData {
    /// The input to the runnable, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    /// The output of the runnable, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    /// A streaming chunk, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk: Option<Value>,
    /// An error message, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A stream event emitted during runnable execution.
///
/// Events follow the LangChain v2 streaming event protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StreamEvent {
    /// A standard event with structured `EventData`.
    Standard {
        event: String,
        name: String,
        run_id: String,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        metadata: HashMap<String, Value>,
        #[serde(default)]
        parent_ids: Vec<String>,
        data: EventData,
    },
    /// A custom event with arbitrary JSON data.
    Custom {
        event: String,
        name: String,
        run_id: String,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        metadata: HashMap<String, Value>,
        #[serde(default)]
        parent_ids: Vec<String>,
        data: Value,
    },
}
