//! Log stream tracer for streaming run logs.
//!
//! Mirrors Python `langchain_core.tracers.log_stream`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// A single entry in the run log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// The unique ID of this log entry's run.
    pub id: String,
    /// The name of the component.
    pub name: String,
    /// The type of run.
    pub run_type: String,
    /// Tags associated with this run.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Metadata about the run.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
    /// Start time as ISO string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start_time: Option<String>,
    /// End time as ISO string.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub end_time: Option<String>,
    /// Streamed output chunks.
    #[serde(default)]
    pub streamed_output: Vec<Value>,
    /// Streamed output string (for LLM output).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub streamed_output_str: Option<Vec<String>>,
    /// Final output.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_output: Option<Value>,
}

/// The state of a run, containing all log entries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RunState {
    /// The unique ID of the root run.
    pub id: String,
    /// All streamed output from the root run.
    #[serde(default)]
    pub streamed_output: Vec<Value>,
    /// Final output of the root run.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_output: Option<Value>,
    /// Log entries for all nested runs, keyed by run name.
    #[serde(default)]
    pub logs: HashMap<String, LogEntry>,
    /// The name of the root runnable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The type of the root runnable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_type: Option<String>,
}

/// A JSON Patch operation for incremental log updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonPatchOp {
    /// The operation type: "add", "replace", "remove".
    pub op: String,
    /// The JSON Pointer path to the target location.
    pub path: String,
    /// The value for add/replace operations.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Value>,
}

/// A collection of JSON Patch operations representing an incremental update.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunLogPatch {
    /// The list of patch operations.
    pub ops: Vec<JsonPatchOp>,
}

impl RunLogPatch {
    /// Create a new patch from a list of operations.
    pub fn new(ops: Vec<JsonPatchOp>) -> Self {
        Self { ops }
    }

    /// Create a patch that adds a new log entry.
    pub fn add_log_entry(name: &str, entry: &LogEntry) -> Self {
        Self::new(vec![JsonPatchOp {
            op: "add".to_string(),
            path: format!("/logs/{}", name),
            value: Some(serde_json::to_value(entry).unwrap_or_default()),
        }])
    }

    /// Create a patch that adds streamed output.
    pub fn add_streamed_output(index: usize, value: Value) -> Self {
        Self::new(vec![JsonPatchOp {
            op: "add".to_string(),
            path: format!("/streamed_output/{}", index),
            value: Some(value),
        }])
    }

    /// Create a patch that sets the final output.
    pub fn set_final_output(value: Value) -> Self {
        Self::new(vec![JsonPatchOp {
            op: "replace".to_string(),
            path: "/final_output".to_string(),
            value: Some(value),
        }])
    }
}
