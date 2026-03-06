//! Tracer schemas and run types.
//!
//! Mirrors Python `langchain_core.tracers.schemas`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

/// The type of component that produced a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunType {
    Llm,
    ChatModel,
    Chain,
    Tool,
    Retriever,
}

/// Represents a single execution run in a trace.
///
/// A run captures the inputs, outputs, timing, and metadata for one
/// invocation of a component (LLM, chain, tool, retriever).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Run {
    /// Unique identifier for this run.
    pub id: Uuid,
    /// Human-readable name for the run.
    pub name: String,
    /// Type of component.
    pub run_type: RunType,
    /// Inputs provided to the component.
    #[serde(default)]
    pub inputs: Value,
    /// Outputs produced by the component (None if not yet complete).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outputs: Option<Value>,
    /// Error message if the run failed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Parent run ID for nested runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent_run_id: Option<Uuid>,
    /// Trace ID grouping related runs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trace_id: Option<Uuid>,
    /// Additional metadata.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub extra: HashMap<String, Value>,
    /// Serialized representation of the component.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serialized: Option<Value>,
    /// Child runs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub child_runs: Vec<Run>,
    /// Tags associated with this run.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tags: Vec<String>,
}

impl Run {
    pub fn new(id: Uuid, name: impl Into<String>, run_type: RunType, inputs: Value) -> Self {
        Self {
            id,
            name: name.into(),
            run_type,
            inputs,
            outputs: None,
            error: None,
            parent_run_id: None,
            trace_id: None,
            extra: HashMap::new(),
            serialized: None,
            child_runs: Vec::new(),
            tags: Vec::new(),
        }
    }
}
