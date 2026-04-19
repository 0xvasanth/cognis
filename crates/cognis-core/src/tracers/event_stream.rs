//! Event stream tracer for async event-based tracing.
//!
//! Mirrors Python `langchain_core.tracers.event_stream`.
//!
//! Provides an [`EventStreamCallbackHandler`] that collects callback events
//! into an async channel, allowing consumers to receive a stream of
//! [`StreamEvent`] values as a run progresses.

use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::agents::{AgentAction, AgentFinish};
use crate::callbacks::{CallbackHandler, ToolEndEvent, ToolErrorEvent, ToolStartEvent};
use crate::documents::Document;
use crate::error::Result;
use crate::messages::Message;
use crate::outputs::LLMResult;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// The type of event emitted during a run.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    /// LLM (non-chat) started.
    #[serde(rename = "on_llm_start")]
    OnLlmStart,
    /// LLM streaming token.
    #[serde(rename = "on_llm_stream")]
    OnLlmStream,
    /// LLM ended.
    #[serde(rename = "on_llm_end")]
    OnLlmEnd,
    /// LLM error.
    #[serde(rename = "on_llm_error")]
    OnLlmError,
    /// Chat model started.
    #[serde(rename = "on_chat_model_start")]
    OnChatModelStart,
    /// Chat model streaming token.
    #[serde(rename = "on_chat_model_stream")]
    OnChatModelStream,
    /// Chat model ended.
    #[serde(rename = "on_chat_model_end")]
    OnChatModelEnd,
    /// Chain started.
    #[serde(rename = "on_chain_start")]
    OnChainStart,
    /// Chain streaming chunk.
    #[serde(rename = "on_chain_stream")]
    OnChainStream,
    /// Chain ended.
    #[serde(rename = "on_chain_end")]
    OnChainEnd,
    /// Chain error.
    #[serde(rename = "on_chain_error")]
    OnChainError,
    /// Tool started.
    #[serde(rename = "on_tool_start")]
    OnToolStart,
    /// Tool ended.
    #[serde(rename = "on_tool_end")]
    OnToolEnd,
    /// Tool error.
    #[serde(rename = "on_tool_error")]
    OnToolError,
    /// Retriever started.
    #[serde(rename = "on_retriever_start")]
    OnRetrieverStart,
    /// Retriever ended.
    #[serde(rename = "on_retriever_end")]
    OnRetrieverEnd,
    /// Retriever error.
    #[serde(rename = "on_retriever_error")]
    OnRetrieverError,
    /// Agent decided to take an action (invoke a tool).
    ///
    /// Agent lifecycle (start/end) is represented via the enclosing
    /// `OnChainStart`/`OnChainEnd` events on the agent executor itself,
    /// so dedicated `OnAgentStart`/`OnAgentEnd` variants are intentionally
    /// omitted.
    #[serde(rename = "on_agent_action")]
    OnAgentAction,
    /// Agent finished (produced final answer).
    #[serde(rename = "on_agent_finish")]
    OnAgentFinish,
    /// Custom event.
    #[serde(rename = "on_custom_event")]
    OnCustomEvent,
}

impl std::fmt::Display for EventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = serde_json::to_value(self)
            .ok()
            .and_then(|v| v.as_str().map(String::from))
            .unwrap_or_else(|| format!("{:?}", self));
        write!(f, "{}", s)
    }
}

/// Data payload for a streaming event.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EventData {
    /// Input data for the event (present on start/end events).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<Value>,
    /// Output data for the event (present on end events).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output: Option<Value>,
    /// Streaming chunk (present on stream events).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chunk: Option<Value>,
    /// Structured artifact payload from tools (present on tool end events).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub artifact: Option<Value>,
    /// Error description (present on error events).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// A single streaming event emitted by the event stream tracer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    /// The kind of event.
    pub event: EventType,
    /// Human-readable name of the component.
    pub name: String,
    /// Data payload.
    pub data: EventData,
    /// Unique run identifier.
    pub run_id: String,
    /// Parent run IDs from root to immediate parent.
    #[serde(default)]
    pub parent_ids: Vec<String>,
    /// Tags associated with this run.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Metadata associated with this run.
    #[serde(default)]
    pub metadata: HashMap<String, Value>,
}

/// Information about an active run, kept in the handler's run map.
#[derive(Debug, Clone)]
pub struct RunInfo {
    /// Human-readable name.
    pub name: String,
    /// Tags associated with the run.
    pub tags: Vec<String>,
    /// Metadata associated with the run.
    pub metadata: HashMap<String, Value>,
    /// The run type string (e.g. "llm", "chat_model", "chain", "tool", "retriever").
    pub run_type: String,
    /// Parent run ID, if any.
    pub parent_run_id: Option<Uuid>,
    /// Inputs to the run (optional).
    pub inputs: Option<Value>,
}

// ---------------------------------------------------------------------------
// Root event filter
// ---------------------------------------------------------------------------

/// Filters which events are sent through the stream based on name, type, and
/// tag inclusion/exclusion lists.
#[derive(Debug, Clone, Default)]
pub struct RootEventFilter {
    /// Only include events whose name is in this list (if non-empty).
    pub include_names: Vec<String>,
    /// Only include events whose run type is in this list (if non-empty).
    pub include_types: Vec<String>,
    /// Only include events that carry at least one of these tags (if non-empty).
    pub include_tags: Vec<String>,
    /// Exclude events whose name is in this list.
    pub exclude_names: Vec<String>,
    /// Exclude events whose run type is in this list.
    pub exclude_types: Vec<String>,
    /// Exclude events that carry any of these tags.
    pub exclude_tags: Vec<String>,
}

impl RootEventFilter {
    /// Returns `true` if the event should be included in the stream.
    pub fn include_event(&self, event: &StreamEvent, run_type: &str) -> bool {
        // Exclusion checks
        if !self.exclude_names.is_empty() && self.exclude_names.contains(&event.name) {
            return false;
        }
        if !self.exclude_types.is_empty() && self.exclude_types.contains(&run_type.to_string()) {
            return false;
        }
        if !self.exclude_tags.is_empty() && event.tags.iter().any(|t| self.exclude_tags.contains(t))
        {
            return false;
        }

        // If no inclusion filters are set, include everything that survived
        // the exclusion checks.
        let has_include_filter = !self.include_names.is_empty()
            || !self.include_types.is_empty()
            || !self.include_tags.is_empty();

        if !has_include_filter {
            return true;
        }

        // Inclusion checks — the event must match at least one of the active
        // inclusion filters.
        if !self.include_names.is_empty() && self.include_names.contains(&event.name) {
            return true;
        }
        if !self.include_types.is_empty() && self.include_types.contains(&run_type.to_string()) {
            return true;
        }
        if !self.include_tags.is_empty() && event.tags.iter().any(|t| self.include_tags.contains(t))
        {
            return true;
        }

        false
    }
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Resolve a human-readable name from the explicit `name` parameter or the
/// serialized representation.
fn assign_name(name: Option<&str>, serialized: &Value) -> String {
    if let Some(n) = name {
        return n.to_string();
    }
    if let Some(obj) = serialized.as_object() {
        if let Some(Value::String(n)) = obj.get("name") {
            return n.clone();
        }
        if let Some(Value::Array(ids)) = obj.get("id") {
            if let Some(Value::String(last)) = ids.last() {
                return last.clone();
            }
        }
    }
    "Unnamed".to_string()
}

// ---------------------------------------------------------------------------
// EventStreamCallbackHandler
// ---------------------------------------------------------------------------

/// Callback handler that collects events into an async mpsc channel.
///
/// Consumers receive [`StreamEvent`] values through the
/// [`mpsc::Receiver`](tokio::sync::mpsc::Receiver) returned by
/// [`take_receiver`](EventStreamCallbackHandler::take_receiver).
///
/// This is the Rust equivalent of the Python
/// `_AstreamEventsCallbackHandler`.
pub struct EventStreamCallbackHandler {
    /// Map from run ID to run metadata.
    run_map: Mutex<HashMap<Uuid, RunInfo>>,
    /// Map from child run ID to parent run ID.
    parent_map: Mutex<HashMap<Uuid, Option<Uuid>>>,
    /// Sender half of the event channel.
    sender: mpsc::Sender<StreamEvent>,
    /// Receiver half — taken once by the consumer via [`take_receiver`].
    receiver: Mutex<Option<mpsc::Receiver<StreamEvent>>>,
    /// Event filter.
    filter: RootEventFilter,
}

impl EventStreamCallbackHandler {
    /// Create a new handler with the given channel buffer size and optional
    /// event filter.
    pub fn new(buffer: usize, filter: RootEventFilter) -> Self {
        let (tx, rx) = mpsc::channel(buffer);
        Self {
            run_map: Mutex::new(HashMap::new()),
            parent_map: Mutex::new(HashMap::new()),
            sender: tx,
            receiver: Mutex::new(Some(rx)),
            filter,
        }
    }

    /// Create a handler with default buffer (256) and no filters.
    pub fn with_defaults() -> Self {
        Self::new(256, RootEventFilter::default())
    }

    /// Take the receiver half of the event channel.
    ///
    /// This can only be called once. Subsequent calls return `None`.
    pub fn take_receiver(&self) -> Option<mpsc::Receiver<StreamEvent>> {
        self.receiver.lock().unwrap().take()
    }

    /// Build the parent-ID chain for `run_id`, from root to immediate parent.
    fn get_parent_ids(&self, run_id: Uuid) -> Vec<String> {
        let parent_map = self.parent_map.lock().unwrap();
        let mut ids = Vec::new();
        let mut current = run_id;
        while let Some(Some(parent)) = parent_map.get(&current) {
            let s = parent.to_string();
            if ids.contains(&s) {
                // Cycle detected — should not happen.
                break;
            }
            ids.push(s);
            current = *parent;
        }
        ids.reverse();
        ids
    }

    /// Send an event through the channel if it passes the filter.
    fn send_event(&self, event: StreamEvent, run_type: &str) {
        if self.filter.include_event(&event, run_type) {
            // Best-effort send; if the receiver is dropped we silently drop.
            let _ = self.sender.try_send(event);
        }
    }

    /// Record run metadata at the start of a run.
    #[allow(clippy::too_many_arguments)]
    fn write_run_start_info(
        &self,
        run_id: Uuid,
        name: String,
        run_type: String,
        tags: Vec<String>,
        metadata: HashMap<String, Value>,
        parent_run_id: Option<Uuid>,
        inputs: Option<Value>,
    ) {
        let info = RunInfo {
            name,
            tags,
            metadata,
            run_type,
            parent_run_id,
            inputs,
        };
        self.run_map.lock().unwrap().insert(run_id, info);
        self.parent_map
            .lock()
            .unwrap()
            .insert(run_id, parent_run_id);
    }

    /// Remove and return the `RunInfo` for a completed run.
    fn pop_run_info(&self, run_id: Uuid) -> Option<RunInfo> {
        self.run_map.lock().unwrap().remove(&run_id)
    }
}

// ---------------------------------------------------------------------------
// CallbackHandler implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl CallbackHandler for EventStreamCallbackHandler {
    async fn on_llm_start(
        &self,
        serialized: &Value,
        prompts: &[String],
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let name = assign_name(None, serialized);
        let run_type = "llm".to_string();
        let inputs = serde_json::to_value(prompts).ok();

        self.write_run_start_info(
            run_id,
            name.clone(),
            run_type.clone(),
            Vec::new(),
            HashMap::new(),
            parent_run_id,
            inputs.clone(),
        );

        let event = StreamEvent {
            event: EventType::OnLlmStart,
            name,
            data: EventData {
                input: inputs,
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: Vec::new(),
            metadata: HashMap::new(),
        };
        self.send_event(event, &run_type);
        Ok(())
    }

    async fn on_llm_new_token(
        &self,
        token: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let run_map = self.run_map.lock().unwrap();
        let run_info = match run_map.get(&run_id) {
            Some(info) => info.clone(),
            None => return Ok(()),
        };
        drop(run_map);

        let (event_type, chunk_value) = if run_info.run_type == "chat_model" {
            (
                EventType::OnChatModelStream,
                Value::String(token.to_string()),
            )
        } else {
            (EventType::OnLlmStream, Value::String(token.to_string()))
        };

        let event = StreamEvent {
            event: event_type,
            name: run_info.name,
            data: EventData {
                chunk: Some(chunk_value),
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: run_info.tags,
            metadata: run_info.metadata,
        };
        self.send_event(event, &run_info.run_type);
        Ok(())
    }

    async fn on_llm_end(
        &self,
        response: &LLMResult,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let run_info = match self.pop_run_info(run_id) {
            Some(info) => info,
            None => return Ok(()),
        };

        let event_type = if run_info.run_type == "chat_model" {
            EventType::OnChatModelEnd
        } else {
            EventType::OnLlmEnd
        };

        let output = serde_json::to_value(response).unwrap_or(Value::Null);

        let event = StreamEvent {
            event: event_type,
            name: run_info.name,
            data: EventData {
                output: Some(output),
                input: run_info.inputs,
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: run_info.tags,
            metadata: run_info.metadata,
        };
        self.send_event(event, &run_info.run_type);
        Ok(())
    }

    async fn on_llm_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let run_info = match self.pop_run_info(run_id) {
            Some(info) => info,
            None => return Ok(()),
        };

        let event = StreamEvent {
            event: EventType::OnLlmError,
            name: run_info.name,
            data: EventData {
                error: Some(error.to_string()),
                input: run_info.inputs,
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: run_info.tags,
            metadata: run_info.metadata,
        };
        self.send_event(event, &run_info.run_type);
        Ok(())
    }

    async fn on_chat_model_start(
        &self,
        serialized: &Value,
        messages: &[Vec<Message>],
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let name = assign_name(None, serialized);
        let run_type = "chat_model".to_string();
        let inputs = serde_json::to_value(messages).ok();

        self.write_run_start_info(
            run_id,
            name.clone(),
            run_type.clone(),
            Vec::new(),
            HashMap::new(),
            parent_run_id,
            inputs.clone(),
        );

        let event = StreamEvent {
            event: EventType::OnChatModelStart,
            name,
            data: EventData {
                input: inputs,
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: Vec::new(),
            metadata: HashMap::new(),
        };
        self.send_event(event, &run_type);
        Ok(())
    }

    async fn on_chain_start(
        &self,
        serialized: &Value,
        inputs: &Value,
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let name = assign_name(None, serialized);
        let run_type = "chain".to_string();

        self.write_run_start_info(
            run_id,
            name.clone(),
            run_type.clone(),
            Vec::new(),
            HashMap::new(),
            parent_run_id,
            Some(inputs.clone()),
        );

        let event = StreamEvent {
            event: EventType::OnChainStart,
            name,
            data: EventData {
                input: Some(inputs.clone()),
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: Vec::new(),
            metadata: HashMap::new(),
        };
        self.send_event(event, &run_type);
        Ok(())
    }

    async fn on_chain_end(
        &self,
        outputs: &Value,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let run_info = match self.pop_run_info(run_id) {
            Some(info) => info,
            None => return Ok(()),
        };

        let event = StreamEvent {
            event: EventType::OnChainEnd,
            name: run_info.name,
            data: EventData {
                output: Some(outputs.clone()),
                input: run_info.inputs,
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: run_info.tags,
            metadata: run_info.metadata,
        };
        self.send_event(event, &run_info.run_type);
        Ok(())
    }

    async fn on_chain_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let run_info = match self.pop_run_info(run_id) {
            Some(info) => info,
            None => return Ok(()),
        };

        let event = StreamEvent {
            event: EventType::OnChainError,
            name: run_info.name,
            data: EventData {
                error: Some(error.to_string()),
                input: run_info.inputs,
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: run_info.tags,
            metadata: run_info.metadata,
        };
        self.send_event(event, &run_info.run_type);
        Ok(())
    }

    async fn on_tool_start(&self, event: ToolStartEvent) -> Result<()> {
        let name = assign_name(None, &event.serialized);
        let run_type = "tool".to_string();
        let inputs = Some(Value::String(event.input_str.clone()));

        self.write_run_start_info(
            event.run_id,
            name.clone(),
            run_type.clone(),
            Vec::new(),
            HashMap::new(),
            event.parent_run_id,
            inputs.clone(),
        );

        let stream_event = StreamEvent {
            event: EventType::OnToolStart,
            name,
            data: EventData {
                input: inputs,
                ..Default::default()
            },
            run_id: event.run_id.to_string(),
            parent_ids: self.get_parent_ids(event.run_id),
            tags: Vec::new(),
            metadata: HashMap::new(),
        };
        self.send_event(stream_event, &run_type);
        Ok(())
    }

    async fn on_tool_end(&self, event: ToolEndEvent) -> Result<()> {
        let run_info = match self.pop_run_info(event.run_id) {
            Some(info) => info,
            None => return Ok(()),
        };

        let stream_event = StreamEvent {
            event: EventType::OnToolEnd,
            name: run_info.name,
            data: EventData {
                output: Some(event.output_value),
                artifact: event.artifact,
                input: run_info.inputs,
                ..Default::default()
            },
            run_id: event.run_id.to_string(),
            parent_ids: self.get_parent_ids(event.run_id),
            tags: run_info.tags,
            metadata: run_info.metadata,
        };
        self.send_event(stream_event, &run_info.run_type);
        Ok(())
    }

    async fn on_tool_error(&self, event: ToolErrorEvent) -> Result<()> {
        let run_info = match self.pop_run_info(event.run_id) {
            Some(info) => info,
            None => return Ok(()),
        };

        let stream_event = StreamEvent {
            event: EventType::OnToolError,
            name: run_info.name,
            data: EventData {
                error: Some(event.error.clone()),
                input: run_info.inputs,
                ..Default::default()
            },
            run_id: event.run_id.to_string(),
            parent_ids: self.get_parent_ids(event.run_id),
            tags: run_info.tags,
            metadata: run_info.metadata,
        };
        self.send_event(stream_event, &run_info.run_type);
        Ok(())
    }

    async fn on_agent_action(
        &self,
        action: &AgentAction,
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        // Record parent link so get_parent_ids() works.
        {
            let mut parent_map = self.parent_map.lock().unwrap();
            parent_map.insert(run_id, parent_run_id);
        }
        let event = StreamEvent {
            event: EventType::OnAgentAction,
            name: action.tool.clone(),
            data: EventData {
                input: Some(action.tool_input.clone()),
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: Vec::new(),
            metadata: HashMap::new(),
        };
        self.send_event(event, "agent");
        Ok(())
    }

    async fn on_agent_finish(
        &self,
        finish: &AgentFinish,
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        {
            let mut parent_map = self.parent_map.lock().unwrap();
            parent_map.insert(run_id, parent_run_id);
        }
        let output_value =
            serde_json::to_value(&finish.return_values).unwrap_or(serde_json::Value::Null);
        let event = StreamEvent {
            event: EventType::OnAgentFinish,
            name: "AgentExecutor".to_string(),
            data: EventData {
                output: Some(output_value),
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: Vec::new(),
            metadata: HashMap::new(),
        };
        self.send_event(event, "agent");
        Ok(())
    }

    async fn on_retriever_start(
        &self,
        serialized: &Value,
        query: &str,
        run_id: Uuid,
        parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let name = assign_name(None, serialized);
        let run_type = "retriever".to_string();
        let inputs = Some(Value::String(query.to_string()));

        self.write_run_start_info(
            run_id,
            name.clone(),
            run_type.clone(),
            Vec::new(),
            HashMap::new(),
            parent_run_id,
            inputs.clone(),
        );

        let event = StreamEvent {
            event: EventType::OnRetrieverStart,
            name,
            data: EventData {
                input: inputs,
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: Vec::new(),
            metadata: HashMap::new(),
        };
        self.send_event(event, &run_type);
        Ok(())
    }

    async fn on_retriever_end(
        &self,
        documents: &[Document],
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let run_info = match self.pop_run_info(run_id) {
            Some(info) => info,
            None => return Ok(()),
        };

        let output = serde_json::to_value(documents).unwrap_or(Value::Null);

        let event = StreamEvent {
            event: EventType::OnRetrieverEnd,
            name: run_info.name,
            data: EventData {
                output: Some(output),
                input: run_info.inputs,
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: run_info.tags,
            metadata: run_info.metadata,
        };
        self.send_event(event, &run_info.run_type);
        Ok(())
    }

    async fn on_retriever_error(
        &self,
        error: &str,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let run_info = match self.pop_run_info(run_id) {
            Some(info) => info,
            None => return Ok(()),
        };

        let event = StreamEvent {
            event: EventType::OnRetrieverError,
            name: run_info.name,
            data: EventData {
                error: Some(error.to_string()),
                input: run_info.inputs,
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: run_info.tags,
            metadata: run_info.metadata,
        };
        self.send_event(event, &run_info.run_type);
        Ok(())
    }

    async fn on_custom_event(
        &self,
        name: &str,
        data: &Value,
        run_id: Uuid,
        _parent_run_id: Option<Uuid>,
    ) -> Result<()> {
        let event = StreamEvent {
            event: EventType::OnCustomEvent,
            name: name.to_string(),
            data: EventData {
                input: Some(data.clone()),
                ..Default::default()
            },
            run_id: run_id.to_string(),
            parent_ids: self.get_parent_ids(run_id),
            tags: Vec::new(),
            metadata: HashMap::new(),
        };
        self.send_event(event, name);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_serialized(name: &str) -> Value {
        json!({"name": name})
    }

    fn start_event(
        serialized: &Value,
        input: &str,
        run_id: Uuid,
        parent: Option<Uuid>,
    ) -> ToolStartEvent {
        ToolStartEvent {
            tool: serialized
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string(),
            serialized: serialized.clone(),
            input_str: input.to_string(),
            inputs: Value::Null,
            tool_call_id: None,
            run_id,
            parent_run_id: parent,
            tags: vec![],
            metadata: HashMap::new(),
        }
    }

    fn end_event(out: &str, run_id: Uuid) -> ToolEndEvent {
        ToolEndEvent {
            tool: "".into(),
            output_str: out.into(),
            output_value: Value::String(out.into()),
            artifact: None,
            tool_call_id: None,
            run_id,
            parent_run_id: None,
        }
    }

    fn error_event(err: &str, run_id: Uuid) -> ToolErrorEvent {
        ToolErrorEvent {
            tool: "".into(),
            error: err.into(),
            error_kind: crate::callbacks::ToolErrorKind::Execution,
            tool_call_id: None,
            run_id,
            parent_run_id: None,
        }
    }

    #[tokio::test]
    async fn test_llm_start_end_events() {
        let handler = EventStreamCallbackHandler::with_defaults();
        let mut rx = handler.take_receiver().unwrap();
        let run_id = Uuid::new_v4();

        handler
            .on_llm_start(
                &make_serialized("my-llm"),
                &["Hello".to_string()],
                run_id,
                None,
            )
            .await
            .unwrap();

        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnLlmStart);
        assert_eq!(evt.name, "my-llm");
        assert_eq!(evt.run_id, run_id.to_string());
        assert!(evt.data.input.is_some());

        let response = LLMResult {
            generations: vec![],
            llm_output: None,
            run: None,
        };
        handler.on_llm_end(&response, run_id, None).await.unwrap();

        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnLlmEnd);
        assert_eq!(evt.name, "my-llm");
    }

    #[tokio::test]
    async fn test_chain_start_end_events() {
        let handler = EventStreamCallbackHandler::with_defaults();
        let mut rx = handler.take_receiver().unwrap();
        let run_id = Uuid::new_v4();

        handler
            .on_chain_start(
                &make_serialized("my-chain"),
                &json!({"key": "val"}),
                run_id,
                None,
            )
            .await
            .unwrap();

        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnChainStart);
        assert_eq!(evt.name, "my-chain");
        assert_eq!(evt.data.input, Some(json!({"key": "val"})));

        handler
            .on_chain_end(&json!({"result": 42}), run_id, None)
            .await
            .unwrap();

        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnChainEnd);
        assert_eq!(evt.data.output, Some(json!({"result": 42})));
        // Input should be echoed back on end events.
        assert_eq!(evt.data.input, Some(json!({"key": "val"})));
    }

    #[tokio::test]
    async fn test_tool_start_end_events() {
        let handler = EventStreamCallbackHandler::with_defaults();
        let mut rx = handler.take_receiver().unwrap();
        let run_id = Uuid::new_v4();

        handler
            .on_tool_start(start_event(
                &make_serialized("my-tool"),
                "tool input",
                run_id,
                None,
            ))
            .await
            .unwrap();

        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnToolStart);
        assert_eq!(evt.name, "my-tool");

        handler
            .on_tool_end(end_event("tool output", run_id))
            .await
            .unwrap();

        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnToolEnd);
        assert_eq!(
            evt.data.output,
            Some(Value::String("tool output".to_string()))
        );
    }

    #[tokio::test]
    async fn on_tool_end_preserves_structured_value() {
        let handler = EventStreamCallbackHandler::with_defaults();
        let mut rx = handler.take_receiver().unwrap();
        let run_id = Uuid::new_v4();

        handler
            .on_tool_start(ToolStartEvent {
                tool: "search".into(),
                serialized: json!({"name": "search"}),
                input_str: "\"q\"".into(),
                inputs: json!("q"),
                tool_call_id: None,
                run_id,
                parent_run_id: None,
                tags: vec![],
                metadata: HashMap::new(),
            })
            .await
            .unwrap();

        let typed = json!({"hits": [1, 2, 3]});
        handler
            .on_tool_end(ToolEndEvent {
                tool: "search".into(),
                output_str: typed.to_string(),
                output_value: typed.clone(),
                artifact: Some(json!({"source": "cache"})),
                tool_call_id: None,
                run_id,
                parent_run_id: None,
            })
            .await
            .unwrap();

        // Drain the start event; the end event is what matters.
        let _ = rx.recv().await.unwrap();
        let end_event = rx.recv().await.unwrap();
        assert_eq!(end_event.event, EventType::OnToolEnd);
        assert_eq!(end_event.data.output, Some(typed));
        assert_eq!(end_event.data.artifact, Some(json!({"source": "cache"})));
    }

    #[tokio::test]
    async fn test_retriever_start_end_events() {
        let handler = EventStreamCallbackHandler::with_defaults();
        let mut rx = handler.take_receiver().unwrap();
        let run_id = Uuid::new_v4();

        handler
            .on_retriever_start(
                &make_serialized("my-retriever"),
                "search query",
                run_id,
                None,
            )
            .await
            .unwrap();

        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnRetrieverStart);
        assert_eq!(evt.name, "my-retriever");

        let docs = vec![Document {
            page_content: "result".to_string(),
            id: None,
            metadata: HashMap::new(),
            doc_type: None,
        }];
        handler.on_retriever_end(&docs, run_id, None).await.unwrap();

        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnRetrieverEnd);
        assert!(evt.data.output.is_some());
    }

    #[tokio::test]
    async fn test_llm_new_token_stream_event() {
        let handler = EventStreamCallbackHandler::with_defaults();
        let mut rx = handler.take_receiver().unwrap();
        let run_id = Uuid::new_v4();

        // Must start the run first so token events can find the run info.
        handler
            .on_llm_start(
                &make_serialized("streamer"),
                &["prompt".to_string()],
                run_id,
                None,
            )
            .await
            .unwrap();
        let _ = rx.try_recv(); // consume start event

        handler
            .on_llm_new_token("hello", run_id, None)
            .await
            .unwrap();

        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnLlmStream);
        assert_eq!(evt.data.chunk, Some(Value::String("hello".to_string())));
    }

    #[tokio::test]
    async fn test_chat_model_stream_event() {
        let handler = EventStreamCallbackHandler::with_defaults();
        let mut rx = handler.take_receiver().unwrap();
        let run_id = Uuid::new_v4();

        handler
            .on_chat_model_start(&make_serialized("chat"), &[], run_id, None)
            .await
            .unwrap();
        let _ = rx.try_recv(); // consume start event

        handler
            .on_llm_new_token("world", run_id, None)
            .await
            .unwrap();

        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnChatModelStream);
    }

    #[tokio::test]
    async fn test_error_events() {
        let handler = EventStreamCallbackHandler::with_defaults();
        let mut rx = handler.take_receiver().unwrap();

        // LLM error
        let run_id = Uuid::new_v4();
        handler
            .on_llm_start(&make_serialized("err-llm"), &[], run_id, None)
            .await
            .unwrap();
        let _ = rx.try_recv();
        handler
            .on_llm_error("llm failed", run_id, None)
            .await
            .unwrap();
        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnLlmError);
        assert_eq!(evt.data.error, Some("llm failed".to_string()));

        // Chain error
        let run_id = Uuid::new_v4();
        handler
            .on_chain_start(&make_serialized("err-chain"), &json!({}), run_id, None)
            .await
            .unwrap();
        let _ = rx.try_recv();
        handler
            .on_chain_error("chain failed", run_id, None)
            .await
            .unwrap();
        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnChainError);

        // Tool error
        let run_id = Uuid::new_v4();
        handler
            .on_tool_start(start_event(&make_serialized("err-tool"), "", run_id, None))
            .await
            .unwrap();
        let _ = rx.try_recv();
        handler
            .on_tool_error(error_event("tool failed", run_id))
            .await
            .unwrap();
        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnToolError);

        // Retriever error
        let run_id = Uuid::new_v4();
        handler
            .on_retriever_start(&make_serialized("err-ret"), "", run_id, None)
            .await
            .unwrap();
        let _ = rx.try_recv();
        handler
            .on_retriever_error("ret failed", run_id, None)
            .await
            .unwrap();
        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnRetrieverError);
    }

    #[tokio::test]
    async fn test_parent_ids_chain() {
        let handler = EventStreamCallbackHandler::with_defaults();
        let mut rx = handler.take_receiver().unwrap();

        let root_id = Uuid::new_v4();
        let child_id = Uuid::new_v4();
        let grandchild_id = Uuid::new_v4();

        // Root chain
        handler
            .on_chain_start(&make_serialized("root"), &json!({}), root_id, None)
            .await
            .unwrap();
        let _ = rx.try_recv();

        // Child chain
        handler
            .on_chain_start(
                &make_serialized("child"),
                &json!({}),
                child_id,
                Some(root_id),
            )
            .await
            .unwrap();
        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.parent_ids, vec![root_id.to_string()]);

        // Grandchild tool
        handler
            .on_tool_start(start_event(
                &make_serialized("grandchild"),
                "",
                grandchild_id,
                Some(child_id),
            ))
            .await
            .unwrap();
        let evt = rx.try_recv().unwrap();
        assert_eq!(
            evt.parent_ids,
            vec![root_id.to_string(), child_id.to_string()]
        );
    }

    #[tokio::test]
    async fn test_custom_event() {
        let handler = EventStreamCallbackHandler::with_defaults();
        let mut rx = handler.take_receiver().unwrap();
        let run_id = Uuid::new_v4();

        handler
            .on_custom_event("my_event", &json!({"foo": "bar"}), run_id, None)
            .await
            .unwrap();

        let evt = rx.try_recv().unwrap();
        assert_eq!(evt.event, EventType::OnCustomEvent);
        assert_eq!(evt.name, "my_event");
        assert_eq!(evt.data.input, Some(json!({"foo": "bar"})));
    }

    #[tokio::test]
    async fn test_assign_name_from_serialized() {
        assert_eq!(assign_name(Some("explicit"), &json!({})), "explicit");
        assert_eq!(
            assign_name(None, &json!({"name": "from_name"})),
            "from_name"
        );
        assert_eq!(
            assign_name(None, &json!({"id": ["module", "ClassName"]})),
            "ClassName"
        );
        assert_eq!(assign_name(None, &json!({})), "Unnamed");
    }

    #[test]
    fn test_root_event_filter_no_filters() {
        let filter = RootEventFilter::default();
        let event = StreamEvent {
            event: EventType::OnLlmStart,
            name: "test".to_string(),
            data: EventData::default(),
            run_id: Uuid::new_v4().to_string(),
            parent_ids: vec![],
            tags: vec![],
            metadata: HashMap::new(),
        };
        assert!(filter.include_event(&event, "llm"));
    }

    #[test]
    fn test_root_event_filter_exclude_names() {
        let filter = RootEventFilter {
            exclude_names: vec!["hidden".to_string()],
            ..Default::default()
        };
        let event = StreamEvent {
            event: EventType::OnLlmStart,
            name: "hidden".to_string(),
            data: EventData::default(),
            run_id: Uuid::new_v4().to_string(),
            parent_ids: vec![],
            tags: vec![],
            metadata: HashMap::new(),
        };
        assert!(!filter.include_event(&event, "llm"));
    }

    #[test]
    fn test_root_event_filter_include_types() {
        let filter = RootEventFilter {
            include_types: vec!["llm".to_string()],
            ..Default::default()
        };
        let llm_event = StreamEvent {
            event: EventType::OnLlmStart,
            name: "test".to_string(),
            data: EventData::default(),
            run_id: Uuid::new_v4().to_string(),
            parent_ids: vec![],
            tags: vec![],
            metadata: HashMap::new(),
        };
        let chain_event = StreamEvent {
            event: EventType::OnChainStart,
            name: "test".to_string(),
            data: EventData::default(),
            run_id: Uuid::new_v4().to_string(),
            parent_ids: vec![],
            tags: vec![],
            metadata: HashMap::new(),
        };
        assert!(filter.include_event(&llm_event, "llm"));
        assert!(!filter.include_event(&chain_event, "chain"));
    }

    #[test]
    fn test_root_event_filter_include_tags() {
        let filter = RootEventFilter {
            include_tags: vec!["important".to_string()],
            ..Default::default()
        };
        let tagged = StreamEvent {
            event: EventType::OnLlmStart,
            name: "test".to_string(),
            data: EventData::default(),
            run_id: Uuid::new_v4().to_string(),
            parent_ids: vec![],
            tags: vec!["important".to_string()],
            metadata: HashMap::new(),
        };
        let untagged = StreamEvent {
            event: EventType::OnLlmStart,
            name: "test".to_string(),
            data: EventData::default(),
            run_id: Uuid::new_v4().to_string(),
            parent_ids: vec![],
            tags: vec![],
            metadata: HashMap::new(),
        };
        assert!(filter.include_event(&tagged, "llm"));
        assert!(!filter.include_event(&untagged, "llm"));
    }

    #[tokio::test]
    async fn test_take_receiver_only_once() {
        let handler = EventStreamCallbackHandler::with_defaults();
        assert!(handler.take_receiver().is_some());
        assert!(handler.take_receiver().is_none());
    }

    #[test]
    fn test_event_type_serialization() {
        let val = serde_json::to_value(EventType::OnLlmStart).unwrap();
        assert_eq!(val, json!("on_llm_start"));

        let val = serde_json::to_value(EventType::OnChatModelStream).unwrap();
        assert_eq!(val, json!("on_chat_model_stream"));

        let deserialized: EventType = serde_json::from_str("\"on_tool_end\"").unwrap();
        assert_eq!(deserialized, EventType::OnToolEnd);
    }

    #[test]
    fn test_stream_event_serialization_roundtrip() {
        let event = StreamEvent {
            event: EventType::OnChainEnd,
            name: "my-chain".to_string(),
            data: EventData {
                output: Some(json!({"answer": 42})),
                ..Default::default()
            },
            run_id: Uuid::nil().to_string(),
            parent_ids: vec!["parent-1".to_string()],
            tags: vec!["tag1".to_string()],
            metadata: HashMap::new(),
        };

        let json_str = serde_json::to_string(&event).unwrap();
        let deserialized: StreamEvent = serde_json::from_str(&json_str).unwrap();
        assert_eq!(deserialized.event, EventType::OnChainEnd);
        assert_eq!(deserialized.name, "my-chain");
        assert_eq!(deserialized.data.output, Some(json!({"answer": 42})));
    }

    #[test]
    fn test_agent_event_types_serialize() {
        assert_eq!(
            serde_json::to_string(&EventType::OnAgentAction).unwrap(),
            "\"on_agent_action\""
        );
        assert_eq!(
            serde_json::to_string(&EventType::OnAgentFinish).unwrap(),
            "\"on_agent_finish\""
        );
    }

    #[test]
    fn test_agent_event_types_display() {
        assert_eq!(EventType::OnAgentAction.to_string(), "on_agent_action");
    }

    #[tokio::test]
    async fn test_handler_receives_agent_events() {
        use crate::agents::{AgentAction, AgentFinish};
        use std::collections::HashMap;

        let handler = EventStreamCallbackHandler::with_defaults();
        let mut rx = handler.take_receiver().unwrap();
        let run_id = Uuid::new_v4();

        let action = AgentAction::new(
            "calculator",
            serde_json::json!({"a": 1, "b": 2}),
            "thought: use calculator",
        );
        handler
            .on_agent_action(&action, run_id, None)
            .await
            .unwrap();

        let mut rv = HashMap::new();
        rv.insert("output".to_string(), serde_json::json!("done"));
        let finish = AgentFinish::new(rv, "final answer");
        handler
            .on_agent_finish(&finish, run_id, None)
            .await
            .unwrap();

        // Drop handler to close sender so rx.recv() returns None at end.
        drop(handler);

        let ev1 = rx.recv().await.unwrap();
        assert_eq!(ev1.event, EventType::OnAgentAction);
        assert_eq!(ev1.name, "calculator");
        assert_eq!(ev1.data.input, Some(serde_json::json!({"a": 1, "b": 2})));

        let ev2 = rx.recv().await.unwrap();
        assert_eq!(ev2.event, EventType::OnAgentFinish);
        assert_eq!(ev2.name, "AgentExecutor");
        assert!(ev2.data.output.is_some());
    }
}
