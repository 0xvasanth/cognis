//! Core types for LangGraph.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Stream mode for graph execution output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StreamMode {
    /// Stream the values of the state after each step.
    Values,
    /// Stream only the updates to the state after each step.
    Updates,
    /// Stream checkpoint information.
    Checkpoints,
    /// Stream task information.
    Tasks,
    /// Stream debug information.
    Debug,
    /// Stream messages.
    Messages,
    /// Custom stream mode.
    Custom,
}

/// Retry policy for node execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Initial interval in seconds between retries.
    pub initial_interval_secs: f64,
    /// Backoff factor for exponential backoff.
    pub backoff_factor: f64,
    /// Maximum interval in seconds between retries.
    pub max_interval_secs: f64,
    /// Maximum number of retry attempts.
    pub max_attempts: u32,
    /// Whether to add jitter to retry intervals.
    pub jitter: bool,
    /// Optional list of error message patterns that should trigger a retry.
    /// If None, all errors trigger retries.
    #[serde(skip)]
    pub retry_on: Option<Vec<String>>,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            initial_interval_secs: 0.5,
            backoff_factor: 2.0,
            max_interval_secs: 128.0,
            max_attempts: 3,
            jitter: true,
            retry_on: None,
        }
    }
}

/// Cache policy for node results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachePolicy {
    /// Time-to-live in seconds. `None` means no expiration.
    pub ttl: Option<u64>,
    /// Optional custom cache key function name.
    #[serde(default)]
    pub key_func: Option<String>,
}

/// An update emitted during streaming graph execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamUpdate {
    /// The node that produced this update.
    pub node: String,
    /// The data produced (depends on stream mode).
    pub data: Value,
    /// Which stream mode this update belongs to.
    pub mode: StreamMode,
}

/// An interrupt raised during graph execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Interrupt {
    /// The value associated with the interrupt.
    pub value: Value,
    /// Unique identifier for the interrupt.
    pub id: String,
}

impl Interrupt {
    /// Create an interrupt scoped to a namespace.
    pub fn from_ns(value: Value, ns: &str) -> Self {
        Self {
            value,
            id: format!("{}/{}", ns, uuid::Uuid::new_v4()),
        }
    }
}

/// A message to send to a specific node with an argument.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Send {
    /// The target node name.
    pub node: String,
    /// The argument to send to the node.
    pub arg: Value,
}

impl Eq for Send {}

impl Hash for Send {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.node.hash(state);
        // Hash the JSON string representation for the Value
        let s = serde_json::to_string(&self.arg).unwrap_or_default();
        s.hash(state);
    }
}

/// A goto target in a command: either a node name or a Send.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CommandGoto {
    /// Go to a named node.
    Node(String),
    /// Send a value to a node.
    SendTo(Send),
}

/// A command that controls graph execution flow.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Command {
    /// Target graph (use `Command::PARENT` for parent graph).
    pub graph: Option<String>,
    /// State update to apply.
    pub update: Option<Value>,
    /// Value to resume with after an interrupt.
    pub resume: Option<Value>,
    /// Nodes to go to next.
    pub goto: Vec<CommandGoto>,
}

impl Command {
    /// Constant for referencing the parent graph.
    pub const PARENT: &'static str = "__parent__";

    /// Create a new empty command.
    pub fn new() -> Self {
        Self {
            graph: None,
            update: None,
            resume: None,
            goto: Vec::new(),
        }
    }
}

impl Command {
    /// Convert the update field to a list of (key, value) tuples.
    pub fn update_as_tuples(&self) -> Vec<(String, Value)> {
        match &self.update {
            Some(Value::Object(map)) => map.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            _ => Vec::new(),
        }
    }
}

impl Default for Command {
    fn default() -> Self {
        Self::new()
    }
}

/// The type of interrupt that occurred during graph execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterruptType {
    /// Execution paused before the node ran.
    Before,
    /// Execution paused after the node ran.
    After,
}

/// State captured when graph execution is interrupted for human-in-the-loop.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterruptedState {
    /// The current graph state at the point of interruption.
    pub state: Value,
    /// The node at which execution was interrupted.
    pub interrupted_at: String,
    /// Whether the interrupt occurred before or after the node executed.
    pub interrupt_type: InterruptType,
    /// The nodes that would execute next if resumed.
    pub next_nodes: Vec<String>,
}

/// Result of invoking a graph that may be interrupted.
#[derive(Debug, Clone)]
pub enum InvokeResult {
    /// Graph ran to completion.
    Complete(Value),
    /// Graph was interrupted and awaits human input.
    Interrupted(InterruptedState),
}

/// An overwrite value for a channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Overwrite {
    /// The value to overwrite with.
    pub value: Value,
}

/// A state update to apply to the graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateUpdate {
    /// The values to update.
    pub values: Option<HashMap<String, Value>>,
    /// Apply the update as if it came from this node.
    pub as_node: Option<String>,
    /// The task ID associated with this update.
    pub task_id: Option<String>,
}

/// A task in the Pregel execution engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PregelTask {
    /// Unique identifier for the task.
    pub id: String,
    /// Name of the node this task belongs to.
    pub name: String,
    /// Path in the graph.
    pub path: Vec<String>,
    /// Error message if the task failed.
    pub error: Option<String>,
    /// Interrupts raised during task execution.
    pub interrupts: Vec<Interrupt>,
    /// The result of the task execution.
    pub result: Option<Value>,
}

/// A snapshot of the graph state at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateSnapshot {
    /// The current state values.
    pub values: Value,
    /// The next nodes to execute.
    pub next: Vec<String>,
    /// Metadata associated with the snapshot.
    pub metadata: Option<HashMap<String, Value>>,
    /// When the snapshot was created.
    pub created_at: Option<String>,
    /// Configuration of the parent checkpoint.
    pub parent_config: Option<HashMap<String, Value>>,
    /// Tasks in the snapshot.
    pub tasks: Vec<PregelTask>,
    /// Interrupts in the snapshot.
    pub interrupts: Vec<Interrupt>,
}

/// Durability mode for graph execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Durability {
    /// Synchronous durability — wait for persistence before continuing.
    Sync,
    /// Asynchronous durability — persist in the background.
    Async,
    /// Persist only on exit.
    Exit,
}

/// An executable task in the Pregel engine with all execution context.
#[derive(Debug, Clone)]
pub struct PregelExecutableTask {
    /// Unique identifier for the task.
    pub id: String,
    /// Name of the node this task belongs to.
    pub name: String,
    /// The input value for the task.
    pub input: Value,
    /// Path in the graph.
    pub path: Vec<String>,
    /// Retry policy for the task.
    pub retry_policy: Option<RetryPolicy>,
    /// Channel triggers that activated this task.
    pub triggers: Vec<String>,
}

/// A key for caching node results.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CacheKey {
    /// Namespace for the cache entry.
    pub ns: String,
    /// The cache key.
    pub key: String,
    /// Time-to-live in seconds.
    pub ttl: Option<u64>,
}
