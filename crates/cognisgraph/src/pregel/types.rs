//! Pregel-specific types for the execution engine.
//!
//! These types support the Pregel execution model — a step-based, channel-driven
//! computation framework inspired by Google's Pregel and Apache Beam.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;

use crate::errors::LangGraphError;

/// Sentinel type indicating "all nodes" for interrupt configuration.
///
/// When used with `interrupt_before` or `interrupt_after`, this means
/// the graph should interrupt before/after every node execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct All;

/// Specifies which nodes to interrupt on — either specific nodes or all of them.
#[derive(Debug, Clone, Default)]
pub enum InterruptConfig {
    /// No interrupts.
    #[default]
    None,
    /// Interrupt on all nodes.
    All,
    /// Interrupt on specific named nodes.
    Nodes(Vec<String>),
}

impl InterruptConfig {
    /// Check if a node name should trigger an interrupt.
    pub fn should_interrupt(&self, node_name: &str) -> bool {
        match self {
            Self::None => false,
            Self::All => true,
            Self::Nodes(names) => names.iter().any(|n| n == node_name),
        }
    }
}

/// Description of a task to be executed in a Pregel step.
#[derive(Debug, Clone)]
pub struct PregelTaskDescription {
    /// The name of the node to execute.
    pub name: String,
    /// Input data for the task.
    pub input: Value,
    /// Triggers that caused this task to be scheduled.
    pub triggers: Vec<String>,
    /// Unique identifier for the task.
    pub id: String,
}

/// A write operation produced by a node: (channel_name, value).
pub type ChannelWrite = (String, Value);

/// The result of executing a single Pregel task.
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// The task that was executed.
    pub task: PregelTaskDescription,
    /// The writes produced by the task, as (channel_name, value) pairs.
    pub writes: Vec<ChannelWrite>,
    /// Error, if the task failed.
    pub error: Option<String>,
}

/// Status of the Pregel execution loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PregelStatus {
    /// Waiting for input.
    Pending,
    /// Execution is running.
    Running,
    /// Execution completed successfully.
    Done,
    /// Interrupted before a node.
    InterruptBefore,
    /// Interrupted after a node.
    InterruptAfter,
    /// Recursion limit reached.
    OutOfSteps,
}

/// A stream chunk emitted during Pregel execution.
///
/// Contains the namespace path, stream mode identifier, and the payload.
#[derive(Debug, Clone)]
pub struct StreamChunk {
    /// Namespace path (for subgraph support).
    pub ns: Vec<String>,
    /// The stream mode this chunk belongs to.
    pub mode: String,
    /// The payload data.
    pub data: Value,
}

/// Configuration for a single Pregel execution run.
#[derive(Debug, Clone)]
pub struct PregelConfig {
    /// Maximum number of super-steps before raising a recursion error.
    pub recursion_limit: usize,
    /// Interrupt before executing these nodes.
    pub interrupt_before: InterruptConfig,
    /// Interrupt after executing these nodes.
    pub interrupt_after: InterruptConfig,
    /// Stream modes to enable.
    pub stream_modes: Vec<crate::types::StreamMode>,
    /// Additional metadata for the run.
    pub metadata: HashMap<String, Value>,
}

impl Default for PregelConfig {
    fn default() -> Self {
        Self {
            recursion_limit: 25,
            interrupt_before: InterruptConfig::None,
            interrupt_after: InterruptConfig::None,
            stream_modes: vec![crate::types::StreamMode::Values],
            metadata: HashMap::new(),
        }
    }
}

/// An async node function that takes state and returns channel writes.
///
/// Unlike `AsyncNodeAction` (which returns state updates), this returns
/// explicit channel writes for the Pregel execution model.
pub type PregelNodeAction = Arc<
    dyn Fn(
            Value,
        ) -> Pin<
            Box<dyn std::future::Future<Output = Result<Vec<ChannelWrite>, LangGraphError>> + Send>,
        > + Send
        + Sync,
>;

/// Specification for a node in the Pregel execution engine.
pub struct PregelNodeSpec {
    /// The name of this node.
    pub name: String,
    /// Channels that trigger this node's execution.
    pub triggers: Vec<String>,
    /// Channels this node reads from.
    pub channels: Vec<String>,
    /// The action to execute.
    pub action: PregelNodeAction,
    /// Channels this node writes to (if empty, uses default mapping).
    pub writers: Vec<String>,
    /// Optional retry policy.
    pub retry_policy: Option<crate::types::RetryPolicy>,
    /// Optional cache policy for this node's results.
    pub cache_policy: Option<crate::types::CachePolicy>,
    /// Tags for this node (e.g., for filtering in streaming).
    pub tags: Vec<String>,
    /// Metadata for this node.
    pub metadata: HashMap<String, Value>,
}

impl fmt::Debug for PregelNodeSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PregelNodeSpec")
            .field("name", &self.name)
            .field("triggers", &self.triggers)
            .field("channels", &self.channels)
            .field("writers", &self.writers)
            .field("retry_policy", &self.retry_policy)
            .field("cache_policy", &self.cache_policy)
            .field("tags", &self.tags)
            .field("metadata", &self.metadata)
            .field("action", &"<PregelNodeAction>")
            .finish()
    }
}
