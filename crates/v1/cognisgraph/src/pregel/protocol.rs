//! PregelProtocol trait — the abstract interface for Pregel-style graph execution.
//!
//! This trait defines the core operations that any Pregel execution engine must
//! support: invoking the graph, streaming results, reading/updating state, and
//! managing interrupts.

use async_trait::async_trait;
use serde_json::Value;
use std::collections::HashMap;

use crate::errors::LangGraphError;
use crate::types::{StateSnapshot, StateUpdate, StreamMode};

use super::types::{InterruptConfig, StreamChunk};

/// The core protocol for Pregel-style graph execution.
///
/// Implementors provide the full lifecycle of graph execution:
/// - `invoke` / `ainvoke` — run to completion and return final state
/// - `stream` — run and yield intermediate outputs
/// - `get_state` / `update_state` — inspect and modify graph state
///
/// This trait mirrors the Python `PregelProtocol` abstract base class.
#[async_trait]
pub trait PregelProtocol: Send + Sync {
    /// Invoke the graph with an input, running to completion.
    ///
    /// Returns the final state after all nodes have executed and the graph
    /// reaches the END node or has no more tasks to run.
    async fn invoke(
        &self,
        input: Value,
        config: Option<HashMap<String, Value>>,
        interrupt_before: InterruptConfig,
        interrupt_after: InterruptConfig,
    ) -> Result<Value, LangGraphError>;

    /// Stream execution outputs as the graph runs.
    ///
    /// Returns a vector of stream chunks produced during execution.
    /// Each chunk contains the namespace, stream mode, and data payload.
    async fn stream(
        &self,
        input: Value,
        config: Option<HashMap<String, Value>>,
        stream_mode: Vec<StreamMode>,
        interrupt_before: InterruptConfig,
        interrupt_after: InterruptConfig,
    ) -> Result<Vec<StreamChunk>, LangGraphError>;

    /// Get the current state snapshot.
    ///
    /// Returns a snapshot of the graph's state including current values,
    /// pending tasks, and metadata.
    async fn get_state(
        &self,
        config: Option<HashMap<String, Value>>,
    ) -> Result<StateSnapshot, LangGraphError>;

    /// Update the graph state externally.
    ///
    /// This allows injecting state changes as if they came from a specific node,
    /// which is useful for human-in-the-loop workflows.
    async fn update_state(
        &self,
        values: StateUpdate,
        config: Option<HashMap<String, Value>>,
    ) -> Result<(), LangGraphError>;
}
