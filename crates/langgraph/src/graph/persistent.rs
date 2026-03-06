//! Persistent graph execution with checkpoint integration.
//!
//! [`PersistentGraph`] wraps a [`CompiledStateGraph`] with a
//! [`CheckpointSaver`] backend so that every invocation automatically
//! saves/restores state, enabling durable, resumable workflows keyed by a
//! `thread_id`.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::errors::{LangGraphError, Result};
use crate::pregel::checkpoint::{
    empty_checkpoint, Checkpoint, CheckpointMetadata, CheckpointSaver,
};

use super::state::CompiledStateGraph;

/// A persistent wrapper around [`CompiledStateGraph`] that automatically
/// saves and restores graph state using a [`CheckpointSaver`] backend.
///
/// Each instance is bound to a specific `thread_id`. Successive calls to
/// [`invoke`](PersistentGraph::invoke) load the latest checkpoint for that
/// thread, merge it with the new input, execute the graph, and persist the
/// resulting state as a new checkpoint.
pub struct PersistentGraph {
    /// The underlying compiled graph.
    graph: CompiledStateGraph,
    /// The checkpoint persistence backend.
    saver: Arc<dyn CheckpointSaver>,
    /// The thread identifier used to key checkpoints.
    thread_id: String,
}

impl PersistentGraph {
    /// Create a new `PersistentGraph`.
    ///
    /// # Arguments
    ///
    /// * `graph` - A compiled state graph to execute.
    /// * `saver` - A checkpoint saver backend (e.g. `InMemoryCheckpointSaver`).
    /// * `thread_id` - A unique identifier for the conversation / execution thread.
    pub fn new(
        graph: CompiledStateGraph,
        saver: Arc<dyn CheckpointSaver>,
        thread_id: impl Into<String>,
    ) -> Self {
        Self {
            graph,
            saver,
            thread_id: thread_id.into(),
        }
    }

    /// Return the thread ID this graph is bound to.
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    // ---------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------

    /// Build the config map used to key checkpoints for this thread.
    fn thread_config(&self) -> HashMap<String, Value> {
        let mut config = HashMap::new();
        config.insert(
            "thread_id".to_string(),
            Value::String(self.thread_id.clone()),
        );
        config
    }

    /// Merge two JSON values. Keys from `overlay` overwrite keys in `base`.
    /// If either value is not an object the overlay wins outright.
    fn merge_values(base: &Value, overlay: &Value) -> Value {
        match (base, overlay) {
            (Value::Object(b), Value::Object(o)) => {
                let mut merged = b.clone();
                for (k, v) in o {
                    merged.insert(k.clone(), v.clone());
                }
                Value::Object(merged)
            }
            _ => overlay.clone(),
        }
    }

    /// Reconstruct a flat `Value::Object` from a checkpoint's `channel_values`.
    fn state_from_checkpoint(checkpoint: &Checkpoint) -> Value {
        Value::Object(
            checkpoint
                .channel_values
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    }

    /// Save the given result state as a new checkpoint and return the state.
    async fn save_checkpoint(&self, state: &Value, step: i64, source: &str) -> Result<()> {
        let mut cp = empty_checkpoint();

        // Populate channel_values from the state object.
        if let Value::Object(map) = state {
            for (k, v) in map {
                cp.channel_values.insert(k.clone(), v.clone());
            }
        }

        let metadata = CheckpointMetadata {
            source: source.to_string(),
            step,
            writes: None,
            extra: HashMap::new(),
        };

        let config = self.thread_config();
        self.saver.put(&config, cp, metadata).await?;
        Ok(())
    }

    // ---------------------------------------------------------------
    // Public API
    // ---------------------------------------------------------------

    /// Invoke the graph, resuming from the latest checkpoint if one exists.
    ///
    /// 1. Load the latest checkpoint for `thread_id`.
    /// 2. If a checkpoint exists, merge `input` on top of the saved state.
    /// 3. Execute the graph.
    /// 4. Save the result as a new checkpoint.
    /// 5. Return the result.
    pub async fn invoke(&self, input: Value) -> Result<Value> {
        let config = self.thread_config();
        let existing = self.saver.get(&config).await?;

        let effective_input = match existing {
            Some(tuple) => {
                let saved = Self::state_from_checkpoint(&tuple.checkpoint);
                Self::merge_values(&saved, &input)
            }
            None => input,
        };

        let result = self.graph.invoke(effective_input).await?;
        self.save_checkpoint(&result, 1, "invoke").await?;
        Ok(result)
    }

    /// Invoke the graph from scratch, ignoring any existing checkpoint.
    ///
    /// The result is still saved as a new checkpoint so subsequent calls to
    /// [`invoke`](Self::invoke) can resume from it.
    pub async fn invoke_fresh(&self, input: Value) -> Result<Value> {
        let result = self.graph.invoke(input).await?;
        self.save_checkpoint(&result, 0, "invoke_fresh").await?;
        Ok(result)
    }

    /// Resume execution from a specific checkpoint.
    ///
    /// # Arguments
    ///
    /// * `checkpoint_id` - The ID of the checkpoint to resume from.
    /// * `update` - Optional value to merge on top of the checkpoint state
    ///   before executing.
    pub async fn resume_from_checkpoint(
        &self,
        checkpoint_id: &str,
        update: Option<Value>,
    ) -> Result<Value> {
        let mut config = self.thread_config();
        config.insert(
            "checkpoint_id".to_string(),
            Value::String(checkpoint_id.to_string()),
        );

        let tuple = self.saver.get_tuple(&config).await?.ok_or_else(|| {
            LangGraphError::Other(format!("Checkpoint not found: {}", checkpoint_id))
        })?;

        let mut state = Self::state_from_checkpoint(&tuple.checkpoint);
        if let Some(upd) = update {
            state = Self::merge_values(&state, &upd);
        }

        let result = self.graph.invoke(state).await?;

        let step = tuple.metadata.as_ref().map(|m| m.step + 1).unwrap_or(0);
        self.save_checkpoint(&result, step, "resume").await?;
        Ok(result)
    }

    /// Get the latest persisted state for this thread.
    ///
    /// Returns `None` if no checkpoint has been saved yet.
    pub async fn get_state(&self) -> Result<Option<Value>> {
        let config = self.thread_config();
        let existing = self.saver.get(&config).await?;
        Ok(existing.map(|t| Self::state_from_checkpoint(&t.checkpoint)))
    }

    /// List checkpoint history for this thread.
    ///
    /// Checkpoints are returned newest-first. If `limit` is `Some(n)`, at most
    /// `n` entries are returned.
    pub async fn get_history(&self, limit: Option<usize>) -> Result<Vec<Checkpoint>> {
        let config = self.thread_config();
        let tuples = self.saver.list(&config, limit).await?;
        Ok(tuples.into_iter().map(|t| t.checkpoint).collect())
    }

    /// Fork execution from a specific checkpoint into a new thread.
    ///
    /// Loads the checkpoint identified by `checkpoint_id`, saves it under
    /// `new_thread_id`, and returns a new `PersistentGraph` bound to that
    /// thread.
    pub async fn fork(&self, checkpoint_id: &str, new_thread_id: &str) -> Result<PersistentGraph> {
        let mut config = self.thread_config();
        config.insert(
            "checkpoint_id".to_string(),
            Value::String(checkpoint_id.to_string()),
        );

        let tuple = self.saver.get_tuple(&config).await?.ok_or_else(|| {
            LangGraphError::Other(format!("Checkpoint not found for fork: {}", checkpoint_id))
        })?;

        // Save the checkpoint state under the new thread.
        let mut new_config = HashMap::new();
        new_config.insert(
            "thread_id".to_string(),
            Value::String(new_thread_id.to_string()),
        );

        let metadata = CheckpointMetadata {
            source: "fork".to_string(),
            step: tuple.metadata.as_ref().map(|m| m.step).unwrap_or(0),
            writes: None,
            extra: HashMap::new(),
        };

        self.saver
            .put(&new_config, tuple.checkpoint, metadata)
            .await?;

        Ok(PersistentGraph {
            graph: self.graph.clone(),
            saver: Arc::clone(&self.saver),
            thread_id: new_thread_id.to_string(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc as StdArc;

    use crate::graph::state::{AsyncNodeAction, StateGraph};
    use crate::pregel::checkpoint::InMemoryCheckpointSaver;
    use serde_json::json;

    /// Helper: build a simple two-node graph that adds `step` counts.
    fn build_test_graph() -> CompiledStateGraph {
        let action: AsyncNodeAction = StdArc::new(|state: Value| {
            Box::pin(async move {
                let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "count": count + 1 }))
            })
        });

        let graph = StateGraph::new()
            .add_node("add_one", action)
            .add_edge("__start__", "add_one")
            .add_edge("add_one", "__end__")
            .compile()
            .unwrap();

        graph
    }

    #[tokio::test]
    async fn test_invoke_saves_checkpoint() {
        let graph = build_test_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());
        let pg = PersistentGraph::new(graph, saver.clone(), "thread-1");

        let result = pg.invoke(json!({ "count": 0 })).await.unwrap();
        assert_eq!(result["count"], 1);

        // Verify a checkpoint was saved.
        let state = pg.get_state().await.unwrap();
        assert!(state.is_some());
        assert_eq!(state.unwrap()["count"], 1);
    }

    #[tokio::test]
    async fn test_second_invoke_loads_checkpoint() {
        let graph = build_test_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());
        let pg = PersistentGraph::new(graph, saver.clone(), "thread-1");

        // First invoke: 0 -> 1
        let r1 = pg.invoke(json!({ "count": 0 })).await.unwrap();
        assert_eq!(r1["count"], 1);

        // Second invoke without explicit count — should load saved count=1
        // and merge with empty input, then run add_one: 1 -> 2.
        let r2 = pg.invoke(json!({})).await.unwrap();
        assert_eq!(r2["count"], 2);
    }

    #[tokio::test]
    async fn test_invoke_fresh_ignores_checkpoint() {
        let graph = build_test_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());
        let pg = PersistentGraph::new(graph, saver.clone(), "thread-1");

        // First invoke: 0 -> 1
        pg.invoke(json!({ "count": 10 })).await.unwrap();

        // invoke_fresh should ignore the saved count=11 and start from 0.
        let result = pg.invoke_fresh(json!({ "count": 0 })).await.unwrap();
        assert_eq!(result["count"], 1);
    }

    #[tokio::test]
    async fn test_get_state_returns_latest() {
        let graph = build_test_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());
        let pg = PersistentGraph::new(graph, saver.clone(), "thread-1");

        // No state yet.
        assert!(pg.get_state().await.unwrap().is_none());

        pg.invoke(json!({ "count": 0 })).await.unwrap();
        let state = pg.get_state().await.unwrap().unwrap();
        assert_eq!(state["count"], 1);

        pg.invoke(json!({})).await.unwrap();
        let state = pg.get_state().await.unwrap().unwrap();
        assert_eq!(state["count"], 2);
    }

    #[tokio::test]
    async fn test_get_history_returns_ordered_list() {
        let graph = build_test_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());
        let pg = PersistentGraph::new(graph, saver.clone(), "thread-1");

        // Run three invocations to build history.
        pg.invoke(json!({ "count": 0 })).await.unwrap();
        pg.invoke(json!({})).await.unwrap();
        pg.invoke(json!({})).await.unwrap();

        let history = pg.get_history(None).await.unwrap();
        assert_eq!(history.len(), 3);

        // History is newest-first (from saver.list), so first entry has highest count.
        assert_eq!(history[0].channel_values["count"], json!(3));
        assert_eq!(history[1].channel_values["count"], json!(2));
        assert_eq!(history[2].channel_values["count"], json!(1));

        // Test with limit.
        let limited = pg.get_history(Some(2)).await.unwrap();
        assert_eq!(limited.len(), 2);
    }

    #[tokio::test]
    async fn test_fork_creates_independent_copy() {
        let graph = build_test_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());
        let pg = PersistentGraph::new(graph, saver.clone(), "thread-1");

        // Build some state.
        pg.invoke(json!({ "count": 0 })).await.unwrap(); // count = 1
        pg.invoke(json!({})).await.unwrap(); // count = 2

        // Get the latest checkpoint ID to fork from.
        let history = pg.get_history(Some(1)).await.unwrap();
        let checkpoint_id = &history[0].id;

        // Fork into a new thread.
        let forked = pg.fork(checkpoint_id, "thread-2").await.unwrap();
        assert_eq!(forked.thread_id(), "thread-2");

        // The forked thread should have the same state.
        let forked_state = forked.get_state().await.unwrap().unwrap();
        assert_eq!(forked_state["count"], 2);

        // Invoke on forked thread should continue independently.
        let forked_result = forked.invoke(json!({})).await.unwrap();
        assert_eq!(forked_result["count"], 3);

        // Original thread should still be at count=2.
        let original_state = pg.get_state().await.unwrap().unwrap();
        assert_eq!(original_state["count"], 2);
    }
}
