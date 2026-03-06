//! Time-travel engine for checkpoint replay and forking.
//!
//! [`TimeTravelEngine`] wraps a [`CompiledStateGraph`] and a
//! [`CheckpointSaver`] backend to provide history browsing, state inspection,
//! replay-from-checkpoint, and fork-with-modification capabilities.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use uuid::Uuid;

use crate::errors::{LangGraphError, Result};
use crate::pregel::checkpoint::{
    CheckpointEntry, CheckpointMetadata, CheckpointSaver, empty_checkpoint,
};

use super::state::CompiledStateGraph;

/// Engine for time-travel operations over a checkpointed graph execution.
///
/// Binds a compiled graph, a checkpoint saver, and a thread identifier together
/// so callers can list past checkpoints, inspect state at any point, replay
/// execution from a previous checkpoint, or fork into a new timeline with
/// modified state.
pub struct TimeTravelEngine {
    /// The compiled graph used for (re-)execution.
    graph: CompiledStateGraph,
    /// Checkpoint persistence backend.
    saver: Arc<dyn CheckpointSaver>,
    /// Thread identifier that scopes all checkpoint operations.
    thread_id: String,
}

impl TimeTravelEngine {
    /// Create a new `TimeTravelEngine`.
    ///
    /// # Arguments
    ///
    /// * `graph` - A compiled state graph to execute.
    /// * `saver` - A checkpoint saver backend (e.g. `InMemoryCheckpointSaver`).
    /// * `thread_id` - The thread whose checkpoints this engine operates on.
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

    /// Return the thread ID this engine is bound to.
    pub fn thread_id(&self) -> &str {
        &self.thread_id
    }

    // ---------------------------------------------------------------
    // History & inspection
    // ---------------------------------------------------------------

    /// List all checkpoints for this thread, ordered by timestamp (oldest first).
    ///
    /// Each entry contains the checkpoint ID, the node that produced it, a
    /// timestamp, and the full state snapshot at that point.
    pub async fn list_checkpoints(&self) -> Result<Vec<CheckpointEntry>> {
        self.saver.list_checkpoints(&self.thread_id).await
    }

    /// Return the state at a specific checkpoint without executing anything.
    ///
    /// # Errors
    ///
    /// Returns an error if the checkpoint is not found.
    pub async fn get_state_at(&self, checkpoint_id: &str) -> Result<Value> {
        let tuple = self.load_checkpoint(checkpoint_id).await?;
        Ok(Self::state_from_channel_values(&tuple.checkpoint.channel_values))
    }

    // ---------------------------------------------------------------
    // Replay & fork
    // ---------------------------------------------------------------

    /// Replay the graph from the state captured at `checkpoint_id`.
    ///
    /// The graph is re-executed from that state, and new checkpoints are saved
    /// under a freshly generated fork thread ID so the original history is
    /// preserved. Returns the final state.
    pub async fn replay_from(&self, checkpoint_id: &str) -> Result<Value> {
        let tuple = self.load_checkpoint(checkpoint_id).await?;
        let state = Self::state_from_channel_values(&tuple.checkpoint.channel_values);

        let fork_thread = self.new_fork_thread_id();
        let result = self.graph.invoke(state).await?;
        self.save_result(&fork_thread, &result, "replay").await?;
        Ok(result)
    }

    /// Fork from a checkpoint with state modifications.
    ///
    /// Loads the state at `checkpoint_id`, deep-merges `state_modifications`
    /// on top, re-executes the graph, and saves the result under a new fork
    /// thread. Returns the final state.
    pub async fn fork_from(
        &self,
        checkpoint_id: &str,
        state_modifications: Value,
    ) -> Result<Value> {
        let tuple = self.load_checkpoint(checkpoint_id).await?;
        let base = Self::state_from_channel_values(&tuple.checkpoint.channel_values);
        let modified = Self::merge_values(&base, &state_modifications);

        let fork_thread = self.new_fork_thread_id();
        let result = self.graph.invoke(modified).await?;
        self.save_result(&fork_thread, &result, "fork").await?;
        Ok(result)
    }

    // ---------------------------------------------------------------
    // Internal helpers
    // ---------------------------------------------------------------

    /// Build a config map for a given thread.
    fn config_for(thread_id: &str) -> HashMap<String, Value> {
        let mut config = HashMap::new();
        config.insert(
            "thread_id".to_string(),
            Value::String(thread_id.to_string()),
        );
        config
    }

    /// Load a specific checkpoint tuple by ID, returning an error if not found.
    async fn load_checkpoint(
        &self,
        checkpoint_id: &str,
    ) -> Result<crate::pregel::checkpoint::CheckpointTuple> {
        let mut config = Self::config_for(&self.thread_id);
        config.insert(
            "checkpoint_id".to_string(),
            Value::String(checkpoint_id.to_string()),
        );

        self.saver.get_tuple(&config).await?.ok_or_else(|| {
            LangGraphError::Other(format!(
                "Checkpoint not found: {}",
                checkpoint_id
            ))
        })
    }

    /// Reconstruct a flat `Value::Object` from checkpoint channel values.
    fn state_from_channel_values(channel_values: &HashMap<String, Value>) -> Value {
        Value::Object(
            channel_values
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect(),
        )
    }

    /// Merge two JSON values. Keys from `overlay` overwrite keys in `base`.
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

    /// Generate a new fork thread ID derived from the current thread.
    fn new_fork_thread_id(&self) -> String {
        format!("{}-fork-{}", self.thread_id, Uuid::now_v7())
    }

    /// Save a result state as a checkpoint under the given thread.
    async fn save_result(
        &self,
        thread_id: &str,
        state: &Value,
        source: &str,
    ) -> Result<()> {
        let mut cp = empty_checkpoint();

        if let Value::Object(map) = state {
            for (k, v) in map {
                cp.channel_values.insert(k.clone(), v.clone());
            }
        }

        let metadata = CheckpointMetadata {
            source: source.to_string(),
            step: 0,
            writes: None,
            extra: HashMap::new(),
        };

        let config = Self::config_for(thread_id);
        self.saver.put(&config, cp, metadata).await?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use serde_json::json;

    use crate::graph::state::{AsyncNodeAction, StateGraph};
    use crate::pregel::checkpoint::InMemoryCheckpointSaver;

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /// Build a simple single-node graph: increments `count` by 1.
    fn build_add_one_graph() -> CompiledStateGraph {
        let action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "count": count + 1 }))
            })
        });

        StateGraph::new()
            .add_node("add_one", action)
            .add_edge("__start__", "add_one")
            .add_edge("add_one", "__end__")
            .compile()
            .unwrap()
    }

    /// Build a two-node graph: node_a doubles `value`, node_b adds 10.
    fn build_multi_node_graph() -> CompiledStateGraph {
        let double: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let v = state.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "value": v * 2 }))
            })
        });

        let add_ten: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let v = state.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "value": v + 10 }))
            })
        });

        StateGraph::new()
            .add_node("double", double)
            .add_node("add_ten", add_ten)
            .add_edge("__start__", "double")
            .add_edge("double", "add_ten")
            .add_edge("add_ten", "__end__")
            .compile()
            .unwrap()
    }

    /// Run the graph through PersistentGraph to populate checkpoints, then
    /// return the saver and the checkpoint IDs in order.
    async fn populate_checkpoints(
        graph: CompiledStateGraph,
        saver: Arc<InMemoryCheckpointSaver>,
        thread_id: &str,
        inputs: Vec<Value>,
    ) -> Vec<String> {
        use crate::graph::persistent::PersistentGraph;

        let pg = PersistentGraph::new(graph, saver.clone(), thread_id);
        for input in inputs {
            pg.invoke(input).await.unwrap();
        }

        let entries = saver.list_checkpoints(thread_id).await.unwrap();
        entries.into_iter().map(|e| e.checkpoint_id).collect()
    }

    // ------------------------------------------------------------------
    // Tests
    // ------------------------------------------------------------------

    #[tokio::test]
    async fn test_list_checkpoints_after_execution() {
        let graph = build_add_one_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());
        let thread = "tt-list";

        let ids = populate_checkpoints(graph.clone(), saver.clone(), thread, vec![
            json!({"count": 0}),
            json!({}),
            json!({}),
        ])
        .await;

        let engine = TimeTravelEngine::new(graph, saver, thread);
        let entries = engine.list_checkpoints().await.unwrap();

        assert_eq!(entries.len(), 3);
        // Oldest first.
        assert_eq!(entries[0].checkpoint_id, ids[0]);
        assert_eq!(entries[2].checkpoint_id, ids[2]);
        // State values should reflect successive increments.
        assert_eq!(entries[0].state["count"], json!(1));
        assert_eq!(entries[1].state["count"], json!(2));
        assert_eq!(entries[2].state["count"], json!(3));
    }

    #[tokio::test]
    async fn test_get_state_at_specific_checkpoint() {
        let graph = build_add_one_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());
        let thread = "tt-state-at";

        let ids = populate_checkpoints(graph.clone(), saver.clone(), thread, vec![
            json!({"count": 0}),
            json!({}),
        ])
        .await;

        let engine = TimeTravelEngine::new(graph, saver, thread);

        let state0 = engine.get_state_at(&ids[0]).await.unwrap();
        assert_eq!(state0["count"], json!(1));

        let state1 = engine.get_state_at(&ids[1]).await.unwrap();
        assert_eq!(state1["count"], json!(2));
    }

    #[tokio::test]
    async fn test_replay_from_middle_checkpoint() {
        let graph = build_add_one_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());
        let thread = "tt-replay";

        // Build history: count goes 0->1, 1->2, 2->3
        let ids = populate_checkpoints(graph.clone(), saver.clone(), thread, vec![
            json!({"count": 0}),
            json!({}),
            json!({}),
        ])
        .await;

        let engine = TimeTravelEngine::new(graph, saver, thread);

        // Replay from the first checkpoint (count=1). The graph adds 1, so result=2.
        let result = engine.replay_from(&ids[0]).await.unwrap();
        assert_eq!(result["count"], json!(2));
    }

    #[tokio::test]
    async fn test_fork_with_modified_state_produces_different_result() {
        let graph = build_add_one_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());
        let thread = "tt-fork-mod";

        let ids = populate_checkpoints(graph.clone(), saver.clone(), thread, vec![
            json!({"count": 0}),
        ])
        .await;

        let engine = TimeTravelEngine::new(graph, saver, thread);

        // The checkpoint has count=1. Override count to 100. Graph adds 1 => 101.
        let result = engine
            .fork_from(&ids[0], json!({"count": 100}))
            .await
            .unwrap();
        assert_eq!(result["count"], json!(101));
    }

    #[tokio::test]
    async fn test_fork_creates_new_thread() {
        let graph = build_add_one_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());
        let thread = "tt-fork-thread";

        let ids = populate_checkpoints(graph.clone(), saver.clone(), thread, vec![
            json!({"count": 0}),
        ])
        .await;

        let engine = TimeTravelEngine::new(graph, saver.clone(), thread);

        // Fork should save under a new thread (not the original).
        engine
            .fork_from(&ids[0], json!({"count": 50}))
            .await
            .unwrap();

        // The original thread should still have exactly 1 checkpoint.
        let original_entries = saver.list_checkpoints(thread).await.unwrap();
        assert_eq!(original_entries.len(), 1);
    }

    #[tokio::test]
    async fn test_empty_checkpoint_list_for_new_thread() {
        let graph = build_add_one_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());

        let engine = TimeTravelEngine::new(graph, saver, "nonexistent-thread");
        let entries = engine.list_checkpoints().await.unwrap();
        assert!(entries.is_empty());
    }

    #[tokio::test]
    async fn test_replay_from_latest_checkpoint() {
        let graph = build_add_one_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());
        let thread = "tt-replay-latest";

        let ids = populate_checkpoints(graph.clone(), saver.clone(), thread, vec![
            json!({"count": 0}),
            json!({}),
            json!({}),
        ])
        .await;

        let engine = TimeTravelEngine::new(graph, saver, thread);

        // Replay from latest (count=3). Graph adds 1 => 4.
        let result = engine.replay_from(ids.last().unwrap()).await.unwrap();
        assert_eq!(result["count"], json!(4));
    }

    #[tokio::test]
    async fn test_time_travel_with_multi_node_graph() {
        let graph = build_multi_node_graph();
        let saver = Arc::new(InMemoryCheckpointSaver::new());
        let thread = "tt-multi-node";

        // value=5 => double(5)=10 => add_ten(10)=20
        let ids = populate_checkpoints(graph.clone(), saver.clone(), thread, vec![
            json!({"value": 5}),
        ])
        .await;

        let engine = TimeTravelEngine::new(graph.clone(), saver.clone(), thread);

        // Checkpoint state should be {value: 20}.
        let state = engine.get_state_at(&ids[0]).await.unwrap();
        assert_eq!(state["value"], json!(20));

        // Replay: 20 => double(20)=40 => add_ten(40)=50.
        let replayed = engine.replay_from(&ids[0]).await.unwrap();
        assert_eq!(replayed["value"], json!(50));

        // Fork with modification: set value=3 => double(3)=6 => add_ten(6)=16.
        let forked = engine
            .fork_from(&ids[0], json!({"value": 3}))
            .await
            .unwrap();
        assert_eq!(forked["value"], json!(16));
    }
}
