//! Core Pregel execution engine.
//!
//! The [`PregelExecutor`] implements the Pregel-style step-based execution model:
//!
//! 1. **Prepare tasks** — determine which nodes should run based on channel triggers
//! 2. **Execute tasks** — run node actions concurrently, collecting channel writes
//! 3. **Apply writes** — update channels with task outputs
//! 4. **Check termination** — stop if no tasks remain, END is reached, or recursion limit hit
//!
//! This is the heart of LangGraph's execution model, inspired by Google's Pregel
//! and Apache Beam.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

use crate::channels::BaseChannel;
use crate::constants::END;
use crate::errors::{create_error_message, ErrorCode, LangGraphError};
use crate::types::StreamMode;

use super::types::{
    ChannelWrite, PregelConfig, PregelNodeAction, PregelNodeSpec, PregelStatus,
    PregelTaskDescription, StreamChunk, TaskResult,
};

/// The core Pregel execution engine.
///
/// Manages channels, nodes, and the step-based execution loop. Each "super-step"
/// consists of: collecting triggered tasks, executing them, applying their writes
/// to channels, and checking for termination.
///
/// # Channel-Based State Management
///
/// State flows through channels (`Box<dyn BaseChannel>`). Nodes read from and
/// write to channels. The channel type determines how concurrent writes are
/// resolved (last-value, topic/append, binary-operator aggregation, etc.).
///
/// # Trigger Model
///
/// Nodes are triggered based on which channels were **updated** in the current
/// step (not merely available). On the first step after `apply_input`, the input
/// channels count as updated. On subsequent steps, only channels that received
/// writes from the previous step's tasks are considered updated. This prevents
/// infinite loops when using persistent channels like `LastValue`.
///
/// # Example
///
/// ```rust,no_run
/// use langgraph::pregel::executor::PregelExecutor;
/// use langgraph::pregel::types::{PregelNodeSpec, PregelConfig};
/// use langgraph::channels::LastValue;
///
/// let mut executor = PregelExecutor::new(PregelConfig::default());
/// // Add channels, nodes, then run
/// ```
pub struct PregelExecutor {
    /// The channels that hold graph state between steps.
    channels: HashMap<String, Box<dyn BaseChannel>>,
    /// The nodes in the graph.
    nodes: HashMap<String, PregelNodeSpec>,
    /// Execution configuration.
    config: PregelConfig,
    /// Current step number (0-based).
    step: usize,
    /// Current execution status.
    status: PregelStatus,
    /// Stream chunks collected during execution.
    stream_output: Vec<StreamChunk>,
    /// Channels that were updated in the current step (used for triggering).
    updated_channels: HashSet<String>,
}

impl std::fmt::Debug for PregelExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PregelExecutor")
            .field("channels", &self.channels.keys().collect::<Vec<_>>())
            .field("nodes", &self.nodes.keys().collect::<Vec<_>>())
            .field("config", &self.config)
            .field("step", &self.step)
            .field("status", &self.status)
            .finish()
    }
}

impl PregelExecutor {
    /// Create a new Pregel executor with the given configuration.
    pub fn new(config: PregelConfig) -> Self {
        Self {
            channels: HashMap::new(),
            nodes: HashMap::new(),
            config,
            step: 0,
            status: PregelStatus::Pending,
            stream_output: Vec::new(),
            updated_channels: HashSet::new(),
        }
    }

    /// Add a channel to the executor.
    ///
    /// Channels are the state containers that nodes read from and write to.
    /// Each channel has a unique key name.
    pub fn add_channel(&mut self, channel: Box<dyn BaseChannel>) {
        self.channels.insert(channel.key().to_string(), channel);
    }

    /// Add a node to the executor.
    ///
    /// Nodes define the computation. Each node specifies which channels trigger
    /// it, which channels it reads, and what action to perform.
    pub fn add_node(&mut self, node: PregelNodeSpec) {
        self.nodes.insert(node.name.clone(), node);
    }

    /// Get the current values of all available channels as a JSON object.
    pub fn read_channels(&self) -> Value {
        let mut state = serde_json::Map::new();
        for (key, channel) in &self.channels {
            if channel.is_available() {
                if let Ok(value) = channel.get() {
                    state.insert(key.clone(), value);
                }
            }
        }
        Value::Object(state)
    }

    /// Read a specific set of channels and return them as a JSON object.
    pub fn read_channels_by_keys(&self, keys: &[String]) -> Value {
        let mut state = serde_json::Map::new();
        for key in keys {
            if let Some(channel) = self.channels.get(key) {
                if channel.is_available() {
                    if let Ok(value) = channel.get() {
                        state.insert(key.clone(), value);
                    }
                }
            }
        }
        Value::Object(state)
    }

    /// Get the current execution step number.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Get the current execution status.
    pub fn status(&self) -> PregelStatus {
        self.status
    }

    /// Get collected stream output.
    pub fn stream_output(&self) -> &[StreamChunk] {
        &self.stream_output
    }

    /// Apply input to the graph's input channels.
    ///
    /// This writes the input values to matching channels to kick off the
    /// first super-step. Channels that receive input are marked as updated
    /// so they can trigger nodes.
    pub fn apply_input(&mut self, input: Value) -> Result<(), LangGraphError> {
        match input {
            Value::Object(map) => {
                for (key, value) in map {
                    if let Some(channel) = self.channels.get_mut(&key) {
                        if channel.update(vec![value])? {
                            self.updated_channels.insert(key);
                        }
                    }
                    // Silently ignore keys without matching channels.
                }
            }
            _other => {
                // Non-object input: no channel mapping possible without a
                // dedicated START channel, silently skip.
            }
        }
        Ok(())
    }

    /// Prepare tasks for the current step by checking which nodes are triggered.
    ///
    /// A node is triggered when any of its trigger channels were updated in the
    /// previous step (tracked in `updated_channels`). Returns a list of task
    /// descriptions.
    pub fn prepare_next_tasks(&self) -> Vec<PregelTaskDescription> {
        let mut tasks = Vec::new();

        for (name, node) in &self.nodes {
            // Check if any of the node's triggers were updated this step.
            let active_triggers: Vec<String> = node
                .triggers
                .iter()
                .filter(|trigger| self.updated_channels.contains(trigger.as_str()))
                .cloned()
                .collect();

            if !active_triggers.is_empty() {
                // Gather input from the node's read channels.
                let input = if node.channels.is_empty() {
                    // Default: read all available channels.
                    self.read_channels()
                } else {
                    self.read_channels_by_keys(&node.channels)
                };

                let task_id = format!("{}-step{}-{}", name, self.step, uuid::Uuid::new_v4());

                tasks.push(PregelTaskDescription {
                    name: name.clone(),
                    input,
                    triggers: active_triggers,
                    id: task_id,
                });
            }
        }

        tasks
    }

    /// Execute a single task by running the node's action.
    ///
    /// Returns a `TaskResult` containing the writes produced by the node.
    pub async fn execute_task(
        &self,
        task: &PregelTaskDescription,
    ) -> Result<TaskResult, LangGraphError> {
        let node = self.nodes.get(&task.name).ok_or_else(|| {
            LangGraphError::TaskNotFound(format!("Node '{}' not found", task.name))
        })?;

        match (node.action)(task.input.clone()).await {
            Ok(writes) => Ok(TaskResult {
                task: task.clone(),
                writes,
                error: None,
            }),
            Err(e) => Ok(TaskResult {
                task: task.clone(),
                writes: Vec::new(),
                error: Some(e.to_string()),
            }),
        }
    }

    /// Apply writes from task results to channels.
    ///
    /// For each (channel_name, value) pair, the corresponding channel is updated.
    /// Returns the set of channel names that changed.
    pub fn apply_writes(&mut self, writes: &[ChannelWrite]) -> Result<Vec<String>, LangGraphError> {
        let mut changed = Vec::new();

        for (channel_name, value) in writes {
            if channel_name == END {
                // Writes to END signal graph completion; don't update a channel.
                continue;
            }

            if let Some(channel) = self.channels.get_mut(channel_name) {
                let did_change = channel.update(vec![value.clone()])?;
                if did_change {
                    changed.push(channel_name.clone());
                }
            }
            // Writes to non-existent channels are silently ignored,
            // matching Python behavior.
        }

        Ok(changed)
    }

    /// Consume ephemeral channels after tasks have read from them.
    ///
    /// This calls `consume()` on all channels, which resets ephemeral channels
    /// for the next step.
    pub fn consume_channels(&mut self) {
        for channel in self.channels.values_mut() {
            channel.consume();
        }
    }

    /// Emit a stream chunk if the given mode is enabled.
    fn emit_stream(&mut self, mode: StreamMode, mode_str: &str, data: Value) {
        if self.config.stream_modes.contains(&mode) {
            self.stream_output.push(StreamChunk {
                ns: Vec::new(),
                mode: mode_str.to_string(),
                data,
            });
        }
    }

    /// Run the Pregel execution loop to completion.
    ///
    /// This is the main entry point for executing a graph. It:
    /// 1. Applies the input to channels
    /// 2. Loops through super-steps until termination
    /// 3. Returns the final state from all available channels
    ///
    /// # Errors
    ///
    /// Returns `LangGraphError::GraphRecursionError` if the recursion limit is
    /// exceeded.
    /// Returns `LangGraphError::GraphInterrupt` if an interrupt is triggered.
    pub async fn run(&mut self, input: Value) -> Result<Value, LangGraphError> {
        self.status = PregelStatus::Running;
        self.step = 0;
        self.stream_output.clear();
        self.updated_channels.clear();

        // Step 0: Apply input to channels.
        self.apply_input(input)?;

        // Emit initial state.
        let initial_state = self.read_channels();
        self.emit_stream(StreamMode::Values, "values", initial_state);

        // Main execution loop.
        loop {
            // Check recursion limit.
            if self.step >= self.config.recursion_limit {
                self.status = PregelStatus::OutOfSteps;
                return Err(LangGraphError::GraphRecursionError(create_error_message(
                    &format!(
                        "Recursion limit of {} reached without hitting a stop condition. \
                         You can increase the limit by setting `recursion_limit` in the config.",
                        self.config.recursion_limit
                    ),
                    ErrorCode::GraphRecursionLimit,
                )));
            }

            // Prepare tasks for this step.
            let tasks = self.prepare_next_tasks();

            // No tasks means we're done.
            if tasks.is_empty() {
                self.status = PregelStatus::Done;
                break;
            }

            // Check interrupt_before.
            let interrupt_before_tasks: Vec<&PregelTaskDescription> = tasks
                .iter()
                .filter(|t| self.config.interrupt_before.should_interrupt(&t.name))
                .collect();

            if !interrupt_before_tasks.is_empty() {
                self.status = PregelStatus::InterruptBefore;
                let interrupts = interrupt_before_tasks
                    .iter()
                    .map(|t| crate::types::Interrupt {
                        value: serde_json::json!({"node": t.name, "step": self.step}),
                        id: t.id.clone(),
                    })
                    .collect();
                return Err(LangGraphError::GraphInterrupt(interrupts));
            }

            // Execute all tasks for this step.
            let mut all_writes: Vec<ChannelWrite> = Vec::new();
            let mut step_updates = serde_json::Map::new();

            for task in &tasks {
                let result = self.execute_task(task).await?;

                if let Some(ref error) = result.error {
                    step_updates.insert(task.name.clone(), serde_json::json!({"error": error}));
                } else {
                    // Collect updates for streaming.
                    let task_writes: serde_json::Map<String, Value> =
                        result.writes.iter().cloned().collect();
                    step_updates.insert(task.name.clone(), Value::Object(task_writes));

                    all_writes.extend(result.writes);
                }
            }

            // Emit updates stream chunk.
            self.emit_stream(StreamMode::Updates, "updates", Value::Object(step_updates));

            // Consume ephemeral channels from this step's reads.
            self.consume_channels();

            // Apply all writes from this step to channels.
            // Track which channels changed for the next step's triggering.
            let changed = self.apply_writes(&all_writes)?;
            self.updated_channels = changed.into_iter().collect();

            // Emit values stream chunk with updated state.
            let current_state = self.read_channels();
            self.emit_stream(StreamMode::Values, "values", current_state);

            // Check interrupt_after.
            let interrupt_after_tasks: Vec<&PregelTaskDescription> = tasks
                .iter()
                .filter(|t| self.config.interrupt_after.should_interrupt(&t.name))
                .collect();

            if !interrupt_after_tasks.is_empty() {
                self.status = PregelStatus::InterruptAfter;
                let interrupts = interrupt_after_tasks
                    .iter()
                    .map(|t| crate::types::Interrupt {
                        value: serde_json::json!({"node": t.name, "step": self.step}),
                        id: t.id.clone(),
                    })
                    .collect();
                return Err(LangGraphError::GraphInterrupt(interrupts));
            }

            // Finish channels (for named barriers, etc.).
            for channel in self.channels.values_mut() {
                channel.finish();
            }

            self.step += 1;
        }

        Ok(self.read_channels())
    }

    /// Run the graph and return stream chunks along with the final state.
    ///
    /// This is similar to `run` but returns both the stream output and final value.
    pub async fn run_with_stream(
        &mut self,
        input: Value,
    ) -> Result<(Value, Vec<StreamChunk>), LangGraphError> {
        let result = self.run(input).await?;
        let chunks = std::mem::take(&mut self.stream_output);
        Ok((result, chunks))
    }

    /// Create a checkpoint of all channel states.
    ///
    /// Returns a map of channel_name -> checkpoint_value for channels that
    /// have state to persist.
    pub fn checkpoint(&self) -> HashMap<String, Value> {
        let mut ckpt = HashMap::new();
        for (key, channel) in &self.channels {
            if let Some(value) = channel.checkpoint() {
                ckpt.insert(key.clone(), value);
            }
        }
        ckpt
    }

    /// Restore channels from a checkpoint.
    ///
    /// For each channel, if the checkpoint contains a value for that channel's key,
    /// the channel is restored from that checkpoint value.
    pub fn restore_from_checkpoint(
        &mut self,
        checkpoint: &HashMap<String, Value>,
    ) -> Result<(), LangGraphError> {
        let mut restored_channels = HashMap::new();
        for (key, channel) in &self.channels {
            let ckpt_value = checkpoint.get(key).cloned();
            let restored = channel.from_checkpoint(ckpt_value);
            restored_channels.insert(key.clone(), restored);
        }
        self.channels = restored_channels;
        Ok(())
    }

    /// Get a reference to the channels.
    pub fn channels(&self) -> &HashMap<String, Box<dyn BaseChannel>> {
        &self.channels
    }

    /// Get a mutable reference to the channels.
    pub fn channels_mut(&mut self) -> &mut HashMap<String, Box<dyn BaseChannel>> {
        &mut self.channels
    }

    /// Get a reference to the nodes.
    pub fn nodes(&self) -> &HashMap<String, PregelNodeSpec> {
        &self.nodes
    }

    /// Get the current config.
    pub fn config(&self) -> &PregelConfig {
        &self.config
    }
}

/// Helper to create a `PregelNodeAction` from an async closure that returns
/// state updates (a JSON object). The updates are automatically converted to
/// channel writes where each key becomes a channel write.
pub fn state_update_action(action: crate::graph::state::AsyncNodeAction) -> PregelNodeAction {
    std::sync::Arc::new(move |input: Value| {
        let action = action.clone();
        Box::pin(async move {
            let update = action(input).await?;
            // Convert state update (JSON object) to channel writes.
            match update {
                Value::Object(map) => {
                    let writes: Vec<ChannelWrite> = map.into_iter().collect();
                    Ok(writes)
                }
                other => {
                    // Single value update goes to a default channel.
                    Ok(vec![("__default__".to_string(), other)])
                }
            }
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::LastValue;
    use crate::pregel::types::InterruptConfig;
    use serde_json::json;
    use std::sync::Arc;

    /// Helper: create a PregelNodeAction that writes to specific channels.
    fn make_write_action(writes: Vec<(String, Value)>) -> PregelNodeAction {
        Arc::new(move |_input: Value| {
            let writes = writes.clone();
            Box::pin(async move { Ok(writes) })
        })
    }

    /// Helper: create a PregelNodeAction that reads input and transforms it.
    fn make_transform_action<F>(f: F) -> PregelNodeAction
    where
        F: Fn(Value) -> Result<Vec<ChannelWrite>, LangGraphError> + Send + Sync + 'static,
    {
        Arc::new(move |input: Value| {
            let result = f(input);
            Box::pin(async move { result })
        })
    }

    /// Helper: build a simple PregelNodeSpec.
    fn make_node(
        name: &str,
        triggers: Vec<&str>,
        channels: Vec<&str>,
        action: PregelNodeAction,
    ) -> PregelNodeSpec {
        PregelNodeSpec {
            name: name.to_string(),
            triggers: triggers.into_iter().map(|s| s.to_string()).collect(),
            channels: channels.into_iter().map(|s| s.to_string()).collect(),
            action,
            writers: Vec::new(),
            retry_policy: None,
            cache_policy: None,
            tags: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    #[tokio::test]
    async fn test_simple_linear_execution() {
        // Graph: input -> node_a -> node_b -> done
        let mut executor = PregelExecutor::new(PregelConfig::default());

        executor.add_channel(Box::new(LastValue::new("input")));
        executor.add_channel(Box::new(LastValue::new("a_output")));
        executor.add_channel(Box::new(LastValue::new("result")));

        executor.add_node(make_node(
            "node_a",
            vec!["input"],
            vec!["input"],
            make_write_action(vec![("a_output".to_string(), json!("from_a"))]),
        ));

        executor.add_node(make_node(
            "node_b",
            vec!["a_output"],
            vec!["a_output"],
            make_write_action(vec![("result".to_string(), json!("from_b"))]),
        ));

        let result = executor.run(json!({"input": "hello"})).await.unwrap();
        assert_eq!(result["result"], json!("from_b"));
        assert_eq!(result["a_output"], json!("from_a"));
        assert_eq!(result["input"], json!("hello"));
        assert_eq!(executor.status(), PregelStatus::Done);
    }

    #[tokio::test]
    async fn test_no_tasks_completes_immediately() {
        let mut executor = PregelExecutor::new(PregelConfig::default());
        executor.add_channel(Box::new(LastValue::new("unused")));

        let result = executor.run(json!({})).await.unwrap();
        assert_eq!(result, json!({}));
        assert_eq!(executor.status(), PregelStatus::Done);
    }

    #[tokio::test]
    async fn test_recursion_limit() {
        // A node that keeps triggering itself by writing to its own trigger channel.
        let mut executor = PregelExecutor::new(PregelConfig {
            recursion_limit: 3,
            ..Default::default()
        });

        executor.add_channel(Box::new(LastValue::new("counter")));

        executor.add_node(make_node(
            "loop_node",
            vec!["counter"],
            vec!["counter"],
            make_transform_action(|input| {
                let count = input.get("counter").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(vec![("counter".to_string(), json!(count + 1))])
            }),
        ));

        let result = executor.run(json!({"counter": 0})).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            LangGraphError::GraphRecursionError(msg) => {
                assert!(msg.contains("Recursion limit of 3"));
            }
            other => panic!("Expected GraphRecursionError, got: {other:?}"),
        }
        assert_eq!(executor.status(), PregelStatus::OutOfSteps);
    }

    #[tokio::test]
    async fn test_interrupt_before() {
        let mut executor = PregelExecutor::new(PregelConfig {
            interrupt_before: InterruptConfig::Nodes(vec!["critical_node".to_string()]),
            ..Default::default()
        });

        executor.add_channel(Box::new(LastValue::new("input")));
        executor.add_channel(Box::new(LastValue::new("output")));

        executor.add_node(make_node(
            "critical_node",
            vec!["input"],
            vec!["input"],
            make_write_action(vec![("output".to_string(), json!("done"))]),
        ));

        let result = executor.run(json!({"input": "go"})).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            LangGraphError::GraphInterrupt(interrupts) => {
                assert_eq!(interrupts.len(), 1);
                assert_eq!(interrupts[0].value["node"], json!("critical_node"));
            }
            other => panic!("Expected GraphInterrupt, got: {other:?}"),
        }
        assert_eq!(executor.status(), PregelStatus::InterruptBefore);
    }

    #[tokio::test]
    async fn test_interrupt_after() {
        let mut executor = PregelExecutor::new(PregelConfig {
            interrupt_after: InterruptConfig::Nodes(vec!["tracked_node".to_string()]),
            ..Default::default()
        });

        executor.add_channel(Box::new(LastValue::new("input")));
        executor.add_channel(Box::new(LastValue::new("output")));

        executor.add_node(make_node(
            "tracked_node",
            vec!["input"],
            vec!["input"],
            make_write_action(vec![("output".to_string(), json!("done"))]),
        ));

        let result = executor.run(json!({"input": "go"})).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            LangGraphError::GraphInterrupt(interrupts) => {
                assert_eq!(interrupts.len(), 1);
                assert_eq!(interrupts[0].value["node"], json!("tracked_node"));
            }
            other => panic!("Expected GraphInterrupt, got: {other:?}"),
        }
        assert_eq!(executor.status(), PregelStatus::InterruptAfter);
    }

    #[tokio::test]
    async fn test_interrupt_all() {
        let mut executor = PregelExecutor::new(PregelConfig {
            interrupt_before: InterruptConfig::All,
            ..Default::default()
        });

        executor.add_channel(Box::new(LastValue::new("input")));
        executor.add_node(make_node(
            "any_node",
            vec!["input"],
            vec!["input"],
            make_write_action(vec![]),
        ));

        let result = executor.run(json!({"input": "go"})).await;
        assert!(matches!(result, Err(LangGraphError::GraphInterrupt(_))));
    }

    #[tokio::test]
    async fn test_channel_checkpoint_and_restore() {
        let mut executor = PregelExecutor::new(PregelConfig::default());
        executor.add_channel(Box::new(LastValue::new("state")));

        executor.apply_input(json!({"state": "hello"})).unwrap();

        let ckpt = executor.checkpoint();
        assert!(ckpt.contains_key("state"));
        assert_eq!(ckpt["state"], json!("hello"));

        let mut executor2 = PregelExecutor::new(PregelConfig::default());
        executor2.add_channel(Box::new(LastValue::new("state")));
        executor2.restore_from_checkpoint(&ckpt).unwrap();

        let state = executor2.read_channels();
        assert_eq!(state["state"], json!("hello"));
    }

    #[tokio::test]
    async fn test_multi_channel_execution() {
        let mut executor = PregelExecutor::new(PregelConfig::default());

        executor.add_channel(Box::new(LastValue::new("name")));
        executor.add_channel(Box::new(LastValue::new("age")));
        executor.add_channel(Box::new(LastValue::new("greeting")));

        executor.add_node(make_node(
            "greeter",
            vec!["name"],
            vec!["name", "age"],
            make_transform_action(|input| {
                let name = input
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown");
                let age = input.get("age").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(vec![(
                    "greeting".to_string(),
                    json!(format!("Hello {name}, you are {age}!")),
                )])
            }),
        ));

        let result = executor
            .run(json!({"name": "Alice", "age": 30}))
            .await
            .unwrap();

        assert_eq!(result["greeting"], json!("Hello Alice, you are 30!"));
    }

    #[tokio::test]
    async fn test_prepare_next_tasks_no_triggers() {
        let executor = PregelExecutor::new(PregelConfig::default());
        let tasks = executor.prepare_next_tasks();
        assert!(tasks.is_empty());
    }

    #[tokio::test]
    async fn test_apply_writes_to_end() {
        let mut executor = PregelExecutor::new(PregelConfig::default());
        executor.add_channel(Box::new(LastValue::new("state")));

        let changed = executor
            .apply_writes(&[(END.to_string(), json!("done"))])
            .unwrap();
        assert!(changed.is_empty());
    }

    #[tokio::test]
    async fn test_apply_writes_to_nonexistent_channel() {
        let mut executor = PregelExecutor::new(PregelConfig::default());

        let changed = executor
            .apply_writes(&[("nonexistent".to_string(), json!("value"))])
            .unwrap();
        assert!(changed.is_empty());
    }

    #[tokio::test]
    async fn test_stream_output_collected() {
        let mut executor = PregelExecutor::new(PregelConfig {
            stream_modes: vec![StreamMode::Values, StreamMode::Updates],
            ..Default::default()
        });

        executor.add_channel(Box::new(LastValue::new("input")));
        executor.add_channel(Box::new(LastValue::new("output")));

        executor.add_node(make_node(
            "worker",
            vec!["input"],
            vec!["input"],
            make_write_action(vec![("output".to_string(), json!("result"))]),
        ));

        let (_result, chunks) = executor
            .run_with_stream(json!({"input": "go"}))
            .await
            .unwrap();

        let value_chunks: Vec<_> = chunks.iter().filter(|c| c.mode == "values").collect();
        let update_chunks: Vec<_> = chunks.iter().filter(|c| c.mode == "updates").collect();

        assert!(
            !value_chunks.is_empty(),
            "Should have at least one values chunk"
        );
        assert!(
            !update_chunks.is_empty(),
            "Should have at least one updates chunk"
        );
    }

    #[tokio::test]
    async fn test_node_error_handling() {
        let mut executor = PregelExecutor::new(PregelConfig::default());

        executor.add_channel(Box::new(LastValue::new("input")));
        executor.add_channel(Box::new(LastValue::new("output")));

        let error_action: PregelNodeAction = Arc::new(|_input: Value| {
            Box::pin(async { Err(LangGraphError::Other("node failed".to_string())) })
        });

        executor.add_node(make_node(
            "failing_node",
            vec!["input"],
            vec!["input"],
            error_action,
        ));

        // The failing node produces no writes, so no channels are updated
        // and the loop terminates on the next step.
        let result = executor.run(json!({"input": "go"})).await.unwrap();
        assert!(result.get("output").is_none());
    }

    #[tokio::test]
    async fn test_state_update_action_helper() {
        let action: crate::graph::state::AsyncNodeAction =
            Arc::new(|_state: Value| Box::pin(async { Ok(json!({"key": "value"})) }));

        let pregel_action = state_update_action(action);
        let writes = pregel_action(json!({})).await.unwrap();

        assert_eq!(writes.len(), 1);
        assert_eq!(writes[0].0, "key");
        assert_eq!(writes[0].1, json!("value"));
    }

    #[tokio::test]
    async fn test_two_step_chain() {
        let mut executor = PregelExecutor::new(PregelConfig::default());

        executor.add_channel(Box::new(LastValue::new("input")));
        executor.add_channel(Box::new(LastValue::new("middle")));
        executor.add_channel(Box::new(LastValue::new("final_out")));

        executor.add_node(make_node(
            "node_a",
            vec!["input"],
            vec!["input"],
            make_transform_action(|input| {
                let val = input.get("input").cloned().unwrap_or(json!(null));
                Ok(vec![("middle".to_string(), json!({"processed": val}))])
            }),
        ));

        executor.add_node(make_node(
            "node_b",
            vec!["middle"],
            vec!["middle"],
            make_transform_action(|input| {
                let val = input.get("middle").cloned().unwrap_or(json!(null));
                Ok(vec![("final_out".to_string(), json!({"result": val}))])
            }),
        ));

        let result = executor.run(json!({"input": "start"})).await.unwrap();

        assert!(result.get("final_out").is_some());
        assert_eq!(result["final_out"]["result"]["processed"], json!("start"));
        assert_eq!(executor.step(), 2);
    }

    #[tokio::test]
    async fn test_consume_clears_ephemeral() {
        use crate::channels::EphemeralValue;

        let mut executor = PregelExecutor::new(PregelConfig::default());

        executor.add_channel(Box::new(EphemeralValue::new("trigger")));
        executor.add_channel(Box::new(LastValue::new("output")));

        executor.add_node(make_node(
            "worker",
            vec!["trigger"],
            vec!["trigger"],
            make_write_action(vec![("output".to_string(), json!("done"))]),
        ));

        let result = executor.run(json!({"trigger": true})).await.unwrap();
        assert_eq!(result["output"], json!("done"));
        assert!(!executor.channels().get("trigger").unwrap().is_available());
    }

    #[tokio::test]
    async fn test_read_channels_empty() {
        let executor = PregelExecutor::new(PregelConfig::default());
        let state = executor.read_channels();
        assert_eq!(state, json!({}));
    }

    #[tokio::test]
    async fn test_step_counter_increments() {
        let mut executor = PregelExecutor::new(PregelConfig::default());

        executor.add_channel(Box::new(LastValue::new("input")));
        executor.add_channel(Box::new(LastValue::new("output")));

        executor.add_node(make_node(
            "node",
            vec!["input"],
            vec!["input"],
            make_write_action(vec![("output".to_string(), json!("done"))]),
        ));

        assert_eq!(executor.step(), 0);
        executor.run(json!({"input": "go"})).await.unwrap();
        assert!(executor.step() >= 1);
    }

    #[tokio::test]
    async fn test_self_loop_terminates_on_recursion_limit() {
        let mut executor = PregelExecutor::new(PregelConfig {
            recursion_limit: 5,
            ..Default::default()
        });

        executor.add_channel(Box::new(LastValue::new("state")));

        executor.add_node(make_node(
            "self_loop",
            vec!["state"],
            vec!["state"],
            make_write_action(vec![("state".to_string(), json!("same"))]),
        ));

        let result = executor.run(json!({"state": "init"})).await;
        assert!(matches!(
            result,
            Err(LangGraphError::GraphRecursionError(_))
        ));
    }

    #[tokio::test]
    async fn test_no_writes_terminates() {
        let mut executor = PregelExecutor::new(PregelConfig::default());

        executor.add_channel(Box::new(LastValue::new("input")));

        executor.add_node(make_node(
            "sink",
            vec!["input"],
            vec!["input"],
            make_write_action(vec![]),
        ));

        let result = executor.run(json!({"input": "go"})).await.unwrap();
        assert_eq!(result["input"], json!("go"));
        assert_eq!(executor.status(), PregelStatus::Done);
    }

    #[tokio::test]
    async fn test_diamond_graph() {
        let mut executor = PregelExecutor::new(PregelConfig::default());

        executor.add_channel(Box::new(LastValue::new("input")));
        executor.add_channel(Box::new(LastValue::new("a_out")));
        executor.add_channel(Box::new(LastValue::new("b_out")));
        executor.add_channel(Box::new(LastValue::new("merged")));

        executor.add_node(make_node(
            "node_a",
            vec!["input"],
            vec!["input"],
            make_write_action(vec![("a_out".to_string(), json!("from_a"))]),
        ));

        executor.add_node(make_node(
            "node_b",
            vec!["input"],
            vec!["input"],
            make_write_action(vec![("b_out".to_string(), json!("from_b"))]),
        ));

        executor.add_node(make_node(
            "merge",
            vec!["a_out", "b_out"],
            vec!["a_out", "b_out"],
            make_transform_action(|input| {
                let a = input.get("a_out").cloned().unwrap_or(json!(null));
                let b = input.get("b_out").cloned().unwrap_or(json!(null));
                Ok(vec![("merged".to_string(), json!({"a": a, "b": b}))])
            }),
        ));

        let result = executor.run(json!({"input": "start"})).await.unwrap();
        assert_eq!(result["merged"]["a"], json!("from_a"));
        assert_eq!(result["merged"]["b"], json!("from_b"));
    }

    #[tokio::test]
    async fn test_topic_channel_accumulation() {
        use crate::channels::Topic;

        let mut executor = PregelExecutor::new(PregelConfig::default());

        executor.add_channel(Box::new(LastValue::new("trigger")));
        executor.add_channel(Box::new(Topic::new("messages").with_accumulate(true)));
        executor.add_channel(Box::new(LastValue::new("done")));

        executor.add_node(make_node(
            "writer",
            vec!["trigger"],
            vec!["trigger"],
            make_write_action(vec![
                ("messages".to_string(), json!("hello")),
                ("done".to_string(), json!(true)),
            ]),
        ));

        let result = executor.run(json!({"trigger": "go"})).await.unwrap();
        let messages = result["messages"].as_array().unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0], json!("hello"));
    }
}
