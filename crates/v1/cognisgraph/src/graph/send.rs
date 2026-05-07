//! Send API for fan-out/fan-in and map-reduce patterns.
//!
//! This module provides primitives for dispatching work to specific graph nodes
//! with custom inputs, executing them in parallel, and merging the results back
//! together. It enables map-reduce style workflows on top of [`CompiledStateGraph`].
//!
//! # Key types
//!
//! - [`SendCommand`] — a directive to invoke a specific node with a given input.
//! - [`MapReduceGraph`] — wraps a compiled graph to split input, fan-out to a
//!   map node, and reduce the results via a reduce node.
//!
//! # Key functions
//!
//! - [`send_to`] — convenience constructor for [`SendCommand`].
//! - [`fan_out`] — execute multiple [`SendCommand`]s in parallel.
//! - [`fan_in`] — merge a vector of results using a caller-supplied function.

use std::sync::Arc;

use serde_json::Value;

use crate::errors::LangGraphError;
use crate::graph::state::CompiledStateGraph;

/// A command to dispatch work to a specific node with a custom input.
///
/// `SendCommand` decouples the *what* (which node to run) from the *where*
/// (the current position in the graph), allowing callers to programmatically
/// fan work out to arbitrary nodes.
#[derive(Debug, Clone)]
pub struct SendCommand {
    /// The name of the target node to invoke.
    pub node: String,
    /// The input value to pass to the target node.
    pub input: Value,
}

impl SendCommand {
    /// Create a new `SendCommand`.
    pub fn new(node: impl Into<String>, input: Value) -> Self {
        Self {
            node: node.into(),
            input,
        }
    }
}

/// Convenience function to create a [`SendCommand`].
///
/// # Example
///
/// ```rust
/// use serde_json::json;
/// use cognisgraph::graph::send::send_to;
///
/// let cmd = send_to("summarize", json!({"text": "hello world"}));
/// assert_eq!(cmd.node, "summarize");
/// ```
pub fn send_to(node_name: impl Into<String>, input: Value) -> SendCommand {
    SendCommand::new(node_name, input)
}

/// Execute a list of [`SendCommand`]s in parallel against a compiled graph.
///
/// Each command invokes the named node's action with the command's input.
/// All invocations run concurrently via [`tokio::spawn`] and the results are
/// collected in order.
///
/// Returns a `Vec<Value>` with one result per command, preserving the input
/// order.
///
/// # Errors
///
/// Returns an error if any node is not found in the graph or if any node
/// action fails.
pub async fn fan_out(
    graph: &CompiledStateGraph,
    commands: Vec<SendCommand>,
) -> Result<Vec<Value>, LangGraphError> {
    if commands.is_empty() {
        return Ok(Vec::new());
    }

    let mut handles = Vec::with_capacity(commands.len());

    for cmd in commands {
        let node = graph.nodes.get(&cmd.node).ok_or_else(|| {
            LangGraphError::Other(format!(
                "fan_out: node '{}' not found in compiled graph",
                cmd.node
            ))
        })?;
        let action = node.action.clone();
        let input = cmd.input;

        handles.push(tokio::spawn(async move { (action)(input).await }));
    }

    let mut results = Vec::with_capacity(handles.len());
    for handle in handles {
        let result = handle
            .await
            .map_err(|e| LangGraphError::Other(format!("fan_out: task join error: {e}")))??;
        results.push(result);
    }

    Ok(results)
}

/// Merge a vector of [`Value`]s into a single value using a caller-supplied
/// merge function.
///
/// This is the complement of [`fan_out`]: after fanning work out to multiple
/// nodes, `fan_in` combines their outputs.
///
/// If no custom merge function is needed, use [`deep_merge_values`] as the
/// default strategy.
pub fn fan_in<F>(values: Vec<Value>, merge_fn: F) -> Value
where
    F: FnOnce(Vec<Value>) -> Value,
{
    merge_fn(values)
}

/// Default deep-merge strategy for fan-in.
///
/// Objects are merged key-by-key (later values overwrite earlier ones for the
/// same key). Non-object values are collected into a JSON array.
pub fn deep_merge_values(values: Vec<Value>) -> Value {
    if values.is_empty() {
        return Value::Null;
    }

    let all_objects = values.iter().all(|v| v.is_object());

    if all_objects {
        let mut merged = serde_json::Map::new();
        for val in values {
            if let Value::Object(map) = val {
                for (k, v) in map {
                    merged.insert(k, v);
                }
            }
        }
        Value::Object(merged)
    } else {
        Value::Array(values)
    }
}

/// A merge function type used by [`MapReduceGraph`].
pub type MergeFn = Arc<dyn Fn(Vec<Value>) -> Value + Send + Sync>;

/// A split function type used by [`MapReduceGraph`].
pub type SplitFn = Arc<dyn Fn(Value) -> Vec<Value> + Send + Sync>;

/// A high-level map-reduce executor built on top of a [`CompiledStateGraph`].
///
/// `MapReduceGraph` splits an input into chunks, fans each chunk out to a
/// *map node*, collects the results, and then passes the merged output through
/// a *reduce node*.
///
/// # Concurrency
///
/// By default all map invocations run in parallel. Set `max_concurrency` to
/// bound the number of concurrent tasks.
pub struct MapReduceGraph {
    /// The underlying compiled graph whose nodes are used for map and reduce.
    pub graph: CompiledStateGraph,
    /// The node to fan-out to (map phase).
    pub map_node: String,
    /// The node to combine results (reduce phase).
    pub reduce_node: String,
    /// Splits the input into chunks for the map phase.
    pub split_fn: SplitFn,
    /// Merges the map results before passing to the reduce node.
    pub merge_fn: MergeFn,
    /// Optional upper bound on parallel map tasks.
    pub max_concurrency: Option<usize>,
}

impl MapReduceGraph {
    /// Create a new `MapReduceGraph`.
    pub fn new(
        graph: CompiledStateGraph,
        map_node: impl Into<String>,
        reduce_node: impl Into<String>,
        split_fn: SplitFn,
        merge_fn: MergeFn,
    ) -> Self {
        Self {
            graph,
            map_node: map_node.into(),
            reduce_node: reduce_node.into(),
            split_fn,
            merge_fn,
            max_concurrency: None,
        }
    }

    /// Set the maximum concurrency for the map phase.
    pub fn with_max_concurrency(mut self, max: usize) -> Self {
        self.max_concurrency = Some(max);
        self
    }

    /// Execute the map-reduce pipeline.
    ///
    /// 1. Split the input using `split_fn`.
    /// 2. Fan-out: invoke `map_node` for each chunk (respecting `max_concurrency`).
    /// 3. Merge the map results using `merge_fn`.
    /// 4. Invoke `reduce_node` with the merged value.
    pub async fn execute(&self, input: Value) -> Result<Value, LangGraphError> {
        let chunks = (self.split_fn)(input);

        // Build SendCommands for the map phase.
        let commands: Vec<SendCommand> = chunks
            .into_iter()
            .map(|chunk| SendCommand::new(self.map_node.clone(), chunk))
            .collect();

        let map_results = match self.max_concurrency {
            Some(max) => self.fan_out_bounded(commands, max).await?,
            None => fan_out(&self.graph, commands).await?,
        };

        // Merge the map results.
        let merged = (self.merge_fn)(map_results);

        // Reduce phase: invoke the reduce node with the merged value.
        let reduce_node = self.graph.nodes.get(&self.reduce_node).ok_or_else(|| {
            LangGraphError::Other(format!(
                "MapReduceGraph: reduce node '{}' not found",
                self.reduce_node
            ))
        })?;

        (reduce_node.action)(merged).await
    }

    /// Fan-out with a bounded number of concurrent tasks.
    async fn fan_out_bounded(
        &self,
        commands: Vec<SendCommand>,
        max_concurrency: usize,
    ) -> Result<Vec<Value>, LangGraphError> {
        use tokio::sync::Semaphore;

        if commands.is_empty() {
            return Ok(Vec::new());
        }

        let semaphore = Arc::new(Semaphore::new(max_concurrency));
        let mut handles = Vec::with_capacity(commands.len());

        for cmd in commands {
            let node = self.graph.nodes.get(&cmd.node).ok_or_else(|| {
                LangGraphError::Other(format!(
                    "fan_out_bounded: node '{}' not found in compiled graph",
                    cmd.node
                ))
            })?;
            let action = node.action.clone();
            let input = cmd.input;
            let sem = semaphore.clone();

            handles.push(tokio::spawn(async move {
                let _permit = sem
                    .acquire()
                    .await
                    .map_err(|e| LangGraphError::Other(format!("semaphore error: {e}")))?;
                (action)(input).await
            }));
        }

        let mut results = Vec::with_capacity(handles.len());
        for handle in handles {
            let result = handle.await.map_err(|e| {
                LangGraphError::Other(format!("fan_out_bounded: task join error: {e}"))
            })??;
            results.push(result);
        }

        Ok(results)
    }
}

impl std::fmt::Debug for MapReduceGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MapReduceGraph")
            .field("map_node", &self.map_node)
            .field("reduce_node", &self.reduce_node)
            .field("max_concurrency", &self.max_concurrency)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;

    use crate::graph::state::{AsyncNodeAction, StateGraph};

    /// Helper: build a simple graph with the given named nodes and finish points.
    fn build_graph(nodes: Vec<(&str, AsyncNodeAction)>) -> CompiledStateGraph {
        let mut builder = StateGraph::new();
        let first_name = nodes[0].0.to_string();
        for (name, action) in &nodes {
            builder = builder.add_node(name, action.clone());
        }
        builder = builder.set_entry_point(&first_name);
        for (name, _) in &nodes {
            builder = builder.set_finish_point(name);
        }
        builder.compile().unwrap()
    }

    // -----------------------------------------------------------------------
    // Test 1: SendCommand creation and fields
    // -----------------------------------------------------------------------
    #[test]
    fn test_send_command_creation_and_fields() {
        let cmd = SendCommand::new("my_node", json!({"key": "value"}));
        assert_eq!(cmd.node, "my_node");
        assert_eq!(cmd.input, json!({"key": "value"}));
    }

    // -----------------------------------------------------------------------
    // Test 2: send_to convenience function
    // -----------------------------------------------------------------------
    #[test]
    fn test_send_to_convenience() {
        let cmd = send_to("target", json!(42));
        assert_eq!(cmd.node, "target");
        assert_eq!(cmd.input, json!(42));
    }

    // -----------------------------------------------------------------------
    // Test 3: fan_out executes nodes in parallel
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_fan_out_executes_nodes_in_parallel() {
        let action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let n = state.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({"result": n * 2}))
            })
        });

        let graph = build_graph(vec![("double", action)]);

        let commands = vec![
            send_to("double", json!({"n": 1})),
            send_to("double", json!({"n": 2})),
            send_to("double", json!({"n": 3})),
        ];

        let results = fan_out(&graph, commands).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], json!({"result": 2}));
        assert_eq!(results[1], json!({"result": 4}));
        assert_eq!(results[2], json!({"result": 6}));
    }

    // -----------------------------------------------------------------------
    // Test 4: fan_in merges results
    // -----------------------------------------------------------------------
    #[test]
    fn test_fan_in_merges_results() {
        let values = vec![json!({"a": 1}), json!({"b": 2}), json!({"c": 3})];

        let merged = fan_in(values, deep_merge_values);
        assert_eq!(merged, json!({"a": 1, "b": 2, "c": 3}));
    }

    // -----------------------------------------------------------------------
    // Test 5: fan_in with custom merge function
    // -----------------------------------------------------------------------
    #[test]
    fn test_fan_in_custom_merge() {
        let values = vec![json!(1), json!(2), json!(3)];

        let sum = fan_in(values, |vals| {
            let total: i64 = vals.iter().filter_map(|v| v.as_i64()).sum();
            json!(total)
        });
        assert_eq!(sum, json!(6));
    }

    // -----------------------------------------------------------------------
    // Test 6: MapReduceGraph splits, maps, and reduces
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_map_reduce_graph_basic() {
        // Map node: doubles the "n" field.
        let map_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let n = state.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({"value": n * 2}))
            })
        });

        // Reduce node: sums the "values" array.
        let reduce_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let values = state
                    .get("values")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let total: i64 = values
                    .iter()
                    .filter_map(|v| v.get("value").and_then(|n| n.as_i64()))
                    .sum();
                Ok(json!({"total": total}))
            })
        });

        let graph = build_graph(vec![("mapper", map_action), ("reducer", reduce_action)]);

        let mr = MapReduceGraph::new(
            graph,
            "mapper",
            "reducer",
            Arc::new(|input| {
                // Split: extract "items" array into individual chunks.
                input
                    .get("items")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|item| json!({"n": item}))
                    .collect()
            }),
            Arc::new(|results| {
                // Merge: wrap results in a "values" key.
                json!({"values": results})
            }),
        );

        let result = mr.execute(json!({"items": [1, 2, 3, 4]})).await.unwrap();
        // Each item doubled: 2 + 4 + 6 + 8 = 20
        assert_eq!(result, json!({"total": 20}));
    }

    // -----------------------------------------------------------------------
    // Test 7: MapReduceGraph with concurrency limit
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_map_reduce_graph_with_concurrency_limit() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        let peak = Arc::new(AtomicUsize::new(0));
        let current = Arc::new(AtomicUsize::new(0));
        let peak_clone = peak.clone();
        let current_clone = current.clone();

        let map_action: AsyncNodeAction = Arc::new(move |state: Value| {
            let peak = peak_clone.clone();
            let current = current_clone.clone();
            Box::pin(async move {
                let count = current.fetch_add(1, Ordering::SeqCst) + 1;
                peak.fetch_max(count, Ordering::SeqCst);
                // Yield to let other tasks run.
                tokio::task::yield_now().await;
                current.fetch_sub(1, Ordering::SeqCst);
                let n = state.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({"result": n}))
            })
        });

        let reduce_action: AsyncNodeAction =
            Arc::new(|state: Value| Box::pin(async move { Ok(state) }));

        let graph = build_graph(vec![("mapper", map_action), ("reducer", reduce_action)]);

        let mr = MapReduceGraph::new(
            graph,
            "mapper",
            "reducer",
            Arc::new(|input| {
                let count = input.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                (0..count).map(|i| json!({"n": i})).collect()
            }),
            Arc::new(|results| json!({"results": results})),
        )
        .with_max_concurrency(2);

        let result = mr.execute(json!({"count": 6})).await.unwrap();

        let results = result.get("results").and_then(|v| v.as_array()).unwrap();
        assert_eq!(results.len(), 6);

        // Peak concurrency should not exceed 2.
        assert!(peak.load(Ordering::SeqCst) <= 2);
    }

    // -----------------------------------------------------------------------
    // Test 8: fan_out with different inputs per node
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_fan_out_different_inputs_per_node() {
        let action_a: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let x = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({"from_a": x + 10}))
            })
        });

        let action_b: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let y = state.get("y").and_then(|v| v.as_str()).unwrap_or("");
                Ok(json!({"from_b": format!("hello {}", y)}))
            })
        });

        let graph = build_graph(vec![("node_a", action_a), ("node_b", action_b)]);

        let commands = vec![
            send_to("node_a", json!({"x": 5})),
            send_to("node_b", json!({"y": "world"})),
        ];

        let results = fan_out(&graph, commands).await.unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0], json!({"from_a": 15}));
        assert_eq!(results[1], json!({"from_b": "hello world"}));
    }

    // -----------------------------------------------------------------------
    // Test 9: empty send commands
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_fan_out_empty_commands() {
        let action: AsyncNodeAction = Arc::new(|_| Box::pin(async { Ok(json!({})) }));
        let graph = build_graph(vec![("noop", action)]);

        let results = fan_out(&graph, vec![]).await.unwrap();
        assert!(results.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test 10: map reduce with multi-step graph (map then transform then reduce)
    // -----------------------------------------------------------------------
    #[tokio::test]
    async fn test_map_reduce_multi_step() {
        // Map node: squares the input.
        let map_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let n = state.get("n").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({"squared": n * n}))
            })
        });

        // Transform node: adds 1 to each squared value.
        let transform_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let items = state
                    .get("items")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let transformed: Vec<Value> = items
                    .iter()
                    .filter_map(|v| v.get("squared").and_then(|n| n.as_i64()))
                    .map(|n| json!({"val": n + 1}))
                    .collect();
                Ok(json!({"transformed": transformed}))
            })
        });

        // Reduce node: sums transformed values.
        let reduce_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let items = state
                    .get("transformed")
                    .and_then(|v| v.as_array())
                    .cloned()
                    .unwrap_or_default();
                let total: i64 = items
                    .iter()
                    .filter_map(|v| v.get("val").and_then(|n| n.as_i64()))
                    .sum();
                Ok(json!({"total": total}))
            })
        });

        let graph = build_graph(vec![
            ("mapper", map_action),
            ("transform", transform_action),
            ("reducer", reduce_action),
        ]);

        // Step 1: Fan-out to mapper.
        let commands: Vec<SendCommand> = (1..=4)
            .map(|i| send_to("mapper", json!({"n": i})))
            .collect();
        let map_results = fan_out(&graph, commands).await.unwrap();

        // Step 2: Fan-in map results, then send to transform.
        let merged_for_transform = json!({"items": map_results});
        let transform_commands = vec![send_to("transform", merged_for_transform)];
        let transform_results = fan_out(&graph, transform_commands).await.unwrap();

        // Step 3: Send transform output to reducer.
        let reduce_commands = vec![send_to("reducer", transform_results[0].clone())];
        let final_results = fan_out(&graph, reduce_commands).await.unwrap();

        // 1^2+1 + 2^2+1 + 3^2+1 + 4^2+1 = 2+5+10+17 = 34
        assert_eq!(final_results[0], json!({"total": 34}));
    }

    // -----------------------------------------------------------------------
    // Test 11: deep_merge_values with non-object values
    // -----------------------------------------------------------------------
    #[test]
    fn test_deep_merge_non_objects() {
        let values = vec![json!(1), json!("hello"), json!(true)];
        let merged = deep_merge_values(values.clone());
        assert_eq!(merged, Value::Array(values));
    }

    // -----------------------------------------------------------------------
    // Test 12: deep_merge_values with empty vec
    // -----------------------------------------------------------------------
    #[test]
    fn test_deep_merge_empty() {
        let merged = deep_merge_values(vec![]);
        assert_eq!(merged, Value::Null);
    }
}
