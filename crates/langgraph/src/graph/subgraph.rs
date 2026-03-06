//! Subgraph composition support.
//!
//! [`SubgraphNode`] wraps a [`CompiledStateGraph`] so it can be used as a node
//! inside another [`StateGraph`]. This enables hierarchical graph composition
//! where an entire compiled graph executes as a single step in a parent graph.
//!
//! State isolation is enforced: the subgraph receives only the keys specified in
//! its `input_mapping`, and only the keys specified in `output_mapping` are
//! written back to the parent state. Internal subgraph keys that are not in the
//! output mapping do not leak into the parent.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use super::state::{AsyncNodeAction, CompiledStateGraph};

/// A wrapper that allows a [`CompiledStateGraph`] to be used as a node action
/// inside another [`StateGraph`].
///
/// When invoked, the `SubgraphNode`:
/// 1. Extracts keys from the parent state according to `input_mapping`
/// 2. Invokes the inner graph with the extracted (mapped) state
/// 3. Maps output keys back according to `output_mapping`
/// 4. Returns only the mapped output, ensuring state isolation
///
/// If no mappings are provided, the subgraph receives the full parent state and
/// its full output is merged back (pass-through mode).
///
/// # Example
///
/// ```rust
/// use std::collections::HashMap;
/// use std::sync::Arc;
/// use serde_json::{json, Value};
/// use langgraph::graph::state::{StateGraph, AsyncNodeAction};
/// use langgraph::graph::subgraph::SubgraphNode;
/// use langgraph::errors::LangGraphError;
///
/// let action: AsyncNodeAction = Arc::new(|_s: Value| {
///     Box::pin(async { Ok(json!({"inner_done": true})) })
/// });
///
/// let inner = StateGraph::new()
///     .add_node("inner_a", action)
///     .set_entry_point("inner_a")
///     .set_finish_point("inner_a")
///     .compile()
///     .unwrap();
///
/// let outer = StateGraph::new()
///     .add_subgraph("my_subgraph", inner)
///     .set_entry_point("my_subgraph")
///     .set_finish_point("my_subgraph")
///     .compile()
///     .unwrap();
/// ```
pub struct SubgraphNode {
    /// The compiled subgraph to execute.
    graph: Arc<CompiledStateGraph>,
    /// Mapping from parent state keys to subgraph input keys.
    /// If empty, the full parent state is passed through.
    input_mapping: HashMap<String, String>,
    /// Mapping from subgraph output keys to parent state keys.
    /// If empty, the full subgraph output is passed through.
    output_mapping: HashMap<String, String>,
}

impl SubgraphNode {
    /// Create a new `SubgraphNode` wrapping the given compiled graph.
    ///
    /// Without mappings, the subgraph receives the full parent state and its
    /// full output is merged back (pass-through mode).
    pub fn new(graph: CompiledStateGraph) -> Self {
        Self {
            graph: Arc::new(graph),
            input_mapping: HashMap::new(),
            output_mapping: HashMap::new(),
        }
    }

    /// Create a new `SubgraphNode` with explicit input and output state mappings.
    ///
    /// - `input_mapping`: `parent_key -> subgraph_key` — which keys from the
    ///   parent state are extracted and renamed for the subgraph input.
    /// - `output_mapping`: `subgraph_key -> parent_key` — which keys from the
    ///   subgraph output are extracted and renamed for the parent state.
    ///
    /// This ensures state isolation: the subgraph only sees mapped inputs and
    /// only mapped outputs propagate back to the parent.
    pub fn with_mapping(
        graph: CompiledStateGraph,
        input_mapping: HashMap<String, String>,
        output_mapping: HashMap<String, String>,
    ) -> Self {
        Self {
            graph: Arc::new(graph),
            input_mapping,
            output_mapping,
        }
    }

    /// Convert this `SubgraphNode` into an [`AsyncNodeAction`] that can be
    /// added to a [`StateGraph`].
    pub fn into_action(self) -> AsyncNodeAction {
        let graph = self.graph;
        let input_mapping = self.input_mapping;
        let output_mapping = self.output_mapping;

        Arc::new(move |state: Value| {
            let graph = graph.clone();
            let input_mapping = input_mapping.clone();
            let output_mapping = output_mapping.clone();

            Box::pin(async move {
                // Build subgraph input from parent state.
                let subgraph_input = if input_mapping.is_empty() {
                    state
                } else {
                    map_state_keys(&state, &input_mapping)
                };

                // Invoke the subgraph.
                let subgraph_output = graph.invoke(subgraph_input).await?;

                // Build parent output from subgraph output.
                if output_mapping.is_empty() {
                    Ok(subgraph_output)
                } else {
                    Ok(map_state_keys(&subgraph_output, &output_mapping))
                }
            })
        })
    }
}

/// Extract and rename keys from a JSON object according to the given mapping.
///
/// For each `(source_key, target_key)` pair in the mapping, the value at
/// `source_key` in `state` is placed at `target_key` in the result. Keys not
/// present in the source state are silently skipped.
fn map_state_keys(state: &Value, mapping: &HashMap<String, String>) -> Value {
    let mut result = serde_json::Map::new();
    if let Value::Object(obj) = state {
        for (source_key, target_key) in mapping {
            if let Some(value) = obj.get(source_key) {
                result.insert(target_key.clone(), value.clone());
            }
        }
    }
    Value::Object(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::state::{AsyncNodeAction, StateGraph};
    use serde_json::json;

    /// Helper: build a simple single-node subgraph that applies the given action.
    fn build_single_node_subgraph(action: AsyncNodeAction) -> CompiledStateGraph {
        StateGraph::new()
            .add_node("inner", action)
            .set_entry_point("inner")
            .set_finish_point("inner")
            .compile()
            .unwrap()
    }

    /// Test 1: Simple subgraph that transforms a value.
    #[tokio::test]
    async fn test_simple_subgraph_transforms_value() {
        let inner_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "x": val * 2 }))
            })
        });

        let inner = build_single_node_subgraph(inner_action);

        let outer = StateGraph::new()
            .add_subgraph("sub", inner)
            .set_entry_point("sub")
            .set_finish_point("sub")
            .compile()
            .unwrap();

        let result = outer.invoke(json!({ "x": 5 })).await.unwrap();
        assert_eq!(result["x"], 10);
    }

    /// Test 2: Subgraph with input/output mapping.
    #[tokio::test]
    async fn test_subgraph_with_input_output_mapping() {
        // The subgraph expects "input_val" and produces "output_val".
        let inner_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("input_val").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "output_val": val + 100 }))
            })
        });

        let inner = build_single_node_subgraph(inner_action);

        // Parent state has "parent_x", map it to subgraph's "input_val".
        // Subgraph produces "output_val", map it back to "parent_result".
        let mut input_map = HashMap::new();
        input_map.insert("parent_x".to_string(), "input_val".to_string());

        let mut output_map = HashMap::new();
        output_map.insert("output_val".to_string(), "parent_result".to_string());

        let sub_node = SubgraphNode::with_mapping(inner, input_map, output_map);
        let outer = StateGraph::new()
            .add_node("sub", sub_node.into_action())
            .set_entry_point("sub")
            .set_finish_point("sub")
            .compile()
            .unwrap();

        let result = outer
            .invoke(json!({ "parent_x": 42, "other_key": "hello" }))
            .await
            .unwrap();

        // parent_result should be 42 + 100 = 142
        assert_eq!(result["parent_result"], 142);
        // The original parent keys should still be present (merged state).
        assert_eq!(result["parent_x"], 42);
        assert_eq!(result["other_key"], "hello");
    }

    /// Test 3: Subgraph with its own internal multi-step nodes.
    #[tokio::test]
    async fn test_subgraph_multi_step() {
        let step1: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "x": val + 1 }))
            })
        });

        let step2: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "x": val * 3 }))
            })
        });

        let inner = StateGraph::new()
            .add_node("step1", step1)
            .add_node("step2", step2)
            .set_entry_point("step1")
            .add_edge("step1", "step2")
            .set_finish_point("step2")
            .compile()
            .unwrap();

        let outer = StateGraph::new()
            .add_subgraph("sub", inner)
            .set_entry_point("sub")
            .set_finish_point("sub")
            .compile()
            .unwrap();

        // x starts at 5, step1 -> 6, step2 -> 18
        let result = outer.invoke(json!({ "x": 5 })).await.unwrap();
        assert_eq!(result["x"], 18);
    }

    /// Test 4: Nested subgraphs (subgraph within subgraph).
    #[tokio::test]
    async fn test_nested_subgraphs() {
        let leaf_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("v").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "v": val + 10 }))
            })
        });

        // Inner-most subgraph: adds 10
        let innermost = build_single_node_subgraph(leaf_action);

        // Middle subgraph: contains the innermost subgraph as a node
        let middle = StateGraph::new()
            .add_subgraph("nested_inner", innermost)
            .set_entry_point("nested_inner")
            .set_finish_point("nested_inner")
            .compile()
            .unwrap();

        // Outer graph: contains the middle subgraph
        let outer = StateGraph::new()
            .add_subgraph("nested_middle", middle)
            .set_entry_point("nested_middle")
            .set_finish_point("nested_middle")
            .compile()
            .unwrap();

        let result = outer.invoke(json!({ "v": 5 })).await.unwrap();
        assert_eq!(result["v"], 15);
    }

    /// Test 5: State isolation — subgraph internal state does not leak to parent.
    #[tokio::test]
    async fn test_state_isolation() {
        // Subgraph sets an internal key "internal_secret" that should not
        // propagate to the parent when output mapping is used.
        let inner_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({
                    "a": val * 2,
                    "result": val * 2,
                    "internal_secret": "should_not_leak"
                }))
            })
        });

        let inner = build_single_node_subgraph(inner_action);

        let mut input_map = HashMap::new();
        input_map.insert("a".to_string(), "a".to_string());

        let mut output_map = HashMap::new();
        output_map.insert("result".to_string(), "result".to_string());
        // Deliberately do NOT map "internal_secret"

        let sub_node = SubgraphNode::with_mapping(inner, input_map, output_map);
        let outer = StateGraph::new()
            .add_node("sub", sub_node.into_action())
            .set_entry_point("sub")
            .set_finish_point("sub")
            .compile()
            .unwrap();

        let result = outer.invoke(json!({ "a": 7 })).await.unwrap();

        // "result" should be present (mapped output).
        assert_eq!(result["result"], 14);
        // "internal_secret" should NOT be present (state isolation).
        assert!(result.get("internal_secret").is_none());
        // Original parent key "a" should still be 7 (not overwritten by subgraph).
        assert_eq!(result["a"], 7);
    }

    /// Test 6: Subgraph in a larger pipeline with other nodes before and after.
    #[tokio::test]
    async fn test_subgraph_in_pipeline() {
        let pre_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "x": val + 1, "stage": "pre" }))
            })
        });

        let sub_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "x": val * 10 }))
            })
        });

        let post_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("x").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "x": val + 1000, "stage": "post" }))
            })
        });

        let inner = build_single_node_subgraph(sub_action);

        let outer = StateGraph::new()
            .add_node("pre", pre_action)
            .add_subgraph("sub", inner)
            .add_node("post", post_action)
            .set_entry_point("pre")
            .add_edge("pre", "sub")
            .add_edge("sub", "post")
            .set_finish_point("post")
            .compile()
            .unwrap();

        // x=2 -> pre: 3 -> sub: 30 -> post: 1030
        let result = outer.invoke(json!({ "x": 2 })).await.unwrap();
        assert_eq!(result["x"], 1030);
        assert_eq!(result["stage"], "post");
    }

    /// Test 7: add_subgraph_with_mapping convenience method on StateGraph.
    #[tokio::test]
    async fn test_add_subgraph_with_mapping_convenience() {
        let inner_action: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let val = state.get("sub_input").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "sub_output": val + 50 }))
            })
        });

        let inner = build_single_node_subgraph(inner_action);

        let mut input_map = HashMap::new();
        input_map.insert("parent_val".to_string(), "sub_input".to_string());

        let mut output_map = HashMap::new();
        output_map.insert("sub_output".to_string(), "parent_result".to_string());

        let outer = StateGraph::new()
            .add_subgraph_with_mapping("sub", inner, input_map, output_map)
            .set_entry_point("sub")
            .set_finish_point("sub")
            .compile()
            .unwrap();

        let result = outer.invoke(json!({ "parent_val": 25 })).await.unwrap();

        assert_eq!(result["parent_result"], 75);
        assert_eq!(result["parent_val"], 25);
    }
}
