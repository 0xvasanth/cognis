//! Fine-grained graph streaming events.
//!
//! Provides [`GraphStreamEvent`], a structured event type that exposes the
//! full lifecycle of a graph execution (graph start/end, node start/end,
//! state updates, errors, and custom events).
//!
//! The [`stream_graph_events`] function drives a [`CompiledStateGraph`] and
//! returns an async [`Stream`] of [`GraphStreamEvent`] values.

use std::pin::Pin;

use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::errors::LangGraphError;

use super::state::CompiledStateGraph;

// ---------------------------------------------------------------------------
// GraphStreamEvent
// ---------------------------------------------------------------------------

/// A fine-grained event emitted during graph execution.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type")]
pub enum GraphStreamEvent {
    /// Emitted once at the very start of graph execution.
    #[serde(rename = "graph_start")]
    GraphStart {
        /// The initial input to the graph.
        input: Value,
    },

    /// Emitted once when graph execution completes successfully.
    #[serde(rename = "graph_end")]
    GraphEnd {
        /// The final output (merged state) of the graph.
        output: Value,
    },

    /// Emitted when a node begins execution.
    #[serde(rename = "node_start")]
    NodeStart {
        /// The name of the node.
        node: String,
        /// The input passed to the node.
        input: Value,
    },

    /// Emitted when a node finishes execution successfully.
    #[serde(rename = "node_end")]
    NodeEnd {
        /// The name of the node.
        node: String,
        /// The output (update) returned by the node.
        output: Value,
    },

    /// Emitted when a node encounters an error during execution.
    #[serde(rename = "node_error")]
    NodeError {
        /// The name of the node that failed.
        node: String,
        /// A description of the error.
        error: String,
    },

    /// Emitted after a node's output has been merged into the graph state.
    #[serde(rename = "state_update")]
    StateUpdate {
        /// The full graph state after the merge.
        state: Value,
    },

    /// An extensible custom event for user-defined instrumentation.
    #[serde(rename = "custom")]
    Custom {
        /// A user-defined event type string.
        event_type: String,
        /// Arbitrary data payload.
        data: Value,
    },
}

// ---------------------------------------------------------------------------
// stream_graph_events
// ---------------------------------------------------------------------------

/// Drive a compiled graph to completion, emitting fine-grained
/// [`GraphStreamEvent`] values as an async [`Stream`].
///
/// The returned stream emits events in the following order:
///
/// 1. [`GraphStreamEvent::GraphStart`] — once, with the initial input.
/// 2. For each executed node:
///    - [`GraphStreamEvent::NodeStart`]
///    - [`GraphStreamEvent::NodeEnd`] (on success) **or** [`GraphStreamEvent::NodeError`] (on failure)
///    - [`GraphStreamEvent::StateUpdate`] (on success, after state merge)
/// 3. [`GraphStreamEvent::GraphEnd`] — once, with the final state.
///
/// If a node fails, the stream emits `NodeError` and then terminates
/// (no `GraphEnd` is emitted).
pub fn stream_graph_events(
    graph: &CompiledStateGraph,
    input: Value,
    _config: Option<Value>,
) -> Pin<Box<dyn Stream<Item = Result<GraphStreamEvent, LangGraphError>> + Send>> {
    use std::collections::HashMap;

    use crate::constants::{END, START};

    use super::state::AsyncNodeAction;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<GraphStreamEvent, LangGraphError>>(64);

    // Clone the pieces we need into the spawned task (same pattern as
    // CompiledStateGraph::stream).
    let nodes: HashMap<String, AsyncNodeAction> = graph
        .nodes
        .iter()
        .map(|(k, v)| (k.clone(), v.action.clone()))
        .collect();

    let graph_clone = graph.clone();
    let input_clone = input.clone();

    tokio::spawn(async move {
        // Emit GraphStart.
        if tx
            .send(Ok(GraphStreamEvent::GraphStart {
                input: input_clone.clone(),
            }))
            .await
            .is_err()
        {
            return;
        }

        let mut state = input_clone;
        let mut step_count: usize = 0;
        let recursion_limit = 25 * nodes.len().max(1);

        // Resolve the initial nodes from START using the static helper.
        let mut current_nodes =
            match CompiledStateGraph::get_next_nodes_static_pub(&graph_clone, START, &state) {
                Ok(n) => n,
                Err(e) => {
                    let _ = tx.send(Err(e)).await;
                    return;
                }
            };

        loop {
            if current_nodes.is_empty() {
                break;
            }

            let executable: Vec<String> = current_nodes
                .iter()
                .filter(|n| n.as_str() != END)
                .cloned()
                .collect();

            if executable.is_empty() {
                break;
            }

            let mut next_nodes: Vec<String> = Vec::new();

            for node_name in &executable {
                step_count += 1;
                if step_count > recursion_limit {
                    let _ = tx
                        .send(Err(LangGraphError::GraphRecursionError(format!(
                            "Recursion limit of {} reached after {} steps",
                            recursion_limit, step_count
                        ))))
                        .await;
                    return;
                }

                let action = match nodes.get(node_name) {
                    Some(a) => a.clone(),
                    None => {
                        let _ = tx
                            .send(Err(LangGraphError::Other(format!(
                                "Node '{}' not found in compiled graph",
                                node_name
                            ))))
                            .await;
                        return;
                    }
                };

                // Emit NodeStart.
                if tx
                    .send(Ok(GraphStreamEvent::NodeStart {
                        node: node_name.clone(),
                        input: state.clone(),
                    }))
                    .await
                    .is_err()
                {
                    return;
                }

                // Execute the node action.
                let update = match action(state.clone()).await {
                    Ok(u) => u,
                    Err(e) => {
                        // Emit NodeError.
                        let _ = tx
                            .send(Ok(GraphStreamEvent::NodeError {
                                node: node_name.clone(),
                                error: e.to_string(),
                            }))
                            .await;
                        // Terminate with the underlying error.
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                };

                // Emit NodeEnd.
                if tx
                    .send(Ok(GraphStreamEvent::NodeEnd {
                        node: node_name.clone(),
                        output: update.clone(),
                    }))
                    .await
                    .is_err()
                {
                    return;
                }

                // Merge update into state.
                match CompiledStateGraph::merge_state_pub(&mut state, update) {
                    Ok(()) => {}
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                }

                // Emit StateUpdate.
                if tx
                    .send(Ok(GraphStreamEvent::StateUpdate {
                        state: state.clone(),
                    }))
                    .await
                    .is_err()
                {
                    return;
                }

                // Resolve next nodes.
                match CompiledStateGraph::get_next_nodes_static_pub(&graph_clone, node_name, &state)
                {
                    Ok(mut successors) => next_nodes.append(&mut successors),
                    Err(e) => {
                        let _ = tx.send(Err(e)).await;
                        return;
                    }
                }
            }

            current_nodes = next_nodes;
        }

        // Emit GraphEnd.
        let _ = tx
            .send(Ok(GraphStreamEvent::GraphEnd { output: state }))
            .await;
    });

    Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
}

// ---------------------------------------------------------------------------
// GraphEventCollector
// ---------------------------------------------------------------------------

/// Utility for collecting all graph stream events into a `Vec`.
///
/// Useful for testing and debugging.
pub struct GraphEventCollector;

impl GraphEventCollector {
    /// Consume the entire event stream and collect successful events into a
    /// `Vec`. Returns an error if any event in the stream is an `Err`.
    pub async fn collect_all(
        stream: Pin<Box<dyn Stream<Item = Result<GraphStreamEvent, LangGraphError>> + Send>>,
    ) -> Result<Vec<GraphStreamEvent>, LangGraphError> {
        use futures::StreamExt;

        let mut events = Vec::new();
        let mut stream = stream;
        while let Some(item) = stream.next().await {
            events.push(item?);
        }
        Ok(events)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;

    use crate::graph::branch::RouterResult;
    use crate::graph::state::StateGraph;

    /// Helper: build a simple linear graph A -> B -> END.
    fn build_linear_graph() -> CompiledStateGraph {
        StateGraph::new()
            .add_node(
                "A",
                Arc::new(|state: Value| {
                    Box::pin(async move {
                        let mut s = state;
                        if let Some(obj) = s.as_object_mut() {
                            obj.insert("a".into(), json!("done"));
                        }
                        Ok(s)
                    })
                }),
            )
            .add_node(
                "B",
                Arc::new(|state: Value| {
                    Box::pin(async move {
                        let mut s = state;
                        if let Some(obj) = s.as_object_mut() {
                            obj.insert("b".into(), json!("done"));
                        }
                        Ok(s)
                    })
                }),
            )
            .add_edge("__start__", "A")
            .add_edge("A", "B")
            .add_edge("B", "__end__")
            .compile()
            .unwrap()
    }

    /// Helper: build a conditional graph START -> router -> (A | B) -> END.
    fn build_conditional_graph(route_to: &'static str) -> CompiledStateGraph {
        StateGraph::new()
            .add_node(
                "router",
                Arc::new(|state: Value| Box::pin(async move { Ok(state) })),
            )
            .add_node(
                "A",
                Arc::new(|state: Value| {
                    Box::pin(async move {
                        let mut s = state;
                        if let Some(obj) = s.as_object_mut() {
                            obj.insert("branch".into(), json!("A"));
                        }
                        Ok(s)
                    })
                }),
            )
            .add_node(
                "B",
                Arc::new(|state: Value| {
                    Box::pin(async move {
                        let mut s = state;
                        if let Some(obj) = s.as_object_mut() {
                            obj.insert("branch".into(), json!("B"));
                        }
                        Ok(s)
                    })
                }),
            )
            .add_edge("__start__", "router")
            .add_conditional_edges(
                "router",
                Arc::new(move |_state: &Value| RouterResult::Single(route_to.to_string())),
                None,
            )
            .add_edge("A", "__end__")
            .add_edge("B", "__end__")
            .compile()
            .unwrap()
    }

    /// Helper: build a graph with a node that errors.
    fn build_error_graph() -> CompiledStateGraph {
        StateGraph::new()
            .add_node(
                "fail_node",
                Arc::new(|_state: Value| {
                    Box::pin(
                        async move { Err(LangGraphError::Other("intentional failure".into())) },
                    )
                }),
            )
            .add_edge("__start__", "fail_node")
            .add_edge("fail_node", "__end__")
            .compile()
            .unwrap()
    }

    // ------------------------------------------------------------------
    // Test 1: Linear graph emits correct event sequence
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn test_linear_graph_event_sequence() {
        let graph = build_linear_graph();
        let stream = stream_graph_events(&graph, json!({}), None);
        let events = GraphEventCollector::collect_all(stream).await.unwrap();

        // Expected: GraphStart, NodeStart(A), NodeEnd(A), StateUpdate, NodeStart(B), NodeEnd(B), StateUpdate, GraphEnd
        assert!(
            events.len() >= 8,
            "expected at least 8 events, got {}",
            events.len()
        );

        assert!(matches!(&events[0], GraphStreamEvent::GraphStart { .. }));

        // Find NodeStart/NodeEnd pairs for A and B in order.
        let node_names: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                GraphStreamEvent::NodeStart { node, .. } => Some(node.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(node_names, vec!["A", "B"]);

        assert!(matches!(
            events.last().unwrap(),
            GraphStreamEvent::GraphEnd { .. }
        ));
    }

    // ------------------------------------------------------------------
    // Test 2: Conditional graph emits events for taken branch only
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn test_conditional_graph_takes_branch_a() {
        let graph = build_conditional_graph("A");
        let stream = stream_graph_events(&graph, json!({}), None);
        let events = GraphEventCollector::collect_all(stream).await.unwrap();

        let node_starts: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                GraphStreamEvent::NodeStart { node, .. } => Some(node.as_str()),
                _ => None,
            })
            .collect();

        // Should execute "router" then "A", but NOT "B".
        assert!(node_starts.contains(&"router"));
        assert!(node_starts.contains(&"A"));
        assert!(!node_starts.contains(&"B"));
    }

    // ------------------------------------------------------------------
    // Test 3: GraphStart and GraphEnd wrap all node events
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn test_graph_start_end_wrap_node_events() {
        let graph = build_linear_graph();
        let stream = stream_graph_events(&graph, json!({"init": true}), None);
        let events = GraphEventCollector::collect_all(stream).await.unwrap();

        // First event must be GraphStart.
        match &events[0] {
            GraphStreamEvent::GraphStart { input } => {
                assert_eq!(input, &json!({"init": true}));
            }
            other => panic!("expected GraphStart, got {:?}", other),
        }

        // Last event must be GraphEnd.
        match events.last().unwrap() {
            GraphStreamEvent::GraphEnd { output } => {
                // Output should contain the merged state.
                assert!(output.is_object());
            }
            other => panic!("expected GraphEnd, got {:?}", other),
        }

        // All NodeStart/NodeEnd events must be between GraphStart and GraphEnd.
        let first_node_idx = events
            .iter()
            .position(|e| matches!(e, GraphStreamEvent::NodeStart { .. }))
            .unwrap();
        let last_node_idx = events
            .iter()
            .rposition(|e| matches!(e, GraphStreamEvent::NodeEnd { .. }))
            .unwrap();

        assert!(first_node_idx > 0, "NodeStart should come after GraphStart");
        assert!(
            last_node_idx < events.len() - 1,
            "NodeEnd should come before GraphEnd"
        );
    }

    // ------------------------------------------------------------------
    // Test 4: NodeStart/NodeEnd pairs for each executed node
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn test_node_start_end_pairs() {
        let graph = build_linear_graph();
        let stream = stream_graph_events(&graph, json!({}), None);
        let events = GraphEventCollector::collect_all(stream).await.unwrap();

        // Collect NodeStart and NodeEnd events and verify pairing.
        let mut starts: Vec<String> = Vec::new();
        let mut ends: Vec<String> = Vec::new();

        for e in &events {
            match e {
                GraphStreamEvent::NodeStart { node, .. } => starts.push(node.clone()),
                GraphStreamEvent::NodeEnd { node, .. } => ends.push(node.clone()),
                _ => {}
            }
        }

        assert_eq!(starts, vec!["A", "B"]);
        assert_eq!(ends, vec!["A", "B"]);

        // For each node, NodeStart must come before NodeEnd.
        for name in &["A", "B"] {
            let start_idx = events
                .iter()
                .position(|e| matches!(e, GraphStreamEvent::NodeStart { node, .. } if node == name))
                .unwrap();
            let end_idx = events
                .iter()
                .position(|e| matches!(e, GraphStreamEvent::NodeEnd { node, .. } if node == name))
                .unwrap();
            assert!(
                start_idx < end_idx,
                "NodeStart({name}) should come before NodeEnd({name})"
            );
        }
    }

    // ------------------------------------------------------------------
    // Test 5: Error in node produces NodeError event
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn test_node_error_event() {
        let graph = build_error_graph();
        let stream = stream_graph_events(&graph, json!({}), None);

        // Collect events — the collector will return an error because the
        // stream includes an Err item after the NodeError event.
        use futures::StreamExt;
        let mut stream = stream;
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(event) => events.push(event),
                Err(_) => break,
            }
        }

        // Should have GraphStart, NodeStart(fail_node), NodeError(fail_node).
        assert!(matches!(&events[0], GraphStreamEvent::GraphStart { .. }));

        let has_node_error = events.iter().any(|e| {
            matches!(e, GraphStreamEvent::NodeError { node, error } if node == "fail_node" && error.contains("intentional failure"))
        });
        assert!(has_node_error, "expected a NodeError event for fail_node");

        // Should NOT have a GraphEnd event (error terminates early).
        let has_graph_end = events
            .iter()
            .any(|e| matches!(e, GraphStreamEvent::GraphEnd { .. }));
        assert!(!has_graph_end, "should not have GraphEnd after error");
    }

    // ------------------------------------------------------------------
    // Test 6: Event collector utility works
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn test_event_collector_success() {
        let graph = build_linear_graph();
        let stream = stream_graph_events(&graph, json!({}), None);
        let events = GraphEventCollector::collect_all(stream).await.unwrap();

        // Should have collected all events.
        assert!(!events.is_empty());
        assert!(matches!(&events[0], GraphStreamEvent::GraphStart { .. }));
        assert!(matches!(
            events.last().unwrap(),
            GraphStreamEvent::GraphEnd { .. }
        ));
    }

    #[tokio::test]
    async fn test_event_collector_propagates_error() {
        let graph = build_error_graph();
        let stream = stream_graph_events(&graph, json!({}), None);
        let result = GraphEventCollector::collect_all(stream).await;

        // Should propagate the error from the stream.
        assert!(result.is_err());
    }

    // ------------------------------------------------------------------
    // Test 7: Multi-node graph has correct event ordering
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn test_multi_node_event_ordering() {
        // Build a 3-node linear graph: A -> B -> C -> END.
        let graph = StateGraph::new()
            .add_node(
                "A",
                Arc::new(|state: Value| {
                    Box::pin(async move {
                        let mut s = state;
                        if let Some(obj) = s.as_object_mut() {
                            obj.insert("step".into(), json!(1));
                        }
                        Ok(s)
                    })
                }),
            )
            .add_node(
                "B",
                Arc::new(|state: Value| {
                    Box::pin(async move {
                        let mut s = state;
                        if let Some(obj) = s.as_object_mut() {
                            obj.insert("step".into(), json!(2));
                        }
                        Ok(s)
                    })
                }),
            )
            .add_node(
                "C",
                Arc::new(|state: Value| {
                    Box::pin(async move {
                        let mut s = state;
                        if let Some(obj) = s.as_object_mut() {
                            obj.insert("step".into(), json!(3));
                        }
                        Ok(s)
                    })
                }),
            )
            .add_edge("__start__", "A")
            .add_edge("A", "B")
            .add_edge("B", "C")
            .add_edge("C", "__end__")
            .compile()
            .unwrap();
        let stream = stream_graph_events(&graph, json!({}), None);
        let events = GraphEventCollector::collect_all(stream).await.unwrap();

        // Expected pattern: GraphStart, then for each of A, B, C:
        // (NodeStart, NodeEnd, StateUpdate), then GraphEnd.
        // Total = 1 + 3*3 + 1 = 11 events.
        assert_eq!(events.len(), 11);

        // Verify the repeating pattern.
        let expected_types = vec![
            "graph_start",
            "node_start",
            "node_end",
            "state_update",
            "node_start",
            "node_end",
            "state_update",
            "node_start",
            "node_end",
            "state_update",
            "graph_end",
        ];

        let actual_types: Vec<&str> = events
            .iter()
            .map(|e| match e {
                GraphStreamEvent::GraphStart { .. } => "graph_start",
                GraphStreamEvent::GraphEnd { .. } => "graph_end",
                GraphStreamEvent::NodeStart { .. } => "node_start",
                GraphStreamEvent::NodeEnd { .. } => "node_end",
                GraphStreamEvent::NodeError { .. } => "node_error",
                GraphStreamEvent::StateUpdate { .. } => "state_update",
                GraphStreamEvent::Custom { .. } => "custom",
            })
            .collect();

        assert_eq!(actual_types, expected_types);
    }

    // ------------------------------------------------------------------
    // Test 8: State updates reflect actual state changes
    // ------------------------------------------------------------------
    #[tokio::test]
    async fn test_state_updates_reflect_changes() {
        let graph = build_linear_graph();
        let stream = stream_graph_events(&graph, json!({}), None);
        let events = GraphEventCollector::collect_all(stream).await.unwrap();

        let state_updates: Vec<&Value> = events
            .iter()
            .filter_map(|e| match e {
                GraphStreamEvent::StateUpdate { state } => Some(state),
                _ => None,
            })
            .collect();

        // After node A: state should have "a" key.
        assert_eq!(state_updates.len(), 2);
        assert!(
            state_updates[0].get("a").is_some(),
            "state after A should contain 'a'"
        );

        // After node B: state should have both "a" and "b" keys.
        assert!(
            state_updates[1].get("a").is_some(),
            "state after B should still contain 'a'"
        );
        assert!(
            state_updates[1].get("b").is_some(),
            "state after B should contain 'b'"
        );
    }
}
