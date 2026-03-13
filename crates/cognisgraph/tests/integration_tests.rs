//! Comprehensive integration tests for the LangGraph state-graph framework.
//!
//! These tests exercise the full public API surface: linear graphs, conditional
//! branching, cycles, interrupts (before/after), streaming, Mermaid diagram
//! generation, tool agents, subgraph composition, fan-out via Send, and
//! stress-testing with many nodes.

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
use serde_json::{json, Value};

use cognisgraph::constants::{END, START};
use cognisgraph::errors::LangGraphError;
use cognisgraph::graph::branch::RouterResult;
use cognisgraph::graph::state::{AsyncNodeAction, CompiledStateGraph, StateGraph};
use cognisgraph::types::{InterruptType, InvokeResult, Send as GraphSend, StreamMode};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create an async node action that sets a key to a value in state.
fn set_key(key: &str, value: Value) -> AsyncNodeAction {
    let key = key.to_string();
    Arc::new(move |_state: Value| {
        let key = key.clone();
        let value = value.clone();
        Box::pin(async move { Ok(json!({ key: value })) })
    })
}

/// Create an async node action that transforms state via a closure.
fn transform<F>(f: F) -> AsyncNodeAction
where
    F: Fn(Value) -> Result<Value, LangGraphError> + Send + Sync + 'static,
{
    Arc::new(move |state: Value| {
        let result = f(state);
        Box::pin(async move { result })
    })
}

// ---------------------------------------------------------------------------
// 1. Linear graph execution
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_linear_graph_execution() {
    // A -> B -> C, each appending a marker to state.
    let graph = StateGraph::new()
        .add_node(
            "a",
            transform(|state| {
                let mut trail: Vec<String> = state
                    .get("trail")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                trail.push("a".into());
                Ok(json!({ "trail": trail, "a_ran": true }))
            }),
        )
        .add_node(
            "b",
            transform(|state| {
                let mut trail: Vec<String> = state
                    .get("trail")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                trail.push("b".into());
                Ok(json!({ "trail": trail, "b_ran": true }))
            }),
        )
        .add_node(
            "c",
            transform(|state| {
                let mut trail: Vec<String> = state
                    .get("trail")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                trail.push("c".into());
                Ok(json!({ "trail": trail, "c_ran": true }))
            }),
        )
        .set_entry_point("a")
        .add_edge("a", "b")
        .add_edge("b", "c")
        .set_finish_point("c")
        .compile()
        .unwrap();

    let result = graph.invoke(json!({ "trail": [] })).await.unwrap();

    assert_eq!(result["a_ran"], json!(true));
    assert_eq!(result["b_ran"], json!(true));
    assert_eq!(result["c_ran"], json!(true));
    assert_eq!(result["trail"], json!(["a", "b", "c"]));
}

// ---------------------------------------------------------------------------
// 2. Conditional branching
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_conditional_branching() {
    let graph = StateGraph::new()
        .add_node("agent", set_key("agent_ran", json!(true)))
        .add_node("positive", set_key("sentiment", json!("positive")))
        .add_node("negative", set_key("sentiment", json!("negative")))
        .set_entry_point("agent")
        .add_conditional_edges(
            "agent",
            Arc::new(|state: &Value| {
                let score = state.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                if score >= 0.0 {
                    RouterResult::Single("positive".into())
                } else {
                    RouterResult::Single("negative".into())
                }
            }),
            None,
        )
        .set_finish_point("positive")
        .set_finish_point("negative")
        .compile()
        .unwrap();

    // Positive path
    let result = graph.invoke(json!({ "score": 0.8 })).await.unwrap();
    assert_eq!(result["sentiment"], json!("positive"));
    assert_eq!(result["agent_ran"], json!(true));

    // Negative path
    let result = graph.invoke(json!({ "score": -0.5 })).await.unwrap();
    assert_eq!(result["sentiment"], json!("negative"));
}

// ---------------------------------------------------------------------------
// 3. Graph with cycle
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_graph_with_cycle() {
    let graph = StateGraph::new()
        .add_node(
            "increment",
            transform(|state| {
                let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "count": count + 1 }))
            }),
        )
        .add_node("done", set_key("finished", json!(true)))
        .set_entry_point("increment")
        .add_conditional_edges(
            "increment",
            Arc::new(|state: &Value| {
                let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                if count >= 5 {
                    RouterResult::Single("done".into())
                } else {
                    RouterResult::Single("increment".into())
                }
            }),
            None,
        )
        .set_finish_point("done")
        .compile()
        .unwrap();

    let result = graph.invoke(json!({ "count": 0 })).await.unwrap();
    assert_eq!(result["count"], json!(5));
    assert_eq!(result["finished"], json!(true));
}

// ---------------------------------------------------------------------------
// 4. Interrupt before and resume
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_interrupt_before_and_resume() {
    let graph = StateGraph::new()
        .add_node("pre", set_key("pre_ran", json!(true)))
        .add_node("review", set_key("review_ran", json!(true)))
        .add_node("post", set_key("post_ran", json!(true)))
        .set_entry_point("pre")
        .add_edge("pre", "review")
        .add_edge("review", "post")
        .set_finish_point("post")
        .interrupt_before(vec!["review"])
        .compile()
        .unwrap();

    // First invocation should interrupt before "review".
    let result = graph
        .invoke_with_interrupt(json!({ "input": "hello" }))
        .await
        .unwrap();

    match result {
        InvokeResult::Interrupted(interrupted) => {
            assert_eq!(interrupted.interrupted_at, "review");
            assert_eq!(interrupted.interrupt_type, InterruptType::Before);
            // "pre" should have run.
            assert_eq!(interrupted.state["pre_ran"], json!(true));
            // "review" should NOT have run yet.
            assert!(interrupted.state.get("review_ran").is_none());

            // Resume with a human-provided update.
            let resumed = graph
                .resume(interrupted, Some(json!({ "human_approved": true })))
                .await
                .unwrap();

            match resumed {
                InvokeResult::Complete(state) => {
                    assert_eq!(state["review_ran"], json!(true));
                    assert_eq!(state["post_ran"], json!(true));
                    assert_eq!(state["human_approved"], json!(true));
                }
                other => panic!("Expected Complete after resume, got: {:?}", other),
            }
        }
        other => panic!("Expected Interrupted, got: {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 5. Interrupt after and resume
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_interrupt_after_and_resume() {
    let graph = StateGraph::new()
        .add_node("step_one", set_key("step_one_ran", json!(true)))
        .add_node("step_two", set_key("step_two_ran", json!(true)))
        .set_entry_point("step_one")
        .add_edge("step_one", "step_two")
        .set_finish_point("step_two")
        .interrupt_after(vec!["step_one"])
        .compile()
        .unwrap();

    let result = graph.invoke_with_interrupt(json!({})).await.unwrap();

    match result {
        InvokeResult::Interrupted(interrupted) => {
            assert_eq!(interrupted.interrupted_at, "step_one");
            assert_eq!(interrupted.interrupt_type, InterruptType::After);
            // "step_one" should have run.
            assert_eq!(interrupted.state["step_one_ran"], json!(true));
            // "step_two" should not have run yet.
            assert!(interrupted.state.get("step_two_ran").is_none());

            // Resume execution.
            let resumed = graph.resume(interrupted, None).await.unwrap();

            match resumed {
                InvokeResult::Complete(state) => {
                    assert_eq!(state["step_one_ran"], json!(true));
                    assert_eq!(state["step_two_ran"], json!(true));
                }
                other => panic!("Expected Complete after resume, got: {:?}", other),
            }
        }
        other => panic!("Expected Interrupted, got: {:?}", other),
    }
}

// ---------------------------------------------------------------------------
// 6. Streaming — Values mode
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_streaming_values_mode() {
    let graph = StateGraph::new()
        .add_node("a", set_key("a", json!(1)))
        .add_node("b", set_key("b", json!(2)))
        .set_entry_point("a")
        .add_edge("a", "b")
        .set_finish_point("b")
        .compile()
        .unwrap();

    let stream = graph
        .stream(json!({ "init": true }), StreamMode::Values)
        .await
        .unwrap();

    let updates: Vec<_> = stream.collect().await;

    assert_eq!(updates.len(), 2);

    // First update: after node "a", full state should include "a" key.
    let first = updates[0].as_ref().unwrap();
    assert_eq!(first.node, "a");
    assert_eq!(first.mode, StreamMode::Values);
    assert_eq!(first.data["a"], json!(1));
    assert_eq!(first.data["init"], json!(true));

    // Second update: after node "b", full state should include both.
    let second = updates[1].as_ref().unwrap();
    assert_eq!(second.node, "b");
    assert_eq!(second.data["a"], json!(1));
    assert_eq!(second.data["b"], json!(2));
}

// ---------------------------------------------------------------------------
// 7. Streaming — Updates mode
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_streaming_updates_mode() {
    let graph = StateGraph::new()
        .add_node("x", set_key("x", json!("hello")))
        .add_node("y", set_key("y", json!("world")))
        .set_entry_point("x")
        .add_edge("x", "y")
        .set_finish_point("y")
        .compile()
        .unwrap();

    let stream = graph.stream(json!({}), StreamMode::Updates).await.unwrap();

    let updates: Vec<_> = stream.collect().await;

    assert_eq!(updates.len(), 2);

    // Updates mode yields only the delta from each node.
    let first = updates[0].as_ref().unwrap();
    assert_eq!(first.node, "x");
    assert_eq!(first.mode, StreamMode::Updates);
    assert_eq!(first.data, json!({ "x": "hello" }));

    let second = updates[1].as_ref().unwrap();
    assert_eq!(second.node, "y");
    assert_eq!(second.data, json!({ "y": "world" }));
}

// ---------------------------------------------------------------------------
// 8. Mermaid diagram generation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_mermaid_diagram_generation() {
    let graph = StateGraph::new()
        .add_node("alpha", set_key("a", json!(1)))
        .add_node("beta", set_key("b", json!(2)))
        .add_node("gamma", set_key("c", json!(3)))
        .set_entry_point("alpha")
        .add_edge("alpha", "beta")
        .add_edge("beta", "gamma")
        .set_finish_point("gamma")
        .interrupt_before(vec!["beta"])
        .compile()
        .unwrap();

    let mermaid = graph.draw_mermaid();

    // Check structural requirements.
    assert!(mermaid.starts_with("graph TD"));
    assert!(mermaid.contains("alpha"));
    assert!(mermaid.contains("beta"));
    assert!(mermaid.contains("gamma"));
    assert!(mermaid.contains(START));
    assert!(mermaid.contains(END));
    // Direct edges present.
    assert!(mermaid.contains("alpha --> beta"));
    assert!(mermaid.contains("beta --> gamma"));
    // Interrupt node styling.
    assert!(mermaid.contains("style beta"));
}

// ---------------------------------------------------------------------------
// 9. Tool agent end-to-end
// ---------------------------------------------------------------------------

mod tool_agent_helpers {
    use super::*;
    use async_trait::async_trait;
    use cognis_core::language_models::fake::FakeMessagesListChatModel;
    use cognis_core::messages::{AIMessage, Message, ToolCall};
    use cognis_core::tools::types::{ToolInput, ToolOutput};
    use cognis_core::tools::BaseTool;

    pub struct MockTool {
        pub tool_name: String,
        pub result: String,
    }

    #[async_trait]
    impl BaseTool for MockTool {
        fn name(&self) -> &str {
            &self.tool_name
        }

        fn description(&self) -> &str {
            "mock tool for testing"
        }

        async fn _run(&self, _input: ToolInput) -> cognis_core::error::Result<ToolOutput> {
            Ok(ToolOutput::Content(Value::String(self.result.clone())))
        }
    }

    pub fn make_tool_agent_test() -> CompiledStateGraph {
        // Turn 1: model returns a tool call.
        // Turn 2: model returns a final answer.
        let tc = ToolCall {
            name: "lookup".into(),
            args: {
                let mut m = HashMap::new();
                m.insert("query".into(), json!("test"));
                m
            },
            id: Some("call_1".into()),
        };
        let mut ai_with_tc = AIMessage::new("");
        ai_with_tc.tool_calls = vec![tc];

        let model = Arc::new(FakeMessagesListChatModel::new(vec![
            Message::Ai(ai_with_tc),
            Message::Ai(AIMessage::new("Final answer: 42")),
        ]));

        let tool: Arc<dyn BaseTool> = Arc::new(MockTool {
            tool_name: "lookup".into(),
            result: "42".into(),
        });

        cognisgraph::prebuilt::create_tool_agent(model, vec![tool], None).unwrap()
    }
}

#[tokio::test]
async fn test_tool_agent_end_to_end() {
    let graph = tool_agent_helpers::make_tool_agent_test();

    let input = json!({
        "messages": [
            { "type": "human", "content": "What is the answer?" }
        ]
    });

    let result = graph.invoke(input).await.unwrap();
    let messages = result["messages"].as_array().unwrap();

    // human + ai(tool_call) + tool_result + ai(final)
    assert_eq!(messages.len(), 4);

    // Check final message content.
    let last: cognis_core::messages::Message =
        serde_json::from_value(messages.last().unwrap().clone()).unwrap();
    assert_eq!(last.content().text(), "Final answer: 42");
}

// ---------------------------------------------------------------------------
// 10. Subgraph composition
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_subgraph_composition() {
    // Inner graph: doubles a counter.
    let inner = StateGraph::new()
        .add_node(
            "double",
            transform(|state| {
                let val = state.get("value").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "value": val * 2 }))
            }),
        )
        .set_entry_point("double")
        .set_finish_point("double")
        .compile()
        .unwrap();

    // Outer graph: sets initial value, calls subgraph, then adds a marker.
    let outer = StateGraph::new()
        .add_node("init", set_key("value", json!(5)))
        .add_subgraph("sub", inner)
        .add_node("finalize", set_key("done", json!(true)))
        .set_entry_point("init")
        .add_edge("init", "sub")
        .add_edge("sub", "finalize")
        .set_finish_point("finalize")
        .compile()
        .unwrap();

    let result = outer.invoke(json!({})).await.unwrap();
    assert_eq!(result["value"], json!(10)); // 5 * 2
    assert_eq!(result["done"], json!(true));
}

// ---------------------------------------------------------------------------
// 11. Parallel fan-out via Send API
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_parallel_fan_out() {
    // A dispatcher node fans out work to a "worker" node via Send instructions.
    // Each Send carries a custom input. All results are merged back.
    let graph = StateGraph::new()
        .add_node("dispatcher", set_key("dispatched", json!(true)))
        .add_node(
            "worker",
            transform(|state| {
                // Each worker invocation receives a custom input with a "task_id".
                let task_id = state.get("task_id").and_then(|v| v.as_i64()).unwrap_or(-1);
                Ok(json!({ "last_task": task_id }))
            }),
        )
        .add_node("collector", set_key("collected", json!(true)))
        .set_entry_point("dispatcher")
        .add_conditional_edges(
            "dispatcher",
            Arc::new(|_state: &Value| {
                RouterResult::Sends(vec![
                    GraphSend {
                        node: "worker".into(),
                        arg: json!({ "task_id": 1 }),
                    },
                    GraphSend {
                        node: "worker".into(),
                        arg: json!({ "task_id": 2 }),
                    },
                    GraphSend {
                        node: "worker".into(),
                        arg: json!({ "task_id": 3 }),
                    },
                ])
            }),
            None,
        )
        .add_edge("worker", "collector")
        .set_finish_point("collector")
        .compile()
        .unwrap();

    let result = graph.invoke(json!({})).await.unwrap();
    assert_eq!(result["dispatched"], json!(true));
    assert_eq!(result["collected"], json!(true));
    // The last worker to execute determines `last_task` (sequential execution).
    assert!(result.get("last_task").is_some());
}

// ---------------------------------------------------------------------------
// 12. Stress test — many-node DAG
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_graph_with_many_nodes() {
    let node_count = 15;
    let mut builder = StateGraph::new();

    // Create nodes: each increments a counter and records its name.
    for i in 0..node_count {
        let name = format!("node_{}", i);
        let name_clone = name.clone();
        builder = builder.add_node(
            &name,
            transform(move |state: Value| {
                let count = state.get("steps").and_then(|v| v.as_i64()).unwrap_or(0);
                let mut visited: Vec<String> = state
                    .get("visited")
                    .and_then(|v| serde_json::from_value(v.clone()).ok())
                    .unwrap_or_default();
                visited.push(name_clone.clone());
                Ok(json!({ "steps": count + 1, "visited": visited }))
            }),
        );
    }

    // Wire them in sequence: node_0 -> node_1 -> ... -> node_14
    builder = builder.set_entry_point("node_0");
    for i in 0..node_count - 1 {
        builder = builder.add_edge(&format!("node_{}", i), &format!("node_{}", i + 1));
    }
    builder = builder.set_finish_point(&format!("node_{}", node_count - 1));

    let graph = builder.compile().unwrap();

    let result = graph
        .invoke(json!({ "steps": 0, "visited": [] }))
        .await
        .unwrap();

    assert_eq!(result["steps"], json!(node_count as i64));
    let visited: Vec<String> = serde_json::from_value(result["visited"].clone()).unwrap();
    assert_eq!(visited.len(), node_count);
    for i in 0..node_count {
        assert_eq!(visited[i], format!("node_{}", i));
    }
}

// ---------------------------------------------------------------------------
// Additional edge-case tests
// ---------------------------------------------------------------------------

#[tokio::test]
async fn test_add_sequence_helper() {
    // Verify the add_sequence convenience method.
    let graph = StateGraph::new()
        .add_sequence(vec![
            ("first", set_key("first", json!(true))),
            ("second", set_key("second", json!(true))),
            ("third", set_key("third", json!(true))),
        ])
        .set_entry_point("first")
        .set_finish_point("third")
        .compile()
        .unwrap();

    let result = graph.invoke(json!({})).await.unwrap();
    assert_eq!(result["first"], json!(true));
    assert_eq!(result["second"], json!(true));
    assert_eq!(result["third"], json!(true));
}

#[tokio::test]
async fn test_recursion_limit_triggers() {
    let graph = StateGraph::new()
        .with_recursion_limit(3)
        .add_node("looper", set_key("x", json!(1)))
        .set_entry_point("looper")
        .add_edge("looper", "looper")
        .compile()
        .unwrap();

    let result = graph.invoke(json!({})).await;
    assert!(result.is_err());
    match result.unwrap_err() {
        LangGraphError::GraphRecursionError(msg) => {
            assert!(msg.contains("Recursion limit of 3"));
        }
        other => panic!("Expected GraphRecursionError, got: {:?}", other),
    }
}

#[tokio::test]
async fn test_streaming_debug_mode() {
    let graph = StateGraph::new()
        .add_node("step", set_key("val", json!(42)))
        .set_entry_point("step")
        .set_finish_point("step")
        .compile()
        .unwrap();

    let stream = graph.stream(json!({}), StreamMode::Debug).await.unwrap();

    let updates: Vec<_> = stream.collect().await;
    assert_eq!(updates.len(), 1);

    let update = updates[0].as_ref().unwrap();
    assert_eq!(update.mode, StreamMode::Debug);
    // Debug mode includes step, elapsed_ms, update, and state.
    assert!(update.data.get("step").is_some());
    assert!(update.data.get("elapsed_ms").is_some());
    assert!(update.data.get("update").is_some());
    assert!(update.data.get("state").is_some());
}

#[tokio::test]
async fn test_node_names_and_introspection() {
    let graph = StateGraph::new()
        .add_node("alpha", set_key("a", json!(1)))
        .add_node("beta", set_key("b", json!(2)))
        .add_node("gamma", set_key("c", json!(3)))
        .set_entry_point("alpha")
        .add_edge("alpha", "beta")
        .add_edge("beta", "gamma")
        .set_finish_point("gamma")
        .compile()
        .unwrap();

    let mut names = graph.node_names();
    names.sort();
    assert_eq!(names, vec!["alpha", "beta", "gamma"]);
    assert_eq!(graph.recursion_limit(), 25); // default
}
