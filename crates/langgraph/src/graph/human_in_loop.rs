//! Human-in-the-loop approval patterns for graph execution.
//!
//! This module provides a high-level API for building workflows that require
//! human approval at specific nodes. It wraps a [`CompiledStateGraph`] and a
//! [`CheckpointSaver`] to pause execution, present an [`ApprovalRequest`] to
//! a human, and then resume based on the [`HumanAction`] taken.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

use crate::errors::LangGraphError;
use crate::pregel::checkpoint::{CheckpointMetadata, CheckpointSaver};
use crate::types::{InterruptedState, InvokeResult};

use super::state::CompiledStateGraph;

/// An action a human can take in response to an approval request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HumanAction {
    /// Approve — continue execution as-is.
    Approve,
    /// Reject — abort execution with a reason.
    Reject { reason: String },
    /// Edit — modify the state before continuing.
    Edit { modifications: Value },
    /// Feedback — add a feedback message to state and continue.
    Feedback { message: String },
}

/// A request for human approval at an interrupt point.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApprovalRequest {
    /// Which node is requesting approval.
    pub node_name: String,
    /// The state at the interrupt point.
    pub current_state: Value,
    /// Execution thread identifier.
    pub thread_id: String,
    /// Checkpoint to resume from.
    pub checkpoint_id: String,
    /// The interrupted state metadata needed to resume execution.
    #[serde(skip)]
    pub(crate) interrupted_state: Option<InterruptedState>,
    /// The interrupt nodes configured for this execution.
    #[serde(skip)]
    pub(crate) interrupt_before_nodes: Vec<String>,
}

/// Result of a human-in-the-loop execution step.
#[derive(Debug, Clone)]
pub enum HumanInTheLoopResult {
    /// Execution completed successfully.
    Complete(Value),
    /// Execution was interrupted and awaits human approval.
    PendingApproval(ApprovalRequest),
    /// Execution was rejected by a human.
    Rejected { reason: String, state: Value },
}

/// High-level wrapper for human-in-the-loop approval workflows.
///
/// Wraps a [`CompiledStateGraph`] and a [`CheckpointSaver`] to support
/// pause-approve-resume patterns at configurable interrupt points.
///
/// # Example
///
/// ```rust,no_run
/// use std::sync::Arc;
/// use serde_json::{json, Value};
/// use langgraph::graph::state::{StateGraph, AsyncNodeAction};
/// use langgraph::graph::human_in_loop::{HumanInTheLoop, HumanAction};
/// use langgraph::pregel::checkpoint::InMemoryCheckpointSaver;
///
/// # async fn example() -> Result<(), langgraph::errors::LangGraphError> {
/// let action: AsyncNodeAction = Arc::new(|state: Value| {
///     Box::pin(async move { Ok(json!({"processed": true})) })
/// });
///
/// let graph = StateGraph::new()
///     .add_node("step", action)
///     .set_entry_point("step")
///     .set_finish_point("step")
///     .compile()?;
///
/// let saver = Arc::new(InMemoryCheckpointSaver::new());
/// let hitl = HumanInTheLoop::new(graph, saver);
///
/// let result = hitl.execute_with_approval(
///     json!({"input": "data"}),
///     vec!["step".to_string()],
/// ).await?;
/// # Ok(())
/// # }
/// ```
pub struct HumanInTheLoop {
    /// The compiled state graph to execute.
    graph: CompiledStateGraph,
    /// Checkpoint saver for persisting state at interrupt points.
    saver: Arc<dyn CheckpointSaver>,
}

impl HumanInTheLoop {
    /// Create a new `HumanInTheLoop` wrapper.
    pub fn new(graph: CompiledStateGraph, saver: Arc<dyn CheckpointSaver>) -> Self {
        Self { graph, saver }
    }

    /// Execute the graph, pausing at the specified interrupt nodes for approval.
    ///
    /// If an interrupt is hit, returns [`HumanInTheLoopResult::PendingApproval`]
    /// with an [`ApprovalRequest`] containing the current state and checkpoint
    /// information. If no interrupt is triggered, returns
    /// [`HumanInTheLoopResult::Complete`].
    pub async fn execute_with_approval(
        &self,
        input: Value,
        interrupt_before_nodes: Vec<String>,
    ) -> Result<HumanInTheLoopResult, LangGraphError> {
        // Build a graph with the requested interrupt points.
        let graph = self.graph_with_interrupts(&interrupt_before_nodes);

        let thread_id = Uuid::now_v7().to_string();

        // Save initial checkpoint.
        let config = self.make_config(&thread_id, None);
        let initial_checkpoint = crate::pregel::checkpoint::empty_checkpoint();
        let metadata = CheckpointMetadata {
            source: "input".to_string(),
            step: 0,
            writes: None,
            extra: HashMap::new(),
        };
        self.saver
            .put(&config, initial_checkpoint, metadata)
            .await
            .map_err(|e| LangGraphError::Other(format!("Failed to save checkpoint: {e}")))?;

        // Run graph with interrupt support.
        let result = graph.invoke_with_interrupt(input).await?;

        match result {
            InvokeResult::Complete(state) => {
                // Save final checkpoint.
                self.save_checkpoint(&thread_id, &state, "complete", 1)
                    .await?;
                Ok(HumanInTheLoopResult::Complete(state))
            }
            InvokeResult::Interrupted(interrupted) => {
                // Save the interrupted state as a checkpoint.
                let cp_id = self
                    .save_checkpoint(&thread_id, &interrupted.state, "interrupt", 1)
                    .await?;

                Ok(HumanInTheLoopResult::PendingApproval(ApprovalRequest {
                    node_name: interrupted.interrupted_at.clone(),
                    current_state: interrupted.state.clone(),
                    thread_id,
                    checkpoint_id: cp_id,
                    interrupted_state: Some(interrupted),
                    interrupt_before_nodes,
                }))
            }
        }
    }

    /// Respond to an approval request with a [`HumanAction`].
    ///
    /// - [`HumanAction::Approve`]: resumes execution from the checkpoint.
    /// - [`HumanAction::Reject`]: returns a rejection result immediately.
    /// - [`HumanAction::Edit`]: merges modifications into state, then resumes.
    /// - [`HumanAction::Feedback`]: adds a feedback message to state, then resumes.
    pub async fn respond(
        &self,
        request: ApprovalRequest,
        action: HumanAction,
    ) -> Result<HumanInTheLoopResult, LangGraphError> {
        let interrupted = request.interrupted_state.ok_or_else(|| {
            LangGraphError::Other("ApprovalRequest missing interrupted state metadata".to_string())
        })?;

        let graph = self.graph_with_interrupts(&request.interrupt_before_nodes);

        match action {
            HumanAction::Approve => {
                let result = graph.resume(interrupted, None).await?;
                self.handle_invoke_result(
                    result,
                    &request.thread_id,
                    &request.interrupt_before_nodes,
                )
                .await
            }
            HumanAction::Reject { reason } => Ok(HumanInTheLoopResult::Rejected {
                reason,
                state: request.current_state,
            }),
            HumanAction::Edit { modifications } => {
                let result = graph.resume(interrupted, Some(modifications)).await?;
                self.handle_invoke_result(
                    result,
                    &request.thread_id,
                    &request.interrupt_before_nodes,
                )
                .await
            }
            HumanAction::Feedback { message } => {
                let feedback_update = serde_json::json!({ "feedback": message });
                let result = graph.resume(interrupted, Some(feedback_update)).await?;
                self.handle_invoke_result(
                    result,
                    &request.thread_id,
                    &request.interrupt_before_nodes,
                )
                .await
            }
        }
    }

    /// Execute the graph to completion, using a closure to decide the action
    /// at each approval point.
    ///
    /// The `action_fn` is called with each [`ApprovalRequest`] and must return
    /// a [`HumanAction`]. The loop continues until the graph completes or is
    /// rejected.
    pub async fn execute_to_completion<F>(
        &self,
        input: Value,
        interrupt_nodes: Vec<String>,
        action_fn: F,
    ) -> Result<HumanInTheLoopResult, LangGraphError>
    where
        F: Fn(&ApprovalRequest) -> HumanAction,
    {
        let mut result = self
            .execute_with_approval(input, interrupt_nodes.clone())
            .await?;

        loop {
            match result {
                HumanInTheLoopResult::Complete(_) | HumanInTheLoopResult::Rejected { .. } => {
                    return Ok(result);
                }
                HumanInTheLoopResult::PendingApproval(request) => {
                    let action = action_fn(&request);
                    result = self.respond(request, action).await?;
                }
            }
        }
    }

    // --- Private helpers ---

    /// Create a copy of the graph with the given interrupt_before nodes set.
    fn graph_with_interrupts(&self, interrupt_before_nodes: &[String]) -> CompiledStateGraph {
        let mut graph = self.graph.clone();
        for node in interrupt_before_nodes {
            graph.interrupt_before.insert(node.clone());
        }
        graph
    }

    /// Build a config map for the checkpoint saver.
    fn make_config(&self, thread_id: &str, checkpoint_id: Option<&str>) -> HashMap<String, Value> {
        let mut config = HashMap::new();
        config.insert(
            "thread_id".to_string(),
            Value::String(thread_id.to_string()),
        );
        if let Some(cp_id) = checkpoint_id {
            config.insert(
                "checkpoint_id".to_string(),
                Value::String(cp_id.to_string()),
            );
        }
        config
    }

    /// Save a checkpoint and return its ID.
    async fn save_checkpoint(
        &self,
        thread_id: &str,
        state: &Value,
        source: &str,
        step: i64,
    ) -> Result<String, LangGraphError> {
        let mut checkpoint = crate::pregel::checkpoint::empty_checkpoint();
        if let Value::Object(map) = state {
            for (k, v) in map {
                checkpoint.channel_values.insert(k.clone(), v.clone());
            }
        }
        let cp_id = checkpoint.id.clone();

        let config = self.make_config(thread_id, None);
        let metadata = CheckpointMetadata {
            source: source.to_string(),
            step,
            writes: None,
            extra: HashMap::new(),
        };
        self.saver
            .put(&config, checkpoint, metadata)
            .await
            .map_err(|e| LangGraphError::Other(format!("Failed to save checkpoint: {e}")))?;

        Ok(cp_id)
    }

    /// Convert an InvokeResult into a HumanInTheLoopResult, saving checkpoints
    /// and creating new approval requests as needed.
    async fn handle_invoke_result(
        &self,
        result: InvokeResult,
        thread_id: &str,
        interrupt_before_nodes: &[String],
    ) -> Result<HumanInTheLoopResult, LangGraphError> {
        match result {
            InvokeResult::Complete(state) => {
                self.save_checkpoint(thread_id, &state, "complete", 2)
                    .await?;
                Ok(HumanInTheLoopResult::Complete(state))
            }
            InvokeResult::Interrupted(interrupted) => {
                let cp_id = self
                    .save_checkpoint(thread_id, &interrupted.state, "interrupt", 2)
                    .await?;

                Ok(HumanInTheLoopResult::PendingApproval(ApprovalRequest {
                    node_name: interrupted.interrupted_at.clone(),
                    current_state: interrupted.state.clone(),
                    thread_id: thread_id.to_string(),
                    checkpoint_id: cp_id,
                    interrupted_state: Some(interrupted),
                    interrupt_before_nodes: interrupt_before_nodes.to_vec(),
                }))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::state::{AsyncNodeAction, StateGraph};
    use crate::pregel::checkpoint::InMemoryCheckpointSaver;
    use serde_json::json;

    /// Helper: build a simple two-node graph: step1 -> step2
    fn build_two_node_graph() -> CompiledStateGraph {
        let step1: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "count": count + 1, "step1": true }))
            })
        });

        let step2: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "count": count + 10, "step2": true }))
            })
        });

        StateGraph::new()
            .add_node("step1", step1)
            .add_node("step2", step2)
            .set_entry_point("step1")
            .add_edge("step1", "step2")
            .set_finish_point("step2")
            .compile()
            .unwrap()
    }

    /// Helper: build a three-node graph: a -> b -> c
    fn build_three_node_graph() -> CompiledStateGraph {
        let node_a: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "count": count + 1, "a_ran": true }))
            })
        });

        let node_b: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "count": count + 10, "b_ran": true }))
            })
        });

        let node_c: AsyncNodeAction = Arc::new(|state: Value| {
            Box::pin(async move {
                let count = state.get("count").and_then(|v| v.as_i64()).unwrap_or(0);
                Ok(json!({ "count": count + 100, "c_ran": true }))
            })
        });

        StateGraph::new()
            .add_node("a", node_a)
            .add_node("b", node_b)
            .add_node("c", node_c)
            .set_entry_point("a")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .set_finish_point("c")
            .compile()
            .unwrap()
    }

    fn make_saver() -> Arc<dyn CheckpointSaver> {
        Arc::new(InMemoryCheckpointSaver::new())
    }

    // ---- Test 1: Execute with approval at a node, approve, complete ----
    #[tokio::test]
    async fn test_approve_and_complete() {
        let graph = build_two_node_graph();
        let saver = make_saver();
        let hitl = HumanInTheLoop::new(graph, saver);

        let result = hitl
            .execute_with_approval(json!({"count": 0}), vec!["step2".to_string()])
            .await
            .unwrap();

        // Should be pending approval at step2.
        let request = match result {
            HumanInTheLoopResult::PendingApproval(req) => req,
            other => panic!("Expected PendingApproval, got {:?}", other),
        };
        assert_eq!(request.node_name, "step2");

        // Approve and continue.
        let final_result = hitl.respond(request, HumanAction::Approve).await.unwrap();

        match final_result {
            HumanInTheLoopResult::Complete(state) => {
                // step1 adds 1, step2 adds 10 => count = 11
                assert_eq!(state.get("count").unwrap(), 11);
                assert_eq!(state.get("step1").unwrap(), true);
                assert_eq!(state.get("step2").unwrap(), true);
            }
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    // ---- Test 2: Execute with approval, reject, get rejection ----
    #[tokio::test]
    async fn test_reject_execution() {
        let graph = build_two_node_graph();
        let saver = make_saver();
        let hitl = HumanInTheLoop::new(graph, saver);

        let result = hitl
            .execute_with_approval(json!({"count": 0}), vec!["step2".to_string()])
            .await
            .unwrap();

        let request = match result {
            HumanInTheLoopResult::PendingApproval(req) => req,
            other => panic!("Expected PendingApproval, got {:?}", other),
        };

        let final_result = hitl
            .respond(
                request,
                HumanAction::Reject {
                    reason: "Not safe to proceed".to_string(),
                },
            )
            .await
            .unwrap();

        match final_result {
            HumanInTheLoopResult::Rejected { reason, state } => {
                assert_eq!(reason, "Not safe to proceed");
                // State should be the state at the interrupt point (step1 ran).
                assert_eq!(state.get("count").unwrap(), 1);
            }
            other => panic!("Expected Rejected, got {:?}", other),
        }
    }

    // ---- Test 3: Execute with edit, modified state flows through ----
    #[tokio::test]
    async fn test_edit_state_before_continuing() {
        let graph = build_two_node_graph();
        let saver = make_saver();
        let hitl = HumanInTheLoop::new(graph, saver);

        let result = hitl
            .execute_with_approval(json!({"count": 0}), vec!["step2".to_string()])
            .await
            .unwrap();

        let request = match result {
            HumanInTheLoopResult::PendingApproval(req) => req,
            other => panic!("Expected PendingApproval, got {:?}", other),
        };

        // Edit: override count to 100 before step2 runs.
        let final_result = hitl
            .respond(
                request,
                HumanAction::Edit {
                    modifications: json!({"count": 100}),
                },
            )
            .await
            .unwrap();

        match final_result {
            HumanInTheLoopResult::Complete(state) => {
                // step2 adds 10 to the edited count of 100 => 110
                assert_eq!(state.get("count").unwrap(), 110);
            }
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    // ---- Test 4: Execute with feedback ----
    #[tokio::test]
    async fn test_feedback_added_to_state() {
        let graph = build_two_node_graph();
        let saver = make_saver();
        let hitl = HumanInTheLoop::new(graph, saver);

        let result = hitl
            .execute_with_approval(json!({"count": 0}), vec!["step2".to_string()])
            .await
            .unwrap();

        let request = match result {
            HumanInTheLoopResult::PendingApproval(req) => req,
            other => panic!("Expected PendingApproval, got {:?}", other),
        };

        let final_result = hitl
            .respond(
                request,
                HumanAction::Feedback {
                    message: "Looks good, proceed carefully".to_string(),
                },
            )
            .await
            .unwrap();

        match final_result {
            HumanInTheLoopResult::Complete(state) => {
                // Feedback message should be in the state.
                assert_eq!(
                    state.get("feedback").unwrap(),
                    "Looks good, proceed carefully"
                );
                // step2 should also have run.
                assert_eq!(state.get("step2").unwrap(), true);
            }
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    // ---- Test 5: Multiple approval points in sequence ----
    #[tokio::test]
    async fn test_multiple_approval_points() {
        let graph = build_three_node_graph();
        let saver = make_saver();
        let hitl = HumanInTheLoop::new(graph, saver);

        // Interrupt before both b and c.
        let result = hitl
            .execute_with_approval(json!({"count": 0}), vec!["b".to_string(), "c".to_string()])
            .await
            .unwrap();

        // First interrupt should be at b (after a ran).
        let request1 = match result {
            HumanInTheLoopResult::PendingApproval(req) => req,
            other => panic!("Expected PendingApproval at b, got {:?}", other),
        };
        assert_eq!(request1.node_name, "b");
        assert_eq!(request1.current_state.get("count").unwrap(), 1);
        assert_eq!(request1.current_state.get("a_ran").unwrap(), true);

        // Approve b.
        let result2 = hitl.respond(request1, HumanAction::Approve).await.unwrap();

        // Second interrupt should be at c (after b ran).
        let request2 = match result2 {
            HumanInTheLoopResult::PendingApproval(req) => req,
            other => panic!("Expected PendingApproval at c, got {:?}", other),
        };
        assert_eq!(request2.node_name, "c");
        assert_eq!(request2.current_state.get("count").unwrap(), 11);
        assert_eq!(request2.current_state.get("b_ran").unwrap(), true);

        // Approve c.
        let final_result = hitl.respond(request2, HumanAction::Approve).await.unwrap();

        match final_result {
            HumanInTheLoopResult::Complete(state) => {
                assert_eq!(state.get("count").unwrap(), 111);
                assert_eq!(state.get("c_ran").unwrap(), true);
            }
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    // ---- Test 6: execute_to_completion with auto-approve ----
    #[tokio::test]
    async fn test_execute_to_completion_auto_approve() {
        let graph = build_three_node_graph();
        let saver = make_saver();
        let hitl = HumanInTheLoop::new(graph, saver);

        let result = hitl
            .execute_to_completion(
                json!({"count": 0}),
                vec!["b".to_string(), "c".to_string()],
                |_request| HumanAction::Approve,
            )
            .await
            .unwrap();

        match result {
            HumanInTheLoopResult::Complete(state) => {
                // a: +1, b: +10, c: +100 => 111
                assert_eq!(state.get("count").unwrap(), 111);
            }
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    // ---- Test 7: Approval request contains correct state ----
    #[tokio::test]
    async fn test_approval_request_contains_correct_state() {
        let graph = build_two_node_graph();
        let saver = make_saver();
        let hitl = HumanInTheLoop::new(graph, saver);

        let result = hitl
            .execute_with_approval(
                json!({"count": 5, "extra": "data"}),
                vec!["step2".to_string()],
            )
            .await
            .unwrap();

        let request = match result {
            HumanInTheLoopResult::PendingApproval(req) => req,
            other => panic!("Expected PendingApproval, got {:?}", other),
        };

        // After step1: count = 5 + 1 = 6, step1 = true, extra should still be there.
        assert_eq!(request.node_name, "step2");
        assert_eq!(request.current_state.get("count").unwrap(), 6);
        assert_eq!(request.current_state.get("step1").unwrap(), true);
        // The merge overwrites at key level, but "extra" was in the original
        // state and step1 doesn't touch it — however, merge_state replaces the
        // whole object when the node returns a new object. Let's verify what
        // actually happens: step1 returns {"count":6,"step1":true} which merges
        // into the state, preserving "extra".
        assert_eq!(request.current_state.get("extra").unwrap(), "data");

        // Thread and checkpoint IDs should be populated.
        assert!(!request.thread_id.is_empty());
        assert!(!request.checkpoint_id.is_empty());
    }

    // ---- Test 8: No interrupt nodes means normal execution ----
    #[tokio::test]
    async fn test_no_interrupt_nodes_normal_execution() {
        let graph = build_two_node_graph();
        let saver = make_saver();
        let hitl = HumanInTheLoop::new(graph, saver);

        // No interrupt nodes — should run to completion.
        let result = hitl
            .execute_with_approval(json!({"count": 0}), vec![])
            .await
            .unwrap();

        match result {
            HumanInTheLoopResult::Complete(state) => {
                assert_eq!(state.get("count").unwrap(), 11);
                assert_eq!(state.get("step1").unwrap(), true);
                assert_eq!(state.get("step2").unwrap(), true);
            }
            other => panic!("Expected Complete, got {:?}", other),
        }
    }

    // ---- Test 9: execute_to_completion with reject stops early ----
    #[tokio::test]
    async fn test_execute_to_completion_with_reject() {
        let graph = build_three_node_graph();
        let saver = make_saver();
        let hitl = HumanInTheLoop::new(graph, saver);

        let result = hitl
            .execute_to_completion(
                json!({"count": 0}),
                vec!["b".to_string(), "c".to_string()],
                |request| {
                    if request.node_name == "c" {
                        HumanAction::Reject {
                            reason: "Stopped at c".to_string(),
                        }
                    } else {
                        HumanAction::Approve
                    }
                },
            )
            .await
            .unwrap();

        match result {
            HumanInTheLoopResult::Rejected { reason, state } => {
                assert_eq!(reason, "Stopped at c");
                // a ran (+1), b ran (+10), c did not run
                assert_eq!(state.get("count").unwrap(), 11);
                assert!(state.get("c_ran").is_none());
            }
            other => panic!("Expected Rejected, got {:?}", other),
        }
    }
}
