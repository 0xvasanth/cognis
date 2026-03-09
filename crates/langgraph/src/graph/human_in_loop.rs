//! Human-in-the-loop approval patterns for graph execution.
//!
//! This module provides a high-level API for building workflows that require
//! human approval at specific nodes. It wraps a [`CompiledStateGraph`] and a
//! [`CheckpointSaver`] to pause execution, present an [`ApprovalRequest`] to
//! a human, and then resume based on the [`HumanAction`] taken.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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

// ===========================================================================
// Human Approval, Policies, Review Queues, and Feedback Logging
// ===========================================================================

// ---------------------------------------------------------------------------
// HumanApproval
// ---------------------------------------------------------------------------

/// The outcome of a human review decision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HumanApproval {
    /// The proposed action was approved.
    Approved,
    /// The proposed action was rejected with a reason.
    Rejected { reason: String },
    /// The proposed value was modified by the reviewer.
    Modified { new_value: Value },
    /// The review timed out without a decision.
    Timeout,
}

impl HumanApproval {
    /// Returns `true` if the decision is `Approved`.
    pub fn is_approved(&self) -> bool {
        matches!(self, Self::Approved)
    }

    /// Returns `true` if the decision is `Rejected`.
    pub fn is_rejected(&self) -> bool {
        matches!(self, Self::Rejected { .. })
    }

    /// Serialize this approval decision to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        match self {
            Self::Approved => json!({"decision": "approved"}),
            Self::Rejected { reason } => json!({"decision": "rejected", "reason": reason}),
            Self::Modified { new_value } => {
                json!({"decision": "modified", "new_value": new_value})
            }
            Self::Timeout => json!({"decision": "timeout"}),
        }
    }
}

impl fmt::Display for HumanApproval {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Approved => write!(f, "Approved"),
            Self::Rejected { reason } => write!(f, "Rejected({})", reason),
            Self::Modified { .. } => write!(f, "Modified"),
            Self::Timeout => write!(f, "Timeout"),
        }
    }
}

// ---------------------------------------------------------------------------
// ReviewApprovalRequest
// ---------------------------------------------------------------------------

/// A request for human review at a specific point in graph execution.
///
/// Unlike [`ApprovalRequest`], which is tightly coupled to checkpointed graph
/// execution, `ReviewApprovalRequest` is a standalone, serializable request
/// suitable for external review systems and queues.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReviewApprovalRequest {
    /// Unique identifier for this request.
    pub id: String,
    /// The node that triggered the review.
    pub node_name: String,
    /// A human-readable description of the action to be approved.
    pub action_description: String,
    /// The value proposed for approval.
    pub proposed_value: Value,
    /// Arbitrary metadata attached to this request.
    pub metadata: HashMap<String, Value>,
    /// ISO-8601 timestamp when the request was created.
    pub created_at: String,
    /// Optional timeout in milliseconds.
    pub timeout_ms: Option<u64>,
}

impl ReviewApprovalRequest {
    /// Create a new builder for a `ReviewApprovalRequest`.
    pub fn builder(
        node_name: impl Into<String>,
        action_description: impl Into<String>,
    ) -> ReviewApprovalRequestBuilder {
        ReviewApprovalRequestBuilder {
            node_name: node_name.into(),
            action_description: action_description.into(),
            proposed_value: Value::Null,
            metadata: HashMap::new(),
            timeout_ms: None,
        }
    }

    /// Serialize this request to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "node_name": self.node_name,
            "action_description": self.action_description,
            "proposed_value": self.proposed_value,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "timeout_ms": self.timeout_ms,
        })
    }
}

/// Builder for [`ReviewApprovalRequest`].
pub struct ReviewApprovalRequestBuilder {
    node_name: String,
    action_description: String,
    proposed_value: Value,
    metadata: HashMap<String, Value>,
    timeout_ms: Option<u64>,
}

impl ReviewApprovalRequestBuilder {
    /// Set the proposed value for review.
    pub fn proposed_value(mut self, value: Value) -> Self {
        self.proposed_value = value;
        self
    }

    /// Add a metadata entry.
    pub fn metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Set the timeout in milliseconds.
    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = Some(ms);
        self
    }

    /// Build the `ReviewApprovalRequest`.
    pub fn build(self) -> ReviewApprovalRequest {
        ReviewApprovalRequest {
            id: Uuid::now_v7().to_string(),
            node_name: self.node_name,
            action_description: self.action_description,
            proposed_value: self.proposed_value,
            metadata: self.metadata,
            created_at: chrono::Utc::now().to_rfc3339(),
            timeout_ms: self.timeout_ms,
        }
    }
}

// ---------------------------------------------------------------------------
// ApprovalPolicy
// ---------------------------------------------------------------------------

/// Policy that determines whether a value requires human approval.
pub enum ApprovalPolicy {
    /// Always approve without human review.
    AlwaysApprove,
    /// Always reject without human review.
    AlwaysReject,
    /// Always require human review.
    RequireHuman,
    /// Automatically approve if the predicate returns `true`, otherwise require
    /// human review.
    AutoApproveIf(Box<dyn Fn(&Value) -> bool + Send + Sync>),
}

impl fmt::Debug for ApprovalPolicy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AlwaysApprove => write!(f, "AlwaysApprove"),
            Self::AlwaysReject => write!(f, "AlwaysReject"),
            Self::RequireHuman => write!(f, "RequireHuman"),
            Self::AutoApproveIf(_) => write!(f, "AutoApproveIf(<fn>)"),
        }
    }
}

impl ApprovalPolicy {
    /// Evaluate the policy against a value and return a [`HumanApproval`].
    ///
    /// - `AlwaysApprove` returns `Approved`.
    /// - `AlwaysReject` returns `Rejected` with a generic reason.
    /// - `RequireHuman` returns `Timeout` (indicating no automatic decision).
    /// - `AutoApproveIf(f)` returns `Approved` if `f` returns `true`,
    ///   otherwise `Timeout` (awaiting human).
    pub fn evaluate(&self, value: &Value) -> HumanApproval {
        match self {
            Self::AlwaysApprove => HumanApproval::Approved,
            Self::AlwaysReject => HumanApproval::Rejected {
                reason: "Policy: always reject".to_string(),
            },
            Self::RequireHuman => HumanApproval::Timeout,
            Self::AutoApproveIf(predicate) => {
                if predicate(value) {
                    HumanApproval::Approved
                } else {
                    HumanApproval::Timeout
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// HumanReviewPoint
// ---------------------------------------------------------------------------

/// A point in the graph where human review is required.
///
/// Combines a node name with an [`ApprovalPolicy`] to generate
/// [`ReviewApprovalRequest`]s and apply review decisions.
pub struct HumanReviewPoint {
    /// The name of the node requiring review.
    pub node_name: String,
    /// The policy governing automatic vs. human review.
    pub policy: ApprovalPolicy,
}

impl HumanReviewPoint {
    /// Create a new review point for the given node with the specified policy.
    pub fn new(node_name: impl Into<String>, policy: ApprovalPolicy) -> Self {
        Self {
            node_name: node_name.into(),
            policy,
        }
    }

    /// Generate an [`ReviewApprovalRequest`] for the given proposed value.
    pub fn request_approval(&self, value: &Value) -> ReviewApprovalRequest {
        ReviewApprovalRequest::builder(
            &self.node_name,
            format!("Review action at node '{}'", self.node_name),
        )
        .proposed_value(value.clone())
        .build()
    }

    /// Apply a human decision to a proposed value, returning the final value
    /// or an error.
    ///
    /// - `Approved` returns the value unchanged.
    /// - `Modified` returns the modified value.
    /// - `Rejected` returns an error.
    /// - `Timeout` returns an error.
    pub fn apply_decision(&self, decision: HumanApproval, value: Value) -> Result<Value, String> {
        match decision {
            HumanApproval::Approved => Ok(value),
            HumanApproval::Modified { new_value } => Ok(new_value),
            HumanApproval::Rejected { reason } => {
                Err(format!("Rejected at node '{}': {}", self.node_name, reason))
            }
            HumanApproval::Timeout => Err(format!(
                "Timed out waiting for approval at node '{}'",
                self.node_name
            )),
        }
    }
}

impl fmt::Debug for HumanReviewPoint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HumanReviewPoint")
            .field("node_name", &self.node_name)
            .field("policy", &self.policy)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// ReviewQueue
// ---------------------------------------------------------------------------

/// A queue of pending [`ReviewApprovalRequest`]s awaiting human review.
#[derive(Debug)]
pub struct ReviewQueue {
    pending: VecDeque<ReviewApprovalRequest>,
    resolved: HashMap<String, HumanApproval>,
}

impl ReviewQueue {
    /// Create a new, empty review queue.
    pub fn new() -> Self {
        Self {
            pending: VecDeque::new(),
            resolved: HashMap::new(),
        }
    }

    /// Add a request to the queue.
    pub fn enqueue(&mut self, request: ReviewApprovalRequest) {
        self.pending.push_back(request);
    }

    /// Remove and return the next pending request (FIFO order).
    pub fn dequeue(&mut self) -> Option<ReviewApprovalRequest> {
        self.pending.pop_front()
    }

    /// Peek at the next pending request without removing it.
    pub fn peek(&self) -> Option<&ReviewApprovalRequest> {
        self.pending.front()
    }

    /// Resolve a pending request by its ID with the given decision.
    ///
    /// Returns an error if no request with the given ID is found.
    pub fn resolve(&mut self, id: &str, decision: HumanApproval) -> Result<(), String> {
        let idx = self.pending.iter().position(|r| r.id == id);

        match idx {
            Some(i) => {
                self.pending.remove(i);
                self.resolved.insert(id.to_string(), decision);
                Ok(())
            }
            None => Err(format!("No pending request with id '{}'", id)),
        }
    }

    /// Return the number of pending (unresolved) requests.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Return the number of resolved requests.
    pub fn resolved_count(&self) -> usize {
        self.resolved.len()
    }

    /// Returns `true` if there are no pending requests.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }
}

impl Default for ReviewQueue {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// HumanInLoopConfig
// ---------------------------------------------------------------------------

/// Configuration for human-in-the-loop behavior.
#[derive(Debug)]
pub struct HumanInLoopConfig {
    /// The default policy for nodes without explicit configuration.
    pub default_policy: ApprovalPolicy,
    /// Default timeout in milliseconds for approval requests.
    pub timeout_ms: u64,
    /// Nodes that require human review.
    pub review_nodes: Vec<String>,
}

impl HumanInLoopConfig {
    /// Create a new builder for `HumanInLoopConfig`.
    pub fn builder() -> HumanInLoopConfigBuilder {
        HumanInLoopConfigBuilder {
            default_policy: ApprovalPolicy::RequireHuman,
            timeout_ms: 300_000,
            review_nodes: Vec::new(),
        }
    }
}

impl Default for HumanInLoopConfig {
    fn default() -> Self {
        Self {
            default_policy: ApprovalPolicy::RequireHuman,
            timeout_ms: 300_000,
            review_nodes: Vec::new(),
        }
    }
}

/// Builder for [`HumanInLoopConfig`].
pub struct HumanInLoopConfigBuilder {
    default_policy: ApprovalPolicy,
    timeout_ms: u64,
    review_nodes: Vec<String>,
}

impl HumanInLoopConfigBuilder {
    /// Set the default approval policy.
    pub fn default_policy(mut self, policy: ApprovalPolicy) -> Self {
        self.default_policy = policy;
        self
    }

    /// Set the default timeout in milliseconds.
    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Add a node that requires human review.
    pub fn review_node(mut self, node: impl Into<String>) -> Self {
        self.review_nodes.push(node.into());
        self
    }

    /// Set all review nodes at once.
    pub fn review_nodes(mut self, nodes: Vec<String>) -> Self {
        self.review_nodes = nodes;
        self
    }

    /// Build the `HumanInLoopConfig`.
    pub fn build(self) -> HumanInLoopConfig {
        HumanInLoopConfig {
            default_policy: self.default_policy,
            timeout_ms: self.timeout_ms,
            review_nodes: self.review_nodes,
        }
    }
}

// ---------------------------------------------------------------------------
// HumanFeedback
// ---------------------------------------------------------------------------

/// A record of feedback provided by a human reviewer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanFeedback {
    /// The ID of the request this feedback is for.
    pub request_id: String,
    /// The decision made by the reviewer.
    pub decision: HumanApproval,
    /// The reviewer's identity, if known.
    pub reviewer: Option<String>,
    /// Free-form feedback text.
    pub feedback_text: Option<String>,
    /// ISO-8601 timestamp when the feedback was provided.
    pub timestamp: String,
    /// The node name associated with this feedback (for filtering).
    #[serde(default)]
    pub node_name: Option<String>,
}

impl HumanFeedback {
    /// Create a new feedback record.
    pub fn new(request_id: impl Into<String>, decision: HumanApproval) -> Self {
        Self {
            request_id: request_id.into(),
            decision,
            reviewer: None,
            feedback_text: None,
            timestamp: chrono::Utc::now().to_rfc3339(),
            node_name: None,
        }
    }

    /// Set the reviewer identity.
    pub fn with_reviewer(mut self, reviewer: impl Into<String>) -> Self {
        self.reviewer = Some(reviewer.into());
        self
    }

    /// Set feedback text.
    pub fn with_feedback_text(mut self, text: impl Into<String>) -> Self {
        self.feedback_text = Some(text.into());
        self
    }

    /// Set the node name this feedback is associated with.
    pub fn with_node_name(mut self, node: impl Into<String>) -> Self {
        self.node_name = Some(node.into());
        self
    }

    /// Serialize this feedback to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "request_id": self.request_id,
            "decision": self.decision.to_json(),
            "reviewer": self.reviewer,
            "feedback_text": self.feedback_text,
            "timestamp": self.timestamp,
            "node_name": self.node_name,
        })
    }
}

// ---------------------------------------------------------------------------
// FeedbackLog
// ---------------------------------------------------------------------------

/// A log of all human feedback decisions, with query and statistics methods.
#[derive(Debug, Default)]
pub struct FeedbackLog {
    entries: Vec<HumanFeedback>,
}

impl FeedbackLog {
    /// Create a new, empty feedback log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Record a feedback entry.
    pub fn record(&mut self, feedback: HumanFeedback) {
        self.entries.push(feedback);
    }

    /// Return all feedback entries for the given node name.
    pub fn by_node(&self, node_name: &str) -> Vec<&HumanFeedback> {
        self.entries
            .iter()
            .filter(|f| f.node_name.as_deref() == Some(node_name))
            .collect()
    }

    /// Calculate the approval rate as a fraction in [0.0, 1.0].
    ///
    /// Returns 0.0 if there are no entries.
    pub fn approval_rate(&self) -> f64 {
        if self.entries.is_empty() {
            return 0.0;
        }
        let approved = self
            .entries
            .iter()
            .filter(|f| f.decision.is_approved())
            .count();
        approved as f64 / self.entries.len() as f64
    }

    /// Return the most recent `n` feedback entries (newest last).
    pub fn recent(&self, n: usize) -> Vec<&HumanFeedback> {
        let start = self.entries.len().saturating_sub(n);
        self.entries[start..].iter().collect()
    }

    /// Return the number of entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Serialize the entire log to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let entries: Vec<Value> = self.entries.iter().map(|f| f.to_json()).collect();
        json!({
            "entries": entries,
            "count": self.entries.len(),
            "approval_rate": self.approval_rate(),
        })
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

    // ===================================================================
    // HumanApproval tests
    // ===================================================================

    #[test]
    fn test_human_approval_approved() {
        let approval = HumanApproval::Approved;
        assert!(approval.is_approved());
        assert!(!approval.is_rejected());
    }

    #[test]
    fn test_human_approval_rejected() {
        let approval = HumanApproval::Rejected {
            reason: "bad data".to_string(),
        };
        assert!(!approval.is_approved());
        assert!(approval.is_rejected());
    }

    #[test]
    fn test_human_approval_modified() {
        let approval = HumanApproval::Modified {
            new_value: json!({"fixed": true}),
        };
        assert!(!approval.is_approved());
        assert!(!approval.is_rejected());
    }

    #[test]
    fn test_human_approval_timeout() {
        let approval = HumanApproval::Timeout;
        assert!(!approval.is_approved());
        assert!(!approval.is_rejected());
    }

    #[test]
    fn test_human_approval_approved_to_json() {
        let j = HumanApproval::Approved.to_json();
        assert_eq!(j["decision"], "approved");
    }

    #[test]
    fn test_human_approval_rejected_to_json() {
        let j = HumanApproval::Rejected {
            reason: "unsafe".to_string(),
        }
        .to_json();
        assert_eq!(j["decision"], "rejected");
        assert_eq!(j["reason"], "unsafe");
    }

    #[test]
    fn test_human_approval_modified_to_json() {
        let j = HumanApproval::Modified {
            new_value: json!(42),
        }
        .to_json();
        assert_eq!(j["decision"], "modified");
        assert_eq!(j["new_value"], 42);
    }

    #[test]
    fn test_human_approval_timeout_to_json() {
        let j = HumanApproval::Timeout.to_json();
        assert_eq!(j["decision"], "timeout");
    }

    #[test]
    fn test_human_approval_display() {
        assert_eq!(HumanApproval::Approved.to_string(), "Approved");
        assert_eq!(
            HumanApproval::Rejected {
                reason: "no".to_string()
            }
            .to_string(),
            "Rejected(no)"
        );
        assert_eq!(
            HumanApproval::Modified {
                new_value: json!(1)
            }
            .to_string(),
            "Modified"
        );
        assert_eq!(HumanApproval::Timeout.to_string(), "Timeout");
    }

    // ===================================================================
    // ReviewApprovalRequest tests
    // ===================================================================

    #[test]
    fn test_review_approval_request_builder_basic() {
        let req = ReviewApprovalRequest::builder("node1", "Do something")
            .proposed_value(json!({"key": "value"}))
            .build();

        assert_eq!(req.node_name, "node1");
        assert_eq!(req.action_description, "Do something");
        assert_eq!(req.proposed_value, json!({"key": "value"}));
        assert!(!req.id.is_empty());
        assert!(!req.created_at.is_empty());
        assert!(req.timeout_ms.is_none());
        assert!(req.metadata.is_empty());
    }

    #[test]
    fn test_review_approval_request_builder_with_metadata() {
        let req = ReviewApprovalRequest::builder("node2", "Check data")
            .metadata("priority", json!("high"))
            .metadata("source", json!("test"))
            .build();

        assert_eq!(req.metadata.len(), 2);
        assert_eq!(req.metadata["priority"], json!("high"));
        assert_eq!(req.metadata["source"], json!("test"));
    }

    #[test]
    fn test_review_approval_request_builder_with_timeout() {
        let req = ReviewApprovalRequest::builder("node3", "Approve action")
            .timeout_ms(60000)
            .build();

        assert_eq!(req.timeout_ms, Some(60000));
    }

    #[test]
    fn test_review_approval_request_to_json() {
        let req = ReviewApprovalRequest::builder("node4", "Review")
            .proposed_value(json!({"x": 1}))
            .timeout_ms(5000)
            .build();

        let j = req.to_json();
        assert_eq!(j["node_name"], "node4");
        assert_eq!(j["action_description"], "Review");
        assert_eq!(j["proposed_value"], json!({"x": 1}));
        assert_eq!(j["timeout_ms"], 5000);
        assert!(!j["id"].as_str().unwrap().is_empty());
        assert!(!j["created_at"].as_str().unwrap().is_empty());
    }

    #[test]
    fn test_review_approval_request_default_proposed_value() {
        let req = ReviewApprovalRequest::builder("n", "d").build();
        assert!(req.proposed_value.is_null());
    }

    // ===================================================================
    // ApprovalPolicy tests
    // ===================================================================

    #[test]
    fn test_policy_always_approve() {
        let policy = ApprovalPolicy::AlwaysApprove;
        let result = policy.evaluate(&json!({"any": "value"}));
        assert!(result.is_approved());
    }

    #[test]
    fn test_policy_always_reject() {
        let policy = ApprovalPolicy::AlwaysReject;
        let result = policy.evaluate(&json!({"any": "value"}));
        assert!(result.is_rejected());
    }

    #[test]
    fn test_policy_require_human() {
        let policy = ApprovalPolicy::RequireHuman;
        let result = policy.evaluate(&json!({}));
        assert!(matches!(result, HumanApproval::Timeout));
    }

    #[test]
    fn test_policy_auto_approve_if_true() {
        let policy = ApprovalPolicy::AutoApproveIf(Box::new(|v| {
            v.get("safe").and_then(|s| s.as_bool()).unwrap_or(false)
        }));
        let result = policy.evaluate(&json!({"safe": true}));
        assert!(result.is_approved());
    }

    #[test]
    fn test_policy_auto_approve_if_false() {
        let policy = ApprovalPolicy::AutoApproveIf(Box::new(|v| {
            v.get("safe").and_then(|s| s.as_bool()).unwrap_or(false)
        }));
        let result = policy.evaluate(&json!({"safe": false}));
        assert!(matches!(result, HumanApproval::Timeout));
    }

    #[test]
    fn test_policy_auto_approve_if_missing_field() {
        let policy = ApprovalPolicy::AutoApproveIf(Box::new(|v| {
            v.get("safe").and_then(|s| s.as_bool()).unwrap_or(false)
        }));
        let result = policy.evaluate(&json!({}));
        assert!(matches!(result, HumanApproval::Timeout));
    }

    #[test]
    fn test_policy_debug_format() {
        assert!(format!("{:?}", ApprovalPolicy::AlwaysApprove).contains("AlwaysApprove"));
        assert!(format!("{:?}", ApprovalPolicy::AlwaysReject).contains("AlwaysReject"));
        assert!(format!("{:?}", ApprovalPolicy::RequireHuman).contains("RequireHuman"));
        let policy = ApprovalPolicy::AutoApproveIf(Box::new(|_| true));
        assert!(format!("{:?}", policy).contains("AutoApproveIf"));
    }

    // ===================================================================
    // HumanReviewPoint tests
    // ===================================================================

    #[test]
    fn test_review_point_request_approval() {
        let rp = HumanReviewPoint::new("execute_tool", ApprovalPolicy::RequireHuman);
        let req = rp.request_approval(&json!({"tool": "delete_file"}));

        assert_eq!(req.node_name, "execute_tool");
        assert_eq!(req.proposed_value, json!({"tool": "delete_file"}));
        assert!(req.action_description.contains("execute_tool"));
    }

    #[test]
    fn test_review_point_apply_approved() {
        let rp = HumanReviewPoint::new("node", ApprovalPolicy::AlwaysApprove);
        let result = rp.apply_decision(HumanApproval::Approved, json!({"data": 1}));
        assert_eq!(result.unwrap(), json!({"data": 1}));
    }

    #[test]
    fn test_review_point_apply_modified() {
        let rp = HumanReviewPoint::new("node", ApprovalPolicy::RequireHuman);
        let result = rp.apply_decision(
            HumanApproval::Modified {
                new_value: json!({"data": 99}),
            },
            json!({"data": 1}),
        );
        assert_eq!(result.unwrap(), json!({"data": 99}));
    }

    #[test]
    fn test_review_point_apply_rejected() {
        let rp = HumanReviewPoint::new("dangerous_node", ApprovalPolicy::RequireHuman);
        let result = rp.apply_decision(
            HumanApproval::Rejected {
                reason: "too risky".to_string(),
            },
            json!({}),
        );
        let err = result.unwrap_err();
        assert!(err.contains("dangerous_node"));
        assert!(err.contains("too risky"));
    }

    #[test]
    fn test_review_point_apply_timeout() {
        let rp = HumanReviewPoint::new("slow_node", ApprovalPolicy::RequireHuman);
        let result = rp.apply_decision(HumanApproval::Timeout, json!({}));
        let err = result.unwrap_err();
        assert!(err.contains("Timed out"));
        assert!(err.contains("slow_node"));
    }

    #[test]
    fn test_review_point_debug() {
        let rp = HumanReviewPoint::new("n", ApprovalPolicy::AlwaysApprove);
        let dbg = format!("{:?}", rp);
        assert!(dbg.contains("HumanReviewPoint"));
        assert!(dbg.contains("AlwaysApprove"));
    }

    // ===================================================================
    // ReviewQueue tests
    // ===================================================================

    #[test]
    fn test_review_queue_new_empty() {
        let q = ReviewQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.pending_count(), 0);
        assert_eq!(q.resolved_count(), 0);
    }

    #[test]
    fn test_review_queue_enqueue_dequeue() {
        let mut q = ReviewQueue::new();
        let req = ReviewApprovalRequest::builder("n1", "action1").build();
        q.enqueue(req);

        assert_eq!(q.pending_count(), 1);
        assert!(!q.is_empty());

        let dequeued = q.dequeue().unwrap();
        assert_eq!(dequeued.node_name, "n1");
        assert!(q.is_empty());
    }

    #[test]
    fn test_review_queue_fifo_order() {
        let mut q = ReviewQueue::new();
        let r1 = ReviewApprovalRequest::builder("first", "a1").build();
        let r2 = ReviewApprovalRequest::builder("second", "a2").build();
        q.enqueue(r1);
        q.enqueue(r2);

        assert_eq!(q.dequeue().unwrap().node_name, "first");
        assert_eq!(q.dequeue().unwrap().node_name, "second");
    }

    #[test]
    fn test_review_queue_peek() {
        let mut q = ReviewQueue::new();
        assert!(q.peek().is_none());

        let req = ReviewApprovalRequest::builder("peek_node", "peek action").build();
        q.enqueue(req);

        assert_eq!(q.peek().unwrap().node_name, "peek_node");
        assert_eq!(q.pending_count(), 1); // peek does not remove
    }

    #[test]
    fn test_review_queue_resolve_existing() {
        let mut q = ReviewQueue::new();
        let req = ReviewApprovalRequest::builder("rn", "ra").build();
        let id = req.id.clone();
        q.enqueue(req);

        let result = q.resolve(&id, HumanApproval::Approved);
        assert!(result.is_ok());
        assert_eq!(q.pending_count(), 0);
        assert_eq!(q.resolved_count(), 1);
    }

    #[test]
    fn test_review_queue_resolve_nonexistent() {
        let mut q = ReviewQueue::new();
        let result = q.resolve("nonexistent-id", HumanApproval::Approved);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("nonexistent-id"));
    }

    #[test]
    fn test_review_queue_dequeue_empty() {
        let mut q = ReviewQueue::new();
        assert!(q.dequeue().is_none());
    }

    #[test]
    fn test_review_queue_multiple_resolve() {
        let mut q = ReviewQueue::new();
        let r1 = ReviewApprovalRequest::builder("n1", "a1").build();
        let r2 = ReviewApprovalRequest::builder("n2", "a2").build();
        let id1 = r1.id.clone();
        let id2 = r2.id.clone();
        q.enqueue(r1);
        q.enqueue(r2);

        q.resolve(
            &id2,
            HumanApproval::Rejected {
                reason: "no".to_string(),
            },
        )
        .unwrap();
        assert_eq!(q.pending_count(), 1);
        assert_eq!(q.resolved_count(), 1);

        q.resolve(&id1, HumanApproval::Approved).unwrap();
        assert_eq!(q.pending_count(), 0);
        assert_eq!(q.resolved_count(), 2);
    }

    #[test]
    fn test_review_queue_default() {
        let q = ReviewQueue::default();
        assert!(q.is_empty());
    }

    // ===================================================================
    // HumanInLoopConfig tests
    // ===================================================================

    #[test]
    fn test_config_default() {
        let config = HumanInLoopConfig::default();
        assert_eq!(config.timeout_ms, 300_000);
        assert!(config.review_nodes.is_empty());
        assert!(matches!(
            config.default_policy,
            ApprovalPolicy::RequireHuman
        ));
    }

    #[test]
    fn test_config_builder_basic() {
        let config = HumanInLoopConfig::builder()
            .timeout_ms(60_000)
            .review_node("step1")
            .review_node("step2")
            .build();

        assert_eq!(config.timeout_ms, 60_000);
        assert_eq!(config.review_nodes, vec!["step1", "step2"]);
    }

    #[test]
    fn test_config_builder_with_policy() {
        let config = HumanInLoopConfig::builder()
            .default_policy(ApprovalPolicy::AlwaysApprove)
            .build();

        assert!(matches!(
            config.default_policy,
            ApprovalPolicy::AlwaysApprove
        ));
    }

    #[test]
    fn test_config_builder_review_nodes_bulk() {
        let config = HumanInLoopConfig::builder()
            .review_nodes(vec!["a".to_string(), "b".to_string(), "c".to_string()])
            .build();

        assert_eq!(config.review_nodes.len(), 3);
    }

    // ===================================================================
    // HumanFeedback tests
    // ===================================================================

    #[test]
    fn test_human_feedback_new() {
        let fb = HumanFeedback::new("req-123", HumanApproval::Approved);
        assert_eq!(fb.request_id, "req-123");
        assert!(fb.decision.is_approved());
        assert!(fb.reviewer.is_none());
        assert!(fb.feedback_text.is_none());
        assert!(!fb.timestamp.is_empty());
    }

    #[test]
    fn test_human_feedback_with_reviewer() {
        let fb = HumanFeedback::new("req-456", HumanApproval::Approved).with_reviewer("alice");
        assert_eq!(fb.reviewer, Some("alice".to_string()));
    }

    #[test]
    fn test_human_feedback_with_feedback_text() {
        let fb = HumanFeedback::new(
            "req-789",
            HumanApproval::Rejected {
                reason: "bad".to_string(),
            },
        )
        .with_feedback_text("Needs more context");
        assert_eq!(fb.feedback_text, Some("Needs more context".to_string()));
    }

    #[test]
    fn test_human_feedback_with_node_name() {
        let fb = HumanFeedback::new("req-1", HumanApproval::Approved).with_node_name("tool_call");
        assert_eq!(fb.node_name, Some("tool_call".to_string()));
    }

    #[test]
    fn test_human_feedback_to_json() {
        let fb = HumanFeedback::new("req-j", HumanApproval::Approved)
            .with_reviewer("bob")
            .with_feedback_text("looks good");
        let j = fb.to_json();
        assert_eq!(j["request_id"], "req-j");
        assert_eq!(j["decision"]["decision"], "approved");
        assert_eq!(j["reviewer"], "bob");
        assert_eq!(j["feedback_text"], "looks good");
    }

    #[test]
    fn test_human_feedback_builder_chain() {
        let fb = HumanFeedback::new(
            "r1",
            HumanApproval::Modified {
                new_value: json!(42),
            },
        )
        .with_reviewer("carol")
        .with_feedback_text("changed value")
        .with_node_name("compute");
        assert_eq!(fb.request_id, "r1");
        assert_eq!(fb.reviewer, Some("carol".to_string()));
        assert_eq!(fb.feedback_text, Some("changed value".to_string()));
        assert_eq!(fb.node_name, Some("compute".to_string()));
    }

    // ===================================================================
    // FeedbackLog tests
    // ===================================================================

    #[test]
    fn test_feedback_log_new_empty() {
        let log = FeedbackLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        assert_eq!(log.approval_rate(), 0.0);
    }

    #[test]
    fn test_feedback_log_record_and_len() {
        let mut log = FeedbackLog::new();
        log.record(HumanFeedback::new("r1", HumanApproval::Approved));
        log.record(HumanFeedback::new(
            "r2",
            HumanApproval::Rejected {
                reason: "x".to_string(),
            },
        ));
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
    }

    #[test]
    fn test_feedback_log_by_node() {
        let mut log = FeedbackLog::new();
        log.record(HumanFeedback::new("r1", HumanApproval::Approved).with_node_name("nodeA"));
        log.record(HumanFeedback::new("r2", HumanApproval::Approved).with_node_name("nodeB"));
        log.record(
            HumanFeedback::new(
                "r3",
                HumanApproval::Rejected {
                    reason: "r".to_string(),
                },
            )
            .with_node_name("nodeA"),
        );

        assert_eq!(log.by_node("nodeA").len(), 2);
        assert_eq!(log.by_node("nodeB").len(), 1);
        assert_eq!(log.by_node("nodeC").len(), 0);
    }

    #[test]
    fn test_feedback_log_by_node_no_node_name() {
        let mut log = FeedbackLog::new();
        log.record(HumanFeedback::new("r1", HumanApproval::Approved));
        assert_eq!(log.by_node("anything").len(), 0);
    }

    #[test]
    fn test_feedback_log_approval_rate_all_approved() {
        let mut log = FeedbackLog::new();
        log.record(HumanFeedback::new("r1", HumanApproval::Approved));
        log.record(HumanFeedback::new("r2", HumanApproval::Approved));
        assert!((log.approval_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_feedback_log_approval_rate_none_approved() {
        let mut log = FeedbackLog::new();
        log.record(HumanFeedback::new(
            "r1",
            HumanApproval::Rejected {
                reason: "a".to_string(),
            },
        ));
        log.record(HumanFeedback::new("r2", HumanApproval::Timeout));
        assert!((log.approval_rate() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_feedback_log_approval_rate_mixed() {
        let mut log = FeedbackLog::new();
        log.record(HumanFeedback::new("r1", HumanApproval::Approved));
        log.record(HumanFeedback::new(
            "r2",
            HumanApproval::Rejected {
                reason: "x".to_string(),
            },
        ));
        log.record(HumanFeedback::new("r3", HumanApproval::Approved));
        log.record(HumanFeedback::new("r4", HumanApproval::Timeout));
        assert!((log.approval_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_feedback_log_recent() {
        let mut log = FeedbackLog::new();
        for i in 0..5 {
            log.record(HumanFeedback::new(
                format!("r{}", i),
                HumanApproval::Approved,
            ));
        }
        let recent = log.recent(3);
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].request_id, "r2");
        assert_eq!(recent[1].request_id, "r3");
        assert_eq!(recent[2].request_id, "r4");
    }

    #[test]
    fn test_feedback_log_recent_more_than_available() {
        let mut log = FeedbackLog::new();
        log.record(HumanFeedback::new("r1", HumanApproval::Approved));
        let recent = log.recent(10);
        assert_eq!(recent.len(), 1);
    }

    #[test]
    fn test_feedback_log_recent_empty() {
        let log = FeedbackLog::new();
        assert!(log.recent(5).is_empty());
    }

    #[test]
    fn test_feedback_log_to_json() {
        let mut log = FeedbackLog::new();
        log.record(HumanFeedback::new("r1", HumanApproval::Approved));
        log.record(HumanFeedback::new(
            "r2",
            HumanApproval::Rejected {
                reason: "no".to_string(),
            },
        ));

        let j = log.to_json();
        assert_eq!(j["count"], 2);
        assert!((j["approval_rate"].as_f64().unwrap() - 0.5).abs() < f64::EPSILON);
        assert_eq!(j["entries"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn test_feedback_log_default() {
        let log = FeedbackLog::default();
        assert!(log.is_empty());
    }
}
