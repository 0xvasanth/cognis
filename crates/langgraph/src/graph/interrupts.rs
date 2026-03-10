//! Interrupt/resume system for human-in-the-loop workflows.
//!
//! This module provides the building blocks for pausing graph execution,
//! collecting human (or external system) input, and resuming execution
//! with the collected response. Key types:
//!
//! - [`InterruptType`] — the kind of interrupt (approval, input, review, custom)
//! - [`InterruptRequest`] — a request to pause execution at a node
//! - [`InterruptResponse`] — a response from a human or external system
//! - [`InterruptPolicy`] — controls when interrupts are triggered
//! - [`InterruptQueue`] — manages pending requests and their responses
//! - [`InterruptHandler`] — trait for handling interrupt lifecycle events
//! - [`InterruptAction`] — action to take when an interrupt fires
//! - [`InterruptLog`] — audit log for all interrupt events
//! - [`TimeoutPolicy`] — auto-resolve interrupts after a timeout

use std::collections::HashMap;
use std::fmt;

use serde_json::{json, Value};

use crate::errors::{LangGraphError, Result};

// ---------------------------------------------------------------------------
// InterruptType
// ---------------------------------------------------------------------------

/// The kind of interrupt requested during graph execution.
#[derive(Debug, Clone)]
pub enum InterruptType {
    /// Request human approval (yes/no) before continuing.
    Approval,
    /// Request human input with a prompt string.
    Input { prompt: String },
    /// Request human review of arbitrary data.
    Review { data: Value },
    /// A custom interrupt with a user-defined kind and payload.
    Custom { kind: String, payload: Value },
}

impl InterruptType {
    /// Return a short name describing this interrupt kind.
    pub fn name(&self) -> &str {
        match self {
            Self::Approval => "approval",
            Self::Input { .. } => "input",
            Self::Review { .. } => "review",
            Self::Custom { kind, .. } => kind.as_str(),
        }
    }

    /// Serialize this interrupt type to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        match self {
            Self::Approval => json!({ "type": "approval" }),
            Self::Input { prompt } => json!({ "type": "input", "prompt": prompt }),
            Self::Review { data } => json!({ "type": "review", "data": data }),
            Self::Custom { kind, payload } => {
                json!({ "type": "custom", "kind": kind, "payload": payload })
            }
        }
    }
}

impl fmt::Display for InterruptType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Approval => write!(f, "Approval"),
            Self::Input { prompt } => write!(f, "Input({})", prompt),
            Self::Review { .. } => write!(f, "Review"),
            Self::Custom { kind, .. } => write!(f, "Custom({})", kind),
        }
    }
}

// ---------------------------------------------------------------------------
// InterruptRequest
// ---------------------------------------------------------------------------

/// A request to pause graph execution at a particular node.
#[derive(Debug, Clone)]
pub struct InterruptRequest {
    /// Unique identifier for this request.
    pub id: String,
    /// The node where the interrupt was triggered.
    pub node_id: String,
    /// The kind of interrupt.
    pub interrupt_type: InterruptType,
    /// Additional context associated with the interrupt.
    pub context: HashMap<String, Value>,
    /// ISO-8601 timestamp when the request was created.
    pub created_at: String,
}

impl InterruptRequest {
    /// Create a new `InterruptRequest` for the given node and type.
    ///
    /// The `id` is generated as a simple incrementing placeholder (callers
    /// can replace it). `created_at` is set to a fixed placeholder; in
    /// production this would use the real clock.
    pub fn new(node_id: impl Into<String>, interrupt_type: InterruptType) -> Self {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self {
            id: format!("int-{}", seq),
            node_id: node_id.into(),
            interrupt_type,
            context: HashMap::new(),
            created_at: "2026-03-10T00:00:00Z".to_string(),
        }
    }

    /// Add a key-value pair to the context (builder pattern).
    pub fn with_context(mut self, key: impl Into<String>, value: Value) -> Self {
        self.context.insert(key.into(), value);
        self
    }

    /// Serialize this request to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "node_id": self.node_id,
            "interrupt_type": self.interrupt_type.to_json(),
            "context": self.context,
            "created_at": self.created_at,
        })
    }
}

// ---------------------------------------------------------------------------
// InterruptResponse
// ---------------------------------------------------------------------------

/// A response from a human or external system to an interrupt request.
#[derive(Debug, Clone)]
pub struct InterruptResponse {
    /// The ID of the [`InterruptRequest`] this responds to.
    pub request_id: String,
    /// Whether the request was approved.
    pub approved: bool,
    /// Optional data provided with the response.
    pub data: Option<Value>,
    /// ISO-8601 timestamp when the response was created.
    pub responded_at: String,
    /// Optional identifier for the person/system that responded.
    pub responder: Option<String>,
}

impl InterruptResponse {
    /// Create an approval response for the given request ID.
    pub fn approve(request_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            approved: true,
            data: None,
            responded_at: "2026-03-10T00:00:00Z".to_string(),
            responder: None,
        }
    }

    /// Create a rejection response for the given request ID.
    pub fn reject(request_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            approved: false,
            data: None,
            responded_at: "2026-03-10T00:00:00Z".to_string(),
            responder: None,
        }
    }

    /// Create an approval response with additional data.
    pub fn approve_with_data(request_id: impl Into<String>, data: Value) -> Self {
        Self {
            request_id: request_id.into(),
            approved: true,
            data: Some(data),
            responded_at: "2026-03-10T00:00:00Z".to_string(),
            responder: None,
        }
    }

    /// Serialize this response to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "request_id": self.request_id,
            "approved": self.approved,
            "data": self.data,
            "responded_at": self.responded_at,
            "responder": self.responder,
        })
    }
}

// ---------------------------------------------------------------------------
// InterruptPolicy
// ---------------------------------------------------------------------------

/// Controls when interrupts are triggered for specific nodes.
#[derive(Debug, Clone, Default)]
pub struct InterruptPolicy {
    /// Nodes that should always be interrupted (both before and after).
    always: Vec<String>,
    /// Nodes that should be interrupted before execution.
    before: Vec<String>,
    /// Nodes that should be interrupted after execution.
    after: Vec<String>,
    /// Nodes with conditional interrupts: `(node_id, condition_name)`.
    conditional: Vec<(String, String)>,
}

impl InterruptPolicy {
    /// Create a new, empty interrupt policy.
    pub fn new() -> Self {
        Self::default()
    }

    /// Always interrupt at the given node (before and after).
    pub fn always_interrupt(mut self, node_id: impl Into<String>) -> Self {
        self.always.push(node_id.into());
        self
    }

    /// Interrupt before executing the given node.
    pub fn interrupt_before(mut self, node_id: impl Into<String>) -> Self {
        self.before.push(node_id.into());
        self
    }

    /// Interrupt after executing the given node.
    pub fn interrupt_after(mut self, node_id: impl Into<String>) -> Self {
        self.after.push(node_id.into());
        self
    }

    /// Interrupt at the given node when a named condition is met.
    pub fn conditional_interrupt(
        mut self,
        node_id: impl Into<String>,
        condition_name: impl Into<String>,
    ) -> Self {
        self.conditional
            .push((node_id.into(), condition_name.into()));
        self
    }

    /// Check whether the policy requires an interrupt before the given node.
    pub fn should_interrupt_before(&self, node_id: &str) -> bool {
        self.always.iter().any(|n| n == node_id)
            || self.before.iter().any(|n| n == node_id)
            || self.conditional.iter().any(|(n, _)| n == node_id)
    }

    /// Check whether the policy requires an interrupt after the given node.
    pub fn should_interrupt_after(&self, node_id: &str) -> bool {
        self.always.iter().any(|n| n == node_id) || self.after.iter().any(|n| n == node_id)
    }

    /// Serialize this policy to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "always": self.always,
            "before": self.before,
            "after": self.after,
            "conditional": self.conditional.iter()
                .map(|(n, c)| json!({"node_id": n, "condition": c}))
                .collect::<Vec<_>>(),
        })
    }
}

// ---------------------------------------------------------------------------
// InterruptQueue
// ---------------------------------------------------------------------------

/// A queue managing pending interrupt requests and their responses.
#[derive(Debug, Default)]
pub struct InterruptQueue {
    /// Pending requests in FIFO order.
    pending: Vec<InterruptRequest>,
    /// Responses keyed by the request ID.
    responses: HashMap<String, InterruptResponse>,
}

impl InterruptQueue {
    /// Create a new, empty queue.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a request to the back of the queue.
    pub fn enqueue(&mut self, request: InterruptRequest) {
        self.pending.push(request);
    }

    /// Remove and return the request at the front of the queue.
    pub fn dequeue(&mut self) -> Option<InterruptRequest> {
        if self.pending.is_empty() {
            None
        } else {
            Some(self.pending.remove(0))
        }
    }

    /// Peek at the request at the front of the queue without removing it.
    pub fn peek(&self) -> Option<&InterruptRequest> {
        self.pending.first()
    }

    /// Return the number of pending requests.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Return whether the queue has no pending requests.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Submit a response for a request identified by `request_id`.
    ///
    /// Returns an error if no pending request with that ID exists.
    pub fn respond(&mut self, request_id: &str, response: InterruptResponse) -> Result<()> {
        let exists = self
            .pending
            .iter()
            .any(|r| r.id == request_id)
            || self.responses.contains_key(request_id);

        if !exists {
            return Err(LangGraphError::Other(format!(
                "No interrupt request found with id: {}",
                request_id
            )));
        }
        self.responses.insert(request_id.to_string(), response);
        Ok(())
    }

    /// Get the response for a given request ID, if one has been submitted.
    pub fn get_response(&self, request_id: &str) -> Option<&InterruptResponse> {
        self.responses.get(request_id)
    }

    /// Return the number of pending (un-responded) requests.
    pub fn pending_count(&self) -> usize {
        self.pending
            .iter()
            .filter(|r| !self.responses.contains_key(&r.id))
            .count()
    }

    /// Return the number of requests that have received a response.
    pub fn responded_count(&self) -> usize {
        self.responses.len()
    }
}

// ---------------------------------------------------------------------------
// InterruptAction
// ---------------------------------------------------------------------------

/// Action to take when an interrupt is triggered.
#[derive(Debug, Clone)]
pub enum InterruptAction {
    /// Pause execution and wait for a human response.
    Pause,
    /// Automatically approve the interrupt without waiting.
    AutoApprove,
    /// Automatically reject the interrupt without waiting.
    AutoReject,
    /// Wait for a response up to the given number of seconds, then auto-resolve.
    Timeout { seconds: u64 },
}

// ---------------------------------------------------------------------------
// InterruptHandler trait
// ---------------------------------------------------------------------------

/// Trait for components that handle interrupt lifecycle events.
pub trait InterruptHandler: Send + Sync {
    /// Called when an interrupt is triggered. Returns the action to take.
    fn on_interrupt(&self, request: &InterruptRequest) -> InterruptAction;

    /// Called when execution resumes after an interrupt has been responded to.
    fn on_resume(&self, request: &InterruptRequest, response: &InterruptResponse);
}

// ---------------------------------------------------------------------------
// InterruptLogEntry
// ---------------------------------------------------------------------------

/// A single entry in the interrupt audit log.
#[derive(Debug, Clone)]
pub struct InterruptLogEntry {
    /// The kind of event.
    pub event: String,
    /// The request ID.
    pub request_id: String,
    /// The node associated with the event.
    pub node_id: String,
    /// Additional details serialized as JSON.
    pub details: Value,
    /// ISO-8601 timestamp.
    pub timestamp: String,
}

// ---------------------------------------------------------------------------
// InterruptLog
// ---------------------------------------------------------------------------

/// Audit log that records all interrupt-related events (requests and responses).
#[derive(Debug, Default)]
pub struct InterruptLog {
    entries: Vec<InterruptLogEntry>,
}

impl InterruptLog {
    /// Create a new, empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an interrupt request event.
    pub fn record_request(&mut self, request: &InterruptRequest) {
        self.entries.push(InterruptLogEntry {
            event: "request".to_string(),
            request_id: request.id.clone(),
            node_id: request.node_id.clone(),
            details: request.to_json(),
            timestamp: request.created_at.clone(),
        });
    }

    /// Record a response event for a previously recorded request.
    pub fn record_response(&mut self, request_id: &str, response: &InterruptResponse) {
        self.entries.push(InterruptLogEntry {
            event: "response".to_string(),
            request_id: request_id.to_string(),
            node_id: String::new(),
            details: response.to_json(),
            timestamp: response.responded_at.clone(),
        });
    }

    /// Return references to all log entries.
    pub fn entries(&self) -> &[InterruptLogEntry] {
        &self.entries
    }

    /// Return entries associated with the given node.
    pub fn filter_by_node(&self, node_id: &str) -> Vec<&InterruptLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.node_id == node_id)
            .collect()
    }

    /// Serialize the entire log to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        let entries: Vec<Value> = self
            .entries
            .iter()
            .map(|e| {
                json!({
                    "event": e.event,
                    "request_id": e.request_id,
                    "node_id": e.node_id,
                    "details": e.details,
                    "timestamp": e.timestamp,
                })
            })
            .collect();
        json!({ "entries": entries })
    }

    /// Return the number of log entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ---------------------------------------------------------------------------
// TimeoutPolicy
// ---------------------------------------------------------------------------

/// Policy for automatically resolving interrupts after a timeout.
#[derive(Debug, Clone)]
pub struct TimeoutPolicy {
    /// Default timeout in seconds for all nodes.
    default_timeout_secs: u64,
    /// Per-node timeout overrides.
    overrides: HashMap<String, u64>,
}

impl TimeoutPolicy {
    /// Create a new timeout policy with the given default timeout (in seconds).
    pub fn new(default_timeout_secs: u64) -> Self {
        Self {
            default_timeout_secs,
            overrides: HashMap::new(),
        }
    }

    /// Set a per-node timeout override.
    pub fn set_timeout(&mut self, node_id: impl Into<String>, secs: u64) {
        self.overrides.insert(node_id.into(), secs);
    }

    /// Get the effective timeout for a given node.
    pub fn get_timeout(&self, node_id: &str) -> u64 {
        self.overrides
            .get(node_id)
            .copied()
            .unwrap_or(self.default_timeout_secs)
    }

    /// Check whether an interrupt request has expired according to this policy.
    ///
    /// This performs a simple string-based comparison against `created_at`.
    /// In production the timestamps would be parsed properly; here we compare
    /// the `created_at` field against a hardcoded "now" value for determinism.
    pub fn is_expired(&self, request: &InterruptRequest) -> bool {
        // For a real implementation this would parse the ISO-8601 timestamp and
        // compare against the current wall-clock time.  For this module we use
        // a simplified heuristic: the request is expired if the timeout is 0.
        let timeout = self.get_timeout(&request.node_id);
        timeout == 0
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // InterruptType tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_interrupt_type_approval_name() {
        assert_eq!(InterruptType::Approval.name(), "approval");
    }

    #[test]
    fn test_interrupt_type_input_name() {
        let it = InterruptType::Input {
            prompt: "Enter value".to_string(),
        };
        assert_eq!(it.name(), "input");
    }

    #[test]
    fn test_interrupt_type_review_name() {
        let it = InterruptType::Review {
            data: json!({"x": 1}),
        };
        assert_eq!(it.name(), "review");
    }

    #[test]
    fn test_interrupt_type_custom_name() {
        let it = InterruptType::Custom {
            kind: "escalation".to_string(),
            payload: json!({}),
        };
        assert_eq!(it.name(), "escalation");
    }

    #[test]
    fn test_interrupt_type_approval_display() {
        assert_eq!(InterruptType::Approval.to_string(), "Approval");
    }

    #[test]
    fn test_interrupt_type_input_display() {
        let it = InterruptType::Input {
            prompt: "Name?".to_string(),
        };
        assert_eq!(it.to_string(), "Input(Name?)");
    }

    #[test]
    fn test_interrupt_type_review_display() {
        let it = InterruptType::Review { data: json!({}) };
        assert_eq!(it.to_string(), "Review");
    }

    #[test]
    fn test_interrupt_type_custom_display() {
        let it = InterruptType::Custom {
            kind: "foo".to_string(),
            payload: json!(null),
        };
        assert_eq!(it.to_string(), "Custom(foo)");
    }

    #[test]
    fn test_interrupt_type_approval_to_json() {
        let j = InterruptType::Approval.to_json();
        assert_eq!(j["type"], "approval");
    }

    #[test]
    fn test_interrupt_type_input_to_json() {
        let it = InterruptType::Input {
            prompt: "Enter".to_string(),
        };
        let j = it.to_json();
        assert_eq!(j["type"], "input");
        assert_eq!(j["prompt"], "Enter");
    }

    #[test]
    fn test_interrupt_type_review_to_json() {
        let it = InterruptType::Review {
            data: json!({"score": 5}),
        };
        let j = it.to_json();
        assert_eq!(j["type"], "review");
        assert_eq!(j["data"]["score"], 5);
    }

    #[test]
    fn test_interrupt_type_custom_to_json() {
        let it = InterruptType::Custom {
            kind: "alert".to_string(),
            payload: json!({"level": "high"}),
        };
        let j = it.to_json();
        assert_eq!(j["type"], "custom");
        assert_eq!(j["kind"], "alert");
        assert_eq!(j["payload"]["level"], "high");
    }

    // -----------------------------------------------------------------------
    // InterruptRequest tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_request_new() {
        let req = InterruptRequest::new("node_a", InterruptType::Approval);
        assert_eq!(req.node_id, "node_a");
        assert!(req.id.starts_with("int-"));
        assert!(req.context.is_empty());
    }

    #[test]
    fn test_request_with_context() {
        let req = InterruptRequest::new("n1", InterruptType::Approval)
            .with_context("user", json!("alice"))
            .with_context("priority", json!(1));
        assert_eq!(req.context.len(), 2);
        assert_eq!(req.context["user"], json!("alice"));
        assert_eq!(req.context["priority"], json!(1));
    }

    #[test]
    fn test_request_to_json() {
        let req = InterruptRequest::new("n1", InterruptType::Approval)
            .with_context("key", json!("val"));
        let j = req.to_json();
        assert_eq!(j["node_id"], "n1");
        assert_eq!(j["interrupt_type"]["type"], "approval");
        assert_eq!(j["context"]["key"], "val");
        assert!(!j["id"].as_str().unwrap().is_empty());
        assert!(!j["created_at"].as_str().unwrap().is_empty());
    }

    #[test]
    fn test_request_unique_ids() {
        let r1 = InterruptRequest::new("a", InterruptType::Approval);
        let r2 = InterruptRequest::new("b", InterruptType::Approval);
        assert_ne!(r1.id, r2.id);
    }

    #[test]
    fn test_request_input_type() {
        let req = InterruptRequest::new(
            "ask",
            InterruptType::Input {
                prompt: "Your name?".to_string(),
            },
        );
        assert_eq!(req.interrupt_type.name(), "input");
    }

    // -----------------------------------------------------------------------
    // InterruptResponse tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_response_approve() {
        let resp = InterruptResponse::approve("int-1");
        assert_eq!(resp.request_id, "int-1");
        assert!(resp.approved);
        assert!(resp.data.is_none());
        assert!(resp.responder.is_none());
    }

    #[test]
    fn test_response_reject() {
        let resp = InterruptResponse::reject("int-2");
        assert_eq!(resp.request_id, "int-2");
        assert!(!resp.approved);
        assert!(resp.data.is_none());
    }

    #[test]
    fn test_response_approve_with_data() {
        let resp = InterruptResponse::approve_with_data("int-3", json!({"comment": "looks good"}));
        assert!(resp.approved);
        assert_eq!(resp.data.as_ref().unwrap()["comment"], "looks good");
    }

    #[test]
    fn test_response_to_json_approve() {
        let resp = InterruptResponse::approve("int-1");
        let j = resp.to_json();
        assert_eq!(j["request_id"], "int-1");
        assert_eq!(j["approved"], true);
        assert!(j["data"].is_null());
    }

    #[test]
    fn test_response_to_json_reject() {
        let resp = InterruptResponse::reject("int-2");
        let j = resp.to_json();
        assert_eq!(j["approved"], false);
    }

    #[test]
    fn test_response_to_json_with_data() {
        let resp = InterruptResponse::approve_with_data("int-3", json!(42));
        let j = resp.to_json();
        assert_eq!(j["data"], 42);
    }

    #[test]
    fn test_response_responder_field() {
        let mut resp = InterruptResponse::approve("r1");
        resp.responder = Some("admin".to_string());
        let j = resp.to_json();
        assert_eq!(j["responder"], "admin");
    }

    // -----------------------------------------------------------------------
    // InterruptPolicy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_policy_new_empty() {
        let policy = InterruptPolicy::new();
        assert!(!policy.should_interrupt_before("any"));
        assert!(!policy.should_interrupt_after("any"));
    }

    #[test]
    fn test_policy_always_interrupt() {
        let policy = InterruptPolicy::new().always_interrupt("critical");
        assert!(policy.should_interrupt_before("critical"));
        assert!(policy.should_interrupt_after("critical"));
    }

    #[test]
    fn test_policy_interrupt_before() {
        let policy = InterruptPolicy::new().interrupt_before("step1");
        assert!(policy.should_interrupt_before("step1"));
        assert!(!policy.should_interrupt_after("step1"));
    }

    #[test]
    fn test_policy_interrupt_after() {
        let policy = InterruptPolicy::new().interrupt_after("step2");
        assert!(!policy.should_interrupt_before("step2"));
        assert!(policy.should_interrupt_after("step2"));
    }

    #[test]
    fn test_policy_conditional_interrupt() {
        let policy =
            InterruptPolicy::new().conditional_interrupt("review", "needs_human_review");
        assert!(policy.should_interrupt_before("review"));
        // conditional only triggers before
        assert!(!policy.should_interrupt_after("review"));
    }

    #[test]
    fn test_policy_combined() {
        let policy = InterruptPolicy::new()
            .always_interrupt("critical")
            .interrupt_before("step1")
            .interrupt_after("step2");
        assert!(policy.should_interrupt_before("critical"));
        assert!(policy.should_interrupt_after("critical"));
        assert!(policy.should_interrupt_before("step1"));
        assert!(!policy.should_interrupt_after("step1"));
        assert!(!policy.should_interrupt_before("step2"));
        assert!(policy.should_interrupt_after("step2"));
    }

    #[test]
    fn test_policy_no_match() {
        let policy = InterruptPolicy::new().interrupt_before("a");
        assert!(!policy.should_interrupt_before("b"));
        assert!(!policy.should_interrupt_after("a"));
    }

    #[test]
    fn test_policy_to_json() {
        let policy = InterruptPolicy::new()
            .always_interrupt("c")
            .interrupt_before("b")
            .interrupt_after("a")
            .conditional_interrupt("d", "check");
        let j = policy.to_json();
        assert_eq!(j["always"], json!(["c"]));
        assert_eq!(j["before"], json!(["b"]));
        assert_eq!(j["after"], json!(["a"]));
        assert_eq!(j["conditional"][0]["node_id"], "d");
        assert_eq!(j["conditional"][0]["condition"], "check");
    }

    #[test]
    fn test_policy_multiple_before() {
        let policy = InterruptPolicy::new()
            .interrupt_before("a")
            .interrupt_before("b");
        assert!(policy.should_interrupt_before("a"));
        assert!(policy.should_interrupt_before("b"));
        assert!(!policy.should_interrupt_before("c"));
    }

    // -----------------------------------------------------------------------
    // InterruptQueue tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_queue_new_empty() {
        let queue = InterruptQueue::new();
        assert_eq!(queue.len(), 0);
        assert!(queue.is_empty());
        assert!(queue.peek().is_none());
    }

    #[test]
    fn test_queue_enqueue_dequeue() {
        let mut queue = InterruptQueue::new();
        let req = InterruptRequest::new("n1", InterruptType::Approval);
        let req_id = req.id.clone();
        queue.enqueue(req);

        assert_eq!(queue.len(), 1);
        assert!(!queue.is_empty());

        let dequeued = queue.dequeue().unwrap();
        assert_eq!(dequeued.id, req_id);
        assert!(queue.is_empty());
    }

    #[test]
    fn test_queue_fifo_order() {
        let mut queue = InterruptQueue::new();
        let r1 = InterruptRequest::new("first", InterruptType::Approval);
        let r2 = InterruptRequest::new("second", InterruptType::Approval);
        let id1 = r1.id.clone();
        let id2 = r2.id.clone();
        queue.enqueue(r1);
        queue.enqueue(r2);

        assert_eq!(queue.dequeue().unwrap().id, id1);
        assert_eq!(queue.dequeue().unwrap().id, id2);
    }

    #[test]
    fn test_queue_peek() {
        let mut queue = InterruptQueue::new();
        let req = InterruptRequest::new("n1", InterruptType::Approval);
        let req_id = req.id.clone();
        queue.enqueue(req);

        assert_eq!(queue.peek().unwrap().id, req_id);
        // peek should not remove the item
        assert_eq!(queue.len(), 1);
    }

    #[test]
    fn test_queue_dequeue_empty() {
        let mut queue = InterruptQueue::new();
        assert!(queue.dequeue().is_none());
    }

    #[test]
    fn test_queue_respond_success() {
        let mut queue = InterruptQueue::new();
        let req = InterruptRequest::new("n1", InterruptType::Approval);
        let req_id = req.id.clone();
        queue.enqueue(req);

        let resp = InterruptResponse::approve(&req_id);
        assert!(queue.respond(&req_id, resp).is_ok());
        assert!(queue.get_response(&req_id).is_some());
        assert!(queue.get_response(&req_id).unwrap().approved);
    }

    #[test]
    fn test_queue_respond_unknown_id() {
        let mut queue = InterruptQueue::new();
        let resp = InterruptResponse::approve("nonexistent");
        let result = queue.respond("nonexistent", resp);
        assert!(result.is_err());
    }

    #[test]
    fn test_queue_pending_count() {
        let mut queue = InterruptQueue::new();
        let r1 = InterruptRequest::new("a", InterruptType::Approval);
        let r2 = InterruptRequest::new("b", InterruptType::Approval);
        let id1 = r1.id.clone();
        queue.enqueue(r1);
        queue.enqueue(r2);

        assert_eq!(queue.pending_count(), 2);

        let resp = InterruptResponse::approve(&id1);
        queue.respond(&id1, resp).unwrap();
        assert_eq!(queue.pending_count(), 1);
    }

    #[test]
    fn test_queue_responded_count() {
        let mut queue = InterruptQueue::new();
        let r1 = InterruptRequest::new("a", InterruptType::Approval);
        let id1 = r1.id.clone();
        queue.enqueue(r1);

        assert_eq!(queue.responded_count(), 0);

        let resp = InterruptResponse::approve(&id1);
        queue.respond(&id1, resp).unwrap();
        assert_eq!(queue.responded_count(), 1);
    }

    #[test]
    fn test_queue_get_response_none() {
        let queue = InterruptQueue::new();
        assert!(queue.get_response("anything").is_none());
    }

    // -----------------------------------------------------------------------
    // InterruptAction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_interrupt_action_pause() {
        let action = InterruptAction::Pause;
        assert!(matches!(action, InterruptAction::Pause));
    }

    #[test]
    fn test_interrupt_action_auto_approve() {
        let action = InterruptAction::AutoApprove;
        assert!(matches!(action, InterruptAction::AutoApprove));
    }

    #[test]
    fn test_interrupt_action_auto_reject() {
        let action = InterruptAction::AutoReject;
        assert!(matches!(action, InterruptAction::AutoReject));
    }

    #[test]
    fn test_interrupt_action_timeout() {
        let action = InterruptAction::Timeout { seconds: 30 };
        if let InterruptAction::Timeout { seconds } = action {
            assert_eq!(seconds, 30);
        } else {
            panic!("Expected Timeout variant");
        }
    }

    // -----------------------------------------------------------------------
    // InterruptHandler trait test (via mock)
    // -----------------------------------------------------------------------

    struct MockHandler {
        action: InterruptAction,
    }

    impl InterruptHandler for MockHandler {
        fn on_interrupt(&self, _request: &InterruptRequest) -> InterruptAction {
            self.action.clone()
        }
        fn on_resume(&self, _request: &InterruptRequest, _response: &InterruptResponse) {}
    }

    #[test]
    fn test_handler_on_interrupt_pause() {
        let handler = MockHandler {
            action: InterruptAction::Pause,
        };
        let req = InterruptRequest::new("n1", InterruptType::Approval);
        let action = handler.on_interrupt(&req);
        assert!(matches!(action, InterruptAction::Pause));
    }

    #[test]
    fn test_handler_on_interrupt_auto_approve() {
        let handler = MockHandler {
            action: InterruptAction::AutoApprove,
        };
        let req = InterruptRequest::new("n1", InterruptType::Approval);
        let action = handler.on_interrupt(&req);
        assert!(matches!(action, InterruptAction::AutoApprove));
    }

    #[test]
    fn test_handler_on_resume() {
        let handler = MockHandler {
            action: InterruptAction::Pause,
        };
        let req = InterruptRequest::new("n1", InterruptType::Approval);
        let resp = InterruptResponse::approve(&req.id);
        // Just verify it doesn't panic
        handler.on_resume(&req, &resp);
    }

    // -----------------------------------------------------------------------
    // InterruptLog tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_log_new_empty() {
        let log = InterruptLog::new();
        assert_eq!(log.len(), 0);
        assert!(log.is_empty());
        assert!(log.entries().is_empty());
    }

    #[test]
    fn test_log_record_request() {
        let mut log = InterruptLog::new();
        let req = InterruptRequest::new("node_a", InterruptType::Approval);
        log.record_request(&req);

        assert_eq!(log.len(), 1);
        assert_eq!(log.entries()[0].event, "request");
        assert_eq!(log.entries()[0].node_id, "node_a");
    }

    #[test]
    fn test_log_record_response() {
        let mut log = InterruptLog::new();
        let resp = InterruptResponse::approve("int-1");
        log.record_response("int-1", &resp);

        assert_eq!(log.len(), 1);
        assert_eq!(log.entries()[0].event, "response");
        assert_eq!(log.entries()[0].request_id, "int-1");
    }

    #[test]
    fn test_log_filter_by_node() {
        let mut log = InterruptLog::new();
        let r1 = InterruptRequest::new("node_a", InterruptType::Approval);
        let r2 = InterruptRequest::new("node_b", InterruptType::Approval);
        let r3 = InterruptRequest::new("node_a", InterruptType::Review { data: json!({}) });
        log.record_request(&r1);
        log.record_request(&r2);
        log.record_request(&r3);

        assert_eq!(log.filter_by_node("node_a").len(), 2);
        assert_eq!(log.filter_by_node("node_b").len(), 1);
        assert_eq!(log.filter_by_node("node_c").len(), 0);
    }

    #[test]
    fn test_log_to_json() {
        let mut log = InterruptLog::new();
        let req = InterruptRequest::new("n1", InterruptType::Approval);
        log.record_request(&req);

        let j = log.to_json();
        assert!(j["entries"].is_array());
        assert_eq!(j["entries"].as_array().unwrap().len(), 1);
        assert_eq!(j["entries"][0]["event"], "request");
    }

    #[test]
    fn test_log_mixed_events() {
        let mut log = InterruptLog::new();
        let req = InterruptRequest::new("n1", InterruptType::Approval);
        let req_id = req.id.clone();
        log.record_request(&req);

        let resp = InterruptResponse::approve(&req_id);
        log.record_response(&req_id, &resp);

        assert_eq!(log.len(), 2);
        assert_eq!(log.entries()[0].event, "request");
        assert_eq!(log.entries()[1].event, "response");
    }

    #[test]
    fn test_log_default() {
        let log = InterruptLog::default();
        assert!(log.is_empty());
    }

    // -----------------------------------------------------------------------
    // TimeoutPolicy tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_timeout_policy_default_timeout() {
        let policy = TimeoutPolicy::new(60);
        assert_eq!(policy.get_timeout("any_node"), 60);
    }

    #[test]
    fn test_timeout_policy_per_node_override() {
        let mut policy = TimeoutPolicy::new(60);
        policy.set_timeout("critical", 120);
        assert_eq!(policy.get_timeout("critical"), 120);
        assert_eq!(policy.get_timeout("other"), 60);
    }

    #[test]
    fn test_timeout_policy_is_expired_zero_timeout() {
        let mut policy = TimeoutPolicy::new(60);
        policy.set_timeout("instant", 0);
        let req = InterruptRequest::new("instant", InterruptType::Approval);
        assert!(policy.is_expired(&req));
    }

    #[test]
    fn test_timeout_policy_is_expired_nonzero() {
        let policy = TimeoutPolicy::new(60);
        let req = InterruptRequest::new("normal", InterruptType::Approval);
        assert!(!policy.is_expired(&req));
    }

    #[test]
    fn test_timeout_policy_multiple_overrides() {
        let mut policy = TimeoutPolicy::new(30);
        policy.set_timeout("fast", 5);
        policy.set_timeout("slow", 300);
        assert_eq!(policy.get_timeout("fast"), 5);
        assert_eq!(policy.get_timeout("slow"), 300);
        assert_eq!(policy.get_timeout("default"), 30);
    }

    // -----------------------------------------------------------------------
    // Integration tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_interrupt_lifecycle() {
        // 1. Create policy
        let policy = InterruptPolicy::new()
            .interrupt_before("review_node")
            .interrupt_after("deploy_node");

        // 2. Check policy
        assert!(policy.should_interrupt_before("review_node"));
        assert!(policy.should_interrupt_after("deploy_node"));

        // 3. Create request
        let req = InterruptRequest::new("review_node", InterruptType::Approval)
            .with_context("reason", json!("needs human review"));

        // 4. Enqueue
        let mut queue = InterruptQueue::new();
        let req_id = req.id.clone();
        queue.enqueue(req);
        assert_eq!(queue.pending_count(), 1);

        // 5. Log the request
        let mut log = InterruptLog::new();
        log.record_request(queue.peek().unwrap());

        // 6. Respond
        let resp = InterruptResponse::approve_with_data(&req_id, json!({"reviewer": "alice"}));
        queue.respond(&req_id, resp.clone()).unwrap();

        // 7. Log the response
        log.record_response(&req_id, &resp);

        // 8. Verify final state
        assert_eq!(queue.responded_count(), 1);
        assert_eq!(log.len(), 2);
        assert!(queue.get_response(&req_id).unwrap().approved);
    }

    #[test]
    fn test_multiple_requests_lifecycle() {
        let mut queue = InterruptQueue::new();
        let mut log = InterruptLog::new();

        // Create three requests
        let r1 = InterruptRequest::new("node_a", InterruptType::Approval);
        let r2 = InterruptRequest::new("node_b", InterruptType::Input {
            prompt: "Enter value".to_string(),
        });
        let r3 = InterruptRequest::new("node_a", InterruptType::Review {
            data: json!({"doc": "content"}),
        });

        let id1 = r1.id.clone();
        let id2 = r2.id.clone();
        let id3 = r3.id.clone();

        log.record_request(&r1);
        log.record_request(&r2);
        log.record_request(&r3);
        queue.enqueue(r1);
        queue.enqueue(r2);
        queue.enqueue(r3);

        assert_eq!(queue.len(), 3);
        assert_eq!(queue.pending_count(), 3);

        // Respond to first two
        let resp1 = InterruptResponse::approve(&id1);
        let resp2 = InterruptResponse::reject(&id2);
        queue.respond(&id1, resp1.clone()).unwrap();
        queue.respond(&id2, resp2.clone()).unwrap();

        log.record_response(&id1, &resp1);
        log.record_response(&id2, &resp2);

        assert_eq!(queue.responded_count(), 2);
        assert_eq!(queue.pending_count(), 1);
        assert_eq!(log.filter_by_node("node_a").len(), 2);
        assert_eq!(log.filter_by_node("node_b").len(), 1);

        // Verify the unanswered one
        assert!(queue.get_response(&id3).is_none());
    }

    #[test]
    fn test_policy_with_timeout() {
        let policy = InterruptPolicy::new()
            .interrupt_before("step1")
            .interrupt_after("step2");
        let mut timeout = TimeoutPolicy::new(300);
        timeout.set_timeout("step1", 60);

        assert_eq!(timeout.get_timeout("step1"), 60);
        assert_eq!(timeout.get_timeout("step2"), 300);
        assert!(policy.should_interrupt_before("step1"));
    }
}
