//! Collaboration protocols for deep agents.
//!
//! This module enables agents to share work, delegate tasks, and coordinate on
//! complex problems through structured protocols such as round-robin scheduling,
//! hierarchical delegation, consensus voting, and pipeline processing.
//!
//! # Components
//!
//! - [`CollaborationProtocol`] -- defines how agents coordinate (round-robin, hierarchical, etc.)
//! - [`SharedWorkspace`] -- shared key-value state that multiple agents can read and write
//! - [`DelegationRequest`] / [`DelegationResponse`] -- structured task delegation between agents
//! - [`TaskPriority`] -- priority levels for delegation requests
//! - [`DelegationStatus`] -- lifecycle status of a delegation
//! - [`ConsensusVote`] / [`ConsensusRound`] / [`ConsensusResult`] -- consensus-based decision making
//! - [`CollaborationLog`] -- audit log of collaboration events
//!
//! # Example
//!
//! ```rust
//! use cognisagent::collaboration::*;
//! use serde_json::json;
//!
//! // Create a shared workspace
//! let mut workspace = SharedWorkspace::new("project-alpha".into());
//! workspace.put("plan".into(), json!({"steps": ["research", "implement"]}));
//!
//! // Delegate a task
//! let request = DelegationRequest::new("planner", "researcher", "Find relevant papers")
//!     .with_priority(TaskPriority::High)
//!     .with_context("topic", json!("LLM agents"));
//!
//! // Run a consensus round
//! let mut round = ConsensusRound::new("Should we proceed?".into(), 2);
//! round.cast_vote(ConsensusVote::new("agent-a", true, 0.9));
//! round.cast_vote(ConsensusVote::new("agent-b", true, 0.8));
//! assert!(round.has_quorum());
//! ```

use std::collections::HashMap;
use std::fmt;

use serde_json::{json, Value};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// CollaborationProtocol
// ---------------------------------------------------------------------------

/// Defines how a group of agents coordinate their work.
#[derive(Debug, Clone, PartialEq)]
pub enum CollaborationProtocol {
    /// Each agent takes turns processing work in order.
    RoundRobin,
    /// One agent leads and delegates to others.
    Hierarchical {
        /// The agent id of the leader.
        leader: String,
    },
    /// Agents vote to reach agreement; requires a minimum number of votes.
    Consensus {
        /// Minimum number of votes required.
        quorum: usize,
    },
    /// Work flows through agents sequentially, each transforming the output.
    Pipeline,
    /// Work is sent to all agents simultaneously.
    Broadcast,
}

impl CollaborationProtocol {
    /// Returns a human-readable name for the protocol.
    pub fn name(&self) -> &str {
        match self {
            Self::RoundRobin => "round_robin",
            Self::Hierarchical { .. } => "hierarchical",
            Self::Consensus { .. } => "consensus",
            Self::Pipeline => "pipeline",
            Self::Broadcast => "broadcast",
        }
    }

    /// Serializes the protocol to a JSON value.
    pub fn to_json(&self) -> Value {
        match self {
            Self::RoundRobin => json!({"protocol": "round_robin"}),
            Self::Hierarchical { leader } => {
                json!({"protocol": "hierarchical", "leader": leader})
            }
            Self::Consensus { quorum } => {
                json!({"protocol": "consensus", "quorum": quorum})
            }
            Self::Pipeline => json!({"protocol": "pipeline"}),
            Self::Broadcast => json!({"protocol": "broadcast"}),
        }
    }
}

impl fmt::Display for CollaborationProtocol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RoundRobin => write!(f, "RoundRobin"),
            Self::Hierarchical { leader } => write!(f, "Hierarchical(leader={leader})"),
            Self::Consensus { quorum } => write!(f, "Consensus(quorum={quorum})"),
            Self::Pipeline => write!(f, "Pipeline"),
            Self::Broadcast => write!(f, "Broadcast"),
        }
    }
}

// ---------------------------------------------------------------------------
// SharedWorkspace
// ---------------------------------------------------------------------------

/// A shared key-value store that multiple agents can read from and write to.
#[derive(Debug, Clone)]
pub struct SharedWorkspace {
    /// Name of the workspace.
    pub name: String,
    data: HashMap<String, Value>,
}

impl SharedWorkspace {
    /// Creates a new empty workspace with the given name.
    pub fn new(name: String) -> Self {
        Self {
            name,
            data: HashMap::new(),
        }
    }

    /// Inserts or updates a value in the workspace.
    pub fn put(&mut self, key: String, value: Value) {
        self.data.insert(key, value);
    }

    /// Returns a reference to the value associated with the key, if any.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.data.get(key)
    }

    /// Removes and returns the value associated with the key, if any.
    pub fn remove(&mut self, key: &str) -> Option<Value> {
        self.data.remove(key)
    }

    /// Returns an iterator over all keys in the workspace.
    pub fn keys(&self) -> impl Iterator<Item = &String> {
        self.data.keys()
    }

    /// Returns the number of entries in the workspace.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` if the workspace contains no entries.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a clone of the entire workspace data.
    pub fn snapshot(&self) -> HashMap<String, Value> {
        self.data.clone()
    }

    /// Serializes the workspace to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "name": self.name,
            "entries": self.data.len(),
            "data": self.data,
        })
    }
}

// ---------------------------------------------------------------------------
// TaskPriority
// ---------------------------------------------------------------------------

/// Priority level for a delegation request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskPriority {
    /// Must be handled immediately -- failure is catastrophic.
    Critical,
    /// Important, should be handled soon.
    High,
    /// Default priority.
    Normal,
    /// Can be deferred.
    Low,
}

impl TaskPriority {
    /// Returns a numeric weight (higher = more urgent).
    pub fn weight(self) -> u32 {
        match self {
            Self::Critical => 4,
            Self::High => 3,
            Self::Normal => 2,
            Self::Low => 1,
        }
    }
}

impl PartialOrd for TaskPriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TaskPriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.weight().cmp(&other.weight())
    }
}

impl fmt::Display for TaskPriority {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Critical => write!(f, "Critical"),
            Self::High => write!(f, "High"),
            Self::Normal => write!(f, "Normal"),
            Self::Low => write!(f, "Low"),
        }
    }
}

// ---------------------------------------------------------------------------
// DelegationRequest
// ---------------------------------------------------------------------------

/// A request from one agent to another to perform a task.
#[derive(Debug, Clone)]
pub struct DelegationRequest {
    /// Unique identifier for this request.
    pub id: String,
    /// The agent sending the request.
    pub from_agent: String,
    /// The agent receiving the request.
    pub to_agent: String,
    /// Description of the task to perform.
    pub task: String,
    /// Priority of the task.
    pub priority: TaskPriority,
    /// Additional context for the task.
    pub context: HashMap<String, Value>,
    /// Optional deadline (ISO 8601 string).
    pub deadline: Option<String>,
}

impl DelegationRequest {
    /// Creates a new delegation request with default priority and no context.
    pub fn new(from: &str, to: &str, task: &str) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            from_agent: from.to_string(),
            to_agent: to.to_string(),
            task: task.to_string(),
            priority: TaskPriority::Normal,
            context: HashMap::new(),
            deadline: None,
        }
    }

    /// Sets the priority of this request (builder pattern).
    pub fn with_priority(mut self, priority: TaskPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Adds a key-value pair to the request context (builder pattern).
    pub fn with_context(mut self, key: &str, value: Value) -> Self {
        self.context.insert(key.to_string(), value);
        self
    }

    /// Sets the deadline for this request (builder pattern).
    pub fn with_deadline(mut self, deadline: &str) -> Self {
        self.deadline = Some(deadline.to_string());
        self
    }

    /// Serializes the request to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "task": self.task,
            "priority": self.priority.to_string(),
            "context": self.context,
            "deadline": self.deadline,
        })
    }
}

// ---------------------------------------------------------------------------
// DelegationStatus
// ---------------------------------------------------------------------------

/// Lifecycle status of a delegation request.
#[derive(Debug, Clone, PartialEq)]
pub enum DelegationStatus {
    /// The request has been sent but not yet acknowledged.
    Pending,
    /// The receiving agent has accepted the request.
    Accepted,
    /// The receiving agent has rejected the request with a reason.
    Rejected(String),
    /// The task has been completed successfully.
    Completed,
    /// The task has failed with an error description.
    Failed(String),
}

impl DelegationStatus {
    /// Returns `true` if the status represents a terminal state
    /// (Completed, Rejected, or Failed).
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Completed | Self::Rejected(_) | Self::Failed(_))
    }
}

impl fmt::Display for DelegationStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "Pending"),
            Self::Accepted => write!(f, "Accepted"),
            Self::Rejected(reason) => write!(f, "Rejected({reason})"),
            Self::Completed => write!(f, "Completed"),
            Self::Failed(error) => write!(f, "Failed({error})"),
        }
    }
}

// ---------------------------------------------------------------------------
// DelegationResponse
// ---------------------------------------------------------------------------

/// Response to a [`DelegationRequest`].
#[derive(Debug, Clone)]
pub struct DelegationResponse {
    /// The id of the request this responds to.
    pub request_id: String,
    /// Current status.
    pub status: DelegationStatus,
    /// Optional result payload (set on completion).
    pub result: Option<Value>,
    /// Optional error message (set on failure).
    pub error: Option<String>,
    /// Timestamp of when the response was created.
    pub responded_at: String,
}

impl DelegationResponse {
    /// Creates an acceptance response.
    pub fn accept(request_id: &str) -> Self {
        Self {
            request_id: request_id.to_string(),
            status: DelegationStatus::Accepted,
            result: None,
            error: None,
            responded_at: current_timestamp(),
        }
    }

    /// Creates a rejection response with a reason.
    pub fn reject(request_id: &str, reason: &str) -> Self {
        Self {
            request_id: request_id.to_string(),
            status: DelegationStatus::Rejected(reason.to_string()),
            result: None,
            error: None,
            responded_at: current_timestamp(),
        }
    }

    /// Creates a completion response with a result payload.
    pub fn complete(request_id: &str, result: Value) -> Self {
        Self {
            request_id: request_id.to_string(),
            status: DelegationStatus::Completed,
            result: Some(result),
            error: None,
            responded_at: current_timestamp(),
        }
    }

    /// Creates a failure response with an error message.
    pub fn fail(request_id: &str, error: &str) -> Self {
        Self {
            request_id: request_id.to_string(),
            status: DelegationStatus::Failed(error.to_string()),
            result: None,
            error: Some(error.to_string()),
            responded_at: current_timestamp(),
        }
    }

    /// Serializes the response to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "request_id": self.request_id,
            "status": self.status.to_string(),
            "result": self.result,
            "error": self.error,
            "responded_at": self.responded_at,
        })
    }
}

// ---------------------------------------------------------------------------
// ConsensusVote
// ---------------------------------------------------------------------------

/// A single vote cast during a consensus round.
#[derive(Debug, Clone)]
pub struct ConsensusVote {
    /// The agent casting the vote.
    pub voter: String,
    /// `true` for approve, `false` for reject.
    pub decision: bool,
    /// Confidence level in the decision, between 0.0 and 1.0.
    pub confidence: f64,
    /// Optional reasoning behind the vote.
    pub reasoning: Option<String>,
    /// Timestamp of when the vote was cast.
    pub timestamp: String,
}

impl ConsensusVote {
    /// Creates a new vote.
    pub fn new(voter: &str, decision: bool, confidence: f64) -> Self {
        Self {
            voter: voter.to_string(),
            decision,
            confidence: confidence.clamp(0.0, 1.0),
            timestamp: current_timestamp(),
            reasoning: None,
        }
    }

    /// Attaches reasoning to the vote (builder pattern).
    pub fn with_reasoning(mut self, reasoning: &str) -> Self {
        self.reasoning = Some(reasoning.to_string());
        self
    }

    /// Serializes the vote to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "voter": self.voter,
            "decision": self.decision,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
        })
    }
}

// ---------------------------------------------------------------------------
// ConsensusResult
// ---------------------------------------------------------------------------

/// The outcome of a completed consensus round.
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// Whether the proposal was approved (majority voted `true`).
    pub approved: bool,
    /// Number of votes in favour.
    pub votes_for: usize,
    /// Number of votes against.
    pub votes_against: usize,
    /// Average confidence across all votes.
    pub average_confidence: f64,
}

impl ConsensusResult {
    /// Serializes the result to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "approved": self.approved,
            "votes_for": self.votes_for,
            "votes_against": self.votes_against,
            "average_confidence": self.average_confidence,
        })
    }
}

// ---------------------------------------------------------------------------
// ConsensusRound
// ---------------------------------------------------------------------------

/// Manages a round of consensus voting among agents.
#[derive(Debug, Clone)]
pub struct ConsensusRound {
    /// The topic being voted on.
    pub topic: String,
    /// Minimum number of votes required to reach quorum.
    pub quorum: usize,
    votes: Vec<ConsensusVote>,
}

impl ConsensusRound {
    /// Creates a new consensus round.
    pub fn new(topic: String, quorum: usize) -> Self {
        Self {
            topic,
            quorum,
            votes: Vec::new(),
        }
    }

    /// Casts a vote in this round.
    pub fn cast_vote(&mut self, vote: ConsensusVote) {
        self.votes.push(vote);
    }

    /// Returns `true` if enough votes have been cast to meet the quorum.
    pub fn has_quorum(&self) -> bool {
        self.votes.len() >= self.quorum
    }

    /// Computes the consensus result if quorum has been reached.
    pub fn result(&self) -> Option<ConsensusResult> {
        if !self.has_quorum() {
            return None;
        }

        let votes_for = self.votes.iter().filter(|v| v.decision).count();
        let votes_against = self.votes.len() - votes_for;
        let average_confidence = if self.votes.is_empty() {
            0.0
        } else {
            self.votes.iter().map(|v| v.confidence).sum::<f64>() / self.votes.len() as f64
        };

        Some(ConsensusResult {
            approved: votes_for > votes_against,
            votes_for,
            votes_against,
            average_confidence,
        })
    }

    /// Returns a slice of all votes cast so far.
    pub fn votes(&self) -> &[ConsensusVote] {
        &self.votes
    }

    /// Returns the participation rate as a fraction of `total_voters`.
    pub fn participation_rate(&self, total_voters: usize) -> f64 {
        if total_voters == 0 {
            return 0.0;
        }
        self.votes.len() as f64 / total_voters as f64
    }

    /// Serializes the round to a JSON value.
    pub fn to_json(&self) -> Value {
        let result_json = self.result().map(|r| r.to_json());
        json!({
            "topic": self.topic,
            "quorum": self.quorum,
            "votes_cast": self.votes.len(),
            "has_quorum": self.has_quorum(),
            "result": result_json,
            "votes": self.votes.iter().map(|v| v.to_json()).collect::<Vec<_>>(),
        })
    }
}

// ---------------------------------------------------------------------------
// CollaborationLog
// ---------------------------------------------------------------------------

/// A single entry in the collaboration log.
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// The kind of event.
    pub event_type: String,
    /// The agent(s) involved.
    pub agents: Vec<String>,
    /// Details payload.
    pub details: Value,
    /// Timestamp of the event.
    pub timestamp: String,
}

/// An append-only log of collaboration events for auditing and debugging.
#[derive(Debug, Clone)]
pub struct CollaborationLog {
    entries: Vec<LogEntry>,
}

impl CollaborationLog {
    /// Creates a new empty log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Records a delegation request event.
    pub fn record_delegation(&mut self, request: &DelegationRequest) {
        self.entries.push(LogEntry {
            event_type: "delegation_request".to_string(),
            agents: vec![request.from_agent.clone(), request.to_agent.clone()],
            details: request.to_json(),
            timestamp: current_timestamp(),
        });
    }

    /// Records a delegation response event.
    pub fn record_response(&mut self, response: &DelegationResponse) {
        self.entries.push(LogEntry {
            event_type: "delegation_response".to_string(),
            agents: vec![],
            details: response.to_json(),
            timestamp: current_timestamp(),
        });
    }

    /// Records a consensus vote event.
    pub fn record_vote(&mut self, round_topic: &str, vote: &ConsensusVote) {
        self.entries.push(LogEntry {
            event_type: "consensus_vote".to_string(),
            agents: vec![vote.voter.clone()],
            details: json!({
                "round_topic": round_topic,
                "vote": vote.to_json(),
            }),
            timestamp: current_timestamp(),
        });
    }

    /// Returns a slice of all log entries.
    pub fn entries(&self) -> &[LogEntry] {
        &self.entries
    }

    /// Returns entries that involve the given agent.
    pub fn filter_by_agent(&self, agent: &str) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| e.agents.iter().any(|a| a == agent))
            .collect()
    }

    /// Serializes the log to a JSON value.
    pub fn to_json(&self) -> Value {
        json!({
            "total_entries": self.entries.len(),
            "entries": self.entries.iter().map(|e| json!({
                "event_type": e.event_type,
                "agents": e.agents,
                "details": e.details,
                "timestamp": e.timestamp,
            })).collect::<Vec<_>>(),
        })
    }
}

impl Default for CollaborationLog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns a simple timestamp string for logging purposes.
fn current_timestamp() -> String {
    // Using a simple counter-based approach since we avoid external crates.
    // In production this would use chrono or similar.
    "2026-03-11T00:00:00Z".to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -- CollaborationProtocol tests --

    #[test]
    fn test_protocol_round_robin_name() {
        assert_eq!(CollaborationProtocol::RoundRobin.name(), "round_robin");
    }

    #[test]
    fn test_protocol_hierarchical_name() {
        let p = CollaborationProtocol::Hierarchical {
            leader: "boss".into(),
        };
        assert_eq!(p.name(), "hierarchical");
    }

    #[test]
    fn test_protocol_consensus_name() {
        let p = CollaborationProtocol::Consensus { quorum: 3 };
        assert_eq!(p.name(), "consensus");
    }

    #[test]
    fn test_protocol_pipeline_name() {
        assert_eq!(CollaborationProtocol::Pipeline.name(), "pipeline");
    }

    #[test]
    fn test_protocol_broadcast_name() {
        assert_eq!(CollaborationProtocol::Broadcast.name(), "broadcast");
    }

    #[test]
    fn test_protocol_round_robin_display() {
        assert_eq!(
            format!("{}", CollaborationProtocol::RoundRobin),
            "RoundRobin"
        );
    }

    #[test]
    fn test_protocol_hierarchical_display() {
        let p = CollaborationProtocol::Hierarchical {
            leader: "alpha".into(),
        };
        assert_eq!(format!("{p}"), "Hierarchical(leader=alpha)");
    }

    #[test]
    fn test_protocol_consensus_display() {
        let p = CollaborationProtocol::Consensus { quorum: 5 };
        assert_eq!(format!("{p}"), "Consensus(quorum=5)");
    }

    #[test]
    fn test_protocol_pipeline_display() {
        assert_eq!(format!("{}", CollaborationProtocol::Pipeline), "Pipeline");
    }

    #[test]
    fn test_protocol_broadcast_display() {
        assert_eq!(format!("{}", CollaborationProtocol::Broadcast), "Broadcast");
    }

    #[test]
    fn test_protocol_round_robin_to_json() {
        let j = CollaborationProtocol::RoundRobin.to_json();
        assert_eq!(j["protocol"], "round_robin");
    }

    #[test]
    fn test_protocol_hierarchical_to_json() {
        let p = CollaborationProtocol::Hierarchical {
            leader: "leader-1".into(),
        };
        let j = p.to_json();
        assert_eq!(j["protocol"], "hierarchical");
        assert_eq!(j["leader"], "leader-1");
    }

    #[test]
    fn test_protocol_consensus_to_json() {
        let p = CollaborationProtocol::Consensus { quorum: 7 };
        let j = p.to_json();
        assert_eq!(j["protocol"], "consensus");
        assert_eq!(j["quorum"], 7);
    }

    #[test]
    fn test_protocol_equality() {
        assert_eq!(
            CollaborationProtocol::RoundRobin,
            CollaborationProtocol::RoundRobin
        );
        assert_ne!(
            CollaborationProtocol::RoundRobin,
            CollaborationProtocol::Pipeline
        );
    }

    // -- SharedWorkspace tests --

    #[test]
    fn test_workspace_new_is_empty() {
        let ws = SharedWorkspace::new("test".into());
        assert!(ws.is_empty());
        assert_eq!(ws.len(), 0);
        assert_eq!(ws.name, "test");
    }

    #[test]
    fn test_workspace_put_and_get() {
        let mut ws = SharedWorkspace::new("ws".into());
        ws.put("key1".into(), json!("value1"));
        assert_eq!(ws.get("key1"), Some(&json!("value1")));
        assert_eq!(ws.len(), 1);
    }

    #[test]
    fn test_workspace_get_missing() {
        let ws = SharedWorkspace::new("ws".into());
        assert_eq!(ws.get("missing"), None);
    }

    #[test]
    fn test_workspace_put_overwrite() {
        let mut ws = SharedWorkspace::new("ws".into());
        ws.put("k".into(), json!(1));
        ws.put("k".into(), json!(2));
        assert_eq!(ws.get("k"), Some(&json!(2)));
        assert_eq!(ws.len(), 1);
    }

    #[test]
    fn test_workspace_remove() {
        let mut ws = SharedWorkspace::new("ws".into());
        ws.put("k".into(), json!(42));
        let removed = ws.remove("k");
        assert_eq!(removed, Some(json!(42)));
        assert!(ws.is_empty());
    }

    #[test]
    fn test_workspace_remove_missing() {
        let mut ws = SharedWorkspace::new("ws".into());
        assert_eq!(ws.remove("nope"), None);
    }

    #[test]
    fn test_workspace_keys() {
        let mut ws = SharedWorkspace::new("ws".into());
        ws.put("a".into(), json!(1));
        ws.put("b".into(), json!(2));
        let mut keys: Vec<&String> = ws.keys().collect();
        keys.sort();
        assert_eq!(keys, vec!["a", "b"]);
    }

    #[test]
    fn test_workspace_snapshot() {
        let mut ws = SharedWorkspace::new("ws".into());
        ws.put("x".into(), json!(10));
        let snap = ws.snapshot();
        assert_eq!(snap.get("x"), Some(&json!(10)));
        // Snapshot is independent.
        ws.put("x".into(), json!(20));
        assert_eq!(snap.get("x"), Some(&json!(10)));
    }

    #[test]
    fn test_workspace_to_json() {
        let mut ws = SharedWorkspace::new("myws".into());
        ws.put("foo".into(), json!("bar"));
        let j = ws.to_json();
        assert_eq!(j["name"], "myws");
        assert_eq!(j["entries"], 1);
    }

    // -- TaskPriority tests --

    #[test]
    fn test_priority_weights() {
        assert_eq!(TaskPriority::Critical.weight(), 4);
        assert_eq!(TaskPriority::High.weight(), 3);
        assert_eq!(TaskPriority::Normal.weight(), 2);
        assert_eq!(TaskPriority::Low.weight(), 1);
    }

    #[test]
    fn test_priority_ordering() {
        assert!(TaskPriority::Critical > TaskPriority::High);
        assert!(TaskPriority::High > TaskPriority::Normal);
        assert!(TaskPriority::Normal > TaskPriority::Low);
    }

    #[test]
    fn test_priority_display() {
        assert_eq!(format!("{}", TaskPriority::Critical), "Critical");
        assert_eq!(format!("{}", TaskPriority::Low), "Low");
    }

    #[test]
    fn test_priority_sort() {
        let mut priorities = vec![
            TaskPriority::Low,
            TaskPriority::Critical,
            TaskPriority::Normal,
            TaskPriority::High,
        ];
        priorities.sort();
        assert_eq!(
            priorities,
            vec![
                TaskPriority::Low,
                TaskPriority::Normal,
                TaskPriority::High,
                TaskPriority::Critical,
            ]
        );
    }

    // -- DelegationRequest tests --

    #[test]
    fn test_delegation_request_new() {
        let req = DelegationRequest::new("agent-a", "agent-b", "do something");
        assert_eq!(req.from_agent, "agent-a");
        assert_eq!(req.to_agent, "agent-b");
        assert_eq!(req.task, "do something");
        assert_eq!(req.priority, TaskPriority::Normal);
        assert!(req.context.is_empty());
        assert!(req.deadline.is_none());
        assert!(!req.id.is_empty());
    }

    #[test]
    fn test_delegation_request_with_priority() {
        let req = DelegationRequest::new("a", "b", "task").with_priority(TaskPriority::Critical);
        assert_eq!(req.priority, TaskPriority::Critical);
    }

    #[test]
    fn test_delegation_request_with_context() {
        let req = DelegationRequest::new("a", "b", "task").with_context("key", json!("val"));
        assert_eq!(req.context.get("key"), Some(&json!("val")));
    }

    #[test]
    fn test_delegation_request_with_deadline() {
        let req = DelegationRequest::new("a", "b", "task").with_deadline("2026-12-31T23:59:59Z");
        assert_eq!(req.deadline, Some("2026-12-31T23:59:59Z".to_string()));
    }

    #[test]
    fn test_delegation_request_builder_chain() {
        let req = DelegationRequest::new("from", "to", "task")
            .with_priority(TaskPriority::High)
            .with_context("a", json!(1))
            .with_context("b", json!(2))
            .with_deadline("2026-06-01");
        assert_eq!(req.priority, TaskPriority::High);
        assert_eq!(req.context.len(), 2);
        assert_eq!(req.deadline, Some("2026-06-01".to_string()));
    }

    #[test]
    fn test_delegation_request_to_json() {
        let req = DelegationRequest::new("x", "y", "my task");
        let j = req.to_json();
        assert_eq!(j["from_agent"], "x");
        assert_eq!(j["to_agent"], "y");
        assert_eq!(j["task"], "my task");
        assert_eq!(j["priority"], "Normal");
    }

    #[test]
    fn test_delegation_request_unique_ids() {
        let r1 = DelegationRequest::new("a", "b", "t");
        let r2 = DelegationRequest::new("a", "b", "t");
        assert_ne!(r1.id, r2.id);
    }

    // -- DelegationStatus tests --

    #[test]
    fn test_status_pending_not_terminal() {
        assert!(!DelegationStatus::Pending.is_terminal());
    }

    #[test]
    fn test_status_accepted_not_terminal() {
        assert!(!DelegationStatus::Accepted.is_terminal());
    }

    #[test]
    fn test_status_rejected_is_terminal() {
        assert!(DelegationStatus::Rejected("busy".into()).is_terminal());
    }

    #[test]
    fn test_status_completed_is_terminal() {
        assert!(DelegationStatus::Completed.is_terminal());
    }

    #[test]
    fn test_status_failed_is_terminal() {
        assert!(DelegationStatus::Failed("oops".into()).is_terminal());
    }

    #[test]
    fn test_status_display() {
        assert_eq!(format!("{}", DelegationStatus::Pending), "Pending");
        assert_eq!(format!("{}", DelegationStatus::Accepted), "Accepted");
        assert_eq!(
            format!("{}", DelegationStatus::Rejected("no".into())),
            "Rejected(no)"
        );
        assert_eq!(format!("{}", DelegationStatus::Completed), "Completed");
        assert_eq!(
            format!("{}", DelegationStatus::Failed("err".into())),
            "Failed(err)"
        );
    }

    // -- DelegationResponse tests --

    #[test]
    fn test_response_accept() {
        let resp = DelegationResponse::accept("req-1");
        assert_eq!(resp.request_id, "req-1");
        assert_eq!(resp.status, DelegationStatus::Accepted);
        assert!(resp.result.is_none());
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_response_reject() {
        let resp = DelegationResponse::reject("req-2", "too busy");
        assert_eq!(resp.request_id, "req-2");
        assert_eq!(resp.status, DelegationStatus::Rejected("too busy".into()));
    }

    #[test]
    fn test_response_complete() {
        let resp = DelegationResponse::complete("req-3", json!({"answer": 42}));
        assert_eq!(resp.status, DelegationStatus::Completed);
        assert_eq!(resp.result, Some(json!({"answer": 42})));
        assert!(resp.error.is_none());
    }

    #[test]
    fn test_response_fail() {
        let resp = DelegationResponse::fail("req-4", "timeout");
        assert_eq!(resp.status, DelegationStatus::Failed("timeout".into()));
        assert_eq!(resp.error, Some("timeout".into()));
    }

    #[test]
    fn test_response_to_json() {
        let resp = DelegationResponse::complete("r1", json!("done"));
        let j = resp.to_json();
        assert_eq!(j["request_id"], "r1");
        assert_eq!(j["status"], "Completed");
        assert_eq!(j["result"], "done");
    }

    // -- ConsensusVote tests --

    #[test]
    fn test_vote_new() {
        let vote = ConsensusVote::new("voter-1", true, 0.85);
        assert_eq!(vote.voter, "voter-1");
        assert!(vote.decision);
        assert!((vote.confidence - 0.85).abs() < f64::EPSILON);
        assert!(vote.reasoning.is_none());
    }

    #[test]
    fn test_vote_confidence_clamped() {
        let v1 = ConsensusVote::new("v", true, 1.5);
        assert!((v1.confidence - 1.0).abs() < f64::EPSILON);
        let v2 = ConsensusVote::new("v", false, -0.5);
        assert!((v2.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_vote_with_reasoning() {
        let vote = ConsensusVote::new("v", true, 0.9).with_reasoning("looks good");
        assert_eq!(vote.reasoning, Some("looks good".to_string()));
    }

    #[test]
    fn test_vote_to_json() {
        let vote = ConsensusVote::new("agent-x", false, 0.3);
        let j = vote.to_json();
        assert_eq!(j["voter"], "agent-x");
        assert_eq!(j["decision"], false);
    }

    // -- ConsensusRound tests --

    #[test]
    fn test_round_new() {
        let round = ConsensusRound::new("topic".into(), 3);
        assert_eq!(round.topic, "topic");
        assert_eq!(round.quorum, 3);
        assert!(round.votes().is_empty());
        assert!(!round.has_quorum());
    }

    #[test]
    fn test_round_cast_vote() {
        let mut round = ConsensusRound::new("t".into(), 1);
        round.cast_vote(ConsensusVote::new("v1", true, 0.9));
        assert_eq!(round.votes().len(), 1);
    }

    #[test]
    fn test_round_quorum_reached() {
        let mut round = ConsensusRound::new("t".into(), 2);
        assert!(!round.has_quorum());
        round.cast_vote(ConsensusVote::new("v1", true, 0.9));
        assert!(!round.has_quorum());
        round.cast_vote(ConsensusVote::new("v2", false, 0.7));
        assert!(round.has_quorum());
    }

    #[test]
    fn test_round_result_none_before_quorum() {
        let round = ConsensusRound::new("t".into(), 2);
        assert!(round.result().is_none());
    }

    #[test]
    fn test_round_result_approved() {
        let mut round = ConsensusRound::new("t".into(), 2);
        round.cast_vote(ConsensusVote::new("a", true, 0.9));
        round.cast_vote(ConsensusVote::new("b", true, 0.8));
        let result = round.result().unwrap();
        assert!(result.approved);
        assert_eq!(result.votes_for, 2);
        assert_eq!(result.votes_against, 0);
    }

    #[test]
    fn test_round_result_rejected() {
        let mut round = ConsensusRound::new("t".into(), 2);
        round.cast_vote(ConsensusVote::new("a", false, 0.9));
        round.cast_vote(ConsensusVote::new("b", false, 0.8));
        let result = round.result().unwrap();
        assert!(!result.approved);
        assert_eq!(result.votes_for, 0);
        assert_eq!(result.votes_against, 2);
    }

    #[test]
    fn test_round_result_average_confidence() {
        let mut round = ConsensusRound::new("t".into(), 2);
        round.cast_vote(ConsensusVote::new("a", true, 0.8));
        round.cast_vote(ConsensusVote::new("b", true, 0.6));
        let result = round.result().unwrap();
        assert!((result.average_confidence - 0.7).abs() < f64::EPSILON);
    }

    #[test]
    fn test_round_participation_rate() {
        let mut round = ConsensusRound::new("t".into(), 1);
        round.cast_vote(ConsensusVote::new("a", true, 0.5));
        assert!((round.participation_rate(4) - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn test_round_participation_rate_zero_voters() {
        let round = ConsensusRound::new("t".into(), 1);
        assert!((round.participation_rate(0) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_round_to_json() {
        let mut round = ConsensusRound::new("should we ship?".into(), 1);
        round.cast_vote(ConsensusVote::new("a", true, 1.0));
        let j = round.to_json();
        assert_eq!(j["topic"], "should we ship?");
        assert_eq!(j["quorum"], 1);
        assert_eq!(j["has_quorum"], true);
        assert_eq!(j["votes_cast"], 1);
    }

    // -- ConsensusResult tests --

    #[test]
    fn test_consensus_result_to_json() {
        let r = ConsensusResult {
            approved: true,
            votes_for: 3,
            votes_against: 1,
            average_confidence: 0.75,
        };
        let j = r.to_json();
        assert_eq!(j["approved"], true);
        assert_eq!(j["votes_for"], 3);
        assert_eq!(j["votes_against"], 1);
    }

    // -- CollaborationLog tests --

    #[test]
    fn test_log_new_is_empty() {
        let log = CollaborationLog::new();
        assert!(log.entries().is_empty());
    }

    #[test]
    fn test_log_default() {
        let log = CollaborationLog::default();
        assert!(log.entries().is_empty());
    }

    #[test]
    fn test_log_record_delegation() {
        let mut log = CollaborationLog::new();
        let req = DelegationRequest::new("a", "b", "task");
        log.record_delegation(&req);
        assert_eq!(log.entries().len(), 1);
        assert_eq!(log.entries()[0].event_type, "delegation_request");
        assert_eq!(log.entries()[0].agents, vec!["a", "b"]);
    }

    #[test]
    fn test_log_record_response() {
        let mut log = CollaborationLog::new();
        let resp = DelegationResponse::accept("req-1");
        log.record_response(&resp);
        assert_eq!(log.entries().len(), 1);
        assert_eq!(log.entries()[0].event_type, "delegation_response");
    }

    #[test]
    fn test_log_record_vote() {
        let mut log = CollaborationLog::new();
        let vote = ConsensusVote::new("voter-1", true, 0.9);
        log.record_vote("topic-1", &vote);
        assert_eq!(log.entries().len(), 1);
        assert_eq!(log.entries()[0].event_type, "consensus_vote");
        assert_eq!(log.entries()[0].agents, vec!["voter-1"]);
    }

    #[test]
    fn test_log_filter_by_agent() {
        let mut log = CollaborationLog::new();
        let req1 = DelegationRequest::new("alice", "bob", "task1");
        let req2 = DelegationRequest::new("charlie", "dave", "task2");
        log.record_delegation(&req1);
        log.record_delegation(&req2);
        let alice_entries = log.filter_by_agent("alice");
        assert_eq!(alice_entries.len(), 1);
        let dave_entries = log.filter_by_agent("dave");
        assert_eq!(dave_entries.len(), 1);
        let nobody = log.filter_by_agent("nobody");
        assert_eq!(nobody.len(), 0);
    }

    #[test]
    fn test_log_filter_by_agent_multiple_matches() {
        let mut log = CollaborationLog::new();
        let req1 = DelegationRequest::new("alice", "bob", "task1");
        let req2 = DelegationRequest::new("alice", "charlie", "task2");
        log.record_delegation(&req1);
        log.record_delegation(&req2);
        let alice_entries = log.filter_by_agent("alice");
        assert_eq!(alice_entries.len(), 2);
    }

    #[test]
    fn test_log_to_json() {
        let mut log = CollaborationLog::new();
        let req = DelegationRequest::new("a", "b", "t");
        log.record_delegation(&req);
        let j = log.to_json();
        assert_eq!(j["total_entries"], 1);
    }

    #[test]
    fn test_log_mixed_events() {
        let mut log = CollaborationLog::new();
        let req = DelegationRequest::new("a", "b", "t");
        log.record_delegation(&req);
        let resp = DelegationResponse::accept(&req.id);
        log.record_response(&resp);
        let vote = ConsensusVote::new("a", true, 0.9);
        log.record_vote("topic", &vote);
        assert_eq!(log.entries().len(), 3);
    }

    // -- Integration / cross-cutting tests --

    #[test]
    fn test_full_delegation_workflow() {
        let mut log = CollaborationLog::new();

        // Create and log a request.
        let req = DelegationRequest::new("planner", "worker", "build feature")
            .with_priority(TaskPriority::High)
            .with_context("spec", json!({"version": 2}));
        log.record_delegation(&req);

        // Accept.
        let accept = DelegationResponse::accept(&req.id);
        assert_eq!(accept.status, DelegationStatus::Accepted);
        assert!(!accept.status.is_terminal());
        log.record_response(&accept);

        // Complete.
        let done = DelegationResponse::complete(&req.id, json!("feature built"));
        assert!(done.status.is_terminal());
        log.record_response(&done);

        assert_eq!(log.entries().len(), 3);
        let planner_entries = log.filter_by_agent("planner");
        assert_eq!(planner_entries.len(), 1);
    }

    #[test]
    fn test_full_consensus_workflow() {
        let mut round = ConsensusRound::new("deploy v2?".into(), 3);
        let mut log = CollaborationLog::new();

        let voters = vec![
            ("alice", true, 0.95, Some("tests pass")),
            ("bob", true, 0.80, None),
            ("charlie", false, 0.60, Some("needs more testing")),
            ("dave", true, 0.70, None),
        ];

        for (name, decision, conf, reasoning) in voters {
            let mut vote = ConsensusVote::new(name, decision, conf);
            if let Some(r) = reasoning {
                vote = vote.with_reasoning(r);
            }
            log.record_vote(&round.topic, &vote);
            round.cast_vote(vote);
        }

        assert!(round.has_quorum());
        let result = round.result().unwrap();
        assert!(result.approved);
        assert_eq!(result.votes_for, 3);
        assert_eq!(result.votes_against, 1);
        assert!((round.participation_rate(5) - 0.8).abs() < f64::EPSILON);
        assert_eq!(log.entries().len(), 4);
    }

    #[test]
    fn test_workspace_with_delegation_context() {
        let mut ws = SharedWorkspace::new("project".into());
        ws.put("requirements".into(), json!(["fast", "reliable"]));

        let req = DelegationRequest::new("pm", "dev", "implement")
            .with_context("workspace_data", ws.to_json());

        let ctx = &req.context["workspace_data"];
        assert_eq!(ctx["name"], "project");
    }

    #[test]
    fn test_protocol_clone() {
        let p = CollaborationProtocol::Hierarchical {
            leader: "boss".into(),
        };
        let p2 = p.clone();
        assert_eq!(p, p2);
    }

    #[test]
    fn test_workspace_clone_independence() {
        let mut ws = SharedWorkspace::new("ws".into());
        ws.put("k".into(), json!(1));
        let ws2 = ws.clone();
        ws.put("k".into(), json!(2));
        assert_eq!(ws2.get("k"), Some(&json!(1)));
    }
}
