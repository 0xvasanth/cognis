//! Async approval-gate middleware for tools that declare `requires_approval`.
//!
//! This middleware is the agent-level counterpart of
//! [`cognisgraph::graph::human_in_loop`]. When a tool marked
//! `requires_approval: true` is about to execute, the middleware:
//!
//! 1. Creates an [`ApprovalToken`] and stores a [`tokio::sync::oneshot`]
//!    resolver in its [`ApprovalRegistry`].
//! 2. Emits an [`AgentEventType::PendingApproval`] event with the token, tool
//!    name, and input.
//! 3. Awaits the external consumer calling
//!    [`ApprovalRegistry::resolve`] with the matching token.
//! 4. Emits an [`AgentEventType::ApprovalResolved`] event and returns a
//!    [`ToolGateDecision`]: `Continue` on approve, `Reject { observation }` on
//!    reject.
//!
//! Unlike the synchronous
//! [`cognis::tools::human::HumanApprovalTool`], this gate never blocks on
//! stdin — resolution happens asynchronously from a web UI, RPC handler, or
//! any other consumer of the [`EventBus`].
//!
//! # Example
//!
//! ```rust,ignore
//! use std::sync::Arc;
//! use cognisagent::events::EventBus;
//! use cognisagent::middleware::approval_gate::{
//!     ApprovalDecision, ApprovalGateMiddleware, ApprovalRegistry,
//! };
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let bus = EventBus::new();
//! let registry = Arc::new(ApprovalRegistry::new());
//! let gate = ApprovalGateMiddleware::new(registry.clone(), bus.clone());
//!
//! // Somewhere else (UI/RPC handler) resolve pending approvals:
//! // registry.resolve(token, ApprovalDecision::Approve)?;
//! # Ok(()) }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::oneshot;
use uuid::Uuid;

use crate::agent::DeepAgentError;
use crate::events::{AgentEventType, EventBus};

use super::{AgentState, Middleware, Result, ToolGateDecision};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A decision made by a human in response to an [`ApprovalRequest`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ApprovalDecision {
    /// Approve the pending tool invocation; it will execute normally.
    Approve,
    /// Reject the pending tool invocation with a reason. The reason is
    /// surfaced to the agent as part of the user-rejection observation.
    Reject { reason: String },
}

impl ApprovalDecision {
    /// Convenience constructor for a rejection with a reason.
    pub fn reject(reason: impl Into<String>) -> Self {
        Self::Reject {
            reason: reason.into(),
        }
    }

    /// Whether this decision approves the invocation.
    pub fn is_approved(&self) -> bool {
        matches!(self, Self::Approve)
    }
}

/// An opaque token identifying a pending approval. Consumers pass this back
/// to [`ApprovalRegistry::resolve`] to approve or reject the invocation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ApprovalToken(pub String);

impl ApprovalToken {
    /// Generate a fresh random token.
    pub fn new() -> Self {
        Self(Uuid::new_v4().to_string())
    }

    /// Construct a token from an existing identifier (for deterministic tests
    /// or when propagating tokens across systems).
    pub fn from_string(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// The underlying string representation.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl Default for ApprovalToken {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for ApprovalToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Metadata about a pending approval, suitable for surfacing to a UI.
#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    /// The unique token for this request.
    pub token: ApprovalToken,
    /// The name of the tool awaiting approval.
    pub tool_name: String,
    /// The JSON input that would be passed to the tool.
    pub tool_input: Value,
}

/// Error returned when resolving an unknown or already-resolved token.
#[derive(Debug, thiserror::Error)]
pub enum ApprovalError {
    /// No pending approval matches the supplied token.
    #[error("unknown approval token: {0}")]
    UnknownToken(String),
    /// The pending approval was dropped before a decision was received.
    #[error("approval resolver was dropped before a decision was made")]
    ResolverDropped,
    /// Waiting for resolution timed out.
    #[error("timed out waiting for approval decision after {0:?}")]
    Timeout(Duration),
}

// ---------------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------------

/// Thread-safe registry of pending approval requests.
///
/// The middleware registers pending approvals here when it intercepts a
/// gated tool; external consumers (UI handlers, RPC endpoints) resolve them
/// by token.
#[derive(Default)]
pub struct ApprovalRegistry {
    pending: Mutex<HashMap<ApprovalToken, PendingEntry>>,
}

struct PendingEntry {
    sender: oneshot::Sender<ApprovalDecision>,
    tool_name: String,
    tool_input: Value,
}

impl fmt::Debug for ApprovalRegistry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pending = self.pending.lock().unwrap();
        f.debug_struct("ApprovalRegistry")
            .field("pending_count", &pending.len())
            .finish()
    }
}

impl ApprovalRegistry {
    /// Create a new empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new pending approval, returning the token and a receiver
    /// the caller should `.await` to get the decision.
    pub fn register(
        &self,
        tool_name: impl Into<String>,
        tool_input: Value,
    ) -> (ApprovalToken, oneshot::Receiver<ApprovalDecision>) {
        let token = ApprovalToken::new();
        let (tx, rx) = oneshot::channel();
        let entry = PendingEntry {
            sender: tx,
            tool_name: tool_name.into(),
            tool_input,
        };
        self.pending.lock().unwrap().insert(token.clone(), entry);
        (token, rx)
    }

    /// Resolve a pending approval. Returns `UnknownToken` if no matching
    /// pending approval exists (already resolved or never registered).
    pub fn resolve(
        &self,
        token: &ApprovalToken,
        decision: ApprovalDecision,
    ) -> std::result::Result<(), ApprovalError> {
        let entry = self
            .pending
            .lock()
            .unwrap()
            .remove(token)
            .ok_or_else(|| ApprovalError::UnknownToken(token.as_str().to_string()))?;
        // The receiver may have been dropped (e.g. the middleware was cancelled);
        // surface that as an error rather than silently succeeding.
        entry
            .sender
            .send(decision)
            .map_err(|_| ApprovalError::ResolverDropped)
    }

    /// List all currently pending approvals.
    pub fn list_pending(&self) -> Vec<ApprovalRequest> {
        self.pending
            .lock()
            .unwrap()
            .iter()
            .map(|(token, entry)| ApprovalRequest {
                token: token.clone(),
                tool_name: entry.tool_name.clone(),
                tool_input: entry.tool_input.clone(),
            })
            .collect()
    }

    /// Number of pending approvals.
    pub fn pending_count(&self) -> usize {
        self.pending.lock().unwrap().len()
    }

    /// Whether a given token corresponds to a still-pending approval.
    pub fn is_pending(&self, token: &ApprovalToken) -> bool {
        self.pending.lock().unwrap().contains_key(token)
    }

    /// Drop all pending approvals. Any receivers awaiting resolution will
    /// observe [`ApprovalError::ResolverDropped`].
    pub fn clear(&self) {
        self.pending.lock().unwrap().clear();
    }
}

// ---------------------------------------------------------------------------
// Middleware
// ---------------------------------------------------------------------------

/// Predicate selecting which tool invocations require approval.
///
/// By default the middleware consults a static set of tool names; callers can
/// supply a custom predicate for more elaborate policy (e.g. consult the
/// [`cognis_core::tools::base::BaseTool::requires_approval`] flag via a tool
/// registry).
pub type GatePredicate = Arc<dyn Fn(&str, &Value) -> bool + Send + Sync>;

/// Middleware that gates configured tools behind asynchronous human approval.
pub struct ApprovalGateMiddleware {
    registry: Arc<ApprovalRegistry>,
    event_bus: EventBus,
    predicate: GatePredicate,
    /// Optional upper bound on how long to wait for a decision. `None` means
    /// wait forever (until the registry is cleared).
    timeout: Option<Duration>,
}

impl fmt::Debug for ApprovalGateMiddleware {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ApprovalGateMiddleware")
            .field("registry", &self.registry)
            .field("event_bus", &self.event_bus)
            .field("timeout", &self.timeout)
            .finish()
    }
}

impl ApprovalGateMiddleware {
    /// Build a middleware that gates every tool invocation. Mostly useful for
    /// testing; production use should prefer [`with_tool_names`](Self::with_tool_names)
    /// or [`with_predicate`](Self::with_predicate).
    pub fn new(registry: Arc<ApprovalRegistry>, event_bus: EventBus) -> Self {
        Self {
            registry,
            event_bus,
            predicate: Arc::new(|_name, _input| true),
            timeout: None,
        }
    }

    /// Gate only the listed tool names.
    pub fn with_tool_names(
        registry: Arc<ApprovalRegistry>,
        event_bus: EventBus,
        names: impl IntoIterator<Item = impl Into<String>>,
    ) -> Self {
        let gated: std::collections::HashSet<String> = names.into_iter().map(Into::into).collect();
        let predicate: GatePredicate = Arc::new(move |name, _input| gated.contains(name));
        Self {
            registry,
            event_bus,
            predicate,
            timeout: None,
        }
    }

    /// Gate tools selected by a custom predicate.
    pub fn with_predicate(
        registry: Arc<ApprovalRegistry>,
        event_bus: EventBus,
        predicate: GatePredicate,
    ) -> Self {
        Self {
            registry,
            event_bus,
            predicate,
            timeout: None,
        }
    }

    /// Set an upper bound on how long to wait for a decision. On timeout the
    /// gate rejects with a timeout observation.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Shared access to the backing registry.
    pub fn registry(&self) -> Arc<ApprovalRegistry> {
        self.registry.clone()
    }

    /// The event bus this middleware emits to.
    pub fn event_bus(&self) -> EventBus {
        self.event_bus.clone()
    }

    async fn wait_for_decision(
        &self,
        rx: oneshot::Receiver<ApprovalDecision>,
    ) -> std::result::Result<ApprovalDecision, ApprovalError> {
        match self.timeout {
            Some(duration) => match tokio::time::timeout(duration, rx).await {
                Ok(Ok(decision)) => Ok(decision),
                Ok(Err(_)) => Err(ApprovalError::ResolverDropped),
                Err(_) => Err(ApprovalError::Timeout(duration)),
            },
            None => rx.await.map_err(|_| ApprovalError::ResolverDropped),
        }
    }
}

#[async_trait]
impl Middleware for ApprovalGateMiddleware {
    fn name(&self) -> &str {
        "approval_gate"
    }

    async fn gate_tool(
        &self,
        _state: &mut AgentState,
        tool_name: &str,
        tool_input: &Value,
    ) -> Result<ToolGateDecision> {
        if !(self.predicate)(tool_name, tool_input) {
            return Ok(ToolGateDecision::Continue);
        }

        let (token, rx) = self.registry.register(tool_name, tool_input.clone());

        // Emit before awaiting so consumers see the pending approval even if
        // the decision is delivered synchronously from the same task.
        self.event_bus
            .emit(AgentEventType::PendingApproval {
                token: token.as_str().to_string(),
                tool: tool_name.to_string(),
                input: tool_input.clone(),
            })
            .await
            .map_err(|e| DeepAgentError::Other(format!("event emit failed: {}", e)))?;

        let decision = match self.wait_for_decision(rx).await {
            Ok(d) => d,
            Err(ApprovalError::Timeout(d)) => {
                // Cancel the pending entry so the registry stays clean.
                let _ = self.registry.pending.lock().unwrap().remove(&token);
                ApprovalDecision::reject(format!("approval timed out after {:?}", d))
            }
            Err(ApprovalError::ResolverDropped) => {
                ApprovalDecision::reject("approval resolver was dropped before a decision was made")
            }
            Err(ApprovalError::UnknownToken(_)) => {
                ApprovalDecision::reject("approval registry lost track of this token")
            }
        };

        let (approved, reason) = match &decision {
            ApprovalDecision::Approve => (true, None),
            ApprovalDecision::Reject { reason } => (false, Some(reason.clone())),
        };
        self.event_bus
            .emit(AgentEventType::ApprovalResolved {
                token: token.as_str().to_string(),
                tool: tool_name.to_string(),
                approved,
                reason: reason.clone(),
            })
            .await
            .map_err(|e| DeepAgentError::Other(format!("event emit failed: {}", e)))?;

        Ok(match decision {
            ApprovalDecision::Approve => ToolGateDecision::Continue,
            ApprovalDecision::Reject { reason } => ToolGateDecision::Reject {
                observation: format!("Action denied by human: {}", reason),
            },
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::{AgentEvent, EventHandler};
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct RecordingHandler {
        events: Mutex<Vec<AgentEventType>>,
        count: AtomicUsize,
    }

    impl RecordingHandler {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                events: Mutex::new(Vec::new()),
                count: AtomicUsize::new(0),
            })
        }

        fn events(&self) -> Vec<AgentEventType> {
            self.events.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl EventHandler for RecordingHandler {
        async fn handle(
            &self,
            event: &AgentEvent,
        ) -> std::result::Result<(), Box<dyn std::error::Error + Send + Sync>> {
            self.events.lock().unwrap().push(event.event_type.clone());
            self.count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    fn setup() -> (
        Arc<ApprovalRegistry>,
        EventBus,
        Arc<RecordingHandler>,
        ApprovalGateMiddleware,
    ) {
        let registry = Arc::new(ApprovalRegistry::new());
        let bus = EventBus::new();
        let handler = RecordingHandler::new();
        bus.subscribe(handler.clone());
        let gate = ApprovalGateMiddleware::new(registry.clone(), bus.clone());
        (registry, bus, handler, gate)
    }

    #[tokio::test]
    async fn approve_allows_execution() {
        let (registry, _bus, handler, gate) = setup();
        let mut state = Value::Null;

        let gate_fut = async {
            gate.gate_tool(&mut state, "shell", &json!({"cmd": "ls"}))
                .await
                .unwrap()
        };

        let resolve_fut = async {
            // Spin briefly until the middleware has registered the pending
            // approval, then resolve it.
            loop {
                if let Some(req) = registry.list_pending().into_iter().next() {
                    registry
                        .resolve(&req.token, ApprovalDecision::Approve)
                        .unwrap();
                    break;
                }
                tokio::task::yield_now().await;
            }
        };

        let (decision, _) = tokio::join!(gate_fut, resolve_fut);
        assert!(matches!(decision, ToolGateDecision::Continue));
        assert_eq!(registry.pending_count(), 0);

        let events = handler.events();
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEventType::PendingApproval { .. })));
        assert!(events
            .iter()
            .any(|e| matches!(e, AgentEventType::ApprovalResolved { approved: true, .. })));
    }

    #[tokio::test]
    async fn reject_short_circuits_with_observation() {
        let (registry, _bus, handler, gate) = setup();
        let mut state = Value::Null;

        let gate_fut = async {
            gate.gate_tool(&mut state, "shell", &json!({"cmd": "rm -rf /"}))
                .await
                .unwrap()
        };

        let resolve_fut = async {
            loop {
                if let Some(req) = registry.list_pending().into_iter().next() {
                    registry
                        .resolve(&req.token, ApprovalDecision::reject("destructive"))
                        .unwrap();
                    break;
                }
                tokio::task::yield_now().await;
            }
        };

        let (decision, _) = tokio::join!(gate_fut, resolve_fut);
        match decision {
            ToolGateDecision::Reject { observation } => {
                assert!(observation.contains("Action denied by human"));
                assert!(observation.contains("destructive"));
            }
            other => panic!("expected Reject, got {:?}", other),
        }

        let resolved = handler
            .events()
            .into_iter()
            .find(|e| matches!(e, AgentEventType::ApprovalResolved { .. }))
            .expect("ApprovalResolved emitted");
        match resolved {
            AgentEventType::ApprovalResolved {
                approved, reason, ..
            } => {
                assert!(!approved);
                assert_eq!(reason.as_deref(), Some("destructive"));
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn predicate_skips_ungated_tools() {
        let registry = Arc::new(ApprovalRegistry::new());
        let bus = EventBus::new();
        let gate = ApprovalGateMiddleware::with_tool_names(registry.clone(), bus, ["shell"]);

        let decision = gate
            .gate_tool(&mut Value::Null, "calculator", &json!({"a": 1}))
            .await
            .unwrap();
        assert!(matches!(decision, ToolGateDecision::Continue));
        assert_eq!(registry.pending_count(), 0);
    }

    #[tokio::test]
    async fn custom_predicate_gates_by_input() {
        let registry = Arc::new(ApprovalRegistry::new());
        let bus = EventBus::new();
        let predicate: GatePredicate = Arc::new(|_name, input| {
            input
                .get("destructive")
                .and_then(|v| v.as_bool())
                .unwrap_or(false)
        });
        let gate = ApprovalGateMiddleware::with_predicate(registry.clone(), bus, predicate);

        let benign = gate
            .gate_tool(&mut Value::Null, "shell", &json!({"destructive": false}))
            .await
            .unwrap();
        assert!(matches!(benign, ToolGateDecision::Continue));
        assert_eq!(registry.pending_count(), 0);
    }

    #[tokio::test]
    async fn timeout_rejects_with_timeout_message() {
        let registry = Arc::new(ApprovalRegistry::new());
        let bus = EventBus::new();
        let gate = ApprovalGateMiddleware::new(registry.clone(), bus)
            .with_timeout(Duration::from_millis(20));

        let decision = gate
            .gate_tool(&mut Value::Null, "shell", &json!({}))
            .await
            .unwrap();
        match decision {
            ToolGateDecision::Reject { observation } => {
                assert!(observation.to_lowercase().contains("timed out"));
            }
            other => panic!("expected Reject, got {:?}", other),
        }
        // Registry should be cleaned up after timeout.
        assert_eq!(registry.pending_count(), 0);
    }

    #[tokio::test]
    async fn resolve_unknown_token_errors() {
        let registry = ApprovalRegistry::new();
        let err = registry
            .resolve(
                &ApprovalToken::from_string("nope"),
                ApprovalDecision::Approve,
            )
            .unwrap_err();
        match err {
            ApprovalError::UnknownToken(t) => assert_eq!(t, "nope"),
            other => panic!("expected UnknownToken, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn register_returns_working_token() {
        let registry = ApprovalRegistry::new();
        let (token, rx) = registry.register("shell", json!({}));
        assert!(registry.is_pending(&token));
        assert_eq!(registry.pending_count(), 1);

        registry.resolve(&token, ApprovalDecision::Approve).unwrap();
        let decision = rx.await.unwrap();
        assert!(decision.is_approved());
        assert_eq!(registry.pending_count(), 0);
    }

    #[test]
    fn approval_decision_helpers() {
        assert!(ApprovalDecision::Approve.is_approved());
        let rej = ApprovalDecision::reject("nope");
        assert!(!rej.is_approved());
        match rej {
            ApprovalDecision::Reject { reason } => assert_eq!(reason, "nope"),
            _ => unreachable!(),
        }
    }

    #[test]
    fn tool_gate_decision_is_rejected() {
        assert!(!ToolGateDecision::Continue.is_rejected());
        assert!(ToolGateDecision::Reject {
            observation: "x".into()
        }
        .is_rejected());
    }

    #[test]
    fn approval_token_roundtrip() {
        let t = ApprovalToken::from_string("abc-123");
        assert_eq!(t.as_str(), "abc-123");
        assert_eq!(t.to_string(), "abc-123");
    }
}
