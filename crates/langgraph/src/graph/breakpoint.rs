//! Breakpoint system for human-in-the-loop graph execution.
//!
//! This module provides a flexible breakpoint mechanism that allows pausing
//! graph execution before or after specific nodes, conditionally, or at every
//! step. A [`BreakpointManager`] coordinates breakpoint registration, matching,
//! handler dispatch, and history tracking.

use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;

use async_trait::async_trait;
use serde_json::Value;

use crate::errors::Result;

// ---------------------------------------------------------------------------
// BreakpointType
// ---------------------------------------------------------------------------

/// Describes *when* a breakpoint should trigger.
pub enum BreakpointType {
    /// Pause **before** executing the named node.
    BeforeNode(String),
    /// Pause **after** executing the named node.
    AfterNode(String),
    /// Pause before the named node only when `condition` returns `true`.
    Conditional {
        node: String,
        condition: Arc<dyn Fn(&Value) -> bool + Send + Sync>,
    },
    /// Pause at **every** node.
    Always,
}

impl fmt::Debug for BreakpointType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BeforeNode(n) => f.debug_tuple("BeforeNode").field(n).finish(),
            Self::AfterNode(n) => f.debug_tuple("AfterNode").field(n).finish(),
            Self::Conditional { node, .. } => {
                f.debug_struct("Conditional").field("node", node).finish()
            }
            Self::Always => write!(f, "Always"),
        }
    }
}

impl BreakpointType {
    /// Return the node name this breakpoint is associated with, if any.
    pub fn node_name(&self) -> Option<&str> {
        match self {
            Self::BeforeNode(n) | Self::AfterNode(n) | Self::Conditional { node: n, .. } => {
                Some(n.as_str())
            }
            Self::Always => None,
        }
    }
}

// ---------------------------------------------------------------------------
// BreakpointState
// ---------------------------------------------------------------------------

/// Tracks the lifecycle state of a breakpoint hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakpointState {
    /// Execution is paused at this breakpoint.
    Paused,
    /// Execution has been resumed past this breakpoint.
    Resumed,
    /// This breakpoint was skipped (node was not executed).
    Skipped,
}

// ---------------------------------------------------------------------------
// BreakpointEvent
// ---------------------------------------------------------------------------

/// Record of a breakpoint being triggered during graph execution.
#[derive(Debug, Clone)]
pub struct BreakpointEvent {
    /// The type of breakpoint that fired.
    pub breakpoint_type: String,
    /// The node where the breakpoint was hit.
    pub node: String,
    /// The execution step number.
    pub step: usize,
    /// A snapshot of the graph state at the breakpoint.
    pub state: Value,
    /// When the breakpoint was triggered.
    pub timestamp: SystemTime,
}

impl BreakpointEvent {
    /// Create a new event from a [`BreakpointType`] reference.
    pub fn new(bp: &BreakpointType, node: &str, step: usize, state: &Value) -> Self {
        let breakpoint_type = match bp {
            BreakpointType::BeforeNode(_) => "BeforeNode".to_string(),
            BreakpointType::AfterNode(_) => "AfterNode".to_string(),
            BreakpointType::Conditional { .. } => "Conditional".to_string(),
            BreakpointType::Always => "Always".to_string(),
        };
        Self {
            breakpoint_type,
            node: node.to_string(),
            step,
            state: state.clone(),
            timestamp: SystemTime::now(),
        }
    }
}

// ---------------------------------------------------------------------------
// BreakpointAction
// ---------------------------------------------------------------------------

/// The action a handler instructs the runtime to take when a breakpoint fires.
#[derive(Debug, Clone)]
pub enum BreakpointAction {
    /// Resume execution normally.
    Continue,
    /// Resume execution with a modified state.
    ContinueWithState(Value),
    /// Skip the node entirely.
    Skip,
    /// Abort execution with the given reason.
    Abort(String),
}

// ---------------------------------------------------------------------------
// BreakpointHandler trait
// ---------------------------------------------------------------------------

/// Async handler invoked when a breakpoint fires.
#[async_trait]
pub trait BreakpointHandler: Send + Sync {
    /// Decide what action to take in response to a breakpoint event.
    async fn on_breakpoint(&self, event: &BreakpointEvent) -> Result<BreakpointAction>;
}

// ---------------------------------------------------------------------------
// Built-in handlers
// ---------------------------------------------------------------------------

/// A handler that always returns [`BreakpointAction::Continue`].
#[derive(Debug, Clone, Default)]
pub struct AutoApproveHandler;

#[async_trait]
impl BreakpointHandler for AutoApproveHandler {
    async fn on_breakpoint(&self, _event: &BreakpointEvent) -> Result<BreakpointAction> {
        Ok(BreakpointAction::Continue)
    }
}

/// A handler that logs the event (via `println!`) and returns [`BreakpointAction::Continue`].
#[derive(Debug, Clone, Default)]
pub struct LoggingBreakpointHandler;

#[async_trait]
impl BreakpointHandler for LoggingBreakpointHandler {
    async fn on_breakpoint(&self, event: &BreakpointEvent) -> Result<BreakpointAction> {
        println!(
            "[breakpoint] type={} node={} step={}",
            event.breakpoint_type, event.node, event.step
        );
        Ok(BreakpointAction::Continue)
    }
}

// ---------------------------------------------------------------------------
// BreakpointManager
// ---------------------------------------------------------------------------

/// Central coordinator for breakpoint registration, matching, dispatch, and
/// history tracking.
pub struct BreakpointManager {
    breakpoints: Vec<BreakpointType>,
    handler: Option<Arc<dyn BreakpointHandler>>,
    history: Vec<BreakpointEvent>,
}

impl fmt::Debug for BreakpointManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BreakpointManager")
            .field("breakpoints", &self.breakpoints)
            .field("history_len", &self.history.len())
            .finish()
    }
}

impl Default for BreakpointManager {
    fn default() -> Self {
        Self::new()
    }
}

impl BreakpointManager {
    /// Create an empty manager with no breakpoints and no handler.
    pub fn new() -> Self {
        Self {
            breakpoints: Vec::new(),
            handler: None,
            history: Vec::new(),
        }
    }

    /// Register a breakpoint.
    pub fn add_breakpoint(&mut self, bp: BreakpointType) {
        self.breakpoints.push(bp);
    }

    /// Remove all breakpoints associated with the given node name.
    ///
    /// `Always` breakpoints are **not** removed by this method since they are
    /// not associated with a specific node.
    pub fn remove_breakpoint(&mut self, node: &str) {
        self.breakpoints.retain(|bp| bp.node_name() != Some(node));
    }

    /// Remove every registered breakpoint.
    pub fn clear_breakpoints(&mut self) {
        self.breakpoints.clear();
    }

    /// Set (or replace) the handler that will be invoked on breakpoint hits.
    pub fn set_handler(&mut self, handler: Arc<dyn BreakpointHandler>) {
        self.handler = Some(handler);
    }

    /// Check whether any registered breakpoint matches the current node and
    /// state. If a match is found, a [`BreakpointEvent`] is created, appended
    /// to the history, and returned.
    pub fn check_breakpoint(
        &mut self,
        node: &str,
        step: usize,
        state: &Value,
    ) -> Option<BreakpointEvent> {
        for bp in &self.breakpoints {
            let matched = match bp {
                BreakpointType::BeforeNode(n) | BreakpointType::AfterNode(n) => n == node,
                BreakpointType::Conditional { node: n, condition } => n == node && condition(state),
                BreakpointType::Always => true,
            };

            if matched {
                let event = BreakpointEvent::new(bp, node, step, state);
                self.history.push(event.clone());
                return Some(event);
            }
        }
        None
    }

    /// Invoke the registered handler for the given event.
    ///
    /// Returns [`BreakpointAction::Continue`] if no handler has been set.
    pub async fn handle_breakpoint(&self, event: &BreakpointEvent) -> Result<BreakpointAction> {
        match &self.handler {
            Some(h) => h.on_breakpoint(event).await,
            None => Ok(BreakpointAction::Continue),
        }
    }

    /// Return references to all registered breakpoints.
    pub fn list_breakpoints(&self) -> Vec<&BreakpointType> {
        self.breakpoints.iter().collect()
    }

    /// Return the full history of triggered breakpoint events.
    pub fn history(&self) -> &[BreakpointEvent] {
        &self.history
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::errors::LangGraphError;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // -- helpers -----------------------------------------------------------

    /// A handler that records how many times it was called and always continues.
    struct CountingHandler {
        count: AtomicUsize,
    }

    impl CountingHandler {
        fn new() -> Self {
            Self {
                count: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.count.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl BreakpointHandler for CountingHandler {
        async fn on_breakpoint(&self, _event: &BreakpointEvent) -> Result<BreakpointAction> {
            self.count.fetch_add(1, Ordering::SeqCst);
            Ok(BreakpointAction::Continue)
        }
    }

    /// A handler that returns a specific action.
    struct FixedActionHandler {
        action: BreakpointAction,
    }

    #[async_trait]
    impl BreakpointHandler for FixedActionHandler {
        async fn on_breakpoint(&self, _event: &BreakpointEvent) -> Result<BreakpointAction> {
            Ok(self.action.clone())
        }
    }

    /// A handler that aborts with an error.
    struct ErrorHandler;

    #[async_trait]
    impl BreakpointHandler for ErrorHandler {
        async fn on_breakpoint(&self, _event: &BreakpointEvent) -> Result<BreakpointAction> {
            Err(LangGraphError::Other("handler error".to_string()))
        }
    }

    // -- 1. BeforeNode matches correct node --------------------------------
    #[tokio::test]
    async fn test_before_node_matches() {
        let mut mgr = BreakpointManager::new();
        mgr.add_breakpoint(BreakpointType::BeforeNode("step1".to_string()));

        let state = json!({"count": 0});
        let event = mgr.check_breakpoint("step1", 0, &state);
        assert!(event.is_some());
        let event = event.unwrap();
        assert_eq!(event.node, "step1");
        assert_eq!(event.breakpoint_type, "BeforeNode");
    }

    // -- 2. BeforeNode does not match wrong node ---------------------------
    #[tokio::test]
    async fn test_before_node_no_match() {
        let mut mgr = BreakpointManager::new();
        mgr.add_breakpoint(BreakpointType::BeforeNode("step1".to_string()));

        assert!(mgr.check_breakpoint("step2", 0, &json!({})).is_none());
    }

    // -- 3. AfterNode matches correct node ---------------------------------
    #[tokio::test]
    async fn test_after_node_matches() {
        let mut mgr = BreakpointManager::new();
        mgr.add_breakpoint(BreakpointType::AfterNode("step2".to_string()));

        let event = mgr.check_breakpoint("step2", 1, &json!({"done": true}));
        assert!(event.is_some());
        assert_eq!(event.unwrap().breakpoint_type, "AfterNode");
    }

    // -- 4. Conditional breakpoint: condition true -------------------------
    #[tokio::test]
    async fn test_conditional_true() {
        let mut mgr = BreakpointManager::new();
        mgr.add_breakpoint(BreakpointType::Conditional {
            node: "check".to_string(),
            condition: Arc::new(|state: &Value| {
                state.get("risk").and_then(|v| v.as_bool()).unwrap_or(false)
            }),
        });

        let event = mgr.check_breakpoint("check", 2, &json!({"risk": true}));
        assert!(event.is_some());
        assert_eq!(event.unwrap().breakpoint_type, "Conditional");
    }

    // -- 5. Conditional breakpoint: condition false -------------------------
    #[tokio::test]
    async fn test_conditional_false() {
        let mut mgr = BreakpointManager::new();
        mgr.add_breakpoint(BreakpointType::Conditional {
            node: "check".to_string(),
            condition: Arc::new(|state: &Value| {
                state.get("risk").and_then(|v| v.as_bool()).unwrap_or(false)
            }),
        });

        assert!(mgr
            .check_breakpoint("check", 2, &json!({"risk": false}))
            .is_none());
    }

    // -- 6. Conditional breakpoint: wrong node -----------------------------
    #[tokio::test]
    async fn test_conditional_wrong_node() {
        let mut mgr = BreakpointManager::new();
        mgr.add_breakpoint(BreakpointType::Conditional {
            node: "check".to_string(),
            condition: Arc::new(|_| true),
        });

        assert!(mgr.check_breakpoint("other", 0, &json!({})).is_none());
    }

    // -- 7. Always breakpoint matches any node -----------------------------
    #[tokio::test]
    async fn test_always_matches_any_node() {
        let mut mgr = BreakpointManager::new();
        mgr.add_breakpoint(BreakpointType::Always);

        assert!(mgr.check_breakpoint("a", 0, &json!({})).is_some());
        assert!(mgr.check_breakpoint("b", 1, &json!({})).is_some());
        assert!(mgr.check_breakpoint("z", 99, &json!({})).is_some());
    }

    // -- 8. remove_breakpoint removes by node name -------------------------
    #[tokio::test]
    async fn test_remove_breakpoint() {
        let mut mgr = BreakpointManager::new();
        mgr.add_breakpoint(BreakpointType::BeforeNode("a".to_string()));
        mgr.add_breakpoint(BreakpointType::AfterNode("a".to_string()));
        mgr.add_breakpoint(BreakpointType::BeforeNode("b".to_string()));

        mgr.remove_breakpoint("a");
        assert_eq!(mgr.list_breakpoints().len(), 1);
        // Only "b" remains.
        assert!(mgr.check_breakpoint("a", 0, &json!({})).is_none());
        assert!(mgr.check_breakpoint("b", 0, &json!({})).is_some());
    }

    // -- 9. remove_breakpoint does not remove Always -----------------------
    #[tokio::test]
    async fn test_remove_does_not_affect_always() {
        let mut mgr = BreakpointManager::new();
        mgr.add_breakpoint(BreakpointType::Always);
        mgr.add_breakpoint(BreakpointType::BeforeNode("x".to_string()));

        mgr.remove_breakpoint("x");
        // Always should still be there.
        assert_eq!(mgr.list_breakpoints().len(), 1);
        assert!(mgr.check_breakpoint("anything", 0, &json!({})).is_some());
    }

    // -- 10. clear_breakpoints empties everything --------------------------
    #[tokio::test]
    async fn test_clear_breakpoints() {
        let mut mgr = BreakpointManager::new();
        mgr.add_breakpoint(BreakpointType::Always);
        mgr.add_breakpoint(BreakpointType::BeforeNode("a".to_string()));
        mgr.add_breakpoint(BreakpointType::AfterNode("b".to_string()));

        mgr.clear_breakpoints();
        assert!(mgr.list_breakpoints().is_empty());
        assert!(mgr.check_breakpoint("a", 0, &json!({})).is_none());
    }

    // -- 11. AutoApproveHandler returns Continue ---------------------------
    #[tokio::test]
    async fn test_auto_approve_handler() {
        let handler = AutoApproveHandler;
        let event = BreakpointEvent::new(&BreakpointType::Always, "node", 0, &json!({}));
        let action = handler.on_breakpoint(&event).await.unwrap();
        assert!(matches!(action, BreakpointAction::Continue));
    }

    // -- 12. LoggingBreakpointHandler returns Continue ---------------------
    #[tokio::test]
    async fn test_logging_handler() {
        let handler = LoggingBreakpointHandler;
        let event = BreakpointEvent::new(
            &BreakpointType::BeforeNode("n".to_string()),
            "n",
            5,
            &json!({"x": 1}),
        );
        let action = handler.on_breakpoint(&event).await.unwrap();
        assert!(matches!(action, BreakpointAction::Continue));
    }

    // -- 13. handle_breakpoint delegates to handler ------------------------
    #[tokio::test]
    async fn test_handle_breakpoint_delegates() {
        let counter = Arc::new(CountingHandler::new());
        let mut mgr = BreakpointManager::new();
        mgr.set_handler(counter.clone());

        let event = BreakpointEvent::new(&BreakpointType::Always, "x", 0, &json!({}));
        let action = mgr.handle_breakpoint(&event).await.unwrap();
        assert!(matches!(action, BreakpointAction::Continue));
        assert_eq!(counter.calls(), 1);
    }

    // -- 14. handle_breakpoint with no handler returns Continue -------------
    #[tokio::test]
    async fn test_handle_breakpoint_no_handler() {
        let mgr = BreakpointManager::new();
        let event = BreakpointEvent::new(&BreakpointType::Always, "x", 0, &json!({}));
        let action = mgr.handle_breakpoint(&event).await.unwrap();
        assert!(matches!(action, BreakpointAction::Continue));
    }

    // -- 15. history records triggered breakpoints -------------------------
    #[tokio::test]
    async fn test_history_records_events() {
        let mut mgr = BreakpointManager::new();
        mgr.add_breakpoint(BreakpointType::Always);

        mgr.check_breakpoint("a", 0, &json!({"step": 0}));
        mgr.check_breakpoint("b", 1, &json!({"step": 1}));
        mgr.check_breakpoint("c", 2, &json!({"step": 2}));

        let history = mgr.history();
        assert_eq!(history.len(), 3);
        assert_eq!(history[0].node, "a");
        assert_eq!(history[1].node, "b");
        assert_eq!(history[2].node, "c");
    }

    // -- 16. BreakpointAction::Abort carries reason ------------------------
    #[tokio::test]
    async fn test_abort_action() {
        let handler = FixedActionHandler {
            action: BreakpointAction::Abort("bad state".to_string()),
        };
        let mut mgr = BreakpointManager::new();
        mgr.set_handler(Arc::new(handler));

        let event = BreakpointEvent::new(&BreakpointType::Always, "x", 0, &json!({}));
        let action = mgr.handle_breakpoint(&event).await.unwrap();
        match action {
            BreakpointAction::Abort(reason) => assert_eq!(reason, "bad state"),
            other => panic!("Expected Abort, got {:?}", other),
        }
    }

    // -- 17. BreakpointAction::ContinueWithState carries value -------------
    #[tokio::test]
    async fn test_continue_with_state_action() {
        let new_state = json!({"override": true});
        let handler = FixedActionHandler {
            action: BreakpointAction::ContinueWithState(new_state.clone()),
        };
        let mut mgr = BreakpointManager::new();
        mgr.set_handler(Arc::new(handler));

        let event = BreakpointEvent::new(&BreakpointType::Always, "x", 0, &json!({}));
        let action = mgr.handle_breakpoint(&event).await.unwrap();
        match action {
            BreakpointAction::ContinueWithState(v) => assert_eq!(v, new_state),
            other => panic!("Expected ContinueWithState, got {:?}", other),
        }
    }

    // -- 18. BreakpointAction::Skip is returned correctly ------------------
    #[tokio::test]
    async fn test_skip_action() {
        let handler = FixedActionHandler {
            action: BreakpointAction::Skip,
        };
        let mut mgr = BreakpointManager::new();
        mgr.set_handler(Arc::new(handler));

        let event = BreakpointEvent::new(&BreakpointType::Always, "x", 0, &json!({}));
        let action = mgr.handle_breakpoint(&event).await.unwrap();
        assert!(matches!(action, BreakpointAction::Skip));
    }

    // -- 19. Handler returning error propagates correctly ------------------
    #[tokio::test]
    async fn test_handler_error_propagates() {
        let mut mgr = BreakpointManager::new();
        mgr.set_handler(Arc::new(ErrorHandler));

        let event = BreakpointEvent::new(&BreakpointType::Always, "x", 0, &json!({}));
        let result = mgr.handle_breakpoint(&event).await;
        assert!(result.is_err());
    }

    // -- 20. Event state snapshot captures current state -------------------
    #[tokio::test]
    async fn test_event_state_snapshot() {
        let mut mgr = BreakpointManager::new();
        mgr.add_breakpoint(BreakpointType::BeforeNode("n".to_string()));

        let state = json!({"count": 42, "items": [1, 2, 3]});
        let event = mgr.check_breakpoint("n", 7, &state).unwrap();
        assert_eq!(event.state, state);
        assert_eq!(event.step, 7);
    }

    // -- 21. list_breakpoints returns correct count ------------------------
    #[tokio::test]
    async fn test_list_breakpoints() {
        let mut mgr = BreakpointManager::new();
        assert!(mgr.list_breakpoints().is_empty());

        mgr.add_breakpoint(BreakpointType::BeforeNode("a".to_string()));
        mgr.add_breakpoint(BreakpointType::AfterNode("b".to_string()));
        mgr.add_breakpoint(BreakpointType::Always);
        assert_eq!(mgr.list_breakpoints().len(), 3);
    }

    // -- 22. BreakpointState enum values -----------------------------------
    #[test]
    fn test_breakpoint_state_enum() {
        let paused = BreakpointState::Paused;
        let resumed = BreakpointState::Resumed;
        let skipped = BreakpointState::Skipped;

        assert_eq!(paused, BreakpointState::Paused);
        assert_eq!(resumed, BreakpointState::Resumed);
        assert_eq!(skipped, BreakpointState::Skipped);
        assert_ne!(paused, resumed);
        assert_ne!(resumed, skipped);
    }
}
