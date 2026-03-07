//! Human-in-the-loop interrupt system for graph execution.
//!
//! This module provides a flexible interrupt mechanism that allows pausing graph
//! execution at designated points for human input before resuming. An
//! [`InterruptManager`] coordinates interrupt registration, handler dispatch,
//! and history tracking via [`InterruptLog`].
//!
//! # Built-in handlers
//!
//! - [`AutoResumeHandler`] — always returns [`ResumeAction::Continue`]
//! - [`DefaultOptionHandler`] — selects the first option if available, otherwise continues
//! - [`TimeoutHandler`] — continues after a duration or aborts on timeout

use std::collections::HashMap;
use std::fmt;
use std::time::{Duration, Instant};

use serde_json::{json, Value};

use crate::errors::Result;

// ---------------------------------------------------------------------------
// InterruptType
// ---------------------------------------------------------------------------

/// Describes *when* an interrupt should trigger during graph execution.
#[derive(Debug, Clone)]
pub enum InterruptType {
    /// Pause **before** executing the named node.
    BeforeNode(String),
    /// Pause **after** executing the named node.
    AfterNode(String),
    /// Pause at the named node only when the named condition is met.
    Conditional {
        node: String,
        condition: String,
    },
    /// A manually triggered interrupt (not tied to any node).
    Manual,
    /// An interrupt that fires after a specified duration has elapsed.
    Timeout(Duration),
}

impl fmt::Display for InterruptType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BeforeNode(n) => write!(f, "BeforeNode({})", n),
            Self::AfterNode(n) => write!(f, "AfterNode({})", n),
            Self::Conditional { node, condition } => {
                write!(f, "Conditional(node={}, condition={})", node, condition)
            }
            Self::Manual => write!(f, "Manual"),
            Self::Timeout(d) => write!(f, "Timeout({:?})", d),
        }
    }
}

impl InterruptType {
    /// Return the node name this interrupt is associated with, if any.
    pub fn node_name(&self) -> Option<&str> {
        match self {
            Self::BeforeNode(n) | Self::AfterNode(n) | Self::Conditional { node: n, .. } => {
                Some(n.as_str())
            }
            Self::Manual | Self::Timeout(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// InterruptPayload
// ---------------------------------------------------------------------------

/// Payload delivered when an interrupt fires, containing the interrupt context.
#[derive(Debug, Clone)]
pub struct InterruptPayload {
    /// The type of interrupt that was triggered.
    pub interrupt_type: InterruptType,
    /// A snapshot of the graph state at the interrupt point.
    pub state: Value,
    /// An optional human-readable message describing what input is needed.
    pub message: Option<String>,
    /// A list of options the human can choose from.
    pub options: Vec<String>,
    /// When the interrupt was triggered.
    pub timestamp: Instant,
}

impl InterruptPayload {
    /// Create a new `InterruptPayload` with the given type and state.
    pub fn new(interrupt_type: InterruptType, state: Value) -> Self {
        Self {
            interrupt_type,
            state,
            message: None,
            options: Vec::new(),
            timestamp: Instant::now(),
        }
    }

    /// Set a human-readable message on this payload (builder pattern).
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Set the list of selectable options on this payload (builder pattern).
    pub fn with_options(mut self, opts: Vec<String>) -> Self {
        self.options = opts;
        self
    }

    /// Serialize the payload to a JSON [`Value`].
    pub fn to_json(&self) -> Value {
        json!({
            "interrupt_type": self.interrupt_type.to_string(),
            "state": self.state,
            "message": self.message,
            "options": self.options,
        })
    }
}

// ---------------------------------------------------------------------------
// ResumeAction
// ---------------------------------------------------------------------------

/// Action returned by an [`InterruptHandler`] telling the runtime how to
/// proceed after an interrupt.
#[derive(Debug, Clone)]
pub enum ResumeAction {
    /// Continue execution normally.
    Continue,
    /// Continue execution with a modified state.
    ContinueWithState(Value),
    /// Retry the current node.
    Retry,
    /// Skip the current node.
    Skip,
    /// Abort execution with the given reason.
    Abort(String),
    /// Select the option at the given index from the payload's options list.
    SelectOption(usize),
}

impl ResumeAction {
    /// Returns `true` if this action terminates graph execution.
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Abort(_))
    }
}

// ---------------------------------------------------------------------------
// InterruptHandler trait
// ---------------------------------------------------------------------------

/// Trait for handlers that decide how to resume after an interrupt.
pub trait InterruptHandler: Send + Sync {
    /// Given an interrupt payload, decide what action to take.
    fn handle(&self, payload: &InterruptPayload) -> Result<ResumeAction>;
}

// ---------------------------------------------------------------------------
// Built-in: AutoResumeHandler
// ---------------------------------------------------------------------------

/// A handler that always returns [`ResumeAction::Continue`].
#[derive(Debug, Clone, Default)]
pub struct AutoResumeHandler;

impl InterruptHandler for AutoResumeHandler {
    fn handle(&self, _payload: &InterruptPayload) -> Result<ResumeAction> {
        Ok(ResumeAction::Continue)
    }
}

// ---------------------------------------------------------------------------
// Built-in: DefaultOptionHandler
// ---------------------------------------------------------------------------

/// A handler that selects the first option if available, otherwise continues.
#[derive(Debug, Clone, Default)]
pub struct DefaultOptionHandler;

impl InterruptHandler for DefaultOptionHandler {
    fn handle(&self, payload: &InterruptPayload) -> Result<ResumeAction> {
        if payload.options.is_empty() {
            Ok(ResumeAction::Continue)
        } else {
            Ok(ResumeAction::SelectOption(0))
        }
    }
}

// ---------------------------------------------------------------------------
// Built-in: TimeoutHandler
// ---------------------------------------------------------------------------

/// A handler that returns [`ResumeAction::Continue`] if the interrupt was
/// triggered within the timeout window, or [`ResumeAction::Abort`] if the
/// timeout has been exceeded.
#[derive(Debug, Clone)]
pub struct TimeoutHandler {
    timeout: Duration,
}

impl TimeoutHandler {
    /// Create a new `TimeoutHandler` with the given timeout duration.
    pub fn new(timeout: Duration) -> Self {
        Self { timeout }
    }
}

impl InterruptHandler for TimeoutHandler {
    fn handle(&self, payload: &InterruptPayload) -> Result<ResumeAction> {
        if payload.timestamp.elapsed() > self.timeout {
            Ok(ResumeAction::Abort("Interrupt timeout exceeded".to_string()))
        } else {
            Ok(ResumeAction::Continue)
        }
    }
}

// ---------------------------------------------------------------------------
// InterruptManager
// ---------------------------------------------------------------------------

/// Central coordinator for interrupt point registration, handler dispatch,
/// and per-node handler overrides.
pub struct InterruptManager {
    interrupts: Vec<InterruptType>,
    handler: Option<Box<dyn InterruptHandler>>,
    node_handlers: HashMap<String, Box<dyn InterruptHandler>>,
}

impl fmt::Debug for InterruptManager {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InterruptManager")
            .field("interrupt_count", &self.interrupts.len())
            .field("has_handler", &self.handler.is_some())
            .field("node_handlers", &self.node_handlers.len())
            .finish()
    }
}

impl Default for InterruptManager {
    fn default() -> Self {
        Self::new()
    }
}

impl InterruptManager {
    /// Create an empty manager with no interrupt points and no handler.
    pub fn new() -> Self {
        Self {
            interrupts: Vec::new(),
            handler: None,
            node_handlers: HashMap::new(),
        }
    }

    /// Register an interrupt point.
    pub fn add_interrupt(&mut self, interrupt_type: InterruptType) {
        self.interrupts.push(interrupt_type);
    }

    /// Set (or replace) the global handler invoked when an interrupt fires.
    pub fn set_handler(&mut self, handler: Box<dyn InterruptHandler>) {
        self.handler = Some(handler);
    }

    /// Set (or replace) a per-node handler that takes precedence over the
    /// global handler for the given node.
    pub fn set_node_handler(&mut self, node: &str, handler: Box<dyn InterruptHandler>) {
        self.node_handlers.insert(node.to_string(), handler);
    }

    /// Check whether execution should pause at the given node and phase.
    ///
    /// The `phase` argument should be `"before"` or `"after"` to distinguish
    /// between `BeforeNode` and `AfterNode` interrupt types. Returns `true`
    /// if any registered interrupt matches.
    pub fn should_interrupt(&self, node: &str, phase: &str) -> bool {
        self.interrupts.iter().any(|it| match it {
            InterruptType::BeforeNode(n) => n == node && phase == "before",
            InterruptType::AfterNode(n) => n == node && phase == "after",
            InterruptType::Conditional { node: n, .. } => n == node,
            InterruptType::Manual => false,
            InterruptType::Timeout(_) => false,
        })
    }

    /// Trigger an interrupt and get the resume action from the appropriate
    /// handler (per-node handler takes precedence over the global handler).
    ///
    /// Returns [`ResumeAction::Continue`] if no handler has been set.
    pub fn trigger(&self, payload: InterruptPayload) -> Result<ResumeAction> {
        // Check for a per-node handler first.
        if let Some(node) = payload.interrupt_type.node_name() {
            if let Some(handler) = self.node_handlers.get(node) {
                return handler.handle(&payload);
            }
        }

        // Fall back to the global handler.
        match &self.handler {
            Some(h) => h.handle(&payload),
            None => Ok(ResumeAction::Continue),
        }
    }

    /// Return the number of registered interrupt points.
    pub fn interrupt_count(&self) -> usize {
        self.interrupts.len()
    }

    /// Remove all registered interrupt points and handlers.
    pub fn clear(&mut self) {
        self.interrupts.clear();
        self.handler = None;
        self.node_handlers.clear();
    }

    /// Return references to all pending (registered) interrupt types.
    pub fn pending_interrupts(&self) -> Vec<&InterruptType> {
        self.interrupts.iter().collect()
    }
}

// ---------------------------------------------------------------------------
// InterruptLogEntry
// ---------------------------------------------------------------------------

/// A single entry in the interrupt log, recording an interrupt event and
/// the action that was taken.
#[derive(Debug, Clone)]
pub struct InterruptLogEntry {
    /// The type of interrupt that was triggered.
    pub interrupt_type: String,
    /// The resume action that was taken.
    pub action: String,
    /// When the interrupt was triggered.
    pub timestamp: Instant,
    /// The node associated with the interrupt, if any.
    pub node: Option<String>,
}

// ---------------------------------------------------------------------------
// InterruptLog
// ---------------------------------------------------------------------------

/// Records a history of interrupt events and their resolutions.
#[derive(Debug, Default)]
pub struct InterruptLog {
    entries: Vec<InterruptLogEntry>,
}

impl InterruptLog {
    /// Create a new, empty interrupt log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Record an interrupt event and the action that was taken.
    pub fn record(&mut self, payload: &InterruptPayload, action: &ResumeAction) {
        let entry = InterruptLogEntry {
            interrupt_type: payload.interrupt_type.to_string(),
            action: format!("{:?}", action),
            timestamp: payload.timestamp,
            node: payload.interrupt_type.node_name().map(|s| s.to_string()),
        };
        self.entries.push(entry);
    }

    /// Return references to all log entries.
    pub fn entries(&self) -> Vec<&InterruptLogEntry> {
        self.entries.iter().collect()
    }

    /// Return entries associated with the given node.
    pub fn entries_for_node(&self, node: &str) -> Vec<&InterruptLogEntry> {
        self.entries
            .iter()
            .filter(|e| e.node.as_deref() == Some(node))
            .collect()
    }

    /// Return the number of entries in the log.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all log entries.
    pub fn clear(&mut self) {
        self.entries.clear();
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

    // -----------------------------------------------------------------------
    // InterruptType tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_interrupt_type_before_node_display() {
        let it = InterruptType::BeforeNode("step1".to_string());
        assert_eq!(it.to_string(), "BeforeNode(step1)");
    }

    #[test]
    fn test_interrupt_type_after_node_display() {
        let it = InterruptType::AfterNode("step2".to_string());
        assert_eq!(it.to_string(), "AfterNode(step2)");
    }

    #[test]
    fn test_interrupt_type_conditional_display() {
        let it = InterruptType::Conditional {
            node: "check".to_string(),
            condition: "is_risky".to_string(),
        };
        assert_eq!(
            it.to_string(),
            "Conditional(node=check, condition=is_risky)"
        );
    }

    #[test]
    fn test_interrupt_type_manual_display() {
        let it = InterruptType::Manual;
        assert_eq!(it.to_string(), "Manual");
    }

    #[test]
    fn test_interrupt_type_timeout_display() {
        let it = InterruptType::Timeout(Duration::from_secs(30));
        assert_eq!(it.to_string(), "Timeout(30s)");
    }

    #[test]
    fn test_interrupt_type_node_name_before() {
        let it = InterruptType::BeforeNode("n1".to_string());
        assert_eq!(it.node_name(), Some("n1"));
    }

    #[test]
    fn test_interrupt_type_node_name_after() {
        let it = InterruptType::AfterNode("n2".to_string());
        assert_eq!(it.node_name(), Some("n2"));
    }

    #[test]
    fn test_interrupt_type_node_name_conditional() {
        let it = InterruptType::Conditional {
            node: "n3".to_string(),
            condition: "cond".to_string(),
        };
        assert_eq!(it.node_name(), Some("n3"));
    }

    #[test]
    fn test_interrupt_type_node_name_manual() {
        assert_eq!(InterruptType::Manual.node_name(), None);
    }

    #[test]
    fn test_interrupt_type_node_name_timeout() {
        let it = InterruptType::Timeout(Duration::from_secs(5));
        assert_eq!(it.node_name(), None);
    }

    // -----------------------------------------------------------------------
    // InterruptPayload tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_payload_new() {
        let payload = InterruptPayload::new(
            InterruptType::BeforeNode("step1".to_string()),
            json!({"count": 1}),
        );
        assert!(matches!(payload.interrupt_type, InterruptType::BeforeNode(_)));
        assert_eq!(payload.state, json!({"count": 1}));
        assert!(payload.message.is_none());
        assert!(payload.options.is_empty());
    }

    #[test]
    fn test_payload_with_message() {
        let payload = InterruptPayload::new(InterruptType::Manual, json!({}))
            .with_message("Please review");
        assert_eq!(payload.message, Some("Please review".to_string()));
    }

    #[test]
    fn test_payload_with_options() {
        let payload = InterruptPayload::new(InterruptType::Manual, json!({}))
            .with_options(vec!["approve".to_string(), "reject".to_string()]);
        assert_eq!(payload.options.len(), 2);
        assert_eq!(payload.options[0], "approve");
        assert_eq!(payload.options[1], "reject");
    }

    #[test]
    fn test_payload_to_json() {
        let payload = InterruptPayload::new(
            InterruptType::BeforeNode("n".to_string()),
            json!({"x": 42}),
        )
        .with_message("check this")
        .with_options(vec!["yes".to_string(), "no".to_string()]);

        let j = payload.to_json();
        assert_eq!(j["interrupt_type"], "BeforeNode(n)");
        assert_eq!(j["state"], json!({"x": 42}));
        assert_eq!(j["message"], "check this");
        assert_eq!(j["options"], json!(["yes", "no"]));
    }

    #[test]
    fn test_payload_to_json_no_message() {
        let payload = InterruptPayload::new(InterruptType::Manual, json!({}));
        let j = payload.to_json();
        assert!(j["message"].is_null());
        assert_eq!(j["options"], json!([]));
    }

    #[test]
    fn test_payload_builder_chain() {
        let payload = InterruptPayload::new(InterruptType::Manual, json!({"a": 1}))
            .with_message("msg")
            .with_options(vec!["opt1".to_string()]);
        assert_eq!(payload.message, Some("msg".to_string()));
        assert_eq!(payload.options.len(), 1);
        assert_eq!(payload.state, json!({"a": 1}));
    }

    // -----------------------------------------------------------------------
    // ResumeAction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resume_action_continue_not_terminal() {
        assert!(!ResumeAction::Continue.is_terminal());
    }

    #[test]
    fn test_resume_action_continue_with_state_not_terminal() {
        assert!(!ResumeAction::ContinueWithState(json!({})).is_terminal());
    }

    #[test]
    fn test_resume_action_retry_not_terminal() {
        assert!(!ResumeAction::Retry.is_terminal());
    }

    #[test]
    fn test_resume_action_skip_not_terminal() {
        assert!(!ResumeAction::Skip.is_terminal());
    }

    #[test]
    fn test_resume_action_abort_is_terminal() {
        assert!(ResumeAction::Abort("reason".to_string()).is_terminal());
    }

    #[test]
    fn test_resume_action_select_option_not_terminal() {
        assert!(!ResumeAction::SelectOption(0).is_terminal());
    }

    // -----------------------------------------------------------------------
    // AutoResumeHandler tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_auto_resume_handler() {
        let handler = AutoResumeHandler;
        let payload = InterruptPayload::new(InterruptType::Manual, json!({}));
        let action = handler.handle(&payload).unwrap();
        assert!(matches!(action, ResumeAction::Continue));
    }

    #[test]
    fn test_auto_resume_handler_with_options() {
        let handler = AutoResumeHandler;
        let payload = InterruptPayload::new(InterruptType::Manual, json!({}))
            .with_options(vec!["a".to_string()]);
        let action = handler.handle(&payload).unwrap();
        // AutoResume always returns Continue regardless of options
        assert!(matches!(action, ResumeAction::Continue));
    }

    // -----------------------------------------------------------------------
    // DefaultOptionHandler tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_option_handler_with_options() {
        let handler = DefaultOptionHandler;
        let payload = InterruptPayload::new(InterruptType::Manual, json!({}))
            .with_options(vec!["first".to_string(), "second".to_string()]);
        let action = handler.handle(&payload).unwrap();
        assert!(matches!(action, ResumeAction::SelectOption(0)));
    }

    #[test]
    fn test_default_option_handler_empty_options() {
        let handler = DefaultOptionHandler;
        let payload = InterruptPayload::new(InterruptType::Manual, json!({}));
        let action = handler.handle(&payload).unwrap();
        assert!(matches!(action, ResumeAction::Continue));
    }

    #[test]
    fn test_default_option_handler_single_option() {
        let handler = DefaultOptionHandler;
        let payload = InterruptPayload::new(InterruptType::Manual, json!({}))
            .with_options(vec!["only".to_string()]);
        let action = handler.handle(&payload).unwrap();
        assert!(matches!(action, ResumeAction::SelectOption(0)));
    }

    // -----------------------------------------------------------------------
    // TimeoutHandler tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_timeout_handler_within_timeout() {
        let handler = TimeoutHandler::new(Duration::from_secs(60));
        let payload = InterruptPayload::new(InterruptType::Manual, json!({}));
        // Payload was just created so timestamp is fresh — well within 60s.
        let action = handler.handle(&payload).unwrap();
        assert!(matches!(action, ResumeAction::Continue));
    }

    #[test]
    fn test_timeout_handler_exceeded() {
        let handler = TimeoutHandler::new(Duration::from_nanos(0));
        // Create a payload and immediately check — even 0ns will have elapsed.
        let payload = InterruptPayload::new(InterruptType::Manual, json!({}));
        // We need at least a tiny bit of elapsed time; spin briefly.
        std::thread::sleep(Duration::from_millis(1));
        let action = handler.handle(&payload).unwrap();
        assert!(matches!(action, ResumeAction::Abort(_)));
        if let ResumeAction::Abort(reason) = action {
            assert!(reason.contains("timeout"));
        }
    }

    // -----------------------------------------------------------------------
    // InterruptManager tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_manager_new_empty() {
        let mgr = InterruptManager::new();
        assert_eq!(mgr.interrupt_count(), 0);
        assert!(mgr.pending_interrupts().is_empty());
    }

    #[test]
    fn test_manager_add_interrupt() {
        let mut mgr = InterruptManager::new();
        mgr.add_interrupt(InterruptType::BeforeNode("a".to_string()));
        mgr.add_interrupt(InterruptType::AfterNode("b".to_string()));
        assert_eq!(mgr.interrupt_count(), 2);
    }

    #[test]
    fn test_manager_should_interrupt_before() {
        let mut mgr = InterruptManager::new();
        mgr.add_interrupt(InterruptType::BeforeNode("step1".to_string()));

        assert!(mgr.should_interrupt("step1", "before"));
        assert!(!mgr.should_interrupt("step1", "after"));
        assert!(!mgr.should_interrupt("other", "before"));
    }

    #[test]
    fn test_manager_should_interrupt_after() {
        let mut mgr = InterruptManager::new();
        mgr.add_interrupt(InterruptType::AfterNode("step2".to_string()));

        assert!(mgr.should_interrupt("step2", "after"));
        assert!(!mgr.should_interrupt("step2", "before"));
    }

    #[test]
    fn test_manager_should_interrupt_conditional() {
        let mut mgr = InterruptManager::new();
        mgr.add_interrupt(InterruptType::Conditional {
            node: "review".to_string(),
            condition: "needs_approval".to_string(),
        });

        // Conditional matches any phase for the node
        assert!(mgr.should_interrupt("review", "before"));
        assert!(mgr.should_interrupt("review", "after"));
        assert!(!mgr.should_interrupt("other", "before"));
    }

    #[test]
    fn test_manager_should_interrupt_manual_never_matches() {
        let mut mgr = InterruptManager::new();
        mgr.add_interrupt(InterruptType::Manual);

        assert!(!mgr.should_interrupt("any_node", "before"));
        assert!(!mgr.should_interrupt("any_node", "after"));
    }

    #[test]
    fn test_manager_trigger_no_handler() {
        let mgr = InterruptManager::new();
        let payload = InterruptPayload::new(InterruptType::Manual, json!({}));
        let action = mgr.trigger(payload).unwrap();
        assert!(matches!(action, ResumeAction::Continue));
    }

    #[test]
    fn test_manager_trigger_global_handler() {
        let mut mgr = InterruptManager::new();
        mgr.set_handler(Box::new(AutoResumeHandler));

        let payload = InterruptPayload::new(InterruptType::Manual, json!({}));
        let action = mgr.trigger(payload).unwrap();
        assert!(matches!(action, ResumeAction::Continue));
    }

    #[test]
    fn test_manager_trigger_per_node_handler_override() {
        struct AbortHandler;
        impl InterruptHandler for AbortHandler {
            fn handle(&self, _payload: &InterruptPayload) -> Result<ResumeAction> {
                Ok(ResumeAction::Abort("node override".to_string()))
            }
        }

        let mut mgr = InterruptManager::new();
        mgr.set_handler(Box::new(AutoResumeHandler));
        mgr.set_node_handler("critical", Box::new(AbortHandler));

        // For the "critical" node, the per-node handler should take precedence.
        let payload = InterruptPayload::new(
            InterruptType::BeforeNode("critical".to_string()),
            json!({}),
        );
        let action = mgr.trigger(payload).unwrap();
        assert!(matches!(action, ResumeAction::Abort(_)));

        // For a different node, the global handler should be used.
        let payload2 = InterruptPayload::new(
            InterruptType::BeforeNode("normal".to_string()),
            json!({}),
        );
        let action2 = mgr.trigger(payload2).unwrap();
        assert!(matches!(action2, ResumeAction::Continue));
    }

    #[test]
    fn test_manager_clear() {
        let mut mgr = InterruptManager::new();
        mgr.add_interrupt(InterruptType::BeforeNode("a".to_string()));
        mgr.add_interrupt(InterruptType::AfterNode("b".to_string()));
        mgr.set_handler(Box::new(AutoResumeHandler));
        mgr.set_node_handler("a", Box::new(AutoResumeHandler));

        mgr.clear();
        assert_eq!(mgr.interrupt_count(), 0);
        assert!(mgr.pending_interrupts().is_empty());
    }

    #[test]
    fn test_manager_pending_interrupts() {
        let mut mgr = InterruptManager::new();
        mgr.add_interrupt(InterruptType::BeforeNode("a".to_string()));
        mgr.add_interrupt(InterruptType::Manual);
        mgr.add_interrupt(InterruptType::Timeout(Duration::from_secs(10)));

        let pending = mgr.pending_interrupts();
        assert_eq!(pending.len(), 3);
    }

    #[test]
    fn test_manager_default() {
        let mgr = InterruptManager::default();
        assert_eq!(mgr.interrupt_count(), 0);
    }

    #[test]
    fn test_manager_handler_error_propagates() {
        struct ErrorHandler;
        impl InterruptHandler for ErrorHandler {
            fn handle(&self, _payload: &InterruptPayload) -> Result<ResumeAction> {
                Err(LangGraphError::Other("handler failed".to_string()))
            }
        }

        let mut mgr = InterruptManager::new();
        mgr.set_handler(Box::new(ErrorHandler));

        let payload = InterruptPayload::new(InterruptType::Manual, json!({}));
        let result = mgr.trigger(payload);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("handler failed"));
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
    fn test_log_record_and_entries() {
        let mut log = InterruptLog::new();
        let payload = InterruptPayload::new(
            InterruptType::BeforeNode("step1".to_string()),
            json!({"x": 1}),
        );
        let action = ResumeAction::Continue;
        log.record(&payload, &action);

        assert_eq!(log.len(), 1);
        assert!(!log.is_empty());

        let entries = log.entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].node, Some("step1".to_string()));
        assert!(entries[0].interrupt_type.contains("BeforeNode"));
        assert!(entries[0].action.contains("Continue"));
    }

    #[test]
    fn test_log_entries_for_node() {
        let mut log = InterruptLog::new();

        let p1 = InterruptPayload::new(
            InterruptType::BeforeNode("nodeA".to_string()),
            json!({}),
        );
        log.record(&p1, &ResumeAction::Continue);

        let p2 = InterruptPayload::new(
            InterruptType::AfterNode("nodeB".to_string()),
            json!({}),
        );
        log.record(&p2, &ResumeAction::Skip);

        let p3 = InterruptPayload::new(
            InterruptType::BeforeNode("nodeA".to_string()),
            json!({}),
        );
        log.record(&p3, &ResumeAction::Retry);

        assert_eq!(log.entries_for_node("nodeA").len(), 2);
        assert_eq!(log.entries_for_node("nodeB").len(), 1);
        assert_eq!(log.entries_for_node("nodeC").len(), 0);
    }

    #[test]
    fn test_log_entries_for_manual_interrupt() {
        let mut log = InterruptLog::new();
        let payload = InterruptPayload::new(InterruptType::Manual, json!({}));
        log.record(&payload, &ResumeAction::Continue);

        // Manual interrupts have no node, so entries_for_node should not find them.
        assert_eq!(log.entries_for_node("anything").len(), 0);
        assert_eq!(log.len(), 1);
        assert!(log.entries()[0].node.is_none());
    }

    #[test]
    fn test_log_clear() {
        let mut log = InterruptLog::new();
        let payload = InterruptPayload::new(InterruptType::Manual, json!({}));
        log.record(&payload, &ResumeAction::Continue);
        log.record(&payload, &ResumeAction::Skip);
        assert_eq!(log.len(), 2);

        log.clear();
        assert_eq!(log.len(), 0);
        assert!(log.is_empty());
    }

    #[test]
    fn test_log_multiple_actions_recorded() {
        let mut log = InterruptLog::new();

        let actions: Vec<ResumeAction> = vec![
            ResumeAction::Continue,
            ResumeAction::ContinueWithState(json!({"new": true})),
            ResumeAction::Retry,
            ResumeAction::Skip,
            ResumeAction::Abort("reason".to_string()),
            ResumeAction::SelectOption(2),
        ];

        for action in &actions {
            let payload = InterruptPayload::new(
                InterruptType::BeforeNode("n".to_string()),
                json!({}),
            );
            log.record(&payload, action);
        }

        assert_eq!(log.len(), 6);
        let entries = log.entries();
        assert!(entries[0].action.contains("Continue"));
        assert!(entries[1].action.contains("ContinueWithState"));
        assert!(entries[2].action.contains("Retry"));
        assert!(entries[3].action.contains("Skip"));
        assert!(entries[4].action.contains("Abort"));
        assert!(entries[5].action.contains("SelectOption"));
    }

    // -----------------------------------------------------------------------
    // Integration: manager + log
    // -----------------------------------------------------------------------

    #[test]
    fn test_manager_and_log_integration() {
        let mut mgr = InterruptManager::new();
        mgr.add_interrupt(InterruptType::BeforeNode("review".to_string()));
        mgr.set_handler(Box::new(DefaultOptionHandler));

        let mut log = InterruptLog::new();

        // Simulate an interrupt with options
        let payload = InterruptPayload::new(
            InterruptType::BeforeNode("review".to_string()),
            json!({"draft": true}),
        )
        .with_message("Review this draft")
        .with_options(vec!["approve".to_string(), "reject".to_string()]);

        let action = mgr.trigger(payload.clone()).unwrap();
        log.record(&payload, &action);

        assert!(matches!(action, ResumeAction::SelectOption(0)));
        assert_eq!(log.len(), 1);
        assert_eq!(log.entries_for_node("review").len(), 1);
    }

    #[test]
    fn test_multiple_interrupt_points() {
        let mut mgr = InterruptManager::new();
        mgr.add_interrupt(InterruptType::BeforeNode("step1".to_string()));
        mgr.add_interrupt(InterruptType::AfterNode("step1".to_string()));
        mgr.add_interrupt(InterruptType::BeforeNode("step2".to_string()));

        assert!(mgr.should_interrupt("step1", "before"));
        assert!(mgr.should_interrupt("step1", "after"));
        assert!(mgr.should_interrupt("step2", "before"));
        assert!(!mgr.should_interrupt("step2", "after"));
        assert!(!mgr.should_interrupt("step3", "before"));
    }

    #[test]
    fn test_log_default() {
        let log = InterruptLog::default();
        assert!(log.is_empty());
    }
}
