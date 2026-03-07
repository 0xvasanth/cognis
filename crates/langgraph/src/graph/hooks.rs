//! Execution hooks / lifecycle system for intercepting graph execution.
//!
//! This module provides a flexible hook system that allows users to intercept
//! graph execution at various lifecycle points — before/after nodes, edges,
//! and the full graph — and optionally modify state or control flow.
//!
//! # Built-in hooks
//!
//! - [`LoggingHook`] — logs all lifecycle phases to stderr
//! - [`StateValidationHook`] — validates that state contains required keys
//! - [`TimingHook`] — tracks execution time per node
//! - [`StateSnapshotHook`] — captures state snapshots at each node for debugging

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde_json::Value;
use tokio::sync::Mutex;

use crate::errors::{LangGraphError, Result};

// ---------------------------------------------------------------------------
// HookPhase
// ---------------------------------------------------------------------------

/// Lifecycle phase at which a hook fires.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum HookPhase {
    /// Before the graph starts executing.
    BeforeGraph,
    /// After the graph finishes executing.
    AfterGraph,
    /// Before a node executes.
    BeforeNode,
    /// After a node executes.
    AfterNode,
    /// Before an edge is traversed.
    BeforeEdge,
    /// After an edge is traversed.
    AfterEdge,
    /// When an error occurs during execution.
    OnError,
}

// ---------------------------------------------------------------------------
// HookContext
// ---------------------------------------------------------------------------

/// Context passed to hooks at each lifecycle event.
#[derive(Debug, Clone)]
pub struct HookContext {
    /// The current lifecycle phase.
    pub phase: HookPhase,
    /// The node name, if applicable (present for `BeforeNode`, `AfterNode`, `OnError`).
    pub node: Option<String>,
    /// The edge (from, to), if applicable (present for `BeforeEdge`, `AfterEdge`).
    pub edge: Option<(String, String)>,
    /// The current graph state.
    pub state: Value,
    /// The current execution step number.
    pub step: usize,
    /// Arbitrary metadata associated with this event.
    pub metadata: HashMap<String, Value>,
}

impl HookContext {
    /// Create a new `HookContext`.
    pub fn new(phase: HookPhase, state: Value, step: usize) -> Self {
        Self {
            phase,
            node: None,
            edge: None,
            state,
            step,
            metadata: HashMap::new(),
        }
    }

    /// Set the node name.
    pub fn with_node(mut self, node: impl Into<String>) -> Self {
        self.node = Some(node.into());
        self
    }

    /// Set the edge.
    pub fn with_edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        self.edge = Some((from.into(), to.into()));
        self
    }

    /// Add a metadata entry.
    pub fn with_metadata(mut self, key: impl Into<String>, value: Value) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }
}

// ---------------------------------------------------------------------------
// HookAction
// ---------------------------------------------------------------------------

/// Action returned by a hook to control execution flow.
#[derive(Debug, Clone)]
pub enum HookAction {
    /// Continue execution normally.
    Continue,
    /// Continue execution with modified state.
    ModifyState(Value),
    /// Skip the current node (only valid for `BeforeNode`).
    SkipNode,
    /// Abort graph execution with the given reason.
    Abort(String),
}

// ---------------------------------------------------------------------------
// ExecutionHook trait
// ---------------------------------------------------------------------------

/// Trait for hooks that intercept graph execution at lifecycle points.
#[async_trait]
pub trait ExecutionHook: Send + Sync {
    /// Called when a matching lifecycle event occurs.
    ///
    /// Return a [`HookAction`] to control execution flow.
    async fn on_event(&self, ctx: &HookContext) -> Result<HookAction>;

    /// Return the set of phases this hook listens to.
    ///
    /// Only events matching one of these phases will be dispatched to this hook.
    fn phases(&self) -> Vec<HookPhase>;
}

// ---------------------------------------------------------------------------
// HookRegistry
// ---------------------------------------------------------------------------

/// Registry that manages execution hooks and dispatches events.
#[derive(Default)]
pub struct HookRegistry {
    hooks: Vec<Arc<dyn ExecutionHook>>,
}

impl HookRegistry {
    /// Create a new empty `HookRegistry`.
    pub fn new() -> Self {
        Self { hooks: Vec::new() }
    }

    /// Register a hook.
    pub fn register(&mut self, hook: Arc<dyn ExecutionHook>) {
        self.hooks.push(hook);
    }

    /// Unregister the hook at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn unregister(&mut self, index: usize) {
        self.hooks.remove(index);
    }

    /// Return the number of registered hooks.
    pub fn len(&self) -> usize {
        self.hooks.len()
    }

    /// Return whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// Return all hooks that listen to the given phase.
    pub fn hooks_for_phase(&self, phase: &HookPhase) -> Vec<Arc<dyn ExecutionHook>> {
        self.hooks
            .iter()
            .filter(|h| h.phases().contains(phase))
            .cloned()
            .collect()
    }

    /// Dispatch an event to all matching hooks.
    ///
    /// Hooks are called in registration order. The first hook that returns
    /// a non-[`HookAction::Continue`] action short-circuits and that action
    /// is returned. If all hooks return `Continue`, `Continue` is returned.
    pub async fn dispatch(&self, ctx: &HookContext) -> Result<HookAction> {
        let matching = self.hooks_for_phase(&ctx.phase);
        for hook in matching {
            let action = hook.on_event(ctx).await?;
            match &action {
                HookAction::Continue => continue,
                _ => return Ok(action),
            }
        }
        Ok(HookAction::Continue)
    }
}

// ---------------------------------------------------------------------------
// Built-in: LoggingHook
// ---------------------------------------------------------------------------

/// Built-in hook that logs all lifecycle phases to stderr.
pub struct LoggingHook;

#[async_trait]
impl ExecutionHook for LoggingHook {
    async fn on_event(&self, ctx: &HookContext) -> Result<HookAction> {
        match &ctx.phase {
            HookPhase::BeforeGraph => {
                eprintln!("[hooks] BeforeGraph (step {})", ctx.step);
            }
            HookPhase::AfterGraph => {
                eprintln!("[hooks] AfterGraph (step {})", ctx.step);
            }
            HookPhase::BeforeNode => {
                eprintln!(
                    "[hooks] BeforeNode '{}' (step {})",
                    ctx.node.as_deref().unwrap_or("?"),
                    ctx.step
                );
            }
            HookPhase::AfterNode => {
                eprintln!(
                    "[hooks] AfterNode '{}' (step {})",
                    ctx.node.as_deref().unwrap_or("?"),
                    ctx.step
                );
            }
            HookPhase::BeforeEdge => {
                if let Some((from, to)) = &ctx.edge {
                    eprintln!("[hooks] BeforeEdge '{}' -> '{}' (step {})", from, to, ctx.step);
                }
            }
            HookPhase::AfterEdge => {
                if let Some((from, to)) = &ctx.edge {
                    eprintln!("[hooks] AfterEdge '{}' -> '{}' (step {})", from, to, ctx.step);
                }
            }
            HookPhase::OnError => {
                eprintln!(
                    "[hooks] OnError at node '{}' (step {})",
                    ctx.node.as_deref().unwrap_or("?"),
                    ctx.step
                );
            }
        }
        Ok(HookAction::Continue)
    }

    fn phases(&self) -> Vec<HookPhase> {
        vec![
            HookPhase::BeforeGraph,
            HookPhase::AfterGraph,
            HookPhase::BeforeNode,
            HookPhase::AfterNode,
            HookPhase::BeforeEdge,
            HookPhase::AfterEdge,
            HookPhase::OnError,
        ]
    }
}

// ---------------------------------------------------------------------------
// Built-in: StateValidationHook
// ---------------------------------------------------------------------------

/// Built-in hook that validates state has required keys at each step.
pub struct StateValidationHook {
    required_keys: Vec<String>,
}

impl StateValidationHook {
    /// Create a new `StateValidationHook` that checks for the given keys.
    pub fn new(required_keys: Vec<String>) -> Self {
        Self { required_keys }
    }
}

#[async_trait]
impl ExecutionHook for StateValidationHook {
    async fn on_event(&self, ctx: &HookContext) -> Result<HookAction> {
        if let Some(obj) = ctx.state.as_object() {
            for key in &self.required_keys {
                if !obj.contains_key(key) {
                    return Ok(HookAction::Abort(format!(
                        "State validation failed: missing required key '{}'",
                        key
                    )));
                }
            }
        } else if !self.required_keys.is_empty() {
            return Ok(HookAction::Abort(
                "State validation failed: state is not an object".to_string(),
            ));
        }
        Ok(HookAction::Continue)
    }

    fn phases(&self) -> Vec<HookPhase> {
        vec![HookPhase::BeforeNode, HookPhase::AfterNode]
    }
}

// ---------------------------------------------------------------------------
// Built-in: TimingHook
// ---------------------------------------------------------------------------

/// Built-in hook that tracks execution time per node.
///
/// Records the timestamp when `BeforeNode` fires and computes the elapsed
/// duration when `AfterNode` fires for the same node.
pub struct TimingHook {
    /// Per-node accumulated durations.
    timings: Mutex<HashMap<String, Vec<Duration>>>,
    /// In-flight node start times.
    pending: Mutex<HashMap<String, Instant>>,
}

impl TimingHook {
    /// Create a new `TimingHook`.
    pub fn new() -> Self {
        Self {
            timings: Mutex::new(HashMap::new()),
            pending: Mutex::new(HashMap::new()),
        }
    }

    /// Return a snapshot of per-node durations.
    pub async fn get_timings(&self) -> HashMap<String, Vec<Duration>> {
        self.timings.lock().await.clone()
    }

    /// Return per-node total durations.
    pub async fn total_durations(&self) -> HashMap<String, Duration> {
        let timings = self.timings.lock().await;
        timings
            .iter()
            .map(|(k, v)| (k.clone(), v.iter().sum()))
            .collect()
    }
}

impl Default for TimingHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ExecutionHook for TimingHook {
    async fn on_event(&self, ctx: &HookContext) -> Result<HookAction> {
        match &ctx.phase {
            HookPhase::BeforeNode => {
                if let Some(node) = &ctx.node {
                    let mut pending = self.pending.lock().await;
                    pending.insert(node.clone(), Instant::now());
                }
            }
            HookPhase::AfterNode => {
                if let Some(node) = &ctx.node {
                    let mut pending = self.pending.lock().await;
                    if let Some(start) = pending.remove(node) {
                        let elapsed = start.elapsed();
                        let mut timings = self.timings.lock().await;
                        timings.entry(node.clone()).or_default().push(elapsed);
                    }
                }
            }
            _ => {}
        }
        Ok(HookAction::Continue)
    }

    fn phases(&self) -> Vec<HookPhase> {
        vec![HookPhase::BeforeNode, HookPhase::AfterNode]
    }
}

// ---------------------------------------------------------------------------
// Built-in: StateSnapshotHook
// ---------------------------------------------------------------------------

/// Built-in hook that captures state snapshots at each node for debugging.
///
/// Stores a snapshot of the state before and after each node execution.
pub struct StateSnapshotHook {
    /// Collected snapshots: (node_name, phase, step, state).
    snapshots: Mutex<Vec<StateSnapshot>>,
}

/// A single state snapshot captured by [`StateSnapshotHook`].
#[derive(Debug, Clone)]
pub struct StateSnapshot {
    /// The node name.
    pub node: String,
    /// The phase at which the snapshot was taken.
    pub phase: HookPhase,
    /// The step number.
    pub step: usize,
    /// The state at this point.
    pub state: Value,
}

impl StateSnapshotHook {
    /// Create a new `StateSnapshotHook`.
    pub fn new() -> Self {
        Self {
            snapshots: Mutex::new(Vec::new()),
        }
    }

    /// Return all captured snapshots.
    pub async fn get_snapshots(&self) -> Vec<StateSnapshot> {
        self.snapshots.lock().await.clone()
    }

    /// Return snapshots for a specific node.
    pub async fn snapshots_for_node(&self, node: &str) -> Vec<StateSnapshot> {
        self.snapshots
            .lock()
            .await
            .iter()
            .filter(|s| s.node == node)
            .cloned()
            .collect()
    }

    /// Clear all captured snapshots.
    pub async fn clear(&self) {
        self.snapshots.lock().await.clear();
    }
}

impl Default for StateSnapshotHook {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ExecutionHook for StateSnapshotHook {
    async fn on_event(&self, ctx: &HookContext) -> Result<HookAction> {
        if let Some(node) = &ctx.node {
            let snapshot = StateSnapshot {
                node: node.clone(),
                phase: ctx.phase.clone(),
                step: ctx.step,
                state: ctx.state.clone(),
            };
            self.snapshots.lock().await.push(snapshot);
        }
        Ok(HookAction::Continue)
    }

    fn phases(&self) -> Vec<HookPhase> {
        vec![HookPhase::BeforeNode, HookPhase::AfterNode]
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    // -----------------------------------------------------------------------
    // HookPhase tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hook_phase_equality() {
        assert_eq!(HookPhase::BeforeGraph, HookPhase::BeforeGraph);
        assert_ne!(HookPhase::BeforeGraph, HookPhase::AfterGraph);
        assert_ne!(HookPhase::BeforeNode, HookPhase::AfterNode);
        assert_ne!(HookPhase::BeforeEdge, HookPhase::AfterEdge);
    }

    #[test]
    fn test_hook_phase_clone() {
        let phase = HookPhase::OnError;
        let cloned = phase.clone();
        assert_eq!(phase, cloned);
    }

    // -----------------------------------------------------------------------
    // HookContext tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hook_context_builder() {
        let ctx = HookContext::new(HookPhase::BeforeNode, json!({"x": 1}), 5)
            .with_node("my_node")
            .with_metadata("key", json!("value"));

        assert_eq!(ctx.phase, HookPhase::BeforeNode);
        assert_eq!(ctx.node, Some("my_node".to_string()));
        assert_eq!(ctx.step, 5);
        assert_eq!(ctx.state, json!({"x": 1}));
        assert_eq!(ctx.metadata.get("key"), Some(&json!("value")));
        assert!(ctx.edge.is_none());
    }

    #[test]
    fn test_hook_context_with_edge() {
        let ctx = HookContext::new(HookPhase::BeforeEdge, json!({}), 0)
            .with_edge("a", "b");

        assert_eq!(ctx.edge, Some(("a".to_string(), "b".to_string())));
        assert!(ctx.node.is_none());
    }

    // -----------------------------------------------------------------------
    // HookAction tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_hook_action_continue() {
        let action = HookAction::Continue;
        assert!(matches!(action, HookAction::Continue));
    }

    #[test]
    fn test_hook_action_modify_state() {
        let action = HookAction::ModifyState(json!({"modified": true}));
        if let HookAction::ModifyState(v) = action {
            assert_eq!(v, json!({"modified": true}));
        } else {
            panic!("Expected ModifyState");
        }
    }

    #[test]
    fn test_hook_action_skip_node() {
        let action = HookAction::SkipNode;
        assert!(matches!(action, HookAction::SkipNode));
    }

    #[test]
    fn test_hook_action_abort() {
        let action = HookAction::Abort("reason".to_string());
        if let HookAction::Abort(reason) = action {
            assert_eq!(reason, "reason");
        } else {
            panic!("Expected Abort");
        }
    }

    // -----------------------------------------------------------------------
    // HookRegistry tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_registry_new_is_empty() {
        let registry = HookRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
    }

    #[tokio::test]
    async fn test_registry_register() {
        let mut registry = HookRegistry::new();
        registry.register(Arc::new(LoggingHook));
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
    }

    #[tokio::test]
    async fn test_registry_unregister() {
        let mut registry = HookRegistry::new();
        registry.register(Arc::new(LoggingHook));
        registry.register(Arc::new(TimingHook::new()));
        assert_eq!(registry.len(), 2);
        registry.unregister(0);
        assert_eq!(registry.len(), 1);
    }

    #[tokio::test]
    async fn test_registry_hooks_for_phase() {
        let mut registry = HookRegistry::new();
        // LoggingHook listens to all phases
        registry.register(Arc::new(LoggingHook));
        // TimingHook listens to BeforeNode and AfterNode only
        registry.register(Arc::new(TimingHook::new()));

        let before_node_hooks = registry.hooks_for_phase(&HookPhase::BeforeNode);
        assert_eq!(before_node_hooks.len(), 2);

        let before_graph_hooks = registry.hooks_for_phase(&HookPhase::BeforeGraph);
        assert_eq!(before_graph_hooks.len(), 1); // only LoggingHook
    }

    #[tokio::test]
    async fn test_registry_dispatch_continue() {
        let mut registry = HookRegistry::new();
        registry.register(Arc::new(LoggingHook));

        let ctx = HookContext::new(HookPhase::BeforeGraph, json!({}), 0);
        let action = registry.dispatch(&ctx).await.unwrap();
        assert!(matches!(action, HookAction::Continue));
    }

    #[tokio::test]
    async fn test_registry_dispatch_no_matching_hooks() {
        let registry = HookRegistry::new();
        let ctx = HookContext::new(HookPhase::BeforeNode, json!({}), 0);
        let action = registry.dispatch(&ctx).await.unwrap();
        assert!(matches!(action, HookAction::Continue));
    }

    #[tokio::test]
    async fn test_registry_dispatch_abort_short_circuits() {
        /// A hook that always aborts.
        struct AbortHook;

        #[async_trait]
        impl ExecutionHook for AbortHook {
            async fn on_event(&self, _ctx: &HookContext) -> Result<HookAction> {
                Ok(HookAction::Abort("aborted".to_string()))
            }
            fn phases(&self) -> Vec<HookPhase> {
                vec![HookPhase::BeforeNode]
            }
        }

        /// A hook that tracks if it was called.
        struct TrackingHook {
            called: Mutex<bool>,
        }

        #[async_trait]
        impl ExecutionHook for TrackingHook {
            async fn on_event(&self, _ctx: &HookContext) -> Result<HookAction> {
                *self.called.lock().await = true;
                Ok(HookAction::Continue)
            }
            fn phases(&self) -> Vec<HookPhase> {
                vec![HookPhase::BeforeNode]
            }
        }

        let tracking = Arc::new(TrackingHook {
            called: Mutex::new(false),
        });

        let mut registry = HookRegistry::new();
        registry.register(Arc::new(AbortHook));
        registry.register(tracking.clone());

        let ctx = HookContext::new(HookPhase::BeforeNode, json!({}), 0)
            .with_node("test");
        let action = registry.dispatch(&ctx).await.unwrap();
        assert!(matches!(action, HookAction::Abort(_)));
        // The tracking hook should NOT have been called due to short-circuit.
        assert!(!*tracking.called.lock().await);
    }

    #[tokio::test]
    async fn test_registry_dispatch_modify_state() {
        struct ModifyHook;

        #[async_trait]
        impl ExecutionHook for ModifyHook {
            async fn on_event(&self, _ctx: &HookContext) -> Result<HookAction> {
                Ok(HookAction::ModifyState(json!({"injected": true})))
            }
            fn phases(&self) -> Vec<HookPhase> {
                vec![HookPhase::BeforeNode]
            }
        }

        let mut registry = HookRegistry::new();
        registry.register(Arc::new(ModifyHook));

        let ctx = HookContext::new(HookPhase::BeforeNode, json!({}), 0);
        let action = registry.dispatch(&ctx).await.unwrap();
        if let HookAction::ModifyState(v) = action {
            assert_eq!(v, json!({"injected": true}));
        } else {
            panic!("Expected ModifyState");
        }
    }

    #[tokio::test]
    async fn test_registry_dispatch_skip_node() {
        struct SkipHook;

        #[async_trait]
        impl ExecutionHook for SkipHook {
            async fn on_event(&self, _ctx: &HookContext) -> Result<HookAction> {
                Ok(HookAction::SkipNode)
            }
            fn phases(&self) -> Vec<HookPhase> {
                vec![HookPhase::BeforeNode]
            }
        }

        let mut registry = HookRegistry::new();
        registry.register(Arc::new(SkipHook));

        let ctx = HookContext::new(HookPhase::BeforeNode, json!({}), 0)
            .with_node("skip_me");
        let action = registry.dispatch(&ctx).await.unwrap();
        assert!(matches!(action, HookAction::SkipNode));
    }

    // -----------------------------------------------------------------------
    // LoggingHook tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_logging_hook_all_phases() {
        let hook = LoggingHook;
        let phases = hook.phases();
        assert_eq!(phases.len(), 7);
        assert!(phases.contains(&HookPhase::BeforeGraph));
        assert!(phases.contains(&HookPhase::AfterGraph));
        assert!(phases.contains(&HookPhase::BeforeNode));
        assert!(phases.contains(&HookPhase::AfterNode));
        assert!(phases.contains(&HookPhase::BeforeEdge));
        assert!(phases.contains(&HookPhase::AfterEdge));
        assert!(phases.contains(&HookPhase::OnError));
    }

    #[tokio::test]
    async fn test_logging_hook_returns_continue() {
        let hook = LoggingHook;
        let ctx = HookContext::new(HookPhase::BeforeNode, json!({}), 1)
            .with_node("test_node");
        let action = hook.on_event(&ctx).await.unwrap();
        assert!(matches!(action, HookAction::Continue));
    }

    #[tokio::test]
    async fn test_logging_hook_edge_event() {
        let hook = LoggingHook;
        let ctx = HookContext::new(HookPhase::BeforeEdge, json!({}), 2)
            .with_edge("a", "b");
        let action = hook.on_event(&ctx).await.unwrap();
        assert!(matches!(action, HookAction::Continue));
    }

    // -----------------------------------------------------------------------
    // StateValidationHook tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_state_validation_passes() {
        let hook = StateValidationHook::new(vec!["name".to_string(), "age".to_string()]);
        let ctx = HookContext::new(
            HookPhase::BeforeNode,
            json!({"name": "Alice", "age": 30}),
            0,
        );
        let action = hook.on_event(&ctx).await.unwrap();
        assert!(matches!(action, HookAction::Continue));
    }

    #[tokio::test]
    async fn test_state_validation_missing_key() {
        let hook = StateValidationHook::new(vec!["name".to_string(), "missing_key".to_string()]);
        let ctx = HookContext::new(HookPhase::BeforeNode, json!({"name": "Alice"}), 0);
        let action = hook.on_event(&ctx).await.unwrap();
        if let HookAction::Abort(reason) = action {
            assert!(reason.contains("missing_key"));
        } else {
            panic!("Expected Abort");
        }
    }

    #[tokio::test]
    async fn test_state_validation_non_object_state() {
        let hook = StateValidationHook::new(vec!["key".to_string()]);
        let ctx = HookContext::new(HookPhase::BeforeNode, json!("not an object"), 0);
        let action = hook.on_event(&ctx).await.unwrap();
        assert!(matches!(action, HookAction::Abort(_)));
    }

    #[tokio::test]
    async fn test_state_validation_empty_keys() {
        let hook = StateValidationHook::new(vec![]);
        let ctx = HookContext::new(HookPhase::BeforeNode, json!({}), 0);
        let action = hook.on_event(&ctx).await.unwrap();
        assert!(matches!(action, HookAction::Continue));
    }

    #[tokio::test]
    async fn test_state_validation_phases() {
        let hook = StateValidationHook::new(vec![]);
        let phases = hook.phases();
        assert_eq!(phases.len(), 2);
        assert!(phases.contains(&HookPhase::BeforeNode));
        assert!(phases.contains(&HookPhase::AfterNode));
    }

    // -----------------------------------------------------------------------
    // TimingHook tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_timing_hook_records_duration() {
        let hook = TimingHook::new();

        // Simulate BeforeNode
        let ctx_before = HookContext::new(HookPhase::BeforeNode, json!({}), 1)
            .with_node("node_a");
        hook.on_event(&ctx_before).await.unwrap();

        // Small delay to ensure measurable duration
        tokio::time::sleep(Duration::from_millis(5)).await;

        // Simulate AfterNode
        let ctx_after = HookContext::new(HookPhase::AfterNode, json!({}), 1)
            .with_node("node_a");
        hook.on_event(&ctx_after).await.unwrap();

        let timings = hook.get_timings().await;
        assert!(timings.contains_key("node_a"));
        assert_eq!(timings["node_a"].len(), 1);
        assert!(timings["node_a"][0] >= Duration::from_millis(1));
    }

    #[tokio::test]
    async fn test_timing_hook_multiple_invocations() {
        let hook = TimingHook::new();

        for _ in 0..3 {
            let before = HookContext::new(HookPhase::BeforeNode, json!({}), 1)
                .with_node("counter");
            hook.on_event(&before).await.unwrap();

            let after = HookContext::new(HookPhase::AfterNode, json!({}), 1)
                .with_node("counter");
            hook.on_event(&after).await.unwrap();
        }

        let timings = hook.get_timings().await;
        assert_eq!(timings["counter"].len(), 3);
    }

    #[tokio::test]
    async fn test_timing_hook_total_durations() {
        let hook = TimingHook::new();

        for _ in 0..2 {
            let before = HookContext::new(HookPhase::BeforeNode, json!({}), 1)
                .with_node("n");
            hook.on_event(&before).await.unwrap();
            let after = HookContext::new(HookPhase::AfterNode, json!({}), 1)
                .with_node("n");
            hook.on_event(&after).await.unwrap();
        }

        let totals = hook.total_durations().await;
        assert!(totals.contains_key("n"));
    }

    #[tokio::test]
    async fn test_timing_hook_phases() {
        let hook = TimingHook::new();
        let phases = hook.phases();
        assert_eq!(phases.len(), 2);
        assert!(phases.contains(&HookPhase::BeforeNode));
        assert!(phases.contains(&HookPhase::AfterNode));
    }

    // -----------------------------------------------------------------------
    // StateSnapshotHook tests
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_snapshot_hook_captures() {
        let hook = StateSnapshotHook::new();

        let ctx = HookContext::new(HookPhase::BeforeNode, json!({"x": 1}), 1)
            .with_node("my_node");
        hook.on_event(&ctx).await.unwrap();

        let ctx2 = HookContext::new(HookPhase::AfterNode, json!({"x": 2}), 1)
            .with_node("my_node");
        hook.on_event(&ctx2).await.unwrap();

        let snapshots = hook.get_snapshots().await;
        assert_eq!(snapshots.len(), 2);
        assert_eq!(snapshots[0].node, "my_node");
        assert_eq!(snapshots[0].phase, HookPhase::BeforeNode);
        assert_eq!(snapshots[0].state, json!({"x": 1}));
        assert_eq!(snapshots[1].phase, HookPhase::AfterNode);
        assert_eq!(snapshots[1].state, json!({"x": 2}));
    }

    #[tokio::test]
    async fn test_snapshot_hook_for_node() {
        let hook = StateSnapshotHook::new();

        let ctx_a = HookContext::new(HookPhase::BeforeNode, json!({}), 1)
            .with_node("a");
        hook.on_event(&ctx_a).await.unwrap();

        let ctx_b = HookContext::new(HookPhase::BeforeNode, json!({}), 2)
            .with_node("b");
        hook.on_event(&ctx_b).await.unwrap();

        let a_snaps = hook.snapshots_for_node("a").await;
        assert_eq!(a_snaps.len(), 1);
        let b_snaps = hook.snapshots_for_node("b").await;
        assert_eq!(b_snaps.len(), 1);
    }

    #[tokio::test]
    async fn test_snapshot_hook_clear() {
        let hook = StateSnapshotHook::new();

        let ctx = HookContext::new(HookPhase::BeforeNode, json!({}), 1)
            .with_node("a");
        hook.on_event(&ctx).await.unwrap();
        assert_eq!(hook.get_snapshots().await.len(), 1);

        hook.clear().await;
        assert_eq!(hook.get_snapshots().await.len(), 0);
    }

    #[tokio::test]
    async fn test_snapshot_hook_phases() {
        let hook = StateSnapshotHook::new();
        let phases = hook.phases();
        assert_eq!(phases.len(), 2);
        assert!(phases.contains(&HookPhase::BeforeNode));
        assert!(phases.contains(&HookPhase::AfterNode));
    }

    #[tokio::test]
    async fn test_snapshot_hook_ignores_no_node() {
        let hook = StateSnapshotHook::new();
        // Context without a node name — should not capture
        let ctx = HookContext::new(HookPhase::BeforeNode, json!({}), 1);
        hook.on_event(&ctx).await.unwrap();
        assert_eq!(hook.get_snapshots().await.len(), 0);
    }

    // -----------------------------------------------------------------------
    // Integration: multiple hooks in registry
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_multiple_hooks_in_registry() {
        let timing = Arc::new(TimingHook::new());
        let snapshot = Arc::new(StateSnapshotHook::new());

        let mut registry = HookRegistry::new();
        registry.register(timing.clone());
        registry.register(snapshot.clone());
        registry.register(Arc::new(LoggingHook));

        // Simulate BeforeNode
        let ctx = HookContext::new(HookPhase::BeforeNode, json!({"step": "before"}), 1)
            .with_node("test_node");
        let action = registry.dispatch(&ctx).await.unwrap();
        assert!(matches!(action, HookAction::Continue));

        // Simulate AfterNode
        let ctx2 = HookContext::new(HookPhase::AfterNode, json!({"step": "after"}), 1)
            .with_node("test_node");
        let action2 = registry.dispatch(&ctx2).await.unwrap();
        assert!(matches!(action2, HookAction::Continue));

        // Verify timing and snapshot hooks recorded data
        let timings = timing.get_timings().await;
        assert!(timings.contains_key("test_node"));

        let snapshots = snapshot.get_snapshots().await;
        assert_eq!(snapshots.len(), 2);
    }

    #[tokio::test]
    async fn test_registry_default() {
        let registry = HookRegistry::default();
        assert!(registry.is_empty());
    }

    // -----------------------------------------------------------------------
    // Hook error propagation
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn test_hook_error_propagation() {
        struct ErrorHook;

        #[async_trait]
        impl ExecutionHook for ErrorHook {
            async fn on_event(&self, _ctx: &HookContext) -> Result<HookAction> {
                Err(LangGraphError::Other("hook error".to_string()))
            }
            fn phases(&self) -> Vec<HookPhase> {
                vec![HookPhase::BeforeNode]
            }
        }

        let mut registry = HookRegistry::new();
        registry.register(Arc::new(ErrorHook));

        let ctx = HookContext::new(HookPhase::BeforeNode, json!({}), 0);
        let result = registry.dispatch(&ctx).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("hook error"));
    }
}
