//! Node trait + per-superstep context + helper closure adapter.

use std::sync::Arc;

use async_trait::async_trait;
use cognis2_core::stream::Observer;
use cognis2_core::{Event, Result, RunnableConfig};
use uuid::Uuid;

use crate::goto::Goto;
use crate::state::GraphState;

/// Output of a node's `execute`: a typed state delta + where to go next.
pub struct NodeOut<S: GraphState> {
    /// State update — applied via per-field reducers in this superstep's atomic merge.
    pub update: S::Update,
    /// Routing decision.
    pub goto: Goto,
}

impl<S: GraphState> NodeOut<S> {
    /// Convenience: terminal node with a state update.
    pub fn end_with(update: S::Update) -> Self {
        Self {
            update,
            goto: Goto::End,
        }
    }

    /// Convenience: route somewhere with no state delta (Default::default()).
    pub fn goto_only(goto: Goto) -> Self {
        Self {
            update: S::Update::default(),
            goto,
        }
    }
}

/// Per-superstep context handed to every `Node::execute` call. Carries
/// run-correlation metadata and the active `RunnableConfig`. The lifetime
/// is the superstep — don't hold across awaits beyond `execute` returning.
pub struct NodeCtx<'a> {
    /// Correlation ID for this run.
    pub run_id: Uuid,
    /// Superstep counter (0-indexed).
    pub step: u64,
    /// The active runnable config (recursion_limit, observers, cancel_token, …).
    pub config: &'a RunnableConfig,
    /// Per-target payload when this node is invoked as a `Goto::Send` target.
    /// `None` for all other dispatch types.
    payload: Option<&'a serde_json::Value>,
}

impl<'a> NodeCtx<'a> {
    /// Create a new `NodeCtx`. Primarily used by the engine; exposed publicly
    /// so node implementations in external crates can construct test contexts.
    pub fn new(run_id: Uuid, step: u64, config: &'a RunnableConfig) -> Self {
        Self {
            run_id,
            step,
            config,
            payload: None,
        }
    }

    /// Engine-internal: attach a Send payload.
    pub(crate) fn with_payload(mut self, payload: &'a serde_json::Value) -> Self {
        self.payload = Some(payload);
        self
    }

    /// The Send payload accompanying this dispatch, if any. Returns `None`
    /// when the node is invoked via `Goto::Node` or `Goto::Multiple`.
    pub fn payload(&self) -> Option<&serde_json::Value> {
        self.payload
    }

    /// Notify every observer in `config.observers` of an event.
    pub fn emit(&self, event: &Event) {
        self.config.emit(event);
    }

    /// True if the run was cancelled.
    pub fn is_cancelled(&self) -> bool {
        self.config.is_cancelled()
    }

    /// Convenience accessor for observers.
    pub fn observers(&self) -> &[Arc<dyn Observer>] {
        &self.config.observers
    }
}

/// Per-task retry policy. The engine wraps `Node::execute` calls in a
/// retry loop when the node returns a `Some` policy and the call fails
/// with a [`cognis2_core::CognisError`] whose `is_retryable()` is true.
#[derive(Debug, Clone, Copy)]
pub struct NodeRetryPolicy {
    /// Maximum total attempts (including the first).
    pub max_attempts: u32,
    /// Initial backoff before the first retry (milliseconds).
    pub initial_delay_ms: u64,
    /// Multiplier applied to the delay after each failed attempt.
    pub backoff_multiplier: f64,
    /// Cap on per-attempt delay (milliseconds).
    pub max_delay_ms: u64,
}

impl Default for NodeRetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 100,
            backoff_multiplier: 2.0,
            max_delay_ms: 30_000,
        }
    }
}

/// The unit of computation in a graph. Async, takes a `&S` snapshot of state
/// + per-step context, returns a delta + a routing decision.
#[async_trait]
pub trait Node<S: GraphState>: Send + Sync {
    /// Execute one superstep of this node.
    async fn execute(&self, state: &S, ctx: &NodeCtx<'_>) -> Result<NodeOut<S>>;

    /// Friendly name for telemetry / logging. Default uses the type name.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Per-task retry policy. Default `None` means "no retry — propagate
    /// the error". When `Some`, the engine retries `execute` on retryable
    /// errors with exponential backoff.
    fn retry_policy(&self) -> Option<NodeRetryPolicy> {
        None
    }
}

/// Closure adapter — wrap any `Fn(&S, &NodeCtx) -> Future` as a `Node`.
pub struct NodeFn<S, F> {
    name: String,
    f: F,
    _state: std::marker::PhantomData<fn() -> S>,
}

/// Build a `NodeFn` from a closure. The closure receives `(&S, &NodeCtx)`
/// and returns `Future<Output = Result<NodeOut<S>>>`.
pub fn node_fn<S, F, Fut>(name: impl Into<String>, f: F) -> NodeFn<S, F>
where
    S: GraphState,
    F: Fn(&S, &NodeCtx<'_>) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<NodeOut<S>>> + Send,
{
    NodeFn {
        name: name.into(),
        f,
        _state: std::marker::PhantomData,
    }
}

#[async_trait]
impl<S, F, Fut> Node<S> for NodeFn<S, F>
where
    S: GraphState,
    F: Fn(&S, &NodeCtx<'_>) -> Fut + Send + Sync + 'static,
    Fut: std::future::Future<Output = Result<NodeOut<S>>> + Send,
{
    async fn execute(&self, state: &S, ctx: &NodeCtx<'_>) -> Result<NodeOut<S>> {
        (self.f)(state, ctx).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::goto::Goto;
    use crate::state::GraphState;

    #[derive(Default, Clone, Debug, PartialEq)]
    struct S {
        n: u32,
    }

    #[derive(Default)]
    struct SU {
        n: u32,
    }

    impl GraphState for S {
        type Update = SU;
        fn apply(&mut self, update: Self::Update) {
            self.n += update.n;
        }
    }

    #[tokio::test]
    async fn node_fn_executes() {
        let n = node_fn::<S, _, _>("incr", |state, _ctx| {
            let cur = state.n;
            async move {
                Ok(NodeOut {
                    update: SU { n: cur + 1 },
                    goto: Goto::end(),
                })
            }
        });
        let cfg = RunnableConfig::default();
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        let s = S { n: 5 };
        let out = n.execute(&s, &ctx).await.unwrap();
        assert_eq!(out.update.n, 6);
        assert!(out.goto.is_end());
        assert_eq!(n.name(), "incr");
    }

    #[test]
    fn node_ctx_payload_default_none() {
        let cfg = RunnableConfig::default();
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg);
        assert!(ctx.payload().is_none());
    }

    #[test]
    fn node_ctx_with_payload() {
        let cfg = RunnableConfig::default();
        let payload = serde_json::json!({"x": 42});
        let ctx = NodeCtx::new(Uuid::nil(), 0, &cfg).with_payload(&payload);
        assert_eq!(ctx.payload().unwrap()["x"], 42);
    }

    #[test]
    fn nodeout_constructors() {
        let upd = SU { n: 10 };
        let no: NodeOut<S> = NodeOut::end_with(upd);
        assert!(no.goto.is_end());

        let no2: NodeOut<S> = NodeOut::goto_only(Goto::node("next"));
        assert_eq!(no2.update.n, 0);
        assert_eq!(no2.goto, Goto::node("next"));
    }
}
