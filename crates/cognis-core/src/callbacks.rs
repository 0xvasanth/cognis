//! Typed callback-handler taxonomy. A higher-level alternative to the
//! flat [`crate::Observer`] trait — handlers receive *typed* lifecycle
//! calls (`on_chain_start`, `on_llm_token`, `on_tool_end`, …) instead
//! of a single `&Event` enum match.
//!
//! Two complementary integration paths:
//!
//! - Continue using [`crate::Observer`] for cheap, generic event sinks.
//! - Use [`CallbackHandler`] when you want strongly-named hook methods
//!   (typical for ports of LangChain-style integrations).
//!
//! Bridge: every `CallbackHandler` is automatically usable as an
//! `Observer` via [`HandlerObserver`], so the two systems compose. A
//! [`CallbackManager`] owns a pool of handlers and dispatches them via
//! a single observer entry on `RunnableConfig`.
//!
//! Customization: implement [`CallbackHandler`] for full control. For
//! one-off cases use [`HandlerBuilder`] to assemble a handler from
//! individual closures.

use std::sync::Arc;

use uuid::Uuid;

use crate::stream::{Event, Observer};

/// Typed lifecycle callbacks. All methods have default no-op impls;
/// implement only what you need.
pub trait CallbackHandler: Send + Sync {
    /// A `Runnable` (chain / graph) started.
    fn on_chain_start(&self, _runnable: &str, _input: &serde_json::Value, _run_id: Uuid) {}
    /// A `Runnable` finished successfully.
    fn on_chain_end(&self, _runnable: &str, _output: &serde_json::Value, _run_id: Uuid) {}
    /// A `Runnable` errored.
    fn on_chain_error(&self, _runnable: &str, _error: &str, _run_id: Uuid) {}

    /// LLM / chat-model invocation started.
    fn on_llm_start(&self, _model: &str, _prompt: &serde_json::Value, _run_id: Uuid) {}
    /// LLM emitted a streamed token.
    fn on_llm_token(&self, _token: &str, _run_id: Uuid) {}
    /// LLM finished a generation.
    fn on_llm_end(&self, _model: &str, _output: &serde_json::Value, _run_id: Uuid) {}
    /// LLM errored.
    fn on_llm_error(&self, _model: &str, _error: &str, _run_id: Uuid) {}

    /// Tool execution started.
    fn on_tool_start(&self, _tool: &str, _args: &serde_json::Value, _run_id: Uuid) {}
    /// Tool execution finished.
    fn on_tool_end(&self, _tool: &str, _result: &serde_json::Value, _run_id: Uuid) {}
    /// Tool errored.
    fn on_tool_error(&self, _tool: &str, _error: &str, _run_id: Uuid) {}

    /// Graph node started.
    fn on_node_start(&self, _node: &str, _step: u64, _run_id: Uuid) {}
    /// Graph node finished.
    fn on_node_end(&self, _node: &str, _step: u64, _output: &serde_json::Value, _run_id: Uuid) {}

    /// Graph engine persisted a checkpoint.
    fn on_checkpoint(&self, _step: u64, _run_id: Uuid) {}

    /// Custom node-emitted event.
    fn on_custom(&self, _kind: &str, _payload: &serde_json::Value, _run_id: Uuid) {}

    /// Friendly name for diagnostics.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

// ---------------------------------------------------------------------------
// HandlerObserver — adapt a CallbackHandler to the Observer trait.
// ---------------------------------------------------------------------------

/// Wraps a [`CallbackHandler`] as an [`Observer`] so existing event
/// plumbing routes through the typed handler.
pub struct HandlerObserver<H: CallbackHandler>(pub H);

impl<H: CallbackHandler> Observer for HandlerObserver<H> {
    fn on_event(&self, event: &Event) {
        match event {
            Event::OnStart {
                runnable,
                run_id,
                input,
            } => self.0.on_chain_start(runnable, input, *run_id),
            Event::OnEnd {
                runnable,
                run_id,
                output,
            } => self.0.on_chain_end(runnable, output, *run_id),
            Event::OnError { error, run_id } => self.0.on_chain_error("", error, *run_id),
            Event::OnLlmToken { token, run_id } => self.0.on_llm_token(token, *run_id),
            Event::OnToolStart { tool, args, run_id } => self.0.on_tool_start(tool, args, *run_id),
            Event::OnToolEnd {
                tool,
                result,
                run_id,
            } => self.0.on_tool_end(tool, result, *run_id),
            Event::OnNodeStart { node, step, run_id } => self.0.on_node_start(node, *step, *run_id),
            Event::OnNodeEnd {
                node,
                step,
                output,
                run_id,
            } => self.0.on_node_end(node, *step, output, *run_id),
            Event::OnCheckpoint { step, run_id } => self.0.on_checkpoint(*step, *run_id),
            Event::Custom {
                kind,
                payload,
                run_id,
            } => self.0.on_custom(kind, payload, *run_id),
        }
    }
}

// ---------------------------------------------------------------------------
// CallbackManager — pool of handlers exposed as a single observer.
// ---------------------------------------------------------------------------

/// Pool of [`CallbackHandler`]s. Use to bundle multiple handlers and
/// expose them as a single [`Observer`] entry on `RunnableConfig`.
#[derive(Default)]
pub struct CallbackManager {
    handlers: Vec<Arc<dyn CallbackHandler>>,
}

impl CallbackManager {
    /// Empty manager.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a handler.
    pub fn push(mut self, h: Arc<dyn CallbackHandler>) -> Self {
        self.handlers.push(h);
        self
    }

    /// Number of registered handlers.
    pub fn len(&self) -> usize {
        self.handlers.len()
    }

    /// True when no handlers are registered.
    pub fn is_empty(&self) -> bool {
        self.handlers.is_empty()
    }

    /// Borrow the registered handlers (read-only).
    pub fn handlers(&self) -> &[Arc<dyn CallbackHandler>] {
        &self.handlers
    }
}

impl Observer for CallbackManager {
    fn on_event(&self, event: &Event) {
        for h in &self.handlers {
            HandlerObserver(h.clone()).on_event(event);
        }
    }
}

// Implement CallbackHandler for Arc<dyn CallbackHandler> so we can wrap
// trait objects directly.
impl CallbackHandler for Arc<dyn CallbackHandler> {
    fn on_chain_start(&self, runnable: &str, input: &serde_json::Value, run_id: Uuid) {
        self.as_ref().on_chain_start(runnable, input, run_id)
    }
    fn on_chain_end(&self, runnable: &str, output: &serde_json::Value, run_id: Uuid) {
        self.as_ref().on_chain_end(runnable, output, run_id)
    }
    fn on_chain_error(&self, runnable: &str, error: &str, run_id: Uuid) {
        self.as_ref().on_chain_error(runnable, error, run_id)
    }
    fn on_llm_start(&self, model: &str, prompt: &serde_json::Value, run_id: Uuid) {
        self.as_ref().on_llm_start(model, prompt, run_id)
    }
    fn on_llm_token(&self, token: &str, run_id: Uuid) {
        self.as_ref().on_llm_token(token, run_id)
    }
    fn on_llm_end(&self, model: &str, output: &serde_json::Value, run_id: Uuid) {
        self.as_ref().on_llm_end(model, output, run_id)
    }
    fn on_llm_error(&self, model: &str, error: &str, run_id: Uuid) {
        self.as_ref().on_llm_error(model, error, run_id)
    }
    fn on_tool_start(&self, tool: &str, args: &serde_json::Value, run_id: Uuid) {
        self.as_ref().on_tool_start(tool, args, run_id)
    }
    fn on_tool_end(&self, tool: &str, result: &serde_json::Value, run_id: Uuid) {
        self.as_ref().on_tool_end(tool, result, run_id)
    }
    fn on_tool_error(&self, tool: &str, error: &str, run_id: Uuid) {
        self.as_ref().on_tool_error(tool, error, run_id)
    }
    fn on_node_start(&self, node: &str, step: u64, run_id: Uuid) {
        self.as_ref().on_node_start(node, step, run_id)
    }
    fn on_node_end(&self, node: &str, step: u64, output: &serde_json::Value, run_id: Uuid) {
        self.as_ref().on_node_end(node, step, output, run_id)
    }
    fn on_checkpoint(&self, step: u64, run_id: Uuid) {
        self.as_ref().on_checkpoint(step, run_id)
    }
    fn on_custom(&self, kind: &str, payload: &serde_json::Value, run_id: Uuid) {
        self.as_ref().on_custom(kind, payload, run_id)
    }
    fn name(&self) -> &str {
        self.as_ref().name()
    }
}

// ---------------------------------------------------------------------------
// HandlerBuilder — assemble from individual closures.
// ---------------------------------------------------------------------------

type ChainStartFn = Arc<dyn Fn(&str, &serde_json::Value, Uuid) + Send + Sync>;
type ChainEndFn = Arc<dyn Fn(&str, &serde_json::Value, Uuid) + Send + Sync>;
type ChainErrFn = Arc<dyn Fn(&str, &str, Uuid) + Send + Sync>;
type LlmStartFn = Arc<dyn Fn(&str, &serde_json::Value, Uuid) + Send + Sync>;
type LlmEndFn = Arc<dyn Fn(&str, &serde_json::Value, Uuid) + Send + Sync>;
type LlmTokenFn = Arc<dyn Fn(&str, Uuid) + Send + Sync>;
type LlmErrFn = Arc<dyn Fn(&str, &str, Uuid) + Send + Sync>;
type ToolStartFn = Arc<dyn Fn(&str, &serde_json::Value, Uuid) + Send + Sync>;
type ToolEndFn = Arc<dyn Fn(&str, &serde_json::Value, Uuid) + Send + Sync>;
type ToolErrFn = Arc<dyn Fn(&str, &str, Uuid) + Send + Sync>;
type NodeStartFn = Arc<dyn Fn(&str, u64, Uuid) + Send + Sync>;
type NodeEndFn = Arc<dyn Fn(&str, u64, &serde_json::Value, Uuid) + Send + Sync>;
type CheckpointFn = Arc<dyn Fn(u64, Uuid) + Send + Sync>;
type CustomFn = Arc<dyn Fn(&str, &serde_json::Value, Uuid) + Send + Sync>;

/// Build a [`CallbackHandler`] from any subset of closures. Methods not
/// configured fall through to the trait's no-op defaults.
#[derive(Default)]
pub struct HandlerBuilder {
    chain_start: Option<ChainStartFn>,
    chain_end: Option<ChainEndFn>,
    chain_error: Option<ChainErrFn>,
    llm_start: Option<LlmStartFn>,
    llm_token: Option<LlmTokenFn>,
    llm_end: Option<LlmEndFn>,
    llm_error: Option<LlmErrFn>,
    tool_start: Option<ToolStartFn>,
    tool_end: Option<ToolEndFn>,
    tool_error: Option<ToolErrFn>,
    node_start: Option<NodeStartFn>,
    node_end: Option<NodeEndFn>,
    checkpoint: Option<CheckpointFn>,
    custom: Option<CustomFn>,
    name: Option<String>,
}

impl HandlerBuilder {
    /// Empty builder.
    pub fn new() -> Self {
        Self::default()
    }
    /// Override the handler's reported name.
    pub fn with_name(mut self, n: impl Into<String>) -> Self {
        self.name = Some(n.into());
        self
    }
    /// Set the on_chain_start closure.
    pub fn on_chain_start<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &serde_json::Value, Uuid) + Send + Sync + 'static,
    {
        self.chain_start = Some(Arc::new(f));
        self
    }
    /// Set the on_chain_end closure.
    pub fn on_chain_end<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &serde_json::Value, Uuid) + Send + Sync + 'static,
    {
        self.chain_end = Some(Arc::new(f));
        self
    }
    /// Set the on_chain_error closure.
    pub fn on_chain_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &str, Uuid) + Send + Sync + 'static,
    {
        self.chain_error = Some(Arc::new(f));
        self
    }
    /// Set the on_llm_start closure.
    pub fn on_llm_start<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &serde_json::Value, Uuid) + Send + Sync + 'static,
    {
        self.llm_start = Some(Arc::new(f));
        self
    }
    /// Set the on_llm_token closure.
    pub fn on_llm_token<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, Uuid) + Send + Sync + 'static,
    {
        self.llm_token = Some(Arc::new(f));
        self
    }
    /// Set the on_llm_end closure.
    pub fn on_llm_end<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &serde_json::Value, Uuid) + Send + Sync + 'static,
    {
        self.llm_end = Some(Arc::new(f));
        self
    }
    /// Set the on_llm_error closure.
    pub fn on_llm_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &str, Uuid) + Send + Sync + 'static,
    {
        self.llm_error = Some(Arc::new(f));
        self
    }
    /// Set the on_tool_start closure.
    pub fn on_tool_start<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &serde_json::Value, Uuid) + Send + Sync + 'static,
    {
        self.tool_start = Some(Arc::new(f));
        self
    }
    /// Set the on_tool_end closure.
    pub fn on_tool_end<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &serde_json::Value, Uuid) + Send + Sync + 'static,
    {
        self.tool_end = Some(Arc::new(f));
        self
    }
    /// Set the on_tool_error closure.
    pub fn on_tool_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &str, Uuid) + Send + Sync + 'static,
    {
        self.tool_error = Some(Arc::new(f));
        self
    }
    /// Set the on_node_start closure.
    pub fn on_node_start<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, u64, Uuid) + Send + Sync + 'static,
    {
        self.node_start = Some(Arc::new(f));
        self
    }
    /// Set the on_node_end closure.
    pub fn on_node_end<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, u64, &serde_json::Value, Uuid) + Send + Sync + 'static,
    {
        self.node_end = Some(Arc::new(f));
        self
    }
    /// Set the on_checkpoint closure.
    pub fn on_checkpoint<F>(mut self, f: F) -> Self
    where
        F: Fn(u64, Uuid) + Send + Sync + 'static,
    {
        self.checkpoint = Some(Arc::new(f));
        self
    }
    /// Set the on_custom closure.
    pub fn on_custom<F>(mut self, f: F) -> Self
    where
        F: Fn(&str, &serde_json::Value, Uuid) + Send + Sync + 'static,
    {
        self.custom = Some(Arc::new(f));
        self
    }
    /// Finalize.
    pub fn build(self) -> BuiltHandler {
        BuiltHandler { inner: self }
    }
}

/// Handler constructed via [`HandlerBuilder`].
pub struct BuiltHandler {
    inner: HandlerBuilder,
}

impl CallbackHandler for BuiltHandler {
    fn on_chain_start(&self, runnable: &str, input: &serde_json::Value, run_id: Uuid) {
        if let Some(f) = &self.inner.chain_start {
            f(runnable, input, run_id);
        }
    }
    fn on_chain_end(&self, runnable: &str, output: &serde_json::Value, run_id: Uuid) {
        if let Some(f) = &self.inner.chain_end {
            f(runnable, output, run_id);
        }
    }
    fn on_chain_error(&self, runnable: &str, error: &str, run_id: Uuid) {
        if let Some(f) = &self.inner.chain_error {
            f(runnable, error, run_id);
        }
    }
    fn on_llm_start(&self, model: &str, prompt: &serde_json::Value, run_id: Uuid) {
        if let Some(f) = &self.inner.llm_start {
            f(model, prompt, run_id);
        }
    }
    fn on_llm_token(&self, token: &str, run_id: Uuid) {
        if let Some(f) = &self.inner.llm_token {
            f(token, run_id);
        }
    }
    fn on_llm_end(&self, model: &str, output: &serde_json::Value, run_id: Uuid) {
        if let Some(f) = &self.inner.llm_end {
            f(model, output, run_id);
        }
    }
    fn on_llm_error(&self, model: &str, error: &str, run_id: Uuid) {
        if let Some(f) = &self.inner.llm_error {
            f(model, error, run_id);
        }
    }
    fn on_tool_start(&self, tool: &str, args: &serde_json::Value, run_id: Uuid) {
        if let Some(f) = &self.inner.tool_start {
            f(tool, args, run_id);
        }
    }
    fn on_tool_end(&self, tool: &str, result: &serde_json::Value, run_id: Uuid) {
        if let Some(f) = &self.inner.tool_end {
            f(tool, result, run_id);
        }
    }
    fn on_tool_error(&self, tool: &str, error: &str, run_id: Uuid) {
        if let Some(f) = &self.inner.tool_error {
            f(tool, error, run_id);
        }
    }
    fn on_node_start(&self, node: &str, step: u64, run_id: Uuid) {
        if let Some(f) = &self.inner.node_start {
            f(node, step, run_id);
        }
    }
    fn on_node_end(&self, node: &str, step: u64, output: &serde_json::Value, run_id: Uuid) {
        if let Some(f) = &self.inner.node_end {
            f(node, step, output, run_id);
        }
    }
    fn on_checkpoint(&self, step: u64, run_id: Uuid) {
        if let Some(f) = &self.inner.checkpoint {
            f(step, run_id);
        }
    }
    fn on_custom(&self, kind: &str, payload: &serde_json::Value, run_id: Uuid) {
        if let Some(f) = &self.inner.custom {
            f(kind, payload, run_id);
        }
    }
    fn name(&self) -> &str {
        self.inner.name.as_deref().unwrap_or("BuiltHandler")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn handler_observer_routes_typed_events() {
        struct H {
            chain: Arc<AtomicUsize>,
            tool: Arc<AtomicUsize>,
            checkpoint: Arc<AtomicUsize>,
            custom: Arc<AtomicUsize>,
        }
        impl CallbackHandler for H {
            fn on_chain_start(&self, _: &str, _: &serde_json::Value, _: Uuid) {
                self.chain.fetch_add(1, Ordering::SeqCst);
            }
            fn on_tool_start(&self, _: &str, _: &serde_json::Value, _: Uuid) {
                self.tool.fetch_add(1, Ordering::SeqCst);
            }
            fn on_checkpoint(&self, _: u64, _: Uuid) {
                self.checkpoint.fetch_add(1, Ordering::SeqCst);
            }
            fn on_custom(&self, _: &str, _: &serde_json::Value, _: Uuid) {
                self.custom.fetch_add(1, Ordering::SeqCst);
            }
        }

        let h = H {
            chain: Arc::new(AtomicUsize::new(0)),
            tool: Arc::new(AtomicUsize::new(0)),
            checkpoint: Arc::new(AtomicUsize::new(0)),
            custom: Arc::new(AtomicUsize::new(0)),
        };
        let chain = h.chain.clone();
        let tool = h.tool.clone();
        let cp = h.checkpoint.clone();
        let custom = h.custom.clone();

        let obs = HandlerObserver(h);
        let id = Uuid::nil();

        obs.on_event(&Event::OnStart {
            runnable: "r".into(),
            run_id: id,
            input: serde_json::Value::Null,
        });
        obs.on_event(&Event::OnToolStart {
            tool: "t".into(),
            args: serde_json::Value::Null,
            run_id: id,
        });
        obs.on_event(&Event::OnCheckpoint {
            step: 0,
            run_id: id,
        });
        obs.on_event(&Event::Custom {
            kind: "k".into(),
            payload: serde_json::json!({"x": 1}),
            run_id: id,
        });

        assert_eq!(chain.load(Ordering::SeqCst), 1);
        assert_eq!(tool.load(Ordering::SeqCst), 1);
        assert_eq!(cp.load(Ordering::SeqCst), 1);
        assert_eq!(custom.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn manager_dispatches_to_all_handlers() {
        let count = Arc::new(AtomicUsize::new(0));
        struct H(Arc<AtomicUsize>);
        impl CallbackHandler for H {
            fn on_chain_start(&self, _: &str, _: &serde_json::Value, _: Uuid) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let mgr = CallbackManager::new()
            .push(Arc::new(H(count.clone())))
            .push(Arc::new(H(count.clone())));
        mgr.on_event(&Event::OnStart {
            runnable: "r".into(),
            run_id: Uuid::nil(),
            input: serde_json::Value::Null,
        });
        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn handler_builder_assembles_from_closures() {
        let starts = Arc::new(AtomicUsize::new(0));
        let s2 = starts.clone();
        let h: BuiltHandler = HandlerBuilder::new()
            .on_chain_start(move |_, _, _| {
                s2.fetch_add(1, Ordering::SeqCst);
            })
            .with_name("test")
            .build();
        h.on_chain_start("r", &serde_json::Value::Null, Uuid::nil());
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert_eq!(h.name(), "test");
    }
}
