//! Runnable-level middleware — generic before/after/error hooks that
//! wrap any [`Runnable`] without subclassing.
//!
//! Differs from the agent-orchestration middleware in `cognis::middleware`:
//! that one is specific to LLM-call shapes; this one is generic over
//! `(I, O)` and works for any runnable in the framework.
//!
//! Three integration points:
//! - [`Middleware<I, O>`] trait — implement to plug behavior in.
//! - [`WithMiddleware<R, I, O>`] wrapper — wraps a single runnable with
//!   one or more middlewares (executed in registration order).
//! - [`MiddlewareStack<I, O>`] — explicit ordered list, useful when a
//!   caller wants to compose middlewares once and reuse them.
//!
//! Customization: implement `Middleware<I, O>` for full control. For
//! lighter cases, use [`fn_middleware`] to lift a closure or
//! [`InspectMiddleware`] to attach pure read-only observers.

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;

use crate::runnable::{Runnable, RunnableConfig};
use crate::{CognisError, Result};

/// Object-safe middleware trait. The three hooks fire in this order:
///
/// 1. `before_invoke` — receives the input by `&mut`, may rewrite it or
///    short-circuit by returning an error.
/// 2. (the wrapped runnable runs)
/// 3. `after_invoke` (on `Ok`) — receives the output by `&mut`, may
///    rewrite it.
/// 4. `on_error` (on `Err`) — receives the error by `&mut`, may rewrite
///    or substitute. Returning `Ok(Some(new_output))` recovers the call.
///
/// All hooks have default no-op impls; implement only what you need.
#[async_trait]
pub trait Middleware<I, O>: Send + Sync
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Called with the input before the wrapped runnable is invoked.
    /// Mutate `input` in place to rewrite it. Return `Err(...)` to
    /// short-circuit (the wrapped runnable will not be called).
    async fn before_invoke(&self, _input: &mut I, _config: &RunnableConfig) -> Result<()> {
        Ok(())
    }

    /// Called on the success path with the output. Mutate to rewrite.
    async fn after_invoke(&self, _output: &mut O, _config: &RunnableConfig) -> Result<()> {
        Ok(())
    }

    /// Called on the error path. Returning `Ok(Some(o))` recovers — the
    /// invocation completes successfully with that output. Returning
    /// `Ok(None)` (the default) re-propagates the error.
    /// Returning `Err(e)` substitutes the original error.
    async fn on_error(
        &self,
        _err: &mut CognisError,
        _config: &RunnableConfig,
    ) -> Result<Option<O>> {
        Ok(None)
    }

    /// Friendly name for telemetry / diagnostics.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

// ---------------------------------------------------------------------------
// fn_middleware — lift a closure into a Middleware.
// ---------------------------------------------------------------------------

/// Build a middleware from a `before_invoke` closure only. For the
/// after / error hooks, use [`Middleware`] directly.
pub fn fn_middleware<I, O, F>(before: F) -> FnMiddleware<I, O, F>
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(&mut I, &RunnableConfig) -> Result<()> + Send + Sync + 'static,
{
    FnMiddleware {
        before,
        _t: PhantomData,
    }
}

/// Closure-backed middleware (before-only).
pub struct FnMiddleware<I, O, F> {
    before: F,
    _t: PhantomData<fn(I) -> O>,
}

#[async_trait]
impl<I, O, F> Middleware<I, O> for FnMiddleware<I, O, F>
where
    I: Send + 'static,
    O: Send + 'static,
    F: Fn(&mut I, &RunnableConfig) -> Result<()> + Send + Sync + 'static,
{
    async fn before_invoke(&self, input: &mut I, config: &RunnableConfig) -> Result<()> {
        (self.before)(input, config)
    }
}

// ---------------------------------------------------------------------------
// InspectMiddleware — pure read-only observation, no mutation.
// ---------------------------------------------------------------------------

/// Read-only inspector. Closure receives `(&I, &RunnableConfig)` after
/// each successful invocation (purely for telemetry / metrics).
pub struct InspectMiddleware<I, O, F> {
    on_ok: F,
    _t: PhantomData<fn(I) -> O>,
}

impl<I, O, F> InspectMiddleware<I, O, F>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    F: Fn(&O, &RunnableConfig) + Send + Sync + 'static,
{
    /// Build an inspector that fires on `after_invoke`.
    pub fn new(on_ok: F) -> Self {
        Self {
            on_ok,
            _t: PhantomData,
        }
    }
}

#[async_trait]
impl<I, O, F> Middleware<I, O> for InspectMiddleware<I, O, F>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    F: Fn(&O, &RunnableConfig) + Send + Sync + 'static,
{
    async fn after_invoke(&self, output: &mut O, config: &RunnableConfig) -> Result<()> {
        (self.on_ok)(output, config);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MiddlewareStack — explicit ordered list of middlewares.
// ---------------------------------------------------------------------------

/// Ordered list of middlewares. `before_invoke` runs first→last;
/// `after_invoke` runs in reverse (last registered → first registered),
/// matching standard middleware ordering.
pub struct MiddlewareStack<I, O> {
    inner: Vec<Arc<dyn Middleware<I, O>>>,
}

impl<I, O> Default for MiddlewareStack<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<I, O> Clone for MiddlewareStack<I, O> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<I, O> MiddlewareStack<I, O>
where
    I: Send + 'static,
    O: Send + 'static,
{
    /// Empty stack.
    pub fn new() -> Self {
        Self { inner: Vec::new() }
    }

    /// Append a middleware to the end of the chain.
    pub fn push(mut self, m: Arc<dyn Middleware<I, O>>) -> Self {
        self.inner.push(m);
        self
    }

    /// Number of registered middlewares.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// True if no middlewares are registered.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Borrow the registered middlewares (read-only).
    pub fn middlewares(&self) -> &[Arc<dyn Middleware<I, O>>] {
        &self.inner
    }
}

// ---------------------------------------------------------------------------
// WithMiddleware — wrap a Runnable with a MiddlewareStack.
// ---------------------------------------------------------------------------

/// Wraps a `Runnable<I, O>` with a stack of middlewares.
pub struct WithMiddleware<R, I, O> {
    inner: R,
    stack: MiddlewareStack<I, O>,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<R, I, O> WithMiddleware<R, I, O>
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Send + 'static,
{
    /// Wrap with an empty stack — equivalent to `inner` itself but
    /// chainable via [`Self::push`].
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            stack: MiddlewareStack::new(),
            _phantom: PhantomData,
        }
    }

    /// Wrap with an existing stack.
    pub fn with_stack(inner: R, stack: MiddlewareStack<I, O>) -> Self {
        Self {
            inner,
            stack,
            _phantom: PhantomData,
        }
    }

    /// Append a middleware.
    pub fn push(mut self, m: Arc<dyn Middleware<I, O>>) -> Self {
        self.stack = self.stack.push(m);
        self
    }
}

#[async_trait]
impl<R, I, O> Runnable<I, O> for WithMiddleware<R, I, O>
where
    R: Runnable<I, O>,
    I: Send + 'static,
    O: Send + 'static,
{
    async fn invoke(&self, mut input: I, config: RunnableConfig) -> Result<O> {
        // before — first → last
        for m in self.stack.inner.iter() {
            m.before_invoke(&mut input, &config).await?;
        }
        let result = self.inner.invoke(input, config.clone()).await;
        match result {
            Ok(mut output) => {
                // after — last → first (standard onion order)
                for m in self.stack.inner.iter().rev() {
                    m.after_invoke(&mut output, &config).await?;
                }
                Ok(output)
            }
            Err(mut err) => {
                // on_error — last → first; first to recover wins.
                for m in self.stack.inner.iter().rev() {
                    if let Some(o) = m.on_error(&mut err, &config).await? {
                        return Ok(o);
                    }
                }
                Err(err)
            }
        }
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn input_schema(&self) -> Option<serde_json::Value> {
        self.inner.input_schema()
    }

    fn output_schema(&self) -> Option<serde_json::Value> {
        self.inner.output_schema()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    struct Echo;

    #[async_trait]
    impl Runnable<String, String> for Echo {
        async fn invoke(&self, input: String, _: RunnableConfig) -> Result<String> {
            Ok(input)
        }
    }

    struct UppercaseInput;

    #[async_trait]
    impl Middleware<String, String> for UppercaseInput {
        async fn before_invoke(&self, input: &mut String, _: &RunnableConfig) -> Result<()> {
            *input = input.to_uppercase();
            Ok(())
        }
    }

    struct AppendOutput(&'static str);

    #[async_trait]
    impl Middleware<String, String> for AppendOutput {
        async fn after_invoke(&self, output: &mut String, _: &RunnableConfig) -> Result<()> {
            output.push_str(self.0);
            Ok(())
        }
    }

    #[tokio::test]
    async fn before_invoke_rewrites_input() {
        let chain = WithMiddleware::new(Echo).push(Arc::new(UppercaseInput));
        let out = chain
            .invoke("hello".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out, "HELLO");
    }

    #[tokio::test]
    async fn after_invoke_rewrites_output() {
        let chain = WithMiddleware::new(Echo).push(Arc::new(AppendOutput("!")));
        let out = chain
            .invoke("hi".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out, "hi!");
    }

    #[tokio::test]
    async fn middlewares_run_in_onion_order() {
        // before order: outer → inner. after order: inner → outer.
        // Starting input "x" → outer adds "(", then inner adds "[":
        //   before(outer): "(x"
        //   before(inner): "([x"
        //   echo:          "([x"
        //   after(inner):  "([x]"
        //   after(outer):  "([x])"
        struct Outer;
        struct Inner;

        #[async_trait]
        impl Middleware<String, String> for Outer {
            async fn before_invoke(&self, input: &mut String, _: &RunnableConfig) -> Result<()> {
                *input = format!("({input}");
                Ok(())
            }
            async fn after_invoke(&self, output: &mut String, _: &RunnableConfig) -> Result<()> {
                output.push(')');
                Ok(())
            }
        }
        #[async_trait]
        impl Middleware<String, String> for Inner {
            async fn before_invoke(&self, input: &mut String, _: &RunnableConfig) -> Result<()> {
                // Insert `[` right after the leading `(` placed by Outer.
                // Input arrives as "(x"; we want "([x".
                if let Some(idx) = input.find('(') {
                    input.insert(idx + 1, '[');
                }
                Ok(())
            }
            async fn after_invoke(&self, output: &mut String, _: &RunnableConfig) -> Result<()> {
                output.push(']');
                Ok(())
            }
        }

        let chain = WithMiddleware::new(Echo)
            .push(Arc::new(Outer))
            .push(Arc::new(Inner));
        let out = chain
            .invoke("x".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out, "([x])");
    }

    #[tokio::test]
    async fn before_invoke_can_short_circuit() {
        struct Reject;
        #[async_trait]
        impl Middleware<String, String> for Reject {
            async fn before_invoke(&self, _: &mut String, _: &RunnableConfig) -> Result<()> {
                Err(CognisError::Configuration("rejected by middleware".into()))
            }
        }
        let chain = WithMiddleware::new(Echo).push(Arc::new(Reject));
        let err = chain
            .invoke("x".into(), RunnableConfig::default())
            .await
            .unwrap_err();
        assert!(matches!(err, CognisError::Configuration(_)));
    }

    #[tokio::test]
    async fn on_error_can_recover() {
        struct Failing;
        #[async_trait]
        impl Runnable<String, String> for Failing {
            async fn invoke(&self, _: String, _: RunnableConfig) -> Result<String> {
                Err(CognisError::Internal("boom".into()))
            }
        }
        struct Recover;
        #[async_trait]
        impl Middleware<String, String> for Recover {
            async fn on_error(
                &self,
                _: &mut CognisError,
                _: &RunnableConfig,
            ) -> Result<Option<String>> {
                Ok(Some("recovered".into()))
            }
        }
        let chain = WithMiddleware::new(Failing).push(Arc::new(Recover));
        let out = chain
            .invoke("x".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out, "recovered");
    }

    #[tokio::test]
    async fn fn_middleware_lifts_closure() {
        let saw = Arc::new(AtomicBool::new(false));
        let saw_for_mw = saw.clone();
        let mw = fn_middleware::<String, String, _>(move |input, _| {
            saw_for_mw.store(true, Ordering::SeqCst);
            input.push('!');
            Ok(())
        });
        let chain = WithMiddleware::new(Echo).push(Arc::new(mw));
        let out = chain
            .invoke("hi".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert!(saw.load(Ordering::SeqCst));
        assert_eq!(out, "hi!");
    }

    #[tokio::test]
    async fn inspect_middleware_does_not_mutate() {
        let count = Arc::new(AtomicUsize::new(0));
        let count_for_mw = count.clone();
        let inspector = InspectMiddleware::<String, String, _>::new(move |_out, _cfg| {
            count_for_mw.fetch_add(1, Ordering::SeqCst);
        });
        let chain = WithMiddleware::new(Echo).push(Arc::new(inspector));
        let out = chain
            .invoke("hi".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out, "hi");
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn middleware_stack_clone_independent() {
        let stack = MiddlewareStack::<String, String>::new()
            .push(Arc::new(UppercaseInput))
            .push(Arc::new(AppendOutput("!")));
        let chain1 = WithMiddleware::with_stack(Echo, stack.clone());
        let chain2 = WithMiddleware::with_stack(Echo, stack);
        let cfg = RunnableConfig::default();
        let o1 = chain1.invoke("hi".into(), cfg.clone()).await.unwrap();
        let o2 = chain2.invoke("hi".into(), cfg).await.unwrap();
        assert_eq!(o1, "HI!");
        assert_eq!(o2, "HI!");
    }
}
