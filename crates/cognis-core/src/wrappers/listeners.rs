//! Typed lifecycle listeners — fire on the boundaries of any
//! [`Runnable<I, O>`] invocation with the actual `I`/`O` values
//! (vs. the broad serialized [`crate::Event`] stream).
//!
//! Differences from observers:
//! - **Observers** receive serialized [`crate::Event`]s for everything
//!   running under a config. Cheap to attach globally.
//! - **Listeners** are typed, scoped to a single wrapped runnable, and
//!   see the exact `(I, O)` types — useful when you need to write a
//!   strongly-typed metric/decision (e.g. token-count of an output
//!   message) without round-tripping through JSON.
//!
//! Customization: implement [`Listener<I, O>`] for full control. For
//! drop-in cases use [`fn_listener`] or build from individual closures
//! via [`ListenerBuilder`].

use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;

use crate::runnable::{Runnable, RunnableConfig};
use crate::{CognisError, Result};

/// Object-safe listener trait. All hooks have default no-op impls;
/// override only what you need.
#[async_trait]
pub trait Listener<I, O>: Send + Sync
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Fired before `invoke` runs. Cannot mutate input — for that, use
    /// [`super::middleware::Middleware`].
    async fn on_start(&self, _input: &I, _config: &RunnableConfig) {}

    /// Fired on the success path with the produced output.
    async fn on_end(&self, _input: &I, _output: &O, _config: &RunnableConfig) {}

    /// Fired on the error path with the original input and the produced
    /// error. Cannot recover — for that, use middleware.
    async fn on_error(&self, _input: &I, _err: &CognisError, _config: &RunnableConfig) {}

    /// Friendly name for telemetry / diagnostics.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

// ---------------------------------------------------------------------------
// fn_listener — lift simple closures.
// ---------------------------------------------------------------------------

/// Lift an `on_end`-only closure into a [`Listener`].
pub fn fn_listener<I, O, F>(on_end: F) -> FnListener<I, O, F>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    F: Fn(&I, &O, &RunnableConfig) + Send + Sync + 'static,
{
    FnListener {
        on_end,
        _t: PhantomData,
    }
}

/// Closure-backed listener (on_end-only).
pub struct FnListener<I, O, F> {
    on_end: F,
    _t: PhantomData<fn(I) -> O>,
}

#[async_trait]
impl<I, O, F> Listener<I, O> for FnListener<I, O, F>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
    F: Fn(&I, &O, &RunnableConfig) + Send + Sync + 'static,
{
    async fn on_end(&self, input: &I, output: &O, config: &RunnableConfig) {
        (self.on_end)(input, output, config);
    }
}

// ---------------------------------------------------------------------------
// ListenerBuilder — compose listener from individual closures.
// ---------------------------------------------------------------------------

type StartFn<I> = Arc<dyn Fn(&I, &RunnableConfig) + Send + Sync>;
type EndFn<I, O> = Arc<dyn Fn(&I, &O, &RunnableConfig) + Send + Sync>;
type ErrorFn<I> = Arc<dyn Fn(&I, &CognisError, &RunnableConfig) + Send + Sync>;

/// Build a listener from any subset of closures. Useful when a custom
/// trait impl is overkill.
pub struct ListenerBuilder<I, O> {
    on_start: Option<StartFn<I>>,
    on_end: Option<EndFn<I, O>>,
    on_error: Option<ErrorFn<I>>,
    name: Option<String>,
    _t: PhantomData<fn(I) -> O>,
}

impl<I, O> Default for ListenerBuilder<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    fn default() -> Self {
        Self {
            on_start: None,
            on_end: None,
            on_error: None,
            name: None,
            _t: PhantomData,
        }
    }
}

impl<I, O> ListenerBuilder<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the listener's reported name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the on_start closure.
    pub fn on_start<F>(mut self, f: F) -> Self
    where
        F: Fn(&I, &RunnableConfig) + Send + Sync + 'static,
    {
        self.on_start = Some(Arc::new(f));
        self
    }

    /// Set the on_end closure.
    pub fn on_end<F>(mut self, f: F) -> Self
    where
        F: Fn(&I, &O, &RunnableConfig) + Send + Sync + 'static,
    {
        self.on_end = Some(Arc::new(f));
        self
    }

    /// Set the on_error closure.
    pub fn on_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&I, &CognisError, &RunnableConfig) + Send + Sync + 'static,
    {
        self.on_error = Some(Arc::new(f));
        self
    }

    /// Finalize into a [`Listener`].
    pub fn build(self) -> BuiltListener<I, O> {
        BuiltListener {
            on_start: self.on_start,
            on_end: self.on_end,
            on_error: self.on_error,
            name: self.name.unwrap_or_else(|| "BuiltListener".to_string()),
            _t: PhantomData,
        }
    }
}

/// Listener constructed via [`ListenerBuilder`].
pub struct BuiltListener<I, O> {
    on_start: Option<StartFn<I>>,
    on_end: Option<EndFn<I, O>>,
    on_error: Option<ErrorFn<I>>,
    name: String,
    _t: PhantomData<fn(I) -> O>,
}

#[async_trait]
impl<I, O> Listener<I, O> for BuiltListener<I, O>
where
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    async fn on_start(&self, input: &I, config: &RunnableConfig) {
        if let Some(f) = &self.on_start {
            f(input, config);
        }
    }
    async fn on_end(&self, input: &I, output: &O, config: &RunnableConfig) {
        if let Some(f) = &self.on_end {
            f(input, output, config);
        }
    }
    async fn on_error(&self, input: &I, err: &CognisError, config: &RunnableConfig) {
        if let Some(f) = &self.on_error {
            f(input, err, config);
        }
    }
    fn name(&self) -> &str {
        &self.name
    }
}

// ---------------------------------------------------------------------------
// WithListeners — wrap a Runnable.
// ---------------------------------------------------------------------------

/// Wraps a `Runnable<I, O>` with one or more [`Listener<I, O>`]s.
pub struct WithListeners<R, I, O> {
    inner: R,
    listeners: Vec<Arc<dyn Listener<I, O>>>,
    _phantom: PhantomData<fn(I) -> O>,
}

impl<R, I, O> WithListeners<R, I, O>
where
    R: Runnable<I, O>,
    I: Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    /// Wrap with no listeners.
    pub fn new(inner: R) -> Self {
        Self {
            inner,
            listeners: Vec::new(),
            _phantom: PhantomData,
        }
    }

    /// Append a listener.
    pub fn push(mut self, l: Arc<dyn Listener<I, O>>) -> Self {
        self.listeners.push(l);
        self
    }

    /// Number of registered listeners.
    pub fn len(&self) -> usize {
        self.listeners.len()
    }

    /// True if no listeners.
    pub fn is_empty(&self) -> bool {
        self.listeners.is_empty()
    }
}

#[async_trait]
impl<R, I, O> Runnable<I, O> for WithListeners<R, I, O>
where
    R: Runnable<I, O>,
    I: Clone + Send + Sync + 'static,
    O: Send + Sync + 'static,
{
    async fn invoke(&self, input: I, config: RunnableConfig) -> Result<O> {
        for l in self.listeners.iter() {
            l.on_start(&input, &config).await;
        }
        // Clone input so each listener can see the original on_start
        // value even if the inner runnable consumes it.
        let input_for_listeners = input.clone();
        let result = self.inner.invoke(input, config.clone()).await;
        match &result {
            Ok(o) => {
                for l in self.listeners.iter() {
                    l.on_end(&input_for_listeners, o, &config).await;
                }
            }
            Err(e) => {
                for l in self.listeners.iter() {
                    l.on_error(&input_for_listeners, e, &config).await;
                }
            }
        }
        result
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
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Echo;
    #[async_trait]
    impl Runnable<String, String> for Echo {
        async fn invoke(&self, input: String, _: RunnableConfig) -> Result<String> {
            Ok(input)
        }
    }

    struct Failing;
    #[async_trait]
    impl Runnable<String, String> for Failing {
        async fn invoke(&self, _: String, _: RunnableConfig) -> Result<String> {
            Err(CognisError::Internal("boom".into()))
        }
    }

    #[tokio::test]
    async fn fn_listener_fires_on_success() {
        let saw = Arc::new(AtomicUsize::new(0));
        let saw_for_l = saw.clone();
        let l = fn_listener::<String, String, _>(move |_, _, _| {
            saw_for_l.fetch_add(1, Ordering::SeqCst);
        });
        let chain = WithListeners::new(Echo).push(Arc::new(l));
        let _ = chain.invoke("x".into(), RunnableConfig::default()).await;
        assert_eq!(saw.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn builder_fires_each_phase() {
        let starts = Arc::new(AtomicUsize::new(0));
        let ends = Arc::new(AtomicUsize::new(0));
        let errs = Arc::new(AtomicUsize::new(0));
        let s2 = starts.clone();
        let e2 = ends.clone();
        let er2 = errs.clone();
        let l: BuiltListener<String, String> = ListenerBuilder::new()
            .on_start(move |_, _| {
                s2.fetch_add(1, Ordering::SeqCst);
            })
            .on_end(move |_, _, _| {
                e2.fetch_add(1, Ordering::SeqCst);
            })
            .on_error(move |_, _, _| {
                er2.fetch_add(1, Ordering::SeqCst);
            })
            .with_name("test-listener")
            .build();

        let chain = WithListeners::new(Echo).push(Arc::new(l));
        let _ = chain.invoke("x".into(), RunnableConfig::default()).await;
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert_eq!(ends.load(Ordering::SeqCst), 1);
        assert_eq!(errs.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn on_error_fires_on_failure() {
        let errs = Arc::new(AtomicUsize::new(0));
        let er2 = errs.clone();
        let l: BuiltListener<String, String> = ListenerBuilder::new()
            .on_error(move |_, _, _| {
                er2.fetch_add(1, Ordering::SeqCst);
            })
            .build();
        let chain = WithListeners::new(Failing).push(Arc::new(l));
        let res = chain.invoke("x".into(), RunnableConfig::default()).await;
        assert!(res.is_err());
        assert_eq!(errs.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn multiple_listeners_all_fire() {
        let count = Arc::new(AtomicUsize::new(0));
        let c1 = count.clone();
        let c2 = count.clone();
        let l1 = fn_listener::<String, String, _>(move |_, _, _| {
            c1.fetch_add(1, Ordering::SeqCst);
        });
        let l2 = fn_listener::<String, String, _>(move |_, _, _| {
            c2.fetch_add(10, Ordering::SeqCst);
        });
        let chain = WithListeners::new(Echo)
            .push(Arc::new(l1))
            .push(Arc::new(l2));
        let _ = chain.invoke("x".into(), RunnableConfig::default()).await;
        assert_eq!(count.load(Ordering::SeqCst), 11);
    }

    #[tokio::test]
    async fn listeners_do_not_alter_output() {
        let l = fn_listener::<String, String, _>(|_, _, _| {});
        let chain = WithListeners::new(Echo).push(Arc::new(l));
        let out = chain
            .invoke("hi".into(), RunnableConfig::default())
            .await
            .unwrap();
        assert_eq!(out, "hi");
    }
}
