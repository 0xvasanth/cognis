//! Interceptor — chat-shape before/after/error hooks for any
//! [`LLMProvider`].
//!
//! Differences from the generic `cognis_core::Middleware`:
//! - Operates on chat-specific types ([`ChatOptions`], [`ChatResponse`],
//!   [`Vec<Message>`]) rather than `(I, O)`.
//! - Hooks fire on every provider entry-point (`chat_completion`,
//!   `chat_completion_stream`, `chat_completion_with_tools`) — no need
//!   to write three sets of generic hooks.
//!
//! Customization:
//! - Implement [`ChatInterceptor`] for full control.
//! - Use [`FnChatInterceptor`] to assemble from individual closures.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{CognisError, Result, RunnableStream};

use crate::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk};
use crate::provider::{LLMProvider, Provider};
use crate::tools::ToolDefinition;
use crate::Message;

/// Chat-shape hooks. All have default no-op impls.
#[async_trait]
pub trait ChatInterceptor: Send + Sync {
    /// Fired before the inner provider is called. Mutate the request in
    /// place. Return `Err(...)` to short-circuit — the provider will
    /// not be called.
    async fn before_call(
        &self,
        _messages: &mut Vec<Message>,
        _opts: &mut ChatOptions,
    ) -> Result<()> {
        Ok(())
    }

    /// Fired on a successful response. Mutate the response in place.
    async fn after_call(&self, _response: &mut ChatResponse) -> Result<()> {
        Ok(())
    }

    /// Fired on an error. Returning `Ok(Some(resp))` substitutes a
    /// recovered response. `Ok(None)` re-propagates the error. `Err(...)`
    /// substitutes the error.
    async fn on_error(&self, _err: &mut CognisError) -> Result<Option<ChatResponse>> {
        Ok(None)
    }

    /// Friendly name for diagnostics.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
}

// ---------------------------------------------------------------------------
// FnChatInterceptor — assemble from individual closures.
// ---------------------------------------------------------------------------

type BeforeFn = Arc<
    dyn for<'a> Fn(
            &'a mut Vec<Message>,
            &'a mut ChatOptions,
        ) -> futures::future::BoxFuture<'a, Result<()>>
        + Send
        + Sync,
>;
type AfterFn = Arc<
    dyn for<'a> Fn(&'a mut ChatResponse) -> futures::future::BoxFuture<'a, Result<()>>
        + Send
        + Sync,
>;
type ErrFn = Arc<
    dyn for<'a> Fn(
            &'a mut CognisError,
        ) -> futures::future::BoxFuture<'a, Result<Option<ChatResponse>>>
        + Send
        + Sync,
>;

/// Closure-assembled interceptor. All hooks are optional.
#[derive(Default)]
pub struct FnChatInterceptor {
    before: Option<BeforeFn>,
    after: Option<AfterFn>,
    on_err: Option<ErrFn>,
    name: Option<String>,
}

impl FnChatInterceptor {
    /// Empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the friendly name.
    pub fn with_name(mut self, n: impl Into<String>) -> Self {
        self.name = Some(n.into());
        self
    }

    /// Sync `before_call` shorthand. The closure runs synchronously and
    /// is wrapped in `async`.
    pub fn before<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut Vec<Message>, &mut ChatOptions) -> Result<()> + Send + Sync + 'static,
    {
        self.before = Some(Arc::new(move |msgs, opts| {
            let res = f(msgs, opts);
            Box::pin(async move { res })
        }));
        self
    }

    /// Sync `after_call` shorthand.
    pub fn after<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut ChatResponse) -> Result<()> + Send + Sync + 'static,
    {
        self.after = Some(Arc::new(move |resp| {
            let res = f(resp);
            Box::pin(async move { res })
        }));
        self
    }

    /// Sync `on_error` shorthand.
    pub fn on_error<F>(mut self, f: F) -> Self
    where
        F: Fn(&mut CognisError) -> Result<Option<ChatResponse>> + Send + Sync + 'static,
    {
        self.on_err = Some(Arc::new(move |e| {
            let res = f(e);
            Box::pin(async move { res })
        }));
        self
    }
}

#[async_trait]
impl ChatInterceptor for FnChatInterceptor {
    async fn before_call(&self, messages: &mut Vec<Message>, opts: &mut ChatOptions) -> Result<()> {
        if let Some(f) = &self.before {
            f(messages, opts).await
        } else {
            Ok(())
        }
    }
    async fn after_call(&self, response: &mut ChatResponse) -> Result<()> {
        if let Some(f) = &self.after {
            f(response).await
        } else {
            Ok(())
        }
    }
    async fn on_error(&self, err: &mut CognisError) -> Result<Option<ChatResponse>> {
        if let Some(f) = &self.on_err {
            f(err).await
        } else {
            Ok(None)
        }
    }
    fn name(&self) -> &str {
        self.name.as_deref().unwrap_or("FnChatInterceptor")
    }
}

// ---------------------------------------------------------------------------
// InterceptorProvider — wrap a provider with a chain of interceptors.
// ---------------------------------------------------------------------------

/// Wraps an [`LLMProvider`] with one or more [`ChatInterceptor`]s.
///
/// Execution order:
/// - `before_call` runs first → last (registration order).
/// - The inner provider runs.
/// - `after_call` runs last → first (onion-out).
/// - `on_error` runs last → first; first to recover wins.
pub struct InterceptorProvider {
    inner: Arc<dyn LLMProvider>,
    chain: Vec<Arc<dyn ChatInterceptor>>,
    name: String,
}

impl InterceptorProvider {
    /// Wrap with no interceptors (no-op until configured).
    pub fn new(inner: Arc<dyn LLMProvider>) -> Self {
        let name = inner.name().to_string();
        Self {
            inner,
            chain: Vec::new(),
            name,
        }
    }

    /// Append an interceptor to the chain.
    pub fn push(mut self, ic: Arc<dyn ChatInterceptor>) -> Self {
        self.chain.push(ic);
        self
    }

    /// Borrow the registered interceptors.
    pub fn interceptors(&self) -> &[Arc<dyn ChatInterceptor>] {
        &self.chain
    }

    async fn run_before(&self, messages: &mut Vec<Message>, opts: &mut ChatOptions) -> Result<()> {
        for ic in &self.chain {
            ic.before_call(messages, opts).await?;
        }
        Ok(())
    }

    async fn run_after(&self, resp: &mut ChatResponse) -> Result<()> {
        for ic in self.chain.iter().rev() {
            ic.after_call(resp).await?;
        }
        Ok(())
    }

    async fn run_error(
        &self,
        mut err: CognisError,
    ) -> std::result::Result<ChatResponse, CognisError> {
        for ic in self.chain.iter().rev() {
            match ic.on_error(&mut err).await {
                Ok(Some(r)) => return Ok(r),
                Ok(None) => {}
                Err(e) => err = e,
            }
        }
        Err(err)
    }
}

#[async_trait]
impl LLMProvider for InterceptorProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn provider_type(&self) -> Provider {
        self.inner.provider_type()
    }

    async fn chat_completion(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut messages = messages;
        let mut opts = opts;
        self.run_before(&mut messages, &mut opts).await?;
        match self.inner.chat_completion(messages, opts).await {
            Ok(mut r) => {
                self.run_after(&mut r).await?;
                Ok(r)
            }
            Err(e) => self.run_error(e).await,
        }
    }

    async fn chat_completion_stream(
        &self,
        messages: Vec<Message>,
        opts: ChatOptions,
    ) -> Result<RunnableStream<StreamChunk>> {
        // Streaming hooks only fire `before_call` (the per-chunk path
        // doesn't have a clean after-mutation point).
        let mut messages = messages;
        let mut opts = opts;
        self.run_before(&mut messages, &mut opts).await?;
        self.inner.chat_completion_stream(messages, opts).await
    }

    async fn chat_completion_with_tools(
        &self,
        messages: Vec<Message>,
        tools: Vec<ToolDefinition>,
        opts: ChatOptions,
    ) -> Result<ChatResponse> {
        let mut messages = messages;
        let mut opts = opts;
        self.run_before(&mut messages, &mut opts).await?;
        match self
            .inner
            .chat_completion_with_tools(messages, tools, opts)
            .await
        {
            Ok(mut r) => {
                self.run_after(&mut r).await?;
                Ok(r)
            }
            Err(e) => self.run_error(e).await,
        }
    }

    async fn health_check(&self) -> Result<HealthStatus> {
        self.inner.health_check().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;
    #[async_trait]
    impl LLMProvider for Echo {
        fn name(&self) -> &str {
            "echo"
        }
        fn provider_type(&self) -> Provider {
            Provider::OpenAI
        }
        async fn chat_completion(
            &self,
            messages: Vec<Message>,
            _: ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::ai(
                    messages
                        .last()
                        .map(|m| m.content())
                        .unwrap_or("")
                        .to_string(),
                ),
                usage: None,
                finish_reason: "stop".into(),
                model: "echo".into(),
            })
        }
        async fn chat_completion_stream(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unimplemented!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    struct Failing;
    #[async_trait]
    impl LLMProvider for Failing {
        fn name(&self) -> &str {
            "failing"
        }
        fn provider_type(&self) -> Provider {
            Provider::OpenAI
        }
        async fn chat_completion(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
            Err(CognisError::Internal("boom".into()))
        }
        async fn chat_completion_stream(
            &self,
            _: Vec<Message>,
            _: ChatOptions,
        ) -> Result<RunnableStream<StreamChunk>> {
            unimplemented!()
        }
        async fn health_check(&self) -> Result<HealthStatus> {
            Ok(HealthStatus::Healthy { latency_ms: 0 })
        }
    }

    #[tokio::test]
    async fn before_can_rewrite_messages() {
        let inner = Arc::new(Echo);
        let ic = FnChatInterceptor::new().before(|msgs, _| {
            // Append a sentinel suffix to the last human message.
            if let Some(last) = msgs.last_mut() {
                if matches!(last, Message::Human(_)) {
                    let new = format!("{}!!!", last.content());
                    *last = Message::human(new);
                }
            }
            Ok(())
        });
        let p = InterceptorProvider::new(inner).push(Arc::new(ic));
        let r = p
            .chat_completion(vec![Message::human("hi")], ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(r.message.content(), "hi!!!");
    }

    #[tokio::test]
    async fn after_can_rewrite_response() {
        let inner = Arc::new(Echo);
        let ic = FnChatInterceptor::new().after(|resp| {
            let new = format!("[{}]", resp.message.content());
            resp.message = Message::ai(new);
            Ok(())
        });
        let p = InterceptorProvider::new(inner).push(Arc::new(ic));
        let r = p
            .chat_completion(vec![Message::human("hi")], ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(r.message.content(), "[hi]");
    }

    #[tokio::test]
    async fn on_error_can_recover() {
        let inner = Arc::new(Failing);
        let ic = FnChatInterceptor::new().on_error(|_e| {
            Ok(Some(ChatResponse {
                message: Message::ai("recovered"),
                usage: None,
                finish_reason: "stop".into(),
                model: "n/a".into(),
            }))
        });
        let p = InterceptorProvider::new(inner).push(Arc::new(ic));
        let r = p
            .chat_completion(vec![Message::human("hi")], ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(r.message.content(), "recovered");
    }

    #[tokio::test]
    async fn before_short_circuits_via_err() {
        let inner = Arc::new(Echo);
        let ic = FnChatInterceptor::new()
            .before(|_msgs, _opts| Err(CognisError::Configuration("blocked".into())));
        let p = InterceptorProvider::new(inner).push(Arc::new(ic));
        let err = p
            .chat_completion(vec![Message::human("x")], ChatOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, CognisError::Configuration(_)));
    }

    #[tokio::test]
    async fn onion_order_after_runs_outer_to_inner_reverse() {
        // outer adds (), inner adds [].
        let inner = Arc::new(Echo);
        let outer = FnChatInterceptor::new().after(|r| {
            let n = format!("({})", r.message.content());
            r.message = Message::ai(n);
            Ok(())
        });
        let inner_ic = FnChatInterceptor::new().after(|r| {
            let n = format!("[{}]", r.message.content());
            r.message = Message::ai(n);
            Ok(())
        });
        let p = InterceptorProvider::new(inner)
            .push(Arc::new(outer))
            .push(Arc::new(inner_ic));
        let r = p
            .chat_completion(vec![Message::human("x")], ChatOptions::default())
            .await
            .unwrap();
        // after runs reversed: inner first → "[x]", then outer → "([x])".
        assert_eq!(r.message.content(), "([x])");
    }
}
