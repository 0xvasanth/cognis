//! Inject dynamic context (date, user, locale, tenant) at the head of every
//! request. Idempotent via a hidden marker.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::{Message, Result};
use cognis_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next};

const MARKER: &str = "<!-- cognis:context-inject -->";

/// Source of dynamic context. Implementations can read clocks, env vars,
/// tenant config, etc.
pub trait ContextProvider: Send + Sync {
    /// Render the current context as a system-message body.
    fn render(&self) -> String;
}

/// Closure-backed provider.
pub struct FnContextProvider<F: Fn() -> String + Send + Sync>(pub F);

impl<F: Fn() -> String + Send + Sync> ContextProvider for FnContextProvider<F> {
    fn render(&self) -> String {
        (self.0)()
    }
}

/// Inserts a fresh system message at the head of every request.
pub struct ContextInjection {
    provider: Arc<dyn ContextProvider>,
}

impl ContextInjection {
    /// Build with a provider.
    pub fn new(provider: Arc<dyn ContextProvider>) -> Self {
        Self { provider }
    }
}

#[async_trait]
impl Middleware for ContextInjection {
    async fn call(&self, mut ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        let already = ctx
            .messages
            .iter()
            .any(|m| matches!(m, Message::System(s) if s.content.contains(MARKER)));
        if !already {
            let body = format!("{MARKER}\n{}", self.provider.render());
            ctx.messages.insert(0, Message::system(body));
        }
        next.invoke(ctx).await
    }
    fn name(&self) -> &str {
        "ContextInjection"
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests_util::*;
    use super::*;
    use crate::middleware::MiddlewarePipeline;

    use cognis_llm::chat::ChatOptions;
    use cognis_llm::Client;

    #[tokio::test]
    async fn injects_context_at_head() {
        let rec = make_recording_provider("ok");
        let provider = FnContextProvider(|| "tenant=acme date=2026-05-07".into());
        let pipe = MiddlewarePipeline::new()
            .push(ContextInjection::new(Arc::new(provider)))
            .build(Client::new(rec.clone()));
        let _ = pipe
            .invoke(
                vec![Message::human("hi")],
                Vec::new(),
                ChatOptions::default(),
            )
            .await
            .unwrap();
        let received = rec.received.lock().unwrap();
        assert!(received[0].0[0].content().contains("tenant=acme"));
    }
}
