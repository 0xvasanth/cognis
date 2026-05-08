//! Subagent middleware — route the chat call to one of N inner
//! [`PipelinedClient`]s based on a pluggable [`SubagentRouter`].
//!
//! Distinct from `tools::SubAgentTool` (which exposes a subagent as a
//! tool the LLM can call): this middleware *transparently* dispatches
//! the whole call to a different agent's chain when a router predicate
//! matches, without any tool-call ceremony.
//!
//! Use case: a "supervisor" agent that re-routes certain message
//! patterns to a specialist (e.g. queries containing SQL go to a
//! SQL-tuned agent's pipeline) without surfacing the routing decision
//! to the LLM.
//!
//! Customization:
//! - Implement [`SubagentRouter`] for full control.
//! - Closure routers are supported via blanket impl.

use std::sync::Arc;

use async_trait::async_trait;

use cognis_core::Result;
use cognis_llm::chat::ChatResponse;

use super::{Middleware, MiddlewareCtx, Next, PipelinedClient};

/// Routing decision: index of the chosen subagent in the configured
/// list, or `None` to let the request flow through to the default
/// pipeline (`next.invoke`).
pub trait SubagentRouter: Send + Sync {
    /// Pick a subagent index for the given context.
    fn pick(&self, ctx: &MiddlewareCtx) -> Option<usize>;
}

/// Closure-based router.
impl<F> SubagentRouter for F
where
    F: Fn(&MiddlewareCtx) -> Option<usize> + Send + Sync,
{
    fn pick(&self, ctx: &MiddlewareCtx) -> Option<usize> {
        (self)(ctx)
    }
}

/// Subagent middleware. Holds a slice of subagents (each is a
/// [`PipelinedClient`]) and routes calls via a [`SubagentRouter`].
pub struct SubagentMiddleware {
    subagents: Vec<PipelinedClient>,
    router: Arc<dyn SubagentRouter>,
}

impl SubagentMiddleware {
    /// Build with a router and a list of pipelined clients.
    pub fn new<R: SubagentRouter + 'static>(router: R, subagents: Vec<PipelinedClient>) -> Self {
        Self {
            subagents,
            router: Arc::new(router),
        }
    }

    /// Append a subagent to the list.
    pub fn push(mut self, subagent: PipelinedClient) -> Self {
        self.subagents.push(subagent);
        self
    }

    /// Borrow the configured subagents.
    pub fn subagents(&self) -> &[PipelinedClient] {
        &self.subagents
    }
}

#[async_trait]
impl Middleware for SubagentMiddleware {
    async fn call(&self, ctx: MiddlewareCtx, next: Arc<dyn Next>) -> Result<ChatResponse> {
        match self.router.pick(&ctx) {
            Some(idx) => {
                let sa = self.subagents.get(idx).ok_or_else(|| {
                    cognis_core::CognisError::Configuration(format!(
                        "subagent router returned out-of-range index {idx} (have {})",
                        self.subagents.len()
                    ))
                })?;
                sa.invoke(ctx.messages, ctx.tool_defs, ctx.opts).await
            }
            None => next.invoke(ctx).await,
        }
    }
    fn name(&self) -> &str {
        "SubagentMiddleware"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::middleware::{tests_util::ok_resp, MiddlewarePipeline};
    use cognis_core::{Message, RunnableStream};
    use cognis_llm::chat::{ChatOptions, ChatResponse, HealthStatus, StreamChunk};
    use cognis_llm::provider::{LLMProvider, Provider};
    use cognis_llm::Client;

    use crate::middleware::tests_util::FixedNext;

    /// Provider that returns a fixed response so we can identify which
    /// subagent the middleware dispatched to.
    struct Tagged(&'static str);

    #[async_trait]
    impl LLMProvider for Tagged {
        fn name(&self) -> &str {
            self.0
        }
        fn provider_type(&self) -> Provider {
            Provider::Ollama
        }
        async fn chat_completion(&self, _: Vec<Message>, _: ChatOptions) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::ai(self.0),
                usage: None,
                finish_reason: "stop".into(),
                model: self.0.into(),
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

    fn pipeline_for(tag: &'static str) -> PipelinedClient {
        let client = Client::new(Arc::new(Tagged(tag)));
        MiddlewarePipeline::new().build(client)
    }

    #[tokio::test]
    async fn routes_to_chosen_subagent() {
        let mw = SubagentMiddleware::new(
            |ctx: &MiddlewareCtx| {
                if ctx.messages.iter().any(|m| m.content().contains("sql")) {
                    Some(0)
                } else {
                    None
                }
            },
            vec![pipeline_for("sql-specialist")],
        );
        let next: Arc<dyn Next> = Arc::new(FixedNext(ok_resp("default")));
        let r = mw
            .call(
                MiddlewareCtx::new(
                    vec![Message::human("write a sql query")],
                    vec![],
                    Default::default(),
                ),
                next,
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "sql-specialist");
    }

    #[tokio::test]
    async fn falls_through_when_router_returns_none() {
        let mw = SubagentMiddleware::new(|_ctx: &MiddlewareCtx| None, vec![pipeline_for("never")]);
        let next: Arc<dyn Next> = Arc::new(FixedNext(ok_resp("default-took-over")));
        let r = mw
            .call(
                MiddlewareCtx::new(vec![Message::human("hi")], vec![], Default::default()),
                next,
            )
            .await
            .unwrap();
        assert_eq!(r.message.content(), "default-took-over");
    }

    #[tokio::test]
    async fn out_of_range_index_errors() {
        let mw =
            SubagentMiddleware::new(|_ctx: &MiddlewareCtx| Some(99), vec![pipeline_for("only")]);
        let next: Arc<dyn Next> = Arc::new(FixedNext(ok_resp("never")));
        let res = mw
            .call(
                MiddlewareCtx::new(vec![Message::human("hi")], vec![], Default::default()),
                next,
            )
            .await;
        assert!(res.is_err());
    }
}
